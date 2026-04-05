/*
 * QTest: G233 PWM controller — basic functionality
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * PWM register map (base 0x10015000), 4 channels:
 *   0x00  PWM_GLB      — bits[3:0] CHn_EN mirror, bits[7:4] CHn_DONE (w1c)
 *   Per channel (CHn at 0x10 + n*0x10):
 *     +0x00 CHn_CTRL   — bit0: EN, bit1: POL
 *     +0x04 CHn_PERIOD — period value
 *     +0x08 CHn_DUTY   — duty cycle
 *     +0x0C CHn_CNT    — counter (read-only)
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define PWM_BASE        0x10015000ULL
#define PWM_GLB         (PWM_BASE + 0x00)

#define PWM_CH_CTRL(n)   (PWM_BASE + 0x10 + (n) * 0x10 + 0x00)
#define PWM_CH_PERIOD(n) (PWM_BASE + 0x10 + (n) * 0x10 + 0x04)
#define PWM_CH_DUTY(n)   (PWM_BASE + 0x10 + (n) * 0x10 + 0x08)
#define PWM_CH_CNT(n)    (PWM_BASE + 0x10 + (n) * 0x10 + 0x0C)

/* PWM_GLB bit fields */
#define PWM_GLB_CH_EN(n)    (1u << (n))
#define PWM_GLB_CH_DONE(n)  (1u << (4 + (n)))

/* PWM_CHn_CTRL bit fields */
#define PWM_CTRL_EN   (1u << 0)
#define PWM_CTRL_POL  (1u << 1)

static void test_pwm_config(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, PWM_CH_PERIOD(0), 1000);
    qtest_writel(qts, PWM_CH_DUTY(0), 500);

    g_assert_cmpuint(qtest_readl(qts, PWM_CH_PERIOD(0)), ==, 1000);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_DUTY(0)), ==, 500);

    qtest_quit(qts);
}

static void test_pwm_enable(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, PWM_CH_PERIOD(0), 1000);
    qtest_writel(qts, PWM_CH_DUTY(0), 500);

    /* Enable CH0 */
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CTRL(0)) & PWM_CTRL_EN, ==,
                     PWM_CTRL_EN);

    /* GLB mirror bit should reflect CH0 enabled */
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_EN(0), ==,
                     PWM_GLB_CH_EN(0));

    qtest_quit(qts);
}

static void test_pwm_counter(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, PWM_CH_PERIOD(0), 0xFFFF);
    qtest_writel(qts, PWM_CH_DUTY(0), 0x8000);
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN);

    /* Advance virtual clock to let counter increment */
    qtest_clock_step(qts, 1000000);  /* 1ms */

    uint32_t cnt = qtest_readl(qts, PWM_CH_CNT(0));
    g_assert_cmpuint(cnt, >, 0);

    qtest_quit(qts);
}

static void test_pwm_done_flag(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* Short period to trigger DONE quickly */
    qtest_writel(qts, PWM_CH_PERIOD(0), 10);
    qtest_writel(qts, PWM_CH_DUTY(0), 5);
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN);

    /* Advance enough for period to complete */
    qtest_clock_step(qts, 100000000);  /* 100ms */

    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0), ==,
                     PWM_GLB_CH_DONE(0));

    qtest_quit(qts);
}

static void test_pwm_done_clear(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, PWM_CH_PERIOD(0), 10);
    qtest_writel(qts, PWM_CH_DUTY(0), 5);
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN);
    qtest_clock_step(qts, 100000000);

    /* Verify DONE is set */
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0), !=, 0);

    /* Write 1 to clear */
    qtest_writel(qts, PWM_GLB, PWM_GLB_CH_DONE(0));
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0), ==, 0);

    qtest_quit(qts);
}

static void test_pwm_multi_channel(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    for (int ch = 0; ch < 4; ch++) {
        qtest_writel(qts, PWM_CH_PERIOD(ch), 100 * (ch + 1));
        qtest_writel(qts, PWM_CH_DUTY(ch), 50 * (ch + 1));
    }

    for (int ch = 0; ch < 4; ch++) {
        g_assert_cmpuint(qtest_readl(qts, PWM_CH_PERIOD(ch)), ==,
                         100 * (ch + 1));
        g_assert_cmpuint(qtest_readl(qts, PWM_CH_DUTY(ch)), ==,
                         50 * (ch + 1));
    }

    qtest_quit(qts);
}

static void test_pwm_polarity(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_POL);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CTRL(0)) & PWM_CTRL_POL, ==,
                     PWM_CTRL_POL);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/pwm/config", test_pwm_config);
    qtest_add_func("g233/pwm/enable", test_pwm_enable);
    qtest_add_func("g233/pwm/counter", test_pwm_counter);
    qtest_add_func("g233/pwm/done_flag", test_pwm_done_flag);
    qtest_add_func("g233/pwm/done_clear", test_pwm_done_clear);
    qtest_add_func("g233/pwm/multi_channel", test_pwm_multi_channel);
    qtest_add_func("g233/pwm/polarity", test_pwm_polarity);

    return g_test_run();
}
