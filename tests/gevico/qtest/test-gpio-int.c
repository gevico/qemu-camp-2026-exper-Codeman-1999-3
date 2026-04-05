/*
 * QTest: G233 GPIO controller — interrupt functionality
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * GPIO register map: see test-gpio-basic.c
 * GPIO PLIC IRQ: 2
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define GPIO_BASE   0x10012000ULL
#define GPIO_DIR    (GPIO_BASE + 0x00)
#define GPIO_OUT    (GPIO_BASE + 0x04)
#define GPIO_IN     (GPIO_BASE + 0x08)
#define GPIO_IE     (GPIO_BASE + 0x0C)
#define GPIO_IS     (GPIO_BASE + 0x10)
#define GPIO_TRIG   (GPIO_BASE + 0x14)
#define GPIO_POL    (GPIO_BASE + 0x18)

#define PLIC_BASE       0x0C000000ULL
#define PLIC_PENDING    (PLIC_BASE + 0x1000)

#define GPIO_PLIC_IRQ   2

static inline bool plic_irq_pending(QTestState *qts, int irq)
{
    uint32_t word = qtest_readl(qts, PLIC_PENDING + (irq / 32) * 4);
    return (word >> (irq % 32)) & 1;
}

/* Edge-triggered, rising polarity: IS set on 0→1 transition */
static void test_gpio_edge_rising(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* Pin 0: output, edge-triggered, rising polarity, interrupt enabled */
    qtest_writel(qts, GPIO_DIR,  0x1);
    qtest_writel(qts, GPIO_TRIG, 0x0);  /* edge */
    qtest_writel(qts, GPIO_POL,  0x1);  /* rising */
    qtest_writel(qts, GPIO_IE,   0x1);

    /* Start low */
    qtest_writel(qts, GPIO_OUT, 0x0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS) & 0x1, ==, 0);

    /* 0→1 transition → interrupt */
    qtest_writel(qts, GPIO_OUT, 0x1);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS) & 0x1, ==, 0x1);

    qtest_quit(qts);
}

/* Level-triggered, high polarity: IS stays set while pin is high */
static void test_gpio_level_high(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, GPIO_DIR,  0x1);
    qtest_writel(qts, GPIO_TRIG, 0x1);  /* level */
    qtest_writel(qts, GPIO_POL,  0x1);  /* high */
    qtest_writel(qts, GPIO_IE,   0x1);

    /* Drive high → interrupt active */
    qtest_writel(qts, GPIO_OUT, 0x1);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS) & 0x1, ==, 0x1);

    /* Drive low → interrupt clears */
    qtest_writel(qts, GPIO_OUT, 0x0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS) & 0x1, ==, 0);

    qtest_quit(qts);
}

/* Write-1-to-clear on GPIO_IS (edge mode) */
static void test_gpio_is_clear(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, GPIO_DIR,  0x1);
    qtest_writel(qts, GPIO_TRIG, 0x0);  /* edge */
    qtest_writel(qts, GPIO_POL,  0x1);  /* rising */
    qtest_writel(qts, GPIO_IE,   0x1);

    qtest_writel(qts, GPIO_OUT, 0x0);
    qtest_writel(qts, GPIO_OUT, 0x1);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS) & 0x1, ==, 0x1);

    /* Write 1 to clear */
    qtest_writel(qts, GPIO_IS, 0x1);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS) & 0x1, ==, 0);

    qtest_quit(qts);
}

/* IE mask: disabled interrupt should not set IS */
static void test_gpio_ie_mask(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, GPIO_DIR,  0x1);
    qtest_writel(qts, GPIO_TRIG, 0x0);
    qtest_writel(qts, GPIO_POL,  0x1);
    qtest_writel(qts, GPIO_IE,   0x0);  /* disabled */

    qtest_writel(qts, GPIO_OUT, 0x0);
    qtest_writel(qts, GPIO_OUT, 0x1);

    /* IS should NOT be set when IE is disabled */
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS) & 0x1, ==, 0);

    qtest_quit(qts);
}

/* GPIO interrupt should appear on PLIC IRQ 2 */
static void test_gpio_plic(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, GPIO_DIR,  0x1);
    qtest_writel(qts, GPIO_TRIG, 0x0);
    qtest_writel(qts, GPIO_POL,  0x1);
    qtest_writel(qts, GPIO_IE,   0x1);

    qtest_writel(qts, GPIO_OUT, 0x0);
    qtest_writel(qts, GPIO_OUT, 0x1);

    g_assert_true(plic_irq_pending(qts, GPIO_PLIC_IRQ));

    /* Clear and verify PLIC pending clears */
    qtest_writel(qts, GPIO_IS, 0x1);
    g_assert_false(plic_irq_pending(qts, GPIO_PLIC_IRQ));

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/gpio-int/edge_rising", test_gpio_edge_rising);
    qtest_add_func("g233/gpio-int/level_high", test_gpio_level_high);
    qtest_add_func("g233/gpio-int/is_clear", test_gpio_is_clear);
    qtest_add_func("g233/gpio-int/ie_mask", test_gpio_ie_mask);
    qtest_add_func("g233/gpio-int/plic", test_gpio_plic);

    return g_test_run();
}
