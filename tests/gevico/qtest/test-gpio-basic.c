/*
 * QTest: G233 GPIO controller — basic functionality
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * GPIO register map (base 0x10012000):
 *   0x00  GPIO_DIR   — direction (0=input, 1=output)
 *   0x04  GPIO_OUT   — output data
 *   0x08  GPIO_IN    — input data (read-only; reflects OUT when DIR=output)
 *   0x0C  GPIO_IE    — interrupt enable
 *   0x10  GPIO_IS    — interrupt status (write-1-to-clear)
 *   0x14  GPIO_TRIG  — trigger type (0=edge, 1=level)
 *   0x18  GPIO_POL   — polarity (0=low/falling, 1=high/rising)
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

static void test_gpio_reset_value(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    g_assert_cmpuint(qtest_readl(qts, GPIO_DIR),  ==, 0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_OUT),  ==, 0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IN),   ==, 0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IE),   ==, 0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IS),   ==, 0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_TRIG), ==, 0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_POL),  ==, 0);

    qtest_quit(qts);
}

static void test_gpio_direction(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* Set pin 0 to output */
    qtest_writel(qts, GPIO_DIR, 0x1);
    g_assert_cmpuint(qtest_readl(qts, GPIO_DIR), ==, 0x1);

    /* Set all 32 pins to output */
    qtest_writel(qts, GPIO_DIR, 0xFFFFFFFF);
    g_assert_cmpuint(qtest_readl(qts, GPIO_DIR), ==, 0xFFFFFFFF);

    qtest_quit(qts);
}

static void test_gpio_output(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* Configure pin 0 as output */
    qtest_writel(qts, GPIO_DIR, 0x1);

    /* Drive high */
    qtest_writel(qts, GPIO_OUT, 0x1);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IN) & 0x1, ==, 0x1);

    /* Drive low */
    qtest_writel(qts, GPIO_OUT, 0x0);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IN) & 0x1, ==, 0x0);

    qtest_quit(qts);
}

static void test_gpio_multi_pin(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    const uint32_t pins = (1u << 0) | (1u << 7) | (1u << 15) | (1u << 31);

    qtest_writel(qts, GPIO_DIR, pins);
    g_assert_cmpuint(qtest_readl(qts, GPIO_DIR), ==, pins);

    qtest_writel(qts, GPIO_OUT, pins);
    g_assert_cmpuint(qtest_readl(qts, GPIO_IN) & pins, ==, pins);

    /* Clear only bit 7 */
    qtest_writel(qts, GPIO_OUT, pins & ~(1u << 7));
    uint32_t in_val = qtest_readl(qts, GPIO_IN);
    g_assert_cmpuint(in_val & (1u << 7), ==, 0);
    g_assert_cmpuint(in_val & (1u << 0), ==, (1u << 0));
    g_assert_cmpuint(in_val & (1u << 15), ==, (1u << 15));
    g_assert_cmpuint(in_val & (1u << 31), ==, (1u << 31));

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/gpio/reset_value", test_gpio_reset_value);
    qtest_add_func("g233/gpio/direction", test_gpio_direction);
    qtest_add_func("g233/gpio/output", test_gpio_output);
    qtest_add_func("g233/gpio/multi_pin", test_gpio_multi_pin);

    return g_test_run();
}
