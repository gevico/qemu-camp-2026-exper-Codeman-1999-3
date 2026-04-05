/*
 * QTest: G233 Rust I2C GPIO controller — reset values and prescaler config
 *
 * Copyright (c) 2026 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * I2C GPIO register map (base 0x10013000):
 *   0x00  I2C_CTRL     — bit0: EN, bit1: START, bit2: STOP, bit3: RW
 *   0x04  I2C_STATUS   — bit0: BUSY, bit1: ACK, bit2: DONE
 *   0x08  I2C_ADDR     — 7-bit slave address
 *   0x0C  I2C_DATA     — data register
 *   0x10  I2C_PRESCALE — clock prescaler
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define I2C_BASE        0x10013000ULL

#define I2C_CTRL        (I2C_BASE + 0x00)
#define I2C_STATUS      (I2C_BASE + 0x04)
#define I2C_ADDR        (I2C_BASE + 0x08)
#define I2C_DATA        (I2C_BASE + 0x0C)
#define I2C_PRESCALE    (I2C_BASE + 0x10)

/* All registers should read zero after reset */
static void test_i2c_reset_values(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    g_assert_cmpuint(qtest_readl(qts, I2C_CTRL),     ==, 0);
    g_assert_cmpuint(qtest_readl(qts, I2C_STATUS),   ==, 0);
    g_assert_cmpuint(qtest_readl(qts, I2C_ADDR),     ==, 0);
    g_assert_cmpuint(qtest_readl(qts, I2C_DATA),     ==, 0);
    g_assert_cmpuint(qtest_readl(qts, I2C_PRESCALE), ==, 0);

    qtest_quit(qts);
}

/* Write prescaler value and verify read-back */
static void test_i2c_prescale_config(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* Set prescaler to divide-by-100 */
    qtest_writel(qts, I2C_PRESCALE, 100);
    g_assert_cmpuint(qtest_readl(qts, I2C_PRESCALE), ==, 100);

    /* Change to a different value */
    qtest_writel(qts, I2C_PRESCALE, 0xFF);
    g_assert_cmpuint(qtest_readl(qts, I2C_PRESCALE), ==, 0xFF);

    /* Zero it out */
    qtest_writel(qts, I2C_PRESCALE, 0);
    g_assert_cmpuint(qtest_readl(qts, I2C_PRESCALE), ==, 0);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/rust-i2c/reset_values", test_i2c_reset_values);
    qtest_add_func("g233/rust-i2c/prescale_config", test_i2c_prescale_config);

    return g_test_run();
}
