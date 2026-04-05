/*
 * QTest: G233 Rust I2C GPIO controller — START/STOP/ACK bit-bang protocol
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
 *
 * AT24C02 EEPROM at I2C address 0x50
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define I2C_BASE        0x10013000ULL

#define I2C_CTRL        (I2C_BASE + 0x00)
#define I2C_STATUS      (I2C_BASE + 0x04)
#define I2C_ADDR        (I2C_BASE + 0x08)
#define I2C_DATA        (I2C_BASE + 0x0C)
#define I2C_PRESCALE    (I2C_BASE + 0x10)

#define I2C_CTRL_EN     (1u << 0)
#define I2C_CTRL_START  (1u << 1)
#define I2C_CTRL_STOP   (1u << 2)
#define I2C_CTRL_RW     (1u << 3)   /* 0=write, 1=read */

#define I2C_ST_BUSY     (1u << 0)
#define I2C_ST_ACK      (1u << 1)
#define I2C_ST_DONE     (1u << 2)

#define AT24C02_ADDR    0x50

static void i2c_wait_done(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, I2C_STATUS) & I2C_ST_DONE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
    g_assert_cmpint(timeout, >, 0);
}

/* Issue START, address the EEPROM for write, verify ACK, then STOP */
static void test_i2c_start_stop(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, I2C_PRESCALE, 10);
    qtest_writel(qts, I2C_ADDR, AT24C02_ADDR);

    /* EN + START + write mode */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START);
    i2c_wait_done(qts);

    /* Slave should ACK */
    g_assert_cmpuint(qtest_readl(qts, I2C_STATUS) & I2C_ST_ACK, !=, 0);

    /* Issue STOP */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_STOP);
    i2c_wait_done(qts);

    /* Bus should be idle after STOP */
    g_assert_cmpuint(qtest_readl(qts, I2C_STATUS) & I2C_ST_BUSY, ==, 0);

    qtest_quit(qts);
}

/* Write a single byte to EEPROM address 0x00 and verify ACK */
static void test_i2c_write_byte(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, I2C_PRESCALE, 10);
    qtest_writel(qts, I2C_ADDR, AT24C02_ADDR);

    /* START + write */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START);
    i2c_wait_done(qts);
    g_assert_cmpuint(qtest_readl(qts, I2C_STATUS) & I2C_ST_ACK, !=, 0);

    /* Send memory address byte (0x00) */
    qtest_writel(qts, I2C_DATA, 0x00);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);
    g_assert_cmpuint(qtest_readl(qts, I2C_STATUS) & I2C_ST_ACK, !=, 0);

    /* Send data byte */
    qtest_writel(qts, I2C_DATA, 0xA5);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);
    g_assert_cmpuint(qtest_readl(qts, I2C_STATUS) & I2C_ST_ACK, !=, 0);

    /* STOP */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_STOP);
    i2c_wait_done(qts);

    qtest_quit(qts);
}

/* Read a byte back from EEPROM using repeated START */
static void test_i2c_read_byte(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, I2C_PRESCALE, 10);
    qtest_writel(qts, I2C_ADDR, AT24C02_ADDR);

    /* Write phase: set EEPROM internal address to 0x00 */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START);
    i2c_wait_done(qts);

    qtest_writel(qts, I2C_DATA, 0x00);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);

    /* Repeated START for read */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START | I2C_CTRL_RW);
    i2c_wait_done(qts);
    g_assert_cmpuint(qtest_readl(qts, I2C_STATUS) & I2C_ST_ACK, !=, 0);

    /* Clock in one data byte */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_RW);
    i2c_wait_done(qts);

    /* Read the received byte */
    uint32_t data = qtest_readl(qts, I2C_DATA);
    g_assert_cmpuint(data, <, 256);  /* valid byte range */

    /* STOP */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_STOP);
    i2c_wait_done(qts);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/rust-i2c/start_stop", test_i2c_start_stop);
    qtest_add_func("g233/rust-i2c/write_byte", test_i2c_write_byte);
    qtest_add_func("g233/rust-i2c/read_byte", test_i2c_read_byte);

    return g_test_run();
}
