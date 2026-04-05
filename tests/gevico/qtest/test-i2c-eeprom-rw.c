/*
 * QTest: G233 Rust I2C — AT24C02 EEPROM read/write via I2C controller
 *
 * Copyright (c) 2026 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * AT24C02: 256-byte I2C EEPROM at slave address 0x50.
 * Write: START → addr(W) → mem_addr → data... → STOP
 * Read:  START → addr(W) → mem_addr → rSTART → addr(R) → data... → STOP
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
#define I2C_CTRL_RW     (1u << 3)

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

static void i2c_init(QTestState *qts)
{
    qtest_writel(qts, I2C_PRESCALE, 10);
    qtest_writel(qts, I2C_ADDR, AT24C02_ADDR);
}

static void eeprom_write_byte(QTestState *qts, uint8_t mem_addr, uint8_t val)
{
    /* START + write */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START);
    i2c_wait_done(qts);

    /* Memory address */
    qtest_writel(qts, I2C_DATA, mem_addr);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);

    /* Data byte */
    qtest_writel(qts, I2C_DATA, val);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);

    /* STOP */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_STOP);
    i2c_wait_done(qts);

    /* Write cycle time */
    qtest_clock_step(qts, 5000000);
}

static uint8_t eeprom_read_byte(QTestState *qts, uint8_t mem_addr)
{
    /* START + write to set address */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START);
    i2c_wait_done(qts);

    qtest_writel(qts, I2C_DATA, mem_addr);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);

    /* Repeated START + read */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START | I2C_CTRL_RW);
    i2c_wait_done(qts);

    /* Clock in data */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_RW);
    i2c_wait_done(qts);

    uint8_t val = (uint8_t)qtest_readl(qts, I2C_DATA);

    /* STOP */
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_STOP);
    i2c_wait_done(qts);

    return val;
}

/* Write a byte and read it back */
static void test_eeprom_write_read(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    i2c_init(qts);

    eeprom_write_byte(qts, 0x00, 0xAB);
    g_assert_cmpuint(eeprom_read_byte(qts, 0x00), ==, 0xAB);

    eeprom_write_byte(qts, 0x7F, 0xCD);
    g_assert_cmpuint(eeprom_read_byte(qts, 0x7F), ==, 0xCD);

    /* Verify first byte is still intact */
    g_assert_cmpuint(eeprom_read_byte(qts, 0x00), ==, 0xAB);

    qtest_quit(qts);
}

/* Sequential read: read multiple consecutive addresses */
static void test_eeprom_sequential_read(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    const uint8_t pattern[] = {0x10, 0x20, 0x30, 0x40};
    i2c_init(qts);

    /* Write 4 bytes starting at address 0x10 */
    for (int i = 0; i < 4; i++) {
        eeprom_write_byte(qts, 0x10 + i, pattern[i]);
    }

    /* Read them back and verify */
    for (int i = 0; i < 4; i++) {
        g_assert_cmpuint(eeprom_read_byte(qts, 0x10 + i), ==, pattern[i]);
    }

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/rust-i2c/eeprom_write_read",
                   test_eeprom_write_read);
    qtest_add_func("g233/rust-i2c/eeprom_sequential_read",
                   test_eeprom_sequential_read);

    return g_test_run();
}
