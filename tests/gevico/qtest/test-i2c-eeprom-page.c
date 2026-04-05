/*
 * QTest: G233 Rust I2C — AT24C02 EEPROM page write boundary
 *
 * Copyright (c) 2026 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * AT24C02 has 8-byte page write granularity.  Writes that cross an
 * 8-byte boundary wrap within the current page.
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

#define I2C_ST_DONE     (1u << 2)

#define AT24C02_ADDR    0x50
#define PAGE_SIZE       8

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

/* Page write: START → addr(W) → mem_addr → data[0..n-1] → STOP */
static void eeprom_page_write(QTestState *qts, uint8_t mem_addr,
                              const uint8_t *buf, int len)
{
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START);
    i2c_wait_done(qts);

    qtest_writel(qts, I2C_DATA, mem_addr);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);

    for (int i = 0; i < len; i++) {
        qtest_writel(qts, I2C_DATA, buf[i]);
        qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
        i2c_wait_done(qts);
    }

    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_STOP);
    i2c_wait_done(qts);

    /* Write cycle time */
    qtest_clock_step(qts, 5000000);
}

static uint8_t eeprom_read_byte(QTestState *qts, uint8_t mem_addr)
{
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START);
    i2c_wait_done(qts);

    qtest_writel(qts, I2C_DATA, mem_addr);
    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN);
    i2c_wait_done(qts);

    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_START | I2C_CTRL_RW);
    i2c_wait_done(qts);

    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_RW);
    i2c_wait_done(qts);

    uint8_t val = (uint8_t)qtest_readl(qts, I2C_DATA);

    qtest_writel(qts, I2C_CTRL, I2C_CTRL_EN | I2C_CTRL_STOP);
    i2c_wait_done(qts);

    return val;
}

/* Write a full 8-byte page and read back */
static void test_eeprom_page_write(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    const uint8_t data[PAGE_SIZE] = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08
    };
    i2c_init(qts);

    /* Write page at aligned address 0x00 */
    eeprom_page_write(qts, 0x00, data, PAGE_SIZE);

    for (int i = 0; i < PAGE_SIZE; i++) {
        g_assert_cmpuint(eeprom_read_byte(qts, i), ==, data[i]);
    }

    qtest_quit(qts);
}

/*
 * Page boundary wrap test:
 * Start writing at offset 0x06 within an 8-byte page (page 0: 0x00-0x07).
 * Writing 4 bytes should wrap: bytes land at 0x06, 0x07, 0x00, 0x01.
 */
static void test_eeprom_page_boundary(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    const uint8_t fill = 0xFF;
    const uint8_t data[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    i2c_init(qts);

    /* Pre-fill page 0 with 0xFF */
    uint8_t page_ff[PAGE_SIZE];
    for (int i = 0; i < PAGE_SIZE; i++) {
        page_ff[i] = fill;
    }
    eeprom_page_write(qts, 0x00, page_ff, PAGE_SIZE);

    /* Write 4 bytes starting at offset 6 — should wrap within page */
    eeprom_page_write(qts, 0x06, data, 4);

    /* Expected: 0x00=0xCC, 0x01=0xDD, 0x02-0x05=0xFF, 0x06=0xAA, 0x07=0xBB */
    g_assert_cmpuint(eeprom_read_byte(qts, 0x06), ==, 0xAA);
    g_assert_cmpuint(eeprom_read_byte(qts, 0x07), ==, 0xBB);
    g_assert_cmpuint(eeprom_read_byte(qts, 0x00), ==, 0xCC);
    g_assert_cmpuint(eeprom_read_byte(qts, 0x01), ==, 0xDD);

    /* Untouched bytes in the page should remain 0xFF */
    g_assert_cmpuint(eeprom_read_byte(qts, 0x02), ==, 0xFF);
    g_assert_cmpuint(eeprom_read_byte(qts, 0x05), ==, 0xFF);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/rust-i2c/eeprom_page_write",
                   test_eeprom_page_write);
    qtest_add_func("g233/rust-i2c/eeprom_page_boundary",
                   test_eeprom_page_boundary);

    return g_test_run();
}
