/*
 * QTest: G233 Rust SPI — AT25 SPI EEPROM read/write via Rust SPI controller
 *
 * Copyright (c) 2026 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * AT25 SPI EEPROM on CS0: 256 bytes.
 * Commands:
 *   0x06 — WREN  (write enable)
 *   0x05 — RDSR  (read status register)
 *   0x03 — READ  (read data)
 *   0x02 — WRITE (page program)
 *
 * Status register: bit0=WIP (write-in-progress), bit1=WEL (write-enable-latch)
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define RSPI_BASE       0x10019000ULL

#define RSPI_CR1        (RSPI_BASE + 0x00)
#define RSPI_SR         (RSPI_BASE + 0x04)
#define RSPI_DR         (RSPI_BASE + 0x08)
#define RSPI_CS         (RSPI_BASE + 0x0C)

#define RSPI_CR1_SPE    (1u << 0)
#define RSPI_CR1_MSTR   (1u << 2)

#define RSPI_SR_RXNE    (1u << 0)
#define RSPI_SR_TXE     (1u << 1)

#define AT25_CMD_WREN   0x06
#define AT25_CMD_RDSR   0x05
#define AT25_CMD_READ   0x03
#define AT25_CMD_WRITE  0x02

#define AT25_SR_WIP     (1u << 0)
#define AT25_SR_WEL     (1u << 1)

static void rspi_wait_txe(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, RSPI_SR) & RSPI_SR_TXE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
    g_assert_cmpint(timeout, >, 0);
}

static void rspi_wait_rxne(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, RSPI_SR) & RSPI_SR_RXNE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
    g_assert_cmpint(timeout, >, 0);
}

static uint8_t rspi_xfer(QTestState *qts, uint8_t tx)
{
    rspi_wait_txe(qts);
    qtest_writel(qts, RSPI_DR, tx);
    rspi_wait_rxne(qts);
    return (uint8_t)qtest_readl(qts, RSPI_DR);
}

static void rspi_init(QTestState *qts)
{
    qtest_writel(qts, RSPI_CS, 0);  /* AT25 on CS0 */
    qtest_writel(qts, RSPI_CR1, RSPI_CR1_SPE | RSPI_CR1_MSTR);
}

static void at25_wait_ready(QTestState *qts)
{
    int timeout = 10000;
    uint8_t sr;
    do {
        rspi_xfer(qts, AT25_CMD_RDSR);
        sr = rspi_xfer(qts, 0x00);
        qtest_clock_step(qts, 100000);
    } while ((sr & AT25_SR_WIP) && --timeout);
    g_assert_cmpint(timeout, >, 0);
}

/* Read status register: should be idle after reset */
static void test_rspi_flash_status(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    rspi_init(qts);

    /* Read status register */
    rspi_xfer(qts, AT25_CMD_RDSR);
    uint8_t sr = rspi_xfer(qts, 0x00);

    /* Should not be busy or write-enabled at reset */
    g_assert_cmpuint(sr & AT25_SR_WIP, ==, 0);
    g_assert_cmpuint(sr & AT25_SR_WEL, ==, 0);

    /* Enable write and check WEL */
    rspi_xfer(qts, AT25_CMD_WREN);

    rspi_xfer(qts, AT25_CMD_RDSR);
    sr = rspi_xfer(qts, 0x00);
    g_assert_cmpuint(sr & AT25_SR_WEL, !=, 0);

    qtest_quit(qts);
}

/* Write data to AT25 and read it back */
static void test_rspi_flash_write_read(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    rspi_init(qts);

    /* Write enable */
    rspi_xfer(qts, AT25_CMD_WREN);

    /* Write 4 bytes at address 0x00 */
    rspi_xfer(qts, AT25_CMD_WRITE);
    rspi_xfer(qts, 0x00);  /* address */
    rspi_xfer(qts, 0xDE);
    rspi_xfer(qts, 0xAD);
    rspi_xfer(qts, 0xBE);
    rspi_xfer(qts, 0xEF);

    /* Wait for write to complete */
    at25_wait_ready(qts);

    /* Read back */
    rspi_xfer(qts, AT25_CMD_READ);
    rspi_xfer(qts, 0x00);  /* address */
    uint8_t d0 = rspi_xfer(qts, 0x00);
    uint8_t d1 = rspi_xfer(qts, 0x00);
    uint8_t d2 = rspi_xfer(qts, 0x00);
    uint8_t d3 = rspi_xfer(qts, 0x00);

    g_assert_cmpuint(d0, ==, 0xDE);
    g_assert_cmpuint(d1, ==, 0xAD);
    g_assert_cmpuint(d2, ==, 0xBE);
    g_assert_cmpuint(d3, ==, 0xEF);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/rust-spi/flash_status", test_rspi_flash_status);
    qtest_add_func("g233/rust-spi/flash_write_read",
                   test_rspi_flash_write_read);

    return g_test_run();
}
