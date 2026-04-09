/*
 * QTest: G233 SPI Flash — read/write (polling mode)
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Tests full Flash lifecycle: erase → program → read → verify
 * Flash CS0: W25X16 (2MB)
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define SPI_BASE    0x10018000ULL
#define SPI_CR1     (SPI_BASE + 0x00)
#define SPI_CR2     (SPI_BASE + 0x04)
#define SPI_SR      (SPI_BASE + 0x08)
#define SPI_DR      (SPI_BASE + 0x0C)

#define SPI_CR1_SPE     (1u << 0)
#define SPI_CR1_MSTR    (1u << 2)
#define SPI_SR_RXNE     (1u << 0)
#define SPI_SR_TXE      (1u << 1)

#define FLASH_CMD_WRITE_ENABLE   0x06
#define FLASH_CMD_READ_STATUS    0x05
#define FLASH_CMD_READ_DATA      0x03
#define FLASH_CMD_PAGE_PROGRAM   0x02
#define FLASH_CMD_SECTOR_ERASE   0x20
#define FLASH_CMD_JEDEC_ID       0x9F

#define FLASH_SR_BUSY            0x01

#define PAGE_SIZE   256
#define SECTOR_SIZE 4096

static void spi_wait_txe(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, SPI_SR) & SPI_SR_TXE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
    g_assert_cmpint(timeout, >, 0);
}

static void spi_wait_rxne(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, SPI_SR) & SPI_SR_RXNE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
    g_assert_cmpint(timeout, >, 0);
}

static uint8_t spi_xfer(QTestState *qts, uint8_t tx)
{
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, tx);
    spi_wait_rxne(qts);
    return (uint8_t)qtest_readl(qts, SPI_DR);
}

static void spi_init(QTestState *qts)
{
    qtest_writel(qts, SPI_CR2, 0);  /* CS0 */
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);
}

static uint8_t flash_read_status(QTestState *qts)
{
    qtest_writel(qts, SPI_CR2, 0);
    spi_xfer(qts, FLASH_CMD_READ_STATUS);
    uint8_t sr = spi_xfer(qts, 0x00);
    qtest_writel(qts, SPI_CR2, 1);
    return sr;
}

static void flash_wait_busy(QTestState *qts)
{
    int timeout = 10000;
    while ((flash_read_status(qts) & FLASH_SR_BUSY) && --timeout) {
        qtest_clock_step(qts, 100000);
    }
    g_assert_cmpint(timeout, >, 0);
}

static void flash_write_enable(QTestState *qts)
{
    qtest_writel(qts, SPI_CR2, 0);
    spi_xfer(qts, FLASH_CMD_WRITE_ENABLE);
    qtest_writel(qts, SPI_CR2, 1);
}

static void flash_sector_erase(QTestState *qts, uint32_t addr)
{
    flash_write_enable(qts);
    qtest_writel(qts, SPI_CR2, 0);
    spi_xfer(qts, FLASH_CMD_SECTOR_ERASE);
    spi_xfer(qts, (addr >> 16) & 0xFF);
    spi_xfer(qts, (addr >> 8) & 0xFF);
    spi_xfer(qts, addr & 0xFF);
    qtest_writel(qts, SPI_CR2, 1);
    flash_wait_busy(qts);
}

static void flash_page_program(QTestState *qts, uint32_t addr,
                               const uint8_t *data, int len)
{
    flash_write_enable(qts);
    qtest_writel(qts, SPI_CR2, 0);
    spi_xfer(qts, FLASH_CMD_PAGE_PROGRAM);
    spi_xfer(qts, (addr >> 16) & 0xFF);
    spi_xfer(qts, (addr >> 8) & 0xFF);
    spi_xfer(qts, addr & 0xFF);

    for (int i = 0; i < len; i++) {
        spi_xfer(qts, data[i]);
    }
    qtest_writel(qts, SPI_CR2, 1);
    flash_wait_busy(qts);
}

static void flash_read_data(QTestState *qts, uint32_t addr,
                            uint8_t *buf, int len)
{
    qtest_writel(qts, SPI_CR2, 0);
    spi_xfer(qts, FLASH_CMD_READ_DATA);
    spi_xfer(qts, (addr >> 16) & 0xFF);
    spi_xfer(qts, (addr >> 8) & 0xFF);
    spi_xfer(qts, addr & 0xFF);

    for (int i = 0; i < len; i++) {
        buf[i] = spi_xfer(qts, 0x00);
    }
    qtest_writel(qts, SPI_CR2, 1);
}

/* Test: read status register */
static void test_flash_read_status(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    spi_init(qts);

    uint8_t sr = flash_read_status(qts);
    /* Should not be busy on idle */
    g_assert_cmpuint(sr & FLASH_SR_BUSY, ==, 0);

    qtest_quit(qts);
}

/* Test: sector erase clears data to 0xFF */
static void test_flash_sector_erase(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t buf[16];
    spi_init(qts);

    flash_sector_erase(qts, 0x000000);
    flash_read_data(qts, 0x000000, buf, sizeof(buf));

    for (int i = 0; i < (int)sizeof(buf); i++) {
        g_assert_cmpuint(buf[i], ==, 0xFF);
    }

    qtest_quit(qts);
}

/* Test: page program + read back */
static void test_flash_page_program(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t wbuf[PAGE_SIZE], rbuf[PAGE_SIZE];
    spi_init(qts);

    for (int i = 0; i < PAGE_SIZE; i++) {
        wbuf[i] = (uint8_t)(i & 0xFF);
    }

    flash_sector_erase(qts, 0x000000);
    flash_page_program(qts, 0x000000, wbuf, PAGE_SIZE);
    flash_read_data(qts, 0x000000, rbuf, PAGE_SIZE);

    g_assert_cmpmem(wbuf, PAGE_SIZE, rbuf, PAGE_SIZE);

    qtest_quit(qts);
}

/* Test: full erase → program → read cycle */
static void test_flash_write_test_data(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t wbuf[64], rbuf[64];
    spi_init(qts);

    for (int i = 0; i < 64; i++) {
        wbuf[i] = (uint8_t)(0xA5 ^ i);
    }

    flash_sector_erase(qts, 0x001000);
    flash_page_program(qts, 0x001000, wbuf, 64);
    flash_read_data(qts, 0x001000, rbuf, 64);

    g_assert_cmpmem(wbuf, 64, rbuf, 64);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/flash/read_status", test_flash_read_status);
    qtest_add_func("g233/flash/sector_erase", test_flash_sector_erase);
    qtest_add_func("g233/flash/page_program", test_flash_page_program);
    qtest_add_func("g233/flash/write_test_data", test_flash_write_test_data);

    return g_test_run();
}
