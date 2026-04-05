/*
 * QTest: G233 SPI — dual Flash chip-select control
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * CS0: W25X16 (2MB), JEDEC = {0xEF, 0x30, 0x15}
 * CS1: W25X32 (4MB), JEDEC = {0xEF, 0x30, 0x16}
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

#define FLASH_CMD_WRITE_ENABLE  0x06
#define FLASH_CMD_READ_STATUS   0x05
#define FLASH_CMD_READ_DATA     0x03
#define FLASH_CMD_PAGE_PROGRAM  0x02
#define FLASH_CMD_SECTOR_ERASE  0x20
#define FLASH_CMD_JEDEC_ID      0x9F

#define FLASH_SR_BUSY   0x01

/* W25X16 capacity */
#define CS0_SIZE    (2 * 1024 * 1024)
/* W25X32 capacity */
#define CS1_SIZE    (4 * 1024 * 1024)

static void spi_wait_txe(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, SPI_SR) & SPI_SR_TXE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
}

static void spi_wait_rxne(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, SPI_SR) & SPI_SR_RXNE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
}

static uint8_t spi_xfer(QTestState *qts, uint8_t tx)
{
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, tx);
    spi_wait_rxne(qts);
    return (uint8_t)qtest_readl(qts, SPI_DR);
}

static void spi_select(QTestState *qts, int cs)
{
    qtest_writel(qts, SPI_CR2, cs & 0x3);
}

static void flash_wait_busy(QTestState *qts)
{
    int timeout = 10000;
    uint8_t sr;
    do {
        spi_xfer(qts, FLASH_CMD_READ_STATUS);
        sr = spi_xfer(qts, 0x00);
        qtest_clock_step(qts, 100000);
    } while ((sr & FLASH_SR_BUSY) && --timeout);
}

static uint32_t flash_read_jedec(QTestState *qts)
{
    spi_xfer(qts, FLASH_CMD_JEDEC_ID);
    uint8_t mfr = spi_xfer(qts, 0);
    uint8_t id1 = spi_xfer(qts, 0);
    uint8_t id2 = spi_xfer(qts, 0);
    return (mfr << 16) | (id1 << 8) | id2;
}

static void flash_erase_program_read(QTestState *qts, uint32_t addr,
                                     const uint8_t *wbuf, uint8_t *rbuf,
                                     int len)
{
    /* Erase */
    spi_xfer(qts, FLASH_CMD_WRITE_ENABLE);
    spi_xfer(qts, FLASH_CMD_SECTOR_ERASE);
    spi_xfer(qts, (addr >> 16) & 0xFF);
    spi_xfer(qts, (addr >> 8) & 0xFF);
    spi_xfer(qts, addr & 0xFF);
    flash_wait_busy(qts);

    /* Program */
    spi_xfer(qts, FLASH_CMD_WRITE_ENABLE);
    spi_xfer(qts, FLASH_CMD_PAGE_PROGRAM);
    spi_xfer(qts, (addr >> 16) & 0xFF);
    spi_xfer(qts, (addr >> 8) & 0xFF);
    spi_xfer(qts, addr & 0xFF);
    for (int i = 0; i < len; i++) {
        spi_xfer(qts, wbuf[i]);
    }
    flash_wait_busy(qts);

    /* Read */
    spi_xfer(qts, FLASH_CMD_READ_DATA);
    spi_xfer(qts, (addr >> 16) & 0xFF);
    spi_xfer(qts, (addr >> 8) & 0xFF);
    spi_xfer(qts, addr & 0xFF);
    for (int i = 0; i < len; i++) {
        rbuf[i] = spi_xfer(qts, 0x00);
    }
}

/* Identify both Flash chips */
static void test_flash_identification(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    spi_select(qts, 0);
    g_assert_cmphex(flash_read_jedec(qts), ==, 0xEF3015);

    spi_select(qts, 1);
    g_assert_cmphex(flash_read_jedec(qts), ==, 0xEF3016);

    qtest_quit(qts);
}

/* Write/read each Flash independently */
static void test_individual_flash_operations(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t wbuf[16], rbuf[16];
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    /* CS0: write pattern 0xAA */
    for (int i = 0; i < 16; i++) wbuf[i] = 0xAA;
    spi_select(qts, 0);
    flash_erase_program_read(qts, 0x000000, wbuf, rbuf, 16);
    g_assert_cmpmem(wbuf, 16, rbuf, 16);

    /* CS1: write pattern 0x55 */
    for (int i = 0; i < 16; i++) wbuf[i] = 0x55;
    spi_select(qts, 1);
    flash_erase_program_read(qts, 0x000000, wbuf, rbuf, 16);
    g_assert_cmpmem(wbuf, 16, rbuf, 16);

    qtest_quit(qts);
}

/* Cross-flash: write different patterns and verify isolation */
static void test_cross_flash_operations(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t w0[16], w1[16], r0[16], r1[16];
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    for (int i = 0; i < 16; i++) {
        w0[i] = 0xAA;
        w1[i] = 0x55;
    }

    /* Write to both at same address */
    spi_select(qts, 0);
    flash_erase_program_read(qts, 0x002000, w0, r0, 16);
    spi_select(qts, 1);
    flash_erase_program_read(qts, 0x002000, w1, r1, 16);

    /* Re-read CS0 to make sure it wasn't corrupted */
    spi_select(qts, 0);
    spi_xfer(qts, FLASH_CMD_READ_DATA);
    spi_xfer(qts, 0x00); spi_xfer(qts, 0x20); spi_xfer(qts, 0x00);
    for (int i = 0; i < 16; i++) {
        r0[i] = spi_xfer(qts, 0x00);
    }

    g_assert_cmpmem(w0, 16, r0, 16);
    g_assert_cmpmem(w1, 16, r1, 16);

    qtest_quit(qts);
}

/* Alternate read between two chips */
static void test_alternating_operations(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t w0[4] = {0x11, 0x22, 0x33, 0x44};
    uint8_t w1[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    uint8_t r[4];
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    spi_select(qts, 0);
    flash_erase_program_read(qts, 0x003000, w0, r, 4);

    spi_select(qts, 1);
    flash_erase_program_read(qts, 0x003000, w1, r, 4);

    /* Alternate reads */
    for (int round = 0; round < 3; round++) {
        spi_select(qts, 0);
        spi_xfer(qts, FLASH_CMD_READ_DATA);
        spi_xfer(qts, 0x00); spi_xfer(qts, 0x30); spi_xfer(qts, 0x00);
        for (int i = 0; i < 4; i++) r[i] = spi_xfer(qts, 0);
        g_assert_cmpmem(w0, 4, r, 4);

        spi_select(qts, 1);
        spi_xfer(qts, FLASH_CMD_READ_DATA);
        spi_xfer(qts, 0x00); spi_xfer(qts, 0x30); spi_xfer(qts, 0x00);
        for (int i = 0; i < 4; i++) r[i] = spi_xfer(qts, 0);
        g_assert_cmpmem(w1, 4, r, 4);
    }

    qtest_quit(qts);
}

/* Verify capacity boundaries */
static void test_flash_capacity(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t wbuf[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    uint8_t rbuf[4];
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    /* CS0: write near 2MB boundary */
    uint32_t cs0_addr = CS0_SIZE - 4096;
    spi_select(qts, 0);
    flash_erase_program_read(qts, cs0_addr, wbuf, rbuf, 4);
    g_assert_cmpmem(wbuf, 4, rbuf, 4);

    /* CS1: write near 4MB boundary */
    uint32_t cs1_addr = CS1_SIZE - 4096;
    spi_select(qts, 1);
    flash_erase_program_read(qts, cs1_addr, wbuf, rbuf, 4);
    g_assert_cmpmem(wbuf, 4, rbuf, 4);

    qtest_quit(qts);
}

/* Read status registers from both chips */
static void test_concurrent_status_check(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    spi_select(qts, 0);
    spi_xfer(qts, FLASH_CMD_READ_STATUS);
    uint8_t sr0 = spi_xfer(qts, 0x00);

    spi_select(qts, 1);
    spi_xfer(qts, FLASH_CMD_READ_STATUS);
    uint8_t sr1 = spi_xfer(qts, 0x00);

    /* Neither should be busy at idle */
    g_assert_cmpuint(sr0 & FLASH_SR_BUSY, ==, 0);
    g_assert_cmpuint(sr1 & FLASH_SR_BUSY, ==, 0);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/spi-cs/identification", test_flash_identification);
    qtest_add_func("g233/spi-cs/individual", test_individual_flash_operations);
    qtest_add_func("g233/spi-cs/cross", test_cross_flash_operations);
    qtest_add_func("g233/spi-cs/alternating", test_alternating_operations);
    qtest_add_func("g233/spi-cs/capacity", test_flash_capacity);
    qtest_add_func("g233/spi-cs/concurrent_status",
                   test_concurrent_status_check);

    return g_test_run();
}
