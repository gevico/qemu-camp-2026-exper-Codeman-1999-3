/*
 * QTest: G233 SPI Flash — interrupt-driven transfer
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * SPI PLIC IRQ: 5
 * Tests interrupt-driven SPI transfers with TXE/RXNE interrupts.
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
#define SPI_CR1_ERRIE   (1u << 5)
#define SPI_CR1_RXNEIE  (1u << 6)
#define SPI_CR1_TXEIE   (1u << 7)

#define SPI_SR_RXNE     (1u << 0)
#define SPI_SR_TXE      (1u << 1)

#define PLIC_BASE       0x0C000000ULL
#define PLIC_PENDING    (PLIC_BASE + 0x1000)
#define SPI_PLIC_IRQ    5

#define FLASH_CMD_WRITE_ENABLE  0x06
#define FLASH_CMD_READ_STATUS   0x05
#define FLASH_CMD_READ_DATA     0x03
#define FLASH_CMD_PAGE_PROGRAM  0x02
#define FLASH_CMD_SECTOR_ERASE  0x20
#define FLASH_CMD_JEDEC_ID      0x9F

#define FLASH_SR_BUSY   0x01

static inline bool plic_irq_pending(QTestState *qts, int irq)
{
    uint32_t word = qtest_readl(qts, PLIC_PENDING + (irq / 32) * 4);
    return (word >> (irq % 32)) & 1;
}

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

static void flash_wait_busy(QTestState *qts)
{
    int timeout = 10000;
    uint8_t sr;
    do {
        spi_xfer(qts, FLASH_CMD_READ_STATUS);
        sr = spi_xfer(qts, 0x00);
        qtest_clock_step(qts, 100000);
    } while ((sr & FLASH_SR_BUSY) && --timeout);
    g_assert_cmpint(timeout, >, 0);
}

/* Verify TXEIE generates PLIC interrupt when TXE is set */
static void test_spi_txe_interrupt(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, SPI_CR2, 0);
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR | SPI_CR1_TXEIE);

    /* TX buffer is empty at init → TXE=1 → interrupt should fire */
    qtest_clock_step(qts, 10000);
    g_assert_true(plic_irq_pending(qts, SPI_PLIC_IRQ));

    qtest_quit(qts);
}

/* Verify RXNEIE generates PLIC interrupt after receiving data */
static void test_spi_rxne_interrupt(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, SPI_CR2, 0);
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR | SPI_CR1_RXNEIE);

    /* Send a byte to trigger reception */
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, 0xFF);
    spi_wait_rxne(qts);

    g_assert_true(plic_irq_pending(qts, SPI_PLIC_IRQ));

    /* Read DR to clear RXNE */
    qtest_readl(qts, SPI_DR);

    qtest_quit(qts);
}

/* Interrupt-driven JEDEC ID read */
static void test_flash_jedec_interrupt(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, SPI_CR2, 0);
    qtest_writel(qts, SPI_CR1,
                 SPI_CR1_SPE | SPI_CR1_MSTR | SPI_CR1_RXNEIE);

    /* Send JEDEC ID command */
    spi_xfer(qts, FLASH_CMD_JEDEC_ID);
    uint8_t mfr = spi_xfer(qts, 0x00);
    uint8_t id1 = spi_xfer(qts, 0x00);
    uint8_t id2 = spi_xfer(qts, 0x00);

    g_assert_cmpuint(mfr, ==, 0xEF);
    g_assert_cmpuint(id1, ==, 0x30);
    g_assert_cmpuint(id2, ==, 0x15);

    qtest_quit(qts);
}

/* Interrupt-driven erase → program → read cycle */
static void test_flash_write_test_data(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t wbuf[64], rbuf[64];

    qtest_writel(qts, SPI_CR2, 0);
    qtest_writel(qts, SPI_CR1,
                 SPI_CR1_SPE | SPI_CR1_MSTR | SPI_CR1_RXNEIE);

    for (int i = 0; i < 64; i++) {
        wbuf[i] = (uint8_t)(0x55 ^ i);
    }

    /* Write enable + sector erase */
    spi_xfer(qts, FLASH_CMD_WRITE_ENABLE);
    spi_xfer(qts, FLASH_CMD_SECTOR_ERASE);
    spi_xfer(qts, 0x00);
    spi_xfer(qts, 0x00);
    spi_xfer(qts, 0x00);
    flash_wait_busy(qts);

    /* Write enable + page program */
    spi_xfer(qts, FLASH_CMD_WRITE_ENABLE);
    spi_xfer(qts, FLASH_CMD_PAGE_PROGRAM);
    spi_xfer(qts, 0x00);
    spi_xfer(qts, 0x00);
    spi_xfer(qts, 0x00);
    for (int i = 0; i < 64; i++) {
        spi_xfer(qts, wbuf[i]);
    }
    flash_wait_busy(qts);

    /* Read data */
    spi_xfer(qts, FLASH_CMD_READ_DATA);
    spi_xfer(qts, 0x00);
    spi_xfer(qts, 0x00);
    spi_xfer(qts, 0x00);
    for (int i = 0; i < 64; i++) {
        rbuf[i] = spi_xfer(qts, 0x00);
    }

    g_assert_cmpmem(wbuf, 64, rbuf, 64);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/flash-int/txe_interrupt", test_spi_txe_interrupt);
    qtest_add_func("g233/flash-int/rxne_interrupt", test_spi_rxne_interrupt);
    qtest_add_func("g233/flash-int/jedec_interrupt", test_flash_jedec_interrupt);
    qtest_add_func("g233/flash-int/write_test_data", test_flash_write_test_data);

    return g_test_run();
}
