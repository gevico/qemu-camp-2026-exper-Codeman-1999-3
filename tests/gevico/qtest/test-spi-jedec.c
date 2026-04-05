/*
 * QTest: G233 SPI controller — JEDEC ID read
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * SPI register map (base 0x10018000):
 *   0x00  SPI_CR1  — bit0: SPE, bit2: MSTR, bit5: ERRIE,
 *                     bit6: RXNEIE, bit7: TXEIE
 *   0x04  SPI_CR2  — bits[1:0]: CS select
 *   0x08  SPI_SR   — bit0: RXNE, bit1: TXE, bit4: OVERRUN (w1c)
 *   0x0C  SPI_DR   — data register
 *
 * Flash CS0: W25X16, JEDEC = {0xEF, 0x30, 0x15}
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

#define FLASH_CMD_JEDEC_ID  0x9F

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

static uint8_t spi_transfer_byte(QTestState *qts, uint8_t tx)
{
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, tx);
    spi_wait_rxne(qts);
    return (uint8_t)qtest_readl(qts, SPI_DR);
}

/* Verify SPI init: master mode + enable */
static void test_spi_init(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);
    uint32_t cr1 = qtest_readl(qts, SPI_CR1);
    g_assert_cmpuint(cr1 & SPI_CR1_SPE, ==, SPI_CR1_SPE);
    g_assert_cmpuint(cr1 & SPI_CR1_MSTR, ==, SPI_CR1_MSTR);

    qtest_quit(qts);
}

/* Read JEDEC ID from W25X16 on CS0 */
static void test_jedec_id(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    uint8_t id[3];

    /* Init SPI: master + enable, CS0 */
    qtest_writel(qts, SPI_CR2, 0);  /* CS0 */
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    /* Send JEDEC ID command */
    spi_transfer_byte(qts, FLASH_CMD_JEDEC_ID);

    /* Read 3 bytes of ID */
    id[0] = spi_transfer_byte(qts, 0x00);
    id[1] = spi_transfer_byte(qts, 0x00);
    id[2] = spi_transfer_byte(qts, 0x00);

    /* W25X16: manufacturer=0xEF, device=0x3015 */
    g_assert_cmpuint(id[0], ==, 0xEF);
    g_assert_cmpuint(id[1], ==, 0x30);
    g_assert_cmpuint(id[2], ==, 0x15);

    qtest_quit(qts);
}

/* Verify TXE/RXNE status bits during transfer */
static void test_spi_transfer_byte(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, SPI_CR2, 0);
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    /* TXE should be set initially (TX buffer empty) */
    g_assert_cmpuint(qtest_readl(qts, SPI_SR) & SPI_SR_TXE, !=, 0);

    /* Write data */
    qtest_writel(qts, SPI_DR, 0xAA);

    /* Wait for RXNE */
    spi_wait_rxne(qts);
    g_assert_cmpuint(qtest_readl(qts, SPI_SR) & SPI_SR_RXNE, !=, 0);

    /* Reading DR should clear RXNE */
    qtest_readl(qts, SPI_DR);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/spi/init", test_spi_init);
    qtest_add_func("g233/spi/jedec_id", test_jedec_id);
    qtest_add_func("g233/spi/transfer_byte", test_spi_transfer_byte);

    return g_test_run();
}
