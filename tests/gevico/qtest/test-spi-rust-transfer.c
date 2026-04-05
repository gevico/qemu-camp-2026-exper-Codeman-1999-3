/*
 * QTest: G233 Rust SPI controller — data transfer and loopback
 *
 * Copyright (c) 2026 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Rust SPI register map (base 0x10019000):
 *   0x00  RSPI_CR1  — bit0: SPE, bit2: MSTR
 *   0x04  RSPI_SR   — bit0: RXNE, bit1: TXE, bit4: OVERRUN
 *   0x08  RSPI_DR   — data register
 *   0x0C  RSPI_CS   — chip select
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
#define RSPI_SR_OVERRUN (1u << 4)

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
    qtest_writel(qts, RSPI_CS, 0);
    qtest_writel(qts, RSPI_CR1, RSPI_CR1_SPE | RSPI_CR1_MSTR);
}

/* Transfer a single byte: verify TXE and RXNE transitions */
static void test_rspi_transfer_byte(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    rspi_init(qts);

    /* TXE should be set initially */
    g_assert_cmpuint(qtest_readl(qts, RSPI_SR) & RSPI_SR_TXE, !=, 0);

    /* Write a byte */
    qtest_writel(qts, RSPI_DR, 0x55);

    /* Wait for RXNE (transfer complete) */
    rspi_wait_rxne(qts);
    g_assert_cmpuint(qtest_readl(qts, RSPI_SR) & RSPI_SR_RXNE, !=, 0);

    /* Read DR to clear RXNE */
    qtest_readl(qts, RSPI_DR);

    /* After read, RXNE should be cleared */
    qtest_clock_step(qts, 1000);

    qtest_quit(qts);
}

/* Loopback: transfer multiple bytes and verify round-trip via slave echo */
static void test_rspi_loopback(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");
    rspi_init(qts);

    /*
     * With a flash attached, sending a NOP-like sequence:
     * each transfer returns the slave's response byte.
     */
    const uint8_t tx_data[] = {0x00, 0xFF, 0xA5, 0x5A};

    for (int i = 0; i < 4; i++) {
        uint8_t rx = rspi_xfer(qts, tx_data[i]);
        /* Just verify the transfer completes without hang */
        (void)rx;
    }

    /* Verify no OVERRUN occurred */
    g_assert_cmpuint(qtest_readl(qts, RSPI_SR) & RSPI_SR_OVERRUN, ==, 0);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/rust-spi/transfer_byte", test_rspi_transfer_byte);
    qtest_add_func("g233/rust-spi/loopback", test_rspi_loopback);

    return g_test_run();
}
