/*
 * QTest: G233 Rust SPI controller — register init and enable
 *
 * Copyright (c) 2026 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Rust SPI register map (base 0x10019000):
 *   0x00  RSPI_CR1  — bit0: SPE (enable), bit2: MSTR (master)
 *   0x04  RSPI_SR   — bit0: RXNE, bit1: TXE, bit4: OVERRUN
 *   0x08  RSPI_DR   — data register
 *   0x0C  RSPI_CS   — chip select
 *   PLIC IRQ: 7
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

#define RSPI_SR_TXE     (1u << 1)

/* All registers should read zero after reset */
static void test_rspi_reset_values(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    g_assert_cmpuint(qtest_readl(qts, RSPI_CR1), ==, 0);
    g_assert_cmpuint(qtest_readl(qts, RSPI_SR),  ==, 0);
    g_assert_cmpuint(qtest_readl(qts, RSPI_DR),  ==, 0);
    g_assert_cmpuint(qtest_readl(qts, RSPI_CS),  ==, 0);

    qtest_quit(qts);
}

/* Enable SPI master mode, verify CR1 and TXE status */
static void test_rspi_enable(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* Enable as master */
    qtest_writel(qts, RSPI_CR1, RSPI_CR1_SPE | RSPI_CR1_MSTR);

    uint32_t cr1 = qtest_readl(qts, RSPI_CR1);
    g_assert_cmpuint(cr1 & RSPI_CR1_SPE,  ==, RSPI_CR1_SPE);
    g_assert_cmpuint(cr1 & RSPI_CR1_MSTR, ==, RSPI_CR1_MSTR);

    /* TXE should be set once SPI is enabled (TX buffer empty) */
    g_assert_cmpuint(qtest_readl(qts, RSPI_SR) & RSPI_SR_TXE, !=, 0);

    /* Select CS0 */
    qtest_writel(qts, RSPI_CS, 0);
    g_assert_cmpuint(qtest_readl(qts, RSPI_CS), ==, 0);

    /* Select CS1 */
    qtest_writel(qts, RSPI_CS, 1);
    g_assert_cmpuint(qtest_readl(qts, RSPI_CS), ==, 1);

    /* Disable SPI */
    qtest_writel(qts, RSPI_CR1, 0);
    g_assert_cmpuint(qtest_readl(qts, RSPI_CR1) & RSPI_CR1_SPE, ==, 0);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/rust-spi/reset_values", test_rspi_reset_values);
    qtest_add_func("g233/rust-spi/enable", test_rspi_enable);

    return g_test_run();
}
