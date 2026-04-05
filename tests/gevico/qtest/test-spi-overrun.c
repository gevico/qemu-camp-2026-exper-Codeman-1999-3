/*
 * QTest: G233 SPI — overrun error detection
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Overrun: when RXNE=1 and a new byte completes, SPI_SR.OVERRUN is set.
 * SPI PLIC IRQ: 5
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

#define SPI_SR_RXNE     (1u << 0)
#define SPI_SR_TXE      (1u << 1)
#define SPI_SR_OVERRUN  (1u << 4)

#define PLIC_BASE       0x0C000000ULL
#define PLIC_PENDING    (PLIC_BASE + 0x1000)
#define SPI_PLIC_IRQ    5

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
}

static void spi_wait_rxne(QTestState *qts)
{
    int timeout = 1000;
    while (!(qtest_readl(qts, SPI_SR) & SPI_SR_RXNE) && --timeout) {
        qtest_clock_step(qts, 1000);
    }
}

/*
 * Interrupt-driven overrun detection:
 * Enable ERRIE, send two bytes without reading the first → OVERRUN
 */
static void test_interrupt_overrun_detection(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, SPI_CR2, 0);
    qtest_writel(qts, SPI_CR1,
                 SPI_CR1_SPE | SPI_CR1_MSTR | SPI_CR1_ERRIE);

    /* First byte: fills RX buffer */
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, 0xAA);
    spi_wait_rxne(qts);

    /* Do NOT read DR — RX buffer stays full (RXNE=1) */

    /* Second byte: should cause overrun */
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, 0xBB);
    qtest_clock_step(qts, 100000);

    /* OVERRUN flag should be set */
    g_assert_cmpuint(qtest_readl(qts, SPI_SR) & SPI_SR_OVERRUN, !=, 0);

    /* ERRIE enabled → PLIC interrupt should fire */
    g_assert_true(plic_irq_pending(qts, SPI_PLIC_IRQ));

    qtest_quit(qts);
}

/*
 * Polling-mode overrun detection:
 * No interrupt enable, just poll SPI_SR for OVERRUN, then write-1-to-clear.
 */
static void test_polling_overrun_detection(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, SPI_CR2, 0);
    qtest_writel(qts, SPI_CR1, SPI_CR1_SPE | SPI_CR1_MSTR);

    /* First byte */
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, 0x11);
    spi_wait_rxne(qts);

    /* Don't read DR */

    /* Second byte → overrun */
    spi_wait_txe(qts);
    qtest_writel(qts, SPI_DR, 0x22);
    qtest_clock_step(qts, 100000);

    g_assert_cmpuint(qtest_readl(qts, SPI_SR) & SPI_SR_OVERRUN, !=, 0);

    /* Write 1 to clear OVERRUN */
    qtest_writel(qts, SPI_SR, SPI_SR_OVERRUN);
    g_assert_cmpuint(qtest_readl(qts, SPI_SR) & SPI_SR_OVERRUN, ==, 0);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/spi-overrun/interrupt", test_interrupt_overrun_detection);
    qtest_add_func("g233/spi-overrun/polling", test_polling_overrun_detection);

    return g_test_run();
}
