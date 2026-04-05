/*
 * QTest: G233 board basic functionality
 *
 * Copyright (c) 2025 Chao Liu <chao.liu@yeah.net>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define G233_DRAM_BASE   0x80000000ULL
#define G233_DRAM_SIZE   0x80000000ULL  /* 2GB */
#define G233_UART_BASE   0x10000000ULL
#define G233_PLIC_BASE   0x0C000000ULL
#define G233_CLINT_BASE  0x02000000ULL

/* Verify the machine can boot without crashing */
static void test_board_init(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_quit(qts);
}

/* Verify DRAM is accessible and writable */
static void test_dram_access(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    qtest_writel(qts, G233_DRAM_BASE, 0xDEADBEEF);
    g_assert_cmpuint(qtest_readl(qts, G233_DRAM_BASE), ==, 0xDEADBEEF);

    qtest_writel(qts, G233_DRAM_BASE + 0x1000, 0xCAFEBABE);
    g_assert_cmpuint(qtest_readl(qts, G233_DRAM_BASE + 0x1000), ==,
                     0xCAFEBABE);

    qtest_quit(qts);
}

/* Verify PLIC is present via MMIO read (priority reg at base) */
static void test_plic_present(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* PLIC priority register for source 1 defaults to 0 */
    g_assert_cmpuint(qtest_readl(qts, G233_PLIC_BASE + 0x04), ==, 0);

    /* Write priority and read back */
    qtest_writel(qts, G233_PLIC_BASE + 0x04, 1);
    g_assert_cmpuint(qtest_readl(qts, G233_PLIC_BASE + 0x04), ==, 1);

    qtest_quit(qts);
}

/* Verify CLINT is present */
static void test_clint_present(void)
{
    QTestState *qts = qtest_init("-machine g233 -m 2G");

    /* CLINT mtime register at offset 0xBFF8 should be readable */
    qtest_readq(qts, G233_CLINT_BASE + 0xBFF8);

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("g233/board/init", test_board_init);
    qtest_add_func("g233/board/dram", test_dram_access);
    qtest_add_func("g233/board/plic", test_plic_present);
    qtest_add_func("g233/board/clint", test_clint_present);

    return g_test_run();
}
