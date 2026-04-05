/*
 * Xg233ai instruction test: vmax — INT32 vector max reduction
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

static inline long custom_vmax(const int *src, long n)
{
    long result;
    asm volatile(
        ".insn r 0x7b, 6, 118, %0, %1, %2"
        : "=r"(result)
        : "r"(src), "r"(n)
        : "memory"
    );
    return result;
}

static long software_vmax(const int *src, int n)
{
    int max = src[0];
    for (int i = 1; i < n; i++)
        if (src[i] > max)
            max = src[i];
    /* sign-extend INT32 to XLEN(64) */
    return (long)(int)max;
}

static void test_vmax_positive(void)
{
    int data[] = {5, 23, 1, 99, 42, 7, 88, 3};
    int n = 8;

    long hw = custom_vmax(data, n);
    long sw = software_vmax(data, n);

    if (hw != sw) {
        printf("MISMATCH: hw=%ld sw=%ld\n", hw, sw);
        crt_assert(0);
    }
    crt_assert(sw == 99);
}

static void test_vmax_negative(void)
{
    int data[] = {-5, -23, -1, -99, -42, -7, -88, -3};
    int n = 8;

    long hw = custom_vmax(data, n);
    long sw = software_vmax(data, n);

    if (hw != sw) {
        printf("MISMATCH: hw=%ld sw=%ld\n", hw, sw);
        crt_assert(0);
    }
    crt_assert(sw == -1);
}

int main(void)
{
    test_vmax_positive();
    test_vmax_negative();
    printf("vmax: all tests passed\n");
    return 0;
}
