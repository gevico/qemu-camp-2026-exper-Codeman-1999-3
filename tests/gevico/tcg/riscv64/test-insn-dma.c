/*
 * Xg233ai instruction test: dma — FP32 matrix transpose
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define MAX_N 32

static float src[MAX_N * MAX_N];
static float dst_hw[MAX_N * MAX_N];
static float dst_sw[MAX_N * MAX_N];

static inline void custom_dma(float *dst, const float *src, long grain)
{
    asm volatile(
        ".insn r 0x7b, 6, 6, %0, %1, %2"
        :
        : "r"(dst), "r"(src), "r"(grain)
        : "memory"
    );
}

static void software_transpose(float *dst, const float *src, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[j * n + i] = src[i * n + j];
}

static void compare(const float *hw, const float *sw, int n)
{
    const uint32_t *a = (const uint32_t *)hw;
    const uint32_t *b = (const uint32_t *)sw;
    for (int i = 0; i < n * n; i++) {
        if (a[i] != b[i]) {
            printf("MISMATCH at [%d]: hw=0x%x sw=0x%x\n", i, a[i], b[i]);
            crt_assert(0);
        }
    }
}

static void init_matrix(float *m, int n)
{
    for (int i = 0; i < n * n; i++)
        m[i] = (float)i;
}

static void test_dma_grain_8x8(void)
{
    int n = 8;
    init_matrix(src, n);
    memset(dst_hw, 0, sizeof(dst_hw));
    memset(dst_sw, 0, sizeof(dst_sw));

    custom_dma(dst_hw, src, 0);
    software_transpose(dst_sw, src, n);
    compare(dst_hw, dst_sw, n);
}

static void test_dma_grain_16x16(void)
{
    int n = 16;
    init_matrix(src, n);
    memset(dst_hw, 0, sizeof(dst_hw));
    memset(dst_sw, 0, sizeof(dst_sw));

    custom_dma(dst_hw, src, 1);
    software_transpose(dst_sw, src, n);
    compare(dst_hw, dst_sw, n);
}

static void test_dma_grain_32x32(void)
{
    int n = 32;
    init_matrix(src, n);
    memset(dst_hw, 0, sizeof(dst_hw));
    memset(dst_sw, 0, sizeof(dst_sw));

    custom_dma(dst_hw, src, 2);
    software_transpose(dst_sw, src, n);
    compare(dst_hw, dst_sw, n);
}

int main(void)
{
    test_dma_grain_8x8();
    test_dma_grain_16x16();
    test_dma_grain_32x32();
    printf("dma: all tests passed\n");
    return 0;
}
