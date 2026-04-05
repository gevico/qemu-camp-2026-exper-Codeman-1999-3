/*
 * Xg233ai instruction test: gemm — INT32 4x4 matrix multiply
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define DIM 4

static int mat_a[DIM * DIM];
static int mat_b[DIM * DIM];
static int mat_hw[DIM * DIM];
static int mat_sw[DIM * DIM];

static inline void custom_gemm(int *c, const int *a, const int *b)
{
    asm volatile(
        ".insn r 0x7b, 6, 14, %0, %1, %2"
        :
        : "r"(c), "r"(a), "r"(b)
        : "memory"
    );
}

static void software_gemm(int *c, const int *a, const int *b)
{
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            long acc = 0;
            for (int k = 0; k < DIM; k++)
                acc += (long)a[i * DIM + k] * (long)b[k * DIM + j];
            c[i * DIM + j] = (int)acc;
        }
}

static void compare(const int *hw, const int *sw, int n)
{
    for (int i = 0; i < n; i++) {
        if (hw[i] != sw[i]) {
            printf("MISMATCH at [%d]: hw=%d sw=%d\n", i, hw[i], sw[i]);
            crt_assert(0);
        }
    }
}

static void test_gemm_basic(void)
{
    /* A = {{1..4},{5..8},{9..12},{13..16}} */
    for (int i = 0; i < DIM * DIM; i++)
        mat_a[i] = i + 1;

    /* B = {{17..20},{21..24},{25..28},{29..32}} */
    for (int i = 0; i < DIM * DIM; i++)
        mat_b[i] = i + 17;

    memset(mat_hw, 0, sizeof(mat_hw));
    memset(mat_sw, 0, sizeof(mat_sw));

    custom_gemm(mat_hw, mat_a, mat_b);
    software_gemm(mat_sw, mat_a, mat_b);
    compare(mat_hw, mat_sw, DIM * DIM);

    /* spot-check: C[0][0] = 1*17+2*21+3*25+4*29 = 250 */
    crt_assert(mat_sw[0] == 250);
}

static void test_gemm_identity(void)
{
    int identity[DIM * DIM] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    /* A = {1..16} */
    for (int i = 0; i < DIM * DIM; i++)
        mat_a[i] = i + 1;

    memset(mat_hw, 0, sizeof(mat_hw));

    custom_gemm(mat_hw, mat_a, identity);
    compare(mat_hw, mat_a, DIM * DIM);
}

int main(void)
{
    test_gemm_basic();
    test_gemm_identity();
    printf("gemm: all tests passed\n");
    return 0;
}
