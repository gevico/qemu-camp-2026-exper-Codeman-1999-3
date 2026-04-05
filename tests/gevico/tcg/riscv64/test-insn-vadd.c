/*
 * Xg233ai instruction test: vadd — INT32 vector element-wise addition
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define VEC_LEN 16
#define INT32_MAX_VAL 0x7FFFFFFF

static int dst_hw[VEC_LEN];
static int dst_sw[VEC_LEN];

static inline void custom_vadd(int *c, const int *a, const int *b)
{
    asm volatile(
        ".insn r 0x7b, 6, 30, %0, %1, %2"
        :
        : "r"(c), "r"(a), "r"(b)
        : "memory"
    );
}

static void software_vadd(int *c, const int *a, const int *b)
{
    for (int i = 0; i < VEC_LEN; i++)
        c[i] = a[i] + b[i];
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

static void test_vadd_basic(void)
{
    int a[VEC_LEN], b[VEC_LEN];
    for (int i = 0; i < VEC_LEN; i++) {
        a[i] = i + 1;
        b[i] = (i + 1) * 100;
    }

    custom_vadd(dst_hw, a, b);
    software_vadd(dst_sw, a, b);
    compare(dst_hw, dst_sw, VEC_LEN);

    /* verify: {101, 202, 303, ...} */
    crt_assert(dst_sw[0] == 101 && dst_sw[1] == 202);
}

static void test_vadd_overflow(void)
{
    int a[VEC_LEN], b[VEC_LEN];
    for (int i = 0; i < VEC_LEN; i++) {
        a[i] = INT32_MAX_VAL - i;
        b[i] = i + 1;
    }

    custom_vadd(dst_hw, a, b);
    software_vadd(dst_sw, a, b);
    compare(dst_hw, dst_sw, VEC_LEN);
}

static void test_vadd_inplace(void)
{
    int a[VEC_LEN], b[VEC_LEN];
    for (int i = 0; i < VEC_LEN; i++) {
        a[i] = i + 1;
        b[i] = (i + 1) * 100;
    }

    /* compute fresh software reference */
    software_vadd(dst_sw, a, b);

    /* in-place: c = a, so a += b */
    custom_vadd(a, a, b);
    compare(a, dst_sw, VEC_LEN);
}

int main(void)
{
    test_vadd_basic();
    test_vadd_overflow();
    test_vadd_inplace();
    printf("vadd: all tests passed\n");
    return 0;
}
