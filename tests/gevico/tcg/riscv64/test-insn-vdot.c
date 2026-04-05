/*
 * Xg233ai instruction test: vdot — INT32 vector dot product
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define VEC_LEN 16

static int vec_a[VEC_LEN];
static int vec_b[VEC_LEN];

static inline long custom_vdot(const int *a, const int *b)
{
    long result;
    asm volatile(
        ".insn r 0x7b, 6, 70, %0, %1, %2"
        : "=r"(result)
        : "r"(a), "r"(b)
        : "memory"
    );
    return result;
}

static long software_vdot(const int *a, const int *b)
{
    long acc = 0;
    for (int i = 0; i < VEC_LEN; i++)
        acc += (long)a[i] * (long)b[i];
    return acc;
}

static void test_vdot_basic(void)
{
    /* A = {1, 2, ..., 16}, B = {16, 15, ..., 1} */
    for (int i = 0; i < VEC_LEN; i++) {
        vec_a[i] = i + 1;
        vec_b[i] = VEC_LEN - i;
    }

    long hw = custom_vdot(vec_a, vec_b);
    long sw = software_vdot(vec_a, vec_b);

    if (hw != sw) {
        printf("MISMATCH: hw=%ld sw=%ld\n", hw, sw);
        crt_assert(0);
    }
    /* expected: 816 */
    crt_assert(sw == 816);
}

int main(void)
{
    test_vdot_basic();
    printf("vdot: all tests passed\n");
    return 0;
}
