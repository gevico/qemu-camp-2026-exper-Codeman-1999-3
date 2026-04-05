/*
 * Xg233ai instruction test: vrelu — INT32 vector ReLU activation
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define VEC_LEN 16

static int src_data[VEC_LEN] = {
    -5, 3, -1, 0, 7, -100, 42, -3,
    8, -9, 15, -20, 1, 0, -7, 99
};

static int dst_hw[VEC_LEN];
static int dst_sw[VEC_LEN];

static inline void custom_vrelu(int *dst, const int *src, long n)
{
    asm volatile(
        ".insn r 0x7b, 6, 86, %0, %1, %2"
        :
        : "r"(dst), "r"(src), "r"(n)
        : "memory"
    );
}

static void software_vrelu(int *dst, const int *src, int n)
{
    for (int i = 0; i < n; i++)
        dst[i] = (src[i] > 0) ? src[i] : 0;
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

static void test_vrelu_mixed(void)
{
    memset(dst_hw, 0, sizeof(dst_hw));
    memset(dst_sw, 0, sizeof(dst_sw));

    custom_vrelu(dst_hw, src_data, VEC_LEN);
    software_vrelu(dst_sw, src_data, VEC_LEN);
    compare(dst_hw, dst_sw, VEC_LEN);
}

static void test_vrelu_inplace(void)
{
    int inplace[VEC_LEN];
    memcpy(inplace, src_data, sizeof(src_data));

    custom_vrelu(inplace, inplace, VEC_LEN);
    compare(inplace, dst_sw, VEC_LEN);
}

int main(void)
{
    test_vrelu_mixed();
    test_vrelu_inplace();
    printf("vrelu: all tests passed\n");
    return 0;
}
