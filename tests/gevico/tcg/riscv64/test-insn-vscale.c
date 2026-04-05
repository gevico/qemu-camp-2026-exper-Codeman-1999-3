/*
 * Xg233ai instruction test: vscale — INT32 vector scalar multiply
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define VEC_LEN 16

static int dst_hw[VEC_LEN];
static int dst_sw[VEC_LEN];

static inline void custom_vscale(int *dst, const int *src, long scale)
{
    asm volatile(
        ".insn r 0x7b, 6, 102, %0, %1, %2"
        :
        : "r"(dst), "r"(src), "r"(scale)
        : "memory"
    );
}

static void software_vscale(int *dst, const int *src, long scale)
{
    for (int i = 0; i < VEC_LEN; i++)
        dst[i] = (int)((long)src[i] * scale);
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

static void test_vscale_basic(void)
{
    int src[VEC_LEN];
    for (int i = 0; i < VEC_LEN; i++)
        src[i] = i + 1;

    custom_vscale(dst_hw, src, 3);
    software_vscale(dst_sw, src, 3);
    compare(dst_hw, dst_sw, VEC_LEN);

    /* verify: {3, 6, 9, ..., 48} */
    crt_assert(dst_sw[0] == 3 && dst_sw[15] == 48);
}

static void test_vscale_negative(void)
{
    int src[VEC_LEN] = {
        10, -20, 30, -40, 50, -60, 70, -80,
        90, -100, 110, -120, 130, -140, 150, -160
    };

    custom_vscale(dst_hw, src, -2);
    software_vscale(dst_sw, src, -2);
    compare(dst_hw, dst_sw, VEC_LEN);
}

int main(void)
{
    test_vscale_basic();
    test_vscale_negative();
    printf("vscale: all tests passed\n");
    return 0;
}
