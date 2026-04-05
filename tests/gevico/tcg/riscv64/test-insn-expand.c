/*
 * Xg233ai instruction test: expand — 4-bit to 8-bit decompression
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define SRC_LEN 5
#define DST_LEN 10

static uint8_t src_data[SRC_LEN] = {
    0xAB, 0xCD, 0xEF, 0x12, 0x34
};

static uint8_t dst_hw[DST_LEN];
static uint8_t dst_sw[DST_LEN];

static inline void custom_expand(uint8_t *dst, const uint8_t *src, long n)
{
    asm volatile(
        ".insn r 0x7b, 6, 54, %0, %1, %2"
        :
        : "r"(dst), "r"(src), "r"(n)
        : "memory"
    );
}

static void split_to_4bits(uint8_t *dst, const uint8_t *src, int n)
{
    for (int i = 0; i < n; i++) {
        dst[2 * i]     = src[i] & 0x0F;
        dst[2 * i + 1] = (src[i] >> 4) & 0x0F;
    }
}

static void compare(const uint8_t *hw, const uint8_t *sw, int n)
{
    for (int i = 0; i < n; i++) {
        if (hw[i] != sw[i]) {
            printf("MISMATCH at [%d]: hw=0x%x sw=0x%x\n", i,
                   (int)hw[i], (int)sw[i]);
            crt_assert(0);
        }
    }
}

static void test_expand(void)
{
    memset(dst_hw, 0, sizeof(dst_hw));
    memset(dst_sw, 0, sizeof(dst_sw));

    custom_expand(dst_hw, src_data, SRC_LEN);
    split_to_4bits(dst_sw, src_data, SRC_LEN);
    compare(dst_hw, dst_sw, DST_LEN);
}

int main(void)
{
    test_expand();
    printf("expand: all tests passed\n");
    return 0;
}
