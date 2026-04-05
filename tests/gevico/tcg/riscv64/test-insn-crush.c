/*
 * Xg233ai instruction test: crush — 8-bit to 4-bit compression
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define SRC_LEN 10
#define DST_LEN 5

static uint8_t src_data[SRC_LEN] = {
    0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
    0x0F, 0x01, 0x02, 0x03, 0x04
};

static uint8_t dst_hw[DST_LEN];
static uint8_t dst_sw[DST_LEN];

static inline void custom_crush(uint8_t *dst, const uint8_t *src, long n)
{
    asm volatile(
        ".insn r 0x7b, 6, 38, %0, %1, %2"
        :
        : "r"(dst), "r"(src), "r"(n)
        : "memory"
    );
}

static void pack_low4bits(uint8_t *dst, const uint8_t *src, int n)
{
    int out_len = (n + 1) / 2;
    for (int i = 0; i < n / 2; i++) {
        dst[i]  = src[2 * i] & 0x0F;
        dst[i] |= (src[2 * i + 1] & 0x0F) << 4;
    }
    if (n & 1)
        dst[out_len - 1] = src[n - 1] & 0x0F;
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

static void test_crush(void)
{
    memset(dst_hw, 0, sizeof(dst_hw));
    memset(dst_sw, 0, sizeof(dst_sw));

    custom_crush(dst_hw, src_data, SRC_LEN);
    pack_low4bits(dst_sw, src_data, SRC_LEN);
    compare(dst_hw, dst_sw, DST_LEN);
}

int main(void)
{
    test_crush();
    printf("crush: all tests passed\n");
    return 0;
}
