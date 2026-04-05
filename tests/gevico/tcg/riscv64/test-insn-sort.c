/*
 * Xg233ai instruction test: sort — INT32 bubble sort
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "crt.h"

#define ARRAY_SIZE 32
#define SORT_LEN   16

static int hw_arr[ARRAY_SIZE];
static int sw_arr[ARRAY_SIZE];

static const int init_data[ARRAY_SIZE] = {
    3, 7, 23, 9, 81, 33, 42, 15,
    67, 51, 2, 19, 55, 11, 77, 29,
    64, 88, 5, 39, 71, 14, 46, 27,
    60, 36, 93, 8, 50, 18, 73, 44
};

static inline void custom_sort(long k, const int *arr, long n)
{
    asm volatile(
        ".insn r 0x7b, 6, 22, %0, %1, %2"
        :
        : "r"(k), "r"(arr), "r"(n)
        : "memory"
    );
}

static void bubble_sort(int *arr, int k)
{
    for (int i = 0; i < k - 1; i++)
        for (int j = 0; j < k - i - 1; j++)
            if (arr[j] > arr[j + 1]) {
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
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

static void test_sort(void)
{
    memcpy(hw_arr, init_data, sizeof(init_data));
    memcpy(sw_arr, init_data, sizeof(init_data));

    custom_sort(SORT_LEN, hw_arr, ARRAY_SIZE);
    bubble_sort(sw_arr, SORT_LEN);
    compare(hw_arr, sw_arr, ARRAY_SIZE);
}

int main(void)
{
    test_sort();
    printf("sort: all tests passed\n");
    return 0;
}
