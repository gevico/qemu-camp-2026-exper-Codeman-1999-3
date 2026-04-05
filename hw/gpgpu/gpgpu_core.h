/*
 * QEMU GPGPU - RISC-V SIMT Core
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * 简化的 RV32I 指令解释器，用于 GPU 核心模拟。
 * 参考 NEMU 的设计思想，但独立实现。
 */

#ifndef HW_GPGPU_CORE_H
#define HW_GPGPU_CORE_H

#include "qemu/osdep.h"
#include "fpu/softfloat.h"

/* 前向声明 */
typedef struct GPGPUState GPGPUState;

/*
 * ============================================================================
 * 常量定义
 * ============================================================================
 */
#define GPGPU_WARP_SIZE     32      /* 每个 warp 的 lane 数量 */
#define GPGPU_NUM_REGS      32      /* RISC-V 通用寄存器数量 */
#define GPGPU_NUM_FREGS     32      /* RISC-V 浮点寄存器数量 */

/* 浮点 CSR 地址 */
#define CSR_FFLAGS          0x001
#define CSR_FRM             0x002
#define CSR_FCSR            0x003

/*
 * ============================================================================
 * mhartid CSR 定义
 * ============================================================================
 * 位域布局:
 *   31        13 12     5 4    0
 *   +---------+--------+------+
 *   | block   | warp   | tid  |
 *   | (19bit) | (8bit) | (5b) |
 *   +---------+--------+------+
 */
#define CSR_MHARTID             0xF14

#define MHARTID_THREAD_BITS     5
#define MHARTID_WARP_BITS       8
#define MHARTID_BLOCK_BITS      19
#define MHARTID_THREAD_MASK     0x1F
#define MHARTID_WARP_MASK       0xFF

#define MHARTID_ENCODE(block, warp, thread) \
    (((block) << 13) | ((warp) << 5) | ((thread) & 0x1F))
#define MHARTID_THREAD(id)      ((id) & 0x1F)
#define MHARTID_WARP(id)        (((id) >> 5) & 0xFF)
#define MHARTID_BLOCK(id)       ((id) >> 13)

/*
 * ============================================================================
 * Lane 状态结构
 * ============================================================================
 * 每个 Lane 相当于一个简化的 RISC-V 核心
 */
typedef struct GPGPULane {
    uint32_t gpr[GPGPU_NUM_REGS];   /* 通用寄存器 x0-x31 */
    uint32_t fpr[GPGPU_NUM_FREGS];  /* 浮点寄存器 f0-f31 */
    uint32_t pc;                     /* 程序计数器 */
    uint32_t mhartid;                /* 完整 hart ID (block|warp|lane) */
    uint32_t fcsr;                   /* fflags[4:0] | frm[7:5] */
    float_status fp_status;          /* softfloat 运行状态 */
    bool active;                     /* 是否活跃 */
} GPGPULane;

/*
 * ============================================================================
 * Warp 状态结构
 * ============================================================================
 * 一个 Warp 包含 32 个 Lane，它们锁步执行同一条指令
 */
typedef struct GPGPUWarp {
    GPGPULane lanes[GPGPU_WARP_SIZE];   /* 32 个 lane */
    uint32_t active_mask;                /* 活跃掩码，每位代表一个 lane */
    uint32_t thread_id_base;             /* 这个 warp 的起始 thread_id */
    uint32_t warp_id;                    /* warp 在 block 内的编号 */
    uint32_t block_id[3];                /* 所属 block 的 ID */
} GPGPUWarp;

/*
 * ============================================================================
 * CTRL 设备地址定义 (GPU 核心视角)
 * ============================================================================
 * GPU 核心通过访问这些地址来获取自己的线程 ID
 */
#define GPGPU_CORE_CTRL_BASE        0x80000000  /* CTRL 基地址 */
#define GPGPU_CORE_CTRL_THREAD_ID_X (GPGPU_CORE_CTRL_BASE + 0x00)
#define GPGPU_CORE_CTRL_THREAD_ID_Y (GPGPU_CORE_CTRL_BASE + 0x04)
#define GPGPU_CORE_CTRL_THREAD_ID_Z (GPGPU_CORE_CTRL_BASE + 0x08)
#define GPGPU_CORE_CTRL_BLOCK_ID_X  (GPGPU_CORE_CTRL_BASE + 0x10)
#define GPGPU_CORE_CTRL_BLOCK_ID_Y  (GPGPU_CORE_CTRL_BASE + 0x14)
#define GPGPU_CORE_CTRL_BLOCK_ID_Z  (GPGPU_CORE_CTRL_BASE + 0x18)
#define GPGPU_CORE_CTRL_BLOCK_DIM_X (GPGPU_CORE_CTRL_BASE + 0x20)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Y (GPGPU_CORE_CTRL_BASE + 0x24)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Z (GPGPU_CORE_CTRL_BASE + 0x28)
#define GPGPU_CORE_CTRL_GRID_DIM_X  (GPGPU_CORE_CTRL_BASE + 0x30)
#define GPGPU_CORE_CTRL_GRID_DIM_Y  (GPGPU_CORE_CTRL_BASE + 0x34)
#define GPGPU_CORE_CTRL_GRID_DIM_Z  (GPGPU_CORE_CTRL_BASE + 0x38)

/*
 * ============================================================================
 * 执行引擎 API
 * ============================================================================
 */

/**
 * gpgpu_core_init_warp - 初始化一个 warp
 * @warp: warp 状态指针
 * @pc: 初始程序计数器（内核代码地址）
 * @thread_id_base: 起始线程 ID
 * @block_id: block ID 数组 [x, y, z]
 * @num_threads: 活跃线程数量 (最多 32)
 * @warp_id: warp 在 block 内的编号
 * @block_id_linear: 线性化的 block ID
 */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear);

/**
 * gpgpu_core_exec_warp - 执行一个 warp 直到完成
 * @s: GPGPU 设备状态
 * @warp: warp 状态指针
 * @max_cycles: 最大执行周期数（防止死循环）
 *
 * 返回: 0 成功，-1 错误（如非法指令）
 */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles);

/**
 * gpgpu_core_exec_kernel - 执行完整的 kernel
 * @s: GPGPU 设备状态
 *
 * 根据 s->kernel 中配置的 grid/block 维度执行内核。
 * 返回: 0 成功，-1 错误
 */
int gpgpu_core_exec_kernel(GPGPUState *s);

#endif /* HW_GPGPU_CORE_H */
