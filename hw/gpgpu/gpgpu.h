/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef HW_GPGPU_H
#define HW_GPGPU_H

#include "hw/pci/pci_device.h"
#include "hw/pci/pci_ids.h"
#include "qom/object.h"

/*
 * ============================================================================
 * 设备类型定义
 * ============================================================================
 * TYPE_GPGPU: QOM (QEMU Object Model) 类型名称，用于设备注册和查找
 * GPGPU() 宏: 类型安全的类型转换宏，将 Object* 转换为 GPGPUState*
 */
#define TYPE_GPGPU "gpgpu"
OBJECT_DECLARE_SIMPLE_TYPE(GPGPUState, GPGPU)

/*
 * ============================================================================
 * PCI 配置
 * ============================================================================
 * 定义 PCI 设备的标识符，这些值会出现在 lspci 输出中
 * 驱动程序通过这些 ID 来识别和绑定设备
 */
#define GPGPU_VENDOR_ID         0x1234      /* QEMU 虚拟设备厂商 ID */
#define GPGPU_DEVICE_ID         0x1337      /* 设备 ID (自定义) */
#define GPGPU_REVISION          0x01        /* 硬件版本号 */
#define GPGPU_CLASS_CODE        PCI_CLASS_DISPLAY_3D  /* 0x0302: 3D Controller */

/*
 * ============================================================================
 * BAR 空间大小配置
 * ============================================================================
 * BAR (Base Address Register) 定义了设备的地址空间
 * 操作系统会为每个 BAR 分配物理地址，驱动通过这些地址访问设备
 */
#define GPGPU_CTRL_BAR_SIZE     (1 * 1024 * 1024)   /* BAR0: 控制寄存器 1MB */
#define GPGPU_VRAM_BAR_SIZE     (64 * 1024 * 1024)  /* BAR2: 显存 64MB (默认) */
#define GPGPU_DOORBELL_BAR_SIZE (64 * 1024)         /* BAR4: 门铃寄存器 64KB */

/*
 * ============================================================================
 * 设备默认配置
 * ============================================================================
 * 这些值可以通过 QEMU 命令行参数覆盖
 * 例如: -device gpgpu,num_cus=8,warps_per_cu=8
 */
#define GPGPU_DEFAULT_NUM_CUS       4       /* 默认计算单元数量 */
#define GPGPU_DEFAULT_WARPS_PER_CU  4       /* 每个 CU 的 warp 数量 */
#define GPGPU_DEFAULT_WARP_SIZE     32      /* 每个 warp 的线程数 */
#define GPGPU_DEFAULT_VRAM_SIZE     (64 * 1024 * 1024)  /* 默认显存大小 */

/*
 * ============================================================================
 * 控制寄存器偏移量定义 (BAR 0)
 * ============================================================================
 * 这些定义了 MMIO 空间中各寄存器的地址偏移
 * 驱动程序通过读写这些地址来控制设备
 */

/* 设备信息寄存器组 (0x0000 - 0x00FF): 只读，用于设备识别和能力查询 */
#define GPGPU_REG_DEV_ID            0x0000  /* 设备标识符 */
#define GPGPU_REG_DEV_VERSION       0x0004  /* 版本号 */
#define GPGPU_REG_DEV_CAPS          0x0008  /* 设备能力 */
#define GPGPU_REG_VRAM_SIZE_LO      0x000C  /* 显存大小低 32 位 */
#define GPGPU_REG_VRAM_SIZE_HI      0x0010  /* 显存大小高 32 位 */

/* 全局控制寄存器组 (0x0100 - 0x01FF): 设备使能、复位、状态查询 */
#define GPGPU_REG_GLOBAL_CTRL       0x0100  /* 全局控制 (使能/复位) */
#define GPGPU_REG_GLOBAL_STATUS     0x0104  /* 全局状态 (忙/空闲/错误) */
#define GPGPU_REG_ERROR_STATUS      0x0108  /* 错误状态 (写 1 清除) */

/* 中断控制寄存器组 (0x0200 - 0x02FF): 中断使能和状态管理 */
#define GPGPU_REG_IRQ_ENABLE        0x0200  /* 中断使能掩码 */
#define GPGPU_REG_IRQ_STATUS        0x0204  /* 中断状态 (挂起的中断) */
#define GPGPU_REG_IRQ_ACK           0x0208  /* 中断确认 (写 1 清除) */

/* 内核分发寄存器组 (0x0300 - 0x03FF): 配置和启动 GPU 计算任务 */
#define GPGPU_REG_KERNEL_ADDR_LO    0x0300  /* 内核代码地址低 32 位 */
#define GPGPU_REG_KERNEL_ADDR_HI    0x0304  /* 内核代码地址高 32 位 */
#define GPGPU_REG_KERNEL_ARGS_LO    0x0308  /* 内核参数地址低 32 位 */
#define GPGPU_REG_KERNEL_ARGS_HI    0x030C  /* 内核参数地址高 32 位 */
#define GPGPU_REG_GRID_DIM_X        0x0310  /* Grid X 维度 (Block 数量) */
#define GPGPU_REG_GRID_DIM_Y        0x0314  /* Grid Y 维度 */
#define GPGPU_REG_GRID_DIM_Z        0x0318  /* Grid Z 维度 */
#define GPGPU_REG_BLOCK_DIM_X       0x031C  /* Block X 维度 (线程数量) */
#define GPGPU_REG_BLOCK_DIM_Y       0x0320  /* Block Y 维度 */
#define GPGPU_REG_BLOCK_DIM_Z       0x0324  /* Block Z 维度 */
#define GPGPU_REG_SHARED_MEM_SIZE   0x0328  /* 每个 Block 的共享内存大小 */
#define GPGPU_REG_DISPATCH          0x0330  /* 写任意值启动内核执行 */

/* DMA 引擎寄存器组 (0x0400 - 0x04FF): 主机与显存之间的数据传输 */
#define GPGPU_REG_DMA_SRC_LO        0x0400  /* DMA 源地址低 32 位 */
#define GPGPU_REG_DMA_SRC_HI        0x0404  /* DMA 源地址高 32 位 */
#define GPGPU_REG_DMA_DST_LO        0x0408  /* DMA 目标地址低 32 位 */
#define GPGPU_REG_DMA_DST_HI        0x040C  /* DMA 目标地址高 32 位 */
#define GPGPU_REG_DMA_SIZE          0x0410  /* 传输大小 (字节) */
#define GPGPU_REG_DMA_CTRL          0x0414  /* DMA 控制寄存器 */
#define GPGPU_REG_DMA_STATUS        0x0418  /* DMA 状态寄存器 */

/* 线程上下文寄存器组 (0x1000 - 0x1FFF): GPU 线程读取自身 ID */
#define GPGPU_REG_THREAD_ID_X       0x1000  /* 线程在 Block 中的 X 索引 */
#define GPGPU_REG_THREAD_ID_Y       0x1004  /* 线程在 Block 中的 Y 索引 */
#define GPGPU_REG_THREAD_ID_Z       0x1008  /* 线程在 Block 中的 Z 索引 */
#define GPGPU_REG_BLOCK_ID_X        0x1010  /* Block 在 Grid 中的 X 索引 */
#define GPGPU_REG_BLOCK_ID_Y        0x1014  /* Block 在 Grid 中的 Y 索引 */
#define GPGPU_REG_BLOCK_ID_Z        0x1018  /* Block 在 Grid 中的 Z 索引 */
#define GPGPU_REG_WARP_ID           0x1020  /* Warp 在 Block 中的索引 */
#define GPGPU_REG_LANE_ID           0x1024  /* 线程在 Warp 中的索引 (0-31) */

/* 同步寄存器组 (0x2000 - 0x2FFF): 线程同步原语 */
#define GPGPU_REG_BARRIER           0x2000  /* 写任意值触发 Block 级屏障 */
#define GPGPU_REG_THREAD_MASK       0x2004  /* 活跃线程掩码 */

/*
 * ============================================================================
 * 寄存器位域定义
 * ============================================================================
 */

/* GLOBAL_CTRL 寄存器位 */
#define GPGPU_CTRL_ENABLE           (1 << 0)    /* 设备使能位 */
#define GPGPU_CTRL_RESET            (1 << 1)    /* 软复位 (自动清除) */

/* GLOBAL_STATUS 寄存器位 */
#define GPGPU_STATUS_READY          (1 << 0)    /* 设备就绪 */
#define GPGPU_STATUS_BUSY           (1 << 1)    /* 设备忙 (内核执行中) */
#define GPGPU_STATUS_ERROR          (1 << 2)    /* 发生错误 */

/* ERROR_STATUS 寄存器位 (写 1 清除) */
#define GPGPU_ERR_INVALID_CMD       (1 << 0)    /* 无效命令 */
#define GPGPU_ERR_VRAM_FAULT        (1 << 1)    /* 显存访问越界 */
#define GPGPU_ERR_KERNEL_FAULT      (1 << 2)    /* 内核执行错误 */
#define GPGPU_ERR_DMA_FAULT         (1 << 3)    /* DMA 传输错误 */

/* IRQ 位定义 */
#define GPGPU_IRQ_KERNEL_DONE       (1 << 0)    /* 内核执行完成中断 */
#define GPGPU_IRQ_DMA_DONE          (1 << 1)    /* DMA 传输完成中断 */
#define GPGPU_IRQ_ERROR             (1 << 2)    /* 错误中断 */

/* DMA_CTRL 寄存器位 */
#define GPGPU_DMA_START             (1 << 0)    /* 启动 DMA 传输 */
#define GPGPU_DMA_DIR_TO_VRAM       (0 << 1)    /* 方向: 主机 -> 显存 */
#define GPGPU_DMA_DIR_FROM_VRAM     (1 << 1)    /* 方向: 显存 -> 主机 */
#define GPGPU_DMA_IRQ_ENABLE        (1 << 2)    /* 完成时产生中断 */

/* DMA_STATUS 寄存器位 */
#define GPGPU_DMA_BUSY              (1 << 0)    /* DMA 忙 */
#define GPGPU_DMA_COMPLETE          (1 << 1)    /* DMA 完成 */
#define GPGPU_DMA_ERROR             (1 << 2)    /* DMA 错误 */

/*
 * ============================================================================
 * 设备标识值
 * ============================================================================
 */
#define GPGPU_DEV_ID_VALUE          0x47505055  /* "GPPU" in ASCII */
#define GPGPU_DEV_VERSION_VALUE     0x00010000  /* v1.0.0 */

/*
 * ============================================================================
 * MSI-X 配置
 * ============================================================================
 */
#define GPGPU_MSIX_VECTORS          4           /* MSI-X 向量数量 */
#define GPGPU_MSIX_VEC_KERNEL       0           /* 内核完成中断向量 */
#define GPGPU_MSIX_VEC_DMA          1           /* DMA 完成中断向量 */
#define GPGPU_MSIX_VEC_ERROR        2           /* 错误中断向量 */

/*
 * ============================================================================
 * 内核分发参数结构
 * ============================================================================
 * 用于记录当前配置的内核执行参数
 */
typedef struct GPGPUKernelParams {
    uint64_t kernel_addr;       /* 内核代码在 VRAM 中的地址 */
    uint64_t kernel_args;       /* 内核参数在 VRAM 中的地址 */
    uint32_t grid_dim[3];       /* Grid 维度 [X, Y, Z] */
    uint32_t block_dim[3];      /* Block 维度 [X, Y, Z] */
    uint32_t shared_mem_size;   /* 每个 Block 的共享内存大小 */
} GPGPUKernelParams;

/*
 * ============================================================================
 * DMA 状态结构
 * ============================================================================
 */
typedef struct GPGPUDMAState {
    uint64_t src_addr;          /* 源地址 */
    uint64_t dst_addr;          /* 目标地址 */
    uint32_t size;              /* 传输大小 */
    uint32_t ctrl;              /* 控制寄存器值 */
    uint32_t status;            /* 状态寄存器值 */
} GPGPUDMAState;

/*
 * ============================================================================
 * SIMT 执行上下文 (CTRL 设备核心)
 * ============================================================================
 * 用于跟踪当前执行线程的上下文信息
 * GPU 核心通过 MMIO 读取这些信息来获取自己的 thread_id 等
 *
 * 使用方式:
 * 1. Host 驱动在 dispatch 前设置 grid/block 维度
 * 2. 模拟执行时，设置当前的 thread_id/block_id
 * 3. GPU 线程通过读取 0x1000-0x1FFF 获取自己的 ID
 * 4. 写入 0x2000 触发 barrier 同步
 */
typedef struct GPGPUSIMTContext {
    /* 当前执行的线程位置 */
    uint32_t thread_id[3];      /* threadIdx.x/y/z */
    uint32_t block_id[3];       /* blockIdx.x/y/z */
    uint32_t warp_id;           /* 当前 warp ID */
    uint32_t lane_id;           /* 线程在 warp 中的位置 (0-31) */

    /* Barrier 同步状态 */
    uint32_t barrier_count;     /* 到达 barrier 的线程数 */
    uint32_t barrier_target;    /* 需要到达的线程总数 */
    bool barrier_active;        /* barrier 是否激活 */

    /* 活跃线程掩码 (用于分支分歧) */
    uint32_t thread_mask;       /* 32 位掩码，每位代表一个线程 */
} GPGPUSIMTContext;

/*
 * ============================================================================
 * 设备主状态结构
 * ============================================================================
 * 这是设备的核心数据结构，包含所有设备状态
 * 每个 GPGPU 设备实例都有一个独立的 GPGPUState
 */
struct GPGPUState {
    /*-- 父类 --*/
    PCIDevice parent_obj;           /* 必须是第一个成员，PCI 设备基类 */

    /*-- 内存区域 (对应 BAR 空间) --*/
    MemoryRegion ctrl_mmio;         /* BAR0: 控制寄存器 MMIO 区域 */
    MemoryRegion vram;              /* BAR2: 显存区域 */
    MemoryRegion doorbell_mmio;     /* BAR4: 门铃寄存器区域 */

    /*-- 设备配置 (可通过命令行参数设置) --*/
    uint32_t num_cus;               /* 计算单元数量 */
    uint32_t warps_per_cu;          /* 每个 CU 的 warp 数量 */
    uint32_t warp_size;             /* 每个 warp 的线程数 */
    uint64_t vram_size;             /* 显存大小 (字节) */

    /*-- 显存后端存储 --*/
    uint8_t *vram_ptr;              /* 显存数据指针 */

    /*-- 全局控制寄存器状态 --*/
    uint32_t global_ctrl;           /* GLOBAL_CTRL 寄存器值 */
    uint32_t global_status;         /* GLOBAL_STATUS 寄存器值 */
    uint32_t error_status;          /* ERROR_STATUS 寄存器值 */

    /*-- 中断状态 --*/
    uint32_t irq_enable;            /* IRQ_ENABLE 寄存器值 */
    uint32_t irq_status;            /* IRQ_STATUS 寄存器值 */

    /*-- 内核分发参数 --*/
    GPGPUKernelParams kernel;       /* 当前内核配置 */

    /*-- DMA 引擎状态 --*/
    GPGPUDMAState dma;              /* DMA 状态 */
    QEMUTimer *dma_timer;           /* DMA 完成定时器 (模拟传输延迟) */

    /*-- 内核执行状态 (用于模拟) --*/
    QEMUTimer *kernel_timer;        /* 内核执行完成定时器 */

    /*-- SIMT 执行上下文 (CTRL 设备) --*/
    GPGPUSIMTContext simt;          /* 当前线程的执行上下文 */
};

#endif /* HW_GPGPU_H */
