/*
 * QTest testcase for GPGPU device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "libqtest.h"
#include "libqos/qgraph.h"
#include "libqos/pci.h"

/*
 * GPGPU 设备寄存器定义 (从 hw/gpgpu/gpgpu.h 复制)
 * 为了测试独立性，这里重新定义需要的常量
 */

/* PCI 配置 */
#define GPGPU_VENDOR_ID         0x1234
#define GPGPU_DEVICE_ID         0x1337

/* 设备信息寄存器 (BAR0) */
#define GPGPU_REG_DEV_ID            0x0000
#define GPGPU_REG_DEV_VERSION       0x0004
#define GPGPU_REG_DEV_CAPS          0x0008
#define GPGPU_REG_VRAM_SIZE_LO      0x000C
#define GPGPU_REG_VRAM_SIZE_HI      0x0010

/* 全局控制寄存器 */
#define GPGPU_REG_GLOBAL_CTRL       0x0100
#define GPGPU_REG_GLOBAL_STATUS     0x0104
#define GPGPU_REG_ERROR_STATUS      0x0108

/* 中断寄存器 */
#define GPGPU_REG_IRQ_ENABLE        0x0200
#define GPGPU_REG_IRQ_STATUS        0x0204
#define GPGPU_REG_IRQ_ACK           0x0208

/* 内核分发寄存器 */
#define GPGPU_REG_GRID_DIM_X        0x0310
#define GPGPU_REG_GRID_DIM_Y        0x0314
#define GPGPU_REG_GRID_DIM_Z        0x0318
#define GPGPU_REG_BLOCK_DIM_X       0x031C
#define GPGPU_REG_BLOCK_DIM_Y       0x0320
#define GPGPU_REG_BLOCK_DIM_Z       0x0324

/* DMA 寄存器 */
#define GPGPU_REG_DMA_SRC_LO        0x0400
#define GPGPU_REG_DMA_SRC_HI        0x0404
#define GPGPU_REG_DMA_DST_LO        0x0408
#define GPGPU_REG_DMA_DST_HI        0x040C
#define GPGPU_REG_DMA_SIZE          0x0410
#define GPGPU_REG_DMA_CTRL          0x0414
#define GPGPU_REG_DMA_STATUS        0x0418

/* SIMT 上下文寄存器 (CTRL 设备) */
#define GPGPU_REG_THREAD_ID_X       0x1000
#define GPGPU_REG_THREAD_ID_Y       0x1004
#define GPGPU_REG_THREAD_ID_Z       0x1008
#define GPGPU_REG_BLOCK_ID_X        0x1010
#define GPGPU_REG_BLOCK_ID_Y        0x1014
#define GPGPU_REG_BLOCK_ID_Z        0x1018
#define GPGPU_REG_WARP_ID           0x1020
#define GPGPU_REG_LANE_ID           0x1024

/* 同步寄存器 */
#define GPGPU_REG_BARRIER           0x2000
#define GPGPU_REG_THREAD_MASK       0x2004

/* 内核地址寄存器 */
#define GPGPU_REG_KERNEL_ADDR_LO    0x0300
#define GPGPU_REG_KERNEL_ADDR_HI    0x0304
#define GPGPU_REG_DISPATCH          0x0330

/* 寄存器位定义 */
#define GPGPU_CTRL_ENABLE           (1 << 0)
#define GPGPU_CTRL_RESET            (1 << 1)
#define GPGPU_STATUS_READY          (1 << 0)
#define GPGPU_DMA_START             (1 << 0)
#define GPGPU_DMA_DIR_TO_VRAM       (0 << 1)
#define GPGPU_DMA_DIR_FROM_VRAM     (1 << 1)
#define GPGPU_DMA_COMPLETE          (1 << 1)

/* 设备标识值 */
#define GPGPU_DEV_ID_VALUE          0x47505055  /* "GPPU" */
#define GPGPU_DEV_VERSION_VALUE     0x00010000  /* v1.0.0 */

/* 默认配置 */
#define GPGPU_DEFAULT_VRAM_SIZE     (64 * 1024 * 1024)

typedef struct QGPGPU {
    QOSGraphObject obj;
    QPCIDevice dev;
    QPCIBar bar0;   /* 控制寄存器 */
    QPCIBar bar2;   /* VRAM */
} QGPGPU;

static void *gpgpu_get_driver(void *obj, const char *interface)
{
    QGPGPU *gpgpu = obj;

    if (!g_strcmp0(interface, "pci-device")) {
        return &gpgpu->dev;
    }

    fprintf(stderr, "%s not present in gpgpu\n", interface);
    g_assert_not_reached();
}

static void *gpgpu_create(void *pci_bus, QGuestAllocator *alloc, void *addr)
{
    QGPGPU *gpgpu = g_new0(QGPGPU, 1);
    QPCIBus *bus = pci_bus;

    qpci_device_init(&gpgpu->dev, bus, addr);
    gpgpu->obj.get_driver = gpgpu_get_driver;

    return &gpgpu->obj;
}

/*
 * 测试 1: 设备识别测试
 * 验证设备 ID 和版本号寄存器返回正确的值
 */
static void gpgpu_test_device_id(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t dev_id, version;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 读取设备 ID */
    dev_id = qpci_io_readl(pdev, bar0, GPGPU_REG_DEV_ID);
    g_assert_cmphex(dev_id, ==, GPGPU_DEV_ID_VALUE);

    /* 读取版本号 */
    version = qpci_io_readl(pdev, bar0, GPGPU_REG_DEV_VERSION);
    g_assert_cmphex(version, ==, GPGPU_DEV_VERSION_VALUE);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 2: VRAM 大小寄存器测试
 * 验证显存大小寄存器返回配置的值
 */
static void gpgpu_test_vram_size(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t vram_lo, vram_hi;
    uint64_t vram_size;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 读取 VRAM 大小 */
    vram_lo = qpci_io_readl(pdev, bar0, GPGPU_REG_VRAM_SIZE_LO);
    vram_hi = qpci_io_readl(pdev, bar0, GPGPU_REG_VRAM_SIZE_HI);
    vram_size = ((uint64_t)vram_hi << 32) | vram_lo;

    g_assert_cmpuint(vram_size, ==, GPGPU_DEFAULT_VRAM_SIZE);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 3: 全局控制寄存器测试
 * 验证设备使能和状态寄存器工作正常
 */
static void gpgpu_test_global_ctrl(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t ctrl, status;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 读取初始状态 */
    status = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_STATUS);
    g_assert_cmpuint(status & GPGPU_STATUS_READY, ==, GPGPU_STATUS_READY);

    /* 测试写入控制寄存器 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);
    ctrl = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_CTRL);
    g_assert_cmpuint(ctrl & GPGPU_CTRL_ENABLE, ==, GPGPU_CTRL_ENABLE);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 4: Grid/Block 维度寄存器测试
 * 验证 kernel dispatch 配置寄存器可读写
 */
static void gpgpu_test_dispatch_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 写入 Grid 维度 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_X, 64);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Y, 32);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Z, 1);

    /* 验证读回 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GRID_DIM_X);
    g_assert_cmpuint(val, ==, 64);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GRID_DIM_Y);
    g_assert_cmpuint(val, ==, 32);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GRID_DIM_Z);
    g_assert_cmpuint(val, ==, 1);

    /* 写入 Block 维度 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_X, 256);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Z, 1);

    /* 验证读回 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_BLOCK_DIM_X);
    g_assert_cmpuint(val, ==, 256);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 5: VRAM 读写测试
 * 验证显存区域 (BAR2) 可正确读写
 */
static void gpgpu_test_vram_access(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar2;
    uint32_t test_pattern = 0xDEADBEEF;
    uint32_t read_val;

    qpci_device_enable(pdev);
    bar2 = qpci_iomap(pdev, 2, NULL);

    /* 写入测试数据 */
    qpci_io_writel(pdev, bar2, 0x0, test_pattern);
    qpci_io_writel(pdev, bar2, 0x100, 0x12345678);
    qpci_io_writel(pdev, bar2, 0x1000, 0xCAFEBABE);

    /* 验证读回 */
    read_val = qpci_io_readl(pdev, bar2, 0x0);
    g_assert_cmphex(read_val, ==, test_pattern);

    read_val = qpci_io_readl(pdev, bar2, 0x100);
    g_assert_cmphex(read_val, ==, 0x12345678);

    read_val = qpci_io_readl(pdev, bar2, 0x1000);
    g_assert_cmphex(read_val, ==, 0xCAFEBABE);

    qpci_iounmap(pdev, bar2);
}

/*
 * 测试 6: DMA 寄存器测试
 * 验证 DMA 控制寄存器可正确配置
 */
static void gpgpu_test_dma_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 配置 DMA 源地址 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_SRC_LO, 0x1000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_SRC_HI, 0x0);

    /* 配置 DMA 目标地址 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_DST_LO, 0x2000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_DST_HI, 0x0);

    /* 配置传输大小 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_SIZE, 4096);

    /* 验证读回 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_DMA_SRC_LO);
    g_assert_cmphex(val, ==, 0x1000);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_DMA_DST_LO);
    g_assert_cmphex(val, ==, 0x2000);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_DMA_SIZE);
    g_assert_cmpuint(val, ==, 4096);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 7: 中断寄存器测试
 * 验证中断使能和状态寄存器可正确操作
 */
static void gpgpu_test_irq_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 写入中断使能 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_IRQ_ENABLE, 0x7);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_IRQ_ENABLE);
    g_assert_cmphex(val, ==, 0x7);

    /* 读取中断状态 (应该为 0) */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_IRQ_STATUS);
    g_assert_cmphex(val, ==, 0x0);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 8: SIMT 线程 ID 寄存器测试 (CTRL 设备)
 * 验证 thread_id 寄存器可正确读写
 */
static void gpgpu_test_thread_id_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 初始值应为 0 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_ID_X);
    g_assert_cmpuint(val, ==, 0);

    /* 写入 thread_id.x */
    qpci_io_writel(pdev, bar0, GPGPU_REG_THREAD_ID_X, 15);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_ID_X);
    g_assert_cmpuint(val, ==, 15);

    /* 写入 thread_id.y */
    qpci_io_writel(pdev, bar0, GPGPU_REG_THREAD_ID_Y, 7);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_ID_Y);
    g_assert_cmpuint(val, ==, 7);

    /* 写入 thread_id.z */
    qpci_io_writel(pdev, bar0, GPGPU_REG_THREAD_ID_Z, 3);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_ID_Z);
    g_assert_cmpuint(val, ==, 3);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 9: SIMT Block ID 寄存器测试 (CTRL 设备)
 * 验证 block_id 寄存器可正确读写
 */
static void gpgpu_test_block_id_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 写入 block_id.x */
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_ID_X, 63);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_BLOCK_ID_X);
    g_assert_cmpuint(val, ==, 63);

    /* 写入 block_id.y */
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_ID_Y, 31);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_BLOCK_ID_Y);
    g_assert_cmpuint(val, ==, 31);

    /* 写入 block_id.z */
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_ID_Z, 1);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_BLOCK_ID_Z);
    g_assert_cmpuint(val, ==, 1);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 10: SIMT Warp/Lane ID 寄存器测试 (CTRL 设备)
 * 验证 warp_id 和 lane_id 寄存器可正确读写
 */
static void gpgpu_test_warp_lane_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 写入 warp_id */
    qpci_io_writel(pdev, bar0, GPGPU_REG_WARP_ID, 3);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_WARP_ID);
    g_assert_cmpuint(val, ==, 3);

    /* 写入 lane_id (0-31 范围) */
    qpci_io_writel(pdev, bar0, GPGPU_REG_LANE_ID, 17);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_LANE_ID);
    g_assert_cmpuint(val, ==, 17);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 11: SIMT 线程掩码寄存器测试 (CTRL 设备)
 * 验证 thread_mask 寄存器可正确读写
 */
static void gpgpu_test_thread_mask_reg(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 初始掩码应为 0 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_MASK);
    g_assert_cmphex(val, ==, 0x0);

    /* 写入掩码: 所有 32 个线程活跃 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_THREAD_MASK, 0xFFFFFFFF);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_MASK);
    g_assert_cmphex(val, ==, 0xFFFFFFFF);

    /* 写入掩码: 只有前 16 个线程活跃 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_THREAD_MASK, 0x0000FFFF);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_MASK);
    g_assert_cmphex(val, ==, 0x0000FFFF);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 12: SIMT 上下文复位测试 (CTRL 设备)
 * 验证软复位会清除 SIMT 上下文
 */
static void gpgpu_test_simt_reset(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 设置一些 SIMT 上下文 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_THREAD_ID_X, 123);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_ID_X, 456);
    qpci_io_writel(pdev, bar0, GPGPU_REG_WARP_ID, 7);
    qpci_io_writel(pdev, bar0, GPGPU_REG_THREAD_MASK, 0xDEADBEEF);

    /* 触发软复位 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_RESET);

    /* 验证 SIMT 上下文被清除 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_ID_X);
    g_assert_cmpuint(val, ==, 0);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_BLOCK_ID_X);
    g_assert_cmpuint(val, ==, 0);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_WARP_ID);
    g_assert_cmpuint(val, ==, 0);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_THREAD_MASK);
    g_assert_cmphex(val, ==, 0x0);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 13: 简单内核执行测试
 * 验证 RISC-V 指令解释器能正确执行简单的 kernel
 *
 * Kernel 功能: C[thread_id] = thread_id
 * 每个线程将自己的 thread_id 写入输出数组
 */

/*
 * 简单的 RISC-V 内核代码 (RV32I)
 *
 * 地址布局:
 *   0x0000: kernel 代码
 *   0x1000: 输出数组 C
 *
 * 通过 mhartid CSR 获取线程 ID:
 *   mhartid 位域: [block(19)|warp(8)|tid(5)]
 *
 * 伪代码:
 *   t1 = csrr(mhartid)    // 读取 mhartid
 *   t1 = t1 & 0x1F        // 提取 thread_id (lane within warp)
 *   t2 = t1 << 2          // byte offset
 *   t3 = 0x1000           // 输出地址基址
 *   t3 = t3 + t2          // &C[thread_id]
 *   store(t3, t1)         // C[thread_id] = thread_id
 *   ebreak                // 停止
 */
static const uint32_t simple_kernel[] = {
    0xF1402373,  /* csrrs x6, mhartid, x0  ; t1 = mhartid */
    0x01F37313,  /* andi  x6, x6, 0x1F     ; t1 = thread_id (lane) */
    0x00231393,  /* slli  x7, x6, 2        ; t2 = thread_id * 4 */
    0x00001E37,  /* lui   x28, 1           ; t3 = 0x1000 */
    0x007E0E33,  /* add   x28, x28, x7     ; t3 = &C[thread_id] */
    0x006E2023,  /* sw    x6, 0(x28)       ; C[thread_id] = thread_id */
    0x00100073,  /* ebreak                 ; stop */
};

static void gpgpu_test_kernel_exec(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0, bar2;
    uint32_t val;
    uint32_t num_threads = 8;  /* 测试 8 个线程 */

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);
    bar2 = qpci_iomap(pdev, 2, NULL);

    /* 1. 使能设备 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);

    /* 2. 上传内核代码到 VRAM (地址 0x0000) */
    for (size_t i = 0; i < sizeof(simple_kernel) / sizeof(simple_kernel[0]); i++) {
        qpci_io_writel(pdev, bar2, i * 4, simple_kernel[i]);
    }

    /* 3. 清零输出区域 (地址 0x1000) */
    for (uint32_t i = 0; i < num_threads; i++) {
        qpci_io_writel(pdev, bar2, 0x1000 + i * 4, 0xDEADBEEF);
    }

    /* 4. 配置内核参数 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_X, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Z, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_X, num_threads);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Z, 1);

    /* 5. 触发内核执行 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DISPATCH, 1);

    /* 6. 检查执行完成 (设备不再忙) */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_STATUS);
    g_assert_cmpuint(val & GPGPU_STATUS_READY, ==, GPGPU_STATUS_READY);

    /* 7. 验证输出结果: C[i] 应该等于 i */
    for (uint32_t i = 0; i < num_threads; i++) {
        val = qpci_io_readl(pdev, bar2, 0x1000 + i * 4); 
        g_assert_cmpuint(val, ==, i);
    }

    qpci_iounmap(pdev, bar0);
    qpci_iounmap(pdev, bar2);
}

/*
 * 测试 14: 浮点内核执行测试
 * 验证 RV32F 浮点指令能正确执行
 *
 * Kernel 功能: output[tid] = (int)(tid * 2.0 + 1.0)
 * 期望结果: 0→1, 1→3, 2→5, 3→7, ...
 */
static const uint32_t fp_kernel[] = {
    0xF1402373,  /* csrrs  x6, mhartid, x0    ; x6 = mhartid */
    0x01F37313,  /* andi   x6, x6, 0x1F       ; x6 = tid */
    0xD00300D3,  /* fcvt.s.w f1, x6            ; f1 = (float)tid */
    0x00200493,  /* addi   x9, x0, 2           ; x9 = 2 */
    0xD0048153,  /* fcvt.s.w f2, x9            ; f2 = 2.0 */
    0x00100493,  /* addi   x9, x0, 1           ; x9 = 1 */
    0xD00481D3,  /* fcvt.s.w f3, x9            ; f3 = 1.0 */
    0x10208253,  /* fmul.s f4, f1, f2          ; f4 = tid * 2.0 */
    0x003202D3,  /* fadd.s f5, f4, f3          ; f5 = tid * 2.0 + 1.0 */
    0xC00293D3,  /* fcvt.w.s x7, f5, RTZ       ; x7 = (int)result */
    0x00231413,  /* slli   x8, x6, 2           ; x8 = tid * 4 */
    0x00001E37,  /* lui    x28, 1              ; x28 = 0x1000 */
    0x008E0E33,  /* add    x28, x28, x8        ; x28 = &output[tid] */
    0x007E2023,  /* sw     x7, 0(x28)          ; output[tid] = result */
    0x00100073,  /* ebreak                     ; stop */
};

static void gpgpu_test_fp_kernel_exec(void *obj, void *data,
                                       QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0, bar2;
    uint32_t val;
    uint32_t num_threads = 8;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);
    bar2 = qpci_iomap(pdev, 2, NULL);

    /* 1. 使能设备 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);

    /* 2. 上传内核代码到 VRAM (地址 0x0000) */
    for (size_t i = 0; i < sizeof(fp_kernel) / sizeof(fp_kernel[0]); i++) {
        qpci_io_writel(pdev, bar2, i * 4, fp_kernel[i]);
    }

    /* 3. 清零输出区域 (地址 0x1000) */
    for (uint32_t i = 0; i < num_threads; i++) {
        qpci_io_writel(pdev, bar2, 0x1000 + i * 4, 0xDEADBEEF);
    }

    /* 4. 配置内核参数 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_X, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Z, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_X, num_threads);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Z, 1);

    /* 5. 触发内核执行 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DISPATCH, 1);

    /* 6. 检查执行完成 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_STATUS);
    g_assert_cmpuint(val & GPGPU_STATUS_READY, ==, GPGPU_STATUS_READY);

    /* 7. 验证输出结果: output[i] == 2*i + 1 */
    for (uint32_t i = 0; i < num_threads; i++) {
        val = qpci_io_readl(pdev, bar2, 0x1000 + i * 4);
        g_assert_cmpuint(val, ==, 2 * i + 1);
    }

    qpci_iounmap(pdev, bar0);
    qpci_iounmap(pdev, bar2);
}

/*
 * 测试 15: 低精度浮点转换指令测试
 * 验证 BF16 和 FP8-E4M3 往返转换精度
 *
 * Kernel: 对每个线程:
 *   BF16 round-trip:  42 → f32 → bf16 → f32 → int  (期望 42)
 *   E4M3 round-trip:   2 → f32 → e4m3 → f32 → int  (期望 2)
 *   output[tid*2]   = bf16 结果
 *   output[tid*2+1] = e4m3 结果
 */
static const uint32_t lp_convert_kernel[] = {
    0xF1402373,  /* csrrs  x6, mhartid, x0    ; x6 = mhartid           */
    0x01F37313,  /* andi   x6, x6, 0x1F       ; x6 = tid               */

    /* BF16 round-trip: 42 → f32 → bf16 → f32 → int */
    0x02A00493,  /* addi   x9, x0, 42         ; x9 = 42                */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 42.0             */
    0x44108153,  /* fcvt.bf16.s f2, f1         ; f2[15:0] = bf16(42.0) */
    0x440101D3,  /* fcvt.s.bf16 f3, f2         ; f3 = f32(bf16)        */
    0xC0019553,  /* fcvt.w.s x10, f3, rtz      ; x10 = (int)f3 = 42   */

    /* E4M3 round-trip: 2 → f32 → e4m3 → f32 → int */
    0x00200493,  /* addi   x9, x0, 2          ; x9 = 2                 */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 2.0              */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2[7:0] = e4m3(2.0)  */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = f32(e4m3(2.0))  */
    0xC00195D3,  /* fcvt.w.s x11, f3, rtz      ; x11 = (int)f3 = 2    */

    /* Store: output[tid*2]=x10, output[tid*2+1]=x11 */
    0x00331413,  /* slli   x8, x6, 3          ; x8 = tid * 8           */
    0x00001E37,  /* lui    x28, 1             ; x28 = 0x1000            */
    0x008E0E33,  /* add    x28, x28, x8       ; x28 = &output[tid*2]   */
    0x00AE2023,  /* sw     x10, 0(x28)        ; output[tid*2] = 42     */
    0x00BE2223,  /* sw     x11, 4(x28)        ; output[tid*2+1] = 2    */
    0x00100073,  /* ebreak                    ; stop                    */
};

static void gpgpu_test_lp_convert(void *obj, void *data,
                                   QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0, bar2;
    uint32_t val;
    uint32_t num_threads = 4;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);
    bar2 = qpci_iomap(pdev, 2, NULL);

    /* 1. 使能设备 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);

    /* 2. 上传内核代码到 VRAM */
    for (size_t i = 0; i < sizeof(lp_convert_kernel) /
                            sizeof(lp_convert_kernel[0]); i++) {
        qpci_io_writel(pdev, bar2, i * 4, lp_convert_kernel[i]);
    }

    /* 3. 清零输出区域 (tid*2 个 uint32, 即 num_threads*2) */
    for (uint32_t i = 0; i < num_threads * 2; i++) {
        qpci_io_writel(pdev, bar2, 0x1000 + i * 4, 0xDEADBEEF);
    }

    /* 4. 配置内核参数 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_X, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Z, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_X, num_threads);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Z, 1);

    /* 5. 触发内核执行 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DISPATCH, 1);

    /* 6. 检查执行完成 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_STATUS);
    g_assert_cmpuint(val & GPGPU_STATUS_READY, ==, GPGPU_STATUS_READY);

    /* 7. 验证输出结果 */
    for (uint32_t i = 0; i < num_threads; i++) {
        /* BF16 round-trip: 42 → bf16 → f32 → 42 */
        val = qpci_io_readl(pdev, bar2, 0x1000 + i * 8);
        g_assert_cmpuint(val, ==, 42);

        /* E4M3 round-trip: 2 → e4m3 → f32 → 2 */
        val = qpci_io_readl(pdev, bar2, 0x1000 + i * 8 + 4);
        g_assert_cmpuint(val, ==, 2);
    }

    qpci_iounmap(pdev, bar0);
    qpci_iounmap(pdev, bar2);
}

/*
 * 测试 16: E5M2/E2M1 格式覆盖 + 负数测试
 * 验证 E5M2 / E2M1 往返转换，以及 BF16/E4M3 负数转换
 *
 * output[0] = e5m2  round-trip(4)   → 4
 * output[1] = e2m1  round-trip(2)   → 2
 * output[2] = bf16  round-trip(-3)  → -3
 * output[3] = e4m3  round-trip(-2)  → -2
 */
static const uint32_t lp_convert_e5m2_e2m1_kernel[] = {
    0xF1402373,  /* csrrs  x6, mhartid, x0    ; x6 = mhartid           */
    0x01F37313,  /* andi   x6, x6, 0x1F       ; x6 = tid               */

    /* E5M2 round-trip: 4 → f32 → e5m2 → f32 → int */
    0x00400493,  /* addi   x9, x0, 4          ; x9 = 4                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 4.0              */
    0x48308153,  /* fcvt.e5m2.s f2, f1         ; f2 = e5m2(4.0)        */
    0x482101D3,  /* fcvt.s.e5m2 f3, f2         ; f3 = f32(e5m2)        */
    0xC0019553,  /* fcvt.w.s x10, f3, rtz      ; x10 = 4               */

    /* E2M1 round-trip: 2 → f32 → e2m1 → f32 → int */
    0x00200493,  /* addi   x9, x0, 2          ; x9 = 2                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 2.0              */
    0x4C108153,  /* fcvt.e2m1.s f2, f1         ; f2 = e2m1(2.0)        */
    0x4C0101D3,  /* fcvt.s.e2m1 f3, f2         ; f3 = f32(e2m1)        */
    0xC00195D3,  /* fcvt.w.s x11, f3, rtz      ; x11 = 2               */

    /* BF16 round-trip: -3 → f32 → bf16 → f32 → int */
    0xFFD00493,  /* addi   x9, x0, -3         ; x9 = -3                 */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = -3.0             */
    0x44108153,  /* fcvt.bf16.s f2, f1         ; f2 = bf16(-3.0)        */
    0x440101D3,  /* fcvt.s.bf16 f3, f2         ; f3 = f32(bf16)         */
    0xC0019653,  /* fcvt.w.s x12, f3, rtz      ; x12 = -3              */

    /* E4M3 round-trip: -2 → f32 → e4m3 → f32 → int */
    0xFFE00493,  /* addi   x9, x0, -2         ; x9 = -2                 */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = -2.0             */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(-2.0)       */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = f32(e4m3)        */
    0xC00196D3,  /* fcvt.w.s x13, f3, rtz      ; x13 = -2              */

    /* Store 4 results: tid * 16 offset */
    0x00431413,  /* slli   x8, x6, 4          ; x8 = tid * 16           */
    0x00001E37,  /* lui    x28, 1             ; x28 = 0x1000             */
    0x008E0E33,  /* add    x28, x28, x8       ; x28 = base + offset     */
    0x00AE2023,  /* sw     x10, 0(x28)        ; output[0] = 4           */
    0x00BE2223,  /* sw     x11, 4(x28)        ; output[1] = 2           */
    0x00CE2423,  /* sw     x12, 8(x28)        ; output[2] = -3          */
    0x00DE2623,  /* sw     x13, 12(x28)       ; output[3] = -2          */
    0x00100073,  /* ebreak                    ; stop                     */
};

static void gpgpu_test_lp_convert_e5m2_e2m1(void *obj, void *data,
                                              QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0, bar2;
    uint32_t val;
    uint32_t num_threads = 1;
    int32_t expected[] = { 4, 2, -3, -2 };

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);
    bar2 = qpci_iomap(pdev, 2, NULL);

    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);

    for (size_t i = 0; i < sizeof(lp_convert_e5m2_e2m1_kernel) /
                            sizeof(lp_convert_e5m2_e2m1_kernel[0]); i++) {
        qpci_io_writel(pdev, bar2, i * 4, lp_convert_e5m2_e2m1_kernel[i]);
    }

    for (uint32_t i = 0; i < 4; i++) {
        qpci_io_writel(pdev, bar2, 0x1000 + i * 4, 0xDEADBEEF);
    }

    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_X, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Z, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_X, num_threads);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Z, 1);

    qpci_io_writel(pdev, bar0, GPGPU_REG_DISPATCH, 1);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_STATUS);
    g_assert_cmpuint(val & GPGPU_STATUS_READY, ==, GPGPU_STATUS_READY);

    for (int i = 0; i < 4; i++) {
        val = qpci_io_readl(pdev, bar2, 0x1000 + i * 4);
        g_assert_cmpint((int32_t)val, ==, expected[i]);
    }

    qpci_iounmap(pdev, bar0);
    qpci_iounmap(pdev, bar2);
}

/*
 * 测试 17: 零值、溢出饱和、Inf 饱和测试
 * 验证零值正确转换、超出格式范围的值被饱和到最大值、Inf 也被饱和
 *
 * output[0] = e4m3 round-trip(0)    → 0
 * output[1] = e2m1 round-trip(0)    → 0
 * output[2] = e2m1 round-trip(100)  → 6   (E2M1 max, saturated)
 * output[3] = e4m3 round-trip(1000) → 448 (E4M3 max, saturated)
 * output[4] = e4m3(+Inf)           → 448 (Inf → E4M3 max, saturated)
 */
static const uint32_t lp_convert_saturate_kernel[] = {
    0xF1402373,  /* csrrs  x6, mhartid, x0    ; x6 = mhartid           */
    0x01F37313,  /* andi   x6, x6, 0x1F       ; x6 = tid               */

    /* E4M3 round-trip: 0 → f32 → e4m3 → f32 → int */
    0x00000493,  /* addi   x9, x0, 0          ; x9 = 0                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 0.0              */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(0.0)        */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = f32(e4m3)        */
    0xC0019553,  /* fcvt.w.s x10, f3, rtz      ; x10 = 0               */

    /* E2M1 round-trip: 0 → f32 → e2m1 → f32 → int */
    0x00000493,  /* addi   x9, x0, 0          ; x9 = 0                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 0.0              */
    0x4C108153,  /* fcvt.e2m1.s f2, f1         ; f2 = e2m1(0.0)        */
    0x4C0101D3,  /* fcvt.s.e2m1 f3, f2         ; f3 = f32(e2m1)        */
    0xC00195D3,  /* fcvt.w.s x11, f3, rtz      ; x11 = 0               */

    /* E2M1 round-trip: 100 → saturate → 6 */
    0x06400493,  /* addi   x9, x0, 100        ; x9 = 100                */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 100.0            */
    0x4C108153,  /* fcvt.e2m1.s f2, f1         ; f2 = e2m1(100.0) sat  */
    0x4C0101D3,  /* fcvt.s.e2m1 f3, f2         ; f3 = 6.0              */
    0xC0019653,  /* fcvt.w.s x12, f3, rtz      ; x12 = 6               */

    /* E4M3 round-trip: 1000 → saturate → 448 */
    0x3E800493,  /* addi   x9, x0, 1000       ; x9 = 1000              */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 1000.0           */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(1000) sat   */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = 448.0            */
    0xC00196D3,  /* fcvt.w.s x13, f3, rtz      ; x13 = 448             */

    /* E4M3 of +Inf → saturate → 448 */
    0x7F8004B7,  /* lui    x9, 0x7F800        ; x9 = 0x7F800000 (+Inf) */
    0xF00480D3,  /* fmv.w.x f1, x9            ; f1 = +Inf              */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(Inf) sat    */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = 448.0            */
    0xC0019753,  /* fcvt.w.s x14, f3, rtz      ; x14 = 448             */

    /* Store 5 results: tid * 20 offset (tid=0 → base=0x1000) */
    0x00001E37,  /* lui    x28, 1             ; x28 = 0x1000             */
    0x00AE2023,  /* sw     x10, 0(x28)        ; output[0] = 0           */
    0x00BE2223,  /* sw     x11, 4(x28)        ; output[1] = 0           */
    0x00CE2423,  /* sw     x12, 8(x28)        ; output[2] = 6           */
    0x00DE2623,  /* sw     x13, 12(x28)       ; output[3] = 448         */
    0x00EE2823,  /* sw     x14, 16(x28)       ; output[4] = 448         */
    0x00100073,  /* ebreak                    ; stop                     */
};

static void gpgpu_test_lp_convert_saturate(void *obj, void *data,
                                            QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0, bar2;
    uint32_t val;
    uint32_t num_threads = 1;
    int32_t expected[] = { 0, 0, 6, 448, 448 };
    int num_results = 5;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);
    bar2 = qpci_iomap(pdev, 2, NULL);

    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);

    for (size_t i = 0; i < sizeof(lp_convert_saturate_kernel) /
                            sizeof(lp_convert_saturate_kernel[0]); i++) {
        qpci_io_writel(pdev, bar2, i * 4, lp_convert_saturate_kernel[i]);
    }

    for (int i = 0; i < num_results; i++) {
        qpci_io_writel(pdev, bar2, 0x1000 + i * 4, 0xDEADBEEF);
    }

    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_X, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Z, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_X, num_threads);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Z, 1);

    qpci_io_writel(pdev, bar0, GPGPU_REG_DISPATCH, 1);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_STATUS);
    g_assert_cmpuint(val & GPGPU_STATUS_READY, ==, GPGPU_STATUS_READY);

    for (int i = 0; i < num_results; i++) {
        val = qpci_io_readl(pdev, bar2, 0x1000 + i * 4);
        g_assert_cmpint((int32_t)val, ==, expected[i]);
    }

    qpci_iounmap(pdev, bar0);
    qpci_iounmap(pdev, bar2);
}

static void gpgpu_register_nodes(void)
{
    QOSGraphEdgeOptions opts = {
        .extra_device_opts = "addr=04.0",
    };

    add_qpci_address(&opts, &(QPCIAddress) {
        .devfn = QPCI_DEVFN(4, 0),
        .vendor_id = GPGPU_VENDOR_ID,
        .device_id = GPGPU_DEVICE_ID,
    });

    /* 创建驱动节点 */
    qos_node_create_driver("gpgpu", gpgpu_create);
    qos_node_consumes("gpgpu", "pci-bus", &opts);
    qos_node_produces("gpgpu", "pci-device");

    /* 注册测试用例 */
    qos_add_test("device-id", "gpgpu", gpgpu_test_device_id, NULL);
    qos_add_test("vram-size", "gpgpu", gpgpu_test_vram_size, NULL);
    qos_add_test("global-ctrl", "gpgpu", gpgpu_test_global_ctrl, NULL);
    qos_add_test("dispatch-regs", "gpgpu", gpgpu_test_dispatch_regs, NULL);
    qos_add_test("vram-access", "gpgpu", gpgpu_test_vram_access, NULL);
    qos_add_test("dma-regs", "gpgpu", gpgpu_test_dma_regs, NULL);
    qos_add_test("irq-regs", "gpgpu", gpgpu_test_irq_regs, NULL);

    /* CTRL 设备测试 (SIMT 上下文) */
    qos_add_test("simt-thread-id", "gpgpu", gpgpu_test_thread_id_regs, NULL);
    qos_add_test("simt-block-id", "gpgpu", gpgpu_test_block_id_regs, NULL);
    qos_add_test("simt-warp-lane", "gpgpu", gpgpu_test_warp_lane_regs, NULL);
    qos_add_test("simt-thread-mask", "gpgpu", gpgpu_test_thread_mask_reg, NULL);
    qos_add_test("simt-reset", "gpgpu", gpgpu_test_simt_reset, NULL);

    /* 内核执行测试 */
    qos_add_test("kernel-exec", "gpgpu", gpgpu_test_kernel_exec, NULL);
    qos_add_test("fp-kernel-exec", "gpgpu", gpgpu_test_fp_kernel_exec, NULL);
    qos_add_test("lp-convert", "gpgpu", gpgpu_test_lp_convert, NULL);
    qos_add_test("lp-convert-e5m2-e2m1", "gpgpu",
                 gpgpu_test_lp_convert_e5m2_e2m1, NULL);
    qos_add_test("lp-convert-saturate", "gpgpu",
                 gpgpu_test_lp_convert_saturate, NULL);
}

libqos_init(gpgpu_register_nodes);
