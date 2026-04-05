
GPGPU device
============

..
   Copyright (c) 2024-2025
   This document is licensed under the GPLv2 (or later).

This is an educational GPGPU (General-Purpose Graphics Processing Unit) device
for learning GPU architecture, driver development, and parallel programming.
The device implements a simplified SIMT (Single Instruction, Multiple Threads)
execution model inspired by Vortex GPGPU, using standard RISC-V instructions.

Overview
--------

The GPGPU device is a PCIe device that provides:

- SIMT execution model with configurable warps and threads
- Dedicated video memory (VRAM) for GPU computations
- DMA engine for host-device data transfer
- Command queue for kernel dispatch
- MSI/MSI-X interrupt support

The device reuses the RISC-V RV32IMF instruction set, avoiding custom GPU
instructions. Thread divergence and synchronization are handled through
a dedicated CTRL (Control) unit accessible via memory-mapped I/O.

Architecture
------------

::

    +------------------------------------------------------------------+
    |                        GPGPU Device                              |
    |  +------------------------------------------------------------+  |
    |  |                    PCIe Interface                          |  |
    |  +------------------------------------------------------------+  |
    |         |              |              |              |           |
    |         v              v              v              v           |
    |  +-----------+  +-----------+  +-----------+  +-----------+     |
    |  |  BAR 0    |  |  BAR 2    |  |  BAR 4    |  |    DMA    |     |
    |  |  Control  |  |   VRAM    |  | Doorbell  |  |  Engine   |     |
    |  |   Regs    |  |           |  |           |  |           |     |
    |  +-----------+  +-----------+  +-----------+  +-----------+     |
    |         |              |              |              |           |
    |         v              v              v              v           |
    |  +------------------------------------------------------------+  |
    |  |                  Compute Unit Array                        |  |
    |  |  +----------+  +----------+  +----------+  +----------+    |  |
    |  |  |   CU 0   |  |   CU 1   |  |   CU 2   |  |   CU N   |    |  |
    |  |  | +------+ |  | +------+ |  | +------+ |  | +------+ |    |  |
    |  |  | |Warp 0| |  | |Warp 0| |  | |Warp 0| |  | |Warp 0| |    |  |
    |  |  | +------+ |  | +------+ |  | +------+ |  | +------+ |    |  |
    |  |  | |Warp 1| |  | |Warp 1| |  | |Warp 1| |  | |Warp 1| |    |  |
    |  |  | +------+ |  | +------+ |  | +------+ |  | +------+ |    |  |
    |  |  +----------+  +----------+  +----------+  +----------+    |  |
    |  +------------------------------------------------------------+  |
    |                             |                                    |
    |                             v                                    |
    |  +------------------------------------------------------------+  |
    |  |                    Shared Memory                           |  |
    |  +------------------------------------------------------------+  |
    +------------------------------------------------------------------+

SIMT Execution Model
--------------------

Thread Hierarchy
~~~~~~~~~~~~~~~~

The GPGPU uses a hierarchical thread organization:

Grid
    A grid represents the entire computation, consisting of multiple thread
    blocks. Grid dimensions are specified by ``GRID_DIM_X``, ``GRID_DIM_Y``,
    and ``GRID_DIM_Z`` registers.

Thread Block (Workgroup)
    A thread block is a group of threads that can cooperate via shared memory
    and synchronization barriers. Block dimensions are specified by
    ``BLOCK_DIM_X``, ``BLOCK_DIM_Y``, and ``BLOCK_DIM_Z`` registers.

Warp
    A warp is the basic scheduling unit. All threads in a warp execute the
    same instruction in lockstep (SIMT). The warp size is fixed at 32 threads
    (configurable via device property).

Thread
    The smallest execution unit. Each thread has its own register file and
    program counter (for divergence handling).

Thread Divergence
~~~~~~~~~~~~~~~~~

When threads in a warp take different branch paths, the hardware handles
divergence by:

1. Executing each path sequentially with appropriate thread masks
2. Reconverging threads at the immediate post-dominator
3. Using a divergence stack to track active thread masks

Special Registers (via CTRL unit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Threads can read their identity through CTRL MMIO reads:

- ``THREAD_ID_X/Y/Z``: Thread index within block
- ``BLOCK_ID_X/Y/Z``: Block index within grid
- ``BLOCK_DIM_X/Y/Z``: Block dimensions
- ``GRID_DIM_X/Y/Z``: Grid dimensions
- ``WARP_ID``: Warp index within block
- ``LANE_ID``: Thread index within warp (0-31)

Command Line Options
--------------------

``-device gpgpu[,vram_size=SIZE][,num_cus=N][,warps_per_cu=M][,warp_size=W]``

``vram_size``
    Video memory size in bytes. Default: 64 MiB. Must be power of 2.

``num_cus``
    Number of compute units. Default: 4.

``warps_per_cu``
    Number of warps per compute unit. Default: 4.

``warp_size``
    Number of threads per warp. Default: 32. Must be power of 2.

PCI Configuration
-----------------

PCI IDs
~~~~~~~

=============== ================
Field           Value
=============== ================
Vendor ID       0x1234 (QEMU)
Device ID       0xGPGP (TBD)
Revision        0x01
Class Code      0x030200 (3D Controller)
Subsystem VID   0x1234
Subsystem ID    0x1100
=============== ================

PCI Capabilities
~~~~~~~~~~~~~~~~

- MSI-X capability (64 vectors)
- PCIe capability (Gen3 x16)
- Power Management capability

BAR Layout
~~~~~~~~~~

======= ============= ============= =================================
BAR     Type          Size          Description
======= ============= ============= =================================
BAR 0   Memory, 64b   1 MiB         Control registers (MMIO)
BAR 2   Memory, 64b   64-256 MiB    Video RAM (prefetchable)
BAR 4   Memory, 32b   64 KiB        Doorbell registers
======= ============= ============= =================================

Control Registers (BAR 0)
-------------------------

BAR 0 contains the main control and status registers. All registers are
little-endian. Access size requirements are specified for each register.

Device Information (0x0000 - 0x00FF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Offset
     - Size
     - Access
     - Description
   * - 0x0000
     - 4
     - RO
     - **DEV_ID**: Device identification.
       Value: ``0x47505055`` ("GPPU" in ASCII)
   * - 0x0004
     - 4
     - RO
     - **DEV_VERSION**: Device version.
       Format: ``[31:16]=major, [15:8]=minor, [7:0]=patch``
   * - 0x0008
     - 4
     - RO
     - **DEV_CAPS**: Device capabilities.

       - Bits [7:0]: Number of compute units
       - Bits [15:8]: Warps per compute unit
       - Bits [23:16]: Threads per warp (warp size)
       - Bits [31:24]: Reserved
   * - 0x000C
     - 4
     - RO
     - **VRAM_SIZE_LO**: VRAM size low 32 bits (in bytes)
   * - 0x0010
     - 4
     - RO
     - **VRAM_SIZE_HI**: VRAM size high 32 bits

Global Control (0x0100 - 0x01FF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Offset
     - Size
     - Access
     - Description
   * - 0x0100
     - 4
     - RW
     - **GLOBAL_CTRL**: Global control register.

       - Bit 0: Device enable (1=enabled, 0=disabled)
       - Bit 1: Soft reset (write 1 to reset, auto-clears)
       - Bits [31:2]: Reserved
   * - 0x0104
     - 4
     - RO
     - **GLOBAL_STATUS**: Global status register.

       - Bit 0: Device ready
       - Bit 1: Device busy (kernel executing)
       - Bit 2: Error occurred
       - Bits [31:3]: Reserved
   * - 0x0108
     - 4
     - RW1C
     - **ERROR_STATUS**: Error status (write 1 to clear).

       - Bit 0: Invalid command
       - Bit 1: VRAM access fault
       - Bit 2: Kernel execution error
       - Bit 3: DMA error
       - Bits [31:4]: Reserved

Interrupt Control (0x0200 - 0x02FF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Offset
     - Size
     - Access
     - Description
   * - 0x0200
     - 4
     - RW
     - **IRQ_ENABLE**: Interrupt enable mask.

       - Bit 0: Kernel completion interrupt
       - Bit 1: DMA completion interrupt
       - Bit 2: Error interrupt
       - Bits [31:3]: Reserved
   * - 0x0204
     - 4
     - RO
     - **IRQ_STATUS**: Interrupt status (pending interrupts).
   * - 0x0208
     - 4
     - WO
     - **IRQ_ACK**: Interrupt acknowledge (write 1 to clear).

Kernel Dispatch (0x0300 - 0x03FF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Offset
     - Size
     - Access
     - Description
   * - 0x0300
     - 4
     - RW
     - **KERNEL_ADDR_LO**: Kernel code address in VRAM (low 32 bits)
   * - 0x0304
     - 4
     - RW
     - **KERNEL_ADDR_HI**: Kernel code address (high 32 bits)
   * - 0x0308
     - 4
     - RW
     - **KERNEL_ARGS_LO**: Kernel arguments address in VRAM (low)
   * - 0x030C
     - 4
     - RW
     - **KERNEL_ARGS_HI**: Kernel arguments address (high)
   * - 0x0310
     - 4
     - RW
     - **GRID_DIM_X**: Grid dimension X (number of blocks in X)
   * - 0x0314
     - 4
     - RW
     - **GRID_DIM_Y**: Grid dimension Y
   * - 0x0318
     - 4
     - RW
     - **GRID_DIM_Z**: Grid dimension Z
   * - 0x031C
     - 4
     - RW
     - **BLOCK_DIM_X**: Block dimension X (threads per block in X)
   * - 0x0320
     - 4
     - RW
     - **BLOCK_DIM_Y**: Block dimension Y
   * - 0x0324
     - 4
     - RW
     - **BLOCK_DIM_Z**: Block dimension Z
   * - 0x0328
     - 4
     - RW
     - **SHARED_MEM_SIZE**: Shared memory size per block (bytes)
   * - 0x0330
     - 4
     - WO
     - **DISPATCH**: Write any value to start kernel execution.
       Kernel parameters must be configured before writing.

DMA Engine (0x0400 - 0x04FF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Offset
     - Size
     - Access
     - Description
   * - 0x0400
     - 8
     - RW
     - **DMA_SRC_ADDR**: DMA source address (host physical or VRAM)
   * - 0x0408
     - 8
     - RW
     - **DMA_DST_ADDR**: DMA destination address
   * - 0x0410
     - 4
     - RW
     - **DMA_SIZE**: Transfer size in bytes
   * - 0x0414
     - 4
     - RW
     - **DMA_CTRL**: DMA control register.

       - Bit 0: Start transfer (write 1 to start)
       - Bit 1: Direction (0=host to VRAM, 1=VRAM to host)
       - Bit 2: Interrupt on completion
       - Bits [31:3]: Reserved
   * - 0x0418
     - 4
     - RO
     - **DMA_STATUS**: DMA status.

       - Bit 0: DMA busy
       - Bit 1: DMA complete
       - Bit 2: DMA error

Thread Context Registers (0x1000 - 0x1FFF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These registers provide thread identity information when read by GPU threads.
The values returned depend on the executing thread context.

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Offset
     - Size
     - Access
     - Description
   * - 0x1000
     - 4
     - RO
     - **THREAD_ID_X**: Thread ID within block (X dimension)
   * - 0x1004
     - 4
     - RO
     - **THREAD_ID_Y**: Thread ID within block (Y dimension)
   * - 0x1008
     - 4
     - RO
     - **THREAD_ID_Z**: Thread ID within block (Z dimension)
   * - 0x1010
     - 4
     - RO
     - **BLOCK_ID_X**: Block ID within grid (X dimension)
   * - 0x1014
     - 4
     - RO
     - **BLOCK_ID_Y**: Block ID within grid (Y dimension)
   * - 0x1018
     - 4
     - RO
     - **BLOCK_ID_Z**: Block ID within grid (Z dimension)
   * - 0x1020
     - 4
     - RO
     - **WARP_ID**: Warp index within the block
   * - 0x1024
     - 4
     - RO
     - **LANE_ID**: Lane (thread) index within warp (0 to warp_size-1)

Thread Synchronization (0x2000 - 0x2FFF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Offset
     - Size
     - Access
     - Description
   * - 0x2000
     - 4
     - WO
     - **BARRIER**: Block-level barrier. Write any value to synchronize
       all threads in the current thread block.
   * - 0x2004
     - 4
     - RW
     - **THREAD_MASK**: Active thread mask for warp-level operations
   * - 0x2010
     - 4
     - RW
     - **BALLOT**: Warp vote - returns mask of threads with non-zero value
   * - 0x2020
     - 4
     - RW
     - **SHUFFLE**: Warp shuffle - exchange data between lanes.

       Write: ``[4:0]=source_lane, [31:5]=data``

       Read: returns data from source lane

Video RAM (BAR 2)
-----------------

BAR 2 maps the device's video RAM, which stores:

- Kernel code (RISC-V instructions)
- Kernel arguments
- Input/output data buffers
- Shared memory (mapped per thread block during execution)

Memory Layout (suggested)
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    VRAM Base
    +------------------+ 0x00000000
    |   Kernel Code    |
    +------------------+ 0x00100000 (1 MiB)
    |  Kernel Args     |
    +------------------+ 0x00101000
    |                  |
    |   Data Buffers   |
    |                  |
    +------------------+ VRAM_SIZE

The driver is responsible for managing VRAM allocation.

Doorbell Registers (BAR 4)
--------------------------

BAR 4 provides doorbell registers for command submission (future extension).

::

    0x0000: Command Queue 0 Doorbell (write queue tail pointer)
    0x0004: Command Queue 1 Doorbell
    ...

Interrupt Handling
------------------

The device supports MSI-X interrupts with the following vectors:

======= ================================
Vector  Description
======= ================================
0       Kernel execution complete
1       DMA transfer complete
2       Error interrupt
3-63    Reserved for future use
======= ================================

Programming Model
-----------------

Basic Operation Sequence
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Initialize device**: Set ``GLOBAL_CTRL.enable = 1``

2. **Upload kernel code**: Use DMA to copy kernel binary to VRAM

3. **Upload input data**: Use DMA to copy input buffers to VRAM

4. **Configure kernel dispatch**:

   - Set ``KERNEL_ADDR`` to point to kernel code in VRAM
   - Set ``KERNEL_ARGS`` to point to arguments structure in VRAM
   - Set grid and block dimensions

5. **Dispatch kernel**: Write to ``DISPATCH`` register

6. **Wait for completion**: Poll ``GLOBAL_STATUS.busy`` or wait for interrupt

7. **Download results**: Use DMA to copy output buffers from VRAM

Example: Vector Addition
~~~~~~~~~~~~~~~~~~~~~~~~

Host code (pseudocode)::

    // Upload kernel code
    dma_write(kernel_binary, VRAM_KERNEL_ADDR, kernel_size);

    // Upload input vectors A and B
    dma_write(vector_a, VRAM_DATA_A, vector_size);
    dma_write(vector_b, VRAM_DATA_B, vector_size);

    // Setup arguments (addresses in VRAM)
    struct args {
        uint32_t a_addr = VRAM_DATA_A;
        uint32_t b_addr = VRAM_DATA_B;
        uint32_t c_addr = VRAM_DATA_C;
        uint32_t n = vector_length;
    };
    dma_write(&args, VRAM_ARGS, sizeof(args));

    // Configure dispatch
    write_reg(KERNEL_ADDR_LO, VRAM_KERNEL_ADDR);
    write_reg(KERNEL_ARGS_LO, VRAM_ARGS);
    write_reg(GRID_DIM_X, (vector_length + 255) / 256);
    write_reg(GRID_DIM_Y, 1);
    write_reg(GRID_DIM_Z, 1);
    write_reg(BLOCK_DIM_X, 256);
    write_reg(BLOCK_DIM_Y, 1);
    write_reg(BLOCK_DIM_Z, 1);

    // Dispatch
    write_reg(DISPATCH, 1);

    // Wait for completion
    while (read_reg(GLOBAL_STATUS) & STATUS_BUSY);

    // Download result
    dma_read(VRAM_DATA_C, vector_c, vector_size);

Kernel code (RISC-V assembly concept)::

    # Load thread/block IDs from CTRL registers
    li      t0, CTRL_BASE + THREAD_ID_X
    lw      a0, 0(t0)           # thread_id_x
    li      t0, CTRL_BASE + BLOCK_ID_X
    lw      a1, 0(t0)           # block_id_x
    li      t0, CTRL_BASE + BLOCK_DIM_X
    lw      a2, 0(t0)           # block_dim_x

    # Calculate global thread ID: gid = block_id * block_dim + thread_id
    mul     a3, a1, a2
    add     a3, a3, a0          # a3 = global_id

    # Load kernel arguments
    lw      a4, 0(ARGS_BASE)    # a_addr
    lw      a5, 4(ARGS_BASE)    # b_addr
    lw      a6, 8(ARGS_BASE)    # c_addr
    lw      a7, 12(ARGS_BASE)   # n

    # Bounds check
    bge     a3, a7, done

    # Calculate addresses
    slli    t0, a3, 2           # offset = gid * 4
    add     t1, a4, t0          # &A[gid]
    add     t2, a5, t0          # &B[gid]
    add     t3, a6, t0          # &C[gid]

    # C[gid] = A[gid] + B[gid]
    lw      t4, 0(t1)
    lw      t5, 0(t2)
    add     t6, t4, t5
    sw      t6, 0(t3)

    done:
    # Thread exit (return to scheduler)
    ebreak

Implementation Notes
--------------------

QEMU Device Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The device is implemented in ``hw/gpgpu/gpgpu.c`` with the following components:

- ``GPGPUState``: Main device state structure
- ``gpgpu_mmio_ops``: MMIO operations for BAR 0
- ``gpgpu_vram_ops``: Memory operations for BAR 2
- ``gpgpu_realize()``: Device initialization
- ``gpgpu_dispatch_kernel()``: Kernel execution simulation

Compute Unit Simulation
~~~~~~~~~~~~~~~~~~~~~~~

Each compute unit maintains:

- Warp schedulers
- Register files (per thread)
- Program counters (per warp, with divergence stack)
- Shared memory

The simulation can operate in:

1. **Functional mode**: Execute threads sequentially for correctness
2. **Cycle-approximate mode**: Model warp scheduling and memory latency

Future Extensions
-----------------

- Command queue support (via BAR 4 doorbells)
- Multiple kernel concurrent execution
- Texture sampling units
- L1/L2 cache simulation
- Atomic operations
- Warp shuffle/vote instructions
- Debug registers and breakpoints

References
----------

- Vortex GPGPU: https://github.com/vortexgpgpu/vortex
- RISC-V ISA Specification: https://riscv.org/specifications/
- CUDA Programming Guide (for execution model concepts)
- AMD GCN Architecture Whitepaper
