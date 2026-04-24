/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "hw/pci/msix.h"
#include "fpu/softfloat-helpers.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/* ========================================================================== */
/* Helper: instruction field extraction                                        */
/* ========================================================================== */

#define RV_OPCODE(x)    ((x) & 0x7F)
#define RV_RD(x)        (((x) >> 7) & 0x1F)
#define RV_FUNCT3(x)    (((x) >> 12) & 0x7)
#define RV_RS1(x)       (((x) >> 15) & 0x1F)
#define RV_RS2(x)       (((x) >> 20) & 0x1F)
#define RV_FUNCT7(x)    (((x) >> 25) & 0x7F)

static inline int32_t rv_imm_i(uint32_t x)
{
    return (int32_t)x >> 20;
}

static inline int32_t rv_imm_s(uint32_t x)
{
    return ((int32_t)(x & 0xFE000000) >> 20) | ((x >> 7) & 0x1F);
}

static inline int32_t rv_imm_b(uint32_t x)
{
    int32_t imm = ((x >> 31) & 1) << 12;
    imm |= ((x >> 7) & 1) << 11;
    imm |= ((x >> 25) & 0x3F) << 5;
    imm |= ((x >> 8) & 0xF) << 1;
    return (imm << 19) >> 19;
}

static inline int32_t rv_imm_u(uint32_t x)
{
    return (int32_t)(x & 0xFFFFF000);
}

static inline int32_t rv_imm_j(uint32_t x)
{
    int32_t imm = ((x >> 31) & 1) << 20;
    imm |= ((x >> 12) & 0xFF) << 12;
    imm |= ((x >> 20) & 1) << 11;
    imm |= ((x >> 21) & 0x3FF) << 1;
    return (imm << 11) >> 11;
}

/* ========================================================================== */
/* Helper: VRAM memory access (little-endian)                                  */
/* ========================================================================== */

static uint32_t vram_read32(const uint8_t *vram, uint32_t addr, uint64_t size)
{
    if (addr + 4 > size) {
        return 0;
    }
    return (uint32_t)vram[addr] |
           ((uint32_t)vram[addr + 1] << 8) |
           ((uint32_t)vram[addr + 2] << 16) |
           ((uint32_t)vram[addr + 3] << 24);
}

static void vram_write32(uint8_t *vram, uint32_t addr, uint32_t val,
                          uint64_t size)
{
    if (addr + 4 > size) {
        return;
    }
    vram[addr] = val & 0xFF;
    vram[addr + 1] = (val >> 8) & 0xFF;
    vram[addr + 2] = (val >> 16) & 0xFF;
    vram[addr + 3] = (val >> 24) & 0xFF;
}

static uint16_t vram_read16(const uint8_t *vram, uint32_t addr, uint64_t size)
{
    if (addr + 2 > size) {
        return 0;
    }
    return (uint16_t)vram[addr] | ((uint16_t)vram[addr + 1] << 8);
}

static void vram_write16(uint8_t *vram, uint32_t addr, uint16_t val,
                          uint64_t size)
{
    if (addr + 2 > size) {
        return;
    }
    vram[addr] = val & 0xFF;
    vram[addr + 1] = (val >> 8) & 0xFF;
}

static uint8_t vram_read8(const uint8_t *vram, uint32_t addr, uint64_t size)
{
    if (addr >= size) {
        return 0;
    }
    return vram[addr];
}

static void vram_write8(uint8_t *vram, uint32_t addr, uint8_t val,
                         uint64_t size)
{
    if (addr >= size) {
        return;
    }
    vram[addr] = val;
}

/* ========================================================================== */
/* Helper: RISC-V rounding mode → QEMU FloatRoundMode                         */
/* ========================================================================== */

static FloatRoundMode rv_rm_to_qemu(uint32_t rm, uint32_t fcsr)
{
    if (rm == 7) {
        rm = (fcsr >> 5) & 0x7;
    }
    switch (rm) {
    case 0: return float_round_nearest_even;
    case 1: return float_round_to_zero;
    case 2: return float_round_down;
    case 3: return float_round_up;
    case 4: return float_round_ties_away;
    default: return float_round_nearest_even;
    }
}

/* Helper: accumulate FP exception flags into fcsr */
static void fp_acc_exc(GPGPULane *lane)
{
    uint32_t qemu_flags = get_float_exception_flags(&lane->fp_status) & 0x1F;
    uint32_t rv_flags = 0;
    if (qemu_flags & float_flag_invalid)   rv_flags |= 0x10;
    if (qemu_flags & float_flag_divbyzero) rv_flags |= 0x08;
    if (qemu_flags & float_flag_overflow)  rv_flags |= 0x04;
    if (qemu_flags & float_flag_underflow) rv_flags |= 0x02;
    if (qemu_flags & float_flag_inexact)   rv_flags |= 0x01;
    lane->fcsr = (lane->fcsr & ~0x1F) | ((lane->fcsr & 0x1F) | rv_flags);
    set_float_exception_flags(0, &lane->fp_status);
}

/* Helper: set rounding mode from instruction rm field */
static void fp_set_rm(GPGPULane *lane, uint32_t rm)
{
    set_float_rounding_mode(rv_rm_to_qemu(rm, lane->fcsr), &lane->fp_status);
}

/* ========================================================================== */
/* Helper: CTRL register read (0x80000000+offset)                              */
/* ========================================================================== */

static uint32_t ctrl_read(GPGPUState *s, GPGPUWarp *warp, int lane_id,
                           uint32_t offset)
{
    (void)lane_id;
    uint32_t linear_tid = warp->thread_id_base + lane_id;
    uint32_t bx = s->kernel.block_dim[0];
    uint32_t by = s->kernel.block_dim[1];
    uint32_t bxy = bx * by;

    switch (offset) {
    case 0x00: /* THREAD_ID_X */
        return (by > 0 && bx > 0) ? linear_tid % bx : 0;
    case 0x04: /* THREAD_ID_Y */
        return (by > 0 && bx > 0) ? (linear_tid / bx) % by : 0;
    case 0x08: /* THREAD_ID_Z */
        return (bxy > 0) ? linear_tid / bxy : 0;
    case 0x10: return warp->block_id[0];
    case 0x14: return warp->block_id[1];
    case 0x18: return warp->block_id[2];
    case 0x20: return s->kernel.block_dim[0];
    case 0x24: return s->kernel.block_dim[1];
    case 0x28: return s->kernel.block_dim[2];
    case 0x30: return s->kernel.grid_dim[0];
    case 0x34: return s->kernel.grid_dim[1];
    case 0x38: return s->kernel.grid_dim[2];
    default: return 0;
    }
}

/* ========================================================================== */
/* Helper: memory load/store for lane                                          */
/* ========================================================================== */

static uint32_t lane_mem_read32(GPGPUState *s, GPGPUWarp *warp, int lane_id,
                                 uint32_t addr)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return ctrl_read(s, warp, lane_id, addr - GPGPU_CORE_CTRL_BASE);
    }
    return vram_read32(s->vram_ptr, addr, s->vram_size);
}

static uint16_t lane_mem_read16(GPGPUState *s, uint32_t addr)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return 0; /* CTRL is read-only 32-bit */
    }
    return vram_read16(s->vram_ptr, addr, s->vram_size);
}

static uint8_t lane_mem_read8(GPGPUState *s, uint32_t addr)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return 0;
    }
    return vram_read8(s->vram_ptr, addr, s->vram_size);
}

static void lane_mem_write32(GPGPUState *s, uint32_t addr, uint32_t val)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return; /* CTRL is read-only */
    }
    vram_write32(s->vram_ptr, addr, val, s->vram_size);
}

static void lane_mem_write16(GPGPUState *s, uint32_t addr, uint16_t val)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return;
    }
    vram_write16(s->vram_ptr, addr, val, s->vram_size);
}

static void lane_mem_write8(GPGPUState *s, uint32_t addr, uint8_t val)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return;
    }
    vram_write8(s->vram_ptr, addr, val, s->vram_size);
}

/* ========================================================================== */
/* Helper: E2M1 (FP4) custom conversion                                       */
/* ========================================================================== */

static uint8_t fp32_to_e2m1(float32 f, float_status *status)
{
    bool sign = float32_is_neg(f);
    float32 abs_f = float32_abs(f);

    /* NaN/Inf → saturated max (±6.0) */
    if (float32_is_any_nan(f) || float32_is_infinity(f)) {
        return (sign << 3) | 0x07;
    }

    /* Use C float for threshold comparison (not arithmetic) */
    union { uint32_t i; float f; } u;
    u.i = abs_f;
    float val = u.f;

    /* Saturate values > 6.0 */
    if (val > 6.0f) {
        return (sign << 3) | 0x07;
    }

    /* Threshold rounding (round to nearest even) */
    uint8_t exp, mant;
    if (val >= 5.0f) {
        /* 5.0 is exactly halfway between 4.0 and 6.0; RNE → 4.0 (even) */
        if (val == 5.0f) {
            exp = 3; mant = 0;  /* 4.0 */
        } else {
            exp = 3; mant = 1;  /* 6.0 */
        }
    } else if (val >= 3.5f) {
        exp = 3; mant = 0;  /* 4.0 */
    } else if (val >= 2.5f) {
        exp = 2; mant = 1;  /* 3.0 */
    } else if (val >= 1.75f) {
        exp = 2; mant = 0;  /* 2.0 */
    } else if (val >= 1.25f) {
        exp = 1; mant = 1;  /* 1.5 */
    } else if (val >= 0.75f) {
        exp = 1; mant = 0;  /* 1.0 */
    } else if (val >= 0.25f) {
        exp = 0; mant = 1;  /* 0.5 */
    } else {
        exp = 0; mant = 0;  /* 0.0 */
    }

    return (sign << 3) | (exp << 1) | mant;
}

static float32 e2m1_to_fp32(float4_e2m1 val, float_status *status)
{
    /* Chain: E2M1 → E4M3 → BF16 → FP32 */
    float8_e4m3 e4m3 = float4_e2m1_to_float8_e4m3(val, status);
    bfloat16 bf16 = float8_e4m3_to_bfloat16(e4m3, status);
    return bfloat16_to_float32(bf16, status);
}

/* ========================================================================== */
/* gpgpu_core_init_warp                                                        */
/* ========================================================================== */

void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                           uint32_t thread_id_base, const uint32_t block_id[3],
                           uint32_t num_threads,
                           uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    warp->warp_id = warp_id;
    warp->thread_id_base = thread_id_base;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->gpr[0] = 0;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        lane->active = (i < (int)num_threads);

        /* Initialize FP status */
        set_float_rounding_mode(float_round_nearest_even, &lane->fp_status);
        set_default_nan_mode(true, &lane->fp_status);
        set_float_exception_flags(0, &lane->fp_status);
        lane->fcsr = 0;
    }

    warp->active_mask = (num_threads >= 32) ? 0xFFFFFFFF :
                        ((1U << num_threads) - 1);
}

/* ========================================================================== */
/* gpgpu_core_exec_warp                                                        */
/* ========================================================================== */

int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;

    while (warp->active_mask != 0 && cycles < max_cycles) {
        /* All active lanes share the same PC (SIMT lockstep) */
        int first = __builtin_ctz(warp->active_mask);
        uint32_t pc = warp->lanes[first].pc;

        /* Fetch instruction */
        uint32_t inst = vram_read32(s->vram_ptr, pc, s->vram_size);

        /* Decode common fields once */
        uint32_t opcode = RV_OPCODE(inst);
        uint32_t rd     = RV_RD(inst);
        uint32_t funct3 = RV_FUNCT3(inst);
        uint32_t rs1    = RV_RS1(inst);
        uint32_t rs2    = RV_RS2(inst);
        uint32_t funct7 = RV_FUNCT7(inst);

        /* Execute for each active lane */
        for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (!(warp->active_mask & (1U << i))) {
                continue;
            }

            GPGPULane *lane = &warp->lanes[i];
            uint32_t *gpr = lane->gpr;
            uint32_t *fpr = lane->fpr;

            /* Default: advance PC */
            lane->pc += 4;

            switch (opcode) {
            /* ============================================================ */
            /* LUI                                                           */
            /* ============================================================ */
            case 0x37:
                gpr[rd] = rv_imm_u(inst);
                break;

            /* ============================================================ */
            /* AUIPC                                                         */
            /* ============================================================ */
            case 0x17:
                gpr[rd] = pc + rv_imm_u(inst);
                break;

            /* ============================================================ */
            /* JAL                                                           */
            /* ============================================================ */
            case 0x6F:
                gpr[rd] = lane->pc;  /* pc + 4 already set */
                lane->pc = pc + rv_imm_j(inst);
                break;

            /* ============================================================ */
            /* JALR                                                          */
            /* ============================================================ */
            case 0x67: {
                uint32_t target = (gpr[rs1] + rv_imm_i(inst)) & ~1U;
                gpr[rd] = lane->pc;
                if (target == 0) {
                    /* ret to address 0 → deactivate lane */
                    lane->active = false;
                    warp->active_mask &= ~(1U << i);
                } else {
                    lane->pc = target;
                }
                break;
            }

            /* ============================================================ */
            /* Branch: BEQ/BNE/BLT/BGE/BLTU/BGEU                           */
            /* ============================================================ */
            case 0x63: {
                bool taken = false;
                switch (funct3) {
                case 0: taken = (gpr[rs1] == gpr[rs2]); break;  /* BEQ */
                case 1: taken = (gpr[rs1] != gpr[rs2]); break;  /* BNE */
                case 4: taken = ((int32_t)gpr[rs1] < (int32_t)gpr[rs2]); break;  /* BLT */
                case 5: taken = ((int32_t)gpr[rs1] >= (int32_t)gpr[rs2]); break; /* BGE */
                case 6: taken = (gpr[rs1] < gpr[rs2]); break;  /* BLTU */
                case 7: taken = (gpr[rs1] >= gpr[rs2]); break; /* BGEU */
                default: break;
                }
                if (taken) {
                    lane->pc = pc + rv_imm_b(inst);
                }
                break;
            }

            /* ============================================================ */
            /* Load: LB/LH/LW/LBU/LHU                                       */
            /* ============================================================ */
            case 0x03: {
                uint32_t addr = (uint32_t)((int32_t)gpr[rs1] + rv_imm_i(inst));
                switch (funct3) {
                case 0: /* LB */
                    gpr[rd] = (int8_t)lane_mem_read8(s, addr);
                    break;
                case 1: /* LH */
                    gpr[rd] = (int16_t)lane_mem_read16(s, addr);
                    break;
                case 2: /* LW */
                    gpr[rd] = lane_mem_read32(s, warp, i, addr);
                    break;
                case 4: /* LBU */
                    gpr[rd] = lane_mem_read8(s, addr);
                    break;
                case 5: /* LHU */
                    gpr[rd] = lane_mem_read16(s, addr);
                    break;
                default:
                    break;
                }
                break;
            }

            /* ============================================================ */
            /* Store: SB/SH/SW                                              */
            /* ============================================================ */
            case 0x23: {
                uint32_t addr = (uint32_t)((int32_t)gpr[rs1] + rv_imm_s(inst));
                switch (funct3) {
                case 0: /* SB */
                    lane_mem_write8(s, addr, gpr[rs2] & 0xFF);
                    break;
                case 1: /* SH */
                    lane_mem_write16(s, addr, gpr[rs2] & 0xFFFF);
                    break;
                case 2: /* SW */
                    lane_mem_write32(s, addr, gpr[rs2]);
                    break;
                default:
                    break;
                }
                break;
            }

            /* ============================================================ */
            /* OP-IMM: ADDI/SLTI/SLTIU/XORI/ORI/ANDI/SLLI/SRLI/SRAI         */
            /* ============================================================ */
            case 0x13: {
                int32_t imm = rv_imm_i(inst);
                switch (funct3) {
                case 0: gpr[rd] = (uint32_t)((int32_t)gpr[rs1] + imm); break; /* ADDI */
                case 1: gpr[rd] = gpr[rs1] << (imm & 0x1F); break; /* SLLI */
                case 2: gpr[rd] = ((int32_t)gpr[rs1] < imm) ? 1 : 0; break; /* SLTI */
                case 3: gpr[rd] = (gpr[rs1] < (uint32_t)imm) ? 1 : 0; break; /* SLTIU */
                case 4: gpr[rd] = gpr[rs1] ^ (uint32_t)imm; break; /* XORI */
                case 5:
                    if (funct7 & 0x20) {
                        gpr[rd] = (uint32_t)((int32_t)gpr[rs1] >> (imm & 0x1F)); /* SRAI */
                    } else {
                        gpr[rd] = gpr[rs1] >> (imm & 0x1F); /* SRLI */
                    }
                    break;
                case 6: gpr[rd] = gpr[rs1] | (uint32_t)imm; break; /* ORI */
                case 7: gpr[rd] = gpr[rs1] & (uint32_t)imm; break; /* ANDI */
                default: break;
                }
                break;
            }

            /* ============================================================ */
            /* OP: ADD/SUB/SLL/SLT/SLTU/XOR/SRL/SRA/OR/AND                  */
            /* ============================================================ */
            case 0x33: {
                switch (funct3) {
                case 0:
                    if (funct7 & 0x20) {
                        gpr[rd] = (uint32_t)((int32_t)gpr[rs1] - (int32_t)gpr[rs2]); /* SUB */
                    } else {
                        gpr[rd] = (uint32_t)((int32_t)gpr[rs1] + (int32_t)gpr[rs2]); /* ADD */
                    }
                    break;
                case 1: gpr[rd] = gpr[rs1] << (gpr[rs2] & 0x1F); break; /* SLL */
                case 2: gpr[rd] = ((int32_t)gpr[rs1] < (int32_t)gpr[rs2]) ? 1 : 0; break; /* SLT */
                case 3: gpr[rd] = (gpr[rs1] < gpr[rs2]) ? 1 : 0; break; /* SLTU */
                case 4: gpr[rd] = gpr[rs1] ^ gpr[rs2]; break; /* XOR */
                case 5:
                    if (funct7 & 0x20) {
                        gpr[rd] = (uint32_t)((int32_t)gpr[rs1] >> (gpr[rs2] & 0x1F)); /* SRA */
                    } else {
                        gpr[rd] = gpr[rs1] >> (gpr[rs2] & 0x1F); /* SRL */
                    }
                    break;
                case 6: gpr[rd] = gpr[rs1] | gpr[rs2]; break; /* OR */
                case 7: gpr[rd] = gpr[rs1] & gpr[rs2]; break; /* AND */
                default: break;
                }
                break;
            }

            /* ============================================================ */
            /* SYSTEM: EBREAK/CSR                                           */
            /* ============================================================ */
            case 0x73: {
                if (funct3 == 0) {
                    /* EBREAK or ECALL */
                    lane->active = false;
                    warp->active_mask &= ~(1U << i);
                } else {
                    /* CSR instructions */
                    uint32_t csr_addr = (inst >> 20) & 0xFFF;
                    uint32_t csr_val = 0;
                    uint32_t old_val = 0;

                    /* Read CSR */
                    switch (csr_addr) {
                    case CSR_MHARTID: csr_val = lane->mhartid; break;
                    case CSR_FFLAGS:  csr_val = lane->fcsr & 0x1F; break;
                    case CSR_FRM:     csr_val = (lane->fcsr >> 5) & 0x7; break;
                    case CSR_FCSR:    csr_val = lane->fcsr & 0xFF; break;
                    default: csr_val = 0; break;
                    }

                    /* Compute new CSR value based on funct3 */
                    switch (funct3) {
                    case 1: /* CSRRW */
                        old_val = csr_val;
                        csr_val = gpr[rs1];
                        break;
                    case 2: /* CSRRS */
                        old_val = csr_val;
                        csr_val = csr_val | gpr[rs1];
                        break;
                    case 3: /* CSRRC */
                        old_val = csr_val;
                        csr_val = csr_val & ~gpr[rs1];
                        break;
                    case 5: /* CSRRWI */
                        old_val = csr_val;
                        csr_val = rs1;  /* rs1 field is uimm */
                        break;
                    case 6: /* CSRRSI */
                        old_val = csr_val;
                        csr_val = csr_val | rs1;
                        break;
                    case 7: /* CSRRCI */
                        old_val = csr_val;
                        csr_val = csr_val & ~rs1;
                        break;
                    default:
                        old_val = csr_val;
                        break;
                    }

                    /* Write CSR */
                    switch (csr_addr) {
                    case CSR_MHARTID:
                        /* Read-only, ignore write */
                        break;
                    case CSR_FFLAGS:
                        lane->fcsr = (lane->fcsr & ~0x1F) | (csr_val & 0x1F);
                        set_float_exception_flags(0, &lane->fp_status);
                        break;
                    case CSR_FRM:
                        lane->fcsr = (lane->fcsr & ~0xE0) | ((csr_val & 0x7) << 5);
                        break;
                    case CSR_FCSR:
                        lane->fcsr = csr_val & 0xFF;
                        set_float_exception_flags(0, &lane->fp_status);
                        break;
                    default:
                        break;
                    }

                    /* Write old value to rd */
                    if (rd != 0) {
                        gpr[rd] = old_val;
                    }
                }
                break;
            }

            /* ============================================================ */
            /* OP-FP: RV32F + Low-Precision FP                              */
            /* ============================================================ */
            case 0x53: {
                switch (funct7) {
                /* -------------------------------------------------------- */
                /* FADD.S (funct7=0x00)                                       */
                /* -------------------------------------------------------- */
                case 0x00: {
                    fp_set_rm(lane, funct3);
                    fpr[rd] = float32_add(fpr[rs1], fpr[rs2], &lane->fp_status);
                    fp_acc_exc(lane);
                    break;
                }

                /* -------------------------------------------------------- */
                /* FSUB.S (funct7=0x04)                                       */
                /* -------------------------------------------------------- */
                case 0x04: {
                    fp_set_rm(lane, funct3);
                    fpr[rd] = float32_sub(fpr[rs1], fpr[rs2], &lane->fp_status);
                    fp_acc_exc(lane);
                    break;
                }

                /* -------------------------------------------------------- */
                /* FMUL.S (funct7=0x08)                                       */
                /* -------------------------------------------------------- */
                case 0x08: {
                    fp_set_rm(lane, funct3);
                    fpr[rd] = float32_mul(fpr[rs1], fpr[rs2], &lane->fp_status);
                    fp_acc_exc(lane);
                    break;
                }

                /* -------------------------------------------------------- */
                /* FDIV.S (funct7=0x0C)                                       */
                /* -------------------------------------------------------- */
                case 0x0C: {
                    fp_set_rm(lane, funct3);
                    fpr[rd] = float32_div(fpr[rs1], fpr[rs2], &lane->fp_status);
                    fp_acc_exc(lane);
                    break;
                }

                /* -------------------------------------------------------- */
                /* FSQRT.S (funct7=0x2C, rs2=0)                               */
                /* -------------------------------------------------------- */
                case 0x2C: {
                    fp_set_rm(lane, funct3);
                    fpr[rd] = float32_sqrt(fpr[rs1], &lane->fp_status);
                    fp_acc_exc(lane);
                    break;
                }

                /* -------------------------------------------------------- */
                /* FSGNJ.S/FSGNJN.S/FSGNJX.S (funct7=0x10)                   */
                /* -------------------------------------------------------- */
                case 0x10: {
                    uint32_t a = fpr[rs1], b = fpr[rs2];
                    switch (funct3) {
                    case 0: /* FSGNJ.S */
                        fpr[rd] = (a & 0x7FFFFFFF) | (b & 0x80000000);
                        break;
                    case 1: /* FSGNJN.S */
                        fpr[rd] = (a & 0x7FFFFFFF) | ((~b) & 0x80000000);
                        break;
                    case 2: /* FSGNJX.S */
                        fpr[rd] = a ^ (b & 0x80000000);
                        break;
                    default:
                        break;
                    }
                    break;
                }

                /* -------------------------------------------------------- */
                /* FMIN.S/FMAX.S (funct7=0x14)                                */
                /* -------------------------------------------------------- */
                case 0x14: {
                    switch (funct3) {
                    case 0: /* FMIN.S */
                        fpr[rd] = float32_min(fpr[rs1], fpr[rs2], &lane->fp_status);
                        fp_acc_exc(lane);
                        break;
                    case 1: /* FMAX.S */
                        fpr[rd] = float32_max(fpr[rs1], fpr[rs2], &lane->fp_status);
                        fp_acc_exc(lane);
                        break;
                    default:
                        break;
                    }
                    break;
                }

                /* -------------------------------------------------------- */
                /* FCVT.W.S / FCVT.WU.S (funct7=0x60)                        */
                /* -------------------------------------------------------- */
                case 0x60: {
                    fp_set_rm(lane, funct3);
                    if (rs2 == 0) {
                        /* FCVT.W.S */
                        gpr[rd] = (uint32_t)float32_to_int32(fpr[rs1], &lane->fp_status);
                    } else if (rs2 == 1) {
                        /* FCVT.WU.S */
                        gpr[rd] = float32_to_uint32(fpr[rs1], &lane->fp_status);
                    }
                    fp_acc_exc(lane);
                    break;
                }

                /* -------------------------------------------------------- */
                /* FCVT.S.W / FCVT.S.WU (funct7=0x68)                        */
                /* -------------------------------------------------------- */
                case 0x68: {
                    fp_set_rm(lane, funct3);
                    if (rs2 == 0) {
                        /* FCVT.S.W */
                        fpr[rd] = int32_to_float32((int32_t)gpr[rs1], &lane->fp_status);
                    } else if (rs2 == 1) {
                        /* FCVT.S.WU */
                        fpr[rd] = uint32_to_float32(gpr[rs1], &lane->fp_status);
                    }
                    fp_acc_exc(lane);
                    break;
                }

                /* -------------------------------------------------------- */
                /* FMV.X.W / FCLASS.S (funct7=0x70)                          */
                /* -------------------------------------------------------- */
                case 0x70: {
                    switch (funct3) {
                    case 0: /* FMV.X.W */
                        gpr[rd] = fpr[rs1];
                        break;
                    case 1: { /* FCLASS.S */
                        uint32_t val = fpr[rs1];
                        uint32_t result = 0;
                        bool sign = (val >> 31) & 1;
                        uint32_t exp = (val >> 23) & 0xFF;
                        uint32_t mant = val & 0x7FFFFF;

                        if (exp == 0xFF) {
                            if (mant) {
                                result = (mant & 0x400000) ? (1 << 9) : (1 << 8);
                            } else {
                                result = sign ? (1 << 0) : (1 << 7);
                            }
                        } else if (exp == 0) {
                            if (mant) {
                                result = sign ? (1 << 2) : (1 << 5);
                            } else {
                                result = sign ? (1 << 3) : (1 << 4);
                            }
                        } else {
                            result = sign ? (1 << 1) : (1 << 6);
                        }
                        gpr[rd] = result;
                        break;
                    }
                    default:
                        break;
                    }
                    break;
                }

                /* -------------------------------------------------------- */
                /* FMV.W.X (funct7=0x78, funct3=0)                           */
                /* -------------------------------------------------------- */
                case 0x78: {
                    if (funct3 == 0) {
                        fpr[rd] = gpr[rs1];
                    }
                    break;
                }

                /* -------------------------------------------------------- */
                /* FEQ.S/FLT.S/FLE.S (funct7=0x50)                           */
                /* -------------------------------------------------------- */
                case 0x50: {
                    switch (funct3) {
                    case 0: /* FLE.S */
                        gpr[rd] = float32_le(fpr[rs1], fpr[rs2], &lane->fp_status) ? 1 : 0;
                        fp_acc_exc(lane);
                        break;
                    case 1: /* FLT.S */
                        gpr[rd] = float32_lt(fpr[rs1], fpr[rs2], &lane->fp_status) ? 1 : 0;
                        fp_acc_exc(lane);
                        break;
                    case 2: /* FEQ.S */
                        gpr[rd] = float32_eq_quiet(fpr[rs1], fpr[rs2], &lane->fp_status) ? 1 : 0;
                        fp_acc_exc(lane);
                        break;
                    default:
                        break;
                    }
                    break;
                }

                /* -------------------------------------------------------- */
                /* BF16 conversions (funct7=0x22)                             */
                /* rs2=0: FCVT.S.BF16, rs2=1: FCVT.BF16.S                    */
                /* -------------------------------------------------------- */
                case 0x22: {
                    if (rs2 == 0) {
                        /* FCVT.S.BF16: BF16 → FP32 */
                        bfloat16 bf = (bfloat16)(fpr[rs1] & 0xFFFF);
                        fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
                        fp_acc_exc(lane);
                    } else if (rs2 == 1) {
                        /* FCVT.BF16.S: FP32 → BF16 */
                        fpr[rd] = (uint32_t)float32_to_bfloat16(fpr[rs1], &lane->fp_status);
                        fp_acc_exc(lane);
                    }
                    break;
                }

                /* -------------------------------------------------------- */
                /* FP8 E4M3/E5M2 conversions (funct7=0x24)                    */
                /* rs2=0: FCVT.S.E4M3, rs2=1: FCVT.E4M3.S                    */
                /* rs2=2: FCVT.S.E5M2, rs2=3: FCVT.E5M2.S                    */
                /* -------------------------------------------------------- */
                case 0x24: {
                    switch (rs2) {
                    case 0: { /* FCVT.S.E4M3: E4M3 → FP32 */
                        float8_e4m3 e4 = (float8_e4m3)(fpr[rs1] & 0xFF);
                        bfloat16 bf = float8_e4m3_to_bfloat16(e4, &lane->fp_status);
                        fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
                        fp_acc_exc(lane);
                        break;
                    }
                    case 1: { /* FCVT.E4M3.S: FP32 → E4M3 (sat) */
                        float8_e4m3 e4 = float32_to_float8_e4m3(fpr[rs1], true,
                                                                 &lane->fp_status);
                        fpr[rd] = (uint32_t)e4;
                        fp_acc_exc(lane);
                        break;
                    }
                    case 2: { /* FCVT.S.E5M2: E5M2 → FP32 */
                        float8_e5m2 e5 = (float8_e5m2)(fpr[rs1] & 0xFF);
                        bfloat16 bf = float8_e5m2_to_bfloat16(e5, &lane->fp_status);
                        fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
                        fp_acc_exc(lane);
                        break;
                    }
                    case 3: { /* FCVT.E5M2.S: FP32 → E5M2 (sat) */
                        float8_e5m2 e5 = float32_to_float8_e5m2(fpr[rs1], true,
                                                                 &lane->fp_status);
                        fpr[rd] = (uint32_t)e5;
                        fp_acc_exc(lane);
                        break;
                    }
                    default:
                        break;
                    }
                    break;
                }

                /* -------------------------------------------------------- */
                /* FP4 E2M1 conversions (funct7=0x26)                         */
                /* rs2=0: FCVT.S.E2M1, rs2=1: FCVT.E2M1.S                    */
                /* -------------------------------------------------------- */
                case 0x26: {
                    switch (rs2) {
                    case 0: { /* FCVT.S.E2M1: E2M1 → FP32 */
                        float4_e2m1 e2 = (float4_e2m1)(fpr[rs1] & 0xF);
                        fpr[rd] = e2m1_to_fp32(e2, &lane->fp_status);
                        fp_acc_exc(lane);
                        break;
                    }
                    case 1: { /* FCVT.E2M1.S: FP32 → E2M1 (threshold, sat) */
                        float4_e2m1 e2 = fp32_to_e2m1(fpr[rs1], &lane->fp_status);
                        fpr[rd] = (uint32_t)e2;
                        fp_acc_exc(lane);
                        break;
                    }
                    default:
                        break;
                    }
                    break;
                }

                default:
                    /* Unknown OP-FP instruction, skip */
                    break;
                }
                break;
            }

            /* ============================================================ */
            /* FLW (opcode=0x07)                                            */
            /* ============================================================ */
            case 0x07: {
                uint32_t addr = (uint32_t)((int32_t)gpr[rs1] + rv_imm_i(inst));
                fpr[rd] = lane_mem_read32(s, warp, i, addr);
                break;
            }

            /* ============================================================ */
            /* FSW (opcode=0x27)                                            */
            /* ============================================================ */
            case 0x27: {
                uint32_t addr = (uint32_t)((int32_t)gpr[rs1] + rv_imm_s(inst));
                lane_mem_write32(s, addr, fpr[rs2]);
                break;
            }

            /* ============================================================ */
            /* FMADD.S (opcode=0x43)                                        */
            /* ============================================================ */
            case 0x43: {
                fp_set_rm(lane, funct3);
                uint32_t rs3 = (inst >> 27) & 0x1F;
                fpr[rd] = float32_muladd(fpr[rs1], fpr[rs2], fpr[rs3], 0,
                                         &lane->fp_status);
                fp_acc_exc(lane);
                break;
            }

            /* ============================================================ */
            /* FMSUB.S (opcode=0x47)                                        */
            /* ============================================================ */
            case 0x47: {
                fp_set_rm(lane, funct3);
                uint32_t rs3 = (inst >> 27) & 0x1F;
                fpr[rd] = float32_muladd(fpr[rs1], fpr[rs2], fpr[rs3],
                                         float_muladd_negate_c,
                                         &lane->fp_status);
                fp_acc_exc(lane);
                break;
            }

            /* ============================================================ */
            /* FNMSUB.S (opcode=0x4B)                                       */
            /* ============================================================ */
            case 0x4B: {
                fp_set_rm(lane, funct3);
                uint32_t rs3 = (inst >> 27) & 0x1F;
                fpr[rd] = float32_muladd(fpr[rs1], fpr[rs2], fpr[rs3],
                                         float_muladd_negate_product,
                                         &lane->fp_status);
                fp_acc_exc(lane);
                break;
            }

            /* ============================================================ */
            /* FNMADD.S (opcode=0x4F)                                       */
            /* ============================================================ */
            case 0x4F: {
                fp_set_rm(lane, funct3);
                uint32_t rs3 = (inst >> 27) & 0x1F;
                fpr[rd] = float32_muladd(fpr[rs1], fpr[rs2], fpr[rs3],
                                         float_muladd_negate_result,
                                         &lane->fp_status);
                fp_acc_exc(lane);
                break;
            }

            default:
                /* Unknown opcode, skip */
                break;
            }

            /* x0 is always 0 */
            gpr[0] = 0;
        }

        cycles++;
    }

    return (warp->active_mask == 0) ? 0 : -1;
}

/* ========================================================================== */
/* gpgpu_core_exec_kernel                                                      */
/* ========================================================================== */

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t gx = s->kernel.grid_dim[0] ? s->kernel.grid_dim[0] : 1;
    uint32_t gy = s->kernel.grid_dim[1] ? s->kernel.grid_dim[1] : 1;
    uint32_t gz = s->kernel.grid_dim[2] ? s->kernel.grid_dim[2] : 1;
    uint32_t bx = s->kernel.block_dim[0] ? s->kernel.block_dim[0] : 1;
    uint32_t by = s->kernel.block_dim[1] ? s->kernel.block_dim[1] : 1;
    uint32_t bz = s->kernel.block_dim[2] ? s->kernel.block_dim[2] : 1;
    uint32_t threads_per_block = bx * by * bz;
    uint32_t warp_size = s->warp_size;
    uint32_t pc = (uint32_t)s->kernel.kernel_addr;

    /* Set device BUSY */
    s->global_status &= ~GPGPU_STATUS_READY;
    s->global_status |= GPGPU_STATUS_BUSY;

    int ret = 0;

    for (uint32_t z = 0; z < gz; z++) {
        for (uint32_t y = 0; y < gy; y++) {
            for (uint32_t x = 0; x < gx; x++) {
                uint32_t block_id[3] = { x, y, z };
                uint32_t block_linear = z * gy * gx + y * gx + x;

                uint32_t num_warps =
                    (threads_per_block + warp_size - 1) / warp_size;

                for (uint32_t w = 0; w < num_warps; w++) {
                    uint32_t tid_base = w * warp_size;
                    uint32_t n_threads = threads_per_block - tid_base;
                    if (n_threads > warp_size) {
                        n_threads = warp_size;
                    }

                    GPGPUWarp warp;
                    gpgpu_core_init_warp(&warp, pc, tid_base, block_id,
                                         n_threads, w, block_linear);

                    /* Update SIMT context for external visibility */
                    s->simt.block_id[0] = x;
                    s->simt.block_id[1] = y;
                    s->simt.block_id[2] = z;
                    s->simt.warp_id = w;

                    int r = gpgpu_core_exec_warp(s, &warp, 100000);
                    if (r < 0) {
                        ret = -1;
                    }
                }
            }
        }
    }

    /* Kernel complete: clear BUSY, set READY, set IRQ */
    s->global_status &= ~GPGPU_STATUS_BUSY;
    s->global_status |= GPGPU_STATUS_READY;
    s->irq_status |= GPGPU_IRQ_KERNEL_DONE;

    if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
        msix_notify(PCI_DEVICE(s), GPGPU_MSIX_VEC_KERNEL);
    }

    return ret;
}
