/* file: hdu.v
 * Hazard Detection Unit — 9-stage pipeline (IF1→IF2→ID→EX1→EX2→EX3→EX4→MEM→WB)
 *
 * Version: 2.1
 * Revision history:
 *    - 2.0: 8-stage pipeline (EX1/EX2/EX3).
 *    - 2.1: 9-stage pipeline (EX1/EX2/EX3/EX4).
 *           EX2 = barrel shifter, EX3 = ALU, EX4 = cond eval / branch.
 *           Added EX3→EX1 hazard class, EX4→EX1 hazard class.
 *           Branch now resolved in EX4 → penalty = 6 cycles.
 *           (Feb. 24, 2026)
 *
 * Forwarding is available ONLY from MEM (EX4/MEM reg) and WB (MEM/WB reg).
 * No forwarding from EX1, EX2, EX3, or EX4 — those stages are timing
 * splits only.
 *
 * Data-hazard classes:
 *
 *   1. EX2→EX1 hazard: EX2 writes a register that EX1 reads.
 *      Producer is 3 stages from MEM.  Stall 1 cycle; producer
 *      advances to EX3 → hazard 2 → stall → EX4 → hazard 3 →
 *      stall → MEM for FU.  Self-resolving in 3 stall cycles.
 *
 *   2. EX3→EX1 hazard: EX3 writes a register that EX1 reads.
 *      Producer is 2 stages from MEM.  Stall 1 cycle; producer
 *      advances to EX4 → hazard 3 → stall → MEM for FU.
 *      Self-resolving in 2 stall cycles.
 *
 *   3. EX4→EX1 hazard: EX4 writes a register that EX1 reads.
 *      Producer is 1 stage from MEM.  Stall 1 cycle; result
 *      lands in EX4/MEM for FU.
 *
 *   4. MEM-load→EX1 hazard: a load in MEM has not yet produced
 *      its data (available after WB).  Stall 1 cycle; data
 *      lands in MEM/WB for FU.
 *
 *   5. Multi-cycle (BDT/SWP) in EX4: these instructions perform
 *      ALL register writes through the BDTU in MEM, so their
 *      wr_en1/wr_en2 are 0 in EX4.  Stall 1 cycle so the
 *      instruction reaches MEM and the BDTU starts.
 *
 * Branch penalty: 6 cycles (resolved in EX4, flush IF1–EX3).
 *
 * Priority (highest first):
 *      BDTU busy  >  branch flush  >  data-hazard stall
 */

`ifndef HDU_V
`define HDU_V

`include "define.v"

module hdu (
    // ── EX1/EX2 register: EX2 destination info ──────────
    input wire [3:0] ex1ex2_wd1,
    input wire       ex1ex2_we1,
    input wire [3:0] ex1ex2_wd2,
    input wire       ex1ex2_we2,

    // ── EX2/EX3 register: EX3 destination info ──────────
    //    (NEW in v2.1 — ALU stage)
    input wire [3:0] ex2ex3_wd1,
    input wire       ex2ex3_we1,
    input wire [3:0] ex2ex3_wd2,
    input wire       ex2ex3_we2,

    // ── EX3/EX4 register: EX4 destination info ──────────
    //    (was ex2ex3 in v2.0 — cond eval stage)
    input wire [3:0] ex3ex4_wd1,
    input wire       ex3ex4_we1,
    input wire [3:0] ex3ex4_wd2,
    input wire       ex3ex4_we2,
    input wire       ex3ex4_is_multi_cycle,
    input wire       ex3ex4_valid,

    // ── EX4/MEM register: MEM-stage load info ───────────
    //    (was ex3mem in v2.0)
    input wire       ex4mem_is_load,
    input wire [3:0] ex4mem_wd1,
    input wire       ex4mem_we1,

    // ── EX1 source registers ────────────────────────────
    input wire [3:0] ex1_rn,
    input wire [3:0] ex1_rm,
    input wire [3:0] ex1_rs,
    input wire [3:0] ex1_rd_store,
    input wire       ex1_use_rn,
    input wire       ex1_use_rm,
    input wire       ex1_use_rs,
    input wire       ex1_use_rd_st,

    // ── Branch (resolved in EX4) ────────────────────────
    input wire       branch_taken,

    // ── BDTU busy ───────────────────────────────────────
    input wire       bdtu_busy,

    // ── Pipeline control outputs ────────────────────────
    output wire stall_if1,
    output wire stall_if2,
    output wire stall_id,
    output wire stall_ex1,
    output wire stall_ex2,
    output wire stall_ex3,
    output wire stall_ex4,
    output wire stall_mem,

    output wire flush_if1if2,
    output wire flush_if2id,
    output wire flush_idex1,
    output wire flush_ex1ex2,
    output wire flush_ex2ex3,
    output wire flush_ex3ex4
);

// ────────────────────────────────────────────────────────
// 1. EX2→EX1 data hazard (producer 3 stages from MEM)
//    wr_en signals are UNGATED — conservative.
// ────────────────────────────────────────────────────────
wire ex2_p1_rn = ex1ex2_we1 && ex1_use_rn    && (ex1ex2_wd1 == ex1_rn);
wire ex2_p1_rm = ex1ex2_we1 && ex1_use_rm    && (ex1ex2_wd1 == ex1_rm);
wire ex2_p1_rs = ex1ex2_we1 && ex1_use_rs    && (ex1ex2_wd1 == ex1_rs);
wire ex2_p1_rd = ex1ex2_we1 && ex1_use_rd_st && (ex1ex2_wd1 == ex1_rd_store);

wire ex2_p2_rn = ex1ex2_we2 && ex1_use_rn    && (ex1ex2_wd2 == ex1_rn);
wire ex2_p2_rm = ex1ex2_we2 && ex1_use_rm    && (ex1ex2_wd2 == ex1_rm);
wire ex2_p2_rs = ex1ex2_we2 && ex1_use_rs    && (ex1ex2_wd2 == ex1_rs);
wire ex2_p2_rd = ex1ex2_we2 && ex1_use_rd_st && (ex1ex2_wd2 == ex1_rd_store);

wire ex2_ex1_hazard = ex2_p1_rn | ex2_p1_rm | ex2_p1_rs | ex2_p1_rd
                    | ex2_p2_rn | ex2_p2_rm | ex2_p2_rs | ex2_p2_rd;

// ────────────────────────────────────────────────────────
// 2. EX3→EX1 data hazard (producer 2 stages from MEM)
//    NEW in v2.1 — ALU stage.
//    wr_en signals are UNGATED — conservative.
// ────────────────────────────────────────────────────────
wire ex3_p1_rn = ex2ex3_we1 && ex1_use_rn    && (ex2ex3_wd1 == ex1_rn);
wire ex3_p1_rm = ex2ex3_we1 && ex1_use_rm    && (ex2ex3_wd1 == ex1_rm);
wire ex3_p1_rs = ex2ex3_we1 && ex1_use_rs    && (ex2ex3_wd1 == ex1_rs);
wire ex3_p1_rd = ex2ex3_we1 && ex1_use_rd_st && (ex2ex3_wd1 == ex1_rd_store);

wire ex3_p2_rn = ex2ex3_we2 && ex1_use_rn    && (ex2ex3_wd2 == ex1_rn);
wire ex3_p2_rm = ex2ex3_we2 && ex1_use_rm    && (ex2ex3_wd2 == ex1_rm);
wire ex3_p2_rs = ex2ex3_we2 && ex1_use_rs    && (ex2ex3_wd2 == ex1_rs);
wire ex3_p2_rd = ex2ex3_we2 && ex1_use_rd_st && (ex2ex3_wd2 == ex1_rd_store);

wire ex3_ex1_hazard = ex3_p1_rn | ex3_p1_rm | ex3_p1_rs | ex3_p1_rd
                    | ex3_p2_rn | ex3_p2_rm | ex3_p2_rs | ex3_p2_rd;

// ────────────────────────────────────────────────────────
// 3. EX4→EX1 data hazard (producer 1 stage from MEM)
//    Was EX3→EX1 in v2.0.
//    wr_en signals are UNGATED — conservative.
// ────────────────────────────────────────────────────────
wire ex4_p1_rn = ex3ex4_we1 && ex1_use_rn    && (ex3ex4_wd1 == ex1_rn);
wire ex4_p1_rm = ex3ex4_we1 && ex1_use_rm    && (ex3ex4_wd1 == ex1_rm);
wire ex4_p1_rs = ex3ex4_we1 && ex1_use_rs    && (ex3ex4_wd1 == ex1_rs);
wire ex4_p1_rd = ex3ex4_we1 && ex1_use_rd_st && (ex3ex4_wd1 == ex1_rd_store);

wire ex4_p2_rn = ex3ex4_we2 && ex1_use_rn    && (ex3ex4_wd2 == ex1_rn);
wire ex4_p2_rm = ex3ex4_we2 && ex1_use_rm    && (ex3ex4_wd2 == ex1_rm);
wire ex4_p2_rs = ex3ex4_we2 && ex1_use_rs    && (ex3ex4_wd2 == ex1_rs);
wire ex4_p2_rd = ex3ex4_we2 && ex1_use_rd_st && (ex3ex4_wd2 == ex1_rd_store);

wire ex4_ex1_hazard = ex4_p1_rn | ex4_p1_rm | ex4_p1_rs | ex4_p1_rd
                    | ex4_p2_rn | ex4_p2_rm | ex4_p2_rs | ex4_p2_rd;

// ────────────────────────────────────────────────────────
// 4. MEM-load→EX1 data hazard
//    Load is in MEM; data not available until WB.
// ────────────────────────────────────────────────────────
wire mem_ld_rn = ex4mem_is_load && ex4mem_we1 && ex1_use_rn    && (ex4mem_wd1 == ex1_rn);
wire mem_ld_rm = ex4mem_is_load && ex4mem_we1 && ex1_use_rm    && (ex4mem_wd1 == ex1_rm);
wire mem_ld_rs = ex4mem_is_load && ex4mem_we1 && ex1_use_rs    && (ex4mem_wd1 == ex1_rs);
wire mem_ld_rd = ex4mem_is_load && ex4mem_we1 && ex1_use_rd_st && (ex4mem_wd1 == ex1_rd_store);

wire mem_load_ex1_hazard = mem_ld_rn | mem_ld_rm | mem_ld_rs | mem_ld_rd;

// ────────────────────────────────────────────────────────
// 5. Multi-cycle (BDT/SWP) in EX4
//    (was EX3 in v2.0)
// ────────────────────────────────────────────────────────
wire mc_ex4_hazard = ex3ex4_is_multi_cycle && ex3ex4_valid;

// ────────────────────────────────────────────────────────
// Stall / flush arbitration
// ────────────────────────────────────────────────────────
wire bdtu_stall   = bdtu_busy;
wire branch_flush = branch_taken && !bdtu_stall;
wire hazard_stall = (ex2_ex1_hazard || ex3_ex1_hazard || ex4_ex1_hazard
                     || mem_load_ex1_hazard || mc_ex4_hazard)
                    && !bdtu_stall && !branch_flush;

// ── Stalls ──
// BDTU: freeze entire pipeline (IF1 through MEM).
// Hazard: freeze IF1, IF2, ID, EX1; EX2/EX3/EX4/MEM continue.
assign stall_if1 = bdtu_stall | hazard_stall;
assign stall_if2 = bdtu_stall | hazard_stall;
assign stall_id  = bdtu_stall | hazard_stall;
assign stall_ex1 = bdtu_stall | hazard_stall;
assign stall_ex2 = bdtu_stall;
assign stall_ex3 = bdtu_stall;
assign stall_ex4 = bdtu_stall;
assign stall_mem = bdtu_stall;

// ── Flushes ──
// Branch: kill the 6 younger instructions (IF1/IF2 through EX3/EX4).
// Hazard stall: insert bubble into EX2 (flush EX1/EX2).
assign flush_if1if2 = branch_flush;
assign flush_if2id  = branch_flush;
assign flush_idex1  = branch_flush;
assign flush_ex1ex2 = branch_flush | hazard_stall;
assign flush_ex2ex3 = branch_flush;
assign flush_ex3ex4 = branch_flush;

endmodule

`endif // HDU_V