/* file: hdu.v
 * Hazard Detection Unit — 6-stage pipeline (IF→ID→EX1→EX2→MEM→WB)
 *
 * Three data-hazard classes plus branch and BDTU stalls:
 *
 *   1. EX2→EX1 hazard: EX2 writes a register that EX1 reads.
 *      The ALU result is computed in EX2 but not yet in the
 *      EX2/MEM register, so the forwarding unit cannot supply it
 *      to EX1.  Stall 1 cycle; result lands in EX2/MEM for FU.
 *
 *   2. MEM-load→EX1 hazard: a load in MEM has not yet produced
 *      its data (available after WB).  The FU blocks forwarding
 *      from EX2/MEM when exmem_is_load is set.  Stall 1 cycle;
 *      data lands in MEM/WB for FU.  Combined with hazard 1 this
 *      gives a 2-cycle load-use penalty.
 *
 *   3. Multi-cycle (BDT/SWP) in EX2: these instructions perform
 *      ALL register writes through the BDTU in MEM, so their
 *      wr_en1/wr_en2 are 0 in EX2 — invisible to hazard 1.
 *      Stall 1 cycle so the instruction reaches MEM and the BDTU
 *      starts; the BDTU then stalls the pipeline and its write
 *      ports feed the forwarding unit in EX1.
 *
 *   Priority (highest first):
 *      BDTU busy  >  branch flush  >  data-hazard stall
 */

`ifndef HDU_V
`define HDU_V

`include "define.v"

module hdu (
    // ── EX1/EX2 register: EX2 destination info ──────────
    input wire [3:0] ex1ex2_wd1,            // EX2 primary dest register
    input wire       ex1ex2_we1,            // EX2 primary write enable (ungated)
    input wire [3:0] ex1ex2_wd2,            // EX2 secondary dest register
    input wire       ex1ex2_we2,            // EX2 secondary write enable (ungated)
    input wire       ex1ex2_is_multi_cycle, // EX2 is BDT/SWP (ungated)
    input wire       ex1ex2_valid,          // EX2 holds a valid instruction

    // ── EX2/MEM register: MEM-stage load info ───────────
    input wire       ex2mem_is_load,        // MEM instruction is a load
    input wire [3:0] ex2mem_wd1,            // MEM primary dest register
    input wire       ex2mem_we1,            // MEM primary write enable (gated)

    // ── EX1 source registers ────────────────────────────
    input wire [3:0] ex1_rn,
    input wire [3:0] ex1_rm,
    input wire [3:0] ex1_rs,
    input wire [3:0] ex1_rd_store,
    input wire       ex1_use_rn,
    input wire       ex1_use_rm,
    input wire       ex1_use_rs,
    input wire       ex1_use_rd_st,

    // ── Branch (resolved in EX2) ────────────────────────
    input wire       branch_taken,

    // ── BDTU busy ───────────────────────────────────────
    input wire       bdtu_busy,

    // ── Pipeline control outputs ────────────────────────
    output wire stall_if,
    output wire stall_id,
    output wire stall_ex1,
    output wire stall_ex2,
    output wire stall_mem,

    output wire flush_ifid,
    output wire flush_idex1,
    output wire flush_ex1ex2
);

// ────────────────────────────────────────────────────────
// 1. EX2→EX1 data hazard
//    wr_en signals are UNGATED (condition not yet evaluated
//    in EX2), so this is conservative — may stall when the
//    condition will not be met.  Harmless; minor CPI cost.
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
// 2. MEM-load→EX1 data hazard
//    Load is in MEM; data not available until WB.  The FU
//    will NOT forward from EX2/MEM when exmem_is_load is
//    set, so we must stall one cycle.
// ────────────────────────────────────────────────────────
wire mem_ld_rn = ex2mem_is_load && ex2mem_we1 && ex1_use_rn    && (ex2mem_wd1 == ex1_rn);
wire mem_ld_rm = ex2mem_is_load && ex2mem_we1 && ex1_use_rm    && (ex2mem_wd1 == ex1_rm);
wire mem_ld_rs = ex2mem_is_load && ex2mem_we1 && ex1_use_rs    && (ex2mem_wd1 == ex1_rs);
wire mem_ld_rd = ex2mem_is_load && ex2mem_we1 && ex1_use_rd_st && (ex2mem_wd1 == ex1_rd_store);

wire mem_load_ex1_hazard = mem_ld_rn | mem_ld_rm | mem_ld_rs | mem_ld_rd;

// ────────────────────────────────────────────────────────
// 3. Multi-cycle (BDT/SWP) in EX2
//    BDT/SWP perform register writes through BDTU in MEM,
//    bypassing wr_en1/wr_en2.  Without this stall the next
//    instruction would latch stale operands in EX1/EX2
//    before BDTU even starts.
//
//    Cost: 1 extra bubble per BDT/SWP — negligible versus
//    the multi-cycle operation itself (4–16+ cycles).
// ────────────────────────────────────────────────────────
wire mc_ex2_hazard = ex1ex2_is_multi_cycle && ex1ex2_valid;

// ────────────────────────────────────────────────────────
// Stall / flush arbitration
// ────────────────────────────────────────────────────────
wire bdtu_stall   = bdtu_busy;
wire branch_flush = branch_taken && !bdtu_stall;
wire hazard_stall = (ex2_ex1_hazard || mem_load_ex1_hazard || mc_ex2_hazard)
                    && !bdtu_stall && !branch_flush;

// ── Stalls ──
// BDTU: freeze the entire pipeline (IF through MEM).
// Hazard: freeze IF, ID, EX1; EX2 and MEM continue.
assign stall_if  = bdtu_stall | hazard_stall;
assign stall_id  = bdtu_stall | hazard_stall;
assign stall_ex1 = bdtu_stall | hazard_stall;
assign stall_ex2 = bdtu_stall;
assign stall_mem = bdtu_stall;

// ── Flushes ──
// Branch: kill the three younger instructions (IF/ID, ID/EX1, EX1/EX2).
// Hazard stall: insert bubble into EX2 (flush EX1/EX2).
assign flush_ifid   = branch_flush;
assign flush_idex1  = branch_flush;
assign flush_ex1ex2 = branch_flush | hazard_stall;

endmodule

`endif // HDU_V