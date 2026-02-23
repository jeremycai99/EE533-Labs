/* file: fu.v
 * Forwarding Unit — 6-stage pipeline (IF→ID→EX1→EX2→MEM→WB)
 *
 * The FU sits in the EX1 stage and selects the most-recent
 * value for each source operand.  Forwarding sources:
 *
 *   EX2/MEM register (MEM stage) — result computed 1 cycle ago
 *     Port 1: ALU result (blocked when instruction is a load)
 *     Port 2: base writeback / secondary result (always valid)
 *
 *   MEM/WB register (WB stage) — result computed 2 cycles ago
 *     Port 1: ALU result or load data
 *     Port 2: base writeback / secondary result
 *
 *   BDTU write ports — active during multi-cycle BDT/SWP
 *     Port 1: data register being written this cycle
 *     Port 2: base register writeback
 *
 * The EX2 stage (1 cycle ahead of EX1) is NOT a forwarding
 * source; that case is handled by the HDU stall.
 */

`ifndef FU_V
`define FU_V

`include "define.v"

module fu (
    // ── Source register addresses read in EX1 stage ─────
    input wire [3:0] ex_rn,
    input wire [3:0] ex_rm,
    input wire [3:0] ex_rs,
    input wire [3:0] ex_rd_store,

    input wire ex_use_rn,
    input wire ex_use_rm,
    input wire ex_use_rs,
    input wire ex_use_rd_st,

    // ── EX2/MEM — port 1 (primary: Rd / RdLo) ──────────
    input wire [3:0] exmem_wd1,
    input wire       exmem_we1,
    input wire       exmem_is_load,

    // ── EX2/MEM — port 2 (secondary: base WB / RdHi) ───
    input wire [3:0] exmem_wd2,
    input wire       exmem_we2,

    // ── MEM/WB — port 1 ────────────────────────────────
    input wire [3:0] memwb_wd1,
    input wire       memwb_we1,

    // ── MEM/WB — port 2 ────────────────────────────────
    input wire [3:0] memwb_wd2,
    input wire       memwb_we2,

    // ── BDTU — port 1 (data register) ──────────────────
    input wire [3:0] bdtu_wd1,
    input wire       bdtu_we1,

    // ── BDTU — port 2 (base register) ──────────────────
    input wire [3:0] bdtu_wd2,
    input wire       bdtu_we2,

    // ── Forward select outputs ─────────────────────────
    output reg [2:0] fwd_a,
    output reg [2:0] fwd_b,
    output reg [2:0] fwd_s,
    output reg [2:0] fwd_d
);

// ────────────────────────────────────────────────────────
// Forward select encoding
// ────────────────────────────────────────────────────────
localparam [2:0]
    FWD_NONE     = 3'b000,
    FWD_EXMEM    = 3'b001,  // EX2/MEM port 1 (ALU result)
    FWD_MEMWB    = 3'b010,  // MEM/WB  port 1 (ALU or load data)
    FWD_BDTU_P1  = 3'b011,  // BDTU    port 1 (data register)
    FWD_BDTU_P2  = 3'b100,  // BDTU    port 2 (base register)
    FWD_EXMEM_P2 = 3'b101,  // EX2/MEM port 2 (base WB / RdHi)
    FWD_MEMWB_P2 = 3'b110;  // MEM/WB  port 2 (base WB / RdHi)

// ── Validity checks ────────────────────────────────────
// R15 (PC) is handled by the datapath; never forward it.
// EX2/MEM port 1 is blocked when the instruction is a load
// (data not yet returned from memory).
// EX2/MEM port 2 is always valid when enabled (ALU result).
wire exmem_valid  = exmem_we1 && (exmem_wd1 != 4'd15) && !exmem_is_load;
wire exmem_valid2 = exmem_we2 && (exmem_wd2 != 4'd15);
wire memwb_valid  = memwb_we1 && (memwb_wd1 != 4'd15);
wire memwb_valid2 = memwb_we2 && (memwb_wd2 != 4'd15);
wire bdtu1_valid  = bdtu_we1  && (bdtu_wd1  != 4'd15);
wire bdtu2_valid  = bdtu_we2  && (bdtu_wd2  != 4'd15);

// ────────────────────────────────────────────────────────
// Priority (most-recent instruction wins; within an
// instruction, primary port wins over secondary):
//
//   1. EX2/MEM port 1  (youngest, primary)
//   2. EX2/MEM port 2  (youngest, secondary)
//   3. MEM/WB  port 1  (older,    primary)
//   4. MEM/WB  port 2  (older,    secondary)
//   5. BDTU    port 1  (multi-cycle, data)
//   6. BDTU    port 2  (multi-cycle, base)
// ────────────────────────────────────────────────────────

// Forwarding logic for operand A (Rn)
always @(*) begin
    fwd_a = FWD_NONE;
    if (ex_use_rn && ex_rn != 4'd15) begin
        if      (exmem_valid  && exmem_wd1 == ex_rn) fwd_a = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rn) fwd_a = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1 == ex_rn) fwd_a = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rn) fwd_a = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rn) fwd_a = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rn) fwd_a = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand B (Rm)
always @(*) begin
    fwd_b = FWD_NONE;
    if (ex_use_rm && ex_rm != 4'd15) begin
        if      (exmem_valid  && exmem_wd1 == ex_rm) fwd_b = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rm) fwd_b = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1 == ex_rm) fwd_b = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rm) fwd_b = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rm) fwd_b = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rm) fwd_b = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand S (Rs)
always @(*) begin
    fwd_s = FWD_NONE;
    if (ex_use_rs && ex_rs != 4'd15) begin
        if      (exmem_valid  && exmem_wd1 == ex_rs) fwd_s = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rs) fwd_s = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1 == ex_rs) fwd_s = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rs) fwd_s = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rs) fwd_s = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rs) fwd_s = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand D (store data register)
always @(*) begin
    fwd_d = FWD_NONE;
    if (ex_use_rd_st && ex_rd_store != 4'd15) begin
        if      (exmem_valid  && exmem_wd1 == ex_rd_store) fwd_d = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rd_store) fwd_d = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1 == ex_rd_store) fwd_d = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rd_store) fwd_d = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rd_store) fwd_d = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rd_store) fwd_d = FWD_BDTU_P2;
    end
end

endmodule

`endif // FU_V