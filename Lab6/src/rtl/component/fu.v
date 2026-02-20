/* file: fu.v
 Description: Forwarding unit module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 16, 2026
 Version: 2.0 — added secondary write-port (port 2) forwarding
 */

`ifndef FU_V
`define FU_V

`include "define.v"

module fu (
    // Source register addresses read in EX stage
    input wire [3:0] ex_rn, // Rn address used in EX
    input wire [3:0] ex_rm, // Rm address used in EX
    input wire [3:0] ex_rs, // Rs address used in EX
    input wire [3:0] ex_rd_store, // Rd/Rt address for store data in EX

    // Which sources are actually needed by the EX instruction
    input wire ex_use_rn, // Instruction reads Rn
    input wire ex_use_rm, // Instruction reads Rm
    input wire ex_use_rs, // Instruction reads Rs (register shift)
    input wire ex_use_rd_st, // Instruction is a store and reads Rd

    // ── EX/MEM stage write-back — port 1 (primary: Rd / RdLo) ──
    input wire [3:0] exmem_wd1, // Destination register address
    input wire exmem_we1, // Write-back enable
    input wire exmem_is_load, // 1 = result comes from memory (not yet available)

    // ── EX/MEM stage write-back — port 2 (secondary: base WB / RdHi) ──
    input wire [3:0] exmem_wd2, // Destination register address (port 2)
    input wire exmem_we2, // Write-back enable (port 2)

    // ── MEM/WB stage write-back — port 1 ──
    input wire [3:0] memwb_wd1, // Destination register address
    input wire memwb_we1, // Write-back enable

    // ── MEM/WB stage write-back — port 2 ──
    input wire [3:0] memwb_wd2, // Destination register address (port 2)
    input wire memwb_we2, // Write-back enable (port 2)

    // ── BDTU write-back — port 1 (data register) ──
    input wire [3:0] bdtu_wd1, // BDTU destination register address (port 1)
    input wire bdtu_we1, // BDTU write-back enable (port 1)
    // ── BDTU write-back — port 2 (base register) ──
    input wire [3:0] bdtu_wd2, // BDTU destination register address (port 2)
    input wire bdtu_we2, // BDTU write-back enable (port 2)

    // Forward select outputs
    output reg [2:0] fwd_a, // Forwarding mux select for Rn
    output reg [2:0] fwd_b, // Forwarding mux select for Rm
    output reg [2:0] fwd_s, // Forwarding mux select for Rs
    output reg [2:0] fwd_d  // Forwarding mux select for store data Rd
);

// ────────────────────────────────────────────────────────────
// Forward select encoding
// ────────────────────────────────────────────────────────────
localparam [2:0]
    FWD_NONE = 3'b000, // Use register-file value
    FWD_EXMEM = 3'b001, // Forward from EX/MEM port 1 (ALU result)
    FWD_MEMWB = 3'b010, // Forward from MEM/WB port 1 (ALU result or load data)
    FWD_BDTU_P1 = 3'b011, // Forward from BDTU write port 1 (data)
    FWD_BDTU_P2 = 3'b100, // Forward from BDTU write port 2 (base)
    FWD_EXMEM_P2 = 3'b101, // Forward from EX/MEM port 2 (base WB / RdHi)
    FWD_MEMWB_P2 = 3'b110; // Forward from MEM/WB port 2 (base WB / RdHi)
//
// R15 (PC) is handled separately by the datapath; never forward
// it through the normal forwarding paths.
//
// Port 1 from EX/MEM is NOT valid when the instruction is a load,
// because the load data has not yet returned from memory.
//
// Port 2 from EX/MEM is ALWAYS valid when enabled: its value is
// the ALU result (base writeback) or multiply-hi — both computed
// in EX and already latched in the EX/MEM register.

wire exmem_valid = exmem_we1 && (exmem_wd1 != 4'd15) && !exmem_is_load;
wire exmem_valid2 = exmem_we2 && (exmem_wd2 != 4'd15);
wire memwb_valid = memwb_we1 && (memwb_wd1 != 4'd15);
wire memwb_valid2 = memwb_we2 && (memwb_wd2 != 4'd15);
wire bdtu1_valid = bdtu_we1 && (bdtu_wd1 != 4'd15);
wire bdtu2_valid = bdtu_we2 && (bdtu_wd2 != 4'd15);

// ────────────────────────────────────────────────────────────
// Priority (most-recent instruction wins; within an instruction,
// the primary port wins over the secondary port):
//
//   1. EX/MEM port 1  (youngest, primary  — ALU result)
//   2. EX/MEM port 2  (youngest, secondary — base WB / RdHi)
//   3. MEM/WB port 1  (older,   primary  — ALU result or load)
//   4. MEM/WB port 2  (older,   secondary — base WB / RdHi)
//   5. BDTU   port 1  (oldest,  data register)
//   6. BDTU   port 2  (oldest,  base register)
// ────────────────────────────────────────────────────────────

// Forwarding logic for operand A (Rn)
always @(*) begin
    fwd_a = FWD_NONE;
    if (ex_use_rn && ex_rn != 4'd15) begin
        if      (exmem_valid  && exmem_wd1  == ex_rn) fwd_a = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rn) fwd_a = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1  == ex_rn) fwd_a = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rn) fwd_a = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rn) fwd_a = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rn) fwd_a = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand B (Rm)
always @(*) begin
    fwd_b = FWD_NONE;
    if (ex_use_rm && ex_rm != 4'd15) begin
        if      (exmem_valid  && exmem_wd1  == ex_rm) fwd_b = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rm) fwd_b = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1  == ex_rm) fwd_b = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rm) fwd_b = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rm) fwd_b = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rm) fwd_b = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand S (Rs — shift amount register)
always @(*) begin
    fwd_s = FWD_NONE;
    if (ex_use_rs && ex_rs != 4'd15) begin
        if      (exmem_valid  && exmem_wd1  == ex_rs) fwd_s = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rs) fwd_s = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1  == ex_rs) fwd_s = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rs) fwd_s = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rs) fwd_s = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rs) fwd_s = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand D (store data register)
//   STR Rd, [Rn, Rm] — Rd value needed at MEM stage for the
//   write data bus.  We forward to EX so it can be latched into
//   the EX/MEM pipeline register and presented to the memory.
always @(*) begin
    fwd_d = FWD_NONE;
    if (ex_use_rd_st && ex_rd_store != 4'd15) begin
        if      (exmem_valid  && exmem_wd1  == ex_rd_store) fwd_d = FWD_EXMEM;
        else if (exmem_valid2 && exmem_wd2 == ex_rd_store) fwd_d = FWD_EXMEM_P2;
        else if (memwb_valid  && memwb_wd1  == ex_rd_store) fwd_d = FWD_MEMWB;
        else if (memwb_valid2 && memwb_wd2 == ex_rd_store) fwd_d = FWD_MEMWB_P2;
        else if (bdtu1_valid  && bdtu_wd1  == ex_rd_store) fwd_d = FWD_BDTU_P1;
        else if (bdtu2_valid  && bdtu_wd2  == ex_rd_store) fwd_d = FWD_BDTU_P2;
    end
end

endmodule

`endif // FU_V