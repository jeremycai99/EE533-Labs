/* file: fu.v
 Description: Forwarding unit module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 16, 2026
 Version: 1.0
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

    // EX/MEM stage write-back information
    input wire [3:0] exmem_wd, // Destination register address
    input wire exmem_we, // Write-back enable
    input wire exmem_is_load, // 1 = result comes from memory (not yet available)

    // MEM/WB stage write-back information
    input wire [3:0] memwb_wd, // Destination register address
    input wire memwb_we, // Write-back enable

    // BDTU write-back information (port 1 — data register)
    input wire [3:0] bdtu_wd1, // BDTU destination register address (port 1)
    input wire bdtu_we1, // BDTU write-back enable (port 1)
    // BDTU write-back information (port 2 — base register)
    input wire [3:0] bdtu_wd2, // BDTU destination register address (port 2)
    input wire bdtu_we2, // BDTU write-back enable (port 2)

    // Forward select outputs
    output reg [2:0] fwd_a, // Forwarding mux select for Rn
    output reg [2:0] fwd_b, // Forwarding mux select for Rm
    output reg [2:0] fwd_s, // Forwarding mux select for Rs
    output reg [2:0] fwd_d  // Forwarding mux select for store data Rd
);

// Forward select encoding
localparam [2:0]
    FWD_NONE     = 3'b000, // Use register-file value
    FWD_EXMEM    = 3'b001, // Forward from EX/MEM (ALU result)
    FWD_MEMWB    = 3'b010, // Forward from MEM/WB (ALU result or load data)
    FWD_BDTU_P1  = 3'b011, // Forward from BDTU write port 1 (data)
    FWD_BDTU_P2  = 3'b100; // Forward from BDTU write port 2 (base)

// R15 (PC) is handled separately by the datapath; never forward it
// through the normal forwarding paths. Writes to R15 cause flushes.
wire exmem_valid  = exmem_we  && (exmem_wd  != 4'd15) && !exmem_is_load;
wire memwb_valid  = memwb_we  && (memwb_wd  != 4'd15);
wire bdtu1_valid  = bdtu_we1  && (bdtu_wd1  != 4'd15);
wire bdtu2_valid  = bdtu_we2  && (bdtu_wd2  != 4'd15);

// Priority: EX/MEM > MEM/WB > BDTU (BDTU writes are oldest)
// Within BDTU, port 2 (base WB) is checked after port 1 (data)
// since both could theoretically target the same register.

// Forwarding logic for operand A (Rn)
always @(*) begin
    fwd_a = FWD_NONE;
    if (ex_use_rn && ex_rn != 4'd15) begin
        if      (exmem_valid && exmem_wd == ex_rn) fwd_a = FWD_EXMEM;
        else if (memwb_valid && memwb_wd == ex_rn) fwd_a = FWD_MEMWB;
        else if (bdtu1_valid && bdtu_wd1 == ex_rn) fwd_a = FWD_BDTU_P1;
        else if (bdtu2_valid && bdtu_wd2 == ex_rn) fwd_a = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand B (Rm)
always @(*) begin
    fwd_b = FWD_NONE;
    if (ex_use_rm && ex_rm != 4'd15) begin
        if      (exmem_valid && exmem_wd == ex_rm) fwd_b = FWD_EXMEM;
        else if (memwb_valid && memwb_wd == ex_rm) fwd_b = FWD_MEMWB;
        else if (bdtu1_valid && bdtu_wd1 == ex_rm) fwd_b = FWD_BDTU_P1;
        else if (bdtu2_valid && bdtu_wd2 == ex_rm) fwd_b = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand S (Rs — shift amount register)
always @(*) begin
    fwd_s = FWD_NONE;
    if (ex_use_rs && ex_rs != 4'd15) begin
        if      (exmem_valid && exmem_wd == ex_rs) fwd_s = FWD_EXMEM;
        else if (memwb_valid && memwb_wd == ex_rs) fwd_s = FWD_MEMWB;
        else if (bdtu1_valid && bdtu_wd1 == ex_rs) fwd_s = FWD_BDTU_P1;
        else if (bdtu2_valid && bdtu_wd2 == ex_rs) fwd_s = FWD_BDTU_P2;
    end
end

// Forwarding logic for operand D (store data register)
//   STR Rd, [Rn, Rm] — Rd value needed at MEM stage for the
//   write data bus.  We forward to EX so it can be latched into
//   the EX/MEM pipeline register and presented to the memory.
always @(*) begin
    fwd_d = FWD_NONE;
    if (ex_use_rd_st && ex_rd_store != 4'd15) begin
        if      (exmem_valid && exmem_wd == ex_rd_store) fwd_d = FWD_EXMEM;
        else if (memwb_valid && memwb_wd == ex_rd_store) fwd_d = FWD_MEMWB;
        else if (bdtu1_valid && bdtu_wd1 == ex_rd_store) fwd_d = FWD_BDTU_P1;
        else if (bdtu2_valid && bdtu_wd2 == ex_rd_store) fwd_d = FWD_BDTU_P2;
    end
end

endmodule

`endif // FU_V
