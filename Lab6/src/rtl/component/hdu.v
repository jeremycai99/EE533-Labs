/* file: hdu.v
 Description: Hazard detection unit module for the Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 16, 2026
 Version: 1.0
 */

`ifndef HDU_V
`define HDU_V

`include "define.v"

module hdu (
    input wire idex_is_load, // ID/EX instruction is a load (LDR/LDRB/LDRH etc.)
    input wire [3:0] idex_wd, // ID/EX destination register address (Rd for load)
    input wire idex_we, // ID/EX will write back

    // Source register addresses of the instruction in IF/ID (decode stage)
    input wire [3:0] ifid_rn, // Rn address being decoded
    input wire [3:0] ifid_rm, // Rm address being decoded
    input wire [3:0] ifid_rs, // Rs address being decoded
    input wire [3:0] ifid_rd_store, // Rd for a store instruction being decoded
    input wire ifid_use_rn, // Decoded instruction uses Rn
    input wire ifid_use_rm, // Decoded instruction uses Rm
    input wire ifid_use_rs, // Decoded instruction uses Rs
    input wire ifid_use_rd_st, // Decoded instruction is store using Rd

    // Branch detection
    input wire branch_taken, // Branch resolved as taken in EX

    // Multi-cycle stall
    input wire bdtu_busy, // BDTU is processing (LDM/STM/SWP)

    // Pipeline control outputs
    output wire stall_if, // Stall the IF stage (hold PC and IF/ID register)
    output wire stall_id, // Stall the ID stage (hold IF/ID → ID/EX latch)
    output wire stall_ex, // Stall the EX stage (hold ID/EX → EX/MEM latch)
    output wire stall_mem, // Stall the MEM stage (hold EX/MEM → MEM/WB latch)

    output wire flush_ifid, // Flush IF/ID register (insert bubble into ID)
    output wire flush_idex, // Flush ID/EX register (insert bubble into EX)
    output wire flush_exmem // Flush EX/MEM register (insert bubble into MEM)
);

// Load-use hazard detection

wire load_use_rn = ifid_use_rn && (ifid_rn == idex_wd);
wire load_use_rm = ifid_use_rm && (ifid_rm == idex_wd);
wire load_use_rs = ifid_use_rs && (ifid_rs == idex_wd);
wire load_use_rd = ifid_use_rd_st && (ifid_rd_store == idex_wd);

wire load_use_hazard = idex_is_load && idex_we
                     && (idex_wd != 4'd15)
                     && (load_use_rn | load_use_rm | load_use_rs | load_use_rd);

// Stall / flush generation
//
//   Priority (highest to lowest):
//
//   1. BDTU busy — freezes the entire pipeline upstream of the
//      memory interface.  The BDTU has exclusive use of the
//      register-file write ports and the data memory bus while
//      active, so IF, ID, EX, and MEM must all hold their state.
//      No flushes are needed; the pipeline simply pauses.
//
//   2. Branch taken — the branch is resolved in EX.  The
//      instructions in IF and ID are on the wrong path.
//      Flush IF/ID and ID/EX.  IF is redirected to the branch
//      target by the datapath (PC update logic).
//      No stalls are needed beyond the flush.
//
//   3. Load-use hazard — stall IF and ID for one cycle, bubble
//      into EX (flush ID/EX).

// BDTU stall: freeze everything from IF through MEM.
// No flush needed — contents stay valid; we just hold.
wire bdtu_stall = bdtu_busy;

// Branch flush: kill the two instructions behind the branch.
// IF will fetch from the branch target next cycle (not stalled).
wire branch_flush = branch_taken && !bdtu_stall;

// Load-use stall: freeze IF and ID, bubble into EX.
// Suppressed when BDTU is stalling (BDTU takes precedence) or
// when a branch flush is active (the instruction causing the
// hazard will be flushed anyway).
wire lu_stall = load_use_hazard && !bdtu_stall && !branch_flush;

// Output assignments
// Stall signals (active-high: hold the pipeline register contents)
assign stall_if  = bdtu_stall | lu_stall;
assign stall_id  = bdtu_stall | lu_stall;
assign stall_ex  = bdtu_stall;
assign stall_mem = bdtu_stall;

// Flush signals (active-high: replace pipeline register with bubble)
//   flush_ifid  : on branch taken, kill the instruction in IF/ID
//   flush_idex  : on branch taken OR load-use stall, kill the
//                 instruction in ID/EX (insert NOP/bubble)
//   flush_exmem : on branch taken, kill instruction in EX/MEM
//                 (the branch instruction itself has completed its
//                  useful work in EX; the slot can be a bubble)
//
// NOTE: During a BDTU stall, flushes are suppressed because the
// pipeline is frozen — the branch/load-use cannot be in flight
// simultaneously with an active BDTU (the BDTU stalls earlier
// stages before they can issue new instructions).

assign flush_ifid  = branch_flush;
assign flush_idex  = branch_flush | lu_stall;
assign flush_exmem = branch_flush;

endmodule

`endif // HDU_V
