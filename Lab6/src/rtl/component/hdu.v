/* file: hdu.v
 Description: Hazard detection unit module for the Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 16, 2026
 Version: 2.0 — added secondary write-port (port 2) interface
 */

`ifndef HDU_V
`define HDU_V

`include "define.v"

module hdu (
    input wire       idex_is_load,   // ID/EX instruction is a load (LDR/LDRB etc.)
    input wire [3:0] idex_wd1,        // ID/EX primary dest register (Rd for load)
    input wire       idex_we1,        // ID/EX primary write-back enable

    // ── Secondary write port from ID/EX (base writeback / RdHi) ──
    //
    // These are provided so the HDU has full visibility of every
    // in-flight write.  In the current design they do NOT trigger
    // load-use stalls because port-2's value is always the ALU
    // result (base±offset) or multiply-hi — both computed in EX
    // and available for forwarding from EX/MEM.
    input wire [3:0] idex_wd2,       // ID/EX secondary dest register
    input wire       idex_we2,       // ID/EX secondary write-back enable

    // Source register addresses of the instruction in IF/ID (decode stage)
    input wire [3:0] ifid_rn,        // Rn address being decoded
    input wire [3:0] ifid_rm,        // Rm address being decoded
    input wire [3:0] ifid_rs,        // Rs address being decoded
    input wire [3:0] ifid_rd_store,  // Rd for a store instruction being decoded
    input wire       ifid_use_rn,    // Decoded instruction uses Rn
    input wire       ifid_use_rm,    // Decoded instruction uses Rm
    input wire       ifid_use_rs,    // Decoded instruction uses Rs
    input wire       ifid_use_rd_st, // Decoded instruction is store using Rd

    // Branch detection
    input wire       branch_taken,   // Branch resolved as taken in EX

    // Multi-cycle stall
    input wire       bdtu_busy,      // BDTU is processing (LDM/STM/SWP)

    // Pipeline control outputs
    output wire stall_if,   // Stall the IF stage (hold PC and IF/ID register)
    output wire stall_id,   // Stall the ID stage (hold IF/ID → ID/EX latch)
    output wire stall_ex,   // Stall the EX stage (hold ID/EX → EX/MEM latch)
    output wire stall_mem,  // Stall the MEM stage (hold EX/MEM → MEM/WB latch)

    output wire flush_ifid, // Flush IF/ID register (insert bubble into ID)
    output wire flush_idex, // Flush ID/EX register (insert bubble into EX)
    output wire flush_exmem // Flush EX/MEM register (insert bubble into MEM)
);

// ────────────────────────────────────────────────────────────
// Load-use hazard detection — PRIMARY port (port 1)
// ────────────────────────────────────────────────────────────
// A load-use hazard exists when:
//   • The instruction in ID/EX is a load  (idex_is_load)
//   • It will write back                  (idex_we)
//   • Its destination is not R15           (idex_wd != 15)
//   • The instruction in IF/ID reads that same register

wire load_use_rn = ifid_use_rn    && (ifid_rn       == idex_wd1);
wire load_use_rm = ifid_use_rm    && (ifid_rm       == idex_wd1);
wire load_use_rs = ifid_use_rs    && (ifid_rs       == idex_wd1);
wire load_use_rd = ifid_use_rd_st && (ifid_rd_store == idex_wd1);

wire load_use_hazard = idex_is_load && idex_we1
                     && (idex_wd1 != 4'd15)
                     && (load_use_rn | load_use_rm | load_use_rs | load_use_rd);

// ────────────────────────────────────────────────────────────
// Port-2 note
// ────────────────────────────────────────────────────────────
// No load-use stall is needed for port 2 in the current design:
//
//   • SDT with writeback : wb_data2 = ALU result  (available at EX)
//   • Long multiply      : wb_data2 = RdHi result (available at EX)
//
// Both values can be forwarded from EX/MEM without stalling.
// The forwarding unit (fu.v) handles this via FWD_EXMEM_P2 /
// FWD_MEMWB_P2.
//
//
// wire load_use_p2_rn = ifid_use_rn    && (ifid_rn       == idex_wd2);
// wire load_use_p2_rm = ifid_use_rm    && (ifid_rm       == idex_wd2);
// wire load_use_p2_rs = ifid_use_rs    && (ifid_rs       == idex_wd2);
// wire load_use_p2_rd = ifid_use_rd_st && (ifid_rd_store == idex_wd2);

// wire load_use_hazard_p2 = idex_is_load2 && idex_we2
//                         && (idex_wd2 != 4'd15)
//                         && (load_use_p2_rn | load_use_p2_rm
//                           | load_use_p2_rs | load_use_p2_rd);

// Then OR it into load_use_hazard.

// wire load_use_hazard_full = load_use_hazard | load_use_hazard_p2;

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
wire bdtu_stall = bdtu_busy;

// Branch flush: kill the two instructions behind the branch.
wire branch_flush = branch_taken && !bdtu_stall;

// Load-use stall: freeze IF and ID, bubble into EX.
wire lu_stall = load_use_hazard && !bdtu_stall && !branch_flush;

// ── Output assignments ──

// Stall signals
assign stall_if  = bdtu_stall | lu_stall;
assign stall_id  = bdtu_stall | lu_stall;
assign stall_ex  = bdtu_stall;
assign stall_mem = bdtu_stall;

// Flush signals
assign flush_ifid  = branch_flush;
assign flush_idex  = branch_flush | lu_stall;

// Branch effective in EX stage and flush ex/mem or not doesn't
// matter so hardcode the flush_exmem to 0 for simplicity.
// This fix is intended to fix the multiple branch corner cases.
assign flush_exmem = 1'b0;

endmodule

`endif // HDU_V