/* file: scoreboard.v
 Description: This file implements the scoreboard for the CUDA-like SP (streaming processor) core pipeline.
 The scoreboard is designed to track the status of registers and manage hazards in the pipeline.
 Author: Jeremy Cai
 Date: Feb. 27, 2026
 Version: 1.0
 Revision history:
    - Feb. 27, 2026: Initial implementation of the scoreboard for the CUDA-like SP core pipeline.
*/

`ifndef SCOREBOARD_V
`define SCOREBOARD_V

`include "gpu_define.v"

module scoreboard (
    input  wire clk,
    input  wire rst_n,
    // From decode (ID stage)
    input  wire [3:0] rA_addr, // source A
    input  wire [3:0] rB_addr, // source B (R-type)
    input  wire [3:0] rD_addr, // destination (or source for FMA/ST/STS)
    input  wire uses_rA, // instruction reads rA
    input  wire uses_rB, // instruction reads rB
    input  wire is_fma, // FMA: rD is also a read source
    input  wire is_st, // ST/STS: rD is a read source, not write
    input  wire rf_we, // instruction writes GPR (not NOP/ST/BRA/etc)
    // From SIMT controller
    input  wire [3:0] active_mask,   // current active thread mask. 1111 for now
    // Issue handshake
    input  wire        issue,         // instruction issued this cycle
                                      // (decode_valid & ~stall & ~global_stall)
    // From WB stage (pipeline sideband)
    input  wire [3:0]  wb_rD_addr,    // register being written back
    input  wire        wb_rf_we,      // WB has a valid GPR write
    input  wire [3:0]  wb_active_mask,// active mask snapshot from issue time

    // Outputs
    output wire        stall,         // RAW hazard → freeze IF+ID, bubble EX
    output wire        any_pending    // any register in-flight (for WMMA drain)
);
    // Pending registers: 4 threads × 16 bits = 64 FFs
    reg [15:0] pending [0:3];

    // SET mask: one-hot decode of rD for instruction being issued
    wire [15:0] set_mask = (16'b1 << rD_addr);

    // CLEAR mask: one-hot decode of rD from WB stage
    wire [15:0] clr_mask = (16'b1 << wb_rD_addr);

    // Per-thread SET/CLEAR enables
    wire [3:0] set_en;    // which threads get SET this cycle
    wire [3:0] clr_en;    // which threads get CLEAR this cycle

    // SET fires when: issuing a GPR-writing instruction, per active thread
    assign set_en = {4{issue & rf_we}} & active_mask;

    // CLEAR fires when: WB commits a GPR write, per WB-active thread
    assign clr_en = {4{wb_rf_we}} & wb_active_mask;

    // Pending register update (SET has priority over CLEAR)

    genvar t;
    generate
        for (t = 0; t < 4; t = t + 1) begin : gen_pending
            wire [15:0] clr_vec = clr_en[t] ? clr_mask : 16'b0;
            wire [15:0] set_vec = set_en[t] ? set_mask : 16'b0;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    pending[t] <= 16'b0;
                else
                    pending[t] <= (pending[t] & ~clr_vec) | set_vec;
            end
        end
    endgenerate

    wire [3:0] hazard;

    generate
        for (t = 0; t < 4; t = t + 1) begin : gen_check
            wire [15:0] eff_clr  = clr_en[t] ? clr_mask : 16'b0;
            wire [15:0] pend_eff = pending[t] & ~eff_clr;

            // Source hazard checks
            wire src_a = uses_rA & pend_eff[rA_addr];
            wire src_b = uses_rB & pend_eff[rB_addr];

            // FMA reads rD as accumulator; ST/STS reads rD as store data
            wire src_d = (is_fma | is_st) & pend_eff[rD_addr];

            assign hazard[t] = src_a | src_b | src_d;
        end
    endgenerate

    // Stall if ANY active thread has a hazard
    assign stall = |(active_mask & hazard);

    // Any register pending across all threads (pipeline not drained)
    assign any_pending = |{pending[3], pending[2], pending[1], pending[0]};

endmodule

`endif