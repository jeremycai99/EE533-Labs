/* file: scoreboard.v
 Description: Scoreboard for the CUDA-like SM core pipeline.
 Tracks in-flight register writes and stalls on RAW hazards.
 Author: Jeremy Cai
 Date: Feb. 27, 2026
 Version: 1.1
 Revision history:
    - Feb. 27, 2026: v1.0 — Initial implementation.
    - Mar. 05, 2026: v1.1 — Add synchronous clear input for kernel_start.
      Without this, stale pending bits from a deadlocked or incomplete kernel
      persist across kernel launches, causing permanent front_stall.
*/

`ifndef SCOREBOARD_V
`define SCOREBOARD_V

`include "gpu_define.v"

module scoreboard (
    input  wire clk,
    input  wire rst_n,
    // Synchronous clear (connected to kernel_start)
    input  wire clear,
    // From decode (ID stage)
    input  wire [3:0] rA_addr,
    input  wire [3:0] rB_addr,
    input  wire [3:0] rD_addr,
    input  wire uses_rA,
    input  wire uses_rB,
    input  wire is_fma,
    input  wire is_st,
    input  wire rf_we,
    // From SIMT controller
    input  wire [3:0] active_mask,
    // Issue handshake
    input  wire issue,
    // From WB stage
    input wire [3:0] wb_rD_addr,
    input wire wb_rf_we,
    input wire [3:0] wb_active_mask,

    // Outputs
    output wire stall,
    output wire any_pending
);
    reg [15:0] pending [0:3];

    wire [15:0] set_mask = (16'b1 << rD_addr);
    wire [15:0] clr_mask = (16'b1 << wb_rD_addr);

    wire [3:0] set_en;
    wire [3:0] clr_en;

    assign set_en = {4{issue & rf_we}} & active_mask;
    assign clr_en = {4{wb_rf_we}} & wb_active_mask;

    genvar t;
    generate
        for (t = 0; t < 4; t = t + 1) begin : gen_pending
            wire [15:0] clr_vec = clr_en[t] ? clr_mask : 16'b0;
            wire [15:0] set_vec = set_en[t] ? set_mask : 16'b0;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    pending[t] <= 16'b0;
                else if (clear)
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

            wire src_a = uses_rA & pend_eff[rA_addr];
            wire src_b = uses_rB & pend_eff[rB_addr];
            wire src_d = (is_fma | is_st) & pend_eff[rD_addr];

            assign hazard[t] = src_a | src_b | src_d;
        end
    endgenerate

    assign stall = |(active_mask & hazard);
    assign any_pending = |{pending[3], pending[2], pending[1], pending[0]};

endmodule

`endif