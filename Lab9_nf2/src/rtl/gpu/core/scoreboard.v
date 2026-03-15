/* file: scoreboard.v
 Description: Scoreboard for the CUDA-like SM core pipeline.
 Tracks in-flight register writes and stalls on RAW hazards.
 Author: Jeremy Cai
 Date: Mar. 11, 2026
 Version: 2.1
 Revision history:
    - Feb. 27, 2026: v1.0 — Initial implementation.
    - Mar. 05, 2026: v1.1 — Add synchronous clear input for kernel_start.
      Without this, stale pending bits from a deadlocked or incomplete kernel
      persist across kernel launches, causing permanent front_stall.
    - Mar. 07, 2026: v2.0 — Two-part timing fix targeting the 13-level
      combinational critical path through the stall/issue feedback loop.
    - Mar. 11, 2026: v2.1 — Add predicate write in-flight tracking.
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
    // From WB stage — GPR
    input wire [3:0] wb_rD_addr,
    input wire wb_rf_we,
    input wire [3:0] wb_active_mask,
    // v2.1: Predicate write tracking
    input wire issue_pred,       // pred-writing instr issued (SETP/SET)
    input wire wb_pred_we,       // pred write completed at WB (any lane)

    // Outputs
    output wire stall,
    output wire any_pending
);

    reg [15:0] pending [0:3];

    // ================================================================
    // WB clear path (unchanged — no longer on critical path)
    // ================================================================
    wire [15:0] clr_mask = (16'b1 << wb_rD_addr);
    wire [3:0] clr_en = {4{wb_rf_we}} & wb_active_mask;

    // ================================================================
    // Issue set path — REGISTERED
    // ================================================================
    reg issue_r;
    reg [3:0] issue_rD_addr_r;
    reg [3:0] issue_active_mask_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issue_r <= 1'b0;
            issue_rD_addr_r <= 4'd0;
            issue_active_mask_r <= 4'd0;
        end else if (clear) begin
            issue_r <= 1'b0;
        end else begin
            issue_r <= issue & rf_we;
            issue_rD_addr_r <= rD_addr;
            issue_active_mask_r <= active_mask;
        end
    end

    wire [15:0] set_mask_r = (16'b1 << issue_rD_addr_r);
    wire [3:0] set_en_r = {4{issue_r}} & issue_active_mask_r;

    // ================================================================
    // Pending register update
    // ================================================================
    genvar t;
    generate
        for (t = 0; t < 4; t = t + 1) begin : gen_pending
            wire [15:0] clr_vec = clr_en[t] ? clr_mask : 16'b0;
            wire [15:0] set_vec = set_en_r[t] ? set_mask_r : 16'b0;

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

    // ================================================================
    // Hazard detection
    // ================================================================
    wire [3:0] hazard;

    generate
        for (t = 0; t < 4; t = t + 1) begin : gen_check
            wire [15:0] pend_eff = pending[t];

            wire src_a = uses_rA & pend_eff[rA_addr];
            wire src_b = uses_rB & pend_eff[rB_addr];
            wire src_d = (is_fma | is_st) & pend_eff[rD_addr];

            assign hazard[t] = src_a | src_b | src_d;
        end
    endgenerate

    // ================================================================
    // Bypass comparator
    // ================================================================
    wire bypass_hit_a = uses_rA & (rA_addr == issue_rD_addr_r);
    wire bypass_hit_b = uses_rB & (rB_addr == issue_rD_addr_r);
    wire bypass_hit_d = (is_fma | is_st) & (rD_addr == issue_rD_addr_r);

    wire bypass_thread_overlap = |(active_mask & issue_active_mask_r);

    wire bypass_stall = issue_r & bypass_thread_overlap
                      & (bypass_hit_a | bypass_hit_b | bypass_hit_d);

    // ================================================================
    // Predicate write in-flight counter
    // ================================================================
    reg issue_pred_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)     issue_pred_r <= 1'b0;
        else if (clear) issue_pred_r <= 1'b0;
        else            issue_pred_r <= issue_pred;
    end

    reg [1:0] pred_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pred_cnt <= 2'd0;
        else if (clear)
            pred_cnt <= 2'd0;
        else if (issue_pred_r & ~wb_pred_we)
            pred_cnt <= pred_cnt + 2'd1;
        else if (~issue_pred_r & wb_pred_we & (pred_cnt != 2'd0))
            pred_cnt <= pred_cnt - 2'd1;
    end

    // ================================================================
    // Outputs
    // ================================================================

    // Stall: regular pending-based hazard OR 1-cycle bypass.
    assign stall = |(active_mask & hazard) | bypass_stall;

    // any_pending: GPR pending OR registered GPR issue OR pred in-flight.
    // pred_cnt != 0 covers pred writes between issue+1 and WB.
    // issue_pred_r covers the 1-cycle gap between issue and counter update.
    assign any_pending = |{pending[3], pending[2], pending[1], pending[0]}
                       | issue_r | issue_pred_r | (pred_cnt != 2'd0);

endmodule

`endif // SCOREBOARD_V