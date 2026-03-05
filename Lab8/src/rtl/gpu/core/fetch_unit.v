/* file: fetch_unit.v
Description: Fetch unit for the CUDA-like SM core.
Author: Jeremy Cai
Date: Feb. 28, 2026
Version: 1.2
Revision history:
    - Feb. 28, 2026: v1.0 — Initial implementation.
    - Mar. 04, 2026: v1.1 — Add convergence redirect (conv_redirect /
        conv_target_pc) and pc_out for SIMT convergence checker.
        Remove erroneous self-include.
        PC redirect priority: conv_redirect > branch_taken > PC+1.
    - Mar. 04, 2026: v1.2 — add post_redirect register to
        insert a 1-cycle bubble after any PC redirect that coincides
        with a stall-release transition, ensuring the synchronous BRAM
        output has settled before fetch_valid reasserts.
*/

`ifndef FETCH_UNIT_V
`define FETCH_UNIT_V

`include "gpu_define.v"

module fetch_unit (
    input wire clk,
    input wire rst_n,

    // IMEM interface
    output wire [`GPU_IMEM_ADDR_WIDTH-1:0] imem_addr,

    // Kernel control
    input wire kernel_start,
    input wire [`GPU_PC_WIDTH-1:0] kernel_entry_pc,
    output wire kernel_done,
    output wire running,

    // Branch / flush from ID stage
    input wire branch_taken,
    input wire [`GPU_PC_WIDTH-1:0] branch_target,

    // Convergence redirect from SIMT controller (v1.1)
    input wire conv_redirect,
    input wire [`GPU_PC_WIDTH-1:0] conv_target_pc,

    // Stall
    input wire front_stall,

    // RET detection
    input wire ret_detected,

    // IF/ID pipeline outputs
    output reg [`GPU_PC_WIDTH-1:0] if_id_pc,
    output reg fetch_valid,

    // Current PC (v1.1 — for convergence checker)
    output wire [`GPU_PC_WIDTH-1:0] pc_out
);

    // ================================================================
    // PC Register
    // ================================================================
    reg [`GPU_PC_WIDTH-1:0] pc_reg;
    reg running_r;

    assign running = running_r;
    assign pc_out = pc_reg;

    wire [`GPU_PC_WIDTH-1:0] pc_plus_1 = pc_reg + 1;

    assign imem_addr = pc_reg[`GPU_IMEM_ADDR_WIDTH-1:0];

    // ================================================================
    // Bug A fix: post-redirect bubble for stall-release transitions
    // ================================================================
    // was_stalled: 1-cycle delayed version of front_stall.
    // post_redirect: asserts for 1 cycle after a redirect that fires
    // on the same cycle a stall releases (was_stalled=1).  This gives
    // the synchronous BRAM an extra cycle to present the new address's
    // data.  Normal branches (not preceded by stall) are unaffected.
    reg was_stalled;
    reg post_redirect;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            was_stalled <= 1'b0;
        else
            was_stalled <= front_stall;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            post_redirect <= 1'b0;
        else if (kernel_start)
            post_redirect <= 1'b0;
        else
            post_redirect <= (branch_taken | conv_redirect) & was_stalled;
    end

    // ================================================================
    // PC update + running state
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_reg <= {`GPU_PC_WIDTH{1'b0}};
            running_r <= 1'b0;
        end else if (kernel_start) begin
            pc_reg <= kernel_entry_pc;
            running_r <= 1'b1;
        end else if (ret_detected) begin
            running_r <= 1'b0;
        end else if (running_r && !front_stall) begin
            // Priority: convergence redirect > branch redirect > PC+1
            if (conv_redirect)
                pc_reg <= conv_target_pc;
            else if (branch_taken)
                pc_reg <= branch_target;
            else if (!post_redirect)
                pc_reg <= pc_plus_1;
            // else: post_redirect — hold PC for BRAM settle cycle
        end
    end

    // ================================================================
    // IF/ID Pipeline Register
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            if_id_pc <= {`GPU_PC_WIDTH{1'b0}};
            fetch_valid <= 1'b0;
        end else if (kernel_start) begin
            if_id_pc <= kernel_entry_pc;
            fetch_valid <= 1'b0;
        end else if (front_stall) begin
            // hold
        end else if (post_redirect) begin
            // Bug A fix: extra bubble after redirect-from-stall
            fetch_valid <= 1'b0;
        end else if (conv_redirect) begin
            // Convergence phase-0 redirect: squash instruction at reconv_pc
            fetch_valid <= 1'b0;
        end else if (branch_taken) begin
            fetch_valid <= 1'b0;
        end else if (ret_detected) begin
            fetch_valid <= 1'b0;
        end else begin
            if_id_pc <= pc_reg;
            fetch_valid <= running_r;
        end
    end

    // ================================================================
    // Pipeline drain counter (kernel_done)
    // ================================================================
    reg [2:0] drain_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            drain_counter <= 3'd0;
        else if (kernel_start)
            drain_counter <= 3'd0;
        else if (ret_detected)
            drain_counter <= 3'd6;
        else if (drain_counter != 3'd0)
            drain_counter <= drain_counter - 3'd1;
    end

    assign kernel_done = (drain_counter == 3'd1);

endmodule

`endif // FETCH_UNIT_V