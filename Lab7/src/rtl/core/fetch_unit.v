/* file: fetch_unit.v
 Description: This file implements the fetch unit for the CUDA-like SM core.
 Author: Jeremy Cai
 Date: Feb. 28, 2026
 Version: 1.0
 Revision history:
    - Feb. 28, 2026: Initial implementation of the fetch unit for the CUDA-like SM core.
*/

`ifndef FETCH_UNIT_V
`define FETCH_UNIT_V

`include "gpu_define.v"
`include "fetch_unit.v"

module fetch_unit (
    input wire clk,
    input wire rst_n,

    // IMEM interface (external BRAM)
    output wire [`GPU_IMEM_ADDR_WIDTH-1:0] imem_addr,

    // Kernel control
    input wire kernel_start,
    input wire [`GPU_PC_WIDTH-1:0] kernel_entry_pc,
    output wire kernel_done,
    output wire running,

    // Branch / flush from ID stage
    input wire branch_taken,   // BRA taken (unconditional or PBRA)
    input wire [`GPU_PC_WIDTH-1:0] branch_target,

    // Stall from stall controller
    input wire front_stall,    // hold PC, hold if_id_pc, hold fetch_valid

    // RET detection from decoder
    input wire ret_detected,   // decoder sees valid RET this cycle

    // IF/ID pipeline outputs to decoder
    output reg [`GPU_PC_WIDTH-1:0] if_id_pc,   // PC of instruction in ID stage
    output reg fetch_valid                       // instruction in ID stage is valid
);

    // ================================================================
    // PC Register
    // ================================================================
    reg [`GPU_PC_WIDTH-1:0] pc_reg;
    reg running_r;

    assign running = running_r;

    wire [`GPU_PC_WIDTH-1:0] pc_plus_1 = pc_reg + 1;

    // IMEM address: truncated PC for BRAM indexing
    assign imem_addr = pc_reg[`GPU_IMEM_ADDR_WIDTH-1:0];

    // ================================================================
    // PC update + running state
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_reg <= {`GPU_PC_WIDTH{1'b0}};
            running_r <= 1'b0;
        end else if (kernel_start) begin
            // Kernel launch: load entry PC, start running
            pc_reg <= kernel_entry_pc;
            running_r <= 1'b1;
        end else if (ret_detected) begin
            // RET decoded: stop fetching, pipeline will drain
            running_r <= 1'b0;
        end else if (running_r && !front_stall) begin
            // Normal advance or branch redirect
            if (branch_taken)
                pc_reg <= branch_target;
            else
                pc_reg <= pc_plus_1;
        end
        // else: stalled or not running → hold PC
    end

    // IF/ID Pipeline Register: PC + fetch_valid
    //
    // The instruction word itself does NOT need a register here —
    // the IMEM sync-read BRAM output (imem_rdata) acts as the
    // pipeline register for the instruction. We only register:
    //   - if_id_pc: the PC corresponding to imem_rdata
    //   - fetch_valid: whether imem_rdata is a real instruction
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            if_id_pc <= {`GPU_PC_WIDTH{1'b0}};
            fetch_valid <= 1'b0;
        end else if (kernel_start) begin
            // Startup: first IMEM read in progress, not valid yet
            if_id_pc <= kernel_entry_pc;
            fetch_valid <= 1'b0;
        end else if (front_stall) begin
            // Stalled: hold everything (same instruction re-presented to decoder)
            // if_id_pc and fetch_valid hold their values
        end else if (branch_taken) begin
            // Flush: the instruction arriving next cycle from IMEM is stale
            // PC already updated to branch_target above
            fetch_valid <= 1'b0;
        end else if (ret_detected) begin
            // RET: stop delivering instructions after this cycle
            fetch_valid <= 1'b0;
        end else begin
            // Normal: capture the PC that was sent to IMEM last cycle
            if_id_pc <= pc_reg;
            fetch_valid <= running_r;
        end
    end

    // kernel_done: asserted 1 cycle after pipeline drain completes
    //
    // After RET is decoded, instructions may still be in EX/MEM/WB.
    // Wait for pipeline to drain (worst case: multi-cycle EX + MEM + WB).
    // Use a simple countdown. 5 cycles covers:
    //   RET in ID → (up to) EX(1) → MEM(1) → WB(1) + margin
    // For multi-cycle EX ops before RET, the pipeline would have
    // stalled, so by the time RET reaches ID, earlier ops are done.
    reg [2:0] drain_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain_counter <= 3'd0;
        end else if (kernel_start) begin
            drain_counter <= 3'd0;
        end else if (ret_detected) begin
            drain_counter <= 3'd4; // 4 cycles to drain
        end else if (drain_counter != 3'd0) begin
            drain_counter <= drain_counter - 3'd1;
        end
    end

    assign kernel_done = (drain_counter == 3'd1);

endmodule

`endif // FETCH_UNIT_V