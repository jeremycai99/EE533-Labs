/* file: bf16pe.v
 Description: This file implements the processing element for BF16 format for SM-wise tensor core design.
 Please note that the bf16 arithmetic operations are pipelined: pplbf16mult (3 stages), pplbf16adsub (4 stages)
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.0
 Revision history:
        - Feb. 26, 2026: Initial implementation of processing element for BF16 format.
*/

`ifndef BF16PE_V
`define BF16PE_V
`include "gpu_define.v"
`include "pplbf16mult.v"
`include "pplbf16addsub.v"

module bf16pe (
    input wire clk,
    input wire rst_n,
    // Accumulator control
    input wire acc_load,
    input wire [15:0] acc_in,
    // Data flow
    input wire [15:0] a_in,
    input wire [15:0] b_in,
    input wire valid_in,
    // Pass-through (1-cycle delay)
    output reg [15:0] a_out,
    output reg [15:0] b_out,
    output reg valid_out,
    // Accumulated result
    output reg [15:0] acc_out
);
    // Pass-through registers
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out     <= 16'd0;
            b_out     <= 16'd0;
            valid_out <= 1'b0;
        end else begin
            a_out     <= a_in;
            b_out     <= b_in;
            valid_out <= valid_in;
        end
    end

    // Stage 1: Multiply a_in * b_in (3-stage pipeline)
    wire [15:0] product;
    wire mult_valid;
    pplbf16mult u_mult (
        .clk(clk),
        .rst_n(rst_n),
        .operand_a(a_in),
        .operand_b(b_in),
        .valid_in(valid_in),
        .result(product),
        .valid_out(mult_valid)
    );

    // Stage 2: Add acc_out + product (4-stage pipeline)
    wire [15:0] add_result;
    wire add_valid;
    pplbf16addsub u_add (
        .clk(clk),
        .rst_n(rst_n),
        .operand_a(acc_out),
        .operand_b(product),
        .sub(1'b0),
        .valid_in(mult_valid),
        .result(add_result),
        .valid_out(add_valid)
    );

    // Accumulator: acc_load > add_valid > hold
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_out <= 16'd0;
        else if (acc_load)
            acc_out <= acc_in;
        else if (add_valid)
            acc_out <= add_result;
    end
endmodule

`endif // BF16PE_V
