/* file: pplbf16mult.v
 Description: Wrapper around bf16mult IP for GPU integration.
 IP latency = 1 cycle. Wrapper pipelines valid to match.
 For simulation: compile with test_bf16mult.v (behavioral model).
 For synthesis:  Xilinx IP netlist provides bf16mult.
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 */

`ifndef PPLBF16MULT_V
`define PPLBF16MULT_V

`include "test_bf16mult.v"

module pplbf16mult (
    input wire clk,
    input wire rst_n,
    input wire [15:0] operand_a,
    input wire [15:0] operand_b,
    input wire valid_in,
    output wire [15:0] result,
    output reg valid_out
);

    test_bf16mult u_bf16mult (
        .clk(clk),
        .a(operand_a),
        .b(operand_b),
        .result(result)
    );

    // Valid pipeline: 1 stage to match IP latency
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_out <= 1'b0;
        else
            valid_out <= valid_in;
    end

endmodule

`endif // PPLBF16MULT_V