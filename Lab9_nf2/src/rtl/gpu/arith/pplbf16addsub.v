/* file: pplbf16addsub.v
 Description: Wrapper around bf16addsub IP for GPU integration.
 IP latency = 2 cycles. Wrapper pipelines valid to match.
 For simulation: compile with test_bf16addsub.v (behavioral model).
 For synthesis:  Xilinx IP netlist provides bf16addsub.
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 */

`ifndef PPLBF16ADDSUB_V
`define PPLBF16ADDSUB_V

`include "test_bf16addsub.v"

module pplbf16addsub (
    input wire clk,
    input wire rst_n,
    input wire [15:0] operand_a,
    input wire [15:0] operand_b,
    input wire sub,
    input wire valid_in,
    output wire [15:0] result,
    output reg valid_out
);

    wire [5:0] operation = {5'b00000, sub}; // 000000=add, 000001=sub

    test_bf16addsub u_bf16addsub (
        .clk(clk),
        .operation(operation),
        .a(operand_a),
        .b(operand_b),
        .result(result)
    );

    // Valid pipeline: 2 stages to match IP latency
    reg valid_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            valid_d1 <= valid_in;
            valid_out <= valid_d1;
        end
    end

endmodule

`endif // PPLBF16ADDSUB_V