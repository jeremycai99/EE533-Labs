/* file: test_bf16mult.v
 Description: Simulation/synthesis wrapper for the Xilinx bf16mult IP.
 Matches IP interface: clk, a[15:0], b[15:0], result[15:0].
 Latency: 1 clock cycle.
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 */

`ifndef TEST_BF16MULT_V
`define TEST_BF16MULT_V

`include "xilinx_prim_min.v"
`include "bf16mult.v"

module test_bf16mult (
    clk, a, b, result
);
    input clk;
    input [15:0] a;
    input [15:0] b;
    output [15:0] result;

    bf16mult u_bf16mult (
        .clk(clk),
        .a(a),
        .b(b),
        .result(result)
    );

endmodule

`endif // TEST_BF16MULT_V
