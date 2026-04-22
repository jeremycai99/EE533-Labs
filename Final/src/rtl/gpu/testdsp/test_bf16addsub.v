/* file: test_bf16addsub.v
 Description: Simulation/synthesis wrapper for the Xilinx bf16addsub IP.
 Matches IP interface: clk, operation[5:0], a[15:0], b[15:0], result[15:0].
 Latency: 2 clock cycles.
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 */

`ifndef TEST_BF16ADDSUB_V
`define TEST_BF16ADDSUB_V

`include "xilinx_prim_min.v"
`include "bf16addsub.v"

module test_bf16addsub (
    clk, operation, a, b, result
);
    input clk;
    input [5:0] operation;
    input [15:0] a;
    input [15:0] b;
    output [15:0] result;

    bf16addsub u_bf16addsub (
        .clk(clk),
        .operation(operation),
        .a(a),
        .b(b),
        .result(result)
    );

endmodule

`endif // TEST_BF16ADDSUB_V
