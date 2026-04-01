/* file: pplint16mult.v
 Description: This file implements the 16-bit integer multiplication unit for the GPU.
 The unit is designed to support pipelined execution for high throughput in GPU applications.
 Please note that the int16 multiplication is gonna truncated to 16 bits. Please make sure the range of the input values
    is within the range of -128 to 127 to avoid overflow if accurate results are expected. This design implement a Dadda
    tree multiplier for efficient multiplication of two 16-bit integers, producing a 16-bit result.
 This version of the int16 multiplier is designed to be pipelined, allowing for high throughput in GPU applications.
 The pipeline stages are as follows:
    - Stage 1: Partial Product Generation and Initial Reduction (CSA Tree)
    - Stage 2: Final Addition and Output
 The output is registered and output one clock cycle after the final addition stage's clock
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.0
 Revision history:
    - Feb. 26, 2026: Initial implementation of the pipelined 16-bit integer multiplication unit.
    - Mar. 6, 2026: Updated to use Xilinx Multiplier IP for better performance and resource utilization.
*/

`ifndef PPLINT16MULT_V
`define PPLINT16MULT_V

`include "gpu_define.v"
`include "test_int16mult.v"

module pplint16mult (
    input wire clk,
    input wire rst_n,       // unused — IP has no async reset; kept for interface compat
    input wire [15:0] a,
    input wire [15:0] b,
    output wire [15:0] result
);

    wire [31:0] product;

    test_int16mult u_mult (
        .clk(clk),
        .a(a),
        .b(b),
        .p(product)
    );

    assign result = product[15:0];

endmodule

`endif // PPLINT16MULT_V