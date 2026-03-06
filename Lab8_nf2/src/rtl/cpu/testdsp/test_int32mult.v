/* file: test_int32mult.v
 Description: This file implements a behavioral combinational int32mult unit to be replaced by DSP resource.
 Author: Jeremy Cai
 Date: Feb. 8, 2026
 Version: 1.0
 Revision History:
 */

`ifndef TEST_INT32MULT_V
`define TEST_INT32MULT_V

`include "define.v"

module test_int32mult (
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    input wire is_signed,
    output wire [63:0] product
);

wire signed [63:0] s_product = $signed(operand_a) * $signed(operand_b);
wire [63:0] u_product = operand_a * operand_b;

assign product = is_signed ? s_product : u_product;

endmodule

`endif //TEST_INT32MULT_V
