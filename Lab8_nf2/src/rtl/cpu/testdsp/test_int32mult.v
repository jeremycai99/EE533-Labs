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
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [63:0] p
);

wire signed [63:0] s_product = $signed(a) * $signed(b);

assign p = s_product;

endmodule

`endif //TEST_INT32MULT_V
