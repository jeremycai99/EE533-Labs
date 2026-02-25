/* file: bf16addsub.v
 Description: This file implements the addition and subtraction operations for BF16 format.
 BF16 format: [15] sign, [14:7] exponent (bias=127), [6:0] fraction
 Author: Jeremy Cai
 Date: Feb. 24, 2026
 Version: 1.0
 Revision history:
    - Feb. 24, 2026: Initial implementation of BF16 addition and subtraction.
 */

`ifndef BF16ADDSUB_V
`define BF16ADDSUB_V

`include "gpu_define.v"

module bf16addsub (
    input wire [15:0] operand_a, // First operand in BF16 format
    input wire [15:0] operand_b, // Second operand in BF16 format
    input wire sub, // Control signal: 0 for addition, 1 for subtraction
    output reg [15:0] result // Result of the addition or subtraction in BF16 format
);

localparam EXP_BIAS = 127; // Exponent bias for BF16


// Sign bits of the operands
wire sign_a = operand_a[15];
wire sign_b_raw = operand_b[15]; // Raw sign of operand_b before considering subtraction
wire sign_b = sign_b_raw ^ sub; // Effective sign of operand_b after considering subtraction

// Exponents of the operands
wire [7:0] exp_a = operand_a[14:7];
wire [7:0] exp_b = operand_b[14:7];

// Fractions of the operands (with implicit leading 1 for normalized numbers)
wire [6:0] frac_a = operand_a[6:0];
wire [6:0] frac_b = operand_b[6:0];

// Mantissa alignment and operation
wire [7:0] mant_a = (exp_a != 8'h00) ? {1'b1, frac_a} : {1'b0, frac_a};
wire [7:0] mant_b = (exp_b != 8'h00) ? {1'b1, frac_b} : {1'b0, frac_b};

wire eff_sub = (sign_a != sign_b); // Effective subtraction if signs differ

wire a_lt_b = (exp_a < exp_b) || (exp_a == exp_b && mant_a < mant_b);

wire sign_lg = a_lt_b ? sign_b : sign_a; // Sign of the larger operand
wire [7:0] exp_lg = a_lt_b ? exp_b : exp_a; // Exponent of the larger operand
wire [7:0] exp_sm = a_lt_b ? exp_a : exp_b; // Exponent of the smaller operand
wire [7:0] mant_lg = a_lt_b ? mant_b : mant_a; // Mantissa of the larger operand
wire [7:0] mant_sm = a_lt_b ? mant_a : mant_b; // Mantissa of the smaller operand

wire [7:0] exp_diff = exp_lg - exp_sm; // Exponent difference for alignment







endmodule






`endif // BF16ADDSUB_V