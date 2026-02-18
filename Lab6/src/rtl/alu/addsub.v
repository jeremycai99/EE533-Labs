/* file: addsub.v
 Description: This file implements the addition and subtraction operations for the ALU.
 This design support datawidth up to 64 bits and overflow detection for both addition and subtraction.
 This module handles carry in for addition and subtraction.
 Author: Jeremy Cai
 Date: Feb. 17, 2026
 Version: 1.1
 Revision history:
    - 1.0: Initial version with basic addition and subtraction using Kogge-Stone Adder (Feb. 10, 2026)
    - 1.1: Updated version with more flag detection and carry_in handling (Feb. 17, 2026)
 */

`ifndef ADDSUB_V
`define ADDSUB_V

`include "define.v"
`include "ksa.v"

module addsub (
    input wire [`DATA_WIDTH-1:0] operand_a,     // First operand
    input wire [`DATA_WIDTH-1:0] operand_b,     // Second operand
    input wire sub,                             // Subtract control signal (1 for subtraction, 0 for addition)
    input wire carry_in,                         // Carry input for addition and subtraction
    output wire [`DATA_WIDTH-1:0] result,       // Result of addition or subtraction
    output wire overflow,                       // Overflow flag for addition and subtraction
    output wire carry_out                       // Carry out for addition (not used for subtraction)
);

wire [63:0] operand_a_ext; // Extended first operand for addition/subtraction
wire [63:0] operand_b_ext; // Extended second operand for addition/subtraction

assign operand_a_ext = {{(64-`DATA_WIDTH){operand_a[`DATA_WIDTH-1]}}, operand_a};
assign operand_b_ext = {{(64-`DATA_WIDTH){operand_b[`DATA_WIDTH-1]}}, operand_b};

wire [63:0] operand_b_ext_mod; // Modified second operand for subtraction

// Modify operand_b for subtraction (two's complement)
assign operand_b_ext_mod = sub ? ~operand_b_ext : operand_b_ext;

wire [63:0] sum_ext; // Extended sum output from KSA

// Instantiate the Kogge-Stone Adder
ksa u_ksa (
    .operand_a(operand_a_ext),
    .operand_b(operand_b_ext_mod),
    .cin(carry_in), // Carry input is directly passed to KSA, which will handle it correctly for both addition and subtraction
    .sum(sum_ext),
    .cout(carry_out) //For alu C flag
);

assign result = sum_ext[`DATA_WIDTH-1:0]; // Truncate the result to the defined data width
// Overflow detection for addition and subtraction
assign overflow = (sub) ? ((operand_a[`DATA_WIDTH-1] != operand_b[`DATA_WIDTH-1]) && (result[`DATA_WIDTH-1] != operand_a[`DATA_WIDTH-1])) : // Subtraction overflow
                          ((operand_a[`DATA_WIDTH-1] == operand_b[`DATA_WIDTH-1]) && (result[`DATA_WIDTH-1] != operand_a[`DATA_WIDTH-1])); // Addition overflow
endmodule

`endif //ADDSUB_V