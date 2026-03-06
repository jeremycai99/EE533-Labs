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
    - 1.2: Update the data width (64 -> 32) feeding to ksa
    - 1.3: Replace KSA with operator inference (Mar. 6, 2026).
 */

`ifndef ADDSUB_V
`define ADDSUB_V

`include "define.v"

module addsub (
    input wire [`DATA_WIDTH-1:0] operand_a,
    input wire [`DATA_WIDTH-1:0] operand_b,
    input wire sub,
    input wire carry_in,
    output wire [`DATA_WIDTH-1:0] result,
    output wire overflow,
    output wire carry_out
);

    wire [`DATA_WIDTH-1:0] operand_b_mod = sub ? ~operand_b : operand_b;

    // Infers MUXCY/XORCY carry chain — 32 LUTs + 32 MUXCY (free silicon)
    assign {carry_out, result} = {1'b0, operand_a} + {1'b0, operand_b_mod} + {{`DATA_WIDTH{1'b0}}, carry_in};

    // Overflow detection (unchanged)
    assign overflow = (sub)
        ? ((operand_a[`DATA_WIDTH-1] != operand_b[`DATA_WIDTH-1]) && (result[`DATA_WIDTH-1] != operand_a[`DATA_WIDTH-1]))
        : ((operand_a[`DATA_WIDTH-1] == operand_b[`DATA_WIDTH-1]) && (result[`DATA_WIDTH-1] != operand_a[`DATA_WIDTH-1]));

endmodule

`endif //ADDSUB_V