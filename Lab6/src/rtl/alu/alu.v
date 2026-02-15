/* file: alu.v
 Description: This file implements the main ALU module,
 which performs various arithmetic and logical operations
 based on the provided operation code.
 Author: Jeremy Cai
 Date: Feb. 9, 2026
 Version: 1.0 (for Lab 5 only)
 */

`ifndef ALU_V
`define ALU_V

`include "define.v"
`include "addsub.v"

module alu (
    input wire [`DATA_WIDTH-1:0] operand_a, // First operand
    input wire [`DATA_WIDTH-1:0] operand_b, // Second operand
    input wire [`ALU_OP_WIDTH-1:0] alu_op,  // ALU data processing operation code
    output reg [`DATA_WIDTH-1:0] result,    // Result of the ALU operation
    output wire alu_overflow             // Overflow flag for addition and subtraction
);

wire [`DATA_WIDTH-1:0] addsub_result; // Register to hold the result from add/sub module
wire addsub_overflow, addsub_carry_out; // Overflow and carry out flags from add/sub module

wire sub_en = (alu_op == `ALU_OP_SUB);

addsub u_addsub (
    .operand_a(operand_a),
    .operand_b(operand_b),
    .sub(sub_en), // Control signal for subtraction
    .result(addsub_result), // Output result
    .overflow(addsub_overflow), // Overflow flag (V)
    .carry_out(addsub_carry_out) // Carry out for addition (C)
);

always @(*) begin
    case (alu_op)
        `ALU_OP_ADD: result = addsub_result; // ADD
        `ALU_OP_SUB: result = addsub_result; // SUB
        `ALU_OP_AND: result = operand_a & operand_b; // AND
        `ALU_OP_OR:  result = operand_a | operand_b; // OR
        `ALU_OP_XNOR: result = ~(operand_a ^ operand_b); // XNOR
        `ALU_OP_CMP: result = (operand_a == operand_b); // CMP
        `ALU_OP_LSL: result = operand_a << operand_b[5:0]; // LSL (using lower 6 bits for shift amount)
        `ALU_OP_LSR: result = operand_a >> operand_b[5:0]; // LSR (using lower 6 bits for shift amount)

        // Not standard ALU operation and not used by Arm ISA
        `ALU_OP_SBCMP: result = (operand_a[31:0] == operand_b[31:0]) ? 1 : 0; // Substring Compare (lower 32 bits).
        // `ALU_OP_LSTC: result = (operand_a << operand_b[5:0]) == operand_b ? 1 : 0; // LSTC Left Shift and Compare
        // `ALU_OP_RSTC: result = (operand_a >> operand_b[5:0]) == operand_b ? 1 : 0; // RSTC Right Shift and Compare
        default: result = 0; // Default case to handle undefined operation codes
    endcase
end

assign alu_overflow = ((alu_op == `ALU_OP_ADD || alu_op == `ALU_OP_SUB) && addsub_overflow); // Output the overflow flag from the add/sub module

endmodule

`endif //ALU_V
