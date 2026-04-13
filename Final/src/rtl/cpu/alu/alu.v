/* file: alu.v
 Description: This file implements the main ALU module,
 which performs various arithmetic and logical operations
 based on the provided operation code.
 Author: Jeremy Cai
 Date: Feb. 17, 2026
 Revision history:
    - Feb. 9, 2026: Initial version (1.0) for Lab 5
    - Feb. 17, 2026: Updated for Lab 6 with more DP instruction support for Arm pipeline CPU design
 */

`ifndef ALU_V
`define ALU_V

`include "define.v"
`include "addsub.v"

/* ==========================================================
    *    Sub enable & carry-in computation
    *
    *    Opcode    Operation            sub    carry_in
    *    ───────   ──────────────────   ───    ────────
    *    ADD       a + b                 0       0
    *    ADC       a + b + C             0       C
    *    CMN       a + b   (flags only)  0       0
    *    SUB       a − b                 1       1
    *    SBC       a − b − !C            1       C
    *    CMP       a − b   (flags only)  1       1
    *    RSB       b − a                 1       1      (operands swapped)
    *    RSC       b − a − !C            1       C      (operands swapped)
    * ========================================================== */

module alu (
    input wire [`DATA_WIDTH-1:0] operand_a, // First operand
    input wire [`DATA_WIDTH-1:0] operand_b, // Second operand
    input wire [`ALU_OP_WIDTH-1:0] alu_op,  // ALU data processing operation code
    input wire cin,                        // Carry input for addition and subtraction
    input wire shift_carry_out,              // Carry output from barrel shifter (for shift operations that affect carry)
    output reg [`DATA_WIDTH-1:0] result,    // Result of the ALU operation
    output wire [3:0] alu_flags             // Flags for ALU operations: {N, Z, C, V}
);

wire reverse = (alu_op == `ALU_OP_RSB) | (alu_op == `ALU_OP_RSC); // Control signal to indicate if we are performing reverse subtraction

wire [`DATA_WIDTH-1:0] addsub_operand_b = reverse ? operand_a : operand_b; // If reverse is true, we swap the operands for subtraction
wire [`DATA_WIDTH-1:0] addsub_operand_a = reverse ? operand_b : operand_a; // If reverse is true, we swap the operands for subtraction
wire [`DATA_WIDTH-1:0] addsub_result; // Register to hold the result from add/sub module
wire addsub_overflow, addsub_carry_out; // Overflow and carry out flags from add/sub module

wire sub_en = (alu_op == `ALU_OP_SUB) | (alu_op == `ALU_OP_SBC) 
            | (alu_op == `ALU_OP_RSB) | (alu_op == `ALU_OP_RSC)
            | (alu_op == `ALU_OP_CMP); // Control signal to indicate if we are performing subtraction (including compare which is essentially a subtraction for flag setting)

reg addsub_cin; // Carry input for the add/sub module, used for SBC and RSC operations

// Combinational block to determine the carry input for the add/sub module based on the operation
always @(*) begin
    case (alu_op)
        `ALU_OP_ADD, `ALU_OP_CMN: addsub_cin = 1'b0; // For addition and CMN, carry input is 0
        `ALU_OP_SUB, `ALU_OP_CMP, `ALU_OP_RSB: addsub_cin = 1'b1; // For subtraction, CMP, and RSB, carry input is 1
        `ALU_OP_SBC, `ALU_OP_RSC: addsub_cin = cin; // For SBC and RSC, carry input is the carry flag from the previous operation
        `ALU_OP_ADC: addsub_cin = cin; // For ADC, carry input is the carry flag from the previous operation
        default: addsub_cin = 1'b0; // Default case
    endcase
end

addsub u_addsub (
    .operand_a(addsub_operand_a),
    .operand_b(addsub_operand_b),
    .sub(sub_en), // Control signal for subtraction
    .carry_in(addsub_cin), // Carry input is determined by the operation
    .result(addsub_result), // Output result
    .overflow(addsub_overflow), // Overflow flag (V)
    .carry_out(addsub_carry_out) // Carry out for addition (C)
);

wire arith_ops = (alu_op == `ALU_OP_ADD) | (alu_op == `ALU_OP_ADC) | (alu_op == `ALU_OP_SUB) 
                | (alu_op == `ALU_OP_SBC) | (alu_op == `ALU_OP_RSB) | (alu_op == `ALU_OP_RSC)
                | (alu_op == `ALU_OP_CMP) | (alu_op == `ALU_OP_CMN); // Control signal to indicate if the operation is an arithmetic operation

always @(*) begin
    case (alu_op)
        //Arithmetic operations: result is from the add/sub module
        `ALU_OP_ADD, `ALU_OP_CMN: result = addsub_result; // For ADD and CMN, the result is the output from the add/sub module
        `ALU_OP_SUB, `ALU_OP_CMP, `ALU_OP_RSB: result = addsub_result; // For SUB, CMP, and RSB, the result is the output from the add/sub module
        `ALU_OP_SBC, `ALU_OP_RSC: result = addsub_result; // For SBC and RSC, the result is the output from the add/sub module
        `ALU_OP_ADC: result = addsub_result; // For ADC, the result is the output from the add/sub module
        // Logical operations: result is computed directly from the operands
        `ALU_OP_AND, `ALU_OP_TST: result = operand_a & operand_b; // For AND and TST, the result is the bitwise AND of the operands
        `ALU_OP_EOR, `ALU_OP_TEQ: result = operand_a ^ operand_b; // For EOR and TEQ, the result is the bitwise XOR of the operands
        `ALU_OP_ORR: result = operand_a | operand_b; // For ORR, the result is the bitwise OR of the operands
        `ALU_OP_BIC: result = operand_a & ~operand_b; // For BIC, the result is the bitwise AND of operand_a and the bitwise NOT of operand_b
        // Move operations: result is either operand_b or its bitwise NOT
        `ALU_OP_MOV: result = operand_b; // For MOV, the result is simply operand_b
        `ALU_OP_MVN: result = ~operand_b; // For MVN, the result is the bitwise NOT of operand_b
        default: result = {`DATA_WIDTH{1'b0}}; // Default case to handle undefined operation codes
    endcase
end

wire flag_n = result[`DATA_WIDTH-1]; // N flag is the most significant bit of the result (indicates negative in two's complement)
wire flag_z = (result == 0); // Z flag is set if the result is zero
wire flag_c = arith_ops ? addsub_carry_out : shift_carry_out; // C flag is determined by the type of operation: for arithmetic operations, it is the carry out from the add/sub module; for shift operations, it is the carry out from the barrel shifter
wire flag_v = arith_ops ? addsub_overflow : 1'b0; // V flag is only relevant for arithmetic operations, for logical and shift operations it is set to 0

assign alu_flags = {flag_n, // N flag
                    flag_z, // Z flag
                    flag_c, // C flag
                    flag_v}; // V flag

endmodule

`endif //ALU_V
