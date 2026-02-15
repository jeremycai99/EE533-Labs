/* file: sync_alu.v
 Description: This file implements the integration of asynchronous ALU
 operations with a register at output to create a synchronous ALU module.
 Date: Feb. 8, 2026
 Version: 1.0
 */

`ifndef SYNC_ALU_V
`define SYNC_ALU_V

`include "define.v"
`include "alu.v"

module sync_alu (
    input wire clk, // Clock signal
    input wire rst_n, // Active low reset signal
    input wire [`DATA_WIDTH-1:0] A, // First operand
    input wire [`DATA_WIDTH-1:0] B, // Second operand
    input wire [`ALU_OP_WIDTH-1:0] aluctrl,  // ALU operation code
    output reg [`DATA_WIDTH-1:0] Z,    // Result of the ALU operation
    output reg overflow        // Overflow flag for addition and subtraction
);

wire [`DATA_WIDTH-1:0] alu_result; // Register to hold the ALU result
wire alu_overflow; // Register to hold the ALU overflow flag

alu u_alu (
    .operand_a(A),
    .operand_b(B),
    .alu_op(aluctrl),
    .result(alu_result),
    .alu_overflow(alu_overflow)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        Z <= 0; // Reset the output register
        overflow <= 0; // Reset the overflow flag
    end else begin
        Z <= alu_result; // Update the output register with the ALU result
        overflow <= alu_overflow; // Update the overflow flag
    end
end

endmodule

`endif //SYNC_ALU_V