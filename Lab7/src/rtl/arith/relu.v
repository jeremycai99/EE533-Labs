/* file: relu.v
 Description: This file implements the ReLU operation for BF16 format.
 Author: Jeremy Cai
 Date: Feb. 24, 2026
 Version: 1.0
 Revision history:
    - Feb. 24, 2026: Initial implementation of BF16 ReLU.
 */

`ifndef RELU_V
`define RELU_V

`include "gpu_define.v"

module relu (
    input wire [15:0] operand, // Input operand in BF16 format
    output reg [15:0] result // Result of the ReLU operation in BF16 format
);

always @(*) begin
    // Check the sign bit (bit 15) of the operand
    if (operand[15] == 1'b0) begin
        // If the sign bit is 0, the number is non-negative, so output it directly
        result = operand;
    end else begin
        // If the sign bit is 1, the number is negative, so output zero
        result = 16'b0; // Zero in BF16 format
    end
end

endmodule

`endif // RELU_V