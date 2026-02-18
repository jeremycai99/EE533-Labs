/* file: ppl_reg.v
 Description: Configurable register moduld for pipeline registers
 Author: Jeremy Cai
 Date: Feb. 17, 2026
 Version: 1.1
 Revision history:
 - v1.0: Initial version with basic functionality for Lab 5 only (Feb. 9, 2026)
 - v1.1: Updated version with more features for Lab 6 with flush support (Feb. 17, 2026)
 */

`ifndef PPL_REG_V
`define PPL_REG_V

`include "define.v"

module ppl_reg #(parameter NUM_REG = 32)(
    input wire clk, // Clock signal
    input wire rst_n, // Active low reset signal
    input wire en, // Enable signal for updating the register
    input wire flush, // Flush signal to clear the register (e.g., on branch misprediction)
    input wire [NUM_REG-1:0] D, // Data input
    output reg [NUM_REG-1:0] Q // Data output
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        Q <= {`N{NUM_REG{1'b0}}}; // Reset the register output to 0
    end else if (flush) begin
        Q <= {`N{NUM_REG{1'b0}}}; // Clear the register output on flush
    end else if (en) begin
        Q <= D; // Update the register output with the input data
    end
end

endmodule

`endif //PPL_REG_V