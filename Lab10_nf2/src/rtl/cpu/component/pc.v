/* file: pc.v
 Description: Program counter module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 17, 2026
 Version: 1.1
 Revision history:
    - 1.0: Initial version with basic functionality for Lab 5 only (Feb. 9, 2026)
    - 1.1: Updated version with pc_in for full Arm pipeline design (Feb. 17, 2026)
 */

`ifndef PC_V
`define PC_V

`include "define.v"

module pc (
    input wire clk,
    input wire rst_n,
    input wire en, // Enable signal for updating the PC
    input wire [`PC_WIDTH-1:0] pc_in, // Input value for the PC (for branch/jump instructions)
    output reg [`PC_WIDTH-1:0] pc_out // Current value of the program counter
);

// PC update logic: synchronous update on clock edge
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pc_out <= 0; // Reset PC to 0 on reset
    end else if (en) begin
        pc_out <= pc_in; // Update PC with the input value if enabled
    end
end

endmodule

`endif // PC_V