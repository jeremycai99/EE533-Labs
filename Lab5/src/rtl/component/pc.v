/* file: pc.v
 Description: Program counter module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 10, 2026
 Version: 1.0
 */

`ifndef PC_V
`define PC_V

`include "define.v"

module pc (
    input wire clk,
    input wire rst_n,
    input wire en, // Enable signal for updating the PC
    
    output reg [`PC_WIDTH-1:0] pc_out // Current value of the program counter
);

// PC update logic: synchronous update on clock edge
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pc_out <= 0; // Reset PC to 0 on reset
    end else if (en) begin
        pc_out <= pc_out + 1; // Increment PC by 1 each cycle if enabled
    end


endmodule

`endif // PC_V