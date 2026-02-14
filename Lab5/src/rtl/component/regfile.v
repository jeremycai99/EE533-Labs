/* file: regfile.v
 Description: Register file module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 9, 2026
 Version: 1.0
 */

`ifndef REGFILE_V
`define REGFILE_V

`include "define.v"

module regfile(
    input clk, // Clock signal. No reset signal needed
    input [`REG_ADDR_WIDTH-1:0] r0addr, // Source register 1 address
    input [`REG_ADDR_WIDTH-1:0] r1addr, // Source register 2 address

    input wena, // Write enable signal
    input [`REG_ADDR_WIDTH-1:0] waddr, // Destination register address
    input [`REG_DATA_WIDTH-1:0] wdata, // Data to write to the register

    output reg [`REG_DATA_WIDTH-1:0] r0data, // Data read from source register 1
    output reg [`REG_DATA_WIDTH-1:0] r1data, // Data read from source register 2
    
    // ILA probe signals for debugging
    input [`REG_ADDR_WIDTH-1:0] ila_cpu_reg_addr, // ILA probe address input
    output reg [`REG_DATA_WIDTH-1:0] ila_cpu_reg_data // ILA probe data output
);

reg [`REG_DATA_WIDTH-1:0] regfile [0:(1<<`REG_ADDR_WIDTH)-1]; // Register file array

// Register read logic: combinational read
always @(*) begin
    // do not need to qualify with waddr != 0 since Arm doesn't restrict writes to x0
    r0data = regfile[r0addr]; // Register x0 is hardwired to 0
    r1data = regfile[r1addr]; // Register x0 is hardwired to 0
    ila_cpu_reg_data = regfile[ila_cpu_reg_addr]; // ILA probe data output
end

// Register write logic: synchronous write on clock edge
always @(posedge clk) begin
    if (wena) begin // do not need to qualify with waddr != 0 since Arm doesn't restrict writes to x0
        regfile[waddr] <= wdata;
    end
end
endmodule

`endif //REGFILE_V