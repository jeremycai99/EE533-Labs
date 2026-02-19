/* file: regfile.v
 Description: Register file module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 18, 2026
 Version: 1.2
 Revision history:
    - 1.0: Initial version with basic functionality for Lab 5 only (Feb. 9, 2026)
    - 1.1: Updated version with register forwarding support (Feb. 17, 2026)
    - 1.2: Updated version with support for second write port for multiply instructions (Feb. 18, 2026)
 */

`ifndef REGFILE_V
`define REGFILE_V

`include "define.v"

module regfile(
    input clk, // Clock signal. No reset signal needed
    input [`REG_ADDR_WIDTH-1:0] r1addr, // Source register 1 address
    input [`REG_ADDR_WIDTH-1:0] r2addr, // Source register 2 address
    input [`REG_ADDR_WIDTH-1:0] r3addr, // Source register 3 address (for multiply instructions)

    input wena, // Write enable signal
    input [`REG_ADDR_WIDTH-1:0] wr_addr1, // Destination register address
    input [`REG_DATA_WIDTH-1:0] wr_data1, // Data to write to the register
    input [`REG_ADDR_WIDTH-1:0] wr_addr2, // Destination register address for second write port (for multiply instructions)
    input [`REG_DATA_WIDTH-1:0] wr_data2, // Data to write to the register for second write port (for multiply instructions)

    output wire [`REG_DATA_WIDTH-1:0] r1data, // Data read from source register 1
    output wire [`REG_DATA_WIDTH-1:0] r2data, // Data read from source register 2
    output wire [`REG_DATA_WIDTH-1:0] r3data, // Data read from source register 3 (for multiply instructions)
    // ILA probe signals for debugging
    input [`REG_ADDR_WIDTH-1:0] ila_cpu_reg_addr, // ILA probe address input
    output wire [`REG_DATA_WIDTH-1:0] ila_cpu_reg_data // ILA probe data output
);

reg [`REG_DATA_WIDTH-1:0] regs [0:(1<<`REG_ADDR_WIDTH)-1]; // Register file array

// Register read logic: combinational read
// do not need to qualify with waddr != 0 since Arm doesn't restrict writes to x0
// Implement register forwarding: if the current instruction is writing to a register that is being read
// forward the write data instead of reading from the register file
assign r1data = (wena && wr_addr1 == r1addr) ? wr_data1 : (wena && wr_addr2 == r1addr) ? wr_data2 : regs[r1addr]; // Register x0 is hardwired to 0
assign r2data = (wena && wr_addr1 == r2addr) ? wr_data1 : (wena && wr_addr2 == r2addr) ? wr_data2 : regs[r2addr]; // Register x0 is hardwired to 0
assign r3data = (wena && wr_addr1 == r3addr) ? wr_data1 : (wena && wr_addr2 == r3addr) ? wr_data2 : regs[r3addr]; // Register x0 is hardwired to 0

assign ila_cpu_reg_data = regs[ila_cpu_reg_addr]; // ILA probe data output

// Register write logic: synchronous write on clock edge
always @(posedge clk) begin
    if (wena) begin // do not need to qualify with waddr != 0 since Arm doesn't restrict writes to x0
        regs[wr_addr1] <= wr_data1;
        regs[wr_addr2] <= wr_data2;
    end
end
endmodule

// Register files doesn't need to be initialized since we will initialize registers through software (e.g., by writing to them in the test program)

`endif //REGFILE_V