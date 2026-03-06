/* file: regfile.v
 Description: Register file module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Mar. 6, 2026
 Version: 1.3
 Revision history:
    - 1.0: Initial version with basic functionality for Lab 5 only (Feb. 9, 2026)
    - 1.1: Updated version with register forwarding support (Feb. 17, 2026)
    - 1.2: Updated version with support for second write port for multiply instructions (Feb. 18, 2026)
    - 1.3: Updated version with 4 read ports and forwarding removed for Arm pipeline design (Mar. 6, 2026)
 */

`ifndef REGFILE_V
`define REGFILE_V

`include "define.v"

module regfile(
    input clk,
    input [`REG_ADDR_WIDTH-1:0] r1addr,
    input [`REG_ADDR_WIDTH-1:0] r2addr,
    input [`REG_ADDR_WIDTH-1:0] r3addr,
    input [`REG_ADDR_WIDTH-1:0] r4addr,

    input wena,
    input [`REG_ADDR_WIDTH-1:0] wr_addr1,
    input [`REG_DATA_WIDTH-1:0] wr_data1,
    input [`REG_ADDR_WIDTH-1:0] wr_addr2,
    input [`REG_DATA_WIDTH-1:0] wr_data2,

    output wire [`REG_DATA_WIDTH-1:0] r1data,
    output wire [`REG_DATA_WIDTH-1:0] r2data,
    output wire [`REG_DATA_WIDTH-1:0] r3data,
    output wire [`REG_DATA_WIDTH-1:0] r4data
);

(* ram_style = "distributed" *) reg [`REG_DATA_WIDTH-1:0] regs [0:(1<<`REG_ADDR_WIDTH)-1];

// Combinational read — no forwarding (v1.3).
// EX1 registered bypass in cpu_mt handles WB→EX1 same-thread forwarding.
assign r1data = regs[r1addr];
assign r2data = regs[r2addr];
assign r3data = regs[r3addr];
assign r4data = regs[r4addr];

// Synchronous write — dual port for MLAL (RdHi + RdLo).
// No reset on storage: contents initialized by software.
always @(posedge clk) begin
    if (wena) begin
        regs[wr_addr1] <= wr_data1;
        regs[wr_addr2] <= wr_data2;
    end
end

endmodule

`endif //REGFILE_V