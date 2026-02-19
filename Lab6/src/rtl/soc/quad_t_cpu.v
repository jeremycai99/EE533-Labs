/* file: quad_t_cpu.v
 Description: Quad-threaded 5-stage pipeline Arm CPU module
 Author: Jeremy Cai
 Date: Feb. 18, 2026
 Version: 1.0
 */

`ifndef QUAD_T_CPU_V
`define QUAD_T_CPU_V

`include "define.v"

module quad_t_cpu (
    input wire clk,
    input wire rst_n,
    // Instruction memory interface
    input wire [`INSTR_WIDTH-1:0] i_mem_data_i, // Instruction memory data input
    output wire [`PC_WIDTH-1:0] i_mem_addr_o,   // Instruction memory address output

    // Data memory interface
    input wire [`DATA_WIDTH-1:0] d_mem_data_i,  // Data memory data input (64-bit)
    output wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o, // Data memory address output
    output wire [`DATA_WIDTH-1:0] d_mem_data_o, // Data memory data output (64-bit)
    output wire d_mem_wen_o                    // Data memory write enable
);


endmodule

`endif // QUAD_T_CPU_V

