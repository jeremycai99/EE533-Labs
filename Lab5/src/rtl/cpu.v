/* file: cpu.v
 Description: CPU module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 10, 2026
 Version: 1.2
 */

`ifndef CPU_V
`define CPU_V

`include "define.v"
`include "pc.v"
`include "ppl_reg.v"
`include "regfile.v"

module cpu (
    input wire clk,
    input wire rst_n,
    // Instruction memory interface
    input wire [`INSTR_WIDTH-1:0] i_mem_data_i, // Instruction memory data input (from i_mem)
    output wire [`PC_WIDTH-1:0] i_mem_addr_o, // Instruction memory address output (to i_mem)

    // Data memory interface
    input wire [`DATA_WIDTH-1:0] d_mem_data_i, // Data memory data input (from d_mem)
    output wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o, // Data memory address output (to d_mem)
    output wire [`DATA_WIDTH-1:0] d_mem_data_o, // Data memory data output (to d_mem)
    output wire d_mem_wen_o, // Data memory write enable output (to d_mem)
    output wire cpu_done, // Signal to indicate CPU has completed execution (for testbench control)

    // ILA probe signals for debugging
    input wire [`REG_ADDR_WIDTH-1:0] ila_cpu_reg_addr, // ILA probe address input
    output wire [`REG_DATA_WIDTH-1:0] ila_cpu_reg_data // ILA probe data output
);

assign cpu_done = (pc_if == {`PC_WIDTH{1'b1}});

// CPU internal signal definitions

//IF stage signals
wire [`PC_WIDTH-1:0] pc_if;
wire pc_en;

assign pc_en = 1'b1;

//Instruction memory interface signals
wire [`INSTR_WIDTH-1:0] instr_if;

//ID stage signals
wire [`INSTR_WIDTH-1:0] instr_id;
wire reg_write_id;
wire mem_write_id;

wire [`REG_ADDR_WIDTH-1:0] rdaddr_id;

// Register file interface signals
wire [`REG_ADDR_WIDTH-1:0] r0addr_id;
wire [`REG_ADDR_WIDTH-1:0] r1addr_id;
wire [`REG_ADDR_WIDTH-1:0] reg_write_addr;
wire [`REG_DATA_WIDTH-1:0] reg_write_data;
wire reg_write_en;

wire [`REG_DATA_WIDTH-1:0] r0_out_id;
wire [`REG_DATA_WIDTH-1:0] r1_out_id;

//EX stage signals
wire reg_write_ex;
wire mem_write_ex;
wire [`REG_DATA_WIDTH-1:0] r0_out_ex;
wire [`REG_DATA_WIDTH-1:0] r1_out_ex;
wire [`REG_ADDR_WIDTH-1:0] rdaddr_ex;

//MEM stage signals
wire reg_write_mem;
wire mem_write_mem;
wire [`REG_DATA_WIDTH-1:0] r0_out_mem;
wire [`REG_DATA_WIDTH-1:0] r1_out_mem;
wire [`REG_ADDR_WIDTH-1:0] rdaddr_mem;

// WB stage signals
wire reg_write_wb;
wire [`REG_ADDR_WIDTH-1:0] rdaddr_wb;


// Start core pipeline logic implementation

/* IF STAGE */
pc u_pc (
    .clk(clk),
    .rst_n(rst_n),
    .en(pc_en),
    .pc_out(pc_if)
);

assign i_mem_addr_o = pc_if;
assign instr_if = i_mem_data_i;

/* IF/ID pipeline register */
// create an extra pipeline delay because of the sync-read from instruction memory
// Will dive deep in next lab with full core design.
ppl_reg #(.NUM_REG(`INSTR_WIDTH)) if_id_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1),
    .D(instr_if),
    .Q(instr_id)
);

/* ID STAGE */
// Control Signals
assign mem_write_id = instr_id[31];
assign reg_write_id = instr_id[30];

// Based on soc_tb.v hex codes:
// Load  (40400000): Dest=R2 (Bits 23:21=010), SrcAddr=R0 (Bits 27:24=0000)
// Store (93000000): SrcAddr=R3 (Bits 27:24=0011), SrcData=R2 (Bits 29:27=010)

assign r0addr_id = {1'b0, instr_id[27:24]}; // Address Source (R0 for Load, R3 for Store)
assign r1addr_id = {2'b0, instr_id[29:27]}; // Data Source (R2 for Store)
assign rdaddr_id = {2'b0, instr_id[23:21]}; // Destination (R2/R3 for Load)

regfile u_regfile (
    .clk(clk),
    .r0addr(r0addr_id),
    .r1addr(r1addr_id),
    .waddr(reg_write_addr),
    .wdata(reg_write_data),
    .wena(reg_write_en),
    .r0data(r0_out_id),
    .r1data(r1_out_id),
    .ila_cpu_reg_addr(ila_cpu_reg_addr), // Connect ILA probe address input
    .ila_cpu_reg_data(ila_cpu_reg_data) // Connect ILA probe data output
);

//ID/EX pipeline register
ppl_reg #(.NUM_REG(1+1+`REG_DATA_WIDTH+`REG_DATA_WIDTH+`REG_ADDR_WIDTH)) id_ex_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1),
    .D({reg_write_id, mem_write_id, r0_out_id, r1_out_id, rdaddr_id}),
    .Q({reg_write_ex, mem_write_ex, r0_out_ex, r1_out_ex, rdaddr_ex})
);

/* EX STAGE */
ppl_reg #(.NUM_REG(1+1+`REG_DATA_WIDTH+`REG_DATA_WIDTH+`REG_ADDR_WIDTH)) ex_mem_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1),
    .D({reg_write_ex, mem_write_ex, r0_out_ex, r1_out_ex, rdaddr_ex}),
    .Q({reg_write_mem, mem_write_mem, r0_out_mem, r1_out_mem, rdaddr_mem})
);

/* MEM STAGE */
// Connect data memory interface signals
assign d_mem_addr_o = r0_out_mem; // Address from R0/R3
assign d_mem_data_o = r1_out_mem; // Data to write from R2
assign d_mem_wen_o = mem_write_mem;

// We do NOT latch d_mem_data_i here. It arrives in the next cycle (WB stage).

// MEM/WB pipeline register
ppl_reg #(.NUM_REG(1+`REG_ADDR_WIDTH)) mem_wb_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1),
    .D({reg_write_mem, rdaddr_mem}), // Pass control signals only
    .Q({reg_write_wb, rdaddr_wb})
);

/* WB STAGE */
// Connect d_mem_data_i directly to register write data.
// The data requested in MEM stage is available now.
assign reg_write_data = d_mem_data_i;

assign reg_write_addr = rdaddr_wb;
assign reg_write_en = reg_write_wb;

endmodule

`endif // CPU_V