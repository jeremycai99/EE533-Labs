/* file: cpu.v
 Description: CPU module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 10, 2026
 Version: 1.0 (for Lab 5 only)
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
    output wire d_mem_wen_o // Data memory write enable output (to d_mem)
);

// CPU internal signal definitions

//IF stage signals
wire [`PC_WIDTH-1:0] pc_if;
wire pc_en;

assign pc_en = 1'b1; // For now, always enable PC update (no stalls or branches implemented yet)

//Instruction memory interface signals
wire [`INSTR_WIDTH-1:0] instr_if; // Instruction fetched from instruction memory

//ID stage signals
wire [`INSTR_WIDTH-1:0] instr_id; // Instruction in the ID stage (output of IF/ID register)
wire reg_write_id; // Register write enable signal generated in ID stage
wire mem_write_id; // Memory write enable signal generated in ID stage

wire [`REG_ADDR_WIDTH-1:0] rdaddr_id; // Destination register address in ID stage

// Register file interface signals
wire [`REG_ADDR_WIDTH-1:0] r0addr_id; // Source register
wire [`REG_ADDR_WIDTH-1:0] r1addr_id; // Source register
wire [`REG_ADDR_WIDTH-1:0] reg_write_addr; // Destination register
wire [`REG_DATA_WIDTH-1:0] reg_write_data; // Data to write to register
wire reg_write_en; // Register write enable signal for register file generated in ID stage

wire [`REG_DATA_WIDTH-1:0] r0_out_id; // Data read from r0
wire [`REG_DATA_WIDTH-1:0] r1_out_id; // Data read from r1

//No decoding logic or ALU control logic for now.

//EX stage signals
wire reg_write_ex; // Register write enable signal generated in EX stage
wire mem_write_ex; // Memory write enable signal generated in EX stage
wire [`REG_DATA_WIDTH-1:0] r0_out_ex; // Data read from r0 in EX stage
wire [`REG_DATA_WIDTH-1:0] r1_out_ex; // Data read from r1 in EX stage
wire [`REG_ADDR_WIDTH-1:0] rdaddr_ex; // Destination register address in EX stage

//MEM stage signals
wire reg_write_mem; // Register write enable signal generated in MEM stage
wire mem_write_mem; // Memory write enable signal generated in MEM stage
wire [`REG_DATA_WIDTH-1:0] r0_out_mem; // Data read from r0 in MEM stage
wire [`REG_DATA_WIDTH-1:0] r1_out_mem; // ALU result in MEM stage
wire [`REG_ADDR_WIDTH-1:0] rdaddr_mem; // Destination register address in MEM stage

wire [`DATA_WIDTH-1:0] d_mem_data_mem; // Data read from memory in MEM stage

//Data memory interface signals
//Explicitly defined in the interface section for clarity

//WB stage signals
wire reg_write_wb; // Register write enable signal generated in WB stage
wire [`DATA_WIDTH-1:0] d_mem_data_wb; // Data passed from memory in WB stage
wire [`REG_ADDR_WIDTH-1:0] rdaddr_wb; // Destination register address in WB stage


// Start core pipeline logic implementation

/* IF STAGE */
// Instantiate program counter module
pc u_pc (
    .clk(clk),
    .rst_n(rst_n),
    .en(pc_en),
    .pc_out(pc_if)
);

// Connect PC output to instruction memory address output
assign i_mem_addr_o = pc_if;

assign instr_if = i_mem_data_i; // Connect instruction memory data input to IF stage instruction

/* IF/ID pipeline register */
ppl_reg #(.NUM_REG(`INSTR_WIDTH)) if_id_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1), // Always enable IF/ID register update for now
    .D(instr_if), // Input is the instruction fetched from instruction memory
    .Q(instr_id) // Output is the instruction for the ID stage
);

/* ID STAGE */
// For now, we will just pass the instruction through the pipeline without decoding or executing it.

//minimal control signal generation logic for memmory pass testing logic to connect instruction to regfile and data memory interfaces
assign mem_write_id = instr_id[31]; // Use the most significant bit of the instruction as a dummy memory write enable signal for testing
assign reg_write_id = instr_id[30]; // Use the second most significant bit of the instruction as a dummy register write enable signal for testing
assign r0addr_id = instr_id[29: 29 - `REG_ADDR_WIDTH + 1]; // Use bits [29:29-`REG_ADDR_WIDTH+1] as the source register address for testing
assign r1addr_id = instr_id[29 - `REG_ADDR_WIDTH: 29 - 2*`REG_ADDR_WIDTH + 1]; // Use bits [29-`REG_ADDR_WIDTH:29-2*`REG_ADDR_WIDTH+1] as the second source register address for testing
assign rdaddr_id = instr_id[29 - 2*`REG_ADDR_WIDTH: 29 - 2*`REG_ADDR_WIDTH - `REG_ADDR_WIDTH + 1]; // Use bits [29-2*`REG_ADDR_WIDTH:29-2*`REG_ADDR_WIDTH-`REG_ADDR_WIDTH+1] as the destination register address for testing

// Register file interface signals
regfile u_regfile (
    .clk(clk),
    .r0addr(r0addr_id),
    .r1addr(r1addr_id),
    .waddr(reg_write_addr),
    .wdata(reg_write_data),
    .wena(reg_write_en),
    .r0data(r0_out_id),
    .r1data(r1_out_id)
);

//ID/EX pipeline register to pass control and data signals from ID stage to EX stage
ppl_reg #(.NUM_REG(1+1+`REG_DATA_WIDTH+`REG_DATA_WIDTH+`REG_ADDR_WIDTH)) id_ex_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1), // Always enable ID/EX register update for now
    .D({reg_write_id, mem_write_id, r0_out_id, r1_out_id, rdaddr_id}), // Input is the concatenated control and data signals from ID stage
    .Q({reg_write_ex, mem_write_ex, r0_out_ex, r1_out_ex, rdaddr_ex}) // Output is the separated control and data signals for EX stage
);

/* EX STAGE */
// No ALU or execution logic implemented yet, just passing signals through
ppl_reg #(.NUM_REG(1+1+`REG_DATA_WIDTH+`REG_DATA_WIDTH+`REG_ADDR_WIDTH)) ex_mem_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1), // Always enable EX/MEM register update for now
    .D({reg_write_ex, mem_write_ex, r0_out_ex, r1_out_ex, rdaddr_ex}), // Input is the concatenated control and data signals from EX stage
    .Q({reg_write_mem, mem_write_mem, r0_out_mem, r1_out_mem, rdaddr_mem}) // Output is the separated control and data signals for MEM stage
);

/* MEM STAGE */
// Connect data memory interface signals
assign d_mem_addr_o = r0_out_mem; // For now, just use r0_out_mem as the memory address output
assign d_mem_data_o = r1_out_mem; // For now, just use r1_out_mem as the data output to memory
assign d_mem_wen_o = mem_write_mem; // Connect memory write enable signal from MEM stage
assign d_mem_data_mem = d_mem_data_i; // Connect data memory data input to MEM stage data input

ppl_reg #(.NUM_REG(1+`DATA_WIDTH+`REG_ADDR_WIDTH)) mem_wb_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1), // Always enable MEM/WB register update for now
    .D({reg_write_mem, d_mem_data_mem, rdaddr_mem}), // Input is the concatenated control and data signals from MEM stage
    .Q({reg_write_wb, d_mem_data_wb, rdaddr_wb}) // Output is the separated control and data signals for WB stage
);

/* WB STAGE */
// Connect register file write interface signals
assign reg_write_data = d_mem_data_wb; // For now, just use the data passed from memory as the data to write to the register file
assign reg_write_addr = rdaddr_wb; // Connect destination register address from WB stage to ID stage
assign reg_write_en = reg_write_wb; // Connect register write enable signal from WB stage to ID stage


endmodule

`endif // CPU_V