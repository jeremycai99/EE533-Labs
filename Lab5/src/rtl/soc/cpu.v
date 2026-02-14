/* file: cpu.v
 Description: CPU module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 14, 2026
 Version: 1.3
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
    input wire [`INSTR_WIDTH-1:0] i_mem_data_i, // Instruction memory data input
    output wire [`PC_WIDTH-1:0] i_mem_addr_o,   // Instruction memory address output

    // Data memory interface
    input wire [`DATA_WIDTH-1:0] d_mem_data_i,  // Data memory data input (64-bit)
    output wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o, // Data memory address output
    output wire [`DATA_WIDTH-1:0] d_mem_data_o, // Data memory data output (64-bit)
    output wire d_mem_wen_o,                    // Data memory write enable
    output wire cpu_done,                       // Signal to indicate CPU completion

    // ========================================================================
    // ILA Debug Interface
    // ========================================================================
    // Multiplexed Debug Port (Full 64-bit width)
    // [4] = 0: System Debug (Selects via [3:0])
    // [4] = 1: Register File Debug (Address via [2:0])
    input wire [4:0] ila_debug_sel,
    output reg [`DATA_WIDTH-1:0] ila_debug_data
);

// CPU Done signal: Active when PC reaches max value (all 1s)
assign cpu_done = (pc_if == {`PC_WIDTH{1'b1}});

// CPU internal signal definitions

// IF stage signals
wire [`PC_WIDTH-1:0] pc_if;
wire pc_en;

assign pc_en = 1'b1;

// Instruction memory interface signals
wire [`INSTR_WIDTH-1:0] instr_if;

// ID stage signals
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

// EX stage signals
wire reg_write_ex;
wire mem_write_ex;
wire [`REG_DATA_WIDTH-1:0] r0_out_ex;
wire [`REG_DATA_WIDTH-1:0] r1_out_ex;
wire [`REG_ADDR_WIDTH-1:0] rdaddr_ex;

// MEM stage signals
wire reg_write_mem;
wire mem_write_mem;
wire [`REG_DATA_WIDTH-1:0] r0_out_mem;
wire [`REG_DATA_WIDTH-1:0] r1_out_mem;
wire [`REG_ADDR_WIDTH-1:0] rdaddr_mem;

// WB stage signals
wire reg_write_wb;
wire [`REG_ADDR_WIDTH-1:0] rdaddr_wb;

// Debug signals
wire [`REG_DATA_WIDTH-1:0] debug_reg_out;


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
// Width: Instruction Width (32)
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

// Decoding based on REG_ADDR_WIDTH = 3
assign r0addr_id = instr_id[26:24]; // Source Address (e.g., for Load addr / Store addr)
assign r1addr_id = instr_id[29:27]; // Data Source (e.g., for Store data)
assign rdaddr_id = instr_id[23:21]; // Destination Register

regfile u_regfile (
    .clk(clk),
    .r0addr(r0addr_id),
    .r1addr(r1addr_id),
    .waddr(reg_write_addr),
    .wdata(reg_write_data),
    .wena(reg_write_en),
    .r0data(r0_out_id),
    .r1data(r1_out_id),
    // Map lower bits of debug selector to register address
    .ila_cpu_reg_addr(ila_debug_sel[`REG_ADDR_WIDTH-1:0]), 
    .ila_cpu_reg_data(debug_reg_out) 
);

// ID/EX pipeline register
// Width: 1(RegW) + 1(MemW) + 64(Data0) + 64(Data1) + 3(RdAddr)
ppl_reg #(.NUM_REG(1+1+`REG_DATA_WIDTH+`REG_DATA_WIDTH+`REG_ADDR_WIDTH)) id_ex_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1),
    .D({reg_write_id, mem_write_id, r0_out_id, r1_out_id, rdaddr_id}),
    .Q({reg_write_ex, mem_write_ex, r0_out_ex, r1_out_ex, rdaddr_ex})
);

/* EX STAGE */
// EX/MEM pipeline register
// Width: 1(RegW) + 1(MemW) + 64(ALU/Data0) + 64(Data1) + 3(RdAddr)
ppl_reg #(.NUM_REG(1+1+`REG_DATA_WIDTH+`REG_DATA_WIDTH+`REG_ADDR_WIDTH)) ex_mem_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1),
    .D({reg_write_ex, mem_write_ex, r0_out_ex, r1_out_ex, rdaddr_ex}),
    .Q({reg_write_mem, mem_write_mem, r0_out_mem, r1_out_mem, rdaddr_mem})
);

/* MEM STAGE */
// Connect data memory interface signals
// Note: r0_out_mem is 64-bit, but d_mem_addr_o is 8-bit. Truncation is implicit or explicit.
assign d_mem_addr_o = r0_out_mem[`DMEM_ADDR_WIDTH-1:0]; 
assign d_mem_data_o = r1_out_mem; // Data to write
assign d_mem_wen_o = mem_write_mem;

// MEM/WB pipeline register
// Width: 1(RegW) + 3(RdAddr)
ppl_reg #(.NUM_REG(1+`REG_ADDR_WIDTH)) mem_wb_reg (
    .clk(clk),
    .rst_n(rst_n),
    .en(1'b1),
    .D({reg_write_mem, rdaddr_mem}), 
    .Q({reg_write_wb, rdaddr_wb})
);

/* WB STAGE */
assign reg_write_data = d_mem_data_i; // Data from memory load
assign reg_write_addr = rdaddr_wb;
assign reg_write_en = reg_write_wb;

// Newly added ILA Debug Interface logic
// Now using full `DATA_WIDTH` (64 bits) for output.
// Signals smaller than 64 bits are zero-padded.
always @(*) begin
    if (ila_debug_sel[4]) begin
        // Mode 1: Register File Debug (MSB = 1)
        // Address is taken from ila_debug_sel[2:0] which is already wired to regfile
        ila_debug_data = debug_reg_out;
    end else begin
        // Mode 0: System Debug (MSB = 0)
        case (ila_debug_sel[3:0])
            // 0: Program Counter (Fetch) - 9 bits
            4'd0: ila_debug_data = { {`DATA_WIDTH-`PC_WIDTH{1'b0}}, pc_if };
            
            // 1: Instruction (Decode) - 32 bits
            4'd1: ila_debug_data = { {`DATA_WIDTH-`INSTR_WIDTH{1'b0}}, instr_id };
            
            // 2: Register Read Data A (Decode) - Full 64 bits
            4'd2: ila_debug_data = r0_out_id;
            
            // 3: Register Read Data B (Decode) - Full 64 bits
            4'd3: ila_debug_data = r1_out_id;
            
            // 4: EX Stage Result / Address (Execute) - Full 64 bits
            4'd4: ila_debug_data = r0_out_ex;
            
            // 5: EX Stage Write Data (Execute) - Full 64 bits
            4'd5: ila_debug_data = r1_out_ex;
            
            // 6: Writeback Data / Memory Read Data (Writeback) - Full 64 bits
            4'd6: ila_debug_data = d_mem_data_i;
            
            // 7: Control Signals Vector
            // [0]: Reg Write ID, [1]: Mem Write ID, [2]: Mem Write Mem, [3]: Reg Write WB
            4'd7: ila_debug_data = { {`DATA_WIDTH-4{1'b0}}, reg_write_wb, mem_write_mem, mem_write_id, reg_write_id };

            // 8: Destination Register Address (Decode) - 3 bits
            4'd8: ila_debug_data = { {`DATA_WIDTH-`REG_ADDR_WIDTH{1'b0}}, rdaddr_id };

            // 9: Destination Register Address (Writeback) - 3 bits
            4'd9: ila_debug_data = { {`DATA_WIDTH-`REG_ADDR_WIDTH{1'b0}}, rdaddr_wb };

            default: ila_debug_data = {`DATA_WIDTH{1'b1}}; // All 1s for invalid selection
        endcase
    end
end

endmodule

`endif // CPU_V