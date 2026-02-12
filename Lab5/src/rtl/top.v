`ifndef TOP_V
`define TOP_V

`include "define.v"

`include "i_mem.v"
`include "d_mem.v"
`include "cpu.v"

module top (
    input wire clk,
    input wire rst_n
);

wire [`PC_WIDTH-1:0] i_mem_addr_o;
wire [`INSTR_WIDTH-1:0] i_mem_data_i;

wire d_mem_wea;
wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o;
wire [`DATA_WIDTH-1:0] d_mem_data_i;
wire [`DATA_WIDTH-1:0] d_mem_data_o;

cpu u_cpu (
    .clk(clk),
    .rst_n(rst_n),
    .i_mem_data_i(i_mem_data_i),
    .i_mem_addr_o(i_mem_addr_o),
    .d_mem_addr_o(d_mem_addr_o),
    .d_mem_data_i(d_mem_data_i),
    .d_mem_data_o(d_mem_data_o),
    .d_mem_wen_o(d_mem_wen_o)
);

i_mem u_i_mem (
    .clk(clk),
    .din(), // No writes to instruction memory
    .addr(i_mem_addr_o), // Address from CPU
    .we(1'b0), // Should always be 0 for instruction memory
    .dout(i_mem_data_i)
);


// Data memory port B not used in this design, so tie off the inputs and ignore the outputs

wire d_mem_wea_b = 1'b0;
wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_b = `DMEM_ADDR_WIDTH'b0;
wire [`DATA_WIDTH-1:0] d_mem_dina_b = `DATA_WIDTH'b0;

d_mem u_d_mem (
    .clka(clk),
    .dina(d_mem_data_o), // Data to write to data memory
    .addra(d_mem_addr_o),
    .wea(d_mem_wea), // Write enable from CPU
    .douta(d_mem_data_i), // Data read from data memory
    // Port B (not used)
    .clkb(clk),
    .dinb(d_mem_dina_b),
    .addrb(d_mem_addr_b),
    .web(d_mem_wea_b),
    .doutb() // Unconnected output
);

endmodule


`endif // TOP_V
