/* file: test_i_mem.v
 Description: synchronous read and synchronous write instruction memory test module
 Author: Jeremy Cai
 Date: Feb. 13, 2026
 Version: 1.0
 */

`ifndef TEST_I_MEM_V
`define TEST_I_MEM_V

module test_i_mem (
    input wire clk,
    input wire [`PC_WIDTH-1:0] addr,
    input wire [`INSTR_WIDTH-1:0] din,
    input wire we,
    output reg [`INSTR_WIDTH-1:0] dout
);

reg [`INSTR_WIDTH-1:0] mem [0:(1<<`PC_WIDTH)-1];

// Synchronous read and write logic
always @(posedge clk) begin
    if (we) begin
        mem[addr] <= din; // Write data to memory on write enable
    end
    dout <= mem[addr]; // Read data from memory on clock edge
end

endmodule

`endif // TEST_I_MEM_V
