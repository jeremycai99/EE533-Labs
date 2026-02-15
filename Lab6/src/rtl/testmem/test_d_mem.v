/* file: test_d_mem.v
 Description: synchronous read and synchronous write data memory test module
 Author: Jeremy Cai
 Date: Feb. 13, 2026
 Version: 1.0
 */

`ifndef TEST_D_MEM_V
`define TEST_D_MEM_V

module test_d_mem (
    input wire clka,
    input wire [`DMEM_ADDR_WIDTH-1:0] addra,
    input wire [`DATA_WIDTH-1:0] dina,
    input wire wea,
    output reg [`DATA_WIDTH-1:0] douta,

    input wire clkb,
    input wire [`DMEM_ADDR_WIDTH-1:0] addrb,
    input wire [`DATA_WIDTH-1:0] dinb,
    input wire web,
    output reg [`DATA_WIDTH-1:0] doutb
);

reg [`DATA_WIDTH-1:0] mem [0:(1<<`DMEM_ADDR_WIDTH)-1];
// Synchronous read and write logic for Port A
always @(posedge clka) begin
    if (wea) begin
        mem[addra] <= dina; // Write data to memory on write enable
    end
    douta <= mem[addra]; // Read data from memory on clock edge
end

// Synchronous read and write logic for Port B
always @(posedge clkb) begin
    if (web) begin
        mem[addrb] <= dinb; // Write data to memory on write enable
    end
    doutb <= mem[addrb]; // Read data from memory on clock edge
end

endmodule

`endif // TEST_D_MEM_V