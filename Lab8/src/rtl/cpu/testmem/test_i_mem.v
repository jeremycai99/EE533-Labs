/* file: test_i_mem.v
 Description: synchronous read and synchronous write instruction memory test module
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 Version: 2.0
 Revision History:
    - Version 1.0: Initial implementation (Feb. 13, 2026)
    - Version 2.0: Updated for full pkt_proc connectivity (Mar. 5, 2026)
 */

`ifndef TEST_I_MEM_V
`define TEST_I_MEM_V

module test_i_mem (
    input wire clka,
    input wire [`IMEM_ADDR_WIDTH-1:0] addra,
    input wire [`IMEM_DATA_WIDTH-1:0] dina,
    input wire wea,
    output reg [`IMEM_DATA_WIDTH-1:0] douta,

    input wire clkb,
    input wire [`IMEM_ADDR_WIDTH-1:0] addrb,
    input wire [`IMEM_DATA_WIDTH-1:0] dinb,
    input wire web,
    output reg [`IMEM_DATA_WIDTH-1:0] doutb
);

reg [`IMEM_DATA_WIDTH-1:0] mem [0:(1<<`IMEM_ADDR_WIDTH)-1];
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

`endif // TEST_I_MEM_V
