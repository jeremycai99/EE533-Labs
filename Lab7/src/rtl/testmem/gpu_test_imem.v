/* file: gpu_test_imem.v
 Description: This file implements the wrapper for GPU instruction memory module.
 Author: Jeremy Cai
 Date: Feb. 27, 2026
 Version: 1.0
 Revision history:
    - Feb. 27, 2026: Initial implementation of the wrapper for GPU instruction memory module.
*/

`ifndef GPU_TEST_IMEM_V
`define GPU_TEST_IMEM_V

module gpu_test_imem (
    input wire clka,
    input wire [`GPU_IMEM_ADDR_WIDTH-1:0] addra,
    input wire [`GPU_IMEM_DATA_WIDTH-1:0] dina,
    input wire wea,
    output reg [`GPU_IMEM_DATA_WIDTH-1:0] douta,

    input wire clkb,
    input wire [`GPU_DMEM_ADDR_WIDTH-1:0] addrb,
    input wire [`GPU_DMEM_DATA_WIDTH-1:0] dinb,
    input wire web,
    output reg [`GPU_DMEM_DATA_WIDTH-1:0] doutb
);

reg [`GPU_DMEM_DATA_WIDTH-1:0] mem [0:(1<<`GPU_DMEM_ADDR_WIDTH)-1];
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

`endif // GPU_TEST_IMEM_V