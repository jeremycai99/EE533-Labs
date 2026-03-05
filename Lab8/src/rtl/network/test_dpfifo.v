/* file: test_dpfifo.v
Description: Dual-port BRAM for Convertible FIFO. Default: 256×72b (64b data + 8b ctrl), parameterizable.
Author: Jeremy Cai
Date: Mar. 4, 2026
Version: 1.0
*/


`ifndef TEST_DPFIFO_V
`define TEST_DPFIFO_V

module test_dpfifo #(
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 64
)(
    input wire clka,
    input wire [ADDR_WIDTH-1:0] addra,
    input wire [DATA_WIDTH-1:0] dina,
    input wire wea,
    output reg [DATA_WIDTH-1:0] douta,

    input wire clkb,
    input wire [ADDR_WIDTH-1:0] addrb,
    input wire [DATA_WIDTH-1:0] dinb,
    input wire web,
    output reg [DATA_WIDTH-1:0] doutb
);

    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Port A: synchronous read and write
    always @(posedge clka) begin
        if (wea)
            mem[addra] <= dina;
        douta <= mem[addra];
    end

    // Port B: synchronous read and write
    always @(posedge clkb) begin
        if (web)
            mem[addrb] <= dinb;
        doutb <= mem[addrb];
    end

endmodule

`endif // TEST_DPFIFO_V