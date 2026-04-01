/* file: test_int16mult.v
 Description: Behavioral model of Xilinx int16mult IP for iverilog simulation.
 Matches IP interface: clk, a[15:0], b[15:0], result[31:0]
 Latency: 1 clock cycle
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 */

module test_int16mult (
    input wire clk,
    input wire [15:0] a,
    input wire [15:0] b,
    output reg [31:0] p
);

    // 1-cycle registered output
    always @(posedge clk) begin
        p <= a * b;
    end

endmodule