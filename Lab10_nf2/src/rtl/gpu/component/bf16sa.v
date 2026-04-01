/* file: bf16sa.v
 Description: This file implements the systolic array for BF16 format for SM-wise tensor core design.
 Please note that the bf16 arithmetic operations are pipelined: pplbf16mult (3 stages), pplbf16adsub (4 stages)
 This bf16sa module only consider 4*4 matrix multiplication for simplicity. No parameter controlled array size implemented
 Flattened array design for resolving ISE synthesis issues. 
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.0
 Revision history:
        - Feb. 26, 2026: Initial implementation of systolic array for BF16 format.
*/

`ifndef BF16SA_V
`define BF16SA_V

`include "gpu_define.v"
`include "bf16pe.v"

module bf16sa (
    input wire clk,
    input wire rst_n,
    input wire acc_load,
    input wire [4*4*16-1:0] acc_in,   // C[row][col], [(i*4+j)*16 +: 16]
    input wire [4*16-1:0] a_in,       // a[row],      [i*16 +: 16]
    input wire [3:0] a_valid,         // valid[row]
    input wire [4*16-1:0] b_in,       // b[col],      [j*16 +: 16]
    output wire [4*4*16-1:0] d_out    // D[row][col], [(i*4+j)*16 +: 16]
);
    // Horizontal wires: h_a[row][edge], h_v[row][edge]
    wire [15:0] h_a [0:3][0:4];
    wire h_v [0:3][0:4];
    // Vertical wires: v_b[edge][col]
    wire [15:0] v_b [0:4][0:3];

    // Connect external inputs to left/top edges
    genvar k;
    generate
        for (k = 0; k < 4; k = k + 1) begin : EDGE
            assign h_a[k][0] = a_in[k*16 +: 16];
            assign h_v[k][0] = a_valid[k];
            assign v_b[0][k] = b_in[k*16 +: 16];
        end
    endgenerate

    // 4x4 PE grid
    genvar i, j;
    generate
        for (i = 0; i < 4; i = i + 1) begin : ROW
            for (j = 0; j < 4; j = j + 1) begin : COL
                bf16pe u_pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .acc_load(acc_load),
                    .acc_in(acc_in[(i*4+j)*16 +: 16]),
                    .a_in(h_a[i][j]),
                    .b_in(v_b[i][j]),
                    .valid_in(h_v[i][j]),
                    .a_out(h_a[i][j+1]),
                    .b_out(v_b[i+1][j]),
                    .valid_out(h_v[i][j+1]),
                    .acc_out(d_out[(i*4+j)*16 +: 16])
                );
            end
        end
    endgenerate
endmodule

`endif // BF16SA_V