/* file: pplint16mult.v
 Description: This file implements the 16-bit integer multiplication unit for the GPU.
 The unit is designed to support pipelined execution for high throughput in GPU applications.
 Please note that the int16 multiplication is gonna truncated to 16 bits. Please make sure the range of the input values
    is within the range of -128 to 127 to avoid overflow if accurate results are expected. This design implement a Dadda
    tree multiplier for efficient multiplication of two 16-bit integers, producing a 16-bit result.
 This version of the int16 multiplier is designed to be pipelined, allowing for high throughput in GPU applications.
 The pipeline stages are as follows:
    - Stage 1: Partial Product Generation and Initial Reduction (CSA Tree)
    - Stage 2: Final Addition and Output
 The output is registered and output one clock cycle after the final addition stage's clock
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.0
 Revision history:
    - Feb. 26, 2026: Initial implementation of the pipelined 16-bit integer multiplication unit.
*/

`ifndef PPLINT16MULT_V
`define PPLINT16MULT_V

`include "gpu_define.v"
`include "int16hca.v"

module pplint16mult (
    input wire clk,
    input wire rst_n,
    input wire [15:0] a,
    input wire [15:0] b,
    output reg [15:0] result
);
    // Partial product generation (triangular, truncated to 16 bits)
    // pp[k] = a[k] ? (b << k) : 0   — bits above 15 discarded
    wire [15:0] pp [0:15];
    genvar k;
    generate
        for (k = 0; k < 16; k = k + 1) begin : PP
            assign pp[k] = {16{a[k]}} & (b << k);
        end
    endgenerate

    // CSA tree reduction — Wallace-style, same depth as Dadda
    // CSA(x,y,z): sum = x^y^z, carry = maj(x,y,z) << 1

    // --- Round 1: 16 → 11 (5 CSAs, pp[15] passes through) ---
    wire [15:0] r1s0, r1c0, r1s1, r1c1, r1s2, r1c2, r1s3, r1c3, r1s4, r1c4;

    assign r1s0 = pp[0] ^ pp[1] ^ pp[2];
    assign r1c0 = ((pp[0] & pp[1]) | (pp[0] & pp[2]) | (pp[1] & pp[2])) << 1;

    assign r1s1 = pp[3] ^ pp[4] ^ pp[5];
    assign r1c1 = ((pp[3] & pp[4]) | (pp[3] & pp[5]) | (pp[4] & pp[5])) << 1;

    assign r1s2 = pp[6] ^ pp[7] ^ pp[8];
    assign r1c2 = ((pp[6] & pp[7]) | (pp[6] & pp[8]) | (pp[7] & pp[8])) << 1;

    assign r1s3 = pp[9] ^ pp[10] ^ pp[11];
    assign r1c3 = ((pp[9] & pp[10]) | (pp[9] & pp[11]) | (pp[10] & pp[11])) << 1;

    assign r1s4 = pp[12] ^ pp[13] ^ pp[14];
    assign r1c4 = ((pp[12] & pp[13]) | (pp[12] & pp[14]) | (pp[13] & pp[14])) << 1;
    // pp[15] passes through

    // --- Round 2: 11 → 8 (3 CSAs, r1c4 + pp[15] pass through) ---
    wire [15:0] r2s0, r2c0, r2s1, r2c1, r2s2, r2c2;

    assign r2s0 = r1s0 ^ r1c0 ^ r1s1;
    assign r2c0 = ((r1s0 & r1c0) | (r1s0 & r1s1) | (r1c0 & r1s1)) << 1;

    assign r2s1 = r1c1 ^ r1s2 ^ r1c2;
    assign r2c1 = ((r1c1 & r1s2) | (r1c1 & r1c2) | (r1s2 & r1c2)) << 1;

    assign r2s2 = r1s3 ^ r1c3 ^ r1s4;
    assign r2c2 = ((r1s3 & r1c3) | (r1s3 & r1s4) | (r1c3 & r1s4)) << 1;
    // r1c4, pp[15] pass through

    // --- Round 3: 8 → 6 (2 CSAs, r1c4 + pp[15] pass through) ---
    wire [15:0] r3s0, r3c0, r3s1, r3c1;

    assign r3s0 = r2s0 ^ r2c0 ^ r2s1;
    assign r3c0 = ((r2s0 & r2c0) | (r2s0 & r2s1) | (r2c0 & r2s1)) << 1;

    assign r3s1 = r2c1 ^ r2s2 ^ r2c2;
    assign r3c1 = ((r2c1 & r2s2) | (r2c1 & r2c2) | (r2s2 & r2c2)) << 1;
    // r1c4, pp[15] pass through

    // --- Round 4: 6 -> 4 (2 CSAs) ---
    wire [15:0] r4s0, r4c0, r4s1, r4c1;

    assign r4s0 = r3s0 ^ r3c0 ^ r3s1;
    assign r4c0 = ((r3s0 & r3c0) | (r3s0 & r3s1) | (r3c0 & r3s1)) << 1;

    assign r4s1 = r3c1 ^ r1c4 ^ pp[15];
    assign r4c1 = ((r3c1 & r1c4) | (r3c1 & pp[15]) | (r1c4 & pp[15])) << 1;

    // Pipeline register: 4 rows
    reg [15:0] p_s0, p_c0, p_s1, p_c1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_s0 <= 16'd0;
            p_c0 <= 16'd0;
            p_s1 <= 16'd0;
            p_c1 <= 16'd0;
        end else begin
            p_s0 <= r4s0;
            p_c0 <= r4c0;
            p_s1 <= r4s1;
            p_c1 <= r4c1;
        end
    end

    // --- Round 5: 4 -> 3 (1 CSA, p_c1 passes through) ---
    wire [15:0] r5s, r5c;

    assign r5s = p_s0 ^ p_c0 ^ p_s1;
    assign r5c = ((p_s0 & p_c0) | (p_s0 & p_s1) | (p_c0 & p_s1)) << 1;
    // p_c1 passes through

    // --- Round 6: 3 -> 2 (1 CSA) ---
    wire [15:0] r6s, r6c;

    assign r6s = r5s ^ r5c ^ p_c1;
    assign r6c = ((r5s & r5c) | (r5s & p_c1) | (r5c & p_c1)) << 1;

    // HCA for final addition
    wire [15:0] final_sum;
    wire final_cout;

    int16hca u_mult_int16hca (
        .a(r6s),
        .b(r6c),
        .cin(1'b0),
        .sum(final_sum),
        .cout(final_cout)
    );

    // Final HCA add + output register — result available 1 cycle after final sum is computed
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 16'd0;
        end else begin
            result <= final_sum;
        end
    end
endmodule


`endif // PPLINT16MULT_V