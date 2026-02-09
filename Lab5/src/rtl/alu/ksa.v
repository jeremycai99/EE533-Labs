/* file: ksa.v
 Description: This file implements the 64-bit kogge-stone adder for the ALU.
 Please note that this implementation is for 64 only because of the stages required.
 Author: Jeremy Cai
 Date: Feb. 8, 2026
 Version: 1.0
 */

`ifndef KSA_V
`define KSA_V

`include "define.v"

module ksa (
    input wire [63:0] operand_a,     // First operand
    input wire [63:0] operand_b,     // Second operand
    input wire cin,                  // Carry input for addition
    output wire [63:0] sum,          // Sum output
    output wire cout                 // Carry output for addition
);

//NOTICE: THIS IS A FIXED VALUE FOR ADDER!
//Higher hierarchical logic should padding if needed, and this module will only handle 64 bits of addition.
localparam N = 65; // Total width including carry.

//Generate and Propagate signals
wire [N-1:0] g0, p0;
wire [N-1:0] g1, p1;
wire [N-1:0] g2, p2;
wire [N-1:0] g3, p3;
wire [N-1:0] g4, p4;
wire [N-1:0] g5, p5;
wire [N-1:0] g6, p6;
wire [N-1:0] g7, p7;

// Initial Generate and Propagate
assign g0[0] = cin;
assign p0[0] = 1'b0; // No propagate for carry input

// Stage 0
genvar i0;
generate
    for (i0 = 1; i0 < N; i0 = i0 + 1) begin
        assign g0[i0] = operand_a[i0-1] & operand_b[i0-1]; // Generate
        assign p0[i0] = operand_a[i0-1] ^ operand_b[i0-1]; // Propagate
    end
endgenerate

// Stage 1
genvar i1;
generate
    for (i1 = 0; i1 < N; i1 = i1 + 1) begin
        if (i1 < 1) begin
            assign g1[i1] = g0[i1];
            assign p1[i1] = p0[i1];
        end else begin
            assign g1[i1] = g0[i1] | (p0[i1] & g0[i1-1]);
            assign p1[i1] = p0[i1] & p0[i1-1];
        end
    end
endgenerate

// Stage 2
genvar i2;
generate
    for (i2 = 0; i2 < N; i2 = i2 + 1) begin
        if (i2 < 2) begin
            assign g2[i2] = g1[i2];
            assign p2[i2] = p1[i2];
        end else begin
            assign g2[i2] = g1[i2] | (p1[i2] & g1[i2-2]);
            assign p2[i2] = p1[i2] & p1[i2-2];
        end
    end
endgenerate

// Stage 3
genvar i3;
generate
    for (i3 = 0; i3 < N; i3 = i3 + 1) begin
        if (i3 < 4) begin
        assign g3[i3] = g2[i3];
        assign p3[i3] = p2[i3];
        end else begin
        assign g3[i3] = g2[i3] | (p2[i3] & g2[i3-4]);
        assign p3[i3] = p2[i3] & p2[i3-4];
        end
    end
endgenerate

// Stage 4
genvar i4;
generate
    for (i4 = 0; i4 < N; i4 = i4 + 1) begin
        if (i4 < 8) begin
        assign g4[i4] = g3[i4];
        assign p4[i4] = p3[i4];
        end else begin
        assign g4[i4] = g3[i4] | (p3[i4] & g3[i4-8]);
        assign p4[i4] = p3[i4] & p3[i4-8];
        end
    end
endgenerate

// Stage 5
genvar i5;
generate
    for (i5 = 0; i5 < N; i5 = i5 + 1) begin
        if (i5 < 16) begin
        assign g5[i5] = g4[i5];
        assign p5[i5] = p4[i5];
        end else begin
        assign g5[i5] = g4[i5] | (p4[i5] & g4[i5-16]);
        assign p5[i5] = p4[i5] & p4[i5-16];
        end
    end
endgenerate

// Stage 6
genvar i6;
generate
    for (i6 = 0; i6 < N; i6 = i6 + 1) begin
        if (i6 < 32) begin
        assign g6[i6] = g5[i6];
        assign p6[i6] = p5[i6];
        end else begin
        assign g6[i6] = g5[i6] | (p5[i6] & g5[i6-32]);
        assign p6[i6] = p5[i6] & p5[i6-32];
        end
    end
endgenerate

// Stage 7
genvar i7;
generate
    for (i7 = 0; i7 < N; i7 = i7 + 1) begin
        if (i7 < 64) begin
        assign g7[i7] = g6[i7];
        assign p7[i7] = p6[i7];
        end else begin
        assign g7[i7] = g6[i7] | (p6[i7] & g6[i7-64]);
        assign p7[i7] = p6[i7] & p6[i7-64];
        end
    end
endgenerate

genvar k;
generate
    for (k = 0; k < 64; k = k + 1) begin
        assign sum[k] = p0[k+1] ^ g7[k];
    end
endgenerate

assign cout = g7[64];

endmodule

`endif //KSA_V