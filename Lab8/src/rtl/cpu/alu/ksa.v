/* file: ksa.v
 Description: This file implements the 64-bit kogge-stone adder for the ALU.
 Please note that this implementation is for 64 only because of the stages required.
 Author: Jeremy Cai
 Date: Feb. 8, 2026
 Version: 1.0
 Revision History:
    - Feb. 8, 2026 v1.0: 64 bit adder for 64 bit data
    - Mar. 5, 2026 v1.1: Reduce to 32 bit for low resource usage
 */

`ifndef KSA_V
`define KSA_V

`include "define.v"

module ksa (
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    input wire cin,
    output wire [31:0] sum,
    output wire cout
);

localparam N = 33; // 32 data bits + 1 carry-in position

wire [N-1:0] g0, p0;
wire [N-1:0] g1, p1;
wire [N-1:0] g2, p2;
wire [N-1:0] g3, p3;
wire [N-1:0] g4, p4;
wire [N-1:0] g5, p5;

// Initial Generate and Propagate
assign g0[0] = cin;
assign p0[0] = 1'b0;

// Stage 0
genvar i0;
generate
    for (i0 = 1; i0 < N; i0 = i0 + 1) begin : gen_stage0
        assign g0[i0] = operand_a[i0-1] & operand_b[i0-1];
        assign p0[i0] = operand_a[i0-1] ^ operand_b[i0-1];
    end
endgenerate

// Stage 1 (span 1)
genvar i1;
generate
    for (i1 = 0; i1 < N; i1 = i1 + 1) begin : gen_stage1
        if (i1 < 1) begin : gen_s1_pass
            assign g1[i1] = g0[i1];
            assign p1[i1] = p0[i1];
        end else begin : gen_s1_merge
            assign g1[i1] = g0[i1] | (p0[i1] & g0[i1-1]);
            assign p1[i1] = p0[i1] & p0[i1-1];
        end
    end
endgenerate

// Stage 2 (span 2)
genvar i2;
generate
    for (i2 = 0; i2 < N; i2 = i2 + 1) begin : gen_stage2
        if (i2 < 2) begin : gen_s2_pass
            assign g2[i2] = g1[i2];
            assign p2[i2] = p1[i2];
        end else begin : gen_s2_merge
            assign g2[i2] = g1[i2] | (p1[i2] & g1[i2-2]);
            assign p2[i2] = p1[i2] & p1[i2-2];
        end
    end
endgenerate

// Stage 3 (span 4)
genvar i3;
generate
    for (i3 = 0; i3 < N; i3 = i3 + 1) begin : gen_stage3
        if (i3 < 4) begin : gen_s3_pass
            assign g3[i3] = g2[i3];
            assign p3[i3] = p2[i3];
        end else begin : gen_s3_merge
            assign g3[i3] = g2[i3] | (p2[i3] & g2[i3-4]);
            assign p3[i3] = p2[i3] & p2[i3-4];
        end
    end
endgenerate

// Stage 4 (span 8)
genvar i4;
generate
    for (i4 = 0; i4 < N; i4 = i4 + 1) begin : gen_stage4
        if (i4 < 8) begin : gen_s4_pass
            assign g4[i4] = g3[i4];
            assign p4[i4] = p3[i4];
        end else begin : gen_s4_merge
            assign g4[i4] = g3[i4] | (p3[i4] & g3[i4-8]);
            assign p4[i4] = p3[i4] & p3[i4-8];
        end
    end
endgenerate

// Stage 5 (span 16)
genvar i5;
generate
    for (i5 = 0; i5 < N; i5 = i5 + 1) begin : gen_stage5
        if (i5 < 16) begin : gen_s5_pass
            assign g5[i5] = g4[i5];
            assign p5[i5] = p4[i5];
        end else begin : gen_s5_merge
            assign g5[i5] = g4[i5] | (p4[i5] & g4[i5-16]);
            assign p5[i5] = p4[i5] & p4[i5-16];
        end
    end
endgenerate

// Final sum
genvar k;
generate
    for (k = 0; k < 32; k = k + 1) begin : gen_sum
        assign sum[k] = p0[k+1] ^ g5[k];
    end
endgenerate

assign cout = g5[32];

endmodule

`endif //KSA_V