/* file: int16hca.v
 Description: This file implements the Han-Carlson Adder (HCA) for fast addition of two 16-bit numbers.
 We don't take care of carry-in and carry-out
 Author: Jeremy Cai
 Date: Feb. 25, 2026
 Version: 1.0
 Revision history:
    - Feb. 25, 2026: Initial implementation of Han-Carlson Adder (HCA) for 16-bit numbers.
 */
 
`ifndef INT16HCA_V
`define INT16HCA_V

module int16hca (
    input wire [15:0] a,
    input wire [15:0] b,
    input wire cin,
    output wire [15:0] sum,
    output wire cout
);

    wire [15:0] g_raw = a & b;
    wire [15:0] p_raw = a ^ b;

    wire [15:0] g0;
    wire [15:0] p0;

    assign g0[0] = g_raw[0] | (p_raw[0] & cin);
    assign p0[0] = p_raw[0] ^ cin;

    genvar i;
    generate
        for (i = 1; i < 16; i = i + 1) begin : L0
            assign g0[i] = g_raw[i];
            assign p0[i] = p_raw[i];
        end
    endgenerate

    // =========================================================================
    // Level 1: Distance-1, odd positions only
    // =========================================================================
    wire [15:0] g1, p1;

    generate
        for (i = 0; i < 16; i = i + 1) begin : L1
            if (i % 2 == 1) begin : odd
                assign g1[i] = g0[i] | (p0[i] & g0[i-1]);
                assign p1[i] = p0[i] & p0[i-1];
            end else begin : even
                assign g1[i] = g0[i];
                assign p1[i] = p0[i];
            end
        end
    endgenerate

    // =========================================================================
    // Levels 2-4: Kogge-Stone among odd positions (distance 2, 4, 8)
    // =========================================================================
    wire [15:0] g_ks [0:3];
    wire [15:0] p_ks [0:3];

    assign g_ks[0] = g1;
    assign p_ks[0] = p1;
    
    genvar k;
    generate
        for (k = 1; k <= 3; k = k + 1) begin : KS_LEVEL
            localparam integer DIST = (1 << k);
            for (i = 0; i < 16; i = i + 1) begin : BIT
                if ((i % 2 == 1) && (i >= DIST + 1)) begin : active
                    assign g_ks[k][i] = g_ks[k-1][i] | (p_ks[k-1][i] & g_ks[k-1][i - DIST]);
                    assign p_ks[k][i] = p_ks[k-1][i] & p_ks[k-1][i - DIST];
                end else begin : pass
                    assign g_ks[k][i] = g_ks[k-1][i];
                    assign p_ks[k][i] = p_ks[k-1][i];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Level 5: Even positions from odd neighbors
    // =========================================================================
    wire [15:0] gf;

    assign gf[0] = g0[0];

    generate
        for (i = 1; i < 16; i = i + 1) begin : L5
            if (i % 2 == 1) begin : odd_done
                assign gf[i] = g_ks[3][i];
            end else begin : even_fill
                assign gf[i] = g0[i] | (p0[i] & g_ks[3][i-1]);
            end
        end
    endgenerate

    // =========================================================================
    // Level 6: Post-processing
    // =========================================================================
    assign sum[0] = p_raw[0] ^ cin;

    generate
        for (i = 1; i < 16; i = i + 1) begin : SUM
            assign sum[i] = p_raw[i] ^ gf[i-1];
        end
    endgenerate

    assign cout = gf[15];

    endmodule

`endif // INT16HCA_V