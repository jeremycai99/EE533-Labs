/* file: tensor_core.v
 Description: This file implements the tensor core for BF16 format for SM-wise tensor core design.
 Fully unrolled, no partial register writes — XST-safe.
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.1
 Revision history:
    - Feb. 26, 2026: Initial implementation of tensor core for BF16 format.
    - Mar. 01, 2026: Rewritten for ISE XST synthesis compatibility.
*/

`ifndef TENSOR_CORE_V
`define TENSOR_CORE_V

`include "gpu_define.v"
`include "bf16sa.v"

module tensor_core (
    input wire clk,
    input wire rst_n,
    input wire [4*4*16-1:0] matrix_a,
    input wire [4*4*16-1:0] matrix_b,
    input wire [4*4*16-1:0] matrix_c,
    input wire valid_in,
    output wire [4*4*16-1:0] matrix_d,
    output reg valid_out
);
    localparam S_IDLE  = 2'd0;
    localparam S_LOAD  = 2'd1;
    localparam S_FEED  = 2'd2;
    localparam S_DRAIN = 2'd3;

    reg [1:0] state;
    reg [1:0] round;
    reg [2:0] phase;
    reg [2:0] drain_cnt;
    reg [4*4*16-1:0] A_reg;
    reg [4*4*16-1:0] B_reg;
    reg [4*4*16-1:0] C_reg;
    reg sa_acc_load;
    reg [4*16-1:0] sa_a_in;
    reg [3:0] sa_a_valid;
    reg [4*16-1:0] sa_b_in;

    bf16sa u_sa (
        .clk(clk),
        .rst_n(rst_n),
        .acc_load(sa_acc_load),
        .acc_in(C_reg),
        .a_in(sa_a_in),
        .a_valid(sa_a_valid),
        .b_in(sa_b_in),
        .d_out(matrix_d)
    );

    // ================================================================
    // Mux logic for SA inputs — fully combinational, no partial writes
    // Each row's a/b value selected by round, gated by phase
    // ================================================================

    // Row 0: A[0][round], B[round][0]
    reg [15:0] row0_a, row0_b;
    always @(*) begin
        case (round)
            2'd0: begin row0_a = A_reg[(0*4+0)*16 +: 16]; row0_b = B_reg[(0*4+0)*16 +: 16]; end
            2'd1: begin row0_a = A_reg[(0*4+1)*16 +: 16]; row0_b = B_reg[(1*4+0)*16 +: 16]; end
            2'd2: begin row0_a = A_reg[(0*4+2)*16 +: 16]; row0_b = B_reg[(2*4+0)*16 +: 16]; end
            2'd3: begin row0_a = A_reg[(0*4+3)*16 +: 16]; row0_b = B_reg[(3*4+0)*16 +: 16]; end
        endcase
    end

    // Row 1: A[1][round], B[round][1]
    reg [15:0] row1_a, row1_b;
    always @(*) begin
        case (round)
            2'd0: begin row1_a = A_reg[(1*4+0)*16 +: 16]; row1_b = B_reg[(0*4+1)*16 +: 16]; end
            2'd1: begin row1_a = A_reg[(1*4+1)*16 +: 16]; row1_b = B_reg[(1*4+1)*16 +: 16]; end
            2'd2: begin row1_a = A_reg[(1*4+2)*16 +: 16]; row1_b = B_reg[(2*4+1)*16 +: 16]; end
            2'd3: begin row1_a = A_reg[(1*4+3)*16 +: 16]; row1_b = B_reg[(3*4+1)*16 +: 16]; end
        endcase
    end

    // Row 2: A[2][round], B[round][2]
    reg [15:0] row2_a, row2_b;
    always @(*) begin
        case (round)
            2'd0: begin row2_a = A_reg[(2*4+0)*16 +: 16]; row2_b = B_reg[(0*4+2)*16 +: 16]; end
            2'd1: begin row2_a = A_reg[(2*4+1)*16 +: 16]; row2_b = B_reg[(1*4+2)*16 +: 16]; end
            2'd2: begin row2_a = A_reg[(2*4+2)*16 +: 16]; row2_b = B_reg[(2*4+2)*16 +: 16]; end
            2'd3: begin row2_a = A_reg[(2*4+3)*16 +: 16]; row2_b = B_reg[(3*4+2)*16 +: 16]; end
        endcase
    end

    // Row 3: A[3][round], B[round][3]
    reg [15:0] row3_a, row3_b;
    always @(*) begin
        case (round)
            2'd0: begin row3_a = A_reg[(3*4+0)*16 +: 16]; row3_b = B_reg[(0*4+3)*16 +: 16]; end
            2'd1: begin row3_a = A_reg[(3*4+1)*16 +: 16]; row3_b = B_reg[(1*4+3)*16 +: 16]; end
            2'd2: begin row3_a = A_reg[(3*4+2)*16 +: 16]; row3_b = B_reg[(2*4+3)*16 +: 16]; end
            2'd3: begin row3_a = A_reg[(3*4+3)*16 +: 16]; row3_b = B_reg[(3*4+3)*16 +: 16]; end
        endcase
    end

    // Per-row active: phase[1:0] == row index AND phase < 4
    wire row0_active = (phase[1:0] == 2'd0) && (phase < 3'd4);
    wire row1_active = (phase[1:0] == 2'd1) && (phase < 3'd4);
    wire row2_active = (phase[1:0] == 2'd2) && (phase < 3'd4);
    wire row3_active = (phase[1:0] == 2'd3) && (phase < 3'd4);

    // Build full sa_a_in / sa_b_in / sa_a_valid — always complete assignment
    wire [63:0] feed_a_in = {row3_active ? row3_a : 16'd0,
                             row2_active ? row2_a : 16'd0,
                             row1_active ? row1_a : 16'd0,
                             row0_active ? row0_a : 16'd0};

    wire [63:0] feed_b_in = {row3_active ? row3_b : 16'd0,
                             row2_active ? row2_b : 16'd0,
                             row1_active ? row1_b : 16'd0,
                             row0_active ? row0_b : 16'd0};

    wire [3:0] feed_valid = {row3_active, row2_active, row1_active, row0_active};

    // ================================================================
    // State machine — clean full-width assignments only
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            round <= 2'd0;
            phase <= 3'd0;
            drain_cnt <= 3'd0;
            valid_out <= 1'b0;
            A_reg <= {(4*4*16){1'b0}};
            B_reg <= {(4*4*16){1'b0}};
            C_reg <= {(4*4*16){1'b0}};
            sa_acc_load <= 1'b0;
            sa_a_valid <= 4'd0;
            sa_a_in <= 64'd0;
            sa_b_in <= 64'd0;
        end else begin
            valid_out <= 1'b0;
            sa_acc_load <= 1'b0;
            sa_a_valid <= 4'd0;
            sa_a_in <= 64'd0;
            sa_b_in <= 64'd0;
            case (state)
                S_IDLE: begin
                    if (valid_in) begin
                        A_reg <= matrix_a;
                        B_reg <= matrix_b;
                        C_reg <= matrix_c;
                        state <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    sa_acc_load <= 1'b1;
                    round <= 2'd0;
                    phase <= 3'd0;
                    state <= S_FEED;
                end
                S_FEED: begin
                    sa_a_in <= feed_a_in;
                    sa_b_in <= feed_b_in;
                    sa_a_valid <= feed_valid;
                    if (phase == 3'd6) begin
                        phase <= 3'd0;
                        if (round == 2'd3) begin
                            drain_cnt <= 3'd0;
                            state <= S_DRAIN;
                        end else
                            round <= round + 2'd1;
                    end else
                        phase <= phase + 3'd1;
                end
                S_DRAIN: begin
                    if (drain_cnt == 3'd6) begin
                        valid_out <= 1'b1;
                        state <= S_IDLE;
                    end else
                        drain_cnt <= drain_cnt + 3'd1;
                end
            endcase
        end
    end

endmodule

`endif // TENSOR_CORE_V