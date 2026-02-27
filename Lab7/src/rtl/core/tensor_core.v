/* file: tensor_core.v
 Description: This file implements the tensor core for BF16 format for SM-wise tensor core design.
 Flattened array design for ISE synthesis issues. 
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.0
 Revision history:
        - Feb. 26, 2026: Initial implementation of tensor core for BF16 format.
*/

`ifndef TENSOR_CORE_V
`define TENSOR_CORE_V

`include "gpu_define.v"
`include "bf16sa.v"

module tensor_core (
    input wire clk,
    input wire rst_n,
    input wire [4*4*16-1:0] matrix_a,   // A[(i*4+j)*16 +: 16]
    input wire [4*4*16-1:0] matrix_b,   // B[(i*4+j)*16 +: 16]
    input wire [4*4*16-1:0] matrix_c,   // C[(i*4+j)*16 +: 16]
    input wire valid_in,
    output wire [4*4*16-1:0] matrix_d,  // D[(i*4+j)*16 +: 16]
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

    // State machine
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
        end else begin
            valid_out <= 1'b0;
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
                    round <= 2'd0;
                    phase <= 3'd0;
                    state <= S_FEED;
                end
                S_FEED: begin
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

    // Combinational: SA control signals
    // A[row][col] = A_reg[(row*4+col)*16 +: 16]
    // B[row][col] = B_reg[(row*4+col)*16 +: 16]
    integer idx;
    always @(*) begin
        sa_acc_load = 1'b0;
        sa_a_valid = 4'd0;
        sa_a_in = {(4*16){1'b0}};
        sa_b_in = {(4*16){1'b0}};
        case (state)
            S_LOAD: sa_acc_load = 1'b1;
            S_FEED: begin
                for (idx = 0; idx < 4; idx = idx + 1) begin
                    if (phase[1:0] == idx[1:0] && phase < 3'd4) begin
                        sa_a_in[idx*16 +: 16] = A_reg[(idx*4+round)*16 +: 16];
                        sa_a_valid[idx] = 1'b1;
                        sa_b_in[idx*16 +: 16] = B_reg[(round*4+idx)*16 +: 16];
                    end
                end
            end
            default: ;
        endcase
    end
endmodule
`endif // TENSOR_CORE_V