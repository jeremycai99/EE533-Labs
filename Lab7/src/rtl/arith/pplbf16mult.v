/* file: pplbf16mult.v
 Description: This file implements the pipelined multiplication operation for BF16 format.
 BF16 format: [15] sign, [14:7] exponent (bias=127), [6:0] fraction
 Please note that all the bf16 arithmetic operations are pipelined due to practice in Lab 6 of timing violation issues.
 Pipeline stages:
    - Stage 1: Unpack, special detect, sign XOR
    - Stage 2: 8x8 mantissa multiply + exponent add
    - Stage 3: Normalize (1-bit mux), rounding, special case mux
 Author: Jeremy Cai
 Date: Feb. 24, 2026
 Version: 1.0
 Revision history:
    - Feb. 24, 2026: Initial implementation of pipelined BF16 multiplication.
 */

`ifndef PPLBF16MULT_V
`define PPLBF16MULT_V

`include "gpu_define.v"

module pplbf16mult (
    input wire clk,
    input wire rst_n,
    input wire [15:0] operand_a,
    input wire [15:0] operand_b,
    input wire valid_in,
    output reg [15:0] result,
    output reg valid_out
);

    // =========================================================================
    // Stage 1: Unpack, special detect, sign XOR
    // =========================================================================

    wire sign_a = operand_a[15];
    wire [7:0] exp_a = operand_a[14:7];
    wire [6:0] frac_a = operand_a[6:0];
    wire sign_b = operand_b[15];
    wire [7:0] exp_b = operand_b[14:7];
    wire [6:0] frac_b = operand_b[6:0];

    wire sign_r_w = sign_a ^ sign_b;

    wire is_zero_a = (exp_a == 8'h00) && (frac_a == 7'b0);
    wire is_zero_b = (exp_b == 8'h00) && (frac_b == 7'b0);
    wire is_denorm_a = (exp_a == 8'h00) && (frac_a != 7'b0);
    wire is_denorm_b = (exp_b == 8'h00) && (frac_b != 7'b0);
    wire is_inf_a = (exp_a == 8'hFF) && (frac_a == 7'b0);
    wire is_inf_b = (exp_b == 8'hFF) && (frac_b == 7'b0);
    wire is_nan_a = (exp_a == 8'hFF) && (frac_a != 7'b0);
    wire is_nan_b = (exp_b == 8'hFF) && (frac_b != 7'b0);

    wire [7:0] mant_a_w = (exp_a != 8'h00) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [7:0] mant_b_w = (exp_b != 8'h00) ? {1'b1, frac_b} : {1'b0, frac_b};

    reg special_w;
    reg [15:0] special_result_w;

    always @(*) begin
        special_w = 1'b0;
        special_result_w = 16'h0000;

        if (is_nan_a || is_nan_b) begin
            special_w = 1'b1;
            special_result_w = {1'b0, 8'hFF, 7'h40};
        end
        else if (is_inf_a || is_inf_b) begin
            special_w = 1'b1;
            if ((is_inf_a && (is_zero_b || is_denorm_b)) ||
                (is_inf_b && (is_zero_a || is_denorm_a)))
                special_result_w = {1'b0, 8'hFF, 7'h40};
            else
                special_result_w = {sign_r_w, 8'hFF, 7'h00};
        end
        else if (is_zero_a || is_zero_b || is_denorm_a || is_denorm_b) begin
            special_w = 1'b1;
            special_result_w = {sign_r_w, 8'h00, 7'h00};
        end
    end

    // Stage 1 / Stage 2 pipeline registers
    reg s1_valid;
    reg s1_sign_r;
    reg [7:0] s1_exp_a;
    reg [7:0] s1_exp_b;
    reg [7:0] s1_mant_a;
    reg [7:0] s1_mant_b;
    reg s1_special;
    reg [15:0] s1_special_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_sign_r <= 1'b0;
            s1_exp_a <= 8'b0;
            s1_exp_b <= 8'b0;
            s1_mant_a <= 8'b0;
            s1_mant_b <= 8'b0;
            s1_special <= 1'b0;
            s1_special_result <= 16'b0;
        end else begin
            s1_valid <= valid_in;
            s1_sign_r <= sign_r_w;
            s1_exp_a <= exp_a;
            s1_exp_b <= exp_b;
            s1_mant_a <= mant_a_w;
            s1_mant_b <= mant_b_w;
            s1_special <= special_w;
            s1_special_result <= special_result_w;
        end
    end

    // =========================================================================
    // Stage 2: 8x8 mantissa multiply + exponent add
    // =========================================================================

    wire [15:0] mant_prod_w = s1_mant_a * s1_mant_b;
    wire [9:0] exp_sum_w = {2'b0, s1_exp_a} + {2'b0, s1_exp_b} - 10'd127;

    // Stage 2 / Stage 3 pipeline registers
    reg s2_valid;
    reg s2_sign_r;
    reg [15:0] s2_mant_prod;
    reg [9:0] s2_exp_sum;
    reg s2_special;
    reg [15:0] s2_special_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            s2_sign_r <= 1'b0;
            s2_mant_prod <= 16'b0;
            s2_exp_sum <= 10'b0;
            s2_special <= 1'b0;
            s2_special_result <= 16'b0;
        end else begin
            s2_valid <= s1_valid;
            s2_sign_r <= s1_sign_r;
            s2_mant_prod <= mant_prod_w;
            s2_exp_sum <= exp_sum_w;
            s2_special <= s1_special;
            s2_special_result <= s1_special_result;
        end
    end

    // =========================================================================
    // Stage 3: Normalize (1-bit mux), rounding, special case mux
    // =========================================================================

    wire norm_shift = s2_mant_prod[15];

    wire [6:0] frac_w = norm_shift ? s2_mant_prod[14:8] : s2_mant_prod[13:7];
    wire guard = norm_shift ? s2_mant_prod[7] : s2_mant_prod[6];
    wire round_b = norm_shift ? s2_mant_prod[6] : s2_mant_prod[5];
    wire sticky = norm_shift ? (|s2_mant_prod[5:0]) : (|s2_mant_prod[4:0]);

    wire [9:0] exp_norm = norm_shift ? (s2_exp_sum + 10'd1) : s2_exp_sum;

    wire round_up = guard & (round_b | sticky | frac_w[0]);
    wire [7:0] frac_rounded = {1'b0, frac_w} + {7'b0, round_up};

    wire round_ovf = frac_rounded[7];
    wire [6:0] frac_final = round_ovf ? frac_rounded[7:1] : frac_rounded[6:0];
    wire [9:0] exp_final = round_ovf ? (exp_norm + 10'd1) : exp_norm;

    reg [15:0] result_w;
    always @(*) begin
        if (s2_special)
            result_w = s2_special_result;
        else if (exp_final >= 10'd255)
            result_w = {s2_sign_r, 8'hFF, 7'h00};
        else if (exp_final == 10'd0 || exp_final[9])
            result_w = {s2_sign_r, 8'h00, 7'h00};
        else
            result_w = {s2_sign_r, exp_final[7:0], frac_final};
    end

    // Output register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 16'b0;
            valid_out <= 1'b0;
        end else begin
            result <= result_w;
            valid_out <= s2_valid;
        end
    end

endmodule

`endif // PPLBF16MULT_V