/* file: pplbf16addsub.v
 Description: This file implements the pipelined addition and subtraction operations for BF16 format.
 BF16 format: [15] sign, [14:7] exponent (bias=127), [6:0] fraction
 Please note that all the bf16 arithmetic operations are pipelined due to practice in Lab 6 of timing violation issues.
 Pipeline stages:
    Stage 1: Unpacking and special case detection
    Stage 2: Mantissa alignment and addition/subtraction
    Stage 3: Leading zero count, normalization, and rounding
    Stage 4: rounding, special case maxing and result packing
 Author: Jeremy Cai
 Date: Feb. 24, 2026
 Version: 1.0
 Revision history:
    - Feb. 24, 2026: Initial implementation of pipelined BF16 addition and subtraction.
 */

`ifndef PPLBF16ADDSUB_V
`define PPLBF16ADDSUB_V

`include "gpu_define.v"

module pplbf16addsub (
    input wire clk,
    input wire rst_n,
    input wire [15:0] operand_a,
    input wire [15:0] operand_b,
    input wire sub,
    input wire valid_in,
    output reg [15:0] result,
    output reg valid_out
);

    // =========================================================================
    // Stage 1: Unpack, special detect, exponent compare, swap
    // =========================================================================

    // --- Combinational ---
    wire sign_a = operand_a[15];
    wire [7:0] exp_a = operand_a[14:7];
    wire [6:0] frac_a = operand_a[6:0];
    wire sign_b = operand_b[15];
    wire [7:0] exp_b = operand_b[14:7];
    wire [6:0] frac_b = operand_b[6:0];

    wire sign_b_eff = sign_b ^ sub;

    // Special value detection
    wire is_zero_a = (exp_a == 8'h00) && (frac_a == 7'b0);
    wire is_zero_b = (exp_b == 8'h00) && (frac_b == 7'b0);
    wire is_inf_a = (exp_a == 8'hFF) && (frac_a == 7'b0);
    wire is_inf_b = (exp_b == 8'hFF) && (frac_b == 7'b0);
    wire is_nan_a = (exp_a == 8'hFF) && (frac_a != 7'b0);
    wire is_nan_b = (exp_b == 8'hFF) && (frac_b != 7'b0);

    // Implicit leading bit
    wire [7:0] mant_a = (exp_a != 8'h00) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [7:0] mant_b = (exp_b != 8'h00) ? {1'b1, frac_b} : {1'b0, frac_b};

    // Effective subtraction flag
    wire eff_sub_w = sign_a ^ sign_b_eff;

    // Compare & swap: put larger magnitude first
    wire a_lt_b = (exp_a < exp_b) || ((exp_a == exp_b) && (mant_a < mant_b));

    wire sign_lg_w = a_lt_b ? sign_b_eff : sign_a;
    wire [7:0] exp_lg_w = a_lt_b ? exp_b  : exp_a;
    wire [7:0] mant_lg_w = a_lt_b ? mant_b : mant_a;
    wire [7:0] mant_sm_w = a_lt_b ? mant_a : mant_b;
    wire [7:0] exp_diff_w = (a_lt_b ? exp_b : exp_a) - (a_lt_b ? exp_a : exp_b);

    // Encode special cases into a compact code + payload
    // Avoids re-checking special flags in stage 4
    //   0: normal path
    //   1: result is precomputed (NaN, inf, zero passthrough)
    reg special_w;
    reg [15:0] special_result_w;

    always @(*) begin
        special_w = 1'b0;
        special_result_w = 16'h0000;

        if (is_nan_a || is_nan_b) begin
            special_w = 1'b1;
            special_result_w = {1'b0, 8'hFF, 7'h40};
        end
        else if (is_inf_a && is_inf_b) begin
            special_w = 1'b1;
            special_result_w = eff_sub_w ? {1'b0, 8'hFF, 7'h40}  // inf - inf = NaN
                                         : {sign_a, 8'hFF, 7'h00}; // inf + inf = inf
        end
        else if (is_inf_a) begin
            special_w = 1'b1;
            special_result_w = {sign_a, 8'hFF, 7'h00};
        end
        else if (is_inf_b) begin
            special_w = 1'b1;
            special_result_w = {sign_b_eff, 8'hFF, 7'h00};
        end
        else if (is_zero_a && is_zero_b) begin
            special_w = 1'b1;
            special_result_w = (sign_a & sign_b_eff) ? 16'h8000 : 16'h0000;
        end
        else if (is_zero_a) begin
            special_w = 1'b1;
            special_result_w = {sign_b_eff, exp_b, frac_b};
        end
        else if (is_zero_b) begin
            special_w = 1'b1;
            special_result_w = operand_a;
        end
    end

    // Stage 1 / Stage 2 pipeline registers
    reg s1_valid;
    reg s1_eff_sub;
    reg s1_sign_lg;
    reg [7:0] s1_exp_lg;
    reg [7:0] s1_mant_lg;
    reg [7:0] s1_mant_sm;
    reg [7:0] s1_exp_diff;
    reg s1_special;
    reg [15:0] s1_special_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_eff_sub <= 1'b0;
            s1_sign_lg <= 1'b0;
            s1_exp_lg <= 8'b0;
            s1_mant_lg <= 8'b0;
            s1_mant_sm <= 8'b0;
            s1_exp_diff <= 8'b0;
            s1_special <= 1'b0;
            s1_special_result <= 16'b0;
        end else begin
            s1_valid <= valid_in;
            s1_eff_sub <= eff_sub_w;
            s1_sign_lg <= sign_lg_w;
            s1_exp_lg <= exp_lg_w;
            s1_mant_lg <= mant_lg_w;
            s1_mant_sm <= mant_sm_w;
            s1_exp_diff <= exp_diff_w;
            s1_special <= special_w;
            s1_special_result <= special_result_w;
        end
    end

    // =========================================================================
    // Stage 2: Alignment barrel shift + mantissa add/sub
    // =========================================================================

    // Right-shift small mantissa by exp_diff, capture G/R/S
    // {mant_sm[7:0], guard, round, sticky_bits[7:0]} = 18 bits
    wire [17:0] sm_ext = {s1_mant_sm, 10'b0};
    wire [3:0] shift_amt = (s1_exp_diff > 8'd15) ? 4'd15 : s1_exp_diff[3:0];
    wire [17:0] sm_shifted = sm_ext >> shift_amt;

    wire [7:0] mant_sm_al = sm_shifted[17:10];
    wire guard_w = sm_shifted[9];
    wire round_w = sm_shifted[8];
    wire sticky_w = (|sm_shifted[7:0]) | (s1_exp_diff > 8'd15);

    // 10-bit mantissa add/sub (2 extra MSBs for carry)
    wire [9:0] mant_sum_w = s1_eff_sub ? ({2'b0, s1_mant_lg} - {2'b0, mant_sm_al})
                                        : ({2'b0, s1_mant_lg} + {2'b0, mant_sm_al});

    // --- Stage 2 / Stage 3 pipeline registers ---
    reg s2_valid;
    reg s2_sign_lg;
    reg [7:0] s2_exp_lg;
    reg [9:0] s2_mant_sum;
    reg s2_guard;
    reg s2_round;
    reg s2_sticky;
    reg s2_special;
    reg [15:0] s2_special_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            s2_sign_lg <= 1'b0;
            s2_exp_lg <= 8'b0;
            s2_mant_sum <= 10'b0;
            s2_guard <= 1'b0;
            s2_round <= 1'b0;
            s2_sticky <= 1'b0;
            s2_special <= 1'b0;
            s2_special_result <= 16'b0;
        end else begin
            s2_valid <= s1_valid;
            s2_sign_lg <= s1_sign_lg;
            s2_exp_lg <= s1_exp_lg;
            s2_mant_sum <= mant_sum_w;
            s2_guard <= guard_w;
            s2_round <= round_w;
            s2_sticky <= sticky_w;
            s2_special <= s1_special;
            s2_special_result <= s1_special_result;
        end
    end

    // =========================================================================
    // Stage 3: Leading-zero count, normalization shift, exponent adjust
    // =========================================================================

    // Priority encoder: find leading one
    reg [3:0] lead_pos;
    always @(*) begin
        casez (s2_mant_sum)
            10'b1?_????_????: lead_pos = 4'd9;
            10'b01_????_????: lead_pos = 4'd8;
            10'b00_1???_????: lead_pos = 4'd7;
            10'b00_01??_????: lead_pos = 4'd6;
            10'b00_001?_????: lead_pos = 4'd5;
            10'b00_0001_????: lead_pos = 4'd4;
            10'b00_0000_1???: lead_pos = 4'd3;
            10'b00_0000_01??: lead_pos = 4'd2;
            10'b00_0000_001?: lead_pos = 4'd1;
            10'b00_0000_0001: lead_pos = 4'd0;
            default:          lead_pos = 4'd0;
        endcase
    end

    wire sum_zero = (s2_mant_sum == 10'd0);

    // Shifts to place leading 1 at bit position 7
    wire [3:0] lshift_need = (lead_pos < 4'd7) ? (4'd7 - lead_pos) : 4'd0;
    wire [3:0] rshift_need = (lead_pos > 4'd7) ? (lead_pos - 4'd7) : 4'd0;

    // Cap left shift so exponent stays >= 1
    wire [7:0] max_lshift = (s2_exp_lg > 8'd1) ? (s2_exp_lg - 8'd1) : 8'd0;
    wire [3:0] act_lshift = (lshift_need > max_lshift[3:0]) ? max_lshift[3:0] : lshift_need;

    // Apply normalization shift
    wire [9:0] mant_norm_w = (rshift_need != 4'd0) ? (s2_mant_sum >> rshift_need) :
                             (act_lshift  != 4'd0) ? (s2_mant_sum << act_lshift)  :
                             s2_mant_sum;

    // Adjusted exponent
    wire [8:0] exp_norm_w = {1'b0, s2_exp_lg} + {5'b0, rshift_need} - {5'b0, act_lshift};

    // Guard/round/sticky after normalization
    // Right shift pushes mantissa bits into G/R/S positions
    wire g_w = (rshift_need != 4'd0) ? s2_mant_sum[rshift_need - 4'd1] : s2_guard;
    wire r_w = (rshift_need >= 4'd2) ? s2_mant_sum[0] :
               (rshift_need == 4'd1) ? s2_guard        : s2_round;
    wire s_w = (rshift_need != 4'd0) ? (s2_guard | s2_round | s2_sticky) : s2_sticky;

    // --- Stage 3 / Stage 4 pipeline registers ---
    reg s3_valid;
    reg s3_sign_lg;
    reg [7:0] s3_mant_norm;
    reg [8:0] s3_exp_norm;
    reg s3_guard;
    reg s3_round;
    reg s3_sticky;
    reg s3_sum_zero;
    reg s3_special;
    reg [15:0] s3_special_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid <= 1'b0;
            s3_sign_lg <= 1'b0;
            s3_mant_norm <= 8'b0;
            s3_exp_norm <= 9'b0;
            s3_guard <= 1'b0;
            s3_round <= 1'b0;
            s3_sticky <= 1'b0;
            s3_sum_zero <= 1'b0;
            s3_special <= 1'b0;
            s3_special_result <= 16'b0;
        end else begin
            s3_valid <= s2_valid;
            s3_sign_lg <= s2_sign_lg;
            s3_mant_norm <= mant_norm_w[7:0];
            s3_exp_norm <= exp_norm_w;
            s3_guard <= g_w;
            s3_round <= r_w;
            s3_sticky <= s_w;
            s3_sum_zero <= sum_zero;
            s3_special <= s2_special;
            s3_special_result <= s2_special_result;
        end
    end

    // =========================================================================
    // Stage 4: Rounding, special case mux, result assembly
    // =========================================================================

    // Round to nearest, ties to even
    wire round_up = s3_guard & (s3_round | s3_sticky | s3_mant_norm[0]);
    wire [8:0] mant_rounded = {1'b0, s3_mant_norm} + {8'b0, round_up};

    // Rounding overflow: 1.1111111 + 1 -> 10.0000000
    wire round_ovf = mant_rounded[8];
    wire [6:0] frac_final = round_ovf ? mant_rounded[7:1] : mant_rounded[6:0];
    wire [8:0] exp_final  = round_ovf ? (s3_exp_norm + 9'd1) : s3_exp_norm;

    // Final result mux
    reg [15:0] result_w;
    always @(*) begin
        if (s3_special) begin
            result_w = s3_special_result;
        end
        else if (s3_sum_zero) begin
            result_w = 16'h0000;
        end
        else if (exp_final >= 9'd255) begin
            result_w = {s3_sign_lg, 8'hFF, 7'h00}; // overflow -> inf
        end
        else if (exp_final == 9'd0 || exp_final[8]) begin
            result_w = {s3_sign_lg, 15'b0};         // underflow -> zero
        end
        else begin
            result_w = {s3_sign_lg, exp_final[7:0], frac_final};
        end
    end

    // Output register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 16'b0;
            valid_out <= 1'b0;
        end else begin
            result <= result_w;
            valid_out <= s3_valid;
        end
    end

endmodule

`endif // PPLBF16ADDSUB_V