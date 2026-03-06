/* file: test_bf16addsub.v
 Description: Behavioral model of Xilinx bf16addsub IP for iverilog simulation.
 Matches IP interface: clk, operation[5:0], a[15:0], b[15:0], result[15:0]
 Operation encoding: 000000=Add, 000001=Subtract
 Latency: 2 clock cycles
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 */

module test_bf16addsub (
    clk, operation, a, b, result
);
    input clk;
    input [5:0] operation;
    input [15:0] a;
    input [15:0] b;
    output [15:0] result;

    // ============================================================
    // Combinational: BF16 add/sub logic
    // ============================================================
    wire sub = operation[0];

    wire sign_a = a[15];
    wire [7:0] exp_a = a[14:7];
    wire [6:0] frac_a = a[6:0];
    wire sign_b = b[15];
    wire [7:0] exp_b = b[14:7];
    wire [6:0] frac_b = b[6:0];
    wire sign_b_eff = sign_b ^ sub;

    // Special detection
    wire is_zero_a = (exp_a == 8'h00) && (frac_a == 7'b0);
    wire is_zero_b = (exp_b == 8'h00) && (frac_b == 7'b0);
    wire is_inf_a = (exp_a == 8'hFF) && (frac_a == 7'b0);
    wire is_inf_b = (exp_b == 8'hFF) && (frac_b == 7'b0);
    wire is_nan_a = (exp_a == 8'hFF) && (frac_a != 7'b0);
    wire is_nan_b = (exp_b == 8'hFF) && (frac_b != 7'b0);

    // Implicit leading bit
    wire [7:0] mant_a = (exp_a != 8'h00) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [7:0] mant_b = (exp_b != 8'h00) ? {1'b1, frac_b} : {1'b0, frac_b};

    wire eff_sub = sign_a ^ sign_b_eff;
    wire a_lt_b = (exp_a < exp_b) || ((exp_a == exp_b) && (mant_a < mant_b));
    wire sign_lg = a_lt_b ? sign_b_eff : sign_a;
    wire [7:0] exp_lg = a_lt_b ? exp_b : exp_a;
    wire [7:0] mant_lg = a_lt_b ? mant_b : mant_a;
    wire [7:0] mant_sm = a_lt_b ? mant_a : mant_b;
    wire [7:0] exp_diff = (a_lt_b ? exp_b : exp_a) - (a_lt_b ? exp_a : exp_b);

    // Special-case result
    reg special;
    reg [15:0] special_result;
    always @(*) begin
        special = 1'b0;
        special_result = 16'h0000;
        if (is_nan_a || is_nan_b) begin
            special = 1'b1;
            special_result = {1'b0, 8'hFF, 7'h40};
        end else if (is_inf_a && is_inf_b) begin
            special = 1'b1;
            special_result = eff_sub ? {1'b0, 8'hFF, 7'h40}
                                     : {sign_a, 8'hFF, 7'h00};
        end else if (is_inf_a) begin
            special = 1'b1;
            special_result = {sign_a, 8'hFF, 7'h00};
        end else if (is_inf_b) begin
            special = 1'b1;
            special_result = {sign_b_eff, 8'hFF, 7'h00};
        end else if (is_zero_a && is_zero_b) begin
            special = 1'b1;
            special_result = (sign_a & sign_b_eff) ? 16'h8000 : 16'h0000;
        end else if (is_zero_a) begin
            special = 1'b1;
            special_result = {sign_b_eff, exp_b, frac_b};
        end else if (is_zero_b) begin
            special = 1'b1;
            special_result = a;
        end
    end

    // Mantissa alignment
    wire [17:0] sm_ext = {mant_sm, 10'b0};
    wire [3:0] shift_amt = (exp_diff > 8'd15) ? 4'd15 : exp_diff[3:0];
    wire [17:0] sm_shifted = sm_ext >> shift_amt;
    wire [7:0] mant_sm_al = sm_shifted[17:10];
    wire guard = sm_shifted[9];
    wire round_bit = sm_shifted[8];
    wire sticky = (|sm_shifted[7:0]) | (exp_diff > 8'd15);

    // Add / subtract
    wire [9:0] mant_sum = eff_sub ? ({2'b0, mant_lg} - {2'b0, mant_sm_al})
                                  : ({2'b0, mant_lg} + {2'b0, mant_sm_al});

    // Leading-one detection
    reg [3:0] lead_pos;
    always @(*) begin
        casez (mant_sum)
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

    wire sum_zero = (mant_sum == 10'd0);
    wire [3:0] lshift_need = (lead_pos < 4'd7) ? (4'd7 - lead_pos) : 4'd0;
    wire [3:0] rshift_need = (lead_pos > 4'd7) ? (lead_pos - 4'd7) : 4'd0;
    wire [7:0] max_lshift = (exp_lg > 8'd1) ? (exp_lg - 8'd1) : 8'd0;
    wire [3:0] act_lshift = ({4'b0, lshift_need} > max_lshift) ? max_lshift[3:0] : lshift_need;

    wire [9:0] mant_norm = (rshift_need != 4'd0) ? (mant_sum >> rshift_need) :
                           (act_lshift  != 4'd0) ? (mant_sum << act_lshift)  :
                           mant_sum;
    wire [8:0] exp_norm = {1'b0, exp_lg} + {5'b0, rshift_need} - {5'b0, act_lshift};

    // Rounding bits after normalization
    wire g_n = (rshift_need != 4'd0) ? mant_sum[rshift_need - 4'd1] : guard;
    wire r_n = (rshift_need >= 4'd2) ? mant_sum[0] :
               (rshift_need == 4'd1) ? guard       : round_bit;
    wire s_n = (rshift_need != 4'd0) ? (guard | round_bit | sticky) : sticky;

    // Round-to-nearest-even
    wire round_up = g_n & (r_n | s_n | mant_norm[0]);
    wire [8:0] mant_rounded = {1'b0, mant_norm[7:0]} + {8'b0, round_up};
    wire round_ovf = mant_rounded[8];
    wire [6:0] frac_final = round_ovf ? mant_rounded[7:1] : mant_rounded[6:0];
    wire [8:0] exp_final = round_ovf ? (exp_norm + 9'd1) : exp_norm;

    // Final result mux
    reg [15:0] result_comb;
    always @(*) begin
        if (special)
            result_comb = special_result;
        else if (sum_zero)
            result_comb = 16'h0000;
        else if (exp_final >= 9'd255)
            result_comb = {sign_lg, 8'hFF, 7'h00};
        else if (exp_final == 9'd0 || exp_final[8])
            result_comb = {sign_lg, 15'b0};
        else
            result_comb = {sign_lg, exp_final[7:0], frac_final};
    end

    // ============================================================
    // 2-stage pipeline registers (match IP latency = 2)
    // ============================================================
    reg [15:0] pipe_d1;
    reg [15:0] pipe_d2;

    always @(posedge clk) begin
        pipe_d1 <= result_comb;
        pipe_d2 <= pipe_d1;
    end

    assign result = pipe_d2;

endmodule