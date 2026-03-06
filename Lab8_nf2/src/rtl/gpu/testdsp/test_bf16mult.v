/* file: test_bf16mult.v
 Description: Behavioral model of Xilinx bf16mult IP for iverilog simulation.
 Matches IP interface: clk, a[15:0], b[15:0], result[15:0]
 Latency: 1 clock cycle
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 */

module test_bf16mult (
    clk, a, b, result
);
    input clk;
    input [15:0] a;
    input [15:0] b;
    output [15:0] result;

    // ============================================================
    // Combinational: BF16 multiply logic
    // ============================================================
    wire sign_a = a[15];
    wire [7:0] exp_a = a[14:7];
    wire [6:0] frac_a = a[6:0];
    wire sign_b = b[15];
    wire [7:0] exp_b = b[14:7];
    wire [6:0] frac_b = b[6:0];

    wire sign_r = sign_a ^ sign_b;

    // Special detection
    wire is_zero_a = (exp_a == 8'h00) && (frac_a == 7'b0);
    wire is_zero_b = (exp_b == 8'h00) && (frac_b == 7'b0);
    wire is_inf_a = (exp_a == 8'hFF) && (frac_a == 7'b0);
    wire is_inf_b = (exp_b == 8'hFF) && (frac_b == 7'b0);
    wire is_nan_a = (exp_a == 8'hFF) && (frac_a != 7'b0);
    wire is_nan_b = (exp_b == 8'hFF) && (frac_b != 7'b0);
    wire is_denorm_a = (exp_a == 8'h00);
    wire is_denorm_b = (exp_b == 8'h00);

    // Implicit leading bit
    wire [7:0] mant_a = is_denorm_a ? {1'b0, frac_a} : {1'b1, frac_a};
    wire [7:0] mant_b = is_denorm_b ? {1'b0, frac_b} : {1'b1, frac_b};

    // 8x8 mantissa product
    wire [15:0] mant_prod = mant_a * mant_b;

    // Exponent: exp_a + exp_b - 127
    wire [9:0] exp_sum = {2'b0, exp_a} + {2'b0, exp_b} - 10'd127;

    // Normalize: check if product overflows into bit [15]
    wire prod_ovf = mant_prod[15];
    wire [7:0] mant_norm = prod_ovf ? mant_prod[15:8] : mant_prod[14:7];
    wire [9:0] exp_norm = prod_ovf ? (exp_sum + 10'd1) : exp_sum;

    // Rounding bits
    wire guard = prod_ovf ? mant_prod[7] : mant_prod[6];
    wire round_bit = prod_ovf ? mant_prod[6] : mant_prod[5];
    wire sticky = prod_ovf ? (|mant_prod[5:0]) : (|mant_prod[4:0]);

    // Round-to-nearest-even
    wire round_up = guard & (round_bit | sticky | mant_norm[0]);
    wire [8:0] mant_rounded = {1'b0, mant_norm} + {8'b0, round_up};
    wire round_ovf = mant_rounded[8];
    wire [6:0] frac_final = round_ovf ? mant_rounded[7:1] : mant_rounded[6:0];
    wire [9:0] exp_final = round_ovf ? (exp_norm + 10'd1) : exp_norm;

    // Final result mux
    reg [15:0] result_comb;
    always @(*) begin
        if (is_nan_a || is_nan_b)
            result_comb = {1'b0, 8'hFF, 7'h40};
        else if (is_inf_a || is_inf_b) begin
            if (is_zero_a || is_zero_b)
                result_comb = {1'b0, 8'hFF, 7'h40}; // inf * 0 = NaN
            else
                result_comb = {sign_r, 8'hFF, 7'h00};
        end else if (is_zero_a || is_zero_b)
            result_comb = {sign_r, 15'b0};
        else if (exp_final >= 10'd255)
            result_comb = {sign_r, 8'hFF, 7'h00};
        else if (exp_final == 10'd0 || exp_final[9])
            result_comb = {sign_r, 15'b0};
        else
            result_comb = {sign_r, exp_final[7:0], frac_final};
    end

    // ============================================================
    // 1-stage pipeline register (match IP latency = 1)
    // ============================================================
    reg [15:0] pipe_d1;

    always @(posedge clk) begin
        pipe_d1 <= result_comb;
    end

    assign result = pipe_d1;

endmodule