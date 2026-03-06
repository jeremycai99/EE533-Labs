/* file: pplbfintcvt.v
 Description: This file implements the BF16 to INT16 and INT16 to BF16 conversion unit for the GPU design
 This coversion unit is a pipelined design to make sure no timing violations in fpga implementation.
 Author: Jeremy Cai
 Date: Feb. 27, 2026
 Version: 1.0
 Revision history:
    - Feb. 27, 2026: Initial implementation of the BF16 to INT16 and INT16 to BF16 conversion unit.
*/

`ifndef PPLBFINTCVT_V
`define PPLBFINTCVT_V

`include "gpu_define.v"

module pplbfintcvt (
    input wire clk,
    input wire rst_n,
    input wire dt,         // DT_INT16=int16→bf16, DT_BF16=bf16→int16
    input wire [15:0] in,         // Input operand
    output reg [15:0] out         // Result (valid 2 cycles after input)
);

    // Stage 1 — combinational

    // INT16 → BF16 path (DT=0): conditional negate + priority encoder
    wire        i2f_sign = in[15];
    wire [15:0] i2f_mag  = i2f_sign ? (~in + 16'd1) : in;
    wire        i2f_zero = (in == 16'd0);

    // Priority encoder: find position of leading one in magnitude
    reg [3:0] i2f_lzc;
    always @(*) begin
        casez (i2f_mag)
            16'b1???????????????: i2f_lzc = 4'd15;
            16'b01??????????????: i2f_lzc = 4'd14;
            16'b001?????????????: i2f_lzc = 4'd13;
            16'b0001????????????: i2f_lzc = 4'd12;
            16'b00001???????????: i2f_lzc = 4'd11;
            16'b000001??????????: i2f_lzc = 4'd10;
            16'b0000001?????????: i2f_lzc = 4'd9;
            16'b00000001????????: i2f_lzc = 4'd8;
            16'b000000001???????: i2f_lzc = 4'd7;
            16'b0000000001??????: i2f_lzc = 4'd6;
            16'b00000000001?????: i2f_lzc = 4'd5;
            16'b000000000001????: i2f_lzc = 4'd4;
            16'b0000000000001???: i2f_lzc = 4'd3;
            16'b00000000000001??: i2f_lzc = 4'd2;
            16'b000000000000001?: i2f_lzc = 4'd1;
            16'b0000000000000001: i2f_lzc = 4'd0;
            default:              i2f_lzc = 4'd0;  // zero case (guarded by i2f_zero)
        endcase
    end

    // BF16 → INT16 path (DT=1): extract fields + barrel shift

    wire f2i_sign = in[15];
    wire [7:0] f2i_exp = in[14:7];
    wire [6:0] f2i_man = in[6:0];

    wire f2i_is_zero = (f2i_exp == 8'd0); // zero + denormals
    wire f2i_is_special = (f2i_exp == 8'hFF);  // inf or NaN
    wire f2i_is_nan = f2i_is_special & (f2i_man != 7'd0);
    wire f2i_is_inf = f2i_is_special & (f2i_man == 7'd0);
    wire f2i_underflow = (f2i_exp < 8'd127) & ~f2i_is_zero; // 0 < |val| < 1

    // Barrel shift: place {1, mantissa} at correct integer position
    // full_man = 1.man × 2^7 in fixed-point = {1, man, 8'b0}
    // Integer value = full_man >> (142 - exp) where 142 = 127 + 15
    // Valid shift range: 0-15. exp > 142 → overflow, exp < 127 → underflow.
    wire [15:0] f2i_full = {1'b1, f2i_man, 8'b0};
    wire [7:0] f2i_rshift_raw = 8'd142 - f2i_exp;
    wire [3:0] f2i_rshift = f2i_rshift_raw[3:0];
    wire f2i_overflow = (~f2i_is_special) & (f2i_exp > 8'd142);
    wire [15:0] f2i_shifted = f2i_full >> f2i_rshift;

    // Pipeline register (stage 1 → stage 2)
    reg dt_r;
    // INT16 → BF16 registered fields
    reg i2f_sign_r;
    reg [15:0] i2f_mag_r;
    reg [3:0] i2f_lzc_r;
    reg i2f_zero_r;
    // BF16 → INT16 registered fields
    reg f2i_sign_r;
    reg [15:0] f2i_shifted_r;
    reg f2i_is_zero_r;
    reg f2i_is_nan_r;
    reg f2i_is_inf_r;
    reg f2i_underflow_r;
    reg f2i_overflow_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dt_r <= 1'b0;
            i2f_sign_r <= 1'b0;
            i2f_mag_r <= 16'd0;
            i2f_lzc_r <= 4'd0;
            i2f_zero_r <= 1'b1;
            f2i_sign_r <= 1'b0;
            f2i_shifted_r <= 16'd0;
            f2i_is_zero_r <= 1'b1;
            f2i_is_nan_r <= 1'b0;
            f2i_is_inf_r <= 1'b0;
            f2i_underflow_r <= 1'b0;
            f2i_overflow_r <= 1'b0;
        end else begin
            dt_r <= dt;
            // I2F path
            i2f_sign_r <= i2f_sign;
            i2f_mag_r <= i2f_mag;
            i2f_lzc_r <= i2f_lzc;
            i2f_zero_r <= i2f_zero;
            // F2I path
            f2i_sign_r <= f2i_sign;
            f2i_shifted_r <= f2i_shifted;
            f2i_is_zero_r <= f2i_is_zero;
            f2i_is_nan_r <= f2i_is_nan;
            f2i_is_inf_r <= f2i_is_inf;
            f2i_underflow_r <= f2i_underflow;
            f2i_overflow_r <= f2i_overflow;
        end
    end

    // Stage 2 — combinational → registered output

    // INT16 → BF16 (stage 2): normalize + exponent + assemble
    wire [3:0] s2_nshift = 4'd15 - i2f_lzc_r;
    wire [15:0] s2_norm = i2f_mag_r << s2_nshift;
    wire [7:0] s2_exp = 8'd127 + {4'd0, i2f_lzc_r};
    wire [6:0] s2_man = s2_norm[14:8];
    wire [15:0] s2_i2f = i2f_zero_r ? 16'h0000
                                        : {i2f_sign_r, s2_exp, s2_man};

    // BF16 → INT16 (stage 2): conditional negate + saturation
    wire s2_pos_ovf = ~f2i_sign_r & (f2i_shifted_r > 16'd32767);
    wire s2_neg_ovf =  f2i_sign_r & (f2i_shifted_r > 16'd32768);
    wire [15:0] s2_negated = ~f2i_shifted_r + 16'd1;

    reg [15:0] s2_f2i;
    always @(*) begin
        if (f2i_is_zero_r | f2i_underflow_r)
            s2_f2i = 16'd0;
        else if (f2i_is_nan_r)
            s2_f2i = 16'd0;                        // NaN → 0
        else if (f2i_is_inf_r & ~f2i_sign_r)
            s2_f2i = 16'h7FFF;                     // +Inf → +32767
        else if (f2i_is_inf_r & f2i_sign_r)
            s2_f2i = 16'h8000;                     // -Inf → -32768
        else if (f2i_overflow_r & ~f2i_sign_r)
            s2_f2i = 16'h7FFF;                     // positive overflow → +32767
        else if (f2i_overflow_r & f2i_sign_r)
            s2_f2i = 16'h8000;                     // negative overflow → -32768
        else if (s2_pos_ovf)
            s2_f2i = 16'h7FFF;                     // magnitude > 32767 positive
        else if (s2_neg_ovf)
            s2_f2i = 16'h8000;                     // magnitude > 32768 negative
        else if (f2i_sign_r)
            s2_f2i = s2_negated;                   // normal negative
        else
            s2_f2i = f2i_shifted_r;                // normal positive
    end

    // Output register: mux by dt_r
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            out <= 16'd0;
        else
            out <= dt_r ? s2_f2i : s2_i2f;
    end

endmodule

`endif // PPLBFINTCVT_V