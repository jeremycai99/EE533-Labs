/* file: bf16comp.v
 Description: This file implements the comparison operation for BF16 format.
 This module is a generalized form for ReLU and Max/Min operations.
 MIN/MAX/SET instructions can be implemented by using this module to compare two BF16 numbers and select
 Date: Feb. 27, 2026
 Version: 1.0
 Revision history:
    - Feb. 27, 2026: Initial implementation of comparison operation for BF16 format.
*/

`ifndef BF16COMP_V
`define BF16COMP_V

`include "gpu_define.v"

module bf16comp (
    input wire [15:0] a,
    input wire [15:0] b,
    output wire eq, // a == b
    output wire ne, // a != b
    output wire lt, // a <  b
    output wire le  // a <= b
);
    // Sign, exponent, mantissa
    wire a_sign = a[15];
    wire b_sign = b[15];
    wire [7:0] a_exp = a[14:7];
    wire [7:0] b_exp = b[14:7];
    wire [6:0] a_man = a[6:0];
    wire [6:0] b_man = b[6:0];

    // NaN detection: exp = 0xFF and mantissa != 0
    wire a_nan = (&a_exp) & (|a_man);
    wire b_nan = (&b_exp) & (|b_man);
    wire has_nan = a_nan | b_nan;

    // Zero detection: exp = 0 and mantissa = 0 (either sign)
    wire a_zero = ~(|a[14:0]);
    wire b_zero = ~(|b[14:0]);
    wire both_zero = a_zero & b_zero;

    // Magnitude comparison on bits [14:0]
    wire [14:0] a_mag = a[14:0];
    wire [14:0] b_mag = b[14:0];
    wire mag_eq = (a_mag == b_mag);
    wire mag_lt = (a_mag < b_mag);
    wire mag_gt = (a_mag > b_mag);

    // Ordered equality: same bits, or both zero
    wire ord_eq = (a == b) | both_zero;

    // Ordered less-than (assuming no NaN)
    // Case 1: a negative, b positive (neither zero) → a < b
    // Case 2: both positive → smaller magnitude is less
    // Case 3: both negative → larger magnitude is less
    // Case 4: a positive, b negative → a >= b
    wire ord_lt;
    assign ord_lt = both_zero          ? 1'b0 :
                    (a_sign & ~b_sign) ? 1'b1 : // a<0, b>=0
                    (~a_sign & b_sign) ? 1'b0 : // a>=0, b<0
                    (~a_sign)          ? mag_lt : // both positive
                                         mag_gt; // both negative

    // Final outputs with NaN handling
    assign eq = ~has_nan & ord_eq;
    assign ne = has_nan | ~ord_eq;
    assign lt = ~has_nan & ord_lt;
    assign le = ~has_nan & (ord_eq | ord_lt);

endmodule

`endif // BF16COMP_V