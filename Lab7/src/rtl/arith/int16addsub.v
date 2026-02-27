/* file: int16addsub.v
 Description: This file implements the addition and subtraction operations for 16-bit integers using the Han-Carlson Adder (HCA).
 We don't take care of carry-out
 Author: Jeremy Cai
 Date: Feb. 25, 2026
 Version: 1.0
 Revision history:
    - Feb. 25, 2026: Initial implementation of addition and subtraction for 16-bit integers using the Han-Carlson Adder (HCA).
 */

`ifndef INT16ADDSUB_V
`define INT16ADDSUB_V

module int16addsub (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire sub,
    output wire [15:0] result
);

    wire [15:0] b_mod = sub ? ~b : b; // If subtraction, take two's complement of b
    wire cout; // Carry out from the HCA, not used in this design. Keep floating.

    int16hca u_int16hca (
        .a(a),
        .b(b_mod),
        .cin(sub), // For subtraction, we need to add 1 (two's complement), so cin is set to sub
        .sum(result),
        .cout(cout)
    );

endmodule

`endif // INT16ADDSUB_V