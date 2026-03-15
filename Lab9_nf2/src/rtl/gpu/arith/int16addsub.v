/* file: int16addsub.v
 Description: This file implements the addition and subtraction operations for 16-bit integers using the Han-Carlson Adder (HCA).
 We don't take care of carry-out
 Author: Jeremy Cai
 Date: Mar. 6, 2026
 Version: 1.1
 Revision history:
    - Feb. 25, 2026: Initial implementation of addition and subtraction for 16-bit integers using the Han-Carlson Adder (HCA).
    - Mar. 6, 2026: Updated to replace HCA with operator inference for better performance and resource utilization.
 */

`ifndef INT16ADDSUB_V
`define INT16ADDSUB_V

`include "gpu_define.v"

module int16addsub (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire sub,
    output wire [15:0] result
);

    wire [15:0] b_mod = sub ? ~b : b;
    assign result = a + b_mod + {15'd0, sub};

endmodule

`endif // INT16ADDSUB_V