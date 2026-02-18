/* file: cond_eval.v
 Description: Condition evaluation module for the Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 17, 2026
 Version: 1.0
 */

`ifndef COND_EVAL_V
`define COND_EVAL_V

`include "define.v"

module cond_eval (
    input wire [COND_WIDTH-1:0] cond_code, // Condition code from instruction
    input wire [3:0] flags, // Current CPU flags: [N, Z, C, V]
    output wire cond_met // Output: 1 if condition is met, 0 otherwise
);

wire n = flags[`FLAG_N]; // Negative flag
wire z = flags[`FLAG_Z]; // Zero flag
wire c = flags[`FLAG_C]; // Carry flag
wire v = flags[`FLAG_V]; // Overflow flag

always @(*) begin
    case(cond_code)
        `COND_EQ: cond_met = z; // Equal (Z set)
        `COND_NE: cond_met = ~z; // Not equal (Z clear)
        `COND_CS: cond_met = c; // Carry set (unsigned higher or same)
        `COND_CC: cond_met = ~c; // Carry clear (unsigned lower)
        `COND_MI: cond_met = n; // Minus/negative (N set)
        `COND_PL: cond_met = ~n; // Plus/positive or zero (N clear)
        `COND_VS: cond_met = v; // Overflow set
        `COND_VC: cond_met = ~v; // Overflow clear
        `COND_HI: cond_met = c & ~z; // Unsigned higher (C set and Z clear)
        `COND_LS: cond_met = ~c | z; // Unsigned lower or same (C clear or Z set)
        `COND_GE: cond_met = (n == v); // Signed greater than or equal (N equals V)
        `COND_LT: cond_met = (n != v); // Signed less than (N not equal to V)
        `COND_GT: cond_met = ~z & (n == v); // Signed greater than (Z clear and N equals V)
        `COND_LE: cond_met = z | (n != v); // Signed less than or equal (Z set or N not equal to V)
        `COND_AL: cond_met = 1'b1; // Always
        default:  cond_met = 1'b0; // Undefined condition codes treated as not met
    endcase
end
endmodule

`endif // COND_EVAL_V