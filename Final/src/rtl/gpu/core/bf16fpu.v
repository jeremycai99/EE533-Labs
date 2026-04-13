/* file: bf16fpu.v
 Description: This file implements the BF16 floating-point unit (FPU) for addition, subtraction, multiplication, and comparison operations.
 The FPU is designed to support only BF16 data type and due to complexity of pipeline arithmetic units, we implement a FSM inside the FPU
    to control the multi-cycle execution of these operations.
 Author: Jeremy Cai
 Date: Feb. 27, 2026
 Version: 1.0
 Revision history:
    - Feb. 27, 2026: Initial implementation of the pipelined BF16 FPU for addition, subtraction, multiplication, and comparison operations.
    - Feb. 28, 2026: Fix bug for buzy signals for multi-cycle operations.
*/

`ifndef BF16FPU_V
`define BF16FPU_V

`include "gpu_define.v"
`include "pplbf16mult.v"
`include "pplbf16addsub.v"
`include "bf16comp.v"

module bf16fpu (
    input wire clk,
    input wire rst_n,
    input wire [4:0] alu_op,
    input wire valid_in,
    input wire [1:0] cmp_mode,
    input wire pred_val,
    input wire [15:0] op_a,
    input wire [15:0] op_b,
    input wire [15:0] op_c,
    output wire [15:0] result,
    output wire valid_out,
    output wire busy,
    output wire cmp_eq,
    output wire cmp_ne,
    output wire cmp_lt,
    output wire cmp_le
);

    wire is_mult_op = (alu_op == `OP_MUL) | (alu_op == `OP_MULI);
    wire is_add_op  = (alu_op == `OP_ADD) | (alu_op == `OP_SUB) |
                      (alu_op == `OP_ADDI);
    wire is_fma_op  = (alu_op == `OP_FMA);
    wire is_multi_cycle = is_mult_op | is_add_op | is_fma_op;
    wire is_1cyc_op = ~is_multi_cycle;

    localparam [2:0] S_IDLE  = 3'd0,
                     S_MULT  = 3'd1,
                     S_ADD   = 3'd2,
                     S_FMA_M = 3'd3,
                     S_FMA_A = 3'd4;

    reg [2:0] state;
    reg [15:0] op_c_reg;

    wire mult_valid_out;
    wire add_valid_out;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            op_c_reg <= 16'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (valid_in & is_mult_op)
                        state <= S_MULT;
                    else if (valid_in & is_add_op)
                        state <= S_ADD;
                    else if (valid_in & is_fma_op) begin
                        state    <= S_FMA_M;
                        op_c_reg <= op_c;
                    end
                end
                S_MULT:  if (mult_valid_out) state <= S_IDLE;
                S_ADD:   if (add_valid_out)  state <= S_IDLE;
                S_FMA_M: if (mult_valid_out) state <= S_FMA_A;
                S_FMA_A: if (add_valid_out)  state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

    // Busy: include combinational term for multi-cycle ops so busy goes
    // high the SAME cycle the op enters EX (prevents flush_id gap).
    assign busy = (state != S_IDLE) | (valid_in & is_multi_cycle);

    wire mult_feed = valid_in & (state == S_IDLE) & (is_mult_op | is_fma_op);
    wire [15:0] mult_result;

    pplbf16mult u_mult (
        .clk(clk), .rst_n(rst_n),
        .operand_a(op_a), .operand_b(op_b),
        .valid_in(mult_feed),
        .result(mult_result), .valid_out(mult_valid_out)
    );

    wire fma_inject = (state == S_FMA_M) & mult_valid_out;
    wire add_feed = (valid_in & (state == S_IDLE) & is_add_op) | fma_inject;

    wire [15:0] addsub_a = fma_inject ? mult_result : op_a;
    wire [15:0] addsub_b = fma_inject ? op_c_reg : op_b;
    wire addsub_sub = ~fma_inject & (alu_op == `OP_SUB);

    wire [15:0] add_result;

    pplbf16addsub u_addsub (
        .clk(clk), .rst_n(rst_n),
        .operand_a(addsub_a), .operand_b(addsub_b),
        .sub(addsub_sub), .valid_in(add_feed),
        .result(add_result), .valid_out(add_valid_out)
    );

    bf16comp u_comp (
        .a(op_a), .b(op_b),
        .eq(cmp_eq), .ne(cmp_ne), .lt(cmp_lt), .le(cmp_le)
    );

    wire set_val = (cmp_mode == `COMP_EQ) ? cmp_eq :
                   (cmp_mode == `COMP_NE) ? cmp_ne :
                   (cmp_mode == `COMP_LT) ? cmp_lt : cmp_le;

    reg [15:0] result_1cyc;
    always @(*) begin
        result_1cyc = 16'd0;
        case (alu_op)
            `OP_ABS: result_1cyc = {1'b0, op_a[14:0]};
            `OP_NEG: result_1cyc = {~op_a[15], op_a[14:0]};
            `OP_MAX: result_1cyc = cmp_lt ? op_b : op_a;
            `OP_MIN: result_1cyc = cmp_lt ? op_a : op_b;
            `OP_SELP: result_1cyc = pred_val ? op_a : op_b;
            `OP_SET,
            `OP_SETP: result_1cyc = {15'd0, set_val};
            default: result_1cyc = 16'd0;
        endcase
    end

    wire valid_1cyc = (state == S_IDLE) & valid_in & is_1cyc_op;
    wire valid_mult = (state == S_MULT) & mult_valid_out;
    wire valid_add  = (state == S_ADD)  & add_valid_out;
    wire valid_fma  = (state == S_FMA_A) & add_valid_out;

    assign valid_out = valid_1cyc | valid_mult | valid_add | valid_fma;

    assign result = valid_mult             ? mult_result  :
                    (valid_add | valid_fma) ? add_result   :
                                             result_1cyc;

endmodule

`endif // BF16FPU_V