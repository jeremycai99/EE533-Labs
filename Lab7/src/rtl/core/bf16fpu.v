/* file: bf16fpu.v
 Description: This file implements the BF16 floating-point unit (FPU) for addition, subtraction, multiplication, and comparison operations.
 The FPU is designed to support only BF16 data type and due to complexity of pipeline arithmetic units, we implement a FSM inside the FPU
    to control the multi-cycle execution of these operations.
 Author: Jeremy Cai
 Date: Feb. 27, 2026
 Version: 1.0
 Revision history:
    - Feb. 27, 2026: Initial implementation of the pipelined BF16 FPU for addition, subtraction, multiplication, and comparison operations.
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
    // Control
    input wire [4:0] alu_op, // ISA opcode (13 valid opcodes)
    input wire valid_in,    // New operation valid
    input wire [1:0] cmp_mode, // COMP_EQ/NE/LT/LE for SET/SETP
    input wire pred_val, // Predicate value for SELP
    // Operands
    input wire [15:0] op_a, // rA value
    input wire [15:0] op_b, // rB value or imm16
    input wire [15:0] op_c, // rC for FMA accumulator
    // Outputs
    output wire [15:0] result, // FPU result
    output wire valid_out, // Result valid this cycle
    output wire busy, // Cannot accept new instruction
    // Comparison flags (always computed on op_a vs op_b, BF16 semantics)
    output wire cmp_eq,
    output wire cmp_ne,
    output wire cmp_lt,
    output wire cmp_le
);

    // Opcode classification
    wire is_mult_op = (alu_op == `OP_MUL) | (alu_op == `OP_MULI);
    wire is_add_op  = (alu_op == `OP_ADD) | (alu_op == `OP_SUB) |
                      (alu_op == `OP_ADDI);
    wire is_fma_op  = (alu_op == `OP_FMA);
    wire is_multi_cycle = is_mult_op | is_add_op | is_fma_op;
    wire is_1cyc_op = ~is_multi_cycle;

    // State machine
    //   IDLE:  accept new instructions (1-cycle ops resolve immediately)
    //   MULT:  MUL/MULI in pplbf16mult pipeline (3 cycles)
    //   ADD:   ADD/SUB/ADDI in pplbf16addsub pipeline (4 cycles)
    //   FMA_M: FMA mult phase in pplbf16mult (3 cycles)
    //   FMA_A: FMA add phase in pplbf16addsub (4 cycles)
    localparam [2:0] S_IDLE  = 3'd0,
                     S_MULT  = 3'd1,
                     S_ADD   = 3'd2,
                     S_FMA_M = 3'd3,
                     S_FMA_A = 3'd4;

    reg [2:0] state;

    // Registered FMA accumulator operand (captured at accept)
    reg [15:0] op_c_reg;

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
                    // 1-cycle ops: stay in IDLE
                end
                S_MULT:  if (mult_valid_out) state <= S_IDLE;
                S_ADD:   if (add_valid_out)  state <= S_IDLE;
                S_FMA_M: if (mult_valid_out) state <= S_FMA_A;
                S_FMA_A: if (add_valid_out)  state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

    assign busy = (state != S_IDLE);

    // Multiplier (3-stage pipelined, free-running)
    wire mult_feed = valid_in & ~busy & (is_mult_op | is_fma_op);
    wire [15:0] mult_result;
    wire mult_valid_out;

    pplbf16mult u_mult (
        .clk(clk),
        .rst_n(rst_n),
        .operand_a(op_a),
        .operand_b(op_b),
        .valid_in(mult_feed),
        .result(mult_result),
        .valid_out(mult_valid_out)
    );

    // Addsub (4-stage pipelined, shared between direct ADD/SUB and FMA)
    //   Direct: op_a ± op_b when accepting ADD/SUB/ADDI
    //   FMA:    mult_result + op_c_reg when mult completes during FMA
    //   Mutually exclusive: busy blocks new ADD while FMA in progress.
    wire fma_inject = (state == S_FMA_M) & mult_valid_out;
    wire add_feed = (valid_in & ~busy & is_add_op) | fma_inject;

    wire [15:0] addsub_a = fma_inject ? mult_result : op_a;
    wire [15:0] addsub_b = fma_inject ? op_c_reg : op_b;
    wire addsub_sub = ~fma_inject & (alu_op == `OP_SUB);

    wire [15:0] add_result;
    wire add_valid_out;

    pplbf16addsub u_addsub (
        .clk(clk),
        .rst_n(rst_n),
        .operand_a(addsub_a),
        .operand_b(addsub_b),
        .sub(addsub_sub),
        .valid_in(add_feed),
        .result(add_result),
        .valid_out(add_valid_out)
    );

    // BF16 comparator (combinational, always active)
    //   Inputs stable during multi-cycle stalls (upstream holds operands).
    bf16comp u_comp (
        .a(op_a),
        .b(op_b),
        .eq(cmp_eq),
        .ne(cmp_ne),
        .lt(cmp_lt),
        .le(cmp_le)
    );

    // Compare result mux for SET/SETP
    wire set_val = (cmp_mode == `COMP_EQ) ? cmp_eq :
                   (cmp_mode == `COMP_NE) ? cmp_ne :
                   (cmp_mode == `COMP_LT) ? cmp_lt : cmp_le;

    // 1-cycle result mux (combinational)
    //   BF16-specific: ABS/NEG are sign-bit manipulations only.
    //   MAX/MIN use bf16comp (IEEE sign-magnitude ordering).
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

    // Valid output
    //   1-cycle: valid same cycle as valid_in (while IDLE)
    //   Multi-cycle: valid when pipeline completes (signaled by valid_out
    //   from mult/addsub, gated by state to distinguish MUL vs FMA).
    wire valid_1cyc = (state == S_IDLE) & valid_in & is_1cyc_op;
    wire valid_mult = (state == S_MULT) & mult_valid_out;
    wire valid_add  = (state == S_ADD)  & add_valid_out;
    wire valid_fma  = (state == S_FMA_A) & add_valid_out;

    assign valid_out = valid_1cyc | valid_mult | valid_add | valid_fma;

    // Result output mux
    assign result = valid_mult             ? mult_result  :
                    (valid_add | valid_fma) ? add_result   :
                                             result_1cyc;

endmodule

`endif // BF16FPU_V