/* file: int16alu.v
 Description: This file implements the CUDA-like integer 16-bit ALU core pipeline design.
 This ALU is a sequential design because of the pipeline multiplier design and FMA support.
 Please use the output only when valid_out is high, and check busy before sending new instruction.
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.0
 Revision history:
    - Feb. 26, 2026: Initial implementation of the CUDA-like integer 16-bit ALU core pipeline.
*/

`ifndef INT16ALU_V
`define INT16ALU_V

`include "gpu_define.v"
`include "pplint16mult.v"
`include "int16addsub.v"

module int16alu (
    input wire clk,
    input wire rst_n,
    // Control
    input wire [4:0] alu_op, // ISA opcode (24 valid opcodes)
    input wire valid_in, // New instruction valid
    input wire [1:0] cmp_mode, // COMP_EQ/NE/LT/LE for SET/SETP
    input wire pred_val, // Predicate value for SELP
    // Operands
    input wire [15:0] op_a, // rA value
    input wire [15:0] op_b, // rB value or imm16
    input wire [15:0] op_c, // rC for FMA accumulator
    // Outputs
    output wire [15:0] result, // ALU result
    output wire valid_out, // Result valid this cycle
    output wire busy, // Cannot accept new instruction
    // Comparison flags (for SETP/SET, always computed on op_a vs op_b)
    output wire cmp_eq,
    output wire cmp_ne,
    output wire cmp_lt,
    output wire cmp_le
);

    wire is_mult_op = (alu_op == `OP_MUL) | (alu_op == `OP_MULI) |
                      (alu_op == `OP_FMA);

    // Multiplier pipeline tracking
    //   mult_valid[0]: mult stage 1 in progress (busy)
    //   mult_valid[1]: mult result ready this cycle
    reg [1:0] mult_valid;
    reg is_fma_reg;
    reg [15:0] op_c_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_valid <= 2'b00;
            is_fma_reg <= 1'b0;
            op_c_reg   <= 16'd0;
        end else begin
            mult_valid[0] <= valid_in & is_mult_op;
            mult_valid[1] <= mult_valid[0];
            if (valid_in & is_mult_op) begin
                is_fma_reg <= (alu_op == `OP_FMA);
                op_c_reg   <= op_c;
            end
        end
    end

    assign busy = mult_valid[0];
    wire mult_done = mult_valid[1];

    // Multiplier (2-stage pipeline, free-running)
    wire [15:0] mult_result;

    pplint16mult u_mult (
        .clk(clk),
        .rst_n(rst_n),
        .a(op_a),
        .b(op_b),
        .result(mult_result)
    );

    // Addsub input mux — time-shared:
    //   Normal: 1-cycle ops (ADD/SUB/ADDI/NEG/ABS/addr calc)
    //   FMA:    mult_result + op_c_reg (when mult completes)
    // Mutually exclusive because busy stalls pipeline during mult ops.
    reg [15:0] add_a, add_b;
    reg add_sub;

    always @(*) begin
        add_a = op_a;
        add_b = op_b;
        add_sub = 1'b0;

        if (mult_done & is_fma_reg) begin
            // FMA accumulation: mult_result + op_c
            add_a = mult_result;
            add_b = op_c_reg;
            add_sub = 1'b0;
        end else begin
            case (alu_op)
                `OP_SUB: begin
                    add_a = op_a;
                    add_b = op_b;
                    add_sub = 1'b1;
                end
                `OP_NEG, `OP_ABS: begin
                    add_a = 16'd0;
                    add_b = op_a;
                    add_sub = 1'b1;     // 0 - rA = -rA
                end
                default: begin          // ADD, ADDI, LD, ST, LDS, STS
                    add_a = op_a;
                    add_b = op_b;
                    add_sub = 1'b0;
                end
            endcase
        end
    end

    wire [15:0] addsub_result;

    int16addsub u_addsub (
        .a(add_a),
        .b(add_b),
        .sub(add_sub),
        .result(addsub_result)
    );

    // Signed comparison (always active, combinational)
    wire signed [15:0] signed_a = op_a;
    wire signed [15:0] signed_b = op_b;

    assign cmp_eq = (op_a == op_b);
    assign cmp_ne = (op_a != op_b);
    assign cmp_lt = (signed_a < signed_b);
    assign cmp_le = (signed_a <= signed_b);

    // Compare result mux for SET/SETP
    wire set_val = (cmp_mode == `COMP_EQ) ? cmp_eq :
                   (cmp_mode == `COMP_NE) ? cmp_ne :
                   (cmp_mode == `COMP_LT) ? cmp_lt : cmp_le;

    // Barrel shifter (combinational)
    wire [3:0] shamt = op_b[3:0];
    wire [15:0] shl_result = op_a << shamt;
    wire signed [15:0] shr_result = $signed(op_a) >>> shamt;  // arithmetic

    // Result mux
    reg [15:0] result_comb;

    always @(*) begin
        result_comb = 16'd0;

        if (mult_done) begin
            result_comb = is_fma_reg ? addsub_result : mult_result;
        end else begin
            case (alu_op)
                // Pass-through
                `OP_MOV: result_comb = op_a;
                `OP_MOVI: result_comb = op_b;

                // Addsub path
                `OP_ADD, `OP_ADDI,
                `OP_LD, `OP_ST, `OP_LDS, `OP_STS:  result_comb = addsub_result;
                `OP_SUB: result_comb = addsub_result;
                `OP_NEG: result_comb = addsub_result;
                `OP_ABS: result_comb = op_a[15] ? addsub_result : op_a;

                // Logic
                `OP_AND: result_comb = op_a & op_b;
                `OP_OR: result_comb = op_a | op_b;
                `OP_XOR: result_comb = op_a ^ op_b;

                // Shift
                `OP_SHL: result_comb = shl_result;
                `OP_SHR: result_comb = shr_result;

                // Min/Max (signed)
                `OP_MAX: result_comb = (signed_a >= signed_b) ? op_a : op_b;
                `OP_MIN: result_comb = (signed_a <= signed_b) ? op_a : op_b;

                // Predicate select
                `OP_SELP: result_comb = pred_val ? op_a : op_b;

                // Set from compare
                `OP_SET, `OP_SETP: result_comb = {15'd0, set_val};

                default: result_comb = 16'd0;
            endcase
        end
    end

    assign result = result_comb;

    assign valid_out = (valid_in & ~is_mult_op) | mult_done;

endmodule

`endif // INT16ALU_V