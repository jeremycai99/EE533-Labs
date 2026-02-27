/* file: int16alu_tb.v
 Description: Comprehensive testbench for INT16 ALU.
   Tests all 24 opcodes: arithmetic (1-cycle and 2-cycle),
   logic, shifts, comparisons, MOV/MOVI, SELP, address calc.
   Golden model comparison for directed + random vectors.
   Global cycle counter reported for every test.
   MLA test has two cycles because the final add is combinational.
   May need to revise due to timing
 Author: Jeremy Cai
 Date: Feb. 27, 2026
*/

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "int16alu.v"

module int16alu_tb;
    reg clk, rst_n;
    reg [4:0] alu_op;
    reg valid_in;
    reg [1:0] cmp_mode;
    reg pred_val;
    reg [15:0] op_a, op_b, op_c;
    wire [15:0] result;
    wire valid_out, busy;
    wire cmp_eq, cmp_ne, cmp_lt, cmp_le;

    int16alu dut (
        .clk(clk), .rst_n(rst_n),
        .alu_op(alu_op), .valid_in(valid_in),
        .cmp_mode(cmp_mode), .pred_val(pred_val),
        .op_a(op_a), .op_b(op_b), .op_c(op_c),
        .result(result), .valid_out(valid_out), .busy(busy),
        .cmp_eq(cmp_eq), .cmp_ne(cmp_ne),
        .cmp_lt(cmp_lt), .cmp_le(cmp_le)
    );

    always #5 clk = ~clk;

    // =========================================================================
    // Global cycle counter
    // =========================================================================
    integer cycle;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle <= 0;
        else
            cycle <= cycle + 1;
    end

    integer pass_count, fail_count;
    integer t_start;        // Captured at issue time
    reg verbose;            // 1=print PASS, 0=print FAIL only (for random)

    // =========================================================================
    // 1-cycle test task
    // =========================================================================
    task test_1cyc;
        input [4:0] op;
        input [15:0] a, b, c;
        input [1:0] cmode;
        input pval;
        input [15:0] expected;
        input [8*40-1:0] label;
        begin
            @(posedge clk); #1;
            t_start = cycle;
            alu_op = op; op_a = a; op_b = b; op_c = c;
            cmp_mode = cmode; pred_val = pval;
            valid_in = 1'b1;
            @(posedge clk); #1;
            valid_in = 1'b0;
            if (!valid_out) begin
                $display("[cyc %0d->%0d] FAIL %s: valid_out not asserted",
                    t_start, cycle, label);
                fail_count = fail_count + 1;
            end else if (result !== expected) begin
                $display("[cyc %0d->%0d] FAIL %s: got=%04h exp=%04h (a=%04h b=%04h c=%04h)",
                    t_start, cycle, label, result, expected, a, b, c);
                fail_count = fail_count + 1;
            end else begin
                if (verbose)
                    $display("[cyc %0d->%0d] PASS %s: res=%04h",
                        t_start, cycle, label, result);
                pass_count = pass_count + 1;
            end
        end
    endtask

    // Shorthand: no compare/pred
    task t1;
        input [4:0] op;
        input [15:0] a, b, expected;
        input [8*40-1:0] label;
        begin
            test_1cyc(op, a, b, 16'd0, 2'b00, 1'b0, expected, label);
        end
    endtask

    // =========================================================================
    // 2-cycle test task (waits for valid_out)
    // =========================================================================
    integer t2_wait;
    task test_2cyc;
        input [4:0] op;
        input [15:0] a, b, c;
        input [15:0] expected;
        input [8*40-1:0] label;
        begin
            @(posedge clk); #1;
            t_start = cycle;
            alu_op = op; op_a = a; op_b = b; op_c = c;
            cmp_mode = 2'b00; pred_val = 1'b0;
            valid_in = 1'b1;
            @(posedge clk); #1;
            valid_in = 1'b0;
            t2_wait = 0;
            while (!valid_out) begin
                @(posedge clk); #1;
                t2_wait = t2_wait + 1;
                if (t2_wait > 10) begin
                    $display("[cyc %0d->%0d] TIMEOUT %s", t_start, cycle, label);
                    fail_count = fail_count + 1;
                    disable test_2cyc;
                end
            end
            if (result !== expected) begin
                $display("[cyc %0d->%0d] FAIL %s: got=%04h exp=%04h (a=%04h b=%04h c=%04h)",
                    t_start, cycle, label, result, expected, a, b, c);
                fail_count = fail_count + 1;
            end else begin
                if (verbose)
                    $display("[cyc %0d->%0d] PASS %s: res=%04h",
                        t_start, cycle, label, result);
                pass_count = pass_count + 1;
            end
        end
    endtask

    // =========================================================================
    // Compare flag test task (for SETP)
    // =========================================================================
    task test_cmp;
        input [15:0] a, b;
        input exp_eq, exp_ne, exp_lt, exp_le;
        input [8*40-1:0] label;
        begin
            @(posedge clk); #1;
            t_start = cycle;
            alu_op = `OP_SETP; op_a = a; op_b = b; op_c = 16'd0;
            cmp_mode = 2'b00; pred_val = 1'b0;
            valid_in = 1'b1;
            @(posedge clk); #1;
            valid_in = 1'b0;
            if (cmp_eq !== exp_eq || cmp_ne !== exp_ne ||
                cmp_lt !== exp_lt || cmp_le !== exp_le) begin
                $display("[cyc %0d->%0d] FAIL %s: a=%04h b=%04h eq=%b/%b ne=%b/%b lt=%b/%b le=%b/%b",
                    t_start, cycle, label, a, b,
                    cmp_eq, exp_eq, cmp_ne, exp_ne,
                    cmp_lt, exp_lt, cmp_le, exp_le);
                fail_count = fail_count + 1;
            end else begin
                if (verbose)
                    $display("[cyc %0d->%0d] PASS %s: eq=%b ne=%b lt=%b le=%b",
                        t_start, cycle, label, cmp_eq, cmp_ne, cmp_lt, cmp_le);
                pass_count = pass_count + 1;
            end
        end
    endtask

    // Golden models
    reg [15:0] gold;
    reg signed [15:0] sa, sb;
    reg [31:0] rand_a, rand_b, rand_c;
    integer i;

    initial begin
        $dumpfile("int16alu_tb.vcd");
        $dumpvars(0, int16alu_tb);
        clk = 0; rst_n = 0; valid_in = 0;
        alu_op = 5'd0; op_a = 0; op_b = 0; op_c = 0;
        cmp_mode = 2'b00; pred_val = 0;
        pass_count = 0; fail_count = 0;
        verbose = 1;    // Directed tests: print all

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        $display("=========================================================");
        $display(" INT16 ALU Testbench — 24 Opcodes");
        $display("=========================================================");

        // =================================================================
        // MOV (0x03): rD <- rA
        // =================================================================
        $display("\n--- MOV ---");
        t1(`OP_MOV, 16'h1234, 16'h0000, 16'h1234, "mov_basic");
        t1(`OP_MOV, 16'h0000, 16'hFFFF, 16'h0000, "mov_zero");
        t1(`OP_MOV, 16'hFFFF, 16'h0000, 16'hFFFF, "mov_ffff");
        t1(`OP_MOV, 16'h8000, 16'h0000, 16'h8000, "mov_8000");

        // =================================================================
        // MOVI (0x04): rD <- imm16 (passed as op_b)
        // =================================================================
        $display("\n--- MOVI ---");
        t1(`OP_MOVI, 16'h0000, 16'hABCD, 16'hABCD, "movi_basic");
        t1(`OP_MOVI, 16'h0000, 16'h0000, 16'h0000, "movi_zero");
        t1(`OP_MOVI, 16'h0000, 16'hFFFF, 16'hFFFF, "movi_ffff");

        // =================================================================
        // ADD (0x06): rD <- rA + rB
        // =================================================================
        $display("\n--- ADD ---");
        t1(`OP_ADD, 16'd0,     16'd0,     16'd0,     "add_0+0");
        t1(`OP_ADD, 16'd100,   16'd200,   16'd300,   "add_100+200");
        t1(`OP_ADD, 16'hFFFF,  16'd1,     16'h0000,  "add_wrap");
        t1(`OP_ADD, 16'h7FFF,  16'd1,     16'h8000,  "add_overflow");
        t1(`OP_ADD, 16'h8000,  16'h8000,  16'h0000,  "add_neg+neg");

        // =================================================================
        // SUB (0x07): rD <- rA - rB
        // =================================================================
        $display("\n--- SUB ---");
        t1(`OP_SUB, 16'd300,   16'd100,   16'd200,   "sub_300-100");
        t1(`OP_SUB, 16'd0,     16'd1,     16'hFFFF,  "sub_underflow");
        t1(`OP_SUB, 16'd100,   16'd100,   16'd0,     "sub_equal");
        t1(`OP_SUB, 16'h8000,  16'd1,     16'h7FFF,  "sub_8000-1");

        // =================================================================
        // ADDI (0x13): rD <- rA + imm16
        // =================================================================
        $display("\n--- ADDI ---");
        t1(`OP_ADDI, 16'd50,    16'd50,    16'd100,   "addi_50+50");
        t1(`OP_ADDI, 16'hFFF0,  16'h0020,  16'h0010,  "addi_wrap");

        // =================================================================
        // NEG (0x0D): rD <- -rA
        // =================================================================
        $display("\n--- NEG ---");
        t1(`OP_NEG, 16'd1,     16'd0, 16'hFFFF,  "neg_1");
        t1(`OP_NEG, 16'hFFFF,  16'd0, 16'd1,     "neg_-1");
        t1(`OP_NEG, 16'd0,     16'd0, 16'd0,     "neg_0");
        t1(`OP_NEG, 16'h8000,  16'd0, 16'h8000,  "neg_min");
        t1(`OP_NEG, 16'd100,   16'd0, 16'hFF9C,  "neg_100");

        // =================================================================
        // ABS (0x0C): rD <- |rA|
        // =================================================================
        $display("\n--- ABS ---");
        t1(`OP_ABS, 16'd100,   16'd0, 16'd100,   "abs_pos");
        t1(`OP_ABS, 16'hFF9C,  16'd0, 16'd100,   "abs_-100");
        t1(`OP_ABS, 16'd0,     16'd0, 16'd0,     "abs_0");
        t1(`OP_ABS, 16'hFFFF,  16'd0, 16'd1,     "abs_-1");
        t1(`OP_ABS, 16'h8000,  16'd0, 16'h8000,  "abs_min");

        // =================================================================
        // AND (0x0E)
        // =================================================================
        $display("\n--- AND ---");
        t1(`OP_AND, 16'hFF00, 16'h0FF0, 16'h0F00, "and_basic");
        t1(`OP_AND, 16'hFFFF, 16'hFFFF, 16'hFFFF, "and_all1");
        t1(`OP_AND, 16'hAAAA, 16'h5555, 16'h0000, "and_disjoint");

        // =================================================================
        // OR (0x0F)
        // =================================================================
        $display("\n--- OR ---");
        t1(`OP_OR, 16'hFF00, 16'h00FF, 16'hFFFF, "or_basic");
        t1(`OP_OR, 16'h0000, 16'h0000, 16'h0000, "or_zero");
        t1(`OP_OR, 16'hAAAA, 16'h5555, 16'hFFFF, "or_fill");

        // =================================================================
        // XOR (0x10)
        // =================================================================
        $display("\n--- XOR ---");
        t1(`OP_XOR, 16'hFFFF, 16'hFFFF, 16'h0000, "xor_cancel");
        t1(`OP_XOR, 16'hFF00, 16'h0FF0, 16'hF0F0, "xor_basic");
        t1(`OP_XOR, 16'h1234, 16'h0000, 16'h1234, "xor_ident");

        // =================================================================
        // SHL (0x11): rD <- rA << imm[3:0]
        // =================================================================
        $display("\n--- SHL ---");
        t1(`OP_SHL, 16'h0001, 16'd0,  16'h0001, "shl_0");
        t1(`OP_SHL, 16'h0001, 16'd1,  16'h0002, "shl_1");
        t1(`OP_SHL, 16'h0001, 16'd15, 16'h8000, "shl_15");
        t1(`OP_SHL, 16'hFFFF, 16'd4,  16'hFFF0, "shl_4");
        t1(`OP_SHL, 16'h00FF, 16'd8,  16'hFF00, "shl_8");

        // =================================================================
        // SHR (0x12): arithmetic shift right
        // =================================================================
        $display("\n--- SHR (arithmetic) ---");
        t1(`OP_SHR, 16'h8000, 16'd1,  16'hC000, "shr_sign_ext");
        t1(`OP_SHR, 16'h8000, 16'd15, 16'hFFFF, "shr_15_neg");
        t1(`OP_SHR, 16'h7FFF, 16'd1,  16'h3FFF, "shr_pos");
        t1(`OP_SHR, 16'h7FFF, 16'd15, 16'h0000, "shr_15_pos");
        t1(`OP_SHR, 16'hFF00, 16'd4,  16'hFFF0, "shr_4_neg");
        t1(`OP_SHR, 16'h00FF, 16'd4,  16'h000F, "shr_4_pos");

        // =================================================================
        // MAX (0x0A): signed max
        // =================================================================
        $display("\n--- MAX ---");
        t1(`OP_MAX, 16'd100,   16'd200,   16'd200,  "max_pos");
        t1(`OP_MAX, 16'hFFFF,  16'd0,     16'd0,    "max_neg_vs_0");
        t1(`OP_MAX, 16'h8000,  16'h7FFF,  16'h7FFF, "max_extremes");
        t1(`OP_MAX, 16'hFFFE,  16'hFFFF,  16'hFFFF, "max_-2_vs_-1");
        t1(`OP_MAX, 16'd50,    16'd50,    16'd50,   "max_equal");

        // =================================================================
        // MIN (0x0B): signed min
        // =================================================================
        $display("\n--- MIN ---");
        t1(`OP_MIN, 16'd100,   16'd200,   16'd100,  "min_pos");
        t1(`OP_MIN, 16'hFFFF,  16'd0,     16'hFFFF, "min_neg_vs_0");
        t1(`OP_MIN, 16'h8000,  16'h7FFF,  16'h8000, "min_extremes");
        t1(`OP_MIN, 16'd50,    16'd50,    16'd50,   "min_equal");

        // =================================================================
        // SELP (0x16): rD <- P ? rA : rB
        // =================================================================
        $display("\n--- SELP ---");
        test_1cyc(`OP_SELP, 16'hAAAA, 16'h5555, 16'd0, 2'b00, 1'b1, 16'hAAAA, "selp_true");
        test_1cyc(`OP_SELP, 16'hAAAA, 16'h5555, 16'd0, 2'b00, 1'b0, 16'h5555, "selp_false");

        // =================================================================
        // SET (0x1A): rD <- cmp(rA, rB) ? 1 : 0
        // =================================================================
        $display("\n--- SET ---");
        test_1cyc(`OP_SET, 16'd5,    16'd5,    16'd0, `COMP_EQ, 1'b0, 16'd1, "set_eq_true");
        test_1cyc(`OP_SET, 16'd5,    16'd6,    16'd0, `COMP_EQ, 1'b0, 16'd0, "set_eq_false");
        test_1cyc(`OP_SET, 16'd5,    16'd6,    16'd0, `COMP_NE, 1'b0, 16'd1, "set_ne_true");
        test_1cyc(`OP_SET, 16'd5,    16'd5,    16'd0, `COMP_NE, 1'b0, 16'd0, "set_ne_false");
        test_1cyc(`OP_SET, 16'hFFFF, 16'd0,    16'd0, `COMP_LT, 1'b0, 16'd1, "set_lt_signed");
        test_1cyc(`OP_SET, 16'd0,    16'hFFFF, 16'd0, `COMP_LT, 1'b0, 16'd0, "set_lt_false");
        test_1cyc(`OP_SET, 16'd5,    16'd5,    16'd0, `COMP_LE, 1'b0, 16'd1, "set_le_equal");
        test_1cyc(`OP_SET, 16'd6,    16'd5,    16'd0, `COMP_LE, 1'b0, 16'd0, "set_le_false");

        // =================================================================
        // SETP (0x15): comparison flags
        // =================================================================
        $display("\n--- SETP (flags) ---");
        test_cmp(16'd5,    16'd5,    1,0,0,1, "cmp_equal");
        test_cmp(16'd5,    16'd10,   0,1,1,1, "cmp_lt");
        test_cmp(16'd10,   16'd5,    0,1,0,0, "cmp_gt");
        test_cmp(16'hFFFF, 16'd0,    0,1,1,1, "cmp_neg_vs_0");
        test_cmp(16'h8000, 16'h7FFF, 0,1,1,1, "cmp_min_vs_max");
        test_cmp(16'd0,    16'd0,    1,0,0,1, "cmp_0_0");

        // =================================================================
        // Address calculation: LD/ST/LDS/STS (reuse ADD path)
        // =================================================================
        $display("\n--- Address calc (LD/ST/LDS/STS) ---");
        t1(`OP_LD,  16'h1000, 16'h0008, 16'h1008, "ld_addr");
        t1(`OP_ST,  16'h2000, 16'h0010, 16'h2010, "st_addr");
        t1(`OP_LDS, 16'h0100, 16'h0004, 16'h0104, "lds_addr");
        t1(`OP_STS, 16'h0200, 16'h0008, 16'h0208, "sts_addr");
        t1(`OP_LD,  16'hFFFC, 16'h0008, 16'h0004, "ld_addr_wrap");

        // =================================================================
        // MUL (0x08): 2-cycle
        // =================================================================
        $display("\n--- MUL (2-cycle) ---");
        test_2cyc(`OP_MUL, 16'd0,    16'd0,    16'd0, 16'd0,     "mul_0x0");
        test_2cyc(`OP_MUL, 16'd1,    16'd1,    16'd0, 16'd1,     "mul_1x1");
        test_2cyc(`OP_MUL, 16'd3,    16'd7,    16'd0, 16'd21,    "mul_3x7");
        test_2cyc(`OP_MUL, 16'd100,  16'd200,  16'd0, 16'd20000, "mul_100x200");
        test_2cyc(`OP_MUL, 16'd256,  16'd256,  16'd0, 16'h0000,  "mul_overflow");
        test_2cyc(`OP_MUL, 16'hFFFF, 16'hFFFF, 16'd0, 16'h0001,  "mul_-1x-1");
        test_2cyc(`OP_MUL, 16'hFFFF, 16'd2,    16'd0, 16'hFFFE,  "mul_-1x2");
        test_2cyc(`OP_MUL, 16'd255,  16'd255,  16'd0, 16'hFE01,  "mul_255x255");

        // =================================================================
        // MULI (0x14): 2-cycle
        // =================================================================
        $display("\n--- MULI (2-cycle) ---");
        test_2cyc(`OP_MULI, 16'd10,   16'd5,    16'd0, 16'd50,   "muli_10x5");
        test_2cyc(`OP_MULI, 16'd8,    16'd3,    16'd0, 16'd24,   "muli_8x3");
        test_2cyc(`OP_MULI, 16'h0100, 16'h0100, 16'd0, 16'h0000, "muli_overflow");

        // =================================================================
        // FMA (0x09): 2-cycle, rD <- (rA * rB)[15:0] + rC
        // =================================================================
        $display("\n--- FMA (2-cycle) ---");
        test_2cyc(`OP_FMA, 16'd3,    16'd7,  16'd10,    16'd31,   "fma_3x7+10");
        test_2cyc(`OP_FMA, 16'd0,    16'd5,  16'd99,    16'd99,   "fma_0x5+99");
        test_2cyc(`OP_FMA, 16'd10,   16'd10, 16'd0,     16'd100,  "fma_10x10+0");
        test_2cyc(`OP_FMA, 16'd2,    16'd3,  16'hFFFA,  16'd0,    "fma_2x3+-6=0");
        test_2cyc(`OP_FMA, 16'hFFFF, 16'd1,  16'd1,     16'd0,    "fma_-1x1+1=0");
        test_2cyc(`OP_FMA, 16'd100,  16'd100, 16'd1000, 16'h2AF8, "fma_100x100+1000");

        $display("\n  Directed: %0d passed, %0d failed @ cycle %0d",
            pass_count, fail_count, cycle);

        // =================================================================
        // Random 1-cycle tests (FAIL only printed)
        // =================================================================
        $display("\n--- Random 1-cycle tests (5000 × 10 ops = 50000 vectors) ---");
        verbose = 0;
        for (i = 0; i < 5000; i = i + 1) begin
            rand_a = $random;
            rand_b = $random;
            sa = rand_a[15:0];
            sb = rand_b[15:0];

            gold = rand_a[15:0] + rand_b[15:0];
            t1(`OP_ADD, rand_a[15:0], rand_b[15:0], gold, "rand_add");

            gold = rand_a[15:0] - rand_b[15:0];
            t1(`OP_SUB, rand_a[15:0], rand_b[15:0], gold, "rand_sub");

            gold = rand_a[15:0] & rand_b[15:0];
            t1(`OP_AND, rand_a[15:0], rand_b[15:0], gold, "rand_and");

            gold = rand_a[15:0] | rand_b[15:0];
            t1(`OP_OR,  rand_a[15:0], rand_b[15:0], gold, "rand_or");

            gold = rand_a[15:0] ^ rand_b[15:0];
            t1(`OP_XOR, rand_a[15:0], rand_b[15:0], gold, "rand_xor");

            gold = rand_a[15:0] << rand_b[3:0];
            t1(`OP_SHL, rand_a[15:0], {12'd0, rand_b[3:0]}, gold, "rand_shl");

            gold = $signed(rand_a[15:0]) >>> rand_b[3:0];
            t1(`OP_SHR, rand_a[15:0], {12'd0, rand_b[3:0]}, gold, "rand_shr");

            gold = -rand_a[15:0];
            t1(`OP_NEG, rand_a[15:0], 16'd0, gold, "rand_neg");

            gold = (sa >= sb) ? rand_a[15:0] : rand_b[15:0];
            t1(`OP_MAX, rand_a[15:0], rand_b[15:0], gold, "rand_max");

            gold = (sa <= sb) ? rand_a[15:0] : rand_b[15:0];
            t1(`OP_MIN, rand_a[15:0], rand_b[15:0], gold, "rand_min");
        end
        $display("  Random 1-cyc: %0d passed, %0d failed @ cycle %0d",
            pass_count, fail_count, cycle);

        // =================================================================
        // Random 2-cycle tests (FAIL only printed)
        // =================================================================
        $display("\n--- Random 2-cycle tests (2000 × 2 ops = 4000 vectors) ---");
        for (i = 0; i < 2000; i = i + 1) begin
            rand_a = $random;
            rand_b = $random;
            rand_c = $random;

            gold = rand_a[15:0] * rand_b[15:0];
            test_2cyc(`OP_MUL, rand_a[15:0], rand_b[15:0], 16'd0, gold, "rand_mul");

            gold = (rand_a[15:0] * rand_b[15:0]) + rand_c[15:0];
            test_2cyc(`OP_FMA, rand_a[15:0], rand_b[15:0], rand_c[15:0], gold, "rand_fma");
        end
        $display("  Random 2-cyc: %0d passed, %0d failed @ cycle %0d",
            pass_count, fail_count, cycle);

        // =================================================================
        // Back-to-back: mixed 1-cycle / 2-cycle
        // =================================================================
        $display("\n--- Back-to-back pipeline ---");
        verbose = 1;
        t1(`OP_ADD, 16'd10, 16'd20, 16'd30, "b2b_add");
        test_2cyc(`OP_MUL, 16'd5, 16'd6, 16'd0, 16'd30, "b2b_mul");
        t1(`OP_SUB, 16'd100, 16'd50, 16'd50, "b2b_sub");
        test_2cyc(`OP_FMA, 16'd2, 16'd3, 16'd4, 16'd10, "b2b_fma");
        t1(`OP_ADD, 16'd1, 16'd1, 16'd2, "b2b_add2");

        // =================================================================
        // Final summary
        // =================================================================
        repeat (5) @(posedge clk);
        $display("\n=========================================================");
        $display(" Results: %0d PASSED, %0d FAILED / %0d total",
            pass_count, fail_count, pass_count + fail_count);
        $display(" Total cycles: %0d", cycle);
        $display("=========================================================");
        $finish;
    end

    initial begin
        #50000000;
        $display("WATCHDOG timeout @ cycle %0d", cycle);
        $finish;
    end
endmodule