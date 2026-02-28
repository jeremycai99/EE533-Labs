/* file: bf16fpu_tb.v
 * Description: Testbench for the BF16 FPU module.
 *   - Tests 1-cycle ops: ABS, NEG, MAX, MIN, SELP, SET, SETP
 *   - Tests multi-cycle ops: MUL (3 cyc), ADD (4 cyc), SUB (4 cyc),
 *                            ADDI (4 cyc), MULI (3 cyc), FMA (3+4=7 cyc)
 *   - Verifies busy/valid handshaking
 *   - Uses known BF16 bit-patterns for golden-value checks
 *   - Tests edge cases: zero, negative, back-to-back ops
 *
 * Pipeline latencies (from valid_in assertion to valid_out assertion):
 *   pplbf16mult:   3 clock cycles  (3 pipeline registers)
 *   pplbf16addsub: 4 clock cycles  (4 pipeline registers)
 *   FMA:           3 + 4 = 7 clock cycles (mult then addsub)
 *
 * Author: Claude (generated for Jeremy Cai's EE533 GPU project)
 * Date: Feb. 27, 2026
 */

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "bf16fpu.v"

module bf16fpu_tb;

    // ═══════════════════════════════════════════════════════════════════════
    // BF16 Constants: {sign[15], exp[14:7], frac[6:0]}
    // ═══════════════════════════════════════════════════════════════════════
    localparam [15:0] BF16_POS_ZERO  = 16'h0000;  //  +0.0
    localparam [15:0] BF16_NEG_ZERO  = 16'h8000;  //  -0.0
    localparam [15:0] BF16_ONE       = 16'h3F80;  //   1.0
    localparam [15:0] BF16_NEG_ONE   = 16'hBF80;  //  -1.0
    localparam [15:0] BF16_TWO       = 16'h4000;  //   2.0
    localparam [15:0] BF16_NEG_TWO   = 16'hC000;  //  -2.0
    localparam [15:0] BF16_THREE     = 16'h4040;  //   3.0
    localparam [15:0] BF16_NEG_THREE = 16'hC040;  //  -3.0
    localparam [15:0] BF16_FOUR      = 16'h4080;  //   4.0
    localparam [15:0] BF16_FIVE      = 16'h40A0;  //   5.0
    localparam [15:0] BF16_SIX       = 16'h40C0;  //   6.0
    localparam [15:0] BF16_SEVEN     = 16'h40E0;  //   7.0
    localparam [15:0] BF16_TEN       = 16'h4120;  //  10.0
    localparam [15:0] BF16_HALF      = 16'h3F00;  //   0.5
    localparam [15:0] BF16_ONE_HALF  = 16'h3FC0;  //   1.5
    localparam [15:0] BF16_POS_INF   = 16'h7F80;  //  +Inf
    localparam [15:0] BF16_NEG_INF   = 16'hFF80;  //  -Inf
    localparam [15:0] BF16_QNAN      = 16'h7FC0;  //  QNaN (canonical)

    // ═══════════════════════════════════════════════════════════════════════
    // DUT Signals
    // ═══════════════════════════════════════════════════════════════════════
    reg         clk;
    reg         rst_n;
    reg  [4:0]  alu_op;
    reg         valid_in;
    reg  [1:0]  cmp_mode;
    reg         pred_val;
    reg  [15:0] op_a;
    reg  [15:0] op_b;
    reg  [15:0] op_c;

    wire [15:0] result;
    wire        valid_out;
    wire        busy;
    wire        cmp_eq, cmp_ne, cmp_lt, cmp_le;

    // ═══════════════════════════════════════════════════════════════════════
    // Clock: 10 ns period (100 MHz)
    // ═══════════════════════════════════════════════════════════════════════
    initial clk = 0;
    always #5 clk = ~clk;

    // ═══════════════════════════════════════════════════════════════════════
    // DUT Instantiation
    // ═══════════════════════════════════════════════════════════════════════
    bf16fpu u_dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .alu_op   (alu_op),
        .valid_in (valid_in),
        .cmp_mode (cmp_mode),
        .pred_val (pred_val),
        .op_a     (op_a),
        .op_b     (op_b),
        .op_c     (op_c),
        .result   (result),
        .valid_out(valid_out),
        .busy     (busy),
        .cmp_eq   (cmp_eq),
        .cmp_ne   (cmp_ne),
        .cmp_lt   (cmp_lt),
        .cmp_le   (cmp_le)
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Test Bookkeeping
    // ═══════════════════════════════════════════════════════════════════════
    integer test_num;
    integer pass_count;
    integer fail_count;

    // ═══════════════════════════════════════════════════════════════════════
    // Helper Tasks
    // ═══════════════════════════════════════════════════════════════════════

    // Clear all inputs to idle state
    task clear_inputs;
        begin
            alu_op   = `OP_NOP;
            valid_in = 1'b0;
            cmp_mode = 2'b00;
            pred_val = 1'b0;
            op_a     = 16'd0;
            op_b     = 16'd0;
            op_c     = 16'd0;
        end
    endtask

    // Wait for valid_out with timeout. Returns on the posedge where valid_out is high.
    task wait_for_valid;
        input integer timeout_cycles;
        integer cnt;
        begin
            cnt = 0;
            // If valid_out is already high (1-cycle op), don't wait
            if (!valid_out) begin
                while (cnt < timeout_cycles) begin
                    @(posedge clk);
                    #1; // allow combinational settle
                    cnt = cnt + 1;
                    if (valid_out) begin
                        cnt = timeout_cycles; // break
                    end
                end
            end
        end
    endtask

    // Check result and report
    task check;
        input [15:0]  expected;
        input [255:0] name;
        begin
            test_num = test_num + 1;
            if (result === expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %2d: %-30s | result=0x%04h  expected=0x%04h",
                         test_num, name, result, expected);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %2d: %-30s | result=0x%04h  expected=0x%04h  <<<",
                         test_num, name, result, expected);
            end
        end
    endtask

    // Check a comparison flag
    task check_flag;
        input         actual;
        input         expected;
        input [255:0] name;
        begin
            test_num = test_num + 1;
            if (actual === expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %2d: %-30s | flag=%b  expected=%b",
                         test_num, name, actual, expected);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %2d: %-30s | flag=%b  expected=%b  <<<",
                         test_num, name, actual, expected);
            end
        end
    endtask

    // ─────────────────────────────────────────────────────────────────────
    // Test 1-cycle operations
    //   valid_out is combinational: (state==IDLE) & valid_in & is_1cyc_op
    //   So we assert valid_in at negedge, and sample at the SAME posedge.
    // ─────────────────────────────────────────────────────────────────────
    task test_1cyc_op;
        input [4:0]   t_op;
        input [15:0]  t_a, t_b;
        input [1:0]   t_cmp;
        input         t_pred;
        input [15:0]  expected;
        input [255:0] name;
        begin
            @(negedge clk);
            alu_op   = t_op;
            op_a     = t_a;
            op_b     = t_b;
            op_c     = 16'd0;
            cmp_mode = t_cmp;
            pred_val = t_pred;
            valid_in = 1'b1;
            @(posedge clk);
            #1;
            // valid_out should be high now (combinational)
            if (!valid_out) begin
                test_num = test_num + 1;
                fail_count = fail_count + 1;
                $display("[FAIL] Test %2d: %-30s | valid_out not asserted for 1-cyc op  <<<", test_num, name);
            end else begin
                check(expected, name);
            end
            @(negedge clk);
            valid_in = 1'b0;
            @(negedge clk); // gap cycle
        end
    endtask

    // ─────────────────────────────────────────────────────────────────────
    // Test multi-cycle operations (MUL, ADD, SUB, ADDI, MULI)
    //   Assert valid_in for 1 cycle, then wait for valid_out from pipeline.
    // ─────────────────────────────────────────────────────────────────────
    task test_multicyc_op;
        input [4:0]   t_op;
        input [15:0]  t_a, t_b, t_c;
        input [15:0]  expected;
        input [255:0] name;
        begin
            // Assert valid_in for one cycle
            @(negedge clk);
            alu_op   = t_op;
            op_a     = t_a;
            op_b     = t_b;
            op_c     = t_c;
            cmp_mode = 2'b00;
            pred_val = 1'b0;
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;
            // Wait for valid_out
            wait_for_valid(20);
            if (!valid_out) begin
                test_num = test_num + 1;
                fail_count = fail_count + 1;
                $display("[FAIL] Test %2d: %-30s | TIMEOUT waiting for valid_out  <<<", test_num, name);
            end else begin
                check(expected, name);
            end
            // Wait for state to return to IDLE
            @(negedge clk);
            @(negedge clk);
        end
    endtask

    // ═══════════════════════════════════════════════════════════════════════
    // Main Stimulus
    // ═══════════════════════════════════════════════════════════════════════
    initial begin
        $dumpfile("bf16fpu_tb.vcd");
        $dumpvars(0, bf16fpu_tb);

        // Initialization
        test_num   = 0;
        pass_count = 0;
        fail_count = 0;
        rst_n      = 0;
        clear_inputs;

        // Hold reset for 4 cycles
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        $display("═══════════════════════════════════════════════════════════");
        $display("  BF16 FPU Testbench - Starting Tests");
        $display("═══════════════════════════════════════════════════════════");

        // ═════════════════════════════════════════════════════════════════
        // Section 1: Comparator flag tests (combinational, always active)
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 1: Comparator Flags ---");

        // Drive operands and check flags (no valid_in needed for comparator)
        @(negedge clk);
        op_a = BF16_ONE; op_b = BF16_ONE; valid_in = 0;
        @(posedge clk); #1;
        check_flag(cmp_eq, 1'b1, "CMP: 1.0 == 1.0  (eq)");
        check_flag(cmp_ne, 1'b0, "CMP: 1.0 == 1.0  (ne)");
        check_flag(cmp_lt, 1'b0, "CMP: 1.0 == 1.0  (lt)");
        check_flag(cmp_le, 1'b1, "CMP: 1.0 == 1.0  (le)");

        @(negedge clk);
        op_a = BF16_ONE; op_b = BF16_TWO;
        @(posedge clk); #1;
        check_flag(cmp_eq, 1'b0, "CMP: 1.0 vs 2.0  (eq)");
        check_flag(cmp_ne, 1'b1, "CMP: 1.0 vs 2.0  (ne)");
        check_flag(cmp_lt, 1'b1, "CMP: 1.0 vs 2.0  (lt)");
        check_flag(cmp_le, 1'b1, "CMP: 1.0 vs 2.0  (le)");

        @(negedge clk);
        op_a = BF16_THREE; op_b = BF16_ONE;
        @(posedge clk); #1;
        check_flag(cmp_lt, 1'b0, "CMP: 3.0 vs 1.0  (lt)");
        check_flag(cmp_le, 1'b0, "CMP: 3.0 vs 1.0  (le)");

        @(negedge clk);
        op_a = BF16_NEG_TWO; op_b = BF16_ONE;
        @(posedge clk); #1;
        check_flag(cmp_lt, 1'b1, "CMP: -2.0 vs 1.0 (lt)");

        @(negedge clk);
        op_a = BF16_POS_ZERO; op_b = BF16_NEG_ZERO;
        @(posedge clk); #1;
        check_flag(cmp_eq, 1'b1, "CMP: +0 vs -0     (eq)");

        @(negedge clk);
        op_a = BF16_QNAN; op_b = BF16_ONE;
        @(posedge clk); #1;
        check_flag(cmp_eq, 1'b0, "CMP: NaN vs 1.0  (eq)");
        check_flag(cmp_lt, 1'b0, "CMP: NaN vs 1.0  (lt)");
        check_flag(cmp_ne, 1'b1, "CMP: NaN vs 1.0  (ne)");

        @(negedge clk); clear_inputs; @(negedge clk);

        // ═════════════════════════════════════════════════════════════════
        // Section 2: 1-cycle operations
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 2: 1-Cycle Operations ---");

        // ABS(-2.0) = 2.0
        test_1cyc_op(`OP_ABS, BF16_NEG_TWO, 16'h0000, 2'b00, 1'b0,
                     BF16_TWO, "ABS(-2.0) = 2.0");

        // ABS(3.0) = 3.0 (already positive)
        test_1cyc_op(`OP_ABS, BF16_THREE, 16'h0000, 2'b00, 1'b0,
                     BF16_THREE, "ABS(3.0) = 3.0");

        // NEG(3.0) = -3.0
        test_1cyc_op(`OP_NEG, BF16_THREE, 16'h0000, 2'b00, 1'b0,
                     BF16_NEG_THREE, "NEG(3.0) = -3.0");

        // NEG(-1.0) = 1.0
        test_1cyc_op(`OP_NEG, BF16_NEG_ONE, 16'h0000, 2'b00, 1'b0,
                     BF16_ONE, "NEG(-1.0) = 1.0");

        // MAX(1.0, 2.0) = 2.0
        test_1cyc_op(`OP_MAX, BF16_ONE, BF16_TWO, 2'b00, 1'b0,
                     BF16_TWO, "MAX(1.0, 2.0) = 2.0");

        // MAX(5.0, 3.0) = 5.0
        test_1cyc_op(`OP_MAX, BF16_FIVE, BF16_THREE, 2'b00, 1'b0,
                     BF16_FIVE, "MAX(5.0, 3.0) = 5.0");

        // MAX(-2.0, -1.0) = -1.0  (cmp_lt: -2 < -1 is true, so result = op_b)
        test_1cyc_op(`OP_MAX, BF16_NEG_TWO, BF16_NEG_ONE, 2'b00, 1'b0,
                     BF16_NEG_ONE, "MAX(-2.0, -1.0) = -1.0");

        // MIN(1.0, 2.0) = 1.0
        test_1cyc_op(`OP_MIN, BF16_ONE, BF16_TWO, 2'b00, 1'b0,
                     BF16_ONE, "MIN(1.0, 2.0) = 1.0");

        // MIN(5.0, 3.0) = 3.0
        test_1cyc_op(`OP_MIN, BF16_FIVE, BF16_THREE, 2'b00, 1'b0,
                     BF16_THREE, "MIN(5.0, 3.0) = 3.0");

        // SELP(3.0, 4.0, pred=1) = 3.0
        test_1cyc_op(`OP_SELP, BF16_THREE, BF16_FOUR, 2'b00, 1'b1,
                     BF16_THREE, "SELP pred=1 -> op_a=3.0");

        // SELP(3.0, 4.0, pred=0) = 4.0
        test_1cyc_op(`OP_SELP, BF16_THREE, BF16_FOUR, 2'b00, 1'b0,
                     BF16_FOUR, "SELP pred=0 -> op_b=4.0");

        // SET with COMP_EQ: 1.0 == 1.0 → 1
        test_1cyc_op(`OP_SET, BF16_ONE, BF16_ONE, `COMP_EQ, 1'b0,
                     16'h0001, "SET EQ: 1.0==1.0 -> 1");

        // SET with COMP_EQ: 1.0 == 2.0 → 0
        test_1cyc_op(`OP_SET, BF16_ONE, BF16_TWO, `COMP_EQ, 1'b0,
                     16'h0000, "SET EQ: 1.0==2.0 -> 0");

        // SET with COMP_LT: 1.0 < 2.0 → 1
        test_1cyc_op(`OP_SET, BF16_ONE, BF16_TWO, `COMP_LT, 1'b0,
                     16'h0001, "SET LT: 1.0<2.0 -> 1");

        // SET with COMP_LT: 3.0 < 2.0 → 0
        test_1cyc_op(`OP_SET, BF16_THREE, BF16_TWO, `COMP_LT, 1'b0,
                     16'h0000, "SET LT: 3.0<2.0 -> 0");

        // SET with COMP_NE: 1.0 != 2.0 → 1
        test_1cyc_op(`OP_SET, BF16_ONE, BF16_TWO, `COMP_NE, 1'b0,
                     16'h0001, "SET NE: 1.0!=2.0 -> 1");

        // SET with COMP_LE: 2.0 <= 2.0 → 1
        test_1cyc_op(`OP_SET, BF16_TWO, BF16_TWO, `COMP_LE, 1'b0,
                     16'h0001, "SET LE: 2.0<=2.0 -> 1");

        // SETP with COMP_LT: 1.0 < 5.0 → 1
        test_1cyc_op(`OP_SETP, BF16_ONE, BF16_FIVE, `COMP_LT, 1'b0,
                     16'h0001, "SETP LT: 1.0<5.0 -> 1");

        // ═════════════════════════════════════════════════════════════════
        // Section 3: Multi-cycle MUL (3-cycle pipeline)
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 3: Multiplication (3-cycle pipeline) ---");

        // 2.0 * 3.0 = 6.0
        test_multicyc_op(`OP_MUL, BF16_TWO, BF16_THREE, 16'h0000,
                         BF16_SIX, "MUL: 2.0 * 3.0 = 6.0");

        // 1.0 * 1.0 = 1.0
        test_multicyc_op(`OP_MUL, BF16_ONE, BF16_ONE, 16'h0000,
                         BF16_ONE, "MUL: 1.0 * 1.0 = 1.0");

        // -2.0 * 3.0 = -6.0
        test_multicyc_op(`OP_MUL, BF16_NEG_TWO, BF16_THREE, 16'h0000,
                         16'hC0C0, "MUL: -2.0 * 3.0 = -6.0");

        // -1.0 * -1.0 = 1.0
        test_multicyc_op(`OP_MUL, BF16_NEG_ONE, BF16_NEG_ONE, 16'h0000,
                         BF16_ONE, "MUL: -1.0 * -1.0 = 1.0");

        // 2.0 * 0.5 = 1.0
        test_multicyc_op(`OP_MUL, BF16_TWO, BF16_HALF, 16'h0000,
                         BF16_ONE, "MUL: 2.0 * 0.5 = 1.0");

        // 0.0 * 5.0 = 0.0
        test_multicyc_op(`OP_MUL, BF16_POS_ZERO, BF16_FIVE, 16'h0000,
                         BF16_POS_ZERO, "MUL: 0.0 * 5.0 = 0.0");

        // MULI: 3.0 * 2.0 = 6.0 (same pipeline as MUL)
        test_multicyc_op(`OP_MULI, BF16_THREE, BF16_TWO, 16'h0000,
                         BF16_SIX, "MULI: 3.0 * 2.0 = 6.0");

        // ═════════════════════════════════════════════════════════════════
        // Section 4: Multi-cycle ADD/SUB (4-cycle pipeline)
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 4: Addition/Subtraction (4-cycle pipeline) ---");

        // 1.0 + 2.0 = 3.0
        test_multicyc_op(`OP_ADD, BF16_ONE, BF16_TWO, 16'h0000,
                         BF16_THREE, "ADD: 1.0 + 2.0 = 3.0");

        // 3.0 + 4.0 = 7.0
        test_multicyc_op(`OP_ADD, BF16_THREE, BF16_FOUR, 16'h0000,
                         BF16_SEVEN, "ADD: 3.0 + 4.0 = 7.0");

        // -1.0 + 1.0 = 0.0
        test_multicyc_op(`OP_ADD, BF16_NEG_ONE, BF16_ONE, 16'h0000,
                         BF16_POS_ZERO, "ADD: -1.0 + 1.0 = 0.0");

        // -2.0 + -3.0 = -5.0
        test_multicyc_op(`OP_ADD, BF16_NEG_TWO, BF16_NEG_THREE, 16'h0000,
                         16'hC0A0, "ADD: -2.0 + -3.0 = -5.0");

        // 1.0 + 0.5 = 1.5
        test_multicyc_op(`OP_ADD, BF16_ONE, BF16_HALF, 16'h0000,
                         BF16_ONE_HALF, "ADD: 1.0 + 0.5 = 1.5");

        // SUB: 3.0 - 1.0 = 2.0
        test_multicyc_op(`OP_SUB, BF16_THREE, BF16_ONE, 16'h0000,
                         BF16_TWO, "SUB: 3.0 - 1.0 = 2.0");

        // SUB: 5.0 - 5.0 = 0.0
        test_multicyc_op(`OP_SUB, BF16_FIVE, BF16_FIVE, 16'h0000,
                         BF16_POS_ZERO, "SUB: 5.0 - 5.0 = 0.0");

        // SUB: 1.0 - 3.0 = -2.0
        test_multicyc_op(`OP_SUB, BF16_ONE, BF16_THREE, 16'h0000,
                         BF16_NEG_TWO, "SUB: 1.0 - 3.0 = -2.0");

        // ADDI: 4.0 + 3.0 = 7.0
        test_multicyc_op(`OP_ADDI, BF16_FOUR, BF16_THREE, 16'h0000,
                         BF16_SEVEN, "ADDI: 4.0 + 3.0 = 7.0");

        // ═════════════════════════════════════════════════════════════════
        // Section 5: FMA (3-cycle mult + 4-cycle add = 7 cycles)
        //   FMA computes: op_a * op_b + op_c
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 5: Fused Multiply-Add (7-cycle total) ---");

        // FMA: 2.0 * 3.0 + 4.0 = 10.0
        test_multicyc_op(`OP_FMA, BF16_TWO, BF16_THREE, BF16_FOUR,
                         BF16_TEN, "FMA: 2*3+4 = 10.0");

        // FMA: 1.0 * 1.0 + 1.0 = 2.0
        test_multicyc_op(`OP_FMA, BF16_ONE, BF16_ONE, BF16_ONE,
                         BF16_TWO, "FMA: 1*1+1 = 2.0");

        // FMA: 2.0 * 2.0 + 3.0 = 7.0
        test_multicyc_op(`OP_FMA, BF16_TWO, BF16_TWO, BF16_THREE,
                         BF16_SEVEN, "FMA: 2*2+3 = 7.0");

        // FMA: 3.0 * 1.0 + 0.0 = 3.0  (FMA as pure multiply)
        test_multicyc_op(`OP_FMA, BF16_THREE, BF16_ONE, BF16_POS_ZERO,
                         BF16_THREE, "FMA: 3*1+0 = 3.0");

        // FMA: -1.0 * 2.0 + 5.0 = 3.0
        test_multicyc_op(`OP_FMA, BF16_NEG_ONE, BF16_TWO, BF16_FIVE,
                         BF16_THREE, "FMA: -1*2+5 = 3.0");

        // ═════════════════════════════════════════════════════════════════
        // Section 6: Busy signal and handshake verification
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 6: Busy/Valid Handshake ---");

        // Verify busy goes high during multi-cycle op and returns low after
        begin : busy_test_block
            integer cycle_count;

            // Start a MUL and count cycles
            @(negedge clk);
            alu_op   = `OP_MUL;
            op_a     = BF16_TWO;
            op_b     = BF16_THREE;
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            // busy should be high now
            @(posedge clk); #1;
            test_num = test_num + 1;
            if (busy) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %2d: %-30s | busy=%b", test_num, "BUSY high during MUL", busy);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %2d: %-30s | busy=%b  <<<", test_num, "BUSY high during MUL", busy);
            end

            // Count cycles until valid_out
            cycle_count = 1;
            while (!valid_out && cycle_count < 20) begin
                @(posedge clk); #1;
                cycle_count = cycle_count + 1;
            end

            // Check result
            check(BF16_SIX, "BUSY test MUL result=6.0");

            // After valid_out, next cycle busy should drop
            @(posedge clk); #1;
            @(posedge clk); #1;
            test_num = test_num + 1;
            if (!busy) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %2d: %-30s | busy=%b", test_num, "BUSY low after MUL done", busy);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %2d: %-30s | busy=%b  <<<", test_num, "BUSY low after MUL done", busy);
            end

            @(negedge clk); clear_inputs; @(negedge clk);
        end

        // ═════════════════════════════════════════════════════════════════
        // Section 7: Back-to-back operations
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 7: Back-to-Back Operations ---");

        // 1-cycle op right after reset idle
        test_1cyc_op(`OP_ABS, BF16_NEG_ONE, 16'h0000, 2'b00, 1'b0,
                     BF16_ONE, "B2B: ABS(-1) after idle");

        // MUL followed immediately by another 1-cycle op (should wait)
        test_multicyc_op(`OP_MUL, BF16_ONE, BF16_TWO, 16'h0000,
                         BF16_TWO, "B2B: MUL 1*2=2.0");

        test_1cyc_op(`OP_NEG, BF16_FIVE, 16'h0000, 2'b00, 1'b0,
                     16'hC0A0, "B2B: NEG(5) after MUL");

        // ADD followed by MUL
        test_multicyc_op(`OP_ADD, BF16_ONE, BF16_ONE, 16'h0000,
                         BF16_TWO, "B2B: ADD 1+1=2.0");

        test_multicyc_op(`OP_MUL, BF16_THREE, BF16_THREE, 16'h0000,
                         16'h4110, "B2B: MUL 3*3=9.0");
        // 3.0*3.0 = 9.0: sign=0, exp=130=10000010, frac=0010000 -> 0_10000010_0010000 = 0x4110

        // FMA followed by 1-cycle op
        test_multicyc_op(`OP_FMA, BF16_ONE, BF16_TWO, BF16_THREE,
                         BF16_FIVE, "B2B: FMA 1*2+3=5.0");

        test_1cyc_op(`OP_MAX, BF16_ONE, BF16_SEVEN, 2'b00, 1'b0,
                     BF16_SEVEN, "B2B: MAX(1,7) after FMA");

        // ═════════════════════════════════════════════════════════════════
        // Section 8: Special values
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 8: Special Value Handling ---");

        // MUL: Inf * 2.0 = Inf
        test_multicyc_op(`OP_MUL, BF16_POS_INF, BF16_TWO, 16'h0000,
                         BF16_POS_INF, "MUL: Inf * 2.0 = Inf");

        // MUL: Inf * 0.0 = NaN  (special: inf * zero)
        test_multicyc_op(`OP_MUL, BF16_POS_INF, BF16_POS_ZERO, 16'h0000,
                         BF16_QNAN, "MUL: Inf * 0.0 = NaN");

        // MUL: NaN * 1.0 = NaN
        test_multicyc_op(`OP_MUL, BF16_QNAN, BF16_ONE, 16'h0000,
                         BF16_QNAN, "MUL: NaN * 1.0 = NaN");

        // ADD: Inf + 1.0 = Inf
        test_multicyc_op(`OP_ADD, BF16_POS_INF, BF16_ONE, 16'h0000,
                         BF16_POS_INF, "ADD: Inf + 1.0 = Inf");

        // ADD: Inf + -Inf = NaN
        test_multicyc_op(`OP_ADD, BF16_POS_INF, BF16_NEG_INF, 16'h0000,
                         BF16_QNAN, "ADD: Inf + -Inf = NaN");

        // SUB: Inf - Inf = NaN
        test_multicyc_op(`OP_SUB, BF16_POS_INF, BF16_POS_INF, 16'h0000,
                         BF16_QNAN, "SUB: Inf - Inf = NaN");

        // ADD: 0.0 + 0.0 = 0.0
        test_multicyc_op(`OP_ADD, BF16_POS_ZERO, BF16_POS_ZERO, 16'h0000,
                         BF16_POS_ZERO, "ADD: 0.0 + 0.0 = 0.0");

        // ADD: NaN + 1.0 = NaN
        test_multicyc_op(`OP_ADD, BF16_QNAN, BF16_ONE, 16'h0000,
                         BF16_QNAN, "ADD: NaN + 1.0 = NaN");

        // ═════════════════════════════════════════════════════════════════
        // Section 9: 1-cycle ops on special values
        // ═════════════════════════════════════════════════════════════════
        $display("\n--- Section 9: 1-Cycle Ops on Special Values ---");

        // ABS(-Inf) = +Inf
        test_1cyc_op(`OP_ABS, BF16_NEG_INF, 16'h0000, 2'b00, 1'b0,
                     BF16_POS_INF, "ABS(-Inf) = +Inf");

        // NEG(+Inf) = -Inf
        test_1cyc_op(`OP_NEG, BF16_POS_INF, 16'h0000, 2'b00, 1'b0,
                     BF16_NEG_INF, "NEG(+Inf) = -Inf");

        // ABS(0.0) = 0.0
        test_1cyc_op(`OP_ABS, BF16_POS_ZERO, 16'h0000, 2'b00, 1'b0,
                     BF16_POS_ZERO, "ABS(+0) = +0");

        // ABS(-0.0) = +0.0
        test_1cyc_op(`OP_ABS, BF16_NEG_ZERO, 16'h0000, 2'b00, 1'b0,
                     BF16_POS_ZERO, "ABS(-0) = +0");

        // NEG(0.0) = -0.0
        test_1cyc_op(`OP_NEG, BF16_POS_ZERO, 16'h0000, 2'b00, 1'b0,
                     BF16_NEG_ZERO, "NEG(+0) = -0");

        // ═══════════════════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════════════════
        $display("\n═══════════════════════════════════════════════════════════");
        $display("  Test Summary: %0d PASSED, %0d FAILED out of %0d total",
                 pass_count, fail_count, test_num);
        if (fail_count == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** SOME TESTS FAILED ***");
        $display("═══════════════════════════════════════════════════════════");

        #100;
        $finish;
    end

    // ═══════════════════════════════════════════════════════════════════════
    // Watchdog timer
    // ═══════════════════════════════════════════════════════════════════════
    initial begin
        #100000;
        $display("[WATCHDOG] Simulation timed out at %0t", $time);
        $finish;
    end

endmodule