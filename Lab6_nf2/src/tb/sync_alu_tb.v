/* file: sync_alu_tb.v
 * Description: Testbench for sync_alu module.
 *              Verifies all ALU operations and synchronous output behavior.
 * Date: Feb. 9, 2026
 * Author: Jeremy Cai
 * Version: 1.0
 */

`timescale 1ns / 1ps

`include "define.v"
`include "sync_alu.v"

module sync_alu_tb;

    // Clock period
    localparam CLK_PERIOD = 10; //100 MHz clock

    // Testbench signals
    reg clk;
    reg rst_n;
    reg [`DATA_WIDTH-1:0] A;
    reg [`DATA_WIDTH-1:0] B;
    reg [`ALU_OP_WIDTH-1:0] aluctrl;
    wire [`DATA_WIDTH-1:0] Z;
    wire overflow;

    // Bookkeeping
    integer test_num;
    integer pass_count;
    integer fail_count;

    // DUT instantiation
    sync_alu uut (
        .clk(clk),
        .rst_n(rst_n),
        .A(A),
        .B(B),
        .aluctrl(aluctrl),
        .Z(Z),
        .overflow(overflow)
    );

    // Clock generation
    initial clk = 0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    // ---------------------------------------------------------------
    // Task: apply inputs, wait one rising edge for the registered
    //       output to update, then check result on the next edge.
    // ---------------------------------------------------------------
    task apply_and_check;
        input [`DATA_WIDTH-1:0] in_a;
        input [`DATA_WIDTH-1:0] in_b;
        input [`ALU_OP_WIDTH-1:0] op;
        input [`DATA_WIDTH-1:0] expected_z;
        input expected_ovf;
        input [255:0] op_name; // ASCII string for display
        begin
            // Drive inputs before the rising edge
            @(negedge clk);
            A      = in_a;
            B      = in_b;
            aluctrl = op;

            // Wait for the rising edge to latch, then a small
            // settling delay so Z and overflow are stable
            @(posedge clk);
            #1;

            test_num = test_num + 1;

            if (Z !== expected_z || overflow !== expected_ovf) begin
                $display("FAIL test %0d [%0s]: A=0x%016h B=0x%016h | Z=0x%016h (exp 0x%016h) ovf=%0b (exp %0b)",
                         test_num, op_name, in_a, in_b, Z, expected_z, overflow, expected_ovf);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS test %0d [%0s]: A=0x%016h B=0x%016h | Z=0x%016h ovf=%0b",
                         test_num, op_name, in_a, in_b, Z, overflow);
                pass_count = pass_count + 1;
            end
        end
    endtask

    // ---------------------------------------------------------------
    // Main stimulus
    // ---------------------------------------------------------------
    initial begin
        // Init
        test_num   = 0;
        pass_count = 0;
        fail_count = 0;
        A          = 0;
        B          = 0;
        aluctrl    = 0;
        rst_n      = 1;

        // ------ Reset test ------
        $display("\n===== Reset Test =====");
        @(negedge clk);
        rst_n = 0;
        @(posedge clk);
        #1;
        test_num = test_num + 1;
        if (Z !== 0 || overflow !== 0) begin
            $display("FAIL test %0d [RESET]: Z=0x%016h ovf=%0b (expected all zero)", test_num, Z, overflow);
            fail_count = fail_count + 1;
        end else begin
            $display("PASS test %0d [RESET]: Z=0x%016h ovf=%0b", test_num, Z, overflow);
            pass_count = pass_count + 1;
        end
        @(negedge clk);
        rst_n = 1;

        // ------ ADD tests ------
        $display("\n===== ADD Tests =====");
        apply_and_check(
            64'h0000_0000_0000_0001,
            64'h0000_0000_0000_0002,
            `ALU_OP_ADD,
            64'h0000_0000_0000_0003,
            1'b0,
            "ADD basic"
        );

        apply_and_check(
            64'hFFFF_FFFF_FFFF_FFFF,
            64'h0000_0000_0000_0001,
            `ALU_OP_ADD,
            64'h0000_0000_0000_0000,
            1'b0,  // unsigned wrap, no signed overflow
            "ADD wrap"
        );

        // Signed overflow: two large positive numbers whose sum overflows
        apply_and_check(
            64'h7FFF_FFFF_FFFF_FFFF,
            64'h0000_0000_0000_0001,
            `ALU_OP_ADD,
            64'h8000_0000_0000_0000,
            1'b1,  // signed overflow
            "ADD overflow"
        );

        // ------ SUB tests ------
        $display("\n===== SUB Tests =====");
        apply_and_check(
            64'h0000_0000_0000_0005,
            64'h0000_0000_0000_0003,
            `ALU_OP_SUB,
            64'h0000_0000_0000_0002,
            1'b0,
            "SUB basic"
        );

        apply_and_check(
            64'h0000_0000_0000_0000,
            64'h0000_0000_0000_0001,
            `ALU_OP_SUB,
            64'hFFFF_FFFF_FFFF_FFFF,
            1'b0,  // no signed overflow (0 - 1 = -1, representable)
            "SUB underflow"
        );

        // Signed overflow: positive - negative overflows
        apply_and_check(
            64'h7FFF_FFFF_FFFF_FFFF,
            64'hFFFF_FFFF_FFFF_FFFF, // -1 in signed
            `ALU_OP_SUB,
            64'h8000_0000_0000_0000,
            1'b1,  // signed overflow
            "SUB overflow"
        );

        // ------ AND test ------
        $display("\n===== AND Tests =====");
        apply_and_check(
            64'hFF00_FF00_FF00_FF00,
            64'h0F0F_0F0F_0F0F_0F0F,
            `ALU_OP_AND,
            64'h0F00_0F00_0F00_0F00,
            1'b0,
            "AND"
        );

        apply_and_check(
            64'hFFFF_FFFF_FFFF_FFFF,
            64'h0000_0000_0000_0000,
            `ALU_OP_AND,
            64'h0000_0000_0000_0000,
            1'b0,
            "AND zero"
        );

        // ------ OR test ------
        $display("\n===== OR Tests =====");
        apply_and_check(
            64'hFF00_FF00_FF00_FF00,
            64'h0F0F_0F0F_0F0F_0F0F,
            `ALU_OP_OR,
            64'hFF0F_FF0F_FF0F_FF0F,
            1'b0,
            "OR"
        );

        apply_and_check(
            64'h0000_0000_0000_0000,
            64'h0000_0000_0000_0000,
            `ALU_OP_OR,
            64'h0000_0000_0000_0000,
            1'b0,
            "OR zero"
        );

        // ------ XNOR test ------
        $display("\n===== XNOR Tests =====");
        apply_and_check(
            64'hAAAA_AAAA_AAAA_AAAA,
            64'hAAAA_AAAA_AAAA_AAAA,
            `ALU_OP_XNOR,
            64'hFFFF_FFFF_FFFF_FFFF, // identical inputs -> all ones
            1'b0,
            "XNOR same"
        );

        apply_and_check(
            64'hAAAA_AAAA_AAAA_AAAA,
            64'h5555_5555_5555_5555,
            `ALU_OP_XNOR,
            64'h0000_0000_0000_0000, // bitwise complement -> all zeros
            1'b0,
            "XNOR complement"
        );

        // ------ CMP test ------
        $display("\n===== CMP Tests =====");
        apply_and_check(
            64'hDEAD_BEEF_DEAD_BEEF,
            64'hDEAD_BEEF_DEAD_BEEF,
            `ALU_OP_CMP,
            64'h0000_0000_0000_0001, // equal -> 1
            1'b0,
            "CMP equal"
        );

        apply_and_check(
            64'hDEAD_BEEF_DEAD_BEEF,
            64'h0000_0000_0000_0001,
            `ALU_OP_CMP,
            64'h0000_0000_0000_0000, // not equal -> 0
            1'b0,
            "CMP not equal"
        );

        // ------ LSL test ------
        $display("\n===== LSL Tests =====");
        apply_and_check(
            64'h0000_0000_0000_0001,
            64'h0000_0000_0000_0004, // shift left by 4
            `ALU_OP_LSL,
            64'h0000_0000_0000_0010,
            1'b0,
            "LSL by 4"
        );

        apply_and_check(
            64'h0000_0000_0000_0001,
            64'h0000_0000_0000_003F, // shift left by 63
            `ALU_OP_LSL,
            64'h8000_0000_0000_0000,
            1'b0,
            "LSL by 63"
        );

        apply_and_check(
            64'hFFFF_FFFF_FFFF_FFFF,
            64'h0000_0000_0000_0000, // shift left by 0
            `ALU_OP_LSL,
            64'hFFFF_FFFF_FFFF_FFFF,
            1'b0,
            "LSL by 0"
        );

        // ------ LSR test ------
        $display("\n===== LSR Tests =====");
        apply_and_check(
            64'h8000_0000_0000_0000,
            64'h0000_0000_0000_003F, // shift right by 63
            `ALU_OP_LSR,
            64'h0000_0000_0000_0001,
            1'b0,
            "LSR by 63"
        );

        apply_and_check(
            64'hFFFF_FFFF_FFFF_FF00,
            64'h0000_0000_0000_0008, // shift right by 8
            `ALU_OP_LSR,
            64'h00FF_FFFF_FFFF_FFFF,
            1'b0,
            "LSR by 8"
        );

        // ------ SBCMP test ------
        $display("\n===== SBCMP Tests =====");
        // Lower 32 bits match, upper 32 bits differ
        apply_and_check(
            64'hAAAA_AAAA_1234_5678,
            64'hBBBB_BBBB_1234_5678,
            `ALU_OP_SBCMP,
            64'h0000_0000_0000_0001, // lower 32 match -> 1
            1'b0,
            "SBCMP match"
        );

        // Lower 32 bits differ
        apply_and_check(
            64'hAAAA_AAAA_1234_5678,
            64'hAAAA_AAAA_1234_5679,
            `ALU_OP_SBCMP,
            64'h0000_0000_0000_0000, // lower 32 differ -> 0
            1'b0,
            "SBCMP no match"
        );

        // ------ Pipeline latency check ------
        // Verify the output is delayed by exactly one clock cycle
        $display("\n===== Pipeline Latency Test =====");
        @(negedge clk);
        A       = 64'h0000_0000_0000_000A;
        B       = 64'h0000_0000_0000_0014;
        aluctrl = `ALU_OP_ADD;

        // Immediately after driving, before next posedge, Z should
        // still hold the PREVIOUS result (SBCMP no match = 0)
        #1;
        test_num = test_num + 1;
        if (Z !== 64'h0000_0000_0000_0000) begin
            $display("FAIL test %0d [LATENCY pre-edge]: Z updated before clock edge (Z=0x%016h)", test_num, Z);
            fail_count = fail_count + 1;
        end else begin
            $display("PASS test %0d [LATENCY pre-edge]: Z still holds previous value", test_num);
            pass_count = pass_count + 1;
        end

        // Now wait for the posedge to latch new result
        @(posedge clk);
        #1;
        test_num = test_num + 1;
        if (Z !== 64'h0000_0000_0000_001E) begin
            $display("FAIL test %0d [LATENCY post-edge]: Z=0x%016h (exp 0x%016h)", test_num, Z, 64'h1E);
            fail_count = fail_count + 1;
        end else begin
            $display("PASS test %0d [LATENCY post-edge]: Z=0x%016h", test_num, Z);
            pass_count = pass_count + 1;
        end

        // ------ Summary ------
        $display("\n===================================");
        $display("  Total : %0d", test_num);
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        $display("===================================\n");

        $finish;
    end

    // Optional: dump waveform
    initial begin
        $dumpfile("sync_alu_tb.vcd");
        $dumpvars(0, sync_alu_tb);
    end

endmodule