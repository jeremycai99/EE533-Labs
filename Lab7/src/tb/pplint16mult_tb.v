/* file: pplint16mult_tb.v
 Description: Testbench for 2-stage pipelined int16 multiplier.
   Directed tests + random test with golden model comparison.
   Latency: 2 cycles (result appears 2 clocks after input).
 Author: Jeremy Cai
 Date: Feb. 26, 2026
*/

`timescale 1ns / 1ps

`include "pplint16mult.v"

module pplint16mult_tb;
    reg clk, rst_n;
    reg [15:0] a, b;
    wire [15:0] result;

    pplint16mult dut (
        .clk(clk), .rst_n(rst_n),
        .a(a), .b(b),
        .result(result)
    );

    always #5 clk = ~clk;
    integer pass_count, fail_count;

    // Pipeline tracking: 2-stage delay for expected values
    reg [15:0] exp_pipe [0:1];
    reg [15:0] a_pipe [0:1];
    reg [15:0] b_pipe [0:1];
    reg valid_pipe [0:1];
    wire [15:0] expected = exp_pipe[1];
    wire pipe_valid = valid_pipe[1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exp_pipe[0]   <= 16'd0; exp_pipe[1]   <= 16'd0;
            a_pipe[0]     <= 16'd0; a_pipe[1]     <= 16'd0;
            b_pipe[0]     <= 16'd0; b_pipe[1]     <= 16'd0;
            valid_pipe[0] <= 1'b0;  valid_pipe[1] <= 1'b0;
        end else begin
            exp_pipe[0]   <= a * b;    // golden model: Verilog * gives 32-bit, [15:0] truncates
            a_pipe[0]     <= a;
            b_pipe[0]     <= b;
            valid_pipe[0] <= 1'b1;
            exp_pipe[1]   <= exp_pipe[0];
            a_pipe[1]     <= a_pipe[0];
            b_pipe[1]     <= b_pipe[0];
            valid_pipe[1] <= valid_pipe[0];
        end
    end

    // Check on every cycle once pipeline is full
    always @(posedge clk) begin
        if (pipe_valid && rst_n) begin
            if (result === expected)
                pass_count = pass_count + 1;
            else begin
                $display("  FAIL: %0d * %0d = %0d (0x%04h), expected %0d (0x%04h)",
                    a_pipe[1], b_pipe[1], result, result, expected, expected);
                fail_count = fail_count + 1;
            end
        end
    end

    // Directed test helper: set inputs, wait 1 cycle
    task drive;
        input [15:0] ta, tb;
        begin
            a = ta; b = tb;
            @(posedge clk); #1;
        end
    endtask

    integer i;
    reg [31:0] rand_a, rand_b;

    initial begin
        $dumpfile("pplint16mult_tb.vcd");
        $dumpvars(0, pplint16mult_tb);
        clk = 0; rst_n = 0;
        a = 0; b = 0;
        pass_count = 0; fail_count = 0;

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        $display("=========================================================");
        $display(" int16mult Testbench (2-stage pipeline, lower 16 bits)");
        $display("=========================================================");

        // --- Directed tests ---
        $display("\n--- Directed tests ---");

        // Basic
        drive(16'd0,     16'd0);       // 0 * 0 = 0
        drive(16'd1,     16'd1);       // 1 * 1 = 1
        drive(16'd2,     16'd3);       // 2 * 3 = 6
        drive(16'd7,     16'd11);      // 7 * 11 = 77
        drive(16'd100,   16'd200);     // 100 * 200 = 20000
        drive(16'd255,   16'd255);     // 255 * 255 = 65025

        // Powers of 2
        drive(16'd1,     16'd32768);   // 1 * 2^15 = 32768
        drive(16'd2,     16'd16384);   // 2 * 2^14 = 32768
        drive(16'd256,   16'd256);     // 2^8 * 2^8 = 2^16 → low 16 = 0
        drive(16'd1024,  16'd64);      // 2^10 * 2^6 = 2^16 → 0

        // Overflow (wraps mod 2^16)
        drive(16'hFFFF,  16'hFFFF);    // 65535*65535 = 0xFFFE0001 → 0x0001
        drive(16'hFFFF,  16'd2);       // 65535*2 = 0x1FFFE → 0xFFFE
        drive(16'h8000,  16'd2);       // 32768*2 = 65536 → 0
        drive(16'hFFFF,  16'd1);       // 65535*1 = 65535
        drive(16'd1,     16'hFFFF);    // commutative check

        // One operand zero
        drive(16'd0,     16'hFFFF);    // 0 * max = 0
        drive(16'hFFFF,  16'd0);       // max * 0 = 0
        drive(16'd0,     16'd1);       // 0 * 1 = 0

        // Signed interpretation (lower 16 identical for signed/unsigned)
        // -1 * -1 = 1 (0xFFFF * 0xFFFF = ...0001)
        drive(16'hFFFF,  16'hFFFF);
        // -1 * 2 = -2 (0xFFFF * 2 = 0xFFFE)
        drive(16'hFFFF,  16'd2);
        // -128 * 256 = -32768 (0xFF80 * 0x0100 = 0x8000)
        drive(16'hFF80,  16'h0100);

        // Square numbers
        drive(16'd100,   16'd100);     // 10000
        drive(16'd181,   16'd181);     // 32761

        // Alternating bits
        drive(16'hAAAA,  16'h5555);
        drive(16'h5555,  16'hAAAA);
        drive(16'hAAAA,  16'hAAAA);
        drive(16'h5555,  16'h5555);

        // Walking ones
        drive(16'h0001,  16'hFFFF);
        drive(16'h0002,  16'hFFFF);
        drive(16'h0004,  16'hFFFF);
        drive(16'h0008,  16'hFFFF);
        drive(16'h8000,  16'hFFFF);

        // Flush pipeline
        drive(16'd0, 16'd0);
        drive(16'd0, 16'd0);
        drive(16'd0, 16'd0);

        $display("  Directed: %0d passed, %0d failed",
            pass_count, fail_count);

        // --- Random tests ---
        $display("\n--- Random tests (1000 vectors) ---");
        for (i = 0; i < 1000; i = i + 1) begin
            rand_a = $random;
            rand_b = $random;
            drive(rand_a[15:0], rand_b[15:0]);
        end

        // Flush pipeline
        drive(16'd0, 16'd0);
        drive(16'd0, 16'd0);
        drive(16'd0, 16'd0);

        repeat (5) @(posedge clk);
        $display("\n=========================================================");
        $display(" Results: %0d PASSED, %0d FAILED / %0d total",
            pass_count, fail_count, pass_count + fail_count);
        $display("=========================================================");
        $finish;
    end

    initial begin
        #60000;
        $display("WATCHDOG timeout");
        $finish;
    end
endmodule