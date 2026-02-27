`timescale 1ns / 1ps

`include "int16hca.v"

module int16hca_tb;

    reg [15:0] a, b;
    reg cin;
    wire [15:0] sum;
    wire cout;

    int16hca dut (
        .a(a),
        .b(b),
        .cin(cin),
        .sum(sum),
        .cout(cout)
    );

    integer pass_count, fail_count, i;
    reg [16:0] expected; // 17-bit to capture carry

    task check;
        input [8*20-1:0] label;
        begin
            #1;
            expected = {1'b0, a} + {1'b0, b} + {16'b0, cin};
            if ({cout, sum} === expected[16:0]) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                if (fail_count <= 20)
                    $display("FAIL %s: 0x%04h + 0x%04h + %0b = {%b, 0x%04h} (expected {%b, 0x%04h})",
                        label, a, b, cin, cout, sum, expected[16], expected[15:0]);
            end
        end
    endtask

    initial begin
        $dumpfile("hca16_tb.vcd");
        $dumpvars(0, int16hca_tb);

        pass_count = 0;
        fail_count = 0;

        $display("=========================================================");
        $display(" 16-bit Han-Carlson Adder Testbench (cin/cout)");
        $display("=========================================================");

        // --- ADD edge cases (cin=0) ---
        $display("--- ADD (cin=0) ---");
        a = 16'h0000; b = 16'h0000; cin = 0; check("0+0");
        a = 16'hFFFF; b = 16'h0000; cin = 0; check("FFFF+0");
        a = 16'hFFFF; b = 16'h0001; cin = 0; check("FFFF+1");
        a = 16'hFFFF; b = 16'hFFFF; cin = 0; check("FFFF+FFFF");
        a = 16'h7FFF; b = 16'h0001; cin = 0; check("7FFF+1");
        a = 16'hAAAA; b = 16'h5555; cin = 0; check("AAAA+5555");
        a = 16'h8000; b = 16'h8000; cin = 0; check("8000+8000");

        // --- ADD with cin=1 ---
        $display("--- ADD (cin=1) ---");
        a = 16'h0000; b = 16'h0000; cin = 1; check("0+0+1");
        a = 16'hFFFF; b = 16'h0000; cin = 1; check("FFFF+0+1");
        a = 16'hFFFF; b = 16'hFFFF; cin = 1; check("FFFF+FFFF+1");
        a = 16'h7FFF; b = 16'h0000; cin = 1; check("7FFF+0+1");

        // --- SUB (a + ~b + 1) via cin=1 ---
        $display("--- SUB (a + ~b + cin=1) ---");
        // 5 - 3 = 2
        a = 16'h0005; b = ~16'h0003; cin = 1; check("5-3");
        expected = 17'h0002;
        // 0 - 1 = FFFF (-1)
        a = 16'h0000; b = ~16'h0001; cin = 1; check("0-1");
        // 100 - 100 = 0
        a = 16'h0064; b = ~16'h0064; cin = 1; check("100-100");
        // 1 - 0 = 1
        a = 16'h0001; b = ~16'h0000; cin = 1; check("1-0");
        // 8000 - 7FFF = 1
        a = 16'h8000; b = ~16'h7FFF; cin = 1; check("8000-7FFF");

        $display("Directed tests: %0d passed, %0d failed", pass_count, fail_count);

        // --- Random ADD (cin=0) ---
        $display("Running 50 random ADD vectors...");
        for (i = 0; i < 50; i = i + 1) begin
            a = $random; b = $random; cin = 0;
            check("rand_add");
        end

        // --- Random ADD (cin=1) ---
        $display("Running 50 random ADD+cin vectors...");
        for (i = 0; i < 50; i = i + 1) begin
            a = $random; b = $random; cin = 1;
            check("rand_add_cin");
        end

        // --- Random SUB (a + ~b + 1) ---
        $display("Running 50 random SUB vectors...");
        for (i = 0; i < 50; i = i + 1) begin
            a = $random; b = ~($random); cin = 1;
            check("rand_sub");
        end

        $display("=========================================================");
        $display(" Results: %0d PASSED, %0d FAILED",
            pass_count, fail_count);
        $display("=========================================================");
        $finish;
    end

endmodule