`timescale 1ns / 1ps

`include "pplbf16addsub.v"

module pplbf16addsub_tb;

    reg clk;
    reg rst_n;
    reg [15:0] operand_a;
    reg [15:0] operand_b;
    reg sub;
    reg valid_in;
    wire [15:0] result;
    wire valid_out;

    pplbf16addsub dut (
        .clk(clk),
        .rst_n(rst_n),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .sub(sub),
        .valid_in(valid_in),
        .result(result),
        .valid_out(valid_out)
    );

    always #5 clk = ~clk;

    integer pass_count, fail_count, test_num;

    reg [15:0] expect_queue [0:31];
    reg [8*28-1:0] name_queue [0:31];
    reg [4:0] queue_wr, queue_rd;

    task push_test;
        input [15:0] a;
        input [15:0] b;
        input is_sub;
        input [15:0] expected;
        input [8*28-1:0] name;
        begin
            @(posedge clk);
            #1;
            operand_a = a;
            operand_b = b;
            sub = is_sub;
            valid_in = 1'b1;
            expect_queue[queue_wr] = expected;
            name_queue[queue_wr] = name;
            queue_wr = queue_wr + 5'd1;
        end
    endtask

    always @(negedge clk) begin
        if (valid_out) begin
            test_num = test_num + 1;
            if (result === expect_queue[queue_rd]) begin
                $display("PASS %2d: %-28s | result=0x%04h",
                    test_num, name_queue[queue_rd], result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL %2d: %-28s | result=0x%04h (expected 0x%04h)",
                    test_num, name_queue[queue_rd], result, expect_queue[queue_rd]);
                fail_count = fail_count + 1;
            end
            queue_rd = queue_rd + 5'd1;
        end
    end

    initial begin
        $dumpfile("pplbf16addsub_tb.vcd");
        $dumpvars(0, pplbf16addsub_tb);

        clk = 0; rst_n = 0; valid_in = 0;
        operand_a = 0; operand_b = 0; sub = 0;
        pass_count = 0; fail_count = 0; test_num = 0;
        queue_wr = 0; queue_rd = 0;

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("=========================================================");
        $display(" Pipelined BF16 Add/Sub (4-stage pipeline) Testbench");
        $display("=========================================================");

        // Addition
        push_test(16'h3F80, 16'h3F80, 0, 16'h4000, "1.0 + 1.0 = 2.0");
        push_test(16'h3F80, 16'h4000, 0, 16'h4040, "1.0 + 2.0 = 3.0");
        push_test(16'h3F00, 16'h3F00, 0, 16'h3F80, "0.5 + 0.5 = 1.0");
        push_test(16'h3FC0, 16'h4020, 0, 16'h4080, "1.5 + 2.5 = 4.0");
        push_test(16'hBF80, 16'hBF80, 0, 16'hC000, "-1.0 + -1.0 = -2.0");

        // Subtraction
        push_test(16'h4040, 16'h3F80, 1, 16'h4000, "3.0 - 1.0 = 2.0");
        push_test(16'h4000, 16'h4000, 1, 16'h0000, "2.0 - 2.0 = 0.0");
        push_test(16'h3F80, 16'h4000, 1, 16'hBF80, "1.0 - 2.0 = -1.0");
        push_test(16'hBF80, 16'hBF80, 1, 16'h0000, "-1.0 - -1.0 = 0.0");

        // Zero
        push_test(16'h0000, 16'h0000, 0, 16'h0000, "0 + 0 = 0");
        push_test(16'h0000, 16'h3F80, 0, 16'h3F80, "0 + 1.0 = 1.0");
        push_test(16'h3F80, 16'h0000, 0, 16'h3F80, "1.0 + 0 = 1.0");
        push_test(16'h8000, 16'h8000, 0, 16'h8000, "-0 + -0 = -0");

        // Infinity
        push_test(16'h7F80, 16'h3F80, 0, 16'h7F80, "inf + 1.0 = inf");
        push_test(16'h7F80, 16'h7F80, 1, 16'h7FC0, "inf - inf = NaN");
        push_test(16'hFF80, 16'hFF80, 0, 16'hFF80, "-inf + -inf = -inf");

        // NaN
        push_test(16'h7FC0, 16'h3F80, 0, 16'h7FC0, "NaN + 1.0 = NaN");
        push_test(16'h3F80, 16'h7FC0, 0, 16'h7FC0, "1.0 + NaN = NaN");

        // Large exp diff
        push_test(16'h4380, 16'h3E80, 0, 16'h4380, "256 + 0.25 = 256");

        @(posedge clk);
        #1;
        valid_in = 1'b0;
        repeat (10) @(posedge clk);

        $display("=========================================================");
        $display(" Results: %0d PASSED, %0d FAILED / %0d total",
            pass_count, fail_count, test_num);
        $display("=========================================================");
        $finish;
    end

endmodule