`timescale 1 ns / 1 ps

module bf16_ip_compare_addsub_tb;
    reg clk = 1'b0;
    reg [5:0] operation = 6'b000000;
    reg [15:0] a = 16'h0000;
    reg [15:0] b = 16'h0000;
    wire [15:0] ip_result;
    wire [15:0] model_result;

    integer i;
    integer mismatches = 0;
    integer tests = 0;

    bf16addsub u_ip (
        .clk(clk),
        .operation(operation),
        .a(a),
        .b(b),
        .result(ip_result)
    );

    test_bf16addsub u_model (
        .clk(clk),
        .operation(operation),
        .a(a),
        .b(b),
        .result(model_result)
    );

    always #5 clk = ~clk;

    task check;
        input [15:0] ta;
        input [15:0] tb;
        input tsub;
        begin
            @(negedge clk);
            a = ta;
            b = tb;
            operation = {5'b00000, tsub};
            @(posedge clk);
            @(posedge clk);
            #1;
            tests = tests + 1;
            if (ip_result !== model_result) begin
                mismatches = mismatches + 1;
                if (mismatches <= 60)
                    $display("MISMATCH addsub sub=%0d a=%04h b=%04h ip=%04h model=%04h",
                             tsub, ta, tb, ip_result, model_result);
            end
        end
    endtask

    task check_expect;
        input [15:0] ta;
        input [15:0] tb;
        input tsub;
        input [15:0] expected;
        begin
            @(negedge clk);
            a = ta;
            b = tb;
            operation = {5'b00000, tsub};
            @(posedge clk);
            @(posedge clk);
            #1;
            tests = tests + 1;
            if (ip_result !== expected || model_result !== expected) begin
                mismatches = mismatches + 1;
                $display("MISMATCH addsub-exp sub=%0d a=%04h b=%04h ip=%04h model=%04h exp=%04h",
                         tsub, ta, tb, ip_result, model_result, expected);
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);

        check_expect(16'h4000, 16'h4040, 1'b0, 16'h40a0); // 2 + 3
        check_expect(16'h40c0, 16'h3f80, 1'b0, 16'h40e0); // 6 + 1
        check_expect(16'h40c0, 16'h3f80, 1'b1, 16'h40a0); // 6 - 1
        check_expect(16'hbf65, 16'h3e1f, 1'b0, 16'hbf3d); // ANN drift pair
        check(16'h0001, 16'h3f80, 1'b0); // subnormal behavior
        check(16'h007f, 16'h0001, 1'b0);

        for (i = 0; i < 20000; i = i + 1) begin
            check(((i * 16'h43fd) + 16'h1234) & 16'hffff,
                  ((i * 16'h2c57) + 16'h89ab) & 16'hffff,
                  i[0]);
        end

        $display("SUMMARY addsub tests=%0d mismatches=%0d", tests, mismatches);
        $finish;
    end
endmodule
