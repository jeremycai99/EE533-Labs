`timescale 1 ns / 1 ps

module bf16_ip_compare_mult_tb;
    reg clk = 1'b0;
    reg [15:0] a = 16'h0000;
    reg [15:0] b = 16'h0000;
    wire [15:0] ip_result;
    wire [15:0] model_result;

    integer i;
    integer mismatches = 0;
    integer tests = 0;

    bf16mult u_ip (
        .clk(clk),
        .a(a),
        .b(b),
        .result(ip_result)
    );

    test_bf16mult u_model (
        .clk(clk),
        .a(a),
        .b(b),
        .result(model_result)
    );

    always #5 clk = ~clk;

    task check;
        input [15:0] ta;
        input [15:0] tb;
        begin
            @(negedge clk);
            a = ta;
            b = tb;
            @(posedge clk);
            #1;
            tests = tests + 1;
            if (ip_result !== model_result) begin
                mismatches = mismatches + 1;
                if (mismatches <= 40)
                    $display("MISMATCH mult a=%04h b=%04h ip=%04h model=%04h",
                             ta, tb, ip_result, model_result);
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);

        check(16'h3f80, 16'h3f80); // 1 * 1
        check(16'h4000, 16'h4040); // 2 * 3
        check(16'hbf80, 16'h3f80); // -1 * 1
        check(16'h0001, 16'h3f80); // subnormal behavior
        check(16'h007f, 16'h3f80);

        for (i = 0; i < 20000; i = i + 1) begin
            check(((i * 16'h43fd) + 16'h1234) & 16'hffff,
                  ((i * 16'h2c57) + 16'h89ab) & 16'hffff);
        end

        $display("SUMMARY mult tests=%0d mismatches=%0d", tests, mismatches);
        $finish;
    end
endmodule
