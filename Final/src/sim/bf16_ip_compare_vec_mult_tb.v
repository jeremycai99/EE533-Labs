`timescale 1 ns / 1 ps

module bf16_ip_compare_vec_mult_tb;
    parameter MAX_VECS = 100000;

    reg clk = 1'b0;
    reg [15:0] a = 16'h0000;
    reg [15:0] b = 16'h0000;
    wire [15:0] ip_result;
    wire [15:0] model_result;

    reg [31:0] vecs [0:MAX_VECS-1];
    reg [1023:0] vecfile;
    integer nvec = 0;
    integer i;
    integer mismatches = 0;

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
            if (ip_result !== model_result) begin
                mismatches = mismatches + 1;
                if (mismatches <= 40)
                    $display("MISMATCH ann-mult a=%04h b=%04h ip=%04h model=%04h",
                             ta, tb, ip_result, model_result);
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("vecfile=%s", vecfile)) begin
            $display("ERROR: missing +vecfile=<path>");
            $finish;
        end
        if (!$value$plusargs("nvec=%d", nvec)) begin
            $display("ERROR: missing +nvec=<count>");
            $finish;
        end
        $readmemh(vecfile, vecs);
        repeat (4) @(posedge clk);
        for (i = 0; i < nvec; i = i + 1)
            check(vecs[i][31:16], vecs[i][15:0]);
        $display("SUMMARY ann-mult tests=%0d mismatches=%0d", nvec, mismatches);
        $finish;
    end
endmodule
