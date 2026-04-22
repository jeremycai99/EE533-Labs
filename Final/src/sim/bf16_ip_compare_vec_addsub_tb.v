`timescale 1 ns / 1 ps

module bf16_ip_compare_vec_addsub_tb;
    parameter MAX_VECS = 100000;

    reg clk = 1'b0;
    reg [5:0] operation = 6'b000000;
    reg [15:0] a = 16'h0000;
    reg [15:0] b = 16'h0000;
    wire [15:0] ip_result;
    wire [15:0] model_result;

    reg [32:0] vecs [0:MAX_VECS-1];
    reg [1023:0] vecfile;
    integer nvec = 0;
    integer i;
    integer mismatches = 0;

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
            if (ip_result !== model_result) begin
                mismatches = mismatches + 1;
                if (mismatches <= 60)
                    $display("MISMATCH ann-addsub sub=%0d a=%04h b=%04h ip=%04h model=%04h",
                             tsub, ta, tb, ip_result, model_result);
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
            check(vecs[i][31:16], vecs[i][15:0], vecs[i][32]);
        $display("SUMMARY ann-addsub tests=%0d mismatches=%0d", nvec, mismatches);
        $finish;
    end
endmodule
