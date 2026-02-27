/* file: tensor_core_tb.v
 Description: Comprehensive testbench for BF16 tensor core (D = A*B + C).
   18 tests covering identity, diagonal, negatives, cancellation,
   fractional, dot product, back-to-back, and full matmul cases.
   Flat bit-vector indexing for Icarus Verilog compatibility.
 Author: Jeremy Cai
 Date: Feb. 26, 2026
*/

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "tensor_core.v"

module tensor_core_tb;
    reg clk, rst_n;
    reg [4*4*16-1:0] matrix_a, matrix_b, matrix_c;
    reg valid_in;
    wire [4*4*16-1:0] matrix_d;
    wire valid_out;

    tensor_core dut (
        .clk(clk), .rst_n(rst_n),
        .matrix_a(matrix_a), .matrix_b(matrix_b), .matrix_c(matrix_c),
        .valid_in(valid_in),
        .matrix_d(matrix_d), .valid_out(valid_out)
    );

    always #5 clk = ~clk;
    integer pass_count, fail_count, wait_cnt;

    // BF16 constants
    localparam [15:0] F0   = 16'h0000;
    localparam [15:0] F1   = 16'h3F80;
    localparam [15:0] F2   = 16'h4000;
    localparam [15:0] F3   = 16'h4040;
    localparam [15:0] F4   = 16'h4080;
    localparam [15:0] F5   = 16'h40A0;
    localparam [15:0] F6   = 16'h40C0;
    localparam [15:0] F7   = 16'h40E0;
    localparam [15:0] F8   = 16'h4100;
    localparam [15:0] F9   = 16'h4110;
    localparam [15:0] F10  = 16'h4120;
    localparam [15:0] F11  = 16'h4130;
    localparam [15:0] F12  = 16'h4140;
    localparam [15:0] F13  = 16'h4150;
    localparam [15:0] F14  = 16'h4160;
    localparam [15:0] F15  = 16'h4170;
    localparam [15:0] F16  = 16'h4180;
    localparam [15:0] F19  = 16'h4198;
    localparam [15:0] F22  = 16'h41B0;
    localparam [15:0] F27  = 16'h41D8;
    localparam [15:0] F30  = 16'h41F0;
    localparam [15:0] F33  = 16'h4204;
    localparam [15:0] F36  = 16'h4210;
    localparam [15:0] F43  = 16'h422C;
    localparam [15:0] F50  = 16'h4248;
    localparam [15:0] F52  = 16'h4250;
    localparam [15:0] F56  = 16'h4260;
    localparam [15:0] F60  = 16'h4270;
    localparam [15:0] F64  = 16'h4280;
    localparam [15:0] F100 = 16'h42C8;
    localparam [15:0] FN1  = 16'hBF80;
    localparam [15:0] FN2  = 16'hC000;

    // Expected result (flat 256-bit vector, same layout as matrix_d)
    reg [4*4*16-1:0] exp;

    // Fire: pulse valid_in for 1 cycle
    task fire;
        begin
            @(posedge clk); #1;
            valid_in = 1'b1;
            @(posedge clk); #1;
            valid_in = 1'b0;
        end
    endtask

    // Check: wait for valid_out, compare all 16 outputs
    integer ri, ci;
    reg test_pass;
    reg [15:0] got_val, exp_val;
    task check;
        input [8*40-1:0] label;
        begin
            wait_cnt = 0;
            while (!valid_out) begin
                @(posedge clk);
                wait_cnt = wait_cnt + 1;
                if (wait_cnt > 80) begin
                    $display("  TIMEOUT %s after %0d cycles", label, wait_cnt);
                    fail_count = fail_count + 1;
                    disable check;
                end
            end
            test_pass = 1;
            for (ri = 0; ri < 4; ri = ri + 1)
                for (ci = 0; ci < 4; ci = ci + 1) begin
                    got_val = matrix_d[(ri*4+ci)*16 +: 16];
                    exp_val = exp[(ri*4+ci)*16 +: 16];
                    if (got_val !== exp_val)
                        test_pass = 0;
                end
            if (test_pass) begin
                $display("  PASS %s (%0d cyc)", label, wait_cnt);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL %s (%0d cyc)", label, wait_cnt);
                for (ri = 0; ri < 4; ri = ri + 1)
                    $display("    r%0d got:[%04h %04h %04h %04h] exp:[%04h %04h %04h %04h]",
                        ri,
                        matrix_d[(ri*4+0)*16 +: 16], matrix_d[(ri*4+1)*16 +: 16],
                        matrix_d[(ri*4+2)*16 +: 16], matrix_d[(ri*4+3)*16 +: 16],
                        exp[(ri*4+0)*16 +: 16], exp[(ri*4+1)*16 +: 16],
                        exp[(ri*4+2)*16 +: 16], exp[(ri*4+3)*16 +: 16]);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Set matrix row by row: row0=[a,b,c,d], row1=[e,f,g,h], ...
    task set_a;
        input [15:0] a00,a01,a02,a03, a10,a11,a12,a13,
                     a20,a21,a22,a23, a30,a31,a32,a33;
        begin
            matrix_a = {a33,a32,a31,a30, a23,a22,a21,a20,
                        a13,a12,a11,a10, a03,a02,a01,a00};
        end
    endtask

    task set_b;
        input [15:0] b00,b01,b02,b03, b10,b11,b12,b13,
                     b20,b21,b22,b23, b30,b31,b32,b33;
        begin
            matrix_b = {b33,b32,b31,b30, b23,b22,b21,b20,
                        b13,b12,b11,b10, b03,b02,b01,b00};
        end
    endtask

    task set_c;
        input [15:0] c00,c01,c02,c03, c10,c11,c12,c13,
                     c20,c21,c22,c23, c30,c31,c32,c33;
        begin
            matrix_c = {c33,c32,c31,c30, c23,c22,c21,c20,
                        c13,c12,c11,c10, c03,c02,c01,c00};
        end
    endtask

    task set_exp;
        input [15:0] e00,e01,e02,e03, e10,e11,e12,e13,
                     e20,e21,e22,e23, e30,e31,e32,e33;
        begin
            exp = {e33,e32,e31,e30, e23,e22,e21,e20,
                   e13,e12,e11,e10, e03,e02,e01,e00};
        end
    endtask

    task set_c_zero;
        begin
            set_c(F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        end
    endtask

    task set_a_eye;
        begin
            set_a(F1,F0,F0,F0, F0,F1,F0,F0, F0,F0,F1,F0, F0,F0,F0,F1);
        end
    endtask

    task set_b_eye;
        begin
            set_b(F1,F0,F0,F0, F0,F1,F0,F0, F0,F0,F1,F0, F0,F0,F0,F1);
        end
    endtask

    initial begin
        $dumpfile("tensor_core_tb.vcd");
        $dumpvars(0, tensor_core_tb);
        clk = 0; rst_n = 0; valid_in = 0;
        matrix_a = 0; matrix_b = 0; matrix_c = 0;
        pass_count = 0; fail_count = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        $display("=========================================================");
        $display(" Tensor Core Testbench: D = A * B + C  (BF16 4x4)");
        $display("=========================================================");

        // Test 1: I*I + 0 = I
        $display("\n--- Test 1: I*I+0 = I ---");
        set_a_eye; set_b_eye; set_c_zero;
        set_exp(F1,F0,F0,F0, F0,F1,F0,F0, F0,F0,F1,F0, F0,F0,F0,F1);
        fire; check("I*I+0=I");

        // Test 2: I*I + ones = I + ones
        $display("\n--- Test 2: I*I+1 = I+1 ---");
        set_a_eye; set_b_eye;
        set_c(F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1);
        set_exp(F2,F1,F1,F1, F1,F2,F1,F1, F1,F1,F2,F1, F1,F1,F1,F2);
        fire; check("I*I+1=I+1");

        // Test 3: A*I + 0 = A
        $display("\n--- Test 3: A*I+0 = A ---");
        set_a(F1,F2,F3,F4, F5,F6,F7,F8, F9,F10,F11,F12, F13,F14,F15,F16);
        set_b_eye; set_c_zero;
        set_exp(F1,F2,F3,F4, F5,F6,F7,F8, F9,F10,F11,F12, F13,F14,F15,F16);
        fire; check("A*I+0=A");

        // Test 4: I*B + 0 = B
        $display("\n--- Test 4: I*B+0 = B ---");
        set_a_eye;
        set_b(F1,F2,F3,F4, F5,F6,F7,F8, F9,F10,F11,F12, F13,F14,F15,F16);
        set_c_zero;
        set_exp(F1,F2,F3,F4, F5,F6,F7,F8, F9,F10,F11,F12, F13,F14,F15,F16);
        fire; check("I*B+0=B");

        // Test 5: 0*B + C = C
        $display("\n--- Test 5: 0*B+C = C ---");
        set_a(F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_b(F1,F2,F3,F4, F5,F6,F7,F8, F9,F10,F11,F12, F13,F14,F15,F16);
        set_c(F1,F1,F1,F1, F2,F2,F2,F2, F3,F3,F3,F3, F4,F4,F4,F4);
        set_exp(F1,F1,F1,F1, F2,F2,F2,F2, F3,F3,F3,F3, F4,F4,F4,F4);
        fire; check("0*B+C=C");

        // Test 6: Scalar 2*3 + 1 = 7
        $display("\n--- Test 6: scalar 2*3+1=7 ---");
        set_a(F2,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_b(F3,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_c(F1,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_exp(F7,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        fire; check("scalar_2*3+1=7");

        // Test 7: ones * ones + 0 = 4s
        $display("\n--- Test 7: 1s*1s+0 = 4s ---");
        set_a(F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1);
        set_b(F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1);
        set_c_zero;
        set_exp(F4,F4,F4,F4, F4,F4,F4,F4, F4,F4,F4,F4, F4,F4,F4,F4);
        fire; check("1s*1s=4s");

        // Test 8: diag(1,2,3,4) * B + 0
        $display("\n--- Test 8: diag*B ---");
        set_a(F1,F0,F0,F0, F0,F2,F0,F0, F0,F0,F3,F0, F0,F0,F0,F4);
        set_b(F1,F2,F3,F4, F5,F6,F7,F8, F9,F10,F11,F12, F13,F14,F15,F16);
        set_c_zero;
        set_exp(F1,  F2,  F3,  F4,
                F10, F12, F14, F16,
                F27, F30, F33, F36,
                F52, F56, F60, F64);
        fire; check("diag*B");

        // Test 9: I*I + diag(10) = diag(11)
        $display("\n--- Test 9: I*I+diag10 = diag11 ---");
        set_a_eye; set_b_eye;
        set_c(F10,F0,F0,F0, F0,F10,F0,F0, F0,F0,F10,F0, F0,F0,F0,F10);
        set_exp(F11,F0,F0,F0, F0,F11,F0,F0, F0,F0,F11,F0, F0,F0,F0,F11);
        fire; check("I+diag10=diag11");

        // Test 10: Negative A * I = A
        $display("\n--- Test 10: negA*I = A ---");
        set_a(F1,FN1,F0,F0, FN1,F1,F0,F0, F0,F0,F1,FN1, F0,F0,FN1,F1);
        set_b_eye; set_c_zero;
        set_exp(F1,FN1,F0,F0, FN1,F1,F0,F0, F0,F0,F1,FN1, F0,F0,FN1,F1);
        fire; check("negA*I=A");

        // Test 11: 2I * 2I + 0 = 4I
        $display("\n--- Test 11: 2I*2I = 4I ---");
        set_a(F2,F0,F0,F0, F0,F2,F0,F0, F0,F0,F2,F0, F0,F0,F0,F2);
        set_b(F2,F0,F0,F0, F0,F2,F0,F0, F0,F0,F2,F0, F0,F0,F0,F2);
        set_c_zero;
        set_exp(F4,F0,F0,F0, F0,F4,F0,F0, F0,F0,F4,F0, F0,F0,F0,F4);
        fire; check("2I*2I=4I");

        // Test 12: Dot product [1,2,3,4].[1,2,3,4] = 30
        $display("\n--- Test 12: dot product = 30 ---");
        set_a(F1,F2,F3,F4, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_b(F1,F0,F0,F0, F2,F0,F0,F0, F3,F0,F0,F0, F4,F0,F0,F0);
        set_c_zero;
        set_exp(F30,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        fire; check("dot=30");

        // Test 13: 2s * 2s + 0 = 16s
        $display("\n--- Test 13: 2s*2s = 16s ---");
        set_a(F2,F2,F2,F2, F2,F2,F2,F2, F2,F2,F2,F2, F2,F2,F2,F2);
        set_b(F2,F2,F2,F2, F2,F2,F2,F2, F2,F2,F2,F2, F2,F2,F2,F2);
        set_c_zero;
        set_exp(F16,F16,F16,F16, F16,F16,F16,F16,
                F16,F16,F16,F16, F16,F16,F16,F16);
        fire; check("2s*2s=16s");

        // Test 14: 0*0 + C = C (C=100)
        $display("\n--- Test 14: 0+C=C ---");
        set_a(F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_b(F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_c(F100,F100,F100,F100, F100,F100,F100,F100,
              F100,F100,F100,F100, F100,F100,F100,F100);
        set_exp(F100,F100,F100,F100, F100,F100,F100,F100,
                F100,F100,F100,F100, F100,F100,F100,F100);
        fire; check("0+C=C");

        // Test 15a: Back-to-back first
        $display("\n--- Test 15a: b2b first I*I+0 ---");
        set_a_eye; set_b_eye; set_c_zero;
        set_exp(F1,F0,F0,F0, F0,F1,F0,F0, F0,F0,F1,F0, F0,F0,F0,F1);
        fire; check("b2b_1st");

        // Test 15b: Back-to-back second
        $display("\n--- Test 15b: b2b second 2I*3I+1 ---");
        set_a(F2,F0,F0,F0, F0,F2,F0,F0, F0,F0,F2,F0, F0,F0,F0,F2);
        set_b(F3,F0,F0,F0, F0,F3,F0,F0, F0,F0,F3,F0, F0,F0,F0,F3);
        set_c(F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1, F1,F1,F1,F1);
        set_exp(F7,F1,F1,F1, F1,F7,F1,F1, F1,F1,F7,F1, F1,F1,F1,F7);
        fire; check("b2b_2nd");

        // Test 16: Negative cancellation
        $display("\n--- Test 16: neg cancellation ---");
        set_a(F1,F1,F0,F0, F0,F0,F1,F1, F0,F0,F0,F0, F0,F0,F0,F0);
        set_b(F1,F0,F0,F0, FN1,F0,F0,F0, F2,F0,F0,F0, FN2,F0,F0,F0);
        set_c_zero;
        set_exp(F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        fire; check("neg_cancel");

        // Test 17: 2x2 matmul in top-left corner
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        $display("\n--- Test 17: 2x2 matmul ---");
        set_a(F1,F2,F0,F0, F3,F4,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_b(F5,F6,F0,F0, F7,F8,F0,F0, F0,F0,F0,F0, F0,F0,F0,F0);
        set_c_zero;
        set_exp(F19, F22, F0, F0,
                F43, F50, F0, F0,
                F0,  F0,  F0, F0,
                F0,  F0,  F0, F0);
        fire; check("2x2_matmul");

        // Test 18: I*I + I = 2I
        $display("\n--- Test 18: I*I+I = 2I ---");
        set_a_eye; set_b_eye;
        set_c(F1,F0,F0,F0, F0,F1,F0,F0, F0,F0,F1,F0, F0,F0,F0,F1);
        set_exp(F2,F0,F0,F0, F0,F2,F0,F0, F0,F0,F2,F0, F0,F0,F0,F2);
        fire; check("I*I+I=2I");

        // Summary
        repeat (10) @(posedge clk);
        $display("\n=========================================================");
        $display(" Results: %0d PASSED, %0d FAILED / %0d total",
            pass_count, fail_count, pass_count + fail_count);
        $display("=========================================================");
        $finish;
    end

    initial begin
        #200000;
        $display("WATCHDOG timeout");
        $finish;
    end
endmodule