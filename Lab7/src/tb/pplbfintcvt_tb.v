/* file: pplbfintcvt_tb.v
 Description: Testbench for bidirectional BF16 ↔ INT16 converter.
   Tests both directions (DT=0: int16→bf16, DT=1: bf16→int16).
   Directed cases: zero, small/large values, negatives, powers of 2,
   edge cases (NaN, ±Inf, denormals, overflow, saturation).
   Random tests with golden model.
   Global cycle counter for each test.
 Author: Jeremy Cai
 Date: Feb. 27, 2026
*/

`timescale 1ns / 1ps
`include "gpu_define.v"
`include "pplbfintcvt.v"

module pplbfintcvt_tb;
    reg         clk, rst_n;
    reg         dt;
    reg  [15:0] in;
    wire [15:0] out;

    pplbfintcvt dut (
        .clk(clk), .rst_n(rst_n),
        .dt(dt), .in(in), .out(out)
    );

    always #5 clk = ~clk;

    // Global cycle counter
    integer cycle;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cycle <= 0;
        else        cycle <= cycle + 1;
    end

    integer pass_count, fail_count;
    integer t_start;
    reg verbose;

    // Failure log — stores first 16 failures, reprinted at end
    reg [8*40-1:0] fail_labels [0:15];
    reg [15:0]     fail_in     [0:15];
    reg [15:0]     fail_got    [0:15];
    reg [15:0]     fail_exp    [0:15];
    reg            fail_dir    [0:15];
    integer        fail_cyc    [0:15];

    // =====================================================================
    // Test task: drive input, wait 2 posedges, check output
    // =====================================================================
    task test_cvt;
        input        dir;          // 0=i2f, 1=f2i
        input [15:0] val;
        input [15:0] expected;
        input [8*40-1:0] label;
        begin
            @(posedge clk); #1;
            t_start = cycle;
            dt = dir;
            in = val;
            @(posedge clk); #1;     // stage 1 latches
            in = 16'd0; dt = 1'b0;  // clear (free-running, don't care)
            @(posedge clk); #1;     // stage 2 latches → output valid
            if (out !== expected) begin
                $display(">>> [cyc %0d->%0d] FAIL %s: dt=%0b in=%04h got=%04h exp=%04h <<<",
                    t_start, cycle, label, dir, val, out, expected);
                if (fail_count < 16) begin
                    fail_labels[fail_count] = label;
                    fail_in[fail_count]     = val;
                    fail_got[fail_count]    = out;
                    fail_exp[fail_count]    = expected;
                    fail_dir[fail_count]    = dir;
                    fail_cyc[fail_count]    = t_start;
                end
                fail_count = fail_count + 1;
            end else begin
                if (verbose)
                    $display("[cyc %0d->%0d] PASS %s: in=%04h out=%04h",
                        t_start, cycle, label, val, out);
                pass_count = pass_count + 1;
            end
        end
    endtask

    // =====================================================================
    // INT16 → BF16 golden model (for random tests)
    //   Mirrors RTL: negate → priority encode → normalize → assemble
    // =====================================================================
    reg        gm_sign;
    reg [15:0] gm_mag;
    reg [3:0]  gm_lzc;
    reg [15:0] gm_norm;
    reg [7:0]  gm_exp;
    reg [6:0]  gm_man;
    reg [15:0] gm_i2f;

    task golden_i2f;
        input [15:0] val;
        begin
            gm_sign = val[15];
            gm_mag  = gm_sign ? (~val + 16'd1) : val;
            casez (gm_mag)
                16'b1???????????????: gm_lzc = 4'd15;
                16'b01??????????????: gm_lzc = 4'd14;
                16'b001?????????????: gm_lzc = 4'd13;
                16'b0001????????????: gm_lzc = 4'd12;
                16'b00001???????????: gm_lzc = 4'd11;
                16'b000001??????????: gm_lzc = 4'd10;
                16'b0000001?????????: gm_lzc = 4'd9;
                16'b00000001????????: gm_lzc = 4'd8;
                16'b000000001???????: gm_lzc = 4'd7;
                16'b0000000001??????: gm_lzc = 4'd6;
                16'b00000000001?????: gm_lzc = 4'd5;
                16'b000000000001????: gm_lzc = 4'd4;
                16'b0000000000001???: gm_lzc = 4'd3;
                16'b00000000000001??: gm_lzc = 4'd2;
                16'b000000000000001?: gm_lzc = 4'd1;
                16'b0000000000000001: gm_lzc = 4'd0;
                default:              gm_lzc = 4'd0;
            endcase
            gm_norm = gm_mag << (4'd15 - gm_lzc);
            gm_exp  = 8'd127 + {4'd0, gm_lzc};
            gm_man  = gm_norm[14:8];
            gm_i2f  = (val == 16'd0) ? 16'h0000
                                      : {gm_sign, gm_exp, gm_man};
        end
    endtask

    // =====================================================================
    // BF16 → INT16 golden model
    // =====================================================================
    reg        gm_f_sign;
    reg [7:0]  gm_f_exp;
    reg [6:0]  gm_f_man;
    reg [15:0] gm_f_full;
    reg [7:0]  gm_f_rshift;
    reg [15:0] gm_f_shifted;
    reg [15:0] gm_f2i;

    task golden_f2i;
        input [15:0] val;
        begin
            gm_f_sign = val[15];
            gm_f_exp  = val[14:7];
            gm_f_man  = val[6:0];
            gm_f_full = {1'b1, gm_f_man, 8'b0};
            gm_f_rshift = 8'd142 - gm_f_exp;
            gm_f_shifted = gm_f_full >> gm_f_rshift[3:0];

            if (gm_f_exp == 8'd0)                          // zero / denormal
                gm_f2i = 16'd0;
            else if (gm_f_exp == 8'hFF && gm_f_man != 0)   // NaN
                gm_f2i = 16'd0;
            else if (gm_f_exp == 8'hFF && !gm_f_sign)      // +Inf
                gm_f2i = 16'h7FFF;
            else if (gm_f_exp == 8'hFF && gm_f_sign)       // -Inf
                gm_f2i = 16'h8000;
            else if (gm_f_exp < 8'd127)                    // |val| < 1
                gm_f2i = 16'd0;
            else if (gm_f_exp > 8'd142 && !gm_f_sign)      // positive overflow
                gm_f2i = 16'h7FFF;
            else if (gm_f_exp > 8'd142 && gm_f_sign)       // negative overflow
                gm_f2i = 16'h8000;
            else if (!gm_f_sign && gm_f_shifted > 16'd32767)
                gm_f2i = 16'h7FFF;
            else if (gm_f_sign && gm_f_shifted > 16'd32768)
                gm_f2i = 16'h8000;
            else if (gm_f_sign)
                gm_f2i = ~gm_f_shifted + 16'd1;
            else
                gm_f2i = gm_f_shifted;
        end
    endtask

    // =====================================================================
    // Main test
    // =====================================================================
    integer i;
    reg [31:0] rand_v;

    initial begin
        $dumpfile("pplbfintcvt_tb.vcd");
        $dumpvars(0, pplbfintcvt_tb);
        clk = 0; rst_n = 0; dt = 0; in = 0;
        pass_count = 0; fail_count = 0;
        verbose = 1;

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        $display("=========================================================");
        $display(" pplbfintcvt Testbench — Bidirectional BF16 ↔ INT16");
        $display("=========================================================");

        // =============================================================
        // INT16 → BF16 (DT=0) directed tests
        // =============================================================
        $display("\n=== INT16 → BF16 (DT=0) ===");

        $display("\n--- Zero ---");
        test_cvt(0, 16'd0,      16'h0000, "i2f_zero");

        $display("\n--- Small positives ---");
        test_cvt(0, 16'd1,      16'h3F80, "i2f_1");       // 1.0
        test_cvt(0, 16'd2,      16'h4000, "i2f_2");       // 2.0
        test_cvt(0, 16'd3,      16'h4040, "i2f_3");       // 3.0
        test_cvt(0, 16'd7,      16'h40E0, "i2f_7");       // 7.0
        test_cvt(0, 16'd10,     16'h4120, "i2f_10");      // 10.0
        test_cvt(0, 16'd42,     16'h4228, "i2f_42");      // 42.0
        test_cvt(0, 16'd100,    16'h42C8, "i2f_100");     // 100.0
        test_cvt(0, 16'd127,    16'h42FE, "i2f_127");     // 127.0
        test_cvt(0, 16'd128,    16'h4300, "i2f_128");     // 128.0
        test_cvt(0, 16'd255,    16'h437F, "i2f_255");     // 255.0
        test_cvt(0, 16'd256,    16'h4380, "i2f_256");     // 256.0

        $display("\n--- Powers of 2 ---");
        test_cvt(0, 16'd4,      16'h4080, "i2f_4");
        test_cvt(0, 16'd8,      16'h4100, "i2f_8");
        test_cvt(0, 16'd16,     16'h4180, "i2f_16");
        test_cvt(0, 16'd32,     16'h4200, "i2f_32");
        test_cvt(0, 16'd64,     16'h4280, "i2f_64");
        test_cvt(0, 16'd512,    16'h4400, "i2f_512");
        test_cvt(0, 16'd1024,   16'h4480, "i2f_1024");
        test_cvt(0, 16'd4096,   16'h4580, "i2f_4096");
        test_cvt(0, 16'd8192,   16'h4600, "i2f_8192");
        test_cvt(0, 16'd16384,  16'h4680, "i2f_16384");

        $display("\n--- Negatives ---");
        test_cvt(0, -16'd1,     16'hBF80, "i2f_-1");      // -1.0
        test_cvt(0, -16'd2,     16'hC000, "i2f_-2");      // -2.0
        test_cvt(0, -16'd10,    16'hC120, "i2f_-10");     // -10.0
        test_cvt(0, -16'd100,   16'hC2C8, "i2f_-100");    // -100.0
        test_cvt(0, -16'd128,   16'hC300, "i2f_-128");    // -128.0
        test_cvt(0, -16'd256,   16'hC380, "i2f_-256");    // -256.0

        $display("\n--- Extremes ---");
        test_cvt(0, 16'h7FFF,   16'h46FF, "i2f_32767");   // 32767 → truncated
        test_cvt(0, 16'h8000,   16'hC700, "i2f_-32768");  // -32768.0

        // =============================================================
        // BF16 → INT16 (DT=1) directed tests
        // =============================================================
        $display("\n=== BF16 → INT16 (DT=1) ===");

        $display("\n--- Zero ---");
        test_cvt(1, 16'h0000,   16'd0,     "f2i_+0");
        test_cvt(1, 16'h8000,   16'd0,     "f2i_-0");

        $display("\n--- Small positives ---");
        test_cvt(1, 16'h3F80,   16'd1,     "f2i_1.0");
        test_cvt(1, 16'h4000,   16'd2,     "f2i_2.0");
        test_cvt(1, 16'h4040,   16'd3,     "f2i_3.0");
        test_cvt(1, 16'h4120,   16'd10,    "f2i_10.0");
        test_cvt(1, 16'h42C8,   16'd100,   "f2i_100.0");
        test_cvt(1, 16'h4300,   16'd128,   "f2i_128.0");
        test_cvt(1, 16'h4380,   16'd256,   "f2i_256.0");

        $display("\n--- Powers of 2 ---");
        test_cvt(1, 16'h4080,   16'd4,     "f2i_4.0");
        test_cvt(1, 16'h4100,   16'd8,     "f2i_8.0");
        test_cvt(1, 16'h4180,   16'd16,    "f2i_16.0");
        test_cvt(1, 16'h4200,   16'd32,    "f2i_32.0");
        test_cvt(1, 16'h4480,   16'd1024,  "f2i_1024.0");
        test_cvt(1, 16'h4680,   16'd16384, "f2i_16384.0");

        $display("\n--- Negatives ---");
        test_cvt(1, 16'hBF80,   -16'd1,    "f2i_-1.0");
        test_cvt(1, 16'hC000,   -16'd2,    "f2i_-2.0");
        test_cvt(1, 16'hC120,   -16'd10,   "f2i_-10.0");
        test_cvt(1, 16'hC2C8,   -16'd100,  "f2i_-100.0");
        test_cvt(1, 16'hC300,   -16'd128,  "f2i_-128.0");

        $display("\n--- Truncation (fractional part dropped) ---");
        // 1.5 = 3F_C0 → 1
        test_cvt(1, 16'h3FC0,   16'd1,     "f2i_1.5");
        // 2.75 = 40_30 → 2
        test_cvt(1, 16'h4030,   16'd2,     "f2i_2.75");
        // 9.5 = 41_18 → 9
        test_cvt(1, 16'h4118,   16'd9,     "f2i_9.5");
        // -1.5 = BF_C0 → -1
        test_cvt(1, 16'hBFC0,   -16'd1,    "f2i_-1.5");
        // -9.5 = C1_18 → -9
        test_cvt(1, 16'hC118,   -16'd9,    "f2i_-9.5");

        $display("\n--- Underflow (|val| < 1 → 0) ---");
        // 0.5 = 3F_00
        test_cvt(1, 16'h3F00,   16'd0,     "f2i_0.5");
        // 0.25 = 3E_80
        test_cvt(1, 16'h3E80,   16'd0,     "f2i_0.25");
        // -0.5 = BF_00
        test_cvt(1, 16'hBF00,   16'd0,     "f2i_-0.5");

        $display("\n--- Overflow / saturation ---");
        // 32768.0 = 47_00 → saturate +32767
        test_cvt(1, 16'h4700,   16'h7FFF,  "f2i_32768_sat");
        // -32769 → saturate -32768 (exp=143, man=0000001)
        // Actually 65536.0 = 47_80 neg = C7_80
        test_cvt(1, 16'hC780,   16'h8000,  "f2i_-65536_sat");

        $display("\n--- Special values ---");
        // +Inf = 7F_80
        test_cvt(1, 16'h7F80,   16'h7FFF,  "f2i_+inf");
        // -Inf = FF_80
        test_cvt(1, 16'hFF80,   16'h8000,  "f2i_-inf");
        // NaN  = 7F_C0 (quiet NaN)
        test_cvt(1, 16'h7FC0,   16'd0,     "f2i_nan");
        // NaN  = FF_81 (signaling NaN, negative)
        test_cvt(1, 16'hFF81,   16'd0,     "f2i_snan");
        // Denormal = 00_01 → 0
        test_cvt(1, 16'h0001,   16'd0,     "f2i_denorm");
        // Denormal negative = 80_01 → 0
        test_cvt(1, 16'h8001,   16'd0,     "f2i_neg_denorm");

        // =============================================================
        // Round-trip: int16 → bf16 → int16 (exact for |val| ≤ 128)
        // =============================================================
        $display("\n=== Round-trip (exact for |val| <= 128) ===");
        test_cvt(0, 16'd50,     16'h4248, "rt_50_i2f");
        test_cvt(1, 16'h4248,   16'd50,   "rt_50_f2i");
        test_cvt(0, 16'd1,      16'h3F80, "rt_1_i2f");
        test_cvt(1, 16'h3F80,   16'd1,    "rt_1_f2i");
        test_cvt(0, -16'd64,    16'hC280, "rt_-64_i2f");
        test_cvt(1, 16'hC280,   -16'd64,  "rt_-64_f2i");
        test_cvt(0, -16'd128,   16'hC300, "rt_-128_i2f");
        test_cvt(1, 16'hC300,   -16'd128, "rt_-128_f2i");

        $display("\n  Directed: %0d passed, %0d failed @ cycle %0d",
            pass_count, fail_count, cycle);

        // =============================================================
        // Random INT16 → BF16 (5000 vectors)
        // =============================================================
        $display("\n--- Random i2f (5000 vectors) ---");
        verbose = 0;
        for (i = 0; i < 5000; i = i + 1) begin
            rand_v = $random;
            golden_i2f(rand_v[15:0]);
            test_cvt(0, rand_v[15:0], gm_i2f, "rand_i2f");
        end
        $display("  Random i2f: %0d passed, %0d failed @ cycle %0d",
            pass_count, fail_count, cycle);

        // =============================================================
        // Random BF16 → INT16 (5000 vectors)
        // =============================================================
        $display("\n--- Random f2i (5000 vectors) ---");
        for (i = 0; i < 5000; i = i + 1) begin
            rand_v = $random;
            golden_f2i(rand_v[15:0]);
            test_cvt(1, rand_v[15:0], gm_f2i, "rand_f2i");
        end
        $display("  Random f2i: %0d passed, %0d failed @ cycle %0d",
            pass_count, fail_count, cycle);

        // =============================================================
        // Random round-trip (exact values: |val| <= 128)
        // =============================================================
        $display("\n--- Random round-trip |val| <= 128 (1000 vectors) ---");
        for (i = 0; i < 1000; i = i + 1) begin
            rand_v = $random;
            // Clamp to -128..128
            rand_v[15:0] = ($signed(rand_v[15:0]) > 128)  ? 16'd128 :
                           ($signed(rand_v[15:0]) < -128) ? -16'd128 :
                           rand_v[15:0];
            golden_i2f(rand_v[15:0]);
            test_cvt(0, rand_v[15:0], gm_i2f, "rt_i2f");
            golden_f2i(gm_i2f);
            test_cvt(1, gm_i2f, rand_v[15:0], "rt_f2i");
        end
        $display("  Round-trip: %0d passed, %0d failed @ cycle %0d",
            pass_count, fail_count, cycle);

        // =============================================================
        // Back-to-back: alternating directions
        // =============================================================
        $display("\n--- Back-to-back alternating ---");
        verbose = 1;
        test_cvt(0, 16'd42,     16'h4228, "b2b_i2f_42");
        test_cvt(1, 16'h4228,   16'd42,   "b2b_f2i_42");
        test_cvt(0, -16'd10,    16'hC120, "b2b_i2f_-10");
        test_cvt(1, 16'h7F80,   16'h7FFF, "b2b_f2i_inf");
        test_cvt(0, 16'd0,      16'h0000, "b2b_i2f_0");

        // Summary
        repeat (5) @(posedge clk);
        $display("\n=========================================================");
        $display(" Results: %0d PASSED, %0d FAILED / %0d total",
            pass_count, fail_count, pass_count + fail_count);
        $display(" Total cycles: %0d", cycle);
        if (fail_count > 0) begin
            $display("---------------------------------------------------------");
            $display(" FAILURE LOG (first %0d):", (fail_count > 16) ? 16 : fail_count);
            for (i = 0; i < fail_count && i < 16; i = i + 1) begin
                $display("  #%0d [cyc %0d] %s: dt=%0b in=%04h got=%04h exp=%04h",
                    i, fail_cyc[i], fail_labels[i], fail_dir[i],
                    fail_in[i], fail_got[i], fail_exp[i]);
            end
            if (fail_count > 16)
                $display("  ... and %0d more", fail_count - 16);
        end
        $display("=========================================================");
        $finish;
    end

    initial begin
        #10000000;
        $display("WATCHDOG timeout @ cycle %0d", cycle);
        $finish;
    end
endmodule