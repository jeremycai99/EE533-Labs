/* file: conv_fifo_tb.v
 * Testbench for conv_fifo v2.0 — 4096×64b Convertible FIFO.
 *
 * Author: Jeremy Cai
 * Date: Mar. 5, 2026
 * Version: 2.0
 */

`timescale 1ns / 1ps

`include "conv_fifo.v"

module conv_fifo_tb;

    localparam ADDR_WIDTH = 12;
    localparam DATA_WIDTH = 64;
    localparam CTRL_WIDTH = 8;

    reg clk, rst_n;
    localparam CLK_PERIOD = 10;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // DUT I/O
    reg [1:0] mode;
    reg [DATA_WIDTH-1:0] in_data;
    reg [CTRL_WIDTH-1:0] in_ctrl;
    reg in_wr;
    wire in_rdy;
    wire [DATA_WIDTH-1:0] out_data;
    wire [CTRL_WIDTH-1:0] out_ctrl;
    wire out_wr;
    reg out_rdy;
    reg tx_start;
    reg [1:0] tx_port;
    wire tx_done;
    reg [ADDR_WIDTH-1:0] sram_addr;
    reg [DATA_WIDTH-1:0] sram_wdata;
    reg sram_we;
    wire [DATA_WIDTH-1:0] sram_rdata;
    reg [ADDR_WIDTH-1:0] head_ptr_in, tail_ptr_in;
    reg head_ptr_wr, tail_ptr_wr;
    wire [ADDR_WIDTH-1:0] head_ptr_out, tail_ptr_out, pkt_end_ptr;
    wire pkt_ready, nearly_full, fifo_empty, fifo_full;

    conv_fifo #(
        .ADDR_WIDTH(ADDR_WIDTH), .DATA_WIDTH(DATA_WIDTH),
        .CTRL_WIDTH(CTRL_WIDTH), .NEARLY_FULL_THRESH(4)
    ) u_dut (
        .clk(clk), .rst_n(rst_n), .mode(mode),
        .in_data(in_data), .in_ctrl(in_ctrl), .in_wr(in_wr), .in_rdy(in_rdy),
        .out_data(out_data), .out_ctrl(out_ctrl), .out_wr(out_wr), .out_rdy(out_rdy),
        .tx_start(tx_start), .tx_port(tx_port), .tx_done(tx_done),
        .sram_addr(sram_addr), .sram_wdata(sram_wdata), .sram_we(sram_we),
        .sram_rdata(sram_rdata),
        .head_ptr_in(head_ptr_in), .head_ptr_wr(head_ptr_wr),
        .tail_ptr_in(tail_ptr_in), .tail_ptr_wr(tail_ptr_wr),
        .head_ptr_out(head_ptr_out), .tail_ptr_out(tail_ptr_out),
        .pkt_end_ptr(pkt_end_ptr),
        .pkt_ready(pkt_ready), .nearly_full(nearly_full),
        .fifo_empty(fifo_empty), .fifo_full(fifo_full)
    );

    integer pass_count = 0, fail_count = 0, test_num = 0;

    // TX capture
    reg [DATA_WIDTH-1:0] tx_cap_data [0:63];
    reg [CTRL_WIDTH-1:0] tx_cap_ctrl [0:63];
    integer tx_cap_cnt;

    task tick; begin @(posedge clk); #1; end endtask

    task check1;
        input val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = %b, exp %b", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task check8;
        input [7:0] val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = 0x%02h, exp 0x%02h", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task check12;
        input [11:0] val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = 0x%03h, exp 0x%03h", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task check64;
        input [63:0] val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = 0x%016h, exp 0x%016h", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task rx_word;
        input [DATA_WIDTH-1:0] data;
        input [CTRL_WIDTH-1:0] ctrl;
    begin in_data = data; in_ctrl = ctrl; in_wr = 1; tick; end
    endtask

    task rx_end;
    begin in_wr = 0; in_data = 0; in_ctrl = 0; tick; end
    endtask

    task capture_tx;
        input integer max_cycles;
        integer cyc;
    begin
        tx_cap_cnt = 0; cyc = 0;
        while (!tx_done && cyc < max_cycles) begin
            if (out_wr && out_rdy) begin
                tx_cap_data[tx_cap_cnt] = out_data;
                tx_cap_ctrl[tx_cap_cnt] = out_ctrl;
                tx_cap_cnt = tx_cap_cnt + 1;
            end
            tick; cyc = cyc + 1;
        end
        if (cyc >= max_cycles) $display("[TIMEOUT] TX drain");
    end endtask

    task sram_write;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
    begin sram_addr = addr; sram_wdata = data; sram_we = 1; tick; sram_we = 0; end
    endtask

    task sram_read;
        input [ADDR_WIDTH-1:0] addr;
    begin sram_addr = addr; sram_we = 0; tick; tick; end
    endtask

    initial begin $dumpfile("conv_fifo_tb.vcd"); $dumpvars(0, conv_fifo_tb); end
    initial begin #500000; $display("[TIMEOUT]"); $finish; end

    initial begin
        $display("============================================");
        $display("  Conv FIFO v2.0 Testbench — Starting");
        $display("  (4096×64b, no per-word ctrl)");
        $display("============================================");

        rst_n = 0; mode = 0;
        in_data = 0; in_ctrl = 0; in_wr = 0;
        out_rdy = 1; tx_start = 0; tx_port = 0;
        sram_addr = 0; sram_wdata = 0; sram_we = 0;
        head_ptr_in = 0; head_ptr_wr = 0;
        tail_ptr_in = 0; tail_ptr_wr = 0;
        repeat (3) tick;
        rst_n = 1; tick;

        // ── Test 1: Reset ────────────────────────────────
        $display("\n--- Test 1: Reset state ---");
        check12(head_ptr_out, 12'd0, "head=0");
        check12(tail_ptr_out, 12'd0, "tail=0");
        check1(fifo_empty, 1'b1, "fifo_empty");
        check1(pkt_ready, 1'b0, "!pkt_ready");
        check1(in_rdy, 1'b1, "in_rdy");

        // ── Test 2: RX 4-word packet ─────────────────────
        $display("\n--- Test 2: RX_FIFO receive (4 words) ---");
        mode = 2'd0;
        rx_word(64'hAAAA_BBBB_CCCC_DDDD, 8'hFF); // word 0: module header
        rx_word(64'h1111_1111_1111_1111, 8'h00);
        rx_word(64'h2222_2222_2222_2222, 8'h00);
        rx_word(64'h3333_3333_3333_3333, 8'h00);
        rx_end;

        check12(tail_ptr_out, 12'd4, "tail=4");
        check1(pkt_ready, 1'b1, "pkt_ready");
        check12(pkt_end_ptr, 12'd3, "pkt_end=3");

        // ── Test 3: SRAM read of RX data ─────────────────
        $display("\n--- Test 3: SRAM read ---");
        mode = 2'd1;
        sram_read(12'd0);
        check64(sram_rdata, 64'hAAAA_BBBB_CCCC_DDDD, "SRAM[0]");
        sram_read(12'd2);
        check64(sram_rdata, 64'h2222_2222_2222_2222, "SRAM[2]");
        sram_read(12'd3);
        check64(sram_rdata, 64'h3333_3333_3333_3333, "SRAM[3]");

        // ── Test 4: SRAM write + readback ────────────────
        $display("\n--- Test 4: SRAM write + readback ---");
        sram_write(12'd100, 64'hDEAD_BEEF_CAFE_1234);
        sram_read(12'd100);
        check64(sram_rdata, 64'hDEAD_BEEF_CAFE_1234, "SRAM[100]");

        // ── Test 5: TX_DRAIN (port 2) ────────────────────
        $display("\n--- Test 5: TX_DRAIN ---");
        mode = 2'd2;
        out_rdy = 1;
        tx_port = 2'd2;  // MAC port 2 → out_ctrl=8'h10 on word 0
        tx_start = 1; tick; tx_start = 0;
        capture_tx(100);

        check1(tx_done, 1'b1, "tx_done");
        check1(pkt_ready, 1'b0, "pkt_ready cleared");

        test_num = test_num + 1;
        if (tx_cap_cnt == 4) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: TX count=%0d, exp 4", test_num, tx_cap_cnt); fail_count = fail_count + 1; end

        check64(tx_cap_data[0], 64'hAAAA_BBBB_CCCC_DDDD, "TX[0] data");
        check8(tx_cap_ctrl[0], 8'h10, "TX[0] ctrl (port 2 one-hot)");
        check64(tx_cap_data[1], 64'h1111_1111_1111_1111, "TX[1] data");
        check8(tx_cap_ctrl[1], 8'h00, "TX[1] ctrl=0");
        check64(tx_cap_data[2], 64'h2222_2222_2222_2222, "TX[2] data");
        check64(tx_cap_data[3], 64'h3333_3333_3333_3333, "TX[3] data");
        check12(head_ptr_out, 12'd4, "head advanced to 4");

        // ── Test 6: TX with backpressure ─────────────────
        $display("\n--- Test 6: TX with backpressure ---");
        mode = 2'd0;
        rx_word(64'hF0F0_F0F0_F0F0_F0F0, 8'h02);
        rx_word(64'hA5A5_A5A5_A5A5_A5A5, 8'h00);
        rx_word(64'h5A5A_5A5A_5A5A_5A5A, 8'h00);
        rx_end;

        mode = 2'd2;
        out_rdy = 0;
        tx_port = 2'd0;
        tx_start = 1; tick; tx_start = 0;

        tx_cap_cnt = 0;
        begin : bp_block
            integer cyc;
            for (cyc = 0; cyc < 50; cyc = cyc + 1) begin
                out_rdy = (cyc % 2 == 0) ? 1'b1 : 1'b0;
                if (out_wr && out_rdy) begin
                    tx_cap_data[tx_cap_cnt] = out_data;
                    tx_cap_ctrl[tx_cap_cnt] = out_ctrl;
                    tx_cap_cnt = tx_cap_cnt + 1;
                end
                tick;
                if (tx_done) disable bp_block;
            end
        end
        out_rdy = 1;

        test_num = test_num + 1;
        if (tx_cap_cnt == 3) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: BP TX count=%0d, exp 3", test_num, tx_cap_cnt); fail_count = fail_count + 1; end

        check64(tx_cap_data[0], 64'hF0F0_F0F0_F0F0_F0F0, "BP TX[0]");
        check8(tx_cap_ctrl[0], 8'h01, "BP TX[0] ctrl (port 0 one-hot)");
        check8(tx_cap_ctrl[1], 8'h00, "BP TX[1] ctrl=0");

        // ── Test 7: Pointer write ────────────────────────
        $display("\n--- Test 7: Pointer write ---");
        mode = 2'd1;
        head_ptr_in = 12'd50; head_ptr_wr = 1;
        tail_ptr_in = 12'd60; tail_ptr_wr = 1;
        tick;
        head_ptr_wr = 0; tail_ptr_wr = 0;
        check12(head_ptr_out, 12'd50, "head=50");
        check12(tail_ptr_out, 12'd60, "tail=60");
        check1(fifo_empty, 1'b0, "!empty");

        // ── Test 8: Single-word packet ───────────────────
        $display("\n--- Test 8: Single-word packet ---");
        head_ptr_in = 12'd0; head_ptr_wr = 1;
        tail_ptr_in = 12'd0; tail_ptr_wr = 1;
        tick; head_ptr_wr = 0; tail_ptr_wr = 0;

        mode = 2'd0;
        rx_word(64'h9999_9999_9999_9999, 8'h01);
        rx_end;

        check1(pkt_ready, 1'b1, "single pkt_ready");
        check12(pkt_end_ptr, 12'd0, "pkt_end=0");

        mode = 2'd2;
        tx_port = 2'd1;
        tx_start = 1; tick; tx_start = 0;
        capture_tx(50);

        test_num = test_num + 1;
        if (tx_cap_cnt == 1) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: 1w TX count=%0d", test_num, tx_cap_cnt); fail_count = fail_count + 1; end

        check64(tx_cap_data[0], 64'h9999_9999_9999_9999, "1w TX data");
        check8(tx_cap_ctrl[0], 8'h04, "1w TX ctrl (port 1 one-hot)");

        // ── Test 9: Large address (verify 12-bit works) ──
        $display("\n--- Test 9: Large address (>256) ---");
        mode = 2'd1;
        head_ptr_in = 12'd0; head_ptr_wr = 1;
        tail_ptr_in = 12'd0; tail_ptr_wr = 1;
        tick; head_ptr_wr = 0; tail_ptr_wr = 0;

        sram_write(12'd2000, 64'hBEEF_DEAD_1234_5678);
        sram_write(12'd4000, 64'hCAFE_BABE_8765_4321);
        sram_read(12'd2000);
        check64(sram_rdata, 64'hBEEF_DEAD_1234_5678, "SRAM[2000]");
        sram_read(12'd4000);
        check64(sram_rdata, 64'hCAFE_BABE_8765_4321, "SRAM[4000]");

        // ── Summary ──────────────────────────────────────
        $display("\n============================================");
        $display("  Conv FIFO v2.0 Testbench — Summary");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        $display("  TOTAL:  %0d", pass_count + fail_count);
        $display("============================================");
        if (fail_count == 0) $display(">>> ALL TESTS PASSED <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $finish;
    end

endmodule