/* file: pkt_proc_tb.v
 * Testbench for pkt_proc v2.0 — NOP, LOAD_IMEM, LOAD_DMEM, CPU_START.
 *
 * Author: Jeremy Cai
 * Date: Mar. 5, 2026
 * Version: 2.0
 */

`timescale 1ns / 1ps

`include "pkt_proc.v"

module pkt_proc_tb;

    localparam FIFO_AW = 12;
    localparam IMEM_AW = 10;
    localparam DMEM_AW = 12;

    reg clk, rst_n;
    localparam CLK_PERIOD = 10;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // DUT I/O
    wire [FIFO_AW-1:0] fifo_addr;
    wire [63:0] fifo_rdata;
    wire [1:0] fifo_mode;
    reg [FIFO_AW-1:0] fifo_head_ptr, fifo_pkt_end;
    reg fifo_pkt_ready;

    wire [IMEM_AW-1:0] imem_addr;
    wire [31:0] imem_din;
    wire imem_we;

    wire [DMEM_AW-1:0] dmem_addr;
    wire [31:0] dmem_din;
    wire dmem_we;

    wire cpu_rst_n_out;
    wire cpu_start_out;
    wire [31:0] entry_pc;
    reg cpu_done;
    wire active, owns_port_b;

    pkt_proc #(
        .FIFO_ADDR_WIDTH(FIFO_AW),
        .IMEM_ADDR_WIDTH(IMEM_AW),
        .DMEM_ADDR_WIDTH(DMEM_AW)
    ) u_dut (
        .clk(clk), .rst_n(rst_n),
        .fifo_addr(fifo_addr), .fifo_rdata(fifo_rdata),
        .fifo_mode(fifo_mode),
        .fifo_head_ptr(fifo_head_ptr), .fifo_pkt_end(fifo_pkt_end),
        .fifo_pkt_ready(fifo_pkt_ready),
        .imem_addr(imem_addr), .imem_din(imem_din), .imem_we(imem_we),
        .dmem_addr(dmem_addr), .dmem_din(dmem_din), .dmem_we(dmem_we),
        .cpu_rst_n(cpu_rst_n_out), .cpu_start(cpu_start_out),
        .entry_pc(entry_pc), .cpu_done(cpu_done),
        .active(active), .owns_port_b(owns_port_b)
    );

    // FIFO BRAM model (4096×64b, sync read)
    reg [63:0] fifo_mem [0:4095];
    reg [63:0] fifo_rdata_r;
    always @(posedge clk) begin
        fifo_rdata_r <= fifo_mem[fifo_addr];
    end
    assign fifo_rdata = fifo_rdata_r;

    // IMEM BRAM model (1024×32b, write-only)
    reg [31:0] imem_mem [0:1023];
    always @(posedge clk) begin
        if (imem_we) imem_mem[imem_addr] <= imem_din;
    end

    // DMEM BRAM model (4096×32b, write-only for this TB)
    reg [31:0] dmem_mem [0:4095];
    always @(posedge clk) begin
        if (dmem_we) dmem_mem[dmem_addr] <= dmem_din;
    end

    // Test infrastructure
    integer pass_count = 0, fail_count = 0, test_num = 0;

    task tick; begin @(posedge clk); #1; end endtask

    task check32;
        input [31:0] val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = 0x%08h, exp 0x%08h", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task check1;
        input val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = %b, exp %b", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    function [63:0] cmd_word;
        input [3:0] cmd;
        input [11:0] addr;
        input [15:0] cnt;
        input [31:0] param;
    begin
        cmd_word = {cmd, addr, cnt, param};
    end
    endfunction

    task wait_idle;
        input integer max_cycles;
        integer cyc;
    begin
        cyc = 0;
        while (active && cyc < max_cycles) begin tick; cyc = cyc + 1; end
        if (cyc >= max_cycles) $display("[TIMEOUT] wait_idle");
    end endtask

    task wait_cpu_running;
        input integer max_cycles;
        integer cyc;
    begin
        cyc = 0;
        while (!cpu_rst_n_out && cyc < max_cycles) begin tick; cyc = cyc + 1; end
        if (cyc >= max_cycles) $display("[TIMEOUT] wait_cpu_running");
    end endtask

    initial begin $dumpfile("pkt_proc_tb.vcd"); $dumpvars(0, pkt_proc_tb); end
    initial begin #1000000; $display("[TIMEOUT] Global"); $finish; end

    initial begin
        $display("============================================");
        $display("  pkt_proc v2.0 Testbench — Starting");
        $display("  (NOP, LOAD_IMEM, LOAD_DMEM, CPU_START)");
        $display("============================================");

        rst_n = 0;
        fifo_head_ptr = 0; fifo_pkt_end = 0;
        fifo_pkt_ready = 0; cpu_done = 0;
        repeat (3) tick;
        rst_n = 1; tick;

        // ── Test 1: LOAD_IMEM — 4 instructions ──────────
        $display("\n--- Test 1: LOAD_IMEM ---");

        fifo_mem[0] = cmd_word(4'h1, 12'h000, 16'd2, 32'h0);
        fifo_mem[1] = {32'hBBBB_BBBB, 32'hAAAA_AAAA};
        fifo_mem[2] = {32'hDDDD_DDDD, 32'hCCCC_CCCC};

        fifo_head_ptr = 12'd0;
        fifo_pkt_end = 12'd2;
        fifo_pkt_ready = 1; tick; fifo_pkt_ready = 0;

        wait_idle(100);

        check32(imem_mem[0], 32'hAAAA_AAAA, "IMEM[0]");
        check32(imem_mem[1], 32'hBBBB_BBBB, "IMEM[1]");
        check32(imem_mem[2], 32'hCCCC_CCCC, "IMEM[2]");
        check32(imem_mem[3], 32'hDDDD_DDDD, "IMEM[3]");

        // ── Test 2: LOAD_DMEM — 4 data words ────────────
        $display("\n--- Test 2: LOAD_DMEM ---");

        fifo_mem[0] = cmd_word(4'h2, 12'h100, 16'd2, 32'h0);
        fifo_mem[1] = {32'h2222_2222, 32'h1111_1111};
        fifo_mem[2] = {32'h4444_4444, 32'h3333_3333};

        fifo_head_ptr = 12'd0;
        fifo_pkt_end = 12'd2;
        fifo_pkt_ready = 1; tick; fifo_pkt_ready = 0;

        wait_idle(100);

        check32(dmem_mem[12'h100], 32'h1111_1111, "DMEM[0x100]");
        check32(dmem_mem[12'h101], 32'h2222_2222, "DMEM[0x101]");
        check32(dmem_mem[12'h102], 32'h3333_3333, "DMEM[0x102]");
        check32(dmem_mem[12'h103], 32'h4444_4444, "DMEM[0x103]");

        // ── Test 3: Multi-command + CPU_START ────────────
        $display("\n--- Test 3: LOAD_IMEM + LOAD_DMEM + CPU_START ---");

        fifo_mem[0] = cmd_word(4'h1, 12'h000, 16'd1, 32'h0);
        fifo_mem[1] = {32'hFEED_FACE, 32'hDEAD_BEEF};
        fifo_mem[2] = cmd_word(4'h2, 12'h000, 16'd1, 32'h0);
        fifo_mem[3] = {32'hCAFE_BABE, 32'hBAAD_F00D};
        fifo_mem[4] = cmd_word(4'h3, 12'h000, 16'd0, 32'h0000_0040);

        fifo_head_ptr = 12'd0;
        fifo_pkt_end = 12'd4;
        fifo_pkt_ready = 1; tick; fifo_pkt_ready = 0;

        wait_cpu_running(100);
        check1(cpu_rst_n_out, 1'b1, "cpu_rst_n released");
        check1(cpu_start_out, 1'b1, "cpu_start pulse HIGH");
        check32(entry_pc, 32'h0000_0040, "entry_pc=0x40");

        // Verify pulse auto-clears next cycle
        tick;
        check1(cpu_start_out, 1'b0, "cpu_start auto-cleared");

        check32(imem_mem[0], 32'hDEAD_BEEF, "IMEM[0] multi");
        check32(imem_mem[1], 32'hFEED_FACE, "IMEM[1] multi");
        check32(dmem_mem[0], 32'hBAAD_F00D, "DMEM[0] multi");
        check32(dmem_mem[1], 32'hCAFE_BABE, "DMEM[1] multi");

        check1(owns_port_b, 1'b0, "port_b released during CPU_RUN");
        check1(active, 1'b1, "still active (waiting)");
        check1(fifo_mode == 2'd0, 1'b1, "fifo_mode=RX during CPU_RUN");

        cpu_done = 1; tick; cpu_done = 0;
        wait_idle(100);
        check1(active, 1'b0, "idle after cpu_done");

        // ── Test 4: NOP command ──────────────────────────
        $display("\n--- Test 4: NOP ---");

        fifo_mem[0] = cmd_word(4'h0, 12'h0, 16'd0, 32'h0);
        fifo_mem[1] = cmd_word(4'h1, 12'h010, 16'd1, 32'h0);
        fifo_mem[2] = {32'h9999_9999, 32'h8888_8888};

        fifo_head_ptr = 12'd0;
        fifo_pkt_end = 12'd2;
        fifo_pkt_ready = 1; tick; fifo_pkt_ready = 0;

        wait_idle(100);

        check32(imem_mem[10'h010], 32'h8888_8888, "NOP skip: IMEM[0x10]");
        check32(imem_mem[10'h011], 32'h9999_9999, "NOP skip: IMEM[0x11]");

        // ── Test 5: Non-zero base address ────────────────
        $display("\n--- Test 5: Non-zero base addr ---");

        fifo_mem[0] = cmd_word(4'h1, 12'h100, 16'd1, 32'h0);
        fifo_mem[1] = {32'h5555_5555, 32'h4444_4444};

        fifo_head_ptr = 12'd0;
        fifo_pkt_end = 12'd1;
        fifo_pkt_ready = 1; tick; fifo_pkt_ready = 0;

        wait_idle(100);

        check32(imem_mem[10'h100], 32'h4444_4444, "IMEM[0x100]");
        check32(imem_mem[10'h101], 32'h5555_5555, "IMEM[0x101]");

        // ── Test 6: CPU_START at end of packet → idle ────
        $display("\n--- Test 6: CPU_START then idle ---");

        fifo_mem[0] = cmd_word(4'h3, 12'h000, 16'd0, 32'h0000_0000);

        fifo_head_ptr = 12'd0;
        fifo_pkt_end = 12'd0;
        fifo_pkt_ready = 1; tick; fifo_pkt_ready = 0;

        wait_cpu_running(100);
        check1(cpu_rst_n_out, 1'b1, "cpu running");
        check1(fifo_mode == 2'd0, 1'b1, "FIFO released");
        tick;
        check1(cpu_start_out, 1'b0, "start pulse cleared");
        check1(cpu_rst_n_out, 1'b1, "rst_n still high");

        cpu_done = 1; tick; cpu_done = 0;
        wait_idle(100);
        check1(active, 1'b0, "idle after solo CPU_START");

        // ── Test 7: Non-zero head_ptr (mid-FIFO packet) ─
        $display("\n--- Test 7: Non-zero head_ptr ---");

        fifo_mem[100] = cmd_word(4'h2, 12'h500, 16'd1, 32'h0);
        fifo_mem[101] = {32'hFFFF_FFFF, 32'hEEEE_EEEE};

        fifo_head_ptr = 12'd100;
        fifo_pkt_end = 12'd101;
        fifo_pkt_ready = 1; tick; fifo_pkt_ready = 0;

        wait_idle(100);

        check32(dmem_mem[12'h500], 32'hEEEE_EEEE, "DMEM[0x500] mid-FIFO");
        check32(dmem_mem[12'h501], 32'hFFFF_FFFF, "DMEM[0x501] mid-FIFO");

        // ── Summary ──────────────────────────────────────
        $display("\n============================================");
        $display("  pkt_proc v2.0 Testbench — Summary");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        $display("  TOTAL:  %0d", pass_count + fail_count);
        $display("============================================");
        if (fail_count == 0) $display(">>> ALL TESTS PASSED <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $finish;
    end

endmodule