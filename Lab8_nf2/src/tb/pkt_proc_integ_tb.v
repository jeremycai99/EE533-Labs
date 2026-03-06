/* file: pkt_proc_integ_tb.v
 * Integration testbench: conv_fifo + pkt_proc.
 * Comprehensive test coverage for the full packet processing pipeline.
 *
 * Author: Jeremy Cai
 * Date: Mar. 5, 2026
 * Version: 4.0
 */

`timescale 1ns / 1ps

`include "conv_fifo.v"
`include "pkt_proc.v"

module pkt_proc_integ_tb;

    localparam FIFO_AW = 12;
    localparam IMEM_AW = 10;
    localparam DMEM_AW = 12;
    localparam DATA_WIDTH = 64;
    localparam CTRL_WIDTH = 8;

    reg clk, rst_n;
    localparam CLK_PERIOD = 10;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── NetFPGA RX / TX ─────────────────────────────────
    reg [DATA_WIDTH-1:0] in_data;
    reg [CTRL_WIDTH-1:0] in_ctrl;
    reg in_wr;
    wire in_rdy;
    wire [DATA_WIDTH-1:0] out_data;
    wire [CTRL_WIDTH-1:0] out_ctrl;
    wire out_wr;
    reg out_rdy;

    // ── Interconnect ────────────────────────────────────
    wire [FIFO_AW-1:0] pp_fifo_addr;
    wire [63:0] pp_fifo_wdata;
    wire pp_fifo_we;
    wire [63:0] fifo_sram_rdata;
    wire [1:0] pp_fifo_mode;
    wire [FIFO_AW-1:0] pp_head_wr_data, pp_tail_wr_data;
    wire pp_head_wr, pp_tail_wr;
    wire pp_tx_start, pp_pkt_ack;
    wire [FIFO_AW-1:0] fifo_head_out, fifo_tail_out, fifo_pkt_end;
    wire fifo_pkt_ready, fifo_tx_done;
    wire fifo_nearly_full, fifo_empty, fifo_full;
    wire [IMEM_AW-1:0] imem_addr;
    wire [31:0] imem_din;
    wire imem_we;
    wire [DMEM_AW-1:0] dmem_addr;
    wire [31:0] dmem_din;
    wire dmem_we;
    wire [31:0] dmem_dout;
    wire cpu_rst_n_out, cpu_start_out;
    wire [31:0] entry_pc;
    reg cpu_done;
    wire pp_active, pp_owns_port_b;

    // ── DUT: conv_fifo ──────────────────────────────────
    conv_fifo #(
        .ADDR_WIDTH(FIFO_AW), .DATA_WIDTH(DATA_WIDTH),
        .CTRL_WIDTH(CTRL_WIDTH), .NEARLY_FULL_THRESH(4)
    ) u_fifo (
        .clk(clk), .rst_n(rst_n), .mode(pp_fifo_mode),
        .in_data(in_data), .in_ctrl(in_ctrl), .in_wr(in_wr), .in_rdy(in_rdy),
        .out_data(out_data), .out_ctrl(out_ctrl), .out_wr(out_wr), .out_rdy(out_rdy),
        .tx_start(pp_tx_start), .pkt_ack(pp_pkt_ack), .tx_done(fifo_tx_done),
        .sram_addr(pp_fifo_addr), .sram_wdata(pp_fifo_wdata),
        .sram_we(pp_fifo_we), .sram_rdata(fifo_sram_rdata),
        .head_ptr_in(pp_head_wr_data), .head_ptr_wr(pp_head_wr),
        .tail_ptr_in(pp_tail_wr_data), .tail_ptr_wr(pp_tail_wr),
        .head_ptr_out(fifo_head_out), .tail_ptr_out(fifo_tail_out),
        .pkt_end_ptr(fifo_pkt_end),
        .pkt_ready(fifo_pkt_ready), .nearly_full(fifo_nearly_full),
        .fifo_empty(fifo_empty), .fifo_full(fifo_full)
    );

    // ── DUT: pkt_proc ───────────────────────────────────
    pkt_proc #(
        .FIFO_ADDR_WIDTH(FIFO_AW), .IMEM_ADDR_WIDTH(IMEM_AW),
        .DMEM_ADDR_WIDTH(DMEM_AW)
    ) u_pp (
        .clk(clk), .rst_n(rst_n),
        .fifo_addr(pp_fifo_addr), .fifo_wdata(pp_fifo_wdata),
        .fifo_we(pp_fifo_we), .fifo_rdata(fifo_sram_rdata),
        .fifo_mode(pp_fifo_mode),
        .fifo_head_wr_data(pp_head_wr_data), .fifo_head_wr(pp_head_wr),
        .fifo_tail_wr_data(pp_tail_wr_data), .fifo_tail_wr(pp_tail_wr),
        .fifo_tx_start(pp_tx_start),
        .fifo_head_ptr(fifo_head_out), .fifo_pkt_end(fifo_pkt_end),
        .fifo_pkt_ready(fifo_pkt_ready), .fifo_pkt_ack(pp_pkt_ack),
        .fifo_tx_done(fifo_tx_done),
        .imem_addr(imem_addr), .imem_din(imem_din), .imem_we(imem_we),
        .dmem_addr(dmem_addr), .dmem_din(dmem_din), .dmem_we(dmem_we),
        .dmem_dout(dmem_dout),
        .cpu_rst_n(cpu_rst_n_out), .cpu_start(cpu_start_out),
        .entry_pc(entry_pc), .cpu_done(cpu_done),
        .active(pp_active), .owns_port_b(pp_owns_port_b)
    );

    // ── IMEM BRAM (1024×32b) ────────────────────────────
    reg [31:0] imem_mem [0:1023];
    always @(posedge clk) begin
        if (imem_we) imem_mem[imem_addr] <= imem_din;
    end

    // ── DMEM BRAM (4096×32b, sync read) ─────────────────
    reg [31:0] dmem_mem [0:4095];
    reg [31:0] dmem_dout_r;
    always @(posedge clk) begin
        if (dmem_we) dmem_mem[dmem_addr] <= dmem_din;
        dmem_dout_r <= dmem_mem[dmem_addr];
    end
    assign dmem_dout = dmem_dout_r;

    // ════════════════════════════════════════════════════
    // Test infrastructure
    // ════════════════════════════════════════════════════
    integer pass_count = 0, fail_count = 0, test_num = 0;
    integer i;
    reg [31:0] tmp_lo, tmp_hi;  // temps for loop data generation

    reg [DATA_WIDTH-1:0] tx_cap_data [0:127];
    reg [CTRL_WIDTH-1:0] tx_cap_ctrl [0:127];
    integer tx_cap_cnt;

    task tick; begin @(posedge clk); #1; end endtask

    task check32;
        input [31:0] val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = 0x%08h, exp 0x%08h", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task check64;
        input [63:0] val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = 0x%016h, exp 0x%016h", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task check8;
        input [7:0] val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = 0x%02h, exp 0x%02h", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task check1;
        input val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = %b, exp %b", test_num, name, val, expected); fail_count = fail_count + 1; end
    end endtask

    task checkN;
        input integer val, expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) pass_count = pass_count + 1;
        else begin $display("[FAIL] T%0d: %0s = %0d, exp %0d", test_num, name, val, expected); fail_count = fail_count + 1; end
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

    task rx_word;
        input [DATA_WIDTH-1:0] data;
        input [CTRL_WIDTH-1:0] ctrl;
    begin in_data = data; in_ctrl = ctrl; in_wr = 1; tick; end
    endtask

    task rx_end;
    begin in_wr = 0; in_data = 0; in_ctrl = 0; tick; end
    endtask

    task dump_state;
    begin
        $display("  [DBG] FIFO: head=%0d tail=%0d pkt_end=%0d pkt_rdy=%b mode=%0d",
            fifo_head_out, fifo_tail_out, fifo_pkt_end, fifo_pkt_ready, pp_fifo_mode);
        $display("  [DBG] PP: state=%0d rd_ptr=%0d active=%b owns_pb=%b",
            u_pp.state, u_pp.rd_ptr, pp_active, pp_owns_port_b);
    end
    endtask

    task wait_cpu_running;
        input integer max_cycles;
        integer cyc;
    begin
        cyc = 0;
        while (!cpu_rst_n_out && cyc < max_cycles) begin tick; cyc = cyc + 1; end
        if (cyc >= max_cycles) begin $display("[TIMEOUT] wait_cpu_running"); dump_state; end
    end endtask

    task wait_complete;
        input integer max_cycles;
        integer cyc;
    begin
        cyc = 0;
        while (!pp_active && cyc < max_cycles) begin tick; cyc = cyc + 1; end
        if (cyc >= max_cycles) begin $display("[TIMEOUT] wait_complete: not started"); dump_state; end
        else begin
            while (pp_active && cyc < max_cycles) begin tick; cyc = cyc + 1; end
            if (cyc >= max_cycles) begin $display("[TIMEOUT] wait_complete: not finished"); dump_state; end
        end
    end endtask

    task do_cpu_done;
    begin
        repeat (3) tick;
        cpu_done = 1; tick; cpu_done = 0;
    end endtask

    task capture_tx;
        input integer max_cycles;
        integer cyc;
    begin
        tx_cap_cnt = 0; cyc = 0;
        while (pp_active && cyc < max_cycles) begin
            if (out_wr && out_rdy) begin
                tx_cap_data[tx_cap_cnt] = out_data;
                tx_cap_ctrl[tx_cap_cnt] = out_ctrl;
                tx_cap_cnt = tx_cap_cnt + 1;
            end
            tick; cyc = cyc + 1;
        end
        if (cyc >= max_cycles) begin $display("[TIMEOUT] capture_tx"); dump_state; end
    end endtask

    // Settle between tests — ensure pkt_proc is idle and pkt_ready cleared
    task settle;
    begin
        repeat (5) tick;
    end endtask

    initial begin $dumpfile("pkt_proc_integ_tb.vcd"); $dumpvars(0, pkt_proc_integ_tb); end
    initial begin #5000000; $display("[TIMEOUT] Global"); $finish; end

    // ════════════════════════════════════════════════════
    // Main test sequence
    // ════════════════════════════════════════════════════
    initial begin
        $display("============================================");
        $display("  pkt_proc + conv_fifo Integration TB v4");
        $display("  Comprehensive test suite (12 tests)");
        $display("============================================");

        rst_n = 0;
        in_data = 0; in_ctrl = 0; in_wr = 0;
        out_rdy = 1; cpu_done = 0;
        repeat (3) tick;
        rst_n = 1; tick;

        // ────────────────────────────────────────────────
        // Test 1: Full end-to-end (ctrl=0x04)
        //   LOAD_IMEM(2) + LOAD_DMEM(2) + CPU_START + READBACK(1) + SEND
        // ────────────────────────────────────────────────
        $display("\n--- Test 1: Full end-to-end (ctrl=0x04) ---");

        rx_word(cmd_word(4'h1, 12'h000, 16'd2, 32'h0), 8'h04);
        rx_word({32'hBBBB_BBBB, 32'hAAAA_AAAA}, 8'h00);
        rx_word({32'hDDDD_DDDD, 32'hCCCC_CCCC}, 8'h00);
        rx_word(cmd_word(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx_word({32'h2222_2222, 32'h1111_1111}, 8'h00);
        rx_word({32'h4444_4444, 32'h3333_3333}, 8'h00);
        rx_word(cmd_word(4'h3, 12'h000, 16'd0, 32'h0000_0000), 8'h00);
        rx_word(cmd_word(4'h4, 12'h100, 16'd1, 32'h0), 8'h00);
        rx_word(cmd_word(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_cpu_running(200);

        check32(imem_mem[0], 32'hAAAA_AAAA, "IMEM[0]");
        check32(imem_mem[1], 32'hBBBB_BBBB, "IMEM[1]");
        check32(imem_mem[2], 32'hCCCC_CCCC, "IMEM[2]");
        check32(imem_mem[3], 32'hDDDD_DDDD, "IMEM[3]");
        check32(dmem_mem[0], 32'h1111_1111, "DMEM[0]");
        check32(dmem_mem[1], 32'h2222_2222, "DMEM[1]");
        check32(dmem_mem[2], 32'h3333_3333, "DMEM[2]");
        check32(dmem_mem[3], 32'h4444_4444, "DMEM[3]");

        // Simulate CPU results
        dmem_mem[12'h100] = 32'hFACE_0000;
        dmem_mem[12'h101] = 32'hFACE_0001;
        do_cpu_done;

        out_rdy = 1;
        capture_tx(500);

        checkN(tx_cap_cnt, 1, "TX word count");
        check64(tx_cap_data[0], {32'hFACE_0001, 32'hFACE_0000}, "TX[0] data");
        check8(tx_cap_ctrl[0], 8'h04, "TX[0] ctrl passthrough");
        check1(pp_active, 1'b0, "idle after T1");

        settle;

        // ────────────────────────────────────────────────
        // Test 2: Different ctrl (0x40), multi-word readback
        // ────────────────────────────────────────────────
        $display("\n--- Test 2: Different ctrl (0x40) ---");

        dmem_mem[12'h200] = 32'hD00D_0000;
        dmem_mem[12'h201] = 32'hD00D_0001;
        dmem_mem[12'h202] = 32'hD00D_0002;
        dmem_mem[12'h203] = 32'hD00D_0003;

        rx_word(cmd_word(4'h3, 12'h0, 16'd0, 32'h0), 8'h40);
        rx_word(cmd_word(4'h4, 12'h200, 16'd2, 32'h0), 8'h00);
        rx_word(cmd_word(4'h5, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_cpu_running(200);
        do_cpu_done;

        capture_tx(500);

        checkN(tx_cap_cnt, 2, "TX word count");
        check64(tx_cap_data[0], {32'hD00D_0001, 32'hD00D_0000}, "TX[0]");
        check64(tx_cap_data[1], {32'hD00D_0003, 32'hD00D_0002}, "TX[1]");
        check8(tx_cap_ctrl[0], 8'h40, "TX[0] ctrl=0x40");
        check8(tx_cap_ctrl[1], 8'h00, "TX[1] ctrl=0x00");

        settle;

        // ────────────────────────────────────────────────
        // Test 3: Load-only (no CPU_START, no readback)
        // ────────────────────────────────────────────────
        $display("\n--- Test 3: Load-only ---");

        rx_word(cmd_word(4'h1, 12'h200, 16'd1, 32'h0), 8'hFF);
        rx_word({32'h9999_9999, 32'h8888_8888}, 8'h00);
        rx_end;

        wait_complete(200);

        check32(imem_mem[10'h200], 32'h8888_8888, "IMEM[0x200]");
        check32(imem_mem[10'h201], 32'h9999_9999, "IMEM[0x201]");
        // Original IMEM persists
        check32(imem_mem[0], 32'hAAAA_AAAA, "IMEM[0] persists");

        settle;

        // ────────────────────────────────────────────────
        // Test 4: NOP commands (skip + then real command)
        // ────────────────────────────────────────────────
        $display("\n--- Test 4: NOPs ---");

        rx_word(cmd_word(4'h0, 12'h0, 16'd0, 32'h0), 8'h01);
        rx_word(cmd_word(4'h0, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_word(cmd_word(4'h2, 12'hF00, 16'd1, 32'h0), 8'h00);
        rx_word({32'h7777_7777, 32'h6666_6666}, 8'h00);
        rx_end;

        wait_complete(200);

        check32(dmem_mem[12'hF00], 32'h6666_6666, "DMEM[0xF00]");
        check32(dmem_mem[12'hF01], 32'h7777_7777, "DMEM[0xF01]");

        settle;

        // ────────────────────────────────────────────────
        // Test 5: Reuse program, new data
        //   IMEM untouched, only new DMEM + CPU + readback + send
        // ────────────────────────────────────────────────
        $display("\n--- Test 5: Reuse program, new data ---");

        rx_word(cmd_word(4'h2, 12'h000, 16'd1, 32'h0), 8'h04);
        rx_word({32'hBEEF_0001, 32'hBEEF_0000}, 8'h00);
        rx_word(cmd_word(4'h3, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_word(cmd_word(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx_word(cmd_word(4'h5, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_cpu_running(200);

        check32(dmem_mem[0], 32'hBEEF_0000, "DMEM[0] new");
        check32(dmem_mem[1], 32'hBEEF_0001, "DMEM[1] new");
        check32(imem_mem[0], 32'hAAAA_AAAA, "IMEM[0] still persists");

        // In-place result
        dmem_mem[0] = 32'h0ACE_0000;
        dmem_mem[1] = 32'h0ACE_0001;
        do_cpu_done;

        capture_tx(500);

        checkN(tx_cap_cnt, 1, "TX word count");
        check64(tx_cap_data[0], {32'h0ACE_0001, 32'h0ACE_0000}, "TX[0] reuse");
        check8(tx_cap_ctrl[0], 8'h04, "TX[0] ctrl=0x04 reuse");

        settle;

        // ────────────────────────────────────────────────
        // Test 6: TX with backpressure (out_rdy toggling)
        // ────────────────────────────────────────────────
        $display("\n--- Test 6: TX backpressure ---");

        dmem_mem[12'h300] = 32'hAB00_0000;
        dmem_mem[12'h301] = 32'hAB00_0001;
        dmem_mem[12'h302] = 32'hAB00_0002;
        dmem_mem[12'h303] = 32'hAB00_0003;

        rx_word(cmd_word(4'h3, 12'h0, 16'd0, 32'h0), 8'h10);
        rx_word(cmd_word(4'h4, 12'h300, 16'd2, 32'h0), 8'h00);
        rx_word(cmd_word(4'h5, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_cpu_running(200);
        do_cpu_done;

        // Capture with toggling backpressure
        tx_cap_cnt = 0;
        begin : bp_block
            integer cyc;
            for (cyc = 0; cyc < 500; cyc = cyc + 1) begin
                out_rdy = (cyc % 2 == 0) ? 1'b1 : 1'b0;
                if (out_wr && out_rdy) begin
                    tx_cap_data[tx_cap_cnt] = out_data;
                    tx_cap_ctrl[tx_cap_cnt] = out_ctrl;
                    tx_cap_cnt = tx_cap_cnt + 1;
                end
                tick;
                if (!pp_active) disable bp_block;
            end
        end
        out_rdy = 1;

        checkN(tx_cap_cnt, 2, "BP TX word count");
        check64(tx_cap_data[0], {32'hAB00_0001, 32'hAB00_0000}, "BP TX[0]");
        check64(tx_cap_data[1], {32'hAB00_0003, 32'hAB00_0002}, "BP TX[1]");
        check8(tx_cap_ctrl[0], 8'h10, "BP TX[0] ctrl=0x10");
        check8(tx_cap_ctrl[1], 8'h00, "BP TX[1] ctrl=0x00");

        settle;

        // ────────────────────────────────────────────────
        // Test 7: CPU_START only (no load, no readback)
        //   Just run CPU and return to idle
        // ────────────────────────────────────────────────
        $display("\n--- Test 7: CPU_START only ---");

        rx_word(cmd_word(4'h3, 12'h0, 16'd0, 32'h0000_00FC), 8'h01);
        rx_end;

        wait_cpu_running(200);
        check32(entry_pc, 32'h0000_00FC, "entry_pc=0xFC");
        check1(cpu_rst_n_out, 1'b1, "cpu running");
        check1(pp_owns_port_b, 1'b0, "port_b released");

        do_cpu_done;

        // Wait for pkt_proc to return to idle
        begin : t7_wait
            integer cyc;
            for (cyc = 0; cyc < 100; cyc = cyc + 1) begin
                tick;
                if (!pp_active) disable t7_wait;
            end
        end
        check1(pp_active, 1'b0, "idle after CPU-only");

        settle;

        // ────────────────────────────────────────────────
        // Test 8: Large IMEM load (8 data words = 16 instrs)
        // ────────────────────────────────────────────────
        $display("\n--- Test 8: Large IMEM load (16 instrs) ---");

        rx_word(cmd_word(4'h1, 12'h000, 16'd8, 32'h0), 8'h02);
        for (i = 0; i < 8; i = i + 1) begin
            tmp_lo = i * 2;
            tmp_hi = i * 2 + 1;
            rx_word({tmp_hi, tmp_lo}, 8'h00);
        end
        rx_end;

        wait_complete(300);

        for (i = 0; i < 16; i = i + 1) begin
            check32(imem_mem[i], i[31:0], "IMEM[i] large load");
        end

        settle;

        // ────────────────────────────────────────────────
        // Test 9: Large DMEM load (8 data words = 16 vals)
        // ────────────────────────────────────────────────
        $display("\n--- Test 9: Large DMEM load ---");

        rx_word(cmd_word(4'h2, 12'h800, 16'd8, 32'h0), 8'h02);
        for (i = 0; i < 8; i = i + 1) begin
            tmp_lo = 32'hAA00_0000 + i * 2;
            tmp_hi = 32'hAA00_0000 + i * 2 + 1;
            rx_word({tmp_hi, tmp_lo}, 8'h00);
        end
        rx_end;

        wait_complete(300);

        for (i = 0; i < 16; i = i + 1) begin
            check32(dmem_mem[12'h800 + i], 32'hAA00_0000 + i[31:0], "DMEM[0x800+i] large load");
        end

        settle;

        // ────────────────────────────────────────────────
        // Test 10: Multiple LOAD commands in sequence
        //   LOAD_IMEM at 0x000, LOAD_IMEM at 0x100, LOAD_DMEM at 0x000
        // ────────────────────────────────────────────────
        $display("\n--- Test 10: Multi-segment load ---");

        rx_word(cmd_word(4'h1, 12'h000, 16'd1, 32'h0), 8'h08);
        rx_word({32'h1111_1111, 32'h0000_0000}, 8'h00);
        rx_word(cmd_word(4'h1, 12'h100, 16'd1, 32'h0), 8'h00);
        rx_word({32'h3333_3333, 32'h2222_2222}, 8'h00);
        rx_word(cmd_word(4'h2, 12'h000, 16'd1, 32'h0), 8'h00);
        rx_word({32'h5555_5555, 32'h4444_4444}, 8'h00);
        rx_end;

        wait_complete(300);

        check32(imem_mem[10'h000], 32'h0000_0000, "multi IMEM[0x000]");
        check32(imem_mem[10'h001], 32'h1111_1111, "multi IMEM[0x001]");
        check32(imem_mem[10'h100], 32'h2222_2222, "multi IMEM[0x100]");
        check32(imem_mem[10'h101], 32'h3333_3333, "multi IMEM[0x101]");
        check32(dmem_mem[12'h000], 32'h4444_4444, "multi DMEM[0x000]");
        check32(dmem_mem[12'h001], 32'h5555_5555, "multi DMEM[0x001]");

        settle;

        // ────────────────────────────────────────────────
        // Test 11: Back-to-back packets (rapid fire)
        //   Packet A: load + cpu_start + readback + send
        //   Packet B: same pattern, different data
        // ────────────────────────────────────────────────
        $display("\n--- Test 11: Back-to-back packets ---");

        // Packet A
        rx_word(cmd_word(4'h2, 12'h000, 16'd1, 32'h0), 8'h04);
        rx_word({32'h0000_AAAA, 32'h0000_BBBB}, 8'h00);
        rx_word(cmd_word(4'h3, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_word(cmd_word(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx_word(cmd_word(4'h5, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_cpu_running(200);
        dmem_mem[0] = 32'hAAAA_1111;
        dmem_mem[1] = 32'hAAAA_2222;
        do_cpu_done;
        capture_tx(500);

        checkN(tx_cap_cnt, 1, "PktA TX count");
        check64(tx_cap_data[0], {32'hAAAA_2222, 32'hAAAA_1111}, "PktA TX[0]");
        check8(tx_cap_ctrl[0], 8'h04, "PktA ctrl=0x04");

        settle;

        // Packet B (immediately after)
        rx_word(cmd_word(4'h2, 12'h000, 16'd1, 32'h0), 8'h40);
        rx_word({32'h0000_CCCC, 32'h0000_DDDD}, 8'h00);
        rx_word(cmd_word(4'h3, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_word(cmd_word(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx_word(cmd_word(4'h5, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_cpu_running(200);
        dmem_mem[0] = 32'hBBBB_3333;
        dmem_mem[1] = 32'hBBBB_4444;
        do_cpu_done;
        capture_tx(500);

        checkN(tx_cap_cnt, 1, "PktB TX count");
        check64(tx_cap_data[0], {32'hBBBB_4444, 32'hBBBB_3333}, "PktB TX[0]");
        check8(tx_cap_ctrl[0], 8'h40, "PktB ctrl=0x40 (changed)");

        settle;

        // ────────────────────────────────────────────────
        // Test 12: Readback from high DMEM address
        //   Verifies 12-bit address space works end-to-end
        // ────────────────────────────────────────────────
        $display("\n--- Test 12: High DMEM address readback ---");

        dmem_mem[12'hFF0] = 32'hF1F0_0000;
        dmem_mem[12'hFF1] = 32'hF1F0_0001;
        dmem_mem[12'hFF2] = 32'hF1F0_0002;
        dmem_mem[12'hFF3] = 32'hF1F0_0003;

        rx_word(cmd_word(4'h3, 12'h0, 16'd0, 32'h0), 8'h01);
        rx_word(cmd_word(4'h4, 12'hFF0, 16'd2, 32'h0), 8'h00);
        rx_word(cmd_word(4'h5, 12'h0, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_cpu_running(200);
        do_cpu_done;
        capture_tx(500);

        checkN(tx_cap_cnt, 2, "High addr TX count");
        check64(tx_cap_data[0], {32'hF1F0_0001, 32'hF1F0_0000}, "High TX[0]");
        check64(tx_cap_data[1], {32'hF1F0_0003, 32'hF1F0_0002}, "High TX[1]");
        check8(tx_cap_ctrl[0], 8'h01, "High TX ctrl=0x01");

        // ── Summary ──────────────────────────────────────
        $display("\n============================================");
        $display("  Integration TB v4 — Summary");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        $display("  TOTAL:  %0d", pass_count + fail_count);
        $display("============================================");
        if (fail_count == 0) $display(">>> ALL TESTS PASSED <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $finish;
    end

endmodule