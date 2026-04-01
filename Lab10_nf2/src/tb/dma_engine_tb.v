/* file: dma_engine_tb.v
 * Testbench for dma_engine — DMA address sequencer with width conversion.
 * Includes simple BRAM models for CPU DMEM, GPU IMEM, GPU DMEM banks.
 *
 * Author: Jeremy Cai
 * Date: Mar. 4, 2026
 * Version: 1.0
 */

`timescale 1ns / 1ps

`include "dma_engine.v"

module dma_engine_tb;

    // ================================================================
    // Clock / Reset
    // ================================================================
    reg clk, rst_n;
    localparam CLK_PERIOD = 10;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ================================================================
    // DUT I/O
    // ================================================================
    // CP10 interface
    reg [31:0] dma_src_addr, dma_dst_addr;
    reg [15:0] dma_xfer_len;
    reg dma_start, dma_dir, dma_tgt;
    reg [1:0] dma_bank;
    reg dma_auto_inc, dma_burst_all;
    wire dma_busy, dma_error;
    wire [31:0] dma_cur_addr;

    // CPU DMEM Port B
    wire [11:0] cpu_dmem_addr;
    wire [31:0] cpu_dmem_din;
    wire cpu_dmem_we;
    wire [31:0] cpu_dmem_dout;

    // GPU IMEM ext
    wire [7:0] gpu_imem_addr;
    wire [31:0] gpu_imem_din;
    wire gpu_imem_we;

    // GPU DMEM ext
    wire [1:0] gpu_dmem_sel;
    wire [9:0] gpu_dmem_addr;
    wire [15:0] gpu_dmem_din;
    wire gpu_dmem_we;
    wire [15:0] gpu_dmem_dout;

    // ================================================================
    // DUT
    // ================================================================
    dma_engine u_dut (
        .clk(clk), .rst_n(rst_n),
        .dma_src_addr(dma_src_addr), .dma_dst_addr(dma_dst_addr),
        .dma_xfer_len(dma_xfer_len),
        .dma_start(dma_start), .dma_dir(dma_dir), .dma_tgt(dma_tgt),
        .dma_bank(dma_bank), .dma_auto_inc(dma_auto_inc),
        .dma_burst_all(dma_burst_all),
        .dma_busy(dma_busy), .dma_error(dma_error),
        .dma_cur_addr(dma_cur_addr),
        .cpu_dmem_addr(cpu_dmem_addr), .cpu_dmem_din(cpu_dmem_din),
        .cpu_dmem_we(cpu_dmem_we), .cpu_dmem_dout(cpu_dmem_dout),
        .gpu_imem_addr(gpu_imem_addr), .gpu_imem_din(gpu_imem_din),
        .gpu_imem_we(gpu_imem_we),
        .gpu_dmem_sel(gpu_dmem_sel), .gpu_dmem_addr(gpu_dmem_addr),
        .gpu_dmem_din(gpu_dmem_din), .gpu_dmem_we(gpu_dmem_we),
        .gpu_dmem_dout(gpu_dmem_dout)
    );

    // ================================================================
    // BRAM Models
    // ================================================================

    // CPU DMEM: 4096×32b, synchronous read (1-cycle latency)
    reg [31:0] cpu_dmem [0:4095];
    reg [31:0] cpu_dmem_dout_r;
    always @(posedge clk) begin
        if (cpu_dmem_we)
            cpu_dmem[cpu_dmem_addr] <= cpu_dmem_din;
        cpu_dmem_dout_r <= cpu_dmem[cpu_dmem_addr];
    end
    assign cpu_dmem_dout = cpu_dmem_dout_r;

    // GPU IMEM: 256×32b, synchronous write
    reg [31:0] gpu_imem [0:255];
    always @(posedge clk) begin
        if (gpu_imem_we)
            gpu_imem[gpu_imem_addr] <= gpu_imem_din;
    end

    // GPU DMEM: 4 banks × 1024×16b, synchronous read
    reg [15:0] gpu_dmem_bank0 [0:1023];
    reg [15:0] gpu_dmem_bank1 [0:1023];
    reg [15:0] gpu_dmem_bank2 [0:1023];
    reg [15:0] gpu_dmem_bank3 [0:1023];
    reg [15:0] gpu_dmem_dout_r;
    always @(posedge clk) begin
        // Write
        if (gpu_dmem_we) begin
            case (gpu_dmem_sel)
                2'd0: gpu_dmem_bank0[gpu_dmem_addr] <= gpu_dmem_din;
                2'd1: gpu_dmem_bank1[gpu_dmem_addr] <= gpu_dmem_din;
                2'd2: gpu_dmem_bank2[gpu_dmem_addr] <= gpu_dmem_din;
                2'd3: gpu_dmem_bank3[gpu_dmem_addr] <= gpu_dmem_din;
            endcase
        end
        // Read (bank-muxed)
        case (gpu_dmem_sel)
            2'd0: gpu_dmem_dout_r <= gpu_dmem_bank0[gpu_dmem_addr];
            2'd1: gpu_dmem_dout_r <= gpu_dmem_bank1[gpu_dmem_addr];
            2'd2: gpu_dmem_dout_r <= gpu_dmem_bank2[gpu_dmem_addr];
            2'd3: gpu_dmem_dout_r <= gpu_dmem_bank3[gpu_dmem_addr];
        endcase
    end
    assign gpu_dmem_dout = gpu_dmem_dout_r;

    // ================================================================
    // Test Infrastructure
    // ================================================================
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;
    integer i;

    task tick;
    begin
        @(posedge clk); #1;
    end
    endtask

    task wait_idle;
        input integer max_cycles;
        integer cyc;
    begin
        cyc = 0;
        while (dma_busy && cyc < max_cycles) begin
            tick;
            cyc = cyc + 1;
        end
        if (dma_busy)
            $display("[TIMEOUT] DMA still busy after %0d cycles", max_cycles);
    end
    endtask

    task fire_dma;
        input [31:0] src;
        input [31:0] dst;
        input [15:0] len;
        input dir_v;
        input tgt_v;
        input [1:0] bank_v;
        input auto_inc_v;
        input burst_all_v;
    begin
        dma_src_addr = src;
        dma_dst_addr = dst;
        dma_xfer_len = len;
        dma_dir = dir_v;
        dma_tgt = tgt_v;
        dma_bank = bank_v;
        dma_auto_inc = auto_inc_v;
        dma_burst_all = burst_all_v;
        dma_start = 1;
        tick;
        dma_start = 0;
    end
    endtask

    task check32;
        input [31:0] got;
        input [31:0] expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (got === expected) begin
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] T%0d: %0s  got=0x%08h, expected=0x%08h",
                test_num, name, got, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check16;
        input [15:0] got;
        input [15:0] expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (got === expected) begin
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] T%0d: %0s  got=0x%04h, expected=0x%04h",
                test_num, name, got, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check1;
        input val;
        input expected;
        input [64*8-1:0] name;
    begin
        test_num = test_num + 1;
        if (val === expected) begin
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] T%0d: %0s  got=%b, expected=%b",
                test_num, name, val, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    // ================================================================
    // Waveform Dump
    // ================================================================
    initial begin
        $dumpfile("dma_engine_tb.vcd");
        $dumpvars(0, dma_engine_tb);
    end

    initial begin
        #500000;
        $display("[TIMEOUT] Global simulation timeout");
        $finish;
    end

    // ================================================================
    // Main Test Sequence
    // ================================================================
    initial begin
        $display("============================================");
        $display("  DMA Engine Testbench v1.0 — Starting");
        $display("============================================");

        // ── Reset ────────────────────────────────────────
        rst_n = 0;
        dma_start = 0; dma_dir = 0; dma_tgt = 0;
        dma_src_addr = 0; dma_dst_addr = 0; dma_xfer_len = 0;
        dma_bank = 0; dma_auto_inc = 0; dma_burst_all = 0;
        repeat (3) tick;
        rst_n = 1;
        tick;

        // ────────────────────────────────────────────────
        // Test 1: D_IMEM — CPU DMEM → GPU IMEM (32→32)
        // ────────────────────────────────────────────────
        $display("\n--- Test 1: D_IMEM (CPU DMEM → GPU IMEM, 4 words) ---");
        // Preload CPU DMEM[0..3]
        cpu_dmem[0] = 32'hAAAA_0000;
        cpu_dmem[1] = 32'hBBBB_1111;
        cpu_dmem[2] = 32'hCCCC_2222;
        cpu_dmem[3] = 32'hDDDD_3333;

        // dir=0, tgt=1: D_IMEM
        fire_dma(32'd0, 32'd10, 16'd4, 1'b0, 1'b1, 2'd0, 1'b0, 1'b0);
        wait_idle(200);

        check1(dma_busy, 1'b0, "D_IMEM: busy deasserted");
        check1(dma_error, 1'b0, "D_IMEM: no error");
        check32(gpu_imem[10], 32'hAAAA_0000, "GPU IMEM[10]");
        check32(gpu_imem[11], 32'hBBBB_1111, "GPU IMEM[11]");
        check32(gpu_imem[12], 32'hCCCC_2222, "GPU IMEM[12]");
        check32(gpu_imem[13], 32'hDDDD_3333, "GPU IMEM[13]");

        // ────────────────────────────────────────────────
        // Test 2: D_UNPACK — CPU DMEM → GPU DMEM (32→16)
        // ────────────────────────────────────────────────
        $display("\n--- Test 2: D_UNPACK (CPU DMEM → GPU DMEM bank 0, 3 words) ---");
        cpu_dmem[100] = 32'h1234_5678;
        cpu_dmem[101] = 32'hABCD_EF01;
        cpu_dmem[102] = 32'h9876_5432;

        // dir=0, tgt=0, bank=0: D_UNPACK
        fire_dma(32'd100, 32'd0, 16'd3, 1'b0, 1'b0, 2'd0, 1'b0, 1'b0);
        wait_idle(200);

        check1(dma_busy, 1'b0, "D_UNPACK: busy deasserted");
        check1(dma_error, 1'b0, "D_UNPACK: no error");
        // 32'h1234_5678 → lo=0x5678, hi=0x1234
        check16(gpu_dmem_bank0[0], 16'h5678, "GPU DMEM0[0] (lo of word 0)");
        check16(gpu_dmem_bank0[1], 16'h1234, "GPU DMEM0[1] (hi of word 0)");
        // 32'hABCD_EF01 → lo=0xEF01, hi=0xABCD
        check16(gpu_dmem_bank0[2], 16'hEF01, "GPU DMEM0[2] (lo of word 1)");
        check16(gpu_dmem_bank0[3], 16'hABCD, "GPU DMEM0[3] (hi of word 1)");
        // 32'h9876_5432 → lo=0x5432, hi=0x9876
        check16(gpu_dmem_bank0[4], 16'h5432, "GPU DMEM0[4] (lo of word 2)");
        check16(gpu_dmem_bank0[5], 16'h9876, "GPU DMEM0[5] (hi of word 2)");

        // ────────────────────────────────────────────────
        // Test 3: D_UNPACK to bank 2
        // ────────────────────────────────────────────────
        $display("\n--- Test 3: D_UNPACK to bank 2 ---");
        cpu_dmem[200] = 32'hFEDC_BA98;

        fire_dma(32'd200, 32'd0, 16'd1, 1'b0, 1'b0, 2'd2, 1'b0, 1'b0);
        wait_idle(200);

        check16(gpu_dmem_bank2[0], 16'hBA98, "GPU DMEM2[0]");
        check16(gpu_dmem_bank2[1], 16'hFEDC, "GPU DMEM2[1]");

        // ────────────────────────────────────────────────
        // Test 4: D_PACK — GPU DMEM → CPU DMEM (16→32)
        // ────────────────────────────────────────────────
        $display("\n--- Test 4: D_PACK (GPU DMEM bank 0 → CPU DMEM, 3 words) ---");
        // GPU DMEM bank 0 already has data from Test 2
        // Read back 3 CPU words (6 GPU words)

        fire_dma(32'd0, 32'd500, 16'd3, 1'b1, 1'b0, 2'd0, 1'b0, 1'b0);
        wait_idle(200);

        check1(dma_busy, 1'b0, "D_PACK: busy deasserted");
        check1(dma_error, 1'b0, "D_PACK: no error");
        check32(cpu_dmem[500], 32'h1234_5678, "CPU DMEM[500] (pack word 0)");
        check32(cpu_dmem[501], 32'hABCD_EF01, "CPU DMEM[501] (pack word 1)");
        check32(cpu_dmem[502], 32'h9876_5432, "CPU DMEM[502] (pack word 2)");

        // ────────────────────────────────────────────────
        // Test 5: D_IMEM single word
        // ────────────────────────────────────────────────
        $display("\n--- Test 5: D_IMEM single word ---");
        cpu_dmem[50] = 32'hDEAD_BEEF;

        fire_dma(32'd50, 32'd0, 16'd1, 1'b0, 1'b1, 2'd0, 1'b0, 1'b0);
        wait_idle(200);

        check32(gpu_imem[0], 32'hDEAD_BEEF, "GPU IMEM[0] single word");

        // ────────────────────────────────────────────────
        // Test 6: Invalid dir=1, tgt=1 → error
        // ────────────────────────────────────────────────
        $display("\n--- Test 6: Invalid transfer (dir=1, tgt=1) ---");
        fire_dma(32'd0, 32'd0, 16'd1, 1'b1, 1'b1, 2'd0, 1'b0, 1'b0);
        wait_idle(200);

        check1(dma_error, 1'b1, "Invalid mode: error asserted");
        check1(dma_busy, 1'b0, "Invalid mode: busy deasserted");

        // ────────────────────────────────────────────────
        // Test 7: Zero length → immediate done
        // ────────────────────────────────────────────────
        $display("\n--- Test 7: Zero length transfer ---");
        fire_dma(32'd0, 32'd0, 16'd0, 1'b0, 1'b0, 2'd0, 1'b0, 1'b0);
        wait_idle(200);

        check1(dma_busy, 1'b0, "Zero len: busy deasserted");
        check1(dma_error, 1'b0, "Zero len: no error");

        // ────────────────────────────────────────────────
        // Test 8: burst_all — D_UNPACK to all 4 banks
        // ────────────────────────────────────────────────
        $display("\n--- Test 8: burst_all D_UNPACK (4 banks × 2 words) ---");
        // CPU DMEM[300..307]: 8 words → 4 banks × 2 words × 2 GPU words each
        cpu_dmem[300] = 32'hB0_00_B0_01;  // bank0
        cpu_dmem[301] = 32'hB0_02_B0_03;
        cpu_dmem[302] = 32'hB1_00_B1_01;  // bank1
        cpu_dmem[303] = 32'hB1_02_B1_03;
        cpu_dmem[304] = 32'hB2_00_B2_01;  // bank2
        cpu_dmem[305] = 32'hB2_02_B2_03;
        cpu_dmem[306] = 32'hB3_00_B3_01;  // bank3
        cpu_dmem[307] = 32'hB3_02_B3_03;

        // dir=0, tgt=0, bank=0, burst_all=1, len=2 per bank
        fire_dma(32'd300, 32'd20, 16'd2, 1'b0, 1'b0, 2'd0, 1'b0, 1'b1);
        wait_idle(500);

        check1(dma_busy, 1'b0, "burst_all: busy deasserted");
        // Bank 0: word 0 = 32'hB000_B001 → GPU[20]=0xB001, GPU[21]=0xB000
        check16(gpu_dmem_bank0[20], 16'hB001, "burst bank0[20]");
        check16(gpu_dmem_bank0[21], 16'hB000, "burst bank0[21]");
        check16(gpu_dmem_bank0[22], 16'hB003, "burst bank0[22]");
        check16(gpu_dmem_bank0[23], 16'hB002, "burst bank0[23]");
        // Bank 1
        check16(gpu_dmem_bank1[20], 16'hB101, "burst bank1[20]");
        check16(gpu_dmem_bank1[21], 16'hB100, "burst bank1[21]");
        // Bank 2
        check16(gpu_dmem_bank2[20], 16'hB201, "burst bank2[20]");
        check16(gpu_dmem_bank2[21], 16'hB200, "burst bank2[21]");
        // Bank 3
        check16(gpu_dmem_bank3[20], 16'hB301, "burst bank3[20]");
        check16(gpu_dmem_bank3[21], 16'hB300, "burst bank3[21]");

        // ────────────────────────────────────────────────
        // Test 9: auto_inc — verify bank increments after transfer
        // ────────────────────────────────────────────────
        $display("\n--- Test 9: auto_inc ---");
        cpu_dmem[400] = 32'hAAAA_BBBB;

        // bank=1, auto_inc=1
        fire_dma(32'd400, 32'd50, 16'd1, 1'b0, 1'b0, 2'd1, 1'b1, 1'b0);
        wait_idle(200);

        check16(gpu_dmem_bank1[50], 16'hBBBB, "auto_inc: bank1[50]");
        check16(gpu_dmem_bank1[51], 16'hAAAA, "auto_inc: bank1[51]");
        // After done, cur_bank should have incremented to 2 internally
        // (We can't observe cur_bank directly, but the DMA engine stores it)

        // ────────────────────────────────────────────────
        // Test 10: D_PACK burst_all — readback all 4 banks
        // ────────────────────────────────────────────────
        $display("\n--- Test 10: D_PACK burst_all readback ---");
        // Use data from Test 8. Read 2 words from each bank.
        fire_dma(32'd20, 32'd600, 16'd2, 1'b1, 1'b0, 2'd0, 1'b0, 1'b1);
        wait_idle(500);

        check1(dma_busy, 1'b0, "D_PACK burst: busy deasserted");
        // Bank 0
        check32(cpu_dmem[600], 32'hB000_B001, "pack burst bank0 w0");
        check32(cpu_dmem[601], 32'hB002_B003, "pack burst bank0 w1");
        // Bank 1
        check32(cpu_dmem[602], 32'hB100_B101, "pack burst bank1 w0");
        check32(cpu_dmem[603], 32'hB102_B103, "pack burst bank1 w1");
        // Bank 2
        check32(cpu_dmem[604], 32'hB200_B201, "pack burst bank2 w0");
        check32(cpu_dmem[605], 32'hB202_B203, "pack burst bank2 w1");
        // Bank 3
        check32(cpu_dmem[606], 32'hB300_B301, "pack burst bank3 w0");
        check32(cpu_dmem[607], 32'hB302_B303, "pack burst bank3 w1");

        // ────────────────────────────────────────────────
        // Summary
        // ────────────────────────────────────────────────
        $display("\n============================================");
        $display("  DMA Engine Testbench v1.0 — Summary");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        $display("  TOTAL:  %0d", pass_count + fail_count);
        $display("============================================");
        if (fail_count == 0)
            $display(">>> ALL TESTS PASSED <<<");
        else
            $display(">>> SOME TESTS FAILED <<<");
        $finish;
    end

endmodule