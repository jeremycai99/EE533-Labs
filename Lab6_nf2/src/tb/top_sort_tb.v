`timescale 1ns / 1ps

`include "define.v"
`include "top.v"

module top_sort_tb;

    // =========================================================
    // Parameters
    // =========================================================

    localparam CLK_PERIOD     = 10;
    localparam MEM_DEPTH      = 4096;
    localparam ARRAY_SIZE     = 10;
    localparam SCAN_RANGE     = 768;

    localparam [31:0] HALT_INSTR     = 32'hEAFF_FFFE;   // B . (branch-to-self)
    localparam [31:0] HALT_BYTE_ADDR = 32'h0000_0200;
    localparam [31:0] HALT_WORD      = HALT_BYTE_ADDR >> 2; // 128

    localparam [31:0] INIT_SP        = 32'h0000_0800;    // 2048 bytes → stack frame ~words 497-506

    localparam POLL_INTERVAL = 100;
    localparam SORT_TIMEOUT  = 5_000;

    localparam [31:0] IMEM_BASE = 32'h0000_0000;
    localparam [31:0] DMEM_BASE = 32'h8000_0000;

    // ILA register addresses
    localparam ILA_CTRL      = 3'h0;
    localparam ILA_STATUS    = 3'h1;
    localparam ILA_PROBE_SEL = 3'h2;
    localparam ILA_PROBE     = 3'h3;

    // conn_status bit positions
    localparam CONN_BUSY     = 5;
    localparam CONN_FIFO_NE  = 4;

    // ILA probe selectors
    localparam [4:0] PROBE_PC       = 5'h00;
    localparam [4:0] PROBE_REG_BASE = 5'h10;

`ifdef SORT_HEX
    localparam HEX_FILE = `SORT_HEX;
`else
    localparam HEX_FILE = "../hex/sort_imem.txt";
`endif

    // =========================================================
    // DUT Signals
    // =========================================================

    reg                         clk, rst_n, debug_mode;
    reg                         user_valid, user_cmd;
    reg  [`MMIO_ADDR_WIDTH-1:0] user_addr;
    reg  [`MMIO_DATA_WIDTH-1:0] user_wdata;
    wire                        user_ready;
    wire [`MMIO_DATA_WIDTH-1:0] user_rdata;
    wire [`MMIO_ADDR_WIDTH-1:0] status;
    wire [7:0]                  conn_status;
    wire [`MMIO_DATA_WIDTH-1:0] txn_quality;
    wire [`MMIO_DATA_WIDTH-1:0] txn_counters;
    reg                         clear_stats;
    reg  [2:0]                  ila_addr;
    reg  [`MMIO_DATA_WIDTH-1:0] ila_din;
    reg                         ila_we;
    wire [`MMIO_DATA_WIDTH-1:0] ila_dout;

    // =========================================================
    // Test State
    // =========================================================

    reg [`DATA_WIDTH-1:0]      local_mem [0:MEM_DEPTH-1];
    reg [`DATA_WIDTH-1:0]      expected  [0:ARRAY_SIZE-1];
    reg [`MMIO_DATA_WIDTH-1:0] dmem_snap [0:SCAN_RANGE-1];

    integer pass_count = 0;
    integer fail_count = 0;
    integer last_nz;

    // =========================================================
    // DUT
    // =========================================================

    top u_top (
        .clk          (clk),
        .rst_n        (rst_n),
        .debug_mode   (debug_mode),
        .user_valid   (user_valid),
        .user_cmd     (user_cmd),
        .user_addr    (user_addr),
        .user_wdata   (user_wdata),
        .user_ready   (user_ready),
        .user_rdata   (user_rdata),
        .status       (status),
        .conn_status  (conn_status),
        .txn_quality  (txn_quality),
        .txn_counters (txn_counters),
        .clear_stats  (clear_stats),
        .ila_addr     (ila_addr),
        .ila_din      (ila_din),
        .ila_we       (ila_we),
        .ila_dout     (ila_dout)
    );

    // =========================================================
    // Clock
    // =========================================================

    initial clk = 0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    // =========================================================
    // Helper Tasks
    // =========================================================

    task check;
        input cond;
        input [511:0] msg;
        begin
            if (cond) begin
                $display("  [PASS] %0s", msg);
                pass_count = pass_count + 1;
            end else begin
                $display("  [FAIL] %0s", msg);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // --- ILA register access ---

    task ila_write;
        input [2:0] addr;
        input [`MMIO_DATA_WIDTH-1:0] data;
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_din  = data;
            ila_we   = 1;
            @(posedge clk);
            ila_we  = 0;
            ila_din = {`MMIO_DATA_WIDTH{1'b0}};
            @(posedge clk);
        end
    endtask

    task ila_read;
        input  [2:0] addr;
        output [`MMIO_DATA_WIDTH-1:0] data;
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_we   = 0;
            @(posedge clk);
            #1;
            data = ila_dout;
        end
    endtask

    task read_probe;
        input  [4:0] sel;
        output [`MMIO_DATA_WIDTH-1:0] data;
        begin
            ila_write(ILA_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, sel});
            @(posedge clk);
            ila_read(ILA_PROBE, data);
        end
    endtask

    // --- MMIO via soc_driver ---

    task submit_txn;
        input                         cmd;
        input [`MMIO_ADDR_WIDTH-1:0]  addr;
        input [`MMIO_DATA_WIDTH-1:0]  wdata;
        begin
            @(posedge clk);
            while (!user_ready) @(posedge clk);
            user_cmd   = cmd;
            user_addr  = addr;
            user_wdata = wdata;
            user_valid = 1;
            @(posedge clk);
            user_valid = 0;
        end
    endtask

    task wait_done;
        output reg success;
        integer t;
        begin
            success = 0;
            t = 0;
            repeat (2) @(posedge clk);
            while (t < 2000) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NE]) begin
                    success = 1;
                    t = 2000;
                end
                t = t + 1;
            end
            @(posedge clk);
        end
    endtask

    task mmio_wr;
        input [`MMIO_ADDR_WIDTH-1:0] addr;
        input [`MMIO_DATA_WIDTH-1:0] data;
        reg ok;
        begin
            submit_txn(1'b1, addr, data);
            wait_done(ok);
        end
    endtask

    task mmio_rd;
        input  [`MMIO_ADDR_WIDTH-1:0]  addr;
        output [`MMIO_DATA_WIDTH-1:0]  rdata;
        reg ok;
        begin
            submit_txn(1'b0, addr, {`MMIO_DATA_WIDTH{1'b0}});
            wait_done(ok);
            rdata = user_rdata;
        end
    endtask

    // --- CPU register init (hierarchical, simulation only) ---

    task init_cpu_regs;
        input [`DATA_WIDTH-1:0] sp_val;
        input [`DATA_WIDTH-1:0] lr_val;
        integer r;
        begin
            for (r = 0; r < (1 << `REG_ADDR_WIDTH); r = r + 1)
                u_top.u_soc.u_cpu.u_regfile.regs[r] = {`DATA_WIDTH{1'b0}};
            u_top.u_soc.u_cpu.u_regfile.regs[13] = sp_val;
            u_top.u_soc.u_cpu.u_regfile.regs[14] = lr_val;
        end
    endtask

    // =========================================================
    // Main Test
    // =========================================================

    reg [`MMIO_DATA_WIDTH-1:0] rd;
    reg cpu_halted, array_found;
    integer found_at;

    initial begin
        $display("\n########################################");
        $display("###  TOP — BUBBLE SORT INTEGRATION   ###");
        $display("########################################\n");

        // ── Reset ──
        rst_n      = 0;
        debug_mode = 0;
        user_valid = 0;
        user_cmd   = 0;
        user_addr  = 0;
        user_wdata = 0;
        clear_stats = 0;
        ila_addr   = 0;
        ila_din    = 0;
        ila_we     = 0;

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 10);

        #1;
        check(user_ready === 1'b1, "Driver ready after reset");

        // ── Load hex into local array ──
        $display("  Loading hex: %0s", HEX_FILE);
        begin : load_hex
            integer i;
            for (i = 0; i < MEM_DEPTH; i = i + 1)
                local_mem[i] = {`DATA_WIDTH{1'b0}};
            $readmemh(HEX_FILE, local_mem);
            if (local_mem[0] === {`DATA_WIDTH{1'b0}} &&
                local_mem[1] === {`DATA_WIDTH{1'b0}}) begin
                $display("  *** ERROR: hex file empty or missing ***");
                check(1'b0, "Hex file loaded");
                $finish;
            end
        end

        // ── Find extent, patch halt ──
        begin : find_extent
            integer i;
            last_nz = 0;
            for (i = 0; i < MEM_DEPTH; i = i + 1)
                if (local_mem[i] !== {`DATA_WIDTH{1'b0}})
                    last_nz = i;
        end
        local_mem[HALT_WORD] = HALT_INSTR;
        if (HALT_WORD > last_nz) last_nz = HALT_WORD;

        $display("  Extent: word 0..%0d (%0d words)", last_nz, last_nz + 1);
        $display("  First 4: %08H %08H %08H %08H",
                 local_mem[0], local_mem[1], local_mem[2], local_mem[3]);
        $display("  Halt @ word %0d (byte 0x%04H) = 0x%08H",
                 HALT_WORD, HALT_WORD << 2, HALT_INSTR);

        // ── Expected sorted output ──
        expected[0] = 32'hfffffe39;  // -455
        expected[1] = 32'hffffffc8;  //  -56
        expected[2] = 32'h00000000;  //    0
        expected[3] = 32'h00000002;  //    2
        expected[4] = 32'h0000000a;  //   10
        expected[5] = 32'h00000041;  //   65
        expected[6] = 32'h00000062;  //   98
        expected[7] = 32'h0000007b;  //  123
        expected[8] = 32'h0000007d;  //  125
        expected[9] = 32'h00000143;  //  323

        // ── Write IMEM ──
        $display("\n  Writing %0d words to IMEM...", last_nz + 1);
        begin : wr_imem
            integer i, cnt;
            cnt = 0;
            for (i = 0; i <= last_nz; i = i + 1) begin
                mmio_wr(IMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0],
                        {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, local_mem[i]});
                cnt = cnt + 1;
                if (cnt % 50 == 0)
                    $display("    IMEM %0d / %0d", cnt, last_nz + 1);
            end
            $display("    IMEM done (%0d words)", cnt);
        end

        // ── Write DMEM ──
        $display("  Writing %0d words to DMEM...", last_nz + 1);
        begin : wr_dmem
            integer i, cnt;
            cnt = 0;
            for (i = 0; i <= last_nz; i = i + 1) begin
                mmio_wr(DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0],
                        {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, local_mem[i]});
                cnt = cnt + 1;
                if (cnt % 50 == 0)
                    $display("    DMEM %0d / %0d", cnt, last_nz + 1);
            end
            $display("    DMEM done (%0d words)", cnt);
        end
        check(last_nz > 10, "Program loaded to IMEM and DMEM");

        // ── Spot-check ──
        $display("\n  Spot-checking...");
        begin : spotchk
            reg [`MMIO_DATA_WIDTH-1:0] sc;

            mmio_rd(IMEM_BASE | 32'h0, sc);
            check(sc[`DATA_WIDTH-1:0] === local_mem[0], "IMEM[0] matches hex");
            $display("    IMEM[0] = 0x%08H (expected 0x%08H)",
                     sc[`DATA_WIDTH-1:0], local_mem[0]);

            mmio_rd(DMEM_BASE | 32'h0, sc);
            check(sc[`DATA_WIDTH-1:0] === local_mem[0], "DMEM[0] matches hex");

            mmio_rd(IMEM_BASE | HALT_WORD[`MMIO_ADDR_WIDTH-1:0], sc);
            check(sc[`DATA_WIDTH-1:0] === HALT_INSTR, "IMEM halt word = B .");

            mmio_rd(DMEM_BASE | HALT_WORD[`MMIO_ADDR_WIDTH-1:0], sc);
            check(sc[`DATA_WIDTH-1:0] === HALT_INSTR, "DMEM halt word = B .");
        end

        // ── Init CPU registers (SP + LR) ──
        $display("\n  Init CPU regs (SP = 0x%04H, LR = 0x%04H)...",
                 INIT_SP, HALT_BYTE_ADDR);
        init_cpu_regs(INIT_SP, HALT_BYTE_ADDR);
        @(posedge clk); #1;
        init_cpu_regs(INIT_SP, HALT_BYTE_ADDR);

        // ── Start CPU via ILA (cpu_run_level = bit 3) ──
        $display("\n  ████ Starting CPU ████");
        begin : start_cpu
            reg [`MMIO_DATA_WIDTH-1:0] ctrl_word;
            ctrl_word = {`MMIO_DATA_WIDTH{1'b0}};
            ctrl_word[3] = 1'b1;  // cpu_run_level = 1
            ila_write(ILA_CTRL, ctrl_word);
        end

        ila_read(ILA_STATUS, rd);
        check(rd[4] === 1'b1, "cpu_run_level = 1 (CPU running)");

        // ── Poll PC via ILA probes until halt or timeout ──
        $display("  Polling PC (timeout = %0d cycles)...", SORT_TIMEOUT);
        cpu_halted = 0;
        begin : poll_loop
            integer cyc, halt_cnt;
            reg [`MMIO_DATA_WIDTH-1:0] pc;
            cyc      = 0;
            halt_cnt = 0;

            while (cyc < SORT_TIMEOUT) begin
                repeat (POLL_INTERVAL) @(posedge clk);
                cyc = cyc + POLL_INTERVAL;

                read_probe(PROBE_PC, pc);

                // B . makes PC oscillate in [HALT, HALT+12] due to pipeline
                if (pc[`PC_WIDTH-1:0] >= HALT_BYTE_ADDR[`PC_WIDTH-1:0] &&
                    pc[`PC_WIDTH-1:0] <  HALT_BYTE_ADDR[`PC_WIDTH-1:0] + 16 &&
                    cyc > 500) begin
                    halt_cnt = halt_cnt + 1;
                    if (halt_cnt >= 3) begin
                        $display("  CPU halted: PC = 0x%08H (cycle %0d)",
                                 pc, cyc);
                        cpu_halted = 1;
                        cyc = SORT_TIMEOUT;
                    end
                end else begin
                    halt_cnt = 0;
                end

                if (!cpu_halted && cyc < SORT_TIMEOUT && cyc % 20000 == 0)
                    $display("    [%6d] PC = 0x%08H", cyc, pc);
            end

            if (!cpu_halted)
                $display("  *** TIMEOUT: CPU did not halt ***");
        end
        check(cpu_halted, "CPU halted after sort execution");

        // ── Stop CPU via ILA ──
        ila_write(ILA_CTRL, {`MMIO_DATA_WIDTH{1'b0}});  // cpu_run_level = 0
        repeat (10) @(posedge clk);

        ila_read(ILA_STATUS, rd);
        check(rd[4] === 1'b0, "cpu_run_level = 0 (CPU stopped)");

        // Wait for MMIO path to reopen
        begin : wait_rdy
            integer t;
            t = 0;
            while (!user_ready && t < 100) begin
                @(posedge clk); #1;
                t = t + 1;
            end
        end
        check(user_ready === 1'b1, "MMIO available after CPU stop");

        // ── Read DMEM snapshot ──
        $display("\n  Reading %0d DMEM words...", SCAN_RANGE);
        begin : rd_dmem
            integer i;
            for (i = 0; i < SCAN_RANGE; i = i + 1) begin
                mmio_rd(DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0], dmem_snap[i]);
                if ((i + 1) % 200 == 0)
                    $display("    %0d / %0d", i + 1, SCAN_RANGE);
            end
            $display("    Readback complete");
        end

        // ── Search for sorted array ──
        $display("\n  Searching for sorted sequence in %0d DMEM words...",
                 SCAN_RANGE);
        array_found = 0;
        found_at    = 0;
        begin : search
            integer i, j;
            reg match;
            for (i = 0; i < SCAN_RANGE - ARRAY_SIZE + 1; i = i + 1) begin
                if (!array_found) begin
                    match = 1;
                    for (j = 0; j < ARRAY_SIZE; j = j + 1)
                        if (dmem_snap[i + j][`DATA_WIDTH-1:0] !== expected[j])
                            match = 0;
                    if (match) begin
                        array_found = 1;
                        found_at    = i;
                    end
                end
            end
        end

        if (array_found) begin
            $display("  Sorted array at DMEM word %0d (byte 0x%04H):",
                     found_at, found_at << 2);
            begin : show
                integer i;
                for (i = 0; i < ARRAY_SIZE; i = i + 1)
                    $display("    [%0d] = 0x%08H  (%0d)", i,
                             dmem_snap[found_at + i][`DATA_WIDTH-1:0],
                             $signed(dmem_snap[found_at + i][`DATA_WIDTH-1:0]));
            end
        end else begin
            $display("  *** Sorted array NOT found ***");
            $display("  Expected stack frame near words %0d-%0d (SP=0x%04H)",
                     (INIT_SP - 32'h40) >> 2, INIT_SP >> 2, INIT_SP);
            $display("  Non-zero DMEM around stack area and first 64:");
            begin : dump
                integer i, cnt;
                cnt = 0;
                // Show first 20 non-zero in code area
                for (i = 0; i < 130 && cnt < 20; i = i + 1)
                    if (dmem_snap[i] !== {`MMIO_DATA_WIDTH{1'b0}}) begin
                        $display("    [%3d] = 0x%08H  (%0d)", i,
                                 dmem_snap[i][`DATA_WIDTH-1:0],
                                 $signed(dmem_snap[i][`DATA_WIDTH-1:0]));
                        cnt = cnt + 1;
                    end
                // Show everything in the expected stack region
                $display("  --- Stack region (words %0d-%0d) ---",
                         (INIT_SP - 32'h60) >> 2,
                         (INIT_SP >> 2) + 1);
                for (i = (INIT_SP - 32'h60) >> 2;
                     i <= (INIT_SP >> 2) + 1 && i < SCAN_RANGE;
                     i = i + 1)
                    $display("    [%3d] = 0x%08H  (%0d)", i,
                             dmem_snap[i][`DATA_WIDTH-1:0],
                             $signed(dmem_snap[i][`DATA_WIDTH-1:0]));
            end
        end
        check(array_found, "Sorted array found in DMEM");

        // ── Verify ascending signed order ──
        if (array_found) begin
            $display("\n  Verifying ascending order...");
            begin : verify_order
                integer i, errs;
                reg [`DATA_WIDTH-1:0] a, b;
                errs = 0;
                for (i = 0; i < ARRAY_SIZE - 1; i = i + 1) begin
                    a = dmem_snap[found_at + i][`DATA_WIDTH-1:0];
                    b = dmem_snap[found_at + i + 1][`DATA_WIDTH-1:0];
                    if ($signed(a) > $signed(b)) begin
                        $display("    ERROR: [%0d]=%0d > [%0d]=%0d",
                                 i, $signed(a), i + 1, $signed(b));
                        errs = errs + 1;
                    end
                end
                check(errs === 0, "All elements in ascending signed order");
            end
        end

        // ── Verify exact values ──
        if (array_found) begin
            $display("  Comparing against expected...");
            begin : verify_exact
                integer i, errs;
                reg [`DATA_WIDTH-1:0] v;
                errs = 0;
                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                    v = dmem_snap[found_at + i][`DATA_WIDTH-1:0];
                    if (v !== expected[i]) begin
                        $display("    [FAIL] arr[%0d] = %0d, expected %0d",
                                 i, $signed(v), $signed(expected[i]));
                        errs = errs + 1;
                    end else begin
                        $display("    [PASS] arr[%0d] = %0d", i, $signed(v));
                    end
                end
                check(errs === 0, "All elements match expected values");
            end
        end

        // ── CPU register dump (post-sort) ──
        $display("\n  Post-sort register dump:");
        begin : regdump
            integer i;
            reg [`MMIO_DATA_WIDTH-1:0] rv;
            for (i = 0; i < (1 << `REG_ADDR_WIDTH); i = i + 1) begin
                read_probe(PROBE_REG_BASE | i[4:0], rv);
                $display("    R%-2d = 0x%08H  (%0d)", i,
                         rv[`DATA_WIDTH-1:0],
                         $signed(rv[`DATA_WIDTH-1:0]));
            end
        end

        // ── Summary ──
        $display("\n########################################");
        $display("  PASS: %0d   FAIL: %0d", pass_count, fail_count);
        if (fail_count == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else
            $display("  >>> %0d TEST(S) FAILED <<<", fail_count);
        $display("########################################\n");

        if (fail_count > 0) $stop;
        $finish;
    end

    // =========================================================
    // Watchdog
    // =========================================================
    initial begin
        #50_000_000;
        $display("\n[TIMEOUT] 50ms simulation limit exceeded.");
        $display("  PASS: %0d  FAIL: %0d", pass_count, fail_count);
        $finish;
    end

    // =========================================================
    // Waveform Dump
    // =========================================================
    initial begin
        $dumpfile("top_sort_tb.vcd");
        $dumpvars(0, top_sort_tb);
    end

endmodule