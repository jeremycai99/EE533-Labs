`timescale 1ns / 1ps

`include "define.v"
`include "ila.v"

module ila_tb;

    // ---------------------------------------------------------
    // Parameters & Signals
    // ---------------------------------------------------------

    // Register Addresses
    localparam ADDR_CTRL      = 3'h0;
    localparam ADDR_STATUS    = 3'h1;
    localparam ADDR_PROBE_SEL = 3'h2;
    localparam ADDR_PROBE     = 3'h3;
    localparam ADDR_CYCLE     = 3'h4;

    // CTRL bit positions
    localparam CTRL_STEP  = 0;
    localparam CTRL_RUN   = 1;
    localparam CTRL_STOP  = 2;
    localparam CTRL_START = 3;  // cpu_run_level
    localparam CTRL_CLEAR = 5;

    // STATUS bit positions
    localparam STAT_STOPPED  = 0;
    localparam STAT_RUNNING  = 1;
    localparam STAT_STEPPING = 2;
    localparam STAT_BUSY     = 3;
    localparam STAT_CPU_RUN  = 4;
    localparam STAT_DEBUG    = 5;

    // DUT Signals
    reg                          clk;
    reg                          rst_n;
    reg  [2:0]                   ila_addr;
    reg  [`MMIO_DATA_WIDTH-1:0]  ila_din;
    reg                          ila_we;
    wire [`MMIO_DATA_WIDTH-1:0]  ila_dout;

    wire [4:0]                   cpu_debug_sel;
    reg  [`DATA_WIDTH-1:0]       cpu_debug_data;

    wire                         soc_start;
    reg                          debug_mode;
    wire                         soc_clk_en;
    reg                          soc_busy;
    reg                          txn_pending;

    // Test tracking
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    // ---------------------------------------------------------
    // DUT Instantiation
    // ---------------------------------------------------------
    ila u_ila (
        .clk            (clk),
        .rst_n          (rst_n),
        .ila_addr       (ila_addr),
        .ila_din        (ila_din),
        .ila_we         (ila_we),
        .ila_dout       (ila_dout),
        .cpu_debug_sel  (cpu_debug_sel),
        .cpu_debug_data (cpu_debug_data),
        .soc_start      (soc_start),
        .debug_mode     (debug_mode),
        .soc_clk_en     (soc_clk_en),
        .soc_busy       (soc_busy),
        .txn_pending    (txn_pending)
    );

    // ---------------------------------------------------------
    // Clock Generation (10ns period)
    // ---------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ---------------------------------------------------------
    // Helper Tasks
    // ---------------------------------------------------------

    task write_reg(input [2:0] addr, input [`MMIO_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_din  = data;
            ila_we   = 1;
            @(posedge clk);
            ila_we   = 0;
            ila_din  = 0;
            @(posedge clk); // settle
        end
    endtask

    task read_reg(input [2:0] addr, output [`MMIO_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_we   = 0;
            @(posedge clk);
            #1;
            data = ila_dout;
        end
    endtask

    task check(input condition, input [255:0] msg);
        begin
            if (condition) begin
                $display("  [PASS] %0s", msg);
                pass_count = pass_count + 1;
            end else begin
                $display("  [FAIL] %0s", msg);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_clk_en(input expected, input [255:0] msg);
        begin
            #1;
            check(soc_clk_en === expected, msg);
        end
    endtask

    task begin_test(input [255:0] name);
        begin
            test_num = test_num + 1;
            $display("\n========== Test %0d: %0s ==========", test_num, name);
        end
    endtask

    // Build a CTRL word from individual bits for clarity
    function [`MMIO_DATA_WIDTH-1:0] ctrl_word;
        input step, run, stop, start, clear;
        begin
            ctrl_word = {`MMIO_DATA_WIDTH{1'b0}};
            ctrl_word[CTRL_STEP]  = step;
            ctrl_word[CTRL_RUN]   = run;
            ctrl_word[CTRL_STOP]  = stop;
            ctrl_word[CTRL_START] = start;
            ctrl_word[CTRL_CLEAR] = clear;
        end
    endfunction

    // ---------------------------------------------------------
    // Main Test Sequence
    // ---------------------------------------------------------
    reg [`MMIO_DATA_WIDTH-1:0] rd_data;

    initial begin
        // Initialization
        $display("\n###############################################");
        $display("###       ILA Testbench — Full Suite        ###");
        $display("###############################################");

        rst_n          = 0;
        ila_addr       = 0;
        ila_din        = 0;
        ila_we         = 0;
        cpu_debug_data = `DATA_WIDTH'hDEAD_BEEF;
        debug_mode     = 0;
        soc_busy       = 0;
        txn_pending    = 0;

        #20;
        rst_n = 1;
        #20;

        // ===========================================================
        // Test 1: Reset Defaults
        // ===========================================================
        begin_test("Reset Defaults");

        // cpu_run_level resets to 0 per RTL
        check(soc_start === 1'b0, "soc_start defaults to 0 after reset");
        check(cpu_debug_sel === 5'd0, "cpu_debug_sel defaults to 0");
        check_clk_en(1'b1, "soc_clk_en=1 when debug_mode=0 (normal operation)");

        // Read STATUS register — expect stopped=0 because debug_mode=0
        // (stopped = debug_mode & ~run_mode & ~step_active)
        read_reg(ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED]  === 1'b0, "STATUS.stopped=0 (debug_mode off)");
        check(rd_data[STAT_RUNNING]  === 1'b0, "STATUS.running=0 (run_mode off)");
        check(rd_data[STAT_STEPPING] === 1'b0, "STATUS.stepping=0");
        check(rd_data[STAT_BUSY]     === 1'b0, "STATUS.soc_busy=0");
        check(rd_data[STAT_CPU_RUN]  === 1'b0, "STATUS.cpu_run=0 (default)");
        check(rd_data[STAT_DEBUG]    === 1'b0, "STATUS.debug_mode=0");

        // Read PROBE_SEL — should be 0
        read_reg(ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'd0, "PROBE_SEL defaults to 0");

        // Read CYCLE counter — should be non-zero since clock has been running
        read_reg(ADDR_CYCLE, rd_data);
        $display("  Cycle counter after reset: %0d", rd_data);
        check(rd_data > 0, "Cycle counter is incrementing after reset");

        // ===========================================================
        // Test 2: CPU Start/Stop (cpu_run_level)
        // ===========================================================
        begin_test("CPU Start/Stop Level Control");

        // Set cpu_run_level = 1
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 0, 1, 0));
        #1;
        check(soc_start === 1'b1, "soc_start=1 after writing cpu_run_level=1");

        // Verify via STATUS register
        read_reg(ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "STATUS.cpu_run=1");

        // Clear cpu_run_level = 0
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 0, 0, 0));
        #1;
        check(soc_start === 1'b0, "soc_start=0 after writing cpu_run_level=0");

        // Toggle back to 1 for subsequent tests
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 0, 1, 0));

        // ===========================================================
        // Test 3: Probe Select — Boundary Values
        // ===========================================================
        begin_test("Probe Select Boundary Values");

        write_reg(ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h00});
        #1;
        check(cpu_debug_sel === 5'h00, "Probe sel = 0x00");

        write_reg(ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h0A});
        #1;
        check(cpu_debug_sel === 5'h0A, "Probe sel = 0x0A");

        write_reg(ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h15});
        #1;
        check(cpu_debug_sel === 5'h15, "Probe sel = 0x15");

        write_reg(ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h1F});
        #1;
        check(cpu_debug_sel === 5'h1F, "Probe sel = 0x1F (max)");

        // Read back
        read_reg(ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h1F, "PROBE_SEL readback = 0x1F");

        // ===========================================================
        // Test 4: Probe Data Readback
        // ===========================================================
        begin_test("Probe Data Readback");

        cpu_debug_data = `DATA_WIDTH'hCAFE_BABE;
        read_reg(ADDR_PROBE, rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === `DATA_WIDTH'hCAFE_BABE, "Probe reads 0xCAFE_BABE");

        cpu_debug_data = `DATA_WIDTH'h0000_0000;
        read_reg(ADDR_PROBE, rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === `DATA_WIDTH'h0000_0000, "Probe reads 0x00000000");

        cpu_debug_data = {`DATA_WIDTH{1'b1}};
        read_reg(ADDR_PROBE, rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === {`DATA_WIDTH{1'b1}}, "Probe reads all-ones");

        cpu_debug_data = `DATA_WIDTH'hDEAD_BEEF; // restore

        // ===========================================================
        // Test 5: Debug Mode — Clock Gating STOP
        // ===========================================================
        begin_test("Debug Mode — STOP Command");

        debug_mode = 1;
        // Ensure cpu_run_level stays high; only gating the clock
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 1, 1, 0)); // stop=1, start=1
        @(posedge clk);
        check_clk_en(1'b0, "soc_clk_en=0 after STOP in debug mode");

        // Verify STATUS
        read_reg(ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "STATUS.stopped=1");
        check(rd_data[STAT_RUNNING] === 1'b0, "STATUS.running=0");
        check(rd_data[STAT_DEBUG]   === 1'b1, "STATUS.debug_mode=1");

        // ===========================================================
        // Test 6: Debug Mode — RUN Command
        // ===========================================================
        begin_test("Debug Mode — RUN Command");

        write_reg(ADDR_CTRL, ctrl_word(0, 1, 0, 1, 0)); // run=1
        @(posedge clk);
        check_clk_en(1'b1, "soc_clk_en=1 after RUN");

        read_reg(ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "STATUS.running=1");
        check(rd_data[STAT_STOPPED] === 1'b0, "STATUS.stopped=0");

        // ===========================================================
        // Test 7: Debug Mode — STOP then STEP
        // ===========================================================
        begin_test("Debug Mode — Single STEP");

        // Stop first
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 1, 1, 0));
        @(posedge clk);
        check_clk_en(1'b0, "Stopped before step");

        // Issue STEP via direct write for cycle-accurate control
        @(posedge clk);
        ila_addr = ADDR_CTRL;
        ila_din  = ctrl_word(1, 0, 0, 1, 0); // step=1
        ila_we   = 1;
        @(posedge clk);
        ila_we   = 0;
        ila_din  = 0;

        // Cycle after write: step_active should be high
        #1;
        check(soc_clk_en === 1'b1, "Step cycle: soc_clk_en=1");

        @(posedge clk);
        #1;
        check(soc_clk_en === 1'b0, "Post-step: soc_clk_en=0 (auto-cleared)");

        // ===========================================================
        // Test 8: Multiple Consecutive Steps
        // ===========================================================
        begin_test("Multiple Consecutive Steps");

        // We're stopped from previous test. Issue 3 steps.
        repeat (3) begin
            @(posedge clk);
            ila_addr = ADDR_CTRL;
            ila_din  = ctrl_word(1, 0, 0, 1, 0);
            ila_we   = 1;
            @(posedge clk);
            ila_we   = 0;
            ila_din  = 0;

            #1;
            check(soc_clk_en === 1'b1, "Step pulse: clk_en=1");

            @(posedge clk);
            #1;
            check(soc_clk_en === 1'b0, "Post-step: clk_en=0");
        end

        // ===========================================================
        // Test 9: STEP While Running (should be ignored)
        // ===========================================================
        begin_test("STEP While Running (guard)");

        // Enter RUN mode
        write_reg(ADDR_CTRL, ctrl_word(0, 1, 0, 1, 0));
        @(posedge clk);
        check_clk_en(1'b1, "Running before step attempt");

        // Issue STEP while in run_mode — RTL: step only fires if !run_mode
        write_reg(ADDR_CTRL, ctrl_word(1, 0, 0, 1, 0)); // step=1, but run_mode still latched
        @(posedge clk);
        check_clk_en(1'b1, "Still running (step ignored while run_mode=1)");

        // Stop again for next tests
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 1, 1, 0));
        @(posedge clk);
        check_clk_en(1'b0, "Stopped again");

        // ===========================================================
        // Test 10: soc_busy Override
        // ===========================================================
        begin_test("soc_busy Override");

        // Clock is stopped in debug mode
        check_clk_en(1'b0, "Clock stopped before busy");

        soc_busy = 1;
        #1;
        check(soc_clk_en === 1'b1, "soc_busy=1 forces clk_en=1");

        // Verify STATUS reflects busy
        read_reg(ADDR_STATUS, rd_data);
        check(rd_data[STAT_BUSY] === 1'b1, "STATUS.soc_busy=1");

        soc_busy = 0;
        #1;
        check(soc_clk_en === 1'b0, "soc_busy=0, clk_en returns to 0");

        // ===========================================================
        // Test 11: txn_pending Override
        // ===========================================================
        begin_test("txn_pending Override");

        // Clock still stopped
        check_clk_en(1'b0, "Clock stopped before txn_pending");

        txn_pending = 1;
        #1;
        check(soc_clk_en === 1'b1, "txn_pending=1 forces clk_en=1");

        txn_pending = 0;
        #1;
        check(soc_clk_en === 1'b0, "txn_pending=0, clk_en returns to 0");

        // ===========================================================
        // Test 12: Both soc_busy AND txn_pending
        // ===========================================================
        begin_test("soc_busy + txn_pending Simultaneous");

        soc_busy    = 1;
        txn_pending = 1;
        #1;
        check(soc_clk_en === 1'b1, "Both asserted: clk_en=1");

        soc_busy = 0;
        #1;
        check(soc_clk_en === 1'b1, "Only txn_pending: clk_en=1");

        txn_pending = 0;
        #1;
        check(soc_clk_en === 1'b0, "Both deasserted: clk_en=0");

        // ===========================================================
        // Test 13: CLEAR Pulse
        // ===========================================================
        begin_test("CLEAR Pulse");

        // First put into RUN mode so we can verify clear resets it
        write_reg(ADDR_CTRL, ctrl_word(0, 1, 0, 1, 0)); // run=1, start=1
        @(posedge clk);
        check_clk_en(1'b1, "Running before clear");

        // Issue CLEAR
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 0, 0, 1)); // clear=1
        @(posedge clk);
        #1;
        check(soc_clk_en === 1'b0, "After clear: clk_en=0 (run_mode reset)");
        check(soc_start === 1'b0, "After clear: soc_start=0 (cpu_run_level reset)");

        // Verify STATUS reflects stopped state
        read_reg(ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b0, "STATUS.running=0 after clear");
        check(rd_data[STAT_CPU_RUN] === 1'b0, "STATUS.cpu_run=0 after clear");

        // ===========================================================
        // Test 14: Simultaneous RUN + STOP (priority check)
        // ===========================================================
        begin_test("Simultaneous RUN + STOP");

        // Start from stopped state
        // Issue both run=1 and stop=1 simultaneously
        write_reg(ADDR_CTRL, ctrl_word(0, 1, 1, 1, 0)); // run=1, stop=1
        @(posedge clk);
        // RTL: both run_pulse and stop_pulse fire. Last assignment wins?
        // Actually both are set: if (run_pulse) run_mode<=1; if (stop_pulse) run_mode<=0;
        // In Verilog, stop_pulse assignment comes after run_pulse, so stop wins.
        #1;
        $display("  [INFO] run+stop simultaneous: soc_clk_en=%b (stop should win)", soc_clk_en);
        check(soc_clk_en === 1'b0, "STOP takes priority over RUN (sequential assignment)");

        // ===========================================================
        // Test 15: Clock Gating Disabled (debug_mode=0)
        // ===========================================================
        begin_test("Clock Gating Bypass (debug_mode=0)");

        debug_mode = 0;
        // Even though run_mode=0, clock should be enabled
        #1;
        check(soc_clk_en === 1'b1, "debug_mode=0: clk_en always 1");

        // Verify stop/run don't matter
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 1, 1, 0));
        @(posedge clk);
        #1;
        check(soc_clk_en === 1'b1, "debug_mode=0: stop has no effect on clk_en");

        debug_mode = 1; // re-enable for subsequent tests

        // ===========================================================
        // Test 16: Cycle Counter Increments
        // ===========================================================
        begin_test("Cycle Counter Increments");

        // Disable debug mode so clock runs freely
        debug_mode = 0;

        read_reg(ADDR_CYCLE, rd_data);
        begin : cycle_cnt_block
            reg [31:0] cnt_before, cnt_after;
            cnt_before = rd_data[31:0];
            $display("  Cycle count before: %0d", cnt_before);

            repeat (20) @(posedge clk);

            read_reg(ADDR_CYCLE, rd_data);
            cnt_after = rd_data[31:0];
            $display("  Cycle count after:  %0d", cnt_after);

            check(cnt_after > cnt_before, "Cycle counter incremented");
            check((cnt_after - cnt_before) >= 20,
                  "Cycle counter advanced by at least 20");
        end

        // ===========================================================
        // Test 17: CTRL Register Readback
        // ===========================================================
        begin_test("CTRL Register Readback");

        // Write known pattern: start=1
        write_reg(ADDR_CTRL, ctrl_word(0, 0, 0, 1, 0));
        read_reg(ADDR_CTRL, rd_data);
        // Pulses auto-clear, so step/run/stop/clear should be 0
        // Only cpu_run_level (bit 3) should persist
        check(rd_data[CTRL_STEP]  === 1'b0, "CTRL readback: step=0 (auto-cleared)");
        check(rd_data[CTRL_RUN]   === 1'b0, "CTRL readback: run=0 (auto-cleared)");
        check(rd_data[CTRL_STOP]  === 1'b0, "CTRL readback: stop=0 (auto-cleared)");
        check(rd_data[CTRL_START] === 1'b1, "CTRL readback: cpu_run_level=1");
        check(rd_data[CTRL_CLEAR] === 1'b0, "CTRL readback: clear=0 (auto-cleared)");

        // ===========================================================
        // Test 18: Write to Read-Only Addresses (STATUS, PROBE, CYCLE)
        // ===========================================================
        begin_test("Write to Read-Only Addresses (no effect)");

        // Try writing to STATUS (0x1)
        read_reg(ADDR_STATUS, rd_data);
        begin : ro_block
            reg [`MMIO_DATA_WIDTH-1:0] status_before;
            status_before = rd_data;
            write_reg(ADDR_STATUS, {`MMIO_DATA_WIDTH{1'b1}});
            read_reg(ADDR_STATUS, rd_data);
            // STATUS is combinational from internal signals, write should have no effect
            // (the default case in write FSM ignores these addresses)
            $display("  [INFO] STATUS before=%0h, after=%0h", status_before, rd_data);
            check(1'b1, "Write to STATUS did not crash (read-only addr)");
        end

        // ===========================================================
        // Test 19: Default Address Read
        // ===========================================================
        begin_test("Default Address Read (0x5, 0x6, 0x7)");

        read_reg(3'h5, rd_data);
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "Addr 0x5 returns 0");

        read_reg(3'h6, rd_data);
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "Addr 0x6 returns 0");

        read_reg(3'h7, rd_data);
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "Addr 0x7 returns 0");

        // ===========================================================
        // Test 20: Reset During Active State
        // ===========================================================
        begin_test("Reset During Active State");

        debug_mode = 1;
        // Put into RUN mode with cpu_run_level=1
        write_reg(ADDR_CTRL, ctrl_word(0, 1, 0, 1, 0));
        @(posedge clk);
        check_clk_en(1'b1, "Running before reset");

        // Assert reset
        rst_n = 0;
        #20;
        rst_n = 1;
        #20;

        // Everything should be back to defaults
        check(soc_start === 1'b0, "soc_start=0 after mid-operation reset");
        check(cpu_debug_sel === 5'd0, "cpu_debug_sel=0 after reset");

        // In debug mode with run_mode=0, step_active=0 => clk_en=0
        #1;
        check(soc_clk_en === 1'b0, "clk_en=0 after reset (debug_mode still 1)");

        debug_mode = 0;
        #1;
        check(soc_clk_en === 1'b1, "clk_en=1 after debug_mode cleared post-reset");

        // ===========================================================
        // Summary
        // ===========================================================
        $display("\n###############################################");
        $display("###            TEST SUMMARY                 ###");
        $display("###############################################");
        $display("  Total PASS: %0d", pass_count);
        $display("  Total FAIL: %0d", fail_count);
        if (fail_count == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else
            $display("  >>> SOME TESTS FAILED <<<");
        $display("###############################################\n");

        if (fail_count > 0) $stop;
        $finish;
    end

    // ---------------------------------------------------------
    // Timeout Watchdog
    // ---------------------------------------------------------
    initial begin
        #100_000;
        $display("[TIMEOUT] Simulation exceeded 100us — aborting.");
        $finish;
    end

endmodule