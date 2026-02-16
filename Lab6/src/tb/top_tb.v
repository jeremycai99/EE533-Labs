/* file: top_tb.v
 * Description: Enhanced testbench for top module (SoC + ILA + SoC Driver).
 * Dependencies: top.v (which includes ila.v, soc.v, soc_driver.v, cpu.v, etc.)
 * Date: Feb. 16, 2026
 */

`timescale 1ns / 1ps

`include "define.v"
`include "top.v"

module top_tb;

    // =========================================================
    // Parameters
    // =========================================================

    localparam CLK_PERIOD = 10; // 10ns → 100 MHz

    // ILA register addresses
    localparam ILA_ADDR_CTRL      = 3'h0;
    localparam ILA_ADDR_STATUS    = 3'h1;
    localparam ILA_ADDR_PROBE_SEL = 3'h2;
    localparam ILA_ADDR_PROBE     = 3'h3;
    localparam ILA_ADDR_CYCLE     = 3'h4;

    // ILA CTRL bit indices
    localparam CTRL_STEP  = 0;
    localparam CTRL_RUN   = 1;
    localparam CTRL_STOP  = 2;
    localparam CTRL_START = 3; // cpu_run_level
    localparam CTRL_CLEAR = 5;

    // ILA STATUS bit indices
    localparam STAT_STOPPED  = 0;
    localparam STAT_RUNNING  = 1;
    localparam STAT_STEPPING = 2;
    localparam STAT_BUSY     = 3;
    localparam STAT_CPU_RUN  = 4;
    localparam STAT_DEBUG    = 5;

    // SoC memory regions (bits [31:30] of address)
    localparam [31:0] REGION_IMEM_BASE = 32'h0000_0000; // [31:30] = 2'b00
    localparam [31:0] REGION_CTRL_BASE = 32'h4000_0000; // [31:30] = 2'b01
    localparam [31:0] REGION_DMEM_BASE = 32'h8000_0000; // [31:30] = 2'b10

    // Driver conn_status bit indices
    localparam CONN_LINK_ACTIVE    = 0;
    localparam CONN_REQ_TIMEOUT    = 1;
    localparam CONN_RESP_TIMEOUT   = 2;
    localparam CONN_PROTOCOL_ERR   = 3;
    localparam CONN_FIFO_NOT_EMPTY = 4;
    localparam CONN_BUSY           = 5;
    localparam CONN_TXN_TIMEDOUT   = 6;

    // ILA probe selectors (from cpu.v debug mux)
    localparam [4:0] PROBE_PC          = 5'h00; // PC (fetch stage)
    localparam [4:0] PROBE_INSTR_ID    = 5'h01; // Instruction (decode)
    localparam [4:0] PROBE_R0_ID       = 5'h02; // Reg read data A (decode)
    localparam [4:0] PROBE_R1_ID       = 5'h03; // Reg read data B (decode)
    localparam [4:0] PROBE_R0_EX       = 5'h04; // EX stage result
    localparam [4:0] PROBE_R1_EX       = 5'h05; // EX stage write data
    localparam [4:0] PROBE_WB_DATA     = 5'h06; // Writeback data
    localparam [4:0] PROBE_CTRL_VEC    = 5'h07; // Control signals
    localparam [4:0] PROBE_RDADDR_ID   = 5'h08; // Dest reg addr (decode)
    localparam [4:0] PROBE_RDADDR_WB   = 5'h09; // Dest reg addr (WB)
    localparam [4:0] PROBE_REG_BASE    = 5'h10; // Register file mode (bit4=1)

    // Register name aliases for instruction building
    localparam [`REG_ADDR_WIDTH-1:0] R0 = 0;
    localparam [`REG_ADDR_WIDTH-1:0] R1 = 1;
    localparam [`REG_ADDR_WIDTH-1:0] R2 = 2;
    localparam [`REG_ADDR_WIDTH-1:0] R3 = 3;
    localparam [`REG_ADDR_WIDTH-1:0] R4 = 4;

    localparam [`INSTR_WIDTH-1:0] NOP = {`INSTR_WIDTH{1'b0}};

    // =========================================================
    // DUT Signals
    // =========================================================

    reg                          clk;
    reg                          rst_n;
    reg                          debug_mode;

    // User → Driver interface
    reg                          user_valid;
    reg                          user_cmd;
    reg  [`MMIO_ADDR_WIDTH-1:0]  user_addr;
    reg  [`MMIO_DATA_WIDTH-1:0]  user_wdata;
    wire                         user_ready;
    wire [`MMIO_DATA_WIDTH-1:0]  user_rdata;

    // Driver status / quality
    wire [`MMIO_ADDR_WIDTH-1:0]  status;
    wire [7:0]                   conn_status;
    wire [`MMIO_DATA_WIDTH-1:0]  txn_quality;
    wire [`MMIO_DATA_WIDTH-1:0]  txn_counters;
    reg                          clear_stats;

    // ILA direct access
    reg  [2:0]                   ila_addr;
    reg  [`MMIO_DATA_WIDTH-1:0]  ila_din;
    reg                          ila_we;
    wire [`MMIO_DATA_WIDTH-1:0]  ila_dout;

    // Test accounting
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    // =========================================================
    // DUT Instantiation
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
    // Clock Generation
    // =========================================================

    initial clk = 0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    // =========================================================
    // Instruction Builder (matches cpu.v decode exactly)
    // =========================================================
    function [`INSTR_WIDTH-1:0] build_instr;
        input                       mw;
        input                       rw;
        input [`REG_ADDR_WIDTH-1:0] rd;
        input [`REG_ADDR_WIDTH-1:0] r0;
        input [`REG_ADDR_WIDTH-1:0] r1;
        begin
            build_instr = {mw, rw, r0, r1, rd,
                           {(`INSTR_WIDTH - 2 - 3*`REG_ADDR_WIDTH){1'b0}}};
        end
    endfunction

    // ILA CTRL word builder
    // IMPORTANT: cpu_run_level is a LEVEL; every CTRL write overwrites it.
    //            Always include the desired cpu_run_level state.
    function [`MMIO_DATA_WIDTH-1:0] ila_ctrl;
        input step, run, stop, start, clear;
        begin
            ila_ctrl = {`MMIO_DATA_WIDTH{1'b0}};
            ila_ctrl[CTRL_STEP]  = step;
            ila_ctrl[CTRL_RUN]   = run;
            ila_ctrl[CTRL_STOP]  = stop;
            ila_ctrl[CTRL_START] = start;
            ila_ctrl[CTRL_CLEAR] = clear;
        end
    endfunction

    // =========================================================
    // Helper Tasks
    // =========================================================

    // --- Assertion helper ---
    task check(input cond, input [511:0] msg);
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

    task begin_test(input [511:0] name);
        begin
            test_num = test_num + 1;
            $display("\n================================================================");
            $display("  Test %0d: %0s", test_num, name);
            $display("================================================================");
        end
    endtask

    // --- ILA register write (direct, synchronous) ---
    task ila_write(input [2:0] addr, input [`MMIO_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_din  = data;
            ila_we   = 1;
            @(posedge clk);
            ila_we   = 0;
            ila_din  = {`MMIO_DATA_WIDTH{1'b0}};
            @(posedge clk); // settle
        end
    endtask

    // --- ILA register read ---
    task ila_read(input [2:0] addr, output [`MMIO_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_we   = 0;
            @(posedge clk);
            #1;
            data = ila_dout;
        end
    endtask

    // --- ILA probe read (sets PROBE_SEL then reads PROBE) ---
    task read_probe(input [4:0] sel, output [`MMIO_DATA_WIDTH-1:0] data);
        begin
            ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, sel});
            @(posedge clk); // extra settle for probe data propagation
            ila_read(ILA_ADDR_PROBE, data);
        end
    endtask

    // --- Submit one MMIO transaction through the driver ---
    // cmd: 0 = read, 1 = write
    task submit_txn(
        input                         cmd,
        input [`MMIO_ADDR_WIDTH-1:0]  addr,
        input [`MMIO_DATA_WIDTH-1:0]  wdata
    );
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

    // --- Wait for the driver to return to idle (transaction complete) ---
    task wait_txn_done(output reg success);
        integer timeout_cnt;
        begin
            success = 0;
            timeout_cnt = 0;
            repeat (2) @(posedge clk);
            while (timeout_cnt < 2000) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    success = 1;
                    timeout_cnt = 2000;
                end
                timeout_cnt = timeout_cnt + 1;
            end
            @(posedge clk);
        end
    endtask

    // --- Wait with a configurable sim-timeout (for slow scenarios) ---
    task wait_txn_done_long(input integer max_cycles, output reg success);
        integer timeout_cnt;
        begin
            success = 0;
            timeout_cnt = 0;
            repeat (2) @(posedge clk);
            while (timeout_cnt < max_cycles) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    success = 1;
                    timeout_cnt = max_cycles;
                end
                timeout_cnt = timeout_cnt + 1;
            end
            @(posedge clk);
        end
    endtask

    // --- Combined: submit + wait + check status ---
    task do_write(
        input [`MMIO_ADDR_WIDTH-1:0] addr,
        input [`MMIO_DATA_WIDTH-1:0] data,
        input [511:0]                msg
    );
        reg success;
        begin
            submit_txn(1'b1, addr, data);
            wait_txn_done(success);
            check(success, msg);
        end
    endtask

    task do_read(
        input  [`MMIO_ADDR_WIDTH-1:0]  addr,
        output [`MMIO_DATA_WIDTH-1:0]  rdata,
        input  [511:0]                 msg
    );
        reg success;
        begin
            submit_txn(1'b0, addr, {`MMIO_DATA_WIDTH{1'b0}});
            wait_txn_done(success);
            check(success, msg);
            rdata = user_rdata;
        end
    endtask

    // =========================================================
    // Main Test Sequence
    // =========================================================
    reg [`MMIO_DATA_WIDTH-1:0] rd_data;
    reg [`MMIO_DATA_WIDTH-1:0] rd_data2;
    reg [`MMIO_DATA_WIDTH-1:0] rd_data3;
    reg txn_ok;

    initial begin
        $display("\n##################################");
        $display("###     TOP MODULE TESTBENCH     ###");
        $display("##################################\n");

        // ── Initialization ──
        rst_n       = 0;
        debug_mode  = 0;
        user_valid  = 0;
        user_cmd    = 0;
        user_addr   = 0;
        user_wdata  = 0;
        clear_stats = 0;
        ila_addr    = 0;
        ila_din     = 0;
        ila_we      = 0;

        #(CLK_PERIOD * 4);
        rst_n = 1;
        #(CLK_PERIOD * 4);

        // =============================================================
        // Test 1: Reset Defaults
        // =============================================================
        begin_test("Reset Defaults");

        #1;
        check(ila_dout !== {`MMIO_DATA_WIDTH{1'bx}}, "ila_dout is not X after reset");
        check(user_ready === 1'b1, "user_ready=1 (FIFO empty, driver idle)");
        check(conn_status[CONN_BUSY] === 1'b0, "Driver is idle");
        check(conn_status[CONN_FIFO_NOT_EMPTY] === 1'b0, "FIFO is empty");
        check(conn_status[CONN_LINK_ACTIVE] === 1'b0, "link_active=0 (no txns yet)");
        check(conn_status[CONN_REQ_TIMEOUT] === 1'b0, "req_timeout flag clear");
        check(conn_status[CONN_RESP_TIMEOUT] === 1'b0, "resp_timeout flag clear");
        check(conn_status[CONN_PROTOCOL_ERR] === 1'b0, "protocol_error flag clear");
        check(conn_status[7] === 1'b1, "conn_status[7] hardwired to 1");
        check(status === 32'h0, "status is 0 after reset");
        check(txn_counters[15:0] === 16'd0, "total_txn_count=0 after reset");
        check(txn_counters[31:16] === 16'd0, "read_txn_count=0 after reset");
        check(txn_counters[47:32] === 16'd0, "write_txn_count=0 after reset");

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN]  === 1'b0, "ILA STATUS: cpu_run_level=0 after reset");
        check(rd_data[STAT_DEBUG]    === 1'b0, "ILA STATUS: debug_mode=0");
        check(rd_data[STAT_RUNNING]  === 1'b0, "ILA STATUS: run_mode=0");
        check(rd_data[STAT_STEPPING] === 1'b0, "ILA STATUS: step_active=0");
        check(rd_data[STAT_BUSY]     === 1'b0, "ILA STATUS: soc_busy=0");
        check(rd_data[STAT_STOPPED]  === 1'b0, "ILA STATUS: stopped=0 (not in debug mode)");

        ila_read(ILA_ADDR_CYCLE, rd_data);
        check(rd_data > 0, "Cycle counter has started incrementing");

        // =============================================================
        // Test 2: ILA Direct Register Access
        // =============================================================
        begin_test("ILA Direct Register Access");

        // Write & read PROBE_SEL
        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h1A});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h1A, "PROBE_SEL readback = 0x1A");

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h00});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h00, "PROBE_SEL readback = 0x00");

        // Boundary: max selector value
        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h1F});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h1F, "PROBE_SEL readback = 0x1F (max)");

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h00});

        // Cycle counter should be incrementing (debug_mode=0 → clock free-running)
        ila_read(ILA_ADDR_CYCLE, rd_data);
        repeat (10) @(posedge clk);
        ila_read(ILA_ADDR_CYCLE, rd_data2);
        check(rd_data2 > rd_data, "Cycle counter is incrementing");
        $display("  [INFO] Cycle delta = %0d over ~10 cycles", rd_data2 - rd_data);

        // CTRL readback: write cpu_run_level=1, verify via both CTRL and STATUS
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        ila_read(ILA_ADDR_CTRL, rd_data);
        check(rd_data[CTRL_START] === 1'b1, "CTRL readback: cpu_run_level=1");
        check(rd_data[CTRL_STEP] === 1'b0, "CTRL readback: step_pulse auto-cleared");
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "STATUS confirms cpu_run_level=1");
        // Clear it back
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b0, "cpu_run_level=0 after clearing");

        // Write to an undefined ILA address (0x5) — should be benign
        ila_write(3'h5, 64'hCAFE_BABE);
        ila_read(3'h5, rd_data);
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "Undefined ILA addr returns 0 (default case)");

        // =============================================================
        // Test 3: ILA Clock Gating — Stop / Run
        // =============================================================
        begin_test("ILA Clock Gating — Stop / Run");

        debug_mode = 1;
        @(posedge clk);

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_DEBUG] === 1'b1, "STATUS reflects debug_mode=1");

        // STOP the clock
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Clock STOPPED in debug mode");
        check(rd_data[STAT_RUNNING] === 1'b0, "run_mode=0");

        // RUN the clock
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Clock RUNNING after RUN command");
        check(rd_data[STAT_STOPPED] === 1'b0, "stopped=0 when running");

        // STOP again
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Re-stopped after second STOP");

        // Return to normal
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 4: ILA Step — Single-Cycle Verification via PC Probe
        // =============================================================
        begin_test("ILA Step — Single-Cycle Verification via PC Probe");

        // Enter debug mode, start CPU, let it run briefly, then stop and step
        debug_mode = 1;
        @(posedge clk);

        // Start CPU + RUN mode (so gated clock runs and CPU comes out of reset)
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 1, 0));
        repeat (10) @(posedge clk); // Let CPU run ~10 gated cycles

        // STOP (keep cpu_run_level=1 so CPU stays out of reset)
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 1, 0));
        repeat (3) @(posedge clk); // settle

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Clock stopped for step test");
        check(rd_data[STAT_CPU_RUN] === 1'b1, "CPU still out of reset (cpu_run_level=1)");

        // Read current PC
        read_probe(PROBE_PC, rd_data);
        $display("  [INFO] PC before step = %0d", rd_data[`PC_WIDTH-1:0]);

        // Issue STEP (keep cpu_run_level=1)
        ila_write(ILA_ADDR_CTRL, ila_ctrl(1, 0, 0, 1, 0));
        repeat (3) @(posedge clk); // wait for step to execute + settle

        // Read PC again
        read_probe(PROBE_PC, rd_data2);
        $display("  [INFO] PC after 1 step = %0d", rd_data2[`PC_WIDTH-1:0]);
        check(rd_data2[`PC_WIDTH-1:0] === rd_data[`PC_WIDTH-1:0] + 1,
              "PC incremented by exactly 1 after step");

        // Issue a second STEP
        ila_write(ILA_ADDR_CTRL, ila_ctrl(1, 0, 0, 1, 0));
        repeat (3) @(posedge clk);

        read_probe(PROBE_PC, rd_data3);
        $display("  [INFO] PC after 2nd step = %0d", rd_data3[`PC_WIDTH-1:0]);
        check(rd_data3[`PC_WIDTH-1:0] === rd_data2[`PC_WIDTH-1:0] + 1,
              "PC incremented by exactly 1 after second step");

        // Stop CPU and leave debug mode
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 5: ILA Step While Running (Should Be Ignored)
        // =============================================================
        begin_test("ILA Step While Running (No-Op)");

        debug_mode = 1;
        @(posedge clk);

        // Set RUN mode
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Confirmed running before step attempt");

        // Issue step while running — step_active should NOT go high
        // (ILA code: `if (step_pulse && !run_mode) step_active <= 1`)
        // Since run_mode=1, step is suppressed.
        @(posedge clk);
        ila_addr = ILA_ADDR_CTRL;
        ila_din  = ila_ctrl(1, 1, 0, 0, 0); // step=1 but also run=1 to keep run_mode=1
        ila_we   = 1;
        @(posedge clk);
        ila_we = 0;
        ila_din = 0;
        @(posedge clk); // settle

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STEPPING] === 1'b0,
              "step_active=0 (step suppressed while running)");
        check(rd_data[STAT_RUNNING] === 1'b1,
              "run_mode still=1 after step attempt");

        // Cleanup
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 6: ILA Rapid Mode Transitions
        // =============================================================
        begin_test("ILA Rapid Mode Transitions");

        debug_mode = 1;
        @(posedge clk);

        // RUN
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Transition 1: RUN");

        // STOP
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Transition 2: STOP");

        // RUN again
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Transition 3: RUN");

        // STOP + immediate STEP
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk);
        ila_write(ILA_ADDR_CTRL, ila_ctrl(1, 0, 0, 0, 0));
        repeat (3) @(posedge clk);

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Transition 4: Stopped after STOP+STEP sequence");
        check(rd_data[STAT_STEPPING] === 1'b0, "step_active auto-cleared");

        // Both RUN and STOP simultaneously — STOP should win because stop_pulse
        // is processed after run_pulse in the same block
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 1, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        $display("  [INFO] After simultaneous RUN+STOP: running=%b, stopped=%b",
                 rd_data[STAT_RUNNING], rd_data[STAT_STOPPED]);
        // The result depends on the always block priority. Document observed behavior.
        // In the ILA code: `if (run_pulse) run_mode <= 1; if (stop_pulse) run_mode <= 0;`
        // Since stop_pulse assignment comes after run_pulse, stop wins (last assignment wins).
        check(rd_data[STAT_RUNNING] === 1'b0, "Simultaneous RUN+STOP: STOP wins (last write)");

        // Cleanup
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 7: MMIO Write to DMEM
        // =============================================================
        begin_test("MMIO Write to DMEM");

        do_write(REGION_DMEM_BASE | 32'h0, {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'hDEAD},
                 "MMIO write to DMEM[0] completed");
        check(status === 32'hAAAA_AAAA, "Status = 0xAAAA_AAAA (integrity pass)");

        do_write(REGION_DMEM_BASE | 32'h4, {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'hBEEF},
                 "MMIO write to DMEM[4] completed");
        check(status === 32'hAAAA_AAAA, "Status = 0xAAAA_AAAA for DMEM[4]");

        // =============================================================
        // Test 8: MMIO Read from DMEM
        // =============================================================
        begin_test("MMIO Read from DMEM");

        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "MMIO read from DMEM[0] completed");
        $display("  [INFO] DMEM[0] read data = 0x%0h", rd_data);
        check(rd_data[15:0] === 16'hDEAD, "DMEM[0] = 0xDEAD");
        check(status === 32'hAAAA_AAAA, "Status = integrity pass for read");

        do_read(REGION_DMEM_BASE | 32'h4, rd_data, "MMIO read from DMEM[4] completed");
        $display("  [INFO] DMEM[4] read data = 0x%0h", rd_data);
        check(rd_data[15:0] === 16'hBEEF, "DMEM[4] = 0xBEEF");

        // =============================================================
        // Test 9: Walking-Ones Data Pattern (DMEM)
        // =============================================================
        begin_test("Walking-Ones Data Pattern (DMEM)");

        begin : walking_ones_block
            integer bit_idx;
            reg [`MMIO_DATA_WIDTH-1:0] pattern;
            reg [`MMIO_DATA_WIDTH-1:0] readback;
            reg all_pass;
            all_pass = 1;

            for (bit_idx = 0; bit_idx < 16; bit_idx = bit_idx + 1) begin
                pattern = {`MMIO_DATA_WIDTH{1'b0}};
                pattern[bit_idx] = 1'b1;

                // Write walking-one to DMEM[8 + bit_idx]
                do_write(REGION_DMEM_BASE | (32'h8 + bit_idx[`MMIO_ADDR_WIDTH-1:0]),
                         pattern, "Walking-1 write");

                // Read back
                do_read(REGION_DMEM_BASE | (32'h8 + bit_idx[`MMIO_ADDR_WIDTH-1:0]),
                        readback, "Walking-1 read");

                if (readback !== pattern) begin
                    $display("  [FAIL] Bit %0d: wrote 0x%0h, read 0x%0h",
                             bit_idx, pattern, readback);
                    all_pass = 0;
                end
            end
            check(all_pass, "All 16 walking-ones patterns verified");
        end

        // Also test all-ones and all-zeros
        do_write(REGION_DMEM_BASE | 32'h30, {`MMIO_DATA_WIDTH{1'b1}}, "Write all-ones");
        do_read(REGION_DMEM_BASE | 32'h30, rd_data, "Read all-ones");
        check(rd_data === {`MMIO_DATA_WIDTH{1'b1}}, "All-ones pattern verified");

        do_write(REGION_DMEM_BASE | 32'h31, {`MMIO_DATA_WIDTH{1'b0}}, "Write all-zeros");
        do_read(REGION_DMEM_BASE | 32'h31, rd_data, "Read all-zeros");
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "All-zeros pattern verified");

        // =============================================================
        // Test 10: MMIO Write to IMEM (Load a Program)
        // =============================================================
        begin_test("MMIO Write to IMEM — Load Program");

        do_write(REGION_IMEM_BASE | 32'h0,
                 {{(`MMIO_DATA_WIDTH-`INSTR_WIDTH){1'b0}}, build_instr(1'b0, 1'b1, R2, R0, R0)},
                 "IMEM[0] = Load d_mem[R0]->R2");

        do_write(REGION_IMEM_BASE | 32'h1,
                 {{(`MMIO_DATA_WIDTH-`INSTR_WIDTH){1'b0}}, build_instr(1'b0, 1'b1, R3, R0, R0)},
                 "IMEM[1] = Load d_mem[R0]->R3");

        do_write(REGION_IMEM_BASE | 32'h2,
                 {{(`MMIO_DATA_WIDTH-`INSTR_WIDTH){1'b0}}, NOP},
                 "IMEM[2] = NOP");

        do_write(REGION_IMEM_BASE | 32'h3,
                 {{(`MMIO_DATA_WIDTH-`INSTR_WIDTH){1'b0}}, NOP},
                 "IMEM[3] = NOP");

        do_write(REGION_IMEM_BASE | 32'h4,
                 {{(`MMIO_DATA_WIDTH-`INSTR_WIDTH){1'b0}}, NOP},
                 "IMEM[4] = NOP");

        do_write(REGION_IMEM_BASE | 32'h5,
                 {{(`MMIO_DATA_WIDTH-`INSTR_WIDTH){1'b0}}, build_instr(1'b1, 1'b0, R0, R2, R3)},
                 "IMEM[5] = Store R3->d_mem[R2]");

        // =============================================================
        // Test 11: MMIO Read from IMEM (Verify Program)
        // =============================================================
        begin_test("MMIO Read from IMEM — Verify Loaded Program");

        do_read(REGION_IMEM_BASE | 32'h0, rd_data, "Read IMEM[0]");
        check(rd_data[`INSTR_WIDTH-1:0] === build_instr(1'b0, 1'b1, R2, R0, R0),
              "IMEM[0] matches Load R2 instruction");

        do_read(REGION_IMEM_BASE | 32'h1, rd_data, "Read IMEM[1]");
        check(rd_data[`INSTR_WIDTH-1:0] === build_instr(1'b0, 1'b1, R3, R0, R0),
              "IMEM[1] matches Load R3 instruction");

        do_read(REGION_IMEM_BASE | 32'h2, rd_data, "Read IMEM[2]");
        check(rd_data[`INSTR_WIDTH-1:0] === NOP, "IMEM[2] = NOP");

        do_read(REGION_IMEM_BASE | 32'h5, rd_data, "Read IMEM[5]");
        check(rd_data[`INSTR_WIDTH-1:0] === build_instr(1'b1, 1'b0, R0, R2, R3),
              "IMEM[5] matches Store instruction");

        // =============================================================
        // Test 12: MMIO Read from CTRL Region
        // =============================================================
        begin_test("MMIO Read from CTRL — system_active Flag");

        do_read(REGION_CTRL_BASE, rd_data, "Read CTRL region");
        $display("  [INFO] CTRL read data = 0x%0h", rd_data);
        check(rd_data[0] === 1'b0, "system_active=0 (CPU not started yet)");

        // =============================================================
        // Test 13: CPU End-to-End Execution
        // =============================================================
        begin_test("CPU End-to-End Execution");

        // Overwrite DMEM[0] with a small, controlled value
        do_write(REGION_DMEM_BASE | 32'h0,
                 {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, `DATA_WIDTH'd5},
                 "Preload DMEM[0] = 5");

        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Verify DMEM[0]");
        check(rd_data[7:0] === 8'd5, "DMEM[0] = 5 confirmed");

        // Set a canary at DMEM[5]
        do_write(REGION_DMEM_BASE | 32'h5,
                 {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, `DATA_WIDTH'hFFFF_FFFF},
                 "Preload DMEM[5] = 0xFFFF_FFFF (canary)");

        $display("  Starting CPU via ILA cpu_run_level...");
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "cpu_run_level=1, CPU starting");

        // Verify system_active via CTRL region
        // (Note: this MMIO may take a few cycles, and system_active may not be
        //  set yet if CPU hasn't started. The req_rdy might block. Use a longer wait.)
        @(posedge clk); @(posedge clk);

        $display("  Waiting for CPU to complete (~600 cycles)...");
        repeat (700) @(posedge clk);

        // Stop the CPU
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b0, "cpu_run_level=0 after stopping CPU");

        // Read DMEM[5] — should be 5 (CPU stored R3=5 at addr R2=5)
        do_read(REGION_DMEM_BASE | 32'h5, rd_data, "Read DMEM[5] after CPU execution");
        $display("  [INFO] DMEM[5] after CPU = 0x%0h", rd_data);
        check(rd_data == `MMIO_DATA_WIDTH'd5,
              "DMEM[5] = 5 (CPU stored R3=5 at address R2=5)");

        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Read DMEM[0] after CPU execution");
        check(rd_data[7:0] === 8'd5, "DMEM[0] still = 5 (not corrupted)");

        // =============================================================
        // Test 14: CPU Register Readback via ILA Probes (Post-Execution)
        // =============================================================
        begin_test("CPU Register Readback via ILA Probes");

        // After execution, CPU register file should contain:
        //   R0 = 0 (never written)
        //   R2 = 5 (loaded from DMEM[0])
        //   R3 = 5 (loaded from DMEM[0])
        //
        // Use probe register mode (bit4=1) to read register file.

        // Read R0
        read_probe(PROBE_REG_BASE | R0, rd_data);
        $display("  [INFO] R0 = 0x%0h", rd_data);
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "R0 = 0 (never written)");

        // Read R2
        read_probe(PROBE_REG_BASE | R2, rd_data);
        $display("  [INFO] R2 = 0x%0h", rd_data);
        check(rd_data[7:0] === 8'd5, "R2 = 5 (loaded from DMEM[0])");

        // Read R3
        read_probe(PROBE_REG_BASE | R3, rd_data);
        $display("  [INFO] R3 = 0x%0h", rd_data);
        check(rd_data[7:0] === 8'd5, "R3 = 5 (loaded from DMEM[0])");

        // Read R1 (should be 0 — not used by program)
        read_probe(PROBE_REG_BASE | R1, rd_data);
        $display("  [INFO] R1 = 0x%0h", rd_data);
        // R1 was not written by the program, but may be non-zero if register
        // file is not reset. Just verify no X values.
        check(rd_data !== {`MMIO_DATA_WIDTH{1'bx}}, "R1 probe is not X");

        // Read system debug probes
        read_probe(PROBE_PC, rd_data);
        $display("  [INFO] PC (after CPU stopped) = %0d", rd_data[`PC_WIDTH-1:0]);
        // PC should be 0 since cpu_run_level=0 → CPU in reset → PC = 0
        check(rd_data[`PC_WIDTH-1:0] === {`PC_WIDTH{1'b0}},
              "PC = 0 (CPU in reset after stopping)");

        // =============================================================
        // Test 15: Debug Mode — MMIO with Clock Stopped (txn_pending)
        // =============================================================
        begin_test("Debug Mode — MMIO Transactions with Clock Gated");

        debug_mode = 1;
        @(posedge clk);

        // STOP the SoC clock
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "SoC clock stopped");

        // MMIO read while debug-stopped — txn_pending should keep clock alive
        $display("  Attempting MMIO read while debug clock is stopped...");
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "MMIO read DMEM[0] while debug-stopped");
        check(rd_data[7:0] === 8'd5, "DMEM[0] = 5 (txn_pending kept clock alive)");

        // MMIO write while debug-stopped
        do_write(REGION_DMEM_BASE | 32'h7,
                 {{(`MMIO_DATA_WIDTH-8){1'b0}}, 8'h42},
                 "MMIO write DMEM[7]=0x42 while debug-stopped");

        do_read(REGION_DMEM_BASE | 32'h7, rd_data, "MMIO read DMEM[7] while debug-stopped");
        check(rd_data[7:0] === 8'h42, "DMEM[7] = 0x42 (write through gated clock)");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 16: txn_pending Detailed Verification
        // =============================================================
        begin_test("txn_pending Detailed — Observing FIFO & Clock Interaction");

        debug_mode = 1;
        @(posedge clk);

        // STOP the clock
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        repeat (3) @(posedge clk);

        // Submit a transaction — observe FIFO_NOT_EMPTY before completion
        @(posedge clk);
        while (!user_ready) @(posedge clk);
        user_cmd   = 1'b0; // read
        user_addr  = REGION_DMEM_BASE | 32'h0;
        user_wdata = {`MMIO_DATA_WIDTH{1'b0}};
        user_valid = 1;
        @(posedge clk);
        user_valid = 0;

        // Immediately after submission, FIFO should be non-empty
        @(posedge clk);
        #1;
        check(conn_status[CONN_FIFO_NOT_EMPTY] === 1'b1 || conn_status[CONN_BUSY] === 1'b1,
              "FIFO_NOT_EMPTY or BUSY after submit (txn is in flight)");

        // Wait for it to complete (txn_pending drives soc_clk_en)
        begin : txn_pending_wait
            integer tw;
            tw = 0;
            while (tw < 2000) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    tw = 2000;
                end
                tw = tw + 1;
            end
        end

        check(!conn_status[CONN_BUSY], "Transaction completed despite debug clock stopped");
        check(user_rdata[7:0] === 8'd5, "Correct data returned through txn_pending path");

        // Verify status is integrity pass (not timeout)
        check(status === 32'hAAAA_AAAA,
              "Status = 0xAAAA_AAAA (txn completed, not timed out)");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 17: ILA Probe Readback — Multi-Probe Sweep
        // =============================================================
        begin_test("ILA Probe Readback — Multi-Probe Channel Sweep");

        // Sweep all system debug probes (0x00 to 0x09) + register probes (0x10 to 0x17)
        // Verify none return X values (structural connectivity check)
        begin : probe_sweep_block
            integer p;
            reg [`MMIO_DATA_WIDTH-1:0] probe_val;
            reg all_valid;
            all_valid = 1;

            $display("  System probes (0x00 - 0x09):");
            for (p = 0; p < 10; p = p + 1) begin
                read_probe(p[4:0], probe_val);
                $display("    Probe 0x%02h = 0x%016h", p[4:0], probe_val);
                if (probe_val === {`MMIO_DATA_WIDTH{1'bx}}) begin
                    all_valid = 0;
                    $display("    [WARN] Probe 0x%02h returned X!", p[4:0]);
                end
            end

            $display("  Register probes (0x10 - 0x17):");
            for (p = 0; p < (1 << `REG_ADDR_WIDTH); p = p + 1) begin
                read_probe(5'h10 | p[4:0], probe_val);
                $display("    Probe 0x%02h (R%0d) = 0x%016h",
                         5'h10 | p[4:0], p, probe_val);
                if (probe_val === {`MMIO_DATA_WIDTH{1'bx}}) begin
                    all_valid = 0;
                end
            end

            check(all_valid, "All probe channels return non-X data");
        end

        // Verify default probe (probe_sel=0x0F, should hit default case → all 1s)
        read_probe(5'h0F, rd_data);
        $display("  [INFO] Probe 0x0F (default) = 0x%016h", rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === {`DATA_WIDTH{1'b1}},
              "Default probe selector returns all 1s");

        // =============================================================
        // Test 18: SoC Driver State Observation (BUSY During Transaction)
        // =============================================================
        begin_test("SoC Driver State Observation — BUSY Flag");

        // To observe BUSY=1, we need the driver to stay in a non-IDLE state.
        // Strategy: start CPU (system_active=1 → req_rdy=0), submit a txn,
        // then observe BUSY before the timeout expires.

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0)); // cpu_run_level=1
        repeat (5) @(posedge clk);

        // Submit a read — will be stuck in SEND_REQ because req_rdy=0
        @(posedge clk);
        while (!user_ready) @(posedge clk);
        user_cmd   = 1'b0;
        user_addr  = REGION_DMEM_BASE;
        user_wdata = {`MMIO_DATA_WIDTH{1'b0}};
        user_valid = 1;
        @(posedge clk);
        user_valid = 0;

        // Check BUSY immediately
        repeat (5) @(posedge clk);
        #1;
        check(conn_status[CONN_BUSY] === 1'b1,
              "Driver BUSY=1 while waiting for req_rdy");
        check(conn_status[CONN_FIFO_NOT_EMPTY] === 1'b0,
              "FIFO empty (transaction dequeued into FSM)");

        $display("  [INFO] conn_status = 0x%02h (driver stuck in SEND_REQ)", conn_status);

        // Wait for timeout (TIMEOUT_THRESHOLD=1000 + margin)
        begin : state_obs_wait
            integer tw;
            tw = 0;
            while (tw < 1200) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY]) begin
                    tw = 1200;
                end
                tw = tw + 1;
            end
        end

        check(conn_status[CONN_BUSY] === 1'b0, "Driver returned to IDLE after timeout");
        check(status === 32'hDEAD_DEAD, "Status = 0xDEAD_DEAD (timeout)");
        check(conn_status[CONN_REQ_TIMEOUT] === 1'b1, "req_timeout flag set");

        // Stop CPU
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // =============================================================
        // Test 19: Transaction Quality — Max Latency
        // =============================================================
        begin_test("Transaction Quality — Max Latency Tracking");

        // Clear stats to start fresh
        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

        check(txn_quality[31:16] === 16'd0, "max_latency=0 after clear");

        // Do a normal MMIO transaction (cpu stopped, should complete quickly)
        do_write(REGION_DMEM_BASE | 32'h0,
                 {{(`MMIO_DATA_WIDTH-8){1'b0}}, 8'hAB},
                 "Single write for latency measurement");

        @(posedge clk); @(posedge clk);
        begin : max_lat_block
            reg [15:0] max_lat;
            max_lat = txn_quality[31:16];
            $display("  [INFO] max_latency after 1 txn = %0d cycles", max_lat);
            check(max_lat > 16'd0, "max_latency > 0 (transaction took at least 1 cycle)");
            check(max_lat < 16'd100, "max_latency < 100 (no timeout, normal path)");
        end

        // Do more transactions and verify max_latency only grows (or stays same)
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Read for latency tracking");
        begin : max_lat2_block
            reg [15:0] prev_max, new_max;
            prev_max = txn_quality[31:16];
            // One more transaction
            do_write(REGION_DMEM_BASE | 32'h1,
                     {{(`MMIO_DATA_WIDTH-8){1'b0}}, 8'hCD},
                     "Another write for latency");
            @(posedge clk);
            new_max = txn_quality[31:16];
            $display("  [INFO] max_latency: previous=%0d, current=%0d", prev_max, new_max);
            check(new_max >= prev_max, "max_latency monotonically non-decreasing");
        end

        // =============================================================
        // Test 20: Precise Counter Tracking
        // =============================================================
        begin_test("Precise Counter Tracking");

        // Clear all stats
        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

        check(txn_counters[15:0] === 16'd0, "total=0 after clear");
        check(txn_counters[31:16] === 16'd0, "reads=0 after clear");
        check(txn_counters[47:32] === 16'd0, "writes=0 after clear");

        // Do exactly 3 writes
        do_write(REGION_DMEM_BASE | 32'h0, 64'h1, "Counted write 1");
        do_write(REGION_DMEM_BASE | 32'h1, 64'h2, "Counted write 2");
        do_write(REGION_DMEM_BASE | 32'h2, 64'h3, "Counted write 3");

        @(posedge clk); @(posedge clk);
        $display("  [INFO] After 3 writes: total=%0d reads=%0d writes=%0d",
                 txn_counters[15:0], txn_counters[31:16], txn_counters[47:32]);
        check(txn_counters[15:0] === 16'd3, "total=3 after 3 writes");
        check(txn_counters[31:16] === 16'd0, "reads=0 after 3 writes");
        check(txn_counters[47:32] === 16'd3, "writes=3 after 3 writes");

        // Do exactly 2 reads
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Counted read 1");
        do_read(REGION_DMEM_BASE | 32'h1, rd_data, "Counted read 2");

        @(posedge clk); @(posedge clk);
        $display("  [INFO] After +2 reads: total=%0d reads=%0d writes=%0d",
                 txn_counters[15:0], txn_counters[31:16], txn_counters[47:32]);
        check(txn_counters[15:0] === 16'd5, "total=5 after 3W+2R");
        check(txn_counters[31:16] === 16'd2, "reads=2 after 3W+2R");
        check(txn_counters[47:32] === 16'd3, "writes=3 after 3W+2R");

        // Verify read+write=total
        check((txn_counters[31:16] + txn_counters[47:32]) === txn_counters[15:0],
              "reads + writes == total");

        // Verify link_active flag is set after successful transactions
        check(conn_status[CONN_LINK_ACTIVE] === 1'b1, "link_active=1 after successful txns");

        // =============================================================
        // Test 21: Clear Stats & Post-Clear Increment
        // =============================================================
        begin_test("Clear Stats & Post-Clear Increment");

        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        @(posedge clk); @(posedge clk);

        check(txn_counters[15:0] === 16'd0, "Total count = 0 after clear");
        check(txn_counters[31:16] === 16'd0, "Read count = 0 after clear");
        check(txn_counters[47:32] === 16'd0, "Write count = 0 after clear");
        check(txn_quality[31:16] === 16'd0, "max_latency = 0 after clear");
        check(conn_status[CONN_LINK_ACTIVE] === 1'b0, "link_active cleared");
        check(conn_status[CONN_PROTOCOL_ERR] === 1'b0, "protocol_error cleared");

        // One read, verify counters start from 0
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Post-clear read");
        @(posedge clk); @(posedge clk);
        check(txn_counters[15:0] === 16'd1, "Total=1 after one post-clear txn");
        check(txn_counters[31:16] === 16'd1, "Read=1");
        check(txn_counters[47:32] === 16'd0, "Write=0 (only did a read)");
        check(conn_status[CONN_LINK_ACTIVE] === 1'b1, "link_active re-set after 1st txn");

        // =============================================================
        // Test 22: Back-to-Back Transactions (FIFO Stress)
        // =============================================================
        begin_test("Back-to-Back Transactions (FIFO Depth)");

        begin : fifo_stress
            integer i;
            for (i = 0; i < 8; i = i + 1) begin
                @(posedge clk);
                while (!user_ready) @(posedge clk);
                user_cmd   = 1'b1;
                user_addr  = REGION_DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0];
                user_wdata = {{(`MMIO_DATA_WIDTH-8){1'b0}}, i[7:0]};
                user_valid = 1;
                @(posedge clk);
                user_valid = 0;
            end
        end

        $display("  Queued 8 write transactions, waiting for completion...");

        begin : fifo_drain
            integer timeout;
            timeout = 0;
            while (timeout < 5000) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    timeout = 5000;
                end
                timeout = timeout + 1;
            end
        end

        check(!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY],
              "All 8 transactions completed, FIFO drained");

        // Verify written values
        begin : fifo_verify
            integer i;
            reg [`MMIO_DATA_WIDTH-1:0] check_data;
            reg all_ok;
            all_ok = 1;
            for (i = 0; i < 8; i = i + 1) begin
                do_read(REGION_DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0], check_data, "Verify DMEM batch");
                if (check_data[7:0] !== i[7:0]) begin
                    $display("  [FAIL] DMEM[%0d] = 0x%0h, expected 0x%0h",
                             i, check_data[7:0], i[7:0]);
                    all_ok = 0;
                end
            end
            check(all_ok, "All 8 batch-written DMEM values verified");
        end

        // =============================================================
        // Test 23: FIFO Full Behavior
        // =============================================================
        begin_test("FIFO Full Behavior — user_ready Goes Low");

        // Strategy: Start CPU (req_rdy=0) so driver is stuck.
        // Submit 16+ transactions quickly. FIFO (depth 16) should fill up.
        // Observe user_ready → 0.
        // Then stop CPU, let transactions timeout and drain, verify FIFO empties.

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0)); // CPU running → req_rdy=0
        repeat (5) @(posedge clk);

        begin : fifo_full_block
            integer i;
            integer submitted;
            reg saw_not_ready;
            submitted = 0;
            saw_not_ready = 0;

            // Attempt to submit 20 transactions, one per cycle
            for (i = 0; i < 20; i = i + 1) begin
                @(posedge clk);
                #1;
                if (!user_ready) begin
                    saw_not_ready = 1;
                    $display("  [INFO] user_ready=0 at submission %0d (FIFO full)", i);
                    // Stop submitting
                    user_valid = 0;
                    i = 20; // break
                end else begin
                    user_cmd   = 1'b1;
                    user_addr  = REGION_DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0];
                    user_wdata = {{(`MMIO_DATA_WIDTH-8){1'b0}}, i[7:0]};
                    user_valid = 1;
                    submitted = submitted + 1;
                end
            end
            @(posedge clk);
            user_valid = 0;

            $display("  [INFO] Successfully submitted %0d transactions before FIFO full", submitted);

            if (saw_not_ready) begin
                check(1'b1, "user_ready went low (FIFO full detected)");
                // The FIFO depth is 16, but the driver dequeues 1 immediately
                // (into the FSM), so the FIFO holds at most DEPTH-1 before the
                // first one is dequeued. Expected: ~16-17 submissions before full.
                check(submitted >= 2, "At least 2 transactions accepted before full");
            end else begin
                // If we submitted all 20 without seeing FIFO full, the driver
                // must be draining fast enough (unexpected with CPU running)
                check(1'b0, "Expected user_ready to go low but it never did");
            end
        end

        // Now wait for everything to drain (will timeout)
        $display("  Waiting for FIFO to drain (transactions will timeout)...");
        begin : fifo_full_drain
            integer tw;
            tw = 0;
            // Need enough time: up to 17 transactions * 1000 cycle timeout each
            // That's 17000 cycles. Use a generous limit.
            while (tw < 25000) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    tw = 25000;
                end
                tw = tw + 1;
            end
        end

        check(!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY],
              "FIFO fully drained after timeouts");
        check(user_ready === 1'b1, "user_ready=1 after FIFO drain");

        // Stop CPU
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // =============================================================
        // Test 24: Timeout Scenario — Request Phase
        // =============================================================
        begin_test("Timeout — Request Phase (CPU Running, req_rdy Blocked)");

        // Clear stats to get a clean timeout count
        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

        // Start CPU → system_active=1 → req_rdy=0
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        repeat (5) @(posedge clk);

        $display("  Submitting MMIO read while CPU is running...");
        submit_txn(1'b0, REGION_DMEM_BASE | 32'h0, {`MMIO_DATA_WIDTH{1'b0}});

        begin : timeout_wait_24
            integer t;
            t = 0;
            while (t < 1200) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    t = 1200;
                end
                t = t + 1;
            end
        end

        check(status === 32'hDEAD_DEAD, "Status = 0xDEAD_DEAD (req timeout)");
        check(conn_status[CONN_REQ_TIMEOUT] === 1'b1, "req_timeout flag set");
        check(conn_status[CONN_TXN_TIMEDOUT] === 1'b0,
              "txn_timed_out cleared after CHECK→IDLE transition");

        // Verify timeout_count in txn_quality
        $display("  [INFO] txn_quality = 0x%0h", txn_quality);
        $display("  [INFO] timeout_count = %0d", txn_quality[47:32]);
        check(txn_quality[47:32] >= 16'd1, "timeout_count >= 1");
        check(txn_quality[0] === 1'b1, "txn_quality req_timeout flag set");

        // Stop CPU
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // =============================================================
        // Test 25: Multiple Timeouts & Timeout Counter
        // =============================================================
        begin_test("Multiple Timeouts & Timeout Counter Verification");

        // Clear stats
        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

        // Start CPU → req_rdy=0
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        repeat (5) @(posedge clk);

        // Submit 3 transactions that will all timeout
        $display("  Submitting 3 transactions that will timeout...");
        begin : multi_timeout
            integer i;
            for (i = 0; i < 3; i = i + 1) begin
                submit_txn(1'b0, REGION_DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0],
                           {`MMIO_DATA_WIDTH{1'b0}});
            end
        end

        // Wait for all 3 to timeout (3 * ~1000 cycles + margin)
        $display("  Waiting for 3 timeouts (~3200 cycles)...");
        begin : multi_to_wait
            integer tw;
            tw = 0;
            while (tw < 4000) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    tw = 4000;
                end
                tw = tw + 1;
            end
        end

        @(posedge clk); @(posedge clk);
        $display("  [INFO] timeout_count = %0d", txn_quality[47:32]);
        $display("  [INFO] total_txn_count = %0d", txn_counters[15:0]);
        check(txn_quality[47:32] === 16'd3,
              "timeout_count = 3 (all 3 transactions timed out)");
        // Timed-out transactions should NOT increment total_txn_count
        check(txn_counters[15:0] === 16'd0,
              "total_txn_count = 0 (timeouts are not counted as successful)");
        check(conn_status[CONN_REQ_TIMEOUT] === 1'b1, "req_timeout flag still latched");

        // Stop CPU
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // Verify normal transactions still work after timeouts
        do_write(REGION_DMEM_BASE | 32'h0, 64'hCAFE, "Post-timeout write");
        check(status === 32'hAAAA_AAAA, "Normal transaction succeeds after timeout recovery");
        @(posedge clk);
        check(txn_counters[15:0] === 16'd1,
              "total_txn_count incremented for successful post-timeout txn");

        // =============================================================
        // Test 26: ILA CLEAR Command
        // =============================================================
        begin_test("ILA CLEAR Command");

        debug_mode = 1;
        @(posedge clk);

        // Put ILA into RUN mode with cpu_run_level=1
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 1, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Pre-clear: run_mode=1");
        check(rd_data[STAT_CPU_RUN] === 1'b1, "Pre-clear: cpu_run_level=1");

        // Write a non-zero probe_sel
        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h0A});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h0A, "Pre-clear: probe_sel=0x0A");

        // Issue CLEAR
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 1));
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b0, "Post-clear: run_mode=0");
        check(rd_data[STAT_CPU_RUN] === 1'b0, "Post-clear: cpu_run_level=0");
        check(rd_data[STAT_STEPPING] === 1'b0, "Post-clear: step_active=0");
        check(rd_data[STAT_STOPPED] === 1'b1, "Post-clear: stopped=1 (debug mode)");

        // probe_sel should NOT be cleared (CLEAR only resets run_mode/step/cpu_run)
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        $display("  [INFO] probe_sel after clear = 0x%0h", rd_data[4:0]);
        // Note: CLEAR command's scope is run_mode, step_active, cpu_run_level.
        // probe_sel_r is NOT affected by clear_pulse in the ILA code.
        check(rd_data[4:0] === 5'h0A, "probe_sel preserved after CLEAR");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 27: Post-Clear Full System Sanity Check
        // =============================================================
        begin_test("Post-Clear Full System Sanity Check");

        // After all the beating this system has taken, verify basic functionality
        // still works end-to-end.

        // 1. MMIO write & read
        do_write(REGION_DMEM_BASE | 32'h3F,
                 {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'h1234},
                 "Final sanity: write DMEM[63]");
        do_read(REGION_DMEM_BASE | 32'h3F, rd_data, "Final sanity: read DMEM[63]");
        check(rd_data[15:0] === 16'h1234, "DMEM[63] = 0x1234 (write/read works)");
        check(status === 32'hAAAA_AAAA, "Status = integrity pass");

        // 2. ILA register access still works
        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h05});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h05, "ILA PROBE_SEL still writable");

        // 3. Cycle counter still running
        ila_read(ILA_ADDR_CYCLE, rd_data);
        repeat (5) @(posedge clk);
        ila_read(ILA_ADDR_CYCLE, rd_data2);
        check(rd_data2 > rd_data, "Cycle counter still incrementing");

        // 4. No error flags set from normal operation
        check(conn_status[CONN_PROTOCOL_ERR] === 1'b0,
              "No protocol errors throughout entire test suite");

        // 5. Driver is idle
        check(conn_status[CONN_BUSY] === 1'b0, "Driver idle at end of test");
        check(conn_status[CONN_FIFO_NOT_EMPTY] === 1'b0, "FIFO empty at end of test");
        check(user_ready === 1'b1, "user_ready=1 at end of test");

        // =============================================================
        // Summary
        // =============================================================
        $display("\n################################################################");
        $display("###                  TEST SUMMARY                           ###");
        $display("################################################################");
        $display("  Total Tests:  %0d", test_num);
        $display("  Total PASS:   %0d", pass_count);
        $display("  Total FAIL:   %0d", fail_count);
        $display("  Pass Rate:    %0d%%", (pass_count * 100) / (pass_count + fail_count));
        if (fail_count == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else
            $display("  >>> %0d TEST(S) FAILED <<<", fail_count);
        $display("################################################################\n");

        if (fail_count > 0) $stop;
        $finish;
    end

    // =========================================================
    // Timeout Watchdog
    // =========================================================
    initial begin
        #1_000_000; // 1ms absolute timeout (increased for FIFO full tests)
        $display("\n[TIMEOUT] Simulation exceeded 1ms — aborting.");
        $display("  PASS: %0d, FAIL: %0d (at time of abort)", pass_count, fail_count);
        $finish;
    end

    // =========================================================
    // Waveform Dump
    // =========================================================
    initial begin
        $dumpfile("top_tb.vcd");
        $dumpvars(0, top_tb);
    end

endmodule