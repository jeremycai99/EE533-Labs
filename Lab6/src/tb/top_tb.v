/* file: top_tb.v
 * Description: Thorough testbench for top module (SoC + ILA + SoC Driver).
 *
 * Tests:
 *   1.  Reset defaults & static outputs
 *   2.  ILA direct register access (CTRL, PROBE_SEL, STATUS, CYCLE readback)
 *   3.  ILA clock gating (stop / run / step) — standalone
 *   4.  MMIO write → DMEM (single transaction, verify status)
 *   5.  MMIO read  ← DMEM (round-trip data integrity)
 *   6.  MMIO write → IMEM (load a program)
 *   7.  MMIO read  ← IMEM (verify loaded instructions)
 *   8.  MMIO read  ← CTRL (read system_active flag)
 *   9.  CPU end-to-end: load, start via ILA, run, stop, read-back results
 *  10.  Debug mode: clock stopped, MMIO still completes (txn_pending keeps clock alive)
 *  11.  ILA probe readback through top (cpu_debug_data path)
 *  12.  Transaction quality counters & clear_stats
 *  13.  Multiple back-to-back MMIO transactions (FIFO depth stress)
 *  14.  Timeout scenario: start CPU → MMIO blocked → driver times out
 *
 * Dependencies: top.v (which includes ila.v, soc.v, soc_driver.v, cpu.v, etc.)
 *
 * Notes on the CPU pipeline:
 *   - Very limited: no ALU, no branches, no hazard detection.
 *   - Supports: load (reg_write=1, mem_write=0) and store (mem_write=1, reg_write=0).
 *   - cpu_done asserts when PC reaches all-ones ({PC_WIDTH{1'b1}}).
 *   - Instruction encoding:
 *       [31]    mem_write
 *       [30]    reg_write
 *       [29:27] r1addr   (data source for store)
 *       [26:24] r0addr   (address source for load/store)
 *       [23:21] rdaddr   (destination register)
 *       [20:0]  unused
 *
 * Author: Auto-generated testbench
 * Date: Feb. 15, 2026
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
    // Parameters: (mem_write, reg_write, rd, r0, r1)
    // Encoding:   {mw, rw, r0, r1, rd, zeros[20:0]}
    //
    // CPU decodes:
    //   r0addr = instr[26:24]  (address source)
    //   r1addr = instr[29:27]  (data source for stores)
    //   rdaddr = instr[23:21]  (destination register for loads)

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

    // --- Submit one MMIO transaction through the driver ---
    // cmd: 0 = read, 1 = write
    task submit_txn(
        input                         cmd,
        input [`MMIO_ADDR_WIDTH-1:0]  addr,
        input [`MMIO_DATA_WIDTH-1:0]  wdata
    );
        begin
            // Wait until FIFO is not full
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
    // Returns 1 if completed normally, 0 if timed out (in sim, not MMIO timeout)
    task wait_txn_done(output reg success);
        integer timeout_cnt;
        begin
            success = 0;
            timeout_cnt = 0;
            // Wait for driver to leave IDLE (it may already have)
            repeat (2) @(posedge clk);
            // Now wait for driver to return to IDLE with empty FIFO
            while (timeout_cnt < 2000) begin
                @(posedge clk);
                #1;
                if (!conn_status[CONN_BUSY] && !conn_status[CONN_FIFO_NOT_EMPTY]) begin
                    success = 1;
                    timeout_cnt = 2000; // break
                end
                timeout_cnt = timeout_cnt + 1;
            end
            @(posedge clk); // one more settle cycle
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
    reg txn_ok;

    initial begin
        $display("\n################################################################");
        $display("###          TOP MODULE TESTBENCH — Full Suite               ###");
        $display("################################################################");

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
        check(status === 32'h0, "status is 0 after reset");

        // ILA: cpu_run_level defaults to 0, so soc_start=0
        // → system_active=0 in SoC → req_rdy should be 1 (SoC idle, not active)
        // But req_rdy is internal to SoC. We can infer it from driver behavior.

        // Read ILA STATUS register
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN]  === 1'b0, "ILA STATUS: cpu_run_level=0 after reset");
        check(rd_data[STAT_DEBUG]    === 1'b0, "ILA STATUS: debug_mode=0");
        check(rd_data[STAT_RUNNING]  === 1'b0, "ILA STATUS: run_mode=0");
        check(rd_data[STAT_STEPPING] === 1'b0, "ILA STATUS: step_active=0");

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

        // Cycle counter should be incrementing (debug_mode=0 → clock free-running)
        ila_read(ILA_ADDR_CYCLE, rd_data);
        repeat (10) @(posedge clk);
        ila_read(ILA_ADDR_CYCLE, rd_data2);
        check(rd_data2 > rd_data, "Cycle counter is incrementing");

        // CTRL: set cpu_run_level=1, read back, then clear
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "cpu_run_level=1 after CTRL write");
        // Clear it back so MMIO can work
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b0, "cpu_run_level=0 after clearing");

        // =============================================================
        // Test 3: ILA Clock Gating (standalone)
        // =============================================================
        begin_test("ILA Clock Gating — Stop / Run / Step");

        debug_mode = 1;
        @(posedge clk);

        // Read STATUS to confirm debug_mode visible
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_DEBUG] === 1'b1, "STATUS reflects debug_mode=1");

        // STOP the clock
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk);
        // soc_clk_en should be 0 (unless busy/txn_pending)
        // We can check ILA status: stopped should be 1
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Clock STOPPED in debug mode");
        check(rd_data[STAT_RUNNING] === 1'b0, "run_mode=0");

        // RUN the clock
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Clock RUNNING after RUN command");
        check(rd_data[STAT_STOPPED] === 1'b0, "stopped=0");

        // STOP again, then STEP
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk); @(posedge clk);

        // Issue step
        @(posedge clk);
        ila_addr = ILA_ADDR_CTRL;
        ila_din  = ila_ctrl(1, 0, 0, 0, 0);
        ila_we   = 1;
        @(posedge clk);
        ila_we = 0;
        ila_din = 0;

        // step_active should be high for one cycle, then auto-clear
        #1;
        // Read stepping status on the active cycle
        ila_read(ILA_ADDR_STATUS, rd_data);
        // After the read settles, step_active has already auto-cleared
        // Just verify the system didn't crash
        $display("  [INFO] Step command issued, STATUS=0x%0h", rd_data);

        // Return to normal mode
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 4: MMIO Write to DMEM
        // =============================================================
        begin_test("MMIO Write to DMEM");

        // Write value 0xDEAD to DMEM address 0
        do_write(REGION_DMEM_BASE | 32'h0, {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'hDEAD},
                 "MMIO write to DMEM[0] completed");

        check(status === 32'hAAAA_AAAA, "Status = integrity pass (0xAAAA_AAAA)");

        // Write value 0xBEEF to DMEM address 4
        do_write(REGION_DMEM_BASE | 32'h4, {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'hBEEF},
                 "MMIO write to DMEM[4] completed");

        check(status === 32'hAAAA_AAAA, "Status = integrity pass for DMEM[4]");

        // =============================================================
        // Test 5: MMIO Read from DMEM
        // =============================================================
        begin_test("MMIO Read from DMEM");

        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "MMIO read from DMEM[0] completed");
        $display("  [INFO] DMEM[0] read data = 0x%0h", rd_data);
        check(rd_data[15:0] === 16'hDEAD, "DMEM[0] data matches written value 0xDEAD");

        do_read(REGION_DMEM_BASE | 32'h4, rd_data, "MMIO read from DMEM[4] completed");
        $display("  [INFO] DMEM[4] read data = 0x%0h", rd_data);
        check(rd_data[15:0] === 16'hBEEF, "DMEM[4] data matches written value 0xBEEF");

        // =============================================================
        // Test 6: MMIO Write to IMEM (Load a Program)
        // =============================================================
        begin_test("MMIO Write to IMEM — Load Program");

        // Same program as cpu_tb:
        //   Addr 0: Load d_mem[R0] → R2     (rw=1, mw=0, r0=R0, rd=R2)
        //   Addr 1: Load d_mem[R0] → R3     (rw=1, mw=0, r0=R0, rd=R3)
        //   Addr 2: NOP
        //   Addr 3: NOP
        //   Addr 4: NOP
        //   Addr 5: Store R3→d_mem[R2]      (mw=1, rw=0, r0=R2, r1=R3)

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
        // Test 7: MMIO Read from IMEM (Verify Program)
        // =============================================================
        begin_test("MMIO Read from IMEM — Verify Loaded Program");

        do_read(REGION_IMEM_BASE | 32'h0, rd_data, "Read IMEM[0]");
        check(rd_data[`INSTR_WIDTH-1:0] === build_instr(1'b0, 1'b1, R2, R0, R0),
              "IMEM[0] matches Load R2 instruction");

        do_read(REGION_IMEM_BASE | 32'h5, rd_data, "Read IMEM[5]");
        check(rd_data[`INSTR_WIDTH-1:0] === build_instr(1'b1, 1'b0, R0, R2, R3),
              "IMEM[5] matches Store instruction");

        // =============================================================
        // Test 8: MMIO Read from CTRL Region
        // =============================================================
        begin_test("MMIO Read from CTRL — system_active Flag");

        do_read(REGION_CTRL_BASE, rd_data, "Read CTRL region");
        $display("  [INFO] CTRL read data = 0x%0h", rd_data);
        check(rd_data[0] === 1'b0, "system_active=0 (CPU not started yet)");

        // =============================================================
        // Test 9: CPU End-to-End Execution
        // =============================================================
        begin_test("CPU End-to-End Execution");

        // Pre-condition: DMEM[0] = 0xDEAD (written in Test 4)
        //                DMEM[4] = 0xBEEF (written in Test 4)
        //                IMEM loaded with program (Test 6)
        //
        // Program logic:
        //   Cycle 0: Load d_mem[R0=0] → R2   → R2 = 0xDEAD
        //   Cycle 1: Load d_mem[R0=0] → R3   → R3 = 0xDEAD
        //   Cycle 2-4: NOPs (pipeline drain — R2/R3 become available in regfile)
        //   Cycle 5: Store R3 → d_mem[R2]
        //     r0addr=R2 → address = value of R2 = 0xDEAD (truncated to DMEM_ADDR_WIDTH)
        //     r1addr=R3 → data = value of R3 = 0xDEAD
        //
        //   After execution: d_mem[0xDEAD & DMEM_MASK] should contain 0xDEAD
        //
        // BUT the CPU register file starts at 0 on reset. R0=0, so d_mem[0]=0xDEAD is loaded.
        // R2 receives 0xDEAD. Store writes 0xDEAD to address 0xDEAD (truncated).
        //
        // For a simpler verification path, let's also prepare DMEM[0] = small value
        // that fits in DMEM_ADDR_WIDTH.

        // Overwrite DMEM[0] with a small, controlled value that is a valid DMEM address
        do_write(REGION_DMEM_BASE | 32'h0,
                 {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, `DATA_WIDTH'd5},
                 "Preload DMEM[0] = 5 (valid DMEM address)");

        // Verify DMEM[0] was written
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Verify DMEM[0]");
        check(rd_data[7:0] === 8'd5, "DMEM[0] = 5 confirmed");

        // Set a known value at DMEM[5] so we can see it change
        do_write(REGION_DMEM_BASE | 32'h5,
                 {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, `DATA_WIDTH'hFFFF_FFFF},
                 "Preload DMEM[5] = 0xFFFF_FFFF (will be overwritten by CPU)");

        $display("  Starting CPU via ILA cpu_run_level...");

        // Start the CPU: set cpu_run_level = 1 in ILA CTRL
        // This drives soc_start=1 → system_active=1 → CPU comes out of reset
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));

        // Verify system_active is now high by checking ILA STATUS
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "cpu_run_level=1, CPU starting");

        // Let the CPU run. It needs to cycle through all PC values to reach cpu_done.
        // PC_WIDTH bits → 2^PC_WIDTH cycles. Add generous margin.
        // With PC_WIDTH=9 that's 512 cycles + pipeline depth + margin.
        $display("  Waiting for CPU to complete (~600 cycles)...");
        repeat (700) @(posedge clk);

        // Stop the CPU: clear cpu_run_level
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // Verify system_active dropped
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b0, "cpu_run_level=0 after stopping CPU");

        // Now read DMEM[5] — the store instruction should have written there
        // Program stored value_of_R3 at address value_of_R2.
        // R2 was loaded from d_mem[0]=5, R3 was loaded from d_mem[0]=5.
        // So d_mem[5] should now be 5 (the store wrote R3=5 to addr R2=5).
        do_read(REGION_DMEM_BASE | 32'h5, rd_data, "Read DMEM[5] after CPU execution");
        $display("  [INFO] DMEM[5] after CPU = 0x%0h", rd_data);
        check(rd_data == `MMIO_DATA_WIDTH'd5,
              "DMEM[5] = 5 (CPU stored R3=5 at address R2=5)");

        // Also verify DMEM[0] wasn't corrupted
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Read DMEM[0] after CPU execution");
        check(rd_data[7:0] === 8'd5, "DMEM[0] still = 5 (not corrupted by CPU)");

        // =============================================================
        // Test 10: Debug Mode — MMIO with Clock Stopped
        //   (txn_pending / soc_busy should keep gated clock alive)
        // =============================================================
        begin_test("Debug Mode — MMIO Transactions with Clock Gated");

        debug_mode = 1;
        @(posedge clk);

        // STOP the SoC clock
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "SoC clock stopped");

        // Now attempt an MMIO read of DMEM[0].
        // The driver runs on ungated clk, but the SoC runs on gated clock.
        // txn_pending from the driver's FIFO should force soc_clk_en=1,
        // allowing the gated clock to resume for the transaction.
        $display("  Attempting MMIO read while debug clock is stopped...");
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "MMIO read DMEM[0] while debug-stopped");
        check(rd_data[7:0] === 8'd5, "DMEM[0] still = 5 (txn_pending kept clock alive)");

        // Write test too
        do_write(REGION_DMEM_BASE | 32'h7,
                 {{(`MMIO_DATA_WIDTH-8){1'b0}}, 8'h42},
                 "MMIO write DMEM[7]=0x42 while debug-stopped");

        do_read(REGION_DMEM_BASE | 32'h7, rd_data, "MMIO read DMEM[7] while debug-stopped");
        check(rd_data[7:0] === 8'h42, "DMEM[7] = 0x42 (write succeeded through gated clock)");

        // Restore normal mode
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 11: ILA Probe Readback
        // =============================================================
        begin_test("ILA Probe Readback Path");

        // The ILA reads cpu_debug_data from the SoC's CPU.
        // cpu_debug_sel comes from ILA's probe_sel_r.
        // With CPU not running (reset), probe data should be deterministic.

        // Select probe 0 (PC in fetch stage) — should be 0 after reset since CPU is in reset
        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h00});
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_PROBE, rd_data);
        $display("  [INFO] Probe 0x00 (PC) = 0x%0h", rd_data);
        // PC value depends on whether CPU is in reset or not.
        // cpu_run_level=0 → soc_start=0 → cpu_rst_n=0 → PC should be 0
        check(rd_data[`PC_WIDTH-1:0] === {`PC_WIDTH{1'b0}},
              "Probe[0] PC = 0 (CPU in reset)");

        // Select probe for register file mode (bit 4 = 1, addr = 0)
        // R0 should always be 0
        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h10});
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_PROBE, rd_data);
        $display("  [INFO] Probe 0x10 (Reg R0) = 0x%0h", rd_data);
        // After CPU reset, register file values are undefined unless explicitly reset.
        // This is an observability test — just verify the path works without X.
        check(rd_data !== {`MMIO_DATA_WIDTH{1'bx}}, "Probe path for regfile is not X");

        // =============================================================
        // Test 12: Transaction Quality Counters
        // =============================================================
        begin_test("Transaction Quality Counters");

        // We've done many successful MMIO transactions.
        // txn_counters packs: {write_count[47:32], read_count[31:16], total_count[15:0]}
        $display("  [INFO] txn_counters = 0x%0h", txn_counters);
        $display("  [INFO] txn_quality  = 0x%0h", txn_quality);
        $display("  [INFO] conn_status  = 0x%02h", conn_status);

        check(txn_counters[15:0] > 16'd0, "Total transaction count > 0");
        check(conn_status[CONN_LINK_ACTIVE] === 1'b1, "Link active flag set");
        check(conn_status[CONN_PROTOCOL_ERR] === 1'b0, "No protocol errors detected");

        // Verify read + write counts sum to total
        begin : counter_check_block
            reg [15:0] total, reads, writes;
            total  = txn_counters[15:0];
            reads  = txn_counters[31:16];
            writes = txn_counters[47:32];
            $display("  [INFO] Total=%0d, Reads=%0d, Writes=%0d", total, reads, writes);
            check((reads + writes) === total,
                  "read_count + write_count == total_count");
            check(reads > 0 && writes > 0, "Both read and write counts are non-zero");
        end

        // =============================================================
        // Test 13: Clear Stats
        // =============================================================
        begin_test("Clear Stats");

        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        @(posedge clk); @(posedge clk);

        check(txn_counters[15:0] === 16'd0, "Total count = 0 after clear");
        check(txn_counters[31:16] === 16'd0, "Read count = 0 after clear");
        check(txn_counters[47:32] === 16'd0, "Write count = 0 after clear");

        // Do one more transaction and verify counter increments from 0
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Post-clear read DMEM[0]");
        @(posedge clk); @(posedge clk);
        check(txn_counters[15:0] === 16'd1, "Total count = 1 after one post-clear txn");
        check(txn_counters[31:16] === 16'd1, "Read count = 1");

        // =============================================================
        // Test 14: Multiple Back-to-Back Transactions (FIFO Stress)
        // =============================================================
        begin_test("Back-to-Back Transactions (FIFO Depth)");

        // Queue 8 write transactions rapidly — the FIFO (depth 16) should absorb them
        begin : fifo_stress
            integer i;
            for (i = 0; i < 8; i = i + 1) begin
                @(posedge clk);
                while (!user_ready) @(posedge clk);
                user_cmd   = 1'b1; // write
                user_addr  = REGION_DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0];
                user_wdata = {{(`MMIO_DATA_WIDTH-8){1'b0}}, i[7:0]};
                user_valid = 1;
                @(posedge clk);
                user_valid = 0;
            end
        end

        $display("  Queued 8 write transactions, waiting for completion...");

        // Wait for all to drain
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

        // Verify written values by reading back
        begin : fifo_verify
            integer i;
            reg [`MMIO_DATA_WIDTH-1:0] check_data;
            reg all_ok;
            all_ok = 1;
            for (i = 0; i < 8; i = i + 1) begin
                do_read(REGION_DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0], check_data, "Verify DMEM batch");
                if (check_data[7:0] !== i[7:0]) begin
                    $display("  [FAIL] DMEM[%0d] = 0x%0h, expected 0x%0h", i, check_data[7:0], i[7:0]);
                    all_ok = 0;
                end
            end
            check(all_ok, "All 8 batch-written DMEM values verified correctly");
        end

        // =============================================================
        // Test 15: Timeout Scenario — MMIO While CPU Running
        // =============================================================
        begin_test("Timeout — MMIO While CPU Running (req_rdy blocked)");

        // Start the CPU so system_active=1 → SoC req_rdy=0
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0)); // cpu_run_level=1
        repeat (5) @(posedge clk);

        // Attempt an MMIO transaction — SoC will not accept it, driver should timeout
        $display("  Submitting MMIO read while CPU is running (expecting timeout)...");
        submit_txn(1'b0, REGION_DMEM_BASE | 32'h0, {`MMIO_DATA_WIDTH{1'b0}});

        // Wait for the driver's timeout (TIMEOUT_THRESHOLD=1000 cycles + margin)
        begin : timeout_wait
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

        $display("  [INFO] status after timeout = 0x%08h", status);
        check(status === 32'hDEAD_DEAD, "Status = 0xDEAD_DEAD (timeout detected)");
        check(conn_status[CONN_REQ_TIMEOUT] === 1'b1, "conn_status: req_timeout flag set");

        // Stop the CPU
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // =============================================================
        // Test 16: ILA CLEAR Resets Internal State
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

        // Issue CLEAR
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 1));
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b0, "Post-clear: run_mode=0");
        check(rd_data[STAT_CPU_RUN] === 1'b0, "Post-clear: cpu_run_level=0");
        check(rd_data[STAT_STOPPED] === 1'b1, "Post-clear: stopped=1 (debug mode)");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Summary
        // =============================================================
        $display("\n################################################################");
        $display("###                  TEST SUMMARY                           ###");
        $display("################################################################");
        $display("  Total PASS: %0d", pass_count);
        $display("  Total FAIL: %0d", fail_count);
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
        #500_000; // 500us absolute timeout
        $display("\n[TIMEOUT] Simulation exceeded 500us — aborting.");
        $display("  PASS: %0d, FAIL: %0d (at time of abort)", pass_count, fail_count);
        $finish;
    end

    // =========================================================
    // Optional: Waveform Dump
    // =========================================================
    initial begin
        $dumpfile("top_tb.vcd");
        $dumpvars(0, top_tb);
    end

endmodule