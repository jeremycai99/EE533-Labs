/* file: top_tb.v
 * Description: Refined testbench for top module (SoC + ILA + SoC Driver).
 *              Tests 1–9:   MMIO & ILA infrastructure
 *              Tests 10–14: ARM CPU micro-program execution
 *              Tests 15–27: Debug mode, clock gating, txn quality, driver stress
 *              Test  28:    Bubble Sort integration (hex load → run → verify)
 *
 * Dependencies: top.v (which includes ila.v, soc.v, soc_driver.v, cpu.v, etc.)
 * Date: Feb. 21, 2026
 *
 * Changes from previous version:
 *   - Replaced custom build_instr() with real ARM instruction encodings.
 *   - Added init_cpu_regs() task (hierarchical register init via u_top.u_soc…).
 *   - Tests 10–11 now load/verify real ARM instructions (MOV, STR, B .).
 *   - Test 13 runs the ARM micro-program; DMEM[5] canary check proves STR worked.
 *   - Tests 15–16 DMEM[0] expectation updated to 0xDEAD (set by Test 7).
 *   - Test 28 halt detection uses PC-in-range check (handles B . oscillation).
 *   - Bubble sort now calls init_cpu_regs() before starting CPU.
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
    localparam [4:0] PROBE_PC          = 5'h00;
    localparam [4:0] PROBE_INSTR_ID    = 5'h01;
    localparam [4:0] PROBE_R0_ID       = 5'h02;
    localparam [4:0] PROBE_R1_ID       = 5'h03;
    localparam [4:0] PROBE_R0_EX       = 5'h04;
    localparam [4:0] PROBE_R1_EX       = 5'h05;
    localparam [4:0] PROBE_WB_DATA     = 5'h06;
    localparam [4:0] PROBE_CTRL_VEC    = 5'h07;
    localparam [4:0] PROBE_RDADDR_ID   = 5'h08;
    localparam [4:0] PROBE_RDADDR_WB   = 5'h09;
    localparam [4:0] PROBE_REG_BASE    = 5'h10;

    // =========================================================
    // ARM Instruction Constants (replaces old build_instr)
    // =========================================================

    // ARM condition AL = 0xE (always execute)
    localparam [31:0] ARM_NOP         = 32'hE1A0_0000;  // MOV R0, R0
    localparam [31:0] ARM_MOV_R2_20   = 32'hE3A0_2014;  // MOV R2, #20
    localparam [31:0] ARM_MOV_R3_20   = 32'hE3A0_3014;  // MOV R3, #20
    localparam [31:0] ARM_STR_R3_R2   = 32'hE582_3000;  // STR R3, [R2, #0]
    localparam [31:0] ARM_BRANCH_SELF = 32'hEAFF_FFFE;  // B . (branch-to-self)

    // Micro-program layout (loaded into IMEM word 0..7 for Tests 10-14):
    //   [0] MOV R2, #20       — R2 ← 20
    //   [1] MOV R3, #20       — R3 ← 20
    //   [2] NOP               — pipeline safety
    //   [3] NOP
    //   [4] NOP
    //   [5] STR R3, [R2, #0]  — DMEM[byte 20 → word 5] ← 20
    //   [6] NOP               — let store retire
    //   [7] B .               — halt (byte addr 0x1C)
    localparam [31:0] TEST_HALT_WORD = 7;
    localparam [31:0] TEST_HALT_BYTE = 32'h0000_001C;   // 7 × 4

    // =========================================================
    // Bubble Sort Test Parameters
    // =========================================================

    localparam SORT_MEM_DEPTH    = 4096;
    localparam SORT_ARRAY_SIZE   = 10;
    localparam SORT_SCAN_RANGE   = 1024;
    localparam [31:0] SORT_HALT_INSTR     = 32'hEAFF_FFFE;
    localparam [31:0] SORT_HALT_BYTE_ADDR = 32'h0000_0200;
    localparam [31:0] SORT_HALT_WORD      = SORT_HALT_BYTE_ADDR >> 2; // 128
    localparam SORT_POLL_INTERVAL = 500;
    localparam SORT_TIMEOUT       = 200_000;

`ifdef SORT_HEX
    localparam SORT_HEX_FILE = `SORT_HEX;
`else
    localparam SORT_HEX_FILE = "../hex/sort_imem.txt";
`endif

    // =========================================================
    // DUT Signals
    // =========================================================

    reg                          clk;
    reg                          rst_n;
    reg                          debug_mode;

    reg                          user_valid;
    reg                          user_cmd;
    reg  [`MMIO_ADDR_WIDTH-1:0]  user_addr;
    reg  [`MMIO_DATA_WIDTH-1:0]  user_wdata;
    wire                         user_ready;
    wire [`MMIO_DATA_WIDTH-1:0]  user_rdata;

    wire [`MMIO_ADDR_WIDTH-1:0]  status;
    wire [7:0]                   conn_status;
    wire [`MMIO_DATA_WIDTH-1:0]  txn_quality;
    wire [`MMIO_DATA_WIDTH-1:0]  txn_counters;
    reg                          clear_stats;

    reg  [2:0]                   ila_addr;
    reg  [`MMIO_DATA_WIDTH-1:0]  ila_din;
    reg                          ila_we;
    wire [`MMIO_DATA_WIDTH-1:0]  ila_dout;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    // =========================================================
    // Bubble Sort Test Variables
    // =========================================================

    reg [`DATA_WIDTH-1:0]        sort_local_mem  [0:SORT_MEM_DEPTH-1];
    reg [`DATA_WIDTH-1:0]        sort_expected   [0:SORT_ARRAY_SIZE-1];
    reg [`MMIO_DATA_WIDTH-1:0]   sort_dmem_snap  [0:SORT_SCAN_RANGE-1];
    reg                          sort_cpu_halted;
    reg                          sort_array_found;
    integer                      sort_found_at;
    integer                      sort_last_nz;

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
    // ILA CTRL word builder
    // =========================================================

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
    // CPU Register Initialisation (hierarchical — simulation only)
    //   The register file is a plain reg array without reset.
    //   We force-init via hierarchical access while the CPU is
    //   held in reset (cpu_run_level=0 → cpu_rst_n=0).
    // =========================================================

    task init_cpu_regs;
        input [`DATA_WIDTH-1:0] lr_value;
        integer r;
        begin
            for (r = 0; r < `REG_DEPTH; r = r + 1)
                u_top.u_soc.u_cpu.u_regfile.regs[r] = {`DATA_WIDTH{1'b0}};
            u_top.u_soc.u_cpu.u_regfile.regs[14] = lr_value;
        end
    endtask

    // =========================================================
    // Helper Tasks
    // =========================================================

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

    task ila_write(input [2:0] addr, input [`MMIO_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_din  = data;
            ila_we   = 1;
            @(posedge clk);
            ila_we   = 0;
            ila_din  = {`MMIO_DATA_WIDTH{1'b0}};
            @(posedge clk);
        end
    endtask

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

    task read_probe(input [4:0] sel, output [`MMIO_DATA_WIDTH-1:0] data);
        begin
            ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, sel});
            @(posedge clk);
            ila_read(ILA_ADDR_PROBE, data);
        end
    endtask

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

    task mmio_wr;
        input [`MMIO_ADDR_WIDTH-1:0] addr;
        input [`MMIO_DATA_WIDTH-1:0] data;
        reg ok;
        begin
            submit_txn(1'b1, addr, data);
            wait_txn_done(ok);
        end
    endtask

    task mmio_rd;
        input  [`MMIO_ADDR_WIDTH-1:0]  addr;
        output [`MMIO_DATA_WIDTH-1:0]  rdata;
        reg ok;
        begin
            submit_txn(1'b0, addr, {`MMIO_DATA_WIDTH{1'b0}});
            wait_txn_done(ok);
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

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h1A});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h1A, "PROBE_SEL readback = 0x1A");

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h00});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h00, "PROBE_SEL readback = 0x00");

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h1F});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h1F, "PROBE_SEL readback = 0x1F (max)");

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h00});

        ila_read(ILA_ADDR_CYCLE, rd_data);
        repeat (10) @(posedge clk);
        ila_read(ILA_ADDR_CYCLE, rd_data2);
        check(rd_data2 > rd_data, "Cycle counter is incrementing");
        $display("  [INFO] Cycle delta = %0d over ~10 cycles", rd_data2 - rd_data);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        ila_read(ILA_ADDR_CTRL, rd_data);
        check(rd_data[CTRL_START] === 1'b1, "CTRL readback: cpu_run_level=1");
        check(rd_data[CTRL_STEP] === 1'b0, "CTRL readback: step_pulse auto-cleared");
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "STATUS confirms cpu_run_level=1");
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b0, "cpu_run_level=0 after clearing");

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

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Clock STOPPED in debug mode");
        check(rd_data[STAT_RUNNING] === 1'b0, "run_mode=0");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Clock RUNNING after RUN command");
        check(rd_data[STAT_STOPPED] === 1'b0, "stopped=0 when running");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Re-stopped after second STOP");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 4: ILA Step — Single-Cycle Verification via PC Probe
        //         PC is byte-addressed and increments by 4 per instruction.
        // =============================================================
        begin_test("ILA Step — Single-Cycle Verification via PC Probe");

        debug_mode = 1;
        @(posedge clk);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 1, 0));
        repeat (10) @(posedge clk);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 1, 0));
        repeat (3) @(posedge clk);

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Clock stopped for step test");
        check(rd_data[STAT_CPU_RUN] === 1'b1, "CPU still out of reset (cpu_run_level=1)");

        read_probe(PROBE_PC, rd_data);
        $display("  [INFO] PC before step = 0x%0h (%0d)", rd_data[`PC_WIDTH-1:0], rd_data[`PC_WIDTH-1:0]);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(1, 0, 0, 1, 0));
        repeat (3) @(posedge clk);

        read_probe(PROBE_PC, rd_data2);
        $display("  [INFO] PC after 1 step = 0x%0h (%0d)", rd_data2[`PC_WIDTH-1:0], rd_data2[`PC_WIDTH-1:0]);
        check(rd_data2[`PC_WIDTH-1:0] === rd_data[`PC_WIDTH-1:0] + 4,
              "PC incremented by exactly 4 after step (byte-addressed)");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(1, 0, 0, 1, 0));
        repeat (3) @(posedge clk);

        read_probe(PROBE_PC, rd_data3);
        $display("  [INFO] PC after 2nd step = 0x%0h (%0d)", rd_data3[`PC_WIDTH-1:0], rd_data3[`PC_WIDTH-1:0]);
        check(rd_data3[`PC_WIDTH-1:0] === rd_data2[`PC_WIDTH-1:0] + 4,
              "PC incremented by exactly 4 after second step");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 5: ILA Step While Running (Should Be Ignored)
        // =============================================================
        begin_test("ILA Step While Running (No-Op)");

        debug_mode = 1;
        @(posedge clk);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Confirmed running before step attempt");

        @(posedge clk);
        ila_addr = ILA_ADDR_CTRL;
        ila_din  = ila_ctrl(1, 1, 0, 0, 0);
        ila_we   = 1;
        @(posedge clk);
        ila_we = 0;
        ila_din = 0;
        @(posedge clk);

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STEPPING] === 1'b0,
              "step_active=0 (step suppressed while running)");
        check(rd_data[STAT_RUNNING] === 1'b1,
              "run_mode still=1 after step attempt");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 6: ILA Rapid Mode Transitions
        // =============================================================
        begin_test("ILA Rapid Mode Transitions");

        debug_mode = 1;
        @(posedge clk);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Transition 1: RUN");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Transition 2: STOP");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 0, 0));
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Transition 3: RUN");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk);
        ila_write(ILA_ADDR_CTRL, ila_ctrl(1, 0, 0, 0, 0));
        repeat (3) @(posedge clk);

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "Transition 4: Stopped after STOP+STEP sequence");
        check(rd_data[STAT_STEPPING] === 1'b0, "step_active auto-cleared");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 1, 0, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        $display("  [INFO] After simultaneous RUN+STOP: running=%b, stopped=%b",
                 rd_data[STAT_RUNNING], rd_data[STAT_STOPPED]);
        check(rd_data[STAT_RUNNING] === 1'b0, "Simultaneous RUN+STOP: STOP wins (last write)");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 7: MMIO Write to DMEM
        // =============================================================
        begin_test("MMIO Write to DMEM");

        do_write(REGION_DMEM_BASE | 32'h0, {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'hDEAD},
                 "MMIO write to DMEM[word 0] completed");
        check(status === 32'hAAAA_AAAA, "Status = 0xAAAA_AAAA (integrity pass)");

        do_write(REGION_DMEM_BASE | 32'h4, {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'hBEEF},
                 "MMIO write to DMEM[word 4] completed");
        check(status === 32'hAAAA_AAAA, "Status = 0xAAAA_AAAA for DMEM[4]");

        // =============================================================
        // Test 8: MMIO Read from DMEM
        // =============================================================
        begin_test("MMIO Read from DMEM");

        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "MMIO read from DMEM[word 0] completed");
        $display("  [INFO] DMEM[0] read data = 0x%0h", rd_data);
        check(rd_data[15:0] === 16'hDEAD, "DMEM[0] = 0xDEAD");
        check(status === 32'hAAAA_AAAA, "Status = integrity pass for read");

        do_read(REGION_DMEM_BASE | 32'h4, rd_data, "MMIO read from DMEM[word 4] completed");
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

                do_write(REGION_DMEM_BASE | (32'h8 + bit_idx[`MMIO_ADDR_WIDTH-1:0]),
                         pattern, "Walking-1 write");

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

        do_write(REGION_DMEM_BASE | 32'h30, {`MMIO_DATA_WIDTH{1'b1}}, "Write all-ones");
        do_read(REGION_DMEM_BASE | 32'h30, rd_data, "Read all-ones");
        check(rd_data === {`MMIO_DATA_WIDTH{1'b1}}, "All-ones pattern verified");

        do_write(REGION_DMEM_BASE | 32'h31, {`MMIO_DATA_WIDTH{1'b0}}, "Write all-zeros");
        do_read(REGION_DMEM_BASE | 32'h31, rd_data, "Read all-zeros");
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "All-zeros pattern verified");

        // =============================================================
        // Test 10: MMIO Write to IMEM — Load ARM Micro-Program
        //
        //  Word 0: MOV R2, #20       (E3A02014)
        //  Word 1: MOV R3, #20       (E3A03014)
        //  Word 2: NOP — MOV R0,R0   (E1A00000)
        //  Word 3: NOP
        //  Word 4: NOP
        //  Word 5: STR R3, [R2, #0]  (E5823000)  → DMEM[word 5] ← 20
        //  Word 6: NOP
        //  Word 7: B .               (EAFFFFFE)  halt at byte 0x1C
        // =============================================================
        begin_test("MMIO Write to IMEM — Load ARM Micro-Program");

        do_write(REGION_IMEM_BASE | 32'h0,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_MOV_R2_20},
                 "IMEM[0] = MOV R2, #20");

        do_write(REGION_IMEM_BASE | 32'h1,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_MOV_R3_20},
                 "IMEM[1] = MOV R3, #20");

        do_write(REGION_IMEM_BASE | 32'h2,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_NOP},
                 "IMEM[2] = NOP");

        do_write(REGION_IMEM_BASE | 32'h3,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_NOP},
                 "IMEM[3] = NOP");

        do_write(REGION_IMEM_BASE | 32'h4,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_NOP},
                 "IMEM[4] = NOP");

        do_write(REGION_IMEM_BASE | 32'h5,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_STR_R3_R2},
                 "IMEM[5] = STR R3, [R2, #0]");

        do_write(REGION_IMEM_BASE | 32'h6,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_NOP},
                 "IMEM[6] = NOP");

        do_write(REGION_IMEM_BASE | 32'h7,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, ARM_BRANCH_SELF},
                 "IMEM[7] = B . (halt)");

        // =============================================================
        // Test 11: MMIO Read from IMEM — Verify Loaded Program
        // =============================================================
        begin_test("MMIO Read from IMEM — Verify Loaded ARM Program");

        do_read(REGION_IMEM_BASE | 32'h0, rd_data, "Read IMEM[0]");
        check(rd_data[`INSTR_WIDTH-1:0] === ARM_MOV_R2_20[`INSTR_WIDTH-1:0],
              "IMEM[0] = MOV R2, #20 (0xE3A02014)");

        do_read(REGION_IMEM_BASE | 32'h1, rd_data, "Read IMEM[1]");
        check(rd_data[`INSTR_WIDTH-1:0] === ARM_MOV_R3_20[`INSTR_WIDTH-1:0],
              "IMEM[1] = MOV R3, #20 (0xE3A03014)");

        do_read(REGION_IMEM_BASE | 32'h2, rd_data, "Read IMEM[2]");
        check(rd_data[`INSTR_WIDTH-1:0] === ARM_NOP[`INSTR_WIDTH-1:0],
              "IMEM[2] = NOP (0xE1A00000)");

        do_read(REGION_IMEM_BASE | 32'h5, rd_data, "Read IMEM[5]");
        check(rd_data[`INSTR_WIDTH-1:0] === ARM_STR_R3_R2[`INSTR_WIDTH-1:0],
              "IMEM[5] = STR R3,[R2] (0xE5823000)");

        do_read(REGION_IMEM_BASE | 32'h7, rd_data, "Read IMEM[7]");
        check(rd_data[`INSTR_WIDTH-1:0] === ARM_BRANCH_SELF[`INSTR_WIDTH-1:0],
              "IMEM[7] = B . (0xEAFFFFFE)");

        // =============================================================
        // Test 12: MMIO Read from CTRL Region
        // =============================================================
        begin_test("MMIO Read from CTRL — system_active Flag");

        do_read(REGION_CTRL_BASE, rd_data, "Read CTRL region");
        $display("  [INFO] CTRL read data = 0x%0h", rd_data);
        check(rd_data[0] === 1'b0, "system_active=0 (CPU not started yet)");

        // =============================================================
        // Test 13: CPU End-to-End Execution (ARM Micro-Program)
        //
        // The micro-program (loaded in Test 10):
        //   MOV R2, #20       → R2 = 20
        //   MOV R3, #20       → R3 = 20
        //   NOP × 3
        //   STR R3, [R2, #0]  → DMEM[byte 20 >> 2 = word 5] = 20
        //   NOP
        //   B .               → halt at byte 0x1C
        //
        // Verification:
        //   DMEM[word 5] changes from canary (0xDEADBEEF) to 20.
        //   DMEM[word 0] remains 0xDEAD (not corrupted by CPU).
        // =============================================================
        begin_test("CPU End-to-End Execution (ARM)");

        // Preload canary at DMEM[word 5] to prove the CPU overwrites it
        do_write(REGION_DMEM_BASE | 32'h5,
                 {{(`MMIO_DATA_WIDTH-32){1'b0}}, 32'hDEAD_BEEF},
                 "Preload DMEM[word 5] = 0xDEADBEEF (canary)");

        do_read(REGION_DMEM_BASE | 32'h5, rd_data, "Verify canary");
        check(rd_data[31:0] === 32'hDEAD_BEEF, "DMEM[word 5] canary confirmed");

        // Init CPU registers (hierarchical, while CPU is in reset)
        $display("  Initialising CPU registers...");
        init_cpu_regs({`DATA_WIDTH{1'b0}});

        $display("  Starting CPU via ILA cpu_run_level...");
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "cpu_run_level=1, CPU starting");

        // Wait for program to execute (8 instructions, ~20 pipeline cycles + margin)
        $display("  Waiting for CPU to complete (~200 cycles)...");
        repeat (200) @(posedge clk);

        // Read PC via ILA before stopping — should be in B . range
        read_probe(PROBE_PC, rd_data);
        $display("  [INFO] PC during halt loop = 0x%0h", rd_data[`PC_WIDTH-1:0]);
        check(rd_data[`PC_WIDTH-1:0] >= TEST_HALT_BYTE[`PC_WIDTH-1:0] &&
              rd_data[`PC_WIDTH-1:0] <= TEST_HALT_BYTE[`PC_WIDTH-1:0] + 8,
              "PC is in B . oscillation range (0x1C..0x24)");

        // Stop CPU (cpu_run_level=0 → cpu_rst_n=0 → CPU resets, BRAMs preserved)
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b0, "cpu_run_level=0 after stopping CPU");

        // Read DMEM[word 5] — should be 20 (CPU stored R3=20 at byte addr 20)
        do_read(REGION_DMEM_BASE | 32'h5, rd_data, "Read DMEM[word 5] after CPU execution");
        $display("  [INFO] DMEM[word 5] after CPU = 0x%0h", rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === `DATA_WIDTH'd20,
              "DMEM[word 5] = 20 (CPU stored R3=20 at byte addr 20, 20>>2=word 5)");

        // DMEM[word 0] should be untouched (still 0xDEAD from Test 7)
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Read DMEM[word 0] after CPU execution");
        check(rd_data[15:0] === 16'hDEAD, "DMEM[word 0] still = 0xDEAD (not corrupted)");

        // =============================================================
        // Test 14: CPU Register Readback via ILA Probes (Post-Execution)
        //          Register file survives cpu_rst_n de-assertion.
        // =============================================================
        begin_test("CPU Register Readback via ILA Probes (ARM)");

        read_probe(PROBE_REG_BASE | 4'd0, rd_data);
        $display("  [INFO] R0 = 0x%0h", rd_data);
        check(rd_data === {`MMIO_DATA_WIDTH{1'b0}}, "R0 = 0 (never written by program)");

        read_probe(PROBE_REG_BASE | 4'd2, rd_data);
        $display("  [INFO] R2 = 0x%0h", rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === `DATA_WIDTH'd20,
              "R2 = 20 (MOV R2, #20)");

        read_probe(PROBE_REG_BASE | 4'd3, rd_data);
        $display("  [INFO] R3 = 0x%0h", rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === `DATA_WIDTH'd20,
              "R3 = 20 (MOV R3, #20)");

        read_probe(PROBE_REG_BASE | 4'd1, rd_data);
        $display("  [INFO] R1 = 0x%0h", rd_data);
        check(rd_data !== {`MMIO_DATA_WIDTH{1'bx}}, "R1 probe is not X");

        read_probe(PROBE_PC, rd_data);
        $display("  [INFO] PC (after CPU stopped) = %0d", rd_data[`PC_WIDTH-1:0]);
        check(rd_data[`PC_WIDTH-1:0] === {`PC_WIDTH{1'b0}},
              "PC = 0 (CPU in reset after stopping)");

        // =============================================================
        // Test 15: Debug Mode — MMIO with Clock Stopped (txn_pending)
        //          DMEM[0] = 0x0000DEAD (set in Test 7, untouched by CPU)
        // =============================================================
        begin_test("Debug Mode — MMIO Transactions with Clock Gated");

        debug_mode = 1;
        @(posedge clk);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_STOPPED] === 1'b1, "SoC clock stopped");

        $display("  Attempting MMIO read while debug clock is stopped...");
        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "MMIO read DMEM[word 0] while debug-stopped");
        check(rd_data[15:0] === 16'hDEAD, "DMEM[word 0] = 0xDEAD (txn_pending kept clock alive)");

        do_write(REGION_DMEM_BASE | 32'h7,
                 {{(`MMIO_DATA_WIDTH-8){1'b0}}, 8'h42},
                 "MMIO write DMEM[word 7]=0x42 while debug-stopped");

        do_read(REGION_DMEM_BASE | 32'h7, rd_data, "MMIO read DMEM[word 7] while debug-stopped");
        check(rd_data[7:0] === 8'h42, "DMEM[word 7] = 0x42 (write through gated clock)");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 16: txn_pending Detailed Verification
        //          DMEM[0] = 0x0000DEAD (unchanged)
        // =============================================================
        begin_test("txn_pending Detailed — Observing FIFO & Clock Interaction");

        debug_mode = 1;
        @(posedge clk);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 1, 0, 0));
        repeat (3) @(posedge clk);

        @(posedge clk);
        while (!user_ready) @(posedge clk);
        user_cmd   = 1'b0;
        user_addr  = REGION_DMEM_BASE | 32'h0;
        user_wdata = {`MMIO_DATA_WIDTH{1'b0}};
        user_valid = 1;
        @(posedge clk);
        user_valid = 0;

        @(posedge clk);
        #1;
        check(conn_status[CONN_FIFO_NOT_EMPTY] === 1'b1 || conn_status[CONN_BUSY] === 1'b1,
              "FIFO_NOT_EMPTY or BUSY after submit (txn is in flight)");

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
        check(user_rdata[15:0] === 16'hDEAD,
              "Correct data returned through txn_pending path (DMEM[0]=0xDEAD)");

        check(status === 32'hAAAA_AAAA,
              "Status = 0xAAAA_AAAA (txn completed, not timed out)");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 17: ILA Probe Readback — Multi-Probe Sweep
        // =============================================================
        begin_test("ILA Probe Readback — Multi-Probe Channel Sweep");

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

        read_probe(5'h0F, rd_data);
        $display("  [INFO] Probe 0x0F (default) = 0x%016h", rd_data);
        check(rd_data[`DATA_WIDTH-1:0] === {`DATA_WIDTH{1'b1}},
              "Default probe selector returns all 1s");

        // =============================================================
        // Test 18: SoC Driver State Observation (BUSY During Transaction)
        // =============================================================
        begin_test("SoC Driver State Observation — BUSY Flag");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        repeat (5) @(posedge clk);

        @(posedge clk);
        while (!user_ready) @(posedge clk);
        user_cmd   = 1'b0;
        user_addr  = REGION_DMEM_BASE;
        user_wdata = {`MMIO_DATA_WIDTH{1'b0}};
        user_valid = 1;
        @(posedge clk);
        user_valid = 0;

        repeat (5) @(posedge clk);
        #1;
        check(conn_status[CONN_BUSY] === 1'b1,
              "Driver BUSY=1 while waiting for req_rdy");
        check(conn_status[CONN_FIFO_NOT_EMPTY] === 1'b0,
              "FIFO empty (transaction dequeued into FSM)");

        $display("  [INFO] conn_status = 0x%02h (driver stuck in SEND_REQ)", conn_status);

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

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // =============================================================
        // Test 19: Transaction Quality — Max Latency
        // =============================================================
        begin_test("Transaction Quality — Max Latency Tracking");

        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

        check(txn_quality[31:16] === 16'd0, "max_latency=0 after clear");

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

        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Read for latency tracking");
        begin : max_lat2_block
            reg [15:0] prev_max, new_max;
            prev_max = txn_quality[31:16];
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

        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

        check(txn_counters[15:0] === 16'd0, "total=0 after clear");
        check(txn_counters[31:16] === 16'd0, "reads=0 after clear");
        check(txn_counters[47:32] === 16'd0, "writes=0 after clear");

        do_write(REGION_DMEM_BASE | 32'h0, 64'h1, "Counted write 1");
        do_write(REGION_DMEM_BASE | 32'h1, 64'h2, "Counted write 2");
        do_write(REGION_DMEM_BASE | 32'h2, 64'h3, "Counted write 3");

        @(posedge clk); @(posedge clk);
        $display("  [INFO] After 3 writes: total=%0d reads=%0d writes=%0d",
                 txn_counters[15:0], txn_counters[31:16], txn_counters[47:32]);
        check(txn_counters[15:0] === 16'd3, "total=3 after 3 writes");
        check(txn_counters[31:16] === 16'd0, "reads=0 after 3 writes");
        check(txn_counters[47:32] === 16'd3, "writes=3 after 3 writes");

        do_read(REGION_DMEM_BASE | 32'h0, rd_data, "Counted read 1");
        do_read(REGION_DMEM_BASE | 32'h1, rd_data, "Counted read 2");

        @(posedge clk); @(posedge clk);
        $display("  [INFO] After +2 reads: total=%0d reads=%0d writes=%0d",
                 txn_counters[15:0], txn_counters[31:16], txn_counters[47:32]);
        check(txn_counters[15:0] === 16'd5, "total=5 after 3W+2R");
        check(txn_counters[31:16] === 16'd2, "reads=2 after 3W+2R");
        check(txn_counters[47:32] === 16'd3, "writes=3 after 3W+2R");

        check((txn_counters[31:16] + txn_counters[47:32]) === txn_counters[15:0],
              "reads + writes == total");

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

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        repeat (5) @(posedge clk);

        begin : fifo_full_block
            integer i;
            integer submitted;
            reg saw_not_ready;
            submitted = 0;
            saw_not_ready = 0;

            for (i = 0; i < 20; i = i + 1) begin
                @(posedge clk);
                #1;
                if (!user_ready) begin
                    saw_not_ready = 1;
                    $display("  [INFO] user_ready=0 at submission %0d (FIFO full)", i);
                    user_valid = 0;
                    i = 20;
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
                check(submitted >= 2, "At least 2 transactions accepted before full");
            end else begin
                check(1'b0, "Expected user_ready to go low but it never did");
            end
        end

        $display("  Waiting for FIFO to drain (transactions will timeout)...");
        begin : fifo_full_drain
            integer tw;
            tw = 0;
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

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // =============================================================
        // Test 24: Timeout Scenario — Request Phase
        // =============================================================
        begin_test("Timeout — Request Phase (CPU Running, req_rdy Blocked)");

        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

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
              "txn_timed_out cleared after CHECK->IDLE transition");

        $display("  [INFO] txn_quality = 0x%0h", txn_quality);
        $display("  [INFO] timeout_count = %0d", txn_quality[47:32]);
        check(txn_quality[47:32] >= 16'd1, "timeout_count >= 1");
        check(txn_quality[0] === 1'b1, "txn_quality req_timeout flag set");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

        // =============================================================
        // Test 25: Multiple Timeouts & Timeout Counter
        // =============================================================
        begin_test("Multiple Timeouts & Timeout Counter Verification");

        @(posedge clk);
        clear_stats = 1;
        @(posedge clk);
        clear_stats = 0;
        repeat (2) @(posedge clk);

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0));
        repeat (5) @(posedge clk);

        $display("  Submitting 3 transactions that will timeout...");
        begin : multi_timeout
            integer i;
            for (i = 0; i < 3; i = i + 1) begin
                submit_txn(1'b0, REGION_DMEM_BASE | i[`MMIO_ADDR_WIDTH-1:0],
                           {`MMIO_DATA_WIDTH{1'b0}});
            end
        end

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
        check(txn_counters[15:0] === 16'd0,
              "total_txn_count = 0 (timeouts are not counted as successful)");
        check(conn_status[CONN_REQ_TIMEOUT] === 1'b1, "req_timeout flag still latched");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0));
        repeat (5) @(posedge clk);

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

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 1, 0, 1, 0));
        @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b1, "Pre-clear: run_mode=1");
        check(rd_data[STAT_CPU_RUN] === 1'b1, "Pre-clear: cpu_run_level=1");

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h0A});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h0A, "Pre-clear: probe_sel=0x0A");

        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 1));
        @(posedge clk); @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_RUNNING] === 1'b0, "Post-clear: run_mode=0");
        check(rd_data[STAT_CPU_RUN] === 1'b0, "Post-clear: cpu_run_level=0");
        check(rd_data[STAT_STEPPING] === 1'b0, "Post-clear: step_active=0");
        check(rd_data[STAT_STOPPED] === 1'b1, "Post-clear: stopped=1 (debug mode)");

        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        $display("  [INFO] probe_sel after clear = 0x%0h", rd_data[4:0]);
        check(rd_data[4:0] === 5'h0A, "probe_sel preserved after CLEAR");

        debug_mode = 0;
        @(posedge clk);

        // =============================================================
        // Test 27: Post-Clear Full System Sanity Check
        // =============================================================
        begin_test("Post-Clear Full System Sanity Check");

        do_write(REGION_DMEM_BASE | 32'h3F,
                 {{(`MMIO_DATA_WIDTH-16){1'b0}}, 16'h1234},
                 "Final sanity: write DMEM[63]");
        do_read(REGION_DMEM_BASE | 32'h3F, rd_data, "Final sanity: read DMEM[63]");
        check(rd_data[15:0] === 16'h1234, "DMEM[63] = 0x1234 (write/read works)");
        check(status === 32'hAAAA_AAAA, "Status = integrity pass");

        ila_write(ILA_ADDR_PROBE_SEL, {{(`MMIO_DATA_WIDTH-5){1'b0}}, 5'h05});
        ila_read(ILA_ADDR_PROBE_SEL, rd_data);
        check(rd_data[4:0] === 5'h05, "ILA PROBE_SEL still writable");

        ila_read(ILA_ADDR_CYCLE, rd_data);
        repeat (5) @(posedge clk);
        ila_read(ILA_ADDR_CYCLE, rd_data2);
        check(rd_data2 > rd_data, "Cycle counter still incrementing");

        check(conn_status[CONN_PROTOCOL_ERR] === 1'b0,
              "No protocol errors throughout entire test suite");

        check(conn_status[CONN_BUSY] === 1'b0, "Driver idle at end of test");
        check(conn_status[CONN_FIFO_NOT_EMPTY] === 1'b0, "FIFO empty at end of test");
        check(user_ready === 1'b1, "user_ready=1 at end of test");

        // =============================================================
        // Test 28: Bubble Sort via SoC Integration
        //
        //  Flow (mirrors the working soc_tb approach):
        //    1. Full system reset for clean state
        //    2. Load hex file into local array, patch B . at 0x200
        //    3. Write full image to IMEM and DMEM via soc_driver MMIO
        //    4. Spot-check loaded words
        //    5. Init CPU registers (hierarchical), set LR = 0x200
        //    6. Start CPU via ILA (cpu_run_level=1 → soc_start=1)
        //    7. Poll PC via ILA probes — detect B . halt range
        //    8. Stop CPU via ILA (cpu_run_level=0 → CPU resets, BRAMs ok)
        //    9. Read DMEM via MMIO, search for sorted array
        //   10. Verify sort order + expected values
        // =============================================================
        begin_test("Bubble Sort via SoC Integration");

        // ── 28a: System reset for clean state ──
        $display("  Resetting system for sort test...");
        rst_n      = 0;
        debug_mode = 0;
        ila_we     = 0;
        user_valid = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (10) @(posedge clk);

        #1;
        check(user_ready === 1'b1, "Driver ready after sort-test reset");

        // ── 28b: Initialize expected sorted output ──
        // Input:  {323, 123, -455, 2, 98, 125, 10, 65, -56, 0}
        // Sorted: {-455, -56, 0, 2, 10, 65, 98, 123, 125, 323}
        sort_expected[0] = 32'hfffffe39;  // -455
        sort_expected[1] = 32'hffffffc8;  //  -56
        sort_expected[2] = 32'h00000000;  //    0
        sort_expected[3] = 32'h00000002;  //    2
        sort_expected[4] = 32'h0000000a;  //   10
        sort_expected[5] = 32'h00000041;  //   65
        sort_expected[6] = 32'h00000062;  //   98
        sort_expected[7] = 32'h0000007b;  //  123
        sort_expected[8] = 32'h0000007d;  //  125
        sort_expected[9] = 32'h00000143;  //  323

        // ── 28c: Load hex file into local array ──
        begin : sort_load_block
            integer si;
            for (si = 0; si < SORT_MEM_DEPTH; si = si + 1)
                sort_local_mem[si] = {`DATA_WIDTH{1'b0}};

            $readmemh(SORT_HEX_FILE, sort_local_mem);

            if (sort_local_mem[0] === {`DATA_WIDTH{1'b0}} &&
                sort_local_mem[1] === {`DATA_WIDTH{1'b0}}) begin
                $display("  *** WARNING: Hex file appears empty: %0s ***", SORT_HEX_FILE);
                $display("  *** Skipping bubble sort test ***");
                check(1'b0, "Hex file loaded successfully");
            end else begin
                $display("  Loaded hex file: %0s", SORT_HEX_FILE);
            end
        end

        // ── 28d: Find last non-zero word, place halt instruction ──
        begin : sort_find_extent
            integer si;
            sort_last_nz = 0;
            for (si = 0; si < SORT_MEM_DEPTH; si = si + 1) begin
                if (sort_local_mem[si] !== {`DATA_WIDTH{1'b0}})
                    sort_last_nz = si;
            end
        end

        // Place B . at word 128 (byte 0x200) where BX LR lands
        sort_local_mem[SORT_HALT_WORD] = SORT_HALT_INSTR;
        if (SORT_HALT_WORD > sort_last_nz)
            sort_last_nz = SORT_HALT_WORD;

        $display("  Program extent: word 0..%0d  (%0d words)", sort_last_nz, sort_last_nz + 1);
        $display("  First 4 words : %08H %08H %08H %08H",
                 sort_local_mem[0], sort_local_mem[1],
                 sort_local_mem[2], sort_local_mem[3]);
        $display("  Halt (B .)    : word %0d (byte 0x%04H) = 0x%08H",
                 SORT_HALT_WORD, SORT_HALT_WORD << 2, SORT_HALT_INSTR);

        // ── 28e: Write program to IMEM via MMIO ──
        $display("  Loading %0d words into IMEM via MMIO...", sort_last_nz + 1);
        begin : sort_write_imem
            integer si, cnt;
            cnt = 0;
            for (si = 0; si <= sort_last_nz; si = si + 1) begin
                mmio_wr(REGION_IMEM_BASE | si[`MMIO_ADDR_WIDTH-1:0],
                        {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, sort_local_mem[si]});
                cnt = cnt + 1;
                if (cnt % 50 == 0)
                    $display("    ... IMEM %0d / %0d", cnt, sort_last_nz + 1);
            end
            $display("    IMEM: %0d words written", cnt);
        end

        // ── 28f: Write program+data to DMEM via MMIO ──
        $display("  Loading %0d words into DMEM via MMIO...", sort_last_nz + 1);
        begin : sort_write_dmem
            integer si, cnt;
            cnt = 0;
            for (si = 0; si <= sort_last_nz; si = si + 1) begin
                mmio_wr(REGION_DMEM_BASE | si[`MMIO_ADDR_WIDTH-1:0],
                        {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, sort_local_mem[si]});
                cnt = cnt + 1;
                if (cnt % 50 == 0)
                    $display("    ... DMEM %0d / %0d", cnt, sort_last_nz + 1);
            end
            $display("    DMEM: %0d words written", cnt);
        end
        check(sort_last_nz > 10, "Program loaded to IMEM and DMEM");

        // ── 28g: Spot-check loaded words ──
        begin : sort_spotcheck
            reg [`MMIO_DATA_WIDTH-1:0] sc;
            mmio_rd(REGION_IMEM_BASE | 32'h0, sc);
            check(sc[`DATA_WIDTH-1:0] === sort_local_mem[0],
                  "IMEM[0] readback matches hex file");
            $display("  Spot-check IMEM[0] = 0x%08H (expected 0x%08H)",
                     sc[`DATA_WIDTH-1:0], sort_local_mem[0]);

            mmio_rd(REGION_DMEM_BASE | 32'h0, sc);
            check(sc[`DATA_WIDTH-1:0] === sort_local_mem[0],
                  "DMEM[0] readback matches hex file");

            mmio_rd(REGION_IMEM_BASE | SORT_HALT_WORD[`MMIO_ADDR_WIDTH-1:0], sc);
            check(sc[`DATA_WIDTH-1:0] === SORT_HALT_INSTR,
                  "IMEM halt word readback matches B . instruction");

            mmio_rd(REGION_DMEM_BASE | SORT_HALT_WORD[`MMIO_ADDR_WIDTH-1:0], sc);
            check(sc[`DATA_WIDTH-1:0] === SORT_HALT_INSTR,
                  "DMEM halt word readback matches B . instruction");
        end

        // ── 28h: Init CPU registers (hierarchical) ──
        //  The CPU is in reset (cpu_run_level=0 → cpu_rst_n=0).
        //  Register file is a plain reg array — no reset.
        //  We force-init via hierarchical access.
        //  LR is preset to SORT_HALT_BYTE_ADDR as safety (the sort
        //  bootstrap overwrites it with MOV LR, #0x200 anyway).
        $display("  Initialising CPU registers...");
        init_cpu_regs(SORT_HALT_BYTE_ADDR);
        @(posedge clk); #1;
        init_cpu_regs(SORT_HALT_BYTE_ADDR);   // re-init for safety

        // ── 28i: Start CPU via ILA ──
        $display("");
        $display("  ████ Starting CPU — Bubble Sort Execution ████");
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 1, 0)); // cpu_run_level=1
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b1, "cpu_run_level=1 (CPU started)");

        // ── 28j: Poll PC via ILA probes until CPU halts or timeout ──
        //  The B . at byte 0x200 makes PC oscillate among
        //  {0x200, 0x204, 0x208} due to branch delay slots.
        //  We detect halt when PC falls within this range for
        //  consecutive polls (with a minimum cycle threshold to
        //  avoid false positives during early execution).
        $display("  Waiting for CPU to complete (timeout=%0d cycles)...", SORT_TIMEOUT);
        sort_cpu_halted = 1'b0;
        begin : sort_poll_loop
            integer cyc, halt_cnt;
            reg [`MMIO_DATA_WIDTH-1:0] cur_pc;
            cyc      = 0;
            halt_cnt = 0;

            while (cyc < SORT_TIMEOUT) begin
                repeat (SORT_POLL_INTERVAL) @(posedge clk);
                cyc = cyc + SORT_POLL_INTERVAL;

                read_probe(PROBE_PC, cur_pc);

                // Check if PC is in the B . oscillation range
                if (cur_pc[`PC_WIDTH-1:0] >= SORT_HALT_BYTE_ADDR[`PC_WIDTH-1:0] &&
                    cur_pc[`PC_WIDTH-1:0] <  SORT_HALT_BYTE_ADDR[`PC_WIDTH-1:0] + 16 &&
                    cyc > 2000) begin
                    halt_cnt = halt_cnt + 1;
                    if (halt_cnt >= 2) begin
                        $display("  CPU halted: PC=0x%08H (in B . range, confirmed at cycle %0d)",
                                 cur_pc, cyc);
                        sort_cpu_halted = 1'b1;
                        cyc = SORT_TIMEOUT; // break
                    end
                end else begin
                    halt_cnt = 0;
                end

                if (!sort_cpu_halted && cyc < SORT_TIMEOUT && cyc % 20000 == 0)
                    $display("    [%6d cycles] PC = 0x%08H", cyc, cur_pc);
            end

            if (!sort_cpu_halted)
                $display("  *** TIMEOUT: CPU did not halt within %0d cycles ***", SORT_TIMEOUT);
        end

        check(sort_cpu_halted, "CPU halted after sort execution");

        // ── 28k: Stop CPU ──
        ila_write(ILA_ADDR_CTRL, ila_ctrl(0, 0, 0, 0, 0)); // cpu_run_level=0
        repeat (10) @(posedge clk);
        ila_read(ILA_ADDR_STATUS, rd_data);
        check(rd_data[STAT_CPU_RUN] === 1'b0, "cpu_run_level=0 (CPU stopped)");

        // Wait for MMIO to become available (system_active → 0 → req_rdy → 1)
        begin : sort_wait_rdy
            integer tw;
            tw = 0;
            while (!user_ready && tw < 100) begin
                @(posedge clk); #1;
                tw = tw + 1;
            end
            check(user_ready === 1'b1, "MMIO available after CPU stop");
        end

        // ── 28l: Read DMEM snapshot via MMIO ──
        $display("  Reading %0d DMEM words via MMIO...", SORT_SCAN_RANGE);
        begin : sort_readback
            integer si;
            for (si = 0; si < SORT_SCAN_RANGE; si = si + 1) begin
                mmio_rd(REGION_DMEM_BASE | si[`MMIO_ADDR_WIDTH-1:0],
                        sort_dmem_snap[si]);
                if (si % 100 == 99)
                    $display("    ... read %0d / %0d", si + 1, SORT_SCAN_RANGE);
            end
            $display("    DMEM readback complete (%0d words)", SORT_SCAN_RANGE);
        end

        // ── 28m: Search DMEM for sorted array ──
        $display("  Searching DMEM for expected sorted sequence...");
        sort_array_found = 1'b0;
        sort_found_at    = 0;
        begin : sort_search
            integer sw, se;
            reg match_flag;
            for (sw = 0; sw < SORT_SCAN_RANGE - SORT_ARRAY_SIZE + 1; sw = sw + 1) begin
                if (!sort_array_found) begin
                    match_flag = 1'b1;
                    for (se = 0; se < SORT_ARRAY_SIZE; se = se + 1) begin
                        if (sort_dmem_snap[sw + se][`DATA_WIDTH-1:0] !== sort_expected[se])
                            match_flag = 1'b0;
                    end
                    if (match_flag) begin
                        sort_array_found = 1'b1;
                        sort_found_at    = sw;
                    end
                end
            end
        end

        if (sort_array_found) begin
            $display("  Sorted array found at DMEM word %0d (byte 0x%04H):",
                     sort_found_at, sort_found_at << 2);
            begin : sort_display
                integer se;
                for (se = 0; se < SORT_ARRAY_SIZE; se = se + 1) begin
                    $display("    [%0d] = 0x%08H  (%0d)", se,
                             sort_dmem_snap[sort_found_at + se][`DATA_WIDTH-1:0],
                             $signed(sort_dmem_snap[sort_found_at + se][`DATA_WIDTH-1:0]));
                end
            end
        end else begin
            $display("  Sorted array NOT found in DMEM scan range (0..%0d)",
                     SORT_SCAN_RANGE - 1);
            $display("  Non-zero DMEM words (first 64):");
            begin : sort_dump
                integer sw, printed;
                printed = 0;
                for (sw = 0; sw < SORT_SCAN_RANGE; sw = sw + 1) begin
                    if (sort_dmem_snap[sw] !== {`MMIO_DATA_WIDTH{1'b0}} && printed < 64) begin
                        $display("    [%3d] = 0x%08H  (%0d)", sw,
                                 sort_dmem_snap[sw][`DATA_WIDTH-1:0],
                                 $signed(sort_dmem_snap[sw][`DATA_WIDTH-1:0]));
                        printed = printed + 1;
                    end
                end
            end
        end
        check(sort_array_found, "Sorted array found in DMEM");

        // ── 28n: Verify sort order (ascending, signed) ──
        if (sort_array_found) begin
            $display("  Verifying ascending sort order...");
            begin : sort_order_check
                integer si, errs;
                reg [`DATA_WIDTH-1:0] cur_val, nxt_val;
                errs = 0;
                for (si = 0; si < SORT_ARRAY_SIZE - 1; si = si + 1) begin
                    cur_val = sort_dmem_snap[sort_found_at + si][`DATA_WIDTH-1:0];
                    nxt_val = sort_dmem_snap[sort_found_at + si + 1][`DATA_WIDTH-1:0];
                    if ($signed(cur_val) > $signed(nxt_val)) begin
                        $display("    ORDER ERROR: [%0d]=%0d > [%0d]=%0d",
                                 si, $signed(cur_val), si+1, $signed(nxt_val));
                        errs = errs + 1;
                    end
                end
                check(errs === 0, "All elements in ascending signed order");
            end
        end

        // ── 28o: Verify against expected values ──
        if (sort_array_found) begin
            $display("  Comparing each element against expected...");
            begin : sort_expected_check
                integer si, errs;
                reg [`DATA_WIDTH-1:0] val;
                errs = 0;
                for (si = 0; si < SORT_ARRAY_SIZE; si = si + 1) begin
                    val = sort_dmem_snap[sort_found_at + si][`DATA_WIDTH-1:0];
                    if (val !== sort_expected[si]) begin
                        $display("    [FAIL] arr[%0d] = %0d, expected %0d",
                                 si, $signed(val), $signed(sort_expected[si]));
                        errs = errs + 1;
                    end else begin
                        $display("    [PASS] arr[%0d] = %0d", si, $signed(val));
                    end
                end
                check(errs === 0, "All sorted elements match expected values");
            end
        end

        // ── 28p: Read CPU registers via ILA probes (post-sort) ──
        $display("  Post-sort CPU register snapshot:");
        begin : sort_reg_dump
            integer ri;
            reg [`MMIO_DATA_WIDTH-1:0] rv;
            for (ri = 0; ri < (1 << `REG_ADDR_WIDTH); ri = ri + 1) begin
                read_probe(5'h10 | ri[4:0], rv);
                $display("    R%-2d = 0x%08H  (%0d)", ri,
                         rv[`DATA_WIDTH-1:0], $signed(rv[`DATA_WIDTH-1:0]));
            end
        end

        $display("");
        $display("  ████ Bubble Sort Test Complete ████");

        // =============================================================
        // Summary
        // =============================================================
        $display("\n################################################################");
        $display("###                  TEST SUMMARY                           ###");
        $display("################################################################");
        $display("  Total Tests:  %0d", test_num);
        $display("  Total PASS:   %0d", pass_count);
        $display("  Total FAIL:   %0d", fail_count);
        if ((pass_count + fail_count) > 0)
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
    // Timeout Watchdog (sufficient for sort loading + execution)
    // =========================================================
    initial begin
        #50_000_000; // 50ms absolute timeout
        $display("\n[TIMEOUT] Simulation exceeded 50ms — aborting.");
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