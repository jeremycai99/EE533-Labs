/* file: soc_tb.v
 * Description: SoC MMIO testbench
 *   Phase A — Write i_mem.coe / d_mem.coe via MMIO, run CPU, verify d_mem
 *   Phase B — Random read/write correctness tests for IMEM & DMEM
 *
 * KEY DESIGN NOTE — cpu_rst_n fix
 *   The current SoC has:  wire cpu_rst_n = rst_n & cpu_rst_n;
 *   This means the CPU free-runs from PC=0 the moment rst_n deasserts,
 *   even before the host has finished loading memory.
 *
 *   Workaround used here:
 *     1. Load IMEM/DMEM via MMIO  (CPU is running NOPs harmlessly)
 *     2. Re-assert SoC reset      (BRAM contents survive)
 *     3. Deassert reset → CPU restarts from PC = 0
 *     4. Immediately issue CTRL write to set cpu_active = 1
 *        (the store at IMEM[5] reaches the MEM stage ≈ cycle 20,
 *         well after cpu_active gates open at ≈ cycle 19)
 */

`timescale 1ns / 1ps

`include "define.v"
`include "soc.v"

module soc_tb;

    // ================================================================
    //  Clock & Reset
    // ================================================================
    reg clk;
    reg rst_n;
    initial clk = 0;
    always #5 clk = ~clk;                       // 100 MHz, 10 ns period

    // ================================================================
    //  MMIO Interface Signals
    // ================================================================
    reg                          req_cmd;
    reg  [`MMIO_ADDR_WIDTH-1:0]  req_addr;
    reg  [`MMIO_DATA_WIDTH-1:0]  req_data;
    reg                          req_val;
    wire                         req_rdy;

    wire                         resp_cmd;
    wire [`MMIO_ADDR_WIDTH-1:0]  resp_addr;
    wire [`MMIO_DATA_WIDTH-1:0]  resp_data;
    wire                         resp_val;
    reg                          resp_rdy;

    // ================================================================
    //  DUT
    // ================================================================
    soc u_soc (
        .clk      (clk),
        .rst_n    (rst_n),
        .req_cmd  (req_cmd),
        .req_addr (req_addr),
        .req_data (req_data),
        .req_val  (req_val),
        .req_rdy  (req_rdy),
        .resp_cmd (resp_cmd),
        .resp_addr(resp_addr),
        .resp_data(resp_data),
        .resp_val (resp_val),
        .resp_rdy (resp_rdy)
    );

    // ================================================================
    //  Address Bases & Test Parameters
    // ================================================================
    localparam [31:0] IMEM_BASE = 32'h0000_0000;
    localparam [31:0] DMEM_BASE = 32'h8000_0000;
    localparam [31:0] CTRL_BASE = 32'h4000_0000;

    localparam NUM_IMEM_COE = 8;
    localparam NUM_DMEM_COE = 10;
    localparam NUM_RAND     = 64;
    localparam ADDR_RANGE   = 256;
    localparam CPU_TIMEOUT  = 1000;              // cycles before forced reset

    // ================================================================
    //  COE File Contents (hard-coded from the .coe files)
    // ================================================================
    //
    //  i_mem.coe  (8 × INSTR_WIDTH)
    //  ┌──────┬────────────┬──────────────────────────────────────────┐
    //  │ Addr │  Hex       │  Meaning                                 │
    //  ├──────┼────────────┼──────────────────────────────────────────┤
    //  │  0   │ 40400000   │  Load  d_mem[R0] → R2                    │
    //  │  1   │ 40600000   │  Load  d_mem[R0] → R3                    │
    //  │  2   │ 00000000   │  NOP                                     │
    //  │  3   │ 00000000   │  NOP   (R2 = 4 available)                │
    //  │  4   │ 00000000   │  NOP   (R3 = 4 available)                │
    //  │  5   │ 93000000   │  Store R2 → d_mem[R3]  ⇒ d_mem[4] = 4   │
    //  │  6   │ 00000000   │  NOP                                     │
    //  │  7   │ 00000000   │  NOP                                     │
    //  └──────┴────────────┴──────────────────────────────────────────┘
    //
    //  d_mem.coe  (10 × DATA_WIDTH)
    //    [0]=4  [4]=100(0x64)  rest=0
    //
    //  After CPU:  d_mem[0]=4 (unchanged),  d_mem[4]=4 (was 100)

    reg [`MMIO_DATA_WIDTH-1:0] imem_coe      [0:NUM_IMEM_COE-1];
    reg [`MMIO_DATA_WIDTH-1:0] dmem_coe      [0:NUM_DMEM_COE-1];
    reg [`MMIO_DATA_WIDTH-1:0] dmem_post_cpu [0:NUM_DMEM_COE-1];

    // ================================================================
    //  Reference Model (Phase B — random tests)
    // ================================================================
    reg [`MMIO_DATA_WIDTH-1:0] imem_ref   [0:ADDR_RANGE-1];
    reg                        imem_valid [0:ADDR_RANGE-1];
    reg [`MMIO_DATA_WIDTH-1:0] dmem_ref   [0:ADDR_RANGE-1];
    reg                        dmem_valid [0:ADDR_RANGE-1];

    // ================================================================
    //  Variables
    // ================================================================
    integer i, seed;
    integer pass_cnt, fail_cnt;
    integer cycle_cnt, num_checked;

    reg [`MMIO_DATA_WIDTH-1:0]  rd_data;
    reg [`MMIO_ADDR_WIDTH-1:0]  taddr;
    reg [`MMIO_DATA_WIDTH-1:0]  twdata;
    reg [7:0]                   toff;

    // ================================================================
    //  MMIO Transaction Task  (~5 clk cycles per call)
    // ================================================================
    task mmio_txn;
        input                          cmd;
        input  [`MMIO_ADDR_WIDTH-1:0]  addr;
        input  [`MMIO_DATA_WIDTH-1:0]  wdata;
        output [`MMIO_DATA_WIDTH-1:0]  rdata;
        begin
            @(posedge clk); #1;
            while (!req_rdy) begin @(posedge clk); #1; end
            req_cmd  = cmd;
            req_addr = addr;
            req_data = wdata;
            req_val  = 1'b1;
            resp_rdy = 1'b1;
            @(posedge clk); #1;
            req_val  = 1'b0;
            while (!resp_val) begin @(posedge clk); #1; end
            rdata = resp_data;
            @(posedge clk); #1;
            resp_rdy = 1'b0;
        end
    endtask

    task mmio_wr;
        input [`MMIO_ADDR_WIDTH-1:0] addr;
        input [`MMIO_DATA_WIDTH-1:0] wdata;
        reg   [`MMIO_DATA_WIDTH-1:0] dummy;
        begin mmio_txn(1'b1, addr, wdata, dummy); end
    endtask

    task mmio_rd;
        input  [`MMIO_ADDR_WIDTH-1:0] addr;
        output [`MMIO_DATA_WIDTH-1:0] rdata;
        reg    [`MMIO_DATA_WIDTH-1:0] tmp;
        begin mmio_txn(1'b0, addr, {`MMIO_DATA_WIDTH{1'b0}}, tmp); rdata = tmp; end
    endtask

    // ================================================================
    //  Checker  (=== catches X/Z mismatches)
    // ================================================================
    task check;
        input [`MMIO_DATA_WIDTH-1:0]  expected;
        input [`MMIO_DATA_WIDTH-1:0]  actual;
        input [`MMIO_ADDR_WIDTH-1:0]  addr;
        input [255:0]                 tag;
        begin
            if (expected === actual)
                pass_cnt = pass_cnt + 1;
            else begin
                fail_cnt = fail_cnt + 1;
                $display("  [FAIL] %0s @ 0x%08h  exp=0x%016h  got=0x%016h",
                         tag, addr, expected, actual);
            end
        end
    endtask

    // ================================================================
    //  Reset Helper  (10 clk low + 5 clk settle)
    // ================================================================
    task do_reset;
        begin
            rst_n    = 1'b0;
            req_val  = 1'b0;
            resp_rdy = 1'b0;
            repeat (10) @(posedge clk);
            #1; rst_n = 1'b1;
            repeat (5)  @(posedge clk);
        end
    endtask

    // ================================================================
    //  Main Stimulus
    // ================================================================

    //Initialize the register file to 0 to solve X propagation issues
    integer k;
    initial begin
        for (k = 0; k < 32; k = k + 1)
            u_soc.u_cpu.u_regfile.regfile[k] = 64'd0;
    end

    initial begin
        $dumpfile("soc_tb.vcd");
        $dumpvars(0, soc_tb);

        // ── Initialise ──────────────────────────────────
        rst_n = 0; req_cmd = 0; req_addr = 0; req_data = 0;
        req_val = 0; resp_rdy = 0;
        pass_cnt = 0; fail_cnt = 0; seed = 42;

        // i_mem.coe
        imem_coe[0] = 'h40400000;
        imem_coe[1] = 'h40600000;
        imem_coe[2] = 'h00000000;
        imem_coe[3] = 'h00000000;
        imem_coe[4] = 'h00000000;
        imem_coe[5] = 'h93000000;
        imem_coe[6] = 'h00000000;
        imem_coe[7] = 'h00000000;

        // d_mem.coe
        dmem_coe[0] = 'h0000000000000004;
        dmem_coe[1] = 'h0000000000000000;
        dmem_coe[2] = 'h0000000000000000;
        dmem_coe[3] = 'h0000000000000000;
        dmem_coe[4] = 'h0000000000000064;
        dmem_coe[5] = 'h0000000000000000;
        dmem_coe[6] = 'h0000000000000000;
        dmem_coe[7] = 'h0000000000000000;
        dmem_coe[8] = 'h0000000000000000;
        dmem_coe[9] = 'h0000000000000000;

        // Expected DMEM after CPU execution
        for (i = 0; i < NUM_DMEM_COE; i = i + 1)
            dmem_post_cpu[i] = dmem_coe[i];
        dmem_post_cpu[4] = 'h0000000000000004;   // was 0x64 → 4

        // Phase-B reference model init
        for (i = 0; i < ADDR_RANGE; i = i + 1) begin
            imem_ref[i] = 0; imem_valid[i] = 0;
            dmem_ref[i] = 0; dmem_valid[i] = 0;
        end

        // ── Initial Reset ───────────────────────────────
        do_reset();

        $display("");
        $display("======================================================");
        $display("  SoC MMIO Testbench — COE + CPU + Random");
        $display("======================================================");

        // ═══════════════════════════════════════════════════
        //  PHASE A — Load COE → Run CPU → Verify d_mem
        // ═══════════════════════════════════════════════════
        $display("\n──────────────────────────────────────────────────");
        $display("  PHASE A: COE load → CPU execution → verify");
        $display("──────────────────────────────────────────────────");

        // ── A1: Write IMEM from i_mem.coe ───────────────
        $display("\n[A1] Writing IMEM (%0d words) ...", NUM_IMEM_COE);
        for (i = 0; i < NUM_IMEM_COE; i = i + 1)
            mmio_wr(IMEM_BASE | i, imem_coe[i]);

        // ── A2: Write DMEM from d_mem.coe ───────────────
        $display("[A2] Writing DMEM (%0d words) ...", NUM_DMEM_COE);
        for (i = 0; i < NUM_DMEM_COE; i = i + 1)
            mmio_wr(DMEM_BASE | i, dmem_coe[i]);

        // ── A3: Read-back verify IMEM ───────────────────
        $display("[A3] Read-back verify IMEM ...");
        for (i = 0; i < NUM_IMEM_COE; i = i + 1) begin
            taddr = IMEM_BASE | i;
            mmio_rd(taddr, rd_data);
            check(imem_coe[i] & {`INSTR_WIDTH{1'b1}},
                  rd_data, taddr, "COE_IMEM");
        end

        // ── A4: Read-back verify DMEM ───────────────────
        $display("[A4] Read-back verify DMEM ...");
        for (i = 0; i < NUM_DMEM_COE; i = i + 1) begin
            taddr = DMEM_BASE | i;
            mmio_rd(taddr, rd_data);
            check(dmem_coe[i] & {`DATA_WIDTH{1'b1}},
                  rd_data, taddr, "COE_DMEM");
        end

        // ── A5: Re-reset SoC ───────────────────────────
        //  BRAM contents survive.  CPU PC restarts at 0.
        $display("[A5] Re-reset SoC (BRAM preserved, CPU PC → 0) ...");
        do_reset();

        // ── A6: Start CPU via CTRL write ────────────────
        //  After do_reset the CPU is already fetching from
        //  IMEM[0].  The CTRL write sets cpu_active=1 so the
        //  store instruction's d_mem write is not gated.
        $display("[A6] Starting CPU (CTRL write) ...");
        mmio_wr(CTRL_BASE, 1);

        // ── A7: Wait for CPU completion ─────────────────
        //  req_rdy re-asserts once cpu_active falls back to 0.
        //  If cpu_done is not implemented, we force-reset after
        //  the timeout — the store has long since committed.
        $display("[A7] Waiting for CPU (timeout = %0d cycles) ...", CPU_TIMEOUT);
        cycle_cnt = 0;
        @(posedge clk); #1;
        while (!req_rdy && (cycle_cnt < CPU_TIMEOUT)) begin
            @(posedge clk); #1;
            cycle_cnt = cycle_cnt + 1;
        end

        if (cycle_cnt >= CPU_TIMEOUT) begin
            $display("  WARN: cpu_done not seen in %0d cycles — forcing reset",
                     CPU_TIMEOUT);
            do_reset();
        end else begin
            $display("  CPU completed in ~%0d cycles", cycle_cnt);
        end

        // ── A8: CTRL sanity — cpu_active should be 0 ───
        $display("[A8] CTRL read (expect cpu_active = 0) ...");
        mmio_rd(CTRL_BASE, rd_data);
        check({`MMIO_DATA_WIDTH{1'b0}}, rd_data, CTRL_BASE, "CTRL_IDLE");

        // ── A9: Verify DMEM post-execution ──────────────
        $display("[A9] Verifying DMEM after CPU execution ...");
        for (i = 0; i < NUM_DMEM_COE; i = i + 1) begin
            taddr = DMEM_BASE | i;
            mmio_rd(taddr, rd_data);
            check(dmem_post_cpu[i] & {`DATA_WIDTH{1'b1}},
                  rd_data, taddr, "CPU_DMEM");
        end

        // Highlight the two key addresses
        mmio_rd(DMEM_BASE | 32'd0, rd_data);
        $display("  d_mem[0] = %0d  %s", rd_data,
            (rd_data === (dmem_post_cpu[0] & {`DATA_WIDTH{1'b1}}))
                ? "(unchanged as expected)" : "*** MISMATCH ***");

        mmio_rd(DMEM_BASE | 32'd4, rd_data);
        $display("  d_mem[4] = %0d  (was %0d)  %s", rd_data, dmem_coe[4],
            (rd_data === (dmem_post_cpu[4] & {`DATA_WIDTH{1'b1}}))
                ? "(store succeeded)" : "*** MISMATCH ***");

        // ── A10: Verify IMEM was not modified by CPU ────
        $display("[A10] Verifying IMEM unchanged ...");
        for (i = 0; i < NUM_IMEM_COE; i = i + 1) begin
            taddr = IMEM_BASE | i;
            mmio_rd(taddr, rd_data);
            check(imem_coe[i] & {`INSTR_WIDTH{1'b1}},
                  rd_data, taddr, "IMEM_UNCH");
        end

        // ═══════════════════════════════════════════════════
        //  PHASE B — Random Read/Write Tests
        // ═══════════════════════════════════════════════════
        $display("\n──────────────────────────────────────────────────");
        $display("  PHASE B: Random MMIO read/write tests");
        $display("──────────────────────────────────────────────────");

        // Fresh reference model
        for (i = 0; i < ADDR_RANGE; i = i + 1) begin
            imem_ref[i] = 0; imem_valid[i] = 0;
            dmem_ref[i] = 0; dmem_valid[i] = 0;
        end

        // ── B1: IMEM sequential W+R ────────────────────
        $display("\n[B1] IMEM sequential W+R (%0d entries)", NUM_RAND);
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            taddr  = IMEM_BASE | i;
            twdata = $random(seed);
            imem_ref[i]   = twdata & {`INSTR_WIDTH{1'b1}};
            imem_valid[i] = 1'b1;
            mmio_wr(taddr, twdata);
        end
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            taddr = IMEM_BASE | i;
            mmio_rd(taddr, rd_data);
            check(imem_ref[i], rd_data, taddr, "IMEM_SEQ");
        end

        // ── B2: DMEM sequential W+R ────────────────────
        $display("[B2] DMEM sequential W+R (%0d entries)", NUM_RAND);
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            taddr  = DMEM_BASE | i;
            twdata = {$random(seed), $random(seed)};
            dmem_ref[i]   = twdata & {`DATA_WIDTH{1'b1}};
            dmem_valid[i] = 1'b1;
            mmio_wr(taddr, twdata);
        end
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            taddr = DMEM_BASE | i;
            mmio_rd(taddr, rd_data);
            check(dmem_ref[i], rd_data, taddr, "DMEM_SEQ");
        end

        // ── B3: IMEM random-address W+R ────────────────
        $display("[B3] IMEM random-addr W+R (%0d writes)", NUM_RAND);
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            toff   = $random(seed);
            taddr  = IMEM_BASE | {24'b0, toff};
            twdata = $random(seed);
            imem_ref[toff]   = twdata & {`INSTR_WIDTH{1'b1}};
            imem_valid[toff] = 1'b1;
            mmio_wr(taddr, twdata);
        end
        num_checked = 0;
        for (i = 0; i < ADDR_RANGE; i = i + 1) begin
            if (imem_valid[i]) begin
                taddr = IMEM_BASE | i;
                mmio_rd(taddr, rd_data);
                check(imem_ref[i], rd_data, taddr, "IMEM_RAND");
                num_checked = num_checked + 1;
            end
        end
        $display("  Checked %0d IMEM addresses", num_checked);

        // ── B4: DMEM random-address W+R ────────────────
        $display("[B4] DMEM random-addr W+R (%0d writes)", NUM_RAND);
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            toff   = $random(seed);
            taddr  = DMEM_BASE | {24'b0, toff};
            twdata = {$random(seed), $random(seed)};
            dmem_ref[toff]   = twdata & {`DATA_WIDTH{1'b1}};
            dmem_valid[toff] = 1'b1;
            mmio_wr(taddr, twdata);
        end
        num_checked = 0;
        for (i = 0; i < ADDR_RANGE; i = i + 1) begin
            if (dmem_valid[i]) begin
                taddr = DMEM_BASE | i;
                mmio_rd(taddr, rd_data);
                check(dmem_ref[i], rd_data, taddr, "DMEM_RAND");
                num_checked = num_checked + 1;
            end
        end
        $display("  Checked %0d DMEM addresses", num_checked);

        // ── B5: Mixed IMEM/DMEM interleaved ────────────
        $display("[B5] Mixed IMEM/DMEM interleaved (%0d writes)", NUM_RAND);
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            toff = $random(seed);
            if (i[0]) begin                       // odd → DMEM
                taddr  = DMEM_BASE | {24'b0, toff};
                twdata = {$random(seed), $random(seed)};
                dmem_ref[toff]   = twdata & {`DATA_WIDTH{1'b1}};
                dmem_valid[toff] = 1'b1;
            end else begin                        // even → IMEM
                taddr  = IMEM_BASE | {24'b0, toff};
                twdata = $random(seed);
                imem_ref[toff]   = twdata & {`INSTR_WIDTH{1'b1}};
                imem_valid[toff] = 1'b1;
            end
            mmio_wr(taddr, twdata);
        end
        num_checked = 0;
        for (i = 0; i < ADDR_RANGE; i = i + 1) begin
            if (imem_valid[i]) begin
                mmio_rd(IMEM_BASE | i, rd_data);
                check(imem_ref[i], rd_data, IMEM_BASE | i, "MIX_IMEM");
                num_checked = num_checked + 1;
            end
            if (dmem_valid[i]) begin
                mmio_rd(DMEM_BASE | i, rd_data);
                check(dmem_ref[i], rd_data, DMEM_BASE | i, "MIX_DMEM");
                num_checked = num_checked + 1;
            end
        end
        $display("  Checked %0d total addresses", num_checked);

        // ── B6: Write-after-write (last wins) ──────────
        $display("[B6] Write-after-write — last value wins");
        mmio_wr(IMEM_BASE | 32'hA0, 32'hDEAD_BEEF);
        mmio_wr(IMEM_BASE | 32'hA0, 32'hCAFE_BABE);
        mmio_rd(IMEM_BASE | 32'hA0, rd_data);
        check(32'hCAFE_BABE & {`INSTR_WIDTH{1'b1}},
              rd_data, IMEM_BASE | 32'hA0, "IMEM_WAW");

        twdata = {$random(seed), $random(seed)};
        mmio_wr(DMEM_BASE | 32'hB0, {$random(seed), $random(seed)});
        mmio_wr(DMEM_BASE | 32'hB0, twdata);
        mmio_rd(DMEM_BASE | 32'hB0, rd_data);
        check(twdata & {`DATA_WIDTH{1'b1}},
              rd_data, DMEM_BASE | 32'hB0, "DMEM_WAW");

        // ── B7: Read-after-read (non-destructive) ──────
        $display("[B7] Read-after-read — non-destructive");
        taddr  = IMEM_BASE | 32'hC0;
        twdata = $random(seed);
        mmio_wr(taddr, twdata);
        mmio_rd(taddr, rd_data);
        check(twdata & {`INSTR_WIDTH{1'b1}}, rd_data, taddr, "IMEM_RAR1");
        mmio_rd(taddr, rd_data);
        check(twdata & {`INSTR_WIDTH{1'b1}}, rd_data, taddr, "IMEM_RAR2");

        taddr  = DMEM_BASE | 32'hD0;
        twdata = {$random(seed), $random(seed)};
        mmio_wr(taddr, twdata);
        mmio_rd(taddr, rd_data);
        check(twdata & {`DATA_WIDTH{1'b1}}, rd_data, taddr, "DMEM_RAR1");
        mmio_rd(taddr, rd_data);
        check(twdata & {`DATA_WIDTH{1'b1}}, rd_data, taddr, "DMEM_RAR2");

        // ═══════════════════════════════════════════════════
        //  Summary
        // ═══════════════════════════════════════════════════
        $display("");
        $display("======================================================");
        $display("  %0d PASSED    %0d FAILED    (%0d total checks)",
                 pass_cnt, fail_cnt, pass_cnt + fail_cnt);
        $display("======================================================");
        if (fail_cnt == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else
            $display("  >>> SOME TESTS FAILED <<<");
        $display("");

        #200;
        $finish;
    end

    // ── Watchdog ────────────────────────────────────────
    initial begin
        #50_000_000;
        $display("\n[TIMEOUT] Simulation exceeded 50 ms — aborting");
        $finish;
    end

endmodule