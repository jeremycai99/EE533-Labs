/* file: soc_tb.v
 * Description: SoC MMIO testbench
 *   Phase A — Basic MMIO read/write infrastructure tests
 *   Phase B — Bubble Sort integration (hex load via MMIO → run via ext start → verify via MMIO)
 *
 *  The bubble sort program uses a unified memory image. Both IMEM and DMEM
 *  are loaded with the same hex file (code fetched from IMEM, data from DMEM).
 *  The CPU is started via the external `start` pin (not MMIO CTRL) because
 *  the sort program halts with B. (branch-to-self), which does not trigger
 *  `cpu_done`.  PC is monitored through hierarchical access.  After halt
 *  detection, `start` is de-asserted (CPU resets, BRAMs preserved), and DMEM
 *  is read back via MMIO to verify the sorted output.
 *
 *  v2 — Updated for cpu.v v2.4 / bdtu.v v1.5:
 *    - bdtu_state_name widened to 4-bit input (BDTU v1.5 states 0–9)
 *    - Added S_DRAIN (8) and S_MEM_DRAIN (9) state names
 *    - Fixed stall_ex (nonexistent) → stall_ex1 / stall_ex2
 *  v3 — Updated for cpu.v v2.5 (8-stage pipeline):
 *    - pc_if → pc_if1 (IF1 stage PC)
 *    - stall_if → stall_if1 (+ stall_if2)
 *    - Added stall_ex3 to diagnostic displays
 */

`timescale 1ns / 1ps

`include "define.v"
`include "soc.v"

module soc_tb;

    // ================================================================
    //  Parameters
    // ================================================================
    parameter CLK_PERIOD       = 10;
    parameter SORT_ARRAY_SIZE  = 10;
    parameter MAX_SORT_ELEMS   = 64;
    parameter LOAD_EXTENT      = 1024;   // min words to load into IMEM & DMEM

    parameter [31:0] HALT_ADDR  = 32'h0000_0200;
    parameter [31:0] HALT_INSTR = 32'hEAFF_FFFE;  // B . (branch-to-self)
    parameter [31:0] HALT_WORD  = HALT_ADDR >> 2;  // word index 128

    // Hardware BRAM depths — derived from define.v (moved before SORT_MEM_DEPTH)
    localparam IMEM_HW_DEPTH = (1 << `IMEM_ADDR_WIDTH);
    localparam DMEM_HW_DEPTH = (1 << `DMEM_ADDR_WIDTH);

    // Local memory depth must cover the larger of the two BRAMs (min 4096)
    localparam SORT_MEM_DEPTH_MIN = (DMEM_HW_DEPTH > IMEM_HW_DEPTH) ? DMEM_HW_DEPTH : IMEM_HW_DEPTH;
    localparam SORT_MEM_DEPTH     = (SORT_MEM_DEPTH_MIN > 4096) ? SORT_MEM_DEPTH_MIN : 4096;

`ifdef SIM_TIMEOUT
    parameter TIMEOUT = `SIM_TIMEOUT;
`else
    parameter TIMEOUT = 200_000;
`endif

`ifdef MEM_FILE
    parameter FILE_NAME = `MEM_FILE;
`else
    parameter FILE_NAME = "../hex/sort_imem.txt";
`endif

    parameter NUM_RAND         = 64;
    parameter ADDR_RANGE       = 256;
    parameter STATUS_INTERVAL  = 1000;

    // ================================================================
    //  Clock & Reset
    // ================================================================
    reg clk;
    reg rst_n;
    reg start;

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

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
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .req_cmd      (req_cmd),
        .req_addr     (req_addr),
        .req_data     (req_data),
        .req_val      (req_val),
        .req_rdy      (req_rdy),
        .resp_cmd     (resp_cmd),
        .resp_addr    (resp_addr),
        .resp_data    (resp_data),
        .resp_val     (resp_val),
        .resp_rdy     (resp_rdy)
    );

    // ================================================================
    //  Address Bases
    // ================================================================
    localparam [31:0] IMEM_BASE = 32'h0000_0000;
    localparam [31:0] DMEM_BASE = 32'h8000_0000;
    localparam [31:0] CTRL_BASE = 32'h4000_0000;

    // ================================================================
    //  Local Storage
    // ================================================================
    reg [`DATA_WIDTH-1:0]       sort_local_mem  [0:SORT_MEM_DEPTH-1];
    reg [`DATA_WIDTH-1:0]       expected_sorted [0:SORT_ARRAY_SIZE-1];
    reg [`DATA_WIDTH-1:0]       sorted_result   [0:MAX_SORT_ELEMS-1];
    reg [`DATA_WIDTH-1:0]       dmem_snap       [0:SORT_MEM_DEPTH-1];
    integer sort_count;
    integer load_extent_actual;
    integer imem_load_extent;     // always IMEM_HW_DEPTH
    integer dmem_load_extent;     // always DMEM_HW_DEPTH

    // Phase A reference model
    reg [`MMIO_DATA_WIDTH-1:0]  imem_ref   [0:ADDR_RANGE-1];
    reg                         imem_valid [0:ADDR_RANGE-1];
    reg [`MMIO_DATA_WIDTH-1:0]  dmem_ref   [0:ADDR_RANGE-1];
    reg                         dmem_valid [0:ADDR_RANGE-1];

    // ================================================================
    //  Variables
    // ================================================================
    integer i, j, k, seed;
    integer pass_cnt, fail_cnt;
    integer cycle_cnt, num_checked;
    integer stuck_cnt;
    integer run_exit_status;      // 0=halt 1=stuck 2=timeout
    integer total_sort_cycles;
    integer dmem_write_cnt;       // total DMEM writes during execution
    integer dmem_read_cnt;        // total DMEM reads (non-BDTU STR loads)
    integer found_addr;
    reg     found, match;
    reg [`PC_WIDTH-1:0] prev_pc;
    reg [`PC_WIDTH-1:0] cur_pc;

    reg [`MMIO_DATA_WIDTH-1:0]  rd_data;
    reg [`MMIO_ADDR_WIDTH-1:0]  taddr;
    reg [`MMIO_DATA_WIDTH-1:0]  twdata;
    reg [7:0]                   toff;

    // ================================================================
    //  State/Name Decoder — BDTU v1.5 (4-bit state encoding)
    // ================================================================
    function [7*8:1] bdtu_state_name;
        input [3:0] st;
        case (st)
            4'd0: bdtu_state_name = "IDLE   ";
            4'd1: bdtu_state_name = "BDT_XFR";
            4'd2: bdtu_state_name = "BDT_LST";
            4'd3: bdtu_state_name = "BDT_WB ";
            4'd4: bdtu_state_name = "SWP_RD ";
            4'd5: bdtu_state_name = "SWP_WAT";
            4'd6: bdtu_state_name = "SWP_WR ";
            4'd7: bdtu_state_name = "DONE   ";
            4'd8: bdtu_state_name = "DRAIN  ";
            4'd9: bdtu_state_name = "MEMDRAI";
            default: bdtu_state_name = "???    ";
        endcase
    endfunction

    // ================================================================
    //  MMIO Transaction Tasks
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
    //  Checker
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
                $display("  [FAIL] %0s @ 0x%0h  exp=0x%08h  got=0x%08h",
                         tag, addr, expected, actual);
            end
        end
    endtask

    // ================================================================
    //  Reset Helper
    // ================================================================
    task do_reset;
        begin
            rst_n    = 1'b0;
            start    = 1'b0;
            req_val  = 1'b0;
            resp_rdy = 1'b0;
            repeat (10) @(posedge clk);
            #1; rst_n = 1'b1;
            repeat (5) @(posedge clk);
        end
    endtask

    // ================================================================
    //  CPU Register Initialisation (hierarchical — simulation only)
    // ================================================================
    task init_cpu_regs;
        integer r;
        begin
            for (r = 0; r < `REG_DEPTH; r = r + 1)
                u_soc.u_cpu.u_regfile.regs[r] = {`DATA_WIDTH{1'b0}};
            // Safety: pre-set LR to halt address.
            u_soc.u_cpu.u_regfile.regs[14] = HALT_ADDR;
        end
    endtask

    // ================================================================
    //  CPU Register Dump (hierarchical)
    // ================================================================
    task dump_regs;
        integer r;
        begin
            $display("  +-----------------------------------------+");
            $display("  |          Register File Dump             |");
            $display("  +-----------------------------------------+");
            for (r = 0; r < `REG_DEPTH; r = r + 1)
                $display("  |  R%-2d  = 0x%08H  (%0d)", r,
                         u_soc.u_cpu.u_regfile.regs[r],
                         $signed(u_soc.u_cpu.u_regfile.regs[r]));
            $display("  +-----------------------------------------+");
            $display("  |  CPSR  = %04b (N=%b Z=%b C=%b V=%b)    |",
                     u_soc.u_cpu.cpsr_flags,
                     u_soc.u_cpu.cpsr_flags[`FLAG_N],
                     u_soc.u_cpu.cpsr_flags[`FLAG_Z],
                     u_soc.u_cpu.cpsr_flags[`FLAG_C],
                     u_soc.u_cpu.cpsr_flags[`FLAG_V]);
            $display("  +-----------------------------------------+");
        end
    endtask

    // ================================================================
    //  Load Hex File into Local Array
    // ================================================================
    task load_hex_file;
        begin
            for (k = 0; k < SORT_MEM_DEPTH; k = k + 1)
                sort_local_mem[k] = {`DATA_WIDTH{1'b0}};

`ifdef BIN_MODE
            $readmemb(FILE_NAME, sort_local_mem);
            $display("  Loaded (BINARY): %0s", FILE_NAME);
`else
            $readmemh(FILE_NAME, sort_local_mem);
            $display("  Loaded (HEX):    %0s", FILE_NAME);
`endif

            $display("  First 8 words:");
            for (k = 0; k < 8; k = k + 1)
                $display("    [0x%04H] = 0x%08H", k << 2, sort_local_mem[k]);

            if (sort_local_mem[0] === {`DATA_WIDTH{1'b0}} &&
                sort_local_mem[1] === {`DATA_WIDTH{1'b0}})
                $display("  *** WARNING: Memory looks empty — check file path. ***");

            // Place halt instruction at return address
            $display("  Placing halt (B . = 0x%08H) at word %0d (byte 0x%08H)",
                     HALT_INSTR, HALT_WORD, HALT_ADDR);
            sort_local_mem[HALT_WORD] = HALT_INSTR;

            // Determine actual extent (last non-zero word + margin) — diagnostic only
            load_extent_actual = 0;
            for (k = 0; k < SORT_MEM_DEPTH; k = k + 1)
                if (sort_local_mem[k] !== {`DATA_WIDTH{1'b0}})
                    load_extent_actual = k + 1;
            if (load_extent_actual < LOAD_EXTENT)
                load_extent_actual = LOAD_EXTENT;
            $display("  Hex data extent: %0d words (%0d bytes)",
                     load_extent_actual, load_extent_actual << 2);

            // Always load the FULL BRAM to avoid uninitialized locations.
            // sort_local_mem[] is pre-zeroed, so addresses beyond the hex
            // data are written as 0 — eliminating X / stale-data issues
            // that arise when the BRAM is larger than the hex image.
            $display("  Hardware depths: IMEM=%0d words, DMEM=%0d words",
                     IMEM_HW_DEPTH, DMEM_HW_DEPTH);

            imem_load_extent = IMEM_HW_DEPTH;
            dmem_load_extent = DMEM_HW_DEPTH;

            $display("  Will load: IMEM=%0d words, DMEM=%0d words (full BRAM)",
                     imem_load_extent, dmem_load_extent);

            // Safety: verify halt word fits within IMEM
            if (HALT_WORD >= IMEM_HW_DEPTH) begin
                $display("  *** ERROR: HALT_WORD (%0d) >= IMEM depth (%0d) — halt unreachable via IMEM! ***",
                         HALT_WORD, IMEM_HW_DEPTH);
            end
        end
    endtask

    // ================================================================
    //  Verify Array is in Ascending Signed Order
    // ================================================================
    task verify_sort_order;
        integer idx, errors;
        reg [`DATA_WIDTH-1:0] cur, nxt;
        begin
            errors = 0;
            if (sort_count == 0) begin
                $display("  NO DATA — skipping order check");
                fail_cnt = fail_cnt + 1;
            end else begin
                $display("  Verifying sort order (%0d elements)...", sort_count);
                for (idx = 0; idx < sort_count - 1; idx = idx + 1) begin
                    cur = sorted_result[idx];
                    nxt = sorted_result[idx + 1];
                    if ($signed(cur) > $signed(nxt)) begin
                        $display("    *** ORDER ERROR at [%0d]: %0d > [%0d]: %0d",
                                 idx, $signed(cur), idx + 1, $signed(nxt));
                        errors = errors + 1;
                    end
                end
                if (errors == 0) begin
                    $display("    PASS: Correctly sorted in ascending order.");
                    pass_cnt = pass_cnt + 1;
                end else begin
                    $display("    FAIL: %0d order violation(s).", errors);
                    fail_cnt = fail_cnt + 1;
                end
            end
        end
    endtask

    // ================================================================
    //  Compare Against Known Expected Output
    // ================================================================
    task verify_against_expected;
        integer idx, errors;
        begin
            errors = 0;
            if (sort_count == 0) begin
                $display("  NO DATA — skipping expected-value check");
            end else begin
                $display("  Comparing against expected sorted array...");
                for (idx = 0; idx < sort_count && idx < SORT_ARRAY_SIZE; idx = idx + 1) begin
                    if (sorted_result[idx] !== expected_sorted[idx]) begin
                        $display("    [FAIL] arr[%0d] = %0d, expected %0d",
                                 idx, $signed(sorted_result[idx]),
                                 $signed(expected_sorted[idx]));
                        errors  = errors + 1;
                        fail_cnt = fail_cnt + 1;
                    end else begin
                        $display("    [PASS] arr[%0d] = %0d",
                                 idx, $signed(sorted_result[idx]));
                        pass_cnt = pass_cnt + 1;
                    end
                end
                if (errors == 0) $display("    ALL %0d ELEMENTS MATCH.", sort_count);
                else              $display("    %0d MISMATCH(ES).", errors);
            end
        end
    endtask

    // ================================================================
    //  Dump Non-Zero DMEM Snapshot (for debug)
    // ================================================================
    task dump_dmem_nonzero;
        integer w, printed;
        begin
            printed = 0;
            $display("  ── Non-zero DMEM words (first 64 shown) ──");
            for (w = 0; w < dmem_load_extent && printed < 64; w = w + 1) begin
                if (dmem_snap[w] !== {`DATA_WIDTH{1'b0}}) begin
                    $display("    [word %4d / 0x%04H] = 0x%08H  (%0d)",
                             w, w << 2, dmem_snap[w], $signed(dmem_snap[w]));
                    printed = printed + 1;
                end
            end
            if (printed == 0)
                $display("    (all DMEM is zero)");
        end
    endtask

    // ================================================================
    //  Zero CPU Registers at Simulation Start
    // ================================================================
    initial begin
        #1;
        for (k = 0; k < `REG_DEPTH; k = k + 1)
            u_soc.u_cpu.u_regfile.regs[k] = {`DATA_WIDTH{1'b0}};
    end

    // ================================================================
    //  Main Stimulus
    // ================================================================
    initial begin
        $dumpfile("soc_tb.vcd");
        $dumpvars(1, soc_tb);
        $dumpvars(0, u_soc);

        // ── Expected sorted output ──
        //  Input:  {323, 123, -455, 2, 98, 125, 10, 65, -56, 0}
        //  Sorted: {-455, -56, 0, 2, 10, 65, 98, 123, 125, 323}
        expected_sorted[0] = 32'hfffffe39;  // -455
        expected_sorted[1] = 32'hffffffc8;  // -56
        expected_sorted[2] = 32'h00000000;  //    0
        expected_sorted[3] = 32'h00000002;  //    2
        expected_sorted[4] = 32'h0000000a;  //   10
        expected_sorted[5] = 32'h00000041;  //   65
        expected_sorted[6] = 32'h00000062;  //   98
        expected_sorted[7] = 32'h0000007b;  //  123
        expected_sorted[8] = 32'h0000007d;  //  125
        expected_sorted[9] = 32'h00000143;  //  323

        // ── Signal init ──
        rst_n = 0; start = 0;
        req_cmd = 0; req_addr = 0; req_data = 0;
        req_val = 0; resp_rdy = 0;
        pass_cnt = 0; fail_cnt = 0; seed = 42;
        sort_count = 0; cycle_cnt = 0;
        imem_load_extent = 0; dmem_load_extent = 0;

        do_reset();

        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║     SoC MMIO Testbench — MMIO + Bubble Sort Integration    ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        $display("║  Program  : %-48s ║", FILE_NAME);
        $display("║  Halt     : 0x%08H                                     ║", HALT_ADDR);
        $display("║  CPU_DONE : 0x%08H                                     ║", `CPU_DONE_PC);
        $display("║  Timeout  : %0d cycles                                 ║", TIMEOUT);
        $display("║  IMEM     : %0d words (%0d-bit addr)                   ║",
                 IMEM_HW_DEPTH, `IMEM_ADDR_WIDTH);
        $display("║  DMEM     : %0d words (%0d-bit addr)                  ║",
                 DMEM_HW_DEPTH, `DMEM_ADDR_WIDTH);
        $display("╚══════════════════════════════════════════════════════════════╝");

        // ═══════════════════════════════════════════════════
        //  PHASE A — Basic MMIO Read/Write Infrastructure
        // ═══════════════════════════════════════════════════
        $display("\n──────────────────────────────────────────────────");
        $display("  PHASE A: Basic MMIO Infrastructure Tests");
        $display("──────────────────────────────────────────────────");

        // ── A1: CTRL read — idle ────────────────────────
        $display("\n[A1] CTRL read — expect idle (0)");
        mmio_rd(CTRL_BASE, rd_data);
        check({`MMIO_DATA_WIDTH{1'b0}}, rd_data, CTRL_BASE, "CTRL_IDLE");

        // ── A2: IMEM sequential write + readback ────────
        $display("[A2] IMEM sequential W+R (8 words)");
        for (i = 0; i < 8; i = i + 1) begin
            twdata = $random(seed);
            mmio_wr(IMEM_BASE | i, twdata);
            imem_ref[i] = twdata & {`INSTR_WIDTH{1'b1}};
        end
        for (i = 0; i < 8; i = i + 1) begin
            mmio_rd(IMEM_BASE | i, rd_data);
            check(imem_ref[i], rd_data, IMEM_BASE | i, "IMEM_SEQ");
        end

        // ── A3: DMEM sequential write + readback ────────
        $display("[A3] DMEM sequential W+R (8 words)");
        for (i = 0; i < 8; i = i + 1) begin
            twdata = {$random(seed), $random(seed)};
            mmio_wr(DMEM_BASE | i, twdata);
            dmem_ref[i] = twdata & {`DATA_WIDTH{1'b1}};
        end
        for (i = 0; i < 8; i = i + 1) begin
            mmio_rd(DMEM_BASE | i, rd_data);
            check(dmem_ref[i], rd_data, DMEM_BASE | i, "DMEM_SEQ");
        end

        // ── A4: Write-after-write (last wins) ───────────
        $display("[A4] Write-after-write — last value wins");
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

        // ── A5: Read-after-read (non-destructive) ───────
        $display("[A5] Read-after-read — non-destructive");
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

        // ── A6: Random IMEM write + readback ────────────
        $display("[A6] Random IMEM W+R (%0d entries)", NUM_RAND);
        for (i = 0; i < ADDR_RANGE; i = i + 1) begin
            imem_ref[i] = 0; imem_valid[i] = 0;
        end
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
                mmio_rd(IMEM_BASE | i, rd_data);
                check(imem_ref[i], rd_data, IMEM_BASE | i, "IMEM_RAND");
                num_checked = num_checked + 1;
            end
        end
        $display("  Checked %0d IMEM addresses", num_checked);

        // ── A7: Random DMEM write + readback ────────────
        $display("[A7] Random DMEM W+R (%0d entries)", NUM_RAND);
        for (i = 0; i < ADDR_RANGE; i = i + 1) begin
            dmem_ref[i] = 0; dmem_valid[i] = 0;
        end
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
                mmio_rd(DMEM_BASE | i, rd_data);
                check(dmem_ref[i], rd_data, DMEM_BASE | i, "DMEM_RAND");
                num_checked = num_checked + 1;
            end
        end
        $display("  Checked %0d DMEM addresses", num_checked);

        // ── A8: Mixed IMEM/DMEM interleaved ─────────────
        $display("[A8] Mixed IMEM/DMEM interleaved (%0d writes)", NUM_RAND);
        for (i = 0; i < NUM_RAND; i = i + 1) begin
            toff = $random(seed);
            if (i[0]) begin
                taddr  = DMEM_BASE | {24'b0, toff};
                twdata = {$random(seed), $random(seed)};
                dmem_ref[toff]   = twdata & {`DATA_WIDTH{1'b1}};
                dmem_valid[toff] = 1'b1;
            end else begin
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


        // ═══════════════════════════════════════════════════
        //  PHASE B — Bubble Sort via SoC
        // ═══════════════════════════════════════════════════
        $display("\n──────────────────────────────────────────────────");
        $display("  PHASE B: Bubble Sort Integration Test");
        $display("──────────────────────────────────────────────────");

        // ── B1: Load hex file into local array ──────────
        $display("\n[B1] Loading hex file...");
        load_hex_file();

        // ── B2: Reset SoC (clears Phase A BRAM noise) ──
        $display("[B2] Resetting SoC...");
        do_reset();

        // ── B3: Write program to IMEM via MMIO ─────────
        //  The hex file is a unified image (code + data).
        //  Instructions are fetched from IMEM, data from DMEM,
        //  so both get the same image.
        //  Load the FULL BRAM depth so every address is initialised
        //  (sort_local_mem is pre-zeroed beyond the hex data).
        $display("[B3] Writing %0d words to IMEM via MMIO (full BRAM depth)...",
                 imem_load_extent);
        for (i = 0; i < imem_load_extent; i = i + 1)
            mmio_wr(IMEM_BASE | i, sort_local_mem[i]);
        $display("  IMEM load complete.");

        // ── B3.5: Immediate IMEM readback sanity check ──
        $display("[B3.5] Immediate IMEM readback after load...");
        mmio_rd(IMEM_BASE | 0, rd_data);
        $display("  IMEM[0] = 0x%08H (expect 0x%08H)", rd_data,
                 sort_local_mem[0] & {`INSTR_WIDTH{1'b1}});
        if (HALT_WORD < IMEM_HW_DEPTH) begin
            mmio_rd(IMEM_BASE | HALT_WORD, rd_data);
            $display("  IMEM[%0d] = 0x%08H (expect 0x%08H = halt)",
                     HALT_WORD, rd_data, HALT_INSTR & {`INSTR_WIDTH{1'b1}});
        end else begin
            $display("  IMEM[%0d] — SKIPPED (beyond BRAM depth %0d)",
                     HALT_WORD, IMEM_HW_DEPTH);
        end

        // ── B4: Write program to DMEM via MMIO ─────────
        $display("[B4] Writing %0d words to DMEM via MMIO (full BRAM depth)...",
                 dmem_load_extent);
        for (i = 0; i < dmem_load_extent; i = i + 1)
            mmio_wr(DMEM_BASE | i, sort_local_mem[i]);
        $display("  DMEM load complete.");

        // ── B5: Readback spot-check ────────────────────
        $display("[B5] Readback spot-check...");

        // First instruction word
        mmio_rd(IMEM_BASE | 0, rd_data);
        check(sort_local_mem[0] & {`INSTR_WIDTH{1'b1}},
              rd_data, IMEM_BASE | 0, "SORT_IMEM0");
        $display("  IMEM[0]         = 0x%08H (expect 0x%08H)",
                 rd_data, sort_local_mem[0] & {`INSTR_WIDTH{1'b1}});

        // Halt instruction in IMEM (only check if within BRAM range)
        if (HALT_WORD < IMEM_HW_DEPTH) begin
            mmio_rd(IMEM_BASE | HALT_WORD, rd_data);
            check(HALT_INSTR & {`INSTR_WIDTH{1'b1}},
                  rd_data, IMEM_BASE | HALT_WORD, "SORT_HALT_I");
            $display("  IMEM[%0d]      = 0x%08H (expect 0x%08H = B .)",
                     HALT_WORD, rd_data, HALT_INSTR & {`INSTR_WIDTH{1'b1}});
        end else begin
            $display("  IMEM[%0d]      — SKIPPED (word %0d >= IMEM depth %0d)",
                     HALT_WORD, HALT_WORD, IMEM_HW_DEPTH);
            $display("  *** WARNING: Halt instruction is beyond IMEM! CPU cannot reach it. ***");
            fail_cnt = fail_cnt + 1;
        end

        // Halt instruction in DMEM
        if (HALT_WORD < DMEM_HW_DEPTH) begin
            mmio_rd(DMEM_BASE | HALT_WORD, rd_data);
            check(HALT_INSTR & {`DATA_WIDTH{1'b1}},
                  rd_data, DMEM_BASE | HALT_WORD, "SORT_HALT_D");
            $display("  DMEM[%0d]      = 0x%08H (expect 0x%08H = B .)",
                     HALT_WORD, rd_data, HALT_INSTR & {`DATA_WIDTH{1'b1}});
        end else begin
            $display("  DMEM[%0d]      — SKIPPED (word %0d >= DMEM depth %0d)",
                     HALT_WORD, HALT_WORD, DMEM_HW_DEPTH);
        end

        // ── B6: Initialise CPU registers ────────────────
        //  The CPU is in reset (start=0 → cpu_rst_n=0).
        //  Register file is a plain reg array — no reset.
        //  We force-init via hierarchical access.
        $display("[B6] Initialising CPU registers...");
        init_cpu_regs();
        @(posedge clk); #1;
        init_cpu_regs();   // re-init for safety after clock edge

        // ── B7: Assert external start pin ───────────────
        $display("[B7] Asserting external start pin...");
        @(posedge clk); #1;
        start = 1'b1;

        // Verify system is now busy
        @(posedge clk); #1;
        if (!req_rdy)
            $display("  System busy (req_rdy=0) — CPU running");
        else begin
            $display("  WARNING: req_rdy still high after start");
            fail_cnt = fail_cnt + 1;
        end

        // ── B8: Monitor PC via hierarchical access, wait for halt ──
        $display("[B8] Running CPU (timeout=%0d cycles)...", TIMEOUT);
        $display("  halt_addr = 0x%08H", HALT_ADDR);
        $display("");

        cycle_cnt       = 0;
        stuck_cnt       = 0;
        dmem_write_cnt  = 0;
        dmem_read_cnt   = 0;
        prev_pc         = {`PC_WIDTH{1'b0}};
        run_exit_status = 2;       // default: timeout

        begin : sort_run_loop
            forever begin
                @(posedge clk); #1;
                cycle_cnt = cycle_cnt + 1;
                cur_pc = u_soc.u_cpu.pc_if1;

                // ── Early trace (first 50 cycles): pipeline overview ──
                if (cycle_cnt <= 50) begin
                    $display("  C%03d PC=%08H ireg=%08H BDTU=%s busy=%b mc_mem=%b mw_mem=%b wen=%b addr=%08H wdata=%08H SP=%08H",
                             cycle_cnt,
                             u_soc.u_cpu.pc_if1,
                             u_soc.u_cpu.instr_reg_id,
                             bdtu_state_name(u_soc.u_cpu.u_bdtu.state),
                             u_soc.u_cpu.bdtu_busy,
                             u_soc.u_cpu.is_multi_cycle_mem,
                             u_soc.u_cpu.mem_write_mem,
                             u_soc.u_cpu.d_mem_wen_o,
                             u_soc.u_cpu.d_mem_addr_o,
                             u_soc.u_cpu.d_mem_data_o,
                             u_soc.u_cpu.u_regfile.regs[13]);
                    if (u_soc.u_cpu.bdtu_busy)
                        $display("       BDTU: state=%s rem=%04H addr=%08H wr=%b rd=%b wen1=%b wen2=%b waddr1=%0d waddr2=%0d",
                                 bdtu_state_name(u_soc.u_cpu.u_bdtu.state),
                                 u_soc.u_cpu.u_bdtu.remaining,
                                 u_soc.u_cpu.u_bdtu.cur_addr,
                                 u_soc.u_cpu.u_bdtu.mem_wr,
                                 u_soc.u_cpu.u_bdtu.mem_rd,
                                 u_soc.u_cpu.u_bdtu.wr_en1,
                                 u_soc.u_cpu.u_bdtu.wr_en2,
                                 u_soc.u_cpu.u_bdtu.wr_addr1,
                                 u_soc.u_cpu.u_bdtu.wr_addr2);
                    // Pipeline instruction addresses (pc_plus4 - 4)
                    $display("       pipe: ID=%08H EX1=%08H EX2=%08H EX3=%08H MEM=%08H WB=%08H",
                             u_soc.u_cpu.pc_plus4_id  - 32'd4,
                             u_soc.u_cpu.pc_plus4_ex1 - 32'd4,
                             u_soc.u_cpu.pc_plus4_ex2 - 32'd4,
                             u_soc.u_cpu.pc_plus4_ex3 - 32'd4,
                             u_soc.u_cpu.pc_plus4_mem - 32'd4,
                             u_soc.u_cpu.pc_plus4_wb  - 32'd4);
                    $display("       valid: ID=%b EX1=%b EX2=%b EX3=%b  stalls: if1=%b if2=%b id=%b ex1=%b ex2=%b ex3=%b mem=%b",
                             u_soc.u_cpu.if2id_valid,
                             u_soc.u_cpu.valid_ex1,
                             u_soc.u_cpu.valid_ex2,
                             u_soc.u_cpu.valid_ex3,
                             u_soc.u_cpu.stall_if1, u_soc.u_cpu.stall_if2,
                             u_soc.u_cpu.stall_id,
                             u_soc.u_cpu.stall_ex1, u_soc.u_cpu.stall_ex2,
                             u_soc.u_cpu.stall_ex3, u_soc.u_cpu.stall_mem);
                end

                // ── BDTU flush event ──
                if (u_soc.u_cpu.bdtu_done_flush) begin
                    $display("  >>> BDTU FLUSH C%05d: redirecting PC to %08H (pc_plus4_mem)",
                             cycle_cnt, u_soc.u_cpu.pc_plus4_mem);
                end

                // ── DMEM WRITE monitor (ALL writes, all cycles) ──
                if (u_soc.u_cpu.d_mem_wen_o) begin
                    dmem_write_cnt = dmem_write_cnt + 1;
                    $display("  WR C%05d: [%08H]->w%04d = %08H (%0d) %s  (MEM_PC=%08H)",
                             cycle_cnt,
                             u_soc.u_cpu.d_mem_addr_o,
                             u_soc.u_cpu.d_mem_addr_o >> 2,
                             u_soc.u_cpu.d_mem_data_o,
                             $signed(u_soc.u_cpu.d_mem_data_o),
                             u_soc.u_cpu.bdtu_busy ? "BDTU" : "STR ",
                             u_soc.u_cpu.pc_plus4_mem - 32'd4);
                end

                // ── DMEM READ request (MEM stage presents address for LDR) ──
                if (u_soc.u_cpu.mem_read_mem && !u_soc.u_cpu.bdtu_busy && !u_soc.u_cpu.stall_mem) begin
                    dmem_read_cnt = dmem_read_cnt + 1;
                    $display("  RD C%05d: [%08H]->w%04d  dest=R%-2d  (MEM_PC=%08H)",
                             cycle_cnt,
                             u_soc.u_cpu.d_mem_addr_o,
                             u_soc.u_cpu.d_mem_addr_o >> 2,
                             u_soc.u_cpu.wr_addr1_mem,
                             u_soc.u_cpu.pc_plus4_mem - 32'd4);
                end

                // ── Register writeback from WB stage (port 1) ──
                if (u_soc.u_cpu.wb_wr_en1) begin
                    $display("  RF C%05d: R%-2d <- %08H (%0d) [WB]  (WB_PC=%08H)",
                             cycle_cnt,
                             u_soc.u_cpu.wb_wr_addr1,
                             u_soc.u_cpu.wb_wr_data1,
                             $signed(u_soc.u_cpu.wb_wr_data1),
                             u_soc.u_cpu.pc_plus4_wb - 32'd4);
                end
                // ── Register writeback from WB stage (port 2) ──
                if (u_soc.u_cpu.wb_wr_en2) begin
                    $display("  RF C%05d: R%-2d <- %08H (%0d) [WB2] (WB_PC=%08H)",
                             cycle_cnt,
                             u_soc.u_cpu.wb_wr_addr2,
                             u_soc.u_cpu.wb_wr_data2,
                             $signed(u_soc.u_cpu.wb_wr_data2),
                             u_soc.u_cpu.pc_plus4_wb - 32'd4);
                end

                // ── Register writeback from BDTU (port 1) ──
                if (u_soc.u_cpu.bdtu_wr_en1) begin
                    $display("  RF C%05d: R%-2d <- %08H (%0d) [BDTU_P1]",
                             cycle_cnt,
                             u_soc.u_cpu.bdtu_wr_addr1,
                             u_soc.u_cpu.bdtu_wr_data1,
                             $signed(u_soc.u_cpu.bdtu_wr_data1));
                end
                // ── Register writeback from BDTU (port 2) ──
                if (u_soc.u_cpu.bdtu_wr_en2) begin
                    $display("  RF C%05d: R%-2d <- %08H (%0d) [BDTU_P2]",
                             cycle_cnt,
                             u_soc.u_cpu.bdtu_wr_addr2,
                             u_soc.u_cpu.bdtu_wr_data2,
                             $signed(u_soc.u_cpu.bdtu_wr_data2));
                end

                // ── Detection 1: PC reached halt address ──
                if (cur_pc == HALT_ADDR && cycle_cnt > 20) begin
                    $display("");
                    $display("[%0t] PC reached halt address 0x%08H at cycle %0d",
                             $time, HALT_ADDR, cycle_cnt);
                    run_exit_status = 0;
                    repeat (10) @(posedge clk);
                    disable sort_run_loop;
                end

                // ── Stuck-PC detector ──
                if (cur_pc === prev_pc)
                    stuck_cnt = stuck_cnt + 1;
                else
                    stuck_cnt = 0;
                prev_pc = cur_pc;

                // ── Detection 2: stuck at unexpected address ──
                if (stuck_cnt > 500 && prev_pc != HALT_ADDR) begin
                    $display("");
                    $display("  *** STUCK: PC=0x%08H for %0d cycles ***", prev_pc, stuck_cnt);
                    $display("  BDTU state=%s busy=%b stall_if1=%b",
                             bdtu_state_name(u_soc.u_cpu.u_bdtu.state),
                             u_soc.u_cpu.bdtu_busy,
                             u_soc.u_cpu.stall_if1);
                    $display("  Stalls: if1=%b if2=%b id=%b ex1=%b ex2=%b ex3=%b mem=%b",
                             u_soc.u_cpu.stall_if1, u_soc.u_cpu.stall_if2,
                             u_soc.u_cpu.stall_id,
                             u_soc.u_cpu.stall_ex1, u_soc.u_cpu.stall_ex2,
                             u_soc.u_cpu.stall_ex3, u_soc.u_cpu.stall_mem);
                    dump_regs();
                    run_exit_status = 1;
                    repeat (5) @(posedge clk);
                    disable sort_run_loop;
                end

                // ── Detection 3: timeout ──
                if (cycle_cnt >= TIMEOUT) begin
                    $display("");
                    $display("*** TIMEOUT after %0d cycles (PC=0x%08H) ***",
                             TIMEOUT, cur_pc);
                    dump_regs();
                    run_exit_status = 2;
                    disable sort_run_loop;
                end

                // ── Periodic status ──
                if (cycle_cnt % STATUS_INTERVAL == 0) begin
                    $display("  [STATUS C%06d] PC=0x%08H  BDTU=%s  stall_if1=%b",
                             cycle_cnt,
                             u_soc.u_cpu.pc_if1,
                             bdtu_state_name(u_soc.u_cpu.u_bdtu.state),
                             u_soc.u_cpu.stall_if1);
                    $display("    R0=%08H R1=%08H R2=%08H R3=%08H SP=%08H LR=%08H R11=%08H R12=%08H",
                             u_soc.u_cpu.u_regfile.regs[0],
                             u_soc.u_cpu.u_regfile.regs[1],
                             u_soc.u_cpu.u_regfile.regs[2],
                             u_soc.u_cpu.u_regfile.regs[3],
                             u_soc.u_cpu.u_regfile.regs[13],
                             u_soc.u_cpu.u_regfile.regs[14],
                             u_soc.u_cpu.u_regfile.regs[11],
                             u_soc.u_cpu.u_regfile.regs[12]);
                end
            end
        end

        total_sort_cycles = cycle_cnt;
        $display("");
        $display("════════════════════════════════════════════════════════════════");
        $display("  END-OF-RUN: Bubble Sort  (%0d cycles, exit=%0d, dmem_writes=%0d, dmem_reads=%0d)",
                 total_sort_cycles, run_exit_status, dmem_write_cnt, dmem_read_cnt);
        $display("════════════════════════════════════════════════════════════════");
        dump_regs();

        // ── B9: De-assert start (CPU resets, BRAMs preserved) ──
        $display("\n[B9] De-asserting start pin...");
        start = 1'b0;
        @(posedge clk); #1;

        // Wait for MMIO to become available
        cycle_cnt = 0;
        while (!req_rdy && cycle_cnt < 100) begin
            @(posedge clk); #1;
            cycle_cnt = cycle_cnt + 1;
        end
        if (req_rdy)
            $display("  MMIO available (req_rdy=1)");
        else begin
            $display("  WARNING: req_rdy not returning high");
            fail_cnt = fail_cnt + 1;
        end

        // ── B10: Read FULL DMEM via MMIO and search for sorted array ──
        $display("[B10] Reading %0d DMEM words via MMIO (full BRAM)...",
                 dmem_load_extent);
        for (i = 0; i < dmem_load_extent; i = i + 1) begin
            mmio_rd(DMEM_BASE | i, rd_data);
            dmem_snap[i] = rd_data[`DATA_WIDTH-1:0];
        end
        $display("  DMEM readback complete.");

        // Count changed words and show details
        begin : dmem_analysis
            integer changed, w2;
            changed = 0;
            for (w2 = 0; w2 < dmem_load_extent; w2 = w2 + 1)
                if (dmem_snap[w2] !== sort_local_mem[w2])
                    changed = changed + 1;
            $display("  DMEM words changed from original hex: %0d out of %0d", changed, dmem_load_extent);

            // Show ALL changed words with before/after
            $display("  -- ALL changed DMEM words (before -> after) --");
            for (w2 = 0; w2 < dmem_load_extent; w2 = w2 + 1) begin
                if (dmem_snap[w2] !== sort_local_mem[w2])
                    $display("    [w%04d / 0x%05H] %08H -> %08H  (%0d -> %0d)",
                             w2, w2 << 2,
                             sort_local_mem[w2], dmem_snap[w2],
                             $signed(sort_local_mem[w2]), $signed(dmem_snap[w2]));
            end

            $display("  -- DMEM top 16 words (stack region) --");
            for (w2 = dmem_load_extent - 16; w2 < dmem_load_extent; w2 = w2 + 1)
                $display("    [word %4d / 0x%04H] = 0x%08H (%0d)", w2, w2 << 2, dmem_snap[w2], $signed(dmem_snap[w2]));
        end

        // Search for the expected sorted sequence
        $display("  Searching for sorted array in DMEM...");
        found      = 1'b0;
        found_addr = 0;
        sort_count = 0;

        for (i = 0; i < dmem_load_extent - SORT_ARRAY_SIZE + 1; i = i + 1) begin
            if (!found) begin
                match = 1'b1;
                for (j = 0; j < SORT_ARRAY_SIZE; j = j + 1) begin
                    if (dmem_snap[i + j] !== expected_sorted[j])
                        match = 1'b0;
                end
                if (match) begin
                    found      = 1'b1;
                    found_addr = i;
                end
            end
        end

        if (found) begin
            $display("  FOUND sorted array at DMEM word %0d (byte 0x%04H):",
                     found_addr, found_addr << 2);
            for (j = 0; j < SORT_ARRAY_SIZE; j = j + 1) begin
                sorted_result[j] = dmem_snap[found_addr + j];
                $display("    [%0d] = 0x%08H  (%0d)", j,
                         dmem_snap[found_addr + j],
                         $signed(dmem_snap[found_addr + j]));
            end
            sort_count = SORT_ARRAY_SIZE;
        end else begin
            $display("  Sorted array NOT FOUND in DMEM.");
            dump_dmem_nonzero();
        end

        // ── B11: Verify results ────────────────────────
        $display("\n[B11] Verification...");
        verify_sort_order();
        $display("");
        verify_against_expected();


        // ═══════════════════════════════════════════════════
        //  Summary
        // ═══════════════════════════════════════════════════
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                        SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        $display("║  Sort Cycles  : %6d                                     ║",
                 total_sort_cycles);
        case (run_exit_status)
            0: $display("║  Sort Exit    : CLEAN (halt reached)                         ║");
            1: $display("║  Sort Exit    : STUCK (PC looped at unexpected address)       ║");
            2: $display("║  Sort Exit    : TIMEOUT (%0d cycles)                      ║",
                        TIMEOUT);
        endcase
        if (sort_count > 0)
            $display("║  Sort Result  : FOUND (%0d elements)                          ║",
                     sort_count);
        else
            $display("║  Sort Result  : NOT FOUND                                      ║");
        $display("║  IMEM Depth   : %0d words (loaded %0d)                        ║",
                 IMEM_HW_DEPTH, imem_load_extent);
        $display("║  DMEM Depth   : %0d words (loaded %0d)                       ║",
                 DMEM_HW_DEPTH, dmem_load_extent);
        $display("║  Checks       : %0d PASSED, %0d FAILED (%0d total)            ║",
                 pass_cnt, fail_cnt, pass_cnt + fail_cnt);
        $display("╚══════════════════════════════════════════════════════════════╝");
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
        #100_000_000;
        $display("\n[TIMEOUT] Simulation exceeded 100 ms — aborting");
        $finish;
    end

endmodule