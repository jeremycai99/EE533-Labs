/* file: soc_tb.v
 * Description: SoC MMIO testbench
 *   Phase A — Basic MMIO read/write infrastructure tests
 *   Phase B — Bubble Sort integration (hex load via MMIO → run via ext start → verify via MMIO)
 *   Phase C — ILA Debug Interface verification
 *
 *  The bubble sort program uses a unified memory image. Both IMEM and DMEM
 *  are loaded with the same hex file (code fetched from IMEM, data from DMEM).
 *  The CPU is started via the external `start` pin (not MMIO CTRL) because
 *  the sort program halts with B. (branch-to-self), which does not trigger
 *  `cpu_done`.  PC is monitored through the ILA debug port (no MMIO needed
 *  while the CPU is running).  After halt detection, `start` is de-asserted
 *  (CPU resets, BRAMs preserved), and DMEM is read back via MMIO to verify
 *  the sorted output.
 *
 *  COMPILATION:
 *      iverilog -o soc_tb soc_tb.v
 *      vvp soc_tb
 *
 *  To override the hex file:
 *      iverilog -DMEM_FILE=\"my_imem.txt\" -o soc_tb soc_tb.v
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
    parameter SORT_MEM_DEPTH   = 4096;
    parameter LOAD_EXTENT      = 1024;   // min words to load into IMEM & DMEM

    parameter [31:0] HALT_ADDR  = 32'h0000_0200;
    parameter [31:0] HALT_INSTR = 32'hEAFF_FFFE;  // B . (branch-to-self)
    parameter [31:0] HALT_WORD  = HALT_ADDR >> 2;  // word index 128

    // Hardware BRAM depths — derived from define.v
    localparam IMEM_HW_DEPTH = (1 << `IMEM_ADDR_WIDTH);  // e.g. 512 when IMEM_ADDR_WIDTH=9
    localparam DMEM_HW_DEPTH = (1 << `DMEM_ADDR_WIDTH);  // e.g. 1024 when DMEM_ADDR_WIDTH=10

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
    //  ILA Interface Signals
    // ================================================================
    reg  [4:0]                  ila_debug_sel;
    wire [`DATA_WIDTH-1:0]      ila_debug_data;

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
        .resp_rdy     (resp_rdy),
        .ila_debug_sel    (ila_debug_sel),
        .ila_debug_data   (ila_debug_data)
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
    integer imem_load_extent;     // clamped to IMEM_HW_DEPTH
    integer dmem_load_extent;     // clamped to DMEM_HW_DEPTH

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
    integer found_addr;
    reg     found, match;
    reg [`PC_WIDTH-1:0] prev_pc;

    reg [`MMIO_DATA_WIDTH-1:0]  rd_data;
    reg [`MMIO_ADDR_WIDTH-1:0]  taddr;
    reg [`MMIO_DATA_WIDTH-1:0]  twdata;
    reg [7:0]                   toff;

    // ================================================================
    //  State/Name Decoder
    // ================================================================
    function [7*8:1] bdtu_state_name;
        input [2:0] st;
        case (st)
            3'd0: bdtu_state_name = "IDLE   ";
            3'd1: bdtu_state_name = "BDT_XFR";
            3'd2: bdtu_state_name = "BDT_LST";
            3'd3: bdtu_state_name = "BDT_WB ";
            3'd4: bdtu_state_name = "SWP_RD ";
            3'd5: bdtu_state_name = "SWP_WAT";
            3'd6: bdtu_state_name = "SWP_WR ";
            3'd7: bdtu_state_name = "DONE   ";
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

            // Determine actual extent (last non-zero word + margin)
            load_extent_actual = 0;
            for (k = 0; k < SORT_MEM_DEPTH; k = k + 1)
                if (sort_local_mem[k] !== {`DATA_WIDTH{1'b0}})
                    load_extent_actual = k + 1;
            if (load_extent_actual < LOAD_EXTENT)
                load_extent_actual = LOAD_EXTENT;
            $display("  Load extent: %0d words (%0d bytes)",
                     load_extent_actual, load_extent_actual << 2);

            // Clamp to hardware BRAM depths to prevent aliased overwrites
            $display("  Hardware depths: IMEM=%0d words, DMEM=%0d words",
                     IMEM_HW_DEPTH, DMEM_HW_DEPTH);

            if (load_extent_actual > IMEM_HW_DEPTH) begin
                imem_load_extent = IMEM_HW_DEPTH;
                $display("  NOTE: IMEM load clamped from %0d to %0d words (IMEM_ADDR_WIDTH=%0d)",
                         load_extent_actual, IMEM_HW_DEPTH, `IMEM_ADDR_WIDTH);
            end else begin
                imem_load_extent = load_extent_actual;
            end

            if (load_extent_actual > DMEM_HW_DEPTH) begin
                dmem_load_extent = DMEM_HW_DEPTH;
                $display("  NOTE: DMEM load clamped from %0d to %0d words (DMEM_ADDR_WIDTH=%0d)",
                         load_extent_actual, DMEM_HW_DEPTH, `DMEM_ADDR_WIDTH);
            end else begin
                dmem_load_extent = load_extent_actual;
            end

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
        ila_debug_sel = 0;
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
        //  IMPORTANT: Only write up to imem_load_extent words to
        //  avoid aliased overwrites when the image exceeds BRAM depth.
        $display("[B3] Writing %0d words to IMEM via MMIO (BRAM depth=%0d)...",
                 imem_load_extent, IMEM_HW_DEPTH);
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
        $display("[B4] Writing %0d words to DMEM via MMIO (BRAM depth=%0d)...",
                 dmem_load_extent, DMEM_HW_DEPTH);
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

        // ── B8: Monitor PC via ILA, wait for halt ───────
        $display("[B8] Running CPU (timeout=%0d cycles)...", TIMEOUT);
        $display("  halt_addr = 0x%08H", HALT_ADDR);
        $display("");

        ila_debug_sel   = 5'd0;   // Select PC
        cycle_cnt       = 0;
        stuck_cnt       = 0;
        prev_pc         = {`PC_WIDTH{1'b0}};
        run_exit_status = 2;       // default: timeout

        begin : sort_run_loop
            forever begin
                @(posedge clk); #1;
                cycle_cnt = cycle_cnt + 1;

                // ── Detection 1: PC reached halt address ──
                if (ila_debug_data == HALT_ADDR && cycle_cnt > 20) begin
                    $display("");
                    $display("[%0t] PC reached halt address 0x%08H at cycle %0d",
                             $time, HALT_ADDR, cycle_cnt);
                    run_exit_status = 0;
                    repeat (10) @(posedge clk);
                    disable sort_run_loop;
                end

                // ── Stuck-PC detector ──
                if (ila_debug_data === prev_pc)
                    stuck_cnt = stuck_cnt + 1;
                else
                    stuck_cnt = 0;
                prev_pc = ila_debug_data;

                // ── Detection 2: stuck at unexpected address ──
                if (stuck_cnt > 500 && prev_pc != HALT_ADDR) begin
                    $display("");
                    $display("╔══════════════════════════════════════════════════╗");
                    $display("║  *** STUCK: PC=0x%08H for %0d cycles ***",
                             prev_pc, stuck_cnt);
                    $display("╚══════════════════════════════════════════════════╝");
                    $display("  BDTU state=%s busy=%b stall_if=%b",
                             bdtu_state_name(u_soc.u_cpu.u_bdtu.state),
                             u_soc.u_cpu.bdtu_busy,
                             u_soc.u_cpu.stall_if);
                    $display("  Stalls: if=%b id=%b ex=%b mem=%b",
                             u_soc.u_cpu.stall_if, u_soc.u_cpu.stall_id,
                             u_soc.u_cpu.stall_ex, u_soc.u_cpu.stall_mem);
                    dump_regs();
                    run_exit_status = 1;
                    repeat (5) @(posedge clk);
                    disable sort_run_loop;
                end

                // ── Detection 3: timeout ──
                if (cycle_cnt >= TIMEOUT) begin
                    $display("");
                    $display("*** TIMEOUT after %0d cycles (PC=0x%08H) ***",
                             TIMEOUT, ila_debug_data);
                    dump_regs();
                    run_exit_status = 2;
                    disable sort_run_loop;
                end

                // ── Periodic status ──
                if (cycle_cnt % STATUS_INTERVAL == 0) begin
                    $display("  [STATUS C%06d] PC=0x%08H  BDTU=%s  stall_if=%b",
                             cycle_cnt,
                             u_soc.u_cpu.pc_if,
                             bdtu_state_name(u_soc.u_cpu.u_bdtu.state),
                             u_soc.u_cpu.stall_if);
                    $display("    R0=%08H R1=%08H R2=%08H R3=%08H SP=%08H LR=%08H",
                             u_soc.u_cpu.u_regfile.regs[0],
                             u_soc.u_cpu.u_regfile.regs[1],
                             u_soc.u_cpu.u_regfile.regs[2],
                             u_soc.u_cpu.u_regfile.regs[3],
                             u_soc.u_cpu.u_regfile.regs[13],
                             u_soc.u_cpu.u_regfile.regs[14]);
                end
            end
        end

        total_sort_cycles = cycle_cnt;
        $display("");
        $display("════════════════════════════════════════════════════════════════");
        $display("  END-OF-RUN: Bubble Sort  (%0d cycles, exit=%0d)",
                 total_sort_cycles, run_exit_status);
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

        // ── B10: Read DMEM via MMIO and search for sorted array ──
        $display("[B10] Reading %0d DMEM words via MMIO...", dmem_load_extent);
        for (i = 0; i < dmem_load_extent; i = i + 1) begin
            mmio_rd(DMEM_BASE | i, rd_data);
            dmem_snap[i] = rd_data[`DATA_WIDTH-1:0];
        end
        $display("  DMEM readback complete.");

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
        //  PHASE C — ILA Debug Interface Verification
        // ═══════════════════════════════════════════════════
        $display("\n──────────────────────────────────────────────────");
        $display("  PHASE C: ILA Debug Interface Verification");
        $display("──────────────────────────────────────────────────");
        //  The CPU is currently in reset (start=0 → cpu_rst_n=0).
        //  Pipeline registers are cleared, but the register file
        //  retains the values from the just-completed sort execution.

        // ── C1: Read registers via ILA ──────────────────
        $display("\n[C1] Reading registers via ILA debug port...");
        for (i = 0; i < 16; i = i + 1) begin
            ila_debug_sel = 5'b10000 | i[3:0];
            @(posedge clk); #1;
            $display("  ILA Reg[%2d] = 0x%08H  (hier: 0x%08H)",
                     i, ila_debug_data, u_soc.u_cpu.u_regfile.regs[i]);
            // Cross-check ILA port against hierarchical access
            if (ila_debug_data !== u_soc.u_cpu.u_regfile.regs[i]) begin
                $display("    [FAIL] ILA/hier mismatch for R%0d!", i);
                fail_cnt = fail_cnt + 1;
            end else
                pass_cnt = pass_cnt + 1;
        end

        // ── C2: System debug probes ─────────────────────
        $display("\n[C2] System debug probes...");

        ila_debug_sel = 5'd0; @(posedge clk); #1;
        $display("  Sel[0]  PC         = 0x%08H  (CPU in reset, expect 0)",
                 ila_debug_data);
        if (ila_debug_data == {`DATA_WIDTH{1'b0}})
            pass_cnt = pass_cnt + 1;
        else begin
            $display("    [FAIL] PC non-zero while CPU is in reset");
            fail_cnt = fail_cnt + 1;
        end

        ila_debug_sel = 5'd1; @(posedge clk); #1;
        $display("  Sel[1]  Instr      = 0x%08H", ila_debug_data);

        ila_debug_sel = 5'd4; @(posedge clk); #1;
        $display("  Sel[4]  ALU result = 0x%08H", ila_debug_data);

        ila_debug_sel = 5'd7; @(posedge clk); #1;
        $display("  Sel[7]  Controls   = 0x%08H", ila_debug_data);

        ila_debug_sel = 5'd8; @(posedge clk); #1;
        $display("  Sel[8]  CPSR flags = 0x%08H", ila_debug_data);

        // Non-X sanity on the debug bus
        ila_debug_sel = 5'd0; @(posedge clk); #1;
        if (^ila_debug_data === 1'bx) begin
            $display("  [FAIL] Debug bus contains X values");
            fail_cnt = fail_cnt + 1;
        end else begin
            $display("  [PASS] Debug bus free of X");
            pass_cnt = pass_cnt + 1;
        end


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