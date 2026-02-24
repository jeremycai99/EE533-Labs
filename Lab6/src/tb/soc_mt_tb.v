/* soc_mt_tb.v — SoC-level testbench for quad-thread ARMv4T multithreaded CPU
 *
 * Migrated from cpu_mt_tb.v, using MMIO access pattern from soc_tb.v.
 *
 *   Thread 0: network_proc0 — packet header check, field swap, checksum
 *   Thread 1: network_proc1 — XOR encryption with key 0xDEADBEEF
 *   Thread 2: network_proc2 — counter decrement / threshold check
 *   Thread 3: network_proc3 — field comparison against constant 23
 *
 * Phases:
 *   Phase A — Basic MMIO read/write infrastructure tests
 *   Phase B — Multithreaded network processing (4 scenarios loaded/verified via MMIO)
 *   Phase C — Canary / diagnostic tests (replicating socreg_mt script diagnostics)
 *   Phase D — Bootstrap-based MT tests (register init via ARM code, not hierarchical)
 *
 * Memory layout (fits IMEM=1024w / DMEM=4096w from define.v):
 *   IMEM code slots (256 B each):
 *     T0 0x0500-0x05FF   T1 0x0600-0x06FF
 *     T2 0x0700-0x07FF   T3 0x0800-0x08FF
 *   DMEM data (addresses baked into ARM machine code):
 *     T0 data 0x0100   T1 data 0x0200   T2 data 0x0300   T3 data 0x0400
 *   DMEM stacks (set via register init, above code/data region):
 *     T0 SP 0x1000   T1 SP 0x1200   T2 SP 0x1400   T3 SP 0x1600
 *
 * Author: Jeremy Cai
 * Date:   Feb. 23, 2026
 */
`timescale 1ns / 1ps

`include "define.v"
`include "soc_mt.v"

module soc_mt_tb;

// ═══════════════════════════════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════════════════════════════
parameter CLK_PERIOD   = 10;
parameter MEM_DEPTH    = 16384;        // local staging array (16 K words)
parameter MAX_CYCLES   = 10_000;
parameter TRACE_EN     = 1;
parameter TRACE_LIMIT  = 600;

parameter NUM_RAND         = 64;       // Phase-A random iterations
parameter ADDR_RANGE       = 256;      // Phase-A address span (words 0-255, below code)
parameter STATUS_INTERVAL  = 500;

localparam [31:0] SENTINEL = 32'hEAFF_FFFE;   // B . (branch-to-self)

// ── Thread code base addresses (byte addresses) ──────────────────
//    Each thread gets a 256-byte (64-word) IMEM slot.
//    Placed at 0x0500+ so they don't overlap with DMEM data
//    addresses (0x0100-0x0420) in the unified local_mem staging.
localparam [31:0] T0_CODE = 32'h0000_0500;
localparam [31:0] T1_CODE = 32'h0000_0600;
localparam [31:0] T2_CODE = 32'h0000_0700;
localparam [31:0] T3_CODE = 32'h0000_0800;

// ── Thread data base addresses (byte addresses) ──────────────────
//    These are HARDCODED in the ARM machine code (MOV Rd, #imm)
//    and CANNOT be changed without re-assembling the instructions.
localparam [31:0] T0_DATA = 32'h0000_0100;
localparam [31:0] T1_DATA = 32'h0000_0200;
localparam [31:0] T2_DATA = 32'h0000_0300;
localparam [31:0] T3_DATA = 32'h0000_0400;

// ── Thread stack pointers (byte addresses) ───────────────────────
//    Placed well above both code and data to avoid any aliasing.
//    All within DMEM's 4096-word (16 KB) range.
localparam [31:0] T0_SP = 32'h0000_1000;
localparam [31:0] T1_SP = 32'h0000_1200;
localparam [31:0] T2_SP = 32'h0000_1400;
localparam [31:0] T3_SP = 32'h0000_1600;

// ── Thread sentinel (return) addresses ───────────────────────────
//    Last word of each thread's 256-byte IMEM slot.
localparam [31:0] T0_RET = 32'h0000_05FC;
localparam [31:0] T1_RET = 32'h0000_06FC;
localparam [31:0] T2_RET = 32'h0000_07FC;
localparam [31:0] T3_RET = 32'h0000_08FC;

// ── MMIO address bases ────────────────────────────────────────────
localparam [31:0] IMEM_BASE = 32'h0000_0000;
localparam [31:0] DMEM_BASE = 32'h8000_0000;
localparam [31:0] CTRL_BASE = 32'h4000_0000;

// ── Hardware BRAM depths (from define.v) ──────────────────────────
localparam IMEM_HW_DEPTH = (1 << `IMEM_ADDR_WIDTH);   // 1024
localparam DMEM_HW_DEPTH = (1 << `DMEM_ADDR_WIDTH);   // 4096

// ── Minimum depths required by the test ───────────────────────────
//    IMEM must hold code + sentinels up to T3_RET.
//    DMEM must hold data + literal pool + stack accesses up to T3_SP.
localparam IMEM_MIN_DEPTH = (T3_RET >> 2) + 1;        // 576
localparam DMEM_MIN_DEPTH = (T3_SP  >> 2) + 1;        // 1409

// ═══════════════════════════════════════════════════════════════════
//  DUT Signals
// ═══════════════════════════════════════════════════════════════════
reg                          clk, rst_n, start;

// ── MMIO request channel ──────────────────────────────────────────
reg                          req_cmd;
reg  [`MMIO_ADDR_WIDTH-1:0]  req_addr;
reg  [`MMIO_DATA_WIDTH-1:0]  req_data;
reg                          req_val;
wire                         req_rdy;

// ── MMIO response channel ─────────────────────────────────────────
wire                         resp_cmd;
wire [`MMIO_ADDR_WIDTH-1:0]  resp_addr;
wire [`MMIO_DATA_WIDTH-1:0]  resp_data;
wire                         resp_val;
reg                          resp_rdy;

// ═══════════════════════════════════════════════════════════════════
//  DUT Instantiation
// ═══════════════════════════════════════════════════════════════════
soc_mt u_soc_mt (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (start),
    .req_cmd        (req_cmd),
    .req_addr       (req_addr),
    .req_data       (req_data),
    .req_val        (req_val),
    .req_rdy        (req_rdy),
    .resp_cmd       (resp_cmd),
    .resp_addr      (resp_addr),
    .resp_data      (resp_data),
    .resp_val       (resp_val),
    .resp_rdy       (resp_rdy)
);

// ═══════════════════════════════════════════════════════════════════
//  Clock
// ═══════════════════════════════════════════════════════════════════
initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

// ═══════════════════════════════════════════════════════════════════
//  Local Staging Memory  (mirrors cpu_mt_tb's unified mem_array)
// ═══════════════════════════════════════════════════════════════════
reg [31:0] local_mem [0:MEM_DEPTH-1];

// ═══════════════════════════════════════════════════════════════════
//  Bookkeeping
// ═══════════════════════════════════════════════════════════════════
integer total_pass, total_fail, section_pass, section_fail;
integer cycle_cnt;
integer i, j, k, seed;
reg [256*8:1] current_section;

reg [`MMIO_DATA_WIDTH-1:0]  rd_data;
reg [`MMIO_ADDR_WIDTH-1:0]  taddr;
reg [`MMIO_DATA_WIDTH-1:0]  twdata;
reg [7:0]                   toff;
integer num_checked;

// Phase-A reference models
reg [`MMIO_DATA_WIDTH-1:0]  imem_ref   [0:ADDR_RANGE-1];
reg                         imem_valid [0:ADDR_RANGE-1];
reg [`MMIO_DATA_WIDTH-1:0]  dmem_ref   [0:ADDR_RANGE-1];
reg                         dmem_valid [0:ADDR_RANGE-1];


// ═══════════════════════════════════════════════════════════════════
//  MMIO Transaction Tasks  (from soc_tb.v)
// ═══════════════════════════════════════════════════════════════════

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


// ═══════════════════════════════════════════════════════════════════
//  Reset Helper
// ═══════════════════════════════════════════════════════════════════
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


// ═══════════════════════════════════════════════════════════════════
//  Local-Memory Helper Tasks  (same interface as cpu_mt_tb)
// ═══════════════════════════════════════════════════════════════════

task mem_clear;
    integer m;
begin
    for (m = 0; m < MEM_DEPTH; m = m + 1)
        local_mem[m] = 32'h0;
end
endtask

task mem_w;
    input [31:0] byte_addr;
    input [31:0] data;
begin
    local_mem[byte_addr >> 2] = data;
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  SoC Image Loading — write local_mem to IMEM & DMEM via MMIO
// ═══════════════════════════════════════════════════════════════════

task load_image_to_soc;
    integer w, max_extent, imem_ext, dmem_ext;
begin
    // Determine highest non-zero word in local_mem
    max_extent = 0;
    for (w = 0; w < MEM_DEPTH; w = w + 1)
        if (local_mem[w] !== 32'h0)
            max_extent = w;
    max_extent = max_extent + 1;

    // Clamp to hardware depths
    imem_ext = (max_extent < IMEM_HW_DEPTH) ? max_extent : IMEM_HW_DEPTH;
    dmem_ext = (max_extent < DMEM_HW_DEPTH) ? max_extent : DMEM_HW_DEPTH;

    if (max_extent > IMEM_HW_DEPTH)
        $display("  WARNING: Image extent (%0d words) > IMEM depth (%0d). Some code/sentinels may not load!",
                 max_extent, IMEM_HW_DEPTH);
    if (max_extent > DMEM_HW_DEPTH)
        $display("  WARNING: Image extent (%0d words) > DMEM depth (%0d). Some data may not load!",
                 max_extent, DMEM_HW_DEPTH);

    // Write all words 0..extent to IMEM (includes zeros → clears stale data)
    $display("  Loading %0d words to IMEM, %0d words to DMEM...", imem_ext, dmem_ext);
    for (w = 0; w < imem_ext; w = w + 1)
        mmio_wr(IMEM_BASE | w, local_mem[w]);

    for (w = 0; w < dmem_ext; w = w + 1)
        mmio_wr(DMEM_BASE | w, local_mem[w]);

    $display("  Image load complete (extent=%0d words).", max_extent);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  CPU Register Init — hierarchical access during reset (start=0)
// ═══════════════════════════════════════════════════════════════════

task init_cpu_mt_regs;
    integer t, r;
begin
    // Clear all registers for all threads
    for (t = 0; t < 4; t = t + 1)
        for (r = 0; r < `REG_DEPTH; r = r + 1) begin
            case (t)
                0: u_soc_mt.u_cpu_mt.THREAD_RF[0].u_rf.regs[r] = {`DATA_WIDTH{1'b0}};
                1: u_soc_mt.u_cpu_mt.THREAD_RF[1].u_rf.regs[r] = {`DATA_WIDTH{1'b0}};
                2: u_soc_mt.u_cpu_mt.THREAD_RF[2].u_rf.regs[r] = {`DATA_WIDTH{1'b0}};
                3: u_soc_mt.u_cpu_mt.THREAD_RF[3].u_rf.regs[r] = {`DATA_WIDTH{1'b0}};
            endcase
        end

    // Set per-thread SP (R13) and LR (R14)
    u_soc_mt.u_cpu_mt.THREAD_RF[0].u_rf.regs[13] = T0_SP;
    u_soc_mt.u_cpu_mt.THREAD_RF[0].u_rf.regs[14] = T0_RET;
    u_soc_mt.u_cpu_mt.THREAD_RF[1].u_rf.regs[13] = T1_SP;
    u_soc_mt.u_cpu_mt.THREAD_RF[1].u_rf.regs[14] = T1_RET;
    u_soc_mt.u_cpu_mt.THREAD_RF[2].u_rf.regs[13] = T2_SP;
    u_soc_mt.u_cpu_mt.THREAD_RF[2].u_rf.regs[14] = T2_RET;
    u_soc_mt.u_cpu_mt.THREAD_RF[3].u_rf.regs[13] = T3_SP;
    u_soc_mt.u_cpu_mt.THREAD_RF[3].u_rf.regs[14] = T3_RET;
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Hierarchical Register / PC Access  (for monitoring & dumps)
// ═══════════════════════════════════════════════════════════════════

function [31:0] get_reg;
    input [1:0] tid;
    input [3:0] rn;
begin
    case (tid)
        2'd0: get_reg = u_soc_mt.u_cpu_mt.THREAD_RF[0].u_rf.regs[rn];
        2'd1: get_reg = u_soc_mt.u_cpu_mt.THREAD_RF[1].u_rf.regs[rn];
        2'd2: get_reg = u_soc_mt.u_cpu_mt.THREAD_RF[2].u_rf.regs[rn];
        2'd3: get_reg = u_soc_mt.u_cpu_mt.THREAD_RF[3].u_rf.regs[rn];
    endcase
end
endfunction

function [31:0] get_thread_pc;
    input [1:0] tid;
begin
    case (tid)
        2'd0: get_thread_pc = u_soc_mt.u_cpu_mt.pc_thread[0];
        2'd1: get_thread_pc = u_soc_mt.u_cpu_mt.pc_thread[1];
        2'd2: get_thread_pc = u_soc_mt.u_cpu_mt.pc_thread[2];
        2'd3: get_thread_pc = u_soc_mt.u_cpu_mt.pc_thread[3];
    endcase
end
endfunction

function [3:0] get_thread_cpsr;
    input [1:0] tid;
begin
    case (tid)
        2'd0: get_thread_cpsr = u_soc_mt.u_cpu_mt.cpsr_flags[0];
        2'd1: get_thread_cpsr = u_soc_mt.u_cpu_mt.cpsr_flags[1];
        2'd2: get_thread_cpsr = u_soc_mt.u_cpu_mt.cpsr_flags[2];
        2'd3: get_thread_cpsr = u_soc_mt.u_cpu_mt.cpsr_flags[3];
    endcase
end
endfunction


// ═══════════════════════════════════════════════════════════════════
//  MMIO-Based Verification Tasks
// ═══════════════════════════════════════════════════════════════════

// ── Check a DMEM word via MMIO ────────────────────────────────────
task check_mem_mmio;
    input [31:0]    byte_addr;
    input [31:0]    expected;
    input [256*8:1] msg;
    reg [`MMIO_DATA_WIDTH-1:0] raw;
    reg [31:0] actual;
begin
    mmio_rd(DMEM_BASE | (byte_addr >> 2), raw);
    actual = raw[31:0];
    if (actual === expected) begin
        $display("    [PASS] [0x%08H] = 0x%08H  %0s", byte_addr, expected, msg);
        section_pass = section_pass + 1;
        total_pass   = total_pass   + 1;
    end else begin
        $display("    [FAIL] [0x%08H] = 0x%08H, expected 0x%08H  %0s",
                 byte_addr, actual, expected, msg);
        section_fail = section_fail + 1;
        total_fail   = total_fail   + 1;
    end
end
endtask

// ── Check a thread register via hierarchical access ───────────────
task check_reg_hier;
    input [1:0]     tid;
    input [3:0]     rn;
    input [31:0]    expected;
    input [256*8:1] msg;
    reg [31:0] actual;
begin
    actual = get_reg(tid, rn);
    if (actual === expected) begin
        $display("    [PASS] T%0d R%0d = 0x%08H  %0s", tid, rn, expected, msg);
        section_pass = section_pass + 1;
        total_pass   = total_pass   + 1;
    end else begin
        $display("    [FAIL] T%0d R%0d = 0x%08H, expected 0x%08H  %0s",
                 tid, rn, actual, expected, msg);
        section_fail = section_fail + 1;
        total_fail   = total_fail   + 1;
    end
end
endtask

// ── Phase-A style checker ─────────────────────────────────────────
task check;
    input [`MMIO_DATA_WIDTH-1:0]  expected;
    input [`MMIO_DATA_WIDTH-1:0]  actual;
    input [`MMIO_ADDR_WIDTH-1:0]  addr;
    input [255:0]                 tag;
begin
    if (expected === actual) begin
        total_pass = total_pass + 1;
    end else begin
        total_fail = total_fail + 1;
        $display("  [FAIL] %0s @ 0x%0h  exp=0x%08h  got=0x%08h",
                 tag, addr, expected, actual);
    end
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Section Header / Footer
// ═══════════════════════════════════════════════════════════════════

task section_start;
    input [256*8:1] name;
begin
    current_section = name;
    section_pass = 0;
    section_fail = 0;
    $display("");
    $display("┌──────────────────────────────────────────────────────────────┐");
    $display("│  %0s", name);
    $display("└──────────────────────────────────────────────────────────────┘");
end
endtask

task section_end;
begin
    if (section_fail > 0)
        $display("  ** %0s: %0d PASSED, %0d FAILED (%0d cycles) **",
                 current_section, section_pass, section_fail, cycle_cnt);
    else
        $display("  ── %0s: all %0d passed (%0d cycles) ──",
                 current_section, section_pass, cycle_cnt);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Dump Tasks
// ═══════════════════════════════════════════════════════════════════

// ── Dump all thread registers (hierarchical) ──────────────────────
task dump_all_threads;
    integer t, r;
begin
    for (t = 0; t < 4; t = t + 1) begin
        $display("  ┌─ Thread %0d ── PC=0x%08H  CPSR_flags=%04b ─────────────────┐",
                 t, get_thread_pc(t[1:0]), get_thread_cpsr(t[1:0]));
        for (r = 0; r < 16; r = r + 4)
            $display("  │ R%-2d=0x%08H  R%-2d=0x%08H  R%-2d=0x%08H  R%-2d=0x%08H │",
                     r,   get_reg(t[1:0], r[3:0]),
                     r+1, get_reg(t[1:0], (r+1)),
                     r+2, get_reg(t[1:0], (r+2)),
                     r+3, get_reg(t[1:0], (r+3)));
        $display("  └───────────────────────────────────────────────────────────────┘");
    end
end
endtask

// ── Dump a DMEM region via MMIO ───────────────────────────────────
task dump_mem_mmio;
    input [31:0] base_byte;
    input integer count;
    integer d;
    reg [`MMIO_DATA_WIDTH-1:0] val;
begin
    $display("  ┌─ DMEM @ 0x%08H (%0d words) ──────────────────────────────┐",
             base_byte, count);
    for (d = 0; d < count; d = d + 1) begin
        mmio_rd(DMEM_BASE | ((base_byte >> 2) + d), val);
        $display("  │ [0x%08H] = 0x%08H                                          │",
                 base_byte + (d*4), val[31:0]);
    end
    $display("  └───────────────────────────────────────────────────────────────┘");
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Thread Code Loading  (identical to cpu_mt_tb — writes local_mem)
// ═══════════════════════════════════════════════════════════════════

task load_thread0_code;
    reg [31:0] B;
begin
    B = T0_CODE;
    mem_w(B+'h00, 32'hE52DB004);  mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD014);  mem_w(B+'h0C, 32'hE3A03C01);
    mem_w(B+'h10, 32'hE50B3010);  mem_w(B+'h14, 32'hE51B3010);
    mem_w(B+'h18, 32'hE5933004);  mem_w(B+'h1C, 32'hE35300AA);
    mem_w(B+'h20, 32'h0A000003);  mem_w(B+'h24, 32'hE51B3010);
    mem_w(B+'h28, 32'hE3E02000);  mem_w(B+'h2C, 32'hE5832010);
    mem_w(B+'h30, 32'hEA000021);  mem_w(B+'h34, 32'hE51B3010);
    mem_w(B+'h38, 32'hE5933008);  mem_w(B+'h3C, 32'hE50B3014);
    mem_w(B+'h40, 32'hE51B3010);  mem_w(B+'h44, 32'hE593200C);
    mem_w(B+'h48, 32'hE51B3010);  mem_w(B+'h4C, 32'hE5832008);
    mem_w(B+'h50, 32'hE51B3010);  mem_w(B+'h54, 32'hE51B2014);
    mem_w(B+'h58, 32'hE583200C);  mem_w(B+'h5C, 32'hE3A03000);
    mem_w(B+'h60, 32'hE50B3008);  mem_w(B+'h64, 32'hE3A03000);
    mem_w(B+'h68, 32'hE50B300C);  mem_w(B+'h6C, 32'hEA00000B);
    mem_w(B+'h70, 32'hE51B2010);  mem_w(B+'h74, 32'hE51B300C);
    mem_w(B+'h78, 32'hE2833004);  mem_w(B+'h7C, 32'hE1A03103);
    mem_w(B+'h80, 32'hE0823003);  mem_w(B+'h84, 32'hE5933004);
    mem_w(B+'h88, 32'hE51B2008);  mem_w(B+'h8C, 32'hE0823003);
    mem_w(B+'h90, 32'hE50B3008);  mem_w(B+'h94, 32'hE51B300C);
    mem_w(B+'h98, 32'hE2833001);  mem_w(B+'h9C, 32'hE50B300C);
    mem_w(B+'hA0, 32'hE51B300C);  mem_w(B+'hA4, 32'hE3530003);
    mem_w(B+'hA8, 32'hDAFFFFF0);  mem_w(B+'hAC, 32'hE51B3010);
    mem_w(B+'hB0, 32'hE51B2008);  mem_w(B+'hB4, 32'hE5832010);
    mem_w(B+'hB8, 32'hE1A00000);  mem_w(B+'hBC, 32'hE28BD000);
    mem_w(B+'hC0, 32'hE49DB004);  mem_w(B+'hC4, 32'hE12FFF1E);
    $display("  [LOAD] Thread 0 code: 0x%08H - 0x%08H", B, B+'hC4);
end
endtask

task load_thread1_code;
    reg [31:0] B;
begin
    B = T1_CODE;
    mem_w(B+'h00, 32'hE52DB004);  mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C);  mem_w(B+'h0C, 32'hE3A03C02);
    mem_w(B+'h10, 32'hE50B300C);  mem_w(B+'h14, 32'hE51B300C);
    mem_w(B+'h18, 32'hE5933000);  mem_w(B+'h1C, 32'hE3530000);
    mem_w(B+'h20, 32'h1A000016);  mem_w(B+'h24, 32'hE3A03000);
    mem_w(B+'h28, 32'hE50B3008);  mem_w(B+'h2C, 32'hEA00000C);
    mem_w(B+'h30, 32'hE51B300C);  mem_w(B+'h34, 32'hE51B2008);
    mem_w(B+'h38, 32'hE2822002);  mem_w(B+'h3C, 32'hE7932102);
    mem_w(B+'h40, 32'hE59F3048);  mem_w(B+'h44, 32'hE0233002);
    mem_w(B+'h48, 32'hE51B200C);  mem_w(B+'h4C, 32'hE51B1008);
    mem_w(B+'h50, 32'hE2811002);  mem_w(B+'h54, 32'hE7823101);
    mem_w(B+'h58, 32'hE51B3008);  mem_w(B+'h5C, 32'hE2833001);
    mem_w(B+'h60, 32'hE50B3008);  mem_w(B+'h64, 32'hE51B3008);
    mem_w(B+'h68, 32'hE3530003);  mem_w(B+'h6C, 32'hDAFFFFEF);
    mem_w(B+'h70, 32'hE51B300C);  mem_w(B+'h74, 32'hE3A02001);
    mem_w(B+'h78, 32'hE5832000);  mem_w(B+'h7C, 32'hEA000000);
    mem_w(B+'h80, 32'hE1A00000);  mem_w(B+'h84, 32'hE28BD000);
    mem_w(B+'h88, 32'hE49DB004);  mem_w(B+'h8C, 32'hE12FFF1E);
    mem_w(B+'h90, 32'hDEADBEEF);
    $display("  [LOAD] Thread 1 code: 0x%08H - 0x%08H (+literal @0x%08H)",
             B, B+'h8C, B+'h90);
end
endtask

task load_thread2_code;
    reg [31:0] B;
begin
    B = T2_CODE;
    mem_w(B+'h00, 32'hE52DB004);  mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C);  mem_w(B+'h0C, 32'hE3A03C03);
    mem_w(B+'h10, 32'hE50B3008);  mem_w(B+'h14, 32'hE51B3008);
    mem_w(B+'h18, 32'hE5933004);  mem_w(B+'h1C, 32'hE3530001);
    mem_w(B+'h20, 32'h8A000006);  mem_w(B+'h24, 32'hE51B3008);
    mem_w(B+'h28, 32'hE3A02000);  mem_w(B+'h2C, 32'hE5832000);
    mem_w(B+'h30, 32'hE51B3008);  mem_w(B+'h34, 32'hE3A02000);
    mem_w(B+'h38, 32'hE5832004);  mem_w(B+'h3C, 32'hEA000008);
    mem_w(B+'h40, 32'hE51B3008);  mem_w(B+'h44, 32'hE5933004);
    mem_w(B+'h48, 32'hE2432001);  mem_w(B+'h4C, 32'hE51B3008);
    mem_w(B+'h50, 32'hE5832004);  mem_w(B+'h54, 32'hE51B3008);
    mem_w(B+'h58, 32'hE3A02001);  mem_w(B+'h5C, 32'hE5832000);
    mem_w(B+'h60, 32'hE1A00000);  mem_w(B+'h64, 32'hE28BD000);
    mem_w(B+'h68, 32'hE49DB004);  mem_w(B+'h6C, 32'hE12FFF1E);
    $display("  [LOAD] Thread 2 code: 0x%08H - 0x%08H", B, B+'h6C);
end
endtask

task load_thread3_code;
    reg [31:0] B;
begin
    B = T3_CODE;
    mem_w(B+'h00, 32'hE52DB004);  mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C);  mem_w(B+'h0C, 32'hE3A03C04);
    mem_w(B+'h10, 32'hE50B3008);  mem_w(B+'h14, 32'hE51B3008);
    mem_w(B+'h18, 32'hE593300C);  mem_w(B+'h1C, 32'hE3530017);
    mem_w(B+'h20, 32'h1A000003);  mem_w(B+'h24, 32'hE51B3008);
    mem_w(B+'h28, 32'hE3A02002);  mem_w(B+'h2C, 32'hE5832000);
    mem_w(B+'h30, 32'hEA000003);  mem_w(B+'h34, 32'hE51B3008);
    mem_w(B+'h38, 32'hE3A02001);  mem_w(B+'h3C, 32'hE5832000);
    mem_w(B+'h40, 32'hE1A00000);  mem_w(B+'h44, 32'hE28BD000);
    mem_w(B+'h48, 32'hE49DB004);  mem_w(B+'h4C, 32'hE12FFF1E);
    $display("  [LOAD] Thread 3 code: 0x%08H - 0x%08H", B, B+'h4C);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Data Initialisation (scenario-specific, writes local_mem)
// ═══════════════════════════════════════════════════════════════════

task init_data_scenario_A;
begin
    mem_w(32'h0100, 32'h0000_0000); mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'h1111_1111); mem_w(32'h010C, 32'h2222_2222);
    mem_w(32'h0110, 32'h0000_0000); mem_w(32'h0114, 32'h0000_0010);
    mem_w(32'h0118, 32'h0000_0020); mem_w(32'h011C, 32'h0000_0030);
    mem_w(32'h0120, 32'h0000_0040);
    mem_w(32'h0200, 32'h0000_0000); mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hAAAA_AAAA); mem_w(32'h020C, 32'hBBBB_BBBB);
    mem_w(32'h0210, 32'hCCCC_CCCC); mem_w(32'h0214, 32'hDDDD_DDDD);
    mem_w(32'h0300, 32'h0000_0000); mem_w(32'h0304, 32'h0000_0005);
    mem_w(32'h0400, 32'h0000_0000); mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000); mem_w(32'h040C, 32'h0000_0017);
end
endtask

task init_data_scenario_B;
begin
    mem_w(32'h0100, 32'h0000_0000); mem_w(32'h0104, 32'h0000_00BB);
    mem_w(32'h0108, 32'h1111_1111); mem_w(32'h010C, 32'h2222_2222);
    mem_w(32'h0110, 32'h0000_0000); mem_w(32'h0114, 32'h0000_0010);
    mem_w(32'h0118, 32'h0000_0020); mem_w(32'h011C, 32'h0000_0030);
    mem_w(32'h0120, 32'h0000_0040);
    mem_w(32'h0200, 32'h0000_0001); mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hAAAA_AAAA); mem_w(32'h020C, 32'hBBBB_BBBB);
    mem_w(32'h0210, 32'hCCCC_CCCC); mem_w(32'h0214, 32'hDDDD_DDDD);
    mem_w(32'h0300, 32'hFFFF_FFFF); mem_w(32'h0304, 32'h0000_0001);
    mem_w(32'h0400, 32'h0000_0000); mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000); mem_w(32'h040C, 32'h0000_002A);
end
endtask

task init_data_scenario_C;
begin
    mem_w(32'h0100, 32'h0000_0000); mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'hFFFF_FFFF); mem_w(32'h010C, 32'h0000_0001);
    mem_w(32'h0110, 32'h0000_0000); mem_w(32'h0114, 32'h7FFF_FFFF);
    mem_w(32'h0118, 32'h0000_0001); mem_w(32'h011C, 32'h8000_0000);
    mem_w(32'h0120, 32'h0000_0000);
    mem_w(32'h0200, 32'h0000_0000); mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hDEAD_BEEF); mem_w(32'h020C, 32'h0000_0000);
    mem_w(32'h0210, 32'hFFFF_FFFF); mem_w(32'h0214, 32'h1234_5678);
    mem_w(32'h0300, 32'hAAAA_AAAA); mem_w(32'h0304, 32'h0000_0000);
    mem_w(32'h0400, 32'h0000_0000); mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000); mem_w(32'h040C, 32'h0000_0017);
end
endtask

task init_data_scenario_D;
begin
    mem_w(32'h0100, 32'h0000_0000); mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'hAAAA_0000); mem_w(32'h010C, 32'h0000_BBBB);
    mem_w(32'h0110, 32'h0000_0000); mem_w(32'h0114, 32'h0000_0001);
    mem_w(32'h0118, 32'h0000_0002); mem_w(32'h011C, 32'h0000_0003);
    mem_w(32'h0120, 32'h0000_0004);
    mem_w(32'h0200, 32'h0000_0000); mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'h0102_0304); mem_w(32'h020C, 32'h0506_0708);
    mem_w(32'h0210, 32'h090A_0B0C); mem_w(32'h0214, 32'h0D0E_0F10);
    mem_w(32'h0300, 32'h0000_0000); mem_w(32'h0304, 32'h0000_000A);
    mem_w(32'h0400, 32'h0000_0000); mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000); mem_w(32'h040C, 32'h0000_0017);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Prepare Scenario Image — code + sentinels (common to all)
// ═══════════════════════════════════════════════════════════════════

task prepare_common_image;
begin
    mem_clear();
    load_thread0_code();
    load_thread1_code();
    load_thread2_code();
    load_thread3_code();
    mem_w(T0_RET, SENTINEL);
    mem_w(T1_RET, SENTINEL);
    mem_w(T2_RET, SENTINEL);
    mem_w(T3_RET, SENTINEL);
    $display("  [LOAD] Sentinels placed at T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
             T0_RET, T1_RET, T2_RET, T3_RET);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Run Test — load image, init CPU, execute, wait, stop
// ═══════════════════════════════════════════════════════════════════
task run_mt_test_soc;
    output integer cycles_used;
    integer t;
    reg [3:0] sent_ok;
    integer sent_cnt [0:3];
    integer wait_cnt;
begin
    do_reset();
    load_image_to_soc();
    init_cpu_mt_regs();
    @(posedge clk); #1;
    init_cpu_mt_regs();

    @(posedge clk); #1;
    start = 1'b1;

    u_soc_mt.u_cpu_mt.pc_thread[0] = T0_CODE;
    u_soc_mt.u_cpu_mt.pc_thread[1] = T1_CODE;
    u_soc_mt.u_cpu_mt.pc_thread[2] = T2_CODE;
    u_soc_mt.u_cpu_mt.pc_thread[3] = T3_CODE;

    $display("  [RUN]  start=1, CPU executing...");

    cycle_cnt = 0;
    for (t = 0; t < 4; t = t + 1) sent_cnt[t] = 0;
    sent_ok = 4'b0000;

    begin : run_loop
        forever begin
            @(posedge clk); #1;
            cycle_cnt = cycle_cnt + 1;

            if (TRACE_EN && cycle_cnt > 0 && cycle_cnt <= TRACE_LIMIT)
                $display("[C%05d] PC: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         cycle_cnt,
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));

            if (cycle_cnt > 20) begin
                if (get_thread_pc(2'd0) == T0_RET) sent_cnt[0] = sent_cnt[0] + 1;
                else sent_cnt[0] = 0;
                if (get_thread_pc(2'd1) == T1_RET) sent_cnt[1] = sent_cnt[1] + 1;
                else sent_cnt[1] = 0;
                if (get_thread_pc(2'd2) == T2_RET) sent_cnt[2] = sent_cnt[2] + 1;
                else sent_cnt[2] = 0;
                if (get_thread_pc(2'd3) == T3_RET) sent_cnt[3] = sent_cnt[3] + 1;
                else sent_cnt[3] = 0;

                sent_ok[0] = (sent_cnt[0] > 30);
                sent_ok[1] = (sent_cnt[1] > 30);
                sent_ok[2] = (sent_cnt[2] > 30);
                sent_ok[3] = (sent_cnt[3] > 30);
            end

            if (sent_ok == 4'b1111) begin
                $display("  [DONE] All 4 threads at sentinel, cycle %0d", cycle_cnt);
                repeat (10) @(posedge clk);
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            if (cycle_cnt > 0 && (cycle_cnt % STATUS_INTERVAL == 0))
                $display("  [STATUS C%05d] sentinel=%04b  PC: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         cycle_cnt, sent_ok,
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));

            if (cycle_cnt >= MAX_CYCLES) begin
                $display("  *** TIMEOUT after %0d cycles ***", MAX_CYCLES);
                dump_all_threads();
                cycles_used = cycle_cnt;
                disable run_loop;
            end
        end
    end

    start = 1'b0;
    @(posedge clk); #1;
    wait_cnt = 0;
    while (!req_rdy && wait_cnt < 100) begin
        @(posedge clk); #1;
        wait_cnt = wait_cnt + 1;
    end
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  T E S T   S C E N A R I O S   (Phase B — hierarchical init)
// ═══════════════════════════════════════════════════════════════════

task test_scenario_A;
    integer cyc;
begin
    section_start("Scenario A: Normal Path (all four threads)");
    prepare_common_image();
    init_data_scenario_A();
    run_mt_test_soc(cyc);

    check_mem_mmio(32'h0108, 32'h2222_2222, "T0: src<-dst after swap");
    check_mem_mmio(32'h010C, 32'h1111_1111, "T0: dst<-src after swap");
    check_mem_mmio(32'h0110, 32'h0000_00A0, "T0: checksum = 0xA0");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: done flag = 1");
    check_mem_mmio(32'h0208, 32'h7407_1445, "T1: 0xAAAAAAAA ^ 0xDEADBEEF");
    check_mem_mmio(32'h020C, 32'h6516_0554, "T1: 0xBBBBBBBB ^ 0xDEADBEEF");
    check_mem_mmio(32'h0210, 32'h1261_7223, "T1: 0xCCCCCCCC ^ 0xDEADBEEF");
    check_mem_mmio(32'h0214, 32'h0370_6332, "T1: 0xDDDDDDDD ^ 0xDEADBEEF");
    check_mem_mmio(32'h0300, 32'h0000_0001, "T2: status = 1");
    check_mem_mmio(32'h0304, 32'h0000_0004, "T2: counter = 4");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result = 2 (match)");
    check_reg_hier(2'd0, 4'd13, T0_SP, "T0: SP restored");
    check_reg_hier(2'd1, 4'd13, T1_SP, "T1: SP restored");
    check_reg_hier(2'd2, 4'd13, T2_SP, "T2: SP restored");
    check_reg_hier(2'd3, 4'd13, T3_SP, "T3: SP restored");

    section_end();
end
endtask

task test_scenario_B;
    integer cyc;
begin
    section_start("Scenario B: Alternate / Error Paths");
    prepare_common_image();
    init_data_scenario_B();
    run_mt_test_soc(cyc);

    check_mem_mmio(32'h0110, 32'hFFFF_FFFF, "T0: result = -1 (error)");
    check_mem_mmio(32'h0108, 32'h1111_1111, "T0: src unchanged");
    check_mem_mmio(32'h010C, 32'h2222_2222, "T0: dst unchanged");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: flag unchanged = 1");
    check_mem_mmio(32'h0208, 32'hAAAA_AAAA, "T1: data[0] unmodified");
    check_mem_mmio(32'h020C, 32'hBBBB_BBBB, "T1: data[1] unmodified");
    check_mem_mmio(32'h0210, 32'hCCCC_CCCC, "T1: data[2] unmodified");
    check_mem_mmio(32'h0214, 32'hDDDD_DDDD, "T1: data[3] unmodified");
    check_mem_mmio(32'h0300, 32'h0000_0000, "T2: status = 0");
    check_mem_mmio(32'h0304, 32'h0000_0000, "T2: counter = 0");
    check_mem_mmio(32'h0400, 32'h0000_0001, "T3: result = 1 (no match)");

    section_end();
end
endtask

task test_scenario_C;
    integer cyc;
begin
    section_start("Scenario C: Edge-Case Data Values");
    prepare_common_image();
    init_data_scenario_C();
    run_mt_test_soc(cyc);

    check_mem_mmio(32'h0108, 32'h0000_0001, "T0: src<-dst swapped");
    check_mem_mmio(32'h010C, 32'hFFFF_FFFF, "T0: dst<-src swapped");
    check_mem_mmio(32'h0110, 32'h0000_0000, "T0: checksum wraps to 0");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: done flag");
    check_mem_mmio(32'h0208, 32'h0000_0000, "T1: DEADBEEF^DEADBEEF=0");
    check_mem_mmio(32'h020C, 32'hDEAD_BEEF, "T1: 0^DEADBEEF");
    check_mem_mmio(32'h0210, 32'h2152_4110, "T1: FFFFFFFF^DEADBEEF");
    check_mem_mmio(32'h0214, 32'hCC99_E897, "T1: 12345678^DEADBEEF");
    check_mem_mmio(32'h0300, 32'h0000_0000, "T2: status=0");
    check_mem_mmio(32'h0304, 32'h0000_0000, "T2: counter=0");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result=2 (match)");

    section_end();
end
endtask

task test_scenario_D;
    integer cyc;
begin
    section_start("Scenario D: Thread Isolation Verification");
    prepare_common_image();
    init_data_scenario_D();
    run_mt_test_soc(cyc);

    check_mem_mmio(32'h0108, 32'h0000_BBBB, "T0: src<-dst");
    check_mem_mmio(32'h010C, 32'hAAAA_0000, "T0: dst<-src");
    check_mem_mmio(32'h0110, 32'h0000_000A, "T0: checksum=10");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: flag=1");
    check_mem_mmio(32'h0208, 32'hDFAF_BDEB, "T1: 01020304^DEADBEEF");
    check_mem_mmio(32'h020C, 32'hDBAB_B9E7, "T1: 05060708^DEADBEEF");
    check_mem_mmio(32'h0210, 32'hD7A7_B5E3, "T1: 090A0B0C^DEADBEEF");
    check_mem_mmio(32'h0214, 32'hD3A3_B1FF, "T1: 0D0E0F10^DEADBEEF");
    check_mem_mmio(32'h0300, 32'h0000_0001, "T2: status=1");
    check_mem_mmio(32'h0304, 32'h0000_0009, "T2: counter=9");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result=2 (match)");
    check_mem_mmio(32'h0104, 32'h0000_00AA, "T0 header not corrupted");
    check_mem_mmio(32'h040C, 32'h0000_0017, "T3 field not corrupted");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
//  PHASE C & D — BOOTSTRAP / CANARY / DIAGNOSTIC TESTS
//  (Replicating socreg_mt script's canary, rotcanary, widecanary,
//   and bootstrap-based execution flow)
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════

// ── ARM branch encoding: B <target_word> from <from_word> ────────
function [31:0] make_arm_branch_enc;
    input integer from_word;
    input integer to_word;
    reg signed [31:0] off;
begin
    off = (to_word - from_word) * 4 - 8;
    make_arm_branch_enc = {4'hE, 4'hA, off[25:2]};
end
endfunction

// ── Clear all CPU registers to zero (no SP/LR setup) ─────────────
task clear_cpu_regs;
    integer ct, cr;
begin
    for (ct = 0; ct < 4; ct = ct + 1)
        for (cr = 0; cr < 16; cr = cr + 1)
            case (ct)
                0: u_soc_mt.u_cpu_mt.THREAD_RF[0].u_rf.regs[cr] = {`DATA_WIDTH{1'b0}};
                1: u_soc_mt.u_cpu_mt.THREAD_RF[1].u_rf.regs[cr] = {`DATA_WIDTH{1'b0}};
                2: u_soc_mt.u_cpu_mt.THREAD_RF[2].u_rf.regs[cr] = {`DATA_WIDTH{1'b0}};
                3: u_soc_mt.u_cpu_mt.THREAD_RF[3].u_rf.regs[cr] = {`DATA_WIDTH{1'b0}};
            endcase
    for (ct = 0; ct < 4; ct = ct + 1)
        u_soc_mt.u_cpu_mt.cpsr_flags[ct] = 4'b0;
end
endtask

// ── Run CPU for fixed cycles, then stop + wait for MMIO ──────────
task run_fixed_cycles;
    input integer num_cycles;
    integer wc;
begin
    @(posedge clk); #1;
    start = 1'b1;
    repeat (num_cycles) @(posedge clk);
    start = 1'b0;
    @(posedge clk); #1;
    wc = 0;
    while (!req_rdy && wc < 100) begin
        @(posedge clk); #1;
        wc = wc + 1;
    end
    if (!req_rdy)
        $display("  WARNING: req_rdy not high after run_fixed_cycles");
end
endtask

// ═══════════════════════════════════════════════════════════════════
//  C1: CANARY — Minimal CPU Execution (replicates script canary_test)
//  Program: MOV R0,#0xFF; MOV R1,#0; STR R0,[R1]; B .
//  In 4-thread CPU: T0 starts at word 0, T1 at 1, T2 at 2, T3 at 3.
//  T0 writes last (round-robin ordering) so DMEM[0] = T0's value.
// ═══════════════════════════════════════════════════════════════════
task test_canary;
begin
    section_start("C1: CANARY — Minimal CPU Execution");

    $display("  Program: MOV R0,#0xFF; MOV R1,#0; STR R0,[R1]; B .");
    $display("  Expect:  DMEM[byte 0x0000] = 0x000000FF");

    do_reset();

    // Load program to IMEM via MMIO
    mmio_wr(IMEM_BASE | 32'd0, 32'hE3A000FF);  // MOV R0, #0xFF
    mmio_wr(IMEM_BASE | 32'd1, 32'hE3A01000);  // MOV R1, #0
    mmio_wr(IMEM_BASE | 32'd2, 32'hE5810000);  // STR R0, [R1]
    mmio_wr(IMEM_BASE | 32'd3, SENTINEL);       // B .
    for (i = 4; i < 8; i = i + 1)
        mmio_wr(IMEM_BASE | i, SENTINEL);

    // Clear DMEM[0]
    mmio_wr(DMEM_BASE | 32'd0, 32'h0000_0000);

    // Clear all CPU regs; PCs stay at reset values (0, 4, 8, C)
    clear_cpu_regs();

    // Run 200 cycles (plenty for 4 instructions + pipeline drain)
    run_fixed_cycles(200);

    // Verify
    check_mem_mmio(32'h0000, 32'h0000_00FF, "Canary: CPU wrote 0xFF to DMEM[0]");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  C2: MULTI-CANARY — Multiple DMEM stores (replicates script)
//  Verifies CPU can write to several addresses + ALU works (ADD).
// ═══════════════════════════════════════════════════════════════════
task test_multi_canary;
begin
    section_start("C2: MULTI-CANARY — Multiple DMEM Addresses");

    do_reset();

    // Pre-fill DMEM[0..33] with sentinel pattern
    for (i = 0; i < 34; i = i + 1)
        mmio_wr(DMEM_BASE | i, 32'hCAFE_0000 | i);

    // Load program to IMEM
    mmio_wr(IMEM_BASE | 32'd0, 32'hE3A00042);  // MOV R0, #0x42
    mmio_wr(IMEM_BASE | 32'd1, 32'hE3A01000);  // MOV R1, #0
    mmio_wr(IMEM_BASE | 32'd2, 32'hE5810000);  // STR R0, [R1, #0]      → DMEM[0]
    mmio_wr(IMEM_BASE | 32'd3, 32'hE5810004);  // STR R0, [R1, #4]      → DMEM[1]
    mmio_wr(IMEM_BASE | 32'd4, 32'hE5810040);  // STR R0, [R1, #0x40]   → DMEM[16]
    mmio_wr(IMEM_BASE | 32'd5, 32'hE5810080);  // STR R0, [R1, #0x80]   → DMEM[32]
    mmio_wr(IMEM_BASE | 32'd6, 32'hE2800001);  // ADD R0, R0, #1
    mmio_wr(IMEM_BASE | 32'd7, 32'hE5810084);  // STR R0, [R1, #0x84]   → DMEM[33]
    mmio_wr(IMEM_BASE | 32'd8, SENTINEL);       // B .
    for (i = 9; i < 16; i = i + 1)
        mmio_wr(IMEM_BASE | i, SENTINEL);

    clear_cpu_regs();
    run_fixed_cycles(300);

    check_mem_mmio(32'h00, 32'h0000_0042, "STR to byte 0x00");
    check_mem_mmio(32'h04, 32'h0000_0042, "STR to byte 0x04");
    check_mem_mmio(32'h40, 32'h0000_0042, "STR to byte 0x40");
    check_mem_mmio(32'h80, 32'h0000_0042, "STR to byte 0x80");
    check_mem_mmio(32'h84, 32'h0000_0043, "ADD+STR to byte 0x84 (ALU proof)");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  C3: ROTATION CANARY — Barrel shifter rotated-immediate test
//  (Replicates script rotation_canary_test)
//
//  Tests whether MOV Rd, #imm with rotate_imm != 0 works.
//  The bootstrap uses e.g. MOV R13, #0x1000 (rotate=C, imm=0x10).
//  If barrel shifter ignores rotation, R13 gets 0x10 instead of
//  0x1000 and ALL stack accesses go to wrong addresses.
//
//  T0 executes the full program (words 0-18).  T1/T2/T3 start at
//  words 1/2/3 but T0's writes happen last due to round-robin,
//  so DMEM values reflect T0's register state.
// ═══════════════════════════════════════════════════════════════════
task test_rotation_canary;
begin
    section_start("C3: ROTATION CANARY — Barrel Shifter Test");

    $display("  Tests rotated immediates (rotate_imm != 0) in data-processing.");
    $display("  Also tests register-shift (LSL) alternative.");

    do_reset();

    // Clear DMEM[0..7]
    for (i = 0; i < 8; i = i + 1)
        mmio_wr(DMEM_BASE | i, 32'hBBBB_0000 | i);

    // Load rotation test program (19 instructions)
    mmio_wr(IMEM_BASE | 32'd0,  32'hE3A00042);  // MOV R0, #0x42           (rot=0, CONTROL)
    mmio_wr(IMEM_BASE | 32'd1,  32'hE3A01000);  // MOV R1, #0              (rot=0)
    mmio_wr(IMEM_BASE | 32'd2,  32'hE5810000);  // STR R0, [R1, #0]        → DMEM[0]=0x42

    // Test 1: MOV with rotate_imm=C (same as thread code's MOV R3, #0x100)
    mmio_wr(IMEM_BASE | 32'd3,  32'hE3A02C01);  // MOV R2, #0x100          (rot=C, imm=1)
    mmio_wr(IMEM_BASE | 32'd4,  32'hE5812008);  // STR R2, [R1, #8]        → DMEM[2]=0x100

    // Test 2: MOV with rotate_imm=A
    mmio_wr(IMEM_BASE | 32'd5,  32'hE3A03A01);  // MOV R3, #0x1000         (rot=A, imm=1)
    mmio_wr(IMEM_BASE | 32'd6,  32'hE581300C);  // STR R3, [R1, #12]       → DMEM[3]=0x1000

    // Test 3: Exact bootstrap encoding for T0 SP
    mmio_wr(IMEM_BASE | 32'd7,  32'hE3A0DC10);  // MOV R13, rot=C, imm=0x10 → 0x1000
    mmio_wr(IMEM_BASE | 32'd8,  32'hE581D010);  // STR R13, [R1, #16]      → DMEM[4]=0x1000

    // Test 4: Bootstrap LR encoding (MOV+ORR)
    mmio_wr(IMEM_BASE | 32'd9,  32'hE3A0EC05);  // MOV R14, rot=C, imm=5   → 0x500
    mmio_wr(IMEM_BASE | 32'd10, 32'hE38EE0FC);  // ORR R14, R14, #0xFC     → 0x5FC
    mmio_wr(IMEM_BASE | 32'd11, 32'hE581E014);  // STR R14, [R1, #20]      → DMEM[5]=0x5FC

    // Test 5: Register-shift alternative (LSL, no immediate rotation)
    mmio_wr(IMEM_BASE | 32'd12, 32'hE3A04010);  // MOV R4, #0x10           (rot=0, safe)
    mmio_wr(IMEM_BASE | 32'd13, 32'hE1A04404);  // MOV R4, R4, LSL #8     → 0x1000
    mmio_wr(IMEM_BASE | 32'd14, 32'hE5814018);  // STR R4, [R1, #24]      → DMEM[6]=0x1000

    // Completion stamp
    mmio_wr(IMEM_BASE | 32'd15, 32'hE2800001);  // ADD R0, R0, #1
    mmio_wr(IMEM_BASE | 32'd16, 32'hE5810004);  // STR R0, [R1, #4]       → DMEM[1]=0x43
    mmio_wr(IMEM_BASE | 32'd17, SENTINEL);       // B .
    mmio_wr(IMEM_BASE | 32'd18, SENTINEL);
    mmio_wr(IMEM_BASE | 32'd19, SENTINEL);

    clear_cpu_regs();
    run_fixed_cycles(400);

    // Dump results
    $display("");
    $display("  -- Rotation Test Results --");
    dump_mem_mmio(32'h0000, 7);

    // Formal checks
    check_mem_mmio(32'h00, 32'h0000_0042, "Execution proof (MOV #0x42, rot=0)");
    check_mem_mmio(32'h04, 32'h0000_0043, "Completion stamp (ADD+STR)");
    check_mem_mmio(32'h08, 32'h0000_0100, "MOV with rotate_imm=C, imm=1 → 0x100");
    check_mem_mmio(32'h0C, 32'h0000_1000, "MOV with rotate_imm=A, imm=1 → 0x1000");
    check_mem_mmio(32'h10, 32'h0000_1000, "Bootstrap SP encoding (rot=C, imm=0x10)");
    check_mem_mmio(32'h14, 32'h0000_05FC, "Bootstrap LR encoding (MOV rot + ORR)");
    check_mem_mmio(32'h18, 32'h0000_1000, "Register-shift LSL alternative");

    // Diagnostic interpretation
    mmio_rd(DMEM_BASE | 32'd2, rd_data);
    if (rd_data[31:0] == 32'h0000_0001) begin
        $display("");
        $display("  >>> BARREL SHIFTER BUG: rotation IGNORED (got raw imm8) <<<");
        $display("  >>> Bootstrap will set SP/LR incorrectly <<<");
    end

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  C4: WIDE CANARY — Scan DMEM for where CPU writes actually land
//  (Replicates script wide_canary_test)
// ═══════════════════════════════════════════════════════════════════
task test_wide_canary;
    integer changes;
    reg [`MMIO_DATA_WIDTH-1:0] val;
begin
    section_start("C4: WIDE CANARY — DMEM Write Location Scan");

    do_reset();

    // Pre-fill DMEM[0..511] with known pattern
    $display("  Pre-filling DMEM[0..511] with 0xCAFExxxx...");
    for (i = 0; i < 512; i = i + 1)
        mmio_wr(DMEM_BASE | i, 32'hCAFE_0000 | i[15:0]);

    // Program: write marker value using both broken-rotation SP and
    // correct-rotation SP, then stamp completion
    mmio_wr(IMEM_BASE | 32'd0, 32'hE3A000AA);   // MOV R0, #0xAA
    mmio_wr(IMEM_BASE | 32'd1, 32'hE3A01000);   // MOV R1, #0
    mmio_wr(IMEM_BASE | 32'd2, 32'hE3A0D010);   // MOV R13, #0x10  (broken rotation would give this)
    mmio_wr(IMEM_BASE | 32'd3, 32'hE52D0004);   // STR R0, [R13, #-4]!  → byte 0x0C if broken
    mmio_wr(IMEM_BASE | 32'd4, 32'hE3A0DC10);   // MOV R13, #0x1000 (correct rotation)
    mmio_wr(IMEM_BASE | 32'd5, 32'hE52D0004);   // STR R0, [R13, #-4]!  → byte 0xFFC if correct
    mmio_wr(IMEM_BASE | 32'd6, 32'hE3A000BB);   // MOV R0, #0xBB
    mmio_wr(IMEM_BASE | 32'd7, 32'hE5810000);   // STR R0, [R1]  → byte 0x00
    mmio_wr(IMEM_BASE | 32'd8, SENTINEL);
    for (i = 9; i < 16; i = i + 1)
        mmio_wr(IMEM_BASE | i, SENTINEL);

    clear_cpu_regs();
    run_fixed_cycles(300);

    // Scan for changes
    $display("");
    $display("  Scanning DMEM[0..511] for changes from 0xCAFExxxx:");
    changes = 0;
    for (i = 0; i < 512; i = i + 1) begin
        mmio_rd(DMEM_BASE | i, val);
        if (val[31:0] != (32'hCAFE_0000 | i[15:0])) begin
            $display("    [CHANGED] DMEM[W%0d / byte 0x%04H] = 0x%08H (was 0x%08H)",
                     i, i * 4, val[31:0], 32'hCAFE_0000 | i[15:0]);
            changes = changes + 1;
        end
    end
    $display("  Total changes: %0d of 512 words", changes);

    if (changes == 0)
        $display("  WARNING: No changes detected — CPU may not have executed.");
    else begin
        // Interpret results
        mmio_rd(DMEM_BASE | 32'd3, val);  // word 3 = byte 0x0C (broken rotation target)
        if (val[31:0] == 32'h0000_00AA)
            $display("  NOTE: Word 3 (byte 0x0C) has marker — broken rotation confirmed!");
    end

    // Basic check: completion stamp should be at DMEM[0]
    check_mem_mmio(32'h0000, 32'h0000_00BB, "Wide canary: completion stamp at DMEM[0]");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  BOOTSTRAP INJECTION — Write dispatch + init code to IMEM via MMIO
//  (Replicates script inject_bootstrap / gen_thread_bootstrap)
//
//  Dispatch table at IMEM[0..3]: B boot_T0, B boot_T1, ...
//  Per-thread init (17 words each):
//    MOV R0-R12, #0
//    MOV R13, #SP      (rotated immediate)
//    MOV R14, #LR_hi   (rotated immediate)
//    ORR R14, R14, #LR_lo
//    B code_addr
//
//  This writes to IMEM ONLY (not DMEM), so it can be called AFTER
//  load_image_to_soc without corrupting DMEM data regions.
// ═══════════════════════════════════════════════════════════════════
task inject_bootstrap_to_imem;
    integer ti, ri, bs, wi;
    reg [31:0] sp_v, lr_v, code_v;
begin
    $display("  [BOOT] Injecting standard bootstrap (rotated immediates)...");

    for (ti = 0; ti < 4; ti = ti + 1) begin
        bs = 4 + ti * 17;

        // Dispatch branch at word ti
        mmio_wr(IMEM_BASE | ti, make_arm_branch_enc(ti, bs));

        case (ti)
            0: begin sp_v = T0_SP; lr_v = T0_RET; code_v = T0_CODE; end
            1: begin sp_v = T1_SP; lr_v = T1_RET; code_v = T1_CODE; end
            2: begin sp_v = T2_SP; lr_v = T2_RET; code_v = T2_CODE; end
            3: begin sp_v = T3_SP; lr_v = T3_RET; code_v = T3_CODE; end
        endcase

        // MOV R0-R12, #0
        for (ri = 0; ri < 13; ri = ri + 1)
            mmio_wr(IMEM_BASE | (bs + ri), 32'hE3A00000 | (ri << 12));

        // MOV R13, #SP (rotate_imm=C, imm8 = SP[15:8])
        mmio_wr(IMEM_BASE | (bs + 13), 32'hE3A0DC00 | {24'h0, sp_v[15:8]});

        // MOV R14, #LR_hi (rotate_imm=C, imm8 = LR[15:8])
        mmio_wr(IMEM_BASE | (bs + 14), 32'hE3A0EC00 | {24'h0, lr_v[15:8]});

        // ORR R14, R14, #LR_lo
        mmio_wr(IMEM_BASE | (bs + 15), 32'hE38EE000 | {24'h0, lr_v[7:0]});

        // B code_addr
        mmio_wr(IMEM_BASE | (bs + 16), make_arm_branch_enc(bs + 16, code_v / 4));

        $display("    T%0d: dispatch@W%0d -> boot@W%0d..W%0d -> CODE@0x%04H  SP=0x%04H LR=0x%04H",
                 ti, ti, bs, bs + 16, code_v[15:0], sp_v[15:0], lr_v[15:0]);
    end

    $display("  [BOOT] 72 words injected (4 dispatch + 4 x 17 init)");
end
endtask

// ── No-rotate bootstrap variant ──────────────────────────────────
//  Uses MOV Rd, #(val>>8) + MOV Rd, Rd, LSL #8 [+ ORR Rd, Rd, #lo]
//  instead of rotated immediates for SP/LR.
//  19 words per thread (SP low byte = 0 → no ORR; LR low byte != 0 → ORR)
task inject_bootstrap_norotate_to_imem;
    integer ti, ri, bs, wi;
    reg [31:0] sp_v, lr_v, code_v;
    integer bstarts [0:3];
begin
    $display("  [BOOT-NR] Injecting no-rotate bootstrap (MOV+LSL for SP/LR)...");

    // Pre-compute starts: 4 dispatch + 19 per thread
    bstarts[0] = 4;
    bstarts[1] = 4 + 19;
    bstarts[2] = 4 + 38;
    bstarts[3] = 4 + 57;

    // Dispatch table
    for (ti = 0; ti < 4; ti = ti + 1)
        mmio_wr(IMEM_BASE | ti, make_arm_branch_enc(ti, bstarts[ti]));

    for (ti = 0; ti < 4; ti = ti + 1) begin
        wi = bstarts[ti];

        case (ti)
            0: begin sp_v = T0_SP; lr_v = T0_RET; code_v = T0_CODE; end
            1: begin sp_v = T1_SP; lr_v = T1_RET; code_v = T1_CODE; end
            2: begin sp_v = T2_SP; lr_v = T2_RET; code_v = T2_CODE; end
            3: begin sp_v = T3_SP; lr_v = T3_RET; code_v = T3_CODE; end
        endcase

        // MOV R0-R12, #0
        for (ri = 0; ri < 13; ri = ri + 1) begin
            mmio_wr(IMEM_BASE | wi, 32'hE3A00000 | (ri << 12));
            wi = wi + 1;
        end

        // SP: MOV R13, #(SP>>8); MOV R13, R13, LSL #8
        mmio_wr(IMEM_BASE | wi, 32'hE3A0D000 | {24'h0, sp_v[15:8]});  wi = wi + 1;
        mmio_wr(IMEM_BASE | wi, 32'hE1A0D40D);                         wi = wi + 1;  // LSL #8

        // LR: MOV R14, #(LR>>8); MOV R14, R14, LSL #8; ORR R14, R14, #LR_lo
        mmio_wr(IMEM_BASE | wi, 32'hE3A0E000 | {24'h0, lr_v[15:8]});  wi = wi + 1;
        mmio_wr(IMEM_BASE | wi, 32'hE1A0E40E);                         wi = wi + 1;  // LSL #8
        mmio_wr(IMEM_BASE | wi, 32'hE38EE000 | {24'h0, lr_v[7:0]});   wi = wi + 1;  // ORR

        // B code_addr
        mmio_wr(IMEM_BASE | wi, make_arm_branch_enc(wi, code_v / 4));  wi = wi + 1;

        $display("    T%0d: W%0d..W%0d -> CODE@0x%04H  SP=0x%04H LR=0x%04H",
                 ti, bstarts[ti], wi - 1, code_v[15:0], sp_v[15:0], lr_v[15:0]);
    end

    $display("  [BOOT-NR] %0d words injected (4 dispatch + 4 x 19 init)", 4 + 19 * 4);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Bootstrap-based run task — PCs start at 0, 4, 8, C (hardware
//  default).  Bootstrap dispatch → init → B code → execute → BX LR
//  → sentinel.  Same sentinel detection as run_mt_test_soc.
// ═══════════════════════════════════════════════════════════════════
task run_bootstrap_test_soc;
    input integer use_norotate;
    output integer cycles_used;
    integer t;
    reg [3:0] sent_ok;
    integer sent_cnt [0:3];
    integer wait_cnt;
begin
    // Reset
    do_reset();

    // Load code + data + sentinels (from local_mem) to IMEM & DMEM
    load_image_to_soc();

    // Inject bootstrap to IMEM ONLY (overwrites IMEM[0..71/79],
    // does NOT touch DMEM, so data at 0x0100+ is preserved)
    if (use_norotate)
        inject_bootstrap_norotate_to_imem();
    else
        inject_bootstrap_to_imem();

    // Clear all CPU regs (bootstrap code will set them)
    clear_cpu_regs();
    @(posedge clk); #1;
    clear_cpu_regs();

    $display("  [BOOT] PCs at reset defaults: T0@0x0000 T1@0x0004 T2@0x0008 T3@0x000C");
    $display("  [BOOT] Flow: dispatch -> init R0-R14 -> B code -> execute -> BX LR -> sentinel");

    // Start CPU — do NOT override PCs (bootstrap dispatch handles routing)
    @(posedge clk); #1;
    start = 1'b1;

    $display("  [RUN]  start=1, bootstrap + thread code executing...");

    // Sentinel detection (identical to run_mt_test_soc)
    cycle_cnt = 0;
    for (t = 0; t < 4; t = t + 1) sent_cnt[t] = 0;
    sent_ok = 4'b0000;

    begin : boot_run_loop
        forever begin
            @(posedge clk); #1;
            cycle_cnt = cycle_cnt + 1;

            if (TRACE_EN && cycle_cnt > 0 && cycle_cnt <= TRACE_LIMIT)
                $display("[C%05d] PC: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         cycle_cnt,
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));

            // Bootstrap takes ~20 instructions to reach thread code,
            // then thread code runs.  Start sentinel check after 80 cycles.
            if (cycle_cnt > 80) begin
                if (get_thread_pc(2'd0) == T0_RET) sent_cnt[0] = sent_cnt[0] + 1;
                else sent_cnt[0] = 0;
                if (get_thread_pc(2'd1) == T1_RET) sent_cnt[1] = sent_cnt[1] + 1;
                else sent_cnt[1] = 0;
                if (get_thread_pc(2'd2) == T2_RET) sent_cnt[2] = sent_cnt[2] + 1;
                else sent_cnt[2] = 0;
                if (get_thread_pc(2'd3) == T3_RET) sent_cnt[3] = sent_cnt[3] + 1;
                else sent_cnt[3] = 0;

                sent_ok[0] = (sent_cnt[0] > 30);
                sent_ok[1] = (sent_cnt[1] > 30);
                sent_ok[2] = (sent_cnt[2] > 30);
                sent_ok[3] = (sent_cnt[3] > 30);
            end

            if (sent_ok == 4'b1111) begin
                $display("  [DONE] All 4 threads at sentinel via bootstrap, cycle %0d", cycle_cnt);
                repeat (10) @(posedge clk);
                cycles_used = cycle_cnt;
                disable boot_run_loop;
            end

            if (cycle_cnt > 0 && (cycle_cnt % STATUS_INTERVAL == 0))
                $display("  [STATUS C%05d] sentinel=%04b  PC: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         cycle_cnt, sent_ok,
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));

            if (cycle_cnt >= MAX_CYCLES) begin
                $display("  *** BOOTSTRAP TIMEOUT after %0d cycles ***", MAX_CYCLES);
                $display("  Sentinel status: %04b", sent_ok);
                $display("  PCs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));
                dump_all_threads();
                cycles_used = cycle_cnt;
                disable boot_run_loop;
            end
        end
    end

    start = 1'b0;
    @(posedge clk); #1;
    wait_cnt = 0;
    while (!req_rdy && wait_cnt < 100) begin
        @(posedge clk); #1;
        wait_cnt = wait_cnt + 1;
    end
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  D1-D4: BOOTSTRAP SCENARIO TESTS
//  Same thread code + data + expected results as Phase B scenarios,
//  but using bootstrap code for register init instead of
//  hierarchical access.  This is the execution model used by the
//  FPGA (socreg_mt script) where no hierarchical access exists.
// ═══════════════════════════════════════════════════════════════════

task test_bootstrap_scenario_A;
    integer cyc;
begin
    section_start("D1: Bootstrap Scenario A — Normal Path");

    prepare_common_image();
    init_data_scenario_A();
    run_bootstrap_test_soc(0, cyc);   // 0 = use standard (rotated) bootstrap

    $display("  -- Thread 0: Packet Processing (bootstrap) --");
    check_mem_mmio(32'h0108, 32'h2222_2222, "T0: src<-dst after swap");
    check_mem_mmio(32'h010C, 32'h1111_1111, "T0: dst<-src after swap");
    check_mem_mmio(32'h0110, 32'h0000_00A0, "T0: checksum = 0xA0");

    $display("  -- Thread 1: XOR Encryption (bootstrap) --");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: done flag = 1");
    check_mem_mmio(32'h0208, 32'h7407_1445, "T1: 0xAAAAAAAA ^ 0xDEADBEEF");
    check_mem_mmio(32'h020C, 32'h6516_0554, "T1: 0xBBBBBBBB ^ 0xDEADBEEF");
    check_mem_mmio(32'h0210, 32'h1261_7223, "T1: 0xCCCCCCCC ^ 0xDEADBEEF");
    check_mem_mmio(32'h0214, 32'h0370_6332, "T1: 0xDDDDDDDD ^ 0xDEADBEEF");

    $display("  -- Thread 2: Counter Decrement (bootstrap) --");
    check_mem_mmio(32'h0300, 32'h0000_0001, "T2: status = 1");
    check_mem_mmio(32'h0304, 32'h0000_0004, "T2: counter = 4");

    $display("  -- Thread 3: Field Comparison (bootstrap) --");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result = 2 (match)");

    $display("  -- Invariant checks --");
    check_mem_mmio(32'h0104, 32'h0000_00AA, "T0 header preserved");
    check_mem_mmio(32'h040C, 32'h0000_0017, "T3 field preserved");

    section_end();
end
endtask

task test_bootstrap_scenario_B;
    integer cyc;
begin
    section_start("D2: Bootstrap Scenario B — Error / Alternate Paths");

    prepare_common_image();
    init_data_scenario_B();
    run_bootstrap_test_soc(0, cyc);

    check_mem_mmio(32'h0110, 32'hFFFF_FFFF, "T0: result = -1 (error)");
    check_mem_mmio(32'h0108, 32'h1111_1111, "T0: src unchanged");
    check_mem_mmio(32'h010C, 32'h2222_2222, "T0: dst unchanged");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: flag unchanged = 1");
    check_mem_mmio(32'h0208, 32'hAAAA_AAAA, "T1: data[0] unmodified");
    check_mem_mmio(32'h020C, 32'hBBBB_BBBB, "T1: data[1] unmodified");
    check_mem_mmio(32'h0210, 32'hCCCC_CCCC, "T1: data[2] unmodified");
    check_mem_mmio(32'h0214, 32'hDDDD_DDDD, "T1: data[3] unmodified");
    check_mem_mmio(32'h0300, 32'h0000_0000, "T2: status = 0");
    check_mem_mmio(32'h0304, 32'h0000_0000, "T2: counter = 0");
    check_mem_mmio(32'h0400, 32'h0000_0001, "T3: result = 1 (no match)");

    section_end();
end
endtask

task test_bootstrap_scenario_C;
    integer cyc;
begin
    section_start("D3: Bootstrap Scenario C — Edge-Case Values");

    prepare_common_image();
    init_data_scenario_C();
    run_bootstrap_test_soc(0, cyc);

    check_mem_mmio(32'h0108, 32'h0000_0001, "T0: src<-dst swapped");
    check_mem_mmio(32'h010C, 32'hFFFF_FFFF, "T0: dst<-src swapped");
    check_mem_mmio(32'h0110, 32'h0000_0000, "T0: checksum wraps to 0");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: done flag");
    check_mem_mmio(32'h0208, 32'h0000_0000, "T1: DEADBEEF^DEADBEEF=0");
    check_mem_mmio(32'h020C, 32'hDEAD_BEEF, "T1: 0^DEADBEEF");
    check_mem_mmio(32'h0210, 32'h2152_4110, "T1: FFFFFFFF^DEADBEEF");
    check_mem_mmio(32'h0214, 32'hCC99_E897, "T1: 12345678^DEADBEEF");
    check_mem_mmio(32'h0300, 32'h0000_0000, "T2: status=0");
    check_mem_mmio(32'h0304, 32'h0000_0000, "T2: counter=0");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result=2 (match)");

    section_end();
end
endtask

task test_bootstrap_scenario_D;
    integer cyc;
begin
    section_start("D4: Bootstrap Scenario D — Thread Isolation");

    prepare_common_image();
    init_data_scenario_D();
    run_bootstrap_test_soc(0, cyc);

    check_mem_mmio(32'h0108, 32'h0000_BBBB, "T0: src<-dst");
    check_mem_mmio(32'h010C, 32'hAAAA_0000, "T0: dst<-src");
    check_mem_mmio(32'h0110, 32'h0000_000A, "T0: checksum=10");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: flag=1");
    check_mem_mmio(32'h0208, 32'hDFAF_BDEB, "T1: 01020304^DEADBEEF");
    check_mem_mmio(32'h020C, 32'hDBAB_B9E7, "T1: 05060708^DEADBEEF");
    check_mem_mmio(32'h0210, 32'hD7A7_B5E3, "T1: 090A0B0C^DEADBEEF");
    check_mem_mmio(32'h0214, 32'hD3A3_B1FF, "T1: 0D0E0F10^DEADBEEF");
    check_mem_mmio(32'h0300, 32'h0000_0001, "T2: status=1");
    check_mem_mmio(32'h0304, 32'h0000_0009, "T2: counter=9");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result=2 (match)");
    check_mem_mmio(32'h0104, 32'h0000_00AA, "T0 header not corrupted");
    check_mem_mmio(32'h040C, 32'h0000_0017, "T3 field not corrupted");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  D5-D8: NO-ROTATE BOOTSTRAP SCENARIOS (workaround variant)
//  Same tests as D1-D4 but using MOV+LSL instead of rotated MOV.
//  If the rotation canary (C3) fails, these should still pass.
// ═══════════════════════════════════════════════════════════════════

task test_bootstrap_norotate_scenario_A;
    integer cyc;
begin
    section_start("D5: No-Rotate Bootstrap Scenario A");

    prepare_common_image();
    init_data_scenario_A();
    run_bootstrap_test_soc(1, cyc);   // 1 = use no-rotate bootstrap

    check_mem_mmio(32'h0108, 32'h2222_2222, "T0: src<-dst after swap");
    check_mem_mmio(32'h010C, 32'h1111_1111, "T0: dst<-src after swap");
    check_mem_mmio(32'h0110, 32'h0000_00A0, "T0: checksum = 0xA0");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: done flag = 1");
    check_mem_mmio(32'h0208, 32'h7407_1445, "T1: 0xAAAAAAAA ^ 0xDEADBEEF");
    check_mem_mmio(32'h020C, 32'h6516_0554, "T1: 0xBBBBBBBB ^ 0xDEADBEEF");
    check_mem_mmio(32'h0210, 32'h1261_7223, "T1: 0xCCCCCCCC ^ 0xDEADBEEF");
    check_mem_mmio(32'h0214, 32'h0370_6332, "T1: 0xDDDDDDDD ^ 0xDEADBEEF");
    check_mem_mmio(32'h0300, 32'h0000_0001, "T2: status = 1");
    check_mem_mmio(32'h0304, 32'h0000_0004, "T2: counter = 4");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result = 2 (match)");

    section_end();
end
endtask

task test_bootstrap_norotate_scenario_B;
    integer cyc;
begin
    section_start("D6: No-Rotate Bootstrap Scenario B");

    prepare_common_image();
    init_data_scenario_B();
    run_bootstrap_test_soc(1, cyc);

    check_mem_mmio(32'h0110, 32'hFFFF_FFFF, "T0: result = -1");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: flag = 1");
    check_mem_mmio(32'h0208, 32'hAAAA_AAAA, "T1: data[0] unmodified");
    check_mem_mmio(32'h0300, 32'h0000_0000, "T2: status = 0");
    check_mem_mmio(32'h0304, 32'h0000_0000, "T2: counter = 0");
    check_mem_mmio(32'h0400, 32'h0000_0001, "T3: result = 1");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
//  M A I N   S T I M U L U S
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════

initial begin
    $dumpfile("soc_mt_tb.vcd");
    $dumpvars(0, soc_mt_tb);

    total_pass = 0;
    total_fail = 0;
    rst_n = 0;
    start = 0;
    req_cmd = 0; req_addr = 0; req_data = 0;
    req_val = 0; resp_rdy = 0;
    seed = 42;
    cycle_cnt = 0;

    $display("");
    $display("======================================================================");
    $display("   SoC-Level Quad-Thread ARMv4T Multithreaded Pipeline Testbench");
    $display("======================================================================");
    $display("   Phases: A=MMIO  B=MT(hier-init)  C=Canary/Diag  D=MT(bootstrap)");
    $display("----------------------------------------------------------------------");
    $display("   Code:  T0@0x%04H  T1@0x%04H  T2@0x%04H  T3@0x%04H",
             T0_CODE[15:0], T1_CODE[15:0], T2_CODE[15:0], T3_CODE[15:0]);
    $display("   Sent:  T0@0x%04H  T1@0x%04H  T2@0x%04H  T3@0x%04H",
             T0_RET[15:0], T1_RET[15:0], T2_RET[15:0], T3_RET[15:0]);
    $display("   IMEM: %5d words   DMEM: %5d words", IMEM_HW_DEPTH, DMEM_HW_DEPTH);
    $display("======================================================================");

    do_reset();


    // ═══════════════════════════════════════════════════
    //  PHASE A — Basic MMIO Read/Write Infrastructure
    // ═══════════════════════════════════════════════════
    $display("");
    $display("--------------------------------------------------");
    $display("  PHASE A: Basic MMIO Infrastructure Tests");
    $display("--------------------------------------------------");

    $display("\n[A1] CTRL read -- expect idle (0)");
    mmio_rd(CTRL_BASE, rd_data);
    check({`MMIO_DATA_WIDTH{1'b0}}, rd_data, CTRL_BASE, "CTRL_IDLE");

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

    $display("[A4] Write-after-write -- last value wins");
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

    $display("[A5] Read-after-read -- non-destructive");
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

    $display("");
    $display("  Phase A summary: %0d passed, %0d failed",
             total_pass, total_fail);


    // ═══════════════════════════════════════════════════
    //  PHASE B — Multithreaded Network Processing
    //            (hierarchical register init)
    // ═══════════════════════════════════════════════════
    $display("");
    $display("--------------------------------------------------");
    $display("  PHASE B: Multithreaded Network Processing (hier-init)");
    $display("--------------------------------------------------");

    test_scenario_A();
    test_scenario_B();
    test_scenario_C();
    test_scenario_D();


    // ═══════════════════════════════════════════════════
    //  PHASE C — Canary / Diagnostic Tests
    //            (replicating socreg_mt script)
    // ═══════════════════════════════════════════════════
    $display("");
    $display("--------------------------------------------------");
    $display("  PHASE C: Canary / Diagnostic Tests");
    $display("--------------------------------------------------");

    test_canary();
    test_multi_canary();
    test_rotation_canary();
    test_wide_canary();


    // ═══════════════════════════════════════════════════
    //  PHASE D — Bootstrap-Based MT Tests
    //            (register init via ARM code, not hier)
    // ═══════════════════════════════════════════════════
    $display("");
    $display("--------------------------------------------------");
    $display("  PHASE D: Bootstrap-Based MT Tests");
    $display("  (register init via dispatch+init code in IMEM)");
    $display("--------------------------------------------------");

    test_bootstrap_scenario_A();
    test_bootstrap_scenario_B();
    test_bootstrap_scenario_C();
    test_bootstrap_scenario_D();

    // No-rotate variants (workaround for barrel-shifter bugs)
    test_bootstrap_norotate_scenario_A();
    test_bootstrap_norotate_scenario_B();


    // ═══════════════════════════════════════════════════
    //  Final Summary
    // ═══════════════════════════════════════════════════
    $display("");
    $display("======================================================================");
    if (total_fail == 0)
        $display("  *** ALL %4d CHECKS PASSED ***", total_pass);
    else
        $display("  *** %4d PASSED, %4d FAILED ***", total_pass, total_fail);
    $display("  Total checks: %4d", total_pass + total_fail);
    $display("  Phases: A=MMIO  B=MT(hier)  C=Canary  D=MT(bootstrap)");
    $display("======================================================================");
    $display("");

    #(CLK_PERIOD * 5);
    $finish;
end

// ── Watchdog ────────────────────────────────────────────────────
initial begin
    #100_000_000;
    $display("\n[TIMEOUT] Simulation exceeded 100 ms -- aborting");
    $finish;
end

endmodule