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
 * Date:   Feb. 22, 2026
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
//  Base address B is set from T*_CODE localparams.
//  All branch offsets are PC-relative so code is position-independent.
// ═══════════════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────────────
//  Thread 0: network_proc0 — header check, field swap, checksum
// ───────────────────────────────────────────────────────────────────
task load_thread0_code;
    reg [31:0] B;
begin
    B = T0_CODE;
    mem_w(B+'h00, 32'hE52DB004);  // str  fp, [sp, #-4]!
    mem_w(B+'h04, 32'hE28DB000);  // add  fp, sp, #0
    mem_w(B+'h08, 32'hE24DD014);  // sub  sp, sp, #20
    mem_w(B+'h0C, 32'hE3A03C01);  // mov  r3, #256
    mem_w(B+'h10, 32'hE50B3010);  // str  r3, [fp, #-16]
    mem_w(B+'h14, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h18, 32'hE5933004);  // ldr  r3, [r3, #4]
    mem_w(B+'h1C, 32'hE35300AA);  // cmp  r3, #170
    mem_w(B+'h20, 32'h0A000003);  // beq  .L2 (→+0x14)
    mem_w(B+'h24, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h28, 32'hE3E02000);  // mvn  r2, #0
    mem_w(B+'h2C, 32'hE5832010);  // str  r2, [r3, #16]
    mem_w(B+'h30, 32'hEA000021);  // b    .L1 (→+0x8C)
    mem_w(B+'h34, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h38, 32'hE5933008);  // ldr  r3, [r3, #8]
    mem_w(B+'h3C, 32'hE50B3014);  // str  r3, [fp, #-20]
    mem_w(B+'h40, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h44, 32'hE593200C);  // ldr  r2, [r3, #12]
    mem_w(B+'h48, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h4C, 32'hE5832008);  // str  r2, [r3, #8]
    mem_w(B+'h50, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h54, 32'hE51B2014);  // ldr  r2, [fp, #-20]
    mem_w(B+'h58, 32'hE583200C);  // str  r2, [r3, #12]
    mem_w(B+'h5C, 32'hE3A03000);  // mov  r3, #0
    mem_w(B+'h60, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h64, 32'hE3A03000);  // mov  r3, #0
    mem_w(B+'h68, 32'hE50B300C);  // str  r3, [fp, #-12]
    mem_w(B+'h6C, 32'hEA00000B);  // b    .L4 (→+0x34)
    mem_w(B+'h70, 32'hE51B2010);  // ldr  r2, [fp, #-16]
    mem_w(B+'h74, 32'hE51B300C);  // ldr  r3, [fp, #-12]
    mem_w(B+'h78, 32'hE2833004);  // add  r3, r3, #4
    mem_w(B+'h7C, 32'hE1A03103);  // lsl  r3, r3, #2
    mem_w(B+'h80, 32'hE0823003);  // add  r3, r2, r3
    mem_w(B+'h84, 32'hE5933004);  // ldr  r3, [r3, #4]
    mem_w(B+'h88, 32'hE51B2008);  // ldr  r2, [fp, #-8]
    mem_w(B+'h8C, 32'hE0823003);  // add  r3, r2, r3
    mem_w(B+'h90, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h94, 32'hE51B300C);  // ldr  r3, [fp, #-12]
    mem_w(B+'h98, 32'hE2833001);  // add  r3, r3, #1
    mem_w(B+'h9C, 32'hE50B300C);  // str  r3, [fp, #-12]
    mem_w(B+'hA0, 32'hE51B300C);  // ldr  r3, [fp, #-12]
    mem_w(B+'hA4, 32'hE3530003);  // cmp  r3, #3
    mem_w(B+'hA8, 32'hDAFFFFF0);  // ble  .L5 (→-0x38)
    mem_w(B+'hAC, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'hB0, 32'hE51B2008);  // ldr  r2, [fp, #-8]
    mem_w(B+'hB4, 32'hE5832010);  // str  r2, [r3, #16]
    mem_w(B+'hB8, 32'hE1A00000);  // nop
    mem_w(B+'hBC, 32'hE28BD000);  // add  sp, fp, #0
    mem_w(B+'hC0, 32'hE49DB004);  // ldr  fp, [sp], #4
    mem_w(B+'hC4, 32'hE12FFF1E);  // bx   lr
    $display("  [LOAD] Thread 0 code: 0x%08H - 0x%08H", B, B+'hC4);
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Thread 1: network_proc1 — XOR encryption
// ───────────────────────────────────────────────────────────────────
task load_thread1_code;
    reg [31:0] B;
begin
    B = T1_CODE;
    mem_w(B+'h00, 32'hE52DB004);  // str  fp, [sp, #-4]!
    mem_w(B+'h04, 32'hE28DB000);  // add  fp, sp, #0
    mem_w(B+'h08, 32'hE24DD00C);  // sub  sp, sp, #12
    mem_w(B+'h0C, 32'hE3A03C02);  // mov  r3, #512
    mem_w(B+'h10, 32'hE50B300C);  // str  r3, [fp, #-12]
    mem_w(B+'h14, 32'hE51B300C);  // ldr  r3, [fp, #-12]
    mem_w(B+'h18, 32'hE5933000);  // ldr  r3, [r3]
    mem_w(B+'h1C, 32'hE3530000);  // cmp  r3, #0
    mem_w(B+'h20, 32'h1A000016);  // bne  .L6 (→+0x60)
    mem_w(B+'h24, 32'hE3A03000);  // mov  r3, #0
    mem_w(B+'h28, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h2C, 32'hEA00000C);  // b    .L4 (→+0x38)
    mem_w(B+'h30, 32'hE51B300C);  // ldr  r3, [fp, #-12]
    mem_w(B+'h34, 32'hE51B2008);  // ldr  r2, [fp, #-8]
    mem_w(B+'h38, 32'hE2822002);  // add  r2, r2, #2
    mem_w(B+'h3C, 32'hE7932102);  // ldr  r2, [r3, r2, lsl #2]
    mem_w(B+'h40, 32'hE59F3048);  // ldr  r3, [pc, #0x48]  (.L7 literal)
    mem_w(B+'h44, 32'hE0233002);  // eor  r3, r3, r2
    mem_w(B+'h48, 32'hE51B200C);  // ldr  r2, [fp, #-12]
    mem_w(B+'h4C, 32'hE51B1008);  // ldr  r1, [fp, #-8]
    mem_w(B+'h50, 32'hE2811002);  // add  r1, r1, #2
    mem_w(B+'h54, 32'hE7823101);  // str  r3, [r2, r1, lsl #2]
    mem_w(B+'h58, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h5C, 32'hE2833001);  // add  r3, r3, #1
    mem_w(B+'h60, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h64, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h68, 32'hE3530003);  // cmp  r3, #3
    mem_w(B+'h6C, 32'hDAFFFFEF);  // ble  .L5 (→-0x3C)
    mem_w(B+'h70, 32'hE51B300C);  // ldr  r3, [fp, #-12]
    mem_w(B+'h74, 32'hE3A02001);  // mov  r2, #1
    mem_w(B+'h78, 32'hE5832000);  // str  r2, [r3]
    mem_w(B+'h7C, 32'hEA000000);  // b    .L1 (→+0x08)
    mem_w(B+'h80, 32'hE1A00000);  // nop
    mem_w(B+'h84, 32'hE28BD000);  // add  sp, fp, #0
    mem_w(B+'h88, 32'hE49DB004);  // ldr  fp, [sp], #4
    mem_w(B+'h8C, 32'hE12FFF1E);  // bx   lr
    mem_w(B+'h90, 32'hDEADBEEF);  // .word 0xDEADBEEF  (literal pool)
    $display("  [LOAD] Thread 1 code: 0x%08H - 0x%08H (+literal @0x%08H)",
             B, B+'h8C, B+'h90);
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Thread 2: network_proc2 — counter decrement / threshold
// ───────────────────────────────────────────────────────────────────
task load_thread2_code;
    reg [31:0] B;
begin
    B = T2_CODE;
    mem_w(B+'h00, 32'hE52DB004);
    mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C);
    mem_w(B+'h0C, 32'hE3A03C03);  // mov r3, #768
    mem_w(B+'h10, 32'hE50B3008);
    mem_w(B+'h14, 32'hE51B3008);
    mem_w(B+'h18, 32'hE5933004);
    mem_w(B+'h1C, 32'hE3530001);
    mem_w(B+'h20, 32'h8A000006);  // bhi .L2
    mem_w(B+'h24, 32'hE51B3008);
    mem_w(B+'h28, 32'hE3A02000);
    mem_w(B+'h2C, 32'hE5832000);
    mem_w(B+'h30, 32'hE51B3008);
    mem_w(B+'h34, 32'hE3A02000);
    mem_w(B+'h38, 32'hE5832004);
    mem_w(B+'h3C, 32'hEA000008);  // b .L1
    mem_w(B+'h40, 32'hE51B3008);
    mem_w(B+'h44, 32'hE5933004);
    mem_w(B+'h48, 32'hE2432001);
    mem_w(B+'h4C, 32'hE51B3008);
    mem_w(B+'h50, 32'hE5832004);
    mem_w(B+'h54, 32'hE51B3008);
    mem_w(B+'h58, 32'hE3A02001);
    mem_w(B+'h5C, 32'hE5832000);
    mem_w(B+'h60, 32'hE1A00000);
    mem_w(B+'h64, 32'hE28BD000);
    mem_w(B+'h68, 32'hE49DB004);
    mem_w(B+'h6C, 32'hE12FFF1E);
    $display("  [LOAD] Thread 2 code: 0x%08H - 0x%08H", B, B+'h6C);
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Thread 3: network_proc3 — field comparison
// ───────────────────────────────────────────────────────────────────
task load_thread3_code;
    reg [31:0] B;
begin
    B = T3_CODE;
    mem_w(B+'h00, 32'hE52DB004);
    mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C);
    mem_w(B+'h0C, 32'hE3A03C04);  // mov r3, #1024
    mem_w(B+'h10, 32'hE50B3008);
    mem_w(B+'h14, 32'hE51B3008);
    mem_w(B+'h18, 32'hE593300C);
    mem_w(B+'h1C, 32'hE3530017);  // cmp r3, #23
    mem_w(B+'h20, 32'h1A000003);  // bne .L2
    mem_w(B+'h24, 32'hE51B3008);
    mem_w(B+'h28, 32'hE3A02002);
    mem_w(B+'h2C, 32'hE5832000);
    mem_w(B+'h30, 32'hEA000003);
    mem_w(B+'h34, 32'hE51B3008);
    mem_w(B+'h38, 32'hE3A02001);
    mem_w(B+'h3C, 32'hE5832000);
    mem_w(B+'h40, 32'hE1A00000);
    mem_w(B+'h44, 32'hE28BD000);
    mem_w(B+'h48, 32'hE49DB004);
    mem_w(B+'h4C, 32'hE12FFF1E);
    $display("  [LOAD] Thread 3 code: 0x%08H - 0x%08H", B, B+'h4C);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Data Initialisation (scenario-specific, writes local_mem)
//  Data addresses (0x100, 0x200, 0x300, 0x400) are baked into
//  the ARM machine code and must NOT be changed here.
// ═══════════════════════════════════════════════════════════════════

task init_data_scenario_A;
begin
    // T0: header=0xAA, payload 16/32/48/64
    mem_w(32'h0100, 32'h0000_0000);
    mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'h1111_1111);
    mem_w(32'h010C, 32'h2222_2222);
    mem_w(32'h0110, 32'h0000_0000);
    mem_w(32'h0114, 32'h0000_0010);
    mem_w(32'h0118, 32'h0000_0020);
    mem_w(32'h011C, 32'h0000_0030);
    mem_w(32'h0120, 32'h0000_0040);
    // T1: flag=0, encrypt
    mem_w(32'h0200, 32'h0000_0000);
    mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hAAAA_AAAA);
    mem_w(32'h020C, 32'hBBBB_BBBB);
    mem_w(32'h0210, 32'hCCCC_CCCC);
    mem_w(32'h0214, 32'hDDDD_DDDD);
    // T2: counter=5
    mem_w(32'h0300, 32'h0000_0000);
    mem_w(32'h0304, 32'h0000_0005);
    // T3: field=23
    mem_w(32'h0400, 32'h0000_0000);
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_0017);
    $display("  [DATA] Scenario A data loaded into local_mem");
end
endtask

task init_data_scenario_B;
begin
    // T0: bad header
    mem_w(32'h0100, 32'h0000_0000);
    mem_w(32'h0104, 32'h0000_00BB);
    mem_w(32'h0108, 32'h1111_1111);
    mem_w(32'h010C, 32'h2222_2222);
    mem_w(32'h0110, 32'h0000_0000);
    mem_w(32'h0114, 32'h0000_0010);
    mem_w(32'h0118, 32'h0000_0020);
    mem_w(32'h011C, 32'h0000_0030);
    mem_w(32'h0120, 32'h0000_0040);
    // T1: flag=1, skip
    mem_w(32'h0200, 32'h0000_0001);
    mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hAAAA_AAAA);
    mem_w(32'h020C, 32'hBBBB_BBBB);
    mem_w(32'h0210, 32'hCCCC_CCCC);
    mem_w(32'h0214, 32'hDDDD_DDDD);
    // T2: counter=1
    mem_w(32'h0300, 32'hFFFF_FFFF);
    mem_w(32'h0304, 32'h0000_0001);
    // T3: field=42
    mem_w(32'h0400, 32'h0000_0000);
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_002A);
    $display("  [DATA] Scenario B data loaded into local_mem");
end
endtask

task init_data_scenario_C;
begin
    // T0: large payloads
    mem_w(32'h0100, 32'h0000_0000);
    mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'hFFFF_FFFF);
    mem_w(32'h010C, 32'h0000_0001);
    mem_w(32'h0110, 32'h0000_0000);
    mem_w(32'h0114, 32'h7FFF_FFFF);
    mem_w(32'h0118, 32'h0000_0001);
    mem_w(32'h011C, 32'h8000_0000);
    mem_w(32'h0120, 32'h0000_0000);
    // T1: XOR with self
    mem_w(32'h0200, 32'h0000_0000);
    mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hDEAD_BEEF);
    mem_w(32'h020C, 32'h0000_0000);
    mem_w(32'h0210, 32'hFFFF_FFFF);
    mem_w(32'h0214, 32'h1234_5678);
    // T2: counter=0
    mem_w(32'h0300, 32'hAAAA_AAAA);
    mem_w(32'h0304, 32'h0000_0000);
    // T3: field=23
    mem_w(32'h0400, 32'h0000_0000);
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_0017);
    $display("  [DATA] Scenario C data loaded into local_mem");
end
endtask

task init_data_scenario_D;
begin
    // T0: different pattern
    mem_w(32'h0100, 32'h0000_0000);
    mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'hAAAA_0000);
    mem_w(32'h010C, 32'h0000_BBBB);
    mem_w(32'h0110, 32'h0000_0000);
    mem_w(32'h0114, 32'h0000_0001);
    mem_w(32'h0118, 32'h0000_0002);
    mem_w(32'h011C, 32'h0000_0003);
    mem_w(32'h0120, 32'h0000_0004);
    // T1
    mem_w(32'h0200, 32'h0000_0000);
    mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'h0102_0304);
    mem_w(32'h020C, 32'h0506_0708);
    mem_w(32'h0210, 32'h090A_0B0C);
    mem_w(32'h0214, 32'h0D0E_0F10);
    // T2: counter=10
    mem_w(32'h0300, 32'h0000_0000);
    mem_w(32'h0304, 32'h0000_000A);
    // T3: field=23
    mem_w(32'h0400, 32'h0000_0000);
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_0017);
    $display("  [DATA] Scenario D data loaded into local_mem");
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
    // Place sentinel (B .) at each thread's return address
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
    // ── Reset SoC ────────────────────────────────────────────
    do_reset();

    // ── Write image to IMEM & DMEM via MMIO ──────────────────
    load_image_to_soc();

    // ── Init CPU registers (hierarchical, CPU in reset) ──────
    init_cpu_mt_regs();
    @(posedge clk); #1;
    init_cpu_mt_regs();   // re-init after clock edge for safety

    $display("  [INIT] SPs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
             T0_SP, T1_SP, T2_SP, T3_SP);
    $display("  [INIT] LRs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
             T0_RET, T1_RET, T2_RET, T3_RET);

    // ── Assert start ─────────────────────────────────────────
    @(posedge clk); #1;
    start = 1'b1;

    // ── FIX: Initialize per-thread PCs ───────────────────────
    //    Must be AFTER start=1 so the CPU's internal reset
    //    (active while start=0) does not clear them back to 0.
    u_soc_mt.u_cpu_mt.pc_thread[0] = T0_CODE;
    u_soc_mt.u_cpu_mt.pc_thread[1] = T1_CODE;
    u_soc_mt.u_cpu_mt.pc_thread[2] = T2_CODE;
    u_soc_mt.u_cpu_mt.pc_thread[3] = T3_CODE;

    $display("  [RUN]  start=1, CPU executing...");
    $display("  [INIT] PCs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
             T0_CODE, T1_CODE, T2_CODE, T3_CODE);

    // ── Monitor thread PCs until all reach sentinel ──────────
    cycle_cnt = 0;
    for (t = 0; t < 4; t = t + 1) sent_cnt[t] = 0;
    sent_ok = 4'b0000;

    begin : run_loop
        forever begin
            @(posedge clk); #1;
            cycle_cnt = cycle_cnt + 1;

            // ── Per-cycle trace (early cycles) ───────────
            if (TRACE_EN && cycle_cnt > 0 && cycle_cnt <= TRACE_LIMIT) begin
                $display("[C%05d] PC: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         cycle_cnt,
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));
            end

            // ── Sentinel detection ───────────────────────
            if (cycle_cnt > 20) begin
                if (get_thread_pc(2'd0) == T0_RET)
                    sent_cnt[0] = sent_cnt[0] + 1;
                else
                    sent_cnt[0] = 0;

                if (get_thread_pc(2'd1) == T1_RET)
                    sent_cnt[1] = sent_cnt[1] + 1;
                else
                    sent_cnt[1] = 0;

                if (get_thread_pc(2'd2) == T2_RET)
                    sent_cnt[2] = sent_cnt[2] + 1;
                else
                    sent_cnt[2] = 0;

                if (get_thread_pc(2'd3) == T3_RET)
                    sent_cnt[3] = sent_cnt[3] + 1;
                else
                    sent_cnt[3] = 0;

                sent_ok[0] = (sent_cnt[0] > 30);
                sent_ok[1] = (sent_cnt[1] > 30);
                sent_ok[2] = (sent_cnt[2] > 30);
                sent_ok[3] = (sent_cnt[3] > 30);
            end

            // ── All threads done ─────────────────────────
            if (sent_ok == 4'b1111) begin
                $display("  [DONE] All 4 threads at sentinel, cycle %0d", cycle_cnt);
                repeat (10) @(posedge clk);
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            // ── Periodic status ──────────────────────────
            if (cycle_cnt > 0 && (cycle_cnt % STATUS_INTERVAL == 0)) begin
                $display("  [STATUS C%05d] sentinel=%04b  PC: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         cycle_cnt, sent_ok,
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));
            end

            // ── Timeout ──────────────────────────────────
            if (cycle_cnt >= MAX_CYCLES) begin
                $display("  *** TIMEOUT after %0d cycles ***", MAX_CYCLES);
                $display("  Sentinel status: %04b", sent_ok);
                $display("  PCs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         get_thread_pc(2'd0), get_thread_pc(2'd1),
                         get_thread_pc(2'd2), get_thread_pc(2'd3));
                dump_all_threads();
                cycles_used = cycle_cnt;
                disable run_loop;
            end
        end
    end

    // ── De-assert start (CPU resets, BRAMs preserved) ────────
    start = 1'b0;
    @(posedge clk); #1;

    // ── Wait for MMIO to become available ────────────────────
    wait_cnt = 0;
    while (!req_rdy && wait_cnt < 100) begin
        @(posedge clk); #1;
        wait_cnt = wait_cnt + 1;
    end
    if (req_rdy)
        $display("  [STOP] MMIO available (start=0, req_rdy=1)");
    else begin
        $display("  WARNING: req_rdy not returning high after stop");
        total_fail = total_fail + 1;
    end
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  T E S T   S C E N A R I O S
// ═══════════════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────────────
//  Scenario A: Normal path (all four threads)
// ───────────────────────────────────────────────────────────────────
task test_scenario_A;
    integer cyc;
begin
    section_start("Scenario A: Normal Path (all four threads)");

    prepare_common_image();
    init_data_scenario_A();
    run_mt_test_soc(cyc);

    // ── Dump data regions via MMIO ────────────────────────────
    $display("");
    $display("  -- Data Memory After Execution (via MMIO) --");
    dump_mem_mmio(32'h0100, 10);
    dump_mem_mmio(32'h0200, 6);
    dump_mem_mmio(32'h0300, 2);
    dump_mem_mmio(32'h0400, 4);

    // ── Thread 0: swap + checksum ─────────────────────────────
    $display("");
    $display("  -- Thread 0: Packet Processing --");
    check_mem_mmio(32'h0108, 32'h2222_2222, "T0: src<-dst after swap");
    check_mem_mmio(32'h010C, 32'h1111_1111, "T0: dst<-src after swap");
    check_mem_mmio(32'h0110, 32'h0000_00A0, "T0: checksum = 16+32+48+64 = 160 = 0xA0");

    // ── Thread 1: XOR encryption ──────────────────────────────
    $display("");
    $display("  -- Thread 1: XOR Encryption --");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: done flag = 1");
    check_mem_mmio(32'h0208, 32'h7407_1445, "T1: 0xAAAAAAAA ^ 0xDEADBEEF");
    check_mem_mmio(32'h020C, 32'h6516_0554, "T1: 0xBBBBBBBB ^ 0xDEADBEEF");
    check_mem_mmio(32'h0210, 32'h1261_7223, "T1: 0xCCCCCCCC ^ 0xDEADBEEF");
    check_mem_mmio(32'h0214, 32'h0370_6332, "T1: 0xDDDDDDDD ^ 0xDEADBEEF");

    // ── Thread 2: counter decrement ───────────────────────────
    $display("");
    $display("  -- Thread 2: Counter Decrement --");
    check_mem_mmio(32'h0300, 32'h0000_0001, "T2: status = 1 (active)");
    check_mem_mmio(32'h0304, 32'h0000_0004, "T2: counter = 5-1 = 4");

    // ── Thread 3: field comparison ────────────────────────────
    $display("");
    $display("  -- Thread 3: Field Comparison --");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result = 2 (field == 23)");

    // ── Stack pointer restoration (hierarchical) ──────────────
    $display("");
    $display("  -- Stack Pointer Restoration --");
    check_reg_hier(2'd0, 4'd13, T0_SP, "T0: SP restored");
    check_reg_hier(2'd1, 4'd13, T1_SP, "T1: SP restored");
    check_reg_hier(2'd2, 4'd13, T2_SP, "T2: SP restored");
    check_reg_hier(2'd3, 4'd13, T3_SP, "T3: SP restored");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Scenario B: Error / alternate paths
// ───────────────────────────────────────────────────────────────────
task test_scenario_B;
    integer cyc;
begin
    section_start("Scenario B: Alternate / Error Paths");

    prepare_common_image();
    init_data_scenario_B();
    run_mt_test_soc(cyc);

    $display("");
    dump_mem_mmio(32'h0100, 6);
    dump_mem_mmio(32'h0200, 6);
    dump_mem_mmio(32'h0300, 2);
    dump_mem_mmio(32'h0400, 4);

    // ── Thread 0: error path ──────────────────────────────────
    $display("");
    $display("  -- Thread 0: Error Path (bad header) --");
    check_mem_mmio(32'h0110, 32'hFFFF_FFFF, "T0: result = -1 (error)");
    check_mem_mmio(32'h0108, 32'h1111_1111, "T0: src unchanged (no swap)");
    check_mem_mmio(32'h010C, 32'h2222_2222, "T0: dst unchanged (no swap)");

    // ── Thread 1: skip path ───────────────────────────────────
    $display("");
    $display("  -- Thread 1: Skip Path (flag=1) --");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: flag unchanged = 1");
    check_mem_mmio(32'h0208, 32'hAAAA_AAAA, "T1: data[0] unmodified");
    check_mem_mmio(32'h020C, 32'hBBBB_BBBB, "T1: data[1] unmodified");
    check_mem_mmio(32'h0210, 32'hCCCC_CCCC, "T1: data[2] unmodified");
    check_mem_mmio(32'h0214, 32'hDDDD_DDDD, "T1: data[3] unmodified");

    // ── Thread 2: zero path ───────────────────────────────────
    $display("");
    $display("  -- Thread 2: Zero Path (counter<=1) --");
    check_mem_mmio(32'h0300, 32'h0000_0000, "T2: status = 0 (inactive)");
    check_mem_mmio(32'h0304, 32'h0000_0000, "T2: counter = 0");

    // ── Thread 3: no-match ────────────────────────────────────
    $display("");
    $display("  -- Thread 3: No-Match Path (field=42) --");
    check_mem_mmio(32'h0400, 32'h0000_0001, "T3: result = 1 (no match)");

    // ── SPs ───────────────────────────────────────────────────
    $display("");
    check_reg_hier(2'd0, 4'd13, T0_SP, "T0: SP restored");
    check_reg_hier(2'd1, 4'd13, T1_SP, "T1: SP restored");
    check_reg_hier(2'd2, 4'd13, T2_SP, "T2: SP restored");
    check_reg_hier(2'd3, 4'd13, T3_SP, "T3: SP restored");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Scenario C: Edge-case data values
// ───────────────────────────────────────────────────────────────────
task test_scenario_C;
    integer cyc;
begin
    section_start("Scenario C: Edge-Case Data Values");

    prepare_common_image();
    init_data_scenario_C();
    run_mt_test_soc(cyc);

    $display("");
    dump_mem_mmio(32'h0100, 10);
    dump_mem_mmio(32'h0200, 6);
    dump_mem_mmio(32'h0300, 2);
    dump_mem_mmio(32'h0400, 4);

    // ── T0: checksum wrapping ─────────────────────────────────
    $display("");
    $display("  -- Thread 0: Large value checksum (wrapping) --");
    check_mem_mmio(32'h0108, 32'h0000_0001, "T0: src<-dst swapped");
    check_mem_mmio(32'h010C, 32'hFFFF_FFFF, "T0: dst<-src swapped");
    check_mem_mmio(32'h0110, 32'h0000_0000, "T0: checksum wraps to 0");

    // ── T1: XOR edge cases ────────────────────────────────────
    $display("");
    $display("  -- Thread 1: XOR edge cases --");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: done flag");
    check_mem_mmio(32'h0208, 32'h0000_0000, "T1: DEADBEEF^DEADBEEF=0");
    check_mem_mmio(32'h020C, 32'hDEAD_BEEF, "T1: 0^DEADBEEF=DEADBEEF");
    check_mem_mmio(32'h0210, 32'h2152_4110, "T1: FFFFFFFF^DEADBEEF=21524110");
    check_mem_mmio(32'h0214, 32'hCC99_E897, "T1: 12345678^DEADBEEF");

    // ── T2: counter=0 path ────────────────────────────────────
    $display("");
    $display("  -- Thread 2: Counter=0 (zero path) --");
    check_mem_mmio(32'h0300, 32'h0000_0000, "T2: status=0");
    check_mem_mmio(32'h0304, 32'h0000_0000, "T2: counter=0");

    // ── T3: match ─────────────────────────────────────────────
    $display("");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result=2 (match)");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Scenario D: Thread isolation verification
// ───────────────────────────────────────────────────────────────────
task test_scenario_D;
    integer cyc;
begin
    section_start("Scenario D: Thread Isolation Verification");

    prepare_common_image();
    init_data_scenario_D();
    run_mt_test_soc(cyc);

    $display("");
    dump_mem_mmio(32'h0100, 10);
    dump_mem_mmio(32'h0200, 6);

    // ── T0 ────────────────────────────────────────────────────
    $display("");
    $display("  -- Thread 0: Isolation check --");
    check_mem_mmio(32'h0108, 32'h0000_BBBB, "T0: src<-dst");
    check_mem_mmio(32'h010C, 32'hAAAA_0000, "T0: dst<-src");
    check_mem_mmio(32'h0110, 32'h0000_000A, "T0: checksum=1+2+3+4=10");

    // ── T1 ────────────────────────────────────────────────────
    $display("");
    $display("  -- Thread 1: Isolation check --");
    check_mem_mmio(32'h0200, 32'h0000_0001, "T1: flag=1");
    check_mem_mmio(32'h0208, 32'hDFAF_BDEB, "T1: 01020304^DEADBEEF");
    check_mem_mmio(32'h020C, 32'hDBAB_B9E7, "T1: 05060708^DEADBEEF");
    check_mem_mmio(32'h0210, 32'hD7A7_B5E3, "T1: 090A0B0C^DEADBEEF");
    check_mem_mmio(32'h0214, 32'hD3A3_B1FF, "T1: 0D0E0F10^DEADBEEF");

    // ── T2 ────────────────────────────────────────────────────
    $display("");
    $display("  -- Thread 2: Isolation check --");
    check_mem_mmio(32'h0300, 32'h0000_0001, "T2: status=1");
    check_mem_mmio(32'h0304, 32'h0000_0009, "T2: counter=10-1=9");

    // ── T3 ────────────────────────────────────────────────────
    $display("");
    check_mem_mmio(32'h0400, 32'h0000_0002, "T3: result=2 (match)");

    // ── Cross-thread data integrity ───────────────────────────
    $display("");
    $display("  -- Cross-thread data integrity (MMIO) --");
    check_mem_mmio(32'h0104, 32'h0000_00AA, "T0 header not corrupted by other threads");
    check_mem_mmio(32'h0204, 32'h0000_0000, "T1 pad not corrupted");
    check_mem_mmio(32'h040C, 32'h0000_0017, "T3 field not corrupted");

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
    $display("   Thread 0: network_proc0 (packet header/swap/checksum)");
    $display("   Thread 1: network_proc1 (XOR encryption)");
    $display("   Thread 2: network_proc2 (counter decrement)");
    $display("   Thread 3: network_proc3 (field comparison)");
    $display("----------------------------------------------------------------------");
    $display("   Code:  T0@0x%04H  T1@0x%04H  T2@0x%04H  T3@0x%04H",
             T0_CODE[15:0], T1_CODE[15:0], T2_CODE[15:0], T3_CODE[15:0]);
    $display("   Sent:  T0@0x%04H  T1@0x%04H  T2@0x%04H  T3@0x%04H",
             T0_RET[15:0], T1_RET[15:0], T2_RET[15:0], T3_RET[15:0]);
    $display("   Data:  T0@0x%04H  T1@0x%04H  T2@0x%04H  T3@0x%04H",
             T0_DATA[15:0], T1_DATA[15:0], T2_DATA[15:0], T3_DATA[15:0]);
    $display("   Stack: T0@0x%04H  T1@0x%04H  T2@0x%04H  T3@0x%04H",
             T0_SP[15:0], T1_SP[15:0], T2_SP[15:0], T3_SP[15:0]);
    $display("----------------------------------------------------------------------");
    $display("   IMEM: %5d words (%0d-bit addr)   DMEM: %5d words (%0d-bit addr)",
             IMEM_HW_DEPTH, `IMEM_ADDR_WIDTH, DMEM_HW_DEPTH, `DMEM_ADDR_WIDTH);
    $display("   IMEM needed: %0d words   DMEM needed: %0d words",
             IMEM_MIN_DEPTH, DMEM_MIN_DEPTH);
    $display("   MAX_CYCLES: %0d   TRACE_EN: %0d   TRACE_LIMIT: %0d",
             MAX_CYCLES, TRACE_EN, TRACE_LIMIT);
    $display("======================================================================");
    $display("");

    // ── Memory depth checks ──────────────────────────────────
    if (IMEM_HW_DEPTH < IMEM_MIN_DEPTH) begin
        $display("*** FATAL: IMEM depth (%0d) < minimum required (%0d words).",
                 IMEM_HW_DEPTH, IMEM_MIN_DEPTH);
        $display("***        Thread code and/or sentinels will not fit.");
        $display("***        Need IMEM_ADDR_WIDTH >= %0d (currently %0d).",
                 $clog2(IMEM_MIN_DEPTH), `IMEM_ADDR_WIDTH);
    end
    if (DMEM_HW_DEPTH < DMEM_MIN_DEPTH) begin
        $display("*** WARNING: DMEM depth (%0d) < minimum required (%0d words).",
                 DMEM_HW_DEPTH, DMEM_MIN_DEPTH);
        $display("***          Stack accesses (up to word 0x%04H) may alias.",
                 (T3_SP >> 2));
        $display("***          Need DMEM_ADDR_WIDTH >= %0d (currently %0d).",
                 $clog2(DMEM_MIN_DEPTH), `DMEM_ADDR_WIDTH);
    end

    do_reset();


    // ═══════════════════════════════════════════════════
    //  PHASE A — Basic MMIO Read/Write Infrastructure
    // ═══════════════════════════════════════════════════
    $display("");
    $display("--------------------------------------------------");
    $display("  PHASE A: Basic MMIO Infrastructure Tests");
    $display("--------------------------------------------------");

    // ── A1: CTRL read — idle ────────────────────────────
    $display("\n[A1] CTRL read -- expect idle (0)");
    mmio_rd(CTRL_BASE, rd_data);
    check({`MMIO_DATA_WIDTH{1'b0}}, rd_data, CTRL_BASE, "CTRL_IDLE");

    // ── A2: IMEM sequential write + readback ────────────
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

    // ── A3: DMEM sequential write + readback ────────────
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

    // ── A4: Write-after-write (last wins) ───────────────
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

    // ── A5: Read-after-read (non-destructive) ───────────
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

    // ── A6: Random IMEM write + readback ────────────────
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

    // ── A7: Random DMEM write + readback ────────────────
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

    // ── A8: Mixed IMEM/DMEM interleaved ─────────────────
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
    // ═══════════════════════════════════════════════════
    $display("");
    $display("--------------------------------------------------");
    $display("  PHASE B: Multithreaded Network Processing Tests");
    $display("--------------------------------------------------");

    test_scenario_A();
    test_scenario_B();
    test_scenario_C();
    test_scenario_D();


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
    $display("  IMEM: %5d words (%0d-bit)   DMEM: %5d words (%0d-bit)",
             IMEM_HW_DEPTH, `IMEM_ADDR_WIDTH, DMEM_HW_DEPTH, `DMEM_ADDR_WIDTH);
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