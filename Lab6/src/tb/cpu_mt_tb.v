/* cpu_mt_tb.v — Quad-thread ARMv4T multithreaded pipeline testbench
 * Runs four GCC-compiled network-processing programs simultaneously
 * on the zero-overhead multithreaded CPU
 *
 *   Thread 0: network_proc0 — packet header check, field swap, checksum
 *   Thread 1: network_proc1 — XOR encryption with key 0xDEADBEEF
 *   Thread 2: network_proc2 — counter decrement / threshold check
 *   Thread 3: network_proc3 — field comparison against constant 23
 * Author: Jeremy Cai
 * Date:   Feb. 21, 2026
 */
`timescale 1ns / 1ps
`include "define.v"
`include "cpu_mt.v"

module cpu_mt_tb;

// ═══════════════════════════════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════════════════════════════
parameter CLK_PERIOD   = 10;
parameter MEM_DEPTH    = 16384;        // 64 KB (16K words)
parameter MAX_CYCLES   = 10_000;
parameter TRACE_EN     = 1;
parameter TRACE_LIMIT  = 600;
parameter SYNC_MEM     = 1;

localparam [31:0] SENTINEL = 32'hEAFF_FFFE;   // B . (branch to self)

// ── Thread base addresses ─────────────────────────────────────────
localparam [31:0] T0_CODE = 32'h0000_0000;
localparam [31:0] T1_CODE = 32'h0000_1000;
localparam [31:0] T2_CODE = 32'h0000_2000;
localparam [31:0] T3_CODE = 32'h0000_3000;

localparam [31:0] T0_DATA = 32'h0000_0100;    // mov r3, #256
localparam [31:0] T1_DATA = 32'h0000_0200;    // mov r3, #512
localparam [31:0] T2_DATA = 32'h0000_0300;    // mov r3, #768
localparam [31:0] T3_DATA = 32'h0000_0400;    // mov r3, #1024

localparam [31:0] T0_SP = 32'h0000_0800;
localparam [31:0] T1_SP = 32'h0000_0A00;
localparam [31:0] T2_SP = 32'h0000_0C00;
localparam [31:0] T3_SP = 32'h0000_0E00;

localparam [31:0] T0_RET = 32'h0000_0FFC;     // sentinel address
localparam [31:0] T1_RET = 32'h0000_1FFC;
localparam [31:0] T2_RET = 32'h0000_2FFC;
localparam [31:0] T3_RET = 32'h0000_3FFC;

// ═══════════════════════════════════════════════════════════════════
//  DUT Signals
// ═══════════════════════════════════════════════════════════════════
reg                         clk, rst_n;
wire [`PC_WIDTH-1:0]        i_mem_addr;
reg  [`INSTR_WIDTH-1:0]     i_mem_data;
wire [`CPU_DMEM_ADDR_WIDTH-1:0] d_mem_addr;
wire [`DATA_WIDTH-1:0]      d_mem_wdata;
reg  [`DATA_WIDTH-1:0]      d_mem_rdata;
wire                        d_mem_wen;
wire [1:0]                  d_mem_size;
wire                        cpu_done_w;
reg  [1:0]                  ila_thread_sel;
reg  [4:0]                  ila_debug_sel;
wire [`DATA_WIDTH-1:0]      ila_debug_data;

// ═══════════════════════════════════════════════════════════════════
//  Unified Memory
// ═══════════════════════════════════════════════════════════════════
reg [31:0] mem_array [0:MEM_DEPTH-1];

// ═══════════════════════════════════════════════════════════════════
//  DUT Instantiation
// ═══════════════════════════════════════════════════════════════════
cpu_mt u_cpu_mt (
    .clk            (clk),
    .rst_n          (rst_n),
    .i_mem_data_i   (i_mem_data),
    .i_mem_addr_o   (i_mem_addr),
    .d_mem_data_i   (d_mem_rdata),
    .d_mem_addr_o   (d_mem_addr),
    .d_mem_data_o   (d_mem_wdata),
    .d_mem_wen_o    (d_mem_wen),
    .d_mem_size_o   (d_mem_size),
    .cpu_done       (cpu_done_w),
    .ila_thread_sel (ila_thread_sel),
    .ila_debug_sel  (ila_debug_sel),
    .ila_debug_data (ila_debug_data)
);

// ═══════════════════════════════════════════════════════════════════
//  Clock
// ═══════════════════════════════════════════════════════════════════
initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

// ═══════════════════════════════════════════════════════════════════
//  Memory Model  (sync or comb, selectable)
// ═══════════════════════════════════════════════════════════════════
generate
if (SYNC_MEM == 1) begin : gen_sync_mem
    always @(posedge clk) begin
        i_mem_data  <= mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];
        d_mem_rdata <= mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)];
        if (d_mem_wen)
            mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)] <= d_mem_wdata;
    end
end else begin : gen_comb_mem
    always @(*) begin
        i_mem_data  = mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];
        d_mem_rdata = mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)];
    end
    always @(posedge clk) begin
        if (d_mem_wen)
            mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)] <= d_mem_wdata;
    end
end
endgenerate

wire [31:0] instr_at_pc = mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];

// ═══════════════════════════════════════════════════════════════════
//  Per-Thread Sentinel Detection
// ═══════════════════════════════════════════════════════════════════
reg  [7:0] sentinel_cnt [0:3];
wire [3:0] thread_at_sentinel;
integer st;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (st = 0; st < 4; st = st + 1)
            sentinel_cnt[st] <= 8'd0;
    end else if (cycle_cnt > 10) begin
        for (st = 0; st < 4; st = st + 1) begin
            if (mem_array[(u_cpu_mt.pc_thread[st] >> 2) & (MEM_DEPTH-1)] === SENTINEL) begin
                if (sentinel_cnt[st] < 8'd255)
                    sentinel_cnt[st] <= sentinel_cnt[st] + 8'd1;
            end else begin
                if (sentinel_cnt[st] > 0)
                    sentinel_cnt[st] <= sentinel_cnt[st] - 8'd1;
            end
        end
    end
end

assign thread_at_sentinel[0] = (sentinel_cnt[0] > 8'd30);
assign thread_at_sentinel[1] = (sentinel_cnt[1] > 8'd30);
assign thread_at_sentinel[2] = (sentinel_cnt[2] > 8'd30);
assign thread_at_sentinel[3] = (sentinel_cnt[3] > 8'd30);

wire all_threads_done = &thread_at_sentinel;

// ═══════════════════════════════════════════════════════════════════
//  Bookkeeping
// ═══════════════════════════════════════════════════════════════════
integer total_pass, total_fail, section_pass, section_fail;
integer cycle_cnt;
reg [256*8:1] current_section;

// ═══════════════════════════════════════════════════════════════════
//  Per-Cycle Trace
// ═══════════════════════════════════════════════════════════════════
always @(posedge clk) begin
    if (TRACE_EN && rst_n && cycle_cnt > 0 && cycle_cnt <= TRACE_LIMIT) begin
        $display("[C%05d] IF=T%0d PC=0x%08H  @PC=0x%08H | ID=T%0d EX=T%0d MEM=T%0d WB=T%0d | D:a=0x%08H w=%b wd=0x%08H rd=0x%08H",
                 cycle_cnt,
                 u_cpu_mt.tid_if,   i_mem_addr, instr_at_pc,
                 u_cpu_mt.tid_id,   u_cpu_mt.tid_ex,
                 u_cpu_mt.tid_mem,  u_cpu_mt.tid_wb,
                 d_mem_addr, d_mem_wen, d_mem_wdata, d_mem_rdata);
    end
    if (rst_n && d_mem_wen) begin
        $display("         >> DMEM WRITE: [0x%08H] <= 0x%08H (sz=%0d) @ cycle %0d",
                 d_mem_addr, d_mem_wdata, d_mem_size, cycle_cnt);
    end
end

// ═══════════════════════════════════════════════════════════════════
//  Helper Tasks
// ═══════════════════════════════════════════════════════════════════

task mem_clear;
    integer k;
begin
    for (k = 0; k < MEM_DEPTH; k = k + 1)
        mem_array[k] = 32'h0;
end
endtask

task mem_w;
    input [31:0] byte_addr;
    input [31:0] data;
begin
    mem_array[byte_addr >> 2] = data;
end
endtask

// ── Read a thread's register (works around generate-block indexing) ──
function [31:0] get_reg;
    input [1:0] tid;
    input [3:0] rn;
begin
    case (tid)
        2'd0: get_reg = u_cpu_mt.THREAD_RF[0].u_rf.regs[rn];
        2'd1: get_reg = u_cpu_mt.THREAD_RF[1].u_rf.regs[rn];
        2'd2: get_reg = u_cpu_mt.THREAD_RF[2].u_rf.regs[rn];
        2'd3: get_reg = u_cpu_mt.THREAD_RF[3].u_rf.regs[rn];
    endcase
end
endfunction

// ── Check a thread's register ─────────────────────────────────────
task check_reg_t;
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

// ── Check a memory word ───────────────────────────────────────────
task check_mem;
    input [31:0]    byte_addr;
    input [31:0]    expected;
    input [256*8:1] msg;
    reg [31:0] actual;
begin
    actual = mem_array[byte_addr >> 2];
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

// ── Section header / footer ───────────────────────────────────────
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
    dump_all_threads();
    if (section_fail > 0)
        $display("  ** %0s: %0d PASSED, %0d FAILED (%0d cycles) **",
                 current_section, section_pass, section_fail, cycle_cnt);
    else
        $display("  ── %0s: all %0d passed (%0d cycles) ──",
                 current_section, section_pass, cycle_cnt);
end
endtask

// ── Dump all thread registers and PCs ─────────────────────────────
task dump_all_threads;
    integer t, r;
begin
    for (t = 0; t < 4; t = t + 1) begin
        $display("  ┌─ Thread %0d ── PC=0x%08H  CPSR_flags=%04b ─────────────────┐",
                 t, u_cpu_mt.pc_thread[t], u_cpu_mt.cpsr_flags[t]);
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

// ── Dump a memory region ──────────────────────────────────────────
task dump_mem;
    input [31:0] base_byte;
    input integer count;
    integer i;
begin
    $display("  ┌─ Memory @ 0x%08H (%0d words) ─────────────────────────────┐",
             base_byte, count);
    for (i = 0; i < count; i = i + 1)
        $display("  │ [0x%08H] = 0x%08H                                          │",
                 base_byte + (i*4), mem_array[(base_byte>>2) + i]);
    $display("  └───────────────────────────────────────────────────────────────┘");
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Thread Code Loading — Hand-assembled from GCC output
// ═══════════════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────────────
//  Thread 0: network_proc0 — header check, field swap, checksum
//
//  C logic (reconstructed):
//    base = 0x100;
//    if (base[1] != 0xAA) { base[4] = -1; return; }
//    temp = base[2]; base[2] = base[3]; base[3] = temp;
//    chk = 0;
//    for (i=0; i<=3; i++) chk += *(base + (i+4)*4 + 4);
//    base[4] = chk;
// ───────────────────────────────────────────────────────────────────
task load_thread0_code;
    reg [31:0] B;
begin
    B = T0_CODE;
    //        Offset  Instruction               Encoding
    mem_w(B+'h00, 32'hE52DB004);  // str  fp, [sp, #-4]!
    mem_w(B+'h04, 32'hE28DB000);  // add  fp, sp, #0
    mem_w(B+'h08, 32'hE24DD014);  // sub  sp, sp, #20
    mem_w(B+'h0C, 32'hE3A03C01);  // mov  r3, #256         (0x100)
    mem_w(B+'h10, 32'hE50B3010);  // str  r3, [fp, #-16]
    mem_w(B+'h14, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h18, 32'hE5933004);  // ldr  r3, [r3, #4]
    mem_w(B+'h1C, 32'hE35300AA);  // cmp  r3, #170
    mem_w(B+'h20, 32'h0A000003);  // beq  .L2 (→0x34)
    // ── Error path ──
    mem_w(B+'h24, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h28, 32'hE3E02000);  // mvn  r2, #0
    mem_w(B+'h2C, 32'hE5832010);  // str  r2, [r3, #16]
    mem_w(B+'h30, 32'hEA000021);  // b    .L1 (→0xBC)
    // ── .L2: Swap path ──
    mem_w(B+'h34, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h38, 32'hE5933008);  // ldr  r3, [r3, #8]
    mem_w(B+'h3C, 32'hE50B3014);  // str  r3, [fp, #-20]    (temp)
    mem_w(B+'h40, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h44, 32'hE593200C);  // ldr  r2, [r3, #12]
    mem_w(B+'h48, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h4C, 32'hE5832008);  // str  r2, [r3, #8]
    mem_w(B+'h50, 32'hE51B3010);  // ldr  r3, [fp, #-16]
    mem_w(B+'h54, 32'hE51B2014);  // ldr  r2, [fp, #-20]    (temp)
    mem_w(B+'h58, 32'hE583200C);  // str  r2, [r3, #12]
    // ── Init loop ──
    mem_w(B+'h5C, 32'hE3A03000);  // mov  r3, #0            (checksum)
    mem_w(B+'h60, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h64, 32'hE3A03000);  // mov  r3, #0            (i)
    mem_w(B+'h68, 32'hE50B300C);  // str  r3, [fp, #-12]
    mem_w(B+'h6C, 32'hEA00000B);  // b    .L4 (→0xA0)
    // ── .L5: Loop body ──
    mem_w(B+'h70, 32'hE51B2010);  // ldr  r2, [fp, #-16]    (base)
    mem_w(B+'h74, 32'hE51B300C);  // ldr  r3, [fp, #-12]    (i)
    mem_w(B+'h78, 32'hE2833004);  // add  r3, r3, #4
    mem_w(B+'h7C, 32'hE1A03103);  // lsl  r3, r3, #2
    mem_w(B+'h80, 32'hE0823003);  // add  r3, r2, r3
    mem_w(B+'h84, 32'hE5933004);  // ldr  r3, [r3, #4]
    mem_w(B+'h88, 32'hE51B2008);  // ldr  r2, [fp, #-8]     (chk)
    mem_w(B+'h8C, 32'hE0823003);  // add  r3, r2, r3
    mem_w(B+'h90, 32'hE50B3008);  // str  r3, [fp, #-8]     (chk)
    mem_w(B+'h94, 32'hE51B300C);  // ldr  r3, [fp, #-12]    (i)
    mem_w(B+'h98, 32'hE2833001);  // add  r3, r3, #1
    mem_w(B+'h9C, 32'hE50B300C);  // str  r3, [fp, #-12]    (i)
    // ── .L4: Loop condition ──
    mem_w(B+'hA0, 32'hE51B300C);  // ldr  r3, [fp, #-12]    (i)
    mem_w(B+'hA4, 32'hE3530003);  // cmp  r3, #3
    mem_w(B+'hA8, 32'hDAFFFFF0);  // ble  .L5 (→0x70)
    // ── Store result ──
    mem_w(B+'hAC, 32'hE51B3010);  // ldr  r3, [fp, #-16]    (base)
    mem_w(B+'hB0, 32'hE51B2008);  // ldr  r2, [fp, #-8]     (chk)
    mem_w(B+'hB4, 32'hE5832010);  // str  r2, [r3, #16]
    mem_w(B+'hB8, 32'hE1A00000);  // nop
    // ── .L1: Epilogue ──
    mem_w(B+'hBC, 32'hE28BD000);  // add  sp, fp, #0
    mem_w(B+'hC0, 32'hE49DB004);  // ldr  fp, [sp], #4
    mem_w(B+'hC4, 32'hE12FFF1E);  // bx   lr

    $display("  [LOAD] Thread 0 code: 0x%08H – 0x%08H (%0d instructions)",
             B, B+'hC4, (32'hC4/4)+1);
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Thread 1: network_proc1 — XOR encryption
//
//  C logic:
//    base = 0x200;
//    if (base[0] != 0) return; // skip if flag set
//    for (i=0; i<=3; i++)
//        base[i+2] ^= 0xDEADBEEF;
//    base[0] = 1;              // set done flag
// ───────────────────────────────────────────────────────────────────
task load_thread1_code;
    reg [31:0] B;
begin
    B = T1_CODE;
    mem_w(B+'h00, 32'hE52DB004);  // str  fp, [sp, #-4]!
    mem_w(B+'h04, 32'hE28DB000);  // add  fp, sp, #0
    mem_w(B+'h08, 32'hE24DD00C);  // sub  sp, sp, #12
    mem_w(B+'h0C, 32'hE3A03C02);  // mov  r3, #512         (0x200)
    mem_w(B+'h10, 32'hE50B300C);  // str  r3, [fp, #-12]
    mem_w(B+'h14, 32'hE51B300C);  // ldr  r3, [fp, #-12]
    mem_w(B+'h18, 32'hE5933000);  // ldr  r3, [r3]          (flag)
    mem_w(B+'h1C, 32'hE3530000);  // cmp  r3, #0
    mem_w(B+'h20, 32'h1A000016);  // bne  .L6 (→0x80)
    // ── Encryption path ──
    mem_w(B+'h24, 32'hE3A03000);  // mov  r3, #0            (i)
    mem_w(B+'h28, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h2C, 32'hEA00000C);  // b    .L4 (→0x64)
    // ── .L5: Loop body ──
    mem_w(B+'h30, 32'hE51B300C);  // ldr  r3, [fp, #-12]    (base)
    mem_w(B+'h34, 32'hE51B2008);  // ldr  r2, [fp, #-8]     (i)
    mem_w(B+'h38, 32'hE2822002);  // add  r2, r2, #2
    mem_w(B+'h3C, 32'hE7932102);  // ldr  r2, [r3, r2, lsl #2]
    mem_w(B+'h40, 32'hE59F3048);  // ldr  r3, [pc, #0x48]   (.L7 literal)
    mem_w(B+'h44, 32'hE0233002);  // eor  r3, r3, r2
    mem_w(B+'h48, 32'hE51B200C);  // ldr  r2, [fp, #-12]    (base)
    mem_w(B+'h4C, 32'hE51B1008);  // ldr  r1, [fp, #-8]     (i)
    mem_w(B+'h50, 32'hE2811002);  // add  r1, r1, #2
    mem_w(B+'h54, 32'hE7823101);  // str  r3, [r2, r1, lsl #2]
    mem_w(B+'h58, 32'hE51B3008);  // ldr  r3, [fp, #-8]     (i)
    mem_w(B+'h5C, 32'hE2833001);  // add  r3, r3, #1
    mem_w(B+'h60, 32'hE50B3008);  // str  r3, [fp, #-8]     (i)
    // ── .L4: Loop condition ──
    mem_w(B+'h64, 32'hE51B3008);  // ldr  r3, [fp, #-8]     (i)
    mem_w(B+'h68, 32'hE3530003);  // cmp  r3, #3
    mem_w(B+'h6C, 32'hDAFFFFEF);  // ble  .L5 (→0x30)
    // ── Set done flag ──
    mem_w(B+'h70, 32'hE51B300C);  // ldr  r3, [fp, #-12]    (base)
    mem_w(B+'h74, 32'hE3A02001);  // mov  r2, #1
    mem_w(B+'h78, 32'hE5832000);  // str  r2, [r3]           (flag=1)
    mem_w(B+'h7C, 32'hEA000000);  // b    .L1 (→0x84)
    // ── .L6: Skip path ──
    mem_w(B+'h80, 32'hE1A00000);  // nop
    // ── .L1: Epilogue ──
    mem_w(B+'h84, 32'hE28BD000);  // add  sp, fp, #0
    mem_w(B+'h88, 32'hE49DB004);  // ldr  fp, [sp], #4
    mem_w(B+'h8C, 32'hE12FFF1E);  // bx   lr
    // ── .L7: Literal pool ──
    mem_w(B+'h90, 32'hDEADBEEF);  // .word 0xDEADBEEF

    $display("  [LOAD] Thread 1 code: 0x%08H – 0x%08H (%0d instr + literal)",
             B, B+'h90, (32'h8C/4)+1);
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Thread 2: network_proc2 — counter decrement / threshold
//
//  C logic:
//    base = 0x300;
//    if (base[1] <= 1) { base[0]=0; base[1]=0; }
//    else { base[1]--; base[0]=1; }
// ───────────────────────────────────────────────────────────────────
task load_thread2_code;
    reg [31:0] B;
begin
    B = T2_CODE;
    mem_w(B+'h00, 32'hE52DB004);  // str  fp, [sp, #-4]!
    mem_w(B+'h04, 32'hE28DB000);  // add  fp, sp, #0
    mem_w(B+'h08, 32'hE24DD00C);  // sub  sp, sp, #12
    mem_w(B+'h0C, 32'hE3A03C03);  // mov  r3, #768         (0x300)
    mem_w(B+'h10, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h14, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h18, 32'hE5933004);  // ldr  r3, [r3, #4]      (counter)
    mem_w(B+'h1C, 32'hE3530001);  // cmp  r3, #1
    mem_w(B+'h20, 32'h8A000006);  // bhi  .L2 (→0x40)
    // ── Threshold path (counter <= 1): zero out ──
    mem_w(B+'h24, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h28, 32'hE3A02000);  // mov  r2, #0
    mem_w(B+'h2C, 32'hE5832000);  // str  r2, [r3]           (status=0)
    mem_w(B+'h30, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h34, 32'hE3A02000);  // mov  r2, #0
    mem_w(B+'h38, 32'hE5832004);  // str  r2, [r3, #4]       (counter=0)
    mem_w(B+'h3C, 32'hEA000008);  // b    .L1 (→0x64)
    // ── .L2: Decrement path ──
    mem_w(B+'h40, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h44, 32'hE5933004);  // ldr  r3, [r3, #4]       (counter)
    mem_w(B+'h48, 32'hE2432001);  // sub  r2, r3, #1
    mem_w(B+'h4C, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h50, 32'hE5832004);  // str  r2, [r3, #4]       (counter-1)
    mem_w(B+'h54, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h58, 32'hE3A02001);  // mov  r2, #1
    mem_w(B+'h5C, 32'hE5832000);  // str  r2, [r3]           (status=1)
    mem_w(B+'h60, 32'hE1A00000);  // nop
    // ── .L1: Epilogue ──
    mem_w(B+'h64, 32'hE28BD000);  // add  sp, fp, #0
    mem_w(B+'h68, 32'hE49DB004);  // ldr  fp, [sp], #4
    mem_w(B+'h6C, 32'hE12FFF1E);  // bx   lr

    $display("  [LOAD] Thread 2 code: 0x%08H – 0x%08H (%0d instructions)",
             B, B+'h6C, (32'h6C/4)+1);
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Thread 3: network_proc3 — field comparison
//
//  C logic:
//    base = 0x400;
//    if (base[3] == 23) base[0] = 2;
//    else               base[0] = 1;
// ───────────────────────────────────────────────────────────────────
task load_thread3_code;
    reg [31:0] B;
begin
    B = T3_CODE;
    mem_w(B+'h00, 32'hE52DB004);  // str  fp, [sp, #-4]!
    mem_w(B+'h04, 32'hE28DB000);  // add  fp, sp, #0
    mem_w(B+'h08, 32'hE24DD00C);  // sub  sp, sp, #12
    mem_w(B+'h0C, 32'hE3A03C04);  // mov  r3, #1024        (0x400)
    mem_w(B+'h10, 32'hE50B3008);  // str  r3, [fp, #-8]
    mem_w(B+'h14, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h18, 32'hE593300C);  // ldr  r3, [r3, #12]     (field)
    mem_w(B+'h1C, 32'hE3530017);  // cmp  r3, #23
    mem_w(B+'h20, 32'h1A000003);  // bne  .L2 (→0x34)
    // ── Match path ──
    mem_w(B+'h24, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h28, 32'hE3A02002);  // mov  r2, #2
    mem_w(B+'h2C, 32'hE5832000);  // str  r2, [r3]          (result=2)
    mem_w(B+'h30, 32'hEA000003);  // b    .L1 (→0x44)
    // ── .L2: No-match path ──
    mem_w(B+'h34, 32'hE51B3008);  // ldr  r3, [fp, #-8]
    mem_w(B+'h38, 32'hE3A02001);  // mov  r2, #1
    mem_w(B+'h3C, 32'hE5832000);  // str  r2, [r3]          (result=1)
    mem_w(B+'h40, 32'hE1A00000);  // nop
    // ── .L1: Epilogue ──
    mem_w(B+'h44, 32'hE28BD000);  // add  sp, fp, #0
    mem_w(B+'h48, 32'hE49DB004);  // ldr  fp, [sp], #4
    mem_w(B+'h4C, 32'hE12FFF1E);  // bx   lr

    $display("  [LOAD] Thread 3 code: 0x%08H – 0x%08H (%0d instructions)",
             B, B+'h4C, (32'h4C/4)+1);
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Data Initialization
// ═══════════════════════════════════════════════════════════════════

task init_thread_data;
begin
    // ── Thread 0 data: packet buffer at 0x100 ─────────────────────
    //   [0x100] +0 : unused
    //   [0x104] +4 : header (= 0xAA for good path)
    //   [0x108] +8 : src field  → swap with dst
    //   [0x10C] +12: dst field  → swap with src
    //   [0x110] +16: result     → checksum written here
    //   [0x114] +20: payload[0] → checksum input  (via (i+4)*4+4)
    //   [0x118] +24: payload[1]
    //   [0x11C] +28: payload[2]
    //   [0x120] +32: payload[3]
    mem_w(32'h0100, 32'h0000_0000);   // unused
    mem_w(32'h0104, 32'h0000_00AA);   // header = 0xAA (170)
    mem_w(32'h0108, 32'h1111_1111);   // src
    mem_w(32'h010C, 32'h2222_2222);   // dst
    mem_w(32'h0110, 32'h0000_0000);   // result (will be overwritten)
    mem_w(32'h0114, 32'h0000_0010);   // payload[0] = 16
    mem_w(32'h0118, 32'h0000_0020);   // payload[1] = 32
    mem_w(32'h011C, 32'h0000_0030);   // payload[2] = 48
    mem_w(32'h0120, 32'h0000_0040);   // payload[3] = 64

    // ── Thread 1 data: encryption buffer at 0x200 ─────────────────
    //   [0x200] +0 : flag   (0 = encrypt, 1 = done)
    //   [0x204] +4 : unused
    //   [0x208] +8 : data[0]  (i=0, offset (0+2)*4 = 8)
    //   [0x20C] +12: data[1]
    //   [0x210] +16: data[2]
    //   [0x214] +20: data[3]
    mem_w(32'h0200, 32'h0000_0000);   // flag = 0 (trigger encrypt)
    mem_w(32'h0204, 32'h0000_0000);   // unused
    mem_w(32'h0208, 32'hAAAA_AAAA);   // data[0]
    mem_w(32'h020C, 32'hBBBB_BBBB);   // data[1]
    mem_w(32'h0210, 32'hCCCC_CCCC);   // data[2]
    mem_w(32'h0214, 32'hDDDD_DDDD);   // data[3]

    // ── Thread 2 data: counter at 0x300 ───────────────────────────
    //   [0x300] +0 : status
    //   [0x304] +4 : counter
    mem_w(32'h0300, 32'h0000_0000);   // status (overwritten)
    mem_w(32'h0304, 32'h0000_0005);   // counter = 5 (> 1 → decrement)

    // ── Thread 3 data: comparison at 0x400 ────────────────────────
    //   [0x400] +0 : result
    //   [0x404] +4 : (unused)
    //   [0x408] +8 : (unused)
    //   [0x40C] +12: field to compare
    mem_w(32'h0400, 32'h0000_0000);   // result (overwritten)
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_0017);   // field = 23 (match!)

    $display("  [LOAD] Thread data initialized");
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Run Test — Initialize thread state and execute
// ═══════════════════════════════════════════════════════════════════

task run_mt_test;
    output integer cycles_used;
begin
    // ── Assert reset ──
    rst_n = 0;
    cycle_cnt = 0;
    repeat (5) @(posedge clk);
    @(negedge clk);

    // ── Release reset ──
    rst_n = 1;

    // ── Initialize per-thread PCs (direct hierarchical write) ──
    u_cpu_mt.pc_thread[0] = T0_CODE;
    u_cpu_mt.pc_thread[1] = T1_CODE;
    u_cpu_mt.pc_thread[2] = T2_CODE;
    u_cpu_mt.pc_thread[3] = T3_CODE;

    // ── Initialize per-thread SP (R13) and LR (R14) ──
    u_cpu_mt.THREAD_RF[0].u_rf.regs[13] = T0_SP;
    u_cpu_mt.THREAD_RF[0].u_rf.regs[14] = T0_RET;
    u_cpu_mt.THREAD_RF[1].u_rf.regs[13] = T1_SP;
    u_cpu_mt.THREAD_RF[1].u_rf.regs[14] = T1_RET;
    u_cpu_mt.THREAD_RF[2].u_rf.regs[13] = T2_SP;
    u_cpu_mt.THREAD_RF[2].u_rf.regs[14] = T2_RET;
    u_cpu_mt.THREAD_RF[3].u_rf.regs[13] = T3_SP;
    u_cpu_mt.THREAD_RF[3].u_rf.regs[14] = T3_RET;

    // ── Place sentinel instructions at return addresses ──
    mem_w(T0_RET, SENTINEL);
    mem_w(T1_RET, SENTINEL);
    mem_w(T2_RET, SENTINEL);
    mem_w(T3_RET, SENTINEL);

    $display("  [INIT] PCs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
             T0_CODE, T1_CODE, T2_CODE, T3_CODE);
    $display("  [INIT] SPs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
             T0_SP, T1_SP, T2_SP, T3_SP);
    $display("  [INIT] LRs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
             T0_RET, T1_RET, T2_RET, T3_RET);

    // ── Run until all threads reach sentinel or timeout ──
    begin : run_loop
        forever begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;

            // Print thread completion progress periodically
            if (cycle_cnt > 0 && (cycle_cnt % 500 == 0)) begin
                $display("  [PROGRESS] cycle %0d: sentinel status = %04b",
                         cycle_cnt, thread_at_sentinel);
            end

            // All threads done
            if (all_threads_done) begin
                $display("  [DONE] All 4 threads at sentinel, cycle %0d", cycle_cnt);
                repeat (10) @(posedge clk);
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            // Timeout
            if (cycle_cnt >= MAX_CYCLES) begin
                $display("  *** TIMEOUT after %0d cycles ***", MAX_CYCLES);
                $display("  Thread sentinel status: T0=%0d T1=%0d T2=%0d T3=%0d",
                         thread_at_sentinel[0], thread_at_sentinel[1],
                         thread_at_sentinel[2], thread_at_sentinel[3]);
                $display("  PCs: T0=0x%08H T1=0x%08H T2=0x%08H T3=0x%08H",
                         u_cpu_mt.pc_thread[0], u_cpu_mt.pc_thread[1],
                         u_cpu_mt.pc_thread[2], u_cpu_mt.pc_thread[3]);
                dump_all_threads();
                cycles_used = cycle_cnt;
                disable run_loop;
            end
        end
    end
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  T E S T   S C E N A R I O S
// ═══════════════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────────────
//  Scenario A: All four threads, normal (good) path
//    T0: header=0xAA → swap + checksum
//    T1: flag=0 → encrypt
//    T2: counter=5 (>1) → decrement
//    T3: field=23 → match
// ───────────────────────────────────────────────────────────────────
task test_scenario_A;
    integer cyc;
begin
    section_start("Scenario A: Normal Path (all four threads)");
    mem_clear();

    // Load code for all threads
    load_thread0_code();
    load_thread1_code();
    load_thread2_code();
    load_thread3_code();

    // Initialize data
    init_thread_data();

    // Run
    run_mt_test(cyc);

    $display("");
    $display("  ── Data Memory After Execution ──");
    dump_mem(32'h0100, 10);
    dump_mem(32'h0200, 6);
    dump_mem(32'h0300, 2);
    dump_mem(32'h0400, 4);

    // ── Thread 0 checks: swap + checksum ──
    $display("");
    $display("  ── Thread 0: Packet Processing ──");
    check_mem(32'h0108, 32'h2222_2222, "T0: src←dst after swap");
    check_mem(32'h010C, 32'h1111_1111, "T0: dst←src after swap");
    check_mem(32'h0110, 32'h0000_00A0, "T0: checksum = 16+32+48+64 = 160 = 0xA0");

    // ── Thread 1 checks: XOR encryption ──
    $display("");
    $display("  ── Thread 1: XOR Encryption ──");
    check_mem(32'h0200, 32'h0000_0001, "T1: done flag = 1");
    check_mem(32'h0208, 32'h7407_1445, "T1: 0xAAAAAAAA ^ 0xDEADBEEF");
    check_mem(32'h020C, 32'h6516_0554, "T1: 0xBBBBBBBB ^ 0xDEADBEEF");
    check_mem(32'h0210, 32'h1261_7223, "T1: 0xCCCCCCCC ^ 0xDEADBEEF");
    check_mem(32'h0214, 32'h0370_6332, "T1: 0xDDDDDDDD ^ 0xDEADBEEF");

    // ── Thread 2 checks: counter decrement ──
    $display("");
    $display("  ── Thread 2: Counter Decrement ──");
    check_mem(32'h0300, 32'h0000_0001, "T2: status = 1 (active)");
    check_mem(32'h0304, 32'h0000_0004, "T2: counter = 5-1 = 4");

    // ── Thread 3 checks: field comparison ──
    $display("");
    $display("  ── Thread 3: Field Comparison ──");
    check_mem(32'h0400, 32'h0000_0002, "T3: result = 2 (field == 23)");

    // ── Stack pointer restoration ──
    $display("");
    $display("  ── Stack Pointer Restoration ──");
    check_reg_t(2'd0, 4'd13, T0_SP, "T0: SP restored");
    check_reg_t(2'd1, 4'd13, T1_SP, "T1: SP restored");
    check_reg_t(2'd2, 4'd13, T2_SP, "T2: SP restored");
    check_reg_t(2'd3, 4'd13, T3_SP, "T3: SP restored");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Scenario B: Error / alternate paths
//    T0: header != 0xAA → error path, result = -1
//    T1: flag = 1 → skip encryption (NOP path)
//    T2: counter = 1 (<=1) → zero out
//    T3: field != 23 → no-match, result = 1
// ───────────────────────────────────────────────────────────────────
task test_scenario_B;
    integer cyc;
begin
    section_start("Scenario B: Alternate / Error Paths");
    mem_clear();

    // Load code
    load_thread0_code();
    load_thread1_code();
    load_thread2_code();
    load_thread3_code();

    // ── Thread 0 data: wrong header ──
    mem_w(32'h0100, 32'h0000_0000);
    mem_w(32'h0104, 32'h0000_00BB);   // header = 0xBB (NOT 0xAA)
    mem_w(32'h0108, 32'h1111_1111);
    mem_w(32'h010C, 32'h2222_2222);
    mem_w(32'h0110, 32'h0000_0000);
    mem_w(32'h0114, 32'h0000_0010);
    mem_w(32'h0118, 32'h0000_0020);
    mem_w(32'h011C, 32'h0000_0030);
    mem_w(32'h0120, 32'h0000_0040);

    // ── Thread 1 data: flag already set ──
    mem_w(32'h0200, 32'h0000_0001);   // flag = 1 → skip
    mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hAAAA_AAAA);
    mem_w(32'h020C, 32'hBBBB_BBBB);
    mem_w(32'h0210, 32'hCCCC_CCCC);
    mem_w(32'h0214, 32'hDDDD_DDDD);

    // ── Thread 2 data: counter at threshold ──
    mem_w(32'h0300, 32'hFFFF_FFFF);   // status (garbage, will be overwritten)
    mem_w(32'h0304, 32'h0000_0001);   // counter = 1 (<=1 → zero)

    // ── Thread 3 data: field mismatch ──
    mem_w(32'h0400, 32'h0000_0000);
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_002A);   // field = 42 (NOT 23)

    run_mt_test(cyc);

    $display("");
    dump_mem(32'h0100, 6);
    dump_mem(32'h0200, 6);
    dump_mem(32'h0300, 2);
    dump_mem(32'h0400, 4);

    // ── Thread 0: error path ──
    $display("");
    $display("  ── Thread 0: Error Path (bad header) ──");
    check_mem(32'h0110, 32'hFFFF_FFFF, "T0: result = -1 (error)");
    check_mem(32'h0108, 32'h1111_1111, "T0: src unchanged (no swap)");
    check_mem(32'h010C, 32'h2222_2222, "T0: dst unchanged (no swap)");

    // ── Thread 1: skip path ──
    $display("");
    $display("  ── Thread 1: Skip Path (flag=1) ──");
    check_mem(32'h0200, 32'h0000_0001, "T1: flag unchanged = 1");
    check_mem(32'h0208, 32'hAAAA_AAAA, "T1: data[0] unmodified");
    check_mem(32'h020C, 32'hBBBB_BBBB, "T1: data[1] unmodified");
    check_mem(32'h0210, 32'hCCCC_CCCC, "T1: data[2] unmodified");
    check_mem(32'h0214, 32'hDDDD_DDDD, "T1: data[3] unmodified");

    // ── Thread 2: zero path ──
    $display("");
    $display("  ── Thread 2: Zero Path (counter<=1) ──");
    check_mem(32'h0300, 32'h0000_0000, "T2: status = 0 (inactive)");
    check_mem(32'h0304, 32'h0000_0000, "T2: counter = 0");

    // ── Thread 3: no-match ──
    $display("");
    $display("  ── Thread 3: No-Match Path (field=42) ──");
    check_mem(32'h0400, 32'h0000_0001, "T3: result = 1 (no match)");

    // ── SPs ──
    $display("");
    check_reg_t(2'd0, 4'd13, T0_SP, "T0: SP restored");
    check_reg_t(2'd1, 4'd13, T1_SP, "T1: SP restored");
    check_reg_t(2'd2, 4'd13, T2_SP, "T2: SP restored");
    check_reg_t(2'd3, 4'd13, T3_SP, "T3: SP restored");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Scenario C: Edge-case data values
//    T0: header=0xAA, payload = large values (overflow test)
//    T1: flag=0, data = 0xDEADBEEF (XOR with self → 0)
//    T2: counter=0 (<=1 → zero path)
//    T3: field=23 (match)
// ───────────────────────────────────────────────────────────────────
task test_scenario_C;
    integer cyc;
begin
    section_start("Scenario C: Edge-Case Data Values");
    mem_clear();

    load_thread0_code();
    load_thread1_code();
    load_thread2_code();
    load_thread3_code();

    // ── T0: large payload values ──
    mem_w(32'h0100, 32'h0000_0000);
    mem_w(32'h0104, 32'h0000_00AA);   // header OK
    mem_w(32'h0108, 32'hFFFF_FFFF);   // src = -1
    mem_w(32'h010C, 32'h0000_0001);   // dst = 1
    mem_w(32'h0110, 32'h0000_0000);
    mem_w(32'h0114, 32'h7FFF_FFFF);   // payload[0] = MAX_INT
    mem_w(32'h0118, 32'h0000_0001);   // payload[1] = 1
    mem_w(32'h011C, 32'h8000_0000);   // payload[2] = MIN_INT
    mem_w(32'h0120, 32'h0000_0000);   // payload[3] = 0

    // ── T1: XOR with itself ──
    mem_w(32'h0200, 32'h0000_0000);   // flag=0
    mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'hDEAD_BEEF);   // XOR → 0
    mem_w(32'h020C, 32'h0000_0000);   // XOR → 0xDEADBEEF
    mem_w(32'h0210, 32'hFFFF_FFFF);   // XOR → ~0xDEADBEEF = 0x21524110
    mem_w(32'h0214, 32'h1234_5678);   // XOR → specific value

    // ── T2: counter=0 ──
    mem_w(32'h0300, 32'hAAAA_AAAA);
    mem_w(32'h0304, 32'h0000_0000);   // counter = 0 (<=1 → zero)

    // ── T3: field=23 ──
    mem_w(32'h0400, 32'h0000_0000);
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_0017);   // 23

    run_mt_test(cyc);

    $display("");
    dump_mem(32'h0100, 10);
    dump_mem(32'h0200, 6);
    dump_mem(32'h0300, 2);
    dump_mem(32'h0400, 4);

    // ── T0: checksum = 0x7FFFFFFF + 1 + 0x80000000 + 0 = 0x00000000 (wraps) ──
    $display("");
    $display("  ── Thread 0: Large value checksum (wrapping) ──");
    check_mem(32'h0108, 32'h0000_0001, "T0: src←dst swapped");
    check_mem(32'h010C, 32'hFFFF_FFFF, "T0: dst←src swapped");
    check_mem(32'h0110, 32'h0000_0000, "T0: checksum wraps to 0");

    // ── T1: XOR edge cases ──
    $display("");
    $display("  ── Thread 1: XOR edge cases ──");
    check_mem(32'h0200, 32'h0000_0001, "T1: done flag");
    check_mem(32'h0208, 32'h0000_0000, "T1: DEADBEEF^DEADBEEF=0");
    check_mem(32'h020C, 32'hDEAD_BEEF, "T1: 0^DEADBEEF=DEADBEEF");
    check_mem(32'h0210, 32'h2152_4110, "T1: FFFFFFFF^DEADBEEF=21524110");
    check_mem(32'h0214, 32'hCC99_E897, "T1: 12345678^DEADBEEF");

    // ── T2: counter=0 path ──
    $display("");
    $display("  ── Thread 2: Counter=0 (zero path) ──");
    check_mem(32'h0300, 32'h0000_0000, "T2: status=0");
    check_mem(32'h0304, 32'h0000_0000, "T2: counter=0");

    // ── T3: match ──
    $display("");
    check_mem(32'h0400, 32'h0000_0002, "T3: result=2 (match)");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  Scenario D: Verify thread isolation — data independence
//    Run same as scenario A but with different data patterns
//    to confirm threads don't corrupt each other's registers or memory
// ───────────────────────────────────────────────────────────────────
task test_scenario_D;
    integer cyc;
begin
    section_start("Scenario D: Thread Isolation Verification");
    mem_clear();

    load_thread0_code();
    load_thread1_code();
    load_thread2_code();
    load_thread3_code();

    // ── T0 data ──
    mem_w(32'h0100, 32'h0000_0000);
    mem_w(32'h0104, 32'h0000_00AA);   // header OK
    mem_w(32'h0108, 32'hAAAA_0000);   // src
    mem_w(32'h010C, 32'h0000_BBBB);   // dst
    mem_w(32'h0110, 32'h0000_0000);
    mem_w(32'h0114, 32'h0000_0001);   // payload
    mem_w(32'h0118, 32'h0000_0002);
    mem_w(32'h011C, 32'h0000_0003);
    mem_w(32'h0120, 32'h0000_0004);

    // ── T1 data ──
    mem_w(32'h0200, 32'h0000_0000);   // encrypt
    mem_w(32'h0204, 32'h0000_0000);
    mem_w(32'h0208, 32'h0102_0304);
    mem_w(32'h020C, 32'h0506_0708);
    mem_w(32'h0210, 32'h090A_0B0C);
    mem_w(32'h0214, 32'h0D0E_0F10);

    // ── T2 data ──
    mem_w(32'h0300, 32'h0000_0000);
    mem_w(32'h0304, 32'h0000_000A);   // counter=10

    // ── T3 data ──
    mem_w(32'h0400, 32'h0000_0000);
    mem_w(32'h0404, 32'h0000_0000);
    mem_w(32'h0408, 32'h0000_0000);
    mem_w(32'h040C, 32'h0000_0017);   // 23

    run_mt_test(cyc);

    $display("");
    dump_mem(32'h0100, 10);
    dump_mem(32'h0200, 6);

    // ── T0 ──
    $display("");
    $display("  ── Thread 0: Isolation check ──");
    check_mem(32'h0108, 32'h0000_BBBB, "T0: src←dst");
    check_mem(32'h010C, 32'hAAAA_0000, "T0: dst←src");
    check_mem(32'h0110, 32'h0000_000A, "T0: checksum=1+2+3+4=10");

    // ── T1 ──
    $display("");
    $display("  ── Thread 1: Isolation check ──");
    check_mem(32'h0200, 32'h0000_0001, "T1: flag=1");
    check_mem(32'h0208, 32'hDFAF_BDEB, "T1: 01020304^DEADBEEF");
    check_mem(32'h020C, 32'hDBAB_B9E7, "T1: 05060708^DEADBEEF");
    check_mem(32'h0210, 32'hD7A7_B5E3, "T1: 090A0B0C^DEADBEEF");
    check_mem(32'h0214, 32'hD3A3_B1FF, "T1: 0D0E0F10^DEADBEEF");

    // ── T2 ──
    $display("");
    $display("  ── Thread 2: Isolation check ──");
    check_mem(32'h0300, 32'h0000_0001, "T2: status=1");
    check_mem(32'h0304, 32'h0000_0009, "T2: counter=10-1=9");

    // ── T3 ──
    $display("");
    check_mem(32'h0400, 32'h0000_0002, "T3: result=2 (match)");

    // ── Verify threads didn't corrupt each other's data regions ──
    $display("");
    $display("  ── Cross-thread data integrity ──");
    check_mem(32'h0104, 32'h0000_00AA, "T0 header not corrupted by other threads");
    check_mem(32'h0204, 32'h0000_0000, "T1 pad not corrupted");
    check_mem(32'h040C, 32'h0000_0017, "T3 field not corrupted");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
//  M A I N   S T I M U L U S
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════

initial begin
    $dumpfile("cpu_mt_tb.vcd");
    $dumpvars(0, cpu_mt_tb);

    total_pass = 0;
    total_fail = 0;
    ila_thread_sel = 2'd0;
    ila_debug_sel  = 5'd0;
    rst_n = 0;

    $display("");
    $display("╔══════════════════════════════════════════════════════════════════════╗");
    $display("║   Quad-Thread ARMv4T Multithreaded Pipeline Testbench              ║");
    $display("║   SYNC_MEM=%0d  TRACE_EN=%0d  MAX_CYCLES=%0d                          ║",
             SYNC_MEM, TRACE_EN, MAX_CYCLES);
    $display("║                                                                    ║");
    $display("║   Thread 0: network_proc0 (packet header/swap/checksum)            ║");
    $display("║   Thread 1: network_proc1 (XOR encryption)                         ║");
    $display("║   Thread 2: network_proc2 (counter decrement)                      ║");
    $display("║   Thread 3: network_proc3 (field comparison)                       ║");
    $display("╚══════════════════════════════════════════════════════════════════════╝");
    $display("");

    test_scenario_A();
    test_scenario_B();
    test_scenario_C();
    test_scenario_D();

    $display("");
    $display("╔══════════════════════════════════════════════════════════════════════╗");
    if (total_fail == 0)
        $display("║  *** ALL %4d CHECKS PASSED ***                                    ║",
                 total_pass);
    else
        $display("║  *** %4d PASSED, %4d FAILED ***                                   ║",
                 total_pass, total_fail);
    $display("║  Total checks: %4d                                                 ║",
             total_pass + total_fail);
    $display("╚══════════════════════════════════════════════════════════════════════╝");
    $display("");

    #(CLK_PERIOD * 5);
    $finish;
end

endmodule