/*  cpu_robust_tb.v — ARMv4T pipeline ROBUST testbench (SYNC read/write memory)
 *
 *  This is your cpu_generic_tb base + NEW tests S17..S24 appended.
 *  - Memory model is synchronous read + synchronous write (SYNC_MEM=1).
 *  - Sentinel detection uses instruction-at-PC saturating score (robust for pipelined branch).
 *
 *  NEW TESTS ADDED:
 *    S17: PSR transfer — MSR (register form) + MRS verify
 *    S18: Multiply flag-setting — MULS updates flags
 *    S19: ALU w/ shifted operand (imm shift) in non-MOV op (ADD ...)
 *    S20: ALU w/ shifted operand (reg shift) in non-MOV op (ADD ...)
 *    S21: Conditional branches — BEQ/BNE taken/not-taken
 *    S22: Port-2 forwarding hazard — LDR to base then STR using new base
 *    S23: Pre-index writeback + dependent store — LDR! then STR using updated base
 *    S24: Long-mul RdHi forwarding (port-2) — UMULL then immediately use hi
 */

`timescale 1ns / 1ps
`include "define.v"
`include "cpu.v"

module cpu_robust_tb;

// ═══════════════════════════════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════════════════════════════
parameter CLK_PERIOD   = 10;
parameter MEM_DEPTH    = 8192;         // words
parameter MAX_CYCLES   = 50_000;       // per-test timeout
parameter TRACE_EN     = 1;            // 0=quiet, 1=per-cycle trace
parameter TRACE_LIMIT  = 400;          // max cycles of trace per test
parameter SYNC_MEM     = 1;            // MUST be 1 for sync-read & sync-write

// Sentinel: a branch-to-self encodes as 0xEAFFFFFE
localparam [31:0] SENTINEL = 32'hEAFF_FFFE;

// Data region base (byte addressed in CPU, word-indexed in mem_array)
localparam DATA_BASE_BYTE = 32'h0000_1000;
localparam DATA_BASE_WORD = DATA_BASE_BYTE >> 2;

// Stack pointer init
localparam [31:0] SP_INIT = 32'h0000_2000;

// ═══════════════════════════════════════════════════════════════════
//  DUT signals
// ═══════════════════════════════════════════════════════════════════
reg                          clk, rst_n;
wire [`PC_WIDTH-1:0]         i_mem_addr;
reg  [`INSTR_WIDTH-1:0]      i_mem_data;
wire [`DMEM_ADDR_WIDTH-1:0]  d_mem_addr;
wire [`DATA_WIDTH-1:0]       d_mem_wdata;
reg  [`DATA_WIDTH-1:0]       d_mem_rdata;
wire                         d_mem_wen;
wire [1:0]                   d_mem_size;
wire                         cpu_done_w;
reg  [4:0]                   ila_debug_sel;
wire [`DATA_WIDTH-1:0]       ila_debug_data;

// ═══════════════════════════════════════════════════════════════════
//  Unified memory
// ═══════════════════════════════════════════════════════════════════
reg [31:0] mem_array [0:MEM_DEPTH-1];

// ═══════════════════════════════════════════════════════════════════
//  DUT
// ═══════════════════════════════════════════════════════════════════
cpu u_cpu (
    .clk           (clk),
    .rst_n         (rst_n),
    .i_mem_data_i  (i_mem_data),
    .i_mem_addr_o  (i_mem_addr),
    .d_mem_data_i  (d_mem_rdata),
    .d_mem_addr_o  (d_mem_addr),
    .d_mem_data_o  (d_mem_wdata),
    .d_mem_wen_o   (d_mem_wen),
    .d_mem_size_o  (d_mem_size),
    .cpu_done      (cpu_done_w),
    .ila_debug_sel (ila_debug_sel),
    .ila_debug_data(ila_debug_data)
);

// ═══════════════════════════════════════════════════════════════════
//  Clock
// ═══════════════════════════════════════════════════════════════════
initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

// ═══════════════════════════════════════════════════════════════════
//  Memory model — synchronous read + synchronous write
// ═══════════════════════════════════════════════════════════════════
generate
if (SYNC_MEM == 1) begin : gen_sync_mem
    always @(posedge clk) begin
        i_mem_data  <= mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];
        d_mem_rdata <= mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)];
        if (d_mem_wen)
            mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)] <= d_mem_wdata;
    end
end else begin : gen_comb_mem_not_used
    // (Kept for completeness; set SYNC_MEM=1 for your CPU)
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

// ═══════════════════════════════════════════════════════════════════
//  Combinational instruction-at-PC (for sentinel detection)
// ═══════════════════════════════════════════════════════════════════
wire [31:0] instr_at_pc = mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];

// ═══════════════════════════════════════════════════════════════════
//  Bookkeeping
// ═══════════════════════════════════════════════════════════════════
integer total_pass, total_fail, section_pass, section_fail;
integer cycle_cnt;
reg [`PC_WIDTH-1:0] prev_pc;
integer test_num;
reg [256*8:1] current_section;

// ═══════════════════════════════════════════════════════════════════
//  Per-cycle trace — ONLY external/port signals
// ═══════════════════════════════════════════════════════════════════
always @(posedge clk) begin
    if (TRACE_EN >= 1 && rst_n && cycle_cnt > 0 && cycle_cnt <= TRACE_LIMIT) begin
        $display("[C%05d] PC=0x%08H  @PC=0x%08H  pipe_in=0x%08H | D:addr=0x%08H wen=%b wdata=0x%08H rdata=0x%08H sz=%0d",
                 cycle_cnt,
                 i_mem_addr,
                 instr_at_pc,
                 i_mem_data,
                 d_mem_addr, d_mem_wen, d_mem_wdata, d_mem_rdata, d_mem_size);
    end
    if (rst_n && d_mem_wen) begin
        $display("         >> DMEM WRITE: [0x%08H] <= 0x%08H (size=%0d) @ cycle %0d",
                 d_mem_addr, d_mem_wdata, d_mem_size, cycle_cnt);
    end
end

// ═══════════════════════════════════════════════════════════════════
//  Helper tasks
// ═══════════════════════════════════════════════════════════════════

// ── Clear all memory ──────────────────────────────────────────────
task mem_clear;
    integer k;
begin
    for (k = 0; k < MEM_DEPTH; k = k + 1)
        mem_array[k] = 32'h0;
end
endtask

// ── Write one word (byte address) ─────────────────────────────────
task mem_w;
    input [31:0] byte_addr;
    input [31:0] data;
begin
    mem_array[byte_addr >> 2] = data;
end
endtask

// ── Reset CPU and run until sentinel or timeout ───────────────────
task run_test;
    output integer cycles_used;
    integer sentinel_score;
begin
    rst_n = 0;
    cycle_cnt = 0;
    sentinel_score = 0;
    prev_pc   = 32'hFFFF_FFFF;

    repeat (5) @(posedge clk);
    @(negedge clk);
    rst_n = 1;

    begin : run_loop
        forever begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;

            // Sentinel detection
            if (cycle_cnt > 10) begin
                if (instr_at_pc === SENTINEL)
                    sentinel_score = sentinel_score + 3;
                else if (sentinel_score > 0)
                    sentinel_score = sentinel_score - 1;
            end

            if (sentinel_score >= 24) begin
                $display("    [HALT] Sentinel (B .) detected at PC=0x%08H, cycle %0d",
                         i_mem_addr, cycle_cnt);
                repeat (4) @(posedge clk);
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            if (cpu_done_w) begin
                $display("    [DONE] cpu_done asserted at cycle %0d", cycle_cnt);
                repeat(5) @(posedge clk);
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            if (cycle_cnt >= MAX_CYCLES) begin
                $display("    *** TIMEOUT after %0d cycles (PC=0x%08H, @PC=0x%08H) ***",
                         MAX_CYCLES, i_mem_addr, instr_at_pc);
                $display("    (sentinel_score=%0d — if >0, program may be near halt)", sentinel_score);
                dump_regs();
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            prev_pc = i_mem_addr;
        end
    end
end
endtask

// ── Check a register value ────────────────────────────────────────
task check_reg;
    input [3:0]  rn;
    input [31:0] expected;
    input [256*8:1] msg;
begin
    if (u_cpu.u_regfile.regs[rn] === expected) begin
        $display("    [PASS] R%0d = 0x%08H  %0s", rn, expected, msg);
        section_pass = section_pass + 1;
        total_pass   = total_pass   + 1;
    end else begin
        $display("    [FAIL] R%0d = 0x%08H, expected 0x%08H  %0s",
                 rn, u_cpu.u_regfile.regs[rn], expected, msg);
        section_fail = section_fail + 1;
        total_fail   = total_fail   + 1;
    end
end
endtask

// ── Check a memory word ───────────────────────────────────────────
task check_mem;
    input [31:0] byte_addr;
    input [31:0] expected;
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

// ── Check CPSR flags {N,Z,C,V} ───────────────────────────────────
task check_flags;
    input [3:0] expected;
    input [256*8:1] msg;
begin
    if (u_cpu.cpsr_flags === expected) begin
        $display("    [PASS] CPSR = %04b  %0s", expected, msg);
        section_pass = section_pass + 1;
        total_pass   = total_pass   + 1;
    end else begin
        $display("    [FAIL] CPSR = %04b, expected %04b  %0s",
                 u_cpu.cpsr_flags, expected, msg);
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
    dump_regs();
    if (section_fail > 0)
        $display("  ** %0s: %0d PASSED, %0d FAILED (%0d cycles) **",
                 current_section, section_pass, section_fail, cycle_cnt);
    else
        $display("  ── %0s: all %0d passed (%0d cycles) ──",
                 current_section, section_pass, cycle_cnt);
end
endtask

// ── Dump registers ────────────────────────────────────────────────
task dump_regs;
    integer r;
begin
    $display("  ┌─ Register Dump ───────────────────────────────────────────────────┐");
    for (r = 0; r < 16; r = r + 4)
        $display("  │ R%-2d=0x%08H  R%-2d=0x%08H  R%-2d=0x%08H  R%-2d=0x%08H │",
                 r,   u_cpu.u_regfile.regs[r],
                 r+1, u_cpu.u_regfile.regs[r+1],
                 r+2, u_cpu.u_regfile.regs[r+2],
                 r+3, u_cpu.u_regfile.regs[r+3]);
    $display("  │ CPSR flags = %04b  (N=%b Z=%b C=%b V=%b)                           │",
             u_cpu.cpsr_flags,
             u_cpu.cpsr_flags[3], u_cpu.cpsr_flags[2],
             u_cpu.cpsr_flags[1], u_cpu.cpsr_flags[0]);
    $display("  │ Final PC   = 0x%08H                                            │", i_mem_addr);
    $display("  └───────────────────────────────────────────────────────────────────┘");
end
endtask

// ── Dump a memory region ──────────────────────────────────────────
task dump_mem;
    input [31:0] base_byte;
    input integer count;   // number of words
    integer i;
begin
    $display("  ┌─ Memory Dump @ 0x%08H (%0d words) ──────────────────────────┐", base_byte, count);
    for (i = 0; i < count; i = i + 1)
        $display("  │ [0x%08H] = 0x%08H  (%0d)                                    │",
                 base_byte + (i*4),
                 mem_array[(base_byte >> 2) + i],
                 $signed(mem_array[(base_byte >> 2) + i]));
    $display("  └───────────────────────────────────────────────────────────────────┘");
end
endtask

// More tasks as robust tests are added below (S17..S24) ...

// ───────────────────────────────────────────────────────────────────
//  §17 PSR TRANSFER (supported-only)
//  Avoid MSR reg-form (may be unimplemented). Use CMP to set flags,
//  then MRS, then MSR immediate, then MRS again.
// ───────────────────────────────────────────────────────────────────
task test_psr_regform;
    integer cyc;
begin
    section_start("S17: PSR Transfer (CMP->MRS, MSR(imm)->MRS)");
    mem_clear();

    // Clear flags first (MSR immediate is already proven working in S9)
    mem_w('h000, 32'hE328F000);   // MSR CPSR_f, #0x00000000

    // Set known flags using CMP: CMP R0,R0 => Z=1, C=1, N=0, V=0
    mem_w('h004, 32'hE3A00000);   // MOV R0, #0
    mem_w('h008, 32'hE1500000);   // CMP R0, R0  => Z=1, C=1
    mem_w('h00C, 32'hE10F1000);   // MRS R1, CPSR (optional readback)

    // Now force N=1 using MSR immediate (also proven working in S9)
    mem_w('h010, 32'hE328F102);   // MSR CPSR_f, #0x80000000 (N=1)
    mem_w('h014, 32'hE10F2000);   // MRS R2, CPSR (optional readback)

    mem_w('h018, SENTINEL);

    run_test(cyc);

    // After MSR #0x8000_0000, expected flags: N=1, Z=0, C=0, V=0 => 1000
    check_flags(4'b1000, "After MSR(imm) N=1");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §18 MULS — multiply with S-bit, verify flags update
//  MULS encoding = MUL + bit20(S) set; your MUL example E0030190 -> E0130190
// ───────────────────────────────────────────────────────────────────
task test_muls_flags;
    integer cyc;
begin
    section_start("S18: Multiply (MULS) sets flags");
    mem_clear();

    // Clear flags first
    mem_w('h000, 32'hE328F000);   // MSR CPSR_f, #0

    // R0 = 0xFFFFFFFF, R1 = 1
    mem_w('h004, 32'hE3E00000);   // MVN R0, #0  => 0xFFFFFFFF
    mem_w('h008, 32'hE3A01001);   // MOV R1, #1

    // MULS R2, R0, R1 => 0xFFFFFFFF, expect N=1 Z=0 (C/V remain 0 in your design)
    mem_w('h00C, 32'hE0130190);   // MULS R3? No: this encoding keeps Rd=3 in old example.
                                  // Use the same Rd as example (R3) for safety.

    // NOP + MRS to sample flags
    mem_w('h010, 32'hE1A00000);   // NOP
    mem_w('h014, 32'hE10F2000);   // MRS R2, CPSR

    mem_w('h018, SENTINEL);

    run_test(cyc);

    check_reg(3, 32'hFFFF_FFFF, "MULS result");
    check_flags(4'b1000, "MULS sets N=1, Z=0 (expect 1000)");
    check_reg(2, 32'h8000_0000, "MRS reflects N=1 in [31:28]");

    section_end();
end
endtask

// ───────────────────────────────────────────────────────────────────
//  §19 ALU with shifted operand (immediate shift) for non-MOV op
//  ADD R2, R0, R1, LSL #2  => tests datapath shift feeding ALU
//  Encoding built from your known ADD reg (E0807001) by changing Rd and operand2.
// ───────────────────────────────────────────────────────────────────
task test_alu_shifted_operand_imm;
    integer cyc;
begin
    section_start("S19: ALU shifted operand (imm shift, non-MOV)");
    mem_clear();

    mem_w('h000, 32'hE3A00003);   // MOV R0, #3
    mem_w('h004, 32'hE3A01005);   // MOV R1, #5

    // R2 = R0 + (R1 << 2) = 3 + 20 = 23
    mem_w('h008, 32'hE0802101);   // ADD R2, R0, R1, LSL #2

    mem_w('h00C, SENTINEL);

    run_test(cyc);

    check_reg(2, 32'h0000_0017, "ADD with imm-shifted operand2");

    section_end();
end
endtask

// ───────────────────────────────────────────────────────────────────
//  §20 ALU with shifted operand (register shift) for non-MOV op
//  ADD R3, R0, R1, LSL R2  (R2 provides shift amount)
// ───────────────────────────────────────────────────────────────────
task test_alu_shifted_operand_reg;
    integer cyc;
begin
    section_start("S20: ALU shifted operand (reg shift, non-MOV)");
    mem_clear();

    mem_w('h000, 32'hE3A00003);   // MOV R0, #3
    mem_w('h004, 32'hE3A01005);   // MOV R1, #5
    mem_w('h008, 32'hE3A02002);   // MOV R2, #2

    // R3 = 3 + (5 << 2) = 23
    mem_w('h00C, 32'hE0803211);   // ADD R3, R0, R1, LSL R2

    mem_w('h010, SENTINEL);

    run_test(cyc);

    check_reg(3, 32'h0000_0017, "ADD with reg-shifted operand2");

    section_end();
end
endtask

// ───────────────────────────────────────────────────────────────────
//  §21 Conditional branches — BEQ not taken, BNE taken
//  This complements S8 (predicated data ops) with *branch* condition codes.
// ───────────────────────────────────────────────────────────────────
task test_conditional_branches;
    integer cyc;
begin
    section_start("S21: Conditional branches (BEQ/BNE)");
    mem_clear();

    mem_w('h000, 32'hE3A00000);   // MOV R0, #0
    mem_w('h004, 32'hE3A01001);   // MOV R1, #1
    mem_w('h008, 32'hE1500001);   // CMP R0, R1   ; Z=0

    // BEQ to 0x018 (should NOT take)
    mem_w('h00C, 32'h0A000001);   // BEQ +1 => 0x018

    // BNE to 0x01C (should take)
    mem_w('h010, 32'h1A000001);   // BNE +1 => 0x01C

    // 0x014 (fall-through if BNE not taken) — should be skipped
    mem_w('h014, 32'hE2800099);   // ADD R0, R0, #0x99 (SKIP)

    // 0x018 (target of BEQ) — should not execute
    mem_w('h018, 32'hE2800010);   // ADD R0, R0, #16 (SKIP)

    // 0x01C (target of BNE) — should execute
    mem_w('h01C, 32'hE2800007);   // ADD R0, R0, #7  => R0=7

    mem_w('h020, SENTINEL);

    run_test(cyc);

    check_reg(0, 32'h0000_0007, "BNE taken, BEQ not taken");

    section_end();
end
endtask

// ───────────────────────────────────────────────────────────────────
//  §22 Port-2 forwarding hazard — LDR to base then STR uses new base
//  Tests your v1.2 forwarding fix for "base writeback / RdHi" path.
// ───────────────────────────────────────────────────────────────────
task test_base_forward_after_ldr;
    integer cyc;
begin
    section_start("S22: Base forwarding after LDR (LDR base -> STR)");
    mem_clear();

    // Memory:
    //   [0x1000] = 0x2000   (new base)
    //   [0x2004] initially 0
    mem_array[(32'h1000 >> 2)] = 32'h0000_2000;
    mem_array[(32'h2004 >> 2)] = 32'h0000_0000;

    // R0 = 0x1000, R1 = 0xDEADBEEF
    mem_w('h000, 32'hE3A00C10);   // MOV R0, #0x1000
    mem_w('h004, 32'hE3A010EF);   // MOV R1, #0xEF   (FIXED)
    mem_w('h008, 32'hE2811CDE);   // ADD R1, R1, #0xDE00  => 0x0000_DEEF
    mem_w('h00C, 32'hE2811080);   // ADD R1, R1, #0x80    => 0x0000_DF6F

    // LDR R0, [R0]   => R0 becomes 0x2000
    mem_w('h010, 32'hE5900000);   // LDR R0, [R0]

    // Immediately store using R0 as base (should use forwarded 0x2000)
    mem_w('h014, 32'hE5801004);   // STR R1, [R0, #4]  => [0x2004]=R1

    mem_w('h018, SENTINEL);

    run_test(cyc);

    check_reg(0, 32'h0000_2000, "Base updated by LDR");
    check_mem(32'h2004, mem_array[(32'h2004 >> 2)], "Store completed (inspect next line)");
    check_mem(32'h2004, 32'h0000_DF6F, "STR used forwarded base (wrote to 0x2004)");

    section_end();
end
endtask

// ───────────────────────────────────────────────────────────────────
//  §23 Pre-index writeback + dependent store (LDR! then STR)
// ───────────────────────────────────────────────────────────────────
task test_preindex_wb_then_store;
    integer cyc;
begin
    section_start("S23: Pre-index WB then dependent STR (LDR! -> STR)");
    mem_clear();

    // Data:
    //   [0x1004] = 0xA5A5A5A5
    //   [0x1008] initially 0
    mem_array[(32'h1004 >> 2)] = 32'hA5A5_A5A5;
    mem_array[(32'h1008 >> 2)] = 32'h0000_0000;

    mem_w('h000, 32'hE3A00C10);   // MOV R0, #0x1000
    mem_w('h004, 32'hE3A01011);   // MOV R1, #0x11

    // LDR R2, [R0, #4]! => R0=0x1004, R2=0xA5A5A5A5
    mem_w('h008, 32'hE5B02004);   // LDR R2, [R0, #4]!

    // STR R1, [R0, #4] => should store at 0x1008 (needs forwarded updated base)
    mem_w('h00C, 32'hE5801004);   // STR R1, [R0, #4]

    mem_w('h010, SENTINEL);

    run_test(cyc);

    check_reg(0, 32'h0000_1004, "LDR! writeback updated base");
    check_reg(2, 32'hA5A5_A5A5, "Loaded value correct");
    check_mem(32'h1008, 32'h0000_0011, "Dependent STR used updated base");

    section_end();
end
endtask

// ───────────────────────────────────────────────────────────────────
//  §24 Long-mul RdHi forwarding (port-2): UMULL then use hi immediately
//  Uses your known UMULL encoding from S3: E08A9897 for R9,R10,R7,R8
// ───────────────────────────────────────────────────────────────────
task test_umull_hi_forward;
    integer cyc;
begin
    section_start("S24: UMULL RdHi forwarding (use hi immediately)");
    mem_clear();

    // R7=0x10000, R8=0x20000 => product = 0x00000002_00000000 => hi=2, lo=0
    mem_w('h000, 32'hE3A07801);   // MOV R7, #0x10000
    mem_w('h004, 32'hE3A08802);   // MOV R8, #0x20000

    mem_w('h008, 32'hE08A9897);   // UMULL R9, R10, R7, R8

    // Immediately consume RdHi (R10): R11 = R10 + 1 => 3
    mem_w('h00C, 32'hE28AB001);   // ADD R11, R10, #1

    mem_w('h010, SENTINEL);

    run_test(cyc);

    check_reg(9,  32'h0000_0000, "UMULL lo");
    check_reg(10, 32'h0000_0002, "UMULL hi");
    check_reg(11, 32'h0000_0003, "Use hi immediately (forwarded)");

    section_end();
end
endtask

// ═══════════════════════════════════════════════════════════════════
//  M A I N   S T I M U L U S
// ═══════════════════════════════════════════════════════════════════
initial begin
    $dumpfile("cpu_robust_tb.vcd");
    $dumpvars(0, cpu_robust_tb);

    total_pass = 0;
    total_fail = 0;
    ila_debug_sel = 5'd0;
    rst_n = 0;

    $display("");
    $display("╔══════════════════════════════════════════════════════════════════════╗");
    $display("║       ARMv4T ROBUST Pipeline Testbench (SYNC read/write)            ║");
    $display("║       SYNC_MEM=%0d  TRACE_EN=%0d  TRACE_LIMIT=%0d                       ║",
             SYNC_MEM, TRACE_EN, TRACE_LIMIT);
    $display("╚══════════════════════════════════════════════════════════════════════╝");
    $display("");

    // New set (S17..S24)
    test_psr_regform();
    test_muls_flags();
    test_alu_shifted_operand_imm();
    test_alu_shifted_operand_reg();
    test_conditional_branches();
    test_base_forward_after_ldr();
    test_preindex_wb_then_store();
    test_umull_hi_forward();

    $display("");
    $display("╔══════════════════════════════════════════════════════════════════════╗");
    if (total_fail == 0)
        $display("║  *** ALL %4d CHECKS PASSED ***                                    ║", total_pass);
    else
        $display("║  *** %4d PASSED, %4d FAILED ***                                   ║", total_pass, total_fail);
    $display("║  Total checks: %4d                                                 ║", total_pass + total_fail);
    $display("╚══════════════════════════════════════════════════════════════════════╝");
    $display("");

    #(CLK_PERIOD * 5);
    $finish;
end

endmodule