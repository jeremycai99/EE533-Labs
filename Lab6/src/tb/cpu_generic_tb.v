/*  armv4t_tb.v — Comprehensive ARMv4T pipeline testbench (FIXED)
 *
 *  CHANGES vs original:
 *    1. Sentinel detection uses saturating-counter on instruction-at-PC
 *       instead of exact-PC-match (works with pipelined branch delay).
 *    2. Trace uses ONLY external/port signals — no internal CPU paths
 *       that might not exist.  Set TRACE_EN=1 for per-cycle trace.
 *    3. Added dump_regs after every test and on timeout.
 *    4. Added combinational instr_at_pc wire for robust halt detection.
 *    5. Memory model selectable: SYNC_MEM=1 (registered) or 0 (comb).
 *       Try SYNC_MEM=0 if the CPU expects same-cycle memory response.
 */
`timescale 1ns / 1ps
`include "define.v"
`include "cpu.v"

module cpu_generic_tb;

// ═══════════════════════════════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════════════════════════════
parameter CLK_PERIOD   = 10;
parameter MEM_DEPTH    = 8192;         // words
parameter MAX_CYCLES   = 50_000;       // per-test timeout
parameter TRACE_EN     = 1;            // 0=quiet, 1=per-cycle trace
parameter TRACE_LIMIT  = 400;          // max cycles of trace per test
parameter SYNC_MEM     = 1;            // 1=synchronous(registered), 0=combinational

// Sentinel: a branch-to-self encodes as 0xEAFFFFFE
localparam [31:0] SENTINEL = 32'hEAFF_FFFE;

// Data region base (word-addressed) — tests can use 0x1000..0x1FFF
localparam DATA_BASE_BYTE = 32'h0000_1000;
localparam DATA_BASE_WORD = DATA_BASE_BYTE >> 2;

// Stack pointer init for tests that need it
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
//  Memory model — selectable sync vs combinational
// ═══════════════════════════════════════════════════════════════════
//  NOTE: If your CPU expects same-cycle memory reads (combinational),
//        set SYNC_MEM=0.  If it has an internal pipeline stage for
//        memory latency (common for FPGA), keep SYNC_MEM=1.

generate
if (SYNC_MEM == 1) begin : gen_sync_mem
    always @(posedge clk) begin
        i_mem_data  <= mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];
        d_mem_rdata <= mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)];
        if (d_mem_wen)
            mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)] <= d_mem_wdata;
    end
end else begin : gen_comb_mem
    // Combinational reads — data available same cycle as address
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
//  This bypasses the memory pipeline to see what's REALLY at the
//  current PC, regardless of memory model latency.
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
//  Per-cycle trace — uses ONLY external (port) signals
//  No internal CPU hierarchical references that might not compile.
// ═══════════════════════════════════════════════════════════════════
always @(posedge clk) begin
    if (TRACE_EN >= 1 && rst_n && cycle_cnt > 0 && cycle_cnt <= TRACE_LIMIT) begin
        $display("[C%05d] PC=0x%08H  @PC=0x%08H  pipe_in=0x%08H | D:addr=0x%08H wen=%b wdata=0x%08H rdata=0x%08H sz=%0d",
                 cycle_cnt,
                 i_mem_addr,       // address CPU is requesting
                 instr_at_pc,      // instruction AT that address (comb)
                 i_mem_data,       // instruction the CPU is receiving (may lag 1 cycle if SYNC_MEM)
                 d_mem_addr, d_mem_wen, d_mem_wdata, d_mem_rdata, d_mem_size);
    end
    // Always log data memory writes (useful for debugging stores)
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

// ── Write one instruction (byte address) ──────────────────────────
task mem_w;
    input [31:0] byte_addr;
    input [31:0] data;
begin
    mem_array[byte_addr >> 2] = data;
end
endtask

// ── Reset CPU and run until sentinel or timeout ───────────────────
//    FIX: Uses saturating-counter sentinel detection instead of
//    exact-PC-match, which fails on pipelined CPUs where PC
//    oscillates between X and X+4 around a B-to-self.
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

            // ── Sentinel detection (FIXED) ─────────────────────────
            // Look at the instruction sitting at the current PC in
            // memory (combinational, no pipeline delay).  If it's
            // the sentinel (B .), increment a saturating score.
            // With a 2-cycle branch penalty the sentinel is fetched
            // every ~2-3 cycles, so score grows ~+1/cycle net.
            if (cycle_cnt > 10) begin   // let pipeline fill first
                if (instr_at_pc === SENTINEL) begin
                    sentinel_score = sentinel_score + 3;
                end else begin
                    if (sentinel_score > 0)
                        sentinel_score = sentinel_score - 1;
                end
            end

            if (sentinel_score >= 24) begin
                $display("    [HALT] Sentinel (B .) detected at PC=0x%08H, cycle %0d",
                         i_mem_addr, cycle_cnt);
                // Let pipeline drain
                repeat (4) @(posedge clk);
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            // ── cpu_done signal ────────────────────────────────────
            if (cpu_done_w) begin
                $display("    [DONE] cpu_done asserted at cycle %0d", cycle_cnt);
                repeat(5) @(posedge clk);
                cycles_used = cycle_cnt;
                disable run_loop;
            end

            // ── Timeout ────────────────────────────────────────────
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

// ── Dump registers (always available) ─────────────────────────────
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


// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
//  T E S T   S E C T I O N S
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════


// ───────────────────────────────────────────────────────────────────
//  §1  DATA PROCESSING — immediate & register, all 16 opcodes
// ───────────────────────────────────────────────────────────────────
task test_data_processing;
    integer cyc;
begin
    section_start("S1: Data Processing (ALU, 16 opcodes)");
    mem_clear();

    // Setup: R0 = 0xFF, R1 = 0x0F, R2 = 0x80000000
    mem_w('h000, 32'hE3A000FF);   // MOV  R0, #0xFF
    mem_w('h004, 32'hE3A0100F);   // MOV  R1, #0x0F
    mem_w('h008, 32'hE3A02102);   // MOV  R2, #0x80000000  (0x02 ROR 2)

    // AND: R3 = R0 AND R1 = 0x0F
    mem_w('h00C, 32'hE0003001);   // AND  R3, R0, R1

    // EOR: R4 = R0 EOR R1 = 0xF0
    mem_w('h010, 32'hE0204001);   // EOR  R4, R0, R1

    // SUB: R5 = R0 - R1 = 0xF0
    mem_w('h014, 32'hE0405001);   // SUB  R5, R0, R1

    // RSB: R6 = R1 - R0 = 0xFFFFFF10 (-0xF0)
    mem_w('h018, 32'hE0606001);   // RSB  R6, R0, R1

    // ADD: R7 = R0 + R1 = 0x10E
    mem_w('h01C, 32'hE0807001);   // ADD  R7, R0, R1

    // ADC: first set carry with ADDS that overflows
    // ADDS R8, R2, R2 => 0x00000000 with C=1
    mem_w('h020, 32'hE0928002);   // ADDS R8, R2, R2
    // ADC R8, R0, #0 => R0 + 0 + C = 0x100
    mem_w('h024, 32'hE2A08000);   // ADC  R8, R0, #0

    // SBC: R9 = R0 - R1 - !C;  C was set above so !C=0 => 0xF0
    mem_w('h028, 32'hE0C09001);   // SBC  R9, R0, R1

    // RSC: R10 = R1 - R0 - !C = 0x0F - 0xFF - 0 = 0xFFFFFF10
    mem_w('h02C, 32'hE0E0A001);   // RSC  R10, R0, R1

    // TST: R0 AND #0x01 => Z should be clear (0xFF & 0x01 = 1)
    mem_w('h030, 32'hE3100001);   // TST  R0, #1

    // TEQ: R0 EOR R0 => Z should be set
    mem_w('h034, 32'hE1300000);   // TEQ  R0, R0

    // CMP: R1 - R0 => negative, C clear
    mem_w('h038, 32'hE1510000);   // CMP  R1, R0

    // CMN: R0 + R1 => positive, no special flags
    mem_w('h03C, 32'hE1700001);   // CMN  R0, R1

    // ORR: R3 = R0 ORR R1 = 0xFF
    mem_w('h040, 32'hE1803001);   // ORR  R3, R0, R1

    // MOV R4, #0xAB
    mem_w('h044, 32'hE3A040AB);   // MOV  R4, #0xAB

    // BIC: R5 = R0 BIC R1 = 0xFF & ~0x0F = 0xF0
    mem_w('h048, 32'hE1C05001);   // BIC  R5, R0, R1

    // MVN: R6 = ~R1 = 0xFFFFFFF0
    mem_w('h04C, 32'hE1E06001);   // MVN  R6, R1

    // ADD R7, R0, #0x3FC  (0xFF ROR 30 = 0xFF << 2 = 0x3FC)
    mem_w('h050, 32'hE2807FFF);   // ADD R7, R0, #0x3FC

    // Sentinel
    mem_w('h054, SENTINEL);

    run_test(cyc);

    check_reg(0, 32'h0000_00FF, "MOV R0,#0xFF");
    check_reg(1, 32'h0000_000F, "MOV R1,#0x0F");
    check_reg(2, 32'h8000_0000, "MOV R2,#0x80000000");
    check_reg(3, 32'h0000_00FF, "ORR R0,R1");
    check_reg(4, 32'h0000_00AB, "MOV R4,#0xAB");
    check_reg(5, 32'h0000_00F0, "BIC R0, R1");
    check_reg(6, 32'hFFFF_FFF0, "MVN R1");
    check_reg(7, 32'h0000_04FB, "ADD R0,#0x3FC");
    check_reg(8, 32'h0000_0100, "ADC R0,#0 (with C)");
    check_reg(9, 32'h0000_00F0, "SBC R0,R1 (C=1)");
    check_reg(10, 32'hFFFF_FF10, "RSC R0,R1");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §2  BARREL SHIFTER — LSL/LSR/ASR/ROR immediate & register
// ───────────────────────────────────────────────────────────────────
task test_shifter;
    integer cyc;
begin
    section_start("S2: Barrel Shifter (LSL/LSR/ASR/ROR)");
    mem_clear();

    mem_w('h000, 32'hE3A000FF);   // MOV R0, #0xFF
    mem_w('h004, 32'hE3A01004);   // MOV R1, #4
    mem_w('h008, 32'hE3A02102);   // MOV R2, #0x80000000
    mem_w('h00C, 32'hE282200F);   // ADD R2, R2, #0x0F  => R2 = 0x8000000F

    // LSL imm: R3 = R0 LSL #4 = 0xFF0
    mem_w('h010, 32'hE1A03200);   // MOV R3, R0, LSL #4
    // LSL reg: R4 = R0 LSL R1 = 0xFF0
    mem_w('h014, 32'hE1A04110);   // MOV R4, R0, LSL R1
    // LSR imm: R5 = R0 LSR #4 = 0x0F
    mem_w('h018, 32'hE1A05220);   // MOV R5, R0, LSR #4
    // LSR reg: R6 = R0 LSR R1 = 0x0F
    mem_w('h01C, 32'hE1A06130);   // MOV R6, R0, LSR R1
    // ASR imm: R7 = R2 ASR #4 = 0xF8000000
    mem_w('h020, 32'hE1A07242);   // MOV R7, R2, ASR #4
    // ASR reg: R8 = R2 ASR R1 = 0xF8000000
    mem_w('h024, 32'hE1A08152);   // MOV R8, R2, ASR R1
    // ROR imm: R9 = R0 ROR #4 = 0xF000000F
    mem_w('h028, 32'hE1A09260);   // MOV R9, R0, ROR #4
    // ROR reg: R10 = R0 ROR R1 = 0xF000000F
    mem_w('h02C, 32'hE1A0A170);   // MOV R10, R0, ROR R1

    // LSR #32 (encoded as LSR #0): R11 = R0 LSR #32 = 0
    mem_w('h030, 32'hE1A0B020);   // MOV R11, R0, LSR #32
    // ASR #32 (encoded as ASR #0): R12 = R2 ASR #32 = 0xFFFFFFFF
    mem_w('h034, 32'hE1A0C042);   // MOV R12, R2, ASR #32

    // RRX test: set carry first
    mem_w('h038, 32'hE3A03001);   // MOV R3, #1
    mem_w('h03C, 32'hE0934003);   // ADDS R4, R3, R3  ; R4=2, C=0
    mem_w('h040, 32'hE2534000);   // SUBS R4, R3, #0  ; R4=1, C=1 (no borrow)
    // RRX R5 = (C:R0) >> 1 = {1, 0xFF} >> 1 = 0x8000007F
    mem_w('h044, 32'hE1A05060);   // MOV R5, R0, RRX

    mem_w('h048, SENTINEL);

    run_test(cyc);

    check_reg(3, 32'h0000_0001, "setup R3=1");
    check_reg(0, 32'h0000_00FF, "R0 preserved");
    check_reg(5, 32'h8000_007F, "RRX R0 (C=1 in)");
    check_reg(6, 32'h0000_000F, "LSR R0 by R1");
    check_reg(7, 32'hF800_0000, "ASR R2 imm #4");
    check_reg(8, 32'hF800_0000, "ASR R2 by R1");
    check_reg(9, 32'hF000_000F, "ROR R0 imm #4");
    check_reg(10, 32'hF000_000F, "ROR R0 by R1");
    check_reg(11, 32'h0000_0000, "LSR #32");
    check_reg(12, 32'hFFFF_FFFF, "ASR #32 (neg)");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §3  MULTIPLY (MUL, MLA, UMULL, UMLAL, SMULL, SMLAL)
// ───────────────────────────────────────────────────────────────────
task test_multiply;
    integer cyc;
begin
    section_start("S3: Multiply / Multiply-long");
    mem_clear();

    mem_w('h000, 32'hE3A00007);   // MOV R0, #7
    mem_w('h004, 32'hE3A0100B);   // MOV R1, #11
    mem_w('h008, 32'hE3A02003);   // MOV R2, #3

    // MUL R3, R0, R1 => 77 = 0x4D
    mem_w('h00C, 32'hE0030190);   // MUL R3, R0, R1
    // MLA R4, R0, R1, R2 => 7*11 + 3 = 80 = 0x50
    mem_w('h010, 32'hE0242190);   // MLA R4, R0, R1, R2
    // UMULL R5,R6 = R0 * R1 (unsigned) => {0, 77}
    mem_w('h014, 32'hE0865090);   // UMULL R5, R6, R0, R1

    // Larger multiply: R7 = 0x10000, R8 = 0x20000
    mem_w('h018, 32'hE3A07801);   // MOV R7, #0x10000
    mem_w('h01C, 32'hE3A08802);   // MOV R8, #0x20000
    // UMULL R9,R10 = R7*R8 = 0x00000002_00000000
    mem_w('h020, 32'hE08A9897);   // UMULL R9, R10, R7, R8

    // SMULL with negative: R0 = -7 (0xFFFFFFF9)
    mem_w('h024, 32'hE3E00006);   // MVN R0, #6
    // SMULL R3,R4 = R0 * R1 = -7 * 11 = -77
    mem_w('h028, 32'hE0C43190);   // SMULL R3, R4, R0, R1

    // SMLAL with accumulate
    mem_w('h02C, 32'hE3A05001);   // MOV R5, #1
    mem_w('h030, 32'hE3A06000);   // MOV R6, #0
    mem_w('h034, 32'hE0E65190);   // SMLAL R5, R6, R0, R1

    // UMLAL
    mem_w('h038, 32'hE3A05010);   // MOV R5, #0x10
    mem_w('h03C, 32'hE3A06000);   // MOV R6, #0
    mem_w('h040, 32'hE0A65897);   // UMLAL R5, R6, R7, R8

    mem_w('h044, SENTINEL);

    run_test(cyc);

    check_reg(3,  32'hFFFF_FFB3, "SMULL lo: -7*11");
    check_reg(4,  32'hFFFF_FFFF, "SMULL hi: -7*11");
    check_reg(5,  32'h0000_0010, "UMLAL lo: 0x10+0");
    check_reg(6,  32'h0000_0002, "UMLAL hi: 0+2");
    check_reg(9,  32'h0000_0000, "UMULL lo: 0x10000*0x20000");
    check_reg(10, 32'h0000_0002, "UMULL hi: 0x10000*0x20000");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §4  SINGLE DATA TRANSFER — LDR/STR word/byte, pre/post, +/-
// ───────────────────────────────────────────────────────────────────
task test_load_store;
    integer cyc;
begin
    section_start("S4: Single Data Transfer (LDR/STR)");
    mem_clear();

    mem_array[DATA_BASE_WORD + 0] = 32'hDEAD_BEEF;
    mem_array[DATA_BASE_WORD + 1] = 32'h1234_5678;
    mem_array[DATA_BASE_WORD + 2] = 32'hCAFE_BABE;
    mem_array[DATA_BASE_WORD + 3] = 32'h0000_0000;
    mem_array[DATA_BASE_WORD + 4] = 32'h0000_0000;

    mem_w('h000, 32'hE3A00C10);   // MOV R0, #0x1000

    // LDR R1, [R0]
    mem_w('h004, 32'hE5901000);   // LDR R1, [R0, #0]
    // LDR R2, [R0, #4]
    mem_w('h008, 32'hE5902004);   // LDR R2, [R0, #4]
    // LDR R3, [R0, #8]!  (pre + writeback, R0 becomes 0x1008)
    mem_w('h00C, 32'hE5B03008);   // LDR R3, [R0, #8]!
    // STR R1, [R0, #4]   (store DEADBEEF at 0x100C)
    mem_w('h010, 32'hE5801004);   // STR R1, [R0, #4]

    // Restore R0
    mem_w('h014, 32'hE3A00C10);   // MOV R0, #0x1000
    // LDR R4, [R0], #4  (post-indexed)
    mem_w('h018, 32'hE4904004);   // LDR R4, [R0], #4
    // LDRB R5, [R0]
    mem_w('h01C, 32'hE5D05000);   // LDRB R5, [R0, #0]
    // STRB R5, [R0, #16]
    mem_w('h020, 32'hE5C05010);   // STRB R5, [R0, #16]
    // STR R2, [R0, #-4]
    mem_w('h024, 32'hE5002004);   // STR R2, [R0, #-4]

    // LDR with register offset
    mem_w('h028, 32'hE3A06004);   // MOV R6, #4
    mem_w('h02C, 32'hE7907006);   // LDR R7, [R0, R6]
    // LDR with shifted register
    mem_w('h030, 32'hE7908086);   // LDR R8, [R0, R6, LSL #1]
    // STR post-indexed
    mem_w('h034, 32'hE4003004);   // STR R3, [R0], #-4

    mem_w('h038, SENTINEL);

    run_test(cyc);

    check_reg(1, 32'hDEAD_BEEF, "LDR [base+0]");
    check_reg(2, 32'h1234_5678, "LDR [base+4]");
    check_reg(3, 32'hCAFE_BABE, "LDR [base+8]! (pre-wb)");
    check_reg(4, 32'hDEAD_BEEF, "LDR post-indexed");
    check_reg(7, 32'hCAFE_BABE, "LDR reg offset");
    check_mem(32'h100C, 32'hDEAD_BEEF, "STR to [0x100C]");
    check_mem(32'h1000, 32'h1234_5678, "STR neg offset [0x1000]");
    check_mem(32'h1004, 32'hCAFE_BABE, "STR post [0x1004]");
    check_reg(0, 32'h0000_1000, "R0 after post-dec");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §5  HALFWORD & SIGNED TRANSFER (LDRH/STRH/LDRSB/LDRSH)
// ───────────────────────────────────────────────────────────────────
task test_halfword_transfer;
    integer cyc;
begin
    section_start("S5: Halfword & Signed Transfer");
    mem_clear();

    mem_array[DATA_BASE_WORD + 0] = 32'hFF80_9ABC;
    mem_array[DATA_BASE_WORD + 1] = 32'h0000_007F;

    mem_w('h000, 32'hE3A00C10);   // MOV R0, #0x1000
    mem_w('h004, 32'hE1D010B0);   // LDRH R1, [R0, #0]
    mem_w('h008, 32'hE1D020F0);   // LDRSH R2, [R0, #0]
    mem_w('h00C, 32'hE1D030D0);   // LDRSB R3, [R0, #0]
    mem_w('h010, 32'hE1D040B4);   // LDRH R4, [R0, #4]
    mem_w('h014, 32'hE1D050D4);   // LDRSB R5, [R0, #4]
    mem_w('h018, 32'hE1C010B8);   // STRH R1, [R0, #8]

    mem_w('h01C, SENTINEL);

    run_test(cyc);

    check_reg(1, 32'h0000_9ABC, "LDRH unsigned");
    check_reg(2, 32'hFFFF_9ABC, "LDRSH signed neg");
    check_reg(3, 32'hFFFF_FFBC, "LDRSB signed neg byte");
    check_reg(4, 32'h0000_007F, "LDRH unsigned pos");
    check_reg(5, 32'h0000_007F, "LDRSB signed pos byte");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §6  BLOCK DATA TRANSFER — LDM/STM (IA/IB/DA/DB)
// ───────────────────────────────────────────────────────────────────
task test_block_transfer;
    integer cyc;
begin
    section_start("S6: Block Data Transfer (LDM/STM)");
    mem_clear();

    mem_w('h000, 32'hE3A00C10);   // MOV R0, #0x1000
    mem_w('h004, 32'hE3A01001);   // MOV R1, #1
    mem_w('h008, 32'hE3A02002);   // MOV R2, #2
    mem_w('h00C, 32'hE3A03003);   // MOV R3, #3
    mem_w('h010, 32'hE3A04004);   // MOV R4, #4

    // STMIA R0!, {R1-R4}
    mem_w('h014, 32'hE8A0001E);   // STMIA R0!, {R1-R4}
    // LDMDB R0!, {R5-R8}
    mem_w('h018, 32'hE91001E0);   // LDMDB R0!, {R5-R8}
    // STMIB R0, {R1-R4}
    mem_w('h01C, 32'hE980001E);   // STMIB R0, {R1-R4}

    // STMDA
    mem_w('h020, 32'hE3A09C10);   // MOV R9, #0x1000
    mem_w('h024, 32'hE2899020);   // ADD R9, R9, #0x20
    mem_w('h028, 32'hE809001E);   // STMDA R9, {R1-R4}

    // PUSH/POP
    mem_w('h02C, 32'hE3A0DC20);   // MOV SP, #0x2000
    mem_w('h030, 32'hE92D001E);   // PUSH {R1-R4}
    mem_w('h034, 32'hE3A01000);   // MOV R1, #0
    mem_w('h038, 32'hE3A02000);   // MOV R2, #0
    mem_w('h03C, 32'hE3A03000);   // MOV R3, #0
    mem_w('h040, 32'hE3A04000);   // MOV R4, #0
    mem_w('h044, 32'hE8BD001E);   // POP {R1-R4}

    mem_w('h048, SENTINEL);

    run_test(cyc);

    check_reg(1, 32'h0000_0001, "POP R1");
    check_reg(2, 32'h0000_0002, "POP R2");
    check_reg(3, 32'h0000_0003, "POP R3");
    check_reg(4, 32'h0000_0004, "POP R4");
    check_reg(5, 32'h0000_0001, "LDMDB R5");
    check_reg(6, 32'h0000_0002, "LDMDB R6");
    check_reg(7, 32'h0000_0003, "LDMDB R7");
    check_reg(8, 32'h0000_0004, "LDMDB R8");
    check_reg(13, 32'h0000_2000, "SP restored");
    check_mem(32'h1000, 32'h0000_0001, "STMIA [0x1000]=R1");
    check_mem(32'h1004, 32'h0000_0002, "STMIA [0x1004]=R2");
    check_mem(32'h1008, 32'h0000_0003, "STMIA [0x1008]=R3");
    check_mem(32'h100C, 32'h0000_0004, "STMIA [0x100C]=R4");
    check_mem(32'h1020, 32'h0000_0004, "STMIB [0x1010]=R4");
    check_mem(32'h1020, 32'h0000_0004, "STMDA [0x1020]=R4");
    check_mem(32'h101C, 32'h0000_0003, "STMDA [0x101C]=R3");
    check_mem(32'h1018, 32'h0000_0002, "STMDA [0x1018]=R2");
    check_mem(32'h1014, 32'h0000_0001, "STMDA [0x1014]=R1");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §7  BRANCH — B, BL, BX
// ───────────────────────────────────────────────────────────────────
task test_branch;
    integer cyc;
begin
    section_start("S7: Branch (B, BL, BX)");
    mem_clear();

    mem_w('h000, 32'hE3A00000);   // MOV R0, #0
    mem_w('h004, 32'hE1A00000);   // NOP

    // B forward to 0x020
    mem_w('h008, 32'hEA000004);   // B +4 => 0x020
    mem_w('h00C, 32'hE2800099);   // (SKIPPED)
    mem_w('h010, 32'hE2800099);   // (SKIPPED)
    mem_w('h014, 32'hE2800099);   // (SKIPPED)
    mem_w('h018, 32'hE2800099);   // (SKIPPED)
    mem_w('h01C, 32'hE2800099);   // (SKIPPED)

    // 0x020: first landing
    mem_w('h020, 32'hE2800001);   // ADD R0, R0, #1  => R0=1

    // BL to subroutine at 0x080
    mem_w('h024, 32'hEB000015);   // BL +21 => 0x080

    // 0x028: return from BL
    mem_w('h028, 32'hE2800001);   // ADD R0, R0, #1  => R0=3

    // BX register test
    mem_w('h02C, 32'hE3A02040);   // MOV R2, #0x40
    mem_w('h030, 32'hE12FFF12);   // BX  R2
    mem_w('h034, 32'hE2800099);   // (SKIPPED)
    mem_w('h038, 32'hE2800099);   // (SKIPPED)
    mem_w('h03C, 32'hE2800099);   // (SKIPPED)

    // 0x040: BX landing
    mem_w('h040, 32'hE2800001);   // ADD R0, R0, #1  => R0=4

    // B forward to 0x050, then backward to 0x048
    mem_w('h044, 32'hEA000001);   // B +1 => 0x050
    // 0x048: backward-branch target
    mem_w('h048, 32'hE2800001);   // ADD R0, R0, #1  => R0=6
    mem_w('h04C, 32'hEA000004);   // B +4 => 0x064

    // 0x050: forward target
    mem_w('h050, 32'hE2800001);   // ADD R0, R0, #1  => R0=5
    mem_w('h054, 32'hEAFFFFFB);   // B -5 => 0x048

    mem_w('h058, 32'hE2800099);   // (SKIPPED)
    mem_w('h05C, 32'hE2800099);   // (SKIPPED)
    mem_w('h060, 32'hE2800099);   // (SKIPPED)

    // 0x064: final landing
    mem_w('h064, 32'hE1A00000);   // NOP
    mem_w('h068, SENTINEL);

    // Subroutine at 0x080:
    mem_w('h080, 32'hE2800001);   // ADD R0, R0, #1  => R0=2
    mem_w('h084, 32'hE12FFF1E);   // BX  LR

    run_test(cyc);

    check_reg(0, 32'h0000_0006, "R0=6 after all branches");
    check_reg(14, 32'h0000_0028, "LR from BL");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §8  CONDITIONAL EXECUTION — all 15 conditions
// ───────────────────────────────────────────────────────────────────
task test_conditionals;
    integer cyc;
begin
    section_start("S8: Conditional Execution");
    mem_clear();

    mem_w('h000, 32'hE3A00000);   // MOV R0, #0
    mem_w('h004, 32'hE3A01005);   // MOV R1, #5
    mem_w('h008, 32'hE1510001);   // CMP R1, R1   ; Z=1 C=1

    mem_w('h00C, 32'h02800001);   // ADDEQ R0, R0, #1
    mem_w('h010, 32'h12800064);   // ADDNE R0, R0, #100
    mem_w('h014, 32'h22800001);   // ADDCS R0, R0, #1
    mem_w('h018, 32'h32800064);   // ADDCC R0, R0, #100

    mem_w('h01C, 32'hE3A02003);   // MOV R2, #3
    mem_w('h020, 32'hE3A03005);   // MOV R3, #5
    mem_w('h024, 32'hE1520003);   // CMP R2, R3   ; N=1 C=0 Z=0 V=0

    mem_w('h028, 32'h42800001);   // ADDMI R0, R0, #1
    mem_w('h02C, 32'h52800064);   // ADDPL R0, R0, #100
    mem_w('h030, 32'hB2800001);   // ADDLT R0, R0, #1
    mem_w('h034, 32'hA2800064);   // ADDGE R0, R0, #100
    mem_w('h038, 32'hC2800064);   // ADDGT R0, R0, #100
    mem_w('h03C, 32'hD2800001);   // ADDLE R0, R0, #1

    mem_w('h040, 32'hE1510002);   // CMP R1, R2   ; 5-3 => C=1 Z=0

    mem_w('h044, 32'h82800001);   // ADDHI R0, R0, #1
    mem_w('h048, 32'h92800064);   // ADDLS R0, R0, #100

    // Set overflow
    mem_w('h04C, 32'hE3A04102);   // MOV R4, #0x80000000
    mem_w('h050, 32'hE2444001);   // SUB R4, R4, #1     ; R4 = 0x7FFFFFFF
    mem_w('h054, 32'hE2944001);   // ADDS R4, R4, #1    ; V=1, N=1

    mem_w('h058, 32'h62800001);   // ADDVS R0, R0, #1
    mem_w('h05C, 32'h72800064);   // ADDVC R0, R0, #100
    mem_w('h060, 32'hE2800001);   // ADD R0, R0, #1     ; AL

    mem_w('h064, SENTINEL);

    run_test(cyc);

    check_reg(0, 32'h0000_0008, "R0=8 (8 conds executed)");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §9  PSR TRANSFER — MRS, MSR
// ───────────────────────────────────────────────────────────────────
task test_psr;
    integer cyc;
begin
    section_start("S9: PSR Transfer (MRS, MSR)");
    mem_clear();

    mem_w('h000, 32'hE3A00000);   // MOV R0, #0
    mem_w('h004, 32'hE1500000);   // CMP R0, R0    ; Z=1, C=1
    mem_w('h008, 32'hE10F1000);   // MRS R1, CPSR
    mem_w('h00C, 32'hE328F102);   // MSR CPSR_f, #0x80000000 (N=1)
    mem_w('h010, 32'hE10F2000);   // MRS R2, CPSR
    mem_w('h014, 32'hE328F4F0);   // MSR CPSR_f, #0xF0000000 (NZCV)
    mem_w('h018, 32'hE10F3000);   // MRS R3, CPSR

    mem_w('h01C, SENTINEL);

    run_test(cyc);

    $display("    R1 (CPSR after Z=1,C=1) = 0x%08H", u_cpu.u_regfile.regs[1]);
    $display("    R2 (CPSR after N set)   = 0x%08H", u_cpu.u_regfile.regs[2]);
    $display("    R3 (CPSR after NZCV)    = 0x%08H", u_cpu.u_regfile.regs[3]);

    check_flags(4'b1111, "NZCV all set via MSR");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §10 SWAP — SWP, SWPB
// ───────────────────────────────────────────────────────────────────
task test_swap;
    integer cyc;
begin
    section_start("S10: Swap (SWP, SWPB)");
    mem_clear();

    mem_array[DATA_BASE_WORD] = 32'hAAAA_BBBB;

    mem_w('h000, 32'hE3A00C10);   // MOV R0, #0x1000
    mem_w('h004, 32'hE3A01C22);   // MOV R1, #0x2200
    mem_w('h008, 32'hE1002091);   // SWP R2, R1, [R0]

    mem_array[DATA_BASE_WORD+1] = 32'h0000_00FF;
    mem_w('h00C, 32'hE3A03042);   // MOV R3, #0x42
    mem_w('h010, 32'hE2804004);   // ADD R4, R0, #4
    mem_w('h014, 32'hE1445093);   // SWPB R5, R3, [R4]

    mem_w('h018, SENTINEL);

    run_test(cyc);

    check_reg(2, 32'hAAAA_BBBB, "SWP read old value");
    check_mem(32'h1000, 32'h0000_2200, "SWP wrote new value");
    check_reg(5, 32'h0000_00FF, "SWPB read old byte");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §11 PIPELINE HAZARDS — forwarding, load-use, back-to-back
// ───────────────────────────────────────────────────────────────────
task test_hazards;
    integer cyc;
begin
    section_start("S11: Pipeline Hazards");
    mem_clear();

    // RAW forwarding: EX→EX
    mem_w('h000, 32'hE3A00005);   // MOV R0, #5
    mem_w('h004, 32'hE2801003);   // ADD R1, R0, #3
    mem_w('h008, 32'hE0812000);   // ADD R2, R1, R0

    // Triple-dependency chain
    mem_w('h00C, 32'hE0823001);   // ADD R3, R2, R1
    mem_w('h010, 32'hE0434002);   // SUB R4, R3, R2

    // Load-use hazard
    mem_array[DATA_BASE_WORD] = 32'h0000_000A;
    mem_w('h014, 32'hE3A05C10);   // MOV R5, #0x1000
    mem_w('h018, 32'hE5956000);   // LDR R6, [R5]
    mem_w('h01C, 32'hE2867002);   // ADD R7, R6, #2

    // Load followed by store
    mem_array[DATA_BASE_WORD+1] = 32'h0000_0014;
    mem_w('h020, 32'hE5958004);   // LDR R8, [R5, #4]
    mem_w('h024, 32'hE5858008);   // STR R8, [R5, #8]

    // Back-to-back loads
    mem_array[DATA_BASE_WORD+3] = 32'h0000_001E;
    mem_array[DATA_BASE_WORD+4] = 32'h0000_0028;
    mem_w('h028, 32'hE595900C);   // LDR R9, [R5, #12]
    mem_w('h02C, 32'hE595A010);   // LDR R10, [R5, #16]
    mem_w('h030, 32'hE089B00A);   // ADD R11, R9, R10

    // Store after ALU
    mem_w('h034, 32'hE289C001);   // ADD R12, R9, #1
    mem_w('h038, 32'hE585C014);   // STR R12, [R5, #20]

    mem_w('h03C, SENTINEL);

    run_test(cyc);

    check_reg(0, 32'h0000_0005, "MOV #5");
    check_reg(1, 32'h0000_0008, "RAW fwd: 5+3=8");
    check_reg(2, 32'h0000_000D, "RAW fwd: 8+5=13");
    check_reg(3, 32'h0000_0015, "RAW chain: 13+8=21");
    check_reg(4, 32'h0000_0008, "RAW chain: 21-13=8");
    check_reg(6, 32'h0000_000A, "LDR value=10");
    check_reg(7, 32'h0000_000C, "Load-use: 10+2=12");
    check_reg(8, 32'h0000_0014, "LDR value=20");
    check_mem(32'h1008, 32'h0000_0014, "STR after LDR");
    check_reg(9,  32'h0000_001E, "LDR value=30");
    check_reg(10, 32'h0000_0028, "LDR value=40");
    check_reg(11, 32'h0000_0046, "Two load-use: 30+40=70");
    check_reg(12, 32'h0000_001F, "ALU: 30+1=31");
    check_mem(32'h1014, 32'h0000_001F, "STR after ALU fwd");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §12 FLAG-SETTING & SHIFTED OPERAND ALU
// ───────────────────────────────────────────────────────────────────
task test_flag_setting;
    integer cyc;
begin
    section_start("S12: Flag-setting ALU (S-bit, shifted ops)");
    mem_clear();

    mem_w('h000, 32'hE3A00005);   // MOV R0, #5
    mem_w('h004, 32'hE0501000);   // SUBS R1, R0, R0
    mem_w('h008, 32'hE1A00000);   // NOP

    mem_w('h00C, 32'hE3A0200A);   // MOV R2, #10
    mem_w('h010, 32'hE2523014);   // SUBS R3, R2, #20

    mem_w('h014, 32'hE3A04102);   // MOV R4, #0x80000000
    mem_w('h018, 32'hE0945004);   // ADDS R5, R4, R4

    mem_w('h01C, 32'hE3A060FF);   // MOV R6, #0xFF
    mem_w('h020, 32'hE1B07F86);   // MOVS R7, R6, LSL #31

    mem_w('h024, 32'hE3A08000);   // MOV R8, #0
    mem_w('h028, 32'hE21890FF);   // ANDS R9, R8, #0xFF

    mem_w('h02C, 32'hE3A04102);   // MOV R4, #0x80000000
    mem_w('h030, 32'hE2444001);   // SUB R4, R4, #1
    mem_w('h034, 32'hE2945001);   // ADDS R5, R4, #1

    mem_w('h038, 32'hE10FA000);   // MRS R10, CPSR

    mem_w('h03C, SENTINEL);

    run_test(cyc);

    check_reg(1, 32'h0000_0000, "SUBS 5-5=0");
    check_reg(3, 32'hFFFF_FFF6, "SUBS 10-20=-10");
    check_reg(5, 32'h8000_0000, "ADDS overflow");
    check_reg(7, 32'h8000_0000, "MOVS LSL #31");
    check_reg(9, 32'h0000_0000, "ANDS 0 & 0xFF = 0");
    check_flags(4'b1001, "ADDS overflow: N=1 V=1");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §13 LOOP CONSTRUCTS — counting loop, do-while
// ───────────────────────────────────────────────────────────────────
task test_loops;
    integer cyc;
begin
    section_start("S13: Loop Constructs");
    mem_clear();

    mem_w('h000, 32'hE3A00001);   // MOV R0, #1
    mem_w('h004, 32'hE3A01000);   // MOV R1, #0
    mem_w('h008, 32'hE3A0200A);   // MOV R2, #10

    // loop:
    mem_w('h00C, 32'hE0811000);   // ADD R1, R1, R0
    mem_w('h010, 32'hE2800001);   // ADD R0, R0, #1
    mem_w('h014, 32'hE1500002);   // CMP R0, R2
    mem_w('h018, 32'hDAFFFFFB);   // BLE -5 => 0x00C

    // Nested loop
    mem_w('h01C, 32'hE3A03000);   // MOV R3, #0
    mem_w('h020, 32'hE3A05000);   // MOV R5, #0

    // outer_loop:
    mem_w('h024, 32'hE3A04000);   // MOV R4, #0
    // inner_loop:
    mem_w('h028, 32'hE2855001);   // ADD R5, R5, #1
    mem_w('h02C, 32'hE2844001);   // ADD R4, R4, #1
    mem_w('h030, 32'hE3540004);   // CMP R4, #4
    mem_w('h034, 32'hBAFFFFFB);   // BLT -5 => 0x028

    mem_w('h038, 32'hE2833001);   // ADD R3, R3, #1
    mem_w('h03C, 32'hE3530003);   // CMP R3, #3
    mem_w('h040, 32'hBAFFFFF7);   // BLT -9 => 0x024

    mem_w('h044, SENTINEL);

    run_test(cyc);

    check_reg(1, 32'h0000_0037, "Sum 1..10 = 55");
    check_reg(5, 32'h0000_000C, "Nested loop: 3*4=12");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §14 SUBROUTINE CALLING CONVENTION (BL, stack frame)
// ───────────────────────────────────────────────────────────────────
task test_subroutine;
    integer cyc;
begin
    section_start("S14: Subroutine Call/Return with Stack");
    mem_clear();

    mem_w('h000, 32'hE3A0DC20);   // MOV SP, #0x2000
    mem_w('h004, 32'hE3A00005);   // MOV R0, #5
    mem_w('h008, 32'hE3A01007);   // MOV R1, #7
    mem_w('h00C, 32'hEB00000B);   // BL +11 => 0x040

    mem_w('h010, 32'hE1A02000);   // MOV R2, R0

    mem_w('h014, 32'hE3A00064);   // MOV R0, #100
    mem_w('h018, 32'hE3A010C8);   // MOV R1, #200
    mem_w('h01C, 32'hEB000007);   // BL +7 => 0x040

    mem_w('h020, 32'hE1A03000);   // MOV R3, R0
    mem_w('h024, SENTINEL);

    // add_func at 0x040:
    mem_w('h040, 32'hE92D4000);   // PUSH {LR}
    mem_w('h044, 32'hE0800001);   // ADD R0, R0, R1
    mem_w('h048, 32'hE8BD4000);   // POP {LR}
    mem_w('h04C, 32'hE12FFF1E);   // BX LR

    run_test(cyc);

    check_reg(2, 32'h0000_000C, "First call: 5+7=12");
    check_reg(3, 32'h0000_012C, "Second call: 100+200=300");
    check_reg(13, 32'h0000_2000, "SP restored");

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §15 INTEGRATION: Bubble Sort
// ───────────────────────────────────────────────────────────────────
task test_sort_integration;
    integer cyc, idx;
    reg signed [31:0] expected_sort [0:9];

    localparam [31:0] SORT_SP   = 32'h0000_0400;
    localparam [31:0] SORT_FP   = SORT_SP - 32'd4;
    localparam [31:0] SORT_ARR  = SORT_FP - 32'd56;
    localparam [31:0] SORT_ARRW = SORT_ARR >> 2;
begin
    section_start("S15: Integration — Bubble Sort");
    mem_clear();

    mem_w('h000, 32'hE3A0B000);
    mem_w('h004, 32'hE3A0EC02);   // MOV LR, #0x200 (return to sentinel)
    mem_w('h008, 32'hE1A00000);
    mem_w('h00C, 32'hE3A0DB01);
    mem_w('h010, 32'hE92D4800);
    mem_w('h014, 32'hE28DB004);
    mem_w('h018, 32'hE24DD038);
    mem_w('h01C, 32'hE59F3104);
    mem_w('h020, 32'hE24BC038);
    mem_w('h024, 32'hE1A0E003);
    mem_w('h028, 32'hE8BE000F);
    mem_w('h02C, 32'hE8AC000F);
    mem_w('h030, 32'hE8BE000F);
    mem_w('h034, 32'hE8AC000F);
    mem_w('h038, 32'hE89E0003);
    mem_w('h03C, 32'hE88C0003);
    mem_w('h040, 32'hE3A03000);
    mem_w('h044, 32'hE50B3008);
    mem_w('h048, 32'hEA00002E);
    mem_w('h04C, 32'hE51B3008);
    mem_w('h050, 32'hE2833001);
    mem_w('h054, 32'hE50B300C);
    mem_w('h058, 32'hEA000024);
    mem_w('h05C, 32'hE51B300C);
    mem_w('h060, 32'hE1A03103);
    mem_w('h064, 32'hE2433004);
    mem_w('h068, 32'hE083300B);
    mem_w('h06C, 32'hE5132034);
    mem_w('h070, 32'hE51B3008);
    mem_w('h074, 32'hE1A03103);
    mem_w('h078, 32'hE2433004);
    mem_w('h07C, 32'hE083300B);
    mem_w('h080, 32'hE5133034);
    mem_w('h084, 32'hE1520003);
    mem_w('h088, 32'hAA000015);
    mem_w('h08C, 32'hE51B300C);
    mem_w('h090, 32'hE1A03103);
    mem_w('h094, 32'hE2433004);
    mem_w('h098, 32'hE083300B);
    mem_w('h09C, 32'hE5133034);
    mem_w('h0A0, 32'hE50B3010);
    mem_w('h0A4, 32'hE51B3008);
    mem_w('h0A8, 32'hE1A03103);
    mem_w('h0AC, 32'hE2433004);
    mem_w('h0B0, 32'hE083300B);
    mem_w('h0B4, 32'hE5132034);
    mem_w('h0B8, 32'hE51B300C);
    mem_w('h0BC, 32'hE1A03103);
    mem_w('h0C0, 32'hE2433004);
    mem_w('h0C4, 32'hE083300B);
    mem_w('h0C8, 32'hE5032034);
    mem_w('h0CC, 32'hE51B3008);
    mem_w('h0D0, 32'hE1A03103);
    mem_w('h0D4, 32'hE2433004);
    mem_w('h0D8, 32'hE083300B);
    mem_w('h0DC, 32'hE51B2010);
    mem_w('h0E0, 32'hE5032034);
    mem_w('h0E4, 32'hE51B300C);
    mem_w('h0E8, 32'hE2833001);
    mem_w('h0EC, 32'hE50B300C);
    mem_w('h0F0, 32'hE51B300C);
    mem_w('h0F4, 32'hE3530009);
    mem_w('h0F8, 32'hDAFFFFD7);
    mem_w('h0FC, 32'hE51B3008);
    mem_w('h100, 32'hE2833001);
    mem_w('h104, 32'hE50B3008);
    mem_w('h108, 32'hE51B3008);
    mem_w('h10C, 32'hE3530009);
    mem_w('h110, 32'hDAFFFFCD);
    mem_w('h114, 32'hE3A03000);
    mem_w('h118, 32'hE1A00003);
    mem_w('h11C, 32'hE24BD004);
    mem_w('h120, 32'hE8BD4800);
    mem_w('h124, 32'hE12FFF1E);

    mem_w('h128, 32'h0000_012C);

    mem_w('h12C, 32'h0000_0143);
    mem_w('h130, 32'h0000_007B);
    mem_w('h134, 32'hFFFF_FE39);
    mem_w('h138, 32'h0000_0002);
    mem_w('h13C, 32'h0000_0062);
    mem_w('h140, 32'h0000_007D);
    mem_w('h144, 32'h0000_000A);
    mem_w('h148, 32'h0000_0041);
    mem_w('h14C, 32'hFFFF_FFC8);
    mem_w('h150, 32'h0000_0000);

    // Sentinel at return address
    mem_w('h200, SENTINEL);

    expected_sort[0] = -32'sd455;
    expected_sort[1] = -32'sd56;
    expected_sort[2] =  32'sd0;
    expected_sort[3] =  32'sd2;
    expected_sort[4] =  32'sd10;
    expected_sort[5] =  32'sd65;
    expected_sort[6] =  32'sd98;
    expected_sort[7] =  32'sd123;
    expected_sort[8] =  32'sd125;
    expected_sort[9] =  32'sd323;

    run_test(cyc);

    check_reg(0, 32'd0, "R0=0 (return value)");
    check_reg(13, SORT_SP, "SP restored");
    check_reg(11, 32'd0, "FP restored");

    // Dump the array memory region for visibility
    dump_mem(SORT_ARR, 10);

    for (idx = 0; idx < 10; idx = idx + 1) begin
        if ($signed(mem_array[SORT_ARRW + idx]) === expected_sort[idx]) begin
            $display("    [PASS] arr[%0d] = %0d", idx, expected_sort[idx]);
            section_pass = section_pass + 1;
            total_pass   = total_pass   + 1;
        end else begin
            $display("    [FAIL] arr[%0d] = %0d, expected %0d",
                     idx, $signed(mem_array[SORT_ARRW + idx]), expected_sort[idx]);
            section_fail = section_fail + 1;
            total_fail   = total_fail   + 1;
        end
    end

    section_end();
end
endtask


// ───────────────────────────────────────────────────────────────────
//  §16 EDGE CASES
// ───────────────────────────────────────────────────────────────────
task test_edge_cases;
    integer cyc;
begin
    section_start("S16: Edge Cases");
    mem_clear();

    // ADD with PC
    mem_w('h000, 32'hE1A0000F);   // MOV R0, PC
    mem_w('h004, 32'hE3A01000);   // MOV R1, #0
    mem_w('h008, 32'hE3A01000);   // MOV R1, #0

    // Shift by 0
    mem_w('h00C, 32'hE3A020FF);   // MOV R2, #0xFF
    mem_w('h010, 32'hE1A03002);   // MOV R3, R2, LSL #0

    // AND with partial mask
    mem_w('h014, 32'hE20040FF);   // AND R4, R0, #0xFF

    // SUB Rd, Rn, Rn
    mem_w('h018, 32'hE0405000);   // SUB R5, R0, R0

    // MOV Rn, Rn (identity)
    mem_w('h01C, 32'hE3A06042);   // MOV R6, #0x42
    mem_w('h020, 32'hE1A06006);   // MOV R6, R6

    // Large immediate
    mem_w('h024, 32'hE3A074FF);   // MOV R7, #0xFF000000
    mem_w('h028, 32'hE2877801);   // ADD R7, R7, #0x10000

    mem_w('h02C, SENTINEL);

    run_test(cyc);

    check_reg(0, 32'h0000_0008, "MOV R0,PC (=PC+8)");
    check_reg(3, 32'h0000_00FF, "LSL #0 = identity");
    check_reg(4, 32'h0000_0008, "AND R0,#0xFF");
    check_reg(5, 32'h0000_0000, "SUB Rn,Rn = 0");
    check_reg(6, 32'h0000_0042, "MOV Rn,Rn identity");
    check_reg(7, 32'hFF01_0000, "Large imm rotation");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
//  M A I N   S T I M U L U S
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
initial begin
    $dumpfile("cpu_generic_tb.vcd");
    $dumpvars(0, cpu_generic_tb);

    total_pass = 0;
    total_fail = 0;
    ila_debug_sel = 5'd0;
    rst_n = 0;

    $display("");
    $display("╔══════════════════════════════════════════════════════════════════════╗");
    $display("║       ARMv4T Comprehensive Pipeline Testbench  (FIXED)             ║");
    $display("║       SYNC_MEM=%0d  TRACE_EN=%0d  TRACE_LIMIT=%0d                       ║",
             SYNC_MEM, TRACE_EN, TRACE_LIMIT);
    $display("╚══════════════════════════════════════════════════════════════════════╝");
    $display("");

    test_data_processing();
    test_shifter();
    test_multiply();
    test_load_store();
    test_halfword_transfer();
    test_block_transfer();
    test_branch();
    test_conditionals();
    test_psr();
    test_swap();
    test_hazards();
    test_flag_setting();
    test_loops();
    test_subroutine();
    test_edge_cases();
    test_sort_integration();

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