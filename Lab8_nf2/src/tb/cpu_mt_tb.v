/* cpu_mt_tb.v — Quad-thread ARMv4T multithreaded pipeline testbench
 * v2: Added CP10 + DMA test environment, MCR/MRC tests, MUL/MLA tests
 *
 *   Scenarios A–D: Original network-processing programs (4 threads)
 *   Scenario E: CP10 coprocessor MCR/MRC write/readback
 *   Scenario F: MUL/MLA/UMULL multiply instructions
 *
 * Author: Jeremy Cai
 * Date:   Mar. 5, 2026
 */
`timescale 1ns / 1ps
`include "define.v"
`include "cpu_mt.v"
`include "cp10_regfile.v"
`include "dma_engine.v"

module cpu_mt_tb;

// ═══════════════════════════════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════════════════════════════
parameter CLK_PERIOD = 10;
parameter MEM_DEPTH = 16384;
parameter MAX_CYCLES = 10_000;
parameter TRACE_EN = 1;
parameter TRACE_LIMIT = 600;
parameter SYNC_MEM = 1;

localparam [31:0] SENTINEL = 32'hEAFF_FFFE; // B . (branch to self)

localparam [31:0] T0_CODE = 32'h0000_0000;
localparam [31:0] T1_CODE = 32'h0000_1000;
localparam [31:0] T2_CODE = 32'h0000_2000;
localparam [31:0] T3_CODE = 32'h0000_3000;

localparam [31:0] T0_DATA = 32'h0000_0100;
localparam [31:0] T1_DATA = 32'h0000_0200;
localparam [31:0] T2_DATA = 32'h0000_0300;
localparam [31:0] T3_DATA = 32'h0000_0400;

localparam [31:0] T0_SP = 32'h0000_0800;
localparam [31:0] T1_SP = 32'h0000_0A00;
localparam [31:0] T2_SP = 32'h0000_0C00;
localparam [31:0] T3_SP = 32'h0000_0E00;

localparam [31:0] T0_RET = 32'h0000_0FFC;
localparam [31:0] T1_RET = 32'h0000_1FFC;
localparam [31:0] T2_RET = 32'h0000_2FFC;
localparam [31:0] T3_RET = 32'h0000_3FFC;

// ═══════════════════════════════════════════════════════════════════
//  DUT + Peripheral Signals
// ═══════════════════════════════════════════════════════════════════
reg clk, rst_n;
reg cpu_start_tb;
reg [`PC_WIDTH-1:0] entry_pc_tb;
wire [`PC_WIDTH-1:0] i_mem_addr;
reg [`INSTR_WIDTH-1:0] i_mem_data;
wire [`CPU_DMEM_ADDR_WIDTH-1:0] d_mem_addr;
wire [`DATA_WIDTH-1:0] d_mem_wdata;
reg [`DATA_WIDTH-1:0] d_mem_rdata;
wire d_mem_wen;
wire [1:0] d_mem_size;
wire cpu_done_w;

// CPU ↔ CP10 interface
wire cp_wen, cp_ren;
wire [3:0] cp_reg;
wire [31:0] cp_wr_data;
wire [31:0] cp_rd_data;

// CP10 ↔ DMA interface
wire [31:0] dma_src_addr, dma_dst_addr;
wire [15:0] dma_xfer_len;
wire dma_start, dma_dir, dma_tgt, dma_auto_inc, dma_burst_all;
wire [1:0] dma_bank;
wire dma_busy, dma_error;
wire [31:0] dma_cur_addr;

// CP10 ↔ GPU interface (stubbed)
wire gpu_kernel_start, gpu_reset_n;
wire [31:0] gpu_entry_pc, gpu_scratch;
wire [3:0] gpu_thread_mask;

// DMA ↔ CPU DMEM Port B
wire [11:0] dma_cpu_addr;
wire [31:0] dma_cpu_din;
wire dma_cpu_we;
reg [31:0] dma_cpu_dout;

// DMA ↔ GPU IMEM (stubbed)
wire [7:0] dma_gpu_imem_addr;
wire [31:0] dma_gpu_imem_din;
wire dma_gpu_imem_we;

// DMA ↔ GPU DMEM (stubbed)
wire [1:0] dma_gpu_dmem_sel;
wire [9:0] dma_gpu_dmem_addr;
wire [15:0] dma_gpu_dmem_din;
wire dma_gpu_dmem_we;

// ═══════════════════════════════════════════════════════════════════
//  Unified Memory
// ═══════════════════════════════════════════════════════════════════
reg [31:0] mem_array [0:MEM_DEPTH-1];

// ═══════════════════════════════════════════════════════════════════
//  DUT: cpu_mt v2.8
// ═══════════════════════════════════════════════════════════════════
cpu_mt u_cpu_mt (
    .clk(clk), .rst_n(rst_n),
    .cpu_start_i(cpu_start_tb), .entry_pc_i(entry_pc_tb),
    .i_mem_data_i(i_mem_data), .i_mem_addr_o(i_mem_addr),
    .d_mem_data_i(d_mem_rdata), .d_mem_addr_o(d_mem_addr),
    .d_mem_data_o(d_mem_wdata), .d_mem_wen_o(d_mem_wen), .d_mem_size_o(d_mem_size),
    .cp_wen_o(cp_wen), .cp_ren_o(cp_ren), .cp_reg_o(cp_reg),
    .cp_wr_data_o(cp_wr_data), .cp_rd_data_i(cp_rd_data),
    .cpu_done(cpu_done_w)
);

// ═══════════════════════════════════════════════════════════════════
//  CP10 Register File
// ═══════════════════════════════════════════════════════════════════
cp10_regfile u_cp10 (
    .clk(clk), .rst_n(rst_n),
    .cp_wen(cp_wen), .cp_ren(cp_ren), .cp_reg(cp_reg),
    .cp_wdata(cp_wr_data), .cp_rdata(cp_rd_data),
    .dma_src_addr(dma_src_addr), .dma_dst_addr(dma_dst_addr),
    .dma_xfer_len(dma_xfer_len), .dma_start(dma_start),
    .dma_dir(dma_dir), .dma_tgt(dma_tgt), .dma_bank(dma_bank),
    .dma_auto_inc(dma_auto_inc), .dma_burst_all(dma_burst_all),
    .dma_busy(dma_busy), .dma_error(dma_error), .dma_cur_addr(dma_cur_addr),
    .gpu_kernel_start(gpu_kernel_start), .gpu_reset_n(gpu_reset_n),
    .gpu_entry_pc(gpu_entry_pc), .gpu_thread_mask(gpu_thread_mask),
    .gpu_scratch(gpu_scratch),
    .gpu_kernel_done(1'b0), .gpu_active(1'b0)
);

// ═══════════════════════════════════════════════════════════════════
//  DMA Engine
// ═══════════════════════════════════════════════════════════════════
dma_engine u_dma (
    .clk(clk), .rst_n(rst_n),
    .dma_src_addr(dma_src_addr), .dma_dst_addr(dma_dst_addr),
    .dma_xfer_len(dma_xfer_len), .dma_start(dma_start),
    .dma_dir(dma_dir), .dma_tgt(dma_tgt), .dma_bank(dma_bank),
    .dma_auto_inc(dma_auto_inc), .dma_burst_all(dma_burst_all),
    .dma_busy(dma_busy), .dma_error(dma_error), .dma_cur_addr(dma_cur_addr),
    .cpu_dmem_addr(dma_cpu_addr), .cpu_dmem_din(dma_cpu_din),
    .cpu_dmem_we(dma_cpu_we), .cpu_dmem_dout(dma_cpu_dout),
    .gpu_imem_addr(dma_gpu_imem_addr), .gpu_imem_din(dma_gpu_imem_din),
    .gpu_imem_we(dma_gpu_imem_we),
    .gpu_dmem_sel(dma_gpu_dmem_sel), .gpu_dmem_addr(dma_gpu_dmem_addr),
    .gpu_dmem_din(dma_gpu_dmem_din), .gpu_dmem_we(dma_gpu_dmem_we),
    .gpu_dmem_dout(16'd0)
);

// ═══════════════════════════════════════════════════════════════════
//  Clock
// ═══════════════════════════════════════════════════════════════════
initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

// ═══════════════════════════════════════════════════════════════════
//  Memory Model (CPU Port A + DMA Port B)
// ═══════════════════════════════════════════════════════════════════
generate
if (SYNC_MEM == 1) begin : gen_sync_mem
    always @(posedge clk) begin
        // Port A: CPU fetch + data R/W
        i_mem_data <= mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];
        d_mem_rdata <= mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)];
        if (d_mem_wen)
            mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)] <= d_mem_wdata;
        // Port B: DMA R/W
        dma_cpu_dout <= mem_array[dma_cpu_addr & (MEM_DEPTH-1)];
        if (dma_cpu_we)
            mem_array[dma_cpu_addr & (MEM_DEPTH-1)] <= dma_cpu_din;
    end
end else begin : gen_comb_mem
    always @(*) begin
        i_mem_data = mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];
        d_mem_rdata = mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)];
        dma_cpu_dout = mem_array[dma_cpu_addr & (MEM_DEPTH-1)];
    end
    always @(posedge clk) begin
        if (d_mem_wen)
            mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)] <= d_mem_wdata;
        if (dma_cpu_we)
            mem_array[dma_cpu_addr & (MEM_DEPTH-1)] <= dma_cpu_din;
    end
end
endgenerate

wire [31:0] instr_at_pc = mem_array[(i_mem_addr >> 2) & (MEM_DEPTH-1)];

// ═══════════════════════════════════════════════════════════════════
//  Per-Thread Sentinel Detection
// ═══════════════════════════════════════════════════════════════════
reg [7:0] sentinel_cnt [0:3];
wire [3:0] thread_at_sentinel;
integer st;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (st = 0; st < 4; st = st + 1) sentinel_cnt[st] <= 8'd0;
    end else if (cycle_cnt > 10) begin
        for (st = 0; st < 4; st = st + 1) begin
            if (mem_array[(u_cpu_mt.pc_thread[st] >> 2) & (MEM_DEPTH-1)] === SENTINEL) begin
                if (sentinel_cnt[st] < 8'd255) sentinel_cnt[st] <= sentinel_cnt[st] + 8'd1;
            end else begin
                if (sentinel_cnt[st] > 0) sentinel_cnt[st] <= sentinel_cnt[st] - 8'd1;
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
        $display("[C%05d] IF=T%0d PC=0x%08H @PC=0x%08H | ID=T%0d EX1=T%0d EX2=T%0d MEM=T%0d WB=T%0d | D:a=0x%08H w=%b wd=0x%08H",
                 cycle_cnt,
                 u_cpu_mt.tid_if, i_mem_addr, instr_at_pc,
                 u_cpu_mt.tid_id, u_cpu_mt.tid_ex1, u_cpu_mt.tid_ex2,
                 u_cpu_mt.tid_mem, u_cpu_mt.tid_wb,
                 d_mem_addr, d_mem_wen, d_mem_wdata);
    end
    if (TRACE_EN && rst_n && (cp_wen || cp_ren) && cycle_cnt <= TRACE_LIMIT) begin
        if (cp_wen)
            $display("         >> CP10 MCR: CR%0d <= 0x%08H @ cycle %0d", cp_reg, cp_wr_data, cycle_cnt);
        if (cp_ren)
            $display("         >> CP10 MRC: CR%0d => 0x%08H @ cycle %0d", cp_reg, cp_rd_data, cycle_cnt);
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
    for (k = 0; k < MEM_DEPTH; k = k + 1) mem_array[k] = 32'h0;
end
endtask

task mem_w;
    input [31:0] byte_addr;
    input [31:0] data;
begin
    mem_array[byte_addr >> 2] = data;
end
endtask

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

task check_reg_t;
    input [1:0] tid;
    input [3:0] rn;
    input [31:0] expected;
    input [256*8:1] msg;
    reg [31:0] actual;
begin
    actual = get_reg(tid, rn);
    if (actual === expected) begin
        $display("    [PASS] T%0d R%0d = 0x%08H  %0s", tid, rn, expected, msg);
        section_pass = section_pass + 1; total_pass = total_pass + 1;
    end else begin
        $display("    [FAIL] T%0d R%0d = 0x%08H, expected 0x%08H  %0s",
                 tid, rn, actual, expected, msg);
        section_fail = section_fail + 1; total_fail = total_fail + 1;
    end
end
endtask

task check_mem;
    input [31:0] byte_addr;
    input [31:0] expected;
    input [256*8:1] msg;
    reg [31:0] actual;
begin
    actual = mem_array[byte_addr >> 2];
    if (actual === expected) begin
        $display("    [PASS] [0x%08H] = 0x%08H  %0s", byte_addr, expected, msg);
        section_pass = section_pass + 1; total_pass = total_pass + 1;
    end else begin
        $display("    [FAIL] [0x%08H] = 0x%08H, expected 0x%08H  %0s",
                 byte_addr, actual, expected, msg);
        section_fail = section_fail + 1; total_fail = total_fail + 1;
    end
end
endtask

task check_cp10_reg;
    input [31:0] expected;
    input [256*8:1] name;
    input [31:0] actual;
begin
    if (actual === expected) begin
        $display("    [PASS] CP10.%0s = 0x%08H", name, expected);
        section_pass = section_pass + 1; total_pass = total_pass + 1;
    end else begin
        $display("    [FAIL] CP10.%0s = 0x%08H, expected 0x%08H", name, actual, expected);
        section_fail = section_fail + 1; total_fail = total_fail + 1;
    end
end
endtask

task section_start;
    input [256*8:1] name;
begin
    current_section = name;
    section_pass = 0; section_fail = 0;
    $display("");
    $display("================================================================");
    $display("  %0s", name);
    $display("================================================================");
end
endtask

task section_end;
begin
    dump_all_threads();
    if (section_fail > 0)
        $display("  ** %0s: %0d PASSED, %0d FAILED (%0d cycles) **",
                 current_section, section_pass, section_fail, cycle_cnt);
    else
        $display("  -- %0s: all %0d passed (%0d cycles) --",
                 current_section, section_pass, cycle_cnt);
end
endtask

task dump_all_threads;
    integer t, r;
begin
    for (t = 0; t < 4; t = t + 1) begin
        $display("  Thread %0d: PC=0x%08H CPSR=%04b", t, u_cpu_mt.pc_thread[t], u_cpu_mt.cpsr_flags[t]);
        for (r = 0; r < 16; r = r + 4)
            $display("    R%-2d=0x%08H  R%-2d=0x%08H  R%-2d=0x%08H  R%-2d=0x%08H",
                     r, get_reg(t[1:0], r[3:0]),
                     r+1, get_reg(t[1:0], (r+1)),
                     r+2, get_reg(t[1:0], (r+2)),
                     r+3, get_reg(t[1:0], (r+3)));
    end
end
endtask

task dump_mem;
    input [31:0] base_byte;
    input integer count;
    integer i;
begin
    $display("  Memory @ 0x%08H (%0d words):", base_byte, count);
    for (i = 0; i < count; i = i + 1)
        $display("    [0x%08H] = 0x%08H", base_byte + (i*4), mem_array[(base_byte>>2) + i]);
end
endtask

// ── Load trivial BX LR + NOP for idle threads ────────────────────
task load_trivial_thread;
    input [31:0] base;
begin
    mem_w(base + 32'h00, 32'hE12FFF1E); // BX LR
    mem_w(base + 32'h04, 32'hE1A00000); // NOP (squashed)
end
endtask

// ═══════════════════════════════════════════════════════════════════
//  Thread Code Loading (Scenarios A–D: original network programs)
// ═══════════════════════════════════════════════════════════════════

task load_thread0_code;
    reg [31:0] B;
begin
    B = T0_CODE;
    mem_w(B+'h00, 32'hE52DB004); mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD014); mem_w(B+'h0C, 32'hE3A03C01);
    mem_w(B+'h10, 32'hE50B3010); mem_w(B+'h14, 32'hE51B3010);
    mem_w(B+'h18, 32'hE5933004); mem_w(B+'h1C, 32'hE35300AA);
    mem_w(B+'h20, 32'h0A000003);
    mem_w(B+'h24, 32'hE51B3010); mem_w(B+'h28, 32'hE3E02000);
    mem_w(B+'h2C, 32'hE5832010); mem_w(B+'h30, 32'hEA000021);
    mem_w(B+'h34, 32'hE51B3010); mem_w(B+'h38, 32'hE5933008);
    mem_w(B+'h3C, 32'hE50B3014); mem_w(B+'h40, 32'hE51B3010);
    mem_w(B+'h44, 32'hE593200C); mem_w(B+'h48, 32'hE51B3010);
    mem_w(B+'h4C, 32'hE5832008); mem_w(B+'h50, 32'hE51B3010);
    mem_w(B+'h54, 32'hE51B2014); mem_w(B+'h58, 32'hE583200C);
    mem_w(B+'h5C, 32'hE3A03000); mem_w(B+'h60, 32'hE50B3008);
    mem_w(B+'h64, 32'hE3A03000); mem_w(B+'h68, 32'hE50B300C);
    mem_w(B+'h6C, 32'hEA00000B);
    mem_w(B+'h70, 32'hE51B2010); mem_w(B+'h74, 32'hE51B300C);
    mem_w(B+'h78, 32'hE2833004); mem_w(B+'h7C, 32'hE1A03103);
    mem_w(B+'h80, 32'hE0823003); mem_w(B+'h84, 32'hE5933004);
    mem_w(B+'h88, 32'hE51B2008); mem_w(B+'h8C, 32'hE0823003);
    mem_w(B+'h90, 32'hE50B3008); mem_w(B+'h94, 32'hE51B300C);
    mem_w(B+'h98, 32'hE2833001); mem_w(B+'h9C, 32'hE50B300C);
    mem_w(B+'hA0, 32'hE51B300C); mem_w(B+'hA4, 32'hE3530003);
    mem_w(B+'hA8, 32'hDAFFFFF0);
    mem_w(B+'hAC, 32'hE51B3010); mem_w(B+'hB0, 32'hE51B2008);
    mem_w(B+'hB4, 32'hE5832010); mem_w(B+'hB8, 32'hE1A00000);
    mem_w(B+'hBC, 32'hE28BD000); mem_w(B+'hC0, 32'hE49DB004);
    mem_w(B+'hC4, 32'hE12FFF1E);
end
endtask

task load_thread1_code;
    reg [31:0] B;
begin
    B = T1_CODE;
    mem_w(B+'h00, 32'hE52DB004); mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C); mem_w(B+'h0C, 32'hE3A03C02);
    mem_w(B+'h10, 32'hE50B300C); mem_w(B+'h14, 32'hE51B300C);
    mem_w(B+'h18, 32'hE5933000); mem_w(B+'h1C, 32'hE3530000);
    mem_w(B+'h20, 32'h1A000016);
    mem_w(B+'h24, 32'hE3A03000); mem_w(B+'h28, 32'hE50B3008);
    mem_w(B+'h2C, 32'hEA00000C);
    mem_w(B+'h30, 32'hE51B300C); mem_w(B+'h34, 32'hE51B2008);
    mem_w(B+'h38, 32'hE2822002); mem_w(B+'h3C, 32'hE7932102);
    mem_w(B+'h40, 32'hE59F3048); mem_w(B+'h44, 32'hE0233002);
    mem_w(B+'h48, 32'hE51B200C); mem_w(B+'h4C, 32'hE51B1008);
    mem_w(B+'h50, 32'hE2811002); mem_w(B+'h54, 32'hE7823101);
    mem_w(B+'h58, 32'hE51B3008); mem_w(B+'h5C, 32'hE2833001);
    mem_w(B+'h60, 32'hE50B3008);
    mem_w(B+'h64, 32'hE51B3008); mem_w(B+'h68, 32'hE3530003);
    mem_w(B+'h6C, 32'hDAFFFFEF);
    mem_w(B+'h70, 32'hE51B300C); mem_w(B+'h74, 32'hE3A02001);
    mem_w(B+'h78, 32'hE5832000); mem_w(B+'h7C, 32'hEA000000);
    mem_w(B+'h80, 32'hE1A00000);
    mem_w(B+'h84, 32'hE28BD000); mem_w(B+'h88, 32'hE49DB004);
    mem_w(B+'h8C, 32'hE12FFF1E);
    mem_w(B+'h90, 32'hDEADBEEF);
end
endtask

task load_thread2_code;
    reg [31:0] B;
begin
    B = T2_CODE;
    mem_w(B+'h00, 32'hE52DB004); mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C); mem_w(B+'h0C, 32'hE3A03C03);
    mem_w(B+'h10, 32'hE50B3008); mem_w(B+'h14, 32'hE51B3008);
    mem_w(B+'h18, 32'hE5933004); mem_w(B+'h1C, 32'hE3530001);
    mem_w(B+'h20, 32'h8A000006);
    mem_w(B+'h24, 32'hE51B3008); mem_w(B+'h28, 32'hE3A02000);
    mem_w(B+'h2C, 32'hE5832000); mem_w(B+'h30, 32'hE51B3008);
    mem_w(B+'h34, 32'hE3A02000); mem_w(B+'h38, 32'hE5832004);
    mem_w(B+'h3C, 32'hEA000008);
    mem_w(B+'h40, 32'hE51B3008); mem_w(B+'h44, 32'hE5933004);
    mem_w(B+'h48, 32'hE2432001); mem_w(B+'h4C, 32'hE51B3008);
    mem_w(B+'h50, 32'hE5832004); mem_w(B+'h54, 32'hE51B3008);
    mem_w(B+'h58, 32'hE3A02001); mem_w(B+'h5C, 32'hE5832000);
    mem_w(B+'h60, 32'hE1A00000);
    mem_w(B+'h64, 32'hE28BD000); mem_w(B+'h68, 32'hE49DB004);
    mem_w(B+'h6C, 32'hE12FFF1E);
end
endtask

task load_thread3_code;
    reg [31:0] B;
begin
    B = T3_CODE;
    mem_w(B+'h00, 32'hE52DB004); mem_w(B+'h04, 32'hE28DB000);
    mem_w(B+'h08, 32'hE24DD00C); mem_w(B+'h0C, 32'hE3A03C04);
    mem_w(B+'h10, 32'hE50B3008); mem_w(B+'h14, 32'hE51B3008);
    mem_w(B+'h18, 32'hE593300C); mem_w(B+'h1C, 32'hE3530017);
    mem_w(B+'h20, 32'h1A000003);
    mem_w(B+'h24, 32'hE51B3008); mem_w(B+'h28, 32'hE3A02002);
    mem_w(B+'h2C, 32'hE5832000); mem_w(B+'h30, 32'hEA000003);
    mem_w(B+'h34, 32'hE51B3008); mem_w(B+'h38, 32'hE3A02001);
    mem_w(B+'h3C, 32'hE5832000); mem_w(B+'h40, 32'hE1A00000);
    mem_w(B+'h44, 32'hE28BD000); mem_w(B+'h48, 32'hE49DB004);
    mem_w(B+'h4C, 32'hE12FFF1E);
end
endtask

task init_thread_data;
begin
    mem_w(32'h0100, 32'h0); mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'h1111_1111); mem_w(32'h010C, 32'h2222_2222);
    mem_w(32'h0110, 32'h0); mem_w(32'h0114, 32'h10);
    mem_w(32'h0118, 32'h20); mem_w(32'h011C, 32'h30); mem_w(32'h0120, 32'h40);
    mem_w(32'h0200, 32'h0); mem_w(32'h0204, 32'h0);
    mem_w(32'h0208, 32'hAAAA_AAAA); mem_w(32'h020C, 32'hBBBB_BBBB);
    mem_w(32'h0210, 32'hCCCC_CCCC); mem_w(32'h0214, 32'hDDDD_DDDD);
    mem_w(32'h0300, 32'h0); mem_w(32'h0304, 32'h5);
    mem_w(32'h0400, 32'h0); mem_w(32'h0404, 32'h0);
    mem_w(32'h0408, 32'h0); mem_w(32'h040C, 32'h17);
end
endtask

// ═══════════════════════════════════════════════════════════════════
//  Run Test
// ═══════════════════════════════════════════════════════════════════

task run_mt_test;
    output integer cycles_used;
begin
    rst_n = 0; cycle_cnt = 0;
    cpu_start_tb = 0; entry_pc_tb = 32'd0;
    repeat (5) @(posedge clk);
    @(negedge clk);
    rst_n = 1;

    // Hierarchical PC + RF init (separate code regions per thread)
    u_cpu_mt.pc_thread[0] = T0_CODE; u_cpu_mt.pc_thread[1] = T1_CODE;
    u_cpu_mt.pc_thread[2] = T2_CODE; u_cpu_mt.pc_thread[3] = T3_CODE;

    u_cpu_mt.THREAD_RF[0].u_rf.regs[13] = T0_SP;
    u_cpu_mt.THREAD_RF[0].u_rf.regs[14] = T0_RET;
    u_cpu_mt.THREAD_RF[1].u_rf.regs[13] = T1_SP;
    u_cpu_mt.THREAD_RF[1].u_rf.regs[14] = T1_RET;
    u_cpu_mt.THREAD_RF[2].u_rf.regs[13] = T2_SP;
    u_cpu_mt.THREAD_RF[2].u_rf.regs[14] = T2_RET;
    u_cpu_mt.THREAD_RF[3].u_rf.regs[13] = T3_SP;
    u_cpu_mt.THREAD_RF[3].u_rf.regs[14] = T3_RET;

    mem_w(T0_RET, SENTINEL); mem_w(T1_RET, SENTINEL);
    mem_w(T2_RET, SENTINEL); mem_w(T3_RET, SENTINEL);

    // v2.8: bypass cpu_start for legacy tests — force running + clear halt
    u_cpu_mt.running = 1'b1;
    u_cpu_mt.halted = 4'b0;
    u_cpu_mt.halt_seen_once = 4'b0;

    begin : run_loop
        forever begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            if (cycle_cnt > 0 && (cycle_cnt % 500 == 0))
                $display("  [PROGRESS] cycle %0d: sentinel = %04b", cycle_cnt, thread_at_sentinel);
            if (all_threads_done) begin
                $display("  [DONE] All threads at sentinel, cycle %0d", cycle_cnt);
                repeat (10) @(posedge clk);
                cycles_used = cycle_cnt; disable run_loop;
            end
            if (cycle_cnt >= MAX_CYCLES) begin
                $display("  *** TIMEOUT after %0d cycles ***", MAX_CYCLES);
                $display("  Sentinel: %04b  PCs: %08H %08H %08H %08H", thread_at_sentinel,
                         u_cpu_mt.pc_thread[0], u_cpu_mt.pc_thread[1],
                         u_cpu_mt.pc_thread[2], u_cpu_mt.pc_thread[3]);
                dump_all_threads();
                cycles_used = cycle_cnt; disable run_loop;
            end
        end
    end
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Scenario A: Normal Path (original 4-thread network processing)
// ═══════════════════════════════════════════════════════════════════
task test_scenario_A;
    integer cyc;
begin
    section_start("Scenario A: Normal Path (all four threads)");
    mem_clear();
    load_thread0_code(); load_thread1_code();
    load_thread2_code(); load_thread3_code();
    init_thread_data();
    run_mt_test(cyc);

    check_mem(32'h0108, 32'h2222_2222, "T0: src<-dst after swap");
    check_mem(32'h010C, 32'h1111_1111, "T0: dst<-src after swap");
    check_mem(32'h0110, 32'h0000_00A0, "T0: checksum = 160");
    check_mem(32'h0200, 32'h0000_0001, "T1: done flag = 1");
    check_mem(32'h0208, 32'h7407_1445, "T1: AAAAAAAA^DEADBEEF");
    check_mem(32'h020C, 32'h6516_0554, "T1: BBBBBBBB^DEADBEEF");
    check_mem(32'h0210, 32'h1261_7223, "T1: CCCCCCCC^DEADBEEF");
    check_mem(32'h0214, 32'h0370_6332, "T1: DDDDDDDD^DEADBEEF");
    check_mem(32'h0300, 32'h0000_0001, "T2: status = 1");
    check_mem(32'h0304, 32'h0000_0004, "T2: counter = 4");
    check_mem(32'h0400, 32'h0000_0002, "T3: result = 2 (match)");
    check_reg_t(2'd0, 4'd13, T0_SP, "T0: SP restored");
    check_reg_t(2'd1, 4'd13, T1_SP, "T1: SP restored");
    check_reg_t(2'd2, 4'd13, T2_SP, "T2: SP restored");
    check_reg_t(2'd3, 4'd13, T3_SP, "T3: SP restored");
    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Scenario B: Error / Alternate Paths
// ═══════════════════════════════════════════════════════════════════
task test_scenario_B;
    integer cyc;
begin
    section_start("Scenario B: Alternate / Error Paths");
    mem_clear();
    load_thread0_code(); load_thread1_code();
    load_thread2_code(); load_thread3_code();

    mem_w(32'h0100, 32'h0); mem_w(32'h0104, 32'h0000_00BB); // bad header
    mem_w(32'h0108, 32'h1111_1111); mem_w(32'h010C, 32'h2222_2222);
    mem_w(32'h0110, 32'h0); mem_w(32'h0114, 32'h10);
    mem_w(32'h0118, 32'h20); mem_w(32'h011C, 32'h30); mem_w(32'h0120, 32'h40);
    mem_w(32'h0200, 32'h1); mem_w(32'h0204, 32'h0); // flag=1 skip
    mem_w(32'h0208, 32'hAAAA_AAAA); mem_w(32'h020C, 32'hBBBB_BBBB);
    mem_w(32'h0210, 32'hCCCC_CCCC); mem_w(32'h0214, 32'hDDDD_DDDD);
    mem_w(32'h0300, 32'hFFFF_FFFF); mem_w(32'h0304, 32'h1); // counter=1
    mem_w(32'h0400, 32'h0); mem_w(32'h0404, 32'h0);
    mem_w(32'h0408, 32'h0); mem_w(32'h040C, 32'h2A); // field=42

    run_mt_test(cyc);

    check_mem(32'h0110, 32'hFFFF_FFFF, "T0: result = -1 (error)");
    check_mem(32'h0108, 32'h1111_1111, "T0: src unchanged");
    check_mem(32'h0200, 32'h0000_0001, "T1: flag unchanged");
    check_mem(32'h0208, 32'hAAAA_AAAA, "T1: data[0] unmodified");
    check_mem(32'h0300, 32'h0000_0000, "T2: status = 0");
    check_mem(32'h0304, 32'h0000_0000, "T2: counter = 0");
    check_mem(32'h0400, 32'h0000_0001, "T3: result = 1 (no match)");
    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Scenario C: Edge-Case Data Values
// ═══════════════════════════════════════════════════════════════════
task test_scenario_C;
    integer cyc;
begin
    section_start("Scenario C: Edge-Case Data Values");
    mem_clear();
    load_thread0_code(); load_thread1_code();
    load_thread2_code(); load_thread3_code();

    mem_w(32'h0100, 32'h0); mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'hFFFF_FFFF); mem_w(32'h010C, 32'h1);
    mem_w(32'h0110, 32'h0);
    mem_w(32'h0114, 32'h7FFF_FFFF); mem_w(32'h0118, 32'h1);
    mem_w(32'h011C, 32'h8000_0000); mem_w(32'h0120, 32'h0);
    mem_w(32'h0200, 32'h0); mem_w(32'h0204, 32'h0);
    mem_w(32'h0208, 32'hDEAD_BEEF); mem_w(32'h020C, 32'h0);
    mem_w(32'h0210, 32'hFFFF_FFFF); mem_w(32'h0214, 32'h1234_5678);
    mem_w(32'h0300, 32'hAAAA_AAAA); mem_w(32'h0304, 32'h0);
    mem_w(32'h0400, 32'h0); mem_w(32'h0404, 32'h0);
    mem_w(32'h0408, 32'h0); mem_w(32'h040C, 32'h17);

    run_mt_test(cyc);

    check_mem(32'h0108, 32'h0000_0001, "T0: src<-dst swapped");
    check_mem(32'h010C, 32'hFFFF_FFFF, "T0: dst<-src swapped");
    check_mem(32'h0110, 32'h0000_0000, "T0: checksum wraps to 0");
    check_mem(32'h0200, 32'h0000_0001, "T1: done flag");
    check_mem(32'h0208, 32'h0000_0000, "T1: DEADBEEF^DEADBEEF=0");
    check_mem(32'h020C, 32'hDEAD_BEEF, "T1: 0^DEADBEEF");
    check_mem(32'h0210, 32'h2152_4110, "T1: FFFFFFFF^DEADBEEF");
    check_mem(32'h0214, 32'hCC99_E897, "T1: 12345678^DEADBEEF");
    check_mem(32'h0300, 32'h0000_0000, "T2: status=0");
    check_mem(32'h0304, 32'h0000_0000, "T2: counter=0");
    check_mem(32'h0400, 32'h0000_0002, "T3: result=2");
    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Scenario D: Thread Isolation
// ═══════════════════════════════════════════════════════════════════
task test_scenario_D;
    integer cyc;
begin
    section_start("Scenario D: Thread Isolation");
    mem_clear();
    load_thread0_code(); load_thread1_code();
    load_thread2_code(); load_thread3_code();

    mem_w(32'h0100, 32'h0); mem_w(32'h0104, 32'h0000_00AA);
    mem_w(32'h0108, 32'hAAAA_0000); mem_w(32'h010C, 32'h0000_BBBB);
    mem_w(32'h0110, 32'h0);
    mem_w(32'h0114, 32'h1); mem_w(32'h0118, 32'h2);
    mem_w(32'h011C, 32'h3); mem_w(32'h0120, 32'h4);
    mem_w(32'h0200, 32'h0); mem_w(32'h0204, 32'h0);
    mem_w(32'h0208, 32'h0102_0304); mem_w(32'h020C, 32'h0506_0708);
    mem_w(32'h0210, 32'h090A_0B0C); mem_w(32'h0214, 32'h0D0E_0F10);
    mem_w(32'h0300, 32'h0); mem_w(32'h0304, 32'hA);
    mem_w(32'h0400, 32'h0); mem_w(32'h0404, 32'h0);
    mem_w(32'h0408, 32'h0); mem_w(32'h040C, 32'h17);

    run_mt_test(cyc);

    check_mem(32'h0108, 32'h0000_BBBB, "T0: src<-dst");
    check_mem(32'h010C, 32'hAAAA_0000, "T0: dst<-src");
    check_mem(32'h0110, 32'h0000_000A, "T0: checksum=10");
    check_mem(32'h0200, 32'h0000_0001, "T1: flag=1");
    check_mem(32'h0208, 32'hDFAF_BDEB, "T1: 01020304^DEADBEEF");
    check_mem(32'h0300, 32'h0000_0001, "T2: status=1");
    check_mem(32'h0304, 32'h0000_0009, "T2: counter=9");
    check_mem(32'h0400, 32'h0000_0002, "T3: result=2");
    check_mem(32'h0104, 32'h0000_00AA, "Cross-thread: T0 header intact");
    check_mem(32'h040C, 32'h0000_0017, "Cross-thread: T3 field intact");
    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Scenario E: CP10 Coprocessor MCR/MRC Write/Readback
//
//  Thread 0 program:
//    MOV R0, #0x100      ; base for results
//    MOV R1, #42         ; test value
//    MCR p10,0,R1,CR8    ; CR8 (scratch) = 42
//    MRC p10,0,R2,CR8    ; R2 <- CR8
//    STR R2, [R0]        ; mem[0x100] = 42
//    MOV R3, #0x500      ; test value (5 rot 24)
//    MCR p10,0,R3,CR0    ; CR0 (DMA src) = 0x500
//    MRC p10,0,R4,CR0    ; R4 <- CR0
//    STR R4, [R0,#4]     ; mem[0x104] = 0x500
//    MOV R5, #0x600      ; test value (6 rot 24)
//    MCR p10,0,R5,CR1    ; CR1 (DMA dst) = 0x600
//    MRC p10,0,R6,CR1    ; R6 <- CR1
//    STR R6, [R0,#8]     ; mem[0x108] = 0x600
//    MOV R7, #0xF
//    MCR p10,0,R7,CR7    ; CR7 (thread mask) = 0xF
//    MRC p10,0,R8,CR7    ; R8 <- CR7
//    STR R8, [R0,#12]    ; mem[0x10C] = 0xF
//    MOV R9, #0xFF
//    MCR p10,0,R9,CR4    ; CR4 (GPU entry PC) = 0xFF
//    MRC p10,0,R10,CR4   ; R10 <- CR4
//    STR R10,[R0,#16]    ; mem[0x110] = 0xFF
//    MRC p10,0,R11,CR6   ; R11 <- CR6 (GPU status, read-only)
//    STR R11,[R0,#20]    ; mem[0x114] = 0 (gpu idle, not done)
//    BX LR
//
//  MCR encoding: EE0{CRn}{Rd}A10   (bit20=0)
//  MRC encoding: EE1{CRn}{Rd}A10   (bit20=1)
// ═══════════════════════════════════════════════════════════════════

task load_cp10_test_code;
    reg [31:0] B;
begin
    B = T0_CODE;
    mem_w(B+'h00, 32'hE3A00C01);  // MOV R0, #0x100
    mem_w(B+'h04, 32'hE3A0102A);  // MOV R1, #42
    mem_w(B+'h08, 32'hEE081A10);  // MCR p10,0,R1,CR8,CR0,0
    mem_w(B+'h0C, 32'hEE182A10);  // MRC p10,0,R2,CR8,CR0,0
    mem_w(B+'h10, 32'hE5802000);  // STR R2, [R0]
    mem_w(B+'h14, 32'hE3A03C05);  // MOV R3, #0x500
    mem_w(B+'h18, 32'hEE003A10);  // MCR p10,0,R3,CR0,CR0,0
    mem_w(B+'h1C, 32'hEE104A10);  // MRC p10,0,R4,CR0,CR0,0
    mem_w(B+'h20, 32'hE5804004);  // STR R4, [R0, #4]
    mem_w(B+'h24, 32'hE3A05C06);  // MOV R5, #0x600
    mem_w(B+'h28, 32'hEE015A10);  // MCR p10,0,R5,CR1,CR0,0
    mem_w(B+'h2C, 32'hEE116A10);  // MRC p10,0,R6,CR1,CR0,0
    mem_w(B+'h30, 32'hE5806008);  // STR R6, [R0, #8]
    mem_w(B+'h34, 32'hE3A0700F);  // MOV R7, #0xF
    mem_w(B+'h38, 32'hEE077A10);  // MCR p10,0,R7,CR7,CR0,0
    mem_w(B+'h3C, 32'hEE178A10);  // MRC p10,0,R8,CR7,CR0,0
    mem_w(B+'h40, 32'hE580800C);  // STR R8, [R0, #12]
    mem_w(B+'h44, 32'hE3A090FF);  // MOV R9, #0xFF
    mem_w(B+'h48, 32'hEE049A10);  // MCR p10,0,R9,CR4,CR0,0
    mem_w(B+'h4C, 32'hEE14AA10);  // MRC p10,0,R10,CR4,CR0,0
    mem_w(B+'h50, 32'hE580A010);  // STR R10, [R0, #16]
    mem_w(B+'h54, 32'hEE16BA10);  // MRC p10,0,R11,CR6,CR0,0
    mem_w(B+'h58, 32'hE580B014);  // STR R11, [R0, #20]
    mem_w(B+'h5C, 32'hE12FFF1E);  // BX LR
    $display("  [LOAD] CP10 test: T0 code 0x%08H-0x%08H (24 instr)", B, B+'h5C);
end
endtask

task test_scenario_E;
    integer cyc;
begin
    section_start("Scenario E: CP10 MCR/MRC Write/Readback");
    mem_clear();

    load_cp10_test_code();
    load_trivial_thread(T1_CODE);
    load_trivial_thread(T2_CODE);
    load_trivial_thread(T3_CODE);

    run_mt_test(cyc);

    $display("");
    $display("  -- Memory results at 0x100 --");
    dump_mem(32'h0100, 6);

    $display("");
    $display("  -- MCR/MRC readback via memory --");
    check_mem(32'h0100, 32'h0000_002A, "CR8 scratch readback = 42");
    check_mem(32'h0104, 32'h0000_0500, "CR0 DMA src readback = 0x500");
    check_mem(32'h0108, 32'h0000_0600, "CR1 DMA dst readback = 0x600");
    check_mem(32'h010C, 32'h0000_000F, "CR7 thread mask readback = 0xF");
    check_mem(32'h0110, 32'h0000_00FF, "CR4 GPU entry PC readback = 0xFF");
    check_mem(32'h0114, 32'h0000_0000, "CR6 GPU status = 0 (idle)");

    $display("");
    $display("  -- CP10 internal register state --");
    check_cp10_reg(32'h0000_0500, "CR0_DMA_SRC", {u_cp10.cr0_dma_src});
    check_cp10_reg(32'h0000_0600, "CR1_DMA_DST", {u_cp10.cr1_dma_dst});
    check_cp10_reg(32'h0000_002A, "CR8_SCRATCH", {u_cp10.cr8_gpu_scratch});
    check_cp10_reg(32'h0000_000F, "CR7_TMASK", {28'd0, u_cp10.cr7_thread_mask});
    check_cp10_reg(32'h0000_00FF, "CR4_GPU_PC", {u_cp10.cr4_gpu_pc});

    $display("");
    $display("  -- Register checks --");
    check_reg_t(2'd0, 4'd2, 32'h0000_002A, "T0 R2 = CR8 readback");
    check_reg_t(2'd0, 4'd4, 32'h0000_0500, "T0 R4 = CR0 readback");
    check_reg_t(2'd0, 4'd6, 32'h0000_0600, "T0 R6 = CR1 readback");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  Scenario F: MUL/MLA/UMULL Multiply Instructions
//
//  Thread 0 program:
//    MOV R1, #7           ; multiplicand
//    MOV R2, #6           ; multiplier
//    MUL R4, R1, R2       ; R4 = 7*6 = 42
//    MOV R3, #10          ; accumulate
//    MLA R5, R1, R2, R3   ; R5 = 7*6+10 = 52
//    MVN R6, #0           ; R6 = 0xFFFFFFFF
//    UMULL R7, R8, R6, R2 ; R8:R7 = 0xFFFFFFFF * 6 = 0x5_FFFFFFFA
//    MOV R0, #0x100       ; base
//    STR R4, [R0]         ; mem[0x100] = 42
//    STR R5, [R0, #4]     ; mem[0x104] = 52
//    STR R7, [R0, #8]     ; mem[0x108] = 0xFFFFFFFA
//    STR R8, [R0, #12]    ; mem[0x10C] = 0x00000005
//    BX LR
//
//  MUL Rd,Rm,Rs:     E0{Rd}0{Rs}9{Rm}  (A=0,S=0)
//  MLA Rd,Rm,Rs,Rn:  E02{Rd}{Rn}{Rs}9{Rm}  (A=1,S=0)
//  UMULL RdLo,RdHi,Rm,Rs: E08{RdHi}{RdLo}{Rs}9{Rm}
// ═══════════════════════════════════════════════════════════════════

task load_mul_test_code;
    reg [31:0] B;
begin
    B = T0_CODE;
    mem_w(B+'h00, 32'hE3A01007);  // MOV R1, #7
    mem_w(B+'h04, 32'hE3A02006);  // MOV R2, #6
    mem_w(B+'h08, 32'hE0040291);  // MUL R4, R1, R2       (42)
    mem_w(B+'h0C, 32'hE3A0300A);  // MOV R3, #10
    mem_w(B+'h10, 32'hE0253291);  // MLA R5, R1, R2, R3   (52)
    mem_w(B+'h14, 32'hE3E06000);  // MVN R6, #0           (0xFFFFFFFF)
    mem_w(B+'h18, 32'hE0887296);  // UMULL R7, R8, R6, R2 (0x5_FFFFFFFA)
    mem_w(B+'h1C, 32'hE3A00C01);  // MOV R0, #0x100
    mem_w(B+'h20, 32'hE5804000);  // STR R4, [R0]
    mem_w(B+'h24, 32'hE5805004);  // STR R5, [R0, #4]
    mem_w(B+'h28, 32'hE5807008);  // STR R7, [R0, #8]
    mem_w(B+'h2C, 32'hE580800C);  // STR R8, [R0, #12]
    mem_w(B+'h30, 32'hE12FFF1E);  // BX LR
    $display("  [LOAD] MUL test: T0 code 0x%08H-0x%08H (13 instr)", B, B+'h30);
end
endtask

task test_scenario_F;
    integer cyc;
begin
    section_start("Scenario F: MUL/MLA/UMULL Instructions");
    mem_clear();

    load_mul_test_code();
    load_trivial_thread(T1_CODE);
    load_trivial_thread(T2_CODE);
    load_trivial_thread(T3_CODE);

    run_mt_test(cyc);

    $display("");
    dump_mem(32'h0100, 4);

    $display("");
    $display("  -- MUL results --");
    check_mem(32'h0100, 32'h0000_002A, "MUL: 7*6 = 42");
    check_mem(32'h0104, 32'h0000_0034, "MLA: 7*6+10 = 52");
    check_mem(32'h0108, 32'hFFFF_FFFA, "UMULL lo: 0xFFFFFFFF*6 = ...FFFFFFFA");
    check_mem(32'h010C, 32'h0000_0005, "UMULL hi: 0xFFFFFFFF*6 = 0x5...");

    $display("");
    $display("  -- Register checks --");
    check_reg_t(2'd0, 4'd4, 32'h0000_002A, "T0 R4 = MUL result");
    check_reg_t(2'd0, 4'd5, 32'h0000_0034, "T0 R5 = MLA result");
    check_reg_t(2'd0, 4'd7, 32'hFFFF_FFFA, "T0 R7 = UMULL lo");
    check_reg_t(2'd0, 4'd8, 32'h0000_0005, "T0 R8 = UMULL hi");

    section_end();
end
endtask


// ═══════════════════════════════════════════════════════════════════
//  M A I N   S T I M U L U S
// ═══════════════════════════════════════════════════════════════════

initial begin
    $dumpfile("cpu_mt_tb.vcd");
    $dumpvars(0, cpu_mt_tb);

    total_pass = 0; total_fail = 0;
    rst_n = 0;
    cpu_start_tb = 0; entry_pc_tb = 32'd0;

    $display("");
    $display("================================================================");
    $display("  Quad-Thread ARMv4T Pipeline Testbench v2");
    $display("  SYNC_MEM=%0d  TRACE_EN=%0d  MAX_CYCLES=%0d", SYNC_MEM, TRACE_EN, MAX_CYCLES);
    $display("  Includes: CP10 + DMA engine + MAC unit");
    $display("================================================================");
    $display("");

    test_scenario_A();
    test_scenario_B();
    test_scenario_C();
    test_scenario_D();
    test_scenario_E();
    test_scenario_F();

    $display("");
    $display("================================================================");
    if (total_fail == 0)
        $display("  *** ALL %0d CHECKS PASSED ***", total_pass);
    else
        $display("  *** %0d PASSED, %0d FAILED ***", total_pass, total_fail);
    $display("  Total checks: %0d", total_pass + total_fail);
    $display("================================================================");
    $display("");

    #(CLK_PERIOD * 5);
    $finish;
end

endmodule