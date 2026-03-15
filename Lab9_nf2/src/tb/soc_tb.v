/* file: soc_tb.v
 * SoC integration testbench — drives ONLY the NetFPGA RX/TX interface.
 * All computation runs inside soc (real CPU, CP10, DMA, GPU).
 *
 * Test 1: CPU ADD — basic packet round-trip
 * Test 2: CPU MUL — MAC unit via packet
 * Test 3: CPU MCR/MRC — CP10 scratch write/readback
 * Test 4: CPU conditional branch (CMP + BEQ taken)
 * Test 5: CPU + DMA + GPU — full heterogeneous compute pipeline
 * Test 6: Back-to-back packets (reuse IMEM, new data)
 * Test 7: CPU MLA — multiply-accumulate
 * Test 8: CPU loop — sum array of 4 elements
 * Test 9: Multiple CP10 register write/readback (CR0,CR1,CR4,CR8)
 * Test 10: NOP command skip (pkt_proc)
 * Test 11: Large DMEM load + readback (16 words)
 * Test 12: CPU conditional branch not-taken (CMP + BEQ NE path)
 * Test 13: GPU LD/ADD/ST with DMA D_UNPACK preload (full heterogeneous data path)
 * Test 14: GPU WMMA.MMA tensor core through full SoC
 * Test 15: GPU per-thread TID differentiation through SoC
 * Test 16: GPU K1 vec_add int16 (LD, D_UNPACK, 10+20=30)
 * Test 17: GPU K2 vec_sub int16 (LD, 100-30=70)
 * Test 18: GPU K3 bf16_mul (LD, 2.0x3.0=6.0)
 * Test 19: GPU K4 bf16_fma (LD, 2.0x3.0+1.0=7.0)
 * Test 20: GPU K5 relu bf16 (LD, per-bank: -1→0, 2→2, -3→0, 5→5)
 * Test 21: GPU K6 WMMA matmul (LD, per-bank identity A, D=I×2=all-2.0)
 * Test 22: Full heterogeneous pipeline (CPU pre→GPU K1×2→GPU K2+10→CPU post)
 * Test 23: CVT round-trip with different rD/rA (int→bf16→int, canary-guarded)
 * Test 24: CVT self-referencing rD==rA (deadlock diagnostic, canary-guarded)
 * Test 25: Full heterogeneous with CVT (CPU pre→GPU CVT×2→GPU CVT+10→CPU post)
 *
 * ARM encoding reference:
 *   MOV Rd, #imm8:        E3A0{Rd}{rot}{imm8}
 *   LDR Rd, [Rn, #off12]: E59{Rn}{Rd}{off12}
 *   STR Rd, [Rn, #off12]: E58{Rn}{Rd}{off12}
 *   ADD Rd, Rn, Rm:       E08{Rn}{Rd}00{Rm}   (0x with Rd at [15:12])
 *   CMP Rn, #imm8:        E35{Rn}0{imm8}
 *   BEQ offset:            0A{offset24}
 *   B offset:              EA{offset24}
 *   B . (halt):            EAFFFFFE
 *   MUL Rd,Rm,Rs:          E0{Rd}00{Rs}9{Rm}
 *   MCR p10,0,Rd,CRn:      EE0{CRn}{Rd}A10
 *   MRC p10,0,Rd,CRn:      EE1{CRn}{Rd}A10
 *   NOP:                   E1A00000
 *
 * Author: Jeremy Cai
 * Date: Mar. 5, 2026
 * Version: 6.0
 */

`timescale 1ns / 1ps
`include "soc.v"

module soc_tb;

localparam CLK_PERIOD = 10;
localparam MAX_CYCLES = 8000;

// ═══════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════
localparam [31:0] ARM_NOP  = 32'hE1A00000;
localparam [31:0] ARM_HALT = 32'hEAFF_FFFE;

// ═══════════════════════════════════════════════════════════════════
//  Clock / Reset
// ═══════════════════════════════════════════════════════════════════
reg clk, rst_n;
initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

// ═══════════════════════════════════════════════════════════════════
//  DUT I/O
// ═══════════════════════════════════════════════════════════════════
reg  [63:0] in_data;
reg  [7:0]  in_ctrl;
reg         in_wr;
wire        in_rdy;

wire [63:0] out_data;
wire [7:0]  out_ctrl;
wire        out_wr;
reg         out_rdy;

// ═══════════════════════════════════════════════════════════════════
//  DUT
// ═══════════════════════════════════════════════════════════════════
soc u_soc (
    .clk(clk), .rst_n(rst_n),
    .in_data(in_data), .in_ctrl(in_ctrl), .in_wr(in_wr), .in_rdy(in_rdy),
    .out_data(out_data), .out_ctrl(out_ctrl), .out_wr(out_wr), .out_rdy(out_rdy)
);

// ═══════════════════════════════════════════════════════════════════
//  Test Infrastructure
// ═══════════════════════════════════════════════════════════════════
integer pass_cnt = 0, fail_cnt = 0, test_id = 0;
integer cycle_cnt;

reg [63:0] tx_data [0:127];
reg [7:0]  tx_ctrl [0:127];
integer tx_cnt;

task tick; begin @(posedge clk); #1; end endtask

// ── Command word builder ─────────────────────────────────────
function [63:0] cmd;
    input [3:0] opcode;
    input [11:0] addr;
    input [15:0] count;
    input [31:0] param;
    cmd = {opcode, addr, count, param};
endfunction

// ── RX word injection ────────────────────────────────────────
task rx;
    input [63:0] data;
    input [7:0]  ctrl;
begin
    in_data = data; in_ctrl = ctrl; in_wr = 1; tick;
end
endtask

task rx_end;
begin
    in_wr = 0; in_data = 64'd0; in_ctrl = 8'd0; tick;
end
endtask

// ── Send N pairs of ARM NOPs as LOAD_IMEM data ──────────────
task send_nop_pairs;
    input integer n;
    integer i;
begin
    for (i = 0; i < n; i = i + 1)
        rx({ARM_NOP, ARM_NOP}, 8'h00);
end
endtask

// ── Universal ARM program for GPU kernel tests ────────────────
// 80 instrs = 40 DWs. Parameterized D_UNPACK xfer_len and D_PACK src.
// Layout: CPU DMEM[0..15]=GPU IMEM, [16..31]=GPU data, [32..47]=readback
//
// Phase 1: DMA D_IMEM (DMEM[0..15]→GPU IMEM, 16 instrs)
// Phase 2: DMA D_UNPACK (DMEM[16..]→GPU DMEM, burst_all, xfer=dunpack_len)
// Phase 3: GPU launch (entry_pc=0, mask=0xF), 24 NOP wait
// Phase 4: DMA D_PACK (GPU DMEM[dpack_src..]→DMEM[32..], burst_all, xfer=2)
// Phase 5: B . (halt)
task send_gpu_arm;
    input [7:0] ctrl;
    input [7:0] dunpack_len; // D_UNPACK xfer_len per bank
    input [7:0] dpack_src;   // D_PACK GPU DMEM source addr
    input [7:0] dpack_len;   // D_PACK xfer_len per bank
begin
    rx(cmd(4'h1, 12'h000, 16'd40, 32'h0), ctrl);
    // Phase 1: D_IMEM [0..7]
    rx({32'hEE000A10, 32'hE3A00000}, 8'h00); // [1]MCR CR0; [0]MOV R0,#0
    rx({32'hE3A01010, 32'hEE010A10}, 8'h00); // [3]MOV R1,#16; [2]MCR CR1
    rx({32'hE3A02005, 32'hEE021A10}, 8'h00); // [5]MOV R2,#5; [4]MCR CR2
    rx({ARM_NOP,      32'hEE032A10}, 8'h00); // [7]NOP; [6]MCR CR3→DMA
    send_nop_pairs(4);                        // [8..15]
    // Phase 2: D_UNPACK [16..23]
    rx({32'hEE000A10, 32'hE3A00010}, 8'h00); // [17]MCR CR0; [16]MOV R0,#16
    rx({32'hEE011A10, 32'hE3A01000}, 8'h00); // [19]MCR CR1; [18]MOV R1,#0
    rx({32'hEE022A10, {24'hE3A020, dunpack_len}}, 8'h00); // [21]MCR CR2; [20]MOV R2,#len
    rx({32'hEE033A10, 32'hE3A03041}, 8'h00); // [23]MCR CR3→DMA; [22]MOV R3,#65
    send_nop_pairs(5);                        // [24..33]
    // Phase 3: GPU launch [34..39]
    rx({32'hEE040A10, 32'hE3A00000}, 8'h00); // [35]MCR CR4; [34]MOV R0,#0
    rx({32'hEE071A10, 32'hE3A0100F}, 8'h00); // [37]MCR CR7; [36]MOV R1,#15
    rx({32'hEE052A10, 32'hE3A02001}, 8'h00); // [39]MCR CR5→launch; [38]MOV R2,#1
    send_nop_pairs(12);                       // [40..63]
    // Phase 4: D_PACK [64..71]
    rx({32'hEE000A10, {24'hE3A000, dpack_src}}, 8'h00); // [65]MCR CR0; [64]MOV R0,#src
    rx({32'hEE011A10, 32'hE3A01020}, 8'h00); // [67]MCR CR1; [66]MOV R1,#32
    rx({32'hEE022A10, {24'hE3A020, dpack_len}}, 8'h00); // [69]MCR CR2; [68]MOV R2,#dpack_len
    rx({32'hEE033A10, 32'hE3A03043}, 8'h00); // [71]MCR CR3→DMA; [70]MOV R3,#67
    send_nop_pairs(3);                        // [72..77]
    rx({ARM_NOP, ARM_HALT}, 8'h00);           // [79]NOP; [78]B .
end
endtask

// ── Send GPU program (16 instrs padded to 16 with NOP) ────────
// Caller provides 8 data words (16 GPU instrs). Unused slots = 0.
// LOAD_DMEM at addr=0, count=8
task send_gpu_imem;
    input [63:0] dw0, dw1, dw2, dw3, dw4, dw5, dw6, dw7;
begin
    rx(cmd(4'h2, 12'h000, 16'd8, 32'h0), 8'h00);
    rx(dw0, 8'h00); rx(dw1, 8'h00); rx(dw2, 8'h00); rx(dw3, 8'h00);
    rx(dw4, 8'h00); rx(dw5, 8'h00); rx(dw6, 8'h00); rx(dw7, 8'h00);
end
endtask

// ── Send readback zeros + commands ────────────────────────────
task send_gpu_tail;
    input integer rb_count; // READBACK count (TX DWs)
begin
    // Zero readback area: LOAD_DMEM addr=32, count=4
    rx(cmd(4'h2, 12'h020, 16'd4, 32'h0), 8'h00);
    rx(64'h0, 8'h00); rx(64'h0, 8'h00);
    rx(64'h0, 8'h00); rx(64'h0, 8'h00);
    // Commands
    rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);             // CPU_START
    rx(cmd(4'h4, 12'h020, rb_count[15:0], 32'h0), 8'h00);    // READBACK
    rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);             // SEND_PKT
    rx_end;
end
endtask

// ── Wait for pkt_proc to activate then return to idle ────────
// Captures all TX output during execution.
task wait_and_capture;
    input integer max_cyc;
    integer c;
begin
    tx_cnt = 0; c = 0;
    // Wait for pkt_proc to start
    while (!u_soc.pp_active && c < max_cyc) begin tick; c = c + 1; end
    // Run until idle, capture TX
    while (u_soc.pp_active && c < max_cyc) begin
        if (out_wr && out_rdy) begin
            tx_data[tx_cnt] = out_data;
            tx_ctrl[tx_cnt] = out_ctrl;
            tx_cnt = tx_cnt + 1;
        end
        tick; c = c + 1;
    end
    if (c >= max_cyc) begin
        $display("    [TIMEOUT] pkt_proc stuck (state=%0d cpu_done=%b halted=%04b running=%b)",
                 u_soc.u_pkt_proc.state, u_soc.cpu_done_w,
                 u_soc.u_cpu_mt.halted, u_soc.u_cpu_mt.running);
        $display("    PCs: %08h %08h %08h %08h",
                 u_soc.u_cpu_mt.pc_thread[0], u_soc.u_cpu_mt.pc_thread[1],
                 u_soc.u_cpu_mt.pc_thread[2], u_soc.u_cpu_mt.pc_thread[3]);
    end
    repeat (5) tick;
end
endtask

// ── Checkers ─────────────────────────────────────────────────
task check64;
    input [63:0] val, exp;
    input [80*8-1:0] name;
begin
    test_id = test_id + 1;
    if (val === exp) begin
        $display("    [PASS] T%0d: %0s = 0x%016h", test_id, name, val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("    [FAIL] T%0d: %0s = 0x%016h, exp 0x%016h", test_id, name, val, exp);
        fail_cnt = fail_cnt + 1;
    end
end
endtask

task check8;
    input [7:0] val, exp;
    input [80*8-1:0] name;
begin
    test_id = test_id + 1;
    if (val === exp) begin
        $display("    [PASS] T%0d: %0s = 0x%02h", test_id, name, val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("    [FAIL] T%0d: %0s = 0x%02h, exp 0x%02h", test_id, name, val, exp);
        fail_cnt = fail_cnt + 1;
    end
end
endtask

task checkN;
    input integer val, exp;
    input [80*8-1:0] name;
begin
    test_id = test_id + 1;
    if (val === exp) begin
        $display("    [PASS] T%0d: %0s = %0d", test_id, name, val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("    [FAIL] T%0d: %0s = %0d, exp %0d", test_id, name, val, exp);
        fail_cnt = fail_cnt + 1;
    end
end
endtask

task settle; begin repeat (15) tick; end endtask

// ═══════════════════════════════════════════════════════════════════
//  Cycle Trace — comprehensive debug
// ═══════════════════════════════════════════════════════════════════
reg trace_en;
initial trace_en = 1;

always @(posedge clk) begin
    if (rst_n && u_soc.pp_active) begin
        cycle_cnt <= cycle_cnt + 1;
        if (trace_en) begin
            // ── Event-driven trace: only print when something interesting happens ──

            // CP10 write (MCR execution)
            if (u_soc.cp_wen)
                $display("  [C%04d] CP10 WR: CR%0d <= 0x%08h", cycle_cnt, u_soc.cp_reg, u_soc.cp_wr_data);

            // DMA start/finish
            if (u_soc.u_dma.state == 0 && u_soc.dma_start)
                $display("  [C%04d] DMA START: dir=%b tgt=%b bank=%0d len=%0d burst=%b src=%0d dst=%0d",
                         cycle_cnt, u_soc.u_dma.dma_dir, u_soc.u_dma.dma_tgt,
                         u_soc.u_dma.dma_bank, u_soc.u_dma.dma_xfer_len,
                         u_soc.u_dma.dma_burst_all,
                         u_soc.u_dma.dma_src_addr, u_soc.u_dma.dma_dst_addr);
            if (u_soc.u_dma.state == 11) // S_DONE
                $display("  [C%04d] DMA DONE", cycle_cnt);

            // DMA bank switch
            if (u_soc.u_dma.state == 10) // S_BANK_NEXT
                $display("  [C%04d] DMA BANK_NEXT: done=%0d cur=%0d gpu_sel=%0d",
                         cycle_cnt, u_soc.u_dma.banks_done, u_soc.u_dma.cur_bank,
                         u_soc.u_dma.gpu_dmem_sel);

            // DMA writes to GPU DMEM
            if (u_soc.dma_gpu_dmem_we)
                $display("  [C%04d] DMA→GPU_DMEM: sel=%0d addr=%0d data=%04h",
                         cycle_cnt, u_soc.dma_gpu_dmem_sel,
                         u_soc.dma_gpu_dmem_addr, u_soc.dma_gpu_dmem_din);

            // DMA writes to GPU IMEM
            if (u_soc.dma_gpu_imem_we)
                $display("  [C%04d] DMA→GPU_IMEM: addr=%0d data=%08h",
                         cycle_cnt, u_soc.dma_gpu_imem_addr, u_soc.dma_gpu_imem_din);

            // DMA writes to CPU DMEM (D_PACK)
            if (u_soc.u_dma.cpu_dmem_we)
                $display("  [C%04d] DMA→CPU_DMEM: addr=%0d data=%08h",
                         cycle_cnt, u_soc.u_dma.cpu_dmem_addr, u_soc.u_dma.cpu_dmem_din);

            // GPU kernel start/done transitions
            if (u_soc.gpu_kernel_start_w)
                $display("  [C%04d] GPU KERNEL_START: entry_pc=%0d mask=%04b rst=%b",
                         cycle_cnt, u_soc.gpu_entry_pc_w,
                         u_soc.gpu_thread_mask_w, u_soc.gpu_rst_gated);
            if (u_soc.gpu_kernel_done_w && u_soc.gpu_active_r)
                $display("  [C%04d] GPU KERNEL_DONE", cycle_cnt);

            // GPU ret detected
            if (u_soc.u_sm_core.ret_detected)
                $display("  [C%04d] GPU RET_DETECTED: amask=%04b",
                         cycle_cnt, u_soc.u_sm_core.active_mask);

            // GPU fetch unit internal — print every cycle when GPU is active
            if (u_soc.gpu_active_r)
                $display("  [C%04d] GPU: run=%b pc=%0d fv=%b ir=%08h amask=%04b front_stall=%b sp_stall=%b ret=%b done=%b | op=%02h de_valid=%b",
                         cycle_cnt,
                         u_soc.u_sm_core.fu_running,
                         u_soc.u_sm_core.fu_pc_out,
                         u_soc.u_sm_core.fetch_valid,
                         u_soc.u_sm_core.ir_latch,
                         u_soc.u_sm_core.active_mask,
                         u_soc.u_sm_core.front_stall,
                         u_soc.u_sm_core.sp_stall,
                         u_soc.u_sm_core.ret_detected,
                         u_soc.u_sm_core.kernel_done,
                         u_soc.u_sm_core.de_opcode,
                         u_soc.u_sm_core.de_valid);

            // Stall decomposition (only when front_stall=1)
            if (u_soc.gpu_active_r && u_soc.u_sm_core.front_stall)
                $display("         STALL: sb=%b sb_gated=%b ex_busy=%b|%b|%b|%b cvt_busy=%b|%b|%b|%b any_ex=%b tc=%b burst=%b",
                         u_soc.u_sm_core.sb_stall,
                         u_soc.u_sm_core.sb_stall_gated,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.ex_busy,
                         u_soc.u_sm_core.SP_LANE[1].u_sp.ex_busy,
                         u_soc.u_sm_core.SP_LANE[2].u_sp.ex_busy,
                         u_soc.u_sm_core.SP_LANE[3].u_sp.ex_busy,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.cvt_busy,
                         u_soc.u_sm_core.SP_LANE[1].u_sp.cvt_busy,
                         u_soc.u_sm_core.SP_LANE[2].u_sp.cvt_busy,
                         u_soc.u_sm_core.SP_LANE[3].u_sp.cvt_busy,
                         u_soc.u_sm_core.any_ex_busy,
                         u_soc.u_sm_core.tc_busy,
                         u_soc.u_sm_core.burst_busy);

            // SP0 pipeline detail (only when stalled or GPU active with valid)
            if (u_soc.gpu_active_r && u_soc.u_sm_core.front_stall)
                $display("         SP0: id_ex_valid=%b id_ex_op=%02h id_ex_dt=%b id_ex_rD=%0d id_ex_launched=%b | cvt_pipe=%02b cvt_done=%b",
                         u_soc.u_sm_core.SP_LANE[0].u_sp.id_ex_valid,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.id_ex_opcode,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.id_ex_dt,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.id_ex_rD_addr,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.id_ex_launched,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.cvt_pipe_valid,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.cvt_done);

            // SP0 EX/MEM/WB pipeline (only when stalled)
            if (u_soc.gpu_active_r && u_soc.u_sm_core.front_stall)
                $display("         SP0 EX/MEM: valid=%b rD=%0d rf_we=%b | MEM/WB: valid=%b rD=%0d rf_we=%b active=%b | WB: w0_we=%b",
                         u_soc.u_sm_core.SP_LANE[0].u_sp.ex_mem_valid,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.ex_mem_rD_addr,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.ex_mem_rf_we,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.mem_wb_valid,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.mem_wb_rD_addr,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.mem_wb_rf_we,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.mem_wb_active,
                         u_soc.u_sm_core.SP_LANE[0].u_sp.w0_we);

            // Scoreboard pending bits (only when stalled)
            if (u_soc.gpu_active_r && u_soc.u_sm_core.front_stall)
                $display("         SB: pending[0]=%04h pending[1]=%04h pending[2]=%04h pending[3]=%04h | wb_rf_we=%b wb_rD=%0d wb_amask=%04b",
                         u_soc.u_sm_core.u_sb.pending[0],
                         u_soc.u_sm_core.u_sb.pending[1],
                         u_soc.u_sm_core.u_sb.pending[2],
                         u_soc.u_sm_core.u_sb.pending[3],
                         u_soc.u_sm_core.u_sb.wb_rf_we,
                         u_soc.u_sm_core.u_sb.wb_rD_addr,
                         u_soc.u_sm_core.u_sb.wb_active_mask);

            // RR stage (only when stalled)
            if (u_soc.gpu_active_r && u_soc.u_sm_core.front_stall)
                $display("         RR: valid=%b op=%02h rD=%0d rf_we=%b dt=%b",
                         u_soc.u_sm_core.rr_valid,
                         u_soc.u_sm_core.rr_opcode,
                         u_soc.u_sm_core.rr_rD_addr,
                         u_soc.u_sm_core.rr_rf_we,
                         u_soc.u_sm_core.rr_dt);

            // GPU DMEM writes (sm_core stores)
            if (|u_soc.core_dmem_we)
                $display("  [C%04d] GPU_SP→DMEM: we=%04b addr0=%0d din0=%04h addr1=%0d din1=%04h",
                         cycle_cnt, u_soc.core_dmem_we,
                         u_soc.core_dmem_addr[0*10 +: 10],
                         u_soc.core_dmem_din[0*16 +: 16],
                         u_soc.core_dmem_addr[1*10 +: 10],
                         u_soc.core_dmem_din[1*16 +: 16]);

            // CPU DMEM writes (ARM stores via Port A)
            if (u_soc.cpu_dmem_wen)
                $display("  [C%04d] CPU→DMEM: byte_addr=%08h data=%08h",
                         cycle_cnt, u_soc.cpu_dmem_byte_addr, u_soc.cpu_dmem_wdata);

            // CPU halt
            if (u_soc.cpu_done_w && u_soc.u_cpu_mt.running)
                $display("  [C%04d] CPU DONE (halted=%04b)", cycle_cnt, u_soc.u_cpu_mt.halted);
        end
    end
end

// ═══════════════════════════════════════════════════════════════════
//  Waveform + Global Timeout
// ═══════════════════════════════════════════════════════════════════
initial begin $dumpfile("soc_tb.vcd"); $dumpvars(0, soc_tb); end
initial begin #40000000; $display("[TIMEOUT] Global"); $finish; end

// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
//  M A I N   T E S T   S E Q U E N C E
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════

initial begin
    $display("");
    $display("================================================================");
    $display("  SoC Integration Testbench v6.0 (25 tests)");
    $display("  Drives NetFPGA RX/TX only — real CPU+CP10+DMA+GPU inside");
    $display("================================================================");
    $display("");

    rst_n = 0; in_data = 0; in_ctrl = 0; in_wr = 0; out_rdy = 1;
    cycle_cnt = 0;
    repeat (5) tick;
    rst_n = 1;
    repeat (5) tick;

    // ────────────────────────────────────────────────────────────
    //  Test 1: CPU ADD — basic round-trip
    //
    //  IMEM (6 instrs at word 0..5):
    //    [0] E3A00000  MOV R0, #0
    //    [1] E5901000  LDR R1, [R0, #0]  → DMEM byte 0 → word 0
    //    [2] E5902004  LDR R2, [R0, #4]  → DMEM byte 4 → word 1
    //    [3] E0813002  ADD R3, R1, R2
    //    [4] E5803008  STR R3, [R0, #8]  → DMEM byte 8 → word 2
    //    [5] EAFFFFFE  B .
    //
    //  DMEM: word0=10, word1=20, word2=0, word3=0
    //  Expected: word2 = 30
    //
    //  Readback addr=0, count=2:
    //    TX[0] = {DMEM[1], DMEM[0]} = {20, 10}
    //    TX[1] = {DMEM[3], DMEM[2]} = {0, 30}
    // ────────────────────────────────────────────────────────────
    begin
        $display("--- Test 1: CPU ADD ---");
        cycle_cnt = 0;

        // LOAD_IMEM: addr=0, count=3 (3 data words = 6 instrs)
        rx(cmd(4'h1, 12'h000, 16'd3, 32'h0), 8'h04);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);   // {instr[1], instr[0]}
        rx({32'hE0813002, 32'hE5902004}, 8'h00);   // {instr[3], instr[2]}
        rx({ARM_HALT,     32'hE5803008}, 8'h00);   // {instr[5], instr[4]}

        // LOAD_DMEM: addr=0, count=2 (4 DMEM words)
        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx({32'h0000_0014, 32'h0000_000A}, 8'h00); // {word1=20, word0=10}
        rx({32'h0000_0000, 32'h0000_0000}, 8'h00); // {word3=0, word2=0}

        // CPU_START: entry_pc = 0 (byte address)
        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);

        // READBACK: addr=0, count=2
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);

        // SEND_PKT
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T1 TX count");
        check64(tx_data[0], {32'h0000_0014, 32'h0000_000A}, "T1 TX[0] inputs 20,10");
        check64(tx_data[1], {32'h0000_0000, 32'h0000_001E}, "T1 TX[1] result 0,30");
        check8(tx_ctrl[0], 8'h04, "T1 TX[0] ctrl passthrough");
        settle;
        trace_en = 0; // disable trace for subsequent tests
    end

    // ────────────────────────────────────────────────────────────
    //  Test 2: CPU MUL — MAC unit
    //
    //  IMEM (6 instrs):
    //    [0] E3A01007  MOV R1, #7
    //    [1] E3A02006  MOV R2, #6
    //    [2] E0030291  MUL R3, R1, R2   → R3 = 42
    //    [3] E3A00000  MOV R0, #0
    //    [4] E5803000  STR R3, [R0]     → DMEM[0] = 42
    //    [5] EAFFFFFE  B .
    //
    //  Readback addr=0, count=1:
    //    TX[0] = {DMEM[1], DMEM[0]} = {0, 42}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 2: CPU MUL ---");
        cycle_cnt = 0;

        rx(cmd(4'h1, 12'h000, 16'd3, 32'h0), 8'h08);
        rx({32'hE3A02006, 32'hE3A01007}, 8'h00);
        rx({32'hE3A00000, 32'hE0030291}, 8'h00);
        rx({ARM_HALT,     32'hE5803000}, 8'h00);

        // Zero DMEM
        rx(cmd(4'h2, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 1, "T2 TX count");
        check64(tx_data[0], {32'h0, 32'h0000_002A}, "T2 MUL 7*6=42");
        check8(tx_ctrl[0], 8'h08, "T2 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 3: CPU MCR/MRC — CP10 scratch register
    //
    //  IMEM (6 instrs):
    //    [0] E3A0102A  MOV R1, #42
    //    [1] EE081A10  MCR p10,0,R1,CR8  → CR8 = 42
    //    [2] EE182A10  MRC p10,0,R2,CR8  → R2 = 42
    //    [3] E3A00000  MOV R0, #0
    //    [4] E5802000  STR R2, [R0]      → DMEM[0] = 42
    //    [5] EAFFFFFE  B .
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 3: CPU MCR/MRC (CP10 scratch) ---");
        cycle_cnt = 0;

        rx(cmd(4'h1, 12'h000, 16'd3, 32'h0), 8'h02);
        rx({32'hEE081A10, 32'hE3A0102A}, 8'h00);
        rx({32'hE3A00000, 32'hEE182A10}, 8'h00);
        rx({ARM_HALT,     32'hE5802000}, 8'h00);

        rx(cmd(4'h2, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 1, "T3 TX count");
        check64(tx_data[0], {32'h0, 32'h0000_002A}, "T3 MCR/MRC CR8=42");
        check8(tx_ctrl[0], 8'h02, "T3 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 4: CPU conditional branch (CMP + BEQ)
    //
    //  IMEM (10 instrs):
    //    [0] E3A00000  MOV R0, #0
    //    [1] E5901000  LDR R1, [R0]       → R1 = DMEM[0] = 5
    //    [2] E3510005  CMP R1, #5
    //    [3] 0A000001  BEQ → [6]           ; offset=(24-12-8)/4=1
    //    [4] E3A02001  MOV R2, #1         (not-equal path)
    //    [5] EA000000  B → [7]              ; offset=(28-20-8)/4=0
    //    [6] E3A02002  MOV R2, #2         (equal path)
    //    [7] E5802004  STR R2, [R0, #4]   → DMEM[1]
    //    [8] EAFFFFFE  B .
    //    [9] E1A00000  NOP (pad)
    //
    //  DMEM: word0=5, word1=0
    //  Expected: DMEM[1] = 2 (equal path taken)
    //  TX[0] = {DMEM[1], DMEM[0]} = {2, 5}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 4: CPU CMP + BEQ ---");
        cycle_cnt = 0;

        // 10 instrs = 5 data words
        rx(cmd(4'h1, 12'h000, 16'd5, 32'h0), 8'h10);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);   // {LDR; MOV R0,#0}
        rx({32'h0A000001, 32'hE3510005}, 8'h00);   // {BEQ+1→[6]; CMP}
        rx({32'hEA000000, 32'hE3A02001}, 8'h00);   // {B+0→[7]; MOV R2,#1}
        rx({32'hE5802004, 32'hE3A02002}, 8'h00);   // {STR; MOV R2,#2}
        rx({ARM_NOP,      ARM_HALT},     8'h00);   // {NOP; B .}

        // DMEM: word0=5
        rx(cmd(4'h2, 12'h000, 16'd1, 32'h0), 8'h00);
        rx({32'h0, 32'h0000_0005}, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 1, "T4 TX count");
        check64(tx_data[0], {32'h0000_0002, 32'h0000_0005}, "T4 BEQ taken, R2=2");
        check8(tx_ctrl[0], 8'h10, "T4 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 5: Full GPU round-trip via CPU + DMA + CP10
    //
    //  CPU DMEM layout (loaded by pkt_proc):
    //    Words 0..7: GPU IMEM program (8 × 32-bit instructions)
    //    Words 32..35: Readback target (zeroed, DMA fills after GPU)
    //
    //  GPU program (MOVI+ADD, stores result+zero to DMEM[0..1]):
    //    [0] 20100064  MOVI R1, 100
    //    [1] 202000C8  MOVI R2, 200
    //    [2] 30312000  ADD  R3, R1, R2     → R3 = 300
    //    [3] 20400000  MOVI R4, 0          (base addr)
    //    [4] 08340000  ST   R3, [R4+0]     → DMEM[0] = 300 (0x012C)
    //    [5] 20500000  MOVI R5, 0
    //    [6] 08540001  ST   R5, [R4+1]     → DMEM[1] = 0
    //    [7] C8000000  RET
    //
    //  ARM program (66 instrs) executed by CPU:
    //    Phase 1: DMA CPU_DMEM[0..7] → GPU IMEM[0..7] (D_IMEM, len=8)
    //    Phase 2: Launch GPU kernel (entry_pc=0, mask=0xF)
    //    Phase 3: DMA GPU DMEM → CPU_DMEM[32..35] (D_PACK, burst_all)
    //    Phase 4: B . (halt)
    //
    //  DMA CR3 encoding: {burst_all, auto_inc, bank[1:0], tgt, dir, start}
    //    D_IMEM (CPU→GPU IMEM): tgt=1, dir=0, start=1 → 0b0000101 = 5
    //    D_PACK+burst (GPU→CPU): burst=1,dir=1,start=1 → 0b1000011 = 67
    //
    //  Expected readback:
    //    Each GPU SP stores 300 to DMEM[0], 0 to DMEM[1].
    //    D_PACK packs {DMEM[1],DMEM[0]} = {0x0000,0x012C} = 0x0000012C
    //    per bank → CPU DMEM[32..35] all = 0x0000012C
    //
    //    TX[0] = {CPU_DMEM[33], CPU_DMEM[32]} = {0x012C, 0x012C}
    //    TX[1] = {CPU_DMEM[35], CPU_DMEM[34]} = {0x012C, 0x012C}
    // ────────────────────────────────────────────────────────────
    begin : test5
        integer i;
        $display("\n--- Test 5: Full GPU round-trip (CPU+DMA+GPU) ---");
        cycle_cnt = 0;

        // ── ARM program (66 instrs = 33 data words) ─────────
        // LOAD_IMEM at addr=0, count=33
        rx(cmd(4'h1, 12'h000, 16'd33, 32'h0), 8'h20);

        // Phase 1: DMA CPU→GPU IMEM (7 instrs)
        // [0] MOV R0, #0         [1] MCR p10,R0,CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [2] MCR p10,R0,CR1     [3] MOV R1, #8
        rx({32'hE3A01008, 32'hEE010A10}, 8'h00);
        // [4] MCR p10,R1,CR2     [5] MOV R2, #5
        rx({32'hE3A02005, 32'hEE021A10}, 8'h00);
        // [6] MCR p10,R2,CR3     [7] NOP
        rx({ARM_NOP,      32'hEE032A10}, 8'h00);

        // NOP padding: instrs [8..16] = 9 NOPs → 4 data words + shared
        send_nop_pairs(4);  // covers [8..15]

        // Phase 2: GPU launch (5 instrs)
        // [16] NOP               [17] MCR p10,R0,CR4
        rx({32'hEE040A10, ARM_NOP},      8'h00);
        // [18] MOV R3, #15       [19] MCR p10,R3,CR7
        rx({32'hEE073A10, 32'hE3A0300F}, 8'h00);
        // [20] MOV R4, #1        [21] MCR p10,R4,CR5
        rx({32'hEE054A10, 32'hE3A04001}, 8'h00);

        // NOP padding: instrs [22..46] = 25 NOPs → 12 DWs + 1 shared
        send_nop_pairs(12); // covers [22..45]

        // Phase 3: DMA readback GPU→CPU (7 instrs)
        // [46] NOP               [47] MCR p10,R0,CR0
        rx({32'hEE000A10, ARM_NOP},      8'h00);
        // [48] MOV R5, #32       [49] MCR p10,R5,CR1
        rx({32'hEE015A10, 32'hE3A05020}, 8'h00);
        // [50] MOV R6, #1        [51] MCR p10,R6,CR2
        rx({32'hEE026A10, 32'hE3A06001}, 8'h00);
        // [52] MOV R7, #67       [53] MCR p10,R7,CR3
        rx({32'hEE037A10, 32'hE3A07043}, 8'h00);

        // NOP padding: instrs [54..63] = 10 NOPs → 5 DWs
        send_nop_pairs(5);

        // [64] B .               [65] NOP (pad)
        rx({ARM_NOP, ARM_HALT}, 8'h00);

        // ── GPU program in CPU DMEM (words 0..7) ────────────
        // LOAD_DMEM at addr=0, count=4 (8 words)
        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        // {GPU[1]=MOVI R2,200, GPU[0]=MOVI R1,100}
        rx({32'h202000C8, 32'h20100064}, 8'h00);
        // {GPU[3]=MOVI R4,0,   GPU[2]=ADD R3,R1,R2}
        rx({32'h20400000, 32'h30312000}, 8'h00);
        // {GPU[5]=MOVI R5,0,   GPU[4]=ST R3,[R4+0]}
        rx({32'h20500000, 32'h08340000}, 8'h00);
        // {GPU[7]=RET,          GPU[6]=ST R5,[R4+1]}
        rx({32'hC8000000, 32'h08540001}, 8'h00);

        // ── Zero readback area (words 32..35) ───────────────
        rx(cmd(4'h2, 12'h020, 16'd2, 32'h0), 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);

        // CPU_START
        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);

        // READBACK addr=32, count=2 (4 words)
        rx(cmd(4'h4, 12'h020, 16'd2, 32'h0), 8'h00);

        // SEND_PKT
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T5 TX count");
        // All 4 GPU SPs computed 100+200=300=0x012C. Each bank packs as 0x0000012C.
        check64(tx_data[0], {32'h0000_012C, 32'h0000_012C}, "T5 TX[0] GPU banks 1,0");
        check64(tx_data[1], {32'h0000_012C, 32'h0000_012C}, "T5 TX[1] GPU banks 3,2");
        check8(tx_ctrl[0], 8'h20, "T5 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 6: Back-to-back packets (reuse Test 1 IMEM, new data)
    //
    //  Packet A: LOAD_DMEM(100+200), CPU_START, READBACK, SEND
    //  Packet B: LOAD_DMEM(1000+2000), CPU_START, READBACK, SEND
    //
    //  IMEM still has the ADD program from Test 1.
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 6A: Back-to-back packet A (100+200) ---");
        cycle_cnt = 0;

        // Reload ADD program (IMEM may have been overwritten by T5)
        rx(cmd(4'h1, 12'h000, 16'd3, 32'h0), 8'h04);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);
        rx({32'hE0813002, 32'hE5902004}, 8'h00);
        rx({ARM_HALT,     32'hE5803008}, 8'h00);

        // Data: 100 + 200
        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx({32'h0000_00C8, 32'h0000_0064}, 8'h00); // {200, 100}
        rx({32'h0, 32'h0}, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T6A TX count");
        check64(tx_data[0], {32'h0000_00C8, 32'h0000_0064}, "T6A inputs 200,100");
        check64(tx_data[1], {32'h0, 32'h0000_012C},         "T6A result 0,300");
        check8(tx_ctrl[0], 8'h04, "T6A ctrl");
        settle;

        $display("\n--- Test 6B: Back-to-back packet B (1000+2000) ---");
        cycle_cnt = 0;

        // Same IMEM, new data: 1000 + 2000
        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h40);
        rx({32'h0000_07D0, 32'h0000_03E8}, 8'h00); // {2000, 1000}
        rx({32'h0, 32'h0}, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T6B TX count");
        check64(tx_data[0], {32'h0000_07D0, 32'h0000_03E8}, "T6B inputs 2000,1000");
        check64(tx_data[1], {32'h0, 32'h0000_0BB8},         "T6B result 0,3000");
        check8(tx_ctrl[0], 8'h40, "T6B ctrl changed");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 7: CPU MLA (multiply-accumulate)
    //
    //  IMEM (8 instrs):
    //    [0] E3A01007  MOV R1, #7
    //    [1] E3A02006  MOV R2, #6
    //    [2] E3A0300A  MOV R3, #10
    //    [3] E0243291  MLA R4, R1, R2, R3   → R4 = 7*6+10 = 52
    //    [4] E3A00000  MOV R0, #0
    //    [5] E5804000  STR R4, [R0]         → DMEM[0] = 52
    //    [6] EAFFFFFE  B .
    //    [7] E1A00000  NOP (pad)
    //
    //  Readback addr=0, count=1:
    //    TX[0] = {DMEM[1]=0, DMEM[0]=52}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 7: CPU MLA (multiply-accumulate) ---");
        cycle_cnt = 0;

        rx(cmd(4'h1, 12'h000, 16'd4, 32'h0), 8'h04);
        rx({32'hE3A02006, 32'hE3A01007}, 8'h00);   // {MOV R2,#6; MOV R1,#7}
        rx({32'hE0243291, 32'hE3A0300A}, 8'h00);   // {MLA R4,R1,R2,R3; MOV R3,#10}
        rx({32'hE5804000, 32'hE3A00000}, 8'h00);   // {STR R4,[R0]; MOV R0,#0}
        rx({ARM_NOP,      ARM_HALT},     8'h00);   // {NOP; B .}

        rx(cmd(4'h2, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 1, "T7 TX count");
        check64(tx_data[0], {32'h0, 32'h0000_0034}, "T7 MLA 7*6+10=52");
        check8(tx_ctrl[0], 8'h04, "T7 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 8: CPU loop — sum array of 4 elements
    //
    //  IMEM (14 instrs):
    //    [0]  E3A00000   MOV R0, #0       ; base ptr (byte 0)
    //    [1]  E3A01000   MOV R1, #0       ; accumulator
    //    [2]  E3A02000   MOV R2, #0       ; index i (byte offset)
    //    [3]  E3A03010   MOV R3, #16      ; limit (4 words × 4 bytes)
    //    [4]  E1520003   CMP R2, R3       ; loop test
    //    [5]  AA000003   BGE → [10]       ; offset=(40-20-8)/4=3
    //    [6]  E7904002   LDR R4, [R0, R2] ; load DMEM[base+i]
    //    [7]  E0811004   ADD R1, R1, R4   ; acc += val
    //    [8]  E2822004   ADD R2, R2, #4   ; i += 4 bytes
    //    [9]  EAFFFFF9   B → [4]          ; offset=(16-36-8)/4=-7
    //    [10] E5801010   STR R1, [R0, #16]; DMEM[4] = sum
    //    [11] EAFFFFFE   B .
    //    [12] E1A00000   NOP (pad)
    //    [13] E1A00000   NOP (pad)
    //
    //  DMEM: [0]=10, [1]=20, [2]=30, [3]=40, [4]=0
    //  Expected: DMEM[4] = 100
    //
    //  Readback addr=0, count=3:
    //    TX[0] = {DMEM[1]=20, DMEM[0]=10}
    //    TX[1] = {DMEM[3]=40, DMEM[2]=30}
    //    TX[2] = {DMEM[5]=0,  DMEM[4]=100}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 8: CPU loop (sum 4 elements) ---");
        cycle_cnt = 0;

        // 14 instrs = 7 data words
        rx(cmd(4'h1, 12'h000, 16'd7, 32'h0), 8'h04);
        rx({32'hE3A01000, 32'hE3A00000}, 8'h00);   // {MOV R1,#0; MOV R0,#0}
        rx({32'hE3A03010, 32'hE3A02000}, 8'h00);   // {MOV R3,#16; MOV R2,#0}
        rx({32'hAA000003, 32'hE1520003}, 8'h00);   // {BGE+3→[10]; CMP R2,R3}
        rx({32'hE0811004, 32'hE7904002}, 8'h00);   // {ADD R1,R1,R4; LDR R4,[R0,R2]}
        rx({32'hEAFFFFF9, 32'hE2822004}, 8'h00);   // {B-7→[4]; ADD R2,R2,#4}
        rx({ARM_HALT,     32'hE5801010}, 8'h00);   // {B .; STR R1,[R0,#16]}
        rx({ARM_NOP,      ARM_NOP},      8'h00);   // {pad; pad}

        // DMEM: words 0-5 (3 data words)
        rx(cmd(4'h2, 12'h000, 16'd3, 32'h0), 8'h00);
        rx({32'h0000_0014, 32'h0000_000A}, 8'h00); // {word1=20, word0=10}
        rx({32'h0000_0028, 32'h0000_001E}, 8'h00); // {word3=40, word2=30}
        rx({32'h0000_0000, 32'h0000_0000}, 8'h00); // {word5=0,  word4=0}

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd3, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 3, "T8 TX count");
        check64(tx_data[0], {32'h0000_0014, 32'h0000_000A}, "T8 TX[0] 20,10");
        check64(tx_data[1], {32'h0000_0028, 32'h0000_001E}, "T8 TX[1] 40,30");
        check64(tx_data[2], {32'h0000_0000, 32'h0000_0064}, "T8 TX[2] sum=100");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 9: Multiple CP10 register write/readback
    //
    //  CPU writes CR0, CR1, CR4, CR8, then reads all back via MRC
    //  and stores to DMEM[0..3].
    //
    //  IMEM (16 instrs):
    //    [0]  E3A01C05   MOV R1, #0x500
    //    [1]  EE001A10   MCR p10,0,R1,CR0       ; CR0=0x500
    //    [2]  E3A02C06   MOV R2, #0x600
    //    [3]  EE012A10   MCR p10,0,R2,CR1       ; CR1=0x600
    //    [4]  E3A030FF   MOV R3, #0xFF
    //    [5]  EE043A10   MCR p10,0,R3,CR4       ; CR4=0xFF
    //    [6]  E3A04C01   MOV R4, #0x100
    //    [7]  EE084A10   MCR p10,0,R4,CR8       ; CR8=0x100
    //    [8]  EE105A10   MRC p10,0,R5,CR0       ; R5=CR0=0x500
    //    [9]  EE116A10   MRC p10,0,R6,CR1       ; R6=CR1=0x600
    //    [10] EE147A10   MRC p10,0,R7,CR4       ; R7=CR4=0xFF
    //    [11] EE188A10   MRC p10,0,R8,CR8       ; R8=CR8=0x100
    //    [12] E3A00000   MOV R0, #0
    //    [13] E5805000   STR R5, [R0, #0]
    //    [14] E5806004   STR R6, [R0, #4]
    //    [15] E5807008   STR R7, [R0, #8]
    //    [16] E580800C   STR R8, [R0, #12]
    //    [17] EAFFFFFE   B .
    //
    //  Readback addr=0, count=2:
    //    TX[0] = {DMEM[1]=0x600, DMEM[0]=0x500}
    //    TX[1] = {DMEM[3]=0x100, DMEM[2]=0xFF}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 9: Multiple CP10 registers ---");
        cycle_cnt = 0;

        // 18 instrs = 9 data words
        rx(cmd(4'h1, 12'h000, 16'd9, 32'h0), 8'h02);
        rx({32'hEE001A10, 32'hE3A01C05}, 8'h00);   // {MCR CR0; MOV R1,#0x500}
        rx({32'hEE012A10, 32'hE3A02C06}, 8'h00);   // {MCR CR1; MOV R2,#0x600}
        rx({32'hEE043A10, 32'hE3A030FF}, 8'h00);   // {MCR CR4; MOV R3,#0xFF}
        rx({32'hEE084A10, 32'hE3A04C01}, 8'h00);   // {MCR CR8; MOV R4,#0x100}
        rx({32'hEE116A10, 32'hEE105A10}, 8'h00);   // {MRC CR1; MRC CR0}
        rx({32'hEE188A10, 32'hEE147A10}, 8'h00);   // {MRC CR8; MRC CR4}
        rx({32'hE5805000, 32'hE3A00000}, 8'h00);   // {STR R5; MOV R0,#0}
        rx({32'hE5807008, 32'hE5806004}, 8'h00);   // {STR R7; STR R6}
        rx({ARM_HALT,     32'hE580800C}, 8'h00);   // {B .; STR R8}

        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T9 TX count");
        check64(tx_data[0], {32'h0000_0600, 32'h0000_0500}, "T9 TX[0] CR0=0x500, CR1=0x600");
        check64(tx_data[1], {32'h0000_0100, 32'h0000_00FF}, "T9 TX[1] CR4=0xFF, CR8=0x100");
        check8(tx_ctrl[0], 8'h02, "T9 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 10: NOP commands are properly skipped
    //
    //  Packet: NOP, NOP, LOAD_DMEM(1 word), NOP, CPU_START,
    //          READBACK, SEND
    //
    //  Uses Test 1 ADD program (still in IMEM from T6A reload).
    //  DMEM: word0=7, word1=3 → result=10 in word2
    //  Readback addr=0, count=2:
    //    TX[0] = {3, 7}
    //    TX[1] = {0, 10}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 10: NOP command skip ---");
        cycle_cnt = 0;

        // Reload IMEM (may have been overwritten)
        rx(cmd(4'h1, 12'h000, 16'd3, 32'h0), 8'h08);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);
        rx({32'hE0813002, 32'hE5902004}, 8'h00);
        rx({ARM_HALT,     32'hE5803008}, 8'h00);

        // NOP, NOP
        rx(cmd(4'h0, 12'h0, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h0, 12'h0, 16'd0, 32'h0), 8'h00);

        // LOAD_DMEM: {3, 7} + {0, 0}
        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx({32'h0000_0003, 32'h0000_0007}, 8'h00);
        rx(64'h0, 8'h00);

        // NOP
        rx(cmd(4'h0, 12'h0, 16'd0, 32'h0), 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T10 TX count");
        check64(tx_data[0], {32'h0000_0003, 32'h0000_0007}, "T10 TX[0] inputs 3,7");
        check64(tx_data[1], {32'h0, 32'h0000_000A},         "T10 TX[1] sum=10");
        check8(tx_ctrl[0], 8'h08, "T10 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 11: Large DMEM load + readback (16 words)
    //
    //  Loads 16 DMEM words (8 data words), runs CPU that sums
    //  word0+word1 and stores to word8, reads back 5 words.
    //
    //  IMEM (8 instrs):
    //    [0] E3A00000  MOV R0, #0
    //    [1] E5901000  LDR R1, [R0, #0]   ; word0
    //    [2] E5902004  LDR R2, [R0, #4]   ; word1
    //    [3] E0813002  ADD R3, R1, R2
    //    [4] E5803020  STR R3, [R0, #32]  ; word8
    //    [5] EAFFFFFE  B .
    //    [6] NOP
    //    [7] NOP
    //
    //  DMEM: word0=0x1000, word1=0x2000, words2..7=0xDEAD (canary)
    //        words8..15=0 (cleared)
    //
    //  Readback addr=0, count=5 (10 words):
    //    TX[0] = {word1, word0} = {0x2000, 0x1000}
    //    TX[1] = {word3, word2} = {canary, canary}
    //    TX[2] = {word5, word4} = {canary, canary}
    //    TX[3] = {word7, word6} = {canary, canary}
    //    TX[4] = {word9=0, word8=0x3000}
    // ────────────────────────────────────────────────────────────
    begin : test11
        integer i;
        $display("\n--- Test 11: Large DMEM load + readback ---");
        cycle_cnt = 0;

        // IMEM: 8 instrs = 4 data words
        rx(cmd(4'h1, 12'h000, 16'd4, 32'h0), 8'h01);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);
        rx({32'hE0813002, 32'hE5902004}, 8'h00);
        rx({ARM_HALT,     32'hE5803020}, 8'h00);
        rx({ARM_NOP,      ARM_NOP},      8'h00);

        // DMEM: 16 words = 8 data words at addr=0
        rx(cmd(4'h2, 12'h000, 16'd8, 32'h0), 8'h00);
        rx({32'h0000_2000, 32'h0000_1000}, 8'h00); // word0=0x1000, word1=0x2000
        rx({32'h0000_DEAD, 32'h0000_DEAD}, 8'h00); // canary
        rx({32'h0000_DEAD, 32'h0000_DEAD}, 8'h00);
        rx({32'h0000_DEAD, 32'h0000_DEAD}, 8'h00);
        rx(64'h0, 8'h00);                           // word8,9 = 0
        rx(64'h0, 8'h00);                           // word10,11 = 0
        rx(64'h0, 8'h00);                           // word12,13 = 0
        rx(64'h0, 8'h00);                           // word14,15 = 0

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd5, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 5, "T11 TX count");
        check64(tx_data[0], {32'h0000_2000, 32'h0000_1000}, "T11 TX[0] inputs");
        check64(tx_data[1], {32'h0000_DEAD, 32'h0000_DEAD}, "T11 TX[1] canary");
        check64(tx_data[2], {32'h0000_DEAD, 32'h0000_DEAD}, "T11 TX[2] canary");
        check64(tx_data[3], {32'h0000_DEAD, 32'h0000_DEAD}, "T11 TX[3] canary");
        check64(tx_data[4], {32'h0, 32'h0000_3000},         "T11 TX[4] sum=0x3000");
        check8(tx_ctrl[0], 8'h01, "T11 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 12: CPU not-equal path (CMP + BNE complement of T4)
    //
    //  Same program layout as T4 but DMEM[0] = 99 (not 5).
    //  CMP R1,#5 → NE → should NOT take BEQ → R2 = 1.
    //
    //  IMEM (10 instrs, same as T4):
    //    [0] MOV R0,#0 [1] LDR R1,[R0]  [2] CMP R1,#5
    //    [3] BEQ → [6] [4] MOV R2,#1    [5] B → [7]
    //    [6] MOV R2,#2 [7] STR R2,[R0,#4]
    //    [8] B .       [9] NOP
    //
    //  DMEM: word0=99, word1=0
    //  Expected: DMEM[1] = 1 (not-equal path)
    //  TX[0] = {1, 99}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 12: CPU CMP + BEQ not-taken (NE path) ---");
        cycle_cnt = 0;

        // Same IMEM as T4
        rx(cmd(4'h1, 12'h000, 16'd5, 32'h0), 8'h10);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);
        rx({32'h0A000001, 32'hE3510005}, 8'h00);
        rx({32'hEA000000, 32'hE3A02001}, 8'h00);
        rx({32'hE5802004, 32'hE3A02002}, 8'h00);
        rx({ARM_NOP,      ARM_HALT},     8'h00);

        // DMEM: word0=99 (NOT 5), word1=0
        rx(cmd(4'h2, 12'h000, 16'd1, 32'h0), 8'h00);
        rx({32'h0, 32'h0000_0063}, 8'h00);  // {0, 99}

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd1, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 1, "T12 TX count");
        check64(tx_data[0], {32'h0000_0001, 32'h0000_0063}, "T12 BEQ not-taken, R2=1");
        check8(tx_ctrl[0], 8'h10, "T12 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 13: GPU LD/ADD/ST with DMA D_UNPACK preload
    //
    //  Full data flow:
    //    1. pkt_proc LOAD_IMEM → ARM IMEM (80 instrs)
    //    2. pkt_proc LOAD_DMEM → CPU DMEM[0..7]   (GPU IMEM instrs)
    //    3. pkt_proc LOAD_DMEM → CPU DMEM[16..23]  (GPU DMEM input, ×4 banks)
    //    4. pkt_proc LOAD_DMEM → CPU DMEM[32..39]  (readback area, zeroed)
    //    5. pkt_proc CPU_START → ARM CPU runs:
    //       Phase 1: MCR DMA D_IMEM: CPU_DMEM[0..7] → GPU IMEM[0..7]
    //       Phase 2: MCR DMA D_UNPACK: CPU_DMEM[16..23] → GPU DMEM (burst_all)
    //       Phase 3: MCR GPU launch (entry_pc=0, mask=0xF)
    //       Phase 4: MCR DMA D_PACK: GPU DMEM → CPU_DMEM[32..39] (burst_all)
    //       Phase 5: B . (halt)
    //    6. pkt_proc READBACK CPU_DMEM[32..39]
    //    7. pkt_proc SEND_PKT
    //
    //  GPU program (8 instrs):
    //    [0] MOVI R0, 0           → 0x20000000
    //    [1] LD R1, [R0+0]       → 0x10100000  ; DMEM[0] = 100
    //    [2] LD R2, [R0+1]       → 0x10200001  ; DMEM[1] = 200
    //    [3] ADD R3, R1, R2      → 0x30312000  ; R3 = 300
    //    [4] ST R3, [R0+2]       → 0x08300002  ; DMEM[2] = 300
    //    [5] MOVI R4, 0          → 0x20400000
    //    [6] ST R4, [R0+3]       → 0x08400003  ; DMEM[3] = 0
    //    [7] RET                 → 0xC8000000
    //
    //  GPU DMEM input per bank (via D_UNPACK 32→16):
    //    CPU word = {high16, low16} → GPU DMEM[n]=low, DMEM[n+1]=high
    //    Word A: {200, 100} = 0x00C80064 → DMEM[0]=100, DMEM[1]=200
    //    Word B: {0, 0}     = 0x00000000 → DMEM[2]=0, DMEM[3]=0
    //    ×4 banks (burst_all): CPU DMEM[16..23]
    //
    //  D_PACK readback (16→32): xfer_len=2, burst_all
    //    Per bank: {DMEM[1],DMEM[0]}, {DMEM[3],DMEM[2]}
    //    Bank 0→CPU[32]: {200,100}=0x00C80064, CPU[33]: {0,300}=0x0000012C
    //    (same all banks)
    //
    //  TX[0] = {CPU[33], CPU[32]} = {0x0000012C, 0x00C80064}
    //  TX[1] = {CPU[35], CPU[34]} = {0x0000012C, 0x00C80064}
    //  TX[2] = {CPU[37], CPU[36]} = {0x0000012C, 0x00C80064}
    //  TX[3] = {CPU[39], CPU[38]} = {0x0000012C, 0x00C80064}
    //
    //  DMA CR3 encoding: {burst_all[6], auto_inc[5], bank[4:3], tgt[2], dir[1], start[0]}
    //    D_IMEM:  tgt=1, dir=0, start=1              = 7'b0000101 = 5
    //    D_UNPACK+burst: burst=1, dir=0, start=1     = 7'b1000001 = 65
    //    D_PACK+burst:   burst=1, dir=1, start=1     = 7'b1000011 = 67
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 13: GPU LD/ADD/ST with D_UNPACK preload ---");
        cycle_cnt = 0;

        // ── ARM IMEM (80 instrs = 40 DWs) ───────────────────
        rx(cmd(4'h1, 12'h000, 16'd40, 32'h0), 8'h04);

        // Phase 1: DMA D_IMEM (CPU_DMEM[0..7] → GPU IMEM)
        // [0] MOV R0,#0  [1] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [2] MCR CR1     [3] MOV R1,#8
        rx({32'hE3A01008, 32'hEE010A10}, 8'h00);
        // [4] MCR CR2     [5] MOV R2,#5
        rx({32'hE3A02005, 32'hEE021A10}, 8'h00);
        // [6] MCR CR3     [7] NOP
        rx({ARM_NOP,      32'hEE032A10}, 8'h00);
        // [8..15] 8 NOPs
        send_nop_pairs(4);

        // Phase 2: DMA D_UNPACK (CPU_DMEM[16..23] → GPU DMEM, burst_all)
        // [16] MOV R0,#16  [17] MCR CR0
        rx({32'hEE000A10, 32'hE3A00010}, 8'h00);
        // [18] MOV R1,#0   [19] MCR CR1
        rx({32'hEE011A10, 32'hE3A01000}, 8'h00);
        // [20] MOV R2,#2   [21] MCR CR2
        rx({32'hEE022A10, 32'hE3A02002}, 8'h00);
        // [22] MOV R3,#65  [23] MCR CR3
        rx({32'hEE033A10, 32'hE3A03041}, 8'h00);
        // [24..33] 10 NOPs
        send_nop_pairs(5);

        // Phase 3: GPU launch
        // [34] MOV R0,#0   [35] MCR CR4
        rx({32'hEE040A10, 32'hE3A00000}, 8'h00);
        // [36] MOV R1,#15  [37] MCR CR7
        rx({32'hEE071A10, 32'hE3A0100F}, 8'h00);
        // [38] MOV R2,#1   [39] MCR CR5
        rx({32'hEE052A10, 32'hE3A02001}, 8'h00);
        // [40..59] 20 NOPs
        send_nop_pairs(10);

        // Phase 4: DMA D_PACK (GPU DMEM → CPU_DMEM[32], burst_all)
        // [60] MOV R0,#0   [61] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [62] MOV R1,#32  [63] MCR CR1
        rx({32'hEE011A10, 32'hE3A01020}, 8'h00);
        // [64] MOV R2,#2   [65] MCR CR2
        rx({32'hEE022A10, 32'hE3A02002}, 8'h00);
        // [66] MOV R3,#67  [67] MCR CR3
        rx({32'hEE033A10, 32'hE3A03043}, 8'h00);
        // [68..77] 10 NOPs
        send_nop_pairs(5);
        // [78] B .  [79] NOP
        rx({ARM_NOP, ARM_HALT}, 8'h00);

        // ── CPU DMEM: GPU IMEM instrs (words 0..7) ──────────
        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        // {GPU[1]=LD R1,[R0+0],  GPU[0]=MOVI R0,0}
        rx({32'h10100000, 32'h20000000}, 8'h00);
        // {GPU[3]=ADD R3,R1,R2,  GPU[2]=LD R2,[R0+1]}
        rx({32'h30312000, 32'h10200001}, 8'h00);
        // {GPU[5]=MOVI R4,0,     GPU[4]=ST R3,[R0+2]}
        rx({32'h20400000, 32'h08300002}, 8'h00);
        // {GPU[7]=RET,           GPU[6]=ST R4,[R0+3]}
        rx({32'hC8000000, 32'h08400003}, 8'h00);

        // ── CPU DMEM: GPU DMEM input data (words 16..23) ─────
        // D_UNPACK burst_all: 4 banks × 2 words each
        // Each pair: {0x00C80064, 0x00000000} → DMEM[0]=100, [1]=200, [2]=0, [3]=0
        rx(cmd(4'h2, 12'h010, 16'd4, 32'h0), 8'h00);
        rx({32'h00000000, 32'h00C80064}, 8'h00);  // bank 0
        rx({32'h00000000, 32'h00C80064}, 8'h00);  // bank 1
        rx({32'h00000000, 32'h00C80064}, 8'h00);  // bank 2
        rx({32'h00000000, 32'h00C80064}, 8'h00);  // bank 3

        // ── CPU DMEM: readback area (words 32..39, zeroed) ───
        rx(cmd(4'h2, 12'h020, 16'd4, 32'h0), 8'h00);
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);

        // ── Commands ─────────────────────────────────────────
        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00); // CPU_START
        rx(cmd(4'h4, 12'h020, 16'd4, 32'h0), 8'h00); // READBACK addr=32, count=4
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00); // SEND_PKT
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T13 TX count");
        // All 4 banks: input {200,100}, result {0,300}
        check64(tx_data[0], {32'h0000012C, 32'h00C80064}, "T13 TX[0] bank0 in+result");
        check64(tx_data[1], {32'h0000012C, 32'h00C80064}, "T13 TX[1] bank1");
        check64(tx_data[2], {32'h0000012C, 32'h00C80064}, "T13 TX[2] bank2");
        check64(tx_data[3], {32'h0000012C, 32'h00C80064}, "T13 TX[3] bank3");
        check8(tx_ctrl[0], 8'h04, "T13 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 14: GPU WMMA.MMA tensor core through full SoC
    //
    //  GPU program (16 instrs):
    //    [0..3]   MOVI R0..R3, 0x3F80  (A = all 1.0 BF16)
    //    [4..7]   MOVI R4..R7, 0x3F80  (B = all 1.0 BF16)
    //    [8..11]  MOVI R8..R11, 0x0000 (C = all 0.0)
    //    [12]     WMMA.MMA R12, R0, R4, R8  → D = A×B+C = 4.0
    //    [13]     MOVI R0, 0               (base addr — R0 free after MMA)
    //    [14]     WMMA.STORE R12, R0, 0    → DMEM[0..3] = [4.0,4.0,4.0,4.0]
    //    [15]     RET
    //
    //  Expected: D[i][j] = sum(A[i][k]*B[k][j]) = 4×(1.0×1.0) = 4.0 = 0x4080
    //  D_PACK per bank: {DMEM[1],DMEM[0]} = {0x4080,0x4080} = 0x40804080
    //                    {DMEM[3],DMEM[2]} = {0x4080,0x4080} = 0x40804080
    //  All 4 TX words = 64'h40804080_40804080
    //
    //  ARM program (72 instrs = 36 DWs):
    //    Phase 1: DMA D_IMEM (16 GPU instrs)
    //    Phase 2: GPU launch + 30 NOP wait (WMMA = 37 cycles)
    //    Phase 3: DMA D_PACK readback (xfer_len=2, burst_all)
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 14: GPU WMMA.MMA tensor core ---");
        cycle_cnt = 0;

        // ── ARM IMEM (72 instrs = 36 DWs) ───────────────────
        rx(cmd(4'h1, 12'h000, 16'd36, 32'h0), 8'h20);

        // Phase 1: DMA D_IMEM (CPU_DMEM[0..15] → GPU IMEM, 16 instrs)
        // [0] MOV R0,#0  [1] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [2] MCR CR1     [3] MOV R1,#16
        rx({32'hE3A01010, 32'hEE010A10}, 8'h00);
        // [4] MCR CR2     [5] MOV R2,#5
        rx({32'hE3A02005, 32'hEE021A10}, 8'h00);
        // [6] MCR CR3     [7] NOP
        rx({ARM_NOP,      32'hEE032A10}, 8'h00);
        // [8..15] 8 NOPs
        send_nop_pairs(4);

        // Phase 2: GPU launch
        // [16] MOV R0,#0  [17] MCR CR4
        rx({32'hEE040A10, 32'hE3A00000}, 8'h00);
        // [18] MOV R1,#15 [19] MCR CR7
        rx({32'hEE071A10, 32'hE3A0100F}, 8'h00);
        // [20] MOV R2,#1  [21] MCR CR5
        rx({32'hEE052A10, 32'hE3A02001}, 8'h00);
        // [22..51] 30 NOPs (WMMA takes ~50 cycles total)
        send_nop_pairs(15);

        // Phase 3: DMA D_PACK readback
        // [52] MOV R0,#0  [53] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [54] MOV R1,#32 [55] MCR CR1
        rx({32'hEE011A10, 32'hE3A01020}, 8'h00);
        // [56] MOV R2,#2  [57] MCR CR2
        rx({32'hEE022A10, 32'hE3A02002}, 8'h00);
        // [58] MOV R3,#67 [59] MCR CR3
        rx({32'hEE033A10, 32'hE3A03043}, 8'h00);
        // [60..69] 10 NOPs
        send_nop_pairs(5);
        // [70] B .  [71] NOP
        rx({ARM_NOP, ARM_HALT}, 8'h00);

        // ── CPU DMEM: GPU IMEM instrs (words 0..15) ─────────
        rx(cmd(4'h2, 12'h000, 16'd8, 32'h0), 8'h00);
        // {GPU[1]=MOVI R1,1.0,  GPU[0]=MOVI R0,1.0}
        rx({32'h20103F80, 32'h20003F80}, 8'h00);
        // {GPU[3]=MOVI R3,1.0,  GPU[2]=MOVI R2,1.0}
        rx({32'h20303F80, 32'h20203F80}, 8'h00);
        // {GPU[5]=MOVI R5,1.0,  GPU[4]=MOVI R4,1.0}
        rx({32'h20503F80, 32'h20403F80}, 8'h00);
        // {GPU[7]=MOVI R7,1.0,  GPU[6]=MOVI R6,1.0}
        rx({32'h20703F80, 32'h20603F80}, 8'h00);
        // {GPU[9]=MOVI R9,0,    GPU[8]=MOVI R8,0}
        rx({32'h20900000, 32'h20800000}, 8'h00);
        // {GPU[11]=MOVI R11,0,  GPU[10]=MOVI R10,0}
        rx({32'h20B00000, 32'h20A00000}, 8'h00);
        // {GPU[13]=MOVI R0,0,  GPU[12]=WMMA.MMA R12,R0,R4,R8}
        rx({32'h20000000, 32'hECC04800}, 8'h00);
        // {GPU[15]=RET,         GPU[14]=WMMA.STORE R12,R0,0}
        rx({32'hC8000000, 32'hFCC00000}, 8'h00);

        // ── CPU DMEM: readback area (words 32..39, zeroed) ───
        rx(cmd(4'h2, 12'h020, 16'd4, 32'h0), 8'h00);
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);

        // ── Commands ─────────────────────────────────────────
        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h020, 16'd4, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        // All elements = 4.0 = 0x4080. Packed: 0x40804080 per word.
        checkN(tx_cnt, 4, "T14 TX count");
        check64(tx_data[0], 64'h40804080_40804080, "T14 TX[0] WMMA D=4.0");
        check64(tx_data[1], 64'h40804080_40804080, "T14 TX[1] WMMA D=4.0");
        check64(tx_data[2], 64'h40804080_40804080, "T14 TX[2] WMMA D=4.0");
        check64(tx_data[3], 64'h40804080_40804080, "T14 TX[3] WMMA D=4.0");
        check8(tx_ctrl[0], 8'h20, "T14 ctrl");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 15: GPU per-thread TID differentiation
    //
    //  GPU program (8 instrs):
    //    [0] MOV.TID R0           → 0x1C000000  ; R0 = lane id (0,1,2,3)
    //    [1] MOVI R1, 100         → 0x20100064
    //    [2] ADD R2, R0, R1       → 0x30201000  ; R2 = TID + 100
    //    [3] MOVI R3, 0           → 0x20300000  ; base addr
    //    [4] ST R2, [R3+0]        → 0x08230000  ; DMEM[0] = TID+100
    //    [5] MOVI R4, 0           → 0x20400000
    //    [6] ST R4, [R3+1]        → 0x08430001  ; DMEM[1] = 0
    //    [7] RET                  → 0xC8000000
    //
    //  D_PACK xfer_len=1, burst_all: 1 CPU word per bank
    //    SP0: {DMEM[1]=0, DMEM[0]=100} = 0x00000064
    //    SP1: {0, 101} = 0x00000065
    //    SP2: {0, 102} = 0x00000066
    //    SP3: {0, 103} = 0x00000067
    //
    //  TX[0] = {CPU[33]=SP1, CPU[32]=SP0} = {0x65, 0x64}
    //  TX[1] = {CPU[35]=SP3, CPU[34]=SP2} = {0x67, 0x66}
    //
    //  ARM program (62 instrs = 31 DWs):
    //    Phase 1: DMA D_IMEM (8 GPU instrs)
    //    Phase 2: GPU launch + 20 NOP wait
    //    Phase 3: DMA D_PACK (xfer_len=1, burst_all)
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 15: GPU per-thread TID ---");
        cycle_cnt = 0;

        // ── ARM IMEM (62 instrs = 31 DWs) ───────────────────
        rx(cmd(4'h1, 12'h000, 16'd31, 32'h0), 8'h08);

        // Phase 1: DMA D_IMEM (CPU_DMEM[0..7] → GPU IMEM, 8 instrs)
        // [0] MOV R0,#0  [1] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [2] MCR CR1     [3] MOV R1,#8
        rx({32'hE3A01008, 32'hEE010A10}, 8'h00);
        // [4] MCR CR2     [5] MOV R2,#5
        rx({32'hE3A02005, 32'hEE021A10}, 8'h00);
        // [6] MCR CR3     [7] NOP
        rx({ARM_NOP,      32'hEE032A10}, 8'h00);
        // [8..15] 8 NOPs
        send_nop_pairs(4);

        // Phase 2: GPU launch
        // [16] MOV R0,#0  [17] MCR CR4
        rx({32'hEE040A10, 32'hE3A00000}, 8'h00);
        // [18] MOV R1,#15 [19] MCR CR7
        rx({32'hEE071A10, 32'hE3A0100F}, 8'h00);
        // [20] MOV R2,#1  [21] MCR CR5
        rx({32'hEE052A10, 32'hE3A02001}, 8'h00);
        // [22..41] 20 NOPs
        send_nop_pairs(10);

        // Phase 3: DMA D_PACK (xfer_len=1, burst_all)
        // [42] MOV R0,#0  [43] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [44] MOV R1,#32 [45] MCR CR1
        rx({32'hEE011A10, 32'hE3A01020}, 8'h00);
        // [46] MOV R2,#1  [47] MCR CR2
        rx({32'hEE022A10, 32'hE3A02001}, 8'h00);
        // [48] MOV R3,#67 [49] MCR CR3
        rx({32'hEE033A10, 32'hE3A03043}, 8'h00);
        // [50..59] 10 NOPs
        send_nop_pairs(5);
        // [60] B .  [61] NOP
        rx({ARM_NOP, ARM_HALT}, 8'h00);

        // ── CPU DMEM: GPU IMEM instrs (words 0..7) ──────────
        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        // {GPU[1]=MOVI R1,100,  GPU[0]=MOV.TID R0}
        rx({32'h20100064, 32'h1C000000}, 8'h00);
        // {GPU[3]=MOVI R3,0,    GPU[2]=ADD R2,R0,R1}
        rx({32'h20300000, 32'h30201000}, 8'h00);
        // {GPU[5]=MOVI R4,0,    GPU[4]=ST R2,[R3+0]}
        rx({32'h20400000, 32'h08230000}, 8'h00);
        // {GPU[7]=RET,          GPU[6]=ST R4,[R3+1]}
        rx({32'hC8000000, 32'h08430001}, 8'h00);

        // ── CPU DMEM: readback area (words 32..35, zeroed) ───
        rx(cmd(4'h2, 12'h020, 16'd2, 32'h0), 8'h00);
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);

        // ── Commands ─────────────────────────────────────────
        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h020, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        // Per-bank D_PACK: SP0→0x64, SP1→0x65, SP2→0x66, SP3→0x67
        checkN(tx_cnt, 2, "T15 TX count");
        check64(tx_data[0], {32'h00000065, 32'h00000064}, "T15 TX[0] SP1=101,SP0=100");
        check64(tx_data[1], {32'h00000067, 32'h00000066}, "T15 TX[1] SP3=103,SP2=102");
        check8(tx_ctrl[0], 8'h08, "T15 ctrl");
        settle;
    end


    // ════════════════════════════════════════════════════════════
    //  GPU Kernel Suite (T16-T21): Full SoC packet flow with LD
    //
    //  GPU programs use LD from DMEM (real hardware data path).
    //  D_UNPACK burst_all preloads all 4 banks — cpu_ptr advances:
    //    Bank k reads CPU[16 + k*dunpack_len .. +dunpack_len-1]
    //  D_PACK burst_all reads back results.
    //
    //  Data format: CPU word = {high16, low16}
    //  D_UNPACK writes: DMEM[2n]=low16(CPU[n]), DMEM[2n+1]=high16(CPU[n])
    // ════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────
    //  Test 16: GPU K1 vec_add int16 (LD 10 + LD 20 = 30)
    //
    //  GPU: MOVI R0,0; LD R1,[0]; LD R2,[1]; ADD R3,R1,R2;
    //       ST R3,[0]; MOVI R4,0; ST R4,[1]; RET
    //
    //  dunpack=1: 1 CPU word per bank → DMEM[0]=A=10, DMEM[1]=B=20
    //    CPU word = {20, 10} = 0x0014000A
    //    4 banks: 4 CPU words at [16..19] → LOAD_DMEM addr=16, count=2
    //  dpack=1: {DMEM[1]=0, DMEM[0]=30} = 0x0000001E all banks
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 16: GPU K1 vec_add int16 (LD) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h10100000, 32'h20000000}, // [1]LD R1,[R0+0];  [0]MOVI R0,0
            {32'h30312000, 32'h10200001}, // [3]ADD R3,R1,R2;  [2]LD R2,[R0+1]
            {32'h20400000, 32'h08300000}, // [5]MOVI R4,0;     [4]ST R3,[R0+0]
            {32'hC8000000, 32'h08400001}, // [7]RET;            [6]ST R4,[R0+1]
            64'h0, 64'h0, 64'h0, 64'h0
        );
        // 4 identical CPU words: LOAD_DMEM count=2 (2 DWs = 4 words)
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h0014000A, 32'h0014000A}, 8'h00); // CPU[17]=bank1, CPU[16]=bank0
        rx({32'h0014000A, 32'h0014000A}, 8'h00); // CPU[19]=bank3, CPU[18]=bank2

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T16 TX count");
        check64(tx_data[0], {32'h0000001E, 32'h0000001E}, "T16 K1 10+20=30");
        check64(tx_data[1], {32'h0000001E, 32'h0000001E}, "T16 all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 17: GPU K2 vec_sub int16 (100 - 30 = 70)
    //  {B=30, A=100} = {0x001E, 0x0064} = 0x001E0064
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 17: GPU K2 vec_sub int16 (LD) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h10100000, 32'h20000000}, // LD R1; MOVI R0,0
            {32'h38312000, 32'h10200001}, // SUB R3,R1,R2; LD R2
            {32'h20400000, 32'h08300000}, // MOVI R4,0; ST R3,[0]
            {32'hC8000000, 32'h08400001}, // RET; ST R4,[1]
            64'h0, 64'h0, 64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h001E0064, 32'h001E0064}, 8'h00);
        rx({32'h001E0064, 32'h001E0064}, 8'h00);

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T17 TX count");
        check64(tx_data[0], {32'h00000046, 32'h00000046}, "T17 K2 100-30=70");
        check64(tx_data[1], {32'h00000046, 32'h00000046}, "T17 all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 18: GPU K3 bf16_mul (2.0 × 3.0 = 6.0)
    //  {B=3.0, A=2.0} = {0x4040, 0x4000} = 0x40404000
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 18: GPU K3 bf16_mul (LD) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h14100000, 32'h20000000}, // LD.f R1; MOVI R0,0
            {32'h44312000, 32'h14200001}, // MUL.f R3,R1,R2; LD.f R2
            {32'h20400000, 32'h0C300000}, // MOVI R4,0; ST.f R3,[0]
            {32'hC8000000, 32'h08400001}, // RET; ST R4,[1]
            64'h0, 64'h0, 64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h40404000, 32'h40404000}, 8'h00);
        rx({32'h40404000, 32'h40404000}, 8'h00);

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T18 TX count");
        check64(tx_data[0], {32'h000040C0, 32'h000040C0}, "T18 K3 2*3=6.0");
        check64(tx_data[1], {32'h000040C0, 32'h000040C0}, "T18 all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 19: GPU K4 bf16_fma (2.0 × 3.0 + 1.0 = 7.0)
    //
    //  dunpack=2: 2 CPU words per bank → DMEM[0..3]
    //    CPU word 0 = {B=3.0, A=2.0} = 0x40404000 → DMEM[0]=2.0, DMEM[1]=3.0
    //    CPU word 1 = {0, C=1.0}     = 0x00003F80 → DMEM[2]=1.0, DMEM[3]=0
    //    4 banks × 2 = 8 CPU words → LOAD_DMEM addr=16, count=4
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 19: GPU K4 bf16_fma (LD) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd2, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h14100000, 32'h20000000}, // [1]LD.f R1,[0]; [0]MOVI R0,0
            {32'h14300002, 32'h14200001}, // [3]LD.f R3,[2]; [2]LD.f R2,[1]
            {32'h0C300000, 32'h4C312000}, // [5]ST.f R3,[0]; [4]FMA.f R3,R1,R2
            {32'hC8000000, 32'h08000001}, // [7]RET;         [6]ST R0,[1]
            64'h0, 64'h0, 64'h0, 64'h0
        );
        // 4 banks × 2 words = 8 CPU words → LOAD_DMEM count=4
        rx(cmd(4'h2, 12'h010, 16'd4, 32'h0), 8'h00);
        rx({32'h00003F80, 32'h40404000}, 8'h00); // bank 0: {0,C}, {B,A}
        rx({32'h00003F80, 32'h40404000}, 8'h00); // bank 1
        rx({32'h00003F80, 32'h40404000}, 8'h00); // bank 2
        rx({32'h00003F80, 32'h40404000}, 8'h00); // bank 3

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T19 TX count");
        check64(tx_data[0], {32'h000040E0, 32'h000040E0}, "T19 K4 2*3+1=7.0");
        check64(tx_data[1], {32'h000040E0, 32'h000040E0}, "T19 all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 20: GPU K5 relu bf16 (per-bank different inputs)
    //
    //  dunpack=1: 1 CPU word per bank (DIFFERENT per bank)
    //    Bank 0: {0, -1.0} = 0x0000BF80 → relu=0
    //    Bank 1: {0, 2.0}  = 0x00004000 → relu=2.0
    //    Bank 2: {0, -3.0} = 0x0000C040 → relu=0
    //    Bank 3: {0, 5.0}  = 0x000040A0 → relu=5.0
    //  D_PACK: {DMEM[1]=0, DMEM[0]=result} per bank
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 20: GPU K5 relu bf16 (per-bank) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h14100000, 32'h20000000}, // LD.f R1; MOVI R0,0
            {32'h54312000, 32'h20200000}, // MAX.f R3,R1,R2; MOVI R2,0
            {32'h08000001, 32'h0C300000}, // ST R0,[1]; ST.f R3,[0]
            {32'hC8000000, 32'h00000000}, // RET; NOP
            64'h0, 64'h0, 64'h0, 64'h0
        );
        // Per-bank data: 4 words at CPU[16..19]
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h00004000, 32'h0000BF80}, 8'h00); // CPU[17]=bank1(2.0), CPU[16]=bank0(-1.0)
        rx({32'h000040A0, 32'h0000C040}, 8'h00); // CPU[19]=bank3(5.0), CPU[18]=bank2(-3.0)

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T20 TX count");
        check64(tx_data[0], {32'h00004000, 32'h00000000}, "T20 relu: bank1=2.0, bank0=0");
        check64(tx_data[1], {32'h000040A0, 32'h00000000}, "T20 relu: bank3=5.0, bank2=0");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 21: GPU K6 WMMA matmul (I × all-2.0 = all-2.0)
    //
    //  dunpack=4: 4 CPU words per bank → DMEM[0..7]
    //    DMEM[0..3]=A row, DMEM[4..7]=B row
    //  dpack=2: 2 packed words per bank → readback=4 (4 TX words)
    //
    //  Identity A per bank, B=all-2.0 → D=all-2.0
    //  D_PACK: {DMEM[1],DMEM[0]}={2.0,2.0}=0x40004000 all banks
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 21: GPU K6 WMMA (I * all-2.0) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h20, 8'd4, 8'd0, 8'd2);
        send_gpu_imem(
            {32'h20100004, 32'h20000000}, // [1]MOVI R1,4;       [0]MOVI R0,0
            {32'hF4810000, 32'hF4400000}, // [3]WMMA.LOAD R8,R1; [2]WMMA.LOAD R4,R0
            {32'h20D00000, 32'h20C00000}, // [5]MOVI R13,0;      [4]MOVI R12,0
            {32'h20F00000, 32'h20E00000}, // [7]MOVI R15,0;      [6]MOVI R14,0
            {32'h20000000, 32'hECC48C00}, // [9]MOVI R0,0;       [8]WMMA.MMA
            {32'hC8000000, 32'hFCC00000}, // [11]RET;             [10]WMMA.STORE
            64'h0, 64'h0                  // pad
        );
        // 4 banks × 4 words = 16 CPU words → LOAD_DMEM addr=16, count=8
        rx(cmd(4'h2, 12'h010, 16'd8, 32'h0), 8'h00);
        // Bank 0: A={1.0,0,0,0}, B={2.0,2.0,2.0,2.0}
        rx({32'h00000000, 32'h00003F80}, 8'h00); // CPU[17]={0,0}; CPU[16]={0,1.0}
        rx({32'h40004000, 32'h40004000}, 8'h00); // CPU[19]={2,2}; CPU[18]={2,2}
        // Bank 1: A={0,1.0,0,0}
        rx({32'h00000000, 32'h3F800000}, 8'h00); // CPU[21]={0,0}; CPU[20]={1.0,0}
        rx({32'h40004000, 32'h40004000}, 8'h00);
        // Bank 2: A={0,0,1.0,0}
        rx({32'h00003F80, 32'h00000000}, 8'h00); // CPU[25]={0,1.0}; CPU[24]={0,0}
        rx({32'h40004000, 32'h40004000}, 8'h00);
        // Bank 3: A={0,0,0,1.0}
        rx({32'h3F800000, 32'h00000000}, 8'h00); // CPU[29]={1.0,0}; CPU[28]={0,0}
        rx({32'h40004000, 32'h40004000}, 8'h00);

        send_gpu_tail(4);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T21 TX count");
        check64(tx_data[0], 64'h40004000_40004000, "T21 WMMA banks 0,1 D=2.0");
        check64(tx_data[1], 64'h40004000_40004000, "T21 WMMA banks 2,3 D=2.0");
        check64(tx_data[2], 64'h40004000_40004000, "T21 WMMA banks 4,5 D=2.0");
        check64(tx_data[3], 64'h40004000_40004000, "T21 WMMA banks 6,7 D=2.0");
        settle;
    end

    // ════════════════════════════════════════════════════════════
    //  Test 22: Full Heterogeneous Compute Pipeline
    //
    //  Complete packet flow:
    //    1. pkt_proc loads ARM IMEM + CPU DMEM (GPU programs + input)
    //    2. CPU pre-processes: adds 5 to each input integer
    //    3. DMA D_IMEM: both GPU kernels → GPU IMEM[0..15]
    //    4. DMA D_UNPACK: pre-processed data → GPU DMEM (per-bank)
    //    5. GPU K1: ADD R,R,R (×2, int16)
    //    6. GPU K2: MOVI+ADD (+10, int16)
    //       (K2 reads K1 results directly from GPU DMEM — no round-trip!)
    //    7. DMA D_PACK: GPU results → CPU DMEM
    //    8. CPU post-processes: adds 7 to each result
    //    9. CPU halts → pkt_proc READBACK → TX
    //
    //  Input:  [10, 20, 30, 40] per bank
    //  Pre +5: [15, 25, 35, 45]
    //  K1 ×2:  [30, 50, 70, 90]
    //  K2 +10: [40, 60, 80, 100]
    //  Post+7: [47, 67, 87, 107]
    //
    //  GPU K1 (IMEM[0..7]): MOVI R0,0; LD R1,[R0]; ADD R2,R1,R1;
    //    ST R2,[R0]; MOVI R3,0; ST R3,[R0+1]; NOP; RET
    //  GPU K2 (IMEM[8..15]): MOVI R0,0; LD R1,[R0]; MOVI R2,10;
    //    ADD R3,R1,R2; ST R3,[R0]; MOVI R4,0; ST R4,[R0+1]; RET
    //
    //  NOTE: Uses pure int16 ops (ADD DT=0, all verified in sm_core_tb K25).
    //  CVT (opcode 0x05) stalls GPU pipeline — separate RTL investigation needed.
    //
    //  CPU DMEM layout:
    //    [0..7]:   GPU K1 program (8 instrs)
    //    [8..15]:  GPU K2 program (8 instrs)
    //    [16..19]: Input data (per bank, 1 word each via D_UNPACK burst_all)
    //    [24..27]: GPU output (D_PACK target)
    //    [32..35]: CPU post-processed output (readback)
    // ════════════════════════════════════════════════════════════
    begin
        $display("\n--- Test 22: Full heterogeneous compute pipeline ---");
        cycle_cnt = 0;
        trace_en = 1; // enable full trace for T22 debug

        // ── ARM IMEM: 134 instrs = 67 DWs ───────────────────

        rx(cmd(4'h1, 12'h000, 16'd67, 32'h0), 8'h04);

        // ── Phase 1: CPU pre-process (add 5 to DMEM[16..19]) ─
        // DMEM word 16 = byte addr 64 = 0x40
        // [0] MOV R0,#64  [1] LDR R1,[R0]
        rx({32'hE5901000, 32'hE3A00040}, 8'h00);
        // [2] ADD R1,R1,#5  [3] STR R1,[R0]
        rx({32'hE5801000, 32'hE2811005}, 8'h00);
        // [4] LDR R1,[R0,#4]  [5] ADD R1,R1,#5
        rx({32'hE2811005, 32'hE5901004}, 8'h00);
        // [6] STR R1,[R0,#4]  [7] LDR R1,[R0,#8]
        rx({32'hE5901008, 32'hE5801004}, 8'h00);
        // [8] ADD R1,R1,#5  [9] STR R1,[R0,#8]
        rx({32'hE5801008, 32'hE2811005}, 8'h00);
        // [10] LDR R1,[R0,#12]  [11] ADD R1,R1,#5
        rx({32'hE2811005, 32'hE590100C}, 8'h00);
        // [12] STR R1,[R0,#12]  [13] NOP
        rx({ARM_NOP, 32'hE580100C}, 8'h00);

        // ── Phase 2: DMA D_IMEM (CPU[0..15] → GPU IMEM) ─────
        // CR0=0(src), CR1=0(dst), CR2=16(len), CR3=5(D_IMEM)
        // [14] MOV R0,#0  [15] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [16] MCR CR1  [17] MOV R1,#16
        rx({32'hE3A01010, 32'hEE010A10}, 8'h00);
        // [18] MCR CR2  [19] MOV R2,#5
        rx({32'hE3A02005, 32'hEE021A10}, 8'h00);
        // [20] MCR CR3  [21] NOP
        rx({ARM_NOP, 32'hEE032A10}, 8'h00);
        send_nop_pairs(5); // [22..31] wait DMA (16 words)

        // ── Phase 3: DMA D_UNPACK (CPU[16..19] → GPU DMEM) ──
        // CR0=16(src), CR1=0(dst), CR2=1(len), CR3=65(burst_all)
        // [32] MOV R0,#16  [33] MCR CR0
        rx({32'hEE000A10, 32'hE3A00010}, 8'h00);
        // [34] MOV R1,#0  [35] MCR CR1
        rx({32'hEE011A10, 32'hE3A01000}, 8'h00);
        // [36] MOV R2,#1  [37] MCR CR2
        rx({32'hEE022A10, 32'hE3A02001}, 8'h00);
        // [38] MOV R3,#65  [39] MCR CR3
        rx({32'hEE033A10, 32'hE3A03041}, 8'h00);
        send_nop_pairs(4); // [40..47] wait DMA

        // ── Phase 4: GPU launch K1 (entry_pc=0) ─────────────
        // [48] MOV R0,#0  [49] MCR CR4
        rx({32'hEE040A10, 32'hE3A00000}, 8'h00);
        // [50] MOV R1,#15  [51] MCR CR7
        rx({32'hEE071A10, 32'hE3A0100F}, 8'h00);
        // [52] MOV R2,#1  [53] MCR CR5 → launch
        rx({32'hEE052A10, 32'hE3A02001}, 8'h00);
        send_nop_pairs(10); // [54..73] wait GPU K1 (8 instrs × 4 threads + pipeline)

        // ── Phase 5: GPU launch K2 (entry_pc=8) ─────────────
        // K2 reads K1 results directly from GPU DMEM — no DMA!
        // [74] MOV R0,#8  [75] MCR CR4
        rx({32'hEE040A10, 32'hE3A00008}, 8'h00);
        // [76] MOV R1,#15  [77] MCR CR7
        rx({32'hEE071A10, 32'hE3A0100F}, 8'h00);
        // [78] MOV R2,#1  [79] MCR CR5 → launch
        rx({32'hEE052A10, 32'hE3A02001}, 8'h00);
        send_nop_pairs(10); // [80..99] wait GPU K2

        // ── Phase 6: DMA D_PACK (GPU → CPU[24..27]) ─────────
        // CR0=0(src), CR1=24(dst), CR2=1(len), CR3=67(burst_all,pack)
        // [100] MOV R0,#0  [101] MCR CR0
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00);
        // [102] MOV R1,#24  [103] MCR CR1
        rx({32'hEE011A10, 32'hE3A01018}, 8'h00);
        // [104] MOV R2,#1  [105] MCR CR2
        rx({32'hEE022A10, 32'hE3A02001}, 8'h00);
        // [106] MOV R3,#67  [107] MCR CR3
        rx({32'hEE033A10, 32'hE3A03043}, 8'h00);
        send_nop_pairs(5); // [108..117] wait DMA

        // ── Phase 7: CPU post-process ────────────────────────
        // Load DMEM[24..27] (byte 96), add 7, store DMEM[32..35] (byte 128)
        // [118] MOV R0,#96  [119] MOV R2,#128
        rx({32'hE3A02080, 32'hE3A00060}, 8'h00);
        // [120] LDR R1,[R0]  [121] ADD R1,R1,#7
        rx({32'hE2811007, 32'hE5901000}, 8'h00);
        // [122] STR R1,[R2]  [123] LDR R1,[R0,#4]
        rx({32'hE5901004, 32'hE5821000}, 8'h00);
        // [124] ADD R1,R1,#7  [125] STR R1,[R2,#4]
        rx({32'hE5821004, 32'hE2811007}, 8'h00);
        // [126] LDR R1,[R0,#8]  [127] ADD R1,R1,#7
        rx({32'hE2811007, 32'hE5901008}, 8'h00);
        // [128] STR R1,[R2,#8]  [129] LDR R1,[R0,#12]
        rx({32'hE590100C, 32'hE5821008}, 8'h00);
        // [130] ADD R1,R1,#7  [131] STR R1,[R2,#12]
        rx({32'hE582100C, 32'hE2811007}, 8'h00);

        // ── Phase 8: Halt ────────────────────────────────────
        // [132] B .  [133] NOP
        rx({ARM_NOP, ARM_HALT}, 8'h00);

        // ── CPU DMEM: GPU K1 program (words 0..7) ────────────
        //   ×2 via int16 ADD to self (no CVT needed)
        //   [0] MOVI R0,0      [1] LD R1,[R0]
        //   [2] ADD R2,R1,R1   [3] ST R2,[R0]
        //   [4] MOVI R3,0      [5] ST R3,[R0+1]
        //   [6] NOP             [7] RET
        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        rx({32'h10100000, 32'h20000000}, 8'h00); // {LD R1;      MOVI R0,0}
        rx({32'h08200000, 32'h30211000}, 8'h00); // {ST R2,[R0]; ADD R2,R1,R1}
        rx({32'h08300001, 32'h20300000}, 8'h00); // {ST R3,[1];  MOVI R3,0}
        rx({32'hC8000000, 32'h00000000}, 8'h00); // {RET;        NOP}

        // ── CPU DMEM: GPU K2 program (words 8..15) ───────────
        //   +10 via int16 MOVI+ADD (no CVT needed)
        //   [8]  MOVI R0,0      [9]  LD R1,[R0]
        //   [10] MOVI R2,10     [11] ADD R3,R1,R2
        //   [12] ST R3,[R0]     [13] MOVI R4,0
        //   [14] ST R4,[R0+1]   [15] RET
        rx(cmd(4'h2, 12'h008, 16'd4, 32'h0), 8'h00);
        rx({32'h10100000, 32'h20000000}, 8'h00); // {LD R1;        MOVI R0,0}
        rx({32'h30312000, 32'h2020000A}, 8'h00); // {ADD R3,R1,R2; MOVI R2,10}
        rx({32'h20400000, 32'h08300000}, 8'h00); // {MOVI R4,0;    ST R3,[R0]}
        rx({32'hC8000000, 32'h08400001}, 8'h00); // {RET;          ST R4,[R0+1]}

        // ── CPU DMEM: Input data (words 16..19) ─────────────
        //   Per-bank: 10, 20, 30, 40 (before CPU pre-processing)
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h00000014, 32'h0000000A}, 8'h00); // {word17=20, word16=10}
        rx({32'h00000028, 32'h0000001E}, 8'h00); // {word19=40, word18=30}

        // ── CPU DMEM: Zero readback targets ──────────────────
        rx(cmd(4'h2, 12'h018, 16'd2, 32'h0), 8'h00); // words 24..27
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);
        rx(cmd(4'h2, 12'h020, 16'd2, 32'h0), 8'h00); // words 32..35
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);

        // ── Commands ─────────────────────────────────────────
        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00); // CPU_START
        rx(cmd(4'h4, 12'h020, 16'd2, 32'h0), 8'h00); // READBACK addr=32, count=2
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00); // SEND_PKT
        rx_end;

        wait_and_capture(20000); // extra cycles for 2 GPU kernels
        trace_en = 0;

        // ── Checkpoint dumps: hierarchical memory read ───────
        $display("");
        $display("  ==== T22 Post-Execution Memory Dump ====");

        // GPU IMEM (verify DMA D_IMEM loaded correctly)
        $display("  GPU IMEM K1 [0..7]:");
        $display("    [0]=%08h [1]=%08h [2]=%08h [3]=%08h",
            u_soc.u_gpu_imem.mem[0], u_soc.u_gpu_imem.mem[1],
            u_soc.u_gpu_imem.mem[2], u_soc.u_gpu_imem.mem[3]);
        $display("    [4]=%08h [5]=%08h [6]=%08h [7]=%08h",
            u_soc.u_gpu_imem.mem[4], u_soc.u_gpu_imem.mem[5],
            u_soc.u_gpu_imem.mem[6], u_soc.u_gpu_imem.mem[7]);
        $display("  GPU IMEM K2 [8..15]:");
        $display("    [8]=%08h [9]=%08h [10]=%08h [11]=%08h",
            u_soc.u_gpu_imem.mem[8], u_soc.u_gpu_imem.mem[9],
            u_soc.u_gpu_imem.mem[10], u_soc.u_gpu_imem.mem[11]);
        $display("    [12]=%08h [13]=%08h [14]=%08h [15]=%08h",
            u_soc.u_gpu_imem.mem[12], u_soc.u_gpu_imem.mem[13],
            u_soc.u_gpu_imem.mem[14], u_soc.u_gpu_imem.mem[15]);

        // CPU DMEM checkpoints (word addresses)
        $display("  CPU DMEM[16..19] (after pre+5):");
        $display("    [16]=%08h [17]=%08h [18]=%08h [19]=%08h",
            u_soc.u_cpu_dmem.mem[16], u_soc.u_cpu_dmem.mem[17],
            u_soc.u_cpu_dmem.mem[18], u_soc.u_cpu_dmem.mem[19]);

        $display("  CPU DMEM[24..27] (after D_PACK):");
        $display("    [24]=%08h [25]=%08h [26]=%08h [27]=%08h",
            u_soc.u_cpu_dmem.mem[24], u_soc.u_cpu_dmem.mem[25],
            u_soc.u_cpu_dmem.mem[26], u_soc.u_cpu_dmem.mem[27]);

        $display("  CPU DMEM[32..35] (final output after post+7):");
        $display("    [32]=%08h [33]=%08h [34]=%08h [35]=%08h",
            u_soc.u_cpu_dmem.mem[32], u_soc.u_cpu_dmem.mem[33],
            u_soc.u_cpu_dmem.mem[34], u_soc.u_cpu_dmem.mem[35]);

        // GPU DMEM bank 0 (final state after both kernels)
        $display("  GPU DMEM bank0 [0..3]: %04h %04h %04h %04h",
            u_soc.GPU_DMEM_BANK[0].u_gpu_dmem.mem[0],
            u_soc.GPU_DMEM_BANK[0].u_gpu_dmem.mem[1],
            u_soc.GPU_DMEM_BANK[0].u_gpu_dmem.mem[2],
            u_soc.GPU_DMEM_BANK[0].u_gpu_dmem.mem[3]);
        $display("  GPU DMEM bank1 [0..3]: %04h %04h %04h %04h",
            u_soc.GPU_DMEM_BANK[1].u_gpu_dmem.mem[0],
            u_soc.GPU_DMEM_BANK[1].u_gpu_dmem.mem[1],
            u_soc.GPU_DMEM_BANK[1].u_gpu_dmem.mem[2],
            u_soc.GPU_DMEM_BANK[1].u_gpu_dmem.mem[3]);
        $display("  GPU DMEM bank2 [0..3]: %04h %04h %04h %04h",
            u_soc.GPU_DMEM_BANK[2].u_gpu_dmem.mem[0],
            u_soc.GPU_DMEM_BANK[2].u_gpu_dmem.mem[1],
            u_soc.GPU_DMEM_BANK[2].u_gpu_dmem.mem[2],
            u_soc.GPU_DMEM_BANK[2].u_gpu_dmem.mem[3]);
        $display("  GPU DMEM bank3 [0..3]: %04h %04h %04h %04h",
            u_soc.GPU_DMEM_BANK[3].u_gpu_dmem.mem[0],
            u_soc.GPU_DMEM_BANK[3].u_gpu_dmem.mem[1],
            u_soc.GPU_DMEM_BANK[3].u_gpu_dmem.mem[2],
            u_soc.GPU_DMEM_BANK[3].u_gpu_dmem.mem[3]);

        // GPU/DMA status (using soc output wires)
        $display("  GPU state: active=%b done=%b rst_gated=%b",
            u_soc.gpu_active_r, u_soc.gpu_kernel_done_w, u_soc.gpu_rst_gated);
        $display("  CP10 outputs: entry_pc=%08h mask=%04b start=%b reset_n=%b",
            u_soc.gpu_entry_pc_w, u_soc.gpu_thread_mask_w,
            u_soc.gpu_kernel_start_w, u_soc.gpu_reset_n_w);
        $display("  DMA: busy=%b src=%08h dst=%08h len=%0d",
            u_soc.dma_busy, u_soc.dma_src_addr, u_soc.dma_dst_addr,
            u_soc.dma_xfer_len);
        $display("  ==== End Dump ====");
        $display("");

        // Expected: [47, 67, 87, 107]
        // TX[0] = {CPU[33]=67, CPU[32]=47} = {0x43, 0x2F}
        // TX[1] = {CPU[35]=107, CPU[34]=87} = {0x6B, 0x57}
        checkN(tx_cnt, 2, "T22 TX count");
        check64(tx_data[0], {32'h00000043, 32'h0000002F}, "T22 full: bank1=67, bank0=47");
        check64(tx_data[1], {32'h0000006B, 32'h00000057}, "T22 full: bank3=107, bank2=87");
        check8(tx_ctrl[0], 8'h04, "T22 ctrl");
        settle;
    end

    // ════════════════════════════════════════════════════════════
    //  CVT Diagnostic Suite (T23-T25)
    // ════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────
    //  Test 23: CVT round-trip with DIFFERENT rD/rA (should pass)
    //
    //  GPU: MOVI R1,15; CVT.i2f R2,R1; CVT.f2i R3,R2; ST R3,[0]
    //  Preload canary 0xBEEF to DMEM[0] via D_UNPACK.
    //  If CVT works: ST overwrites → DMEM[0]=15, D_PACK={0,15}=0xF
    //  If CVT broken: canary survives → D_PACK={0,0xBEEF}
    //
    //  CVT.i2f R2,R1: {00101,1,00,0010,0001,16'h0} = 0x2C210000
    //  CVT.f2i R3,R2: {00101,0,00,0011,0010,16'h0} = 0x28320000
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 23: CVT round-trip (different regs) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h2010000F, 32'h20000000}, // [1]MOVI R1,15;    [0]MOVI R0,0
            {32'h28320000, 32'h2C210000}, // [3]CVT.f2i R3,R2; [2]CVT.i2f R2,R1
            {32'h20400000, 32'h08300000}, // [5]MOVI R4,0;     [4]ST R3,[R0+0]
            {32'hC8000000, 32'h08400001}, // [7]RET;            [6]ST R4,[R0+1]
            64'h0, 64'h0, 64'h0, 64'h0
        );

        // Canary preload: 0xBEEF to DMEM[0] per bank
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h0000BEEF, 32'h0000BEEF}, 8'h00); // bank0, bank1
        rx({32'h0000BEEF, 32'h0000BEEF}, 8'h00); // bank2, bank3

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        // Round-trip: 15 → bf16(15.0) → int(15) = 15 = 0x000F
        checkN(tx_cnt, 2, "T23 TX count");
        check64(tx_data[0], {32'h0000000F, 32'h0000000F}, "T23 CVT round-trip=15");
        check64(tx_data[1], {32'h0000000F, 32'h0000000F}, "T23 all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 24: CVT self-referencing rD==rA (deadlock diagnostic)
    //
    //  GPU: MOVI R1,15; CVT.i2f R1,R1; CVT.f2i R1,R1; ST R1,[0]
    //  Same canary preload.
    //  If works: DMEM[0]=15, D_PACK=0x0000000F
    //  If deadlock: GPU hangs at CVT, ST never fires,
    //    DMEM[0]=0xBEEF (canary), D_PACK=0x0000BEEF
    //
    //  CVT.i2f R1,R1: 0x2C110000 (rD==rA)
    //  CVT.f2i R1,R1: 0x28110000 (rD==rA)
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 24: CVT self-ref rD==rA (deadlock test) ---");
        cycle_cnt = 0;
        trace_en = 1; // trace GPU pipeline to see stall

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h2010000F, 32'h20000000}, // [1]MOVI R1,15;     [0]MOVI R0,0
            {32'h28110000, 32'h2C110000}, // [3]CVT.f2i R1,R1;  [2]CVT.i2f R1,R1 ←deadlock?
            {32'h20200000, 32'h08100000}, // [5]MOVI R2,0;      [4]ST R1,[R0+0]
            {32'hC8000000, 32'h08200001}, // [7]RET;             [6]ST R2,[R0+1]
            64'h0, 64'h0, 64'h0, 64'h0
        );

        // Canary preload
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h0000BEEF, 32'h0000BEEF}, 8'h00);
        rx({32'h0000BEEF, 32'h0000BEEF}, 8'h00);

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);
        trace_en = 0;

        // Dump GPU DMEM to diagnose
        $display("  T24 GPU DMEM bank0[0..1]: %04h %04h",
            u_soc.GPU_DMEM_BANK[0].u_gpu_dmem.mem[0],
            u_soc.GPU_DMEM_BANK[0].u_gpu_dmem.mem[1]);
        $display("  T24 gpu_active=%b done=%b",
            u_soc.gpu_active_r, u_soc.gpu_kernel_done_w);

        checkN(tx_cnt, 2, "T24 TX count");
        // Optimistic: self-ref CVT works → 15
        // If deadlock: will show 0xBEEF (canary)
        check64(tx_data[0], {32'h0000000F, 32'h0000000F}, "T24 CVT self-ref=15 (or 0xBEEF if deadlock)");
        check64(tx_data[1], {32'h0000000F, 32'h0000000F}, "T24 all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 25: Full Heterogeneous Pipeline WITH CVT
    //
    //  Same as T22 but GPU kernels use CVT for type conversion:
    //    K1: LD→CVT.i2f→MUL.f(×2.0)→CVT.f2i→ST
    //    K2: LD→CVT.i2f→ADD.f(+10.0)→CVT.f2i→ST
    //  Uses different rD/rA for CVT (avoids potential deadlock).
    //
    //  Input:  [10, 20, 30, 40]
    //  Pre +5: [15, 25, 35, 45]
    //  K1 CVT×2: [30, 50, 70, 90]
    //  K2 CVT+10: [40, 60, 80, 100]
    //  Post+7: [47, 67, 87, 107]
    //
    //  K1 IMEM[0..7]:
    //    [0] MOVI R0,0      [1] LD R1,[R0]
    //    [2] CVT.i2f R2,R1  [3] MOVI R3,0x4000 (2.0)
    //    [4] MUL.f R4,R2,R3 [5] CVT.f2i R5,R4
    //    [6] ST R5,[R0]     [7] RET
    //
    //  K2 IMEM[8..15]:
    //    [8]  MOVI R0,0      [9]  LD R1,[R0]
    //    [10] CVT.i2f R2,R1  [11] MOVI R3,0x4120 (10.0)
    //    [12] ADD.f R4,R2,R3 [13] CVT.f2i R5,R4
    //    [14] ST R5,[R0]     [15] RET
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 25: Full heterogeneous with CVT ---");
        cycle_cnt = 0;

        // ── ARM IMEM (same structure as T22) ─────────────────
        rx(cmd(4'h1, 12'h000, 16'd61, 32'h0), 8'h04);

        // Phase 1: CPU pre-process (add 5 to DMEM[16..19])
        rx({32'hE5901000, 32'hE3A00040}, 8'h00); // [0]MOV R0,#64; [1]LDR R1,[R0]
        rx({32'hE5801000, 32'hE2811005}, 8'h00); // [2]ADD R1,R1,#5; [3]STR R1,[R0]
        rx({32'hE2811005, 32'hE5901004}, 8'h00); // [4]LDR R1,[R0,#4]; [5]ADD
        rx({32'hE5901008, 32'hE5801004}, 8'h00); // [6]STR [R0,#4]; [7]LDR [R0,#8]
        rx({32'hE5801008, 32'hE2811005}, 8'h00); // [8]ADD; [9]STR [R0,#8]
        rx({32'hE2811005, 32'hE590100C}, 8'h00); // [10]LDR [R0,#12]; [11]ADD
        rx({ARM_NOP, 32'hE580100C}, 8'h00);      // [12]STR [R0,#12]; [13]NOP

        // Phase 2: DMA D_IMEM (CPU[0..15] → GPU IMEM, 16 words)
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00); // [14]MOV R0,#0; [15]MCR CR0
        rx({32'hE3A01010, 32'hEE010A10}, 8'h00); // [16]MCR CR1; [17]MOV R1,#16
        rx({32'hE3A02005, 32'hEE021A10}, 8'h00); // [18]MCR CR2; [19]MOV R2,#5
        rx({ARM_NOP, 32'hEE032A10}, 8'h00);      // [20]MCR CR3→DMA; [21]NOP
        send_nop_pairs(5);                        // [22..31]

        // Phase 3: DMA D_UNPACK (CPU[16..19] → GPU DMEM, burst_all)
        rx({32'hEE000A10, 32'hE3A00010}, 8'h00); // [32]MOV R0,#16; [33]MCR CR0
        rx({32'hEE011A10, 32'hE3A01000}, 8'h00); // [34]MOV R1,#0; [35]MCR CR1
        rx({32'hEE022A10, 32'hE3A02001}, 8'h00); // [36]MOV R2,#1; [37]MCR CR2
        rx({32'hEE033A10, 32'hE3A03041}, 8'h00); // [38]MOV R3,#65; [39]MCR CR3→DMA
        send_nop_pairs(4);                        // [40..47]

        // Phase 4: GPU launch K1 (entry_pc=0)
        rx({32'hEE040A10, 32'hE3A00000}, 8'h00); // [48]MOV R0,#0; [49]MCR CR4
        rx({32'hEE071A10, 32'hE3A0100F}, 8'h00); // [50]MOV R1,#15; [51]MCR CR7
        rx({32'hEE052A10, 32'hE3A02001}, 8'h00); // [52]MOV R2,#1; [53]MCR CR5→launch
        send_nop_pairs(7);                        // [54..67]

        // Phase 5: GPU launch K2 (entry_pc=8)
        rx({32'hEE040A10, 32'hE3A00008}, 8'h00); // [68]MOV R0,#8; [69]MCR CR4
        rx({32'hEE071A10, 32'hE3A0100F}, 8'h00); // [70]MOV R1,#15; [71]MCR CR7
        rx({32'hEE052A10, 32'hE3A02001}, 8'h00); // [72]MOV R2,#1; [73]MCR CR5→launch
        send_nop_pairs(7);                        // [74..87]

        // Phase 6: DMA D_PACK (GPU → CPU[24..27], burst_all)
        rx({32'hEE000A10, 32'hE3A00000}, 8'h00); // [88]MOV R0,#0; [89]MCR CR0
        rx({32'hEE011A10, 32'hE3A01018}, 8'h00); // [90]MOV R1,#24; [91]MCR CR1
        rx({32'hEE022A10, 32'hE3A02001}, 8'h00); // [92]MOV R2,#1; [93]MCR CR2
        rx({32'hEE033A10, 32'hE3A03043}, 8'h00); // [94]MOV R3,#67; [95]MCR CR3→DMA
        send_nop_pairs(5);                        // [96..105]

        // Phase 7: CPU post-process (load DMEM[24..27], add 7, store DMEM[32..35])
        rx({32'hE3A02080, 32'hE3A00060}, 8'h00); // [106]MOV R0,#96; [107]MOV R2,#128
        rx({32'hE2811007, 32'hE5901000}, 8'h00); // [108]LDR R1,[R0]; [109]ADD R1,R1,#7
        rx({32'hE5901004, 32'hE5821000}, 8'h00); // [110]STR R1,[R2]; [111]LDR R1,[R0,#4]
        rx({32'hE5821004, 32'hE2811007}, 8'h00); // [112]ADD; [113]STR [R2,#4]
        rx({32'hE2811007, 32'hE5901008}, 8'h00); // [114]LDR [R0,#8]; [115]ADD
        rx({32'hE590100C, 32'hE5821008}, 8'h00); // [116]STR [R2,#8]; [117]LDR [R0,#12]
        rx({32'hE582100C, 32'hE2811007}, 8'h00); // [118]ADD; [119]STR [R2,#12]
        rx({ARM_NOP, ARM_HALT}, 8'h00);           // [120]B .; [121]NOP

        // ── CPU DMEM: GPU K1 CVT program (words 0..7) ───────
        //   [0] MOVI R0,0       [1] LD R1,[R0]
        //   [2] CVT.i2f R2,R1   [3] MOVI R3,0x4000 (2.0)
        //   [4] MUL.f R4,R2,R3  [5] CVT.f2i R5,R4
        //   [6] ST R5,[R0]      [7] RET
        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        rx({32'h10100000, 32'h20000000}, 8'h00); // {LD; MOVI R0,0}
        rx({32'h20304000, 32'h2C210000}, 8'h00); // {MOVI R3,2.0; CVT.i2f R2,R1}
        rx({32'h28540000, 32'h44423000}, 8'h00); // {CVT.f2i R5,R4; MUL.f R4,R2,R3}
        rx({32'hC8000000, 32'h08500000}, 8'h00); // {RET; ST R5,[R0]}

        // ── CPU DMEM: GPU K2 CVT program (words 8..15) ──────
        //   [8]  MOVI R0,0       [9]  LD R1,[R0]
        //   [10] CVT.i2f R2,R1   [11] MOVI R3,0x4120 (10.0)
        //   [12] ADD.f R4,R2,R3  [13] CVT.f2i R5,R4
        //   [14] ST R5,[R0]      [15] RET
        rx(cmd(4'h2, 12'h008, 16'd4, 32'h0), 8'h00);
        rx({32'h10100000, 32'h20000000}, 8'h00); // {LD; MOVI R0,0}
        rx({32'h20304120, 32'h2C210000}, 8'h00); // {MOVI R3,10.0; CVT.i2f R2,R1}
        rx({32'h28540000, 32'h34423000}, 8'h00); // {CVT.f2i R5,R4; ADD.f R4,R2,R3}
        rx({32'hC8000000, 32'h08500000}, 8'h00); // {RET; ST R5,[R0]}

        // ── CPU DMEM: Input data (words 16..19) ─────────────
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h00000014, 32'h0000000A}, 8'h00); // {word17=20, word16=10}
        rx({32'h00000028, 32'h0000001E}, 8'h00); // {word19=40, word18=30}

        // ── Zero readback targets ────────────────────────────
        rx(cmd(4'h2, 12'h018, 16'd2, 32'h0), 8'h00); // words 24..27
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);
        rx(cmd(4'h2, 12'h020, 16'd2, 32'h0), 8'h00); // words 32..35
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);

        // ── Commands ─────────────────────────────────────────
        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h020, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(20000);

        checkN(tx_cnt, 2, "T25 TX count");
        check64(tx_data[0], {32'h00000043, 32'h0000002F}, "T25 CVT full: bank1=67, bank0=47");
        check64(tx_data[1], {32'h0000006B, 32'h00000057}, "T25 CVT full: bank3=107, bank2=87");
        check8(tx_ctrl[0], 8'h04, "T25 ctrl");
        settle;
    end

    // ════════════════════════════════════════════════════════════
    //  Network Mode Tests (T26-T27)
    //
    //  These tests validate pkt_proc v2.0 network mode:
    //    - EtherType filtering (0x88B5 = SoC, others = passthrough)
    //    - Header skip (3 words: module hdr + Ethernet)
    //    - Passthrough for non-SoC packets
    //    - Header prepend on TX for SoC packets
    //
    //  Network packet layout in FIFO (as seen by pkt_proc):
    //    Word 0: ctrl=port_mask, data=module_header  (NetFPGA adds this)
    //    Word 1: ctrl=0x00, data={DST_MAC[47:0], SRC_MAC[47:32]}
    //    Word 2: ctrl=0x00, data={SRC_MAC[31:0], EtherType, Pad}
    //    Word 3+: ctrl=0x00, data=payload / commands
    // ════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────
    //  Test 26: Network passthrough (non-SoC EtherType)
    //
    //  Sends a 5-word packet with EtherType=0x0800 (IPv4).
    //  pkt_proc should detect non-SoC EtherType and passthrough
    //  the entire packet unchanged via TX drain.
    //
    //  Input packet:
    //    Word 0: ctrl=0x04, data=module_hdr  (port routing)
    //    Word 1: ctrl=0x00, data={DA, SA_hi} (Ethernet)
    //    Word 2: ctrl=0x00, data={SA_lo, 0x0800, 0x0000} (IPv4)
    //    Word 3: ctrl=0x00, data=0xDEADBEEF_CAFEBABE (payload)
    //    Word 4: ctrl=0x00, data=0x12345678_9ABCDEF0 (payload)
    //
    //  Expected TX: all 5 words unchanged, ctrl[0]=0x04
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 26: Network passthrough (non-SoC EtherType) ---");
        cycle_cnt = 0;

        // Simulated network packet: module header + Ethernet + payload
        rx(64'h0001_0002_0003_0004, 8'h04);  // Word 0: module hdr, ctrl=port mask
        rx(64'hFFFF_FFFF_FFFF_0011, 8'h00);  // Word 1: DA=broadcast, SA_hi
        rx(64'h2233_4455_0800_0000, 8'h00);  // Word 2: SA_lo, EtherType=0x0800 (NOT 0x88B5)
        rx(64'hDEAD_BEEF_CAFE_BABE, 8'h00);  // Word 3: payload
        rx(64'h1234_5678_9ABC_DEF0, 8'h00);  // Word 4: payload
        rx_end;

        wait_and_capture(MAX_CYCLES);

        // Passthrough: entire packet should come back unchanged
        checkN(tx_cnt, 5, "T26 TX count (passthrough)");
        if (tx_cnt >= 5) begin
            check64(tx_data[0], 64'h0001_0002_0003_0004, "T26 TX[0] module hdr");
            check64(tx_data[1], 64'hFFFF_FFFF_FFFF_0011, "T26 TX[1] ETH DA+SA");
            check64(tx_data[2], 64'h2233_4455_0800_0000, "T26 TX[2] ETH SA+Type");
            check64(tx_data[3], 64'hDEAD_BEEF_CAFE_BABE, "T26 TX[3] payload");
            check64(tx_data[4], 64'h1234_5678_9ABC_DEF0, "T26 TX[4] payload");
            check8(tx_ctrl[0], 8'h04, "T26 ctrl passthrough");
        end
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 27: Network SoC packet (CPU ADD with Ethernet headers)
    //
    //  Same computation as Test 1 (10+20=30) but wrapped in
    //  Ethernet frame with EtherType=0x88B5.
    //
    //  Input packet:
    //    Word 0:  ctrl=0x04, data=module_hdr
    //    Word 1:  ctrl=0x00, data={DA, SA_hi}
    //    Word 2:  ctrl=0x00, data={SA_lo, 0x88B5, 0x0000}
    //    Word 3:  ctrl=0x00, data=LOAD_IMEM cmd
    //    Word 4:  ctrl=0x00, data={instr1, instr0}
    //    Word 5:  ctrl=0x00, data={instr3, instr2}
    //    Word 6:  ctrl=0x00, data={instr5, instr4}
    //    Word 7:  ctrl=0x00, data=LOAD_DMEM cmd
    //    Word 8:  ctrl=0x00, data={20, 10}
    //    Word 9:  ctrl=0x00, data={0, 0}
    //    Word 10: ctrl=0x00, data=CPU_START cmd
    //    Word 11: ctrl=0x00, data=READBACK cmd
    //    Word 12: ctrl=0x00, data=SEND_PKT cmd
    //
    //  Expected TX:
    //    TX[0]: hdr_word0 (module hdr), ctrl=0x04
    //    TX[1]: hdr_word1 (ETH DA+SA)
    //    TX[2]: hdr_word2 (ETH SA+Type)
    //    TX[3]: {20, 10}  (readback word 0)
    //    TX[4]: {0, 30}   (readback word 1 = result)
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 27: Network SoC packet (CPU ADD via Ethernet) ---");
        cycle_cnt = 0;

        // Module header
        rx(64'h0001_0002_0005_000D, 8'h04);  // Word 0: module hdr, ctrl=0x04

        // Ethernet header
        rx(64'hFFFF_FFFF_FFFF_0011, 8'h00);  // Word 1: DA=broadcast, SA_hi=0x0011
        rx(64'h2233_4455_88B5_0000, 8'h00);  // Word 2: SA_lo, EtherType=0x88B5, pad=0

        // Commands (same as Test 1, ctrl=0x00 for all)
        rx(cmd(4'h1, 12'h000, 16'd3, 32'h0), 8'h00);          // LOAD_IMEM
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);               // instrs [1,0]
        rx({32'hE0813002, 32'hE5902004}, 8'h00);               // instrs [3,2]
        rx({ARM_HALT,     32'hE5803008}, 8'h00);               // instrs [5,4]

        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);          // LOAD_DMEM
        rx({32'h0000_0014, 32'h0000_000A}, 8'h00);             // {20, 10}
        rx({32'h0000_0000, 32'h0000_0000}, 8'h00);             // {0, 0}

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);          // CPU_START
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);          // READBACK addr=0, count=2
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);          // SEND_PKT
        rx_end;

        wait_and_capture(MAX_CYCLES);

        // TX: 3 header words + 2 readback words = 5 total
        checkN(tx_cnt, 5, "T27 TX count (hdr+data)");
        if (tx_cnt >= 5) begin
            // Header replay
            check64(tx_data[0], 64'h0001_0002_0005_000D, "T27 TX[0] module hdr replay");
            check64(tx_data[1], 64'hFFFF_FFFF_FFFF_0011, "T27 TX[1] ETH DA+SA replay");
            check64(tx_data[2], 64'h2233_4455_88B5_0000, "T27 TX[2] ETH Type replay");
            // Readback data (same as Test 1)
            check64(tx_data[3], {32'h0000_0014, 32'h0000_000A}, "T27 TX[3] inputs 20,10");
            check64(tx_data[4], {32'h0000_0000, 32'h0000_001E}, "T27 TX[4] result 0,30");
            // ctrl on word 0 should be the saved port mask
            check8(tx_ctrl[0], 8'h04, "T27 ctrl from module hdr");
        end
        settle;
    end

    // ════════════════════════════════════════════════════════════
    //  ARMv4T ISA Coverage Suite (T28-T34)
    //
    //  Thorough testing of all data processing instructions,
    //  barrel shifter modes, condition codes, and flag operations.
    //  No multiply-family (MUL/MLA/UMULL/SMULL) — tested in T2/T7.
    //
    //  ARM Data Processing encoding:
    //    cond[31:28] | 00 | I[25] | opcode[24:21] | S[20] | Rn[19:16] | Rd[15:12] | operand2[11:0]
    //
    //  Opcodes: 0000=AND 0001=EOR 0010=SUB 0011=RSB 0100=ADD
    //           0101=ADC 0110=SBC 0111=RSC 1000=TST 1001=TEQ
    //           1010=CMP 1011=CMN 1100=ORR 1101=MOV 1110=BIC 1111=MVN
    //
    //  Barrel shifter operand2 (register):
    //    shift_imm[11:7] | shift_type[6:5] | 0 | Rm[3:0]
    //    type: 00=LSL 01=LSR 10=ASR 11=ROR
    // ════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────
    //  Test 28: SUB, RSB — subtraction variants
    //
    //  IMEM (8 instrs):
    //    [0] E3A00000  MOV R0, #0
    //    [1] E5901000  LDR R1, [R0]       ; R1 = 100
    //    [2] E5902004  LDR R2, [R0, #4]   ; R2 = 30
    //    [3] E0413002  SUB R3, R1, R2     ; 100-30 = 70 = 0x46
    //    [4] E0614002  RSB R4, R1, R2     ; 30-100 = -70 = 0xFFFFFFBA
    //    [5] E5803008  STR R3, [R0, #8]
    //    [6] E580400C  STR R4, [R0, #12]
    //    [7] EAFFFFFE  B .
    //
    //  DMEM: [0]=100, [1]=30, [2]=0, [3]=0
    //  TX[0] = {30, 100}
    //  TX[1] = {0xFFFFFFBA, 0x46}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 28: SUB, RSB ---");
        cycle_cnt = 0;

        // 8 instrs = 4 DWs
        rx(cmd(4'h1, 12'h000, 16'd4, 32'h0), 8'h04);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);   // [1]LDR R1,[R0]; [0]MOV R0,#0
        rx({32'hE0413002, 32'hE5902004}, 8'h00);   // [3]SUB R3,R1,R2; [2]LDR R2,[R0,#4]
        rx({32'hE5803008, 32'hE0614002}, 8'h00);   // [5]STR R3,[R0,#8]; [4]RSB R4,R1,R2
        rx({ARM_HALT,     32'hE580400C}, 8'h00);   // [7]B .; [6]STR R4,[R0,#12]

        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx({32'h0000_001E, 32'h0000_0064}, 8'h00); // {word1=30, word0=100}
        rx({32'h0000_0000, 32'h0000_0000}, 8'h00); // {word3=0, word2=0}

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T28 TX count");
        check64(tx_data[0], {32'h0000_001E, 32'h0000_0064}, "T28 TX[0] inputs 30,100");
        check64(tx_data[1], {32'hFFFF_FFBA, 32'h0000_0046}, "T28 TX[1] RSB=-70, SUB=70");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 29: AND, ORR, EOR, BIC, MVN — bitwise operations
    //
    //  IMEM (14 instrs):
    //    [0]  E3A00000  MOV R0, #0
    //    [1]  E5901000  LDR R1, [R0]       ; 0xFF00FF00
    //    [2]  E5902004  LDR R2, [R0, #4]   ; 0x0F0F0F0F
    //    [3]  E0013002  AND R3, R1, R2     ; 0x0F000F00
    //    [4]  E1814002  ORR R4, R1, R2     ; 0xFF0FFF0F
    //    [5]  E0215002  EOR R5, R1, R2     ; 0xF00FF00F
    //    [6]  E1C16002  BIC R6, R1, R2     ; 0xF000F000
    //    [7]  E1E07002  MVN R7, R2         ; 0xF0F0F0F0
    //    [8]  E5803008  STR R3, [R0, #8]
    //    [9]  E580400C  STR R4, [R0, #12]
    //    [10] E5805010  STR R5, [R0, #16]
    //    [11] E5806014  STR R6, [R0, #20]
    //    [12] E5807018  STR R7, [R0, #24]
    //    [13] EAFFFFFE  B .
    //
    //  DMEM: [0]=0xFF00FF00, [1]=0x0F0F0F0F, [2..7]=0
    //  Readback addr=0, count=4:
    //    TX[0] = {0x0F0F0F0F, 0xFF00FF00} inputs
    //    TX[1] = {0xFF0FFF0F, 0x0F000F00} ORR, AND
    //    TX[2] = {0xF000F000, 0xF00FF00F} BIC, EOR
    //    TX[3] = {0x00000000, 0xF0F0F0F0} pad, MVN
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 29: AND, ORR, EOR, BIC, MVN ---");
        cycle_cnt = 0;

        // 14 instrs = 7 DWs
        rx(cmd(4'h1, 12'h000, 16'd7, 32'h0), 8'h04);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);   // [1]LDR R1; [0]MOV R0
        rx({32'hE0013002, 32'hE5902004}, 8'h00);   // [3]AND R3; [2]LDR R2
        rx({32'hE0215002, 32'hE1814002}, 8'h00);   // [5]EOR R5; [4]ORR R4
        rx({32'hE1E07002, 32'hE1C16002}, 8'h00);   // [7]MVN R7; [6]BIC R6
        rx({32'hE580400C, 32'hE5803008}, 8'h00);   // [9]STR R4,[12]; [8]STR R3,[8]
        rx({32'hE5806014, 32'hE5805010}, 8'h00);   // [11]STR R6,[20]; [10]STR R5,[16]
        rx({ARM_HALT,     32'hE5807018}, 8'h00);   // [13]B .; [12]STR R7,[24]

        // DMEM: 4 DWs = 8 words
        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        rx({32'h0F0F0F0F, 32'hFF00FF00}, 8'h00);   // {word1, word0}
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd4, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T29 TX count");
        check64(tx_data[0], {32'h0F0F0F0F, 32'hFF00FF00}, "T29 TX[0] inputs");
        check64(tx_data[1], {32'hFF0FFF0F, 32'h0F000F00}, "T29 TX[1] ORR, AND");
        check64(tx_data[2], {32'hF000F000, 32'hF00FF00F}, "T29 TX[2] BIC, EOR");
        check64(tx_data[3], {32'h0000_0000, 32'hF0F0F0F0}, "T29 TX[3] pad, MVN");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 30: Barrel shifter — LSL, LSR, ASR, ROR
    //
    //  IMEM (14 instrs):
    //    [0]  E3A00000  MOV R0, #0
    //    [1]  E5901000  LDR R1, [R0]       ; 0x0000000F
    //    [2]  E1A02201  MOV R2, R1, LSL #4 ; 0x000000F0
    //    [3]  E1A03401  MOV R3, R1, LSL #8 ; 0x00000F00
    //    [4]  E5904004  LDR R4, [R0, #4]   ; 0x80000001
    //    [5]  E1A050A4  MOV R5, R4, LSR #1 ; 0x40000000
    //    [6]  E1A060C4  MOV R6, R4, ASR #1 ; 0xC0000000
    //    [7]  E1A07264  MOV R7, R4, ROR #4 ; 0x18000000
    //    [8]  E5802008  STR R2, [R0, #8]
    //    [9]  E580300C  STR R3, [R0, #12]
    //    [10] E5805010  STR R5, [R0, #16]
    //    [11] E5806014  STR R6, [R0, #20]
    //    [12] E5807018  STR R7, [R0, #24]
    //    [13] EAFFFFFE  B .
    //
    //  Barrel shifter operand2 encoding:
    //    LSL #4: shift=00100, type=00, Rm=R1 → 0x201
    //    LSL #8: shift=01000, type=00, Rm=R1 → 0x401
    //    LSR #1: shift=00001, type=01, Rm=R4 → 0x0A4
    //    ASR #1: shift=00001, type=10, Rm=R4 → 0x0C4
    //    ROR #4: shift=00100, type=11, Rm=R4 → 0x264
    //
    //  DMEM: [0]=0xF, [1]=0x80000001, [2..7]=0
    //  Readback addr=0, count=4:
    //    TX[0] = {0x80000001, 0x0000000F} inputs
    //    TX[1] = {0x00000F00, 0x000000F0} LSL#8, LSL#4
    //    TX[2] = {0xC0000000, 0x40000000} ASR#1, LSR#1
    //    TX[3] = {0x00000000, 0x18000000} pad, ROR#4
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 30: Barrel shifter (LSL/LSR/ASR/ROR) ---");
        cycle_cnt = 0;

        // 14 instrs = 7 DWs
        rx(cmd(4'h1, 12'h000, 16'd7, 32'h0), 8'h04);
        rx({32'hE5901000, 32'hE3A00000}, 8'h00);   // [1]LDR R1; [0]MOV R0
        rx({32'hE1A03401, 32'hE1A02201}, 8'h00);   // [3]LSL#8; [2]LSL#4
        rx({32'hE1A050A4, 32'hE5904004}, 8'h00);   // [5]LSR#1; [4]LDR R4
        rx({32'hE1A07264, 32'hE1A060C4}, 8'h00);   // [7]ROR#4; [6]ASR#1
        rx({32'hE580300C, 32'hE5802008}, 8'h00);   // [9]STR R3; [8]STR R2
        rx({32'hE5806014, 32'hE5805010}, 8'h00);   // [11]STR R6; [10]STR R5
        rx({ARM_HALT,     32'hE5807018}, 8'h00);   // [13]B .; [12]STR R7

        // DMEM: 4 DWs
        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        rx({32'h80000001, 32'h0000000F}, 8'h00);   // {word1, word0}
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd4, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T30 TX count");
        check64(tx_data[0], {32'h80000001, 32'h0000000F}, "T30 TX[0] inputs");
        check64(tx_data[1], {32'h00000F00, 32'h000000F0}, "T30 TX[1] LSL#8=0xF00, LSL#4=0xF0");
        check64(tx_data[2], {32'hC0000000, 32'h40000000}, "T30 TX[2] ASR#1=0xC.., LSR#1=0x4..");
        check64(tx_data[3], {32'h00000000, 32'h18000000}, "T30 TX[3] pad, ROR#4=0x18..");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 31: ADC, SBC — carry flag operations
    //
    //  IMEM (16 instrs):
    //    [0]  E3A00000  MOV R0, #0
    //    [1]  E3A01000  MOV R1, #0
    //    [2]  E1E01001  MVN R1, R1          ; R1 = 0xFFFFFFFF
    //    [3]  E3A02001  MOV R2, #1
    //    [4]  E0913002  ADDS R3, R1, R2    ; 0xFFFFFFFF+1=0, C=1, Z=1
    //    [5]  E3A04005  MOV R4, #5
    //    [6]  E3A05003  MOV R5, #3
    //    [7]  E0A46005  ADC R6, R4, R5     ; 5+3+C(=1) = 9
    //    [8]  E0557004  SUBS R7, R5, R4    ; 3-5 = -2 = 0xFFFFFFFE, C=0
    //    [9]  E0C48005  SBC R8, R4, R5     ; 5-3-!C = 5-3-1 = 1
    //    [10] E5803000  STR R3, [R0]       ; DMEM[0] = 0
    //    [11] E5806004  STR R6, [R0, #4]   ; DMEM[1] = 9
    //    [12] E5807008  STR R7, [R0, #8]   ; DMEM[2] = 0xFFFFFFFE
    //    [13] E580800C  STR R8, [R0, #12]  ; DMEM[3] = 1
    //    [14] EAFFFFFE  B .
    //    [15] E1A00000  NOP
    //
    //  ADDS: ADD with S=1 → updates CPSR flags
    //  ADC: Rd = Rn + Rm + C
    //  SUBS: SUB with S=1 → C=0 on borrow
    //  SBC: Rd = Rn - Rm - !C
    //
    //  TX[0] = {9, 0}
    //  TX[1] = {1, 0xFFFFFFFE}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 31: ADC, SBC (carry flag) ---");
        cycle_cnt = 0;

        // 16 instrs = 8 DWs
        rx(cmd(4'h1, 12'h000, 16'd8, 32'h0), 8'h04);
        rx({32'hE3A01000, 32'hE3A00000}, 8'h00);   // [1]MOV R1,#0; [0]MOV R0,#0
        rx({32'hE3A02001, 32'hE1E01001}, 8'h00);   // [3]MOV R2,#1; [2]MVN R1,R1
        rx({32'hE3A04005, 32'hE0913002}, 8'h00);   // [5]MOV R4,#5; [4]ADDS R3,R1,R2
        rx({32'hE0A46005, 32'hE3A05003}, 8'h00);   // [7]ADC R6,R4,R5; [6]MOV R5,#3
        rx({32'hE0C48005, 32'hE0557004}, 8'h00);   // [9]SBC R8,R4,R5; [8]SUBS R7,R5,R4
        rx({32'hE5806004, 32'hE5803000}, 8'h00);   // [11]STR R6,[4]; [10]STR R3,[0]
        rx({32'hE580800C, 32'hE5807008}, 8'h00);   // [13]STR R8,[12]; [12]STR R7,[8]
        rx({ARM_NOP,      ARM_HALT},     8'h00);   // [15]NOP; [14]B .

        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T31 TX count");
        check64(tx_data[0], {32'h0000_0009, 32'h0000_0000}, "T31 TX[0] ADC=9, ADDS=0");
        check64(tx_data[1], {32'h0000_0001, 32'hFFFF_FFFE}, "T31 TX[1] SBC=1, SUBS=-2");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 32: Condition codes — EQ/NE/MI/PL/LT/GE
    //
    //  Tests conditional execution after CMP sets flags.
    //
    //  IMEM (26 instrs):
    //    [0]  E3A00000  MOV R0, #0
    //    [1]  E3A03000  MOV R3, #0       ; init result regs
    //    [2]  E3A04000  MOV R4, #0
    //    [3]  E3A05000  MOV R5, #0
    //    [4]  E3A06000  MOV R6, #0
    //    [5]  E3A07000  MOV R7, #0
    //    [6]  E3A08000  MOV R8, #0
    //    [7]  E3A0100A  MOV R1, #10
    //    [8]  E3A0200A  MOV R2, #10
    //    [9]  E1510002  CMP R1, R2       ; 10==10: Z=1, C=1, N=0
    //    [10] 03A03001  MOVEQ R3, #1    ; Z=1 → execute
    //    [11] 13A04001  MOVNE R4, #1    ; Z=0 → skip
    //    [12] E3A01005  MOV R1, #5
    //    [13] E1510002  CMP R1, R2       ; 5<10: Z=0, N=1, C=0, V=0
    //    [14] 43A05001  MOVMI R5, #1    ; N=1 → execute
    //    [15] 53A06001  MOVPL R6, #1    ; N=0 → skip
    //    [16] B3A07001  MOVLT R7, #1    ; N!=V(1!=0) → execute
    //    [17] A3A08001  MOVGE R8, #1    ; N==V(1==0) → skip
    //    [18] E5803000  STR R3, [R0]     ; DMEM[0] = 1 (EQ)
    //    [19] E5804004  STR R4, [R0, #4] ; DMEM[1] = 0 (NE)
    //    [20] E5805008  STR R5, [R0, #8] ; DMEM[2] = 1 (MI)
    //    [21] E580600C  STR R6, [R0, #12]; DMEM[3] = 0 (PL)
    //    [22] E5807010  STR R7, [R0, #16]; DMEM[4] = 1 (LT)
    //    [23] E5808014  STR R8, [R0, #20]; DMEM[5] = 0 (GE)
    //    [24] EAFFFFFE  B .
    //    [25] E1A00000  NOP
    //
    //  TX[0] = {NE=0, EQ=1}
    //  TX[1] = {PL=0, MI=1}
    //  TX[2] = {GE=0, LT=1}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 32: Condition codes (EQ/NE/MI/PL/LT/GE) ---");
        cycle_cnt = 0;

        // 26 instrs = 13 DWs
        rx(cmd(4'h1, 12'h000, 16'd13, 32'h0), 8'h04);
        rx({32'hE3A03000, 32'hE3A00000}, 8'h00);   // [1]MOV R3,#0; [0]MOV R0,#0
        rx({32'hE3A05000, 32'hE3A04000}, 8'h00);   // [3]MOV R5; [2]MOV R4
        rx({32'hE3A07000, 32'hE3A06000}, 8'h00);   // [5]MOV R7; [4]MOV R6
        rx({32'hE3A0100A, 32'hE3A08000}, 8'h00);   // [7]MOV R1,#10; [6]MOV R8
        rx({32'hE1510002, 32'hE3A0200A}, 8'h00);   // [9]CMP R1,R2; [8]MOV R2,#10
        rx({32'h13A04001, 32'h03A03001}, 8'h00);   // [11]MOVNE R4; [10]MOVEQ R3
        rx({32'hE1510002, 32'hE3A01005}, 8'h00);   // [13]CMP R1,R2; [12]MOV R1,#5
        rx({32'h53A06001, 32'h43A05001}, 8'h00);   // [15]MOVPL R6; [14]MOVMI R5
        rx({32'hA3A08001, 32'hB3A07001}, 8'h00);   // [17]MOVGE R8; [16]MOVLT R7
        rx({32'hE5804004, 32'hE5803000}, 8'h00);   // [19]STR R4,[4]; [18]STR R3,[0]
        rx({32'hE580600C, 32'hE5805008}, 8'h00);   // [21]STR R6,[12]; [20]STR R5,[8]
        rx({32'hE5808014, 32'hE5807010}, 8'h00);   // [23]STR R8,[20]; [22]STR R7,[16]
        rx({ARM_NOP,      ARM_HALT},     8'h00);   // [25]NOP; [24]B .

        // Zero DMEM (3 DWs = 6 words)
        rx(cmd(4'h2, 12'h000, 16'd3, 32'h0), 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd3, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 3, "T32 TX count");
        check64(tx_data[0], {32'h0000_0000, 32'h0000_0001}, "T32 TX[0] NE=0, EQ=1");
        check64(tx_data[1], {32'h0000_0000, 32'h0000_0001}, "T32 TX[1] PL=0, MI=1");
        check64(tx_data[2], {32'h0000_0000, 32'h0000_0001}, "T32 TX[2] GE=0, LT=1");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 33: Shifted register in ALU + immediate operand
    //
    //  IMEM (14 instrs):
    //    [0]  E3A00000  MOV R0, #0
    //    [1]  E3A01005  MOV R1, #5
    //    [2]  E3A02003  MOV R2, #3
    //    [3]  E0813102  ADD R3, R1, R2, LSL #2  ; 5+(3<<2)=5+12=17
    //    [4]  E0414102  SUB R4, R1, R2, LSL #2  ; 5-12=-7=0xFFFFFFF9
    //    [5]  E2415001  SUB R5, R1, #1          ; 5-1=4 (immediate)
    //    [6]  E2616000  RSB R6, R1, #0          ; 0-5=-5=0xFFFFFFFB (negate)
    //    [7]  E0017102  AND R7, R1, R2, LSL #2  ; 5 & 12 = 4
    //    [8]  E5803008  STR R3, [R0, #8]
    //    [9]  E580400C  STR R4, [R0, #12]
    //    [10] E5805010  STR R5, [R0, #16]
    //    [11] E5806014  STR R6, [R0, #20]
    //    [12] E5807018  STR R7, [R0, #24]
    //    [13] EAFFFFFE  B .
    //
    //  operand2 for R2, LSL #2:
    //    shift=00010, type=00(LSL), Rm=R2 → 0x102
    //
    //  No DMEM input — all immediates.
    //  DMEM: 4 DWs zeroed
    //  Readback addr=0, count=4:
    //    TX[0] = {DMEM[1]=0, DMEM[0]=0} (unused)
    //    TX[1] = {0xFFFFFFF9, 0x11}  SUB shifted=-7, ADD shifted=17
    //    TX[2] = {0xFFFFFFFB, 0x04}  RSB negate=-5, SUB imm=4
    //    TX[3] = {0, 0x04}            pad, AND shifted=4
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 33: Shifted register ALU + immediate ---");
        cycle_cnt = 0;

        // 14 instrs = 7 DWs
        rx(cmd(4'h1, 12'h000, 16'd7, 32'h0), 8'h04);
        rx({32'hE3A01005, 32'hE3A00000}, 8'h00);   // [1]MOV R1,#5; [0]MOV R0,#0
        rx({32'hE0813102, 32'hE3A02003}, 8'h00);   // [3]ADD R3,R1,R2 LSL#2; [2]MOV R2,#3
        rx({32'hE2415001, 32'hE0414102}, 8'h00);   // [5]SUB R5,R1,#1; [4]SUB R4,R1,R2 LSL#2
        rx({32'hE0017102, 32'hE2616000}, 8'h00);   // [7]AND R7,R1,R2 LSL#2; [6]RSB R6,R1,#0
        rx({32'hE580400C, 32'hE5803008}, 8'h00);   // [9]STR R4,[12]; [8]STR R3,[8]
        rx({32'hE5806014, 32'hE5805010}, 8'h00);   // [11]STR R6,[20]; [10]STR R5,[16]
        rx({ARM_HALT,     32'hE5807018}, 8'h00);   // [13]B .; [12]STR R7,[24]

        rx(cmd(4'h2, 12'h000, 16'd4, 32'h0), 8'h00);
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);
        rx(64'h0, 8'h00); rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd4, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T33 TX count");
        check64(tx_data[0], {32'h0, 32'h0}, "T33 TX[0] unused DMEM");
        check64(tx_data[1], {32'hFFFFFFF9, 32'h00000011}, "T33 TX[1] SUB_sh=-7, ADD_sh=17");
        check64(tx_data[2], {32'hFFFFFFFB, 32'h00000004}, "T33 TX[2] RSB_neg=-5, SUB_imm=4");
        check64(tx_data[3], {32'h00000000, 32'h00000004}, "T33 TX[3] pad, AND_sh=4");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 34: TST, TEQ, CMN — flag-only operations
    //
    //  IMEM (20 instrs):
    //    [0]  E3A00000  MOV R0, #0
    //    [1]  E3A03000  MOV R3, #0       ; init results
    //    [2]  E3A04000  MOV R4, #0
    //    [3]  E3A05000  MOV R5, #0
    //    --- TST: sets Z based on AND ---
    //    [4]  E3A01003  MOV R1, #3       ; 0x03
    //    [5]  E3A02005  MOV R2, #5       ; 0x05
    //    [6]  E1110002  TST R1, R2       ; 3 AND 5 = 1 → Z=0
    //    [7]  13A03001  MOVNE R3, #1     ; Z=0 → R3=1
    //    --- TEQ: sets Z based on EOR ---
    //    [8]  E3A010FF  MOV R1, #0xFF
    //    [9]  E3A020FF  MOV R2, #0xFF
    //    [10] E1310002  TEQ R1, R2       ; 0xFF XOR 0xFF = 0 → Z=1
    //    [11] 03A04001  MOVEQ R4, #1     ; Z=1 → R4=1
    //    --- CMN: sets flags based on ADD ---
    //    [12] E3A01001  MOV R1, #1
    //    [13] E1E02002  MVN R2, R2       ; R2 = ~0xFF = 0xFFFFFF00
    //    [14] E1710002  CMN R1, R2       ; 1+0xFFFFFF00=0xFFFFFF01 → N=1
    //    [15] 43A05001  MOVMI R5, #1     ; N=1 → R5=1
    //    [16] E5803000  STR R3, [R0]     ; TST result = 1
    //    [17] E5804004  STR R4, [R0, #4] ; TEQ result = 1
    //    [18] E5805008  STR R5, [R0, #8] ; CMN result = 1
    //    [19] EAFFFFFE  B .
    //
    //  Encodings:
    //    TST R1, R2:  E1110002  (op=1000, S=1, Rn=R1, Rd=0)
    //    TEQ R1, R2:  E1310002  (op=1001, S=1, Rn=R1, Rd=0)
    //    CMN R1, R2:  E1710002  (op=1011, S=1, Rn=R1, Rd=0)
    //
    //  TX[0] = {TEQ=1, TST=1}
    //  TX[1] = {0, CMN=1}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 34: TST, TEQ, CMN (flag-only ops) ---");
        cycle_cnt = 0;

        // 20 instrs = 10 DWs
        rx(cmd(4'h1, 12'h000, 16'd10, 32'h0), 8'h04);
        rx({32'hE3A03000, 32'hE3A00000}, 8'h00);   // [1]MOV R3,#0; [0]MOV R0,#0
        rx({32'hE3A05000, 32'hE3A04000}, 8'h00);   // [3]MOV R5,#0; [2]MOV R4,#0
        rx({32'hE3A02005, 32'hE3A01003}, 8'h00);   // [5]MOV R2,#5; [4]MOV R1,#3
        rx({32'h13A03001, 32'hE1110002}, 8'h00);   // [7]MOVNE R3; [6]TST R1,R2
        rx({32'hE3A020FF, 32'hE3A010FF}, 8'h00);   // [9]MOV R2,#0xFF; [8]MOV R1,#0xFF
        rx({32'h03A04001, 32'hE1310002}, 8'h00);   // [11]MOVEQ R4; [10]TEQ R1,R2
        rx({32'hE1E02002, 32'hE3A01001}, 8'h00);   // [13]MVN R2,R2; [12]MOV R1,#1
        rx({32'h43A05001, 32'hE1710002}, 8'h00);   // [15]MOVMI R5; [14]CMN R1,R2
        rx({32'hE5804004, 32'hE5803000}, 8'h00);   // [17]STR R4,[4]; [16]STR R3,[0]
        rx({ARM_HALT,     32'hE5805008}, 8'h00);   // [19]B .; [18]STR R5,[8]

        rx(cmd(4'h2, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(64'h0, 8'h00);
        rx(64'h0, 8'h00);

        rx(cmd(4'h3, 12'h000, 16'd0, 32'h0), 8'h00);
        rx(cmd(4'h4, 12'h000, 16'd2, 32'h0), 8'h00);
        rx(cmd(4'h5, 12'h000, 16'd0, 32'h0), 8'h00);
        rx_end;

        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T34 TX count");
        check64(tx_data[0], {32'h0000_0001, 32'h0000_0001}, "T34 TX[0] TEQ=1, TST=1");
        check64(tx_data[1], {32'h0000_0000, 32'h0000_0001}, "T34 TX[1] pad, CMN=1");
        settle;
    end

// ════════════════════════════════════════════════════════════
    //  GPU ISA Coverage Suite (T35-T42)
    //
    //  Tests all remaining GPU opcodes through the full SoC path
    //  (pkt_proc → LOAD_IMEM → ARM → DMA → GPU → DMA → D_PACK → TX).
    //
    //  Encoding reference (verified against sm_core_tb.v enc_* helpers):
    //    enc_r:       {op[4:0], dt, 2'b00, rd, ra, rb, 12'd0}
    //    enc_i:       {op[4:0], dt, 2'b00, rd, ra, imm16}
    //    enc_movi:    {OP_MOVI=00100, 0, 00, rd, 0000, imm16}
    //    enc_m:       {op, 0, 00, rd, ra, offset16}
    //    enc_m_f:     {op, 1, 00, rd, ra, offset16}
    //    enc_setp:    {OP_SETP=10101, dt, cmp[1:0], pd, ra, rb, 12'd0}
    //    enc_set:     {OP_SET=11010, 0, 00, {00,pd}, 0000, 15'd0, val}
    //    enc_selp:    {OP_SELP=10110, 0, pred_sel[1:0], rd, ra, rb, 12'd0}
    //    enc_mov_tid: {OP_MOV=00011, 1, 00, rd, 0000, 16'd0}  (DT=1→TID)
    //    enc_pbra:    {OP_PBRA=11000, pred_sel[1:0], target[12:0], reconv[11:0]}
    //    enc_bra:     {OP_BRA=10111, target[26:0]}
    //
    //  Hex byte prefixes (bit31:24):
    //    NOP=00  ST=08  ST.f=0C  LD=10  LD.f=14  MOV=18  MOV.TID=1C
    //    MOVI=20  CVT.f2i=28  CVT.i2f=2C  ADD=30  ADD.f=34
    //    SUB=38  SUB.f=3C  MUL=40  MUL.f=44  FMA=48  FMA.f=4C
    //    MAX=50  MAX.f=54  MIN=58  MIN.f=5C  ABS=60  ABS.f=64
    //    NEG=68  NEG.f=6C  AND=70  OR=78  XOR=80  SHL=88  SHR=90
    //    ADDI=98  MULI=A0
    //    SETP+EQ=A8  SETP+NE=A9  SETP+LT=AA  SETP+LE=AB
    //    SELP.P0=B0  SELP.P1=B4
    //    BRA=B8  PBRA.P0=C0  RET=C8  SET=D0
    //    WMMA.MMA=EC  WMMA.LD=F0  WMMA.ST=F8
    // ════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════
    //  GPU ISA Coverage Suite (T35-T42)
    //
    //  Encodings verified against sm_core_tb.v enc_* helpers.
    //
    //  D_PACK TX format depends on nw (DMEM words per bank):
    //    nw=2: 2×16=32 bits/bank → 2 banks per TX word
    //      TX[0]={bank1[1:0], bank0[1:0]}, TX[1]={bank3[1:0], bank2[1:0]}
    //    nw=4: 4×16=64 bits/bank → 1 bank per TX word
    //      TX[0]=bank0{[3],[2],[1],[0]}, TX[1]=bank1, TX[2]=bank2, TX[3]=bank3
    //
    //  Encoding reference:
    //    enc_r:    {op[4:0], dt, 2'b00, rd, ra, rb, 12'd0}
    //    enc_i:    {op[4:0], dt, 2'b00, rd, ra, imm16}
    //    enc_m:    {op, 0, 00, rd, ra, offset16}   ⚠ ST: rD=source
    //    enc_setp: {OP_SETP, dt, cmp[1:0], pd, ra, rb, 12'd0}
    //    enc_set:  {OP_SET, 0, 00, {00,pd}, 0000, 15'd0, val}
    //    enc_selp: {OP_SELP, dt, pred_sel, rd, ra, rb, 12'd0}
    //    enc_mov_tid: {OP_MOV, 1, 00, rd, 0000, 16'd0}
    //    enc_bra:  {OP_BRA, target[26:0]}
    //    enc_pbra: {OP_PBRA, pred_sel, target[12:0], reconv[11:0]}
    // ════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────
    //  Test 35: GPU AND, OR, XOR (int16)
    //
    //  Input per bank: 0x0F0FFF00 → DMEM[0]=0xFF00, DMEM[1]=0x0F0F
    //  AND=0x0F00, OR=0xFF0F, XOR=0xF00F
    //  Stores: [0]=AND, [1]=OR, [2]=XOR, [3]=0
    //
    //  nw=4 → TX[n] = {DMEM[3],DMEM[2],DMEM[1],DMEM[0]} per bank
    //        = {0x0000, 0xF00F, 0xFF0F, 0x0F00} = 0x0000F00FFF0F0F00
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 35: GPU AND, OR, XOR (int16) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd2);
        send_gpu_imem(
            {32'h10100000, 32'h20000000}, // [1]LD R1,[R0+0]; [0]MOVI R0,0
            {32'h70312000, 32'h10200001}, // [3]AND R3,R1,R2; [2]LD R2,[R0+1]
            {32'h80512000, 32'h78412000}, // [5]XOR R5,R1,R2; [4]OR R4,R1,R2
            {32'h08400001, 32'h08300000}, // [7]ST R4,[R0+1]; [6]ST R3,[R0+0]
            {32'h20600000, 32'h08500002}, // [9]MOVI R6,0;    [8]ST R5,[R0+2]
            {32'hC8000000, 32'h08600003}, // [11]RET;          [10]ST R6,[R0+3]
            64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h0F0FFF00, 32'h0F0FFF00}, 8'h00);
        rx({32'h0F0FFF00, 32'h0F0FFF00}, 8'h00);

        send_gpu_tail(4);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T35 TX count");
        check64(tx_data[0], 64'h0000F00FFF0F0F00, "T35 TX[0] bank0 {0,XOR,OR,AND}");
        check64(tx_data[1], 64'h0000F00FFF0F0F00, "T35 TX[1] bank1");
        check64(tx_data[2], 64'h0000F00FFF0F0F00, "T35 TX[2] bank2");
        check64(tx_data[3], 64'h0000F00FFF0F0F00, "T35 TX[3] bank3");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 36: GPU SHL, SHR, ADDI, MULI (int16)
    //
    //  Input per bank: 0x00000005 → DMEM[0]=5, DMEM[1]=0
    //  SHL=40, SHR=2, ADDI=15, MULI=35
    //  Stores: [0]=SHL=40, [1]=SHR=2, [2]=ADDI=15, [3]=MULI=35
    //
    //  nw=4 → TX[n] = {35, 15, 2, 40}
    //        = {0x0023, 0x000F, 0x0002, 0x0028} = 0x0023000F00020028
    //
    //  BUG FIX: ST R3,[R0+1] was 0x08310001 (rA=R1). Fixed to 0x08300001 (rA=R0).
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 36: GPU SHL, SHR, ADDI, MULI ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd2);
        send_gpu_imem(
            {32'h10100000, 32'h20000000}, // [1]LD R1,[R0+0]; [0]MOVI R0,0
            {32'h90310001, 32'h88210003}, // [3]SHR R3,R1,1;  [2]SHL R2,R1,3
            {32'hA0510007, 32'h9841000A}, // [5]MULI R5,R1,7; [4]ADDI R4,R1,10
            {32'h08300001, 32'h08200000}, // [7]ST R3,[R0+1]; [6]ST R2,[R0+0]
                                          //  ^^^^^^^^ fixed (was 0x08310001 rA=R1)
            {32'h08500003, 32'h08400002}, // [9]ST R5,[R0+3]; [8]ST R4,[R0+2]
            {32'hC8000000, 32'h00000000}, // [11]RET;          [10]NOP
            64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h00000005, 32'h00000005}, 8'h00);
        rx({32'h00000005, 32'h00000005}, 8'h00);

        send_gpu_tail(4);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T36 TX count");
        check64(tx_data[0], 64'h0023000F00020028, "T36 TX[0] bank0 {MULI,ADDI,SHR,SHL}");
        check64(tx_data[1], 64'h0023000F00020028, "T36 TX[1] bank1");
        check64(tx_data[2], 64'h0023000F00020028, "T36 TX[2] bank2");
        check64(tx_data[3], 64'h0023000F00020028, "T36 TX[3] bank3");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 37: GPU ABS, NEG, MIN (int16, per-bank different)
    //
    //  D_UNPACK per bank:
    //    Bank 0: 0x0014FFF6 → DMEM[0]=-10=FFF6, DMEM[1]=20=0014
    //    Bank 1: 0x000AFFEC → DMEM[0]=-20=FFEC, DMEM[1]=10=000A
    //    Bank 2: 0x001EFFF1 → DMEM[0]=-15=FFF1, DMEM[1]=30=001E
    //    Bank 3: 0x0005FFE2 → DMEM[0]=-30=FFE2, DMEM[1]=5=0005
    //
    //  GPU: ABS(R1)→[0], NEG(R2)→[1], MIN(R1,R2)→[2], 0→[3]
    //
    //  nw=4 → TX[n] = bank_n {DMEM[3],DMEM[2],DMEM[1],DMEM[0]}:
    //    TX[0]=bank0: {0, MIN(-10,20)=-10, NEG(20)=-20, ABS(-10)=10}
    //         = {0000, FFF6, FFEC, 000A} = 0x0000FFF6FFEC000A
    //    TX[1]=bank1: {0, MIN(-20,10)=-20, NEG(10)=-10, ABS(-20)=20}
    //         = {0000, FFEC, FFF6, 0014} = 0x0000FFECFFF60014
    //    TX[2]=bank2: {0, MIN(-15,30)=-15, NEG(30)=-30, ABS(-15)=15}
    //         = {0000, FFF1, FFE2, 000F} = 0x0000FFF1FFE2000F
    //    TX[3]=bank3: {0, MIN(-30,5)=-30, NEG(5)=-5, ABS(-30)=30}
    //         = {0000, FFE2, FFFB, 001E} = 0x0000FFE2FFFB001E
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 37: GPU ABS, NEG, MIN (int16) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd2);
        send_gpu_imem(
            {32'h10100000, 32'h20000000}, // [1]LD R1,[R0+0]; [0]MOVI R0,0
            {32'h60310000, 32'h10200001}, // [3]ABS R3,R1;    [2]LD R2,[R0+1]
            {32'h58512000, 32'h68420000}, // [5]MIN R5,R1,R2; [4]NEG R4,R2
            {32'h08400001, 32'h08300000}, // [7]ST R4,[R0+1]; [6]ST R3,[R0+0]
            {32'h20600000, 32'h08500002}, // [9]MOVI R6,0;    [8]ST R5,[R0+2]
            {32'hC8000000, 32'h08600003}, // [11]RET;          [10]ST R6,[R0+3]
            64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h000AFFEC, 32'h0014FFF6}, 8'h00); // bank1, bank0
        rx({32'h0005FFE2, 32'h001EFFF1}, 8'h00); // bank3, bank2

        send_gpu_tail(4);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T37 TX count");
        check64(tx_data[0], 64'h0000FFF6FFEC000A, "T37 TX[0] bank0 ABS=10,NEG=-20,MIN=-10");
        check64(tx_data[1], 64'h0000FFECFFF60014, "T37 TX[1] bank1 ABS=20,NEG=-10,MIN=-20");
        check64(tx_data[2], 64'h0000FFF1FFE2000F, "T37 TX[2] bank2 ABS=15,NEG=-30,MIN=-15");
        check64(tx_data[3], 64'h0000FFE2FFFB001E, "T37 TX[3] bank3 ABS=30,NEG=-5,MIN=-30");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 38: GPU SUB.f, NEG.f, ABS.f, MIN.f (bf16)
    //
    //  Input per bank: {5.0=40A0, 3.0=4040} → DMEM[0]=3.0, DMEM[1]=5.0
    //  SUB.f(3,5)=-2.0=C000  NEG.f(3)=-3.0=C040
    //  ABS.f(-2)=2.0=4000    MIN.f(3,5)=3.0=4040
    //  Stores: [0]=SUB.f, [1]=NEG.f, [2]=ABS.f, [3]=MIN.f
    //
    //  nw=4 → TX[n] = {MIN.f, ABS.f, NEG.f, SUB.f}
    //        = {4040, 4000, C040, C000} = 0x40404000C040C000
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 38: GPU SUB.f, NEG.f, ABS.f, MIN.f (bf16) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd2);
        send_gpu_imem(
            {32'h14100000, 32'h20000000}, // [1]LD.f R1,[R0+0]; [0]MOVI R0,0
            {32'h3C312000, 32'h14200001}, // [3]SUB.f R3,R1,R2; [2]LD.f R2,[R0+1]
            {32'h64530000, 32'h6C410000}, // [5]ABS.f R5,R3;    [4]NEG.f R4,R1
            {32'h0C400001, 32'h5C612000}, // [7]ST.f R4,[R0+1]; [6]MIN.f R6,R1,R2
            {32'h0C500002, 32'h0C300000}, // [9]ST.f R5,[R0+2]; [8]ST.f R3,[R0+0]
            {32'hC8000000, 32'h0C600003}, // [11]RET;            [10]ST.f R6,[R0+3]
            64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h40A04040, 32'h40A04040}, 8'h00);
        rx({32'h40A04040, 32'h40A04040}, 8'h00);

        send_gpu_tail(4);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 4, "T38 TX count");
        check64(tx_data[0], 64'h40404000C040C000, "T38 TX[0] bank0 {MIN,ABS,NEG,SUB}");
        check64(tx_data[1], 64'h40404000C040C000, "T38 TX[1] bank1");
        check64(tx_data[2], 64'h40404000C040C000, "T38 TX[2] bank2");
        check64(tx_data[3], 64'h40404000C040C000, "T38 TX[3] bank3");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 39: GPU BRA (unconditional branch)
    //
    //  BRA skips two MOVIs. If it works, DMEM[0]=42; if broken, 88.
    //  nw=2 → TX[0]={bank1,bank0}, TX[1]={bank3,bank2}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 39: GPU BRA (unconditional branch) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h2010002A, 32'h20000000}, // [1]MOVI R1,42;    [0]MOVI R0,0
            {32'h20100063, 32'hB8000005}, // [3]MOVI R1,99;    [2]BRA 5
            {32'h08100000, 32'h20100058}, // [5]ST R1,[R0+0];  [4]MOVI R1,88
            {32'h08200001, 32'h20200000}, // [7]ST R2,[R0+1];  [6]MOVI R2,0
            {32'hC8000000, 32'h00000000}, // [9]pad;            [8]RET
            64'h0, 64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h0000BEEF, 32'h0000BEEF}, 8'h00);
        rx({32'h0000BEEF, 32'h0000BEEF}, 8'h00);

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T39 TX count");
        check64(tx_data[0], {32'h0000002A, 32'h0000002A}, "T39 TX[0] BRA: R1=42");
        check64(tx_data[1], {32'h0000002A, 32'h0000002A}, "T39 TX[1] all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 40: GPU MOV reg-to-reg, SETP, SELP
    //
    //  Input: {200,100} = 0x00C80064 → DMEM[0]=100, DMEM[1]=200
    //  MOV R3,R1=100; SETP P0=(100<200)=1; SELP R4=P0?R1:R2=100
    //  Stores: [0]=MOV=100, [1]=SELP=100
    //  nw=2 → TX[0]={bank1,bank0} = {0x00640064, 0x00640064}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 40: GPU MOV reg, SETP, SELP ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        //  [0]  MOVI R0,0           0x20000000
        //  [1]  LD R1,[R0+0]       0x10100000   R1=100
        //  [2]  LD R2,[R0+1]       0x10200001   R2=200
        //  [3]  MOV R3,R1           0x18310000   R3=100
        //  [4]  SETP P0,R1,R2,LT   0xAA012000   P0=(100<200)=1
        //  [5-7] NOP×3             pred WB drain
        //  [8]  SELP R4,R1,R2,P0   0xB0412000   R4=P0?R1:R2=100
        //  [9]  ST R3,[R0+0]       0x08300000
        //  [10] ST R4,[R0+1]       0x08400001
        //  [11] RET                0xC8000000
        send_gpu_imem(
            {32'h10100000, 32'h20000000}, // [1]LD R1;          [0]MOVI R0,0
            {32'h18310000, 32'h10200001}, // [3]MOV R3,R1;      [2]LD R2,[R0+1]
            {32'h00000000, 32'hAA012000}, // [5]NOP;             [4]SETP P0,R1,R2,LT
            {32'h00000000, 32'h00000000}, // [7]NOP;             [6]NOP
            {32'h08300000, 32'hB0412000}, // [9]ST R3,[R0+0];   [8]SELP R4,R1,R2,P0
            {32'hC8000000, 32'h08400001}, // [11]RET;            [10]ST R4,[R0+1]
            64'h0, 64'h0
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h00C80064, 32'h00C80064}, 8'h00);
        rx({32'h00C80064, 32'h00C80064}, 8'h00);

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T40 TX count");
        check64(tx_data[0], {32'h00640064, 32'h00640064}, "T40 TX[0] {SELP=100,MOV=100}");
        check64(tx_data[1], {32'h00640064, 32'h00640064}, "T40 TX[1] all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 41: GPU SET + SELP (predicate set/select)
    //
    //  SET writes P[n] ← literal bit (predicate RF, NOT GPR).
    //  Pattern matches K38 in sm_core_tb: SET → 3 NOPs → SELP.
    //
    //  SET P0=1 → SELP R3=P0?AAAA:BBBB → AAAA
    //  SET P0=0 → SELP R4=P0?AAAA:BBBB → BBBB
    //  nw=2 → TX = {0xBBBBAAAA, 0xBBBBAAAA}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 41: GPU SET + SELP (predicate) ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        //  [0]  MOVI R0,0           [1]  MOVI R1,0xAAAA
        //  [2]  MOVI R2,0xBBBB      [3]  SET P0,1 (0xD0000001)
        //  [4-6] NOP×3              [7]  SELP R3,R1,R2,P0 (0xB0312000)
        //  [8]  SET P0,0 (D0000000) [9-11] NOP×3
        //  [12] SELP R4,R1,R2,P0    [13] ST R3,[R0+0]
        //  [14] ST R4,[R0+1]        [15] RET
        send_gpu_imem(
            {32'h2010AAAA, 32'h20000000}, // [1]MOVI R1,AAAA;  [0]MOVI R0,0
            {32'hD0000001, 32'h2020BBBB}, // [3]SET P0,1;       [2]MOVI R2,BBBB
            {32'h00000000, 32'h00000000}, // [5]NOP;             [4]NOP
            {32'hB0312000, 32'h00000000}, // [7]SELP R3,R1,R2;  [6]NOP
            {32'h00000000, 32'hD0000000}, // [9]NOP;             [8]SET P0,0
            {32'h00000000, 32'h00000000}, // [11]NOP;            [10]NOP
            {32'h08300000, 32'hB0412000}, // [13]ST R3,[R0+0];  [12]SELP R4,R1,R2
            {32'hC8000000, 32'h08400001}  // [15]RET;            [14]ST R4,[R0+1]
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h0000DEAD, 32'h0000DEAD}, 8'h00);
        rx({32'h0000DEAD, 32'h0000DEAD}, 8'h00);

        send_gpu_tail(2);
        wait_and_capture(MAX_CYCLES);

        checkN(tx_cnt, 2, "T41 TX count");
        check64(tx_data[0], {32'hBBBBAAAA, 32'hBBBBAAAA}, "T41 TX[0] {P0=0->BBBB, P0=1->AAAA}");
        check64(tx_data[1], {32'hBBBBAAAA, 32'hBBBBAAAA}, "T41 TX[1] all banks");
        settle;
    end

    // ────────────────────────────────────────────────────────────
    //  Test 42: GPU PBRA — divergent execution + SIMT reconvergence
    //
    //  SETP P0=(TID<2): T0,T1 taken; T2,T3 fall.
    //  Layout matches K31 in sm_core_tb.
    //
    //  [0]  MOVI R0,0           [1]  MOV.TID R1  (0x1C100000)
    //  [2]  MOVI R3,2           [3]  SETP P0,R1,R3,LT (0xAA013000)
    //  [4]  NOP                 [5]  PBRA P0,tgt=9,rc=12 (0xC000900C)
    //  --- Fall (T2,T3, pend_pc=6) ---
    //  [6]  MOVI R2,0xBBBB      [7]  ST R2,[R0+0]   [8]  BRA 12
    //  --- Taken (T0,T1, target=9) ---
    //  [9]  MOVI R2,0xAAAA      [10] ST R2,[R0+0]   [11] NOP
    //  --- Reconvergence [12] ---
    //  [12] MOVI R3,0           [13] ST R3,[R0+1]   [14] RET
    //
    //  nw=2 → TX[0]={T1=AAAA, T0=AAAA}, TX[1]={T3=BBBB, T2=BBBB}
    // ────────────────────────────────────────────────────────────
    begin
        $display("\n--- Test 42: GPU PBRA divergence + SIMT reconvergence ---");
        cycle_cnt = 0;

        send_gpu_arm(8'h04, 8'd1, 8'd0, 8'd1);
        send_gpu_imem(
            {32'h1C100000, 32'h20000000}, // [1]MOV.TID R1;     [0]MOVI R0,0
            {32'hAA013000, 32'h20300002}, // [3]SETP P0,R1,R3,LT; [2]MOVI R3,2
            {32'hC000900C, 32'h00000000}, // [5]PBRA P0,t=9,r=12; [4]NOP
            {32'h08200000, 32'h2020BBBB}, // [7]ST R2,[R0+0];    [6]MOVI R2,BBBB
            {32'h2020AAAA, 32'hB800000C}, // [9]MOVI R2,AAAA;   [8]BRA 12
            {32'h00000000, 32'h08200000}, // [11]NOP;             [10]ST R2,[R0+0]
            {32'h08300001, 32'h20300000}, // [13]ST R3,[R0+1];   [12]MOVI R3,0
            {32'h00000000, 32'hC8000000}  // [15]NOP;             [14]RET
        );
        rx(cmd(4'h2, 12'h010, 16'd2, 32'h0), 8'h00);
        rx({32'h0000DEAD, 32'h0000DEAD}, 8'h00);
        rx({32'h0000DEAD, 32'h0000DEAD}, 8'h00);

        send_gpu_tail(2);
        wait_and_capture(20000);

        checkN(tx_cnt, 2, "T42 TX count");
        check64(tx_data[0], {32'h0000AAAA, 32'h0000AAAA}, "T42 TX[0] T1=AAAA,T0=AAAA (taken)");
        check64(tx_data[1], {32'h0000BBBB, 32'h0000BBBB}, "T42 TX[1] T3=BBBB,T2=BBBB (fall)");
        settle;
    end

    // ═══════════════════════════════════════════════════════════
    //  Summary
    // ═══════════════════════════════════════════════════════════
    $display("");
    $display("================================================================");
    if (fail_cnt == 0)
        $display("  *** ALL %0d CHECKS PASSED ***", pass_cnt);
    else
        $display("  *** %0d PASSED, %0d FAILED ***", pass_cnt, fail_cnt);
    $display("  Total checks: %0d", pass_cnt + fail_cnt);
    $display("================================================================");
    $display("");
    #(CLK_PERIOD * 5);
    $finish;
end

endmodule