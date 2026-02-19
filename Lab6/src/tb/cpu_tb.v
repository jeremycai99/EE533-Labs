/*  cpu_tb.v — Enhanced debug testbench for sort.s
 *  Abundant per-cycle tracing of all pipeline stages, forwarding,
 *  hazard detection, BDTU state, register writes, and memory activity.
 */
`timescale 1ns / 1ps
`include "define.v"

`include "cpu.v"

module cpu_tb;

// ═══════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════
parameter CLK_PERIOD      = 10;
parameter TIMEOUT          = 200_000;
parameter MEM_DEPTH        = 4096;       // words
parameter TRACE_CYCLES     = 120;        // verbose trace window (increased)
parameter STATUS_INTERVAL  = 10_000;     // periodic progress report

parameter [31:0] SP_INIT   = 32'h0000_0400;

localparam [31:0] FP_EXP    = SP_INIT - 32'd4;
localparam [31:0] ARR_BASE  = FP_EXP  - 32'd56;
localparam [31:0] ARR_WBASE = ARR_BASE >> 2;

// ═══════════════════════════════════════════
//  DUT signals
// ═══════════════════════════════════════════
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

// ═══════════════════════════════════════════
//  Unified memory (word-addressed)
// ═══════════════════════════════════════════
reg [31:0] mem_array [0:MEM_DEPTH-1];

// ═══════════════════════════════════════════
//  DUT instantiation
// ═══════════════════════════════════════════
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

// ═══════════════════════════════════════════
//  Clock
// ═══════════════════════════════════════════
initial clk = 1'b0;
always #(CLK_PERIOD/2) clk = ~clk;

// ═══════════════════════════════════════════
//  Synchronous memory model (1-cycle latency)
// ═══════════════════════════════════════════
always @(posedge clk) begin
    i_mem_data  <= mem_array[(i_mem_addr  >> 2) & (MEM_DEPTH-1)];
    d_mem_rdata <= mem_array[(d_mem_addr  >> 2) & (MEM_DEPTH-1)];
    if (d_mem_wen)
        mem_array[(d_mem_addr >> 2) & (MEM_DEPTH-1)] <= d_mem_wdata;
end

// ═══════════════════════════════════════════
//  BDTU state name decoder (for trace)
// ═══════════════════════════════════════════
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

// ═══════════════════════════════════════════
//  Forward select name decoder (for trace)
// ═══════════════════════════════════════════
function [6*8:1] fwd_name;
    input [2:0] sel;
    case (sel)
        3'b000: fwd_name = "REG   ";
        3'b001: fwd_name = "EX/MEM";
        3'b010: fwd_name = "ME/WB ";
        3'b011: fwd_name = "BDTU_1";
        3'b100: fwd_name = "BDTU_2";
        default: fwd_name = "???   ";
    endcase
endfunction

// ═══════════════════════════════════════════
//  Program + data loader
// ═══════════════════════════════════════════
task load_program;
    integer k;
begin
    for (k = 0; k < MEM_DEPTH; k = k + 1)
        mem_array[k] = 32'h0000_0000;

    // ─── BOOTSTRAP ────────────────────────────────────────────
    mem_array['h000 >> 2] = 32'hE3A0B000;  // MOV  R11(FP), #0
    mem_array['h004 >> 2] = 32'hE3E0E000;  // MVN  R14(LR), #0  → 0xFFFFFFFF
    mem_array['h008 >> 2] = 32'hE1A00000;  // NOP  (MOV R0, R0)
    mem_array['h00C >> 2] = 32'hE3A0DB01;  // MOV  R13(SP), #0x400

    // ─── FUNCTION PROLOGUE ────────────────────────────────────
    mem_array['h010 >> 2] = 32'hE92D4800;  // push {fp, lr}
    mem_array['h014 >> 2] = 32'hE28DB004;  // add  fp, sp, #4
    mem_array['h018 >> 2] = 32'hE24DD038;  // sub  sp, sp, #56

    // ─── ARRAY INITIALISATION (.LC0 → stack) ─────────────────
    mem_array['h01C >> 2] = 32'hE59F3104;  // ldr  r3, [pc, #260]
    mem_array['h020 >> 2] = 32'hE24BC038;  // sub  ip, fp, #56
    mem_array['h024 >> 2] = 32'hE1A0E003;  // mov  lr, r3
    mem_array['h028 >> 2] = 32'hE8BE000F;  // ldmia lr!, {r0-r3}
    mem_array['h02C >> 2] = 32'hE8AC000F;  // stmia ip!, {r0-r3}
    mem_array['h030 >> 2] = 32'hE8BE000F;  // ldmia lr!, {r0-r3}
    mem_array['h034 >> 2] = 32'hE8AC000F;  // stmia ip!, {r0-r3}
    mem_array['h038 >> 2] = 32'hE89E0003;  // ldm   lr, {r0, r1}
    mem_array['h03C >> 2] = 32'hE88C0003;  // stm   ip, {r0, r1}

    // ─── OUTER LOOP INIT (i = 0) ─────────────────────────────
    mem_array['h040 >> 2] = 32'hE3A03000;  // mov  r3, #0
    mem_array['h044 >> 2] = 32'hE50B3008;  // str  r3, [fp, #-8]
    mem_array['h048 >> 2] = 32'hEA00002E;  // b    .L2

    // ─── .L6 — outer loop body ───────────────────────────────
    mem_array['h04C >> 2] = 32'hE51B3008;  // ldr  r3, [fp, #-8]
    mem_array['h050 >> 2] = 32'hE2833001;  // add  r3, r3, #1
    mem_array['h054 >> 2] = 32'hE50B300C;  // str  r3, [fp, #-12]
    mem_array['h058 >> 2] = 32'hEA000024;  // b    .L3

    // ─── .L5 — inner loop ────────────────────────────────────
    mem_array['h05C >> 2] = 32'hE51B300C;  // ldr  r3, [fp, #-12]
    mem_array['h060 >> 2] = 32'hE1A03103;  // lsl  r3, r3, #2
    mem_array['h064 >> 2] = 32'hE2433004;  // sub  r3, r3, #4
    mem_array['h068 >> 2] = 32'hE083300B;  // add  r3, r3, fp
    mem_array['h06C >> 2] = 32'hE5132034;  // ldr  r2, [r3, #-52]
    mem_array['h070 >> 2] = 32'hE51B3008;  // ldr  r3, [fp, #-8]
    mem_array['h074 >> 2] = 32'hE1A03103;  // lsl  r3, r3, #2
    mem_array['h078 >> 2] = 32'hE2433004;  // sub  r3, r3, #4
    mem_array['h07C >> 2] = 32'hE083300B;  // add  r3, r3, fp
    mem_array['h080 >> 2] = 32'hE5133034;  // ldr  r3, [r3, #-52]
    mem_array['h084 >> 2] = 32'hE1520003;  // cmp  r2, r3
    mem_array['h088 >> 2] = 32'hAA000015;  // bge  .L4

    // ─── swap ────────────────────────────────────────────────
    mem_array['h08C >> 2] = 32'hE51B300C;  // ldr  r3, [fp, #-12]
    mem_array['h090 >> 2] = 32'hE1A03103;  // lsl  r3, r3, #2
    mem_array['h094 >> 2] = 32'hE2433004;  // sub  r3, r3, #4
    mem_array['h098 >> 2] = 32'hE083300B;  // add  r3, r3, fp
    mem_array['h09C >> 2] = 32'hE5133034;  // ldr  r3, [r3, #-52]
    mem_array['h0A0 >> 2] = 32'hE50B3010;  // str  r3, [fp, #-16]
    mem_array['h0A4 >> 2] = 32'hE51B3008;  // ldr  r3, [fp, #-8]
    mem_array['h0A8 >> 2] = 32'hE1A03103;  // lsl  r3, r3, #2
    mem_array['h0AC >> 2] = 32'hE2433004;  // sub  r3, r3, #4
    mem_array['h0B0 >> 2] = 32'hE083300B;  // add  r3, r3, fp
    mem_array['h0B4 >> 2] = 32'hE5132034;  // ldr  r2, [r3, #-52]
    mem_array['h0B8 >> 2] = 32'hE51B300C;  // ldr  r3, [fp, #-12]
    mem_array['h0BC >> 2] = 32'hE1A03103;  // lsl  r3, r3, #2
    mem_array['h0C0 >> 2] = 32'hE2433004;  // sub  r3, r3, #4
    mem_array['h0C4 >> 2] = 32'hE083300B;  // add  r3, r3, fp
    mem_array['h0C8 >> 2] = 32'hE5032034;  // str  r2, [r3, #-52]
    mem_array['h0CC >> 2] = 32'hE51B3008;  // ldr  r3, [fp, #-8]
    mem_array['h0D0 >> 2] = 32'hE1A03103;  // lsl  r3, r3, #2
    mem_array['h0D4 >> 2] = 32'hE2433004;  // sub  r3, r3, #4
    mem_array['h0D8 >> 2] = 32'hE083300B;  // add  r3, r3, fp
    mem_array['h0DC >> 2] = 32'hE51B2010;  // ldr  r2, [fp, #-16]
    mem_array['h0E0 >> 2] = 32'hE5032034;  // str  r2, [r3, #-52]

    // ─── .L4 — j++ ──────────────────────────────────────────
    mem_array['h0E4 >> 2] = 32'hE51B300C;  // ldr  r3, [fp, #-12]
    mem_array['h0E8 >> 2] = 32'hE2833001;  // add  r3, r3, #1
    mem_array['h0EC >> 2] = 32'hE50B300C;  // str  r3, [fp, #-12]

    // ─── .L3 — inner condition ──────────────────────────────
    mem_array['h0F0 >> 2] = 32'hE51B300C;  // ldr  r3, [fp, #-12]
    mem_array['h0F4 >> 2] = 32'hE3530009;  // cmp  r3, #9
    mem_array['h0F8 >> 2] = 32'hDAFFFFD7;  // ble  .L5

    // ─── i++ / .L2 ──────────────────────────────────────────
    mem_array['h0FC >> 2] = 32'hE51B3008;  // ldr  r3, [fp, #-8]
    mem_array['h100 >> 2] = 32'hE2833001;  // add  r3, r3, #1
    mem_array['h104 >> 2] = 32'hE50B3008;  // str  r3, [fp, #-8]
    mem_array['h108 >> 2] = 32'hE51B3008;  // ldr  r3, [fp, #-8]
    mem_array['h10C >> 2] = 32'hE3530009;  // cmp  r3, #9
    mem_array['h110 >> 2] = 32'hDAFFFFCD;  // ble  .L6

    // ─── EPILOGUE ───────────────────────────────────────────
    mem_array['h114 >> 2] = 32'hE3A03000;  // mov  r3, #0
    mem_array['h118 >> 2] = 32'hE1A00003;  // mov  r0, r3
    mem_array['h11C >> 2] = 32'hE24BD004;  // sub  sp, fp, #4
    mem_array['h120 >> 2] = 32'hE8BD4800;  // pop  {fp, lr}
    mem_array['h124 >> 2] = 32'hE12FFF1E;  // bx   lr

    // ─── LITERAL POOL .L8 ──────────────────────────────────
    mem_array['h128 >> 2] = 32'h0000_012C;

    // ─── .LC0 — unsorted array data ─────────────────────────
    mem_array['h12C >> 2] = 32'h0000_0143; //  323
    mem_array['h130 >> 2] = 32'h0000_007B; //  123
    mem_array['h134 >> 2] = 32'hFFFF_FE39; // -455
    mem_array['h138 >> 2] = 32'h0000_0002; //    2
    mem_array['h13C >> 2] = 32'h0000_0062; //   98
    mem_array['h140 >> 2] = 32'h0000_007D; //  125
    mem_array['h144 >> 2] = 32'h0000_000A; //   10
    mem_array['h148 >> 2] = 32'h0000_0041; //   65
    mem_array['h14C >> 2] = 32'hFFFF_FFC8; //  -56
    mem_array['h150 >> 2] = 32'h0000_0000; //    0
end
endtask

// ═══════════════════════════════════════════
//  Housekeeping
// ═══════════════════════════════════════════
integer cycle_cnt, errors, i;
reg signed [31:0] expected [0:9];
reg [`PC_WIDTH-1:0] prev_pc;
integer stuck_cnt;

// ═══════════════════════════════════════════
//  Per-cycle detailed trace
// ═══════════════════════════════════════════
always @(posedge clk) begin
    if (rst_n && (cycle_cnt < TRACE_CYCLES)) begin

        // ── Header: cycle + pipeline overview ─────────────────
        $display("────────────────────────────────────────────────────────────────────────────────");
        $display("[C%04d] IF: PC=0x%08H  |  ID: instr=0x%08H valid=%b  |  EX: alu_res=0x%08H  |  MEM: addr=0x%08H  |  WB: data1=0x%08H",
                 cycle_cnt,
                 i_mem_addr,
                 u_cpu.instr_id, u_cpu.ifid_valid,
                 u_cpu.alu_result_ex,
                 u_cpu.mem_addr_mem,
                 u_cpu.wb_data1);

        // ── Hazard / stall / flush status ─────────────────────
        $display("        CTRL: stall_if=%b stall_id=%b stall_ex=%b stall_mem=%b | flush_ifid=%b flush_idex=%b flush_exmem=%b",
                 u_cpu.stall_if, u_cpu.stall_id, u_cpu.stall_ex, u_cpu.stall_mem,
                 u_cpu.flush_ifid, u_cpu.flush_idex, u_cpu.flush_exmem);

        // ── HDU detail ────────────────────────────────────────
        $display("        HDU:  lu_hazard=%b (ld_rn=%b ld_rm=%b ld_rs=%b ld_rd=%b) | idex_load=%b idex_wd=R%0d idex_we=%b",
                 u_cpu.u_hdu.load_use_hazard,
                 u_cpu.u_hdu.load_use_rn, u_cpu.u_hdu.load_use_rm,
                 u_cpu.u_hdu.load_use_rs, u_cpu.u_hdu.load_use_rd,
                 u_cpu.mem_read_ex, u_cpu.wr_addr1_ex, u_cpu.wr_en1_ex);

        // ── BDTU state ────────────────────────────────────────
        $display("        BDTU: state=%s busy=%b start=%b | addr=0x%08H rd=%b wr=%b wdata=0x%08H rdata=0x%08H",
                 bdtu_state_name(u_cpu.u_bdtu.state),
                 u_cpu.bdtu_busy, u_cpu.is_multi_cycle_mem,
                 u_cpu.u_bdtu.mem_addr, u_cpu.u_bdtu.mem_rd, u_cpu.u_bdtu.mem_wr,
                 u_cpu.u_bdtu.mem_wdata, d_mem_rdata);

        // ── BDTU register write activity ──────────────────────
        if (u_cpu.bdtu_wr_en1 || u_cpu.bdtu_wr_en2)
            $display("        BDTU WR: port1: R%0d<=0x%08H en=%b | port2: R%0d<=0x%08H en=%b",
                     u_cpu.bdtu_wr_addr1, u_cpu.bdtu_wr_data1, u_cpu.bdtu_wr_en1,
                     u_cpu.bdtu_wr_addr2, u_cpu.bdtu_wr_data2, u_cpu.bdtu_wr_en2);

        // ── Branch status ─────────────────────────────────────
        if (u_cpu.branch_taken_ex)
            $display("        BRANCH: taken=%b target=0x%08H (exchange=%b link=%b)",
                     u_cpu.branch_taken_ex, u_cpu.branch_target_ex,
                     u_cpu.branch_exchange_ex, u_cpu.branch_link_ex);

        // ── ID stage register reads ───────────────────────────
        $display("        ID READ: Rn[R%0d]=0x%08H  Rm[R%0d]=0x%08H  R3port[R%0d]=0x%08H | use: rn=%b rm=%b rs=%b rd=%b",
                 u_cpu.rn_addr_id, u_cpu.rn_data_id,
                 u_cpu.rm_addr_id, u_cpu.rm_data_id,
                 u_cpu.r3addr_mux, u_cpu.r3_data_id,
                 u_cpu.use_rn_id, u_cpu.use_rm_id, u_cpu.use_rs_id, u_cpu.use_rd_id);

        // ── EX stage forwarding ───────────────────────────────
        $display("        EX FWD: fwd_a=%s fwd_b=%s fwd_s=%s fwd_d=%s",
                 fwd_name(u_cpu.fwd_a), fwd_name(u_cpu.fwd_b),
                 fwd_name(u_cpu.fwd_s), fwd_name(u_cpu.fwd_d));
        $display("        EX OPS: rn_fwd=0x%08H rm_fwd=0x%08H rs_fwd=0x%08H rd_st_fwd=0x%08H",
                 u_cpu.rn_fwd, u_cpu.rm_fwd, u_cpu.rs_fwd, u_cpu.rd_store_fwd);
        $display("        EX ALU: a=0x%08H b=0x%08H op=%04b => res=0x%08H flags=%04b | shift_out=0x%08H cout=%b",
                 u_cpu.rn_fwd, u_cpu.alu_src_b_val, u_cpu.alu_op_ex,
                 u_cpu.alu_result_ex, u_cpu.alu_flags_ex,
                 u_cpu.shifted_rm, u_cpu.shifter_cout);

        // ── EX stage control ──────────────────────────────────
        $display("        EX CTL: wr1=R%0d en1=%b wr2=R%0d en2=%b | mem_rd=%b mem_wr=%b wb_sel=%03b | mul_en=%b multi_cyc=%b",
                 u_cpu.wr_addr1_ex, u_cpu.wr_en1_ex,
                 u_cpu.wr_addr2_ex, u_cpu.wr_en2_ex,
                 u_cpu.mem_read_ex, u_cpu.mem_write_ex, u_cpu.wb_sel_ex,
                 u_cpu.mul_en_ex, u_cpu.is_multi_cycle_ex);

        // ── MEM stage ─────────────────────────────────────────
        $display("        MEM: alu_res=0x%08H addr=0x%08H store=0x%08H | rd=%b wr=%b size=%02b signed=%b | wr1=R%0d en1=%b wr2=R%0d en2=%b",
                 u_cpu.alu_result_mem, u_cpu.mem_addr_mem, u_cpu.store_data_mem,
                 u_cpu.mem_read_mem, u_cpu.mem_write_mem, u_cpu.mem_size_mem, u_cpu.mem_signed_mem,
                 u_cpu.wr_addr1_mem, u_cpu.wr_en1_mem,
                 u_cpu.wr_addr2_mem, u_cpu.wr_en2_mem);

        // ── DMEM bus (active transaction) ─────────────────────
        if (d_mem_wen)
            $display("        DMEM WRITE: [0x%08H] <= 0x%08H  size=%02b",
                     d_mem_addr, d_mem_wdata, d_mem_size);
        $display("        DMEM READ:  [0x%08H] => 0x%08H",
                 d_mem_addr, d_mem_rdata);

        // ── WB stage ──────────────────────────────────────────
        $display("        WB: sel=%03b data1=0x%08H data2=0x%08H | wr1=R%0d en1=%b wr2=R%0d en2=%b | load_data=0x%08H",
                 u_cpu.wb_sel_wb, u_cpu.wb_data1, u_cpu.wb_data2,
                 u_cpu.wr_addr1_wb, u_cpu.wr_en1_wb,
                 u_cpu.wr_addr2_wb, u_cpu.wr_en2_wb,
                 u_cpu.load_data_wb);

        // ── Regfile write port (merged) ───────────────────────
        if (u_cpu.rf_wr_en)
            $display("        RF WRITE: port1: R%0d<=0x%08H | port2: R%0d<=0x%08H  (bdtu_prio=%b)",
                     u_cpu.rf_wr_addr1, u_cpu.rf_wr_data1,
                     u_cpu.rf_wr_addr2, u_cpu.rf_wr_data2,
                     u_cpu.bdtu_busy);

        // ── CPSR ──────────────────────────────────────────────
        $display("        CPSR: %04b (N=%b Z=%b C=%b V=%b) | cpsr_wen_ex=%b new_flags=%04b",
                 u_cpu.cpsr_flags,
                 u_cpu.cpsr_flags[3], u_cpu.cpsr_flags[2],
                 u_cpu.cpsr_flags[1], u_cpu.cpsr_flags[0],
                 u_cpu.cpsr_wen_ex, u_cpu.new_flags);

        // ── Key registers snapshot (every 10 cycles or on BDTU/branch events) ──
        if ((cycle_cnt % 10 == 0) || u_cpu.branch_taken_ex || (u_cpu.bdtu_busy && !u_cpu.u_bdtu.state[0]))
            $display("        REGS: R0=%08H R1=%08H R2=%08H R3=%08H SP=%08H FP=%08H LR=%08H",
                     u_cpu.u_regfile.regs[0],  u_cpu.u_regfile.regs[1],
                     u_cpu.u_regfile.regs[2],  u_cpu.u_regfile.regs[3],
                     u_cpu.u_regfile.regs[13], u_cpu.u_regfile.regs[11],
                     u_cpu.u_regfile.regs[14]);
    end

    // ── Compact trace after verbose window ────────────────────
    if (rst_n && (cycle_cnt >= TRACE_CYCLES) && (cycle_cnt < TRACE_CYCLES + 500)) begin
        // Print only on events: branch, BDTU transition, regfile write, or every 50 cycles
        if (u_cpu.branch_taken_ex || (u_cpu.rf_wr_en && !u_cpu.bdtu_busy) ||
            (u_cpu.bdtu_busy && u_cpu.u_bdtu.state == 3'd7) ||
            (cycle_cnt % 50 == 0))
            $display("[C%04d] PC=0x%08H instr=0x%08H | branch=%b bdtu=%s | WR: R%0d<=0x%08H en=%b | CPSR=%04b",
                     cycle_cnt, i_mem_addr, u_cpu.instr_id,
                     u_cpu.branch_taken_ex,
                     bdtu_state_name(u_cpu.u_bdtu.state),
                     u_cpu.rf_wr_addr1, u_cpu.rf_wr_data1, u_cpu.rf_wr_en,
                     u_cpu.cpsr_flags);
    end

    // Periodic status
    if (rst_n && (cycle_cnt > 0) && (cycle_cnt % STATUS_INTERVAL == 0)) begin
        $display("");
        $display("═══════════════════════ STATUS @ C%06d ═══════════════════════", cycle_cnt);
        $display("  PC=0x%08H  BDTU=%s  stall_if=%b",
                 i_mem_addr, bdtu_state_name(u_cpu.u_bdtu.state), u_cpu.stall_if);
        $display("  R0=%08H R1=%08H R2=%08H R3=%08H",
                 u_cpu.u_regfile.regs[0],  u_cpu.u_regfile.regs[1],
                 u_cpu.u_regfile.regs[2],  u_cpu.u_regfile.regs[3]);
        $display("  SP=%08H FP=%08H LR=%08H CPSR=%04b",
                 u_cpu.u_regfile.regs[13], u_cpu.u_regfile.regs[11],
                 u_cpu.u_regfile.regs[14], u_cpu.cpsr_flags);
        $display("═══════════════════════════════════════════════════════════════");
        $display("");
    end
end

// ═══════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════
task dump_regs;
    integer r;
begin
    $display("  +-----------------------------------------+");
    $display("  |          Register File Dump             |");
    $display("  +-----------------------------------------+");
    for (r = 0; r < 16; r = r + 1)
        $display("  |  R%-2d  = 0x%08H  (%0d)", r,
                 u_cpu.u_regfile.regs[r],
                 $signed(u_cpu.u_regfile.regs[r]));
    $display("  +-----------------------------------------+");
    $display("  |  CPSR  = %04b (N=%b Z=%b C=%b V=%b)    |",
             u_cpu.cpsr_flags,
             u_cpu.cpsr_flags[3], u_cpu.cpsr_flags[2],
             u_cpu.cpsr_flags[1], u_cpu.cpsr_flags[0]);
    $display("  +-----------------------------------------+");
end
endtask

task dump_stack_array;
    integer idx;
begin
    $display("  Stack array (base 0x%04H, word idx 0x%03H):", ARR_BASE, ARR_WBASE);
    for (idx = 0; idx < 10; idx = idx + 1)
        $display("    arr[%0d] = %11d  (0x%08H)",
                 idx,
                 $signed(mem_array[ARR_WBASE + idx]),
                 mem_array[ARR_WBASE + idx]);
end
endtask

task dump_stack_frame;
    integer addr;
begin
    $display("  ── Stack frame dump (0x%04H..0x%04H) ──", SP_INIT - 32'h60, SP_INIT);
    for (addr = (SP_INIT - 32'h60) >> 2; addr <= SP_INIT >> 2; addr = addr + 1)
        $display("    [0x%04H] = 0x%08H  (%0d)",
                 addr << 2, mem_array[addr], $signed(mem_array[addr]));
end
endtask

// ═══════════════════════════════════════════
//  Main stimulus
// ═══════════════════════════════════════════
initial begin
    $dumpfile("cpu_tb.vcd");
    $dumpvars(0, cpu_tb);

    $display("");
    $display("════════════════════════════════════════════════════════════════");
    $display("  ARM CPU Testbench  -  Selection Sort (sort.s)");
    $display("  Bootstrap initialises SP/LR/FP via real instructions");
    $display("════════════════════════════════════════════════════════════════");
    $display("  SP_INIT     = 0x%08H", SP_INIT);
    $display("  Expected FP = 0x%08H", FP_EXP);
    $display("  Array base  = 0x%08H  (word 0x%03H)", ARR_BASE, ARR_WBASE);
    $display("════════════════════════════════════════════════════════════════");
    $display("");

    rst_n         = 1'b0;
    ila_debug_sel = 5'd0;
    cycle_cnt     = 0;
    errors        = 0;
    stuck_cnt     = 0;
    prev_pc       = {`PC_WIDTH{1'b0}};

    load_program();

    expected[0] = -32'sd455;
    expected[1] = -32'sd56;
    expected[2] =  32'sd0;
    expected[3] =  32'sd2;
    expected[4] =  32'sd10;
    expected[5] =  32'sd65;
    expected[6] =  32'sd98;
    expected[7] =  32'sd123;
    expected[8] =  32'sd125;
    expected[9] =  32'sd323;

    repeat (5) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    $display("[%0t] Reset released.", $time);
    $display("");

    begin : sim_loop
        forever begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;

            // Stuck-PC detector
            if (i_mem_addr === prev_pc)
                stuck_cnt = stuck_cnt + 1;
            else
                stuck_cnt = 0;
            prev_pc = i_mem_addr;

            if (stuck_cnt > 500) begin
                $display("");
                $display("╔══════════════════════════════════════════════════╗");
                $display("║  *** STUCK: PC=0x%08H for %0d cycles ***", i_mem_addr, stuck_cnt);
                $display("╚══════════════════════════════════════════════════╝");
                $display("  BDTU state=%s busy=%b start(is_mc_mem)=%b",
                         bdtu_state_name(u_cpu.u_bdtu.state),
                         u_cpu.bdtu_busy, u_cpu.is_multi_cycle_mem);
                $display("  Stalls: if=%b id=%b ex=%b mem=%b",
                         u_cpu.stall_if, u_cpu.stall_id, u_cpu.stall_ex, u_cpu.stall_mem);
                $display("  EX: wr1=R%0d en=%b  branch=%b  multi_cyc=%b",
                         u_cpu.wr_addr1_ex, u_cpu.wr_en1_ex,
                         u_cpu.branch_taken_ex, u_cpu.is_multi_cycle_ex);
                $display("  MEM: wr1=R%0d en=%b  rd=%b wr=%b  multi_cyc=%b",
                         u_cpu.wr_addr1_mem, u_cpu.wr_en1_mem,
                         u_cpu.mem_read_mem, u_cpu.mem_write_mem, u_cpu.is_multi_cycle_mem);
                dump_regs();
                dump_stack_frame();
                repeat (5) @(posedge clk);
                disable sim_loop;
            end

            if (cpu_done_w) begin
                $display("");
                $display("[%0t] cpu_done asserted at cycle %0d (PC=0x%08H)",
                         $time, cycle_cnt, i_mem_addr);
                repeat (10) @(posedge clk);
                disable sim_loop;
            end

            if (cycle_cnt >= TIMEOUT) begin
                $display("");
                $display("*** TIMEOUT after %0d cycles (PC=0x%04H) ***",
                         TIMEOUT, i_mem_addr);
                dump_regs();
                dump_stack_array();
                dump_stack_frame();
                disable sim_loop;
            end
        end
    end

    // ═══════════════════════════════════════════════════════
    //  V E R I F I C A T I O N
    // ═══════════════════════════════════════════════════════
    $display("");
    $display("════════════════════════════════════════════════════════════════");
    $display("  VERIFICATION  (%0d cycles)", cycle_cnt);
    $display("════════════════════════════════════════════════════════════════");

    dump_regs();
    $display("");

    // R0 = 0
    if (u_cpu.u_regfile.regs[0] !== 32'd0) begin
        $display("  [FAIL] R0 = 0x%08H, expected 0", u_cpu.u_regfile.regs[0]);
        errors = errors + 1;
    end else
        $display("  [PASS] R0 = 0  (return value)");

    // SP restored
    if (u_cpu.u_regfile.regs[13] !== SP_INIT) begin
        $display("  [FAIL] SP = 0x%08H, expected 0x%08H",
                 u_cpu.u_regfile.regs[13], SP_INIT);
        errors = errors + 1;
    end else
        $display("  [PASS] SP = 0x%08H  (restored)", SP_INIT);

    // FP restored
    if (u_cpu.u_regfile.regs[11] !== 32'd0) begin
        $display("  [FAIL] FP = 0x%08H, expected 0",
                 u_cpu.u_regfile.regs[11]);
        errors = errors + 1;
    end else
        $display("  [PASS] FP = 0  (restored)");

    // LR restored
    if (u_cpu.u_regfile.regs[14] !== 32'hFFFF_FFFF) begin
        $display("  [FAIL] LR = 0x%08H, expected 0xFFFFFFFF",
                 u_cpu.u_regfile.regs[14]);
        errors = errors + 1;
    end else
        $display("  [PASS] LR = 0xFFFFFFFF  (restored)");

    // Sorted array
    $display("");
    $display("  --- Sorted array verification ---");
    for (i = 0; i < 10; i = i + 1) begin
        if ($signed(mem_array[ARR_WBASE + i]) !== expected[i]) begin
            $display("  [FAIL] arr[%0d] = %0d, expected %0d",
                     i, $signed(mem_array[ARR_WBASE + i]), expected[i]);
            errors = errors + 1;
        end else
            $display("  [PASS] arr[%0d] = %0d", i, expected[i]);
    end

    // .LC0 unchanged
    $display("");
    begin : lc0_chk
        reg [31:0] lc0 [0:9];
        integer m;
        reg lc0_ok;
        lc0[0]=32'h143; lc0[1]=32'h7B;   lc0[2]=32'hFFFFFE39;
        lc0[3]=32'h2;   lc0[4]=32'h62;   lc0[5]=32'h7D;
        lc0[6]=32'hA;   lc0[7]=32'h41;   lc0[8]=32'hFFFFFFC8;
        lc0[9]=32'h0;
        lc0_ok = 1'b1;
        for (m = 0; m < 10; m = m + 1) begin
            if (mem_array[('h12C >> 2) + m] !== lc0[m]) begin
                $display("  [FAIL] .LC0[%0d] = 0x%08H, expected 0x%08H",
                         m, mem_array[('h12C >> 2) + m], lc0[m]);
                errors = errors + 1;
                lc0_ok = 1'b0;
            end
        end
        if (lc0_ok)
            $display("  [PASS] .LC0 read-only data unchanged");
    end

    // Stack frame dump for post-mortem
    $display("");
    dump_stack_frame();

    // Summary
    $display("");
    $display("════════════════════════════════════════════════════════════════");
    if (errors == 0)
        $display("  *** ALL %0d CHECKS PASSED ***", 4 + 10 + 10);
    else
        $display("  *** %0d ERROR(S) DETECTED ***", errors);
    $display("════════════════════════════════════════════════════════════════");
    $display("");

    #(CLK_PERIOD * 5);
    $finish;
end

endmodule