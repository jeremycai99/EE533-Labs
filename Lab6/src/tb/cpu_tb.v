/*  cpu_tb.v — Enhanced debug testbench (hex/binary file driven, generic)
 *  Loads program + data from an external file via $readmemh or $readmemb.
 *
 *  FILE FORMATS (standard Verilog memory file):
 *
 *    Hex mode (default) — e.g. program.hex:
 *      E92D4800
 *      E28DB004
 *      E24DD038
 *      ...
 *
 *    Binary mode (-D BIN_MODE) — e.g. program.bin:
 *      11101001001011010100100000000000
 *      11100010100010001101101100000100
 *      11100010010011011101000000111000
 *      ...
 *
 *    Both formats: one 32-bit word per line, optional @<addr> directives.
 *    Word 0 maps to byte address 0x0000, word 1 to 0x0004, etc.
 *
 *  COMPILATION EXAMPLES:
 *      # Hex file (default):
 *      iverilog -o cpu_tb cpu_tb.v -D 'MEM_FILE="sort.hex"'
 *
 *      # Binary file:
 *      iverilog -o cpu_tb cpu_tb.v -D BIN_MODE -D 'MEM_FILE="sort.bin"'
 *
 *      # Override trace depth and timeout:
 *      iverilog -o cpu_tb cpu_tb.v -D 'MEM_FILE="sort.hex"' -D 'TRACE_DEPTH=200' -D 'SIM_TIMEOUT=500000'
 *
 *      vvp cpu_tb
 */
`timescale 1ns / 1ps
`include "define.v"
`include "cpu.v"

module cpu_tb;

// ═══════════════════════════════════════════
//  Parameters (overridable from command line)
// ═══════════════════════════════════════════
parameter CLK_PERIOD      = 10;
parameter MEM_DEPTH       = 4096;       // words

`ifdef SIM_TIMEOUT
parameter TIMEOUT         = `SIM_TIMEOUT;
`else
parameter TIMEOUT         = 200_000;
`endif

`ifdef TRACE_DEPTH
parameter TRACE_CYCLES    = `TRACE_DEPTH;
`else
parameter TRACE_CYCLES    = 120;
`endif

parameter STATUS_INTERVAL = 10_000;     // periodic progress report
parameter COMPACT_WINDOW  = 500;        // compact trace after verbose window

// Memory file to load (override with -D 'MEM_FILE="yourfile"')
`ifdef MEM_FILE
parameter FILE_NAME = "sort.hex";
`else
parameter FILE_NAME = "program.hex";
`endif

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
reg  [`REG_ADDR_WIDTH:0]     ila_debug_sel;
wire [`DATA_WIDTH-1:0]       ila_debug_data;

// ═══════════════════════════════════════════
//  Unified memory (word-addressed)
// ═══════════════════════════════════════════
reg [`DATA_WIDTH-1:0] mem_array [0:MEM_DEPTH-1];

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
function [8*8:1] fwd_name;
    input [2:0] sel;
    case (sel)
        `FWD_NONE:      fwd_name = "REG     ";
        `FWD_EXMEM:     fwd_name = "EX/MEM  ";
        `FWD_MEMWB:     fwd_name = "ME/WB   ";
        `FWD_BDTU_P1:   fwd_name = "BDTU_P1 ";
        `FWD_BDTU_P2:   fwd_name = "BDTU_P2 ";
        `FWD_EXMEM_P2:  fwd_name = "EXMEM_P2";
        `FWD_MEMWB_P2:  fwd_name = "MEMWB_P2";
        default:        fwd_name = "???     ";
    endcase
endfunction

// ═══════════════════════════════════════════
//  Write-back source name decoder (for trace)
// ═══════════════════════════════════════════
function [5*8:1] wb_sel_name;
    input [2:0] sel;
    case (sel)
        `WB_ALU:  wb_sel_name = "ALU  ";
        `WB_MEM:  wb_sel_name = "MEM  ";
        `WB_LINK: wb_sel_name = "LINK ";
        `WB_PSR:  wb_sel_name = "PSR  ";
        `WB_MUL:  wb_sel_name = "MUL  ";
        default:  wb_sel_name = "???  ";
    endcase
endfunction

// ═══════════════════════════════════════════
//  Program + data loader
//   - BIN_MODE defined  → $readmemb (binary)
//   - BIN_MODE undefined → $readmemh (hex)
// ═══════════════════════════════════════════
task load_program;
    integer k;
begin
    // Zero-initialize entire memory
    for (k = 0; k < MEM_DEPTH; k = k + 1)
        mem_array[k] = {`DATA_WIDTH{1'b0}};

`ifdef BIN_MODE
    $readmemb(FILE_NAME, mem_array);
    $display("  Loaded program (BINARY mode) from: %s", FILE_NAME);
`else
    $readmemh(FILE_NAME, mem_array);
    $display("  Loaded program (HEX mode) from: %s", FILE_NAME);
`endif

    // Sanity check: show first 8 loaded words
    $display("  First 8 words loaded:");
    for (k = 0; k < 8; k = k + 1)
        $display("    [0x%04H] = 0x%08H  (bin: %b)", k << 2, mem_array[k], mem_array[k]);

    // Warn if memory looks empty
    if (mem_array[0] === {`DATA_WIDTH{1'b0}} &&
        mem_array[1] === {`DATA_WIDTH{1'b0}} &&
        mem_array[2] === {`DATA_WIDTH{1'b0}} &&
        mem_array[3] === {`DATA_WIDTH{1'b0}}) begin
        $display("");
        $display("  *** WARNING: First 4 words are all zero. Check file path and format. ***");
        $display("  *** HEX format: one 32-bit hex word per line (e.g. E92D4800)         ***");
        $display("  *** BIN format: one 32-bit binary word per line, compile with -D BIN_MODE ***");
        $display("");
    end
end
endtask

// ═══════════════════════════════════════════
//  Housekeeping
// ═══════════════════════════════════════════
integer cycle_cnt, i;
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
        $display("        EX CTL: wr1=R%0d en1=%b wr2=R%0d en2=%b | mem_rd=%b mem_wr=%b wb_sel=%s | mul_en=%b multi_cyc=%b",
                 u_cpu.wr_addr1_ex, u_cpu.wr_en1_ex,
                 u_cpu.wr_addr2_ex, u_cpu.wr_en2_ex,
                 u_cpu.mem_read_ex, u_cpu.mem_write_ex, wb_sel_name(u_cpu.wb_sel_ex),
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
        $display("        WB: sel=%s data1=0x%08H data2=0x%08H | wr1=R%0d en1=%b wr2=R%0d en2=%b | load_data=0x%08H",
                 wb_sel_name(u_cpu.wb_sel_wb), u_cpu.wb_data1, u_cpu.wb_data2,
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
                 u_cpu.cpsr_flags[`FLAG_N], u_cpu.cpsr_flags[`FLAG_Z],
                 u_cpu.cpsr_flags[`FLAG_C], u_cpu.cpsr_flags[`FLAG_V],
                 u_cpu.cpsr_wen_ex, u_cpu.new_flags);

        // ── Key registers snapshot ────────────────────────────
        if ((cycle_cnt % 10 == 0) || u_cpu.branch_taken_ex || (u_cpu.bdtu_busy && !u_cpu.u_bdtu.state[0]))
            $display("        REGS: R0=%08H R1=%08H R2=%08H R3=%08H SP=%08H FP=%08H LR=%08H",
                     u_cpu.u_regfile.regs[0],  u_cpu.u_regfile.regs[1],
                     u_cpu.u_regfile.regs[2],  u_cpu.u_regfile.regs[3],
                     u_cpu.u_regfile.regs[13], u_cpu.u_regfile.regs[11],
                     u_cpu.u_regfile.regs[14]);
    end

    // ── Compact trace after verbose window ────────────────────
    if (rst_n && (cycle_cnt >= TRACE_CYCLES) && (cycle_cnt < TRACE_CYCLES + COMPACT_WINDOW)) begin
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
    for (r = 0; r < `REG_DEPTH; r = r + 1)
        $display("  |  R%-2d  = 0x%08H  (%0d)", r,
                 u_cpu.u_regfile.regs[r],
                 $signed(u_cpu.u_regfile.regs[r]));
    $display("  +-----------------------------------------+");
    $display("  |  CPSR  = %04b (N=%b Z=%b C=%b V=%b)    |",
             u_cpu.cpsr_flags,
             u_cpu.cpsr_flags[`FLAG_N], u_cpu.cpsr_flags[`FLAG_Z],
             u_cpu.cpsr_flags[`FLAG_C], u_cpu.cpsr_flags[`FLAG_V]);
    $display("  +-----------------------------------------+");
end
endtask

task dump_mem_region;
    input [`DMEM_ADDR_WIDTH-1:0] byte_start;
    input [`DMEM_ADDR_WIDTH-1:0] byte_end;
    integer addr;
begin
    $display("  ── Memory dump (0x%04H..0x%04H) ──", byte_start, byte_end);
    for (addr = byte_start >> 2; addr <= byte_end >> 2; addr = addr + 1)
        $display("    [0x%04H] = 0x%08H  (%0d)",
                 addr << 2, mem_array[addr], $signed(mem_array[addr]));
end
endtask

task dump_nonzero_mem;
    integer w, printed;
begin
    printed = 0;
    $display("  ── Non-zero memory words (first 64 shown) ──");
    for (w = 0; w < MEM_DEPTH && printed < 64; w = w + 1) begin
        if (mem_array[w] !== {`DATA_WIDTH{1'b0}}) begin
            $display("    [0x%04H] = 0x%08H  (%0d)",
                     w << 2, mem_array[w], $signed(mem_array[w]));
            printed = printed + 1;
        end
    end
    if (printed == 0)
        $display("    (all memory is zero)");
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
    $display("  ARM CPU Testbench  —  Generic hex/binary file loader");
    $display("  Memory file   : %s", FILE_NAME);
`ifdef BIN_MODE
    $display("  File format   : BINARY ($readmemb)");
`else
    $display("  File format   : HEX ($readmemh)");
`endif
    $display("  CPU_DONE_PC   : 0x%08H", `CPU_DONE_PC);
    $display("  Memory depth  : %0d words (%0d bytes)", MEM_DEPTH, MEM_DEPTH * 4);
    $display("  Timeout       : %0d cycles", TIMEOUT);
    $display("  Trace cycles  : %0d (verbose) + %0d (compact)", TRACE_CYCLES, COMPACT_WINDOW);
    $display("════════════════════════════════════════════════════════════════");
    $display("");

    rst_n         = 1'b0;
    ila_debug_sel = {(`REG_ADDR_WIDTH+1){1'b0}};
    cycle_cnt     = 0;
    stuck_cnt     = 0;
    prev_pc       = {`PC_WIDTH{1'b0}};

    load_program();

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
                dump_nonzero_mem();
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
                dump_nonzero_mem();
                disable sim_loop;
            end
        end
    end

    // ═══════════════════════════════════════════════════════
    //  E N D - O F - R U N   D U M P
    // ═══════════════════════════════════════════════════════
    $display("");
    $display("════════════════════════════════════════════════════════════════");
    $display("  END-OF-RUN DUMP  (%0d cycles)", cycle_cnt);
    $display("════════════════════════════════════════════════════════════════");

    dump_regs();
    $display("");
    dump_nonzero_mem();
    $display("");

    $display("════════════════════════════════════════════════════════════════");
    $display("  Run completed in %0d cycles.  Final PC = 0x%08H", cycle_cnt, i_mem_addr);
    $display("════════════════════════════════════════════════════════════════");
    $display("");

    #(CLK_PERIOD * 5);
    $finish;
end

endmodule