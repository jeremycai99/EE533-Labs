/*  cpu_sort_tb.v — Bubble-sort-only testbench (6-stage pipeline)
 *
 *  Updated for CPU v2.1: EX1/EX2 split pipeline + HDU module.
 *    - Removed ila_debug_sel / ila_debug_data ports (not in CPU).
 *    - Replaced stall_ex / flush_idex / flush_exmem with
 *      stall_ex1, stall_ex2, flush_idex1, flush_ex1ex2.
 *    - Hazard signals now accessed through u_cpu.u_hdu hierarchy
 *      (ex2_ex1_hazard, mem_load_ex1_hazard, mc_ex2_hazard,
 *       hazard_stall, branch_flush, bdtu_stall).
 *    - branch_taken_ex → branch_taken_ex2.
 *    - Trace shows both EX1 (forwarding/shifter) and EX2 (ALU/cond).
 *    - mul_en_ex removed (was already dead in v1.3).
 *
 *  NOTE: The HDU internal signal names referenced via u_cpu.u_hdu.*
 *  (e.g. ex2_ex1_hazard, mc_ex2_hazard, hazard_stall, branch_flush,
 *  bdtu_stall) must match the actual wire names inside hdu.v.
 *  Adjust if your HDU uses different naming.
 *
 *  KEY FIX: Places a halt instruction (B .) at address 0x200,
 *  matching the program's "MOV LR, #0x200".  Without this, BX LR
 *  falls into zeroed memory and the CPU never stops.
 *
 *  COMPILATION:
 *      iverilog -o cpu_sort_tb cpu_sort_tb.v
 *      vvp cpu_sort_tb
 *
 *  To use asynchronous (combinational) memory reads:
 *      iverilog -DASYNC_MEM -o cpu_sort_tb cpu_sort_tb.v
 *
 *  To override the hex file:
 *      iverilog -DMEM_FILE=\"my_imem.txt\" -o cpu_sort_tb cpu_sort_tb.v
 */
`timescale 1ns / 1ps
`include "define.v"
`include "cpu.v"

module cpu_sort_tb;

// ═══════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════
parameter CLK_PERIOD      = 10;
parameter MEM_DEPTH       = 4096;
parameter SORT_ARRAY_SIZE = 10;
parameter MAX_SORT_ELEMS  = 64;

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

parameter STATUS_INTERVAL = 1_000;
parameter COMPACT_WINDOW  = 2000;

// ── Halt configuration ──
parameter [31:0] HALT_ADDR  = 32'h0000_0200;
parameter [31:0] HALT_INSTR = 32'hEAFF_FFFE;  // B . (branch-to-self)

`ifdef MEM_FILE
parameter FILE_NAME = `MEM_FILE;
`else
parameter FILE_NAME = "../hex/sort_imem.txt";
`endif

// ═══════════════════════════════════════════
//  DUT signals
// ═══════════════════════════════════════════
reg                          clk, rst_n;
wire [`PC_WIDTH-1:0]         i_mem_addr;
wire [`CPU_DMEM_ADDR_WIDTH-1:0]  d_mem_addr;
wire [`DATA_WIDTH-1:0]       d_mem_wdata;
wire                         d_mem_wen;
wire [1:0]                   d_mem_size;
wire                         cpu_done_w;

// ═══════════════════════════════════════════
//  Memory model signal declarations
// ═══════════════════════════════════════════
reg  [`INSTR_WIDTH-1:0]      i_mem_data;
reg  [`DATA_WIDTH-1:0]       d_mem_rdata;

// ═══════════════════════════════════════════
//  Memory
// ═══════════════════════════════════════════
reg [`DATA_WIDTH-1:0] mem_array [0:MEM_DEPTH-1];

wire [31:0] i_word_addr = (i_mem_addr >> 2) & (MEM_DEPTH-1);
wire [31:0] d_word_addr = (d_mem_addr >> 2) & (MEM_DEPTH-1);

// ═══════════════════════════════════════════
//  Sorted-result snapshot
// ═══════════════════════════════════════════
reg [`DATA_WIDTH-1:0] sorted_result [0:MAX_SORT_ELEMS-1];
integer sort_count;

// ═══════════════════════════════════════════
//  Expected sorted output (ascending, signed)
//  Input:  {323,123,-455,2,98,125,10,65,-56,0}
//  Sorted: {-455,-56,0,2,10,65,98,123,125,323}
// ═══════════════════════════════════════════
reg [`DATA_WIDTH-1:0] expected_sorted [0:SORT_ARRAY_SIZE-1];

// ═══════════════════════════════════════════
//  DUT — no ila_debug ports in v2.x
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
    .cpu_done      (cpu_done_w)
);

// ═══════════════════════════════════════════
//  Clock
// ═══════════════════════════════════════════
initial clk = 1'b0;
always #(CLK_PERIOD/2) clk = ~clk;

// ═══════════════════════════════════════════════════════════════════
//  Synchronous / Asynchronous Memory Model
// ═══════════════════════════════════════════════════════════════════

`ifdef ASYNC_MEM
    // ── Asynchronous (combinational) reads ──
    always @(*) begin
        i_mem_data  = mem_array[i_word_addr];
        d_mem_rdata = mem_array[d_word_addr];
    end

    always @(posedge clk) begin
        if (d_mem_wen) begin
            case (d_mem_size)
                2'b00: begin
                    case (d_mem_addr[1:0])
                        2'b00: mem_array[d_word_addr][ 7: 0] <= d_mem_wdata[7:0];
                        2'b01: mem_array[d_word_addr][15: 8] <= d_mem_wdata[7:0];
                        2'b10: mem_array[d_word_addr][23:16] <= d_mem_wdata[7:0];
                        2'b11: mem_array[d_word_addr][31:24] <= d_mem_wdata[7:0];
                    endcase
                end
                2'b01: begin
                    if (d_mem_addr[1])
                        mem_array[d_word_addr][31:16] <= d_mem_wdata[15:0];
                    else
                        mem_array[d_word_addr][15: 0] <= d_mem_wdata[15:0];
                end
                default: begin
                    mem_array[d_word_addr] <= d_mem_wdata;
                end
            endcase
        end
    end

    initial begin
        $display("  *** MEMORY MODE: ASYNCHRONOUS (combinational reads) ***");
    end
`else
    // ── Synchronous reads (BRAM-style, 1-cycle latency, read-first) ──
    always @(posedge clk) begin
        i_mem_data  <= mem_array[i_word_addr];
    end

    always @(posedge clk) begin
        d_mem_rdata <= mem_array[d_word_addr];

        if (d_mem_wen) begin
            case (d_mem_size)
                2'b00: begin
                    case (d_mem_addr[1:0])
                        2'b00: mem_array[d_word_addr][ 7: 0] <= d_mem_wdata[7:0];
                        2'b01: mem_array[d_word_addr][15: 8] <= d_mem_wdata[7:0];
                        2'b10: mem_array[d_word_addr][23:16] <= d_mem_wdata[7:0];
                        2'b11: mem_array[d_word_addr][31:24] <= d_mem_wdata[7:0];
                    endcase
                end
                2'b01: begin
                    if (d_mem_addr[1])
                        mem_array[d_word_addr][31:16] <= d_mem_wdata[15:0];
                    else
                        mem_array[d_word_addr][15: 0] <= d_mem_wdata[15:0];
                end
                default: begin
                    mem_array[d_word_addr] <= d_mem_wdata;
                end
            endcase
        end
    end

    initial begin
        $display("  *** MEMORY MODE: SYNCHRONOUS (1-cycle read latency, read-first) ***");
    end
`endif

// ═══════════════════════════════════════════
//  State/name decoders
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
//  Housekeeping
// ═══════════════════════════════════════════
integer cycle_cnt, i;
reg [`PC_WIDTH-1:0] prev_pc;
integer stuck_cnt;
integer total_cycles;
reg trace_enable;
integer run_exit_status;  // 0=done 1=stuck 2=timeout

// ═══════════════════════════════════════════
//  Per-cycle detailed trace (gated by trace_enable)
//  Updated for 6-stage pipeline: EX1 + EX2
//  HDU signals accessed via u_cpu.u_hdu hierarchy (v2.1)
// ═══════════════════════════════════════════
always @(posedge clk) begin
    if (rst_n && trace_enable && (cycle_cnt < TRACE_CYCLES)) begin
        $display("────────────────────────────────────────────────────────────────────────────────");
        $display("[C%04d] IF: PC=0x%08H  |  ID: instr=0x%08H valid=%b  |  EX2: alu_res=0x%08H cond=%b  |  MEM: addr=0x%08H  |  WB: data1=0x%08H",
                 cycle_cnt, i_mem_addr,
                 u_cpu.instr_id, u_cpu.ifid_valid,
                 u_cpu.alu_result_ex2, u_cpu.cond_met_ex2,
                 u_cpu.mem_addr_mem, u_cpu.wb_data1);
        $display("        CTRL: stall_if=%b stall_id=%b stall_ex1=%b stall_ex2=%b stall_mem=%b | flush_ifid=%b flush_idex1=%b flush_ex1ex2=%b",
                 u_cpu.stall_if, u_cpu.stall_id, u_cpu.stall_ex1, u_cpu.stall_ex2, u_cpu.stall_mem,
                 u_cpu.flush_ifid, u_cpu.flush_idex1, u_cpu.flush_ex1ex2);
        $display("        HDU:  ex2_ex1_haz=%b mem_ld_ex1_haz=%b mc_ex2_haz=%b hazard_stall=%b | branch_flush=%b bdtu_busy=%b",
                 u_cpu.u_hdu.ex2_ex1_hazard, u_cpu.u_hdu.mem_load_ex1_hazard,
                 u_cpu.u_hdu.mc_ex2_hazard, u_cpu.u_hdu.hazard_stall,
                 u_cpu.u_hdu.branch_flush, u_cpu.bdtu_busy);
        $display("        BDTU: state=%s busy=%b start=%b | addr=0x%08H rd=%b wr=%b wdata=0x%08H rdata=0x%08H",
                 bdtu_state_name(u_cpu.u_bdtu.state),
                 u_cpu.bdtu_busy, u_cpu.is_multi_cycle_mem,
                 u_cpu.u_bdtu.mem_addr, u_cpu.u_bdtu.mem_rd, u_cpu.u_bdtu.mem_wr,
                 u_cpu.u_bdtu.mem_wdata, d_mem_rdata);
        if (u_cpu.bdtu_wr_en1 || u_cpu.bdtu_wr_en2)
            $display("        BDTU WR: port1: R%0d<=0x%08H en=%b | port2: R%0d<=0x%08H en=%b",
                     u_cpu.bdtu_wr_addr1, u_cpu.bdtu_wr_data1, u_cpu.bdtu_wr_en1,
                     u_cpu.bdtu_wr_addr2, u_cpu.bdtu_wr_data2, u_cpu.bdtu_wr_en2);
        if (u_cpu.branch_taken_ex2)
            $display("        BRANCH: taken=%b target=0x%08H (link=%b)",
                     u_cpu.branch_taken_ex2, u_cpu.branch_target_ex2_wire,
                     u_cpu.branch_link_ex2);
        $display("        ID READ: Rn[R%0d]=0x%08H  Rm[R%0d]=0x%08H  R3port[R%0d]=0x%08H | use: rn=%b rm=%b rs=%b rd=%b",
                 u_cpu.rn_addr_id, u_cpu.rn_data_id,
                 u_cpu.rm_addr_id, u_cpu.rm_data_id,
                 u_cpu.r3addr_mux, u_cpu.r3_data_id,
                 u_cpu.use_rn_id, u_cpu.use_rm_id, u_cpu.use_rs_id, u_cpu.use_rd_id);
        $display("        EX1 FWD: fwd_a=%s fwd_b=%s fwd_s=%s fwd_d=%s",
                 fwd_name(u_cpu.fwd_a), fwd_name(u_cpu.fwd_b),
                 fwd_name(u_cpu.fwd_s), fwd_name(u_cpu.fwd_d));
        $display("        EX1 OPS: rn_fwd=0x%08H rm_fwd=0x%08H rs_fwd=0x%08H rd_st_fwd=0x%08H",
                 u_cpu.rn_fwd, u_cpu.rm_fwd, u_cpu.rs_fwd, u_cpu.rd_store_fwd);
        $display("        EX1 BS:  rm_shifted=0x%08H cout=%b | alu_b_sel=0x%08H | valid=%b cond=%04b",
                 u_cpu.bs_dout, u_cpu.shifter_cout, u_cpu.alu_src_b_val_ex1,
                 u_cpu.valid_ex1, u_cpu.cond_code_ex1);
        $display("        EX2 ALU: a=0x%08H b=0x%08H op=%04b => res=0x%08H flags=%04b | cond=%04b met=%b valid=%b",
                 u_cpu.rn_fwd_ex2, u_cpu.alu_b_ex2, u_cpu.alu_op_ex2,
                 u_cpu.alu_result_ex2, u_cpu.alu_flags_ex2,
                 u_cpu.cond_code_ex2, u_cpu.cond_met_raw_ex2, u_cpu.valid_ex2);
        $display("        EX2 CTL: wr1=R%0d en1=%b(g:%b) wr2=R%0d en2=%b(g:%b) | mem_rd=%b(g:%b) mem_wr=%b(g:%b) wb_sel=%s | multi_cyc=%b(g:%b)",
                 u_cpu.wr_addr1_ex2, u_cpu.wr_en1_ex2, u_cpu.wr_en1_gated_ex2,
                 u_cpu.wr_addr2_ex2, u_cpu.wr_en2_ex2, u_cpu.wr_en2_gated_ex2,
                 u_cpu.mem_read_ex2, u_cpu.mem_read_gated_ex2,
                 u_cpu.mem_write_ex2, u_cpu.mem_write_gated_ex2,
                 wb_sel_name(u_cpu.wb_sel_ex2),
                 u_cpu.is_multi_cycle_ex2, u_cpu.is_multi_cycle_gated_ex2);
        $display("        MEM: alu_res=0x%08H addr=0x%08H store=0x%08H | rd=%b wr=%b size=%02b signed=%b | wr1=R%0d en1=%b wr2=R%0d en2=%b",
                 u_cpu.alu_result_mem, u_cpu.mem_addr_mem, u_cpu.store_data_mem,
                 u_cpu.mem_read_mem, u_cpu.mem_write_mem, u_cpu.mem_size_mem, u_cpu.mem_signed_mem,
                 u_cpu.wr_addr1_mem, u_cpu.wr_en1_mem,
                 u_cpu.wr_addr2_mem, u_cpu.wr_en2_mem);
        if (d_mem_wen)
            $display("        DMEM WRITE: [0x%08H] <= 0x%08H  size=%02b",
                     d_mem_addr, d_mem_wdata, d_mem_size);
        $display("        DMEM READ:  [0x%08H] => 0x%08H", d_mem_addr, d_mem_rdata);
        $display("        WB: sel=%s data1=0x%08H data2=0x%08H | wr1=R%0d en1=%b wr2=R%0d en2=%b | load_data=0x%08H",
                 wb_sel_name(u_cpu.wb_sel_wb), u_cpu.wb_data1, u_cpu.wb_data2,
                 u_cpu.wr_addr1_wb, u_cpu.wr_en1_wb,
                 u_cpu.wr_addr2_wb, u_cpu.wr_en2_wb,
                 u_cpu.load_data_wb);
        if (u_cpu.rf_wr_en)
            $display("        RF WRITE: port1: R%0d<=0x%08H | port2: R%0d<=0x%08H  (bdtu_prio=%b)",
                     u_cpu.rf_wr_addr1, u_cpu.rf_wr_data1,
                     u_cpu.rf_wr_addr2, u_cpu.rf_wr_data2,
                     u_cpu.bdtu_busy);
        $display("        CPSR: %04b (N=%b Z=%b C=%b V=%b) | cpsr_wen_g=%b alu_flags=%04b psr_wr_flags=%b",
                 u_cpu.cpsr_flags,
                 u_cpu.cpsr_flags[`FLAG_N], u_cpu.cpsr_flags[`FLAG_Z],
                 u_cpu.cpsr_flags[`FLAG_C], u_cpu.cpsr_flags[`FLAG_V],
                 u_cpu.cpsr_wen_gated_ex2, u_cpu.alu_flags_ex2,
                 u_cpu.psr_wr_flags_ex2);
        if ((cycle_cnt % 10 == 0) || u_cpu.branch_taken_ex2 || (u_cpu.bdtu_busy && !u_cpu.u_bdtu.state[0]))
            $display("        REGS: R0=%08H R1=%08H R2=%08H R3=%08H SP=%08H FP=%08H LR=%08H",
                     u_cpu.u_regfile.regs[0],  u_cpu.u_regfile.regs[1],
                     u_cpu.u_regfile.regs[2],  u_cpu.u_regfile.regs[3],
                     u_cpu.u_regfile.regs[13], u_cpu.u_regfile.regs[11],
                     u_cpu.u_regfile.regs[14]);
    end

    // Compact trace after verbose window
    if (rst_n && trace_enable && (cycle_cnt >= TRACE_CYCLES) && (cycle_cnt < TRACE_CYCLES + COMPACT_WINDOW)) begin
        if (u_cpu.branch_taken_ex2 || (u_cpu.rf_wr_en && !u_cpu.bdtu_busy) ||
            (u_cpu.bdtu_busy && u_cpu.u_bdtu.state == 3'd7) ||
            (cycle_cnt % 50 == 0))
            $display("[C%04d] PC=0x%08H instr=0x%08H | branch=%b bdtu=%s | WR: R%0d<=0x%08H en=%b | CPSR=%04b",
                     cycle_cnt, i_mem_addr, u_cpu.instr_id,
                     u_cpu.branch_taken_ex2,
                     bdtu_state_name(u_cpu.u_bdtu.state),
                     u_cpu.rf_wr_addr1, u_cpu.rf_wr_data1, u_cpu.rf_wr_en,
                     u_cpu.cpsr_flags);
    end

    // Sparse trace after compact window
    if (rst_n && trace_enable && (cycle_cnt >= TRACE_CYCLES + COMPACT_WINDOW)) begin
        if (u_cpu.branch_taken_ex2 || (cycle_cnt % 200 == 0))
            $display("[C%05d] PC=0x%08H instr=0x%08H | branch=%b | CPSR=%04b | R0=%08H R1=%08H R2=%08H R3=%08H",
                     cycle_cnt, i_mem_addr, u_cpu.instr_id,
                     u_cpu.branch_taken_ex2, u_cpu.cpsr_flags,
                     u_cpu.u_regfile.regs[0], u_cpu.u_regfile.regs[1],
                     u_cpu.u_regfile.regs[2], u_cpu.u_regfile.regs[3]);
    end

    // Periodic status
    if (rst_n && trace_enable && (cycle_cnt > 0) && (cycle_cnt % STATUS_INTERVAL == 0)) begin
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
//  Helper tasks
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
//  Zero all CPU registers via hierarchical access
// ═══════════════════════════════════════════
task init_cpu_regs;
    integer r;
begin
    for (r = 0; r < `REG_DEPTH; r = r + 1)
        u_cpu.u_regfile.regs[r] = {`DATA_WIDTH{1'b0}};
    // Safety: pre-set LR to halt address.
    u_cpu.u_regfile.regs[14] = HALT_ADDR;
end
endtask

// ═══════════════════════════════════════════
//  Load hex/bin file into memory
// ═══════════════════════════════════════════
task load_program_file;
    input [256*8:1] filename;
    integer k;
begin
    for (k = 0; k < MEM_DEPTH; k = k + 1)
        mem_array[k] = {`DATA_WIDTH{1'b0}};

`ifdef BIN_MODE
    $readmemb(filename, mem_array);
    $display("  Loaded (BINARY): %0s", filename);
`else
    $readmemh(filename, mem_array);
    $display("  Loaded (HEX): %0s", filename);
`endif

    $display("  First 8 words:");
    for (k = 0; k < 8; k = k + 1)
        $display("    [0x%04H] = 0x%08H", k << 2, mem_array[k]);

    if (mem_array[0] === {`DATA_WIDTH{1'b0}} &&
        mem_array[1] === {`DATA_WIDTH{1'b0}}) begin
        $display("  *** WARNING: Memory looks empty — check file path. ***");
    end

    // ── KEY FIX: Place halt instruction at return address ──
    $display("");
    $display("  Placing halt (B . = 0x%08H) at address 0x%08H (word %0d)",
             HALT_INSTR, HALT_ADDR, HALT_ADDR >> 2);
    mem_array[HALT_ADDR >> 2] = HALT_INSTR;
end
endtask

// ═══════════════════════════════════════════
//  Search entire memory for the expected
//  sorted sequence and snapshot it
// ═══════════════════════════════════════════
task find_and_snapshot_sorted;
    integer w, e, found_addr;
    reg found, match;
begin
    found      = 1'b0;
    found_addr = 0;

    for (w = 0; w < MEM_DEPTH - SORT_ARRAY_SIZE + 1; w = w + 1) begin
        if (!found) begin
            match = 1'b1;
            for (e = 0; e < SORT_ARRAY_SIZE; e = e + 1) begin
                if (mem_array[w + e] !== expected_sorted[e])
                    match = 1'b0;
            end
            if (match) begin
                found      = 1'b1;
                found_addr = w;
            end
        end
    end

    if (found) begin
        $display("  ✓ FOUND sorted array at word %0d (byte 0x%04H):", found_addr, found_addr << 2);
        for (e = 0; e < SORT_ARRAY_SIZE; e = e + 1) begin
            sorted_result[e] = mem_array[found_addr + e];
            $display("    [%0d] = 0x%08H  (%0d)", e,
                     mem_array[found_addr + e], $signed(mem_array[found_addr + e]));
        end
        sort_count = SORT_ARRAY_SIZE;
    end else begin
        $display("  ✗ Sorted array NOT FOUND in memory.");
        $display("    Dumping non-zero memory for inspection:");
        dump_nonzero_mem();
        sort_count = 0;
    end
end
endtask

// ═══════════════════════════════════════════
//  Run program to completion
// ═══════════════════════════════════════════
task run_program;
begin
    cycle_cnt       = 0;
    stuck_cnt       = 0;
    prev_pc         = {`PC_WIDTH{1'b0}};
    run_exit_status = 2;

    // Assert reset
    rst_n = 1'b0;

    // Zero register file
    init_cpu_regs();

    repeat (5) @(posedge clk);

    // Re-init after clock edges (in case reset logic clears regs)
    init_cpu_regs();

    // Allow sync memory one cycle to register the first instruction
    @(posedge clk);

    // Force LR one final time right before releasing reset
    @(negedge clk);
    u_cpu.u_regfile.regs[14] = HALT_ADDR;
    rst_n = 1'b1;

    $display("[%0t] Reset released — Bubble Sort (6-stage pipeline)", $time);
    $display("  halt_addr = 0x%08H", HALT_ADDR);
`ifdef ASYNC_MEM
    $display("  (async memory — data available same cycle as address)");
`else
    $display("  (sync memory  — data available one cycle after address)");
`endif
    $display("");

    begin : run_loop
        forever begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;

            // Stuck-PC detector
            if (i_mem_addr === prev_pc)
                stuck_cnt = stuck_cnt + 1;
            else
                stuck_cnt = 0;
            prev_pc = i_mem_addr;

            // ── Detection 1: cpu_done signal ──
            if (cpu_done_w) begin
                $display("");
                $display("[%0t] cpu_done asserted at cycle %0d (PC=0x%08H)",
                         $time, cycle_cnt, i_mem_addr);
                run_exit_status = 0;
                repeat (10) @(posedge clk);
                disable run_loop;
            end

            // ── Detection 2: PC reached halt address ──
            if (i_mem_addr == HALT_ADDR && cycle_cnt > 20) begin
                $display("");
                $display("[%0t] PC reached halt address 0x%08H at cycle %0d",
                         $time, HALT_ADDR, cycle_cnt);
                run_exit_status = 0;
                repeat (10) @(posedge clk);
                disable run_loop;
            end

            // ── Detection 3: stuck at unexpected address ──
            if (stuck_cnt > 500) begin
                $display("");
                $display("╔══════════════════════════════════════════════════╗");
                $display("║  *** STUCK: PC=0x%08H for %0d cycles ***", i_mem_addr, stuck_cnt);
                $display("╚══════════════════════════════════════════════════╝");
                $display("  BDTU state=%s busy=%b start(is_mc_mem)=%b",
                         bdtu_state_name(u_cpu.u_bdtu.state),
                         u_cpu.bdtu_busy, u_cpu.is_multi_cycle_mem);
                $display("  Stalls: if=%b id=%b ex1=%b ex2=%b mem=%b",
                         u_cpu.stall_if, u_cpu.stall_id, u_cpu.stall_ex1,
                         u_cpu.stall_ex2, u_cpu.stall_mem);
                $display("  Hazards: ex2_ex1=%b mem_ld_ex1=%b mc_ex2=%b hazard_stall=%b branch_flush=%b",
                         u_cpu.u_hdu.ex2_ex1_hazard, u_cpu.u_hdu.mem_load_ex1_hazard,
                         u_cpu.u_hdu.mc_ex2_hazard, u_cpu.u_hdu.hazard_stall,
                         u_cpu.u_hdu.branch_flush);
                $display("  EX1: wr1=R%0d en=%b  branch_en=%b  multi_cyc=%b  valid=%b",
                         u_cpu.wr_addr1_ex1, u_cpu.wr_en1_ex1,
                         u_cpu.branch_en_ex1, u_cpu.is_multi_cycle_ex1,
                         u_cpu.valid_ex1);
                $display("  EX2: wr1=R%0d en=%b(g:%b) branch=%b cond_met=%b valid=%b multi_cyc=%b(g:%b)",
                         u_cpu.wr_addr1_ex2, u_cpu.wr_en1_ex2, u_cpu.wr_en1_gated_ex2,
                         u_cpu.branch_taken_ex2, u_cpu.cond_met_ex2, u_cpu.valid_ex2,
                         u_cpu.is_multi_cycle_ex2, u_cpu.is_multi_cycle_gated_ex2);
                $display("  MEM: wr1=R%0d en=%b  rd=%b wr=%b  multi_cyc=%b",
                         u_cpu.wr_addr1_mem, u_cpu.wr_en1_mem,
                         u_cpu.mem_read_mem, u_cpu.mem_write_mem, u_cpu.is_multi_cycle_mem);
                dump_regs();
                dump_nonzero_mem();
                run_exit_status = 1;
                repeat (5) @(posedge clk);
                disable run_loop;
            end

            // ── Detection 4: timeout ──
            if (cycle_cnt >= TIMEOUT) begin
                $display("");
                $display("*** TIMEOUT after %0d cycles (PC=0x%04H) ***",
                         TIMEOUT, i_mem_addr);
                dump_regs();
                dump_nonzero_mem();
                run_exit_status = 2;
                disable run_loop;
            end
        end
    end

    $display("");
    $display("════════════════════════════════════════════════════════════════");
    $display("  END-OF-RUN: Bubble Sort  (%0d cycles, exit=%0d)", cycle_cnt,
             run_exit_status);
    $display("════════════════════════════════════════════════════════════════");
    dump_regs();
    $display("");
    dump_nonzero_mem();
    $display("");
end
endtask

// ═══════════════════════════════════════════
//  Verify array is in ascending signed order
// ═══════════════════════════════════════════
task verify_sort_order;
    integer idx, errors;
    reg [`DATA_WIDTH-1:0] cur, nxt;
begin
    errors = 0;
    if (sort_count == 0) begin
        $display("  Bubble Sort: NO DATA — skipping order check");
    end else begin
        $display("  Verifying sort order (%0d elements)...", sort_count);
        for (idx = 0; idx < sort_count - 1; idx = idx + 1) begin
            cur = sorted_result[idx];
            nxt = sorted_result[idx+1];
            if ($signed(cur) > $signed(nxt)) begin
                $display("    *** ORDER ERROR at [%0d]: %0d > [%0d]: %0d",
                         idx, $signed(cur), idx+1, $signed(nxt));
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("    PASS: Correctly sorted in ascending order.");
        else              $display("    FAIL: %0d order violation(s).", errors);
    end
end
endtask

// ═══════════════════════════════════════════
//  Compare against known expected output
// ═══════════════════════════════════════════
task verify_against_expected;
    integer idx, errors;
    reg [`DATA_WIDTH-1:0] val;
begin
    errors = 0;
    if (sort_count == 0) begin
        $display("  Bubble Sort: NO DATA — skipping expected-value check");
    end else begin
        $display("  Comparing output against expected sorted array...");
        for (idx = 0; idx < sort_count && idx < SORT_ARRAY_SIZE; idx = idx + 1) begin
            val = sorted_result[idx];
            if (val !== expected_sorted[idx]) begin
                $display("    [FAIL] arr[%0d] = %0d, expected %0d",
                         idx, $signed(val), $signed(expected_sorted[idx]));
                errors = errors + 1;
            end else begin
                $display("    [PASS] arr[%0d] = %0d", idx, $signed(val));
            end
        end
        if (errors == 0) $display("    ✓ ALL %0d ELEMENTS MATCH.", sort_count);
        else              $display("    ✗ %0d MISMATCH(ES).", errors);
    end
end
endtask

// ═══════════════════════════════════════════
//  Main stimulus
// ═══════════════════════════════════════════
initial begin
    // ── VCD dump — exclude mem_array to avoid multi-GB file ──
    $dumpfile("cpu_sort_tb.vcd");
    $dumpvars(1, cpu_sort_tb);
    $dumpvars(0, u_cpu);

    // Expected sorted output: {-455,-56,0,2,10,65,98,123,125,323}
    expected_sorted[0] = 32'hfffffe39;  // -455
    expected_sorted[1] = 32'hffffffc8;  // -56
    expected_sorted[2] = 32'h00000000;  //  0
    expected_sorted[3] = 32'h00000002;  //  2
    expected_sorted[4] = 32'h0000000a;  //  10
    expected_sorted[5] = 32'h00000041;  //  65
    expected_sorted[6] = 32'h00000062;  //  98
    expected_sorted[7] = 32'h0000007b;  //  123
    expected_sorted[8] = 32'h0000007d;  //  125
    expected_sorted[9] = 32'h00000143;  //  323

    $display("");
    $display("╔══════════════════════════════════════════════════════════════╗");
    $display("║   ARM CPU Testbench — Bubble Sort (6-stage pipeline v2.1)  ║");
    $display("╠══════════════════════════════════════════════════════════════╣");
    $display("║  Pipeline  : IF → ID → EX1 → EX2 → MEM → WB              ║");
    $display("║  Program   : %-45s ║", FILE_NAME);
`ifdef BIN_MODE
    $display("║  Format    : BINARY ($readmemb)                            ║");
`else
    $display("║  Format    : HEX ($readmemh)                               ║");
`endif
`ifdef ASYNC_MEM
    $display("║  Memory    : ASYNC (combinational reads)                   ║");
`else
    $display("║  Memory    : SYNC (1-cycle read latency, read-first)       ║");
`endif
    $display("║  Halt addr : 0x%08H                                    ║", HALT_ADDR);
    $display("║  CPU_DONE  : 0x%08H                                    ║", `CPU_DONE_PC);
    $display("║  Mem depth : %0d words (%0d bytes)                     ║", MEM_DEPTH, MEM_DEPTH * 4);
    $display("║  Timeout   : %0d cycles                                ║", TIMEOUT);
    $display("║  Trace     : %0d (verbose) + %0d (compact)             ║", TRACE_CYCLES, COMPACT_WINDOW);
    $display("╚══════════════════════════════════════════════════════════════╝");
    $display("");

    rst_n         = 1'b0;
    trace_enable  = 1'b1;
    sort_count    = 0;

    // ═══════════════════════════════════════
    //  Load and run Bubble Sort
    // ═══════════════════════════════════════
    $display("████████████████████████████████████████████████████████████████");
    $display("██  Bubble Sort (%0s)", FILE_NAME);
    $display("████████████████████████████████████████████████████████████████");
    $display("");

    load_program_file(FILE_NAME);
    run_program();
    total_cycles = cycle_cnt;

    $display("  --- Searching memory for sorted output ---");
    find_and_snapshot_sorted();

    // ═══════════════════════════════════════
    //  Verification
    // ═══════════════════════════════════════
    $display("");
    $display("╔══════════════════════════════════════════════════════════════╗");
    $display("║                    VERIFICATION                             ║");
    $display("╚══════════════════════════════════════════════════════════════╝");
    $display("");
    verify_sort_order();
    $display("");
    verify_against_expected();

    // ═══════════════════════════════════════
    //  Summary
    // ═══════════════════════════════════════
    $display("");
    $display("╔══════════════════════════════════════════════════════════════╗");
    $display("║                      SUMMARY                                ║");
    $display("╠══════════════════════════════════════════════════════════════╣");
    $display("║  Pipeline  : IF → ID → EX1 → EX2 → MEM → WB  (v2.1)      ║");
    $display("║  Cycles     : %6d                                      ║", total_cycles);
    case (run_exit_status)
        0: $display("║  Exit       : CLEAN (cpu_done / halt reached)               ║");
        1: $display("║  Exit       : STUCK (PC looped at unexpected address)        ║");
        2: $display("║  Exit       : TIMEOUT (%0d cycles)                       ║", TIMEOUT);
    endcase
    if (sort_count > 0)
        $display("║  Sort result: FOUND (%0d elements)                           ║", sort_count);
    else
        $display("║  Sort result: NOT FOUND                                        ║");
    $display("╚══════════════════════════════════════════════════════════════╝");
    $display("");

    #(CLK_PERIOD * 5);
    $finish;
end

endmodule