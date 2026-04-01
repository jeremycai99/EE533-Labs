/* file: sm_core_tb.v
 * Testbench for sm_core — streaming multiprocessor core.
 * Instantiates sm_core + IMEM/DMEM BRAM models, loads test
 * programs, launches kernels, and verifies GPR / DMEM results.
 *
 * CRITICAL: Every @(posedge clk) is followed by #1 to prevent
 * Verilog simulation race conditions between TB and DUT NBAs.
 *
 * Author: Jeremy Cai
 * Date: Mar. 1, 2026
 * Version: 1.5
 * Revision history:
 *   - Mar. 01, 2026: v1.0 — K1–K30 tests (ALU, MEM, BRA, WMMA, PTX).
 *   - Mar. 04, 2026: v1.1 — K31–K37 SIMT divergence/convergence tests.
 *   - Mar. 04, 2026: v1.2 — K32/K33 temp workaround (enc_set→enc_setp).
 *   - Mar. 04, 2026: v1.3 — Reverted K32/K33 to enc_set. Added K38.
 *   - Mar. 05, 2026: v1.4 — Removed debug_rf ports. GPR reads via
 *     override RF read ports. Added K39–K47 CVT tests.
 *   - Mar. 06, 2026: v1.5 — Fixed read_gpr: force ovr_sel=1 during
 *     override RF reads. Fixed K16/K20/K22/K23: zero C matrix (R8-R11)
 *     before WMMA.MMA (stale RF values from prior kernels).
 */

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "sm_core.v"

module sm_core_tb;

    // ================================================================
    // Clock / Reset
    // ================================================================
    reg clk, rst_n;
    localparam CLK_PERIOD = 10;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ================================================================
    // DUT I/O
    // ================================================================
    wire [`GPU_IMEM_ADDR_WIDTH-1:0] imem_addr;
    reg [`GPU_IMEM_DATA_WIDTH-1:0] imem_rdata;

    wire [4*`GPU_DMEM_ADDR_WIDTH-1:0] dmem_addra;
    wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_dina;
    wire [3:0] dmem_wea;
    reg [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_douta;

    reg kernel_start;
    reg [`GPU_PC_WIDTH-1:0] kernel_entry_pc;
    reg [3:0] thread_mask;
    wire kernel_done;

    // ================================================================
    // DUT
    // ================================================================
    sm_core u_dut (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr),
        .imem_rdata(imem_rdata),
        .dmem_addra(dmem_addra),
        .dmem_dina(dmem_dina),
        .dmem_wea(dmem_wea),
        .dmem_douta(dmem_douta),
        .kernel_start(kernel_start),
        .kernel_entry_pc(kernel_entry_pc),
        .thread_mask(thread_mask),
        .kernel_done(kernel_done)
    );

    // ================================================================
    // IMEM BRAM Model
    // ================================================================
    reg [31:0] imem [0:255];

    always @(posedge clk)
        imem_rdata <= imem[imem_addr];

    // ================================================================
    // Per-SP DMEM BRAM Models
    // ================================================================
    reg [15:0] dmem0 [0:1023];
    reg [15:0] dmem1 [0:1023];
    reg [15:0] dmem2 [0:1023];
    reg [15:0] dmem3 [0:1023];

    always @(posedge clk) begin
        if (dmem_wea[0])
            dmem0[dmem_addra[0*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]]
                <= dmem_dina[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        dmem_douta[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]
            <= dmem0[dmem_addra[0*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]];
    end
    always @(posedge clk) begin
        if (dmem_wea[1])
            dmem1[dmem_addra[1*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]]
                <= dmem_dina[1*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        dmem_douta[1*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]
            <= dmem1[dmem_addra[1*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]];
    end
    always @(posedge clk) begin
        if (dmem_wea[2])
            dmem2[dmem_addra[2*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]]
                <= dmem_dina[2*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        dmem_douta[2*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]
            <= dmem2[dmem_addra[2*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]];
    end
    always @(posedge clk) begin
        if (dmem_wea[3])
            dmem3[dmem_addra[3*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]]
                <= dmem_dina[3*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        dmem_douta[3*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]
            <= dmem3[dmem_addra[3*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]];
    end

    // ================================================================
    // Instruction Encoding Helpers
    // ================================================================
    function [31:0] enc_r;
        input [4:0] op; input dt; input [3:0] rd, ra, rb;
        enc_r = {op, dt, 2'b00, rd, ra, rb, 12'd0};
    endfunction

    function [31:0] enc_i;
        input [4:0] op; input dt; input [3:0] rd, ra; input [15:0] imm;
        enc_i = {op, dt, 2'b00, rd, ra, imm};
    endfunction

    function [31:0] enc_movi;
        input [3:0] rd; input [15:0] imm;
        enc_movi = {`OP_MOVI, 1'b0, 2'b00, rd, 4'd0, imm};
    endfunction

    function [31:0] enc_m;
        input [4:0] op; input [3:0] rd, ra; input [15:0] offset;
        enc_m = {op, 1'b0, 2'b00, rd, ra, offset};
    endfunction

    function [31:0] enc_m_f;
        input [4:0] op; input [3:0] rd, ra; input [15:0] offset;
        enc_m_f = {op, 1'b1, 2'b00, rd, ra, offset};
    endfunction

    function [31:0] enc_bra;
        input [26:0] target;
        enc_bra = {`OP_BRA, target};
    endfunction

    function [31:0] enc_setp;
        input dt; input [1:0] cmp; input [3:0] pd, ra, rb;
        enc_setp = {`OP_SETP, dt, cmp, pd, ra, rb, 12'd0};
    endfunction

    function [31:0] enc_pbra;
        input [1:0] pred_sel; input [12:0] branch_target; input [11:0] reconv_pc;
        enc_pbra = {`OP_PBRA, pred_sel, branch_target, reconv_pc};
    endfunction

    function [31:0] enc_set;
        input [1:0] pd; input val;
        enc_set = {`OP_SET, 1'b0, 2'b00, {2'b00, pd}, 4'd0, 15'd0, val};
    endfunction

    function [31:0] enc_mov_tid;
        input [3:0] rd;
        enc_mov_tid = {`OP_MOV, 1'b1, 2'b00, rd, 4'd0, 16'd0};
    endfunction

    function [31:0] enc_wmma_mma;
        input [3:0] rD, rA, rB, rC;
        enc_wmma_mma = {`WMMA_MMA, 1'b1, 2'b00, rD, rA, rB, rC, 8'd0};
    endfunction

    function [31:0] enc_cvt;
        input dt; input [3:0] rd, ra;
        enc_cvt = {`OP_CVT, dt, 2'b00, rd, ra, 16'h0000};
    endfunction

    wire [31:0] INST_RET = {`OP_RET, 27'd0};
    wire [31:0] INST_NOP = {`OP_NOP, 27'd0};

    // ================================================================
    // Test Infrastructure
    // ================================================================
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    task tick;
    begin @(posedge clk); #1; end
    endtask

    task reset_dut;
        integer i;
    begin
        rst_n = 0; kernel_start = 0;
        kernel_entry_pc = 32'd0; thread_mask = 4'b1111;
        for (i = 0; i < 256; i = i + 1) imem[i] = INST_NOP;
        repeat (3) tick;
        rst_n = 1; tick;
    end
    endtask

    task clear_dmem;
        integer i;
    begin
        for (i = 0; i < 1024; i = i + 1) begin
            dmem0[i] = 16'd0; dmem1[i] = 16'd0;
            dmem2[i] = 16'd0; dmem3[i] = 16'd0;
        end
    end
    endtask

    task launch_kernel;
        input [31:0] entry_pc;
    begin
        kernel_entry_pc = entry_pc;
        kernel_start = 1; tick;
        kernel_start = 0;
    end
    endtask

    task wait_kernel_done;
        input integer max_cycles;
        integer cyc;
    begin
        cyc = 0;
        while (!kernel_done && cyc < max_cycles) begin tick; cyc = cyc + 1; end
        if (!kernel_done)
            $display("[TIMEOUT] kernel_done not asserted within %0d cycles", max_cycles);
        tick;
    end
    endtask

    // ================================================================
    // GPR Read via Override RF Port (v1.5: force ovr_sel=1)
    // ================================================================
    reg [15:0] dbg_rf_val [0:3];

    task read_gpr;
        input [3:0] addr;
    begin
        force u_dut.ovr_rf_r0_addr_mux = addr;
        force u_dut.ovr_sel = 1'b1;
        #1;
        dbg_rf_val[0] = u_dut.SP_LANE[0].u_sp.ovr_rf_r0_data;
        dbg_rf_val[1] = u_dut.SP_LANE[1].u_sp.ovr_rf_r0_data;
        dbg_rf_val[2] = u_dut.SP_LANE[2].u_sp.ovr_rf_r0_data;
        dbg_rf_val[3] = u_dut.SP_LANE[3].u_sp.ovr_rf_r0_data;
        release u_dut.ovr_rf_r0_addr_mux;
        release u_dut.ovr_sel;
        #1;
    end
    endtask

    task check_gpr;
        input [1:0] sp; input [3:0] addr; input [15:0] expected;
        input [80*8-1:0] test_name;
        reg [15:0] actual;
    begin
        test_num = test_num + 1;
        read_gpr(addr);
        actual = dbg_rf_val[sp];
        if (actual === expected) begin
            $display("[PASS] T%0d: %0s  SP%0d.R%0d = 0x%04h",
                test_num, test_name, sp, addr, actual);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] T%0d: %0s  SP%0d.R%0d = 0x%04h, expected 0x%04h",
                test_num, test_name, sp, addr, actual, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check_gpr_all;
        input [3:0] addr; input [15:0] expected;
        input [80*8-1:0] test_name;
    begin
        check_gpr(2'd0, addr, expected, test_name);
        check_gpr(2'd1, addr, expected, test_name);
        check_gpr(2'd2, addr, expected, test_name);
        check_gpr(2'd3, addr, expected, test_name);
    end
    endtask

    task check_dmem;
        input [1:0] sp; input [9:0] addr; input [15:0] expected;
        input [80*8-1:0] test_name;
        reg [15:0] actual;
    begin
        test_num = test_num + 1;
        case (sp)
            2'd0: actual = dmem0[addr]; 2'd1: actual = dmem1[addr];
            2'd2: actual = dmem2[addr]; 2'd3: actual = dmem3[addr];
        endcase
        if (actual === expected) begin
            $display("[PASS] T%0d: %0s  SP%0d.DMEM[%0d] = 0x%04h",
                test_num, test_name, sp, addr, actual);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] T%0d: %0s  SP%0d.DMEM[%0d] = 0x%04h, expected 0x%04h",
                test_num, test_name, sp, addr, actual, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check_dmem_all;
        input [9:0] addr; input [15:0] expected;
        input [80*8-1:0] test_name;
    begin
        check_dmem(2'd0, addr, expected, test_name);
        check_dmem(2'd1, addr, expected, test_name);
        check_dmem(2'd2, addr, expected, test_name);
        check_dmem(2'd3, addr, expected, test_name);
    end
    endtask

    // ================================================================
    // Waveform Dump
    // ================================================================
    initial begin
        $dumpfile("sm_core_tb.vcd");
        $dumpvars(0, sm_core_tb);
    end

    initial begin
        #2000000;
        $display("[TIMEOUT] Global simulation timeout reached");
        $finish;
    end

    // ================================================================
    // Pipeline Monitor
    // ================================================================
    integer cycle_count;
    reg trace_en;
    reg cvt_trace_en;
    reg tc_trace_en;

    initial begin cycle_count = 0; trace_en = 0; cvt_trace_en = 0; tc_trace_en = 0; end

    reg [8*8-1:0] opname;
    always @(*) begin
        case (u_dut.dec_opcode)
            `OP_NOP: opname = "NOP     "; `OP_MOV: opname = "MOV     ";
            `OP_MOVI: opname = "MOVI    "; `OP_ADD: opname = "ADD     ";
            `OP_SUB: opname = "SUB     "; `OP_MUL: opname = "MUL     ";
            `OP_FMA: opname = "FMA     "; `OP_MAX: opname = "MAX     ";
            `OP_MIN: opname = "MIN     "; `OP_AND: opname = "AND     ";
            `OP_OR: opname = "OR      "; `OP_XOR: opname = "XOR     ";
            `OP_NEG: opname = "NEG     "; `OP_ABS: opname = "ABS     ";
            `OP_SHL: opname = "SHL     "; `OP_SHR: opname = "SHR     ";
            `OP_ADDI: opname = "ADDI    "; `OP_MULI: opname = "MULI    ";
            `OP_LD: opname = "LD      "; `OP_ST: opname = "ST      ";
            `OP_BRA: opname = "BRA     "; `OP_PBRA: opname = "PBRA    ";
            `OP_SETP: opname = "SETP    "; `OP_SELP: opname = "SELP    ";
            `OP_CVT: opname = "CVT     "; `OP_RET: opname = "RET     ";
            `OP_SET: opname = "SET     "; `OP_LDS: opname = "LDS     ";
            `OP_STS: opname = "STS     "; `WMMA_MMA: opname = "WMMA.MMA";
            `WMMA_LOAD: opname = "WMMA.LD "; `WMMA_STORE: opname = "WMMA.ST ";
            default: opname = "???     ";
        endcase
    end

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (trace_en) begin
            $display("  [C%03d] PC=%0d dec=%08h %0s fv=%b | fstl=%b sb=%b exb=%b drn=%b de_fl=%b | mask=%b stk_e=%b conv=%b%b pbra=%b | done=%b",
                cycle_count, u_dut.u_fetch.pc_reg, u_dut.dec_ir, opname,
                u_dut.fetch_valid, u_dut.front_stall, u_dut.sb_stall,
                u_dut.any_ex_busy, u_dut.pipeline_drained, u_dut.de_flush,
                u_dut.active_mask, u_dut.stack_empty,
                u_dut.conv_phase0_fire, u_dut.conv_phase1_fire,
                u_dut.pbra_fire, kernel_done);
            $display("         SB: pend0=%04h pend1=%04h pend2=%04h pend3=%04h | wb_we_any=%b wb_rD=%0d wb_amsk=%b | wb_we[3:0]=%b%b%b%b | DE: v=%b pbra=%b rD=%0d | stk_sp=%0d tos_rpc=%0d tos_ph=%b",
                u_dut.u_sb.pending[0], u_dut.u_sb.pending[1],
                u_dut.u_sb.pending[2], u_dut.u_sb.pending[3],
                u_dut.sb_wb_rf_we_any, u_dut.sp_wb_rD_addr[0],
                {u_dut.sp_wb_active[3], u_dut.sp_wb_active[2],
                 u_dut.sp_wb_active[1], u_dut.sp_wb_active[0]},
                u_dut.sp_wb_rf_we[3], u_dut.sp_wb_rf_we[2],
                u_dut.sp_wb_rf_we[1], u_dut.sp_wb_rf_we[0],
                u_dut.de_valid, u_dut.de_is_pbra, u_dut.de_rD_addr,
                u_dut.u_simt_stack.sp, u_dut.tos_reconv_pc, u_dut.tos_phase);
        end
        if (cvt_trace_en) begin
            $display("  [C%03d] PC=%0d fv=%b | fstl=%b sp_stl=%b sb=%b exb=%b|%b|%b|%b drn=%b | amask=%b done=%b | de_v=%b de_op=%02h rD=%0d",
                cycle_count, u_dut.u_fetch.pc_reg, u_dut.fetch_valid,
                u_dut.front_stall, u_dut.sp_stall, u_dut.sb_stall,
                u_dut.SP_LANE[0].u_sp.ex_busy, u_dut.SP_LANE[1].u_sp.ex_busy,
                u_dut.SP_LANE[2].u_sp.ex_busy, u_dut.SP_LANE[3].u_sp.ex_busy,
                u_dut.pipeline_drained, u_dut.active_mask, kernel_done,
                u_dut.de_valid, u_dut.de_opcode, u_dut.de_rD_addr);
            if (|dmem_wea)
                $display("         >>> DMEM WRITE: we=%04b addr0=%0d din0=%04h",
                    dmem_wea,
                    dmem_addra[0*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH],
                    dmem_dina[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]);
        end
    end

    task enable_trace; begin trace_en = 1; cycle_count = 0; end endtask
    task disable_trace; begin trace_en = 0; end endtask
    task enable_cvt_trace; begin cvt_trace_en = 1; cycle_count = 0; end endtask
    task disable_cvt_trace; begin cvt_trace_en = 0; end endtask
    task enable_tc_trace; begin tc_trace_en = 1; cycle_count = 0; end endtask
    task disable_tc_trace; begin tc_trace_en = 0; end endtask

    // ================================================================
    // TC state name decode
    // ================================================================
    reg [10*8-1:0] tc_top_state_name;
    always @(*) begin
        case (u_dut.u_tc_top.state)
            3'd0: tc_top_state_name = "IDLE      ";
            3'd1: tc_top_state_name = "GATHER_A  ";
            3'd2: tc_top_state_name = "GATHER_B  ";
            3'd3: tc_top_state_name = "GATHER_C  ";
            3'd4: tc_top_state_name = "COMPUTE   ";
            3'd5: tc_top_state_name = "SCATTER0  ";
            3'd6: tc_top_state_name = "SCATTER1  ";
            3'd7: tc_top_state_name = "SCATTER2  ";
            default: tc_top_state_name = "???       ";
        endcase
    end

    reg [10*8-1:0] tc_core_state_name;
    always @(*) begin
        case (u_dut.u_tc_top.u_tc.state)
            2'd0: tc_core_state_name = "IDLE      ";
            2'd1: tc_core_state_name = "LOAD      ";
            2'd2: tc_core_state_name = "FEED      ";
            2'd3: tc_core_state_name = "DRAIN     ";
            default: tc_core_state_name = "???       ";
        endcase
    end

    // ================================================================
    // TC Trace — deep SA/PE diagnostics
    // ================================================================
    always @(posedge clk) begin
        if (tc_trace_en && (u_dut.u_tc_top.busy || u_dut.u_tc_top.trigger ||
                            u_dut.u_tc_top.scat_w_we != 4'd0 ||
                            u_dut.u_tc_top.u_tc.valid_out)) begin
            $display("");
            $display("  [C%03d] === TC TRACE ===", cycle_count);
            // TC_TOP level
            $display("    TC_TOP: state=%0s scat3=%b busy=%b trigger=%b | rf_ovr=%b r0a=%0d r1a=%0d r2a=%0d r3a=%0d",
                tc_top_state_name,
                u_dut.u_tc_top.scatter3,
                u_dut.u_tc_top.busy,
                u_dut.u_tc_top.trigger,
                u_dut.u_tc_top.rf_addr_override,
                u_dut.u_tc_top.rf_r0_addr,
                u_dut.u_tc_top.rf_r1_addr,
                u_dut.u_tc_top.rf_r2_addr,
                u_dut.u_tc_top.rf_r3_addr);
            $display("    TC_TOP: rA_base=%0d rB_base=%0d rC_base=%0d rD_base=%0d",
                u_dut.u_tc_top.rA_base,
                u_dut.u_tc_top.rB_base,
                u_dut.u_tc_top.rC_base,
                u_dut.u_tc_top.rD_base);

            // tensor_core FSM
            $display("    TC_CORE: state=%0s round=%0d phase=%0d drain=%0d valid_in=%b valid_out=%b",
                tc_core_state_name,
                u_dut.u_tc_top.u_tc.round,
                u_dut.u_tc_top.u_tc.phase,
                u_dut.u_tc_top.u_tc.drain_cnt,
                u_dut.u_tc_top.u_tc.valid_in,
                u_dut.u_tc_top.u_tc.valid_out);

            // SA inputs
            $display("    SA: acc_load=%b a_valid=%04b | a_in=[%04h,%04h,%04h,%04h] b_in=[%04h,%04h,%04h,%04h]",
                u_dut.u_tc_top.u_tc.sa_acc_load,
                u_dut.u_tc_top.u_tc.sa_a_valid,
                u_dut.u_tc_top.u_tc.sa_a_in[0*16 +: 16],
                u_dut.u_tc_top.u_tc.sa_a_in[1*16 +: 16],
                u_dut.u_tc_top.u_tc.sa_a_in[2*16 +: 16],
                u_dut.u_tc_top.u_tc.sa_a_in[3*16 +: 16],
                u_dut.u_tc_top.u_tc.sa_b_in[0*16 +: 16],
                u_dut.u_tc_top.u_tc.sa_b_in[1*16 +: 16],
                u_dut.u_tc_top.u_tc.sa_b_in[2*16 +: 16],
                u_dut.u_tc_top.u_tc.sa_b_in[3*16 +: 16]);

            // Row 0 PEs: trace all 4 columns — shows systolic propagation
            $display("    PE[0][0]: a_in=%04h b_in=%04h v_in=%b | a_out=%04h b_out=%04h v_out=%b | mult_v=%b prod=%04h | add_v=%b add_r=%04h | acc=%04h",
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.a_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.b_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.valid_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.a_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.b_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.valid_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.mult_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.product,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.add_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.add_result,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.acc_out);
            $display("    PE[0][1]: a_in=%04h b_in=%04h v_in=%b | a_out=%04h b_out=%04h v_out=%b | mult_v=%b prod=%04h | add_v=%b add_r=%04h | acc=%04h",
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.a_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.b_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.valid_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.a_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.b_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.valid_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.mult_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.product,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.add_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.add_result,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.acc_out);
            $display("    PE[0][2]: a_in=%04h b_in=%04h v_in=%b | a_out=%04h b_out=%04h v_out=%b | mult_v=%b prod=%04h | add_v=%b add_r=%04h | acc=%04h",
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.a_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.b_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.valid_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.a_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.b_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.valid_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.mult_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.product,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.add_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.add_result,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.acc_out);
            $display("    PE[0][3]: a_in=%04h b_in=%04h v_in=%b | a_out=%04h b_out=%04h v_out=%b | mult_v=%b prod=%04h | add_v=%b add_r=%04h | acc=%04h",
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.a_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.b_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.valid_in,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.a_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.b_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.valid_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.mult_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.product,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.add_valid,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.add_result,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.acc_out);

            // Row 0 accumulators summary (all columns, all from SP0's perspective)
            $display("    ROW0 acc: [%04h, %04h, %04h, %04h]",
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[0].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[1].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[2].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[0].COL[3].u_pe.acc_out);
            $display("    ROW1 acc: [%04h, %04h, %04h, %04h]",
                u_dut.u_tc_top.u_tc.u_sa.ROW[1].COL[0].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[1].COL[1].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[1].COL[2].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[1].COL[3].u_pe.acc_out);
            $display("    ROW2 acc: [%04h, %04h, %04h, %04h]",
                u_dut.u_tc_top.u_tc.u_sa.ROW[2].COL[0].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[2].COL[1].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[2].COL[2].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[2].COL[3].u_pe.acc_out);
            $display("    ROW3 acc: [%04h, %04h, %04h, %04h]",
                u_dut.u_tc_top.u_tc.u_sa.ROW[3].COL[0].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[3].COL[1].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[3].COL[2].u_pe.acc_out,
                u_dut.u_tc_top.u_tc.u_sa.ROW[3].COL[3].u_pe.acc_out);

            // Vertical B propagation — trace b_in at each row boundary for col 0 and col 1
            $display("    B_vert col0: top=%04h r0out=%04h r1out=%04h r2out=%04h r3out=%04h",
                u_dut.u_tc_top.u_tc.u_sa.v_b[0][0],
                u_dut.u_tc_top.u_tc.u_sa.v_b[1][0],
                u_dut.u_tc_top.u_tc.u_sa.v_b[2][0],
                u_dut.u_tc_top.u_tc.u_sa.v_b[3][0],
                u_dut.u_tc_top.u_tc.u_sa.v_b[4][0]);
            $display("    B_vert col1: top=%04h r0out=%04h r1out=%04h r2out=%04h r3out=%04h",
                u_dut.u_tc_top.u_tc.u_sa.v_b[0][1],
                u_dut.u_tc_top.u_tc.u_sa.v_b[1][1],
                u_dut.u_tc_top.u_tc.u_sa.v_b[2][1],
                u_dut.u_tc_top.u_tc.u_sa.v_b[3][1],
                u_dut.u_tc_top.u_tc.u_sa.v_b[4][1]);

            // Horizontal A propagation — trace a values across row 0
            $display("    A_horiz row0: left=%04h c0out=%04h c1out=%04h c2out=%04h c3out=%04h | v: left=%b c0=%b c1=%b c2=%b c3=%b",
                u_dut.u_tc_top.u_tc.u_sa.h_a[0][0],
                u_dut.u_tc_top.u_tc.u_sa.h_a[0][1],
                u_dut.u_tc_top.u_tc.u_sa.h_a[0][2],
                u_dut.u_tc_top.u_tc.u_sa.h_a[0][3],
                u_dut.u_tc_top.u_tc.u_sa.h_a[0][4],
                u_dut.u_tc_top.u_tc.u_sa.h_v[0][0],
                u_dut.u_tc_top.u_tc.u_sa.h_v[0][1],
                u_dut.u_tc_top.u_tc.u_sa.h_v[0][2],
                u_dut.u_tc_top.u_tc.u_sa.h_v[0][3],
                u_dut.u_tc_top.u_tc.u_sa.h_v[0][4]);

            // Scatter output
            if (u_dut.u_tc_top.scat_w_we != 4'd0)
                $display("    SCATTER: addr=%0d we=%04b data=[%04h,%04h,%04h,%04h]",
                    u_dut.u_tc_top.scat_w_addr,
                    u_dut.u_tc_top.scat_w_we,
                    u_dut.u_tc_top.scat_w_data[0*16 +: 16],
                    u_dut.u_tc_top.scat_w_data[1*16 +: 16],
                    u_dut.u_tc_top.scat_w_data[2*16 +: 16],
                    u_dut.u_tc_top.scat_w_data[3*16 +: 16]);

            // d_hold on valid_out
            if (u_dut.u_tc_top.u_tc.valid_out) begin
                $display("    >>> D_OUT ROW0: [%04h, %04h, %04h, %04h]",
                    u_dut.u_tc_top.u_tc.matrix_d[(0*4+0)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(0*4+1)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(0*4+2)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(0*4+3)*16 +: 16]);
                $display("    >>> D_OUT ROW1: [%04h, %04h, %04h, %04h]",
                    u_dut.u_tc_top.u_tc.matrix_d[(1*4+0)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(1*4+1)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(1*4+2)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(1*4+3)*16 +: 16]);
                $display("    >>> D_OUT ROW2: [%04h, %04h, %04h, %04h]",
                    u_dut.u_tc_top.u_tc.matrix_d[(2*4+0)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(2*4+1)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(2*4+2)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(2*4+3)*16 +: 16]);
                $display("    >>> D_OUT ROW3: [%04h, %04h, %04h, %04h]",
                    u_dut.u_tc_top.u_tc.matrix_d[(3*4+0)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(3*4+1)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(3*4+2)*16 +: 16],
                    u_dut.u_tc_top.u_tc.matrix_d[(3*4+3)*16 +: 16]);
            end
        end
    end

    // ================================================================
    // Disassembler
    // ================================================================
    task disasm;
        input integer addr; input [31:0] inst;
        reg [4:0] op; reg dt; reg [3:0] rD, rA, rB, rC; reg [15:0] imm;
    begin
        op = inst[31:27]; dt = inst[26];
        rD = inst[23:20]; rA = inst[19:16]; rB = inst[15:12]; rC = inst[11:8];
        imm = inst[15:0];
        case (op)
            `OP_NOP:  $display("    [%2d] %08h  NOP", addr, inst);
            `OP_MOVI: $display("    [%2d] %08h  MOVI    R%0d, %0d (0x%04h)", addr, inst, rD, imm, imm);
            `OP_MOV:  if (dt) $display("    [%2d] %08h  MOV.TID R%0d", addr, inst, rD);
                      else $display("    [%2d] %08h  MOV     R%0d, R%0d", addr, inst, rD, rA);
            `OP_ADD:  $display("    [%2d] %08h  ADD%s   R%0d, R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA, rB);
            `OP_SUB:  $display("    [%2d] %08h  SUB%s   R%0d, R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA, rB);
            `OP_MUL:  $display("    [%2d] %08h  MUL%s   R%0d, R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA, rB);
            `OP_FMA:  $display("    [%2d] %08h  FMA%s   R%0d, R%0d, R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA, rB, rD);
            `OP_MAX:  $display("    [%2d] %08h  MAX%s   R%0d, R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA, rB);
            `OP_MIN:  $display("    [%2d] %08h  MIN%s   R%0d, R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA, rB);
            `OP_AND:  $display("    [%2d] %08h  AND     R%0d, R%0d, R%0d", addr, inst, rD, rA, rB);
            `OP_OR:   $display("    [%2d] %08h  OR      R%0d, R%0d, R%0d", addr, inst, rD, rA, rB);
            `OP_XOR:  $display("    [%2d] %08h  XOR     R%0d, R%0d, R%0d", addr, inst, rD, rA, rB);
            `OP_NEG:  $display("    [%2d] %08h  NEG%s   R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA);
            `OP_ABS:  $display("    [%2d] %08h  ABS%s   R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA);
            `OP_SHL:  $display("    [%2d] %08h  SHL     R%0d, R%0d, %0d", addr, inst, rD, rA, imm);
            `OP_SHR:  $display("    [%2d] %08h  SHR     R%0d, R%0d, %0d", addr, inst, rD, rA, imm);
            `OP_ADDI: $display("    [%2d] %08h  ADDI%s  R%0d, R%0d, %0d", addr, inst, dt?".f":"  ", rD, rA, imm);
            `OP_MULI: $display("    [%2d] %08h  MULI%s  R%0d, R%0d, %0d", addr, inst, dt?".f":"  ", rD, rA, imm);
            `OP_LD:   $display("    [%2d] %08h  LD      R%0d, [R%0d + %0d]", addr, inst, rD, rA, imm);
            `OP_ST:   $display("    [%2d] %08h  ST      R%0d, [R%0d + %0d]", addr, inst, rD, rA, imm);
            `OP_LDS:  $display("    [%2d] %08h  LDS     R%0d, [R%0d + %0d]", addr, inst, rD, rA, imm);
            `OP_STS:  $display("    [%2d] %08h  STS     R%0d, [R%0d + %0d]", addr, inst, rD, rA, imm);
            `OP_BRA:  $display("    [%2d] %08h  BRA     %0d", addr, inst, inst[26:0]);
            `OP_PBRA: $display("    [%2d] %08h  PBRA    P%0d, target=%0d, reconv=%0d", addr, inst, inst[26:25], inst[24:12], inst[11:0]);
            `OP_RET:  $display("    [%2d] %08h  RET", addr, inst);
            `OP_SETP: $display("    [%2d] %08h  SETP    P%0d, R%0d, R%0d (cmp=%0d)", addr, inst, rD, rA, rB, inst[25:24]);
            `OP_SELP: $display("    [%2d] %08h  SELP    R%0d, R%0d, R%0d (P%0d)", addr, inst, rD, rA, rB, inst[25:24]);
            `OP_SET:  $display("    [%2d] %08h  SET     P%0d, %0d", addr, inst, rD[1:0], imm[0]);
            `OP_CVT:  $display("    [%2d] %08h  CVT.%s  R%0d, R%0d", addr, inst, dt?"i2f":"f2i", rD, rA);
            `WMMA_MMA:   $display("    [%2d] %08h  WMMA.MMA  D=R%0d..%0d, A=R%0d..%0d, B=R%0d..%0d, C=R%0d..%0d",
                           addr, inst, rD, rD+4'd3, rA, rA+4'd3, rB, rB+4'd3, rC, rC+4'd3);
            `WMMA_LOAD:  $display("    [%2d] %08h  WMMA.LOAD R%0d..%0d, [R%0d + %0d]", addr, inst, rD, rD+4'd3, rA, imm);
            `WMMA_STORE: $display("    [%2d] %08h  WMMA.STORE R%0d..%0d, [R%0d + %0d]", addr, inst, rD, rD+4'd3, rA, imm);
            default:  $display("    [%2d] %08h  ??? (op=%02d)", addr, inst, op);
        endcase
    end
    endtask

    task print_program;
        input integer start_addr; input integer length;
        integer pi;
    begin
        $display("  Program listing:");
        for (pi = start_addr; pi < start_addr + length; pi = pi + 1)
            disasm(pi, imem[pi]);
    end
    endtask

    task dump_rf_range;
        input [3:0] r_start; input [3:0] r_end;
        integer ri, si;
    begin
        $write("  RF dump     ");
        for (ri = r_start; ri <= r_end; ri = ri + 1) $write("  R%-2d  ", ri);
        $write("\n");
        for (si = 0; si < 4; si = si + 1) begin
            $write("       SP%0d:  ", si);
            for (ri = r_start; ri <= r_end; ri = ri + 1) begin
                read_gpr(ri[3:0]);
                $write(" %04h ", dbg_rf_val[si]);
            end
            $write("\n");
        end
    end
    endtask

    // ================================================================
    // Main Test Sequence
    // ================================================================
    initial begin
        $display("============================================");
        $display("  SM Core Testbench v1.5 — Starting");
        $display("============================================");

        // ── K1: Basic ALU (MOVI + ADD) ───────────────────
        begin
            $display("\n--- K1: Basic ALU (MOVI + ADD) ---");
            reset_dut; clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd100);
            imem[1] = enc_movi(4'd2, 16'd200);
            imem[2] = enc_r(`OP_ADD, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[3] = INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd1, 16'd100, "K1 R1=100");
            check_gpr_all(4'd2, 16'd200, "K1 R2=200");
            check_gpr_all(4'd3, 16'd300, "K1 R3=300");
        end

        // ── K2: Memory LD / ST ───────────────────────────
        begin
            $display("\n--- K2: Memory LD / ST ---");
            reset_dut; clear_dmem;
            dmem0[10]=16'hABCD; dmem1[10]=16'hABCD; dmem2[10]=16'hABCD; dmem3[10]=16'hABCD;
            imem[0] = enc_movi(4'd1, 16'd10);
            imem[1] = enc_m(`OP_LD, 4'd2, 4'd1, 16'd0);
            imem[2] = enc_i(`OP_ADDI, 1'b0, 4'd3, 4'd2, 16'd1);
            imem[3] = enc_m(`OP_ST, 4'd3, 4'd1, 16'd5);
            imem[4] = INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd2, 16'hABCD, "K2 R2=DMEM[10]");
            check_gpr_all(4'd3, 16'hABCE, "K2 R3=R2+1");
            check_dmem_all(10'd15, 16'hABCE, "K2 DMEM[15]=R3");
        end

        // ── K3: Branch ───────────────────────────────────
        begin
            $display("\n--- K3: Branch (BRA) ---");
            reset_dut; clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd42);
            imem[1] = enc_bra(27'd4);
            imem[2] = enc_movi(4'd1, 16'd99);
            imem[3] = enc_movi(4'd1, 16'd88);
            imem[4] = enc_movi(4'd2, 16'd55);
            imem[5] = INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd1, 16'd42, "K3 R1=42 (not overwritten)");
            check_gpr_all(4'd2, 16'd55, "K3 R2=55 (after branch)");
        end

        // ── K4: MUL + SB ────────────────────────────────
        begin
            $display("\n--- K4: MUL + Scoreboard Stall ---");
            reset_dut; clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd3);
            imem[1] = enc_movi(4'd2, 16'd7);
            imem[2] = enc_r(`OP_MUL, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[3] = enc_r(`OP_ADD, 1'b0, 4'd4, 4'd3, 4'd1);
            imem[4] = INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd3, 16'd21, "K4 R3=3*7=21");
            check_gpr_all(4'd4, 16'd24, "K4 R4=21+3=24");
        end

        // ── K5: MOV.TID ──────────────────────────────────
        begin
            $display("\n--- K5: MOV.TID ---");
            reset_dut; clear_dmem;
            imem[0] = enc_mov_tid(4'd1);
            imem[1] = INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr(2'd0, 4'd1, 16'd0, "K5 SP0.R1=TID0");
            check_gpr(2'd1, 4'd1, 16'd1, "K5 SP1.R1=TID1");
            check_gpr(2'd2, 4'd1, 16'd2, "K5 SP2.R1=TID2");
            check_gpr(2'd3, 4'd1, 16'd3, "K5 SP3.R1=TID3");
        end

        // ── K6: Back-to-back MOVI ────────────────────────
        begin
            $display("\n--- K6: Back-to-back MOVI ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd10); imem[1]=enc_movi(4'd2,16'd20);
            imem[2]=enc_movi(4'd3,16'd30); imem[3]=enc_movi(4'd4,16'd40);
            imem[4]=enc_movi(4'd5,16'd50); imem[5]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd1, 16'd10, "K6 R1=10");
            check_gpr_all(4'd2, 16'd20, "K6 R2=20");
            check_gpr_all(4'd3, 16'd30, "K6 R3=30");
            check_gpr_all(4'd4, 16'd40, "K6 R4=40");
            check_gpr_all(4'd5, 16'd50, "K6 R5=50");
        end

        // ── K7: Per-thread LD/ST ─────────────────────────
        begin
            $display("\n--- K7: Per-thread LD/ST ---");
            reset_dut; clear_dmem;
            imem[0]=enc_mov_tid(4'd1); imem[1]=enc_movi(4'd2,16'hCAFE);
            imem[2]=enc_m(`OP_ST,4'd2,4'd1,16'd100);
            imem[3]=enc_m(`OP_LD,4'd3,4'd1,16'd100);
            imem[4]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_dmem(2'd0,10'd100,16'hCAFE,"K7 SP0 DMEM[100]");
            check_dmem(2'd1,10'd101,16'hCAFE,"K7 SP1 DMEM[101]");
            check_dmem(2'd2,10'd102,16'hCAFE,"K7 SP2 DMEM[102]");
            check_dmem(2'd3,10'd103,16'hCAFE,"K7 SP3 DMEM[103]");
            check_gpr_all(4'd3, 16'hCAFE, "K7 R3=loaded CAFE");
        end

        // ── K8: Logic + Shift ────────────────────────────
        begin
            $display("\n--- K8: Logic + Shift ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'h00FF); imem[1]=enc_movi(4'd2,16'h0F0F);
            imem[2]=enc_r(`OP_AND,1'b0,4'd3,4'd1,4'd2);
            imem[3]=enc_r(`OP_OR, 1'b0,4'd4,4'd1,4'd2);
            imem[4]=enc_r(`OP_XOR,1'b0,4'd5,4'd1,4'd2);
            imem[5]=enc_i(`OP_SHL,1'b0,4'd6,4'd1,16'd4);
            imem[6]=enc_i(`OP_SHR,1'b0,4'd7,4'd1,16'd4);
            imem[7]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd3,16'h000F,"K8 AND");
            check_gpr_all(4'd4,16'h0FFF,"K8 OR");
            check_gpr_all(4'd5,16'h0FF0,"K8 XOR");
            check_gpr_all(4'd6,16'h0FF0,"K8 SHL 4");
            check_gpr_all(4'd7,16'h000F,"K8 SHR 4");
        end

        // ── K9: FMA ──────────────────────────────────────
        begin
            $display("\n--- K9: FMA ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd5); imem[1]=enc_movi(4'd2,16'd6);
            imem[2]=enc_movi(4'd3,16'd10);
            imem[3]=enc_r(`OP_FMA,1'b0,4'd3,4'd1,4'd2);
            imem[4]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd3,16'd40,"K9 FMA 5*6+10=40");
        end

        // ── K10: Non-zero entry PC ───────────────────────
        begin
            $display("\n--- K10: Non-zero entry PC ---");
            reset_dut; clear_dmem;
            imem[10]=enc_movi(4'd1,16'hBEEF); imem[11]=enc_movi(4'd2,16'hDEAD);
            imem[12]=INST_RET;
            launch_kernel(32'd10); wait_kernel_done(100);
            check_gpr_all(4'd1,16'hBEEF,"K10 R1=0xBEEF");
            check_gpr_all(4'd2,16'hDEAD,"K10 R2=0xDEAD");
        end

        // ── K11: SUB + NEG + ABS ─────────────────────────
        begin
            $display("\n--- K11: SUB + NEG + ABS ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd30); imem[1]=enc_movi(4'd2,16'd50);
            imem[2]=enc_r(`OP_SUB,1'b0,4'd3,4'd1,4'd2);
            imem[3]=enc_r(`OP_NEG,1'b0,4'd4,4'd3,4'd0);
            imem[4]=enc_r(`OP_ABS,1'b0,4'd5,4'd3,4'd0);
            imem[5]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd3,16'hFFEC,"K11 SUB 30-50=-20");
            check_gpr_all(4'd4,16'd20,"K11 NEG(-20)=20");
            check_gpr_all(4'd5,16'd20,"K11 ABS(-20)=20");
        end

        // ── K12: ADDI chain ──────────────────────────────
        begin
            $display("\n--- K12: ADDI chain ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd0);
            imem[1]=enc_i(`OP_ADDI,1'b0,4'd1,4'd1,16'd1);
            imem[2]=enc_i(`OP_ADDI,1'b0,4'd1,4'd1,16'd1);
            imem[3]=enc_i(`OP_ADDI,1'b0,4'd1,4'd1,16'd1);
            imem[4]=enc_i(`OP_ADDI,1'b0,4'd1,4'd1,16'd1);
            imem[5]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr_all(4'd1,16'd4,"K12 ADDI chain R1=4");
        end

        // ── K13: MAX / MIN ───────────────────────────────
        begin
            $display("\n--- K13: MAX / MIN ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd100); imem[1]=enc_movi(4'd2,16'd200);
            imem[2]=enc_r(`OP_MAX,1'b0,4'd3,4'd1,4'd2);
            imem[3]=enc_r(`OP_MIN,1'b0,4'd4,4'd1,4'd2);
            imem[4]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd3,16'd200,"K13 MAX(100,200)=200");
            check_gpr_all(4'd4,16'd100,"K13 MIN(100,200)=100");
        end

        // ── K14: WMMA.LOAD ───────────────────────────────
        begin
            $display("\n--- K14: WMMA.LOAD basic ---");
            reset_dut; clear_dmem;
            dmem0[200]=16'h1111;dmem0[201]=16'h2222;dmem0[202]=16'h3333;dmem0[203]=16'h4444;
            dmem1[200]=16'h1111;dmem1[201]=16'h2222;dmem1[202]=16'h3333;dmem1[203]=16'h4444;
            dmem2[200]=16'h1111;dmem2[201]=16'h2222;dmem2[202]=16'h3333;dmem2[203]=16'h4444;
            dmem3[200]=16'h1111;dmem3[201]=16'h2222;dmem3[202]=16'h3333;dmem3[203]=16'h4444;
            imem[0]=enc_movi(4'd15,16'd200);
            imem[1]=enc_m(`WMMA_LOAD,4'd4,4'd15,16'd0);
            imem[2]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd4,16'h1111,"K14 R4=0x1111");
            check_gpr_all(4'd5,16'h2222,"K14 R5=0x2222");
            check_gpr_all(4'd6,16'h3333,"K14 R6=0x3333");
            check_gpr_all(4'd7,16'h4444,"K14 R7=0x4444");
        end

        // ── K15: WMMA.STORE ──────────────────────────────
        begin
            $display("\n--- K15: WMMA.STORE basic ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd0,16'h1234); imem[1]=enc_movi(4'd1,16'h5678);
            imem[2]=enc_movi(4'd2,16'h9ABC); imem[3]=enc_movi(4'd3,16'hDEF0);
            imem[4]=enc_movi(4'd8,16'd300);
            imem[5]=enc_m(`WMMA_STORE,4'd0,4'd8,16'd0);
            imem[6]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(100);
            check_dmem_all(10'd300,16'h1234,"K15 DMEM[300]=0x1234");
            check_dmem_all(10'd301,16'h5678,"K15 DMEM[301]=0x5678");
            check_dmem_all(10'd302,16'h9ABC,"K15 DMEM[302]=0x9ABC");
            check_dmem_all(10'd303,16'hDEF0,"K15 DMEM[303]=0xDEF0");
        end

        // ── K16: WMMA.MMA 1*1+0=4.0 (v1.5: zero C) ─────
        begin
            $display("\n--- K16: WMMA.MMA uniform 1*1+0=4.0 ---");
            reset_dut; clear_dmem;
            enable_tc_trace;
            imem[0]=enc_movi(4'd0,16'h3F80); imem[1]=enc_movi(4'd1,16'h3F80);
            imem[2]=enc_movi(4'd2,16'h3F80); imem[3]=enc_movi(4'd3,16'h3F80);
            imem[4]=enc_movi(4'd4,16'h3F80); imem[5]=enc_movi(4'd5,16'h3F80);
            imem[6]=enc_movi(4'd6,16'h3F80); imem[7]=enc_movi(4'd7,16'h3F80);
            // v1.5: explicitly zero C matrix (R8-R11)
            imem[8]=enc_movi(4'd8,16'h0000); imem[9]=enc_movi(4'd9,16'h0000);
            imem[10]=enc_movi(4'd10,16'h0000); imem[11]=enc_movi(4'd11,16'h0000);
            imem[12]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            disable_tc_trace;
            dump_rf_range(4'd12, 4'd15);
            check_gpr_all(4'd12,16'h4080,"K16 D=4.0");
            check_gpr_all(4'd13,16'h4080,"K16 D=4.0");
            check_gpr_all(4'd14,16'h4080,"K16 D=4.0");
            check_gpr_all(4'd15,16'h4080,"K16 D=4.0");
        end

        // ── K17: WMMA.MMA 2*3+1=25.0 ────────────────────
        begin
            $display("\n--- K17: WMMA.MMA 2*3+1=25.0 ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd0,16'h4000); imem[1]=enc_movi(4'd1,16'h4000);
            imem[2]=enc_movi(4'd2,16'h4000); imem[3]=enc_movi(4'd3,16'h4000);
            imem[4]=enc_movi(4'd4,16'h4040); imem[5]=enc_movi(4'd5,16'h4040);
            imem[6]=enc_movi(4'd6,16'h4040); imem[7]=enc_movi(4'd7,16'h4040);
            imem[8]=enc_movi(4'd8,16'h3F80); imem[9]=enc_movi(4'd9,16'h3F80);
            imem[10]=enc_movi(4'd10,16'h3F80); imem[11]=enc_movi(4'd11,16'h3F80);
            imem[12]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr_all(4'd12,16'h41C8,"K17 D=25.0");
            check_gpr_all(4'd13,16'h41C8,"K17 D=25.0");
            check_gpr_all(4'd14,16'h41C8,"K17 D=25.0");
            check_gpr_all(4'd15,16'h41C8,"K17 D=25.0");
        end

        // ── K18: WMMA LOAD->MMA->STORE ───────────────────
        begin
            $display("\n--- K18: WMMA LOAD->MMA->STORE ---");
            reset_dut; clear_dmem;
            dmem0[200]=16'h3F80;dmem0[201]=16'h0000;dmem0[202]=16'h0000;dmem0[203]=16'h0000;
            dmem1[200]=16'h0000;dmem1[201]=16'h3F80;dmem1[202]=16'h0000;dmem1[203]=16'h0000;
            dmem2[200]=16'h0000;dmem2[201]=16'h0000;dmem2[202]=16'h3F80;dmem2[203]=16'h0000;
            dmem3[200]=16'h0000;dmem3[201]=16'h0000;dmem3[202]=16'h0000;dmem3[203]=16'h3F80;
            dmem0[204]=16'h4000;dmem0[205]=16'h4000;dmem0[206]=16'h4000;dmem0[207]=16'h4000;
            dmem1[204]=16'h4000;dmem1[205]=16'h4000;dmem1[206]=16'h4000;dmem1[207]=16'h4000;
            dmem2[204]=16'h4000;dmem2[205]=16'h4000;dmem2[206]=16'h4000;dmem2[207]=16'h4000;
            dmem3[204]=16'h4000;dmem3[205]=16'h4000;dmem3[206]=16'h4000;dmem3[207]=16'h4000;
            imem[0]=enc_movi(4'd0,16'd200);
            imem[1]=enc_m(`WMMA_LOAD,4'd4,4'd0,16'd0);
            imem[2]=enc_m(`WMMA_LOAD,4'd8,4'd0,16'd4);
            imem[3]=INST_NOP;imem[4]=INST_NOP;imem[5]=INST_NOP;imem[6]=INST_NOP;
            imem[7]=enc_movi(4'd0,16'd0);imem[8]=enc_movi(4'd1,16'd0);
            imem[9]=enc_movi(4'd2,16'd0);imem[10]=enc_movi(4'd3,16'd0);
            imem[11]=enc_wmma_mma(4'd12,4'd4,4'd8,4'd0);
            imem[12]=INST_NOP;imem[13]=INST_NOP;imem[14]=INST_NOP;imem[15]=INST_NOP;
            imem[16]=enc_movi(4'd0,16'd300);
            imem[17]=enc_m(`WMMA_STORE,4'd12,4'd0,16'd0);
            imem[18]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(300);
            check_gpr_all(4'd12,16'h4000,"K18 D=2.0");
            check_gpr_all(4'd13,16'h4000,"K18 D=2.0");
            check_dmem_all(10'd300,16'h4000,"K18 DMEM=2.0");
            check_dmem_all(10'd301,16'h4000,"K18 DMEM=2.0");
        end

        // ── K19: MMA zero passthrough ────────────────────
        begin
            $display("\n--- K19: MMA zero passthrough ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd4,16'h4040);imem[1]=enc_movi(4'd5,16'h4040);
            imem[2]=enc_movi(4'd6,16'h4040);imem[3]=enc_movi(4'd7,16'h4040);
            imem[4]=enc_movi(4'd8,16'h40A0);imem[5]=enc_movi(4'd9,16'h40A0);
            imem[6]=enc_movi(4'd10,16'h40A0);imem[7]=enc_movi(4'd11,16'h40A0);
            imem[8]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[9]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr_all(4'd12,16'h40A0,"K19 D=5.0");
        end

        // ── K20: MMA negative -1*2+0=-8.0 (v1.5: zero C) ──
        begin
            $display("\n--- K20: MMA negative -1*2+0=-8.0 ---");
            reset_dut; clear_dmem;
            enable_tc_trace;
            imem[0]=enc_movi(4'd0,16'hBF80);imem[1]=enc_movi(4'd1,16'hBF80);
            imem[2]=enc_movi(4'd2,16'hBF80);imem[3]=enc_movi(4'd3,16'hBF80);
            imem[4]=enc_movi(4'd4,16'h4000);imem[5]=enc_movi(4'd5,16'h4000);
            imem[6]=enc_movi(4'd6,16'h4000);imem[7]=enc_movi(4'd7,16'h4000);
            // v1.5: explicitly zero C matrix (R8-R11)
            imem[8]=enc_movi(4'd8,16'h0000);imem[9]=enc_movi(4'd9,16'h0000);
            imem[10]=enc_movi(4'd10,16'h0000);imem[11]=enc_movi(4'd11,16'h0000);
            imem[12]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            disable_tc_trace;
            dump_rf_range(4'd12, 4'd15);
            check_gpr_all(4'd12,16'hC100,"K20 D=-8.0");
        end

        // ── K21: MMA mixed 1*1+(-2)=2.0 ─────────────────
        begin
            $display("\n--- K21: MMA mixed 1*1+(-2)=2.0 ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd0,16'h3F80);imem[1]=enc_movi(4'd1,16'h3F80);
            imem[2]=enc_movi(4'd2,16'h3F80);imem[3]=enc_movi(4'd3,16'h3F80);
            imem[4]=enc_movi(4'd4,16'h3F80);imem[5]=enc_movi(4'd5,16'h3F80);
            imem[6]=enc_movi(4'd6,16'h3F80);imem[7]=enc_movi(4'd7,16'h3F80);
            imem[8]=enc_movi(4'd8,16'hC000);imem[9]=enc_movi(4'd9,16'hC000);
            imem[10]=enc_movi(4'd10,16'hC000);imem[11]=enc_movi(4'd11,16'hC000);
            imem[12]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr_all(4'd12,16'h4000,"K21 D=2.0");
        end

        // ── K22: MMA fractional 0.5*0.5+0=1.0 (v1.5: zero C) ──
        begin
            $display("\n--- K22: MMA fractional 0.5*0.5+0=1.0 ---");
            reset_dut; clear_dmem;
            enable_tc_trace;
            imem[0]=enc_movi(4'd0,16'h3F00);imem[1]=enc_movi(4'd1,16'h3F00);
            imem[2]=enc_movi(4'd2,16'h3F00);imem[3]=enc_movi(4'd3,16'h3F00);
            imem[4]=enc_movi(4'd4,16'h3F00);imem[5]=enc_movi(4'd5,16'h3F00);
            imem[6]=enc_movi(4'd6,16'h3F00);imem[7]=enc_movi(4'd7,16'h3F00);
            // v1.5: explicitly zero C matrix (R8-R11)
            imem[8]=enc_movi(4'd8,16'h0000);imem[9]=enc_movi(4'd9,16'h0000);
            imem[10]=enc_movi(4'd10,16'h0000);imem[11]=enc_movi(4'd11,16'h0000);
            imem[12]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            disable_tc_trace;
            dump_rf_range(4'd12, 4'd15);
            check_gpr_all(4'd12,16'h3F80,"K22 D=1.0");
        end

        // ── K23: Chained MMA D1=4.0, D2=16.0 (v1.5: zero C) ──
        begin
            $display("\n--- K23: Chained MMA D1=4.0, D2=16.0 ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd0,16'h3F80);imem[1]=enc_movi(4'd1,16'h3F80);
            imem[2]=enc_movi(4'd2,16'h3F80);imem[3]=enc_movi(4'd3,16'h3F80);
            imem[4]=enc_movi(4'd4,16'h3F80);imem[5]=enc_movi(4'd5,16'h3F80);
            imem[6]=enc_movi(4'd6,16'h3F80);imem[7]=enc_movi(4'd7,16'h3F80);
            // v1.5: explicitly zero C matrix (R8-R11)
            imem[8]=enc_movi(4'd8,16'h0000);imem[9]=enc_movi(4'd9,16'h0000);
            imem[10]=enc_movi(4'd10,16'h0000);imem[11]=enc_movi(4'd11,16'h0000);
            imem[12]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13]=enc_wmma_mma(4'd0, 4'd12, 4'd4, 4'd8);
            imem[14]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(400);
            check_gpr_all(4'd12,16'h4080,"K23 D1=4.0");
            check_gpr_all(4'd0,16'h4180,"K23 D2=16.0");
        end

        // ── K24: Per-thread SIMT diag(2)*3+1=7.0 ────────
        begin
            $display("\n--- K24: Per-thread SIMT diag(2)*3+1=7.0 ---");
            reset_dut; clear_dmem;
            dmem0[100]=16'h4000;dmem0[101]=16'h0000;dmem0[102]=16'h0000;dmem0[103]=16'h0000;
            dmem1[100]=16'h0000;dmem1[101]=16'h4000;dmem1[102]=16'h0000;dmem1[103]=16'h0000;
            dmem2[100]=16'h0000;dmem2[101]=16'h0000;dmem2[102]=16'h4000;dmem2[103]=16'h0000;
            dmem3[100]=16'h0000;dmem3[101]=16'h0000;dmem3[102]=16'h0000;dmem3[103]=16'h4000;
            imem[0]=enc_movi(4'd0,16'd100);
            imem[1]=enc_m(`WMMA_LOAD,4'd0,4'd0,16'd0);
            imem[2]=enc_movi(4'd4,16'h4040);imem[3]=enc_movi(4'd5,16'h4040);
            imem[4]=enc_movi(4'd6,16'h4040);imem[5]=enc_movi(4'd7,16'h4040);
            imem[6]=enc_movi(4'd8,16'h3F80);imem[7]=enc_movi(4'd9,16'h3F80);
            imem[8]=enc_movi(4'd10,16'h3F80);imem[9]=enc_movi(4'd11,16'h3F80);
            imem[10]=enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[11]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(300);
            check_gpr_all(4'd12,16'h40E0,"K24 D=7.0");
        end

        // ── K25–K30: PTX-mapped kernels ──────────────────
        begin
            $display("\n--- K25: PTX K1 vec_add int16 ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'd10;dmem1[2]=16'd20;dmem2[4]=16'd30;dmem3[6]=16'd40;
            dmem0[16]=16'd5;dmem1[18]=16'd15;dmem2[20]=16'd25;dmem3[22]=16'd35;
            imem[0]=enc_mov_tid(4'd0);imem[1]=enc_movi(4'd1,16'd0);
            imem[2]=enc_movi(4'd2,16'd16);imem[3]=enc_movi(4'd3,16'd32);
            imem[4]=enc_i(`OP_SHL,1'b0,4'd4,4'd0,16'd1);
            imem[5]=enc_r(`OP_ADD,1'b0,4'd5,4'd1,4'd4);
            imem[6]=enc_m(`OP_LD,4'd5,4'd5,16'd0);
            imem[7]=enc_r(`OP_ADD,1'b0,4'd6,4'd2,4'd4);
            imem[8]=enc_m(`OP_LD,4'd6,4'd6,16'd0);
            imem[9]=enc_r(`OP_ADD,1'b0,4'd7,4'd6,4'd5);
            imem[10]=enc_r(`OP_ADD,1'b0,4'd8,4'd3,4'd4);
            imem[11]=enc_m(`OP_ST,4'd7,4'd8,16'd0);
            imem[12]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_dmem(2'd0,10'd32,16'd15,"K25 SP0 C=10+5");
            check_dmem(2'd1,10'd34,16'd35,"K25 SP1 C=20+15");
            check_dmem(2'd2,10'd36,16'd55,"K25 SP2 C=30+25");
            check_dmem(2'd3,10'd38,16'd75,"K25 SP3 C=40+35");
        end

        begin
            $display("\n--- K26: PTX K2 vec_sub int16 ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'd100;dmem1[2]=16'd200;dmem2[4]=16'd300;dmem3[6]=16'd400;
            dmem0[16]=16'd30;dmem1[18]=16'd50;dmem2[20]=16'd100;dmem3[22]=16'd150;
            imem[0]=enc_mov_tid(4'd0);imem[1]=enc_movi(4'd1,16'd0);
            imem[2]=enc_movi(4'd2,16'd16);imem[3]=enc_movi(4'd3,16'd32);
            imem[4]=enc_i(`OP_SHL,1'b0,4'd4,4'd0,16'd1);
            imem[5]=enc_r(`OP_ADD,1'b0,4'd5,4'd1,4'd4);
            imem[6]=enc_m(`OP_LD,4'd5,4'd5,16'd0);
            imem[7]=enc_r(`OP_ADD,1'b0,4'd6,4'd2,4'd4);
            imem[8]=enc_m(`OP_LD,4'd6,4'd6,16'd0);
            imem[9]=enc_r(`OP_SUB,1'b0,4'd7,4'd5,4'd6);
            imem[10]=enc_r(`OP_ADD,1'b0,4'd8,4'd3,4'd4);
            imem[11]=enc_m(`OP_ST,4'd7,4'd8,16'd0);
            imem[12]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_dmem(2'd0,10'd32,16'd70,"K26 SP0 C=100-30");
            check_dmem(2'd1,10'd34,16'd150,"K26 SP1 C=200-50");
            check_dmem(2'd2,10'd36,16'd200,"K26 SP2 C=300-100");
            check_dmem(2'd3,10'd38,16'd250,"K26 SP3 C=400-150");
        end

        begin
            $display("\n--- K27: PTX K3 bf16_vector_mul ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'h4000;dmem1[2]=16'h4000;dmem2[4]=16'h4000;dmem3[6]=16'h4000;
            dmem0[16]=16'h4040;dmem1[18]=16'h4040;dmem2[20]=16'h4040;dmem3[22]=16'h4040;
            imem[0]=enc_mov_tid(4'd0);imem[1]=enc_movi(4'd1,16'd0);
            imem[2]=enc_movi(4'd2,16'd16);imem[3]=enc_movi(4'd3,16'd32);
            imem[4]=enc_i(`OP_SHL,1'b0,4'd4,4'd0,16'd1);
            imem[5]=enc_r(`OP_ADD,1'b0,4'd5,4'd1,4'd4);
            imem[6]=enc_m_f(`OP_LD,4'd5,4'd5,16'd0);
            imem[7]=enc_r(`OP_ADD,1'b0,4'd6,4'd2,4'd4);
            imem[8]=enc_m_f(`OP_LD,4'd6,4'd6,16'd0);
            imem[9]=enc_r(`OP_MUL,1'b1,4'd7,4'd5,4'd6);
            imem[10]=enc_r(`OP_ADD,1'b0,4'd8,4'd3,4'd4);
            imem[11]=enc_m_f(`OP_ST,4'd7,4'd8,16'd0);
            imem[12]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr_all(4'd7,16'h40C0,"K27 MUL.f 2*3=6.0");
        end

        begin
            $display("\n--- K28: PTX K4 bf16_fma ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'h4000;dmem1[2]=16'h4000;dmem2[4]=16'h4000;dmem3[6]=16'h4000;
            dmem0[16]=16'h4040;dmem1[18]=16'h4040;dmem2[20]=16'h4040;dmem3[22]=16'h4040;
            dmem0[32]=16'h3F80;dmem1[34]=16'h3F80;dmem2[36]=16'h3F80;dmem3[38]=16'h3F80;
            imem[0]=enc_mov_tid(4'd0);imem[1]=enc_movi(4'd1,16'd0);
            imem[2]=enc_movi(4'd2,16'd16);imem[3]=enc_movi(4'd3,16'd32);
            imem[4]=enc_movi(4'd9,16'd48);
            imem[5]=enc_i(`OP_SHL,1'b0,4'd4,4'd0,16'd1);
            imem[6]=enc_r(`OP_ADD,1'b0,4'd5,4'd1,4'd4);
            imem[7]=enc_m_f(`OP_LD,4'd5,4'd5,16'd0);
            imem[8]=enc_r(`OP_ADD,1'b0,4'd6,4'd2,4'd4);
            imem[9]=enc_m_f(`OP_LD,4'd6,4'd6,16'd0);
            imem[10]=enc_r(`OP_ADD,1'b0,4'd7,4'd3,4'd4);
            imem[11]=enc_m_f(`OP_LD,4'd7,4'd7,16'd0);
            imem[12]=enc_r(`OP_FMA,1'b1,4'd7,4'd5,4'd6);
            imem[13]=enc_r(`OP_ADD,1'b0,4'd8,4'd9,4'd4);
            imem[14]=enc_m_f(`OP_ST,4'd7,4'd8,16'd0);
            imem[15]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr_all(4'd7,16'h40E0,"K28 FMA.f 2*3+1=7.0");
        end

        begin
            $display("\n--- K29: PTX K5 relu bf16 ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'hBF80;dmem1[2]=16'h4000;dmem2[4]=16'hC040;dmem3[6]=16'h40A0;
            imem[0]=enc_mov_tid(4'd0);imem[1]=enc_movi(4'd1,16'd0);
            imem[2]=enc_movi(4'd2,16'd16);
            imem[3]=enc_i(`OP_SHL,1'b0,4'd4,4'd0,16'd1);
            imem[4]=enc_r(`OP_ADD,1'b0,4'd5,4'd1,4'd4);
            imem[5]=enc_m_f(`OP_LD,4'd5,4'd5,16'd0);
            imem[6]=enc_movi(4'd8,16'h0000);
            imem[7]=enc_r(`OP_MAX,1'b1,4'd6,4'd5,4'd8);
            imem[8]=enc_r(`OP_ADD,1'b0,4'd7,4'd2,4'd4);
            imem[9]=enc_m_f(`OP_ST,4'd6,4'd7,16'd0);
            imem[10]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr(2'd0,4'd6,16'h0000,"K29 SP0 relu(-1)=0");
            check_gpr(2'd1,4'd6,16'h4000,"K29 SP1 relu(2)=2.0");
            check_gpr(2'd2,4'd6,16'h0000,"K29 SP2 relu(-3)=0");
            check_gpr(2'd3,4'd6,16'h40A0,"K29 SP3 relu(5)=5.0");
        end

        begin
            $display("\n--- K30: PTX K6 wmma_bf16 matmul ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'h3F80;dmem0[1]=16'h0000;dmem0[2]=16'h0000;dmem0[3]=16'h0000;
            dmem1[8]=16'h0000;dmem1[9]=16'h3F80;dmem1[10]=16'h0000;dmem1[11]=16'h0000;
            dmem2[16]=16'h0000;dmem2[17]=16'h0000;dmem2[18]=16'h3F80;dmem2[19]=16'h0000;
            dmem3[24]=16'h0000;dmem3[25]=16'h0000;dmem3[26]=16'h0000;dmem3[27]=16'h3F80;
            dmem0[32]=16'h4000;dmem0[33]=16'h4000;dmem0[34]=16'h4000;dmem0[35]=16'h4000;
            dmem1[40]=16'h4000;dmem1[41]=16'h4000;dmem1[42]=16'h4000;dmem1[43]=16'h4000;
            dmem2[48]=16'h4000;dmem2[49]=16'h4000;dmem2[50]=16'h4000;dmem2[51]=16'h4000;
            dmem3[56]=16'h4000;dmem3[57]=16'h4000;dmem3[58]=16'h4000;dmem3[59]=16'h4000;
            imem[0]=enc_mov_tid(4'd0);imem[1]=enc_movi(4'd1,16'd0);
            imem[2]=enc_movi(4'd2,16'd32);imem[3]=enc_movi(4'd3,16'd64);
            imem[4]=enc_i(`OP_SHL,1'b0,4'd4,4'd0,16'd3);
            imem[5]=enc_r(`OP_ADD,1'b0,4'd1,4'd1,4'd4);
            imem[6]=enc_r(`OP_ADD,1'b0,4'd2,4'd2,4'd4);
            imem[7]=enc_r(`OP_ADD,1'b0,4'd3,4'd3,4'd4);
            imem[8]=enc_m_f(`WMMA_LOAD,4'd4,4'd1,16'd0);
            imem[9]=enc_m_f(`WMMA_LOAD,4'd8,4'd2,16'd0);
            imem[10]=enc_movi(4'd12,16'h0000);imem[11]=enc_movi(4'd13,16'h0000);
            imem[12]=enc_movi(4'd14,16'h0000);imem[13]=enc_movi(4'd15,16'h0000);
            imem[14]=enc_wmma_mma(4'd12,4'd4,4'd8,4'd12);
            imem[15]=enc_m_f(`WMMA_STORE,4'd12,4'd3,16'd0);
            imem[16]=INST_RET;
            launch_kernel(32'd0); wait_kernel_done(400);
            check_gpr_all(4'd12,16'h4000,"K30 D=2.0");
            check_gpr_all(4'd13,16'h4000,"K30 D=2.0");
            check_dmem(2'd0,10'd64,16'h4000,"K30 SP0 D[0][0]=2.0");
            check_dmem(2'd1,10'd72,16'h4000,"K30 SP1 D[1][0]=2.0");
            check_dmem(2'd2,10'd80,16'h4000,"K30 SP2 D[2][0]=2.0");
            check_dmem(2'd3,10'd88,16'h4000,"K30 SP3 D[3][0]=2.0");
        end

        // ================================================================
        // SIMT Divergence / Convergence Tests (K31–K37)
        // ================================================================

        begin
            $display("\n--- K31: SIMT 2+2 divergence ---");
            reset_dut; clear_dmem; enable_trace;
            imem[0]=enc_mov_tid(4'd0); imem[1]=enc_movi(4'd1,16'd2);
            imem[2]=enc_movi(4'd5,16'd0);
            imem[3]=enc_setp(1'b0,2'd2,4'd0,4'd0,4'd1);
            imem[4]=INST_NOP; imem[5]=INST_NOP; imem[6]=INST_NOP;
            imem[7]=enc_pbra(2'd0,13'd10,12'd12);
            imem[8]=enc_movi(4'd5,16'hBBBB); imem[9]=enc_bra(27'd12);
            imem[10]=enc_movi(4'd5,16'hAAAA); imem[11]=INST_NOP;
            imem[12]=enc_movi(4'd6,16'hCCCC); imem[13]=INST_RET;
            print_program(0,14);
            launch_kernel(32'd0); wait_kernel_done(200); disable_trace;
            check_gpr(2'd0,4'd5,16'hAAAA,"K31 SP0 R5=taken");
            check_gpr(2'd1,4'd5,16'hAAAA,"K31 SP1 R5=taken");
            check_gpr(2'd2,4'd5,16'hBBBB,"K31 SP2 R5=fall");
            check_gpr(2'd3,4'd5,16'hBBBB,"K31 SP3 R5=fall");
            check_gpr_all(4'd6,16'hCCCC,"K31 R6=reconv");
        end

        begin
            $display("\n--- K32: SIMT uniform-taken PBRA ---");
            reset_dut; clear_dmem; enable_trace;
            imem[0]=enc_movi(4'd5,16'd0); imem[1]=enc_movi(4'd6,16'd0);
            imem[2]=enc_set(2'd0,1'b1);
            imem[3]=INST_NOP; imem[4]=INST_NOP; imem[5]=INST_NOP;
            imem[6]=enc_pbra(2'd0,13'd9,12'd11);
            imem[7]=enc_movi(4'd5,16'hBBBB); imem[8]=enc_bra(27'd11);
            imem[9]=enc_movi(4'd5,16'hAAAA); imem[10]=INST_NOP;
            imem[11]=enc_movi(4'd6,16'hCCCC); imem[12]=INST_RET;
            print_program(0,13);
            launch_kernel(32'd0); wait_kernel_done(200); disable_trace;
            check_gpr_all(4'd5,16'hAAAA,"K32 R5=taken path");
            check_gpr_all(4'd6,16'hCCCC,"K32 R6=reconv");
        end

        begin
            $display("\n--- K33: SIMT uniform-fall PBRA ---");
            reset_dut; clear_dmem; enable_trace;
            imem[0]=enc_movi(4'd5,16'd0); imem[1]=enc_movi(4'd6,16'd0);
            imem[2]=enc_set(2'd0,1'b0);
            imem[3]=INST_NOP; imem[4]=INST_NOP; imem[5]=INST_NOP;
            imem[6]=enc_pbra(2'd0,13'd9,12'd11);
            imem[7]=enc_movi(4'd5,16'hBBBB); imem[8]=enc_bra(27'd11);
            imem[9]=enc_movi(4'd5,16'hAAAA); imem[10]=INST_NOP;
            imem[11]=enc_movi(4'd6,16'hCCCC); imem[12]=INST_RET;
            print_program(0,13);
            launch_kernel(32'd0); wait_kernel_done(200); disable_trace;
            check_gpr_all(4'd5,16'hBBBB,"K33 R5=fall path");
            check_gpr_all(4'd6,16'hCCCC,"K33 R6=reconv");
        end

        begin
            $display("\n--- K34: SIMT 1+3 asymmetric split ---");
            reset_dut; clear_dmem; enable_trace;
            imem[0]=enc_mov_tid(4'd0); imem[1]=enc_movi(4'd1,16'd1);
            imem[2]=enc_movi(4'd5,16'd0);
            imem[3]=enc_setp(1'b0,2'd2,4'd0,4'd0,4'd1);
            imem[4]=INST_NOP; imem[5]=INST_NOP; imem[6]=INST_NOP;
            imem[7]=enc_pbra(2'd0,13'd10,12'd12);
            imem[8]=enc_movi(4'd5,16'hBBBB); imem[9]=enc_bra(27'd12);
            imem[10]=enc_movi(4'd5,16'hAAAA); imem[11]=INST_NOP;
            imem[12]=enc_movi(4'd6,16'hCCCC); imem[13]=INST_RET;
            print_program(0,14);
            launch_kernel(32'd0); wait_kernel_done(200); disable_trace;
            check_gpr(2'd0,4'd5,16'hAAAA,"K34 SP0 R5=taken");
            check_gpr(2'd1,4'd5,16'hBBBB,"K34 SP1 R5=fall");
            check_gpr(2'd2,4'd5,16'hBBBB,"K34 SP2 R5=fall");
            check_gpr(2'd3,4'd5,16'hBBBB,"K34 SP3 R5=fall");
            check_gpr_all(4'd6,16'hCCCC,"K34 R6=reconv");
        end

        begin
            $display("\n--- K35: SIMT nested divergence (depth 2) ---");
            reset_dut; clear_dmem; enable_trace;
            imem[0]=enc_mov_tid(4'd0); imem[1]=enc_movi(4'd1,16'd2);
            imem[2]=enc_movi(4'd2,16'd1); imem[3]=enc_movi(4'd5,16'd0);
            imem[4]=enc_setp(1'b0,2'd2,4'd0,4'd0,4'd1);
            imem[5]=INST_NOP; imem[6]=INST_NOP; imem[7]=INST_NOP;
            imem[8]=enc_pbra(2'd0,13'd11,12'd22);
            imem[9]=enc_movi(4'd5,16'h3333); imem[10]=enc_bra(27'd22);
            imem[11]=enc_setp(1'b0,2'd2,4'd1,4'd0,4'd2);
            imem[12]=INST_NOP; imem[13]=INST_NOP; imem[14]=INST_NOP;
            imem[15]=enc_pbra(2'd1,13'd18,12'd20);
            imem[16]=enc_movi(4'd5,16'h2222); imem[17]=enc_bra(27'd20);
            imem[18]=enc_movi(4'd5,16'h1111); imem[19]=INST_NOP;
            imem[20]=INST_NOP; imem[21]=INST_NOP;
            imem[22]=enc_movi(4'd6,16'hDDDD); imem[23]=INST_RET;
            print_program(0,24);
            launch_kernel(32'd0); wait_kernel_done(400); disable_trace;
            dump_rf_range(4'd0,4'd7);
            check_gpr(2'd0,4'd5,16'h1111,"K35 SP0 inner taken");
            check_gpr(2'd1,4'd5,16'h2222,"K35 SP1 inner fall");
            check_gpr(2'd2,4'd5,16'h3333,"K35 SP2 outer fall");
            check_gpr(2'd3,4'd5,16'h3333,"K35 SP3 outer fall");
            check_gpr_all(4'd6,16'hDDDD,"K35 R6=reconv all");
        end

        begin
            $display("\n--- K36: SIMT divergent DMEM store ---");
            reset_dut; clear_dmem; enable_trace;
            imem[0]=enc_mov_tid(4'd0); imem[1]=enc_movi(4'd1,16'd2);
            imem[2]=enc_setp(1'b0,2'd2,4'd0,4'd0,4'd1);
            imem[3]=INST_NOP; imem[4]=INST_NOP; imem[5]=INST_NOP;
            imem[6]=enc_pbra(2'd0,13'd10,12'd13);
            imem[7]=enc_movi(4'd5,16'hBBBB);
            imem[8]=enc_m(`OP_ST,4'd5,4'd0,16'd200); imem[9]=enc_bra(27'd13);
            imem[10]=enc_movi(4'd5,16'hAAAA);
            imem[11]=enc_m(`OP_ST,4'd5,4'd0,16'd200); imem[12]=INST_NOP;
            imem[13]=INST_RET;
            print_program(0,14);
            launch_kernel(32'd0); wait_kernel_done(200); disable_trace;
            check_dmem(2'd0,10'd200,16'hAAAA,"K36 SP0 DMEM=taken");
            check_dmem(2'd1,10'd201,16'hAAAA,"K36 SP1 DMEM=taken");
            check_dmem(2'd2,10'd202,16'hBBBB,"K36 SP2 DMEM=fall");
            check_dmem(2'd3,10'd203,16'hBBBB,"K36 SP3 DMEM=fall");
        end

        begin
            $display("\n--- K37: SIMT PBRA after MUL (drain) ---");
            reset_dut; clear_dmem; enable_trace;
            imem[0]=enc_mov_tid(4'd0); imem[1]=enc_movi(4'd1,16'd3);
            imem[2]=enc_movi(4'd2,16'd7);
            imem[3]=enc_r(`OP_MUL,1'b0,4'd5,4'd1,4'd2);
            imem[4]=enc_setp(1'b0,2'd2,4'd0,4'd0,4'd1);
            imem[5]=INST_NOP; imem[6]=INST_NOP; imem[7]=INST_NOP;
            imem[8]=enc_pbra(2'd0,13'd11,12'd13);
            imem[9]=enc_i(`OP_ADDI,1'b0,4'd5,4'd5,16'd100); imem[10]=enc_bra(27'd13);
            imem[11]=enc_i(`OP_ADDI,1'b0,4'd5,4'd5,16'd10); imem[12]=INST_NOP;
            imem[13]=enc_m(`OP_ST,4'd5,4'd0,16'd300); imem[14]=INST_RET;
            print_program(0,15);
            launch_kernel(32'd0); wait_kernel_done(300); disable_trace;
            check_gpr(2'd0,4'd5,16'd31,"K37 SP0 R5=21+10");
            check_gpr(2'd1,4'd5,16'd31,"K37 SP1 R5=21+10");
            check_gpr(2'd2,4'd5,16'd31,"K37 SP2 R5=21+10");
            check_gpr(2'd3,4'd5,16'd121,"K37 SP3 R5=21+100");
            check_dmem(2'd0,10'd300,16'd31,"K37 SP0 DMEM=31");
            check_dmem(2'd1,10'd301,16'd31,"K37 SP1 DMEM=31");
            check_dmem(2'd2,10'd302,16'd31,"K37 SP2 DMEM=31");
            check_dmem(2'd3,10'd303,16'd121,"K37 SP3 DMEM=121");
        end

        // ── K38: SET instruction unit test ───────────────
        begin
            $display("\n--- K38: SET instruction unit test ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'hAAAA); imem[1]=enc_movi(4'd2,16'hBBBB);
            imem[2]=enc_set(2'd0,1'b1);
            imem[3]=INST_NOP; imem[4]=INST_NOP; imem[5]=INST_NOP;
            imem[6]={`OP_SELP,1'b0,2'd0,4'd3,4'd1,4'd2,12'd0};
            imem[7]=enc_set(2'd0,1'b0);
            imem[8]=INST_NOP; imem[9]=INST_NOP; imem[10]=INST_NOP;
            imem[11]={`OP_SELP,1'b0,2'd0,4'd4,4'd1,4'd2,12'd0};
            imem[12]=enc_set(2'd1,1'b1);
            imem[13]=INST_NOP; imem[14]=INST_NOP; imem[15]=INST_NOP;
            imem[16]={`OP_SELP,1'b0,2'd1,4'd5,4'd1,4'd2,12'd0};
            imem[17]=INST_RET;
            print_program(0,18);
            launch_kernel(32'd0); wait_kernel_done(200);
            check_gpr_all(4'd3,16'hAAAA,"K38 SET P0=1 SELP->R1");
            check_gpr_all(4'd4,16'hBBBB,"K38 SET P0=0 SELP->R2");
            check_gpr_all(4'd5,16'hAAAA,"K38 SET P1=1 SELP->R1");
        end

        // ================================================================
        // CVT Tests (K39–K47)
        // ================================================================

        begin
            $display("\n--- K39: MOVI+ST sanity baseline ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'hDEAD;dmem1[0]=16'hDEAD;dmem2[0]=16'hDEAD;dmem3[0]=16'hDEAD;
            imem[0]=enc_movi(4'd1,16'd15); imem[1]=enc_movi(4'd2,16'd0);
            imem[2]=enc_m(`OP_ST,4'd1,4'd2,16'd0); imem[3]=INST_RET;
            print_program(0,4);
            launch_kernel(32'd0); wait_kernel_done(100);
            check_gpr_all(4'd1,16'd15,"K39 R1=15");
            check_dmem_all(10'd0,16'd15,"K39 DMEM[0]=15");
        end

        begin
            $display("\n--- K40: CVT.i2f basic ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd15);
            imem[1]=INST_NOP;imem[2]=INST_NOP;imem[3]=INST_NOP;imem[4]=INST_NOP;
            imem[5]=INST_NOP;imem[6]=INST_NOP;imem[7]=INST_NOP;imem[8]=INST_NOP;
            imem[9]=enc_cvt(1'b1,4'd2,4'd1);
            imem[10]=INST_NOP;imem[11]=INST_NOP;imem[12]=INST_NOP;imem[13]=INST_NOP;
            imem[14]=INST_NOP;imem[15]=INST_NOP;imem[16]=INST_NOP;imem[17]=INST_NOP;
            imem[18]=enc_movi(4'd3,16'd0);
            imem[19]=enc_m(`OP_ST,4'd2,4'd3,16'd0); imem[20]=INST_RET;
            print_program(0,21);
            launch_kernel(32'd0); wait_kernel_done(200);
            dump_rf_range(4'd0,4'd5);
            check_gpr_all(4'd1,16'd15,"K40 R1=15 (unchanged)");
            check_gpr_all(4'd2,16'h4170,"K40 R2=bf16(15)=0x4170");
            check_dmem_all(10'd0,16'h4170,"K40 DMEM[0]=0x4170");
        end

        begin
            $display("\n--- K41: CVT.f2i basic ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'h4170);
            imem[1]=INST_NOP;imem[2]=INST_NOP;imem[3]=INST_NOP;imem[4]=INST_NOP;
            imem[5]=INST_NOP;imem[6]=INST_NOP;imem[7]=INST_NOP;imem[8]=INST_NOP;
            imem[9]=enc_cvt(1'b0,4'd2,4'd1);
            imem[10]=INST_NOP;imem[11]=INST_NOP;imem[12]=INST_NOP;imem[13]=INST_NOP;
            imem[14]=INST_NOP;imem[15]=INST_NOP;imem[16]=INST_NOP;imem[17]=INST_NOP;
            imem[18]=enc_movi(4'd3,16'd0);
            imem[19]=enc_m(`OP_ST,4'd2,4'd3,16'd0); imem[20]=INST_RET;
            print_program(0,21);
            launch_kernel(32'd0); wait_kernel_done(200);
            dump_rf_range(4'd0,4'd5);
            check_gpr_all(4'd1,16'h4170,"K41 R1=0x4170 (unchanged)");
            check_gpr_all(4'd2,16'd15,"K41 R2=f2i(15.0)=15");
            check_dmem_all(10'd0,16'd15,"K41 DMEM[0]=15");
        end

        begin
            $display("\n--- K42: CVT round-trip (diff regs, NOP padded) ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd15);
            imem[1]=INST_NOP;imem[2]=INST_NOP;imem[3]=INST_NOP;imem[4]=INST_NOP;
            imem[5]=INST_NOP;imem[6]=INST_NOP;imem[7]=INST_NOP;imem[8]=INST_NOP;
            imem[9]=enc_cvt(1'b1,4'd2,4'd1);
            imem[10]=INST_NOP;imem[11]=INST_NOP;imem[12]=INST_NOP;imem[13]=INST_NOP;
            imem[14]=INST_NOP;imem[15]=INST_NOP;imem[16]=INST_NOP;imem[17]=INST_NOP;
            imem[18]=enc_cvt(1'b0,4'd3,4'd2);
            imem[19]=INST_NOP;imem[20]=INST_NOP;imem[21]=INST_NOP;imem[22]=INST_NOP;
            imem[23]=INST_NOP;imem[24]=INST_NOP;imem[25]=INST_NOP;imem[26]=INST_NOP;
            imem[27]=enc_movi(4'd4,16'd0);
            imem[28]=enc_m(`OP_ST,4'd3,4'd4,16'd0); imem[29]=INST_RET;
            print_program(0,30);
            launch_kernel(32'd0); wait_kernel_done(300);
            dump_rf_range(4'd0,4'd5);
            check_gpr_all(4'd1,16'd15,"K42 R1=15");
            check_gpr_all(4'd2,16'h4170,"K42 R2=bf16(15)");
            check_gpr_all(4'd3,16'd15,"K42 R3=round-trip 15");
            check_dmem_all(10'd0,16'd15,"K42 DMEM[0]=15");
        end

        begin
            $display("\n--- K43: CVT round-trip (diff regs, SB stalls) ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd15);
            imem[1]=enc_cvt(1'b1,4'd2,4'd1);
            imem[2]=enc_cvt(1'b0,4'd3,4'd2);
            imem[3]=enc_movi(4'd4,16'd0);
            imem[4]=enc_m(`OP_ST,4'd3,4'd4,16'd0); imem[5]=INST_RET;
            print_program(0,6);
            launch_kernel(32'd0); wait_kernel_done(200);
            dump_rf_range(4'd0,4'd5);
            check_gpr_all(4'd1,16'd15,"K43 R1=15");
            check_gpr_all(4'd2,16'h4170,"K43 R2=bf16(15)");
            check_gpr_all(4'd3,16'd15,"K43 R3=round-trip 15");
            check_dmem_all(10'd0,16'd15,"K43 DMEM[0]=15");
        end

        begin
            $display("\n--- K44: CVT self-ref rD==rA (NOP padded) ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd15);
            imem[1]=INST_NOP;imem[2]=INST_NOP;imem[3]=INST_NOP;imem[4]=INST_NOP;
            imem[5]=INST_NOP;imem[6]=INST_NOP;imem[7]=INST_NOP;imem[8]=INST_NOP;
            imem[9]=enc_cvt(1'b1,4'd1,4'd1);
            imem[10]=INST_NOP;imem[11]=INST_NOP;imem[12]=INST_NOP;imem[13]=INST_NOP;
            imem[14]=INST_NOP;imem[15]=INST_NOP;imem[16]=INST_NOP;imem[17]=INST_NOP;
            imem[18]=enc_movi(4'd5,16'd0);
            imem[19]=enc_m(`OP_ST,4'd1,4'd5,16'd0);
            imem[20]=INST_NOP;imem[21]=INST_NOP;imem[22]=INST_NOP;imem[23]=INST_NOP;
            imem[24]=enc_cvt(1'b0,4'd1,4'd1);
            imem[25]=INST_NOP;imem[26]=INST_NOP;imem[27]=INST_NOP;imem[28]=INST_NOP;
            imem[29]=INST_NOP;imem[30]=INST_NOP;imem[31]=INST_NOP;imem[32]=INST_NOP;
            imem[33]=enc_m(`OP_ST,4'd1,4'd5,16'd1); imem[34]=INST_RET;
            print_program(0,35);
            launch_kernel(32'd0); wait_kernel_done(300);
            dump_rf_range(4'd0,4'd5);
            check_gpr_all(4'd1,16'd15,"K44 R1=15 (round-trip)");
            check_dmem_all(10'd0,16'h4170,"K44 DMEM[0]=bf16(15)=0x4170");
            check_dmem_all(10'd1,16'd15,"K44 DMEM[1]=15");
        end

        begin
            $display("\n--- K45: CVT self-ref rD==rA (SB stalls) ---");
            reset_dut; clear_dmem;
            dmem0[0]=16'hBEEF;dmem1[0]=16'hBEEF;dmem2[0]=16'hBEEF;dmem3[0]=16'hBEEF;
            imem[0]=enc_movi(4'd0,16'd0);
            imem[1]=enc_movi(4'd1,16'd15);
            imem[2]=enc_cvt(1'b1,4'd1,4'd1);
            imem[3]=enc_cvt(1'b0,4'd1,4'd1);
            imem[4]=enc_m(`OP_ST,4'd1,4'd0,16'd0);
            imem[5]=enc_movi(4'd2,16'd0);
            imem[6]=enc_m(`OP_ST,4'd2,4'd0,16'd1); imem[7]=INST_RET;
            print_program(0,8);
            launch_kernel(32'd0); wait_kernel_done(200);
            dump_rf_range(4'd0,4'd5);
            check_gpr_all(4'd1,16'd15,"K45 R1=15 (self-ref round-trip)");
            check_dmem_all(10'd0,16'd15,"K45 DMEM[0]=15");
            check_dmem_all(10'd1,16'd0,"K45 DMEM[1]=0");
        end

        begin
            $display("\n--- K46: CVT.i2f known values sweep ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd0); imem[1]=enc_movi(4'd2,16'd1);
            imem[2]=enc_movi(4'd3,16'd2); imem[3]=enc_movi(4'd4,16'd15);
            imem[4]=enc_movi(4'd5,16'd100);
            imem[5]=INST_NOP;imem[6]=INST_NOP;imem[7]=INST_NOP;imem[8]=INST_NOP;
            imem[9]=INST_NOP;imem[10]=INST_NOP;imem[11]=INST_NOP;imem[12]=INST_NOP;
            imem[13]=enc_cvt(1'b1,4'd6,4'd1);
            imem[14]=enc_cvt(1'b1,4'd7,4'd2);
            imem[15]=enc_cvt(1'b1,4'd8,4'd3);
            imem[16]=enc_cvt(1'b1,4'd9,4'd4);
            imem[17]=enc_cvt(1'b1,4'd10,4'd5);
            imem[18]=INST_NOP;imem[19]=INST_NOP;imem[20]=INST_NOP;imem[21]=INST_NOP;
            imem[22]=INST_NOP;imem[23]=INST_NOP;imem[24]=INST_NOP;imem[25]=INST_NOP;
            imem[26]=INST_NOP;imem[27]=INST_NOP;imem[28]=INST_NOP;imem[29]=INST_NOP;
            imem[30]=enc_m(`OP_ST,4'd6,4'd1,16'd0);
            imem[31]=enc_m(`OP_ST,4'd7,4'd1,16'd1);
            imem[32]=enc_m(`OP_ST,4'd8,4'd1,16'd2);
            imem[33]=enc_m(`OP_ST,4'd9,4'd1,16'd3);
            imem[34]=enc_m(`OP_ST,4'd10,4'd1,16'd4);
            imem[35]=INST_RET;
            print_program(0,36);
            launch_kernel(32'd0); wait_kernel_done(400);
            dump_rf_range(4'd1,4'd10);
            check_gpr_all(4'd6,16'h0000,"K46 cvt(0)=0x0000");
            check_gpr_all(4'd7,16'h3F80,"K46 cvt(1)=0x3F80");
            check_gpr_all(4'd8,16'h4000,"K46 cvt(2)=0x4000");
            check_gpr_all(4'd9,16'h4170,"K46 cvt(15)=0x4170");
            check_gpr_all(4'd10,16'h42C8,"K46 cvt(100)=0x42C8");
            check_dmem_all(10'd0,16'h0000,"K46 DMEM[0] cvt(0)");
            check_dmem_all(10'd1,16'h3F80,"K46 DMEM[1] cvt(1)");
            check_dmem_all(10'd2,16'h4000,"K46 DMEM[2] cvt(2)");
            check_dmem_all(10'd3,16'h4170,"K46 DMEM[3] cvt(15)");
            check_dmem_all(10'd4,16'h42C8,"K46 DMEM[4] cvt(100)");
        end

        begin
            $display("\n--- K47: CVT.i2f + ADD.f cross-check ---");
            reset_dut; clear_dmem;
            imem[0]=enc_movi(4'd1,16'd15);
            imem[1]=INST_NOP;imem[2]=INST_NOP;imem[3]=INST_NOP;imem[4]=INST_NOP;
            imem[5]=INST_NOP;imem[6]=INST_NOP;imem[7]=INST_NOP;imem[8]=INST_NOP;
            imem[9]=enc_cvt(1'b1,4'd2,4'd1);
            imem[10]=INST_NOP;imem[11]=INST_NOP;imem[12]=INST_NOP;imem[13]=INST_NOP;
            imem[14]=INST_NOP;imem[15]=INST_NOP;imem[16]=INST_NOP;imem[17]=INST_NOP;
            imem[18]=enc_r(`OP_ADD,1'b1,4'd3,4'd2,4'd2);
            imem[19]=INST_NOP;imem[20]=INST_NOP;imem[21]=INST_NOP;imem[22]=INST_NOP;
            imem[23]=INST_NOP;imem[24]=INST_NOP;
            imem[25]=enc_movi(4'd4,16'd0);
            imem[26]=enc_m(`OP_ST,4'd2,4'd4,16'd0);
            imem[27]=enc_m(`OP_ST,4'd3,4'd4,16'd1);
            imem[28]=INST_RET;
            print_program(0,29);
            launch_kernel(32'd0); wait_kernel_done(300);
            dump_rf_range(4'd0,4'd5);
            check_gpr_all(4'd2,16'h4170,"K47 R2=bf16(15)=0x4170");
            check_gpr_all(4'd3,16'h41F0,"K47 R3=bf16(30)=0x41F0");
            check_dmem_all(10'd0,16'h4170,"K47 DMEM[0]=0x4170");
            check_dmem_all(10'd1,16'h41F0,"K47 DMEM[1]=0x41F0");
        end

        // ── Summary ──────────────────────────────────────
        $display("\n============================================");
        $display("  SM Core Testbench v1.5 — Summary");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        $display("  TOTAL:  %0d", pass_count + fail_count);
        $display("============================================");
        if (fail_count == 0)
            $display(">>> ALL TESTS PASSED <<<");
        else
            $display(">>> SOME TESTS FAILED <<<");
        $finish;
    end

endmodule