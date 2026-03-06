/* file: sm_core_tb_cvt.v
 * CVT-focused testbench addition for sm_core.
 * Adds K39-K45 CVT tests with deep pipeline tracing.
 *
 * Run standalone or append test cases to sm_core_tb.v
 *
 * Key diagnostic signals traced per cycle:
 *   - RF read address + data (rr_rf_r0_addr → ppl_rf_r0_data)
 *   - id_ex operand A (actual CVT input)
 *   - CVT pipeline: valid_in, pipe_valid, done, result
 *   - EX/MEM capture: result, rf_we
 *   - WB writeback:  w0_we, w0_addr, w0_data
 *   - Scoreboard pending bits
 *
 * Author: Jeremy Cai
 * Date: Mar. 5, 2026
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
    // DUT — sm_core v1.4 (no debug_rf ports)
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
    // IMEM BRAM Model (sync read)
    // ================================================================
    reg [31:0] imem [0:255];
    always @(posedge clk)
        imem_rdata <= imem[imem_addr];

    // ================================================================
    // Per-SP DMEM BRAM Models (sync read + sync write)
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

    function [31:0] enc_mov_tid;
        input [3:0] rd;
        enc_mov_tid = {`OP_MOV, 1'b1, 2'b00, rd, 4'd0, 16'd0};
    endfunction

    // CVT encoding: {OP_CVT, DT, 2'b00, rD, rA, 16'h0}
    //   dt=1: int16 → bf16 (i2f)
    //   dt=0: bf16 → int16 (f2i)
    function [31:0] enc_cvt;
        input dt;
        input [3:0] rd, ra;
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

    task tick; begin @(posedge clk); #1; end endtask

    task reset_dut;
        integer i;
    begin
        rst_n = 0;
        kernel_start = 0;
        kernel_entry_pc = 32'd0;
        thread_mask = 4'b1111;
        for (i = 0; i < 256; i = i + 1) imem[i] = INST_NOP;
        repeat (3) tick;
        rst_n = 1;
        tick;
    end
    endtask

    task clear_dmem;
        integer i;
    begin
        for (i = 0; i < 1024; i = i + 1) begin
            dmem0[i] = 16'd0;
            dmem1[i] = 16'd0;
            dmem2[i] = 16'd0;
            dmem3[i] = 16'd0;
        end
    end
    endtask

    task launch_kernel;
        input [31:0] entry_pc;
    begin
        kernel_entry_pc = entry_pc;
        kernel_start = 1;
        tick;
        kernel_start = 0;
    end
    endtask

    task wait_kernel_done;
        input integer max_cycles;
        integer cyc;
    begin
        cyc = 0;
        while (!kernel_done && cyc < max_cycles) begin
            tick;
            cyc = cyc + 1;
        end
        if (!kernel_done)
            $display("[TIMEOUT] kernel_done not asserted within %0d cycles", max_cycles);
        tick;
    end
    endtask

    // ── GPR read helper (works around Icarus variable-index limitation) ──
    // Reads SP's RF via the override read port (combinational).
    // TC/BU must be idle (they are, after kernel_done).
    reg [15:0] dbg_rf_val [0:3];

    task read_gpr;
        input [3:0] addr;
        // Drive override read address into sm_core's mux input.
        // When TC and BU are idle, ovr_rf_r0_addr_mux = 0, but we
        // need to force the address. Use the TB-level force/release.
    begin
        // Force the override address mux
        force u_dut.ovr_rf_r0_addr_mux = addr;
        #1; // combinational settle
        dbg_rf_val[0] = u_dut.SP_LANE[0].u_sp.ovr_rf_r0_data;
        dbg_rf_val[1] = u_dut.SP_LANE[1].u_sp.ovr_rf_r0_data;
        dbg_rf_val[2] = u_dut.SP_LANE[2].u_sp.ovr_rf_r0_data;
        dbg_rf_val[3] = u_dut.SP_LANE[3].u_sp.ovr_rf_r0_data;
        release u_dut.ovr_rf_r0_addr_mux;
        #1;
    end
    endtask

    // ── GPR check via override read port ─────────────────
    task check_gpr;
        input [1:0] sp;
        input [3:0] addr;
        input [15:0] expected;
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
        input [3:0] addr;
        input [15:0] expected;
        input [80*8-1:0] test_name;
    begin
        check_gpr(2'd0, addr, expected, test_name);
        check_gpr(2'd1, addr, expected, test_name);
        check_gpr(2'd2, addr, expected, test_name);
        check_gpr(2'd3, addr, expected, test_name);
    end
    endtask

    task check_dmem;
        input [1:0] sp;
        input [9:0] addr;
        input [15:0] expected;
        input [80*8-1:0] test_name;
        reg [15:0] actual;
    begin
        test_num = test_num + 1;
        case (sp)
            2'd0: actual = dmem0[addr];
            2'd1: actual = dmem1[addr];
            2'd2: actual = dmem2[addr];
            2'd3: actual = dmem3[addr];
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
        input [9:0] addr;
        input [15:0] expected;
        input [80*8-1:0] test_name;
    begin
        check_dmem(2'd0, addr, expected, test_name);
        check_dmem(2'd1, addr, expected, test_name);
        check_dmem(2'd2, addr, expected, test_name);
        check_dmem(2'd3, addr, expected, test_name);
    end
    endtask

    // ── Dump SP0-3 RF range via override port ───────────
    task dump_rf_range;
        input [3:0] r_start;
        input [3:0] r_end;
        integer ri, si;
    begin
        $write("  RF dump     ");
        for (ri = r_start; ri <= r_end; ri = ri + 1)
            $write("  R%-2d  ", ri);
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

    // ── Print program listing ────────────────────────────
    task disasm;
        input integer addr;
        input [31:0] inst;
        reg [4:0] op;
        reg dt;
        reg [3:0] rD, rA;
        reg [15:0] imm;
    begin
        op = inst[31:27]; dt = inst[26];
        rD = inst[23:20]; rA = inst[19:16]; imm = inst[15:0];
        case (op)
            `OP_NOP:  $display("    [%2d] %08h  NOP", addr, inst);
            `OP_MOVI: $display("    [%2d] %08h  MOVI    R%0d, %0d (0x%04h)", addr, inst, rD, imm, imm);
            `OP_MOV:  begin
                if (dt) $display("    [%2d] %08h  MOV.TID R%0d", addr, inst, rD);
                else    $display("    [%2d] %08h  MOV     R%0d, R%0d", addr, inst, rD, rA);
            end
            `OP_ADD:  $display("    [%2d] %08h  ADD%s   R%0d, R%0d, R%0d", addr, inst, dt?".f":"  ", rD, rA, inst[15:12]);
            `OP_CVT:  $display("    [%2d] %08h  CVT.%s  R%0d, R%0d", addr, inst, dt?"i2f":"f2i", rD, rA);
            `OP_LD:   $display("    [%2d] %08h  LD      R%0d, [R%0d + %0d]", addr, inst, rD, rA, imm);
            `OP_ST:   $display("    [%2d] %08h  ST      R%0d, [R%0d + %0d]", addr, inst, rD, rA, imm);
            `OP_RET:  $display("    [%2d] %08h  RET", addr, inst);
            default:  $display("    [%2d] %08h  op=%0d dt=%b rD=%0d rA=%0d", addr, inst, op, dt, rD, rA);
        endcase
    end
    endtask

    task print_program;
        input integer start_addr;
        input integer length;
        integer pi;
    begin
        $display("  Program listing:");
        for (pi = start_addr; pi < start_addr + length; pi = pi + 1)
            disasm(pi, imem[pi]);
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
        #500000;
        $display("[TIMEOUT] Global simulation timeout reached");
        $finish;
    end

    // ================================================================
    // CVT Deep Diagnostic Trace
    // ================================================================
    integer cycle_count;
    reg trace_en;
    reg cvt_trace_en;

    initial begin
        cycle_count = 0;
        trace_en = 0;
        cvt_trace_en = 0;
    end

    // ── Standard pipeline trace ──────────────────────────
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;

        if (trace_en || cvt_trace_en) begin
            // Line 1: Pipeline overview
            $display("  [C%03d] PC=%0d fv=%b | fstl=%b sp_stl=%b sb=%b exb=%b|%b|%b|%b drn=%b | amask=%b done=%b | de_v=%b de_op=%02h rD=%0d",
                cycle_count,
                u_dut.u_fetch.pc_reg,
                u_dut.fetch_valid,
                u_dut.front_stall,
                u_dut.sp_stall,
                u_dut.sb_stall,
                u_dut.SP_LANE[0].u_sp.ex_busy,
                u_dut.SP_LANE[1].u_sp.ex_busy,
                u_dut.SP_LANE[2].u_sp.ex_busy,
                u_dut.SP_LANE[3].u_sp.ex_busy,
                u_dut.pipeline_drained,
                u_dut.active_mask,
                kernel_done,
                u_dut.de_valid,
                u_dut.de_opcode,
                u_dut.de_rD_addr);

            // Line 2: RR stage
            $display("         RR: v=%b op=%02h rD=%0d we=%b dt=%b | addr0=%0d addr1=%0d",
                u_dut.rr_valid,
                u_dut.rr_opcode,
                u_dut.rr_rD_addr,
                u_dut.rr_rf_we,
                u_dut.rr_dt,
                u_dut.rr_rf_r0_addr,
                u_dut.rr_rf_r1_addr);

            // Line 3: Scoreboard
            $display("         SB: pend[0]=%04h [1]=%04h [2]=%04h [3]=%04h | wb_we=%b wb_rD=%0d",
                u_dut.u_sb.pending[0],
                u_dut.u_sb.pending[1],
                u_dut.u_sb.pending[2],
                u_dut.u_sb.pending[3],
                u_dut.sb_wb_rf_we_any,
                u_dut.sp_wb_rD_addr[0]);
        end

        if (cvt_trace_en) begin
            // Line 4: SP0 id_ex — the actual operand values
            $display("         SP0 id_ex: v=%b op=%02h dt=%b rD=%0d launched=%b | opA=%04h opB=%04h | rf_we=%b active=%b",
                u_dut.SP_LANE[0].u_sp.id_ex_valid,
                u_dut.SP_LANE[0].u_sp.id_ex_opcode,
                u_dut.SP_LANE[0].u_sp.id_ex_dt,
                u_dut.SP_LANE[0].u_sp.id_ex_rD_addr,
                u_dut.SP_LANE[0].u_sp.id_ex_launched,
                u_dut.SP_LANE[0].u_sp.id_ex_opA,
                u_dut.SP_LANE[0].u_sp.id_ex_opB,
                u_dut.SP_LANE[0].u_sp.id_ex_rf_we,
                u_dut.SP_LANE[0].u_sp.id_ex_active);

            // Line 5: CVT pipeline — valid signals + result
            $display("         SP0 CVT: valid_in=%b pipe=%02b done=%b busy=%b | cvt_result=%04h | is_cvt=%b ex_valid=%b",
                u_dut.SP_LANE[0].u_sp.cvt_valid_in,
                u_dut.SP_LANE[0].u_sp.cvt_pipe_valid,
                u_dut.SP_LANE[0].u_sp.cvt_done,
                u_dut.SP_LANE[0].u_sp.cvt_busy,
                u_dut.SP_LANE[0].u_sp.cvt_result,
                u_dut.SP_LANE[0].u_sp.is_cvt,
                u_dut.SP_LANE[0].u_sp.ex_valid_out);

            // Line 6: EX/MEM
            $display("         SP0 EX/MEM: v=%b rD=%0d we=%b result=%04h | ex_result_muxed=%04h",
                u_dut.SP_LANE[0].u_sp.ex_mem_valid,
                u_dut.SP_LANE[0].u_sp.ex_mem_rD_addr,
                u_dut.SP_LANE[0].u_sp.ex_mem_rf_we,
                u_dut.SP_LANE[0].u_sp.ex_mem_result,
                u_dut.SP_LANE[0].u_sp.ex_result_muxed);

            // Line 7: MEM/WB
            $display("         SP0 MEM/WB: v=%b rD=%0d we=%b data=%04h active=%b",
                u_dut.SP_LANE[0].u_sp.mem_wb_valid,
                u_dut.SP_LANE[0].u_sp.mem_wb_rD_addr,
                u_dut.SP_LANE[0].u_sp.mem_wb_rf_we,
                u_dut.SP_LANE[0].u_sp.mem_wb_data,
                u_dut.SP_LANE[0].u_sp.mem_wb_active);

            // Line 8: WB — actual RF write
            $display("         SP0 WB: w0_we=%b w0_addr=%0d w0_data=%04h | wb_data_final=%04h",
                u_dut.SP_LANE[0].u_sp.w0_we,
                u_dut.SP_LANE[0].u_sp.w0_addr,
                u_dut.SP_LANE[0].u_sp.w0_data,
                u_dut.SP_LANE[0].u_sp.wb_data_final);

            // Line 9: RF read port — what the pipeline actually reads
            $display("         SP0 RF_RD: ppl_r0_addr=%0d ppl_r0_data=%04h | stall=%b",
                u_dut.SP_LANE[0].u_sp.ppl_rf_r0_addr,
                u_dut.SP_LANE[0].u_sp.ppl_rf_r0_data,
                u_dut.SP_LANE[0].u_sp.stall);

            // Line 10: ALU/FPU valid (to distinguish from CVT path)
            $display("         SP0 ALU/FPU: alu_v_in=%b alu_v_out=%b fpu_v_in=%b fpu_v_out=%b | set_v=%b",
                u_dut.SP_LANE[0].u_sp.alu_valid_in,
                u_dut.SP_LANE[0].u_sp.alu_valid_out,
                u_dut.SP_LANE[0].u_sp.fpu_valid_in,
                u_dut.SP_LANE[0].u_sp.fpu_valid_out,
                u_dut.SP_LANE[0].u_sp.set_valid);

            // Line 11: DMEM write observation
            if (|dmem_wea)
                $display("         >>> DMEM WRITE: we=%04b addr0=%0d din0=%04h addr1=%0d din1=%04h",
                    dmem_wea,
                    dmem_addra[0*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH],
                    dmem_dina[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH],
                    dmem_addra[1*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH],
                    dmem_dina[1*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]);

            $display(""); // blank line for readability
        end
    end

    // ================================================================
    // Main Test Sequence — CVT-focused tests K39-K45
    // ================================================================
    initial begin
        $display("============================================");
        $display("  SM Core CVT Testbench — Starting");
        $display("  Tests K39-K45: CVT pipeline diagnostics");
        $display("============================================");

        // ════════════════════════════════════════════════════════
        //  K39: MOVI sanity (non-CVT baseline)
        //  Verify basic MOVI+ST works before testing CVT.
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K39: MOVI+ST sanity baseline ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'hDEAD; dmem1[0] = 16'hDEAD;
            dmem2[0] = 16'hDEAD; dmem3[0] = 16'hDEAD;

            imem[0] = enc_movi(4'd1, 16'd15);    // R1 = 15
            imem[1] = enc_movi(4'd2, 16'd0);     // R2 = 0 (base addr)
            imem[2] = enc_m(`OP_ST, 4'd1, 4'd2, 16'd0); // DMEM[0] = R1
            imem[3] = INST_RET;

            print_program(0, 4);
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd1, 16'd15, "K39 R1=15");
            check_dmem_all(10'd0, 16'd15, "K39 DMEM[0]=15");
        end

        // ════════════════════════════════════════════════════════
        //  K40: CVT.i2f basic — different regs, NO scoreboard stall
        //
        //  MOVI R1, 15
        //  NOP × 8  (guarantee R1 written back, SB clear)
        //  CVT.i2f R2, R1  (dt=1, rD≠rA)
        //  NOP × 8  (guarantee R2 written back)
        //  MOVI R3, 0
        //  ST R2, [R3+0]
        //  RET
        //
        //  Expected: R2 = bf16(15.0) = 0x4170
        //            DMEM[0] = 0x4170
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K40: CVT.i2f basic (no stall, diff regs) ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'hDEAD; dmem1[0] = 16'hDEAD;
            dmem2[0] = 16'hDEAD; dmem3[0] = 16'hDEAD;
            cvt_trace_en = 1;
            cycle_count = 0;

            imem[0]  = enc_movi(4'd1, 16'd15);
            imem[1]  = INST_NOP;
            imem[2]  = INST_NOP;
            imem[3]  = INST_NOP;
            imem[4]  = INST_NOP;
            imem[5]  = INST_NOP;
            imem[6]  = INST_NOP;
            imem[7]  = INST_NOP;
            imem[8]  = INST_NOP;
            imem[9]  = enc_cvt(1'b1, 4'd2, 4'd1);   // CVT.i2f R2, R1
            imem[10] = INST_NOP;
            imem[11] = INST_NOP;
            imem[12] = INST_NOP;
            imem[13] = INST_NOP;
            imem[14] = INST_NOP;
            imem[15] = INST_NOP;
            imem[16] = INST_NOP;
            imem[17] = INST_NOP;
            imem[18] = enc_movi(4'd3, 16'd0);
            imem[19] = enc_m(`OP_ST, 4'd2, 4'd3, 16'd0); // DMEM[0] = R2
            imem[20] = INST_RET;

            $display("  bf16(15.0) = 0x4170");
            print_program(0, 21);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            cvt_trace_en = 0;

            $display("  Post-execution RF dump:");
            dump_rf_range(4'd0, 4'd5);

            check_gpr_all(4'd1, 16'd15,    "K40 R1=15 (unchanged)");
            check_gpr_all(4'd2, 16'h4170,  "K40 R2=bf16(15)=0x4170");
            check_dmem_all(10'd0, 16'h4170, "K40 DMEM[0]=0x4170");
        end

        // ════════════════════════════════════════════════════════
        //  K41: CVT.f2i basic — different regs, NO scoreboard stall
        //
        //  MOVI R1, 0x4170  (bf16 for 15.0)
        //  NOP × 8
        //  CVT.f2i R2, R1  (dt=0, rD≠rA)
        //  NOP × 8
        //  MOVI R3, 0
        //  ST R2, [R3+0]
        //  RET
        //
        //  Expected: R2 = 15
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K41: CVT.f2i basic (no stall, diff regs) ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'hDEAD; dmem1[0] = 16'hDEAD;
            dmem2[0] = 16'hDEAD; dmem3[0] = 16'hDEAD;
            cvt_trace_en = 1;
            cycle_count = 0;

            imem[0]  = enc_movi(4'd1, 16'h4170);     // R1 = bf16(15.0)
            imem[1]  = INST_NOP;
            imem[2]  = INST_NOP;
            imem[3]  = INST_NOP;
            imem[4]  = INST_NOP;
            imem[5]  = INST_NOP;
            imem[6]  = INST_NOP;
            imem[7]  = INST_NOP;
            imem[8]  = INST_NOP;
            imem[9]  = enc_cvt(1'b0, 4'd2, 4'd1);   // CVT.f2i R2, R1
            imem[10] = INST_NOP;
            imem[11] = INST_NOP;
            imem[12] = INST_NOP;
            imem[13] = INST_NOP;
            imem[14] = INST_NOP;
            imem[15] = INST_NOP;
            imem[16] = INST_NOP;
            imem[17] = INST_NOP;
            imem[18] = enc_movi(4'd3, 16'd0);
            imem[19] = enc_m(`OP_ST, 4'd2, 4'd3, 16'd0);
            imem[20] = INST_RET;

            print_program(0, 21);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            cvt_trace_en = 0;

            dump_rf_range(4'd0, 4'd5);

            check_gpr_all(4'd1, 16'h4170,  "K41 R1=0x4170 (unchanged)");
            check_gpr_all(4'd2, 16'd15,    "K41 R2=f2i(15.0)=15");
            check_dmem_all(10'd0, 16'd15,   "K41 DMEM[0]=15");
        end

        // ════════════════════════════════════════════════════════
        //  K42: CVT round-trip — different regs, NOP-padded
        //
        //  MOVI R1, 15
        //  NOP × 8
        //  CVT.i2f R2, R1
        //  NOP × 8
        //  CVT.f2i R3, R2
        //  NOP × 8
        //  MOVI R4, 0
        //  ST R3, [R4+0]
        //  RET
        //
        //  Expected: R2=0x4170, R3=15, DMEM[0]=15
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K42: CVT round-trip (diff regs, NOP padded) ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'hBEEF; dmem1[0] = 16'hBEEF;
            dmem2[0] = 16'hBEEF; dmem3[0] = 16'hBEEF;
            cvt_trace_en = 1;
            cycle_count = 0;

            imem[0]  = enc_movi(4'd1, 16'd15);       // R1 = 15
            imem[1]  = INST_NOP; imem[2]  = INST_NOP;
            imem[3]  = INST_NOP; imem[4]  = INST_NOP;
            imem[5]  = INST_NOP; imem[6]  = INST_NOP;
            imem[7]  = INST_NOP; imem[8]  = INST_NOP;
            imem[9]  = enc_cvt(1'b1, 4'd2, 4'd1);   // CVT.i2f R2, R1
            imem[10] = INST_NOP; imem[11] = INST_NOP;
            imem[12] = INST_NOP; imem[13] = INST_NOP;
            imem[14] = INST_NOP; imem[15] = INST_NOP;
            imem[16] = INST_NOP; imem[17] = INST_NOP;
            imem[18] = enc_cvt(1'b0, 4'd3, 4'd2);   // CVT.f2i R3, R2
            imem[19] = INST_NOP; imem[20] = INST_NOP;
            imem[21] = INST_NOP; imem[22] = INST_NOP;
            imem[23] = INST_NOP; imem[24] = INST_NOP;
            imem[25] = INST_NOP; imem[26] = INST_NOP;
            imem[27] = enc_movi(4'd4, 16'd0);
            imem[28] = enc_m(`OP_ST, 4'd3, 4'd4, 16'd0);
            imem[29] = INST_RET;

            print_program(0, 30);
            launch_kernel(32'd0);
            wait_kernel_done(300);
            cvt_trace_en = 0;

            dump_rf_range(4'd0, 4'd5);

            check_gpr_all(4'd1, 16'd15,    "K42 R1=15");
            check_gpr_all(4'd2, 16'h4170,  "K42 R2=bf16(15)");
            check_gpr_all(4'd3, 16'd15,    "K42 R3=round-trip 15");
            check_dmem_all(10'd0, 16'd15,   "K42 DMEM[0]=15");
        end

        // ════════════════════════════════════════════════════════
        //  K43: CVT round-trip — different regs, scoreboard stalls
        //
        //  MOVI R1, 15
        //  CVT.i2f R2, R1   (SB stall: R1 pending from MOVI)
        //  CVT.f2i R3, R2   (SB stall: R2 pending from CVT.i2f)
        //  MOVI R4, 0
        //  ST R3, [R4+0]    (SB stall: R3 pending from CVT.f2i)
        //  RET
        //
        //  Expected: same results, but exercises stall-resume path
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K43: CVT round-trip (diff regs, SB stalls) ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'hBEEF; dmem1[0] = 16'hBEEF;
            dmem2[0] = 16'hBEEF; dmem3[0] = 16'hBEEF;
            cvt_trace_en = 1;
            cycle_count = 0;

            imem[0] = enc_movi(4'd1, 16'd15);
            imem[1] = enc_cvt(1'b1, 4'd2, 4'd1);    // CVT.i2f R2, R1
            imem[2] = enc_cvt(1'b0, 4'd3, 4'd2);    // CVT.f2i R3, R2
            imem[3] = enc_movi(4'd4, 16'd0);
            imem[4] = enc_m(`OP_ST, 4'd3, 4'd4, 16'd0);
            imem[5] = INST_RET;

            print_program(0, 6);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            cvt_trace_en = 0;

            dump_rf_range(4'd0, 4'd5);

            check_gpr_all(4'd1, 16'd15,    "K43 R1=15");
            check_gpr_all(4'd2, 16'h4170,  "K43 R2=bf16(15)");
            check_gpr_all(4'd3, 16'd15,    "K43 R3=round-trip 15");
            check_dmem_all(10'd0, 16'd15,   "K43 DMEM[0]=15");
        end

        // ════════════════════════════════════════════════════════
        //  K44: CVT self-ref (rD==rA) — NOP padded, NO SB stall
        //
        //  MOVI R1, 15
        //  NOP × 8
        //  CVT.i2f R1, R1   (self-referencing, rD=rA=1)
        //  NOP × 8
        //  MOVI R5, 0
        //  ST R1, [R5+0]    (intermediate: should be bf16(15)=0x4170)
        //  NOP × 4
        //  CVT.f2i R1, R1   (self-ref again)
        //  NOP × 8
        //  ST R1, [R5+1]    (final: should be 15)
        //  RET
        //
        //  Expected: DMEM[0]=0x4170, DMEM[1]=15
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K44: CVT self-ref rD==rA (NOP padded) ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'hDEAD; dmem1[0] = 16'hDEAD;
            dmem2[0] = 16'hDEAD; dmem3[0] = 16'hDEAD;
            dmem0[1] = 16'hDEAD; dmem1[1] = 16'hDEAD;
            dmem2[1] = 16'hDEAD; dmem3[1] = 16'hDEAD;
            cvt_trace_en = 1;
            cycle_count = 0;

            imem[0]  = enc_movi(4'd1, 16'd15);
            imem[1]  = INST_NOP; imem[2]  = INST_NOP;
            imem[3]  = INST_NOP; imem[4]  = INST_NOP;
            imem[5]  = INST_NOP; imem[6]  = INST_NOP;
            imem[7]  = INST_NOP; imem[8]  = INST_NOP;
            imem[9]  = enc_cvt(1'b1, 4'd1, 4'd1);   // CVT.i2f R1, R1
            imem[10] = INST_NOP; imem[11] = INST_NOP;
            imem[12] = INST_NOP; imem[13] = INST_NOP;
            imem[14] = INST_NOP; imem[15] = INST_NOP;
            imem[16] = INST_NOP; imem[17] = INST_NOP;
            imem[18] = enc_movi(4'd5, 16'd0);
            imem[19] = enc_m(`OP_ST, 4'd1, 4'd5, 16'd0); // DMEM[0] = R1 (bf16)
            imem[20] = INST_NOP; imem[21] = INST_NOP;
            imem[22] = INST_NOP; imem[23] = INST_NOP;
            imem[24] = enc_cvt(1'b0, 4'd1, 4'd1);   // CVT.f2i R1, R1
            imem[25] = INST_NOP; imem[26] = INST_NOP;
            imem[27] = INST_NOP; imem[28] = INST_NOP;
            imem[29] = INST_NOP; imem[30] = INST_NOP;
            imem[31] = INST_NOP; imem[32] = INST_NOP;
            imem[33] = enc_m(`OP_ST, 4'd1, 4'd5, 16'd1); // DMEM[1] = R1 (int)
            imem[34] = INST_RET;

            print_program(0, 35);
            launch_kernel(32'd0);
            wait_kernel_done(300);
            cvt_trace_en = 0;

            dump_rf_range(4'd0, 4'd5);

            check_gpr_all(4'd1, 16'd15,    "K44 R1=15 (round-trip)");
            check_dmem_all(10'd0, 16'h4170, "K44 DMEM[0]=bf16(15)=0x4170");
            check_dmem_all(10'd1, 16'd15,   "K44 DMEM[1]=15");
        end

        // ════════════════════════════════════════════════════════
        //  K45: CVT self-ref (rD==rA) — with scoreboard stalls
        //
        //  MOVI R1, 15
        //  CVT.i2f R1, R1   (SB stall on R1 from MOVI)
        //  CVT.f2i R1, R1   (SB stall on R1 from CVT.i2f)
        //  MOVI R2, 0
        //  ST R1, [R2+0]    (SB stall on R1 from CVT.f2i)
        //  RET
        //
        //  This is the exact program from soc_tb Test 24.
        //  Expected: DMEM[0]=15 (if working) or 0 (if data bug)
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K45: CVT self-ref rD==rA (SB stalls) ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'hBEEF; dmem1[0] = 16'hBEEF;
            dmem2[0] = 16'hBEEF; dmem3[0] = 16'hBEEF;
            cvt_trace_en = 1;
            cycle_count = 0;

            imem[0] = enc_movi(4'd0, 16'd0);         // R0 = 0 (base addr)
            imem[1] = enc_movi(4'd1, 16'd15);        // R1 = 15
            imem[2] = enc_cvt(1'b1, 4'd1, 4'd1);    // CVT.i2f R1, R1
            imem[3] = enc_cvt(1'b0, 4'd1, 4'd1);    // CVT.f2i R1, R1
            imem[4] = enc_m(`OP_ST, 4'd1, 4'd0, 16'd0); // DMEM[0] = R1
            imem[5] = enc_movi(4'd2, 16'd0);
            imem[6] = enc_m(`OP_ST, 4'd2, 4'd0, 16'd1); // DMEM[1] = 0
            imem[7] = INST_RET;

            print_program(0, 8);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            cvt_trace_en = 0;

            $display("  Post-execution RF dump:");
            dump_rf_range(4'd0, 4'd5);
            $display("  DMEM[0..1]: SP0=%04h,%04h  SP1=%04h,%04h  SP2=%04h,%04h  SP3=%04h,%04h",
                dmem0[0], dmem0[1], dmem1[0], dmem1[1],
                dmem2[0], dmem2[1], dmem3[0], dmem3[1]);

            check_gpr_all(4'd1, 16'd15,   "K45 R1=15 (self-ref round-trip)");
            check_dmem_all(10'd0, 16'd15,  "K45 DMEM[0]=15");
            check_dmem_all(10'd1, 16'd0,   "K45 DMEM[1]=0");
        end

        // ════════════════════════════════════════════════════════
        //  K46: CVT.i2f known values sweep (no stall)
        //
        //  Tests several known int→bf16 conversions:
        //    0  → 0x0000 (0.0)
        //    1  → 0x3F80 (1.0)
        //    2  → 0x4000 (2.0)
        //    15 → 0x4170 (15.0)
        //    100→ 0x42C8 (100.0)
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K46: CVT.i2f known values sweep ---");
            reset_dut;
            clear_dmem;
            cvt_trace_en = 1;
            cycle_count = 0;

            // R1=0, R2=1, R3=2, R4=15, R5=100
            imem[0]  = enc_movi(4'd1, 16'd0);
            imem[1]  = enc_movi(4'd2, 16'd1);
            imem[2]  = enc_movi(4'd3, 16'd2);
            imem[3]  = enc_movi(4'd4, 16'd15);
            imem[4]  = enc_movi(4'd5, 16'd100);
            // NOP padding to drain MOVI writebacks
            imem[5]  = INST_NOP; imem[6]  = INST_NOP;
            imem[7]  = INST_NOP; imem[8]  = INST_NOP;
            imem[9]  = INST_NOP; imem[10] = INST_NOP;
            imem[11] = INST_NOP; imem[12] = INST_NOP;
            // CVT.i2f to R6..R10 (different dest, no SB stall)
            imem[13] = enc_cvt(1'b1, 4'd6, 4'd1);   // R6 = cvt(0)
            imem[14] = enc_cvt(1'b1, 4'd7, 4'd2);   // R7 = cvt(1)
            imem[15] = enc_cvt(1'b1, 4'd8, 4'd3);   // R8 = cvt(2)
            imem[16] = enc_cvt(1'b1, 4'd9, 4'd4);   // R9 = cvt(15)
            imem[17] = enc_cvt(1'b1, 4'd10, 4'd5);  // R10 = cvt(100)
            // NOP padding to drain CVT writebacks
            imem[18] = INST_NOP; imem[19] = INST_NOP;
            imem[20] = INST_NOP; imem[21] = INST_NOP;
            imem[22] = INST_NOP; imem[23] = INST_NOP;
            imem[24] = INST_NOP; imem[25] = INST_NOP;
            imem[26] = INST_NOP; imem[27] = INST_NOP;
            imem[28] = INST_NOP; imem[29] = INST_NOP;
            // Store results to DMEM
            imem[30] = enc_m(`OP_ST, 4'd6, 4'd1, 16'd0); // DMEM[0] = R6
            imem[31] = enc_m(`OP_ST, 4'd7, 4'd1, 16'd1); // DMEM[1] = R7
            imem[32] = enc_m(`OP_ST, 4'd8, 4'd1, 16'd2); // DMEM[2] = R8
            imem[33] = enc_m(`OP_ST, 4'd9, 4'd1, 16'd3); // DMEM[3] = R9
            imem[34] = enc_m(`OP_ST, 4'd10, 4'd1, 16'd4); // DMEM[4] = R10
            imem[35] = INST_RET;

            print_program(0, 36);
            launch_kernel(32'd0);
            wait_kernel_done(400);
            cvt_trace_en = 0;

            dump_rf_range(4'd1, 4'd10);

            check_gpr_all(4'd6,  16'h0000, "K46 cvt(0)=0x0000");
            check_gpr_all(4'd7,  16'h3F80, "K46 cvt(1)=0x3F80");
            check_gpr_all(4'd8,  16'h4000, "K46 cvt(2)=0x4000");
            check_gpr_all(4'd9,  16'h4170, "K46 cvt(15)=0x4170");
            check_gpr_all(4'd10, 16'h42C8, "K46 cvt(100)=0x42C8");

            check_dmem_all(10'd0, 16'h0000, "K46 DMEM[0] cvt(0)");
            check_dmem_all(10'd1, 16'h3F80, "K46 DMEM[1] cvt(1)");
            check_dmem_all(10'd2, 16'h4000, "K46 DMEM[2] cvt(2)");
            check_dmem_all(10'd3, 16'h4170, "K46 DMEM[3] cvt(15)");
            check_dmem_all(10'd4, 16'h42C8, "K46 DMEM[4] cvt(100)");
        end

        // ════════════════════════════════════════════════════════
        //  K47: CVT.i2f with ADD.f verification
        //
        //  If CVT.i2f R2,R1 produces 0x4170 (bf16 15.0),
        //  then ADD.f R3,R2,R2 should produce bf16(30.0) = 0x41F0.
        //  This cross-checks the CVT result against the FPU.
        // ════════════════════════════════════════════════════════
        begin
            $display("\n--- K47: CVT.i2f + ADD.f cross-check ---");
            reset_dut;
            clear_dmem;
            cvt_trace_en = 1;
            cycle_count = 0;

            imem[0]  = enc_movi(4'd1, 16'd15);       // R1 = 15
            imem[1]  = INST_NOP; imem[2]  = INST_NOP;
            imem[3]  = INST_NOP; imem[4]  = INST_NOP;
            imem[5]  = INST_NOP; imem[6]  = INST_NOP;
            imem[7]  = INST_NOP; imem[8]  = INST_NOP;
            imem[9]  = enc_cvt(1'b1, 4'd2, 4'd1);   // CVT.i2f R2, R1
            imem[10] = INST_NOP; imem[11] = INST_NOP;
            imem[12] = INST_NOP; imem[13] = INST_NOP;
            imem[14] = INST_NOP; imem[15] = INST_NOP;
            imem[16] = INST_NOP; imem[17] = INST_NOP;
            imem[18] = enc_r(`OP_ADD, 1'b1, 4'd3, 4'd2, 4'd2); // ADD.f R3 = R2+R2
            imem[19] = INST_NOP; imem[20] = INST_NOP;
            imem[21] = INST_NOP; imem[22] = INST_NOP;
            imem[23] = INST_NOP; imem[24] = INST_NOP;
            imem[25] = enc_movi(4'd4, 16'd0);
            imem[26] = enc_m(`OP_ST, 4'd2, 4'd4, 16'd0); // DMEM[0] = R2 (bf16)
            imem[27] = enc_m(`OP_ST, 4'd3, 4'd4, 16'd1); // DMEM[1] = R3 (bf16)
            imem[28] = INST_RET;

            print_program(0, 29);
            launch_kernel(32'd0);
            wait_kernel_done(300);
            cvt_trace_en = 0;

            dump_rf_range(4'd0, 4'd5);

            check_gpr_all(4'd2, 16'h4170,  "K47 R2=bf16(15)=0x4170");
            check_gpr_all(4'd3, 16'h41F0,  "K47 R3=bf16(30)=0x41F0");
            check_dmem_all(10'd0, 16'h4170, "K47 DMEM[0]=0x4170");
            check_dmem_all(10'd1, 16'h41F0, "K47 DMEM[1]=0x41F0");
        end

        // ════════════════════════════════════════════════════════
        //  Summary
        // ════════════════════════════════════════════════════════
        $display("\n============================================");
        $display("  SM Core CVT Testbench — Summary");
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