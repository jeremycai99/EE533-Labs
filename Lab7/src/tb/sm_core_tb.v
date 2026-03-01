/* file: tb_sm_core.v
 * Testbench for sm_core — streaming multiprocessor core.
 * Instantiates sm_core + IMEM/DMEM BRAM models, loads test
 * programs, launches kernels, and verifies GPR / DMEM results.
 *
 * CRITICAL: Every @(posedge clk) is followed by #1 to prevent
 * Verilog simulation race conditions between TB and DUT NBAs.
 *
 * Author: Jeremy Cai
 * Date: Mar. 1, 2026
 * Version: 1.0
 */

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "sm_core.v"

module tb_sm_core;

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
    wire kernel_done;

    reg [3:0] debug_rf_addr;
    wire [4*`GPU_DMEM_DATA_WIDTH-1:0] debug_rf_data;

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
        .kernel_done(kernel_done),
        .debug_rf_addr(debug_rf_addr),
        .debug_rf_data(debug_rf_data)
    );

    // ================================================================
    // IMEM BRAM Model (sync read, single port)
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

    // SP0 DMEM
    always @(posedge clk) begin
        if (dmem_wea[0])
            dmem0[dmem_addra[0*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]]
                <= dmem_dina[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        dmem_douta[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]
            <= dmem0[dmem_addra[0*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]];
    end
    // SP1 DMEM
    always @(posedge clk) begin
        if (dmem_wea[1])
            dmem1[dmem_addra[1*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]]
                <= dmem_dina[1*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        dmem_douta[1*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]
            <= dmem1[dmem_addra[1*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]];
    end
    // SP2 DMEM
    always @(posedge clk) begin
        if (dmem_wea[2])
            dmem2[dmem_addra[2*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]]
                <= dmem_dina[2*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        dmem_douta[2*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]
            <= dmem2[dmem_addra[2*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]];
    end
    // SP3 DMEM
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
    // R-type: {OPCODE[4:0], DT, 2'b00, rD[3:0], rA[3:0], rB[3:0], 12'd0}
    function [31:0] enc_r;
        input [4:0] op;
        input dt;
        input [3:0] rd, ra, rb;
        enc_r = {op, dt, 2'b00, rd, ra, rb, 12'd0};
    endfunction

    // I-type: {OPCODE, DT, 2'b00, rD, rA, imm16}
    function [31:0] enc_i;
        input [4:0] op;
        input dt;
        input [3:0] rd, ra;
        input [15:0] imm;
        enc_i = {op, dt, 2'b00, rd, ra, imm};
    endfunction

    // MOVI rD, imm16
    function [31:0] enc_movi;
        input [3:0] rd;
        input [15:0] imm;
        enc_movi = {`OP_MOVI, 1'b0, 2'b00, rd, 4'd0, imm};
    endfunction

    // M-type: LD/ST rD, rA, offset16
    function [31:0] enc_m;
        input [4:0] op;
        input [3:0] rd, ra;
        input [15:0] offset;
        enc_m = {op, 1'b0, 2'b00, rd, ra, offset};
    endfunction

    // BRA target (absolute 27-bit)
    function [31:0] enc_bra;
        input [26:0] target;
        enc_bra = {`OP_BRA, target};
    endfunction

    // SETP: {OPCODE, DT, CMP[1:0], pD[3:0], rA[3:0], rB[3:0], 12'd0}
    function [31:0] enc_setp;
        input dt;
        input [1:0] cmp;
        input [3:0] pd, ra, rb;
        enc_setp = {`OP_SETP, dt, cmp, pd, ra, rb, 12'd0};
    endfunction

    // MOV.TID rD (MOV with DT=1, rA=0)
    function [31:0] enc_mov_tid;
        input [3:0] rd;
        enc_mov_tid = {`OP_MOV, 1'b1, 2'b00, rd, 4'd0, 16'd0};
    endfunction

    // WMMA.MMA W-type: {OPCODE, DT=1, RES=00, rD[3:0], rA[3:0], rB[3:0], rC[3:0], 8'd0}
    function [31:0] enc_wmma_mma;
        input [3:0] rD, rA, rB, rC;
        enc_wmma_mma = {`WMMA_MMA, 1'b1, 2'b00, rD, rA, rB, rC, 8'd0};
    endfunction

    // Constants
    wire [31:0] INST_RET = {`OP_RET, 27'd0};
    wire [31:0] INST_NOP = {`OP_NOP, 27'd0};

    // ================================================================
    // Test Infrastructure
    // ================================================================
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    task tick;
    begin
        @(posedge clk); #1;
    end
    endtask

    task reset_dut;
        integer i;
    begin
        rst_n = 0;
        kernel_start = 0;
        kernel_entry_pc = 32'd0;
        debug_rf_addr = 4'd0;
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
        tick; // one extra cycle for final WB to settle
    end
    endtask

    // Check GPR via debug read port (combinational — no clock edge needed)
    task check_gpr;
        input [1:0] sp;
        input [3:0] addr;
        input [15:0] expected;
        input [80*8-1:0] test_name;
        reg [15:0] actual;
    begin
        test_num = test_num + 1;
        debug_rf_addr = addr;
        #1; // combinational settle
        case (sp)
            2'd0: actual = debug_rf_data[0*16 +: 16];
            2'd1: actual = debug_rf_data[1*16 +: 16];
            2'd2: actual = debug_rf_data[2*16 +: 16];
            2'd3: actual = debug_rf_data[3*16 +: 16];
        endcase
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

    // Check GPR across all 4 SPs (same expected value — SIMT)
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

    // Check DMEM value for a specific SP
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

    // Check DMEM across all 4 SPs
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

    // ================================================================
    // Waveform Dump
    // ================================================================
    initial begin
        $dumpfile("tb_sm_core.vcd");
        $dumpvars(0, tb_sm_core);
    end

    // Global timeout
    initial begin
        #500000;
        $display("[TIMEOUT] Global simulation timeout reached");
        $finish;
    end

    // ================================================================
    // Pipeline Monitor — cycle-by-cycle trace
    // ================================================================
    integer cycle_count;
    reg trace_en;

    initial begin
        cycle_count = 0;
        trace_en = 0;
    end

    // --- Opcode name decode (for display) ---
    // Returns a fixed-width string for each opcode (8 chars)
    reg [8*8-1:0] opname;
    always @(*) begin
        case (u_dut.dec_opcode)
            `OP_NOP:      opname = "NOP     ";
            `OP_MOV:      opname = "MOV     ";
            `OP_MOVI:     opname = "MOVI    ";
            `OP_ADD:      opname = "ADD     ";
            `OP_SUB:      opname = "SUB     ";
            `OP_MUL:      opname = "MUL     ";
            `OP_FMA:      opname = "FMA     ";
            `OP_MAX:      opname = "MAX     ";
            `OP_MIN:      opname = "MIN     ";
            `OP_AND:      opname = "AND     ";
            `OP_OR:       opname = "OR      ";
            `OP_XOR:      opname = "XOR     ";
            `OP_NEG:      opname = "NEG     ";
            `OP_ABS:      opname = "ABS     ";
            `OP_SHL:      opname = "SHL     ";
            `OP_SHR:      opname = "SHR     ";
            `OP_ADDI:     opname = "ADDI    ";
            `OP_MULI:     opname = "MULI    ";
            `OP_LD:       opname = "LD      ";
            `OP_ST:       opname = "ST      ";
            `OP_BRA:      opname = "BRA     ";
            `OP_PBRA:     opname = "PBRA    ";
            `OP_SETP:     opname = "SETP    ";
            `OP_SELP:     opname = "SELP    ";
            `OP_CVT:      opname = "CVT     ";
            `OP_RET:      opname = "RET     ";
            `OP_SET:      opname = "SET     ";
            `OP_LDS:      opname = "LDS     ";
            `OP_STS:      opname = "STS     ";
            `WMMA_MMA:    opname = "WMMA.MMA";
            `WMMA_LOAD:   opname = "WMMA.LD ";
            `WMMA_STORE:  opname = "WMMA.ST ";
            default:      opname = "???     ";
        endcase
    end

    // --- TC state name decode ---
    reg [8*8-1:0] tc_state_name;
    always @(*) begin
        case (u_dut.u_tc_top.state)
            3'd0: tc_state_name = "IDLE    ";
            3'd1: tc_state_name = "GATH_A  ";
            3'd2: tc_state_name = "GATH_B  ";
            3'd3: tc_state_name = "GATH_C  ";
            3'd4: tc_state_name = "COMPUTE ";
            3'd5: tc_state_name = "SCAT_0  ";
            3'd6: tc_state_name = "SCAT_1  ";
            default: tc_state_name = "???     ";
        endcase
    end

    // --- Burst state name decode ---
    reg [8*8-1:0] bu_state_name;
    always @(*) begin
        case (u_dut.bu_state)
            3'd0: bu_state_name = "IDLE    ";
            3'd1: bu_state_name = "LD_ADDR ";
            3'd2: bu_state_name = "LD_BEAT ";
            3'd3: bu_state_name = "ST_READ ";
            3'd4: bu_state_name = "ST_BEAT ";
            default: bu_state_name = "???     ";
        endcase
    end

    // --- Enhanced trace ---
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (trace_en) begin
            $display("  [C%03d] PC=%0d ir=%08h dec=%08h %0s fv=%b lat=%b | fstl=%b sb=%b exb=%b sps=%b fl=%b | tc=%0s bu=%0s | pend0=%04h | SP0.w0: we=%b a=%0d d=%04h | done=%b",
                cycle_count,
                u_dut.u_fetch.pc_reg,
                imem_rdata,
                u_dut.dec_ir,
                opname,
                u_dut.fetch_valid,
                u_dut.ir_latched,
                u_dut.front_stall,
                u_dut.sb_stall,
                u_dut.any_ex_busy,
                u_dut.sp_stall,
                u_dut.sp_flush_id,
                tc_state_name,
                bu_state_name,
                u_dut.u_sb.pending[0],
                u_dut.SP_LANE[0].u_sp.w0_we,
                u_dut.SP_LANE[0].u_sp.w0_addr,
                u_dut.SP_LANE[0].u_sp.w0_data,
                kernel_done);
        end
    end

    task enable_trace;
    begin
        trace_en = 1;
        cycle_count = 0;
    end
    endtask

    task disable_trace;
    begin
        trace_en = 0;
    end
    endtask

    // ================================================================
    // Instruction Disassembler — prints one instruction
    // ================================================================
    task disasm;
        input integer addr;
        input [31:0] inst;
        reg [4:0] op;
        reg dt;
        reg [3:0] rD, rA, rB, rC;
        reg [15:0] imm;
    begin
        op  = inst[31:27];
        dt  = inst[26];
        rD  = inst[23:20];
        rA  = inst[19:16];
        rB  = inst[15:12];
        rC  = inst[11:8];
        imm = inst[15:0];
        case (op)
            `OP_NOP:  $display("    [%2d] %08h  NOP", addr, inst);
            `OP_MOVI: $display("    [%2d] %08h  MOVI    R%0d, %0d (0x%04h)", addr, inst, rD, imm, imm);
            `OP_MOV:  begin
                if (dt)
                    $display("    [%2d] %08h  MOV.TID R%0d", addr, inst, rD);
                else
                    $display("    [%2d] %08h  MOV     R%0d, R%0d", addr, inst, rD, rA);
            end
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
            `OP_PBRA: $display("    [%2d] %08h  PBRA    P%0d, %0d", addr, inst, inst[26:25], inst[24:0]);
            `OP_RET:  $display("    [%2d] %08h  RET", addr, inst);
            `OP_SETP: $display("    [%2d] %08h  SETP    P%0d, R%0d, R%0d (cmp=%0d)", addr, inst, rD, rA, rB, inst[25:24]);
            `OP_SELP: $display("    [%2d] %08h  SELP    R%0d, R%0d, R%0d (P%0d)", addr, inst, rD, rA, rB, inst[25:24]);
            `OP_SET:  $display("    [%2d] %08h  SET     P%0d, %0d", addr, inst, rD, imm[0]);
            `WMMA_MMA:   $display("    [%2d] %08h  WMMA.MMA  D=R%0d..%0d, A=R%0d..%0d, B=R%0d..%0d, C=R%0d..%0d",
                           addr, inst, rD, rD+4'd3, rA, rA+4'd3, rB, rB+4'd3, rC, rC+4'd3);
            `WMMA_LOAD:  $display("    [%2d] %08h  WMMA.LOAD R%0d..%0d, [R%0d + %0d]",
                           addr, inst, rD, rD+4'd3, rA, imm);
            `WMMA_STORE: $display("    [%2d] %08h  WMMA.STORE R%0d..%0d, [R%0d + %0d]",
                           addr, inst, rD, rD+4'd3, rA, imm);
            default:  $display("    [%2d] %08h  ??? (op=%02d)", addr, inst, op);
        endcase
    end
    endtask

    // Print full program listing
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
    // BF16 display helpers (for computation documentation)
    // ================================================================
    // bf16 value reference table:
    //   0x0000 = 0.0     0x3F80 = 1.0     0x4000 = 2.0
    //   0x4040 = 3.0     0x4080 = 4.0     0x40A0 = 5.0
    //   0x40C0 = 6.0     0x41C0 = 24.0    0x41C8 = 25.0

    // Dump SP0 RF contents for registers rStart..rEnd
    task dump_rf_range;
        input [3:0] r_start;
        input [3:0] r_end;
        integer ri;
    begin
        $write("  RF dump SP0: ");
        for (ri = r_start; ri <= r_end; ri = ri + 1) begin
            debug_rf_addr = ri[3:0];
            #1;
            $write("R%0d=0x%04h ", ri, debug_rf_data[0*16 +: 16]);
        end
        $write("\n");
    end
    endtask

    // ================================================================
    // Main Test Sequence
    // ================================================================
    initial begin
        $display("============================================");
        $display("  SM Core Testbench — Starting");
        $display("============================================");

        // ── K1: Basic ALU (MOVI + ADD) ───────────────────
        // Program:
        //   addr 0: MOVI R1, 100
        //   addr 1: MOVI R2, 200
        //   addr 2: ADD  R3, R1, R2   → R3 = 300
        //   addr 3: RET
        // Expected: R1=100, R2=200, R3=300 for all SPs
        begin
            $display("\n--- K1: Basic ALU (MOVI + ADD) ---");
            reset_dut;
            clear_dmem;
            enable_trace;
            imem[0] = enc_movi(4'd1, 16'd100);
            imem[1] = enc_movi(4'd2, 16'd200);
            imem[2] = enc_r(`OP_ADD, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[3] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);
            disable_trace;

            check_gpr_all(4'd1, 16'd100, "K1 R1=100");
            check_gpr_all(4'd2, 16'd200, "K1 R2=200");
            check_gpr_all(4'd3, 16'd300, "K1 R3=300");
        end

        // ── K2: Memory LD / ST ───────────────────────────
        // Pre-load: DMEM[10] = 0xABCD (all SPs)
        // Program:
        //   addr 0: MOVI R1, 10         ; base addr
        //   addr 1: LD   R2, R1, 0      ; R2 = DMEM[10] = 0xABCD
        //   addr 2: ADDI R3, R2, 1      ; R3 = 0xABCE
        //   addr 3: ST   R3, R1, 5      ; DMEM[15] = R3
        //   addr 4: RET
        begin
            $display("\n--- K2: Memory LD / ST ---");
            reset_dut;
            clear_dmem;
            dmem0[10] = 16'hABCD;
            dmem1[10] = 16'hABCD;
            dmem2[10] = 16'hABCD;
            dmem3[10] = 16'hABCD;

            imem[0] = enc_movi(4'd1, 16'd10);
            imem[1] = enc_m(`OP_LD, 4'd2, 4'd1, 16'd0);
            imem[2] = enc_i(`OP_ADDI, 1'b0, 4'd3, 4'd2, 16'd1);
            imem[3] = enc_m(`OP_ST, 4'd3, 4'd1, 16'd5);  // ST rD=R3, rA=R1, off=5 → addr=15
            imem[4] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd2, 16'hABCD, "K2 R2=DMEM[10]");
            check_gpr_all(4'd3, 16'hABCE, "K2 R3=R2+1");
            check_dmem_all(10'd15, 16'hABCE, "K2 DMEM[15]=R3");
        end

        // ── K3: Branch (BRA skips instruction) ───────────
        // Program:
        //   addr 0: MOVI R1, 42
        //   addr 1: BRA  4              ; jump to addr 4
        //   addr 2: MOVI R1, 99         ; skipped
        //   addr 3: MOVI R1, 88         ; skipped
        //   addr 4: MOVI R2, 55
        //   addr 5: RET
        begin
            $display("\n--- K3: Branch (BRA) ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd42);
            imem[1] = enc_bra(27'd4);
            imem[2] = enc_movi(4'd1, 16'd99);  // should be skipped
            imem[3] = enc_movi(4'd1, 16'd88);  // should be skipped
            imem[4] = enc_movi(4'd2, 16'd55);
            imem[5] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd1, 16'd42, "K3 R1=42 (not overwritten)");
            check_gpr_all(4'd2, 16'd55, "K3 R2=55 (after branch)");
        end

        // ── K4: Multi-cycle MUL + Scoreboard Stall ───────
        // Program:
        //   addr 0: MOVI R1, 3
        //   addr 1: MOVI R2, 7
        //   addr 2: MUL  R3, R1, R2     ; R3=21 (2-cycle EX)
        //   addr 3: ADD  R4, R3, R1     ; R4=24 (stalls for R3)
        //   addr 4: RET
        begin
            $display("\n--- K4: MUL + Scoreboard Stall ---");
            reset_dut;
            clear_dmem;
            enable_trace;
            imem[0] = enc_movi(4'd1, 16'd3);
            imem[1] = enc_movi(4'd2, 16'd7);
            imem[2] = enc_r(`OP_MUL, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[3] = enc_r(`OP_ADD, 1'b0, 4'd4, 4'd3, 4'd1);
            imem[4] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);
            disable_trace;

            check_gpr_all(4'd3, 16'd21, "K4 R3=3*7=21");
            check_gpr_all(4'd4, 16'd24, "K4 R4=21+3=24");
        end

        // ── K5: MOV.TID — Thread Identity ────────────────
        // Program:
        //   addr 0: MOV.TID R1          ; R1 = thread_id
        //   addr 1: RET
        begin
            $display("\n--- K5: MOV.TID ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_mov_tid(4'd1);
            imem[1] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr(2'd0, 4'd1, 16'd0, "K5 SP0.R1=TID0");
            check_gpr(2'd1, 4'd1, 16'd1, "K5 SP1.R1=TID1");
            check_gpr(2'd2, 4'd1, 16'd2, "K5 SP2.R1=TID2");
            check_gpr(2'd3, 4'd1, 16'd3, "K5 SP3.R1=TID3");
        end

        // ── K6: Back-to-back NOPs + MOVI (no hazard) ────
        // Tests pipeline throughput with no scoreboard stalls.
        // Program:
        //   addr 0: MOVI R1, 10
        //   addr 1: MOVI R2, 20
        //   addr 2: MOVI R3, 30
        //   addr 3: MOVI R4, 40
        //   addr 4: MOVI R5, 50
        //   addr 5: RET
        begin
            $display("\n--- K6: Back-to-back MOVI (no hazard) ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd10);
            imem[1] = enc_movi(4'd2, 16'd20);
            imem[2] = enc_movi(4'd3, 16'd30);
            imem[3] = enc_movi(4'd4, 16'd40);
            imem[4] = enc_movi(4'd5, 16'd50);
            imem[5] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd1, 16'd10, "K6 R1=10");
            check_gpr_all(4'd2, 16'd20, "K6 R2=20");
            check_gpr_all(4'd3, 16'd30, "K6 R3=30");
            check_gpr_all(4'd4, 16'd40, "K6 R4=40");
            check_gpr_all(4'd5, 16'd50, "K6 R5=50");
        end

        // ── K7: Per-thread LD/ST with MOV.TID offset ────
        // Each SP stores to a unique DMEM address.
        // Program:
        //   addr 0: MOV.TID R1           ; R1 = tid
        //   addr 1: MOVI    R2, 0xCAFE
        //   addr 2: ST      R2, R1, 100  ; DMEM[100+tid] = 0xCAFE
        //   addr 3: LD      R3, R1, 100  ; R3 = DMEM[100+tid]
        //   addr 4: RET
        begin
            $display("\n--- K7: Per-thread LD/ST ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_mov_tid(4'd1);
            imem[1] = enc_movi(4'd2, 16'hCAFE);
            imem[2] = enc_m(`OP_ST, 4'd2, 4'd1, 16'd100);
            imem[3] = enc_m(`OP_LD, 4'd3, 4'd1, 16'd100);
            imem[4] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            // SP0: tid=0, stores to DMEM[100], loads from DMEM[100]
            check_dmem(2'd0, 10'd100, 16'hCAFE, "K7 SP0 DMEM[100]");
            check_dmem(2'd1, 10'd101, 16'hCAFE, "K7 SP1 DMEM[101]");
            check_dmem(2'd2, 10'd102, 16'hCAFE, "K7 SP2 DMEM[102]");
            check_dmem(2'd3, 10'd103, 16'hCAFE, "K7 SP3 DMEM[103]");
            check_gpr_all(4'd3, 16'hCAFE, "K7 R3=loaded CAFE");
        end

        // ── K8: Logic + Shift ────────────────────────────
        // Program:
        //   addr 0: MOVI R1, 0x00FF
        //   addr 1: MOVI R2, 0x0F0F
        //   addr 2: AND  R3, R1, R2     ; R3 = 0x000F
        //   addr 3: OR   R4, R1, R2     ; R4 = 0x0FFF
        //   addr 4: XOR  R5, R1, R2     ; R5 = 0x0FF0
        //   addr 5: SHL  R6, R1, 4      ; R6 = 0x0FF0
        //   addr 6: SHR  R7, R1, 4      ; R7 = 0x000F (arithmetic, R1 positive)
        //   addr 7: RET
        begin
            $display("\n--- K8: Logic + Shift ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd1, 16'h00FF);
            imem[1] = enc_movi(4'd2, 16'h0F0F);
            imem[2] = enc_r(`OP_AND, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[3] = enc_r(`OP_OR,  1'b0, 4'd4, 4'd1, 4'd2);
            imem[4] = enc_r(`OP_XOR, 1'b0, 4'd5, 4'd1, 4'd2);
            imem[5] = enc_i(`OP_SHL, 1'b0, 4'd6, 4'd1, 16'd4);
            imem[6] = enc_i(`OP_SHR, 1'b0, 4'd7, 4'd1, 16'd4);
            imem[7] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd3, 16'h000F, "K8 AND");
            check_gpr_all(4'd4, 16'h0FFF, "K8 OR");
            check_gpr_all(4'd5, 16'h0FF0, "K8 XOR");
            check_gpr_all(4'd6, 16'h0FF0, "K8 SHL 4");
            check_gpr_all(4'd7, 16'h000F, "K8 SHR 4");
        end

        // ── K9: FMA (R3 = R1*R2 + R3) ───────────────────
        // FMA is 2-cycle MUL + 1-cycle add = multi-cycle EX
        // Program:
        //   addr 0: MOVI R1, 5
        //   addr 1: MOVI R2, 6
        //   addr 2: MOVI R3, 10          ; accumulator
        //   addr 3: FMA  R3, R1, R2      ; R3 = 5*6 + 10 = 40
        //   addr 4: RET
        // FMA encoding: R-type with rD=R3 (acc), rA=R1, rB=R2
        // dec_is_fma=1 → rD is read source (rf_r2_addr_mux = rD)
        begin
            $display("\n--- K9: FMA ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd5);
            imem[1] = enc_movi(4'd2, 16'd6);
            imem[2] = enc_movi(4'd3, 16'd10);
            imem[3] = enc_r(`OP_FMA, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[4] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd3, 16'd40, "K9 FMA 5*6+10=40");
        end

        // ── K10: Non-zero entry PC ───────────────────────
        // Program loaded at addr 10..12
        // Tests that kernel_entry_pc is respected.
        begin
            $display("\n--- K10: Non-zero entry PC ---");
            reset_dut;
            clear_dmem;
            imem[10] = enc_movi(4'd1, 16'hBEEF);
            imem[11] = enc_movi(4'd2, 16'hDEAD);
            imem[12] = INST_RET;
            launch_kernel(32'd10);
            wait_kernel_done(100);

            check_gpr_all(4'd1, 16'hBEEF, "K10 R1=0xBEEF");
            check_gpr_all(4'd2, 16'hDEAD, "K10 R2=0xDEAD");
        end

        // ── K11: SUB + NEG + ABS ─────────────────────────
        // Program:
        //   addr 0: MOVI R1, 30
        //   addr 1: MOVI R2, 50
        //   addr 2: SUB  R3, R1, R2     ; R3 = 30-50 = -20 = 0xFFEC
        //   addr 3: NEG  R4, R3, x      ; R4 = -R3 = 20
        //   addr 4: ABS  R5, R3, x      ; R5 = |R3| = 20
        //   addr 5: RET
        // NEG and ABS: rA is source, rB unused. use_imm=0 for both.
        begin
            $display("\n--- K11: SUB + NEG + ABS ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd30);
            imem[1] = enc_movi(4'd2, 16'd50);
            imem[2] = enc_r(`OP_SUB, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[3] = enc_r(`OP_NEG, 1'b0, 4'd4, 4'd3, 4'd0);
            imem[4] = enc_r(`OP_ABS, 1'b0, 4'd5, 4'd3, 4'd0);
            imem[5] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd3, 16'hFFEC, "K11 SUB 30-50=-20");
            check_gpr_all(4'd4, 16'd20,   "K11 NEG(-20)=20");
            check_gpr_all(4'd5, 16'd20,   "K11 ABS(-20)=20");
        end

        // ── K12: ADDI chain (dependent, scoreboard stalls) ─
        // Program:
        //   addr 0: MOVI R1, 0
        //   addr 1: ADDI R1, R1, 1      ; R1 = 1 (RAW on R1)
        //   addr 2: ADDI R1, R1, 1      ; R1 = 2
        //   addr 3: ADDI R1, R1, 1      ; R1 = 3
        //   addr 4: ADDI R1, R1, 1      ; R1 = 4
        //   addr 5: RET
        begin
            $display("\n--- K12: ADDI chain (scoreboard stress) ---");
            reset_dut;
            clear_dmem;
            enable_trace;
            imem[0] = enc_movi(4'd1, 16'd0);
            imem[1] = enc_i(`OP_ADDI, 1'b0, 4'd1, 4'd1, 16'd1);
            imem[2] = enc_i(`OP_ADDI, 1'b0, 4'd1, 4'd1, 16'd1);
            imem[3] = enc_i(`OP_ADDI, 1'b0, 4'd1, 4'd1, 16'd1);
            imem[4] = enc_i(`OP_ADDI, 1'b0, 4'd1, 4'd1, 16'd1);
            imem[5] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;

            check_gpr_all(4'd1, 16'd4, "K12 ADDI chain R1=4");
        end

        // ── K13: MAX / MIN ───────────────────────────────
        // Program:
        //   addr 0: MOVI R1, 100
        //   addr 1: MOVI R2, 200
        //   addr 2: MAX  R3, R1, R2     ; R3 = 200
        //   addr 3: MIN  R4, R1, R2     ; R4 = 100
        //   addr 4: RET
        begin
            $display("\n--- K13: MAX / MIN ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd100);
            imem[1] = enc_movi(4'd2, 16'd200);
            imem[2] = enc_r(`OP_MAX, 1'b0, 4'd3, 4'd1, 4'd2);
            imem[3] = enc_r(`OP_MIN, 1'b0, 4'd4, 4'd1, 4'd2);
            imem[4] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd3, 16'd200, "K13 MAX(100,200)=200");
            check_gpr_all(4'd4, 16'd100, "K13 MIN(100,200)=100");
        end

        // ── K14: WMMA.LOAD basic ─────────────────────────────
        // Pre-load DMEM[200..203] = {0x1111,0x2222,0x3333,0x4444}
        // Program:
        //   addr 0: MOVI R15, 200       ; base addr
        //   addr 1: WMMA.LOAD R4, R15, 0 ; R4..R7 ← DMEM[200..203]
        //   addr 2: RET
        begin
            $display("\n--- K14: WMMA.LOAD basic ---");
            reset_dut;
            clear_dmem;
            // Pre-load DMEM for all SPs
            dmem0[200] = 16'h1111; dmem0[201] = 16'h2222;
            dmem0[202] = 16'h3333; dmem0[203] = 16'h4444;
            dmem1[200] = 16'h1111; dmem1[201] = 16'h2222;
            dmem1[202] = 16'h3333; dmem1[203] = 16'h4444;
            dmem2[200] = 16'h1111; dmem2[201] = 16'h2222;
            dmem2[202] = 16'h3333; dmem2[203] = 16'h4444;
            dmem3[200] = 16'h1111; dmem3[201] = 16'h2222;
            dmem3[202] = 16'h3333; dmem3[203] = 16'h4444;
            imem[0] = enc_movi(4'd15, 16'd200);
            imem[1] = enc_m(`WMMA_LOAD, 4'd4, 4'd15, 16'd0);
            imem[2] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd4, 16'h1111, "K14 R4=0x1111");
            check_gpr_all(4'd5, 16'h2222, "K14 R5=0x2222");
            check_gpr_all(4'd6, 16'h3333, "K14 R6=0x3333");
            check_gpr_all(4'd7, 16'h4444, "K14 R7=0x4444");
        end

        // ── K15: WMMA.STORE basic ────────────────────────────
        // Program:
        //   addr 0: MOVI R0, 0x1234
        //   addr 1: MOVI R1, 0x5678
        //   addr 2: MOVI R2, 0x9ABC
        //   addr 3: MOVI R3, 0xDEF0
        //   addr 4: MOVI R8, 300       ; base addr
        //   addr 5: WMMA.STORE R0, R8, 0 ; DMEM[300..303] ← R0..R3
        //   addr 6: RET
        begin
            $display("\n--- K15: WMMA.STORE basic ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd0, 16'h1234);
            imem[1] = enc_movi(4'd1, 16'h5678);
            imem[2] = enc_movi(4'd2, 16'h9ABC);
            imem[3] = enc_movi(4'd3, 16'hDEF0);
            imem[4] = enc_movi(4'd8, 16'd300);
            imem[5] = enc_m(`WMMA_STORE, 4'd0, 4'd8, 16'd0);
            imem[6] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_dmem_all(10'd300, 16'h1234, "K15 DMEM[300]=0x1234");
            check_dmem_all(10'd301, 16'h5678, "K15 DMEM[301]=0x5678");
            check_dmem_all(10'd302, 16'h9ABC, "K15 DMEM[302]=0x9ABC");
            check_dmem_all(10'd303, 16'hDEF0, "K15 DMEM[303]=0xDEF0");
        end

        // ── K16: WMMA.MMA uniform (1.0×1.0 + 0 = 4.0) ──────
        // Math: D[i][j] = sum_{k=0}^{3} A[i][k]*B[k][j] + C[i][j]
        //   A[i][k] = 1.0 for all i,k (bf16 0x3F80)
        //   B[k][j] = 1.0 for all k,j (bf16 0x3F80)
        //   C[i][j] = 0.0 for all i,j (reset default)
        //   D[i][j] = 4 * (1.0*1.0) + 0.0 = 4.0 = bf16(0x4080)
        //
        // Register map: A=R0..R3, B=R4..R7, C=R8..R11, D=R12..R15
        // Thread layout: Thread t holds row t of each matrix
        //   Thread t, R{base+j} = Matrix[t][j]
        begin
            $display("\n--- K16: WMMA.MMA uniform 1*1+0=4.0 ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd0,  16'h3F80); // A[t][0] = 1.0
            imem[1] = enc_movi(4'd1,  16'h3F80); // A[t][1] = 1.0
            imem[2] = enc_movi(4'd2,  16'h3F80); // A[t][2] = 1.0
            imem[3] = enc_movi(4'd3,  16'h3F80); // A[t][3] = 1.0
            imem[4] = enc_movi(4'd4,  16'h3F80); // B[t][0] = 1.0
            imem[5] = enc_movi(4'd5,  16'h3F80); // B[t][1] = 1.0
            imem[6] = enc_movi(4'd6,  16'h3F80); // B[t][2] = 1.0
            imem[7] = enc_movi(4'd7,  16'h3F80); // B[t][3] = 1.0
            // C=R8..R11 = 0 from reset
            imem[8] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[9] = INST_RET;

            $display("  Computation: D = A*B + C = ones(4)*ones(4) + zeros(4) = 4*ones(4)");
            print_program(0, 10);
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;
            $display("  Post-exec RF dump:");
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'h4080, "K16 D[i][0]=4.0");
            check_gpr_all(4'd13, 16'h4080, "K16 D[i][1]=4.0");
            check_gpr_all(4'd14, 16'h4080, "K16 D[i][2]=4.0");
            check_gpr_all(4'd15, 16'h4080, "K16 D[i][3]=4.0");
        end

        // ── K17: WMMA.MMA with accumulate (2×3 + 1 = 25.0) ──
        // Math: D[i][j] = sum_{k=0}^{3} A[i][k]*B[k][j] + C[i][j]
        //   A[i][k] = 2.0 (bf16 0x4000)
        //   B[k][j] = 3.0 (bf16 0x4040)
        //   C[i][j] = 1.0 (bf16 0x3F80)
        //   D[i][j] = 4*(2.0*3.0) + 1.0 = 24+1 = 25.0 = bf16(0x41C8)
        //
        // Register map: A=R0..R3, B=R4..R7, C=R8..R11, D=R12..R15
        begin
            $display("\n--- K17: WMMA.MMA 2*3+1=25.0 ---");
            reset_dut;
            clear_dmem;
            imem[0]  = enc_movi(4'd0,  16'h4000); // A[t][0] = 2.0
            imem[1]  = enc_movi(4'd1,  16'h4000); // A[t][1] = 2.0
            imem[2]  = enc_movi(4'd2,  16'h4000); // A[t][2] = 2.0
            imem[3]  = enc_movi(4'd3,  16'h4000); // A[t][3] = 2.0
            imem[4]  = enc_movi(4'd4,  16'h4040); // B[t][0] = 3.0
            imem[5]  = enc_movi(4'd5,  16'h4040); // B[t][1] = 3.0
            imem[6]  = enc_movi(4'd6,  16'h4040); // B[t][2] = 3.0
            imem[7]  = enc_movi(4'd7,  16'h4040); // B[t][3] = 3.0
            imem[8]  = enc_movi(4'd8,  16'h3F80); // C[t][0] = 1.0
            imem[9]  = enc_movi(4'd9,  16'h3F80); // C[t][1] = 1.0
            imem[10] = enc_movi(4'd10, 16'h3F80); // C[t][2] = 1.0
            imem[11] = enc_movi(4'd11, 16'h3F80); // C[t][3] = 1.0
            imem[12] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13] = INST_RET;

            $display("  Computation: D = A*B + C = 2*ones(4) * 3*ones(4) + ones(4)");
            $display("             = 4*(2*3)*ones(4) + ones(4) = 24+1 = 25.0");
            print_program(0, 14);
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;
            $display("  Post-exec RF dump:");
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'h41C8, "K17 D[i][0]=25.0");
            check_gpr_all(4'd13, 16'h41C8, "K17 D[i][1]=25.0");
            check_gpr_all(4'd14, 16'h41C8, "K17 D[i][2]=25.0");
            check_gpr_all(4'd15, 16'h41C8, "K17 D[i][3]=25.0");
        end

        // ── K18: WMMA full pipeline LOAD → MMA → STORE ──────
        // Pre-load per-SP identity matrix A in DMEM[200..203]:
        //   SP0 row: {1.0, 0, 0, 0}  SP1 row: {0, 1.0, 0, 0}
        //   SP2 row: {0, 0, 1.0, 0}  SP3 row: {0, 0, 0, 1.0}
        // Pre-load B = all bf16(2.0) in DMEM[204..207]
        // Pre-load C = zeros in DMEM[208..211]
        //
        // Math: D = I(4×4) × 2*ones(4×4) + zeros = 2*ones
        //   D[i][j] = 2.0 = bf16(0x4000)
        //
        // Register map: A=R0..R3, B=R4..R7, C=R8..R11, D=R12..R15
        // NOTE: Use R0 (not R15) for store base to avoid clobbering D[3]
        begin
            $display("\n--- K18: WMMA full pipeline LOAD->MMA->STORE ---");
            reset_dut;
            clear_dmem;
            // Identity matrix A (per-SP): SPt row = e_t
            dmem0[200] = 16'h3F80; dmem0[201] = 16'h0000;
            dmem0[202] = 16'h0000; dmem0[203] = 16'h0000;
            dmem1[200] = 16'h0000; dmem1[201] = 16'h3F80;
            dmem1[202] = 16'h0000; dmem1[203] = 16'h0000;
            dmem2[200] = 16'h0000; dmem2[201] = 16'h0000;
            dmem2[202] = 16'h3F80; dmem2[203] = 16'h0000;
            dmem3[200] = 16'h0000; dmem3[201] = 16'h0000;
            dmem3[202] = 16'h0000; dmem3[203] = 16'h3F80;
            // B = all 2.0 (all SPs)
            dmem0[204] = 16'h4000; dmem0[205] = 16'h4000;
            dmem0[206] = 16'h4000; dmem0[207] = 16'h4000;
            dmem1[204] = 16'h4000; dmem1[205] = 16'h4000;
            dmem1[206] = 16'h4000; dmem1[207] = 16'h4000;
            dmem2[204] = 16'h4000; dmem2[205] = 16'h4000;
            dmem2[206] = 16'h4000; dmem2[207] = 16'h4000;
            dmem3[204] = 16'h4000; dmem3[205] = 16'h4000;
            dmem3[206] = 16'h4000; dmem3[207] = 16'h4000;
            // C = zeros (already cleared)

            imem[0] = enc_movi(4'd0, 16'd200);       // base addr for LOADs
            imem[1] = enc_m(`WMMA_LOAD,  4'd4,  4'd0, 16'd0); // A→R4..R7 (using R4 base to avoid R0 clobber)
            imem[2] = enc_m(`WMMA_LOAD,  4'd8,  4'd0, 16'd4); // B→R8..R11
            // Use NOP to let pipeline drain before overwriting R0
            imem[3] = INST_NOP;
            imem[4] = INST_NOP;
            imem[5] = INST_NOP;
            imem[6] = INST_NOP;
            imem[7] = enc_movi(4'd0, 16'd0);         // C[t][0] = 0
            imem[8] = enc_movi(4'd1, 16'd0);         // C[t][1] = 0
            imem[9] = enc_movi(4'd2, 16'd0);         // C[t][2] = 0
            imem[10] = enc_movi(4'd3, 16'd0);        // C[t][3] = 0
            imem[11] = enc_wmma_mma(4'd12, 4'd4, 4'd8, 4'd0); // D=R12..15, A=R4..7, B=R8..11, C=R0..3
            imem[12] = INST_NOP;   // drain pipeline before store setup
            imem[13] = INST_NOP;
            imem[14] = INST_NOP;
            imem[15] = INST_NOP;
            imem[16] = enc_movi(4'd0, 16'd300);      // store base addr (R0 safe, C no longer needed)
            imem[17] = enc_m(`WMMA_STORE, 4'd12, 4'd0, 16'd0); // D → DMEM[300..303]
            imem[18] = INST_RET;

            $display("  DMEM pre-load: A=I(4x4) @200, B=2.0*ones @204, C=0 @208");
            $display("  Computation: D = I*2*ones + 0 = 2*ones => all 2.0 (0x4000)");
            print_program(0, 19);
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(300);
            disable_trace;
            $display("  Post-exec RF dump:");
            dump_rf_range(4'd0, 4'd15);

            // Check D registers (R12-R15)
            check_gpr_all(4'd12, 16'h4000, "K18 D[i][0]=2.0");
            check_gpr_all(4'd13, 16'h4000, "K18 D[i][1]=2.0");
            check_gpr_all(4'd14, 16'h4000, "K18 D[i][2]=2.0");
            check_gpr_all(4'd15, 16'h4000, "K18 D[i][3]=2.0");
            // Check DMEM store results
            check_dmem_all(10'd300, 16'h4000, "K18 DMEM[300]=2.0");
            check_dmem_all(10'd301, 16'h4000, "K18 DMEM[301]=2.0");
            check_dmem_all(10'd302, 16'h4000, "K18 DMEM[302]=2.0");
            check_dmem_all(10'd303, 16'h4000, "K18 DMEM[303]=2.0");
        end

        // ── Summary ──────────────────────────────────────
        $display("\n============================================");
        $display("  SM Core Testbench — Summary");
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