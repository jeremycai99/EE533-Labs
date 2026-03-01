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
 * Version: 1.0
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

    // M-type: LD/ST rD, rA, offset16 (DT=0, int16)
    function [31:0] enc_m;
        input [4:0] op;
        input [3:0] rd, ra;
        input [15:0] offset;
        enc_m = {op, 1'b0, 2'b00, rd, ra, offset};
    endfunction

    // M-type bf16: LD/ST rD, rA, offset16 (DT=1, bf16)
    function [31:0] enc_m_f;
        input [4:0] op;
        input [3:0] rd, ra;
        input [15:0] offset;
        enc_m_f = {op, 1'b1, 2'b00, rd, ra, offset};
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
        $dumpfile("sm_core_tb.vcd");
        $dumpvars(0, sm_core_tb);
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
        integer ri, si;
    begin
        // Header
        $write("  RF dump     ");
        for (ri = r_start; ri <= r_end; ri = ri + 1)
            $write("  R%-2d  ", ri);
        $write("\n");
        // Per-SP rows
        for (si = 0; si < 4; si = si + 1) begin
            $write("       SP%0d:  ", si);
            for (ri = r_start; ri <= r_end; ri = ri + 1) begin
                debug_rf_addr = ri[3:0];
                #1;
                case (si)
                    0: $write(" %04h ", debug_rf_data[0*16 +: 16]);
                    1: $write(" %04h ", debug_rf_data[1*16 +: 16]);
                    2: $write(" %04h ", debug_rf_data[2*16 +: 16]);
                    3: $write(" %04h ", debug_rf_data[3*16 +: 16]);
                endcase
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
        $display("  SM Core Testbench — Starting");
        $display("============================================");

        // ── K1: Basic ALU (MOVI + ADD) ───────────────────
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
            imem[3] = enc_m(`OP_ST, 4'd3, 4'd1, 16'd5);
            imem[4] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd2, 16'hABCD, "K2 R2=DMEM[10]");
            check_gpr_all(4'd3, 16'hABCE, "K2 R3=R2+1");
            check_dmem_all(10'd15, 16'hABCE, "K2 DMEM[15]=R3");
        end

        // ── K3: Branch (BRA skips instruction) ───────────
        begin
            $display("\n--- K3: Branch (BRA) ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd1, 16'd42);
            imem[1] = enc_bra(27'd4);
            imem[2] = enc_movi(4'd1, 16'd99);
            imem[3] = enc_movi(4'd1, 16'd88);
            imem[4] = enc_movi(4'd2, 16'd55);
            imem[5] = INST_RET;
            launch_kernel(32'd0);
            wait_kernel_done(100);

            check_gpr_all(4'd1, 16'd42, "K3 R1=42 (not overwritten)");
            check_gpr_all(4'd2, 16'd55, "K3 R2=55 (after branch)");
        end

        // ── K4: Multi-cycle MUL + Scoreboard Stall ───────
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

        // ── K6: Back-to-back MOVI (no hazard) ────────────
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

            check_dmem(2'd0, 10'd100, 16'hCAFE, "K7 SP0 DMEM[100]");
            check_dmem(2'd1, 10'd101, 16'hCAFE, "K7 SP1 DMEM[101]");
            check_dmem(2'd2, 10'd102, 16'hCAFE, "K7 SP2 DMEM[102]");
            check_dmem(2'd3, 10'd103, 16'hCAFE, "K7 SP3 DMEM[103]");
            check_gpr_all(4'd3, 16'hCAFE, "K7 R3=loaded CAFE");
        end

        // ── K8: Logic + Shift ────────────────────────────
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

        // ── K14: WMMA.LOAD basic ─────────────────────────
        begin
            $display("\n--- K14: WMMA.LOAD basic ---");
            reset_dut;
            clear_dmem;
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

        // ── K15: WMMA.STORE basic ────────────────────────
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

        // ── K16: WMMA.MMA uniform (1.0×1.0 + 0 = 4.0) ──
        begin
            $display("\n--- K16: WMMA.MMA uniform 1*1+0=4.0 ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd0,  16'h3F80);
            imem[1] = enc_movi(4'd1,  16'h3F80);
            imem[2] = enc_movi(4'd2,  16'h3F80);
            imem[3] = enc_movi(4'd3,  16'h3F80);
            imem[4] = enc_movi(4'd4,  16'h3F80);
            imem[5] = enc_movi(4'd5,  16'h3F80);
            imem[6] = enc_movi(4'd6,  16'h3F80);
            imem[7] = enc_movi(4'd7,  16'h3F80);
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
        begin
            $display("\n--- K17: WMMA.MMA 2*3+1=25.0 ---");
            reset_dut;
            clear_dmem;
            imem[0]  = enc_movi(4'd0,  16'h4000);
            imem[1]  = enc_movi(4'd1,  16'h4000);
            imem[2]  = enc_movi(4'd2,  16'h4000);
            imem[3]  = enc_movi(4'd3,  16'h4000);
            imem[4]  = enc_movi(4'd4,  16'h4040);
            imem[5]  = enc_movi(4'd5,  16'h4040);
            imem[6]  = enc_movi(4'd6,  16'h4040);
            imem[7]  = enc_movi(4'd7,  16'h4040);
            imem[8]  = enc_movi(4'd8,  16'h3F80);
            imem[9]  = enc_movi(4'd9,  16'h3F80);
            imem[10] = enc_movi(4'd10, 16'h3F80);
            imem[11] = enc_movi(4'd11, 16'h3F80);
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

        // ── K18: WMMA full pipeline LOAD → MMA → STORE ──
        begin
            $display("\n--- K18: WMMA full pipeline LOAD->MMA->STORE ---");
            reset_dut;
            clear_dmem;
            dmem0[200] = 16'h3F80; dmem0[201] = 16'h0000;
            dmem0[202] = 16'h0000; dmem0[203] = 16'h0000;
            dmem1[200] = 16'h0000; dmem1[201] = 16'h3F80;
            dmem1[202] = 16'h0000; dmem1[203] = 16'h0000;
            dmem2[200] = 16'h0000; dmem2[201] = 16'h0000;
            dmem2[202] = 16'h3F80; dmem2[203] = 16'h0000;
            dmem3[200] = 16'h0000; dmem3[201] = 16'h0000;
            dmem3[202] = 16'h0000; dmem3[203] = 16'h3F80;
            dmem0[204] = 16'h4000; dmem0[205] = 16'h4000;
            dmem0[206] = 16'h4000; dmem0[207] = 16'h4000;
            dmem1[204] = 16'h4000; dmem1[205] = 16'h4000;
            dmem1[206] = 16'h4000; dmem1[207] = 16'h4000;
            dmem2[204] = 16'h4000; dmem2[205] = 16'h4000;
            dmem2[206] = 16'h4000; dmem2[207] = 16'h4000;
            dmem3[204] = 16'h4000; dmem3[205] = 16'h4000;
            dmem3[206] = 16'h4000; dmem3[207] = 16'h4000;

            imem[0] = enc_movi(4'd0, 16'd200);
            imem[1] = enc_m(`WMMA_LOAD,  4'd4,  4'd0, 16'd0);
            imem[2] = enc_m(`WMMA_LOAD,  4'd8,  4'd0, 16'd4);
            imem[3] = INST_NOP;
            imem[4] = INST_NOP;
            imem[5] = INST_NOP;
            imem[6] = INST_NOP;
            imem[7] = enc_movi(4'd0, 16'd0);
            imem[8] = enc_movi(4'd1, 16'd0);
            imem[9] = enc_movi(4'd2, 16'd0);
            imem[10] = enc_movi(4'd3, 16'd0);
            imem[11] = enc_wmma_mma(4'd12, 4'd4, 4'd8, 4'd0);
            imem[12] = INST_NOP;
            imem[13] = INST_NOP;
            imem[14] = INST_NOP;
            imem[15] = INST_NOP;
            imem[16] = enc_movi(4'd0, 16'd300);
            imem[17] = enc_m(`WMMA_STORE, 4'd12, 4'd0, 16'd0);
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

            check_gpr_all(4'd12, 16'h4000, "K18 D[i][0]=2.0");
            check_gpr_all(4'd13, 16'h4000, "K18 D[i][1]=2.0");
            check_gpr_all(4'd14, 16'h4000, "K18 D[i][2]=2.0");
            check_gpr_all(4'd15, 16'h4000, "K18 D[i][3]=2.0");
            check_dmem_all(10'd300, 16'h4000, "K18 DMEM[300]=2.0");
            check_dmem_all(10'd301, 16'h4000, "K18 DMEM[301]=2.0");
            check_dmem_all(10'd302, 16'h4000, "K18 DMEM[302]=2.0");
            check_dmem_all(10'd303, 16'h4000, "K18 DMEM[303]=2.0");
        end

        // ── K19: MMA zero matrix passthrough (0×B + C = C) ──
        begin
            $display("\n--- K19: MMA zero passthrough 0*3+5=5.0 ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd4,  16'h4040);
            imem[1] = enc_movi(4'd5,  16'h4040);
            imem[2] = enc_movi(4'd6,  16'h4040);
            imem[3] = enc_movi(4'd7,  16'h4040);
            imem[4] = enc_movi(4'd8,  16'h40A0);
            imem[5] = enc_movi(4'd9,  16'h40A0);
            imem[6] = enc_movi(4'd10, 16'h40A0);
            imem[7] = enc_movi(4'd11, 16'h40A0);
            imem[8] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[9] = INST_RET;

            $display("  Computation: D = 0*3 + 5 = 5.0 (accumulate passthrough)");
            print_program(0, 10);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'h40A0, "K19 D[i][0]=5.0");
            check_gpr_all(4'd13, 16'h40A0, "K19 D[i][1]=5.0");
            check_gpr_all(4'd14, 16'h40A0, "K19 D[i][2]=5.0");
            check_gpr_all(4'd15, 16'h40A0, "K19 D[i][3]=5.0");
        end

        // ── K20: MMA negative values (-1×2 + 0 = -8.0) ──
        begin
            $display("\n--- K20: MMA negative -1*2+0=-8.0 ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd0,  16'hBF80);
            imem[1] = enc_movi(4'd1,  16'hBF80);
            imem[2] = enc_movi(4'd2,  16'hBF80);
            imem[3] = enc_movi(4'd3,  16'hBF80);
            imem[4] = enc_movi(4'd4,  16'h4000);
            imem[5] = enc_movi(4'd5,  16'h4000);
            imem[6] = enc_movi(4'd6,  16'h4000);
            imem[7] = enc_movi(4'd7,  16'h4000);
            imem[8] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[9] = INST_RET;

            $display("  Computation: D = 4*(-1*2) + 0 = -8.0");
            print_program(0, 10);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'hC100, "K20 D[i][0]=-8.0");
            check_gpr_all(4'd13, 16'hC100, "K20 D[i][1]=-8.0");
            check_gpr_all(4'd14, 16'hC100, "K20 D[i][2]=-8.0");
            check_gpr_all(4'd15, 16'hC100, "K20 D[i][3]=-8.0");
        end

        // ── K21: MMA mixed sign accumulate (1×1 + (-2) = 2.0) ──
        begin
            $display("\n--- K21: MMA mixed 1*1+(-2)=2.0 ---");
            reset_dut;
            clear_dmem;
            imem[0]  = enc_movi(4'd0,  16'h3F80);
            imem[1]  = enc_movi(4'd1,  16'h3F80);
            imem[2]  = enc_movi(4'd2,  16'h3F80);
            imem[3]  = enc_movi(4'd3,  16'h3F80);
            imem[4]  = enc_movi(4'd4,  16'h3F80);
            imem[5]  = enc_movi(4'd5,  16'h3F80);
            imem[6]  = enc_movi(4'd6,  16'h3F80);
            imem[7]  = enc_movi(4'd7,  16'h3F80);
            imem[8]  = enc_movi(4'd8,  16'hC000);
            imem[9]  = enc_movi(4'd9,  16'hC000);
            imem[10] = enc_movi(4'd10, 16'hC000);
            imem[11] = enc_movi(4'd11, 16'hC000);
            imem[12] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[13] = INST_RET;

            $display("  Computation: D = 4*(1*1) + (-2) = 4 - 2 = 2.0");
            print_program(0, 14);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'h4000, "K21 D[i][0]=2.0");
            check_gpr_all(4'd13, 16'h4000, "K21 D[i][1]=2.0");
            check_gpr_all(4'd14, 16'h4000, "K21 D[i][2]=2.0");
            check_gpr_all(4'd15, 16'h4000, "K21 D[i][3]=2.0");
        end

        // ── K22: MMA fractional (0.5×0.5 + 0 = 1.0) ──
        begin
            $display("\n--- K22: MMA fractional 0.5*0.5+0=1.0 ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd0,  16'h3F00);
            imem[1] = enc_movi(4'd1,  16'h3F00);
            imem[2] = enc_movi(4'd2,  16'h3F00);
            imem[3] = enc_movi(4'd3,  16'h3F00);
            imem[4] = enc_movi(4'd4,  16'h3F00);
            imem[5] = enc_movi(4'd5,  16'h3F00);
            imem[6] = enc_movi(4'd6,  16'h3F00);
            imem[7] = enc_movi(4'd7,  16'h3F00);
            imem[8] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[9] = INST_RET;

            $display("  Computation: D = 4*(0.5*0.5) + 0 = 1.0");
            print_program(0, 10);
            launch_kernel(32'd0);
            wait_kernel_done(200);
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'h3F80, "K22 D[i][0]=1.0");
            check_gpr_all(4'd13, 16'h3F80, "K22 D[i][1]=1.0");
            check_gpr_all(4'd14, 16'h3F80, "K22 D[i][2]=1.0");
            check_gpr_all(4'd15, 16'h3F80, "K22 D[i][3]=1.0");
        end

        // ── K23: Chained MMA (D1=A*B, then D2=D1*B) ──
        begin
            $display("\n--- K23: Chained MMA D1=4.0, D2=D1*1+0=16.0 ---");
            reset_dut;
            clear_dmem;
            imem[0] = enc_movi(4'd0,  16'h3F80);
            imem[1] = enc_movi(4'd1,  16'h3F80);
            imem[2] = enc_movi(4'd2,  16'h3F80);
            imem[3] = enc_movi(4'd3,  16'h3F80);
            imem[4] = enc_movi(4'd4,  16'h3F80);
            imem[5] = enc_movi(4'd5,  16'h3F80);
            imem[6] = enc_movi(4'd6,  16'h3F80);
            imem[7] = enc_movi(4'd7,  16'h3F80);
            imem[8] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[9] = enc_wmma_mma(4'd0, 4'd12, 4'd4, 4'd8);
            imem[10] = INST_RET;

            $display("  Step 1: D1 = 4*(1*1)+0 = 4.0");
            $display("  Step 2: D2 = 4*(4*1)+0 = 16.0 (chained)");
            print_program(0, 11);
            launch_kernel(32'd0);
            wait_kernel_done(400);
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'h4080, "K23 D1[i][0]=4.0");
            check_gpr_all(4'd13, 16'h4080, "K23 D1[i][1]=4.0");
            check_gpr_all(4'd14, 16'h4080, "K23 D1[i][2]=4.0");
            check_gpr_all(4'd15, 16'h4080, "K23 D1[i][3]=4.0");
            check_gpr_all(4'd0, 16'h4180, "K23 D2[i][0]=16.0");
            check_gpr_all(4'd1, 16'h4180, "K23 D2[i][1]=16.0");
            check_gpr_all(4'd2, 16'h4180, "K23 D2[i][2]=16.0");
            check_gpr_all(4'd3, 16'h4180, "K23 D2[i][3]=16.0");
        end

        // ── K24: Per-thread SIMT via DMEM (diagonal A, uniform B) ─
        begin
            $display("\n--- K24: Per-thread SIMT diag(2)*3+1=7.0 ---");
            reset_dut;
            clear_dmem;
            dmem0[100] = 16'h4000; dmem0[101] = 16'h0000;
            dmem0[102] = 16'h0000; dmem0[103] = 16'h0000;
            dmem1[100] = 16'h0000; dmem1[101] = 16'h4000;
            dmem1[102] = 16'h0000; dmem1[103] = 16'h0000;
            dmem2[100] = 16'h0000; dmem2[101] = 16'h0000;
            dmem2[102] = 16'h4000; dmem2[103] = 16'h0000;
            dmem3[100] = 16'h0000; dmem3[101] = 16'h0000;
            dmem3[102] = 16'h0000; dmem3[103] = 16'h4000;

            imem[0]  = enc_movi(4'd0, 16'd100);
            imem[1]  = enc_m(`WMMA_LOAD, 4'd0, 4'd0, 16'd0);
            imem[2]  = enc_movi(4'd4,  16'h4040);
            imem[3]  = enc_movi(4'd5,  16'h4040);
            imem[4]  = enc_movi(4'd6,  16'h4040);
            imem[5]  = enc_movi(4'd7,  16'h4040);
            imem[6]  = enc_movi(4'd8,  16'h3F80);
            imem[7]  = enc_movi(4'd9,  16'h3F80);
            imem[8]  = enc_movi(4'd10, 16'h3F80);
            imem[9]  = enc_movi(4'd11, 16'h3F80);
            imem[10] = enc_wmma_mma(4'd12, 4'd0, 4'd4, 4'd8);
            imem[11] = INST_RET;

            $display("  DMEM pre-load: A=diag(2.0) @100 (per-SP rows)");
            $display("  Computation: D = diag(2)*3*ones + ones = 7*ones");
            print_program(0, 12);
            launch_kernel(32'd0);
            wait_kernel_done(300);
            dump_rf_range(4'd0, 4'd15);

            check_gpr_all(4'd12, 16'h40E0, "K24 D[i][0]=7.0");
            check_gpr_all(4'd13, 16'h40E0, "K24 D[i][1]=7.0");
            check_gpr_all(4'd14, 16'h40E0, "K24 D[i][2]=7.0");
            check_gpr_all(4'd15, 16'h40E0, "K24 D[i][3]=7.0");
        end

        // ================================================================
        // PTX-Mapped Kernel Tests (K25–K30)
        // These tests implement the exact ISA assembly from the
        // PTX line-by-line mapping SVG for all 6 CUDA kernels.
        // Each follows: MOV.TID R0 (hw-init), then the SVG assembly.
        // Memory layout: base + tid*2 word-addressing for scalar,
        //                base + tid*8 for WMMA row stride.
        // ================================================================

        // ── K25: PTX K1 — vec_add C[tid]=A[tid]+B[tid] (int16) ──
        // Assembly (12 instr from SVG + 1 MOV.TID):
        //   MOV.TID R0             ; R0 = tid
        //   MOVI R1, 0             ; base_A
        //   MOVI R2, 16            ; base_B
        //   MOVI R3, 32            ; base_C
        //   SHL  R4, R0, 1         ; byte offset = tid*2
        //   ADD  R5, R1, R4        ; &A[tid]
        //   LD   R5, R5, 0         ; R5 = A[tid]
        //   ADD  R6, R2, R4        ; &B[tid]
        //   LD   R6, R6, 0         ; R6 = B[tid]
        //   ADD  R7, R6, R5        ; ★ compute
        //   ADD  R8, R3, R4        ; &C[tid]
        //   ST   R7, R8, 0         ; C[tid] = result
        //   RET
        begin
            $display("\n--- K25: PTX K1 vec_add int16 ---");
            reset_dut;
            clear_dmem;
            // Pre-load A: SP0@[0]=10, SP1@[2]=20, SP2@[4]=30, SP3@[6]=40
            dmem0[0] = 16'd10; dmem1[2] = 16'd20;
            dmem2[4] = 16'd30; dmem3[6] = 16'd40;
            // Pre-load B: SP0@[16]=5, SP1@[18]=15, SP2@[20]=25, SP3@[22]=35
            dmem0[16] = 16'd5;  dmem1[18] = 16'd15;
            dmem2[20] = 16'd25; dmem3[22] = 16'd35;

            imem[0]  = enc_mov_tid(4'd0);
            imem[1]  = enc_movi(4'd1, 16'd0);
            imem[2]  = enc_movi(4'd2, 16'd16);
            imem[3]  = enc_movi(4'd3, 16'd32);
            imem[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            imem[5]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            imem[6]  = enc_m(`OP_LD, 4'd5, 4'd5, 16'd0);
            imem[7]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            imem[8]  = enc_m(`OP_LD, 4'd6, 4'd6, 16'd0);
            imem[9]  = enc_r(`OP_ADD, 1'b0, 4'd7, 4'd6, 4'd5);   // ★ ADD
            imem[10] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd3, 4'd4);
            imem[11] = enc_m(`OP_ST, 4'd7, 4'd8, 16'd0);
            imem[12] = INST_RET;

            print_program(0, 13);
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;

            // SP0: 10+5=15, SP1: 20+15=35, SP2: 30+25=55, SP3: 40+35=75
            check_dmem(2'd0, 10'd32, 16'd15, "K25 SP0 C[0]=10+5");
            check_dmem(2'd1, 10'd34, 16'd35, "K25 SP1 C[1]=20+15");
            check_dmem(2'd2, 10'd36, 16'd55, "K25 SP2 C[2]=30+25");
            check_dmem(2'd3, 10'd38, 16'd75, "K25 SP3 C[3]=40+35");
        end

        // ── K26: PTX K2 — vec_sub C[tid]=A[tid]-B[tid] (int16) ──
        // Identical to K25 except line 9: SUB R7, R5, R6
        begin
            $display("\n--- K26: PTX K2 vec_sub int16 ---");
            reset_dut;
            clear_dmem;
            dmem0[0] = 16'd100; dmem1[2] = 16'd200;
            dmem2[4] = 16'd300; dmem3[6] = 16'd400;
            dmem0[16] = 16'd30; dmem1[18] = 16'd50;
            dmem2[20] = 16'd100; dmem3[22] = 16'd150;

            imem[0]  = enc_mov_tid(4'd0);
            imem[1]  = enc_movi(4'd1, 16'd0);
            imem[2]  = enc_movi(4'd2, 16'd16);
            imem[3]  = enc_movi(4'd3, 16'd32);
            imem[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            imem[5]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            imem[6]  = enc_m(`OP_LD, 4'd5, 4'd5, 16'd0);
            imem[7]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            imem[8]  = enc_m(`OP_LD, 4'd6, 4'd6, 16'd0);
            imem[9]  = enc_r(`OP_SUB, 1'b0, 4'd7, 4'd5, 4'd6);   // ★ SUB
            imem[10] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd3, 4'd4);
            imem[11] = enc_m(`OP_ST, 4'd7, 4'd8, 16'd0);
            imem[12] = INST_RET;

            print_program(0, 13);
            launch_kernel(32'd0);
            wait_kernel_done(200);

            // SP0: 100-30=70, SP1: 200-50=150, SP2: 300-100=200, SP3: 400-150=250
            check_dmem(2'd0, 10'd32, 16'd70,  "K26 SP0 C[0]=100-30");
            check_dmem(2'd1, 10'd34, 16'd150, "K26 SP1 C[1]=200-50");
            check_dmem(2'd2, 10'd36, 16'd200, "K26 SP2 C[2]=300-100");
            check_dmem(2'd3, 10'd38, 16'd250, "K26 SP3 C[3]=400-150");
        end

        // ── K27: PTX K3 — bf16_vector_mul C[tid]=A[tid]*B[tid] ──
        // Key difference: DT=1 for MUL (bf16 datapath)
        // PTX: fma.rn.bf16 + mov.b16(-0) → MUL.f (native)
        // A=2.0(0x4000), B=3.0(0x4040) → C=6.0(0x40C0)
        begin
            $display("\n--- K27: PTX K3 bf16_vector_mul ---");
            reset_dut;
            clear_dmem;
            // A = 2.0 for all threads
            dmem0[0] = 16'h4000; dmem1[2] = 16'h4000;
            dmem2[4] = 16'h4000; dmem3[6] = 16'h4000;
            // B = 3.0 for all threads
            dmem0[16] = 16'h4040; dmem1[18] = 16'h4040;
            dmem2[20] = 16'h4040; dmem3[22] = 16'h4040;

            imem[0]  = enc_mov_tid(4'd0);
            imem[1]  = enc_movi(4'd1, 16'd0);
            imem[2]  = enc_movi(4'd2, 16'd16);
            imem[3]  = enc_movi(4'd3, 16'd32);
            imem[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            imem[5]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            imem[6]  = enc_m_f(`OP_LD, 4'd5, 4'd5, 16'd0);         // LD.bf16
            imem[7]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            imem[8]  = enc_m_f(`OP_LD, 4'd6, 4'd6, 16'd0);         // LD.bf16
            imem[9]  = enc_r(`OP_MUL, 1'b1, 4'd7, 4'd5, 4'd6);   // ★ MUL.f DT=1
            imem[10] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd3, 4'd4);
            imem[11] = enc_m_f(`OP_ST, 4'd7, 4'd8, 16'd0);         // ST.bf16
            imem[12] = INST_RET;

            print_program(0, 13);
            launch_kernel(32'd0);
            wait_kernel_done(200);

            // 2.0 * 3.0 = 6.0 = bf16(0x40C0) for all threads
            check_gpr_all(4'd7, 16'h40C0, "K27 MUL.f 2*3=6.0");
            check_dmem(2'd0, 10'd32, 16'h40C0, "K27 SP0 C=6.0");
            check_dmem(2'd1, 10'd34, 16'h40C0, "K27 SP1 C=6.0");
            check_dmem(2'd2, 10'd36, 16'h40C0, "K27 SP2 C=6.0");
            check_dmem(2'd3, 10'd38, 16'h40C0, "K27 SP3 C=6.0");
        end

        // ── K28: PTX K4 — bf16_fma D[tid]=A[tid]*B[tid]+C[tid] ──
        // 4 pointers, 3 loads, FMA (rD read+write), 1 store
        // A=2.0, B=3.0, C=1.0 → D = 2*3+1 = 7.0 = bf16(0x40E0)
        begin
            $display("\n--- K28: PTX K4 bf16_fma ---");
            reset_dut;
            clear_dmem;
            // A=2.0, B=3.0, C=1.0 per thread
            dmem0[0] = 16'h4000; dmem1[2] = 16'h4000;
            dmem2[4] = 16'h4000; dmem3[6] = 16'h4000;
            dmem0[16] = 16'h4040; dmem1[18] = 16'h4040;
            dmem2[20] = 16'h4040; dmem3[22] = 16'h4040;
            dmem0[32] = 16'h3F80; dmem1[34] = 16'h3F80;
            dmem2[36] = 16'h3F80; dmem3[38] = 16'h3F80;

            imem[0]  = enc_mov_tid(4'd0);
            imem[1]  = enc_movi(4'd1, 16'd0);       // base_A
            imem[2]  = enc_movi(4'd2, 16'd16);      // base_B
            imem[3]  = enc_movi(4'd3, 16'd32);      // base_C
            imem[4]  = enc_movi(4'd9, 16'd48);      // base_D
            imem[5]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            // Load A
            imem[6]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            imem[7]  = enc_m_f(`OP_LD, 4'd5, 4'd5, 16'd0);         // LD.bf16
            // Load B
            imem[8]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            imem[9]  = enc_m_f(`OP_LD, 4'd6, 4'd6, 16'd0);         // LD.bf16
            // Load C into R7 (accumulator for FMA)
            imem[10] = enc_r(`OP_ADD, 1'b0, 4'd7, 4'd3, 4'd4);
            imem[11] = enc_m_f(`OP_LD, 4'd7, 4'd7, 16'd0);         // LD.bf16
            // FMA: R7 = R5*R6 + R7
            imem[12] = enc_r(`OP_FMA, 1'b1, 4'd7, 4'd5, 4'd6);   // ★ FMA.f DT=1
            // Store D
            imem[13] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd9, 4'd4);
            imem[14] = enc_m_f(`OP_ST, 4'd7, 4'd8, 16'd0);         // ST.bf16
            imem[15] = INST_RET;

            $display("  D = A*B + C = 2.0*3.0 + 1.0 = 7.0 (0x40E0)");
            print_program(0, 16);
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;

            check_gpr_all(4'd7, 16'h40E0, "K28 FMA.f 2*3+1=7.0");
            check_dmem(2'd0, 10'd48, 16'h40E0, "K28 SP0 D=7.0");
            check_dmem(2'd1, 10'd50, 16'h40E0, "K28 SP1 D=7.0");
            check_dmem(2'd2, 10'd52, 16'h40E0, "K28 SP2 D=7.0");
            check_dmem(2'd3, 10'd54, 16'h40E0, "K28 SP3 D=7.0");
        end

        // ── K29: PTX K5 — relu out[tid]=max(in[tid],0.0) (bf16) ──
        // Per-thread divergent values: mix of positive and negative
        //   SP0: in=-1.0(0xBF80) → out=0.0(0x0000)
        //   SP1: in= 2.0(0x4000) → out=2.0(0x4000)
        //   SP2: in=-3.0(0xC040) → out=0.0(0x0000)
        //   SP3: in= 5.0(0x40A0) → out=5.0(0x40A0)
        begin
            $display("\n--- K29: PTX K5 relu bf16 ---");
            reset_dut;
            clear_dmem;
            // Input per-thread at base_in=0, offset=tid*2
            dmem0[0] = 16'hBF80; // SP0: -1.0
            dmem1[2] = 16'h4000; // SP1:  2.0
            dmem2[4] = 16'hC040; // SP2: -3.0
            dmem3[6] = 16'h40A0; // SP3:  5.0

            imem[0]  = enc_mov_tid(4'd0);
            imem[1]  = enc_movi(4'd1, 16'd0);       // base_in
            imem[2]  = enc_movi(4'd2, 16'd16);      // base_out
            imem[3]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            // Load input
            imem[4]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            imem[5]  = enc_m_f(`OP_LD, 4'd5, 4'd5, 16'd0);         // LD.bf16
            // bf16 zero constant
            imem[6]  = enc_movi(4'd8, 16'h0000);    // R8 = bf16(0.0)
            // ReLU: max(input, 0.0)
            imem[7]  = enc_r(`OP_MAX, 1'b1, 4'd6, 4'd5, 4'd8);   // ★ MAX.f DT=1
            // Store output
            imem[8]  = enc_r(`OP_ADD, 1'b0, 4'd7, 4'd2, 4'd4);
            imem[9]  = enc_m_f(`OP_ST, 4'd6, 4'd7, 16'd0);         // ST.bf16
            imem[10] = INST_RET;

            $display("  ReLU: max(x, 0) — per-thread divergent inputs");
            print_program(0, 11);
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;

            // GPR checks (per-SP divergent results)
            check_gpr(2'd0, 4'd6, 16'h0000, "K29 SP0 relu(-1)=0");
            check_gpr(2'd1, 4'd6, 16'h4000, "K29 SP1 relu(2)=2.0");
            check_gpr(2'd2, 4'd6, 16'h0000, "K29 SP2 relu(-3)=0");
            check_gpr(2'd3, 4'd6, 16'h40A0, "K29 SP3 relu(5)=5.0");
            // DMEM checks
            check_dmem(2'd0, 10'd16, 16'h0000, "K29 SP0 out=0.0");
            check_dmem(2'd1, 10'd18, 16'h4000, "K29 SP1 out=2.0");
            check_dmem(2'd2, 10'd20, 16'h0000, "K29 SP2 out=0.0");
            check_dmem(2'd3, 10'd22, 16'h40A0, "K29 SP3 out=5.0");
        end

        // ── K30: PTX K6 — wmma_bf16 4×4 matmul D=A*B+C ─────────
        // Exact assembly from SVG K6 (16 instructions + MOV.TID):
        //   MOV.TID R0          ; tid
        //   MOVI R1, base_A     ; 0
        //   MOVI R2, base_B     ; 32
        //   MOVI R3, base_D     ; 64
        //   SHL  R4, R0, 3      ; row_offset = tid*8
        //   ADD  R1, R1, R4     ; R1 = &A[my_row][0]
        //   ADD  R2, R2, R4     ; R2 = &B[my_row][0]
        //   ADD  R3, R3, R4     ; R3 = &D[my_row][0]
        //   WMMA.LOAD R4,R1,0   ; A frag → R4..R7
        //   WMMA.LOAD R8,R2,0   ; B frag → R8..R11
        //   MOVI R12, 0         ; C[0]=0
        //   MOVI R13, 0         ; C[1]=0
        //   MOVI R14, 0         ; C[2]=0
        //   MOVI R15, 0         ; C[3]=0
        //   WMMA.MMA R12,R4,R8,R12  ; D = A*B + C
        //   WMMA.STORE R12,R3,0     ; store D
        //   RET
        //
        // Math: A=I(4×4), B=2*ones(4×4), C=0 → D=2*ones
        // Row stride = tid*8 → SP0@base, SP1@base+8, SP2@base+16, SP3@base+24
        begin
            $display("\n--- K30: PTX K6 wmma_bf16 4x4 matmul ---");
            reset_dut;
            clear_dmem;

            // Pre-load A = identity matrix (per-SP rows)
            // SP0 row 0 @[0..3]: {1.0, 0, 0, 0}
            dmem0[0] = 16'h3F80; dmem0[1] = 16'h0000;
            dmem0[2] = 16'h0000; dmem0[3] = 16'h0000;
            // SP1 row 1 @[8..11]: {0, 1.0, 0, 0}
            dmem1[8] = 16'h0000; dmem1[9] = 16'h3F80;
            dmem1[10] = 16'h0000; dmem1[11] = 16'h0000;
            // SP2 row 2 @[16..19]: {0, 0, 1.0, 0}
            dmem2[16] = 16'h0000; dmem2[17] = 16'h0000;
            dmem2[18] = 16'h3F80; dmem2[19] = 16'h0000;
            // SP3 row 3 @[24..27]: {0, 0, 0, 1.0}
            dmem3[24] = 16'h0000; dmem3[25] = 16'h0000;
            dmem3[26] = 16'h0000; dmem3[27] = 16'h3F80;

            // Pre-load B = 2.0 everywhere (base_B=32)
            // SP0 @[32..35]
            dmem0[32] = 16'h4000; dmem0[33] = 16'h4000;
            dmem0[34] = 16'h4000; dmem0[35] = 16'h4000;
            // SP1 @[40..43]
            dmem1[40] = 16'h4000; dmem1[41] = 16'h4000;
            dmem1[42] = 16'h4000; dmem1[43] = 16'h4000;
            // SP2 @[48..51]
            dmem2[48] = 16'h4000; dmem2[49] = 16'h4000;
            dmem2[50] = 16'h4000; dmem2[51] = 16'h4000;
            // SP3 @[56..59]
            dmem3[56] = 16'h4000; dmem3[57] = 16'h4000;
            dmem3[58] = 16'h4000; dmem3[59] = 16'h4000;

            imem[0]  = enc_mov_tid(4'd0);
            imem[1]  = enc_movi(4'd1, 16'd0);                     // base_A
            imem[2]  = enc_movi(4'd2, 16'd32);                    // base_B
            imem[3]  = enc_movi(4'd3, 16'd64);                    // base_D
            imem[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd3);  // R4 = tid*8
            imem[5]  = enc_r(`OP_ADD, 1'b0, 4'd1, 4'd1, 4'd4);   // R1 = &A[row]
            imem[6]  = enc_r(`OP_ADD, 1'b0, 4'd2, 4'd2, 4'd4);   // R2 = &B[row]
            imem[7]  = enc_r(`OP_ADD, 1'b0, 4'd3, 4'd3, 4'd4);   // R3 = &D[row]
            imem[8]  = enc_m_f(`WMMA_LOAD, 4'd4, 4'd1, 16'd0);    // A→R4..R7 DT=1
            imem[9]  = enc_m_f(`WMMA_LOAD, 4'd8, 4'd2, 16'd0);    // B→R8..R11 DT=1
            imem[10] = enc_movi(4'd12, 16'h0000);                 // C[0]=0
            imem[11] = enc_movi(4'd13, 16'h0000);                 // C[1]=0
            imem[12] = enc_movi(4'd14, 16'h0000);                 // C[2]=0
            imem[13] = enc_movi(4'd15, 16'h0000);                 // C[3]=0
            imem[14] = enc_wmma_mma(4'd12, 4'd4, 4'd8, 4'd12);   // D=A*B+C
            imem[15] = enc_m_f(`WMMA_STORE, 4'd12, 4'd3, 16'd0);  // D→DMEM DT=1
            imem[16] = INST_RET;

            $display("  D = I(4x4) * 2*ones(4x4) + 0 = 2*ones");
            $display("  Row stride = tid*8, base_A=0, base_B=32, base_D=64");
            print_program(0, 17);
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(400);
            disable_trace;
            $display("  Post-exec RF dump:");
            dump_rf_range(4'd0, 4'd15);

            // D = 2.0 everywhere (0x4000)
            check_gpr_all(4'd12, 16'h4000, "K30 D[i][0]=2.0");
            check_gpr_all(4'd13, 16'h4000, "K30 D[i][1]=2.0");
            check_gpr_all(4'd14, 16'h4000, "K30 D[i][2]=2.0");
            check_gpr_all(4'd15, 16'h4000, "K30 D[i][3]=2.0");
            // Check DMEM store results
            // SP0: D row @[64..67], SP1 @[72..75], SP2 @[80..83], SP3 @[88..91]
            check_dmem(2'd0, 10'd64, 16'h4000, "K30 SP0 D[0][0]=2.0");
            check_dmem(2'd0, 10'd65, 16'h4000, "K30 SP0 D[0][1]=2.0");
            check_dmem(2'd0, 10'd66, 16'h4000, "K30 SP0 D[0][2]=2.0");
            check_dmem(2'd0, 10'd67, 16'h4000, "K30 SP0 D[0][3]=2.0");
            check_dmem(2'd1, 10'd72, 16'h4000, "K30 SP1 D[1][0]=2.0");
            check_dmem(2'd1, 10'd73, 16'h4000, "K30 SP1 D[1][1]=2.0");
            check_dmem(2'd1, 10'd74, 16'h4000, "K30 SP1 D[1][2]=2.0");
            check_dmem(2'd1, 10'd75, 16'h4000, "K30 SP1 D[1][3]=2.0");
            check_dmem(2'd2, 10'd80, 16'h4000, "K30 SP2 D[2][0]=2.0");
            check_dmem(2'd2, 10'd81, 16'h4000, "K30 SP2 D[2][1]=2.0");
            check_dmem(2'd2, 10'd82, 16'h4000, "K30 SP2 D[2][2]=2.0");
            check_dmem(2'd2, 10'd83, 16'h4000, "K30 SP2 D[2][3]=2.0");
            check_dmem(2'd3, 10'd88, 16'h4000, "K30 SP3 D[3][0]=2.0");
            check_dmem(2'd3, 10'd89, 16'h4000, "K30 SP3 D[3][1]=2.0");
            check_dmem(2'd3, 10'd90, 16'h4000, "K30 SP3 D[3][2]=2.0");
            check_dmem(2'd3, 10'd91, 16'h4000, "K30 SP3 D[3][3]=2.0");
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