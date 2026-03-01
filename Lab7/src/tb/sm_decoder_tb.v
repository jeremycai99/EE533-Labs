/* file: tb_instruction_decoder.v
 * Testbench for instruction_decoder.
 * Verifies every binary encoding from PTX line-by-line mapping v2
 * (K1–K6) plus edge cases.
 *
 * Author: Jeremy Cai
 * Date: Feb. 28, 2026
 */

`timescale 1ns / 1ps

`include "sm_decoder.v"

module sm_decoder_tb;

    // DUT signals
    reg [31:0] ir;
    wire [4:0] dec_opcode;
    wire dec_dt;
    wire [1:0] dec_cmp_mode;
    wire [3:0] dec_rD_addr, dec_rA_addr, dec_rB_addr, dec_rC_addr;
    wire [15:0] dec_imm16;
    wire dec_rf_we, dec_pred_we;
    wire [1:0] dec_pred_wr_sel, dec_pred_rd_sel;
    wire [2:0] dec_wb_src;
    wire dec_use_imm;
    wire dec_uses_rA, dec_uses_rB, dec_is_fma, dec_is_st;
    wire dec_is_branch, dec_is_pbra, dec_is_ret;
    wire dec_is_ld, dec_is_store, dec_is_lds, dec_is_sts;
    wire dec_is_wmma_mma, dec_is_wmma_load, dec_is_wmma_store;
    wire [1:0] dec_wmma_sel;
    wire [`GPU_PC_WIDTH-1:0] dec_branch_target;

    sm_decoder u_dut (
        .ir(ir),
        .dec_opcode(dec_opcode), .dec_dt(dec_dt), .dec_cmp_mode(dec_cmp_mode),
        .dec_rD_addr(dec_rD_addr), .dec_rA_addr(dec_rA_addr),
        .dec_rB_addr(dec_rB_addr), .dec_rC_addr(dec_rC_addr),
        .dec_imm16(dec_imm16),
        .dec_rf_we(dec_rf_we), .dec_pred_we(dec_pred_we),
        .dec_pred_wr_sel(dec_pred_wr_sel), .dec_pred_rd_sel(dec_pred_rd_sel),
        .dec_wb_src(dec_wb_src), .dec_use_imm(dec_use_imm),
        .dec_uses_rA(dec_uses_rA), .dec_uses_rB(dec_uses_rB),
        .dec_is_fma(dec_is_fma), .dec_is_st(dec_is_st),
        .dec_is_branch(dec_is_branch), .dec_is_pbra(dec_is_pbra),
        .dec_is_ret(dec_is_ret),
        .dec_is_ld(dec_is_ld), .dec_is_store(dec_is_store),
        .dec_is_lds(dec_is_lds), .dec_is_sts(dec_is_sts),
        .dec_is_wmma_mma(dec_is_wmma_mma), .dec_is_wmma_load(dec_is_wmma_load),
        .dec_is_wmma_store(dec_is_wmma_store), .dec_wmma_sel(dec_wmma_sel),
        .dec_branch_target(dec_branch_target)
    );

    // Test counters
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    // Assertion task
    task check;
        input [255:0] name;
        input cond;
        begin
            test_num = test_num + 1;
            if (!cond) begin
                $display("FAIL [%0d] %0s", test_num, name);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
        end
    endtask

    initial begin
        $display("=== Instruction Decoder Testbench ===");
        $display("Verifying against PTX mapping v2 encodings\n");

        // ============================================================
        // K1: vec_add — MOVI, SHL, ADD, LD(int16), ST, RET
        // ============================================================
        $display("--- K1: vec_add ---");

        // MOVI R1, 0x0040 (base_A=64)
        ir = 32'h2010_0040; #1;
        check("K1.MOVI opcode", dec_opcode == `OP_MOVI);
        check("K1.MOVI dt=0", dec_dt == 0);
        check("K1.MOVI rD=R1", dec_rD_addr == 4'd1);
        check("K1.MOVI imm=0x40", dec_imm16 == 16'h0040);
        check("K1.MOVI rf_we=1", dec_rf_we == 1);
        check("K1.MOVI use_imm=1", dec_use_imm == 1);
        check("K1.MOVI uses_rA=0", dec_uses_rA == 0);
        check("K1.MOVI uses_rB=0", dec_uses_rB == 0);
        check("K1.MOVI wb_src=ALU", dec_wb_src == 3'd0);

        // SHL R4, R0, 1 → 0x8840_0001
        ir = 32'h8840_0001; #1;
        check("K1.SHL opcode", dec_opcode == `OP_SHL);
        check("K1.SHL dt=0", dec_dt == 0);
        check("K1.SHL rD=R4", dec_rD_addr == 4'd4);
        check("K1.SHL rA=R0", dec_rA_addr == 4'd0);
        check("K1.SHL imm=1", dec_imm16 == 16'h0001);
        check("K1.SHL rf_we=1", dec_rf_we == 1);
        check("K1.SHL use_imm=1", dec_use_imm == 1);
        check("K1.SHL uses_rA=1", dec_uses_rA == 1);
        check("K1.SHL uses_rB=0", dec_uses_rB == 0);
        check("K1.SHL wb_src=ALU", dec_wb_src == 3'd0);

        // ADD R5, R1, R4 → 0x3051_4000
        ir = 32'h3051_4000; #1;
        check("K1.ADD opcode", dec_opcode == `OP_ADD);
        check("K1.ADD dt=0", dec_dt == 0);
        check("K1.ADD rD=R5", dec_rD_addr == 4'd5);
        check("K1.ADD rA=R1", dec_rA_addr == 4'd1);
        check("K1.ADD rB=R4", dec_rB_addr == 4'd4);
        check("K1.ADD rf_we=1", dec_rf_we == 1);
        check("K1.ADD use_imm=0", dec_use_imm == 0);
        check("K1.ADD uses_rA=1", dec_uses_rA == 1);
        check("K1.ADD uses_rB=1", dec_uses_rB == 1);
        check("K1.ADD is_fma=0", dec_is_fma == 0);

        // LD R5, R5, 0 (int16) → 0x1055_0000
        ir = 32'h1055_0000; #1;
        check("K1.LD opcode", dec_opcode == `OP_LD);
        check("K1.LD dt=0", dec_dt == 0);
        check("K1.LD rD=R5", dec_rD_addr == 4'd5);
        check("K1.LD rA=R5", dec_rA_addr == 4'd5);
        check("K1.LD imm=0", dec_imm16 == 16'h0000);
        check("K1.LD rf_we=1", dec_rf_we == 1);
        check("K1.LD use_imm=1", dec_use_imm == 1);
        check("K1.LD uses_rA=1", dec_uses_rA == 1);
        check("K1.LD is_ld=1", dec_is_ld == 1);
        check("K1.LD wb_src=MEM", dec_wb_src == 3'd1);
        check("K1.LD is_st=0", dec_is_st == 0);

        // ST R7, R8, 0 → 0x0878_0000
        ir = 32'h0878_0000; #1;
        check("K1.ST opcode", dec_opcode == `OP_ST);
        check("K1.ST dt=0", dec_dt == 0);
        check("K1.ST rD=R7(src)", dec_rD_addr == 4'd7);
        check("K1.ST rA=R8(addr)", dec_rA_addr == 4'd8);
        check("K1.ST rf_we=0", dec_rf_we == 0);
        check("K1.ST use_imm=1", dec_use_imm == 1);
        check("K1.ST is_store=1", dec_is_store == 1);
        check("K1.ST is_st=1", dec_is_st == 1);
        check("K1.ST wb_src=STORE", dec_wb_src == 3'd2);

        // RET → 0xC800_0000
        ir = 32'hC800_0000; #1;
        check("K1.RET opcode", dec_opcode == `OP_RET);
        check("K1.RET rf_we=0", dec_rf_we == 0);
        check("K1.RET is_ret=1", dec_is_ret == 1);
        check("K1.RET uses_rA=0", dec_uses_rA == 0);
        check("K1.RET uses_rB=0", dec_uses_rB == 0);

        // ============================================================
        // K2: vec_sub — SUB
        // ============================================================
        $display("--- K2: vec_sub ---");

        // SUB R7, R5, R6 → 0x3875_6000
        ir = 32'h3875_6000; #1;
        check("K2.SUB opcode", dec_opcode == `OP_SUB);
        check("K2.SUB dt=0", dec_dt == 0);
        check("K2.SUB rD=R7", dec_rD_addr == 4'd7);
        check("K2.SUB rA=R5", dec_rA_addr == 4'd5);
        check("K2.SUB rB=R6", dec_rB_addr == 4'd6);
        check("K2.SUB rf_we=1", dec_rf_we == 1);
        check("K2.SUB uses_rA=1", dec_uses_rA == 1);
        check("K2.SUB uses_rB=1", dec_uses_rB == 1);

        // ============================================================
        // K3: bf16_vector_mul — MUL(bf16), LD(bf16), ST(bf16)
        // ============================================================
        $display("--- K3: bf16_vector_mul ---");

        // LD R5, R5, 0 (bf16) → 0x1455_0000
        ir = 32'h1455_0000; #1;
        check("K3.LD.bf16 opcode", dec_opcode == `OP_LD);
        check("K3.LD.bf16 dt=1", dec_dt == 1);
        check("K3.LD.bf16 rD=R5", dec_rD_addr == 4'd5);
        check("K3.LD.bf16 is_ld=1", dec_is_ld == 1);
        check("K3.LD.bf16 wb_src=MEM", dec_wb_src == 3'd1);

        // MUL R7, R5, R6 (bf16) → 0x4475_6000
        ir = 32'h4475_6000; #1;
        check("K3.MUL.bf16 opcode", dec_opcode == `OP_MUL);
        check("K3.MUL.bf16 dt=1", dec_dt == 1);
        check("K3.MUL.bf16 rD=R7", dec_rD_addr == 4'd7);
        check("K3.MUL.bf16 rA=R5", dec_rA_addr == 4'd5);
        check("K3.MUL.bf16 rB=R6", dec_rB_addr == 4'd6);
        check("K3.MUL.bf16 rf_we=1", dec_rf_we == 1);
        check("K3.MUL.bf16 uses_rB=1", dec_uses_rB == 1);
        check("K3.MUL.bf16 is_fma=0", dec_is_fma == 0);

        // ST R7, R8, 0 (bf16) → 0x0C78_0000
        ir = 32'h0C78_0000; #1;
        check("K3.ST.bf16 opcode", dec_opcode == `OP_ST);
        check("K3.ST.bf16 dt=1", dec_dt == 1);
        check("K3.ST.bf16 rD=R7(src)", dec_rD_addr == 4'd7);
        check("K3.ST.bf16 rA=R8(addr)", dec_rA_addr == 4'd8);

        // ============================================================
        // K4: bf16_fma — FMA
        // ============================================================
        $display("--- K4: bf16_fma ---");

        // FMA R7, R5, R6 (bf16) → 0x4C75_6000 (rD=rC=R7)
        ir = 32'h4C75_6000; #1;
        check("K4.FMA.bf16 opcode", dec_opcode == `OP_FMA);
        check("K4.FMA.bf16 dt=1", dec_dt == 1);
        check("K4.FMA.bf16 rD=R7", dec_rD_addr == 4'd7);
        check("K4.FMA.bf16 rA=R5", dec_rA_addr == 4'd5);
        check("K4.FMA.bf16 rB=R6", dec_rB_addr == 4'd6);
        check("K4.FMA.bf16 rf_we=1", dec_rf_we == 1);
        check("K4.FMA.bf16 uses_rA=1", dec_uses_rA == 1);
        check("K4.FMA.bf16 uses_rB=1", dec_uses_rB == 1);
        check("K4.FMA.bf16 is_fma=1", dec_is_fma == 1);
        check("K4.FMA.bf16 wb_src=ALU", dec_wb_src == 3'd0);

        // ============================================================
        // K5: relu — MAX(bf16), MOVI(bf16 zero)
        // ============================================================
        $display("--- K5: relu ---");

        // MOVI R8, 0x0000 (bf16 zero) → 0x2480_0000
        ir = 32'h2480_0000; #1;
        check("K5.MOVI.bf16 opcode", dec_opcode == `OP_MOVI);
        check("K5.MOVI.bf16 dt=1", dec_dt == 1);
        check("K5.MOVI.bf16 rD=R8", dec_rD_addr == 4'd8);
        check("K5.MOVI.bf16 imm=0", dec_imm16 == 16'h0000);
        check("K5.MOVI.bf16 rf_we=1", dec_rf_we == 1);

        // MAX R6, R5, R8 (bf16, relu!) → 0x5465_8000
        ir = 32'h5465_8000; #1;
        check("K5.MAX.bf16 opcode", dec_opcode == `OP_MAX);
        check("K5.MAX.bf16 dt=1", dec_dt == 1);
        check("K5.MAX.bf16 rD=R6", dec_rD_addr == 4'd6);
        check("K5.MAX.bf16 rA=R5", dec_rA_addr == 4'd5);
        check("K5.MAX.bf16 rB=R8", dec_rB_addr == 4'd8);
        check("K5.MAX.bf16 rf_we=1", dec_rf_we == 1);
        check("K5.MAX.bf16 uses_rA=1", dec_uses_rA == 1);
        check("K5.MAX.bf16 uses_rB=1", dec_uses_rB == 1);

        // ============================================================
        // K6: wmma_bf16 — WMMA.LOAD.A/B, WMMA.MMA, WMMA.STORE
        // ============================================================
        $display("--- K6: wmma_bf16 ---");

        // WMMA.LOAD.A R4, R1, 0 → binary: 11110_1_00_0100_0001... = 0xF441_0000
        ir = 32'hF441_0000; #1;
        check("K6.WMMA.LOAD.A opcode", dec_opcode == `WMMA_LOAD);
        check("K6.WMMA.LOAD.A dt=1", dec_dt == 1);
        check("K6.WMMA.LOAD.A rD=R4", dec_rD_addr == 4'd4);
        check("K6.WMMA.LOAD.A rA=R1", dec_rA_addr == 4'd1);
        check("K6.WMMA.LOAD.A sel=00(A)", dec_wmma_sel == 2'b00);
        check("K6.WMMA.LOAD.A is_wmma_load=1", dec_is_wmma_load == 1);
        check("K6.WMMA.LOAD.A rf_we=0", dec_rf_we == 0);
        check("K6.WMMA.LOAD.A use_imm=1", dec_use_imm == 1);

        // WMMA.LOAD.B R8, R2, 0 → binary: 11110_1_01_1000_0010... = 0xF582_0000
        ir = 32'hF582_0000; #1;
        check("K6.WMMA.LOAD.B opcode", dec_opcode == `WMMA_LOAD);
        check("K6.WMMA.LOAD.B dt=1", dec_dt == 1);
        check("K6.WMMA.LOAD.B rD=R8", dec_rD_addr == 4'd8);
        check("K6.WMMA.LOAD.B rA=R2", dec_rA_addr == 4'd2);
        check("K6.WMMA.LOAD.B sel=01(B)", dec_wmma_sel == 2'b01);

        // MOVI R12, 0x0000 (bf16 zero accum) → 0x24C0_0000
        ir = 32'h24C0_0000; #1;
        check("K6.MOVI.R12 opcode", dec_opcode == `OP_MOVI);
        check("K6.MOVI.R12 rD=R12", dec_rD_addr == 4'd12);
        check("K6.MOVI.R12 dt=1", dec_dt == 1);

        // WMMA.MMA R12, R4, R8, R12 → 0xECC4_8C00
        ir = 32'hECC4_8C00; #1;
        check("K6.WMMA.MMA opcode", dec_opcode == `WMMA_MMA);
        check("K6.WMMA.MMA dt=1", dec_dt == 1);
        check("K6.WMMA.MMA rD=R12", dec_rD_addr == 4'd12);
        check("K6.WMMA.MMA rA=R4", dec_rA_addr == 4'd4);
        check("K6.WMMA.MMA rB=R8", dec_rB_addr == 4'd8);
        check("K6.WMMA.MMA rC=R12", dec_rC_addr == 4'd12);
        check("K6.WMMA.MMA is_wmma_mma=1", dec_is_wmma_mma == 1);
        check("K6.WMMA.MMA rf_we=0", dec_rf_we == 0);
        check("K6.WMMA.MMA uses_rA=0", dec_uses_rA == 0);

        // WMMA.STORE R12, R3, 0 → 0xFCC3_0000
        ir = 32'hFCC3_0000; #1;
        check("K6.WMMA.STORE opcode", dec_opcode == `WMMA_STORE);
        check("K6.WMMA.STORE dt=1", dec_dt == 1);
        check("K6.WMMA.STORE rD=R12(src)", dec_rD_addr == 4'd12);
        check("K6.WMMA.STORE rA=R3(addr)", dec_rA_addr == 4'd3);
        check("K6.WMMA.STORE is_wmma_store=1", dec_is_wmma_store == 1);
        check("K6.WMMA.STORE rf_we=0", dec_rf_we == 0);
        check("K6.WMMA.STORE is_st=1", dec_is_st == 1);

        // ============================================================
        // Additional ops not in K1–K6 but in ISA
        // ============================================================
        $display("--- Extra opcodes ---");

        // NOP → 0x0000_0000
        ir = 32'h0000_0000; #1;
        check("NOP opcode", dec_opcode == `OP_NOP);
        check("NOP rf_we=0", dec_rf_we == 0);
        check("NOP uses_rA=0", dec_uses_rA == 0);
        check("NOP uses_rB=0", dec_uses_rB == 0);
        check("NOP is_ret=0", dec_is_ret == 0);

        // MOV R3, R1 → opcode=00011, dt=0, rD=3, rA=1
        // 00011_0_00_0011_0001_0000_0000_0000_0000 = 0x1831_0000
        ir = 32'h1831_0000; #1;
        check("MOV opcode", dec_opcode == `OP_MOV);
        check("MOV dt=0", dec_dt == 0);
        check("MOV rD=R3", dec_rD_addr == 4'd3);
        check("MOV rA=R1", dec_rA_addr == 4'd1);
        check("MOV rf_we=1", dec_rf_we == 1);
        check("MOV uses_rA=1", dec_uses_rA == 1);
        check("MOV uses_rB=0", dec_uses_rB == 0);
        check("MOV use_imm=0", dec_use_imm == 0);

        // MOV.TID R5 → MOV with DT=1: opcode=00011, dt=1
        // 00011_1_00_0101_0000_0000_0000_0000_0000 = 0x1C50_0000
        ir = 32'h1C50_0000; #1;
        check("MOV.TID opcode", dec_opcode == `OP_MOV);
        check("MOV.TID dt=1", dec_dt == 1);
        check("MOV.TID rD=R5", dec_rD_addr == 4'd5);
        check("MOV.TID rA=R0", dec_rA_addr == 4'd0);
        check("MOV.TID rf_we=1", dec_rf_we == 1);

        // CVT R2, R1 → opcode=00101, dt=0 (int→bf16)
        // 00101_0_00_0010_0001_0000_0000_0000_0000 = 0x2821_0000
        ir = 32'h2821_0000; #1;
        check("CVT opcode", dec_opcode == `OP_CVT);
        check("CVT rD=R2", dec_rD_addr == 4'd2);
        check("CVT rA=R1", dec_rA_addr == 4'd1);
        check("CVT rf_we=1", dec_rf_we == 1);
        check("CVT uses_rA=1", dec_uses_rA == 1);

        // MIN R3, R1, R2 (bf16) → opcode=01011, dt=1
        // 01011_1_00_0011_0001_0010_0000_0000_0000 = 0x5C31_2000
        ir = 32'h5C31_2000; #1;
        check("MIN.bf16 opcode", dec_opcode == `OP_MIN);
        check("MIN.bf16 dt=1", dec_dt == 1);
        check("MIN.bf16 rD=R3", dec_rD_addr == 4'd3);
        check("MIN.bf16 uses_rB=1", dec_uses_rB == 1);

        // ABS R2, R1 → opcode=01100, dt=0
        // 01100_0_00_0010_0001_0000_0000_0000_0000 = 0x6021_0000
        ir = 32'h6021_0000; #1;
        check("ABS opcode", dec_opcode == `OP_ABS);
        check("ABS rD=R2", dec_rD_addr == 4'd2);
        check("ABS uses_rA=1", dec_uses_rA == 1);
        check("ABS uses_rB=0", dec_uses_rB == 0);

        // NEG R2, R1 → opcode=01101
        // 01101_0_00_0010_0001_0000_0000_0000_0000 = 0x6821_0000
        ir = 32'h6821_0000; #1;
        check("NEG opcode", dec_opcode == `OP_NEG);
        check("NEG rf_we=1", dec_rf_we == 1);
        check("NEG uses_rB=0", dec_uses_rB == 0);

        // AND R3, R1, R2 → opcode=01110
        // 01110_0_00_0011_0001_0010_0000_0000_0000 = 0x7031_2000
        ir = 32'h7031_2000; #1;
        check("AND opcode", dec_opcode == `OP_AND);
        check("AND uses_rA=1", dec_uses_rA == 1);
        check("AND uses_rB=1", dec_uses_rB == 1);

        // OR R3, R1, R2 → opcode=01111
        // 01111_0_00_0011_0001_0010_0000_0000_0000 = 0x7831_2000
        ir = 32'h7831_2000; #1;
        check("OR opcode", dec_opcode == `OP_OR);

        // XOR R3, R1, R2 → opcode=10000
        // 10000_0_00_0011_0001_0010_0000_0000_0000 = 0x8031_2000
        ir = 32'h8031_2000; #1;
        check("XOR opcode", dec_opcode == `OP_XOR);
        check("XOR uses_rB=1", dec_uses_rB == 1);

        // SHR R4, R1, 3 → opcode=10010
        // 10010_0_00_0100_0001_0000_0000_0000_0011 = 0x9041_0003
        ir = 32'h9041_0003; #1;
        check("SHR opcode", dec_opcode == `OP_SHR);
        check("SHR imm=3", dec_imm16 == 16'h0003);
        check("SHR use_imm=1", dec_use_imm == 1);

        // ADDI R3, R1, 10 → opcode=10011
        // 10011_0_00_0011_0001_0000_0000_0000_1010 = 0x9831_000A
        ir = 32'h9831_000A; #1;
        check("ADDI opcode", dec_opcode == `OP_ADDI);
        check("ADDI imm=10", dec_imm16 == 16'h000A);
        check("ADDI use_imm=1", dec_use_imm == 1);
        check("ADDI uses_rA=1", dec_uses_rA == 1);
        check("ADDI uses_rB=0", dec_uses_rB == 0);

        // MULI R3, R1, 4 → opcode=10100
        // 10100_0_00_0011_0001_0000_0000_0000_0100 = 0xA031_0004
        ir = 32'hA031_0004; #1;
        check("MULI opcode", dec_opcode == `OP_MULI);
        check("MULI use_imm=1", dec_use_imm == 1);

        // SETP P0, R1, R2 (EQ) → opcode=10101, CMP=00, rD[1:0]=00
        // 10101_0_00_0000_0001_0010_0000_0000_0000 = 0xA801_2000
        ir = 32'hA801_2000; #1;
        check("SETP opcode", dec_opcode == `OP_SETP);
        check("SETP rf_we=0", dec_rf_we == 0);
        check("SETP pred_we=1", dec_pred_we == 1);
        check("SETP pred_wr_sel=P0", dec_pred_wr_sel == 2'd0);
        check("SETP cmp_mode=EQ", dec_cmp_mode == 2'b00);
        check("SETP uses_rA=1", dec_uses_rA == 1);
        check("SETP uses_rB=1", dec_uses_rB == 1);

        // SETP P2, R3, R4 (LT) → CMP=10, rD[1:0]=10
        // 10101_0_10_0010_0011_0100_0000_0000_0000 = 0xAA23_4000
        ir = 32'hAA23_4000; #1;
        check("SETP.LT cmp_mode=LT", dec_cmp_mode == 2'b10);
        check("SETP.LT pred_wr_sel=P2", dec_pred_wr_sel == 2'd2);

        // SELP R3, R1, R2 → opcode=10110, pred read from RES[25:24]
        // With pred_rd_sel=01 (P1): 10110_0_01_0011... = 0xB131_2000
        ir = 32'hB131_2000; #1;
        check("SELP opcode", dec_opcode == `OP_SELP);
        check("SELP rf_we=1", dec_rf_we == 1);
        check("SELP uses_rA=1", dec_uses_rA == 1);
        check("SELP uses_rB=1", dec_uses_rB == 1);
        check("SELP pred_rd_sel=P1", dec_pred_rd_sel == 2'b01);

        // BRA target=0x10 → opcode=10111
        // 10111_000_0000_0000_0000_0000_0001_0000 = 0xB800_0010
        ir = 32'hB800_0010; #1;
        check("BRA opcode", dec_opcode == `OP_BRA);
        check("BRA is_branch=1", dec_is_branch == 1);
        check("BRA rf_we=0", dec_rf_we == 0);
        check("BRA target=0x10", dec_branch_target[7:0] == 8'h10);

        // PBRA P1, target=0x20 → opcode=11000, pred_sel[26:25]=01
        // 11000_01_0000_0010_0000_0000_0000_0000 = 0xC200_0000 + target bits
        // [26:25]=01, [24:12]=0x020 (target), [11:0]=reconv
        // 11000_01_0_0000_0010_0000_0000_0000_0000
        ir = {5'b11000, 2'b01, 13'h0020, 12'h000}; #1;
        check("PBRA opcode", dec_opcode == `OP_PBRA);
        check("PBRA is_pbra=1", dec_is_pbra == 1);
        check("PBRA rf_we=0", dec_rf_we == 0);
        check("PBRA pred_rd_sel=P1", dec_pred_rd_sel == 2'b01);

        // SET R5, 1 → opcode=11010, rD=5, imm[15:0] has the value
        // 11010_0_00_0101_0000_0000_0000_0000_0001 = 0xD050_0001
        ir = 32'hD050_0001; #1;
        check("SET opcode", dec_opcode == `OP_SET);
        check("SET rD=R5", dec_rD_addr == 4'd5);
        check("SET rf_we=1", dec_rf_we == 1);
        check("SET use_imm=1", dec_use_imm == 1);
        check("SET uses_rA=0", dec_uses_rA == 0);

        // ============================================================
        // Edge cases
        // ============================================================
        $display("--- Edge cases ---");

        // LD with non-zero offset: LD R6, R1, 8 → addr = R1 + 8
        // 00010_0_00_0110_0001_0000_0000_0000_1000 = 0x1061_0008
        ir = 32'h1061_0008; #1;
        check("LD+offset rD=R6", dec_rD_addr == 4'd6);
        check("LD+offset rA=R1", dec_rA_addr == 4'd1);
        check("LD+offset imm=8", dec_imm16 == 16'h0008);
        check("LD+offset use_imm=1", dec_use_imm == 1);

        // MOVI with large immediate: MOVI R15, 0xFFFF
        // 00100_0_00_1111_1111_1111_1111_1111_1111
        ir = {5'b00100, 1'b0, 2'b00, 4'hF, 20'hFFFFF}; #1;
        check("MOVI.large rD=R15", dec_rD_addr == 4'd15);
        check("MOVI.large imm=0xFFFF", dec_imm16 == 16'hFFFF);

        // ============================================================
        // Summary
        // ============================================================
        $display("");
        $display("=== RESULTS: %0d PASSED, %0d FAILED (of %0d) ===",
                 pass_count, fail_count, test_num);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("*** FAILURES DETECTED ***");

        $finish;
    end

endmodule