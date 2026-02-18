/* file: cu_tb.v
 * Description: Comprehensive testbench for the ARM control unit (cu.v)
 * Author: Jeremy Cai
 * Date: Feb. 18, 2026
 *
 * Structure:
 *   Part A – Every unique instruction from sort.s (program order)
 *   Part B – Instruction types not present in sort.s
 *   Part C – Condition-gating & edge-case tests
 */

`timescale 1ns / 1ps
`include "define.v"
`include "cu.v"

module cu_tb;

    // ============================================================
    // ALU opcode constants (ARM encoding)
    // ============================================================
    localparam [3:0] ALU_AND = 4'b0000,
                     ALU_EOR = 4'b0001,
                     ALU_SUB = 4'b0010,
                     ALU_RSB = 4'b0011,
                     ALU_ADD = 4'b0100,
                     ALU_ADC = 4'b0101,
                     ALU_SBC = 4'b0110,
                     ALU_RSC = 4'b0111,
                     ALU_TST = 4'b1000,
                     ALU_TEQ = 4'b1001,
                     ALU_CMP = 4'b1010,
                     ALU_CMN = 4'b1011,
                     ALU_ORR = 4'b1100,
                     ALU_MOV = 4'b1101,
                     ALU_BIC = 4'b1110,
                     ALU_MVN = 4'b1111;

    // Shift type constants
    localparam [1:0] SH_LSL = 2'b00,
                     SH_LSR = 2'b01,
                     SH_ASR = 2'b10,
                     SH_ROR = 2'b11;

    // Write-back source select constants
    localparam [2:0] WB_ALU  = 3'b000,
                     WB_MEM  = 3'b001,
                     WB_LINK = 3'b010,
                     WB_PSR  = 3'b011,
                     WB_MUL  = 3'b100;

    // ============================================================
    // DUT port wires
    // ============================================================
    reg  [`INSTR_WIDTH-1:0] instr;
    reg                     cond_met;

    wire t_dp_reg, t_dp_imm, t_mul, t_mull, t_swp, t_bx;
    wire t_hdt_rego, t_hdt_immo, t_sdt_rego, t_sdt_immo;
    wire t_bdt, t_br, t_mrs, t_msr_reg, t_msr_imm, t_swi, t_undef;

    wire [3:0]  rn_addr, rd_addr, rs_addr, rm_addr;
    wire [3:0]  wr_addr1, wr_addr2;
    wire        wr_en1, wr_en2;

    wire [3:0]  alu_op;
    wire        alu_src_b, cpsr_wen;

    wire [1:0]  shift_type;
    wire [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount;
    wire        shift_src;

    wire [31:0] imm32;

    wire        mem_read, mem_write;
    wire [1:0]  mem_size;
    wire        mem_signed;

    wire        addr_pre_idx, addr_up, addr_wb;
    wire [2:0]  wb_sel;

    wire        branch_en, branch_link, branch_exchange;

    wire        mul_en, mul_long, mul_signed_out, mul_accumulate;

    wire        psr_rd, psr_wr, psr_field_sel;
    wire [3:0]  psr_mask;

    wire [15:0] bdt_list;
    wire        bdt_load, bdt_s, bdt_wb;

    wire        swap_byte, swi_en;

    wire        use_rn, use_rd, use_rs, use_rm;
    wire        is_multi_cycle;

    // ============================================================
    // Bookkeeping
    // ============================================================
    integer errors;
    integer passes;
    integer checks;
    integer test_num;
    integer test_errors;   // errors within current test

    // ============================================================
    // DUT instantiation
    // ============================================================
    cu dut (
        .instr          (instr),
        .cond_met       (cond_met),

        .t_dp_reg       (t_dp_reg),
        .t_dp_imm       (t_dp_imm),
        .t_mul          (t_mul),
        .t_mull         (t_mull),
        .t_swp          (t_swp),
        .t_bx           (t_bx),
        .t_hdt_rego     (t_hdt_rego),
        .t_hdt_immo     (t_hdt_immo),
        .t_sdt_rego     (t_sdt_rego),
        .t_sdt_immo     (t_sdt_immo),
        .t_bdt          (t_bdt),
        .t_br           (t_br),
        .t_mrs          (t_mrs),
        .t_msr_reg      (t_msr_reg),
        .t_msr_imm      (t_msr_imm),
        .t_swi          (t_swi),
        .t_undef        (t_undef),

        .rn_addr        (rn_addr),
        .rd_addr        (rd_addr),
        .rs_addr        (rs_addr),
        .rm_addr        (rm_addr),

        .wr_addr1       (wr_addr1),
        .wr_en1         (wr_en1),
        .wr_addr2       (wr_addr2),
        .wr_en2         (wr_en2),

        .alu_op         (alu_op),
        .alu_src_b      (alu_src_b),
        .cpsr_wen       (cpsr_wen),

        .shift_type     (shift_type),
        .shift_amount   (shift_amount),
        .shift_src      (shift_src),

        .imm32          (imm32),

        .mem_read       (mem_read),
        .mem_write      (mem_write),
        .mem_size       (mem_size),
        .mem_signed     (mem_signed),

        .addr_pre_idx   (addr_pre_idx),
        .addr_up        (addr_up),
        .addr_wb        (addr_wb),

        .wb_sel         (wb_sel),

        .branch_en      (branch_en),
        .branch_link    (branch_link),
        .branch_exchange(branch_exchange),

        .mul_en         (mul_en),
        .mul_long       (mul_long),
        .mul_signed     (mul_signed_out),
        .mul_accumulate (mul_accumulate),

        .psr_rd         (psr_rd),
        .psr_wr         (psr_wr),
        .psr_field_sel  (psr_field_sel),
        .psr_mask       (psr_mask),

        .bdt_list       (bdt_list),
        .bdt_load       (bdt_load),
        .bdt_s          (bdt_s),
        .bdt_wb         (bdt_wb),

        .swap_byte      (swap_byte),
        .swi_en         (swi_en),

        .use_rn         (use_rn),
        .use_rd         (use_rd),
        .use_rs         (use_rs),
        .use_rm         (use_rm),

        .is_multi_cycle (is_multi_cycle)
    );

    // ============================================================
    // Helper tasks
    // ============================================================
    task apply(input [`INSTR_WIDTH-1:0] instruction, input cm);
        begin
            instr    = instruction;
            cond_met = cm;
            #10;
        end
    endtask

    task check(input [255:0] name, input [31:0] actual, input [31:0] expected);
        begin
            checks = checks + 1;
            if (actual !== expected) begin
                $display("    [FAIL] %0s = 0x%08h, expected 0x%08h", name, actual, expected);
                errors      = errors + 1;
                test_errors = test_errors + 1;
            end else begin
                passes = passes + 1;
            end
        end
    endtask

    task header(input [799:0] desc);
        begin
            // Print result of PREVIOUS test (if any)
            if (test_num > 0) begin
                if (test_errors == 0)
                    $display("  [PASS] Test %0d completed with 0 errors\n", test_num);
                else
                    $display("  [FAIL] Test %0d completed with %0d error(s)\n", test_num, test_errors);
            end
            test_num    = test_num + 1;
            test_errors = 0;
            $display("--- Test %0d: %0s ---", test_num, desc);
        end
    endtask

    // Verify that exactly one type flag is set (no overlap)
    task check_one_type;
        reg [16:0] sum;
        begin
            checks = checks + 1;
            sum = t_dp_reg + t_dp_imm + t_mul + t_mull + t_swp + t_bx +
                  t_hdt_rego + t_hdt_immo + t_sdt_rego + t_sdt_immo +
                  t_bdt + t_br + t_mrs + t_msr_reg + t_msr_imm + t_swi + t_undef;
            if (sum !== 1) begin
                $display("    [FAIL] Exactly one type flag should be set, got %0d (dp_r=%0b dp_i=%0b mul=%0b mull=%0b swp=%0b bx=%0b hdt_r=%0b hdt_i=%0b sdt_r=%0b sdt_i=%0b bdt=%0b br=%0b mrs=%0b msr_r=%0b msr_i=%0b swi=%0b undef=%0b)",
                         sum, t_dp_reg, t_dp_imm, t_mul, t_mull, t_swp, t_bx,
                         t_hdt_rego, t_hdt_immo, t_sdt_rego, t_sdt_immo,
                         t_bdt, t_br, t_mrs, t_msr_reg, t_msr_imm, t_swi, t_undef);
                errors      = errors + 1;
                test_errors = test_errors + 1;
            end else begin
                passes = passes + 1;
            end
        end
    endtask

    // Print final result of the last test
    task finish_last_test;
        begin
            if (test_num > 0) begin
                if (test_errors == 0)
                    $display("  [PASS] Test %0d completed with 0 errors\n", test_num);
                else
                    $display("  [FAIL] Test %0d completed with %0d error(s)\n", test_num, test_errors);
            end
        end
    endtask

    // ============================================================
    // Main stimulus
    // ============================================================
    initial begin
        $dumpfile("cu_tb.vcd");
        $dumpvars(0, cu_tb);

        errors      = 0;
        passes      = 0;
        checks      = 0;
        test_num    = 0;
        test_errors = 0;
        instr       = 32'h0;
        cond_met    = 1;
        #20;

        // ========================================================
        //  PART A: sort.s instruction tests
        // ========================================================
        $display("\n##################################################");
        $display(" PART A: sort.s instruction tests");
        $display("##################################################\n");

        // ----- 0x000: PUSH {fp, lr} = STMDB sp!, {r11,r14} -----
        header("0x000 PUSH {fp,lr} (E92D4800) - BDT store pre-dec WB");
        apply(32'hE92D4800, 1);
        check_one_type;
        check("t_bdt",         t_bdt,          1);
        check("rn_addr",       rn_addr,        4'd13);
        check("bdt_list",      bdt_list,       16'h4800);
        check("bdt_load",      bdt_load,       0);
        check("bdt_wb",        bdt_wb,         1);
        check("addr_pre_idx",  addr_pre_idx,   1);
        check("addr_up",       addr_up,        0);
        check("bdt_s",         bdt_s,          0);
        check("is_multi_cycle",is_multi_cycle, 1);
        check("mem_read",      mem_read,       0);
        check("mem_write",     mem_write,      0);
        check("wr_en1",        wr_en1,         0);
        check("wr_en2",        wr_en2,         0);
        check("use_rn",        use_rn,         1);
        check("branch_en",     branch_en,      0);

        // ----- 0x004: ADD fp, sp, #4 -----
        header("0x004 ADD fp,sp,#4 (E28DB004) - DP imm ADD");
        apply(32'hE28DB004, 1);
        check_one_type;
        check("t_dp_imm",  t_dp_imm,   1);
        check("alu_op",    alu_op,      ALU_ADD);
        check("alu_src_b", alu_src_b,   1);
        check("rn_addr",   rn_addr,     4'd13);
        check("rd_addr",   rd_addr,     4'd11);
        check("imm32",     imm32,       32'd4);
        check("cpsr_wen",  cpsr_wen,    0);
        check("wr_en1",    wr_en1,      1);
        check("wr_addr1",  wr_addr1,    4'd11);
        check("wr_en2",    wr_en2,      0);
        check("wb_sel",    wb_sel,      WB_ALU);
        check("mem_read",  mem_read,    0);
        check("mem_write", mem_write,   0);
        check("branch_en", branch_en,   0);
        check("use_rn",    use_rn,      1);
        check("use_rm",    use_rm,      0);
        check("use_rs",    use_rs,      0);

        // ----- 0x008: SUB sp, sp, #56 -----
        header("0x008 SUB sp,sp,#56 (E24DD038) - DP imm SUB");
        apply(32'hE24DD038, 1);
        check_one_type;
        check("t_dp_imm",  t_dp_imm,   1);
        check("alu_op",    alu_op,      ALU_SUB);
        check("rn_addr",   rn_addr,     4'd13);
        check("rd_addr",   rd_addr,     4'd13);
        check("imm32",     imm32,       32'd56);
        check("wr_en1",    wr_en1,      1);
        check("wr_addr1",  wr_addr1,    4'd13);

        // ----- 0x00C: LDR r3, [pc, #260] -----
        header("0x00C LDR r3,[pc,#260] (E59F3104) - SDT imm LDR U=1");
        apply(32'hE59F3104, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("alu_op",     alu_op,     ALU_ADD);
        check("alu_src_b",  alu_src_b,  1);
        check("rn_addr",    rn_addr,    4'd15);
        check("rd_addr",    rd_addr,    4'd3);
        check("imm32",      imm32,      32'd260);
        check("mem_read",   mem_read,   1);
        check("mem_write",  mem_write,  0);
        check("mem_size",   mem_size,   2'b10);
        check("addr_pre_idx",addr_pre_idx, 1);
        check("addr_up",    addr_up,    1);
        check("addr_wb",    addr_wb,    0);
        check("wr_en1",     wr_en1,     1);
        check("wr_addr1",   wr_addr1,   4'd3);
        check("wr_en2",     wr_en2,     0);
        check("wb_sel",     wb_sel,     WB_MEM);
        check("use_rn",     use_rn,     1);
        check("use_rd",     use_rd,     0);

        // ----- 0x010: SUB ip, fp, #56 -----
        header("0x010 SUB ip,fp,#56 (E24BC038) - DP imm SUB");
        apply(32'hE24BC038, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_SUB);
        check("rn_addr",  rn_addr,  4'd11);
        check("rd_addr",  rd_addr,  4'd12);
        check("imm32",    imm32,    32'd56);
        check("wr_en1",   wr_en1,   1);
        check("wr_addr1", wr_addr1, 4'd12);

        // ----- 0x014: MOV lr, r3 -----
        header("0x014 MOV lr,r3 (E1A0E003) - DP reg MOV");
        apply(32'hE1A0E003, 1);
        check_one_type;
        check("t_dp_reg",    t_dp_reg,    1);
        check("alu_op",      alu_op,      ALU_MOV);
        check("alu_src_b",   alu_src_b,   0);
        check("rd_addr",     rd_addr,     4'd14);
        check("rm_addr",     rm_addr,     4'd3);
        check("shift_type",  shift_type,  SH_LSL);
        check("shift_amount",shift_amount,5'd0);
        check("shift_src",   shift_src,   0);
        check("wr_en1",      wr_en1,      1);
        check("wr_addr1",    wr_addr1,    4'd14);
        check("wb_sel",      wb_sel,      WB_ALU);
        check("mem_read",    mem_read,    0);
        check("mem_write",   mem_write,   0);
        check("use_rm",      use_rm,      1);
        check("use_rs",      use_rs,      0);

        // ----- 0x018: LDMIA lr!, {r0-r3} -----
        header("0x018 LDMIA lr!,{r0-r3} (E8BE000F) - BDT load IA WB");
        apply(32'hE8BE000F, 1);
        check_one_type;
        check("t_bdt",         t_bdt,          1);
        check("rn_addr",       rn_addr,        4'd14);
        check("bdt_list",      bdt_list,       16'h000F);
        check("bdt_load",      bdt_load,       1);
        check("bdt_wb",        bdt_wb,         1);
        check("addr_pre_idx",  addr_pre_idx,   0);
        check("addr_up",       addr_up,        1);
        check("is_multi_cycle",is_multi_cycle, 1);
        check("use_rn",        use_rn,         1);

        // ----- 0x01C: STMIA ip!, {r0-r3} -----
        header("0x01C STMIA ip!,{r0-r3} (E8AC000F) - BDT store IA WB");
        apply(32'hE8AC000F, 1);
        check_one_type;
        check("t_bdt",    t_bdt,    1);
        check("rn_addr",  rn_addr,  4'd12);
        check("bdt_list", bdt_list, 16'h000F);
        check("bdt_load", bdt_load, 0);
        check("bdt_wb",   bdt_wb,   1);
        check("addr_up",  addr_up,  1);
        check("addr_pre_idx", addr_pre_idx, 0);

        // ----- 0x028: LDM lr, {r0,r1} (no writeback) -----
        header("0x028 LDM lr,{r0,r1} (E89E0003) - BDT load no WB");
        apply(32'hE89E0003, 1);
        check_one_type;
        check("t_bdt",    t_bdt,    1);
        check("rn_addr",  rn_addr,  4'd14);
        check("bdt_list", bdt_list, 16'h0003);
        check("bdt_load", bdt_load, 1);
        check("bdt_wb",   bdt_wb,   0);
        check("addr_pre_idx", addr_pre_idx, 0);

        // ----- 0x02C: STM ip, {r0,r1} (no writeback) -----
        header("0x02C STM ip,{r0,r1} (E88C0003) - BDT store no WB");
        apply(32'hE88C0003, 1);
        check_one_type;
        check("t_bdt",    t_bdt,    1);
        check("rn_addr",  rn_addr,  4'd12);
        check("bdt_list", bdt_list, 16'h0003);
        check("bdt_load", bdt_load, 0);
        check("bdt_wb",   bdt_wb,   0);

        // ----- 0x030: MOV r3, #0 -----
        header("0x030 MOV r3,#0 (E3A03000) - DP imm MOV");
        apply(32'hE3A03000, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_MOV);
        check("rd_addr",  rd_addr,  4'd3);
        check("imm32",    imm32,    32'd0);
        check("wr_en1",   wr_en1,   1);
        check("wr_addr1", wr_addr1, 4'd3);
        check("wb_sel",   wb_sel,   WB_ALU);

        // ----- 0x034: STR r3, [fp, #-8] -----
        header("0x034 STR r3,[fp,#-8] (E50B3008) - SDT imm STR U=0");
        apply(32'hE50B3008, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("alu_op",     alu_op,     ALU_SUB);
        check("alu_src_b",  alu_src_b,  1);
        check("rn_addr",    rn_addr,    4'd11);
        check("rd_addr",    rd_addr,    4'd3);
        check("imm32",      imm32,      32'd8);
        check("mem_read",   mem_read,   0);
        check("mem_write",  mem_write,  1);
        check("mem_size",   mem_size,   2'b10);
        check("addr_pre_idx",addr_pre_idx, 1);
        check("addr_up",    addr_up,    0);
        check("addr_wb",    addr_wb,    0);
        check("wr_en1",     wr_en1,     0);
        check("wr_en2",     wr_en2,     0);
        check("use_rn",     use_rn,     1);
        check("use_rd",     use_rd,     1);

        // ----- 0x038: B .L2 (forward, offset=0x2E) -----
        header("0x038 B .L2 (EA00002E) - Branch forward");
        apply(32'hEA00002E, 1);
        check_one_type;
        check("t_br",          t_br,          1);
        check("branch_en",     branch_en,     1);
        check("branch_link",   branch_link,   0);
        check("imm32",         imm32,         32'h000000B8);
        check("wr_en1",        wr_en1,        0);
        check("mem_read",      mem_read,      0);
        check("mem_write",     mem_write,     0);

        // ----- 0x03C: LDR r3, [fp, #-8] -----
        header("0x03C LDR r3,[fp,#-8] (E51B3008) - SDT imm LDR U=0");
        apply(32'hE51B3008, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("alu_op",     alu_op,     ALU_SUB);
        check("rn_addr",    rn_addr,    4'd11);
        check("rd_addr",    rd_addr,    4'd3);
        check("imm32",      imm32,      32'd8);
        check("mem_read",   mem_read,   1);
        check("mem_write",  mem_write,  0);
        check("addr_up",    addr_up,    0);
        check("addr_pre_idx",addr_pre_idx, 1);
        check("addr_wb",    addr_wb,    0);
        check("wr_en1",     wr_en1,     1);
        check("wr_addr1",   wr_addr1,   4'd3);
        check("wb_sel",     wb_sel,     WB_MEM);

        // ----- 0x040: ADD r3, r3, #1 -----
        header("0x040 ADD r3,r3,#1 (E2833001) - DP imm ADD same Rn/Rd");
        apply(32'hE2833001, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_ADD);
        check("rn_addr",  rn_addr,  4'd3);
        check("rd_addr",  rd_addr,  4'd3);
        check("imm32",    imm32,    32'd1);
        check("wr_en1",   wr_en1,   1);
        check("wr_addr1", wr_addr1, 4'd3);
        check("cpsr_wen", cpsr_wen, 0);

        // ----- 0x044: STR r3, [fp, #-12] -----
        header("0x044 STR r3,[fp,#-12] (E50B300C) - SDT imm STR off=12");
        apply(32'hE50B300C, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("rn_addr",    rn_addr,    4'd11);
        check("rd_addr",    rd_addr,    4'd3);
        check("imm32",      imm32,      32'd12);
        check("mem_write",  mem_write,  1);
        check("mem_read",   mem_read,   0);
        check("wr_en1",     wr_en1,     0);
        check("use_rd",     use_rd,     1);

        // ----- 0x048: B .L3 (forward, offset=0x24) -----
        header("0x048 B .L3 (EA000024) - Branch forward");
        apply(32'hEA000024, 1);
        check_one_type;
        check("t_br",        t_br,        1);
        check("branch_en",   branch_en,   1);
        check("branch_link", branch_link, 0);
        check("imm32",       imm32,       32'h00000090);

        // ----- 0x050: LSL r3, r3, #2 = MOV r3, r3, LSL #2 -----
        header("0x050 LSL r3,r3,#2 (E1A03103) - DP reg MOV+LSL");
        apply(32'hE1A03103, 1);
        check_one_type;
        check("t_dp_reg",    t_dp_reg,    1);
        check("alu_op",      alu_op,      ALU_MOV);
        check("alu_src_b",   alu_src_b,   0);
        check("rd_addr",     rd_addr,     4'd3);
        check("rm_addr",     rm_addr,     4'd3);
        check("shift_type",  shift_type,  SH_LSL);
        check("shift_amount",shift_amount,5'd2);
        check("shift_src",   shift_src,   0);
        check("wr_en1",      wr_en1,      1);
        check("wr_addr1",    wr_addr1,    4'd3);
        check("use_rm",      use_rm,      1);
        check("use_rs",      use_rs,      0);

        // ----- 0x054: SUB r3, r3, #4 -----
        header("0x054 SUB r3,r3,#4 (E2433004) - DP imm SUB");
        apply(32'hE2433004, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_SUB);
        check("rn_addr",  rn_addr,  4'd3);
        check("rd_addr",  rd_addr,  4'd3);
        check("imm32",    imm32,    32'd4);
        check("wr_en1",   wr_en1,   1);

        // ----- 0x058: ADD r3, r3, fp -----
        header("0x058 ADD r3,r3,fp (E083300B) - DP reg ADD");
        apply(32'hE083300B, 1);
        check_one_type;
        check("t_dp_reg",    t_dp_reg,    1);
        check("alu_op",      alu_op,      ALU_ADD);
        check("rn_addr",     rn_addr,     4'd3);
        check("rd_addr",     rd_addr,     4'd3);
        check("rm_addr",     rm_addr,     4'd11);
        check("shift_type",  shift_type,  SH_LSL);
        check("shift_amount",shift_amount,5'd0);
        check("wr_en1",      wr_en1,      1);
        check("use_rn",      use_rn,      1);
        check("use_rm",      use_rm,      1);

        // ----- 0x05C: LDR r2, [r3, #-52] -----
        header("0x05C LDR r2,[r3,#-52] (E5132034) - SDT imm off=52 U=0");
        apply(32'hE5132034, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("alu_op",     alu_op,     ALU_SUB);
        check("rn_addr",    rn_addr,    4'd3);
        check("rd_addr",    rd_addr,    4'd2);
        check("imm32",      imm32,      32'd52);
        check("mem_read",   mem_read,   1);
        check("mem_write",  mem_write,  0);
        check("wr_en1",     wr_en1,     1);
        check("wr_addr1",   wr_addr1,   4'd2);
        check("wb_sel",     wb_sel,     WB_MEM);

        // ----- 0x070: LDR r3, [r3, #-52] -----
        header("0x070 LDR r3,[r3,#-52] (E5133034) - SDT same Rn/Rd");
        apply(32'hE5133034, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("rn_addr",    rn_addr,    4'd3);
        check("rd_addr",    rd_addr,    4'd3);
        check("wr_addr1",   wr_addr1,   4'd3);
        check("mem_read",   mem_read,   1);

        // ----- 0x074: CMP r2, r3 -----
        header("0x074 CMP r2,r3 (E1520003) - DP reg CMP");
        apply(32'hE1520003, 1);
        check_one_type;
        check("t_dp_reg",  t_dp_reg,  1);
        check("alu_op",    alu_op,    ALU_CMP);
        check("alu_src_b", alu_src_b, 0);
        check("rn_addr",   rn_addr,   4'd2);
        check("rm_addr",   rm_addr,   4'd3);
        check("cpsr_wen",  cpsr_wen,  1);
        check("wr_en1",    wr_en1,    0);
        check("use_rn",    use_rn,    1);
        check("use_rm",    use_rm,    1);

        // ----- 0x078: BGE .L4 (cond=GE, forward) -----
        header("0x078 BGE .L4 (AA000015) cond MET - Branch fwd");
        apply(32'hAA000015, 1);
        check_one_type;
        check("t_br",        t_br,        1);
        check("branch_en",   branch_en,   1);
        check("branch_link", branch_link, 0);
        check("imm32",       imm32,       32'h00000054);
        check("wr_en1",      wr_en1,      0);

        // Same instruction, cond_met = 0
        header("0x078 BGE .L4 (AA000015) cond NOT MET");
        apply(32'hAA000015, 0);
        check("t_br",      t_br,      1);
        check("branch_en", branch_en, 0);
        check("wr_en1",    wr_en1,    0);
        check("mem_read",  mem_read,  0);
        check("mem_write", mem_write, 0);

        // ----- 0x090: STR r3, [fp, #-16] -----
        header("0x090 STR r3,[fp,#-16] (E50B3010) - SDT STR off=16");
        apply(32'hE50B3010, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("rn_addr",    rn_addr,    4'd11);
        check("rd_addr",    rd_addr,    4'd3);
        check("imm32",      imm32,      32'd16);
        check("mem_write",  mem_write,  1);
        check("mem_read",   mem_read,   0);
        check("wr_en1",     wr_en1,     0);

        // ----- 0x0B8: STR r2, [r3, #-52] -----
        header("0x0B8 STR r2,[r3,#-52] (E5032034) - SDT STR diff Rd");
        apply(32'hE5032034, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("rn_addr",    rn_addr,    4'd3);
        check("rd_addr",    rd_addr,    4'd2);
        check("imm32",      imm32,      32'd52);
        check("mem_write",  mem_write,  1);
        check("use_rd",     use_rd,     1);
        check("use_rn",     use_rn,     1);

        // ----- 0x0CC: LDR r2, [fp, #-16] -----
        header("0x0CC LDR r2,[fp,#-16] (E51B2010) - SDT LDR Rd=R2");
        apply(32'hE51B2010, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("rn_addr",    rn_addr,    4'd11);
        check("rd_addr",    rd_addr,    4'd2);
        check("imm32",      imm32,      32'd16);
        check("mem_read",   mem_read,   1);
        check("mem_write",  mem_write,  0);
        check("wr_addr1",   wr_addr1,   4'd2);
        check("wr_en1",     wr_en1,     1);

        // ----- 0x0E4: CMP r3, #9 -----
        header("0x0E4 CMP r3,#9 (E3530009) - DP imm CMP");
        apply(32'hE3530009, 1);
        check_one_type;
        check("t_dp_imm",  t_dp_imm,  1);
        check("alu_op",    alu_op,    ALU_CMP);
        check("alu_src_b", alu_src_b, 1);
        check("rn_addr",   rn_addr,   4'd3);
        check("imm32",     imm32,     32'd9);
        check("cpsr_wen",  cpsr_wen,  1);
        check("wr_en1",    wr_en1,    0);
        check("use_rn",    use_rn,    1);

        // ----- 0x0E8: BLE .L5 (backward) -----
        // DAFFFFD7  signed_off24=0xFFFFD7 → sign-ext + <<2 = 0xFFFFFF5C
        header("0x0E8 BLE .L5 (DAFFFFD7) cond MET - Branch backward");
        apply(32'hDAFFFFD7, 1);
        check_one_type;
        check("t_br",        t_br,        1);
        check("branch_en",   branch_en,   1);
        check("branch_link", branch_link, 0);
        check("imm32",       imm32,       32'hFFFFFF5C);

        // ----- 0x0FC: CMP r3, #9 (duplicate encoding - verify again) -----
        header("0x0FC CMP r3,#9 (E3530009) - DP imm CMP (dup)");
        apply(32'hE3530009, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_CMP);
        check("cpsr_wen", cpsr_wen, 1);
        check("wr_en1",   wr_en1,   0);

        // ----- 0x100: BLE .L6 (backward) -----
        // DAFFFFCD  signed_off24=0xFFFFCD → sign-ext + <<2 = 0xFFFFFF34
        header("0x100 BLE .L6 (DAFFFFCD) cond MET - Branch backward");
        apply(32'hDAFFFFCD, 1);
        check_one_type;
        check("t_br",      t_br,      1);
        check("branch_en", branch_en, 1);
        check("imm32",     imm32,     32'hFFFFFF34);

        // ----- 0x104: MOV r3, #0 -----
        header("0x104 MOV r3,#0 (E3A03000) - DP imm MOV (dup)");
        apply(32'hE3A03000, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_MOV);
        check("imm32",    imm32,    32'd0);
        check("wr_en1",   wr_en1,   1);

        // ----- 0x108: MOV r0, r3 -----
        header("0x108 MOV r0,r3 (E1A00003) - DP reg MOV");
        apply(32'hE1A00003, 1);
        check_one_type;
        check("t_dp_reg", t_dp_reg, 1);
        check("alu_op",   alu_op,   ALU_MOV);
        check("rd_addr",  rd_addr,  4'd0);
        check("rm_addr",  rm_addr,  4'd3);
        check("wr_en1",   wr_en1,   1);
        check("wr_addr1", wr_addr1, 4'd0);
        check("wb_sel",   wb_sel,   WB_ALU);

        // ----- 0x10C: SUB sp, fp, #4 -----
        header("0x10C SUB sp,fp,#4 (E24BD004) - DP imm SUB");
        apply(32'hE24BD004, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_SUB);
        check("rn_addr",  rn_addr,  4'd11);
        check("rd_addr",  rd_addr,  4'd13);
        check("imm32",    imm32,    32'd4);
        check("wr_en1",   wr_en1,   1);
        check("wr_addr1", wr_addr1, 4'd13);

        // ----- 0x110: POP {fp, lr} -----
        header("0x110 POP {fp,lr} (E8BD4800) - BDT load IA WB");
        apply(32'hE8BD4800, 1);
        check_one_type;
        check("t_bdt",         t_bdt,          1);
        check("rn_addr",       rn_addr,        4'd13);
        check("bdt_list",      bdt_list,       16'h4800);
        check("bdt_load",      bdt_load,       1);
        check("bdt_wb",        bdt_wb,         1);
        check("addr_pre_idx",  addr_pre_idx,   0);
        check("addr_up",       addr_up,        1);
        check("is_multi_cycle",is_multi_cycle, 1);
        check("wr_en1",        wr_en1,         0);

        // ----- 0x114: BX lr -----
        header("0x114 BX lr (E12FFF1E) - Branch exchange");
        apply(32'hE12FFF1E, 1);
        check_one_type;
        check("t_bx",             t_bx,             1);
        check("rm_addr",          rm_addr,          4'd14);
        check("branch_en",        branch_en,        1);
        check("branch_exchange",  branch_exchange,  1);
        check("branch_link",      branch_link,      0);
        check("wr_en1",           wr_en1,           0);
        check("wr_en2",           wr_en2,           0);
        check("mem_read",         mem_read,         0);
        check("mem_write",        mem_write,        0);
        check("use_rm",           use_rm,           1);

        // ========================================================
        //  PART B: Instruction types not in sort.s
        // ========================================================
        $display("\n##################################################");
        $display(" PART B: Additional instruction type coverage");
        $display("##################################################\n");

        // ----- MUL R1, R2, R3 -----
        // E0010392
        header("MUL R1,R2,R3 (E0010392)");
        apply(32'hE0010392, 1);
        check_one_type;
        check("t_mul",         t_mul,         1);
        check("wr_en1",        wr_en1,        1);
        check("wr_addr1",      wr_addr1,      4'd1);
        check("rn_addr",       rn_addr,       4'd1);
        check("rs_addr",       rs_addr,       4'd3);
        check("rm_addr",       rm_addr,       4'd2);
        check("mul_en",        mul_en,        1);
        check("mul_long",      mul_long,      0);
        check("mul_accumulate",mul_accumulate,0);
        check("cpsr_wen",      cpsr_wen,      0);
        check("wb_sel",        wb_sel,        WB_MUL);
        check("mem_read",      mem_read,      0);
        check("mem_write",     mem_write,     0);
        check("branch_en",     branch_en,     0);
        check("use_rm",        use_rm,        1);
        check("use_rs",        use_rs,        1);
        check("use_rd",        use_rd,        0);

        // ----- MULS R1, R2, R3 (S=1) -----
        // E0110392
        header("MULS R1,R2,R3 (E0110392) - MUL + S flag");
        apply(32'hE0110392, 1);
        check_one_type;
        check("t_mul",    t_mul,    1);
        check("cpsr_wen", cpsr_wen, 1);
        check("wr_en1",   wr_en1,   1);

        // ----- MLA R1, R2, R3, R4 -----
        // E0214392
        header("MLA R1,R2,R3,R4 (E0214392)");
        apply(32'hE0214392, 1);
        check_one_type;
        check("t_mul",         t_mul,         1);
        check("mul_accumulate",mul_accumulate,1);
        check("wr_addr1",      wr_addr1,      4'd1);
        check("rd_addr",       rd_addr,       4'd4);
        check("use_rd",        use_rd,        1);
        check("use_rm",        use_rm,        1);
        check("use_rs",        use_rs,        1);

        // ----- UMULL R0, R1, R2, R3 -----
        // E0810392
        header("UMULL R0,R1,R2,R3 (E0810392)");
        apply(32'hE0810392, 1);
        check_one_type;
        check("t_mull",        t_mull,         1);
        check("mul_en",        mul_en,         1);
        check("mul_long",      mul_long,       1);
        check("mul_signed_out",mul_signed_out, 0);
        check("mul_accumulate",mul_accumulate, 0);
        check("wr_en1",        wr_en1,         1);
        check("wr_addr1",      wr_addr1,       4'd0);
        check("wr_en2",        wr_en2,         1);
        check("wr_addr2",      wr_addr2,       4'd1);
        check("wb_sel",        wb_sel,         WB_MUL);
        check("use_rm",        use_rm,         1);
        check("use_rs",        use_rs,         1);
        check("use_rd",        use_rd,         0);
        check("use_rn",        use_rn,         0);

        // ----- SMLAL R0, R1, R2, R3 -----
        // E0E10392
        header("SMLAL R0,R1,R2,R3 (E0E10392)");
        apply(32'hE0E10392, 1);
        check_one_type;
        check("t_mull",        t_mull,         1);
        check("mul_signed_out",mul_signed_out, 1);
        check("mul_accumulate",mul_accumulate, 1);
        check("use_rn",        use_rn,         1);
        check("use_rd",        use_rd,         1);
        check("wr_en1",        wr_en1,         1);
        check("wr_en2",        wr_en2,         1);

        // ----- SMULL R0, R1, R2, R3 (signed, no accumulate) -----
        // E0C10392
        header("SMULL R0,R1,R2,R3 (E0C10392)");
        apply(32'hE0C10392, 1);
        check_one_type;
        check("t_mull",        t_mull,         1);
        check("mul_signed_out",mul_signed_out, 1);
        check("mul_accumulate",mul_accumulate, 0);
        check("use_rn",        use_rn,         0);
        check("use_rd",        use_rd,         0);

        // ----- SWP R1, R2, [R3] -----
        // E1031092
        header("SWP R1,R2,[R3] (E1031092)");
        apply(32'hE1031092, 1);
        check_one_type;
        check("t_swp",         t_swp,          1);
        check("rn_addr",       rn_addr,        4'd3);
        check("rd_addr",       rd_addr,        4'd1);
        check("rm_addr",       rm_addr,        4'd2);
        check("swap_byte",     swap_byte,      0);
        check("is_multi_cycle",is_multi_cycle, 1);
        check("wr_en1",        wr_en1,         0);
        check("use_rm",        use_rm,         1);
        check("mem_read",      mem_read,       0);
        check("mem_write",     mem_write,      0);

        // ----- SWPB R1, R2, [R3] -----
        // E1431092
        header("SWPB R1,R2,[R3] (E1431092)");
        apply(32'hE1431092, 1);
        check_one_type;
        check("t_swp",     t_swp,     1);
        check("swap_byte", swap_byte, 1);
        check("wr_en1",    wr_en1,    0);

        // ----- MRS R0, CPSR -----
        // E10F0000
        header("MRS R0,CPSR (E10F0000)");
        apply(32'hE10F0000, 1);
        check_one_type;
        check("t_mrs",        t_mrs,        1);
        check("rd_addr",      rd_addr,      4'd0);
        check("psr_rd",       psr_rd,       1);
        check("psr_field_sel",psr_field_sel,0);
        check("wr_en1",       wr_en1,       1);
        check("wr_addr1",     wr_addr1,     4'd0);
        check("wb_sel",       wb_sel,       WB_PSR);
        check("mem_read",     mem_read,     0);
        check("mem_write",    mem_write,    0);

        // ----- MRS R0, SPSR -----
        // E14F0000
        header("MRS R0,SPSR (E14F0000)");
        apply(32'hE14F0000, 1);
        check_one_type;
        check("t_mrs",        t_mrs,        1);
        check("psr_field_sel",psr_field_sel,1);
        check("wr_en1",       wr_en1,       1);

        // ----- MSR CPSR_f, R1 -----
        // E128F001
        header("MSR CPSR_f,R1 (E128F001)");
        apply(32'hE128F001, 1);
        check_one_type;
        check("t_msr_reg",    t_msr_reg,    1);
        check("rm_addr",      rm_addr,      4'd1);
        check("psr_wr",       psr_wr,       1);
        check("psr_field_sel",psr_field_sel,0);
        check("psr_mask",     psr_mask,     4'b1000);
        check("wr_en1",       wr_en1,       0);
        check("use_rm",       use_rm,       1);

        // ----- MSR CPSR_f, #0x40 (immediate) -----
        // E328F040
        header("MSR CPSR_f,#0x40 (E328F040)");
        apply(32'hE328F040, 1);
        check_one_type;
        check("t_msr_imm",    t_msr_imm,    1);
        check("alu_src_b",    alu_src_b,     1);
        check("imm32",        imm32,         32'h40);
        check("psr_wr",       psr_wr,        1);
        check("wr_en1",       wr_en1,        0);

        // ----- LDRH R1, [R2, R3] (register offset) -----
        // E19210B3
        header("LDRH R1,[R2,R3] (E19210B3) - HDT reg offset");
        apply(32'hE19210B3, 1);
        check_one_type;
        check("t_hdt_rego", t_hdt_rego, 1);
        check("rn_addr",    rn_addr,    4'd2);
        check("rd_addr",    rd_addr,    4'd1);
        check("rm_addr",    rm_addr,    4'd3);
        check("alu_op",     alu_op,     ALU_ADD);
        check("alu_src_b",  alu_src_b,  0);
        check("mem_read",   mem_read,   1);
        check("mem_write",  mem_write,  0);
        check("mem_size",   mem_size,   2'b01);
        check("mem_signed", mem_signed, 0);
        check("wr_en1",     wr_en1,     1);
        check("wb_sel",     wb_sel,     WB_MEM);
        check("use_rn",     use_rn,     1);
        check("use_rm",     use_rm,     1);

        // ----- LDRH R1, [R2, #5] (immediate offset) -----
        // E1D210B5
        header("LDRH R1,[R2,#5] (E1D210B5) - HDT imm offset");
        apply(32'hE1D210B5, 1);
        check_one_type;
        check("t_hdt_immo", t_hdt_immo, 1);
        check("alu_op",     alu_op,     ALU_ADD);
        check("alu_src_b",  alu_src_b,  1);
        check("imm32",      imm32,      32'd5);
        check("mem_read",   mem_read,   1);
        check("mem_write",  mem_write,  0);
        check("mem_size",   mem_size,   2'b01);
        check("mem_signed", mem_signed, 0);
        check("wr_en1",     wr_en1,     1);

        // ----- STRH R1, [R2, #5] -----
        // E1C210B5
        header("STRH R1,[R2,#5] (E1C210B5) - HDT imm store");
        apply(32'hE1C210B5, 1);
        check_one_type;
        check("t_hdt_immo", t_hdt_immo, 1);
        check("mem_read",   mem_read,   0);
        check("mem_write",  mem_write,  1);
        check("mem_size",   mem_size,   2'b01);
        check("wr_en1",     wr_en1,     0);
        check("use_rd",     use_rd,     1);

        // ----- LDRSB R1, [R2, #5] -----
        // E1D210D5: SH=10
        header("LDRSB R1,[R2,#5] (E1D210D5) - HDT signed byte");
        apply(32'hE1D210D5, 1);
        check_one_type;
        check("t_hdt_immo", t_hdt_immo, 1);
        check("mem_read",   mem_read,   1);
        check("mem_size",   mem_size,   2'b00);
        check("mem_signed", mem_signed, 1);
        check("wr_en1",     wr_en1,     1);

        // ----- LDRSH R1, [R2, #5] -----
        // E1D210F5: SH=11
        header("LDRSH R1,[R2,#5] (E1D210F5) - HDT signed half");
        apply(32'hE1D210F5, 1);
        check_one_type;
        check("t_hdt_immo", t_hdt_immo, 1);
        check("mem_size",   mem_size,   2'b01);
        check("mem_signed", mem_signed, 1);
        check("wr_en1",     wr_en1,     1);

        // ----- LDR R1, [R2, R3, LSL #2] (register offset SDT) -----
        // E7921103
        header("LDR R1,[R2,R3,LSL#2] (E7921103) - SDT reg offset");
        apply(32'hE7921103, 1);
        check_one_type;
        check("t_sdt_rego",  t_sdt_rego,  1);
        check("alu_op",      alu_op,      ALU_ADD);
        check("alu_src_b",   alu_src_b,   0);
        check("rn_addr",     rn_addr,     4'd2);
        check("rd_addr",     rd_addr,     4'd1);
        check("rm_addr",     rm_addr,     4'd3);
        check("shift_type",  shift_type,  SH_LSL);
        check("shift_amount",shift_amount,5'd2);
        check("mem_read",    mem_read,    1);
        check("mem_write",   mem_write,   0);
        check("mem_size",    mem_size,    2'b10);
        check("addr_pre_idx",addr_pre_idx,1);
        check("addr_up",     addr_up,     1);
        check("wr_en1",      wr_en1,      1);
        check("use_rm",      use_rm,      1);
        check("use_rn",      use_rn,      1);

        // ----- STR R1, [R2, R3, LSR #4] (reg offset SDT store) -----
        // E7821223
        header("STR R1,[R2,R3,LSR#4] (E7821223) - SDT reg store LSR");
        apply(32'hE7821223, 1);
        check_one_type;
        check("t_sdt_rego",  t_sdt_rego,  1);
        check("shift_type",  shift_type,  SH_LSR);
        check("shift_amount",shift_amount,5'd4);
        check("mem_write",   mem_write,   1);
        check("mem_read",    mem_read,    0);
        check("wr_en1",      wr_en1,      0);
        check("use_rd",      use_rd,      1);

        // ----- LDRB R1, [R2, #5] -----
        // E5D21005
        header("LDRB R1,[R2,#5] (E5D21005) - SDT byte load");
        apply(32'hE5D21005, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("mem_read",   mem_read,   1);
        check("mem_size",   mem_size,   2'b00);
        check("imm32",      imm32,      32'd5);
        check("wr_en1",     wr_en1,     1);

        // ----- STRB R1, [R2, #5] -----
        // E5C21005
        header("STRB R1,[R2,#5] (E5C21005) - SDT byte store");
        apply(32'hE5C21005, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("mem_write",  mem_write,  1);
        check("mem_read",   mem_read,   0);
        check("mem_size",   mem_size,   2'b00);
        check("wr_en1",     wr_en1,     0);

        // ----- LDR R1, [R2], #5 (post-indexed) -----
        // E4921005: P=0
        header("LDR R1,[R2],#5 (E4921005) - SDT post-indexed");
        apply(32'hE4921005, 1);
        check_one_type;
        check("t_sdt_immo",  t_sdt_immo,  1);
        check("addr_pre_idx",addr_pre_idx,0);
        check("addr_up",     addr_up,     1);
        check("addr_wb",     addr_wb,     1);
        check("wr_en1",      wr_en1,      1);
        check("wr_en2",      wr_en2,      1);
        check("mem_read",    mem_read,    1);
        check("wr_addr1",    wr_addr1,    4'd1);
        check("wr_addr2",    wr_addr2,    4'd2);

        // ----- LDR R1, [R2, #5]! (pre-indexed with WB) -----
        // E5B21005: P=1 W=1
        header("LDR R1,[R2,#5]! (E5B21005) - SDT pre+WB");
        apply(32'hE5B21005, 1);
        check_one_type;
        check("t_sdt_immo",  t_sdt_immo,  1);
        check("addr_pre_idx",addr_pre_idx,1);
        check("addr_wb",     addr_wb,     1);
        check("wr_en1",      wr_en1,      1);
        check("wr_addr1",    wr_addr1,    4'd1);
        check("wr_en2",      wr_en2,      1);
        check("wr_addr2",    wr_addr2,    4'd2);

        // ----- STR R1, [R2, #5]! (pre-indexed store with WB) -----
        // E5A21005
        header("STR R1,[R2,#5]! (E5A21005) - SDT store pre+WB");
        apply(32'hE5A21005, 1);
        check_one_type;
        check("t_sdt_immo", t_sdt_immo, 1);
        check("mem_write",  mem_write,  1);
        check("addr_wb",    addr_wb,    1);
        check("wr_en1",     wr_en1,     0);
        check("wr_en2",     wr_en2,     1);
        check("wr_addr2",   wr_addr2,   4'd2);

        // ----- BL #64 (offset=0x10 words) -----
        // EB000010
        header("BL #64 (EB000010) - Branch and link");
        apply(32'hEB000010, 1);
        check_one_type;
        check("t_br",        t_br,        1);
        check("branch_en",   branch_en,   1);
        check("branch_link", branch_link, 1);
        check("imm32",       imm32,       32'h00000040);
        check("wr_en1",      wr_en1,      1);
        check("wr_addr1",    wr_addr1,    4'd14);
        check("wb_sel",      wb_sel,      WB_LINK);
        check("mem_read",    mem_read,    0);
        check("mem_write",   mem_write,   0);

        // ----- BL backward -----
        // EBFFFFEF  imm24=0xFFFFEF → sign-ext + <<2 = 0xFFFFFFBC
        header("BL backward (EBFFFFEF) - Branch+link backward");
        apply(32'hEBFFFFEF, 1);
        check("t_br",        t_br,        1);
        check("branch_en",   branch_en,   1);
        check("branch_link", branch_link, 1);
        check("imm32",       imm32,       32'hFFFFFFBC);
        check("wr_en1",      wr_en1,      1);
        check("wr_addr1",    wr_addr1,    4'd14);

        // ----- TST R1, #0xFF -----
        // E31100FF
        header("TST R1,#0xFF (E31100FF) - DP imm test no-write");
        apply(32'hE31100FF, 1);
        check_one_type;
        check("t_dp_imm",  t_dp_imm,  1);
        check("alu_op",    alu_op,    ALU_TST);
        check("rn_addr",   rn_addr,   4'd1);
        check("imm32",     imm32,     32'hFF);
        check("cpsr_wen",  cpsr_wen,  1);
        check("wr_en1",    wr_en1,    0);

        // ----- TEQ R1, R2 -----
        // E1310002
        header("TEQ R1,R2 (E1310002) - DP reg test no-write");
        apply(32'hE1310002, 1);
        check_one_type;
        check("t_dp_reg", t_dp_reg, 1);
        check("alu_op",   alu_op,   ALU_TEQ);
        check("cpsr_wen", cpsr_wen, 1);
        check("wr_en1",   wr_en1,   0);

        // ----- CMN R1, #5 -----
        // E3710005
        header("CMN R1,#5 (E3710005) - DP imm CMN");
        apply(32'hE3710005, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_CMN);
        check("cpsr_wen", cpsr_wen, 1);
        check("wr_en1",   wr_en1,   0);

        // ----- ADDS R1, R2, #5 (S=1 for non-test instruction) -----
        // E2921005
        header("ADDS R1,R2,#5 (E2921005) - DP imm ADD+S");
        apply(32'hE2921005, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_ADD);
        check("cpsr_wen", cpsr_wen, 1);
        check("wr_en1",   wr_en1,   1);
        check("wr_addr1", wr_addr1, 4'd1);

        // ----- ADD R1, R2, R3, LSL R4 (register-specified shift) -----
        // E0821413
        header("ADD R1,R2,R3,LSL R4 (E0821413) - DP reg-shift");
        apply(32'hE0821413, 1);
        check_one_type;
        check("t_dp_reg",    t_dp_reg,    1);
        check("alu_op",      alu_op,      ALU_ADD);
        check("rn_addr",     rn_addr,     4'd2);
        check("rd_addr",     rd_addr,     4'd1);
        check("rm_addr",     rm_addr,     4'd3);
        check("rs_addr",     rs_addr,     4'd4);
        check("shift_type",  shift_type,  SH_LSL);
        check("shift_src",   shift_src,   1);
        check("use_rs",      use_rs,      1);
        check("use_rm",      use_rm,      1);
        check("use_rn",      use_rn,      1);
        check("wr_en1",      wr_en1,      1);

        // ----- ORR R1, R2, R3 -----
        // E1821003
        header("ORR R1,R2,R3 (E1821003)");
        apply(32'hE1821003, 1);
        check("t_dp_reg", t_dp_reg, 1);
        check("alu_op",   alu_op,   ALU_ORR);
        check("wr_en1",   wr_en1,   1);

        // ----- AND R1, R2, #0xFF -----
        // E2021003 — wait, that's wrong. Let me recalculate.
        // AND: opcode=0000
        // E2021003 would be: cond=E, 00 1 0000 0 0010 0001 0000 00000011 → AND R1,R2,#3
        header("AND R1,R2,#3 (E2021003)");
        apply(32'hE2021003, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_AND);
        check("imm32",    imm32,    32'd3);
        check("wr_en1",   wr_en1,   1);

        // ----- EOR R1, R2, #0xFF -----
        // E2221003 → cond=E, 00 1 0001 0 0010 0001 ... no.
        // EOR opcode=0001, E2221003: 00 1 0001 0 0010 0001 0000 00000011
        header("EOR R1,R2,#3 (E2221003)");
        apply(32'hE2221003, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_EOR);
        check("wr_en1",   wr_en1,   1);

        // ----- BIC R1, R2, #0xFF -----
        // E3C210FF
        header("BIC R1,R2,#0xFF (E3C210FF)");
        apply(32'hE3C210FF, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_BIC);
        check("imm32",    imm32,    32'hFF);
        check("wr_en1",   wr_en1,   1);

        // ----- MVN R1, R2 -----
        // E1E01002
        header("MVN R1,R2 (E1E01002)");
        apply(32'hE1E01002, 1);
        check("t_dp_reg", t_dp_reg, 1);
        check("alu_op",   alu_op,   ALU_MVN);
        check("rm_addr",  rm_addr,  4'd2);
        check("wr_en1",   wr_en1,   1);

        // ----- RSB R1, R2, #0 (NEG) -----
        // E2621000
        header("RSB R1,R2,#0 (E2621000) - reverse subtract");
        apply(32'hE2621000, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_RSB);
        check("imm32",    imm32,    32'd0);
        check("wr_en1",   wr_en1,   1);

        // ----- MOV R1, #0xFF000000 (rotation=4 → ror 8) -----
        // E3A014FF
        header("MOV R1,#0xFF000000 (E3A014FF) - DP imm rot=8");
        apply(32'hE3A014FF, 1);
        check_one_type;
        check("t_dp_imm", t_dp_imm, 1);
        check("alu_op",   alu_op,   ALU_MOV);
        check("imm32",    imm32,    32'hFF000000);
        check("wr_en1",   wr_en1,   1);

        // ----- MOV R1, #0xFF00 (rotation=12 → ror 24) -----
        // E3A01CFF
        header("MOV R1,#0xFF00 (E3A01CFF) - DP imm rot=24");
        apply(32'hE3A01CFF, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("imm32",    imm32,    32'h0000FF00);

        // ----- SWI #0x123456 -----
        // EF123456
        header("SWI #0x123456 (EF123456)");
        apply(32'hEF123456, 1);
        check_one_type;
        check("t_swi",     t_swi,     1);
        check("swi_en",    swi_en,    1);
        check("wr_en1",    wr_en1,    0);
        check("mem_read",  mem_read,  0);
        check("mem_write", mem_write, 0);
        check("branch_en", branch_en, 0);

        // ----- Undefined instruction -----
        // E6000010: bits[27:25]=011, bit[4]=1
        header("Undefined (E6000010)");
        apply(32'hE6000010, 1);
        check_one_type;
        check("t_undef",   t_undef,   1);
        check("wr_en1",    wr_en1,    0);
        check("wr_en2",    wr_en2,    0);
        check("mem_read",  mem_read,  0);
        check("mem_write", mem_write, 0);
        check("branch_en", branch_en, 0);
        check("mul_en",    mul_en,    0);
        check("swi_en",    swi_en,    0);
        check("psr_wr",    psr_wr,    0);

        // ========================================================
        //  PART C: Condition gating & edge cases
        // ========================================================
        $display("\n##################################################");
        $display(" PART C: Condition gating & edge cases");
        $display("##################################################\n");

        // ----- DP with cond_met=0 -----
        header("ADD fp,sp,#4 cond NOT MET - writes suppressed");
        apply(32'hE28DB004, 0);
        check("t_dp_imm",  t_dp_imm,  1);
        check("alu_op",    alu_op,    ALU_ADD);
        check("wr_en1",    wr_en1,    0);
        check("wr_en2",    wr_en2,    0);
        check("cpsr_wen",  cpsr_wen,  0);
        check("mem_read",  mem_read,  0);
        check("mem_write", mem_write, 0);
        check("branch_en", branch_en, 0);

        // ----- ADDS with cond_met=0 -----
        header("ADDS R1,R2,#5 cond NOT MET - S-flag suppressed");
        apply(32'hE2921005, 0);
        check("t_dp_imm", t_dp_imm, 1);
        check("cpsr_wen", cpsr_wen, 0);
        check("wr_en1",   wr_en1,   0);

        // ----- CMP with cond_met=0 -----
        header("CMP r3,#9 cond NOT MET - flags suppressed");
        apply(32'hE3530009, 0);
        check("t_dp_imm", t_dp_imm, 1);
        check("cpsr_wen", cpsr_wen, 0);
        check("wr_en1",   wr_en1,   0);

        // ----- LDR with cond_met=0 -----
        header("LDR r3,[pc,#260] cond NOT MET");
        apply(32'hE59F3104, 0);
        check("t_sdt_immo", t_sdt_immo, 1);
        check("mem_read",   mem_read,   0);
        check("mem_write",  mem_write,  0);
        check("wr_en1",     wr_en1,     0);
        check("wr_en2",     wr_en2,     0);

        // ----- STR with cond_met=0 -----
        header("STR r3,[fp,#-8] cond NOT MET");
        apply(32'hE50B3008, 0);
        check("t_sdt_immo", t_sdt_immo, 1);
        check("mem_read",   mem_read,   0);
        check("mem_write",  mem_write,  0);
        check("wr_en1",     wr_en1,     0);
        check("wr_en2",     wr_en2,     0);

        // ----- LDR post-indexed with cond_met=0 -----
        header("LDR R1,[R2],#5 cond NOT MET - WB suppressed");
        apply(32'hE4921005, 0);
        check("t_sdt_immo", t_sdt_immo, 1);
        check("mem_read",   mem_read,   0);
        check("wr_en1",     wr_en1,     0);
        check("wr_en2",     wr_en2,     0);
        check("addr_wb",    addr_wb,    1);

        // ----- BDT PUSH with cond_met=0 -----
        header("PUSH {fp,lr} cond NOT MET");
        apply(32'hE92D4800, 0);
        check("t_bdt",          t_bdt,          1);
        check("is_multi_cycle", is_multi_cycle, 0);
        check("wr_en1",         wr_en1,         0);

        // ----- BDT POP with cond_met=0 -----
        header("POP {fp,lr} cond NOT MET");
        apply(32'hE8BD4800, 0);
        check("t_bdt",          t_bdt,          1);
        check("is_multi_cycle", is_multi_cycle, 0);

        // ----- BX with cond_met=0 -----
        header("BX lr cond NOT MET");
        apply(32'hE12FFF1E, 0);
        check("t_bx",            t_bx,            1);
        check("branch_en",       branch_en,       0);
        check("branch_exchange", branch_exchange, 0);
        check("wr_en1",          wr_en1,          0);

        // ----- BL with cond_met=0 -----
        header("BL #64 cond NOT MET - link suppressed");
        apply(32'hEB000010, 0);
        check("t_br",        t_br,        1);
        check("branch_en",   branch_en,   0);
        check("branch_link", branch_link, 0);
        check("wr_en1",      wr_en1,      0);

        // ----- MUL with cond_met=0 -----
        header("MUL R1,R2,R3 cond NOT MET");
        apply(32'hE0010392, 0);
        check("t_mul",  t_mul,  1);
        check("mul_en", mul_en, 0);
        check("wr_en1", wr_en1, 0);

        // ----- UMULL with cond_met=0 -----
        header("UMULL R0,R1,R2,R3 cond NOT MET");
        apply(32'hE0810392, 0);
        check("t_mull", t_mull, 1);
        check("mul_en", mul_en, 0);
        check("wr_en1", wr_en1, 0);
        check("wr_en2", wr_en2, 0);

        // ----- SWP with cond_met=0 -----
        header("SWP R1,R2,[R3] cond NOT MET");
        apply(32'hE1031092, 0);
        check("t_swp",          t_swp,          1);
        check("is_multi_cycle", is_multi_cycle, 0);
        check("wr_en1",         wr_en1,         0);

        // ----- MSR reg with cond_met=0 -----
        header("MSR CPSR_f,R1 cond NOT MET");
        apply(32'hE128F001, 0);
        check("t_msr_reg", t_msr_reg, 1);
        check("psr_wr",    psr_wr,    0);
        check("wr_en1",    wr_en1,    0);

        // ----- MSR imm with cond_met=0 -----
        header("MSR CPSR_f,#0x40 cond NOT MET");
        apply(32'hE328F040, 0);
        check("t_msr_imm", t_msr_imm, 1);
        check("psr_wr",    psr_wr,    0);

        // ----- MRS with cond_met=0 -----
        header("MRS R0,CPSR cond NOT MET");
        apply(32'hE10F0000, 0);
        check("t_mrs",  t_mrs,  1);
        check("wr_en1", wr_en1, 0); 
        check("psr_rd", psr_rd, 1); // Read is non-destructive, so it should still read the PSR value even if condition is not met.

        // ----- SWI with cond_met=0 -----
        header("SWI cond NOT MET");
        apply(32'hEF123456, 0);
        check("t_swi",  t_swi,  1);
        check("swi_en", swi_en, 0);

        // ----- BDT with empty register list (cond met) -----
        header("BDT empty register list (E8AD0000) - edge case");
        apply(32'hE8AD0000, 1);
        check("t_bdt",          t_bdt,          1);
        check("bdt_list",       bdt_list,       16'h0000);
        check("is_multi_cycle", is_multi_cycle, 0);

        // ----- BDT full register list -----
        header("BDT full register list (E8ADFFFF)");
        apply(32'hE8ADFFFF, 1);
        check("t_bdt",          t_bdt,          1);
        check("bdt_list",       bdt_list,       16'hFFFF);
        check("is_multi_cycle", is_multi_cycle, 1);
        check("bdt_load",       bdt_load,       0);
        check("bdt_wb",         bdt_wb,         1);

        // ----- NOP: MOV R0, R0 -----
        header("MOV R0,R0 (E1A00000) - NOP-like");
        apply(32'hE1A00000, 1);
        check_one_type;
        check("t_dp_reg",    t_dp_reg,    1);
        check("alu_op",      alu_op,      ALU_MOV);
        check("rd_addr",     rd_addr,     4'd0);
        check("rm_addr",     rm_addr,     4'd0);
        check("shift_amount",shift_amount,5'd0);
        check("wr_en1",      wr_en1,      1);
        check("wr_addr1",    wr_addr1,    4'd0);

        // ----- DP immediate zero rotation -----
        header("MOV R1,#4 (E3A01004) - zero rotation");
        apply(32'hE3A01004, 1);
        check("t_dp_imm", t_dp_imm, 1);
        check("imm32",    imm32,    32'd4);
        check("wr_en1",   wr_en1,   1);

        // ----- B offset=0 (branch to self+8) -----
        header("B offset=0 (EA000000) - branch self");
        apply(32'hEA000000, 1);
        check("t_br",      t_br,      1);
        check("branch_en", branch_en, 1);
        check("imm32",     imm32,     32'h00000000);

        // ----- B max positive offset -----
        // EA7FFFFF: imm24=0x7FFFFF → +0x1FFFFFC
        header("B max positive (EA7FFFFF)");
        apply(32'hEA7FFFFF, 1);
        check("t_br",      t_br,      1);
        check("branch_en", branch_en, 1);
        check("imm32",     imm32,     32'h01FFFFFC);

        // ----- B max negative offset -----
        // EA800000: imm24=0x800000 → -0x2000000
        header("B max negative (EA800000)");
        apply(32'hEA800000, 1);
        check("t_br",      t_br,      1);
        check("branch_en", branch_en, 1);
        check("imm32",     imm32,     32'hFE000000);

        // ========================================================
        //  Print result for the very last test
        // ========================================================
        finish_last_test;

        // ========================================================
        //  Final Summary
        // ========================================================
        $display("\n##################################################");
        $display(" FINAL SUMMARY");
        $display("##################################################");
        $display("  Total tests:       %0d", test_num);
        $display("  Total checks:      %0d", checks);
        $display("  Checks passed:     %0d", passes);
        $display("  Checks FAILED:     %0d", errors);
        $display("##################################################");
        if (errors == 0)
            $display("  >>> ALL %0d CHECKS PASSED <<<", checks);
        else
            $display("  >>> %0d FAILURE(S) DETECTED <<<", errors);
        $display("##################################################\n");

        $finish;
    end

endmodule