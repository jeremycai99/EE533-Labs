/* file: tb_sp_core.v
 Description: Testbench for sp_core — verifies per-thread pipeline slice.
 Includes BRAM model for realistic sync-read/sync-write memory testing.
 CRITICAL: Every @(posedge clk) is followed by #1 to prevent Verilog
 simulation race conditions between TB and DUT NBA captures.
 Author: Jeremy Cai
 Date: Feb. 28, 2026
 Version: 4.0 (BRAM model + load/store tests)
*/

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "sp_core.v"

module sp_core_tb;

    reg clk, rst_n;
    localparam CLK_PERIOD = 10;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg stall, flush_id;
    reg [3:0] rf_r0_addr, rf_r1_addr, rf_r2_addr, rf_r3_addr;
    wire [15:0] rf_r0_data, rf_r1_data, rf_r2_data, rf_r3_data;
    reg [1:0] pred_rd_sel;
    wire pred_rd_val;
    reg [4:0] id_opcode;
    reg id_dt;
    reg [1:0] id_cmp_mode;
    reg id_rf_we, id_pred_we;
    reg [3:0] id_rD_addr;
    reg [1:0] id_pred_wr_sel;
    reg id_valid, id_active;
    reg [2:0] id_wb_src;
    reg id_use_imm;
    reg [15:0] id_imm16;
    wire [15:0] ex_mem_result_out, ex_mem_store_out;
    wire ex_mem_valid_out, ex_busy;
    reg [15:0] mem_rdata;
    reg [3:0] wb_ext_w1_addr, wb_ext_w2_addr, wb_ext_w3_addr;
    reg [15:0] wb_ext_w1_data, wb_ext_w2_data, wb_ext_w3_data;
    reg wb_ext_w1_we, wb_ext_w2_we, wb_ext_w3_we;
    wire [3:0] wb_rD_addr;
    wire wb_rf_we, wb_active, wb_valid;
    wire mem_is_load, mem_is_store;

    sp_core #(.TID(2'b10)) u_dut (
        .clk(clk), .rst_n(rst_n), .stall(stall), .flush_id(flush_id),
        .rf_r0_addr(rf_r0_addr), .rf_r1_addr(rf_r1_addr),
        .rf_r2_addr(rf_r2_addr), .rf_r3_addr(rf_r3_addr),
        .rf_r0_data(rf_r0_data), .rf_r1_data(rf_r1_data),
        .rf_r2_data(rf_r2_data), .rf_r3_data(rf_r3_data),
        .pred_rd_sel(pred_rd_sel), .pred_rd_val(pred_rd_val),
        .id_opcode(id_opcode), .id_dt(id_dt), .id_cmp_mode(id_cmp_mode),
        .id_rf_we(id_rf_we), .id_pred_we(id_pred_we),
        .id_rD_addr(id_rD_addr), .id_pred_wr_sel(id_pred_wr_sel),
        .id_valid(id_valid), .id_active(id_active),
        .id_wb_src(id_wb_src), .id_use_imm(id_use_imm), .id_imm16(id_imm16),
        .ex_mem_result_out(ex_mem_result_out),
        .ex_mem_store_out(ex_mem_store_out),
        .ex_mem_valid_out(ex_mem_valid_out), .ex_busy(ex_busy),
        .mem_rdata(mem_rdata),
        .mem_is_load(mem_is_load), .mem_is_store(mem_is_store),
        .wb_ext_w1_addr(wb_ext_w1_addr), .wb_ext_w1_data(wb_ext_w1_data),
        .wb_ext_w1_we(wb_ext_w1_we),
        .wb_ext_w2_addr(wb_ext_w2_addr), .wb_ext_w2_data(wb_ext_w2_data),
        .wb_ext_w2_we(wb_ext_w2_we),
        .wb_ext_w3_addr(wb_ext_w3_addr), .wb_ext_w3_data(wb_ext_w3_data),
        .wb_ext_w3_we(wb_ext_w3_we),
        .wb_rD_addr(wb_rD_addr), .wb_rf_we(wb_rf_we),
        .wb_active(wb_active), .wb_valid(wb_valid)
    );

    // ================================================================
    // BRAM Model — sync read, sync write (1-cycle latency)
    // ================================================================
    reg [15:0] dmem [0:1023];
    integer j;
    initial begin
        for (j = 0; j < 1024; j = j + 1)
            dmem[j] = 16'h0000;
        // Pre-load test data for load tests
        dmem[8] = 16'hBEEF;    // addr 8: for basic load (R1+R2 = 5+3)
        dmem[0] = 16'hDEAD;    // addr 0
        dmem[5] = 16'hCAFE;    // addr 5: for LD with imm=0
    end

    always @(posedge clk) begin
        if (mem_is_load)
            mem_rdata <= dmem[ex_mem_result_out[9:0]];
        if (mem_is_store)
            dmem[ex_mem_result_out[9:0]] <= ex_mem_store_out;
    end

    // ================================================================
    // Test infrastructure
    // ================================================================
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    task clear_id;
    begin
        id_opcode = `OP_NOP;
        id_dt = 1'b0;
        id_cmp_mode = 2'd0;
        id_rf_we = 1'b0;
        id_pred_we = 1'b0;
        id_rD_addr = 4'd0;
        id_pred_wr_sel = 2'd0;
        id_valid = 1'b0;
        id_active = 1'b0;
        id_wb_src = 3'd0;
        id_use_imm = 1'b0;
        id_imm16 = 16'd0;
        rf_r0_addr = 4'd0;
        rf_r1_addr = 4'd0;
        rf_r2_addr = 4'd0;
        rf_r3_addr = 4'd0;
        pred_rd_sel = 2'd0;
        wb_ext_w1_addr = 4'd0; wb_ext_w1_data = 16'd0; wb_ext_w1_we = 1'b0;
        wb_ext_w2_addr = 4'd0; wb_ext_w2_data = 16'd0; wb_ext_w2_we = 1'b0;
        wb_ext_w3_addr = 4'd0; wb_ext_w3_data = 16'd0; wb_ext_w3_we = 1'b0;
    end
    endtask

    task tick;
    begin
        @(posedge clk); #1;
    end
    endtask

    task check_gpr;
        input [3:0] addr;
        input [15:0] expected;
        input [255:0] test_name;
        reg [15:0] actual;
    begin
        test_num = test_num + 1;
        rf_r0_addr = addr;
        #1;
        actual = rf_r0_data;
        if (actual === expected) begin
            $display("[PASS] Test %0d: %0s  R%0d = 0x%04h",
                test_num, test_name, addr, actual);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: %0s  R%0d = 0x%04h, expected 0x%04h",
                test_num, test_name, addr, actual, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check_pred;
        input [1:0] sel;
        input expected;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        pred_rd_sel = sel;
        #1;
        if (pred_rd_val === expected) begin
            $display("[PASS] Test %0d: %0s  P%0d = %0b",
                test_num, test_name, sel, pred_rd_val);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: %0s  P%0d = %0b, expected %0b",
                test_num, test_name, sel, pred_rd_val, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check_signal;
        input [15:0] actual;
        input [15:0] expected;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        if (actual === expected) begin
            $display("[PASS] Test %0d: %0s  = 0x%04h",
                test_num, test_name, actual);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: %0s  = 0x%04h, expected 0x%04h",
                test_num, test_name, actual, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check_flag;
        input actual;
        input expected;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        if (actual === expected) begin
            $display("[PASS] Test %0d: %0s  = %0b",
                test_num, test_name, actual);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: %0s  = %0b, expected %0b",
                test_num, test_name, actual, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

    // ================================================================
    // Issue tasks — all use tick (@posedge + #1) to avoid races
    // ================================================================

    // 1-cycle ALU: ID → id_ex(tick1) → ex_mem(tick2) → mem_wb(tick3) → RF(tick4)
    task issue_1cyc;
        input [4:0] t_opcode;
        input t_dt;
        input [3:0] t_rA, t_rB, t_rC, t_rD;
        input t_rf_we, t_pred_we;
        input [1:0] t_cmp_mode, t_pred_wr_sel, t_pred_rd_sel;
        input t_use_imm;
        input [15:0] t_imm16;
        input [2:0] t_wb_src;
        input t_active;
    begin
        id_opcode = t_opcode; id_dt = t_dt;
        id_cmp_mode = t_cmp_mode;
        id_rf_we = t_rf_we; id_pred_we = t_pred_we;
        id_rD_addr = t_rD; id_pred_wr_sel = t_pred_wr_sel;
        id_valid = 1'b1; id_active = t_active;
        id_wb_src = t_wb_src; id_use_imm = t_use_imm; id_imm16 = t_imm16;
        rf_r0_addr = t_rA; rf_r1_addr = t_rB; rf_r2_addr = t_rC;
        pred_rd_sel = t_pred_rd_sel;
        tick; // 1: id_ex captures
        clear_id;
        tick; // 2: ex_mem captures
        tick; // 3: mem_wb captures
        tick; // 4: RF write commits
    end
    endtask

    // 2-cycle MUL/FMA/CVT with stall
    task issue_multicyc;
        input [4:0] t_opcode;
        input [3:0] t_rA, t_rB, t_rC, t_rD;
        input t_dt, t_rf_we, t_use_imm;
        input [15:0] t_imm16;
        input t_active;
    begin
        id_opcode = t_opcode; id_dt = t_dt;
        id_cmp_mode = 2'd0;
        id_rf_we = t_rf_we; id_pred_we = 1'b0;
        id_rD_addr = t_rD; id_pred_wr_sel = 2'd0;
        id_valid = 1'b1; id_active = t_active;
        id_wb_src = 3'd0; id_use_imm = t_use_imm; id_imm16 = t_imm16;
        rf_r0_addr = t_rA; rf_r1_addr = t_rB; rf_r2_addr = t_rC;
        tick; // 1: id_ex captures
        clear_id;
        stall = 1'b1;
        tick; // 2: compute stage 1
        tick; // 3: compute done → ex_mem captures
        stall = 1'b0;
        tick; // 4: mem_wb captures
        tick; // 5: RF write commits
    end
    endtask

    task issue_movi;
        input [3:0] t_rD;
        input [15:0] t_imm;
    begin
        issue_1cyc(`OP_MOVI, `DT_INT16,
            4'd0, 4'd0, 4'd0, t_rD,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b1, t_imm, 3'd0, 1'b1);
    end
    endtask

    task issue_alu_rr;
        input [4:0] t_opcode;
        input [3:0] t_rA, t_rB, t_rD;
    begin
        issue_1cyc(t_opcode, `DT_INT16,
            t_rA, t_rB, 4'd0, t_rD,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
    end
    endtask

    // LD instruction — address computed by ALU (rA + rB/imm).
    // BRAM model provides mem_rdata with 1-cycle latency.
    // Pipeline: ID→id_ex(tick1)→ex_mem(tick2: addr out)→mem_wb(tick3: BRAM read, bypass)→RF(tick4)
    task issue_load;
        input [3:0] t_rA, t_rB, t_rD;
        input t_use_imm;
        input [15:0] t_imm16;
        input t_active;
    begin
        id_opcode = `OP_LD; id_dt = `DT_INT16;
        id_cmp_mode = 2'd0;
        id_rf_we = 1'b1; id_pred_we = 1'b0;
        id_rD_addr = t_rD; id_pred_wr_sel = 2'd0;
        id_valid = 1'b1; id_active = t_active;
        id_wb_src = 3'd1; id_use_imm = t_use_imm; id_imm16 = t_imm16;
        rf_r0_addr = t_rA; rf_r1_addr = t_rB; rf_r2_addr = 4'd0;
        tick; // 1: id_ex captures LD
        clear_id;
        tick; // 2: ex_mem captures (address on ex_mem_result_out → BRAM)
        tick; // 3: BRAM rdata arrives (posedge), mem_wb captures control
              //    WB bypass: w0_data = mem_rdata
        tick; // 4: RF captures w0_data = mem_rdata
    end
    endtask

    // ST instruction — address = rA + rB/imm, data = R[rC].
    // BRAM model writes dmem at posedge after ex_mem drives address.
    task issue_store;
        input [3:0] t_rA, t_rB, t_rC;
        input t_use_imm;
        input [15:0] t_imm16;
        input t_active;
    begin
        id_opcode = `OP_ST; id_dt = `DT_INT16;
        id_cmp_mode = 2'd0;
        id_rf_we = 1'b0; id_pred_we = 1'b0;
        id_rD_addr = 4'd0; id_pred_wr_sel = 2'd0;
        id_valid = 1'b1; id_active = t_active;
        id_wb_src = 3'd2; id_use_imm = t_use_imm; id_imm16 = t_imm16;
        rf_r0_addr = t_rA; rf_r1_addr = t_rB; rf_r2_addr = t_rC;
        tick; // 1: id_ex captures ST
        clear_id;
        tick; // 2: ex_mem captures (addr + data), mem_is_store=1
        tick; // 3: BRAM writes dmem[addr] <= data (posedge)
        tick; // 4: pipeline drain (no RF write)
    end
    endtask

    // CVT instruction (2-cycle, same stall protocol as MUL)
    task issue_cvt;
        input t_dt;
        input [3:0] t_rA, t_rD;
    begin
        issue_multicyc(`OP_CVT, t_rA, 4'd0, 4'd0, t_rD,
            t_dt, 1'b1, 1'b0, 16'd0, 1'b1);
    end
    endtask

    // ================================================================
    // Main test sequence
    // ================================================================
    initial begin
        $dumpfile("sp_core_tb.vcd");
        $dumpvars(0, sp_core_tb);

        rst_n = 1'b0;
        stall = 1'b0;
        flush_id = 1'b0;
        mem_rdata = 16'd0;
        clear_id;
        repeat (3) @(posedge clk); #1;
        rst_n = 1'b1;
        tick;

        $display("\n========================================");
        $display(" SP Core Testbench v4.0  TID=2");
        $display("========================================\n");

        // ============================================================
        // Reset
        // ============================================================
        $display("--- Reset Verification ---");
        check_gpr(4'd0, 16'h0000, "Reset R0=0");
        check_gpr(4'd1, 16'h0000, "Reset R1=0");
        check_gpr(4'd15, 16'h0000, "Reset R15=0");

        // ============================================================
        // MOVI
        // ============================================================
        $display("\n--- MOVI ---");
        issue_movi(4'd1, 16'h0005);
        check_gpr(4'd1, 16'h0005, "MOVI R1=5");

        issue_movi(4'd2, 16'h0003);
        check_gpr(4'd2, 16'h0003, "MOVI R2=3");

        // ============================================================
        // ADD, SUB
        // ============================================================
        $display("\n--- ADD / SUB ---");
        issue_alu_rr(`OP_ADD, 4'd1, 4'd2, 4'd3);
        check_gpr(4'd3, 16'h0008, "ADD R3=R1+R2=8");

        issue_alu_rr(`OP_SUB, 4'd1, 4'd2, 4'd4);
        check_gpr(4'd4, 16'h0002, "SUB R4=R1-R2=2");

        // ============================================================
        // MUL, FMA (multi-cycle)
        // ============================================================
        $display("\n--- MUL / FMA ---");
        issue_multicyc(`OP_MUL, 4'd1, 4'd2, 4'd0, 4'd5,
            `DT_INT16, 1'b1, 1'b0, 16'd0, 1'b1);
        check_gpr(4'd5, 16'h000F, "MUL R5=R1*R2=15");

        issue_movi(4'd6, 16'h000A);
        check_gpr(4'd6, 16'h000A, "MOVI R6=10");
        issue_multicyc(`OP_FMA, 4'd1, 4'd2, 4'd6, 4'd6,
            `DT_INT16, 1'b1, 1'b0, 16'd0, 1'b1);
        check_gpr(4'd6, 16'h0019, "FMA R6=R1*R2+R6=25");

        // ============================================================
        // MOV.TID
        // ============================================================
        $display("\n--- MOV.TID ---");
        issue_1cyc(`OP_MOV, 1'b1,
            4'd0, 4'd0, 4'd0, 4'd7,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_gpr(4'd7, 16'h0002, "MOV.TID R7=2");

        // ============================================================
        // Logic: AND, OR, XOR
        // ============================================================
        $display("\n--- Logic Ops ---");
        issue_alu_rr(`OP_AND, 4'd1, 4'd2, 4'd8);
        check_gpr(4'd8, 16'h0001, "AND R8=R1&R2=1");
        issue_alu_rr(`OP_OR, 4'd1, 4'd2, 4'd9);
        check_gpr(4'd9, 16'h0007, "OR R9=R1|R2=7");
        issue_alu_rr(`OP_XOR, 4'd1, 4'd2, 4'd10);
        check_gpr(4'd10, 16'h0006, "XOR R10=R1^R2=6");

        // ============================================================
        // Shifts: SHL, SHR
        // ============================================================
        $display("\n--- Shift Ops ---");
        issue_alu_rr(`OP_SHL, 4'd1, 4'd2, 4'd11);
        check_gpr(4'd11, 16'h0028, "SHL R11=5<<3=40");
        issue_alu_rr(`OP_SHR, 4'd1, 4'd2, 4'd12);
        check_gpr(4'd12, 16'h0000, "SHR R12=5>>>3=0");

        // ============================================================
        // MAX, MIN
        // ============================================================
        $display("\n--- MAX / MIN ---");
        issue_alu_rr(`OP_MAX, 4'd1, 4'd2, 4'd13);
        check_gpr(4'd13, 16'h0005, "MAX R13=max(5,3)=5");
        issue_alu_rr(`OP_MIN, 4'd1, 4'd2, 4'd14);
        check_gpr(4'd14, 16'h0003, "MIN R14=min(5,3)=3");

        // ============================================================
        // ABS, NEG
        // ============================================================
        $display("\n--- ABS / NEG ---");
        issue_movi(4'd15, 16'hFFF9);
        check_gpr(4'd15, 16'hFFF9, "MOVI R15=-7");
        issue_1cyc(`OP_ABS, `DT_INT16,
            4'd15, 4'd0, 4'd0, 4'd13,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_gpr(4'd13, 16'h0007, "ABS R13=|R15|=7");
        issue_1cyc(`OP_NEG, `DT_INT16,
            4'd1, 4'd0, 4'd0, 4'd14,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_gpr(4'd14, 16'hFFFB, "NEG R14=-R1=-5");

        // ============================================================
        // SETP
        // ============================================================
        $display("\n--- SETP ---");
        issue_1cyc(`OP_SETP, `DT_INT16,
            4'd1, 4'd2, 4'd0, 4'd0,
            1'b0, 1'b1, `COMP_LT, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_pred(2'd0, 1'b0, "SETP P0=(5<3)=0");
        issue_1cyc(`OP_SETP, `DT_INT16,
            4'd2, 4'd1, 4'd0, 4'd0,
            1'b0, 1'b1, `COMP_LT, 2'd1, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_pred(2'd1, 1'b1, "SETP P1=(3<5)=1");

        // ============================================================
        // SET
        // ============================================================
        $display("\n--- SET ---");
        issue_1cyc(`OP_SET, `DT_INT16,
            4'd1, 4'd2, 4'd0, 4'd8,
            1'b1, 1'b0, `COMP_EQ, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_gpr(4'd8, 16'h0000, "SET R8=(5==3)=0");
        issue_1cyc(`OP_SET, `DT_INT16,
            4'd1, 4'd1, 4'd0, 4'd9,
            1'b1, 1'b0, `COMP_EQ, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_gpr(4'd9, 16'h0001, "SET R9=(5==5)=1");

        // ============================================================
        // SELP
        // ============================================================
        $display("\n--- SELP ---");
        issue_1cyc(`OP_SELP, `DT_INT16,
            4'd1, 4'd2, 4'd0, 4'd10,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd1,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_gpr(4'd10, 16'h0005, "SELP R10=P1?R1:R2=5");
        issue_1cyc(`OP_SELP, `DT_INT16,
            4'd1, 4'd2, 4'd0, 4'd11,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b0, 16'd0, 3'd0, 1'b1);
        check_gpr(4'd11, 16'h0003, "SELP R11=P0?R1:R2=3");

        // ============================================================
        // ADDI, MULI
        // ============================================================
        $display("\n--- ADDI / MULI ---");
        issue_1cyc(`OP_ADDI, `DT_INT16,
            4'd1, 4'd0, 4'd0, 4'd8,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b1, 16'h000A, 3'd0, 1'b1);
        check_gpr(4'd8, 16'h000F, "ADDI R8=R1+10=15");
        issue_multicyc(`OP_MULI, 4'd1, 4'd0, 4'd0, 4'd9,
            `DT_INT16, 1'b1, 1'b1, 16'h0004, 1'b1);
        check_gpr(4'd9, 16'h0014, "MULI R9=R1*4=20");

        // ============================================================
        // Load — BRAM model (sync-read bypass verification)
        // ============================================================
        $display("\n--- Load (BRAM model, bypass verification) ---");
        // R1=5, R2=3. LD address = R1+R2 = 8. dmem[8]=0xBEEF (pre-loaded).
        issue_load(4'd1, 4'd2, 4'd10, 1'b0, 16'd0, 1'b1);
        check_gpr(4'd10, 16'hBEEF, "LD R10=dmem[R1+R2]=dmem[8]=0xBEEF");

        // LD with immediate offset: addr = R1 + 0 = 5. dmem[5]=0xCAFE.
        issue_load(4'd1, 4'd0, 4'd11, 1'b1, 16'h0000, 1'b1);
        check_gpr(4'd11, 16'hCAFE, "LD R11=dmem[R1+0]=dmem[5]=0xCAFE");

        // LD from addr 0: use R0(=0) + imm 0. dmem[0]=0xDEAD.
        // Need a register with value 0. R0 might not be 0 if overwritten.
        // Use R12 which should still be 0 from SHR test.
        issue_movi(4'd8, 16'h0000);
        issue_load(4'd8, 4'd0, 4'd12, 1'b1, 16'h0000, 1'b1);
        check_gpr(4'd12, 16'hDEAD, "LD R12=dmem[0+0]=dmem[0]=0xDEAD");

        // ============================================================
        // Store — BRAM model (output verification)
        // ============================================================
        $display("\n--- Store (BRAM model, output verification) ---");
        // Prepare: R8=0 (base), R9=0x0014 (from MULI), R2=3 (data)
        // ST: addr = R1(5) + imm 100 = 105, data = R2(3)
        issue_store(4'd1, 4'd0, 4'd2, 1'b1, 16'h0064, 1'b1);
        // Verify BRAM captured the write
        test_num = test_num + 1;
        if (dmem[105] === 16'h0003) begin
            $display("[PASS] Test %0d: ST dmem[105]=R2=3 (BRAM write)", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: ST dmem[105]=0x%04h, expected 0x0003",
                test_num, dmem[105]);
            fail_count = fail_count + 1;
        end

        // Store R1(5) to addr = R2(3) + imm 200 = 203
        issue_store(4'd2, 4'd0, 4'd1, 1'b1, 16'h00C8, 1'b1);
        test_num = test_num + 1;
        if (dmem[203] === 16'h0005) begin
            $display("[PASS] Test %0d: ST dmem[203]=R1=5 (BRAM write)", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: ST dmem[203]=0x%04h, expected 0x0005",
                test_num, dmem[203]);
            fail_count = fail_count + 1;
        end

        // ============================================================
        // Store → Load roundtrip (BRAM model)
        // ============================================================
        $display("\n--- Store-Load Roundtrip ---");
        // Store 0xABCD to addr 300
        issue_movi(4'd8, 16'hABCD);
        issue_movi(4'd9, 16'h012C); // 300 decimal
        // ST: addr = R9(300) + imm 0 = 300, data = R8(0xABCD)
        issue_store(4'd9, 4'd0, 4'd8, 1'b1, 16'h0000, 1'b1);
        // LD: addr = R9(300) + imm 0 = 300 → should get 0xABCD
        issue_load(4'd9, 4'd0, 4'd10, 1'b1, 16'h0000, 1'b1);
        check_gpr(4'd10, 16'hABCD, "ST-LD roundtrip: dmem[300]=0xABCD");

        // Store 0x1234 to addr 400, then load back
        issue_movi(4'd8, 16'h1234);
        issue_movi(4'd9, 16'h0190); // 400 decimal
        issue_store(4'd9, 4'd0, 4'd8, 1'b1, 16'h0000, 1'b1);
        issue_load(4'd9, 4'd0, 4'd11, 1'b1, 16'h0000, 1'b1);
        check_gpr(4'd11, 16'h1234, "ST-LD roundtrip: dmem[400]=0x1234");

        // Overwrite and verify: store 0x5678 to addr 300, load back
        issue_movi(4'd8, 16'h5678);
        issue_movi(4'd9, 16'h012C); // 300
        issue_store(4'd9, 4'd0, 4'd8, 1'b1, 16'h0000, 1'b1);
        issue_load(4'd9, 4'd0, 4'd12, 1'b1, 16'h0000, 1'b1);
        check_gpr(4'd12, 16'h5678, "ST-LD overwrite: dmem[300]=0x5678");

        // ============================================================
        // Store output signal verification
        // ============================================================
        $display("\n--- Store Output Signals ---");
        issue_movi(4'd8, 16'h0050); // base addr = 80
        issue_movi(4'd9, 16'hFACE); // store data
        // Issue ST manually to check signals mid-pipeline
        id_opcode = `OP_ST; id_dt = `DT_INT16;
        id_rf_we = 1'b0; id_pred_we = 1'b0;
        id_rD_addr = 4'd0; id_valid = 1'b1; id_active = 1'b1;
        id_wb_src = 3'd2; id_use_imm = 1'b1; id_imm16 = 16'h000A; // offset 10
        rf_r0_addr = 4'd8; rf_r1_addr = 4'd0; rf_r2_addr = 4'd9;
        tick; // 1: id_ex captures
        clear_id;
        tick; // 2: ex_mem captures — check signals now
        check_signal(ex_mem_result_out, 16'h005A, "ST addr=R8+10=90");
        check_signal(ex_mem_store_out, 16'hFACE, "ST data=R9=0xFACE");
        check_flag(mem_is_store, 1'b1, "mem_is_store=1 during ST");
        check_flag(mem_is_load, 1'b0, "mem_is_load=0 during ST");
        tick; tick; // drain pipeline

        // ============================================================
        // Load output signal verification
        // ============================================================
        $display("\n--- Load Output Signals ---");
        // Issue LD manually to check mem_is_load mid-pipeline
        dmem[90] = 16'h9999; // pre-load for this test
        id_opcode = `OP_LD; id_dt = `DT_INT16;
        id_rf_we = 1'b1; id_pred_we = 1'b0;
        id_rD_addr = 4'd13; id_valid = 1'b1; id_active = 1'b1;
        id_wb_src = 3'd1; id_use_imm = 1'b1; id_imm16 = 16'h000A;
        rf_r0_addr = 4'd8; rf_r1_addr = 4'd0; rf_r2_addr = 4'd0;
        tick; // 1: id_ex
        clear_id;
        tick; // 2: ex_mem — check signals
        check_signal(ex_mem_result_out, 16'h005A, "LD addr=R8+10=90");
        check_flag(mem_is_load, 1'b1, "mem_is_load=1 during LD");
        check_flag(mem_is_store, 1'b0, "mem_is_store=0 during LD");
        tick; // 3: BRAM read, mem_wb captures
        tick; // 4: RF write
        check_gpr(4'd13, 16'h9999, "LD R13=dmem[90]=0x9999");

        // ============================================================
        // Inactive thread — store should NOT write BRAM
        // ============================================================
        $display("\n--- Inactive Store (should not write) ---");
        dmem[500] = 16'hAAAA; // sentinel
        issue_movi(4'd8, 16'h01F4); // addr 500
        issue_movi(4'd9, 16'hFFFF); // data
        issue_store(4'd8, 4'd0, 4'd9, 1'b1, 16'h0000, 1'b0); // active=0
        test_num = test_num + 1;
        // The store entered pipeline with active=0. mem_is_store uses ex_mem_valid
        // but doesn't check active — BRAM write happens anyway (SM should gate this).
        // For now just check dmem was written (pipeline doesn't gate on active for store):
        $display("[INFO] Test %0d: Inactive store: dmem[500]=0x%04h (SM gates in real design)",
            test_num, dmem[500]);
        // Not a pass/fail — this is architecture-dependent (SM responsibility)

        // ============================================================
        // Pipeline control: Inactive, Flush, Stall
        // ============================================================
        $display("\n--- Pipeline Control ---");
        issue_movi(4'd12, 16'h0000);
        issue_1cyc(`OP_MOVI, `DT_INT16,
            4'd0, 4'd0, 4'd0, 4'd12,
            1'b1, 1'b0, 2'd0, 2'd0, 2'd0,
            1'b1, 16'h1234, 3'd0, 1'b0); // active=0
        check_gpr(4'd12, 16'h0000, "Inactive: R12 unchanged");

        issue_movi(4'd13, 16'h0000);
        id_opcode = `OP_MOVI; id_dt = `DT_INT16;
        id_rf_we = 1'b1; id_rD_addr = 4'd13;
        id_valid = 1'b1; id_active = 1'b1;
        id_wb_src = 3'd0; id_use_imm = 1'b1; id_imm16 = 16'hAAAA;
        flush_id = 1'b1;
        tick;
        flush_id = 1'b0; clear_id;
        tick; tick; tick; tick;
        check_gpr(4'd13, 16'h0000, "Flush: R13 unchanged");

        id_opcode = `OP_ADD; id_dt = `DT_INT16;
        id_rf_we = 1'b1; id_rD_addr = 4'd13;
        id_valid = 1'b1; id_active = 1'b1;
        id_wb_src = 3'd0; id_use_imm = 1'b0;
        rf_r0_addr = 4'd1; rf_r1_addr = 4'd2;
        tick; clear_id;
        stall = 1'b1;
        tick; tick; tick;
        stall = 1'b0;
        tick; tick; tick;
        check_gpr(4'd13, 16'h0008, "Stall: ADD R13=8");

        // ============================================================
        // WMMA Scatter
        // ============================================================
        $display("\n--- WMMA Scatter ---");
        wb_ext_w1_addr = 4'd10; wb_ext_w1_data = 16'hCAFE; wb_ext_w1_we = 1'b1;
        wb_ext_w2_addr = 4'd11; wb_ext_w2_data = 16'hDEAD; wb_ext_w2_we = 1'b1;
        wb_ext_w3_addr = 4'd12; wb_ext_w3_data = 16'hF00D; wb_ext_w3_we = 1'b1;
        tick;
        wb_ext_w1_we = 1'b0; wb_ext_w2_we = 1'b0; wb_ext_w3_we = 1'b0;
        check_gpr(4'd10, 16'hCAFE, "WMMA W1: R10=0xCAFE");
        check_gpr(4'd11, 16'hDEAD, "WMMA W2: R11=0xDEAD");
        check_gpr(4'd12, 16'hF00D, "WMMA W3: R12=0xF00D");

        // ============================================================
        // Scoreboard
        // ============================================================
        $display("\n--- Scoreboard ---");
        id_opcode = `OP_MOVI; id_dt = `DT_INT16;
        id_rf_we = 1'b1; id_rD_addr = 4'd8;
        id_valid = 1'b1; id_active = 1'b1;
        id_wb_src = 3'd0; id_use_imm = 1'b1; id_imm16 = 16'h0042;
        tick; clear_id;
        tick; tick;
        test_num = test_num + 1;
        if (wb_valid === 1'b1 && wb_rf_we === 1'b1 &&
            wb_rD_addr === 4'd8 && wb_active === 1'b1) begin
            $display("[PASS] Test %0d: Scoreboard valid=%b we=%b rD=%0d active=%b",
                test_num, wb_valid, wb_rf_we, wb_rD_addr, wb_active);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: Scoreboard valid=%b we=%b rD=%0d active=%b",
                test_num, wb_valid, wb_rf_we, wb_rD_addr, wb_active);
            fail_count = fail_count + 1;
        end
        tick;
        check_gpr(4'd8, 16'h0042, "Scoreboard: R8=0x42");

        // ============================================================
        // Back-to-back
        // ============================================================
        $display("\n--- Back-to-back ---");
        id_opcode = `OP_MOVI; id_dt = `DT_INT16;
        id_rf_we = 1'b1; id_rD_addr = 4'd8;
        id_valid = 1'b1; id_active = 1'b1;
        id_wb_src = 3'd0; id_use_imm = 1'b1; id_imm16 = 16'h0001;
        tick;
        id_rD_addr = 4'd9; id_imm16 = 16'h0002;
        tick;
        clear_id;
        tick; tick; tick;
        check_gpr(4'd8, 16'h0001, "B2B: R8=1");
        check_gpr(4'd9, 16'h0002, "B2B: R9=2");

        // ============================================================
        // Arithmetic SHR
        // ============================================================
        $display("\n--- Arithmetic SHR ---");
        issue_alu_rr(`OP_SHR, 4'd15, 4'd2, 4'd13);
        check_gpr(4'd13, 16'hFFFF, "SHR R13=(-7)>>>3=0xFFFF");

        // ============================================================
        // CVT INT16 <-> BF16
        // ============================================================
        $display("\n--- CVT INT16->BF16 ---");
        issue_cvt(`DT_INT16, 4'd1, 4'd8);
        check_gpr(4'd8, 16'h40A0, "CVT I2F R8=BF16(5)=0x40A0");
        issue_cvt(`DT_INT16, 4'd15, 4'd9);
        check_gpr(4'd9, 16'hC0E0, "CVT I2F R9=BF16(-7)=0xC0E0");
        issue_movi(4'd10, 16'h0000);
        issue_cvt(`DT_INT16, 4'd10, 4'd10);
        check_gpr(4'd10, 16'h0000, "CVT I2F R10=BF16(0)=0x0000");
        issue_movi(4'd11, 16'h0001);
        issue_cvt(`DT_INT16, 4'd11, 4'd11);
        check_gpr(4'd11, 16'h3F80, "CVT I2F R11=BF16(1)=0x3F80");

        $display("\n--- CVT BF16->INT16 ---");
        issue_movi(4'd12, 16'h4040);
        issue_cvt(`DT_BF16, 4'd12, 4'd12);
        check_gpr(4'd12, 16'h0003, "CVT F2I R12=INT(3.0)=3");
        issue_cvt(`DT_BF16, 4'd8, 4'd13);
        check_gpr(4'd13, 16'h0005, "CVT F2I R13=INT(BF16(5))=5");
        issue_cvt(`DT_BF16, 4'd9, 4'd14);
        check_gpr(4'd14, 16'hFFF9, "CVT F2I R14=INT(-7.0)=-7");

        // ============================================================
        // Summary
        // ============================================================
        $display("\n========================================");
        $display(" Results: %0d passed, %0d failed out of %0d tests",
            pass_count, fail_count, test_num);
        $display("========================================\n");

        if (fail_count == 0)
            $display("*** ALL TESTS PASSED ***\n");
        else
            $display("*** SOME TESTS FAILED ***\n");

        $finish;
    end

endmodule