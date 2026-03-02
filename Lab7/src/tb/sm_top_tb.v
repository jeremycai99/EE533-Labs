/* file: sm_top_tb.v
 * Testbench for sm_top — streaming multiprocessor top wrapper.
 * Loads programs via ext_imem port B, preloads data via ext_dmem
 * port B, launches kernels, and verifies results via DMEM readback.
 *
 * 6 PTX-mapped kernel tests (K1–K6):
 *   K1: vec_add  int16   C[tid] = A[tid] + B[tid]
 *   K2: vec_sub  int16   C[tid] = A[tid] - B[tid]
 *   K3: bf16_mul bf16    C[tid] = A[tid] * B[tid]
 *   K4: bf16_fma bf16    D[tid] = A[tid]*B[tid] + C[tid]
 *   K5: relu     bf16    out[tid] = max(in[tid], 0)
 *   K6: wmma     bf16    D = A*B + C  (4x4 matmul)
 *
 * CRITICAL: Every @(posedge clk) is followed by #1 to prevent
 * Verilog simulation race conditions between TB and DUT NBAs.
 *
 * Author: Jeremy Cai
 * Date: Mar. 1, 2026
 * Version: 1.1 — DMEM-only verification (debug RF removed)
 */

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "sm_top.v"
`include "test_gpu_imem.v"
`include "test_gpu_dmem.v"

module sm_top_tb;

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
    reg kernel_start;
    reg [`GPU_PC_WIDTH-1:0] kernel_entry_pc;
    wire kernel_done;

    // External IMEM port B
    reg [`GPU_IMEM_ADDR_WIDTH-1:0] ext_imem_addr;
    reg [`GPU_IMEM_DATA_WIDTH-1:0] ext_imem_din;
    reg ext_imem_we;
    wire [`GPU_IMEM_DATA_WIDTH-1:0] ext_imem_dout;

    // External DMEM port B
    reg [1:0] ext_dmem_sel;
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] ext_dmem_addr;
    reg [`GPU_DMEM_DATA_WIDTH-1:0] ext_dmem_din;
    reg ext_dmem_we;
    wire [`GPU_DMEM_DATA_WIDTH-1:0] ext_dmem_dout;

    // ================================================================
    // DUT
    // ================================================================
    sm_top u_dut (
        .clk(clk), .rst_n(rst_n),
        .kernel_start(kernel_start),
        .kernel_entry_pc(kernel_entry_pc),
        .kernel_done(kernel_done),
        .ext_imem_addr(ext_imem_addr),
        .ext_imem_din(ext_imem_din),
        .ext_imem_we(ext_imem_we),
        .ext_imem_dout(ext_imem_dout),
        .ext_dmem_sel(ext_dmem_sel),
        .ext_dmem_addr(ext_dmem_addr),
        .ext_dmem_din(ext_dmem_din),
        .ext_dmem_we(ext_dmem_we),
        .ext_dmem_dout(ext_dmem_dout)
    );

    // ================================================================
    // Instruction Encoding Helpers
    // ================================================================
    function [31:0] enc_r;
        input [4:0] op;
        input dt;
        input [3:0] rd, ra, rb;
        enc_r = {op, dt, 2'b00, rd, ra, rb, 12'd0};
    endfunction

    function [31:0] enc_i;
        input [4:0] op;
        input dt;
        input [3:0] rd, ra;
        input [15:0] imm;
        enc_i = {op, dt, 2'b00, rd, ra, imm};
    endfunction

    function [31:0] enc_movi;
        input [3:0] rd;
        input [15:0] imm;
        enc_movi = {`OP_MOVI, 1'b0, 2'b00, rd, 4'd0, imm};
    endfunction

    function [31:0] enc_m;
        input [4:0] op;
        input [3:0] rd, ra;
        input [15:0] offset;
        enc_m = {op, 1'b0, 2'b00, rd, ra, offset};
    endfunction

    function [31:0] enc_m_f;
        input [4:0] op;
        input [3:0] rd, ra;
        input [15:0] offset;
        enc_m_f = {op, 1'b1, 2'b00, rd, ra, offset};
    endfunction

    function [31:0] enc_mov_tid;
        input [3:0] rd;
        enc_mov_tid = {`OP_MOV, 1'b1, 2'b00, rd, 4'd0, 16'd0};
    endfunction

    function [31:0] enc_wmma_mma;
        input [3:0] rD, rA, rB, rC;
        enc_wmma_mma = {`WMMA_MMA, 1'b1, 2'b00, rD, rA, rB, rC, 8'd0};
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
    begin
        @(posedge clk); #1;
    end
    endtask

    task write_imem;
        input [`GPU_IMEM_ADDR_WIDTH-1:0] addr;
        input [`GPU_IMEM_DATA_WIDTH-1:0] data;
    begin
        ext_imem_addr = addr;
        ext_imem_din = data;
        ext_imem_we = 1;
        tick;
        ext_imem_we = 0;
    end
    endtask

    task write_dmem;
        input [1:0] sp;
        input [`GPU_DMEM_ADDR_WIDTH-1:0] addr;
        input [`GPU_DMEM_DATA_WIDTH-1:0] data;
    begin
        ext_dmem_sel = sp;
        ext_dmem_addr = addr;
        ext_dmem_din = data;
        ext_dmem_we = 1;
        tick;
        ext_dmem_we = 0;
    end
    endtask

    task read_dmem;
        input [1:0] sp;
        input [`GPU_DMEM_ADDR_WIDTH-1:0] addr;
        output [`GPU_DMEM_DATA_WIDTH-1:0] data;
    begin
        ext_dmem_sel = sp;
        ext_dmem_addr = addr;
        ext_dmem_we = 0;
        tick;
        data = ext_dmem_dout;
    end
    endtask

    reg [31:0] prog [0:255];
    integer prog_len;

    task load_program;
        input integer len;
        integer i;
    begin
        for (i = 0; i < len; i = i + 1)
            write_imem(i, prog[i]);
        for (i = len; i < 256; i = i + 1)
            write_imem(i, INST_NOP);
    end
    endtask

    task reset_dut;
    begin
        rst_n = 0;
        kernel_start = 0;
        kernel_entry_pc = 0;
        ext_imem_we = 0;
        ext_dmem_we = 0;
        ext_dmem_sel = 0;
        repeat (3) tick;
        rst_n = 1;
        tick;
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

    task check_dmem;
        input [1:0] sp;
        input [`GPU_DMEM_ADDR_WIDTH-1:0] addr;
        input [15:0] expected;
        input [80*8-1:0] test_name;
        reg [15:0] actual;
    begin
        test_num = test_num + 1;
        read_dmem(sp, addr, actual);
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

    // ================================================================
    // Waveform Dump
    // ================================================================
    initial begin
        $dumpfile("sm_top_tb.vcd");
        $dumpvars(0, sm_top_tb);
    end

    initial begin
        #1000000;
        $display("[TIMEOUT] Global simulation timeout reached");
        $finish;
    end

    // ================================================================
    // Pipeline Monitor
    // ================================================================
    integer cycle_count;
    reg trace_en;

    initial begin
        cycle_count = 0;
        trace_en = 0;
    end

    reg [8*8-1:0] opname;
    always @(*) begin
        case (u_dut.u_sm_core.dec_opcode)
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

    reg [8*8-1:0] tc_state_name;
    always @(*) begin
        case (u_dut.u_sm_core.u_tc_top.state)
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

    reg [8*8-1:0] bu_state_name;
    always @(*) begin
        case (u_dut.u_sm_core.bu_state)
            3'd0: bu_state_name = "IDLE    ";
            3'd1: bu_state_name = "LD_ADDR ";
            3'd2: bu_state_name = "LD_BEAT ";
            3'd3: bu_state_name = "ST_READ ";
            3'd4: bu_state_name = "ST_BEAT ";
            default: bu_state_name = "???     ";
        endcase
    end

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (trace_en) begin
            $display("  [C%03d] PC=%0d %0s fv=%b | fstl=%b sb=%b exb=%b fl=%b | tc=%0s bu=%0s | done=%b",
                cycle_count,
                u_dut.u_sm_core.u_fetch.pc_reg,
                opname,
                u_dut.u_sm_core.fetch_valid,
                u_dut.u_sm_core.front_stall,
                u_dut.u_sm_core.sb_stall,
                u_dut.u_sm_core.any_ex_busy,
                u_dut.u_sm_core.sp_flush_id,
                tc_state_name,
                bu_state_name,
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
    // Main Test Sequence
    // ================================================================
    initial begin
        $display("============================================");
        $display("  SM Top Testbench — 6 PTX Kernel Tests");
        $display("  (DMEM-only verification)");
        $display("============================================");

        // ── K1: vec_add C[tid]=A[tid]+B[tid] (int16) ───────────────
        begin
            $display("\n--- K1: vec_add int16 ---");
            reset_dut;

            write_dmem(2'd0, 0, 16'd10);
            write_dmem(2'd1, 2, 16'd20);
            write_dmem(2'd2, 4, 16'd30);
            write_dmem(2'd3, 6, 16'd40);
            write_dmem(2'd0, 16, 16'd5);
            write_dmem(2'd1, 18, 16'd15);
            write_dmem(2'd2, 20, 16'd25);
            write_dmem(2'd3, 22, 16'd35);

            prog[0]  = enc_mov_tid(4'd0);
            prog[1]  = enc_movi(4'd1, 16'd0);
            prog[2]  = enc_movi(4'd2, 16'd16);
            prog[3]  = enc_movi(4'd3, 16'd32);
            prog[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            prog[5]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            prog[6]  = enc_m(`OP_LD, 4'd5, 4'd5, 16'd0);
            prog[7]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            prog[8]  = enc_m(`OP_LD, 4'd6, 4'd6, 16'd0);
            prog[9]  = enc_r(`OP_ADD, 1'b0, 4'd7, 4'd6, 4'd5);
            prog[10] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd3, 4'd4);
            prog[11] = enc_m(`OP_ST, 4'd7, 4'd8, 16'd0);
            prog[12] = INST_RET;
            load_program(13);

            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;

            check_dmem(2'd0, 32, 16'd15, "K1 SP0 C[0]=10+5");
            check_dmem(2'd1, 34, 16'd35, "K1 SP1 C[1]=20+15");
            check_dmem(2'd2, 36, 16'd55, "K1 SP2 C[2]=30+25");
            check_dmem(2'd3, 38, 16'd75, "K1 SP3 C[3]=40+35");
        end

        // ── K2: vec_sub C[tid]=A[tid]-B[tid] (int16) ───────────────
        begin
            $display("\n--- K2: vec_sub int16 ---");
            reset_dut;

            write_dmem(2'd0, 0, 16'd100);
            write_dmem(2'd1, 2, 16'd200);
            write_dmem(2'd2, 4, 16'd300);
            write_dmem(2'd3, 6, 16'd400);
            write_dmem(2'd0, 16, 16'd30);
            write_dmem(2'd1, 18, 16'd50);
            write_dmem(2'd2, 20, 16'd100);
            write_dmem(2'd3, 22, 16'd150);

            prog[0]  = enc_mov_tid(4'd0);
            prog[1]  = enc_movi(4'd1, 16'd0);
            prog[2]  = enc_movi(4'd2, 16'd16);
            prog[3]  = enc_movi(4'd3, 16'd32);
            prog[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            prog[5]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            prog[6]  = enc_m(`OP_LD, 4'd5, 4'd5, 16'd0);
            prog[7]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            prog[8]  = enc_m(`OP_LD, 4'd6, 4'd6, 16'd0);
            prog[9]  = enc_r(`OP_SUB, 1'b0, 4'd7, 4'd5, 4'd6);
            prog[10] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd3, 4'd4);
            prog[11] = enc_m(`OP_ST, 4'd7, 4'd8, 16'd0);
            prog[12] = INST_RET;
            load_program(13);

            launch_kernel(32'd0);
            wait_kernel_done(200);

            check_dmem(2'd0, 32, 16'd70,  "K2 SP0 C[0]=100-30");
            check_dmem(2'd1, 34, 16'd150, "K2 SP1 C[1]=200-50");
            check_dmem(2'd2, 36, 16'd200, "K2 SP2 C[2]=300-100");
            check_dmem(2'd3, 38, 16'd250, "K2 SP3 C[3]=400-150");
        end

        // ── K3: bf16_vector_mul C[tid]=A[tid]*B[tid] ───────────────
        begin
            $display("\n--- K3: bf16_vector_mul ---");
            reset_dut;

            write_dmem(2'd0, 0,  16'h4000);
            write_dmem(2'd1, 2,  16'h4000);
            write_dmem(2'd2, 4,  16'h4000);
            write_dmem(2'd3, 6,  16'h4000);
            write_dmem(2'd0, 16, 16'h4040);
            write_dmem(2'd1, 18, 16'h4040);
            write_dmem(2'd2, 20, 16'h4040);
            write_dmem(2'd3, 22, 16'h4040);

            prog[0]  = enc_mov_tid(4'd0);
            prog[1]  = enc_movi(4'd1, 16'd0);
            prog[2]  = enc_movi(4'd2, 16'd16);
            prog[3]  = enc_movi(4'd3, 16'd32);
            prog[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            prog[5]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            prog[6]  = enc_m_f(`OP_LD, 4'd5, 4'd5, 16'd0);
            prog[7]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            prog[8]  = enc_m_f(`OP_LD, 4'd6, 4'd6, 16'd0);
            prog[9]  = enc_r(`OP_MUL, 1'b1, 4'd7, 4'd5, 4'd6);
            prog[10] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd3, 4'd4);
            prog[11] = enc_m_f(`OP_ST, 4'd7, 4'd8, 16'd0);
            prog[12] = INST_RET;
            load_program(13);

            launch_kernel(32'd0);
            wait_kernel_done(200);

            check_dmem(2'd0, 32, 16'h40C0, "K3 SP0 C=2*3=6.0");
            check_dmem(2'd1, 34, 16'h40C0, "K3 SP1 C=2*3=6.0");
            check_dmem(2'd2, 36, 16'h40C0, "K3 SP2 C=2*3=6.0");
            check_dmem(2'd3, 38, 16'h40C0, "K3 SP3 C=2*3=6.0");
        end

        // ── K4: bf16_fma D[tid]=A[tid]*B[tid]+C[tid] ───────────────
        begin
            $display("\n--- K4: bf16_fma ---");
            reset_dut;

            write_dmem(2'd0, 0,  16'h4000);
            write_dmem(2'd1, 2,  16'h4000);
            write_dmem(2'd2, 4,  16'h4000);
            write_dmem(2'd3, 6,  16'h4000);
            write_dmem(2'd0, 16, 16'h4040);
            write_dmem(2'd1, 18, 16'h4040);
            write_dmem(2'd2, 20, 16'h4040);
            write_dmem(2'd3, 22, 16'h4040);
            write_dmem(2'd0, 32, 16'h3F80);
            write_dmem(2'd1, 34, 16'h3F80);
            write_dmem(2'd2, 36, 16'h3F80);
            write_dmem(2'd3, 38, 16'h3F80);

            prog[0]  = enc_mov_tid(4'd0);
            prog[1]  = enc_movi(4'd1, 16'd0);
            prog[2]  = enc_movi(4'd2, 16'd16);
            prog[3]  = enc_movi(4'd3, 16'd32);
            prog[4]  = enc_movi(4'd9, 16'd48);
            prog[5]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            prog[6]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            prog[7]  = enc_m_f(`OP_LD, 4'd5, 4'd5, 16'd0);
            prog[8]  = enc_r(`OP_ADD, 1'b0, 4'd6, 4'd2, 4'd4);
            prog[9]  = enc_m_f(`OP_LD, 4'd6, 4'd6, 16'd0);
            prog[10] = enc_r(`OP_ADD, 1'b0, 4'd7, 4'd3, 4'd4);
            prog[11] = enc_m_f(`OP_LD, 4'd7, 4'd7, 16'd0);
            prog[12] = enc_r(`OP_FMA, 1'b1, 4'd7, 4'd5, 4'd6);
            prog[13] = enc_r(`OP_ADD, 1'b0, 4'd8, 4'd9, 4'd4);
            prog[14] = enc_m_f(`OP_ST, 4'd7, 4'd8, 16'd0);
            prog[15] = INST_RET;
            load_program(16);

            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;

            check_dmem(2'd0, 48, 16'h40E0, "K4 SP0 D=2*3+1=7.0");
            check_dmem(2'd1, 50, 16'h40E0, "K4 SP1 D=2*3+1=7.0");
            check_dmem(2'd2, 52, 16'h40E0, "K4 SP2 D=2*3+1=7.0");
            check_dmem(2'd3, 54, 16'h40E0, "K4 SP3 D=2*3+1=7.0");
        end

        // ── K5: relu out[tid]=max(in[tid],0.0) (bf16) ──────────────
        begin
            $display("\n--- K5: relu bf16 ---");
            reset_dut;

            write_dmem(2'd0, 0, 16'hBF80);
            write_dmem(2'd1, 2, 16'h4000);
            write_dmem(2'd2, 4, 16'hC040);
            write_dmem(2'd3, 6, 16'h40A0);

            prog[0]  = enc_mov_tid(4'd0);
            prog[1]  = enc_movi(4'd1, 16'd0);
            prog[2]  = enc_movi(4'd2, 16'd16);
            prog[3]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd1);
            prog[4]  = enc_r(`OP_ADD, 1'b0, 4'd5, 4'd1, 4'd4);
            prog[5]  = enc_m_f(`OP_LD, 4'd5, 4'd5, 16'd0);
            prog[6]  = enc_movi(4'd8, 16'h0000);
            prog[7]  = enc_r(`OP_MAX, 1'b1, 4'd6, 4'd5, 4'd8);
            prog[8]  = enc_r(`OP_ADD, 1'b0, 4'd7, 4'd2, 4'd4);
            prog[9]  = enc_m_f(`OP_ST, 4'd6, 4'd7, 16'd0);
            prog[10] = INST_RET;
            load_program(11);

            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(200);
            disable_trace;

            check_dmem(2'd0, 16, 16'h0000, "K5 SP0 relu(-1)=0");
            check_dmem(2'd1, 18, 16'h4000, "K5 SP1 relu(2)=2.0");
            check_dmem(2'd2, 20, 16'h0000, "K5 SP2 relu(-3)=0");
            check_dmem(2'd3, 22, 16'h40A0, "K5 SP3 relu(5)=5.0");
        end

        // ── K6: wmma_bf16 4x4 matmul D=A*B+C ───────────────────────
        begin
            $display("\n--- K6: wmma_bf16 4x4 matmul ---");
            reset_dut;

            // A = identity
            write_dmem(2'd0, 0, 16'h3F80);
            write_dmem(2'd0, 1, 16'h0000);
            write_dmem(2'd0, 2, 16'h0000);
            write_dmem(2'd0, 3, 16'h0000);
            write_dmem(2'd1, 8,  16'h0000);
            write_dmem(2'd1, 9,  16'h3F80);
            write_dmem(2'd1, 10, 16'h0000);
            write_dmem(2'd1, 11, 16'h0000);
            write_dmem(2'd2, 16, 16'h0000);
            write_dmem(2'd2, 17, 16'h0000);
            write_dmem(2'd2, 18, 16'h3F80);
            write_dmem(2'd2, 19, 16'h0000);
            write_dmem(2'd3, 24, 16'h0000);
            write_dmem(2'd3, 25, 16'h0000);
            write_dmem(2'd3, 26, 16'h0000);
            write_dmem(2'd3, 27, 16'h3F80);

            // B = 2.0 everywhere
            write_dmem(2'd0, 32, 16'h4000);
            write_dmem(2'd0, 33, 16'h4000);
            write_dmem(2'd0, 34, 16'h4000);
            write_dmem(2'd0, 35, 16'h4000);
            write_dmem(2'd1, 40, 16'h4000);
            write_dmem(2'd1, 41, 16'h4000);
            write_dmem(2'd1, 42, 16'h4000);
            write_dmem(2'd1, 43, 16'h4000);
            write_dmem(2'd2, 48, 16'h4000);
            write_dmem(2'd2, 49, 16'h4000);
            write_dmem(2'd2, 50, 16'h4000);
            write_dmem(2'd2, 51, 16'h4000);
            write_dmem(2'd3, 56, 16'h4000);
            write_dmem(2'd3, 57, 16'h4000);
            write_dmem(2'd3, 58, 16'h4000);
            write_dmem(2'd3, 59, 16'h4000);

            prog[0]  = enc_mov_tid(4'd0);
            prog[1]  = enc_movi(4'd1, 16'd0);
            prog[2]  = enc_movi(4'd2, 16'd32);
            prog[3]  = enc_movi(4'd3, 16'd64);
            prog[4]  = enc_i(`OP_SHL, 1'b0, 4'd4, 4'd0, 16'd3);
            prog[5]  = enc_r(`OP_ADD, 1'b0, 4'd1, 4'd1, 4'd4);
            prog[6]  = enc_r(`OP_ADD, 1'b0, 4'd2, 4'd2, 4'd4);
            prog[7]  = enc_r(`OP_ADD, 1'b0, 4'd3, 4'd3, 4'd4);
            prog[8]  = enc_m_f(`WMMA_LOAD, 4'd4, 4'd1, 16'd0);
            prog[9]  = enc_m_f(`WMMA_LOAD, 4'd8, 4'd2, 16'd0);
            prog[10] = enc_movi(4'd12, 16'h0000);
            prog[11] = enc_movi(4'd13, 16'h0000);
            prog[12] = enc_movi(4'd14, 16'h0000);
            prog[13] = enc_movi(4'd15, 16'h0000);
            prog[14] = enc_wmma_mma(4'd12, 4'd4, 4'd8, 4'd12);
            prog[15] = enc_m_f(`WMMA_STORE, 4'd12, 4'd3, 16'd0);
            prog[16] = INST_RET;
            load_program(17);

            $display("  D = I(4x4) * 2*ones(4x4) + 0 = 2*ones");
            enable_trace;
            launch_kernel(32'd0);
            wait_kernel_done(400);
            disable_trace;

            // D = 2.0 everywhere (0x4000)
            check_dmem(2'd0, 64, 16'h4000, "K6 SP0 D[0][0]=2.0");
            check_dmem(2'd0, 65, 16'h4000, "K6 SP0 D[0][1]=2.0");
            check_dmem(2'd0, 66, 16'h4000, "K6 SP0 D[0][2]=2.0");
            check_dmem(2'd0, 67, 16'h4000, "K6 SP0 D[0][3]=2.0");
            check_dmem(2'd1, 72, 16'h4000, "K6 SP1 D[1][0]=2.0");
            check_dmem(2'd1, 73, 16'h4000, "K6 SP1 D[1][1]=2.0");
            check_dmem(2'd1, 74, 16'h4000, "K6 SP1 D[1][2]=2.0");
            check_dmem(2'd1, 75, 16'h4000, "K6 SP1 D[1][3]=2.0");
            check_dmem(2'd2, 80, 16'h4000, "K6 SP2 D[2][0]=2.0");
            check_dmem(2'd2, 81, 16'h4000, "K6 SP2 D[2][1]=2.0");
            check_dmem(2'd2, 82, 16'h4000, "K6 SP2 D[2][2]=2.0");
            check_dmem(2'd2, 83, 16'h4000, "K6 SP2 D[2][3]=2.0");
            check_dmem(2'd3, 88, 16'h4000, "K6 SP3 D[3][0]=2.0");
            check_dmem(2'd3, 89, 16'h4000, "K6 SP3 D[3][1]=2.0");
            check_dmem(2'd3, 90, 16'h4000, "K6 SP3 D[3][2]=2.0");
            check_dmem(2'd3, 91, 16'h4000, "K6 SP3 D[3][3]=2.0");
        end

        // ── Summary ─────────────────────────────────────────────────
        $display("\n============================================");
        $display("  SM Top Testbench — Summary");
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