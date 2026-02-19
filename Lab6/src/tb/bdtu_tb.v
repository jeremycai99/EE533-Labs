/* file: bdtu_tb.v
 * Description: Testbench for the Block Data Transfer Unit (BDTU).
 *              Models a synchronous memory (1-cycle read latency) and
 *              a combinational register file to fully exercise the DUT.
 *
 * Usage:
 *   Ensure define.v exists in the include path (required by the DUT).
 *   Compile with:  iverilog -o bdtu_tb bdtu_tb.v bdtu.v
 *   Run with:      vvp bdtu_tb
 *   Or add as testbench in ISE and simulate with ISim.
 *
 * Author: Jeremy Cai
 * Date:   Feb. 18, 2026
 */

`timescale 1ns / 1ps

`include "define.v"
`include "bdtu.v"

module bdtu_tb;

    // ── Clock ─────────────────────────────────────────────────────
    parameter CLK_PERIOD = 10;
    reg clk;
    always #(CLK_PERIOD / 2) clk = ~clk;

    // ── DUT I/O ───────────────────────────────────────────────────
    reg         rst_n;
    reg         start;
    reg         op_bdt;
    reg         op_swp;
    reg [15:0]  reg_list;
    reg         bdt_load;
    reg         bdt_wb;
    reg         pre_index;
    reg         up_down;
    reg         bdt_s;
    reg         swap_byte;
    reg [3:0]   swp_rd;
    reg [3:0]   swp_rm;
    reg [3:0]   base_reg;
    reg [31:0]  base_value;

    wire [3:0]  rf_rd_addr;
    reg  [31:0] rf_rd_data;

    wire [3:0]  wr_addr1;
    wire [31:0] wr_data1;
    wire        wr_en1;
    wire [3:0]  wr_addr2;
    wire [31:0] wr_data2;
    wire        wr_en2;

    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire        mem_rd;
    wire        mem_wr;
    wire [1:0]  mem_size;
    reg  [31:0] mem_rdata;

    wire        busy;

    // ── DUT Instantiation ─────────────────────────────────────────
    bdtu uut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .op_bdt     (op_bdt),
        .op_swp     (op_swp),
        .reg_list   (reg_list),
        .bdt_load   (bdt_load),
        .bdt_wb     (bdt_wb),
        .pre_index  (pre_index),
        .up_down    (up_down),
        .bdt_s      (bdt_s),
        .swap_byte  (swap_byte),
        .swp_rd     (swp_rd),
        .swp_rm     (swp_rm),
        .base_reg   (base_reg),
        .base_value (base_value),
        .rf_rd_addr (rf_rd_addr),
        .rf_rd_data (rf_rd_data),
        .wr_addr1   (wr_addr1),
        .wr_data1   (wr_data1),
        .wr_en1     (wr_en1),
        .wr_addr2   (wr_addr2),
        .wr_data2   (wr_data2),
        .wr_en2     (wr_en2),
        .mem_addr   (mem_addr),
        .mem_wdata  (mem_wdata),
        .mem_rd     (mem_rd),
        .mem_wr     (mem_wr),
        .mem_size   (mem_size),
        .mem_rdata  (mem_rdata),
        .busy       (busy)
    );

    // ── Register File Model (16 x 32-bit, combinational read) ────
    reg [31:0] regfile [0:15];

    always @(*) begin
        rf_rd_data = regfile[rf_rd_addr];
    end

    always @(posedge clk) begin
        if (wr_en1) begin
            regfile[wr_addr1] <= wr_data1;
            $display("  [RF WR1] R%-2d <= 0x%08X  @ %0t", wr_addr1, wr_data1, $time);
        end
        if (wr_en2) begin
            regfile[wr_addr2] <= wr_data2;
            $display("  [RF WR2] R%-2d <= 0x%08X  @ %0t", wr_addr2, wr_data2, $time);
        end
    end

    // ── Synchronous Memory Model (16K x 32-bit = 64 KB) ──────────
    //   Address captured at posedge; read data valid the FOLLOWING
    //   cycle. Matches DUT's sync-memory assumption.
    reg [31:0] memory [0:16383];

    always @(posedge clk) begin
        if (mem_wr) begin
            memory[mem_addr[15:2]] <= mem_wdata;
            $display("  [MEM WR] [0x%08X] <= 0x%08X  size=%b  @ %0t",
                     mem_addr, mem_wdata, mem_size, $time);
        end
        if (mem_rd) begin
            mem_rdata <= memory[mem_addr[15:2]];
            $display("  [MEM RD] [0x%08X] => 0x%08X  @ %0t",
                     mem_addr, memory[mem_addr[15:2]], $time);
        end
    end

    // ── Scoreboard ────────────────────────────────────────────────
    integer errors;
    integer test_num;

    // ── Helper Tasks ──────────────────────────────────────────────

    task do_reset;
        begin
            rst_n      = 0;
            start      = 0;
            op_bdt     = 0;
            op_swp     = 0;
            reg_list   = 16'd0;
            bdt_load   = 0;
            bdt_wb     = 0;
            pre_index  = 0;
            up_down    = 0;
            bdt_s      = 0;
            swap_byte  = 0;
            swp_rd     = 4'd0;
            swp_rm     = 4'd0;
            base_reg   = 4'd0;
            base_value = 32'd0;
            mem_rdata  = 32'd0;
            @(posedge clk);
            @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask

    task init_all;
        integer j;
        begin
            for (j = 0; j < 16384; j = j + 1)
                memory[j] = 32'h0;
            for (j = 0; j < 16; j = j + 1)
                regfile[j] = 32'h0;
        end
    endtask

    // Wait for busy to deassert. Counts posedges while busy is high.
    // Then waits one extra posedge so all NBA register/memory writes
    // from the final active state have settled before check tasks run.
    task wait_done;
        input integer timeout;
        integer cnt;
        begin
            cnt = 0;
            while (busy && cnt < timeout) begin
                @(posedge clk);
                cnt = cnt + 1;
            end
            if (cnt >= timeout) begin
                $display("  ** TIMEOUT after %0d cycles **", cnt);
                errors = errors + 1;
            end else begin
                $display("  -- completed (%0d posedges while busy) --", cnt);
            end
            @(posedge clk);
        end
    endtask

    task check_reg;
        input [3:0]  rnum;
        input [31:0] expected;
        begin
            if (regfile[rnum] !== expected) begin
                $display("  FAIL: R%0d = 0x%08X, expected 0x%08X",
                         rnum, regfile[rnum], expected);
                errors = errors + 1;
            end else begin
                $display("  PASS: R%0d = 0x%08X", rnum, regfile[rnum]);
            end
        end
    endtask

    task check_mem;
        input [31:0] addr;
        input [31:0] expected;
        begin
            if (memory[addr[15:2]] !== expected) begin
                $display("  FAIL: mem[0x%08X] = 0x%08X, expected 0x%08X",
                         addr, memory[addr[15:2]], expected);
                errors = errors + 1;
            end else begin
                $display("  PASS: mem[0x%08X] = 0x%08X", addr, memory[addr[15:2]]);
            end
        end
    endtask

    // Assert start for exactly one posedge, then clear triggers.
    // All configuration inputs must be set BEFORE calling this task.
    task fire_start;
        begin
            start = 1;
            @(negedge clk);
            start  = 0;
            op_bdt = 0;
            op_swp = 0;
        end
    endtask

    // ── Main Test Sequence ────────────────────────────────────────
    integer i;

    initial begin
        $dumpfile("bdtu_tb.vcd");
        $dumpvars(0, bdtu_tb);

        clk    = 0;
        errors = 0;

        do_reset;
        init_all;

        // ═══════════════════════════════════════════════════════════
        // Test 1 — LDMIA R13, {R0, R1, R2}   (P=0 U=1 L=1 W=0)
        //   start_addr = Rn = 0x1000
        //   addrs: 0x1000 0x1004 0x1008
        // ═══════════════════════════════════════════════════════════
        test_num = 1;
        $display("\n===== Test %0d: LDMIA R13, {R0,R1,R2}  no WB =====", test_num);
        init_all;

        memory[32'h1000 >> 2] = 32'hAAAA_0000;
        memory[32'h1004 >> 2] = 32'hBBBB_1111;
        memory[32'h1008 >> 2] = 32'hCCCC_2222;
        regfile[13] = 32'h0000_1000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0007;
        bdt_load   = 1;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd13;  base_value = 32'h0000_1000;
        fire_start;
        wait_done(20);

        check_reg(4'd0,  32'hAAAA_0000);
        check_reg(4'd1,  32'hBBBB_1111);
        check_reg(4'd2,  32'hCCCC_2222);
        check_reg(4'd13, 32'h0000_1000);  // unchanged (no WB)

        // ═══════════════════════════════════════════════════════════
        // Test 2 — STMIA R13, {R0, R1, R2}   (P=0 U=1 L=0 W=0)
        //   start_addr = 0x2000
        //   addrs: 0x2000 0x2004 0x2008
        // ═══════════════════════════════════════════════════════════
        test_num = 2;
        $display("\n===== Test %0d: STMIA R13, {R0,R1,R2}  no WB =====", test_num);
        init_all;

        regfile[0]  = 32'h1111_1111;
        regfile[1]  = 32'h2222_2222;
        regfile[2]  = 32'h3333_3333;
        regfile[13] = 32'h0000_2000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0007;
        bdt_load   = 0;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd13;  base_value = 32'h0000_2000;
        fire_start;
        wait_done(20);

        check_mem(32'h2000, 32'h1111_1111);
        check_mem(32'h2004, 32'h2222_2222);
        check_mem(32'h2008, 32'h3333_3333);
        check_reg(4'd13,   32'h0000_2000);

        // ═══════════════════════════════════════════════════════════
        // Test 3 — LDMIB R13!, {R4, R5}   (P=1 U=1 L=1 W=1)
        //   start_addr = Rn+4 = 0x2004
        //   new_base   = Rn+4N = 0x2008
        // ═══════════════════════════════════════════════════════════
        test_num = 3;
        $display("\n===== Test %0d: LDMIB R13!, {R4,R5}  IB WB =====", test_num);
        init_all;

        memory[32'h2004 >> 2] = 32'hDEAD_BEEF;
        memory[32'h2008 >> 2] = 32'hCAFE_BABE;
        regfile[13] = 32'h0000_2000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0030;
        bdt_load   = 1;  bdt_wb     = 1;
        pre_index  = 1;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd13;  base_value = 32'h0000_2000;
        fire_start;
        wait_done(20);

        check_reg(4'd4,  32'hDEAD_BEEF);
        check_reg(4'd5,  32'hCAFE_BABE);
        check_reg(4'd13, 32'h0000_2008);

        // ═══════════════════════════════════════════════════════════
        // Test 4 — STMDB R13!, {R0-R3}   (P=1 U=0 L=0 W=1)
        //   N=4  start_addr = Rn-4N = 0x2FF0
        //   addrs: 0x2FF0 0x2FF4 0x2FF8 0x2FFC
        //   new_base = 0x2FF0
        // ═══════════════════════════════════════════════════════════
        test_num = 4;
        $display("\n===== Test %0d: STMDB R13!, {R0-R3}  DB WB =====", test_num);
        init_all;

        regfile[0]  = 32'hAA00_AA00;
        regfile[1]  = 32'hBB11_BB11;
        regfile[2]  = 32'hCC22_CC22;
        regfile[3]  = 32'hDD33_DD33;
        regfile[13] = 32'h0000_3000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h000F;
        bdt_load   = 0;  bdt_wb     = 1;
        pre_index  = 1;  up_down    = 0;
        bdt_s      = 0;
        base_reg   = 4'd13;  base_value = 32'h0000_3000;
        fire_start;
        wait_done(20);

        check_mem(32'h2FF0, 32'hAA00_AA00);
        check_mem(32'h2FF4, 32'hBB11_BB11);
        check_mem(32'h2FF8, 32'hCC22_CC22);
        check_mem(32'h2FFC, 32'hDD33_DD33);
        check_reg(4'd13,   32'h0000_2FF0);

        // ═══════════════════════════════════════════════════════════
        // Test 5 — SWP R3, R2, [R1]   (word swap)
        //   mem[0x4000] -> R3,  R2 -> mem[0x4000]
        // ═══════════════════════════════════════════════════════════
        test_num = 5;
        $display("\n===== Test %0d: SWP R3, R2, [R1] =====", test_num);
        init_all;

        regfile[1] = 32'h0000_4000;
        regfile[2] = 32'hABCD_EF00;
        memory[32'h4000 >> 2] = 32'h1234_5678;

        @(negedge clk);
        op_swp     = 1;  swap_byte  = 0;
        swp_rd     = 4'd3;  swp_rm  = 4'd2;
        base_reg   = 4'd1;  base_value = 32'h0000_4000;
        fire_start;
        wait_done(20);

        check_reg(4'd3, 32'h1234_5678);
        check_mem(32'h4000, 32'hABCD_EF00);

        // ═══════════════════════════════════════════════════════════
        // Test 6 — SWPB R5, R4, [R6]   (byte swap)
        //   mem_size should be 2'b00 throughout the operation.
        //   Note: byte-lane extraction is handled externally;
        //   the BDTU passes full 32-bit words through.
        // ═══════════════════════════════════════════════════════════
        test_num = 6;
        $display("\n===== Test %0d: SWPB R5, R4, [R6] =====", test_num);
        init_all;

        regfile[6] = 32'h0000_5000;
        regfile[4] = 32'h0000_00FF;
        memory[32'h5000 >> 2] = 32'h0000_00AB;

        @(negedge clk);
        op_swp     = 1;  swap_byte  = 1;
        swp_rd     = 4'd5;  swp_rm  = 4'd4;
        base_reg   = 4'd6;  base_value = 32'h0000_5000;
        fire_start;
        wait_done(20);

        check_reg(4'd5, 32'h0000_00AB);
        check_mem(32'h5000, 32'h0000_00FF);

        // ═══════════════════════════════════════════════════════════
        // Test 7 — LDMDA R10, {R7}   (P=0 U=0, single register)
        //   N=1  base_dn = 0x5FFC  start_addr = base_dn+4 = 0x6000
        // ═══════════════════════════════════════════════════════════
        test_num = 7;
        $display("\n===== Test %0d: LDMDA R10, {R7}  DA single =====", test_num);
        init_all;

        memory[32'h6000 >> 2] = 32'hFACE_B00C;
        regfile[10] = 32'h0000_6000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0080;
        bdt_load   = 1;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 0;
        bdt_s      = 0;
        base_reg   = 4'd10;  base_value = 32'h0000_6000;
        fire_start;
        wait_done(20);

        check_reg(4'd7, 32'hFACE_B00C);

        // ═══════════════════════════════════════════════════════════
        // Test 8 — STMIA R9, {R8}   (single register, no WB)
        // ═══════════════════════════════════════════════════════════
        test_num = 8;
        $display("\n===== Test %0d: STMIA R9, {R8}  single =====", test_num);
        init_all;

        regfile[8] = 32'h8888_8888;
        regfile[9] = 32'h0000_7000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0100;
        bdt_load   = 0;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd9;  base_value = 32'h0000_7000;
        fire_start;
        wait_done(20);

        check_mem(32'h7000, 32'h8888_8888);

        // ═══════════════════════════════════════════════════════════
        // Test 9 — LDMIA R0!, {R1-R14}   (14 registers, WB)
        //   new_base = 0x100 + 14*4 = 0x138
        // ═══════════════════════════════════════════════════════════
        test_num = 9;
        $display("\n===== Test %0d: LDMIA R0!, {R1-R14}  14 regs WB =====", test_num);
        init_all;

        regfile[0] = 32'h0000_0100;
        for (i = 0; i < 14; i = i + 1)
            memory[(32'h100 >> 2) + i] = 32'h1000_0000 + i;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h7FFE;
        bdt_load   = 1;  bdt_wb     = 1;
        pre_index  = 0;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd0;  base_value = 32'h0000_0100;
        fire_start;
        wait_done(30);

        for (i = 1; i <= 14; i = i + 1)
            check_reg(i[3:0], 32'h1000_0000 + (i - 1));
        check_reg(4'd0, 32'h0000_0138);

        // ═══════════════════════════════════════════════════════════
        // Test 10 — STMIB R5!, {R0, R2, R4}   (P=1 U=1, WB)
        //   start_addr = 0x8004  new_base = 0x800C
        // ═══════════════════════════════════════════════════════════
        test_num = 10;
        $display("\n===== Test %0d: STMIB R5!, {R0,R2,R4}  IB WB =====", test_num);
        init_all;

        regfile[0] = 32'hAAAA_AAAA;
        regfile[2] = 32'hBBBB_BBBB;
        regfile[4] = 32'hCCCC_CCCC;
        regfile[5] = 32'h0000_8000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0015;
        bdt_load   = 0;  bdt_wb     = 1;
        pre_index  = 1;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd5;  base_value = 32'h0000_8000;
        fire_start;
        wait_done(20);

        check_mem(32'h8004, 32'hAAAA_AAAA);
        check_mem(32'h8008, 32'hBBBB_BBBB);
        check_mem(32'h800C, 32'hCCCC_CCCC);
        check_reg(4'd5,    32'h0000_800C);

        // ═══════════════════════════════════════════════════════════
        // Test 11 — LDMDB R3!, {R0, R1}   (P=1 U=0, WB)
        //   N=2  start_addr = Rn-4N = 0x8FF8  new_base = 0x8FF8
        // ═══════════════════════════════════════════════════════════
        test_num = 11;
        $display("\n===== Test %0d: LDMDB R3!, {R0,R1}  DB WB =====", test_num);
        init_all;

        memory[32'h8FF8 >> 2] = 32'h1111_1111;
        memory[32'h8FFC >> 2] = 32'h2222_2222;
        regfile[3] = 32'h0000_9000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0003;
        bdt_load   = 1;  bdt_wb     = 1;
        pre_index  = 1;  up_down    = 0;
        bdt_s      = 0;
        base_reg   = 4'd3;  base_value = 32'h0000_9000;
        fire_start;
        wait_done(20);

        check_reg(4'd0, 32'h1111_1111);
        check_reg(4'd1, 32'h2222_2222);
        check_reg(4'd3, 32'h0000_8FF8);

        // ═══════════════════════════════════════════════════════════
        // Test 12 — Back-to-back: STMIA then LDMIA (same base)
        //   Verifies the BDTU returns cleanly to IDLE and can
        //   immediately accept a new operation.
        // ═══════════════════════════════════════════════════════════
        test_num = 12;
        $display("\n===== Test %0d: Back-to-back STM IA -> LDM IA =====", test_num);
        init_all;

        regfile[0]  = 32'hFEED_FACE;
        regfile[1]  = 32'hDEAD_CAFE;
        regfile[10] = 32'h0000_A000;

        // Part A: store
        $display("  -- Part A: STMIA R10, {R0,R1} --");
        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0003;
        bdt_load   = 0;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd10;  base_value = 32'h0000_A000;
        fire_start;
        wait_done(20);

        check_mem(32'hA000, 32'hFEED_FACE);
        check_mem(32'hA004, 32'hDEAD_CAFE);

        // Part B: load back into different registers
        $display("  -- Part B: LDMIA R10, {R4,R5} --");
        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0030;
        bdt_load   = 1;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd10;  base_value = 32'h0000_A000;
        fire_start;
        wait_done(20);

        check_reg(4'd4, 32'hFEED_FACE);
        check_reg(4'd5, 32'hDEAD_CAFE);

        // ═══════════════════════════════════════════════════════════
        // Test 13 — STMDA R7, {R0, R3, R5}   (P=0 U=0, no WB)
        //   N=3  base_dn = 0xAFF4  start_addr = base_dn+4 = 0xAFF8
        //   addrs: 0xAFF8(R0) 0xAFFC(R3) 0xB000(R5)
        // ═══════════════════════════════════════════════════════════
        test_num = 13;
        $display("\n===== Test %0d: STMDA R7, {R0,R3,R5}  DA no WB =====", test_num);
        init_all;

        regfile[0] = 32'h0000_0000;
        regfile[3] = 32'h3333_3333;
        regfile[5] = 32'h5555_5555;
        regfile[7] = 32'h0000_B000;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0029;
        bdt_load   = 0;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 0;
        bdt_s      = 0;
        base_reg   = 4'd7;  base_value = 32'h0000_B000;
        fire_start;
        wait_done(20);

        check_mem(32'hAFF8, 32'h0000_0000);  // R0 -> lowest address
        check_mem(32'hAFFC, 32'h3333_3333);  // R3
        check_mem(32'hB000, 32'h5555_5555);  // R5 -> highest address

        // ═══════════════════════════════════════════════════════════
        // Test 14 — Empty register list (UNPREDICTABLE per ARM spec)
        //   Verify the FSM does not hang.
        // ═══════════════════════════════════════════════════════════
        test_num = 14;
        $display("\n===== Test %0d: LDMIA R0, {} — empty list =====", test_num);
        init_all;
        regfile[0] = 32'h0000_0200;

        @(negedge clk);
        op_bdt     = 1;  reg_list   = 16'h0000;
        bdt_load   = 1;  bdt_wb     = 0;
        pre_index  = 0;  up_down    = 1;
        bdt_s      = 0;
        base_reg   = 4'd0;  base_value = 32'h0000_0200;
        fire_start;
        wait_done(10);

        $display("  PASS: empty register list completed without hanging");

        // ═══════════════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════════════
        $display("\n============================================");
        if (errors == 0)
            $display("  ALL %0d TESTS PASSED", test_num);
        else
            $display("  FAILED: %0d error(s) across %0d tests", errors, test_num);
        $display("============================================\n");

        $finish;
    end

    // ── Watchdog Timer ────────────────────────────────────────────
    initial begin
        #200000;
        $display("\n** GLOBAL TIMEOUT — simulation killed **\n");
        $finish;
    end
    
endmodule