/* file: cpu_tb.v
 Description: Testbench for the CPU module
 Author: Jeremy Cai
 Date: Feb. 12, 2026
 Version: 1.3
 */

`timescale 1ns/1ps

`include "define.v"
`include "cpu.v"

module cpu_tb;
    reg clk;
    reg rst_n;

    localparam CLK_PERIOD = 10; // Clock period in nanoseconds

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk; // Generate clock signal

    reg [`INSTR_WIDTH-1:0] i_mem_data_i;
    wire [`PC_WIDTH-1:0] i_mem_addr_o;

    reg [`DATA_WIDTH-1:0] d_mem_data_i;
    wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o;
    wire [`DATA_WIDTH-1:0] d_mem_data_o;
    wire d_mem_wen_o;
    
    // Debug/ILA Interface Signals
    reg  [4:0]                  ila_debug_sel;
    wire [`DATA_WIDTH-1:0]      ila_debug_data;
    wire                        cpu_done;

    cpu u_cpu (
        .clk(clk),
        .rst_n(rst_n),
        .i_mem_data_i(i_mem_data_i),
        .i_mem_addr_o(i_mem_addr_o),
        .d_mem_data_i(d_mem_data_i),
        .d_mem_addr_o(d_mem_addr_o),
        .d_mem_data_o(d_mem_data_o),
        .d_mem_wen_o(d_mem_wen_o),
        .ila_debug_sel(ila_debug_sel),
        .ila_debug_data(ila_debug_data),
        .cpu_done(cpu_done)
    );

    // CPU testbench doesn't include actual instruction and data memory models
    localparam I_MEM_DEPTH = 64; // Depth of instruction memory
    reg [`INSTR_WIDTH-1:0] i_mem [0:I_MEM_DEPTH-1];

    always @(*) begin
        i_mem_data_i = i_mem[i_mem_addr_o[5:0]]; // Provide instruction data based on address from CPU
    end

    localparam D_MEM_DEPTH = 64; // Depth of data memory
    reg [`DATA_WIDTH-1:0] d_mem [0:D_MEM_DEPTH-1];
    always @(*) begin
        d_mem_data_i = d_mem[d_mem_addr_o[5:0]]; // Provide data based on address from CPU
    end

    always @(posedge clk) begin
        if (d_mem_wen_o) begin
            d_mem[d_mem_addr_o[5:0]] <= d_mem_data_o; // Write data to memory if write enable is asserted
            $display("Data memory write: Address = %h, Data = %h", d_mem_addr_o, d_mem_data_o);
        end
    end

    localparam [`REG_ADDR_WIDTH-1:0] R0 = 0;
    localparam [`REG_ADDR_WIDTH-1:0] R1 = 1;
    localparam [`REG_ADDR_WIDTH-1:0] R2 = 2;
    localparam [`REG_ADDR_WIDTH-1:0] R3 = 3;

    // Instruction builder function for testing
    function [`INSTR_WIDTH-1:0] build_instr;
        input mw;
        input rw;
        input [`REG_ADDR_WIDTH-1:0] rd;
        input [`REG_ADDR_WIDTH-1:0] r0;
        input [`REG_ADDR_WIDTH-1:0] r1;

        begin
            build_instr = {mw, rw, r0, r1, rd, {(`INSTR_WIDTH - 2 - 3 * `REG_ADDR_WIDTH){1'b0}} }; // Simple instruction format for testing
        end
    endfunction

    localparam [`INSTR_WIDTH-1:0] NOP = {`INSTR_WIDTH{1'b0}};

    // Register file initialization — SEPARATE block
    integer k;
    initial begin
        for (k = 0; k < (1 << `REG_ADDR_WIDTH); k = k + 1) begin
            u_cpu.u_regfile.regs[k] = {`REG_DATA_WIDTH{1'b0}};
        end
    end

    integer i;
    integer cycle_cnt;
    integer error_cnt;

    // Start Test
    initial begin
        $display("Initialize testbench...");
        $dumpfile("cpu_tb.vcd");
        $dumpvars(0, cpu_tb);

        rst_n = 0; // Assert reset
        cycle_cnt = 0;
        error_cnt = 0;
        ila_debug_sel = 5'd0; // Initialize ILA selector (system debug mode)

        for (i = 0; i < I_MEM_DEPTH; i = i + 1) begin
            i_mem[i] = NOP; // Initialize instruction memory with NOPs
        end
        for (i = 0; i < D_MEM_DEPTH; i = i + 1) begin
            d_mem[i] = {`DATA_WIDTH{1'b0}}; // Initialize data memory with zeros
        end
        
        // Preload some data into memory for testing
        d_mem[0] = 32'd4;
        d_mem[4] = 32'd100;

        // Make and load the instruction sequence for testing
        i_mem[0] = build_instr(1'b0, 1'b1, R2, R0, R0);
        i_mem[1] = build_instr(1'b0, 1'b1, R3, R0, R0);
        i_mem[2] = NOP;
        i_mem[3] = NOP;
        i_mem[4] = NOP;
        i_mem[5] = build_instr(1'b1, 1'b0, R0, R2, R3);


        // =============================================
        // Display Test Plan
        // =============================================
        $display("");
        $display("=============================================");
        $display("  CPU Pipeline Testbench");
        $display("=============================================");
        $display("");
        $display("Data Memory (Initial):");
        $display("  Addr | Value");
        $display("  -----|------");
        $display("    0  |   %0d", d_mem[0]);
        $display("    4  | %0d",   d_mem[4]);
        $display("");
        $display("Instruction Memory:");
        $display("  Addr | WME | WRE | REG1 | REG2 | WREG1 | Comments");
        $display("  -----|-----|-----|------|------|-------|---------------------------");
        $display("    0  |  0  |  1  | 000  | XXX  | 002   | Load d_mem[R0] to R2");
        $display("    1  |  0  |  1  | 000  | XXX  | 003   | Load d_mem[R0] to R3");
        $display("    2  |  0  |  0  | XXX  | XXX  | XXX   | Nop");
        $display("    3  |  0  |  0  | XXX  | XXX  | XXX   | Nop - R2=4 by now");
        $display("    4  |  0  |  0  | XXX  | XXX  | XXX   | Nop - R3=4 by now");
        $display("    5  |  1  |  0  | 002  | 003  | XXX   | Store 4 into Mem addr 4");
        $display("");
        $display("Expected: d_mem[4] changes from 100 to 4");
        $display("");

        // =============================================
        // Reset Sequence (hold 3 cycles, release on negedge)
        // =============================================
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        $display("[%0t] Reset de-asserted", $time);
        $display("---------------------------------------------");

        // =============================================
        // Run enough cycles for all instructions to retire
        // (6 instructions + 5 pipeline stages + margin)
        // =============================================
        repeat (20) @(posedge clk);

        // =============================================
        // Self-Checking
        // =============================================
        $display("---------------------------------------------");
        $display("  Phase 1: Memory Verification");
        $display("---------------------------------------------");
        $display("");
        $display("Data Memory (Final):");
        $display("  Addr | Value");
        $display("  -----|------");
        $display("    0  |   %0d", d_mem[0]);
        $display("    4  |   %0d", d_mem[4]);
        $display("");

        if (d_mem[4] !== 32'd4) begin
            $display("FAIL: d_mem[4] = %0d, expected 4", d_mem[4]);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: d_mem[4] = 4 (was 100, store succeeded)");
        end

        if (d_mem[0] !== 32'd4) begin
            $display("FAIL: d_mem[0] = %0d, expected 4 (unchanged)", d_mem[0]);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: d_mem[0] = 4 (unchanged as expected)");
        end

        // =============================================
        // Phase 2: Register Interface Verification
        // =============================================
        $display("");
        $display("---------------------------------------------");
        $display("  Phase 2: Register Interface Verification");
        $display("---------------------------------------------");
        
        // Check R2 (Should contain 4)
        // ila_debug_sel[4]=1 selects register-file mode; [2:0]=register address
        ila_debug_sel = {1'b1, 1'b0, R2};
        #1; // Allow combinational logic to propagate
        if (ila_debug_data !== 64'd4) begin
            $display("FAIL: ILA Read R2: Expected 4, Got %0d", ila_debug_data);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: ILA Read R2: Correctly read %0d", ila_debug_data);
        end

        // Check R3 (Should contain 4)
        ila_debug_sel = {1'b1, 1'b0, R3};
        #1; 
        if (ila_debug_data !== 64'd4) begin
            $display("FAIL: ILA Read R3: Expected 4, Got %0d", ila_debug_data);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: ILA Read R3: Correctly read %0d", ila_debug_data);
        end

        // Check R0 (Should contain 0)
        ila_debug_sel = {1'b1, 1'b0, R0};
        #1;
        if (ila_debug_data !== 64'd0) begin
            $display("FAIL: ILA Read R0: Expected 0, Got %0d", ila_debug_data);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: ILA Read R0: Correctly read %0d", ila_debug_data);
        end

        $display("");
        $display("=============================================");
        if (error_cnt == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  %0d TEST(S) FAILED", error_cnt);
        $display("=============================================");

        $finish;
    end

endmodule