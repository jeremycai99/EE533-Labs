/* file: tb_top.v
 * Description: Top-Level Testbench for SoC.
 *              Fixes the issue where CPU runs immediately and blocks memory loading.
 */

`timescale 1ns/1ps

`ifndef DEFINES_V
`define DEFINES_V
    `define MMIO_ADDR_WIDTH 32
    `define MMIO_DATA_WIDTH 32
    `define DATA_WIDTH 32
    `define INSTR_WIDTH 32
    `define PC_WIDTH 32
    `define DMEM_ADDR_WIDTH 10 
    `define REG_ADDR_WIDTH 5
`endif

`include "top.v"

module tb_top;

    // -------------------------------------------------------------------------
    // Signals & Parameters
    // -------------------------------------------------------------------------
    localparam CLK_PERIOD = 10;

    reg clk;
    reg rst_n;
    reg debug_mode;

    // User Transaction Interface
    reg  user_valid;
    reg  user_cmd; // 0=Read, 1=Write
    reg  [`MMIO_ADDR_WIDTH-1:0] user_addr;
    reg  [`MMIO_DATA_WIDTH-1:0] user_wdata;
    wire user_ready;
    wire [`MMIO_DATA_WIDTH-1:0] user_rdata;

    // Driver Status
    wire [`MMIO_ADDR_WIDTH-1:0] status;
    wire [7:0] conn_status;
    wire [`MMIO_DATA_WIDTH-1:0] txn_quality;
    wire [`MMIO_DATA_WIDTH-1:0] txn_counters;
    reg  clear_stats;

    // ILA Interface
    reg  [2:0] ila_addr;
    reg  [`MMIO_DATA_WIDTH-1:0] ila_din;
    reg  ila_we;
    wire [`MMIO_DATA_WIDTH-1:0] ila_dout;

    // Test Variables
    integer error_cnt = 0;
    
    // Address Map (Matches soc.v decoding)
    localparam ADDR_IMEM = 32'h0000_0000;
    localparam ADDR_DMEM = 32'h8000_0000;
    
    // Register Map
    localparam [`REG_ADDR_WIDTH-1:0] R0 = 0;
    localparam [`REG_ADDR_WIDTH-1:0] R1 = 1;
    localparam [`REG_ADDR_WIDTH-1:0] R2 = 2;
    localparam [`REG_ADDR_WIDTH-1:0] R3 = 3;

    // -------------------------------------------------------------------------
    // DUT Instantiation
    // -------------------------------------------------------------------------
    top u_top (
        .clk(clk),
        .rst_n(rst_n),
        .debug_mode(debug_mode),
        .user_valid(user_valid),
        .user_cmd(user_cmd),
        .user_addr(user_addr),
        .user_wdata(user_wdata),
        .user_ready(user_ready),
        .user_rdata(user_rdata),
        .status(status),
        .conn_status(conn_status),
        .txn_quality(txn_quality),
        .txn_counters(txn_counters),
        .clear_stats(clear_stats),
        .ila_addr(ila_addr),
        .ila_din(ila_din),
        .ila_we(ila_we),
        .ila_dout(ila_dout)
    );

    // -------------------------------------------------------------------------
    // Clock Generation
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Helper Tasks
    // -------------------------------------------------------------------------

    // Task: Write to Memory via MMIO
    task mmio_write;
        input [`MMIO_ADDR_WIDTH-1:0] addr;
        input [`MMIO_DATA_WIDTH-1:0] data;
    begin
        wait(user_ready); // Wait for driver FIFO space
        @(posedge clk);
        user_valid = 1;
        user_cmd   = 1; // Write
        user_addr  = addr;
        user_wdata = data;
        @(posedge clk);
        user_valid = 0;
        
        // Wait for transaction to complete
        // Wait for Busy (Bit 5) to rise then fall
        wait(conn_status[5] == 1); 
        wait(conn_status[5] == 0);
        
        // Check for timeout
        if (conn_status[1]) $display("ERROR: MMIO Write Timeout at addr %h", addr);
    end
    endtask

    // Task: Read from Memory via MMIO
    task mmio_read;
        input [`MMIO_ADDR_WIDTH-1:0] addr;
        output [`MMIO_DATA_WIDTH-1:0] data;
    begin
        wait(user_ready);
        @(posedge clk);
        user_valid = 1;
        user_cmd   = 0; // Read
        user_addr  = addr;
        @(posedge clk);
        user_valid = 0;
        
        wait(conn_status[5] == 1); 
        wait(conn_status[5] == 0);
        @(posedge clk);
        data = user_rdata;
    end
    endtask

    // Task: Write to ILA Register
    task ila_write_reg;
        input [2:0] addr;
        input [`MMIO_DATA_WIDTH-1:0] data;
    begin
        @(posedge clk);
        ila_addr = addr;
        ila_din  = data;
        ila_we   = 1;
        @(posedge clk);
        ila_we   = 0;
        #1;
    end
    endtask

    // Task: Read ILA Register (Probe)
    task ila_read_probe;
        input [4:0] reg_sel;
        output [63:0] probe_data; 
    begin
        // 1. Select the register via PROBE_SEL (Addr 0x2)
        ila_write_reg(3'h2, {27'd0, reg_sel});
        
        // 2. Read PROBE (Addr 0x3)
        @(posedge clk);
        ila_addr = 3'h3;
        ila_we   = 0;
        #1; 
        probe_data = ila_dout; 
    end
    endtask

    // Function: Build Instruction
    function [`INSTR_WIDTH-1:0] build_instr;
        input mw;
        input rw;
        input [`REG_ADDR_WIDTH-1:0] rd;
        input [`REG_ADDR_WIDTH-1:0] r0;
        input [`REG_ADDR_WIDTH-1:0] r1;
        begin
            build_instr = {mw, rw, r0, r1, rd, {(`INSTR_WIDTH - 2 - 3 * `REG_ADDR_WIDTH){1'b0}} };
        end
    endfunction

    localparam [`INSTR_WIDTH-1:0] NOP = {`INSTR_WIDTH{1'b0}};

    // -------------------------------------------------------------------------
    // Main Test Sequence
    // -------------------------------------------------------------------------
    reg [`MMIO_DATA_WIDTH-1:0] read_val;
    reg [63:0] probe_val;

    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);

        // 1. Initialization
        $display("=============================================");
        $display("  Top-Level SoC Testbench (MMIO + ILA)");
        $display("=============================================");
        
        clk = 0;
        rst_n = 0;
        debug_mode = 1; 
        user_valid = 0;
        ila_we = 0;
        clear_stats = 0;

        #50;
        rst_n = 1;
        $display("[%0t] Reset Released. Debug Mode = 1.", $time);
        
        // ---------------------------------------------------------------------
        // CRITICAL FIX: STOP CPU IMMEDIATELY
        // ---------------------------------------------------------------------
        // The CPU starts running by default (ILA Reg 0, Bit 3 = 1).
        // This blocks MMIO writes. We must write 0 to ILA Reg 0 to stop it.
        $display("[%0t] Stopping CPU via ILA to allow MMIO access...", $time);
        ila_write_reg(3'h0, 32'h0000_0000); 
        
        // Wait a few cycles for the CPU to actually stop
        repeat(10) @(posedge clk);

        // ---------------------------------------------------------------------
        // 3. Load Data Memory (via MMIO)
        // ---------------------------------------------------------------------
        $display("\n[%0t] Loading Data Memory...", $time);
        
        // d_mem[0] = 4
        mmio_write(ADDR_DMEM + 0, 32'd4);
        
        // d_mem[4] = 100
        mmio_write(ADDR_DMEM + 4, 32'd100);

        // ---------------------------------------------------------------------
        // 4. Load Instruction Memory (via MMIO)
        // ---------------------------------------------------------------------
        $display("\n[%0t] Loading Instruction Memory...", $time);
        
        // 0: Load d_mem[R0] to R2. (R0=0, Mem[0]=4 -> R2=4)
        mmio_write(ADDR_IMEM + 0, build_instr(1'b0, 1'b1, R2, R0, R0));
        
        // 1: Load d_mem[R0] to R3. (R0=0, Mem[0]=4 -> R3=4)
        mmio_write(ADDR_IMEM + 1, build_instr(1'b0, 1'b1, R3, R0, R0));
        
        // 2-4: NOPs
        mmio_write(ADDR_IMEM + 2, NOP);
        mmio_write(ADDR_IMEM + 3, NOP);
        mmio_write(ADDR_IMEM + 4, NOP);
        
        // 5: Store R3 into Mem[R2]. (Mem[4] = 4)
        mmio_write(ADDR_IMEM + 5, build_instr(1'b1, 1'b0, R0, R2, R3));

        // Wait for MMIO transactions to settle
        #100;

        // ---------------------------------------------------------------------
        // 5. Run CPU (via ILA)
        // ---------------------------------------------------------------------
        $display("\n[%0t] Starting CPU via ILA...", $time);
        
        // Write ILA CTRL (0x0). Set Bit 3 (cpu_run_level) = 1.
        ila_write_reg(3'h0, 32'h0000_0008);

        // Wait for execution (20 cycles)
        repeat (30) @(posedge clk);

        // Stop CPU
        ila_write_reg(3'h0, 32'h0000_0000);
        $display("[%0t] CPU Stopped.", $time);

        // ---------------------------------------------------------------------
        // 6. Verification
        // ---------------------------------------------------------------------
        $display("\n---------------------------------------------");
        $display("  Phase 1: Memory Verification (MMIO Read)");
        $display("---------------------------------------------");

        mmio_read(ADDR_DMEM + 4, read_val);
        if (read_val !== 32'd4) begin
            $display("FAIL: DMEM[4] = %0d, expected 4", read_val);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: DMEM[4] = 4");
        end

        mmio_read(ADDR_DMEM + 0, read_val);
        if (read_val !== 32'd4) begin
            $display("FAIL: DMEM[0] = %0d, expected 4", read_val);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: DMEM[0] = 4");
        end

        $display("\n---------------------------------------------");
        $display("  Phase 2: Register Verification (ILA Probe)");
        $display("---------------------------------------------");

        ila_read_probe(R2, probe_val);
        if (probe_val[31:0] !== 32'd4) begin
            $display("FAIL: ILA Read R2: Expected 4, Got %0d", probe_val[31:0]);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: ILA Read R2: Correctly read %0d", probe_val[31:0]);
        end

        ila_read_probe(R3, probe_val);
        if (probe_val[31:0] !== 32'd4) begin
            $display("FAIL: ILA Read R3: Expected 4, Got %0d", probe_val[31:0]);
            error_cnt = error_cnt + 1;
        end else begin
            $display("PASS: ILA Read R3: Correctly read %0d", probe_val[31:0]);
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