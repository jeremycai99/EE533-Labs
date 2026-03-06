/* file: soc_driver_tb.v
 * Description: Comprehensive Testbench for soc_driver.v
 * Author: Jeremy Cai
 * Date: Feb. 14, 2026
 * Version: 1.4
 */

`timescale 1ns/1ps

`include "soc_driver.v"
`include "define.v"

module soc_driver_tb;

    // ========================================================================
    // Parameters
    // ========================================================================
    localparam TIMEOUT_THRESHOLD = 20; 
    localparam CLK_PERIOD = 10;
    localparam FIFO_DEPTH = 16; 
    localparam MAX_WAIT_CYCLES = 5000; 

    // ========================================================================
    // Signals
    // ========================================================================
    reg clk;
    reg rst_n;
    // reg start; // Removed
    reg clear_stats;

    // User Interface
    reg  user_valid;
    wire user_ready;
    reg  user_cmd; 
    reg  [`MMIO_ADDR_WIDTH-1:0] user_addr;
    reg  [`MMIO_DATA_WIDTH-1:0] user_wdata;
    wire [`MMIO_DATA_WIDTH-1:0] user_rdata;
    wire [`MMIO_ADDR_WIDTH-1:0] status;

    // SoC Interface
    wire soc_req_val;
    reg  soc_req_rdy;
    wire soc_req_cmd;
    wire [`MMIO_ADDR_WIDTH-1:0] soc_req_addr;
    wire [`MMIO_DATA_WIDTH-1:0] soc_req_data;

    reg  soc_resp_val;
    wire soc_resp_rdy;
    reg  soc_resp_cmd;
    reg  [`MMIO_ADDR_WIDTH-1:0] soc_resp_addr;
    reg  [`MMIO_DATA_WIDTH-1:0] soc_resp_data;

    // Monitoring
    wire [7:0] conn_status;
    wire [`MMIO_DATA_WIDTH-1:0] txn_quality;
    wire [`MMIO_DATA_WIDTH-1:0] txn_counters;

    // Loop counters
    integer i;
    integer wait_count;

    // ========================================================================
    // Coverage Counters
    // ========================================================================
    integer cov_cmd_read;
    integer cov_cmd_write;
    integer cov_addr_low;   
    integer cov_addr_high;  
    integer cov_data_zero;
    integer cov_data_max;
    integer cov_fifo_full;
    integer cov_timeout_err;
    integer cov_proto_err;
    integer cov_reset_active;

    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    soc_driver #(
        .TIMEOUT_THRESHOLD(TIMEOUT_THRESHOLD)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        // .start(start), // Removed
        .user_valid(user_valid),
        .user_ready(user_ready),
        .user_cmd(user_cmd),
        .user_addr(user_addr),
        .user_wdata(user_wdata),
        .user_rdata(user_rdata),
        .status(status),
        .soc_req_val(soc_req_val),
        .soc_req_rdy(soc_req_rdy),
        .soc_req_cmd(soc_req_cmd),
        .soc_req_addr(soc_req_addr),
        .soc_req_data(soc_req_data),
        .soc_resp_val(soc_resp_val),
        .soc_resp_rdy(soc_resp_rdy),
        .soc_resp_cmd(soc_resp_cmd),
        .soc_resp_addr(soc_resp_addr),
        .soc_resp_data(soc_resp_data),
        .conn_status(conn_status),
        .txn_quality(txn_quality),
        .txn_counters(txn_counters),
        .clear_stats(clear_stats)
    );

    // ========================================================================
    // Clock Generation
    // ========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // ========================================================================
    // Coverage Monitor Block
    // ========================================================================
    
    always @(posedge clk) begin
        if (rst_n) begin
            // Monitor User Transactions
            if (user_valid && user_ready) begin
                if (user_cmd == 0) cov_cmd_read = cov_cmd_read + 1;
                else               cov_cmd_write = cov_cmd_write + 1;

                if (user_addr < 32'h1000) cov_addr_low = cov_addr_low + 1;
                else                      cov_addr_high = cov_addr_high + 1;

                if (user_wdata == 64'h0) cov_data_zero = cov_data_zero + 1;
                if (user_wdata == {64{1'b1}}) cov_data_max = cov_data_max + 1;
            end

            // Monitor FIFO Status
            if (user_valid && !user_ready) begin
                cov_fifo_full = 1; 
            end

            // Monitor Errors - LATCHING LOGIC
            // We check if bit 0 (Timeout) is high. 
            if (txn_quality[0] === 1'b1) begin
                if (cov_timeout_err == 0) 
                    $display("[MONITOR] Timeout Bit Detected at time %t", $time);
                cov_timeout_err = 1;
            end
            
            if (conn_status[3]) cov_proto_err = 1;
        end
    end

    // Reset Monitor
    always @(negedge rst_n) begin
        if (soc_req_val === 1'b1) begin
            cov_reset_active = 1;
        end
    end

    // ========================================================================
    // Helper Tasks
    // ========================================================================
    
    task init_signals;
        begin
            rst_n <= 0;
            // start <= 0; // Removed
            clear_stats <= 0;
            user_valid <= 0;
            user_cmd <= 0;
            user_addr <= 0;
            user_wdata <= 0;
            soc_req_rdy <= 0;
            soc_resp_val <= 0;
            soc_resp_cmd <= 0;
            soc_resp_addr <= 0;
            soc_resp_data <= 0;
            
            cov_cmd_read = 0;
            cov_cmd_write = 0;
            cov_addr_low = 0;
            cov_addr_high = 0;
            cov_data_zero = 0;
            cov_data_max = 0;
            cov_fifo_full = 0;
            cov_timeout_err = 0;
            cov_proto_err = 0;
            cov_reset_active = 0;
        end
    endtask

    task reset_system;
        begin
            rst_n <= 0;
            repeat(5) @(posedge clk);
            rst_n <= 1;
            repeat(2) @(posedge clk);
        end
    endtask

    task user_push_cmd;
        input cmd;
        input [`MMIO_ADDR_WIDTH-1:0] addr;
        input [`MMIO_DATA_WIDTH-1:0] data;
        integer timeout_ctr;
        begin
            timeout_ctr = 0;
            while (!user_ready && timeout_ctr < MAX_WAIT_CYCLES) begin
                @(posedge clk);
                timeout_ctr = timeout_ctr + 1;
            end
            
            if (timeout_ctr >= MAX_WAIT_CYCLES) begin
                $display("[TB] ERROR: Timeout waiting for user_ready");
                $finish;
            end

            user_valid <= 1;
            user_cmd <= cmd;
            user_addr <= addr;
            user_wdata <= data;
            
            @(posedge clk);
            while (!user_ready) @(posedge clk); 
            user_valid <= 0;
        end
    endtask

    task soc_respond_task;
        input [31:0] delay;
        input corrupt_addr; 
        integer timeout_ctr;
        reg [`MMIO_ADDR_WIDTH-1:0] captured_addr;
        reg captured_cmd;
        begin
            soc_req_rdy <= 1;
            
            timeout_ctr = 0;
            while (!soc_req_val && timeout_ctr < MAX_WAIT_CYCLES) begin
                @(posedge clk);
                timeout_ctr = timeout_ctr + 1;
            end

            if (timeout_ctr >= MAX_WAIT_CYCLES) begin
                $display("[TB] ERROR: Timeout waiting for soc_req_val");
                $finish;
            end

            captured_addr = soc_req_addr;
            captured_cmd  = soc_req_cmd;

            @(posedge clk); 
            soc_req_rdy <= 0; 

            soc_resp_cmd  <= captured_cmd;
            soc_resp_addr <= corrupt_addr ? ~captured_addr : captured_addr;
            soc_resp_data <= 64'hFACE_FEED_CAFE_BABE; 

            repeat(delay) @(posedge clk);

            soc_resp_val <= 1;
            timeout_ctr = 0;
            while (!soc_resp_rdy && timeout_ctr < MAX_WAIT_CYCLES) begin
                @(posedge clk);
                timeout_ctr = timeout_ctr + 1;
            end
            
            @(posedge clk);
            soc_resp_val <= 0;
        end
    endtask

    task print_coverage_report;
        integer total_cmds;
        reg test_passed;
        begin
            total_cmds = cov_cmd_read + cov_cmd_write;
            test_passed = 1; 

            $display("\n============================================================");
            $display(" FUNCTIONAL COVERAGE REPORT");
            $display("============================================================");
            
            $display("Transaction Summary:");
            $write("  - Reads            : %0d", cov_cmd_read);
            if(cov_cmd_read > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end
            
            $write("  - Writes           : %0d", cov_cmd_write);
            if(cov_cmd_write > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end

            $display("");
            $display("Address/Data Coverage:");
            
            $write("  - Low Addr (<0x1K) : %0d", cov_addr_low);
            if(cov_addr_low > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end
            
            $write("  - High Addr        : %0d", cov_addr_high);
            if(cov_addr_high > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end
            
            $write("  - Zero Data        : %0d", cov_data_zero);
            if(cov_data_zero > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end

            $write("  - Max Data (All 1s): %0d", cov_data_max);
            if(cov_data_max > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end

            $display("");
            $display("Corner Cases & Errors:");
            
            $write("  - FIFO Full/Stall  : %0d", cov_fifo_full);
            if(cov_fifo_full > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end
            
            $write("  - Timeout Error    : %0d", cov_timeout_err);
            if(cov_timeout_err > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end
            
            $write("  - Protocol Error   : %0d", cov_proto_err);
            if(cov_proto_err > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end
            
            $write("  - Reset While Busy : %0d", cov_reset_active);
            if(cov_reset_active > 0) $display(" [HIT]"); else begin $display(" [MISS]"); test_passed = 0; end

            $display("============================================================");
            if (test_passed) begin
                $display("  OVERALL RESULT: [ PASS ]");
            end else begin
                $display("  OVERALL RESULT: [ FAIL ]");
            end
            $display("============================================================\n");
        end
    endtask

    // ========================================================================
    // Main Test Sequence
    // ========================================================================
    
    initial begin
        $dumpfile("soc_driver_tb.vcd");
        $dumpvars(0, soc_driver_tb);
        
        init_signals();
        
        $display("============================================================");
        $display(" Starting Coverage Testbench for SoC Driver");
        $display("============================================================");

        // 1. Basic Read/Write
        reset_system();
        // start <= 1; // Removed
        repeat(5) @(posedge clk); 
        
        user_push_cmd(0, 32'h0100, 64'h0);
        soc_respond_task(2, 0);
        
        user_push_cmd(1, 32'h2000, 64'hA);
        soc_respond_task(2, 0);

        // 2. Data Patterns
        reset_system();
        // start <= 1; // Removed
        repeat(5) @(posedge clk);

        user_push_cmd(1, 32'h3000, 64'h0); 
        soc_respond_task(1, 0);

        user_push_cmd(1, 32'h3004, {64{1'b1}}); 
        soc_respond_task(1, 0);

        // 3. FIFO Saturation
        reset_system();
        // start <= 1; // Removed
        repeat(5) @(posedge clk);

        soc_req_rdy <= 0; 
        
        $display("[TB] Filling FIFO...");
        for (i = 0; i < FIFO_DEPTH + 2; i = i + 1) begin
            user_valid <= 1;
            user_cmd <= 1;
            user_addr <= 32'h1000 + i;
            user_wdata <= i;
            @(posedge clk);
            while(!user_ready && i < FIFO_DEPTH) @(posedge clk); 
        end
        user_valid <= 0;
        
        $display("[TB] Draining FIFO...");
        soc_respond_task(0, 0); 

        // 4. Asynchronous Reset
        reset_system();
        // start <= 1; // Removed
        repeat(5) @(posedge clk);

        user_push_cmd(1, 32'hB0, 64'hF);
        
        wait_count = 0;
        while(!soc_req_val && wait_count < MAX_WAIT_CYCLES) begin
            @(posedge clk);
            wait_count = wait_count + 1;
        end
        
        $display("[TB] Triggering Reset during active transaction...");
        rst_n <= 0; 
        repeat(5) @(posedge clk);
        rst_n <= 1;

        // 5. Timeout Error (UPDATED: Request Stall)
        reset_system();
        // start <= 1; // Removed
        repeat(5) @(posedge clk);

        $display("[TB] Testing Timeout (Request Stall)...");
        
        // Push a command to the driver
        user_push_cmd(0, 32'hC0, 64'h0);
        
        // CRITICAL CHANGE: 
        // We do NOT assert soc_req_rdy. We simulate a dead SoC.
        // The driver will try to assert soc_req_val, but we keep rdy low.
        soc_req_rdy <= 0; 
        
        // Wait for the DUT's internal timer to expire.
        // We wait significantly longer than the threshold to be safe.
        wait_count = 0;
        while (txn_quality[0] == 0 && wait_count < (TIMEOUT_THRESHOLD * 5)) begin
            @(posedge clk);
            wait_count = wait_count + 1;
        end
        
        if (txn_quality[0] == 1) 
            $display("[TB] Timeout detected successfully at cycle %0d", wait_count);
        else 
            $display("[TB] WARNING: Timeout signal never went high!");

        // 6. Protocol Error
        reset_system();
        // start <= 1; // Removed
        repeat(5) @(posedge clk);

        $display("[TB] Testing Protocol Error...");
        soc_resp_val <= 1; 
        @(posedge clk);
        soc_resp_val <= 0;
        @(posedge clk);

        // End & Report
        repeat(20) @(posedge clk);
        print_coverage_report();
        $finish;
    end

endmodule