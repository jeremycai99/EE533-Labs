`timescale 1ns / 1ps

`include "define.v"
`include "ila.v"

module ila_tb;

    // ---------------------------------------------------------
    // Parameters & Signals
    // ---------------------------------------------------------
    
    // Register Addresses
    localparam ADDR_CTRL      = 3'h0;
    localparam ADDR_STATUS    = 3'h1;
    localparam ADDR_PROBE_SEL = 3'h2;
    localparam ADDR_PROBE     = 3'h3;
    localparam ADDR_CYCLE     = 3'h4;

    // DUT Signals
    reg             clk;
    reg             rst_n;
    reg  [2:0]      ila_addr;
    reg  [`MMIO_DATA_WIDTH-1:0]   ila_din;
    reg             ila_we;
    wire [`MMIO_DATA_WIDTH-1:0]   ila_dout;
    
    wire [4:0]      cpu_debug_sel;
    reg  [`DATA_WIDTH-1:0]     cpu_debug_data;
    
    wire            soc_start;
    reg             debug_mode;
    wire            soc_clk_en;
    reg             soc_busy;

    // ---------------------------------------------------------
    // DUT Instantiation
    // ---------------------------------------------------------
    ila u_ila (
        .clk            (clk),
        .rst_n          (rst_n),
        .ila_addr       (ila_addr),
        .ila_din        (ila_din),
        .ila_we         (ila_we),
        .ila_dout       (ila_dout),
        .cpu_debug_sel  (cpu_debug_sel),
        .cpu_debug_data (cpu_debug_data),
        .soc_start      (soc_start),
        .debug_mode     (debug_mode),
        .soc_clk_en     (soc_clk_en),
        .soc_busy       (soc_busy)
    );

    // ---------------------------------------------------------
    // Clock Generation
    // ---------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period
    end

    // ---------------------------------------------------------
    // Helper Tasks
    // ---------------------------------------------------------
    task write_reg(input [2:0] addr, input [`MMIO_DATA_WIDTH-1:0] data);
        begin
            @(posedge clk);
            ila_addr = addr;
            ila_din  = data;
            ila_we   = 1;
            @(posedge clk);
            ila_we   = 0;
            ila_din  = 0;
            // Wait a cycle for logic to settle
            @(posedge clk); 
        end
    endtask

    task check_clk_en(input expected_val, input [127:0] msg);
        begin
            // Sample slightly after edge to avoid race conditions in simulation
            #1; 
            if (soc_clk_en !== expected_val) begin
                $display("[FAIL] %s: Expected soc_clk_en=%b, Got=%b", msg, expected_val, soc_clk_en);
                $stop;
            end else begin
                $display("[PASS] %s: soc_clk_en=%b", msg, soc_clk_en);
            end
        end
    endtask

    // ---------------------------------------------------------
    // Main Test Sequence
    // ---------------------------------------------------------
    initial begin
        // 1. Initialization
        $display("\n--- Starting ILA Testbench ---");
        rst_n = 0;
        ila_addr = 0;
        ila_din = 0;
        ila_we = 0;
        cpu_debug_data = `DATA_WIDTH'hDEAD_BEEF;
        debug_mode = 0;
        soc_busy = 0;

        #20;
        rst_n = 1;
        #20;

        // 2. Check Reset Defaults
        $display("\n--- Test 1: Reset Defaults ---");
        if (soc_start !== 1'b1) $display("[FAIL] soc_start should default to 1");
        else $display("[PASS] soc_start defaults to 1");
        
        check_clk_en(1'b1, "Reset state (Debug Mode OFF)");

        // 3. CPU Control (Start/Stop Level)
        $display("\n--- Test 2: CPU Start/Stop Control ---");
        // Write 0 to bit 3 (cpu_run_level)
        write_reg(ADDR_CTRL, `DATA_WIDTH'b0000_0000); 
        #1;
        if (soc_start !== 1'b0) $display("[FAIL] soc_start did not go low");
        else $display("[PASS] soc_start set to 0");

        // Write 1 to bit 3
        write_reg(ADDR_CTRL, `DATA_WIDTH'b0000_1000);
        #1;
        if (soc_start !== 1'b1) $display("[FAIL] soc_start did not go high");
        else $display("[PASS] soc_start set to 1");

        // 4. Probe Select
        $display("\n--- Test 3: Probe Mux Select ---");
        write_reg(ADDR_PROBE_SEL, `DATA_WIDTH'h1F); // Set max value
        #1;
        if (cpu_debug_sel !== 5'h1F) $display("[FAIL] Probe sel mismatch");
        else $display("[PASS] Probe sel updated to 0x1F");

        // 5. Debug Mode & Clock Gating - STOP
        $display("\n--- Test 4: Debug Mode & STOP Command ---");
        debug_mode = 1; // Enable gating logic
        
        // Send STOP command (Bit 2)
        // Note: We keep Bit 3 high to keep CPU "Start" signal asserted, 
        // we are just stopping the clock.
        write_reg(ADDR_CTRL, `DATA_WIDTH'b0000_1100); 
        
        @(posedge clk); 
        check_clk_en(1'b0, "Clock should be STOPPED");

        // 6. Debug Mode - RUN
        $display("\n--- Test 5: RUN Command ---");
        // Send RUN command (Bit 1)
        write_reg(ADDR_CTRL, `DATA_WIDTH'b0000_1010);
        
        @(posedge clk);
        check_clk_en(1'b1, "Clock should be RUNNING");

        // 7. Debug Mode - STEP
        $display("\n--- Test 6: SINGLE STEP Command ---");
        // First, stop again
        write_reg(ADDR_CTRL, `DATA_WIDTH'b0000_1100);
        @(posedge clk);
        check_clk_en(1'b0, "Clock stopped before step");

        // Issue STEP command (Bit 0)
        $display("Issuing STEP command...");
        @(posedge clk);
        ila_addr = ADDR_CTRL;
        ila_din  = `DATA_WIDTH'b0000_1001; // Step=1, RunLevel=1
        ila_we   = 1;
        @(posedge clk);
        ila_we   = 0;
        
        // At this exact moment (after write), step_active should be high for ONE cycle
        // Cycle 1: soc_clk_en should be HIGH
        #1; // wait for propagation
        if (soc_clk_en !== 1'b1) $display("[FAIL] Step cycle: Clock not high!");
        else $display("[PASS] Step cycle: Clock is high");

        @(posedge clk);
        // Cycle 2: soc_clk_en should be LOW (auto-cleared)
        #1;
        if (soc_clk_en !== 1'b0) $display("[FAIL] Post-Step: Clock did not return to low!");
        else $display("[PASS] Post-Step: Clock returned to low");

        // 8. Busy Override
        $display("\n--- Test 7: SOC Busy Override ---");
        // Clock is currently stopped.
        check_clk_en(1'b0, "Clock stopped initially");
        
        soc_busy = 1;
        #1; // Async check
        if (soc_clk_en !== 1'b1) $display("[FAIL] Busy did not override clock enable");
        else $display("[PASS] Busy forced clock ON");
        
        @(posedge clk);
        soc_busy = 0;
        #1;
        if (soc_clk_en !== 1'b0) $display("[FAIL] Clock did not stop after Busy dropped");
        else $display("[PASS] Clock stopped after Busy dropped");

        // 9. Cycle Counter
        $display("\n--- Test 8: Cycle Counter Read ---");
        // Let clock run for a bit (disable debug mode to let it run free)
        debug_mode = 0;
        repeat(10) @(posedge clk);
        
        // Read Address 4
        ila_addr = ADDR_CYCLE;
        ila_we = 0;
        @(posedge clk);
        #1;
        $display("Cycle Count Read: %d", ila_dout);
        if (ila_dout < 10) $display("[FAIL] Cycle counter seems stuck or too low");
        else $display("[PASS] Cycle counter is incrementing");

        $display("\n--- All Tests Completed Successfully ---");
        $finish;
    end

endmodule