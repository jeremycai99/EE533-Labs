/* file: soc_driver_tb.v
   Description: Testbench for soc_driver module
   Author: Jeremy Cai
   Date: Feb. 14, 2026
*/

`timescale 1ns/1ps
`include "define.v"

module soc_driver_tb;

    // Parameters
    localparam CLK_PERIOD = 10;

    // Signals
    reg clk;
    reg rst_n;
    reg start;
    reg user_valid;
    wire user_ready;
    reg user_cmd;
    reg [`MMIO_ADDR_WIDTH-1:0] user_addr;
    reg [`MMIO_DATA_WIDTH-1:0] user_wdata;
    wire [`MMIO_DATA_WIDTH-1:0] user_rdata;
    wire [`MMIO_ADDR_WIDTH-1:0] status;

    wire soc_req_val;
    reg soc_req_rdy;
    wire soc_req_cmd;
    wire [`MMIO_ADDR_WIDTH-1:0] soc_req_addr;
    wire [`MMIO_DATA_WIDTH-1:0] soc_req_data;

    reg soc_resp_val;
    wire soc_resp_rdy;
    reg soc_resp_cmd;
    reg [`MMIO_ADDR_WIDTH-1:0] soc_resp_addr;
    reg [`MMIO_DATA_WIDTH-1:0] soc_resp_data;

    // DUT Instantiation
    soc_driver u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
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
        .soc_resp_data(soc_resp_data)
    );

    // Clock Generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Test Procedure
    initial begin
        // Initialize Inputs
        rst_n = 0;
        start = 0;
        user_valid = 0;
        user_cmd = 0;
        user_addr = 0;
        user_wdata = 0;
        soc_req_rdy = 0;
        soc_resp_val = 0;
        soc_resp_cmd = 0;
        soc_resp_addr = 0;
        soc_resp_data = 0;

        // Reset Sequence
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        $display("=== Test 1: Start Signal Gating ===");
        
        // 1. Push transaction to FIFO while START=0
        $display("Pushing transaction to FIFO while START=0. Expecting no SoC activity.");
        user_valid = 1;
        user_cmd = 1; // Write
        user_addr = 32'h1000;
        user_wdata = 64'hABCDABCDABCDABCD;
        
        @(posedge clk);
        while (!user_ready) @(posedge clk); // Wait for ready
        user_valid = 0;
        
        // Wait a few cycles to ensure DUT stays IDLE
        repeat(5) @(posedge clk);
        
        if (soc_req_val === 1'b0) begin
            $display("PASS: Driver remained idle while START was low.");
        end else begin
            $display("FAIL: Driver asserted soc_req_val despite START=0.");
            $finish;
        end

        // 2. Assert START signal automatically
        $display("Asserting START signal...");
        start = 1; // <--- Automatically asserting start here
        $display("START asserted.");

        // 3. SoC Handshake Simulation
        // Wait for Request
        wait(soc_req_val == 1);
        @(posedge clk);
        $display("SoC received request: CMD=%b ADDR=%h DATA=%h", soc_req_cmd, soc_req_addr, soc_req_data);
        
        // Acknowledge Request
        soc_req_rdy = 1;
        @(posedge clk);
        soc_req_rdy = 0;

        // Send Response
        repeat(2) @(posedge clk);
        soc_resp_val = 1;
        soc_resp_cmd = 1; // Matching CMD
        soc_resp_addr = 32'h1000; // Matching ADDR
        soc_resp_data = 64'h0;
        
        // Wait for Driver to accept response
        wait(soc_resp_rdy == 1);
        @(posedge clk);
        soc_resp_val = 0;

        // Check Status
        repeat(2) @(posedge clk);
        if (status === 32'hAAAAAAAA) begin
            $display("PASS: Transaction completed successfully. Status: %h", status);
        end else begin
            $display("FAIL: Status incorrect. Expected AAAAAAAA, got %h", status);
        end

        $display("=== Test Complete ===");
        $finish;
    end

    // Dump waves
    initial begin
        $dumpfile("soc_driver_tb.vcd");
        $dumpvars(0, soc_driver_tb);
    end

endmodule