/* file: regfile_tb.v
   Description: Testbench for the Register File
*/

`timescale 1ns / 1ps

`include "define.v"
`include "regfile.v"

module regfile_tb;

    // Inputs
    reg clk;
    reg [`REG_ADDR_WIDTH-1:0] r0addr;
    reg [`REG_ADDR_WIDTH-1:0] r1addr;
    reg wena;
    reg [`REG_ADDR_WIDTH-1:0] waddr;
    reg [`REG_DATA_WIDTH-1:0] wdata;
    reg [`REG_ADDR_WIDTH-1:0] ila_cpu_reg_addr;

    // Outputs
    wire [`REG_DATA_WIDTH-1:0] r0data;
    wire [`REG_DATA_WIDTH-1:0] r1data;
    wire [`REG_DATA_WIDTH-1:0] ila_cpu_reg_data;

    // Instantiate the Unit Under Test (UUT)
    regfile uut (
        .clk(clk), 
        .r0addr(r0addr), 
        .r1addr(r1addr), 
        .wena(wena), 
        .waddr(waddr), 
        .wdata(wdata), 
        .r0data(r0data), 
        .r1data(r1data), 
        .ila_cpu_reg_addr(ila_cpu_reg_addr), 
        .ila_cpu_reg_data(ila_cpu_reg_data)
    );

    // Clock generation (10ns period)
    always #5 clk = ~clk;

    initial begin

        // Initialize Inputs
        clk = 0;
        r0addr = 0;
        r1addr = 0;
        wena = 0;
        waddr = 0;
        wdata = 0;
        ila_cpu_reg_addr = 0;

        // Setup VCD dump for waveform viewing
        $dumpfile("regfile_tb.vcd");
        $dumpvars(0, regfile_tb);

        $display("Starting Regfile Testbench...");

        // ------------------------------------------------------------
        // Test 1: Write to Register 1 and Read back
        // ------------------------------------------------------------
        #10;
        waddr = 5'd1;
        wdata = 32'hDEADBEEF;
        wena = 1;
        #10; // Wait for clock edge
        wena = 0; // Disable write
        
        // Read back on Port 0
        r0addr = 5'd1;
        #1; // Wait for combinational read
        if (r0data === 32'hDEADBEEF) 
            $display("[PASS] Test 1: Write/Read Reg 1. Got: %h", r0data);
        else 
            $display("[FAIL] Test 1: Write/Read Reg 1. Exp: DEADBEEF, Got: %h", r0data);

        // ------------------------------------------------------------
        // Test 2: Write to Register 2 and Dual Read (Port 0 & 1)
        // ------------------------------------------------------------
        #10;
        waddr = 5'd2;
        wdata = 32'hCAFEBABE;
        wena = 1;
        #10;
        wena = 0;

        // Read Reg 1 on Port 0, Reg 2 on Port 1
        r0addr = 5'd1;
        r1addr = 5'd2;
        #1;
        if (r0data === 32'hDEADBEEF && r1data === 32'hCAFEBABE)
            $display("[PASS] Test 2: Dual Port Read. R1: %h, R2: %h", r0data, r1data);
        else
            $display("[FAIL] Test 2: Dual Port Read. R1: %h, R2: %h", r0data, r1data);

        // ------------------------------------------------------------
        // Test 3: Write Enable (wena) Logic
        // ------------------------------------------------------------
        #10;
        waddr = 5'd1;       // Target Register 1 again
        wdata = 32'h00000000; // Try to clear it
        wena = 0;           // But wena is LOW
        #10;
        
        r0addr = 5'd1;
        #1;
        if (r0data === 32'hDEADBEEF)
            $display("[PASS] Test 3: Write Enable Low (Data preserved).");
        else
            $display("[FAIL] Test 3: Write Enable Low. Data changed to: %h", r0data);

        // ------------------------------------------------------------
        // Test 4: ILA Debug Port
        // ------------------------------------------------------------
        #10;
        ila_cpu_reg_addr = 5'd2; // Look at Register 2
        #1;
        if (ila_cpu_reg_data === 32'hCAFEBABE)
            $display("[PASS] Test 4: ILA Port Read. Got: %h", ila_cpu_reg_data);
        else
            $display("[FAIL] Test 4: ILA Port Read. Exp: CAFEBABE, Got: %h", ila_cpu_reg_data);

        // ------------------------------------------------------------
        // Test 5: Register 0 Behavior
        // Note: Based on your code, R0 is NOT hardwired to 0.
        // It acts like a normal register.
        // ------------------------------------------------------------
        #10;
        waddr = 5'd0;
        wdata = 32'h12345678;
        wena = 1;
        #10;
        wena = 0;
        
        r0addr = 5'd0;
        #1;
        if (r0data === 32'h12345678)
            $display("[PASS] Test 5: Reg 0 Write/Read (Normal RAM behavior confirmed).");
        else
            $display("[FAIL] Test 5: Reg 0 Write/Read. Got: %h", r0data);

        $display("Testbench Complete.");
        $finish;
    end

endmodule