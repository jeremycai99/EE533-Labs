/* file: cpu_tb.v
 * Description: CPU Testbench
 *   Verifies the CPU standalone using the same instruction encoding
 *   and logic as the SoC testbench.
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

    // ========================================================================
    // CPU Interface Signals
    // ========================================================================
    reg  [`INSTR_WIDTH-1:0]     i_mem_data_i;
    wire [`PC_WIDTH-1:0]        i_mem_addr_o;

    reg  [`DATA_WIDTH-1:0]      d_mem_data_i;
    wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o;
    wire [`DATA_WIDTH-1:0]      d_mem_data_o;
    wire                        d_mem_wen_o;
    wire                        cpu_done;

    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    cpu u_cpu (
        .clk          (clk),
        .rst_n        (rst_n),
        .i_mem_data_i (i_mem_data_i),
        .i_mem_addr_o (i_mem_addr_o),
        .d_mem_data_i (d_mem_data_i),
        .d_mem_addr_o (d_mem_addr_o),
        .d_mem_data_o (d_mem_data_o),
        .d_mem_wen_o  (d_mem_wen_o),
        .cpu_done     (cpu_done)
    );

    // ========================================================================
    // Memory Models
    // ========================================================================
    // Note: Using Word Addressing (Index = Address) to match soc_tb.v behavior.
    
    // --- Instruction Memory ---
    localparam I_MEM_DEPTH = 64; 
    reg [`INSTR_WIDTH-1:0] i_mem [0:I_MEM_DEPTH-1];

    always @(*) begin
        // Async read for IMEM
        i_mem_data_i = i_mem[i_mem_addr_o[5:0]]; 
    end

    // --- Data Memory ---
    localparam D_MEM_DEPTH = 64; 
    reg [`DATA_WIDTH-1:0] d_mem [0:D_MEM_DEPTH-1];

    // Async read for DMEM (CPU pipeline registers handle the timing)
    always @(*) begin
        d_mem_data_i = d_mem[d_mem_addr_o[5:0]];
    end

    // Sync write for DMEM
    always @(posedge clk) begin
        if (d_mem_wen_o) begin
            d_mem[d_mem_addr_o[5:0]] <= d_mem_data_o;
            $display("[%0t] Data Memory WRITE: Addr=%0d Data=%0d (0x%h)", 
                     $time, d_mem_addr_o, d_mem_data_o, d_mem_data_o);
        end
    end

    // ========================================================================
    // Instruction Builder (Matches cpu.v decoding logic)
    // ========================================================================
    localparam [`REG_ADDR_WIDTH-1:0] R0 = 0;
    localparam [`REG_ADDR_WIDTH-1:0] R1 = 1;
    localparam [`REG_ADDR_WIDTH-1:0] R2 = 2;
    localparam [`REG_ADDR_WIDTH-1:0] R3 = 3;

    function [`INSTR_WIDTH-1:0] build_instr;
        input mw;                       // Mem Write (Bit 31)
        input rw;                       // Reg Write (Bit 30)
        input [`REG_ADDR_WIDTH-1:0] rd; // Dest      (Bits 23:21)
        input [`REG_ADDR_WIDTH-1:0] r0; // Addr Base (Bits 27:24)
        input [`REG_ADDR_WIDTH-1:0] r1; // Data Src  (Bits 29:27)
        
        reg [31:0] tmp;
        begin
            tmp = 32'b0;
            tmp[31] = mw;
            tmp[30] = rw;
            
            // r1 (Data Source) -> Bits 29:27
            tmp[29:27] = r1[2:0];
            
            // r0 (Addr Source) -> Bits 27:24
            // Note: Bit 27 is shared. We OR them to combine.
            // This logic works for the specific R2/R3 cases in the test.
            tmp[27:24] = tmp[27:24] | r0[3:0];
            
            // rd (Destination) -> Bits 23:21
            tmp[23:21] = rd[2:0];
            
            build_instr = tmp;
        end
    endfunction

    localparam [`INSTR_WIDTH-1:0] NOP = {`INSTR_WIDTH{1'b0}};

    // ========================================================================
    // Main Test Sequence
    // ========================================================================
    integer k, i;
    integer error_cnt;

    // Initialize Register File (Backdoor access to avoid X propagation)
    initial begin
        for (k = 0; k < 32; k = k + 1) begin
            u_cpu.u_regfile.regfile[k] = 64'd0;
        end
    end

    initial begin
        $dumpfile("cpu_tb.vcd");
        $dumpvars(0, cpu_tb);

        // 1. Initialize
        rst_n = 0;
        error_cnt = 0;
        
        for (i = 0; i < I_MEM_DEPTH; i = i + 1) i_mem[i] = NOP;
        for (i = 0; i < D_MEM_DEPTH; i = i + 1) d_mem[i] = 0;

        // 2. Load Data Memory
        // soc_tb.v setup: d_mem[0]=4, d_mem[4]=100
        d_mem[0] = 32'd4;
        d_mem[4] = 32'd100;

        // 3. Load Instruction Memory
        // Op 0: Load R2 from d_mem[R0] (R0=0 -> Addr 0 -> Val 4)
        //       MW=0, RW=1, RD=R2, R0=R0, R1=0
        i_mem[0] = build_instr(1'b0, 1'b1, R2, R0, R0);

        // Op 1: Load R3 from d_mem[R0] (R0=0 -> Addr 0 -> Val 4)
        //       MW=0, RW=1, RD=R3, R0=R0, R1=0
        i_mem[1] = build_instr(1'b0, 1'b1, R3, R0, R0);

        // NOPs for pipeline hazards
        i_mem[2] = NOP;
        i_mem[3] = NOP;
        i_mem[4] = NOP;

        // Op 5: Store R2 to d_mem[R3] (R3=4 -> Addr 4)
        //       MW=1, RW=0, RD=0, R0=R3(Addr), R1=R2(Data)
        //       Writes value 4 (from R2) to Addr 4.
        i_mem[5] = build_instr(1'b1, 1'b0, R0, R3, R2);

        // 4. Display Test Plan
        $display("");
        $display("==================================================");
        $display("  CPU Testbench (Synced with soc_tb.v decoding)");
        $display("==================================================");
        $display("Initial State:");
        $display("  d_mem[0] = %0d (Addr for indirect access)", d_mem[0]);
        $display("  d_mem[4] = %0d (Target to overwrite)", d_mem[4]);
        $display("--------------------------------------------------");
        
        // 5. Run Test
        repeat (3) @(posedge clk);
        rst_n = 1;
        $display("[%0t] Reset Released", $time);

        // Wait for execution (Fetch + Decode + Ex + Mem + WB)
        repeat (20) @(posedge clk);

        // 6. Verify Results
        $display("--------------------------------------------------");
        $display("Final State:");
        $display("  d_mem[0] = %0d (Expected 4)", d_mem[0]);
        $display("  d_mem[4] = %0d (Expected 4)", d_mem[4]);
        $display("");

        // Check Target
        if (d_mem[4] !== 32'd4) begin
            $display("  [FAIL] d_mem[4] = %0d (Expected 4)", d_mem[4]);
            error_cnt = error_cnt + 1;
        end else begin
            $display("  [PASS] Store successful (100 -> 4).");
        end

        // Check Source Integrity
        if (d_mem[0] !== 32'd4) begin
            $display("  [FAIL] d_mem[0] corrupted! val=%0d", d_mem[0]);
            error_cnt = error_cnt + 1;
        end else begin
            $display("  [PASS] Source memory unchanged.");
        end

        $display("==================================================");
        if (error_cnt == 0) $display("  >>> ALL TESTS PASSED <<<");
        else                $display("  >>> %0d FAILURES <<<", error_cnt);
        $display("==================================================");
        
        $finish;
    end

endmodule