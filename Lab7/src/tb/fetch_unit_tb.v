/* file: tb_fetch_unit.v
 * Testbench for fetch_unit.v
 * Verifies: startup latency, sequential fetch, stall hold,
 * branch bubble, RET/drain/kernel_done, back-to-back kernels.
 *
 * Author: Jeremy Cai
 * Date: Feb. 28, 2026
 */

`timescale 1ns / 1ps

`include "fetch_unit.v"

module fetch_unit_tb;

    reg clk, rst_n;
    reg kernel_start;
    reg [`GPU_PC_WIDTH-1:0] kernel_entry_pc;
    reg branch_taken;
    reg [`GPU_PC_WIDTH-1:0] branch_target;
    reg front_stall;
    reg ret_detected;

    wire [`GPU_IMEM_ADDR_WIDTH-1:0] imem_addr;
    wire kernel_done, running;
    wire [`GPU_PC_WIDTH-1:0] if_id_pc;
    wire fetch_valid;

    fetch_unit u_dut (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr),
        .kernel_start(kernel_start), .kernel_entry_pc(kernel_entry_pc),
        .kernel_done(kernel_done), .running(running),
        .branch_taken(branch_taken), .branch_target(branch_target),
        .front_stall(front_stall),
        .ret_detected(ret_detected),
        .if_id_pc(if_id_pc), .fetch_valid(fetch_valid)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    task check;
        input [255:0] name;
        input cond;
        begin
            test_num = test_num + 1;
            if (!cond) begin
                $display("FAIL [%0d] %0s @ %0t", test_num, name, $time);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
        end
    endtask

    task posedge_clk;
        begin
            @(posedge clk); #1; // settle after edge
        end
    endtask

    task clear_inputs;
        begin
            kernel_start = 0;
            kernel_entry_pc = 0;
            branch_taken = 0;
            branch_target = 0;
            front_stall = 0;
            ret_detected = 0;
        end
    endtask

    integer i;

    initial begin
        $dumpfile("fetch_unit_tb.vcd");
        $dumpvars(0, fetch_unit_tb);

        // Reset
        rst_n = 0;
        clear_inputs;
        repeat(2) posedge_clk;
        rst_n = 1;
        posedge_clk;

        // ========================================================
        // Test 1: Reset state
        // ========================================================
        $display("--- Test 1: Reset state ---");
        check("reset running=0", running == 0);
        check("reset fetch_valid=0", fetch_valid == 0);
        check("reset kernel_done=0", kernel_done == 0);

        // ========================================================
        // Test 2: Kernel startup (2-cycle latency)
        // ========================================================
        $display("--- Test 2: Kernel startup ---");
        kernel_start = 1;
        kernel_entry_pc = 32'h0010;
        posedge_clk;
        kernel_start = 0;
        // After posedge: pc=0x10, running=1, fetch_valid=0
        check("start: running=1", running == 1);
        check("start: fetch_valid=0", fetch_valid == 0);
        check("start: imem_addr=0x10", imem_addr == 8'h10);

        posedge_clk;
        // Cycle 1: fetch_valid=1, if_id_pc=0x10
        check("c1: fetch_valid=1", fetch_valid == 1);
        check("c1: if_id_pc=0x10", if_id_pc == 32'h10);
        check("c1: imem_addr=0x11", imem_addr == 8'h11);

        posedge_clk;
        check("c2: if_id_pc=0x11", if_id_pc == 32'h11);
        check("c2: fetch_valid=1", fetch_valid == 1);

        // ========================================================
        // Test 3: Front stall
        // ========================================================
        $display("--- Test 3: Front stall ---");
        // Currently if_id_pc=0x11
        front_stall = 1;
        posedge_clk;
        check("stall: if_id_pc held=0x11", if_id_pc == 32'h11);
        check("stall: fetch_valid held=1", fetch_valid == 1);
        posedge_clk;
        check("stall2: if_id_pc=0x11", if_id_pc == 32'h11);
        front_stall = 0;

        posedge_clk;
        check("unstall: if_id_pc=0x12", if_id_pc == 32'h12);
        check("unstall: fetch_valid=1", fetch_valid == 1);

        // ========================================================
        // Test 4: Branch taken (1-cycle bubble)
        // ========================================================
        $display("--- Test 4: Branch taken ---");
        posedge_clk; // let it run one more cycle
        branch_taken = 1;
        branch_target = 32'h0050;
        posedge_clk;
        branch_taken = 0;
        check("branch: fetch_valid=0", fetch_valid == 0);
        check("branch: imem_addr=0x50", imem_addr == 8'h50);

        posedge_clk;
        check("post-br: fetch_valid=1", fetch_valid == 1);
        check("post-br: if_id_pc=0x50", if_id_pc == 32'h50);

        posedge_clk;
        check("post-br+1: if_id_pc=0x51", if_id_pc == 32'h51);

        // ========================================================
        // Test 5: RET → drain → kernel_done
        // ========================================================
        $display("--- Test 5: RET and kernel_done ---");
        ret_detected = 1;
        posedge_clk;
        ret_detected = 0;
        check("ret: running=0", running == 0);
        check("ret: fetch_valid=0", fetch_valid == 0);
        check("ret: kernel_done=0", kernel_done == 0);

        posedge_clk;
        check("drain1: done=0", kernel_done == 0);
        posedge_clk;
        check("drain2: done=0", kernel_done == 0);
        posedge_clk;
        check("drain3: done=1", kernel_done == 1);
        posedge_clk;
        check("drain4: done=0", kernel_done == 0);

        // ========================================================
        // Test 6: Back-to-back kernel
        // ========================================================
        $display("--- Test 6: Back-to-back kernel ---");
        clear_inputs;
        kernel_start = 1;
        kernel_entry_pc = 32'h0080;
        posedge_clk;
        kernel_start = 0;
        check("k2: running=1", running == 1);
        check("k2: fetch_valid=0", fetch_valid == 0);

        posedge_clk;
        check("k2c1: fetch_valid=1", fetch_valid == 1);
        check("k2c1: if_id_pc=0x80", if_id_pc == 32'h80);

        posedge_clk;
        check("k2c2: if_id_pc=0x81", if_id_pc == 32'h81);

        // ========================================================
        // Summary
        // ========================================================
        $display("");
        $display("=== RESULTS: %0d PASSED, %0d FAILED (of %0d) ===",
                 pass_count, fail_count, test_num);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("*** FAILURES DETECTED ***");

        $finish;
    end

endmodule