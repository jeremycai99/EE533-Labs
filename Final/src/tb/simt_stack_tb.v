/* file: tb_simt_stack.v
 * Testbench for simt_stack.v
 * Covers:
 *   T1: Reset state (empty, outputs don't-care)
 *   T2: Single push + TOS read
 *   T3: Modify TOS (phase 0→1)
 *   T4: Pop + verify restored TOS
 *   T5: Nested divergence (push × 3, verify LIFO order)
 *   T6: Full stack guard (push at DEPTH=8)
 *   T7: Empty stack guard (pop when empty)
 *   T8: Full divergence/convergence cycle (push → modify → pop)
 */

`timescale 1ns / 1ps

`include "gpu_define.v"
`include "simt_stack.v"

module simt_stack_tb;

    reg clk, rst_n;
    reg push, pop, modify_tos;
    reg [`GPU_PC_WIDTH-1:0] push_reconv_pc, push_pend_pc;
    reg [3:0] push_saved_mask, push_pend_mask;

    wire [`GPU_PC_WIDTH-1:0] tos_reconv_pc, tos_pend_pc;
    wire [3:0] tos_saved_mask, tos_pend_mask;
    wire tos_phase;
    wire stack_empty, stack_full;

    simt_stack #(.DEPTH(8)) uut (
        .clk(clk), .rst_n(rst_n),
        .push(push), .push_reconv_pc(push_reconv_pc),
        .push_saved_mask(push_saved_mask),
        .push_pend_mask(push_pend_mask),
        .push_pend_pc(push_pend_pc),
        .pop(pop), .modify_tos(modify_tos),
        .tos_reconv_pc(tos_reconv_pc),
        .tos_saved_mask(tos_saved_mask),
        .tos_pend_mask(tos_pend_mask),
        .tos_pend_pc(tos_pend_pc),
        .tos_phase(tos_phase),
        .stack_empty(stack_empty),
        .stack_full(stack_full)
    );

    // Clock: 10ns period
    always #5 clk = ~clk;

    integer pass_count, fail_count;

    task check(input [255:0] label, input cond);
    begin
        if (cond) begin
            $display("  [PASS] %0s", label);
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] %0s", label);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task do_push(
        input [`GPU_PC_WIDTH-1:0] reconv,
        input [3:0] saved,
        input [3:0] pend,
        input [`GPU_PC_WIDTH-1:0] pend_pc_val
    );
    begin
        @(negedge clk);
        push <= 1'b1;
        push_reconv_pc <= reconv;
        push_saved_mask <= saved;
        push_pend_mask <= pend;
        push_pend_pc <= pend_pc_val;
        @(posedge clk); #1;
        push <= 1'b0;
    end
    endtask

    task do_pop;
    begin
        @(negedge clk);
        pop <= 1'b1;
        @(posedge clk); #1;
        pop <= 1'b0;
    end
    endtask

    task do_modify;
    begin
        @(negedge clk);
        modify_tos <= 1'b1;
        @(posedge clk); #1;
        modify_tos <= 1'b0;
    end
    endtask

    initial begin
        $dumpfile("simt_stack_tb.vcd");
        $dumpvars(0, simt_stack_tb);

        clk = 0; rst_n = 0;
        push = 0; pop = 0; modify_tos = 0;
        push_reconv_pc = 0; push_saved_mask = 0;
        push_pend_mask = 0; push_pend_pc = 0;
        pass_count = 0; fail_count = 0;

        // Reset
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        // ────────────────────────────────────────────────
        $display("\nT1: Reset state");
        check("stack_empty after reset", stack_empty == 1'b1);
        check("stack_full deasserted",   stack_full == 1'b0);

        // ────────────────────────────────────────────────
        $display("\nT2: Single push + TOS read");
        // Simulate PBRA: reconv=0x20, saved=1111, pend=1100, pend_pc=0x11
        do_push(32'h0000_0020, 4'b1111, 4'b1100, 32'h0000_0011);
        check("not empty after push",       stack_empty == 1'b0);
        check("not full after 1 push",       stack_full == 1'b0);
        check("tos_reconv_pc == 0x20",       tos_reconv_pc == 32'h20);
        check("tos_saved_mask == 4'b1111",   tos_saved_mask == 4'b1111);
        check("tos_pend_mask == 4'b1100",    tos_pend_mask == 4'b1100);
        check("tos_pend_pc == 0x11",         tos_pend_pc == 32'h11);
        check("tos_phase == 0 (taken-first)", tos_phase == 1'b0);

        // ────────────────────────────────────────────────
        $display("\nT3: Modify TOS (phase 0 → 1)");
        do_modify;
        check("tos_phase == 1 after modify", tos_phase == 1'b1);
        check("tos_reconv_pc unchanged",     tos_reconv_pc == 32'h20);
        check("tos_saved_mask unchanged",    tos_saved_mask == 4'b1111);
        check("tos_pend_mask unchanged",     tos_pend_mask == 4'b1100);
        check("tos_pend_pc unchanged",       tos_pend_pc == 32'h11);

        // ────────────────────────────────────────────────
        $display("\nT4: Pop + verify empty");
        do_pop;
        check("stack_empty after pop", stack_empty == 1'b1);

        // ────────────────────────────────────────────────
        $display("\nT5: Nested divergence (3-deep LIFO order)");
        // Level 0: outer if
        do_push(32'h0000_0030, 4'b1111, 4'b1010, 32'h0000_0015);
        // Level 1: inner if (only T0,T2 active → they diverge further)
        do_push(32'h0000_0028, 4'b0101, 4'b0100, 32'h0000_001A);
        // Level 2: deepest branch
        do_push(32'h0000_0025, 4'b0001, 4'b0000, 32'h0000_001D);

        check("TOS is level-2 reconv",   tos_reconv_pc == 32'h25);
        check("TOS is level-2 saved",    tos_saved_mask == 4'b0001);
        check("TOS is level-2 pend",     tos_pend_mask == 4'b0000);

        // Pop level 2
        do_pop;
        check("TOS is level-1 reconv",   tos_reconv_pc == 32'h28);
        check("TOS is level-1 saved",    tos_saved_mask == 4'b0101);
        check("TOS is level-1 pend_mask", tos_pend_mask == 4'b0100);

        // Pop level 1
        do_pop;
        check("TOS is level-0 reconv",   tos_reconv_pc == 32'h30);
        check("TOS is level-0 saved",    tos_saved_mask == 4'b1111);
        check("TOS is level-0 pend_pc",  tos_pend_pc == 32'h15);

        // Pop level 0
        do_pop;
        check("stack_empty after 3 pops", stack_empty == 1'b1);

        // ────────────────────────────────────────────────
        $display("\nT6: Full stack guard (push x8, then push x1 ignored)");
        begin : fill_stack
            integer j;
            for (j = 0; j < 8; j = j + 1)
                do_push(j * 4, 4'b1111, 4'b0011, j * 4 + 1);
        end
        check("stack_full after 8 pushes", stack_full == 1'b1);
        check("TOS is entry 7 reconv",    tos_reconv_pc == 32'd28);

        // 9th push should be silently ignored
        do_push(32'hDEAD, 4'b1111, 4'b1111, 32'hBEEF);
        check("still full (9th push ignored)", stack_full == 1'b1);
        check("TOS unchanged after overflow", tos_reconv_pc == 32'd28);

        // Drain all
        begin : drain_stack
            integer j;
            for (j = 0; j < 8; j = j + 1)
                do_pop;
        end
        check("stack_empty after drain", stack_empty == 1'b1);

        // ────────────────────────────────────────────────
        $display("\nT7: Empty stack guard (pop when empty)");
        do_pop;
        check("still empty after pop-when-empty", stack_empty == 1'b1);

        // ────────────────────────────────────────────────
        $display("\nT8: Full divergence/convergence cycle");
        // PBRA: T0,T1 taken → branch_target; T2,T3 fall → PC+1
        // reconv_pc = 0x40
        do_push(32'h0000_0040, 4'b1111, 4'b1100, 32'h0000_0021);
        check("phase 0: taken path runs",  tos_phase == 1'b0);
        check("saved_mask = full",          tos_saved_mask == 4'b1111);

        // Taken path finishes, PC reaches reconv_pc → phase 0→1
        do_modify;
        check("phase 1: fall path runs",   tos_phase == 1'b1);
        check("pend_mask = 1100",           tos_pend_mask == 4'b1100);
        check("pend_pc = 0x21",             tos_pend_pc == 32'h21);

        // Fall path finishes, PC reaches reconv_pc again → pop
        do_pop;
        check("converged: stack empty",     stack_empty == 1'b1);

        // ────────────────────────────────────────────────
        $display("\n================================================");
        $display("Results: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("================================================\n");

        $finish;
    end

endmodule