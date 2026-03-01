/* tb_scoreboard.v
 * Testbench for per-thread scoreboard (scalar, no WMMA).
 *
 * Test sections:
 *   1. Basic RAW hazard: back-to-back dependent instructions → stall
 *   2. No hazard: independent registers → no stall
 *   3. Same-cycle WB bypass: stall clears on the cycle WB commits
 *   4. SET priority over CLEAR: new issue to same reg keeps it pending
 *   5. FMA: rD read hazard (3-operand)
 *   6. ST/STS: rD-as-source hazard
 *   7. Per-thread divergence: hazard in inactive thread doesn't stall
 *   8. Multiple threads with mixed hazards
 *   9. Back-to-back issue sequence (no stalls)
 *  10. Reset clears all pending bits
 */

`timescale 1ns / 1ps

`include "scoreboard.v"

module scoreboard_tb;

    reg         clk, rst_n;
    reg  [3:0]  rA_addr, rB_addr, rD_addr;
    reg         uses_rA, uses_rB, is_fma, is_st, rf_we;
    reg  [3:0]  active_mask;
    reg         issue;
    reg  [3:0]  wb_rD_addr;
    reg         wb_rf_we;
    reg  [3:0]  wb_active_mask;
    wire        stall;

    scoreboard u_dut (
        .clk(clk), .rst_n(rst_n),
        .rA_addr(rA_addr), .rB_addr(rB_addr), .rD_addr(rD_addr),
        .uses_rA(uses_rA), .uses_rB(uses_rB),
        .is_fma(is_fma), .is_st(is_st), .rf_we(rf_we),
        .active_mask(active_mask),
        .issue(issue),
        .wb_rD_addr(wb_rD_addr), .wb_rf_we(wb_rf_we),
        .wb_active_mask(wb_active_mask),
        .stall(stall)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Test infrastructure ──────────────────────────────────
    integer test_num, pass_count, fail_count;

    task check(input expected_stall, input [255:0] msg);
        begin
            test_num = test_num + 1;
            if (stall === expected_stall) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: %0s | stall=%b expected=%b",
                         test_num, msg, stall, expected_stall);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0s | stall=%b expected=%b  <<<",
                         test_num, msg, stall, expected_stall);
            end
        end
    endtask

    // Helper: clear all inputs to idle state
    task idle;
        begin
            rA_addr = 4'd0; rB_addr = 4'd0; rD_addr = 4'd0;
            uses_rA = 0; uses_rB = 0; is_fma = 0; is_st = 0; rf_we = 0;
            issue = 0;
            wb_rD_addr = 4'd0; wb_rf_we = 0; wb_active_mask = 4'b0000;
        end
    endtask

    // Helper: simulate issuing an R-type instruction (reads rA, rB; writes rD)
    task issue_rtype(input [3:0] rd, ra, rb);
        begin
            @(negedge clk);
            rD_addr = rd; rA_addr = ra; rB_addr = rb;
            uses_rA = 1; uses_rB = 1; is_fma = 0; is_st = 0; rf_we = 1;
            issue = 1;
            @(negedge clk);
            issue = 0;
        end
    endtask

    // Helper: assert WB commit for one cycle
    task wb_commit(input [3:0] rd, input [3:0] mask);
        begin
            @(negedge clk);
            wb_rD_addr = rd; wb_rf_we = 1; wb_active_mask = mask;
            @(negedge clk);
            wb_rf_we = 0; wb_active_mask = 4'b0000;
        end
    endtask

    // ── Main test sequence ───────────────────────────────────
    initial begin
        $dumpfile("scoreboard_tb.vcd");
        $dumpvars(0, scoreboard_tb);

        test_num = 0; pass_count = 0; fail_count = 0;
        idle;
        active_mask = 4'b1111;  // all threads active

        // Reset
        rst_n = 0;
        @(posedge clk); @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("═══════════════════════════════════════════════════");
        $display("  Scoreboard Testbench");
        $display("═══════════════════════════════════════════════════");

        // ═════════════════════════════════════════════════════
        // Section 1: Basic RAW hazard
        //   ADD R3, R1, R2   (writes R3)
        //   SUB R5, R3, R4   (reads R3 → should stall)
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 1: Basic RAW Hazard ---");

        // Issue ADD R3, R1, R2
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1; is_fma = 0; is_st = 0;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Now present SUB R5, R3, R4 (reads R3 which is pending)
        rD_addr = 4'd5; rA_addr = 4'd3; rB_addr = 4'd4;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b1, "RAW on rA: R3 pending, SUB reads R3");

        // Also check rB hazard
        rA_addr = 4'd4; rB_addr = 4'd3;  // swap: R3 now on rB
        @(posedge clk); #1;
        check(1'b1, "RAW on rB: R3 pending, rB=R3");

        // Clean up: WB commit R3
        wb_rD_addr = 4'd3; wb_rf_we = 1; wb_active_mask = 4'b1111;
        @(posedge clk); #1;
        // stall should clear due to same-cycle bypass
        check(1'b0, "WB clears R3 same cycle → no stall");
        @(negedge clk);
        wb_rf_we = 0; wb_active_mask = 4'b0000;
        idle;

        // ═════════════════════════════════════════════════════
        // Section 2: No hazard (independent registers)
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 2: No Hazard ---");

        // Issue ADD R3, R1, R2
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Present MUL R6, R4, R5 (no overlap with R3)
        rD_addr = 4'd6; rA_addr = 4'd4; rB_addr = 4'd5;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b0, "Independent regs: R3 pending but R4,R5 not");

        // Clean up
        wb_commit(4'd3, 4'b1111);
        idle;

        // ═════════════════════════════════════════════════════
        // Section 3: Same-cycle WB bypass
        //   Pipeline trace:
        //   Cyc N:   WB writes R3, ID checks R3 → should NOT stall
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 3: Same-Cycle WB Bypass ---");

        // Set R3 pending
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Verify R3 is stalling
        rA_addr = 4'd3; rB_addr = 4'd4; rD_addr = 4'd5;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b1, "R3 pending → stall");

        // Now WB commits R3 in the same cycle as CHECK
        @(negedge clk);
        rA_addr = 4'd3; uses_rA = 1; uses_rB = 0; rf_we = 1;
        wb_rD_addr = 4'd3; wb_rf_we = 1; wb_active_mask = 4'b1111;
        @(posedge clk); #1;
        check(1'b0, "WB bypass: R3 cleared same cycle → no stall");

        @(negedge clk);
        wb_rf_we = 0;
        idle;

        // ═════════════════════════════════════════════════════
        // Section 4: SET priority over CLEAR
        //   WB clears R3, ID issues new write to R3 → R3 stays pending
        //   Next instruction reading R3 should stall
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 4: SET Priority Over CLEAR ---");

        // Issue first write to R3
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // WB clears R3 while simultaneously issuing a new write to R3
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd4; rB_addr = 4'd5;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;  // new instruction writing R3
        wb_rD_addr = 4'd3; wb_rf_we = 1; wb_active_mask = 4'b1111;
        @(negedge clk);
        issue = 0; wb_rf_we = 0;

        // Now check: R3 should still be pending (new write in flight)
        rD_addr = 4'd6; rA_addr = 4'd3; rB_addr = 4'd7;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b1, "SET>CLEAR: new R3 write keeps it pending");

        wb_commit(4'd3, 4'b1111);
        idle;

        // ═════════════════════════════════════════════════════
        // Section 5: FMA — rD is both read and write
        //   FMA R3, R1, R2  (reads R1, R2, R3; writes R3)
        //   If R3 is pending, FMA should stall on rD read
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 5: FMA rD Read Hazard ---");

        // Issue something that writes R3
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd4; rB_addr = 4'd5;
        uses_rA = 1; uses_rB = 1; rf_we = 1; is_fma = 0;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Present FMA R3, R1, R2 (rD=R3 is read as accumulator)
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; is_fma = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b1, "FMA: rD=R3 pending → stall (accumulator read)");

        // No hazard if only rA/rB are clean and rD is clean
        @(negedge clk);
        wb_rD_addr = 4'd3; wb_rf_we = 1; wb_active_mask = 4'b1111;
        @(negedge clk);
        wb_rf_we = 0;

        rD_addr = 4'd6; rA_addr = 4'd1; rB_addr = 4'd2;
        is_fma = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b0, "FMA: rD=R6 not pending → no stall");

        is_fma = 0;
        idle;

        // ═════════════════════════════════════════════════════
        // Section 6: ST/STS — rD is a source (store data)
        //   If R3 is pending and ST uses rD=R3, should stall
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 6: ST/STS rD-as-Source ---");

        // Issue write to R3
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Present ST: addr=rA(R4), data=rD(R3). rf_we=0 (ST doesn't write GPR)
        rD_addr = 4'd3; rA_addr = 4'd4; rB_addr = 4'd0;
        uses_rA = 1; uses_rB = 0; rf_we = 0; is_st = 1;
        @(posedge clk); #1;
        check(1'b1, "ST: rD=R3 pending as store source → stall");

        // Also check rA hazard for ST
        rA_addr = 4'd3; rD_addr = 4'd7;  // rA=R3 pending, rD=R7 clean
        @(posedge clk); #1;
        check(1'b1, "ST: rA=R3 pending → stall");

        // No stall when neither rA nor rD pending
        rA_addr = 4'd4; rD_addr = 4'd7;
        @(posedge clk); #1;
        check(1'b0, "ST: rA=R4 clean, rD=R7 clean → no stall");

        is_st = 0;
        wb_commit(4'd3, 4'b1111);
        idle;

        // ═════════════════════════════════════════════════════
        // Section 7: Per-thread divergence
        //   Taken path (T2,T3) writes R5. Fall-through (T0,T1)
        //   reads R5 — no real hazard because different threads.
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 7: Divergence — No False Stall ---");

        // Taken path: active_mask = 4'b1100 (T2, T3)
        active_mask = 4'b1100;
        @(negedge clk);
        rD_addr = 4'd5; rA_addr = 4'd6; rB_addr = 4'd7;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Verify: pending[2][R5]=1, pending[3][R5]=1
        //         pending[0][R5]=0, pending[1][R5]=0

        // Switch to fall-through: active_mask = 4'b0011 (T0, T1)
        active_mask = 4'b0011;
        rD_addr = 4'd8; rA_addr = 4'd5; rB_addr = 4'd9;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b0, "Divergence: T0,T1 read R5 — not pending for them");

        // But if T2,T3 were active, R5 IS pending
        active_mask = 4'b1100;
        @(posedge clk); #1;
        check(1'b1, "Divergence: T2,T3 read R5 — pending for them");

        // Full mask: any active thread with pending → stall
        active_mask = 4'b1111;
        @(posedge clk); #1;
        check(1'b1, "Full mask: T2,T3 have R5 pending → stall");

        // Clear R5 for T2,T3
        wb_commit(4'd5, 4'b1100);
        active_mask = 4'b1111;
        idle;

        // ═════════════════════════════════════════════════════
        // Section 8: Multiple pending registers
        //   Issue writes to R3 and R7, then read both
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 8: Multiple Pending Registers ---");

        // Issue write R3
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Issue write R7
        @(negedge clk);
        rD_addr = 4'd7; rA_addr = 4'd4; rB_addr = 4'd5;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Read R3 on rA → stall
        rD_addr = 4'd8; rA_addr = 4'd3; rB_addr = 4'd1;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        @(posedge clk); #1;
        check(1'b1, "R3 pending on rA → stall");

        // Read R7 on rB → stall
        rA_addr = 4'd1; rB_addr = 4'd7;
        @(posedge clk); #1;
        check(1'b1, "R7 pending on rB → stall");

        // Clear R3 only — R7 still pending
        @(negedge clk);
        wb_rD_addr = 4'd3; wb_rf_we = 1; wb_active_mask = 4'b1111;
        rA_addr = 4'd3; rB_addr = 4'd7;
        @(posedge clk); #1;
        check(1'b1, "R3 cleared but R7 still pending on rB → stall");

        @(negedge clk);
        wb_rf_we = 0;

        // Clear R7
        wb_commit(4'd7, 4'b1111);

        rA_addr = 4'd3; rB_addr = 4'd7;
        @(posedge clk); #1;
        check(1'b0, "Both R3 and R7 cleared → no stall");

        idle;

        // ═════════════════════════════════════════════════════
        // Section 9: Back-to-back independent issues (no stalls)
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 9: Back-to-Back Independent Issues ---");

        // Issue R3←R1,R2 then R7←R4,R5 then R8←R9,R10 (all independent reads)
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(posedge clk); #1;
        check(1'b0, "Issue R3←R1,R2: no stall (R1,R2 clean)");

        @(negedge clk);
        rD_addr = 4'd7; rA_addr = 4'd4; rB_addr = 4'd5;
        issue = 1;
        @(posedge clk); #1;
        check(1'b0, "Issue R7←R4,R5: no stall (R4,R5 clean)");

        @(negedge clk);
        rD_addr = 4'd8; rA_addr = 4'd9; rB_addr = 4'd10;
        issue = 1;
        @(posedge clk); #1;
        check(1'b0, "Issue R8←R9,R10: no stall (R9,R10 clean)");

        @(negedge clk);
        issue = 0;

        // Clean up all three
        wb_commit(4'd3, 4'b1111);
        wb_commit(4'd7, 4'b1111);
        wb_commit(4'd8, 4'b1111);
        idle;

        // ═════════════════════════════════════════════════════
        // Section 10: Reset clears all pending bits
        // ═════════════════════════════════════════════════════
        $display("\n--- Section 10: Reset ---");

        // Set R3 pending
        @(negedge clk);
        rD_addr = 4'd3; rA_addr = 4'd1; rB_addr = 4'd2;
        uses_rA = 1; uses_rB = 1; rf_we = 1;
        issue = 1;
        @(negedge clk);
        issue = 0;

        // Verify pending
        rA_addr = 4'd3; uses_rA = 1;
        @(posedge clk); #1;
        check(1'b1, "R3 pending before reset");

        // Assert reset
        @(negedge clk);
        rst_n = 0;
        @(posedge clk); @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        // Check: R3 no longer pending
        rA_addr = 4'd3; uses_rA = 1;
        @(posedge clk); #1;
        check(1'b0, "R3 cleared after reset");

        idle;

        // ═════════════════════════════════════════════════════
        // Summary
        // ═════════════════════════════════════════════════════
        $display("\n═══════════════════════════════════════════════════");
        $display("  Results: %0d PASS, %0d FAIL out of %0d tests",
                 pass_count, fail_count, test_num);
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  *** FAILURES DETECTED ***");
        $display("═══════════════════════════════════════════════════");

        #20;
        $finish;
    end

endmodule