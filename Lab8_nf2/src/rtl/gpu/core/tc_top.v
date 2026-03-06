/* file: tc_top.v
 Description: This file implements the top-level Tensor Core module.
 Author: Jeremy Cai
 Date: Feb. 28, 2026
 Version: 1.1
 Revision history:
    - Feb. 28, 2026: v1.0 — Initial implementation.
    - Mar. 02, 2026: v1.1 — Register rf_addr_override and rf_r*_addr
    - Mar. 06, 2026: v1.2 — Serialize scatter from 3-write×2-cycle to 1-write×4-cycle.
*/

`ifndef TC_TOP_V
`define TC_TOP_V

`include "gpu_define.v"
`include "tensor_core.v"

module tc_top (
    input wire clk,
    input wire rst_n,

    // Trigger from SM decode qualifier
    input wire trigger,

    // Register base addresses (latched from WMMA.MMA)
    input wire [3:0] dec_rA_addr,
    input wire [3:0] dec_rB_addr,
    input wire [3:0] dec_rC_addr,
    input wire [3:0] dec_rD_addr,

    // RF read data from 4 SPs (flat: {SP3,SP2,SP1,SP0})
    input wire [4*16-1:0] sp_rf_r0_data,
    input wire [4*16-1:0] sp_rf_r1_data,
    input wire [4*16-1:0] sp_rf_r2_data,
    input wire [4*16-1:0] sp_rf_r3_data,

    // Status
    output wire busy,

    // RF read address override (REGISTERED — v1.1)
    output reg rf_addr_override,
    output reg [3:0] rf_r0_addr,
    output reg [3:0] rf_r1_addr,
    output reg [3:0] rf_r2_addr,
    output reg [3:0] rf_r3_addr,

    // Scatter write port — single channel (v1.2)
    // Shared addr across all SPs; per-SP data and write-enable
    output reg [3:0] scat_w_addr,
    output reg [4*16-1:0] scat_w_data,
    output reg [3:0] scat_w_we
);

    // ================================================================
    // FSM states
    // ================================================================
    localparam [2:0] TC_IDLE     = 3'd0,
                     TC_GATHER_A = 3'd1,
                     TC_GATHER_B = 3'd2,
                     TC_GATHER_C = 3'd3,
                     TC_COMPUTE  = 3'd4,
                     TC_SCATTER0 = 3'd5,
                     TC_SCATTER1 = 3'd6,
                     TC_SCATTER2 = 3'd7;

    // TC_SCATTER3 uses the 3-bit overflow (3'd0 would alias TC_IDLE),
    // so we use a separate 1-bit flag for the final scatter beat.
    reg scatter3;

    reg [2:0] state;
    assign busy = (state != TC_IDLE) | scatter3;

    // Latched register base addresses
    reg [3:0] rA_base, rB_base, rC_base, rD_base;

    // Gather latches: 4 threads × 4 regs × 16 bits = 256 bits
    reg [255:0] a_hold, b_hold, c_hold;

    // Compute start pulse
    reg compute_start;
    wire tc_valid_in = (state == TC_COMPUTE) & compute_start;

    // ================================================================
    // tensor_core instance
    // ================================================================
    wire [255:0] matrix_d;
    wire tc_valid_out;

    tensor_core u_tc (
        .clk(clk), .rst_n(rst_n),
        .matrix_a(a_hold),
        .matrix_b(b_hold),
        .matrix_c(c_hold),
        .valid_in(tc_valid_in),
        .matrix_d(matrix_d),
        .valid_out(tc_valid_out)
    );

    // ================================================================
    // Latch matrix_d when tensor_core produces valid output.
    // This holds the result stable across all 4 scatter cycles.
    // ================================================================
    reg [255:0] d_hold;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            d_hold <= 256'd0;
        else if (tc_valid_out)
            d_hold <= matrix_d;
    end

    // ================================================================
    // FSM + Registered RF Override (v1.1, unchanged)
    // Scatter serialized to 4 beats (v1.2)
    // ================================================================
    integer gi;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= TC_IDLE;
            scatter3 <= 1'b0;
            rA_base <= 4'd0;
            rB_base <= 4'd0;
            rC_base <= 4'd0;
            rD_base <= 4'd0;
            a_hold <= 256'd0;
            b_hold <= 256'd0;
            c_hold <= 256'd0;
            compute_start <= 1'b0;
            rf_addr_override <= 1'b0;
            rf_r0_addr <= 4'd0;
            rf_r1_addr <= 4'd0;
            rf_r2_addr <= 4'd0;
            rf_r3_addr <= 4'd0;
        end else begin
            compute_start <= 1'b0;
            scatter3 <= 1'b0;
            case (state)
                TC_IDLE: begin
                    if (trigger & ~scatter3) begin
                        rA_base <= dec_rA_addr;
                        rB_base <= dec_rB_addr;
                        rC_base <= dec_rC_addr;
                        rD_base <= dec_rD_addr;
                        // Register override: RF will read rA registers next cycle
                        rf_addr_override <= 1'b1;
                        rf_r0_addr <= dec_rA_addr;
                        rf_r1_addr <= dec_rA_addr + 4'd1;
                        rf_r2_addr <= dec_rA_addr + 4'd2;
                        rf_r3_addr <= dec_rA_addr + 4'd3;
                        state <= TC_GATHER_A;
                    end
                end
                TC_GATHER_A: begin
                    // RF reads from rA+offsets are stable — capture matrix A
                    for (gi = 0; gi < 4; gi = gi + 1) begin
                        a_hold[gi*64 + 0*16 +: 16] <= sp_rf_r0_data[gi*16 +: 16];
                        a_hold[gi*64 + 1*16 +: 16] <= sp_rf_r1_data[gi*16 +: 16];
                        a_hold[gi*64 + 2*16 +: 16] <= sp_rf_r2_data[gi*16 +: 16];
                        a_hold[gi*64 + 3*16 +: 16] <= sp_rf_r3_data[gi*16 +: 16];
                    end
                    // Setup addresses for GATHER_B
                    rf_r0_addr <= rB_base;
                    rf_r1_addr <= rB_base + 4'd1;
                    rf_r2_addr <= rB_base + 4'd2;
                    rf_r3_addr <= rB_base + 4'd3;
                    state <= TC_GATHER_B;
                end
                TC_GATHER_B: begin
                    // RF reads from rB+offsets are stable — capture matrix B
                    for (gi = 0; gi < 4; gi = gi + 1) begin
                        b_hold[gi*64 + 0*16 +: 16] <= sp_rf_r0_data[gi*16 +: 16];
                        b_hold[gi*64 + 1*16 +: 16] <= sp_rf_r1_data[gi*16 +: 16];
                        b_hold[gi*64 + 2*16 +: 16] <= sp_rf_r2_data[gi*16 +: 16];
                        b_hold[gi*64 + 3*16 +: 16] <= sp_rf_r3_data[gi*16 +: 16];
                    end
                    // Setup addresses for GATHER_C
                    rf_r0_addr <= rC_base;
                    rf_r1_addr <= rC_base + 4'd1;
                    rf_r2_addr <= rC_base + 4'd2;
                    rf_r3_addr <= rC_base + 4'd3;
                    state <= TC_GATHER_C;
                end
                TC_GATHER_C: begin
                    // RF reads from rC+offsets are stable — capture matrix C
                    for (gi = 0; gi < 4; gi = gi + 1) begin
                        c_hold[gi*64 + 0*16 +: 16] <= sp_rf_r0_data[gi*16 +: 16];
                        c_hold[gi*64 + 1*16 +: 16] <= sp_rf_r1_data[gi*16 +: 16];
                        c_hold[gi*64 + 2*16 +: 16] <= sp_rf_r2_data[gi*16 +: 16];
                        c_hold[gi*64 + 3*16 +: 16] <= sp_rf_r3_data[gi*16 +: 16];
                    end
                    // Clear override — no more gather reads
                    rf_addr_override <= 1'b0;
                    compute_start <= 1'b1;
                    state <= TC_COMPUTE;
                end
                TC_COMPUTE: begin
                    if (tc_valid_out)
                        state <= TC_SCATTER0;
                end
                TC_SCATTER0: state <= TC_SCATTER1;
                TC_SCATTER1: state <= TC_SCATTER2;
                TC_SCATTER2: begin
                    // Next cycle is the final scatter beat.
                    // State returns to TC_IDLE, but scatter3 keeps busy=1
                    // for one more cycle so the write completes.
                    scatter3 <= 1'b1;
                    state <= TC_IDLE;
                end
                default: state <= TC_IDLE;
            endcase
        end
    end

    // ================================================================
    // Scatter output — serialized 1-write-per-cycle (v1.2)
    //
    // Reads from d_hold (latched on tc_valid_out, stable across all
    // scatter beats). Each beat writes one column of the 4×4 result.
    //
    // Beat mapping (same data layout as v1.0/v1.1):
    //   SCATTER0: rD+0, col 0 — row i data at d_hold[i*64 + 0*16 +: 16]
    //   SCATTER1: rD+1, col 1 — row i data at d_hold[i*64 + 1*16 +: 16]
    //   SCATTER2: rD+2, col 2 — row i data at d_hold[i*64 + 2*16 +: 16]
    //   scatter3: rD+3, col 3 — row i data at d_hold[i*64 + 3*16 +: 16]
    // ================================================================
    always @(*) begin
        scat_w_addr = 4'd0;
        scat_w_data = 64'd0;
        scat_w_we = 4'd0;

        if (state == TC_SCATTER0) begin
            scat_w_addr = rD_base;
            scat_w_we = 4'b1111;
            scat_w_data = {d_hold[3*64+0*16 +: 16], d_hold[2*64+0*16 +: 16],
                           d_hold[1*64+0*16 +: 16], d_hold[0*64+0*16 +: 16]};
        end else if (state == TC_SCATTER1) begin
            scat_w_addr = rD_base + 4'd1;
            scat_w_we = 4'b1111;
            scat_w_data = {d_hold[3*64+1*16 +: 16], d_hold[2*64+1*16 +: 16],
                           d_hold[1*64+1*16 +: 16], d_hold[0*64+1*16 +: 16]};
        end else if (state == TC_SCATTER2) begin
            scat_w_addr = rD_base + 4'd2;
            scat_w_we = 4'b1111;
            scat_w_data = {d_hold[3*64+2*16 +: 16], d_hold[2*64+2*16 +: 16],
                           d_hold[1*64+2*16 +: 16], d_hold[0*64+2*16 +: 16]};
        end else if (scatter3) begin
            scat_w_addr = rD_base + 4'd3;
            scat_w_we = 4'b1111;
            scat_w_data = {d_hold[3*64+3*16 +: 16], d_hold[2*64+3*16 +: 16],
                           d_hold[1*64+3*16 +: 16], d_hold[0*64+3*16 +: 16]};
        end
    end

endmodule

`endif // TC_TOP_V