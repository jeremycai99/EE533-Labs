/* file: simt_stack.v
 * Description: SIMT divergence stack for the SM core.
 *   8-deep LIFO storing divergence context for predicated branches (PBRA).
 *   Each entry holds: reconvergence PC, saved active mask, pending mask,
 *   pending PC (fall-through target), and a phase bit.
 *
 *   Operations:
 *     push       — push new divergence context, SP++
 *     pop        — discard TOS, SP--
 *     modify_tos — update TOS.phase in place (phase 0→1 transition)
 *     clear      — reset stack pointer to 0 (kernel restart)
 *
 *   TOS is always visible combinationally via tos_* outputs.
 *   Stack empty/full status is provided for guard logic in sm_core.
 *
 * Entry format (73 bits @ GPU_PC_WIDTH=32):
 *   {reconv_pc[31:0], saved_mask[3:0], pend_mask[3:0], pend_pc[31:0], phase[0]}
 *
 * Author: Jeremy Cai
 * Date: Mar. 04, 2026
 * Version: 1.1
 * Revision history:
 *   - Mar. 04, 2026: v1.0 — Initial implementation.
 *   - Mar. 04, 2026: v1.1 — Add clear input for kernel_start.
 */

`ifndef SIMT_STACK_V
`define SIMT_STACK_V

`include "gpu_define.v"

module simt_stack #(
    parameter DEPTH = 8    // max nesting depth
)(
    input wire clk,
    input wire rst_n,

    // ── Clear (kernel restart) ─────────────────────────
    input wire clear,

    // ── Push interface (PBRA divergence) ────────────────
    input wire push,
    input wire [`GPU_PC_WIDTH-1:0] push_reconv_pc,
    input wire [3:0] push_saved_mask,
    input wire [3:0] push_pend_mask,
    input wire [`GPU_PC_WIDTH-1:0] push_pend_pc,

    // ── Pop interface (convergence complete) ────────────
    input wire pop,

    // ── Modify TOS (phase 0→1 transition) ──────────────
    input wire modify_tos,

    // ── TOS read (combinational, always valid when ~empty) ─
    output wire [`GPU_PC_WIDTH-1:0] tos_reconv_pc,
    output wire [3:0] tos_saved_mask,
    output wire [3:0] tos_pend_mask,
    output wire [`GPU_PC_WIDTH-1:0] tos_pend_pc,
    output wire tos_phase,

    // ── Status ──────────────────────────────────────────
    output wire stack_empty,
    output wire stack_full
);

    localparam ENTRY_W = `GPU_PC_WIDTH + 4 + 4 + `GPU_PC_WIDTH + 1;

    localparam PHASE_LO      = 0;
    localparam PEND_PC_LO    = 1;
    localparam PEND_PC_HI    = PEND_PC_LO + `GPU_PC_WIDTH - 1;
    localparam PEND_MASK_LO  = PEND_PC_HI + 1;
    localparam PEND_MASK_HI  = PEND_MASK_LO + 3;
    localparam SAVED_MASK_LO = PEND_MASK_HI + 1;
    localparam SAVED_MASK_HI = SAVED_MASK_LO + 3;
    localparam RECONV_PC_LO  = SAVED_MASK_HI + 1;
    localparam RECONV_PC_HI  = RECONV_PC_LO + `GPU_PC_WIDTH - 1;

    reg [ENTRY_W-1:0] stack [0:DEPTH-1];
    reg [3:0] sp;

    assign stack_empty = (sp == 4'd0);
    assign stack_full  = (sp == DEPTH[3:0]);

    wire [3:0] tos_idx = sp - 4'd1;
    wire [ENTRY_W-1:0] tos_entry = stack[tos_idx];

    assign tos_reconv_pc  = tos_entry[RECONV_PC_HI  : RECONV_PC_LO];
    assign tos_saved_mask = tos_entry[SAVED_MASK_HI : SAVED_MASK_LO];
    assign tos_pend_mask  = tos_entry[PEND_MASK_HI  : PEND_MASK_LO];
    assign tos_pend_pc    = tos_entry[PEND_PC_HI    : PEND_PC_LO];
    assign tos_phase      = tos_entry[PHASE_LO];

    wire [ENTRY_W-1:0] push_entry = {push_reconv_pc,
                                      push_saved_mask,
                                      push_pend_mask,
                                      push_pend_pc,
                                      1'b0};

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sp <= 4'd0;
            for (i = 0; i < DEPTH; i = i + 1)
                stack[i] <= {ENTRY_W{1'b0}};
        end else if (clear) begin
            sp <= 4'd0;
        end else if (push && !stack_full) begin
            stack[sp] <= push_entry;
            sp <= sp + 4'd1;
        end else if (pop && !stack_empty) begin
            sp <= sp - 4'd1;
        end else if (modify_tos && !stack_empty) begin
            stack[tos_idx][PHASE_LO] <= 1'b1;
        end
    end

endmodule

`endif // SIMT_STACK_V