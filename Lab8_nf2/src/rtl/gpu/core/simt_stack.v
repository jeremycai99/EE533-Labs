/* file: simt_stack.v
 Description: SIMT divergence stack for the SM core.
 Author: Jeremy Cai
 Date: Mar. 04, 2026
 Version: 1.1
 Revision history:
   - Mar. 04, 2026: v1.0 — Initial implementation.
   - Mar. 04, 2026: v1.1 — Add clear input for kernel_start.
   - Mar. 6, 2026: v1.2 — Rewrite for distributed RAM inference on Virtex-II Pro.
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

    // ================================================================
    //  Storage: unpacked array — XST-inferable as distributed RAM.
    //  NO reset on array contents: entries below sp are never read,
    //  so undefined contents are safe. Only sp needs reset.
    // ================================================================
    (* ram_style = "distributed" *) reg [ENTRY_W-1:0] stack [0:DEPTH-1];

    reg [3:0] sp;

    assign stack_empty = (sp == 4'd0);
    assign stack_full  = (sp == DEPTH[3:0]);

    wire [3:0] tos_idx = sp - 4'd1;

    // ================================================================
    //  Combinational TOS read — separated from write for RAM inference.
    //  Async read through dist RAM read port.
    // ================================================================
    wire [ENTRY_W-1:0] tos_entry = stack[tos_idx];

    assign tos_reconv_pc  = tos_entry[RECONV_PC_HI  : RECONV_PC_LO];
    assign tos_saved_mask = tos_entry[SAVED_MASK_HI : SAVED_MASK_LO];
    assign tos_pend_mask  = tos_entry[PEND_MASK_HI  : PEND_MASK_LO];
    assign tos_pend_pc    = tos_entry[PEND_PC_HI    : PEND_PC_LO];
    assign tos_phase      = tos_entry[PHASE_LO];

    // ================================================================
    //  Push entry construction
    // ================================================================
    wire [ENTRY_W-1:0] push_entry = {push_reconv_pc,
                                      push_saved_mask,
                                      push_pend_mask,
                                      push_pend_pc,
                                      1'b0};

    // ================================================================
    //  Modified TOS entry — full-width write with phase set to 1.
    //  Replaces partial bit write (v1.1: stack[tos_idx][PHASE_LO] <= 1)
    //  which blocked distributed RAM inference.
    // ================================================================
    wire [ENTRY_W-1:0] modified_tos = {tos_entry[ENTRY_W-1 : PHASE_LO+1], 1'b1};

    // ================================================================
    //  Write address / data mux — single write port for RAM inference
    // ================================================================
    reg [3:0] wr_addr;
    reg [ENTRY_W-1:0] wr_data;
    reg wr_en;

    always @(*) begin
        wr_addr = sp;
        wr_data = push_entry;
        wr_en = 1'b0;
        if (push && !stack_full) begin
            wr_addr = sp;
            wr_data = push_entry;
            wr_en = 1'b1;
        end else if (modify_tos && !stack_empty) begin
            wr_addr = tos_idx;
            wr_data = modified_tos;
            wr_en = 1'b1;
        end
    end

    // ================================================================
    //  Synchronous write — clean single-port pattern for dist RAM.
    //  Stack pointer managed separately.
    // ================================================================
    always @(posedge clk) begin
        if (wr_en)
            stack[wr_addr] <= wr_data;
    end

    // ================================================================
    //  Stack pointer — reset and update
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            sp <= 4'd0;
        else if (clear)
            sp <= 4'd0;
        else if (push && !stack_full)
            sp <= sp + 4'd1;
        else if (pop && !stack_empty)
            sp <= sp - 4'd1;
    end

endmodule

`endif // SIMT_STACK_V