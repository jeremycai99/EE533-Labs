/* file: sm_core.v
Description: This file contains the implementation of the CUDA-like SM core, integrating the fetch unit,
decoder, scoreboard, SP cores, tensor core, and SIMT stack.
Author: Jeremy Cai
Date: Mar. 5, 2026
Version: 1.7
Revision history:
    - Feb. 27, 2026: Initial implementation of the CUDA-like SM core.
    - Mar. 4, 2026: v1.0 — Add SIMT stack and PBRA handling logic. Add convergence redirect support
        in fetch unit and SIMT stack.
    - Mar. 4, 2026: v1.1 — Bug fixes
    - Mar. 4, 2026: v1.2 — SoC integration: add thread_mask[3:0] input port (from CP10 CR7).
        Kernel launch uses thread_mask instead of hardcoded 4'b1111 for active_mask init.
    - Mar. 5, 2026: v1.3 — Remove debug_rf_addr/debug_rf_data ports and debug override mux.
    - Mar. 5, 2026: v1.4 — Fix CVT deadlock
    - Mar. 6, 2026: v1.5 — gpr_regfile 4R1W distributed RAM support.
    - Mar. 6, 2026: v1.6 — PBRA pipeline register: splits 15.6ns critical path
    - Mar. 7, 2026: v1.7 — Add SIMT convergence redirect support in fetch unit and SIMT stack. Add conv_redirect
        and conv_target_pc signals to fetch_unit, driven by SIMT stack on reconvergence.
 */

`ifndef SM_CORE_V
`define SM_CORE_V

`include "gpu_define.v"
`include "fetch_unit.v"
`include "sm_decoder.v"
`include "scoreboard.v"
`include "sp_core.v"
`include "tc_top.v"
`include "simt_stack.v"

module sm_core (
    input wire clk,
    input wire rst_n,

    output wire [`GPU_IMEM_ADDR_WIDTH-1:0] imem_addr,
    input wire [`GPU_IMEM_DATA_WIDTH-1:0] imem_rdata,

    output wire [4*`GPU_DMEM_ADDR_WIDTH-1:0] dmem_addra,
    output wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_dina,
    output wire [3:0] dmem_wea,
    input wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_douta,

    input wire kernel_start,
    input wire [`GPU_PC_WIDTH-1:0] kernel_entry_pc,
    input wire [3:0] thread_mask,
    output wire kernel_done
);

    // ================================================================
    // Internal signal declarations
    // ================================================================
    wire [`GPU_PC_WIDTH-1:0] if_id_pc;
    wire fetch_valid;
    wire fu_running;
    wire [`GPU_PC_WIDTH-1:0] fu_pc_out;

    wire [4:0] dec_opcode;
    wire dec_dt;
    wire [1:0] dec_cmp_mode;
    wire [3:0] dec_rD_addr, dec_rA_addr, dec_rB_addr, dec_rC_addr;
    wire [15:0] dec_imm16;
    wire dec_rf_we, dec_pred_we;
    wire [1:0] dec_pred_wr_sel, dec_pred_rd_sel;
    wire [2:0] dec_wb_src;
    wire dec_use_imm;
    wire dec_uses_rA, dec_uses_rB, dec_is_fma, dec_is_st;
    wire dec_is_branch, dec_is_pbra, dec_is_ret;
    wire dec_is_ld, dec_is_store, dec_is_lds, dec_is_sts;
    wire dec_is_wmma_mma, dec_is_wmma_load, dec_is_wmma_store;
    wire [1:0] dec_wmma_sel;
    wire [`GPU_PC_WIDTH-1:0] dec_branch_target;

    wire [15:0] sp_ovr_rf_r0_data [0:3];
    wire [15:0] sp_ovr_rf_r1_data [0:3];
    wire [15:0] sp_ovr_rf_r2_data [0:3];
    wire [15:0] sp_ovr_rf_r3_data [0:3];

    wire sp_pred_rd_val [0:3];
    wire [15:0] sp_ex_mem_result [0:3];
    wire [15:0] sp_ex_mem_store [0:3];
    wire sp_ex_mem_valid [0:3];
    wire sp_ex_busy [0:3];
    wire sp_mem_is_load [0:3];
    wire sp_mem_is_store [0:3];
    wire [3:0] sp_wb_rD_addr [0:3];
    wire sp_wb_rf_we [0:3];
    wire sp_wb_active [0:3];
    wire sp_wb_valid [0:3];
    wire sp_wb_pred_we [0:3];

    wire [`GPU_DMEM_ADDR_WIDTH-1:0] dmem_addr_a [0:3];
    wire [`GPU_DMEM_DATA_WIDTH-1:0] dmem_din_a [0:3];
    wire dmem_we_a [0:3];
    wire [`GPU_DMEM_DATA_WIDTH-1:0] dmem_dout_a [0:3];

    genvar u;
    generate
        for (u = 0; u < 4; u = u + 1) begin : DMEM_IO
            assign dmem_dout_a[u] = dmem_douta[`GPU_DMEM_DATA_WIDTH*u +: `GPU_DMEM_DATA_WIDTH];
            assign dmem_addra[`GPU_DMEM_ADDR_WIDTH*u +: `GPU_DMEM_ADDR_WIDTH] = dmem_addr_a[u];
            assign dmem_dina[`GPU_DMEM_DATA_WIDTH*u +: `GPU_DMEM_DATA_WIDTH] = dmem_din_a[u];
            assign dmem_wea[u] = dmem_we_a[u];
        end
    endgenerate

    reg [3:0] ovr_rf_r0_addr_mux, ovr_rf_r1_addr_mux;
    reg [3:0] ovr_rf_r2_addr_mux, ovr_rf_r3_addr_mux;

    wire sb_stall;
    wire any_ex_busy = sp_ex_busy[0] | sp_ex_busy[1] | sp_ex_busy[2] | sp_ex_busy[3];
    wire tc_busy;
    wire burst_busy;
    wire front_stall;
    wire sp_stall;

    // v1.7c: registered any_ex_busy for convergence path only.
    // Breaks 12.5ns path: id_ex_opcode → ex_busy → any_ex_busy → conv → stack.
    // 1-cycle delay is safe: conv_wait holds PC until conditions are met.
    reg any_ex_busy_r;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) any_ex_busy_r <= 1'b0;
        else any_ex_busy_r <= any_ex_busy;

    // ================================================================
    // Active Mask
    // ================================================================
    reg [3:0] active_mask;

    wire tc_rf_override;
    wire [3:0] tc_rf_r0, tc_rf_r1, tc_rf_r2, tc_rf_r3;

    wire [3:0] tc_w_addr;
    wire [4*16-1:0] tc_w_data;
    wire [3:0] tc_w_we;

    wire [4*16-1:0] flat_ovr_rf_r0 = {sp_ovr_rf_r0_data[3], sp_ovr_rf_r0_data[2],
                                       sp_ovr_rf_r0_data[1], sp_ovr_rf_r0_data[0]};
    wire [4*16-1:0] flat_ovr_rf_r1 = {sp_ovr_rf_r1_data[3], sp_ovr_rf_r1_data[2],
                                       sp_ovr_rf_r1_data[1], sp_ovr_rf_r1_data[0]};
    wire [4*16-1:0] flat_ovr_rf_r2 = {sp_ovr_rf_r2_data[3], sp_ovr_rf_r2_data[2],
                                       sp_ovr_rf_r2_data[1], sp_ovr_rf_r2_data[0]};
    wire [4*16-1:0] flat_ovr_rf_r3 = {sp_ovr_rf_r3_data[3], sp_ovr_rf_r3_data[2],
                                       sp_ovr_rf_r3_data[1], sp_ovr_rf_r3_data[0]};

    // ================================================================
    // SIMT Stack signals
    // ================================================================
    wire [`GPU_PC_WIDTH-1:0] tos_reconv_pc, tos_pend_pc;
    wire [3:0] tos_saved_mask, tos_pend_mask;
    wire tos_phase;
    wire stack_empty, stack_full;

    wire simt_push;
    wire simt_pop;
    wire simt_modify;

    // ================================================================
    // Fetch Unit
    // ================================================================
    wire branch_taken;
    wire [`GPU_PC_WIDTH-1:0] branch_target;
    wire ret_detected;

    wire conv_redirect;
    wire [`GPU_PC_WIDTH-1:0] conv_target_pc;

    fetch_unit u_fetch (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr),
        .kernel_start(kernel_start), .kernel_entry_pc(kernel_entry_pc),
        .kernel_done(kernel_done), .running(fu_running),
        .branch_taken(branch_taken), .branch_target(branch_target),
        .conv_redirect(conv_redirect), .conv_target_pc(conv_target_pc),
        .front_stall(front_stall), .ret_detected(ret_detected),
        .if_id_pc(if_id_pc), .fetch_valid(fetch_valid), .pc_out(fu_pc_out)
    );

    // ================================================================
    // IR Latch
    // ================================================================
    reg [31:0] ir_latch;
    reg ir_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ir_latched <= 1'b0;
            ir_latch <= 32'd0;
        end else if (kernel_start) begin
            ir_latched <= 1'b0;
            ir_latch <= 32'd0;
        end else if (!front_stall) begin
            ir_latched <= 1'b0;
        end else if (front_stall & ~ir_latched) begin
            ir_latch <= imem_rdata;
            ir_latched <= 1'b1;
        end
    end

    wire [31:0] dec_ir = ir_latched ? ir_latch : imem_rdata;

    // ================================================================
    // SM Decoder
    // ================================================================
    sm_decoder u_dec (
        .ir(dec_ir),
        .dec_opcode(dec_opcode), .dec_dt(dec_dt), .dec_cmp_mode(dec_cmp_mode),
        .dec_rD_addr(dec_rD_addr), .dec_rA_addr(dec_rA_addr),
        .dec_rB_addr(dec_rB_addr), .dec_rC_addr(dec_rC_addr),
        .dec_imm16(dec_imm16),
        .dec_rf_we(dec_rf_we), .dec_pred_we(dec_pred_we),
        .dec_pred_wr_sel(dec_pred_wr_sel), .dec_pred_rd_sel(dec_pred_rd_sel),
        .dec_wb_src(dec_wb_src), .dec_use_imm(dec_use_imm),
        .dec_uses_rA(dec_uses_rA), .dec_uses_rB(dec_uses_rB),
        .dec_is_fma(dec_is_fma), .dec_is_st(dec_is_st),
        .dec_is_branch(dec_is_branch), .dec_is_pbra(dec_is_pbra), .dec_is_ret(dec_is_ret),
        .dec_is_ld(dec_is_ld), .dec_is_store(dec_is_store),
        .dec_is_lds(dec_is_lds), .dec_is_sts(dec_is_sts),
        .dec_is_wmma_mma(dec_is_wmma_mma), .dec_is_wmma_load(dec_is_wmma_load),
        .dec_is_wmma_store(dec_is_wmma_store), .dec_wmma_sel(dec_wmma_sel),
        .dec_branch_target(dec_branch_target)
    );

    // ================================================================
    // DE Pipeline Register (Decode -> RF stage)
    // ================================================================
    reg [4:0] de_opcode;
    reg de_dt;
    reg [1:0] de_cmp_mode;
    reg [3:0] de_rD_addr, de_rA_addr, de_rB_addr, de_rC_addr;
    reg [15:0] de_imm16;
    reg de_rf_we, de_pred_we;
    reg [1:0] de_pred_wr_sel, de_pred_rd_sel;
    reg [2:0] de_wb_src;
    reg de_use_imm;
    reg de_uses_rA, de_uses_rB, de_is_fma, de_is_st;
    reg de_is_branch, de_is_pbra, de_is_ret;
    reg de_is_ld, de_is_store, de_is_lds, de_is_sts;
    reg de_is_wmma_mma, de_is_wmma_load, de_is_wmma_store;
    reg [1:0] de_wmma_sel;
    reg [`GPU_PC_WIDTH-1:0] de_branch_target;
    reg de_valid;

    reg [3:0] de_rf_r0_addr, de_rf_r1_addr, de_rf_r2_addr, de_rf_r3_addr;
    reg [`GPU_PC_WIDTH-1:0] de_pc;

    // v1.6: forward-declare pbra_commit for de_flush
    wire pbra_fire;
    reg pbra_commit;
    wire de_flush = branch_taken | ret_detected | pbra_commit;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            de_valid <= 1'b0; de_rf_we <= 1'b0; de_pred_we <= 1'b0;
            de_opcode <= 5'd0; de_dt <= 1'b0; de_cmp_mode <= 2'd0;
            de_rD_addr <= 4'd0; de_rA_addr <= 4'd0; de_rB_addr <= 4'd0; de_rC_addr <= 4'd0;
            de_imm16 <= 16'd0; de_pred_wr_sel <= 2'd0; de_pred_rd_sel <= 2'd0;
            de_wb_src <= 3'd0; de_use_imm <= 1'b0;
            de_uses_rA <= 1'b0; de_uses_rB <= 1'b0; de_is_fma <= 1'b0; de_is_st <= 1'b0;
            de_is_branch <= 1'b0; de_is_pbra <= 1'b0; de_is_ret <= 1'b0;
            de_is_ld <= 1'b0; de_is_store <= 1'b0; de_is_lds <= 1'b0; de_is_sts <= 1'b0;
            de_is_wmma_mma <= 1'b0; de_is_wmma_load <= 1'b0; de_is_wmma_store <= 1'b0;
            de_wmma_sel <= 2'd0; de_branch_target <= {`GPU_PC_WIDTH{1'b0}};
            de_rf_r0_addr <= 4'd0; de_rf_r1_addr <= 4'd0;
            de_rf_r2_addr <= 4'd0; de_rf_r3_addr <= 4'd0;
            de_pc <= {`GPU_PC_WIDTH{1'b0}};
        end else if (kernel_start) begin
            de_valid <= 1'b0; de_rf_we <= 1'b0; de_pred_we <= 1'b0;
        end else if (de_flush) begin
            de_valid <= 1'b0; de_rf_we <= 1'b0; de_pred_we <= 1'b0;
        end else if (!front_stall) begin
            de_opcode <= dec_opcode; de_dt <= dec_dt; de_cmp_mode <= dec_cmp_mode;
            de_rD_addr <= dec_rD_addr; de_rA_addr <= dec_rA_addr;
            de_rB_addr <= dec_rB_addr; de_rC_addr <= dec_rC_addr;
            de_imm16 <= dec_imm16;
            de_rf_we <= dec_rf_we; de_pred_we <= dec_pred_we;
            de_pred_wr_sel <= dec_pred_wr_sel; de_pred_rd_sel <= dec_pred_rd_sel;
            de_wb_src <= dec_wb_src; de_use_imm <= dec_use_imm;
            de_uses_rA <= dec_uses_rA; de_uses_rB <= dec_uses_rB;
            de_is_fma <= dec_is_fma; de_is_st <= dec_is_st;
            de_is_branch <= dec_is_branch; de_is_pbra <= dec_is_pbra; de_is_ret <= dec_is_ret;
            de_is_ld <= dec_is_ld; de_is_store <= dec_is_store;
            de_is_lds <= dec_is_lds; de_is_sts <= dec_is_sts;
            de_is_wmma_mma <= dec_is_wmma_mma; de_is_wmma_load <= dec_is_wmma_load;
            de_is_wmma_store <= dec_is_wmma_store; de_wmma_sel <= dec_wmma_sel;
            de_branch_target <= dec_branch_target;
            de_valid <= fetch_valid;
            de_rf_r0_addr <= dec_rA_addr;
            de_rf_r1_addr <= dec_rB_addr;
            de_rf_r2_addr <= (dec_is_fma | dec_is_st) ? dec_rD_addr : dec_rC_addr;
            de_rf_r3_addr <= 4'd0;
            de_pc <= if_id_pc;
        end
    end

    // ================================================================
    // PBRA reconvergence PC / branch target
    // ================================================================
    wire [`GPU_PC_WIDTH-1:0] de_reconv_pc = {{(`GPU_PC_WIDTH-12){1'b0}}, de_imm16[11:0]};
    wire [`GPU_PC_WIDTH-1:0] de_pend_pc = de_pc + 1;

    wire [`GPU_PC_WIDTH-1:0] de_pbra_target = {{(`GPU_PC_WIDTH-12){1'b0}},
                                                de_rD_addr, de_rA_addr, de_imm16[15:12]};

    // ================================================================
    // Stall / Flush / Issue Control
    // ================================================================
    wire de_wmma_any = de_is_wmma_mma | de_is_wmma_load | de_is_wmma_store;

    wire pipeline_drained;
    wire pipeline_drained_r;  // forward-declared, assigned after pipeline_drained

    wire wmma_drain_wait = de_valid & de_wmma_any & ~pipeline_drained
                         & ~tc_busy & ~burst_busy;

    // drain_waits use combinational pipeline_drained so they stay in sync
    // with front_stall (which uses combinational any_ex_busy).
    wire pbra_drain_wait = (de_valid & de_is_pbra & ~pipeline_drained)
                         | pbra_fire;

    wire sb_stall_gated = sb_stall & de_valid;

    // v1.7c: conv_wait forward-declared here, defined in Convergence Checker.
    wire conv_wait;

    // front_stall MUST use combinational any_ex_busy to stay in sync with
    // sp_stall. Using any_ex_busy_r creates a 1-cycle window where
    // front_stall=0 but sp_stall=1, causing DE→RR to overwrite an
    // instruction stuck in RR.
    assign front_stall = sb_stall_gated | any_ex_busy | tc_busy | burst_busy
                       | wmma_drain_wait | pbra_drain_wait | conv_wait;
    assign sp_stall = any_ex_busy | tc_busy | burst_busy;

    wire de_is_ctrl_special = de_wmma_any | de_is_pbra;
    wire id_can_issue = de_valid & ~front_stall & ~de_is_ctrl_special;
    wire id_issue_ctrl = de_valid & ~front_stall;

    // ================================================================
    // PBRA Handler
    // ================================================================
    wire pbra_pred_override = de_valid & de_is_pbra & pipeline_drained_r;
    reg [1:0] rr_pred_rd_sel;
    wire [1:0] sp_pred_rd_sel_mux = pbra_pred_override ? de_pred_rd_sel
                                                       : rr_pred_rd_sel;

    wire [3:0] pred_vec = {sp_pred_rd_val[3], sp_pred_rd_val[2],
                           sp_pred_rd_val[1], sp_pred_rd_val[0]};
    wire [3:0] taken_mask = active_mask & pred_vec;
    wire [3:0] fall_mask = active_mask & ~pred_vec;

    wire pbra_divergent = (taken_mask != 4'd0) & (fall_mask != 4'd0);

    // v1.6: guard pbra_fire with ~pbra_commit to prevent re-fire
    // v1.7c: use pipeline_drained_r to break ex_busy→pbra_fire timing path
    assign pbra_fire = de_valid & de_is_pbra & pipeline_drained_r
                     & ~tc_busy & ~burst_busy & ~pbra_commit;

    wire [`GPU_PC_WIDTH-1:0] pbra_target = (taken_mask != 4'd0)
                                         ? de_pbra_target : de_pend_pc;

    // ================================================================
    // v1.6: PBRA pipeline register — breaks 15.6ns critical path
    //   Cycle N:   pbra_fire -> pred read -> divergence -> REGISTERED
    //   Cycle N+1: pbra_commit -> SIMT push, mask update, redirect
    // ================================================================
    reg pbra_divergent_r;
    reg [3:0] taken_mask_r, fall_mask_r;
    reg [`GPU_PC_WIDTH-1:0] pbra_target_r;
    reg [`GPU_PC_WIDTH-1:0] pbra_reconv_pc_r, pbra_pend_pc_r;
    reg [3:0] pbra_saved_mask_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pbra_commit <= 1'b0;
            pbra_divergent_r <= 1'b0;
            taken_mask_r <= 4'd0;
            fall_mask_r <= 4'd0;
            pbra_target_r <= {`GPU_PC_WIDTH{1'b0}};
            pbra_reconv_pc_r <= {`GPU_PC_WIDTH{1'b0}};
            pbra_pend_pc_r <= {`GPU_PC_WIDTH{1'b0}};
            pbra_saved_mask_r <= 4'd0;
        end else if (kernel_start) begin
            pbra_commit <= 1'b0;
        end else if (pbra_fire & ~pbra_commit) begin
            pbra_commit <= 1'b1;
            pbra_divergent_r <= pbra_divergent;
            taken_mask_r <= taken_mask;
            fall_mask_r <= fall_mask;
            pbra_target_r <= pbra_target;
            pbra_reconv_pc_r <= de_reconv_pc;
            pbra_pend_pc_r <= de_pend_pc;
            pbra_saved_mask_r <= active_mask;
        end else begin
            pbra_commit <= 1'b0;
        end
    end

    wire pbra_divergent_commit = pbra_commit & pbra_divergent_r;

    // ================================================================
    // Branch / redirect control
    // v1.6: redirect on pbra_commit (cycle N+1), not pbra_fire
    // ================================================================
    assign branch_taken = (id_issue_ctrl & de_is_branch) | pbra_commit;
    assign branch_target = pbra_commit ? pbra_target_r : de_branch_target;
    assign ret_detected = id_issue_ctrl & de_is_ret;

    wire sb_issue = id_can_issue & de_rf_we;
    wire sb_issue_pred = id_can_issue & de_pred_we;

    // ================================================================
    // Scoreboard
    // ================================================================
    wire [3:0] wb_active_mask_sb = {sp_wb_active[3], sp_wb_active[2],
                                    sp_wb_active[1], sp_wb_active[0]};
    wire sb_wb_rf_we_any = sp_wb_rf_we[0] | sp_wb_rf_we[1]
                         | sp_wb_rf_we[2] | sp_wb_rf_we[3];
    
    wire sb_wb_pred_we_any = sp_wb_pred_we[0] | sp_wb_pred_we[1]
                           | sp_wb_pred_we[2] | sp_wb_pred_we[3];

    wire sb_any_pending;

    scoreboard u_sb (
        .clk(clk), .rst_n(rst_n),
        .clear(kernel_start),
        .rA_addr(de_rA_addr), .rB_addr(de_rB_addr), .rD_addr(de_rD_addr),
        .uses_rA(de_uses_rA), .uses_rB(de_uses_rB),
        .is_fma(de_is_fma), .is_st(de_is_st),
        .rf_we(de_rf_we), .active_mask(active_mask),
        .issue(sb_issue),
        .wb_rD_addr(sp_wb_rD_addr[0]), .wb_rf_we(sb_wb_rf_we_any),
        .wb_active_mask(wb_active_mask_sb),
        .issue_pred(sb_issue_pred), .wb_pred_we(sb_wb_pred_we_any),
        .stall(sb_stall), .any_pending(sb_any_pending)
    );

    assign pipeline_drained = ~sb_any_pending & ~any_ex_busy;

    // v1.7c: registered pipeline_drained for trigger/fire signals.
    // Breaks the deep id_ex_opcode→fpu_busy→ex_busy→any_ex_busy path
    // (6 levels, ~4.7ns) that leaks into pbra_fire, tc_trigger, etc.
    // 1-cycle delay is safe: drain_wait stalls hold the frontend.
    // Keep combinational pipeline_drained for drain_wait terms (feed front_stall→FF).
    assign pipeline_drained_r = ~sb_any_pending & ~any_ex_busy_r;

    // ================================================================
    // Tensor Core Top (v1.2)
    // ================================================================
    wire tc_trigger = de_valid & de_is_wmma_mma & ~tc_busy & ~burst_busy
                    & pipeline_drained_r;

    tc_top u_tc_top (
        .clk(clk), .rst_n(rst_n),
        .trigger(tc_trigger),
        .dec_rA_addr(de_rA_addr), .dec_rB_addr(de_rB_addr),
        .dec_rC_addr(de_rC_addr), .dec_rD_addr(de_rD_addr),
        .sp_rf_r0_data(flat_ovr_rf_r0), .sp_rf_r1_data(flat_ovr_rf_r1),
        .sp_rf_r2_data(flat_ovr_rf_r2), .sp_rf_r3_data(flat_ovr_rf_r3),
        .busy(tc_busy),
        .rf_addr_override(tc_rf_override),
        .rf_r0_addr(tc_rf_r0), .rf_r1_addr(tc_rf_r1),
        .rf_r2_addr(tc_rf_r2), .rf_r3_addr(tc_rf_r3),
        .scat_w_addr(tc_w_addr), .scat_w_data(tc_w_data), .scat_w_we(tc_w_we)
    );

    // ================================================================
    // Burst Controller — WMMA.LOAD / WMMA.STORE
    // ================================================================
    localparam [2:0] BU_IDLE       = 3'd0, BU_SETUP      = 3'd1,
                     BU_RREAD      = 3'd2, BU_ADDR       = 3'd3,
                     BU_LOAD_ADDR  = 3'd4, BU_LOAD_BEAT  = 3'd5,
                     BU_STORE_READ = 3'd6, BU_STORE_BEAT = 3'd7;

    reg [2:0] bu_state;
    reg [1:0] bu_beat;
    assign burst_busy = (bu_state != BU_IDLE);

    reg [3:0] bu_rD_base;
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] bu_base_addr [0:3];
    reg [15:0] bu_store_data [0:3][0:3];
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] bu_rf_data [0:3];
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] bu_imm16_r;
    reg bu_is_store;

    wire bu_load_trigger = de_valid & de_is_wmma_load & ~tc_busy
                         & ~burst_busy & pipeline_drained_r;
    wire bu_store_trigger = de_valid & de_is_wmma_store & ~tc_busy
                          & ~burst_busy & pipeline_drained_r;

    reg bu_rf_override_r;
    reg [3:0] bu_rf_r0_r, bu_rf_r1_r, bu_rf_r2_r, bu_rf_r3_r;
    reg [3:0] bu_w1_addr;
    reg bu_w1_we;
    reg bu_dmem_override;

    integer bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bu_state <= BU_IDLE; bu_beat <= 2'd0; bu_rD_base <= 4'd0;
            bu_rf_override_r <= 1'b0;
            bu_rf_r0_r <= 4'd0; bu_rf_r1_r <= 4'd0; bu_rf_r2_r <= 4'd0; bu_rf_r3_r <= 4'd0;
            bu_is_store <= 1'b0; bu_imm16_r <= {`GPU_DMEM_ADDR_WIDTH{1'b0}};
            for (bi = 0; bi < 4; bi = bi + 1) begin
                bu_base_addr[bi] <= {`GPU_DMEM_ADDR_WIDTH{1'b0}};
                bu_rf_data[bi] <= {`GPU_DMEM_ADDR_WIDTH{1'b0}};
                bu_store_data[bi][0] <= 16'd0; bu_store_data[bi][1] <= 16'd0;
                bu_store_data[bi][2] <= 16'd0; bu_store_data[bi][3] <= 16'd0;
            end
        end else begin
            case (bu_state)
                BU_IDLE: begin
                    if (bu_load_trigger | bu_store_trigger) begin
                        bu_rD_base <= de_rD_addr;
                        bu_imm16_r <= de_imm16[`GPU_DMEM_ADDR_WIDTH-1:0];
                        bu_is_store <= bu_store_trigger;
                        bu_rf_override_r <= 1'b1;
                        bu_rf_r0_r <= de_rf_r0_addr;
                        bu_state <= BU_SETUP;
                    end
                end
                BU_SETUP: bu_state <= BU_RREAD;
                BU_RREAD: begin
                    for (bi = 0; bi < 4; bi = bi + 1)
                        bu_rf_data[bi] <= sp_ovr_rf_r0_data[bi][`GPU_DMEM_ADDR_WIDTH-1:0];
                    bu_rf_override_r <= 1'b0;
                    bu_state <= BU_ADDR;
                end
                BU_ADDR: begin
                    for (bi = 0; bi < 4; bi = bi + 1)
                        bu_base_addr[bi] <= bu_rf_data[bi] + bu_imm16_r;
                    if (bu_is_store) begin
                        bu_rf_override_r <= 1'b1;
                        bu_rf_r0_r <= bu_rD_base;
                        bu_rf_r1_r <= bu_rD_base + 4'd1;
                        bu_rf_r2_r <= bu_rD_base + 4'd2;
                        bu_rf_r3_r <= bu_rD_base + 4'd3;
                        bu_state <= BU_STORE_READ;
                    end else begin
                        bu_beat <= 2'd0;
                        bu_state <= BU_LOAD_ADDR;
                    end
                end
                BU_LOAD_ADDR: bu_state <= BU_LOAD_BEAT;
                BU_LOAD_BEAT: begin
                    if (bu_beat == 2'd3) bu_state <= BU_IDLE;
                    else bu_beat <= bu_beat + 2'd1;
                end
                BU_STORE_READ: begin
                    for (bi = 0; bi < 4; bi = bi + 1) begin
                        bu_store_data[bi][0] <= sp_ovr_rf_r0_data[bi];
                        bu_store_data[bi][1] <= sp_ovr_rf_r1_data[bi];
                        bu_store_data[bi][2] <= sp_ovr_rf_r2_data[bi];
                        bu_store_data[bi][3] <= sp_ovr_rf_r3_data[bi];
                    end
                    bu_beat <= 2'd0; bu_state <= BU_STORE_BEAT;
                    bu_rf_override_r <= 1'b0;
                end
                BU_STORE_BEAT: begin
                    if (bu_beat == 2'd3) bu_state <= BU_IDLE;
                    else bu_beat <= bu_beat + 2'd1;
                end
                default: bu_state <= BU_IDLE;
            endcase
        end
    end

    always @(*) begin
        bu_w1_addr = 4'd0; bu_w1_we = 1'b0; bu_dmem_override = 1'b0;
        case (bu_state)
            BU_LOAD_ADDR, BU_LOAD_BEAT: begin
                bu_dmem_override = 1'b1;
                bu_w1_we = (bu_state == BU_LOAD_BEAT);
                bu_w1_addr = bu_rD_base + {2'd0, bu_beat};
            end
            BU_STORE_BEAT: bu_dmem_override = 1'b1;
            default: ;
        endcase
    end

    // ================================================================
    // Override RF Read Address Mux (TC / BU only)
    // ================================================================
    always @(*) begin
        ovr_rf_r0_addr_mux = 4'd0; ovr_rf_r1_addr_mux = 4'd0;
        ovr_rf_r2_addr_mux = 4'd0; ovr_rf_r3_addr_mux = 4'd0;
        if (tc_rf_override) begin
            ovr_rf_r0_addr_mux = tc_rf_r0; ovr_rf_r1_addr_mux = tc_rf_r1;
            ovr_rf_r2_addr_mux = tc_rf_r2; ovr_rf_r3_addr_mux = tc_rf_r3;
        end else if (bu_rf_override_r) begin
            ovr_rf_r0_addr_mux = bu_rf_r0_r; ovr_rf_r1_addr_mux = bu_rf_r1_r;
            ovr_rf_r2_addr_mux = bu_rf_r2_r; ovr_rf_r3_addr_mux = bu_rf_r3_r;
        end
    end

    wire ovr_sel = tc_rf_override | bu_rf_override_r;

    // ================================================================
    // Convergence Checker
    // v1.7c: Break conv→stack critical path using conv_can_fire,
    //   conv_wait in front_stall, and registered any_ex_busy_r.
    // ================================================================
    wire at_reconv = ~stack_empty & fu_running
                   & (fu_pc_out == tos_reconv_pc);

    // v1.7c: convergence uses pipeline_drained_r (defined near line 441).
    // All trigger/fire signals use pipeline_drained_r to avoid the deep
    // id_ex_opcode → ex_busy → any_ex_busy combinational path.

    // ~de_is_pbra replaces ~pbra_fire to avoid combinational leak:
    //   pbra_fire uses pipeline_drained_r (safe), but older versions
    //   leaked through pipeline_drained (comb). Belt-and-suspenders.
    wire conv_can_fire = ~any_ex_busy_r & ~tc_busy & ~burst_busy
                       & ~de_is_pbra & ~pbra_commit
                       & (~de_valid | pipeline_drained_r);

    wire conv_phase0_fire = at_reconv & ~tos_phase & conv_can_fire;
    wire conv_phase1_fire = at_reconv & tos_phase & conv_can_fire;

    // Hold PC at reconv_pc when convergence detected but can't fire yet
    assign conv_wait = at_reconv & ~conv_can_fire;

    assign conv_redirect = conv_phase0_fire;
    assign conv_target_pc = tos_pend_pc;

    // ================================================================
    // SIMT Stack
    // v1.6: push uses registered pbra_commit data
    // ================================================================
    assign simt_push = pbra_divergent_commit;
    assign simt_pop = conv_phase1_fire;
    assign simt_modify = conv_phase0_fire;

    simt_stack #(.DEPTH(8)) u_simt_stack (
        .clk(clk), .rst_n(rst_n),
        .clear(kernel_start),
        .push(simt_push),
        // v1.6: registered push data from pbra pipeline stage
        .push_reconv_pc(pbra_reconv_pc_r), .push_saved_mask(pbra_saved_mask_r),
        .push_pend_mask(fall_mask_r), .push_pend_pc(pbra_pend_pc_r),
        .pop(simt_pop), .modify_tos(simt_modify),
        .tos_reconv_pc(tos_reconv_pc), .tos_saved_mask(tos_saved_mask),
        .tos_pend_mask(tos_pend_mask), .tos_pend_pc(tos_pend_pc),
        .tos_phase(tos_phase),
        .stack_empty(stack_empty), .stack_full(stack_full)
    );

    // ================================================================
    // Active Mask Update
    // v1.6: divergent mask update uses registered taken_mask_r
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) active_mask <= 4'b1111;
        else if (kernel_start) active_mask <= thread_mask;
        else if (conv_phase1_fire) active_mask <= tos_saved_mask;
        else if (conv_phase0_fire) active_mask <= tos_pend_mask;
        else if (pbra_divergent_commit) active_mask <= taken_mask_r;
    end

    // ================================================================
    // RR Pipeline Register (DE -> Register Read)
    // ================================================================
    reg [4:0] rr_opcode;
    reg rr_dt;
    reg [1:0] rr_cmp_mode;
    reg [3:0] rr_rD_addr;
    reg [15:0] rr_imm16;
    reg rr_rf_we, rr_pred_we;
    reg [1:0] rr_pred_wr_sel;
    reg [2:0] rr_wb_src;
    reg rr_use_imm;
    reg rr_valid;
    reg [3:0] rr_rf_r0_addr, rr_rf_r1_addr, rr_rf_r2_addr, rr_rf_r3_addr;
    reg [3:0] rr_active_mask;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rr_valid <= 1'b0; rr_rf_we <= 1'b0; rr_pred_we <= 1'b0;
            rr_opcode <= 5'd0; rr_dt <= 1'b0; rr_cmp_mode <= 2'd0;
            rr_rD_addr <= 4'd0; rr_imm16 <= 16'd0;
            rr_pred_wr_sel <= 2'd0; rr_pred_rd_sel <= 2'd0;
            rr_wb_src <= 3'd0; rr_use_imm <= 1'b0;
            rr_rf_r0_addr <= 4'd0; rr_rf_r1_addr <= 4'd0;
            rr_rf_r2_addr <= 4'd0; rr_rf_r3_addr <= 4'd0;
            rr_active_mask <= 4'b0000;
        end else if (kernel_start) begin
            rr_valid <= 1'b0; rr_rf_we <= 1'b0; rr_pred_we <= 1'b0;
        end else if (!sp_stall) begin
            if (!front_stall) begin
                rr_valid <= de_valid & ~de_is_ctrl_special;
                rr_opcode <= de_opcode; rr_dt <= de_dt; rr_cmp_mode <= de_cmp_mode;
                rr_rD_addr <= de_rD_addr; rr_imm16 <= de_imm16;
                rr_rf_we <= de_rf_we; rr_pred_we <= de_pred_we;
                rr_pred_wr_sel <= de_pred_wr_sel; rr_pred_rd_sel <= de_pred_rd_sel;
                rr_wb_src <= de_wb_src; rr_use_imm <= de_use_imm;
                rr_rf_r0_addr <= de_rf_r0_addr; rr_rf_r1_addr <= de_rf_r1_addr;
                rr_rf_r2_addr <= de_rf_r2_addr; rr_rf_r3_addr <= de_rf_r3_addr;
                rr_active_mask <= active_mask;
            end else begin
                rr_valid <= 1'b0; rr_rf_we <= 1'b0; rr_pred_we <= 1'b0;
            end
        end
    end

    // ================================================================
    // External RF Write Mux — single channel (v1.5)
    // ================================================================
    reg [3:0] ext_w_addr [0:3];
    reg [15:0] ext_w_data [0:3];
    reg ext_w_we [0:3];

    integer wi;
    always @(*) begin
        for (wi = 0; wi < 4; wi = wi + 1) begin
            if (bu_w1_we) begin
                ext_w_addr[wi] = bu_w1_addr;
                ext_w_data[wi] = dmem_douta[wi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
                ext_w_we[wi] = 1'b1;
            end else begin
                ext_w_addr[wi] = tc_w_addr;
                ext_w_data[wi] = tc_w_data[wi*16 +: 16];
                ext_w_we[wi] = tc_w_we[wi];
            end
        end
    end

    // ================================================================
    // 4x SP Core (v1.5)
    // ================================================================
    genvar t;
    generate
        for (t = 0; t < 4; t = t + 1) begin : SP_LANE
            sp_core #(.TID(t[1:0])) u_sp (
                .clk(clk), .rst_n(rst_n),
                .stall(sp_stall), .flush_id(kernel_start),
                .ppl_rf_r0_addr(rr_rf_r0_addr), .ppl_rf_r1_addr(rr_rf_r1_addr),
                .ppl_rf_r2_addr(rr_rf_r2_addr), .ppl_rf_r3_addr(rr_rf_r3_addr),
                .ovr_rf_r0_addr(ovr_rf_r0_addr_mux), .ovr_rf_r1_addr(ovr_rf_r1_addr_mux),
                .ovr_rf_r2_addr(ovr_rf_r2_addr_mux), .ovr_rf_r3_addr(ovr_rf_r3_addr_mux),
                .ovr_rf_r0_data(sp_ovr_rf_r0_data[t]), .ovr_rf_r1_data(sp_ovr_rf_r1_data[t]),
                .ovr_rf_r2_data(sp_ovr_rf_r2_data[t]), .ovr_rf_r3_data(sp_ovr_rf_r3_data[t]),
                .ovr_sel(ovr_sel),
                .pred_rd_sel(sp_pred_rd_sel_mux), .pred_rd_val(sp_pred_rd_val[t]),
                .id_opcode(rr_opcode), .id_dt(rr_dt), .id_cmp_mode(rr_cmp_mode),
                .id_rf_we(rr_rf_we), .id_pred_we(rr_pred_we),
                .id_rD_addr(rr_rD_addr), .id_pred_wr_sel(rr_pred_wr_sel),
                .id_valid(rr_valid), .id_active(rr_active_mask[t]),
                .id_wb_src(rr_wb_src), .id_use_imm(rr_use_imm), .id_imm16(rr_imm16),
                .ex_mem_result_out(sp_ex_mem_result[t]),
                .ex_mem_store_out(sp_ex_mem_store[t]),
                .ex_mem_valid_out(sp_ex_mem_valid[t]),
                .ex_busy(sp_ex_busy[t]),
                .mem_rdata(dmem_dout_a[t]),
                .wb_ext_w_addr(ext_w_addr[t]),
                .wb_ext_w_data(ext_w_data[t]),
                .wb_ext_w_we(ext_w_we[t]),
                .mem_is_load(sp_mem_is_load[t]), .mem_is_store(sp_mem_is_store[t]),
                .wb_rD_addr(sp_wb_rD_addr[t]), .wb_rf_we(sp_wb_rf_we[t]),
                .wb_active(sp_wb_active[t]), .wb_valid(sp_wb_valid[t]),
                .wb_pred_we(sp_wb_pred_we[t])
            );
        end
    endgenerate

    // ================================================================
    // DMEM Port-A Mux (normal vs burst)
    // ================================================================
    genvar d;
    generate
        for (d = 0; d < 4; d = d + 1) begin : DMEM_MUX
            wire bu_load_phase = (bu_state == BU_LOAD_ADDR) | (bu_state == BU_LOAD_BEAT);
            wire bu_store_phase = (bu_state == BU_STORE_BEAT);

            wire [`GPU_DMEM_ADDR_WIDTH-1:0] bu_ld_addr =
                (bu_state == BU_LOAD_ADDR)
                    ? bu_base_addr[d]
                    : bu_base_addr[d] + {{(`GPU_DMEM_ADDR_WIDTH-2){1'b0}}, bu_beat} + {{(`GPU_DMEM_ADDR_WIDTH-1){1'b0}}, 1'b1};

            assign dmem_addr_a[d] = bu_load_phase ? bu_ld_addr :
                                    bu_store_phase ? (bu_base_addr[d] + {{(`GPU_DMEM_ADDR_WIDTH-2){1'b0}}, bu_beat}) :
                                    sp_ex_mem_result[d][`GPU_DMEM_ADDR_WIDTH-1:0];

            assign dmem_din_a[d] = bu_store_phase ? bu_store_data[d][bu_beat] :
                                   sp_ex_mem_store[d];

            assign dmem_we_a[d] = bu_store_phase ? 1'b1 :
                                  (sp_mem_is_store[d] & sp_ex_mem_valid[d] & ~sp_stall);
        end
    endgenerate

endmodule

`endif // SM_CORE_V