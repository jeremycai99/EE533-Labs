/* file: sm_core.v
 Description: This file implements the Streaming Multiprocessor (SM) core, which includes the fetch unit,
    decoder, scoreboard, multiple SP cores, and a tensor core. 
 Date: Feb. 28, 2026
 Version: 1.7
 Revision history:
    - Feb. 28, 2026: Initial implementation of the SM core.
    - Mar. 01, 2026: v1.1 — Register BU RF override signals.
    - Mar. 01, 2026: v1.2 — Insert DE pipeline register stage.
    - Mar. 01, 2026: v1.3 — Restore ir_latch.
    - Mar. 01, 2026: v1.4 — Gate sb_stall with de_valid.
    - Mar. 01, 2026: v1.5 — Insert RR pipeline register stage.
    - Mar. 01, 2026: v1.5a/b — WMMA RF bypass fixes.
    - Mar. 02, 2026: v1.6 — Super-pipelined BU FSM (BU_SETUP/RREAD/ADDR).
    - Mar. 02, 2026: v1.7 — Dual RF read ports + registered TC override.
      Problem: v1.6 critical path -7.5ns slack:
        tc_state(FF) → rf_addr_override(comb) → bypass mux → RF read
        → id_ex_opB(FF) = 15.5ns.
      Three fixes applied:
        1. gpr_regfile v1.1: 8R4W dual read ports (pipeline + override),
           forwarding removed (scoreboard guarantees RAW safety).
        2. tc_top v1.1: registered rf_addr_override and rf_r*_addr.
        3. sm_core v1.7: pipeline reads from rr_rf_r*_addr go directly
           to sp_core pipeline ports — no bypass mux in path.
           Override reads for TC/BU/debug use separate override ports.
      Pipeline critical path:
        rr_rf_r*_addr(FF) → RF 16:1 mux (3 LUTs) → id_ex_opB(FF) ≈ 5-6ns ✓
      Override critical path:
        tc_rf_r*(FF) → addr mux(1 LUT) → RF 16:1 mux → a_hold(FF) ≈ 7ns ✓
      No changes to fetch_unit.v (drain_counter stays 3'd6).
*/

`ifndef SM_CORE_V
`define SM_CORE_V

`include "gpu_define.v"
`include "fetch_unit.v"
`include "sm_decoder.v"
`include "scoreboard.v"
`include "sp_core.v"
`include "tc_top.v"

module sm_core (
    input wire clk,
    input wire rst_n,

    // IMEM BRAM port-A (read-only by GPU)
    output wire [`GPU_IMEM_ADDR_WIDTH-1:0] imem_addr,
    input wire [`GPU_IMEM_DATA_WIDTH-1:0] imem_rdata,

    // Per-SP DMEM BRAM port-A (GPU read/write)
    output wire [4*`GPU_DMEM_ADDR_WIDTH-1:0] dmem_addra,
    output wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_dina,
    output wire [3:0] dmem_wea,
    input wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_douta,

    // Kernel control
    input wire kernel_start,
    input wire [`GPU_PC_WIDTH-1:0] kernel_entry_pc,
    output wire kernel_done,

    // Debug RF read (testbench observation)
    input wire [3:0] debug_rf_addr,
    output wire [4*`GPU_DMEM_DATA_WIDTH-1:0] debug_rf_data
);

    // ================================================================
    // Internal signal declarations
    // ================================================================
    wire [`GPU_PC_WIDTH-1:0] if_id_pc;
    wire fetch_valid;
    wire fu_running;

    // Decoder outputs (combinational from dec_ir)
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

    // Per-SP override RF read data (TC gather / BU data / debug)
    wire [15:0] sp_ovr_rf_r0_data [0:3];
    wire [15:0] sp_ovr_rf_r1_data [0:3];
    wire [15:0] sp_ovr_rf_r2_data [0:3];
    wire [15:0] sp_ovr_rf_r3_data [0:3];

    // Per-SP pipeline signals (arrays)
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

    // Per-SP DMEM internal nets
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

    // Override RF read address bus (TC gather / BU store / debug)
    reg [3:0] ovr_rf_r0_addr_mux, ovr_rf_r1_addr_mux;
    reg [3:0] ovr_rf_r2_addr_mux, ovr_rf_r3_addr_mux;

    // Stall / flush
    wire sb_stall;
    wire any_ex_busy = sp_ex_busy[0] | sp_ex_busy[1] | sp_ex_busy[2] | sp_ex_busy[3];
    wire tc_busy;
    wire burst_busy;
    wire front_stall;
    wire sp_stall;

    // Active mask
    reg [3:0] active_mask;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) active_mask <= 4'b1111;
        else if (kernel_start) active_mask <= 4'b1111;
    end

    // TC top interface wires
    wire tc_rf_override;   // registered inside tc_top v1.1
    wire [3:0] tc_rf_r0, tc_rf_r1, tc_rf_r2, tc_rf_r3;
    wire [3:0] tc_w1_addr, tc_w2_addr, tc_w3_addr;
    wire [4*16-1:0] tc_w1_data, tc_w2_data, tc_w3_data;
    wire [3:0] tc_w1_we, tc_w2_we, tc_w3_we;

    // Flat override RF read buses for tc_top
    wire [4*16-1:0] flat_ovr_rf_r0 = {sp_ovr_rf_r0_data[3], sp_ovr_rf_r0_data[2],
                                       sp_ovr_rf_r0_data[1], sp_ovr_rf_r0_data[0]};
    wire [4*16-1:0] flat_ovr_rf_r1 = {sp_ovr_rf_r1_data[3], sp_ovr_rf_r1_data[2],
                                       sp_ovr_rf_r1_data[1], sp_ovr_rf_r1_data[0]};
    wire [4*16-1:0] flat_ovr_rf_r2 = {sp_ovr_rf_r2_data[3], sp_ovr_rf_r2_data[2],
                                       sp_ovr_rf_r2_data[1], sp_ovr_rf_r2_data[0]};
    wire [4*16-1:0] flat_ovr_rf_r3 = {sp_ovr_rf_r3_data[3], sp_ovr_rf_r3_data[2],
                                       sp_ovr_rf_r3_data[1], sp_ovr_rf_r3_data[0]};

    // ================================================================
    // Fetch Unit
    // ================================================================
    wire branch_taken;
    wire [`GPU_PC_WIDTH-1:0] branch_target;
    wire ret_detected;

    fetch_unit u_fetch (
        .clk (clk), .rst_n (rst_n),
        .imem_addr (imem_addr),
        .kernel_start (kernel_start),
        .kernel_entry_pc (kernel_entry_pc),
        .kernel_done (kernel_done),
        .running (fu_running),
        .branch_taken (branch_taken),
        .branch_target (branch_target),
        .front_stall (front_stall),
        .ret_detected (ret_detected),
        .if_id_pc (if_id_pc),
        .fetch_valid (fetch_valid)
    );

    // ================================================================
    // IR Latch — hold IMEM output during stalls
    // ================================================================
    reg [31:0] ir_latch;
    reg ir_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
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
    // SM Decoder (combinational from dec_ir)
    // ================================================================
    sm_decoder u_dec (
        .ir (dec_ir),
        .dec_opcode (dec_opcode),
        .dec_dt (dec_dt),
        .dec_cmp_mode (dec_cmp_mode),
        .dec_rD_addr (dec_rD_addr),
        .dec_rA_addr (dec_rA_addr),
        .dec_rB_addr (dec_rB_addr),
        .dec_rC_addr (dec_rC_addr),
        .dec_imm16 (dec_imm16),
        .dec_rf_we (dec_rf_we),
        .dec_pred_we (dec_pred_we),
        .dec_pred_wr_sel (dec_pred_wr_sel),
        .dec_pred_rd_sel (dec_pred_rd_sel),
        .dec_wb_src (dec_wb_src),
        .dec_use_imm (dec_use_imm),
        .dec_uses_rA (dec_uses_rA),
        .dec_uses_rB (dec_uses_rB),
        .dec_is_fma (dec_is_fma),
        .dec_is_st (dec_is_st),
        .dec_is_branch (dec_is_branch),
        .dec_is_pbra (dec_is_pbra),
        .dec_is_ret (dec_is_ret),
        .dec_is_ld (dec_is_ld),
        .dec_is_store (dec_is_store),
        .dec_is_lds (dec_is_lds),
        .dec_is_sts (dec_is_sts),
        .dec_is_wmma_mma (dec_is_wmma_mma),
        .dec_is_wmma_load (dec_is_wmma_load),
        .dec_is_wmma_store (dec_is_wmma_store),
        .dec_wmma_sel (dec_wmma_sel),
        .dec_branch_target (dec_branch_target)
    );

    // ================================================================
    // DE Pipeline Register (Decode → RF stage boundary)
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

    // Pre-computed RF read addresses for the pipeline path
    reg [3:0] de_rf_r0_addr;
    reg [3:0] de_rf_r1_addr;
    reg [3:0] de_rf_r2_addr;
    reg [3:0] de_rf_r3_addr;

    // DE flush: invalidate DE on branch redirect or RET
    wire de_flush = branch_taken | ret_detected;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            de_valid <= 1'b0;
            de_rf_we <= 1'b0;
            de_pred_we <= 1'b0;
            de_opcode <= 5'd0;
            de_dt <= 1'b0;
            de_cmp_mode <= 2'd0;
            de_rD_addr <= 4'd0;
            de_rA_addr <= 4'd0;
            de_rB_addr <= 4'd0;
            de_rC_addr <= 4'd0;
            de_imm16 <= 16'd0;
            de_pred_wr_sel <= 2'd0;
            de_pred_rd_sel <= 2'd0;
            de_wb_src <= 3'd0;
            de_use_imm <= 1'b0;
            de_uses_rA <= 1'b0;
            de_uses_rB <= 1'b0;
            de_is_fma <= 1'b0;
            de_is_st <= 1'b0;
            de_is_branch <= 1'b0;
            de_is_pbra <= 1'b0;
            de_is_ret <= 1'b0;
            de_is_ld <= 1'b0;
            de_is_store <= 1'b0;
            de_is_lds <= 1'b0;
            de_is_sts <= 1'b0;
            de_is_wmma_mma <= 1'b0;
            de_is_wmma_load <= 1'b0;
            de_is_wmma_store <= 1'b0;
            de_wmma_sel <= 2'd0;
            de_branch_target <= {`GPU_PC_WIDTH{1'b0}};
            de_rf_r0_addr <= 4'd0;
            de_rf_r1_addr <= 4'd0;
            de_rf_r2_addr <= 4'd0;
            de_rf_r3_addr <= 4'd0;
        end else if (de_flush) begin
            de_valid <= 1'b0;
            de_rf_we <= 1'b0;
            de_pred_we <= 1'b0;
        end else if (!front_stall) begin
            de_opcode <= dec_opcode;
            de_dt <= dec_dt;
            de_cmp_mode <= dec_cmp_mode;
            de_rD_addr <= dec_rD_addr;
            de_rA_addr <= dec_rA_addr;
            de_rB_addr <= dec_rB_addr;
            de_rC_addr <= dec_rC_addr;
            de_imm16 <= dec_imm16;
            de_rf_we <= dec_rf_we;
            de_pred_we <= dec_pred_we;
            de_pred_wr_sel <= dec_pred_wr_sel;
            de_pred_rd_sel <= dec_pred_rd_sel;
            de_wb_src <= dec_wb_src;
            de_use_imm <= dec_use_imm;
            de_uses_rA <= dec_uses_rA;
            de_uses_rB <= dec_uses_rB;
            de_is_fma <= dec_is_fma;
            de_is_st <= dec_is_st;
            de_is_branch <= dec_is_branch;
            de_is_pbra <= dec_is_pbra;
            de_is_ret <= dec_is_ret;
            de_is_ld <= dec_is_ld;
            de_is_store <= dec_is_store;
            de_is_lds <= dec_is_lds;
            de_is_sts <= dec_is_sts;
            de_is_wmma_mma <= dec_is_wmma_mma;
            de_is_wmma_load <= dec_is_wmma_load;
            de_is_wmma_store <= dec_is_wmma_store;
            de_wmma_sel <= dec_wmma_sel;
            de_branch_target <= dec_branch_target;
            de_valid <= fetch_valid;

            // Pre-compute RF addresses
            de_rf_r0_addr <= dec_rA_addr;
            de_rf_r1_addr <= dec_rB_addr;
            de_rf_r2_addr <= (dec_is_fma | dec_is_st) ? dec_rD_addr : dec_rC_addr;
            de_rf_r3_addr <= 4'd0;
        end
        // else: front_stall — DE holds
    end

    // ================================================================
    // Stall / Flush / Issue Control (uses DE-stage signals)
    // ================================================================
    wire de_wmma_any = de_is_wmma_mma | de_is_wmma_load | de_is_wmma_store;

    wire pipeline_drained;
    wire wmma_drain_wait = de_valid & de_wmma_any & ~pipeline_drained
                         & ~tc_busy & ~burst_busy;

    wire sb_stall_gated = sb_stall & de_valid;

    assign front_stall = sb_stall_gated | any_ex_busy | tc_busy | burst_busy
                       | wmma_drain_wait;
    assign sp_stall = any_ex_busy | tc_busy | burst_busy;

    wire id_can_issue = de_valid & ~front_stall & ~de_wmma_any;

    wire id_issue_ctrl = de_valid & ~front_stall;
    assign branch_taken = id_issue_ctrl & de_is_branch;
    assign branch_target = de_branch_target;
    assign ret_detected = id_issue_ctrl & de_is_ret;

    wire sb_issue = id_can_issue & de_rf_we;

    // ================================================================
    // Scoreboard (uses DE-stage addresses — unchanged)
    // ================================================================
    wire [3:0] wb_active_mask_sb = {sp_wb_active[3], sp_wb_active[2],
                                    sp_wb_active[1], sp_wb_active[0]};

    wire sb_any_pending;

    scoreboard u_sb (
        .clk (clk), .rst_n (rst_n),
        .rA_addr (de_rA_addr),
        .rB_addr (de_rB_addr),
        .rD_addr (de_rD_addr),
        .uses_rA (de_uses_rA),
        .uses_rB (de_uses_rB),
        .is_fma (de_is_fma),
        .is_st (de_is_st),
        .rf_we (de_rf_we),
        .active_mask (active_mask),
        .issue (sb_issue),
        .wb_rD_addr (sp_wb_rD_addr[0]),
        .wb_rf_we (sp_wb_rf_we[0]),
        .wb_active_mask (wb_active_mask_sb),
        .stall (sb_stall),
        .any_pending (sb_any_pending)
    );

    // ================================================================
    // Tensor Core Top (uses DE-stage addresses — registered override)
    // ================================================================
    assign pipeline_drained = ~sb_any_pending & ~any_ex_busy;

    wire tc_trigger = de_valid & de_is_wmma_mma & ~tc_busy & ~burst_busy
                    & pipeline_drained;

    tc_top u_tc_top (
        .clk (clk), .rst_n (rst_n),
        .trigger (tc_trigger),
        .dec_rA_addr (de_rA_addr),
        .dec_rB_addr (de_rB_addr),
        .dec_rC_addr (de_rC_addr),
        .dec_rD_addr (de_rD_addr),
        .sp_rf_r0_data (flat_ovr_rf_r0),
        .sp_rf_r1_data (flat_ovr_rf_r1),
        .sp_rf_r2_data (flat_ovr_rf_r2),
        .sp_rf_r3_data (flat_ovr_rf_r3),
        .busy (tc_busy),
        .rf_addr_override (tc_rf_override),
        .rf_r0_addr (tc_rf_r0),
        .rf_r1_addr (tc_rf_r1),
        .rf_r2_addr (tc_rf_r2),
        .rf_r3_addr (tc_rf_r3),
        .scat_w1_addr (tc_w1_addr),
        .scat_w1_data (tc_w1_data),
        .scat_w1_we (tc_w1_we),
        .scat_w2_addr (tc_w2_addr),
        .scat_w2_data (tc_w2_data),
        .scat_w2_we (tc_w2_we),
        .scat_w3_addr (tc_w3_addr),
        .scat_w3_data (tc_w3_data),
        .scat_w3_we (tc_w3_we)
    );

    // ================================================================
    // Burst Controller — WMMA.LOAD / WMMA.STORE (v1.6 pipelined)
    // ================================================================
    localparam [2:0] BU_IDLE       = 3'd0,
                     BU_SETUP      = 3'd1,
                     BU_RREAD      = 3'd2,
                     BU_ADDR       = 3'd3,
                     BU_LOAD_ADDR  = 3'd4,
                     BU_LOAD_BEAT  = 3'd5,
                     BU_STORE_READ = 3'd6,
                     BU_STORE_BEAT = 3'd7;

    reg [2:0] bu_state;
    reg [1:0] bu_beat;
    assign burst_busy = (bu_state != BU_IDLE);

    reg [3:0] bu_rD_base;
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] bu_base_addr [0:3];
    reg [15:0] bu_store_data [0:3][0:3];

    // Pipelined BU registers (v1.6)
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] bu_rf_data [0:3];
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] bu_imm16_r;
    reg bu_is_store;

    wire bu_load_trigger = de_valid & de_is_wmma_load & ~tc_busy
                         & ~burst_busy & pipeline_drained;
    wire bu_store_trigger = de_valid & de_is_wmma_store & ~tc_busy
                          & ~burst_busy & pipeline_drained;

    // Burst RF addr override — REGISTERED
    reg bu_rf_override_r;
    reg [3:0] bu_rf_r0_r, bu_rf_r1_r, bu_rf_r2_r, bu_rf_r3_r;

    // Burst ext write (load beats → W1)
    reg [3:0] bu_w1_addr;
    reg bu_w1_we;

    // Burst DMEM override
    reg bu_dmem_override;

    integer bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bu_state <= BU_IDLE;
            bu_beat <= 2'd0;
            bu_rD_base <= 4'd0;
            bu_rf_override_r <= 1'b0;
            bu_rf_r0_r <= 4'd0;
            bu_rf_r1_r <= 4'd0;
            bu_rf_r2_r <= 4'd0;
            bu_rf_r3_r <= 4'd0;
            bu_is_store <= 1'b0;
            bu_imm16_r <= {`GPU_DMEM_ADDR_WIDTH{1'b0}};
            for (bi = 0; bi < 4; bi = bi + 1) begin
                bu_base_addr[bi] <= {`GPU_DMEM_ADDR_WIDTH{1'b0}};
                bu_rf_data[bi] <= {`GPU_DMEM_ADDR_WIDTH{1'b0}};
                bu_store_data[bi][0] <= 16'd0;
                bu_store_data[bi][1] <= 16'd0;
                bu_store_data[bi][2] <= 16'd0;
                bu_store_data[bi][3] <= 16'd0;
            end
        end else begin
            case (bu_state)
                BU_IDLE: begin
                    if (bu_load_trigger | bu_store_trigger) begin
                        bu_rD_base <= de_rD_addr;
                        bu_imm16_r <= de_imm16[`GPU_DMEM_ADDR_WIDTH-1:0];
                        bu_is_store <= bu_store_trigger;
                        // Set override: read base address register (rA)
                        bu_rf_override_r <= 1'b1;
                        bu_rf_r0_r <= de_rf_r0_addr; // = de_rA_addr
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
                    if (bu_beat == 2'd3)
                        bu_state <= BU_IDLE;
                    else
                        bu_beat <= bu_beat + 2'd1;
                end
                BU_STORE_READ: begin
                    for (bi = 0; bi < 4; bi = bi + 1) begin
                        bu_store_data[bi][0] <= sp_ovr_rf_r0_data[bi];
                        bu_store_data[bi][1] <= sp_ovr_rf_r1_data[bi];
                        bu_store_data[bi][2] <= sp_ovr_rf_r2_data[bi];
                        bu_store_data[bi][3] <= sp_ovr_rf_r3_data[bi];
                    end
                    bu_beat <= 2'd0;
                    bu_state <= BU_STORE_BEAT;
                    bu_rf_override_r <= 1'b0;
                end
                BU_STORE_BEAT: begin
                    if (bu_beat == 2'd3)
                        bu_state <= BU_IDLE;
                    else
                        bu_beat <= bu_beat + 2'd1;
                end
                default: bu_state <= BU_IDLE;
            endcase
        end
    end

    // Burst combinational outputs
    always @(*) begin
        bu_w1_addr = 4'd0;
        bu_w1_we = 1'b0;
        bu_dmem_override = 1'b0;

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
    // Override RF Read Address Mux (for TC/BU/debug — NOT pipeline)
    // ================================================================
    // This mux drives the OVERRIDE read ports of gpr_regfile only.
    // Pipeline read ports are driven by rr_rf_r*_addr directly — no mux.
    // All mux select inputs are registered FFs:
    //   tc_rf_override: registered in tc_top v1.1
    //   bu_rf_override_r: registered in BU FSM
    //   fu_running: registered in fetch_unit
    always @(*) begin
        ovr_rf_r0_addr_mux = 4'd0;
        ovr_rf_r1_addr_mux = 4'd0;
        ovr_rf_r2_addr_mux = 4'd0;
        ovr_rf_r3_addr_mux = 4'd0;

        if (tc_rf_override) begin
            ovr_rf_r0_addr_mux = tc_rf_r0;
            ovr_rf_r1_addr_mux = tc_rf_r1;
            ovr_rf_r2_addr_mux = tc_rf_r2;
            ovr_rf_r3_addr_mux = tc_rf_r3;
        end else if (bu_rf_override_r) begin
            ovr_rf_r0_addr_mux = bu_rf_r0_r;
            ovr_rf_r1_addr_mux = bu_rf_r1_r;
            ovr_rf_r2_addr_mux = bu_rf_r2_r;
            ovr_rf_r3_addr_mux = bu_rf_r3_r;
        end else if (!fu_running) begin
            ovr_rf_r0_addr_mux = debug_rf_addr;
        end
    end

    // ================================================================
    // RR Pipeline Register (DE → Register Read stage boundary)
    // ================================================================
    // Pipeline read addresses go from DE directly to RR — no override
    // mux in this path.  Override mux is only for TC/BU/debug ports.
    reg [4:0] rr_opcode;
    reg rr_dt;
    reg [1:0] rr_cmp_mode;
    reg [3:0] rr_rD_addr;
    reg [15:0] rr_imm16;
    reg rr_rf_we, rr_pred_we;
    reg [1:0] rr_pred_wr_sel, rr_pred_rd_sel;
    reg [2:0] rr_wb_src;
    reg rr_use_imm;
    reg rr_valid;

    // Registered pipeline RF read addresses
    reg [3:0] rr_rf_r0_addr, rr_rf_r1_addr, rr_rf_r2_addr, rr_rf_r3_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rr_valid <= 1'b0;
            rr_rf_we <= 1'b0;
            rr_pred_we <= 1'b0;
            rr_opcode <= 5'd0;
            rr_dt <= 1'b0;
            rr_cmp_mode <= 2'd0;
            rr_rD_addr <= 4'd0;
            rr_imm16 <= 16'd0;
            rr_pred_wr_sel <= 2'd0;
            rr_pred_rd_sel <= 2'd0;
            rr_wb_src <= 3'd0;
            rr_use_imm <= 1'b0;
            rr_rf_r0_addr <= 4'd0;
            rr_rf_r1_addr <= 4'd0;
            rr_rf_r2_addr <= 4'd0;
            rr_rf_r3_addr <= 4'd0;
        end else if (!sp_stall) begin
            if (!front_stall) begin
                // Normal advance: DE → RR
                rr_valid <= de_valid & ~de_wmma_any;
                rr_opcode <= de_opcode;
                rr_dt <= de_dt;
                rr_cmp_mode <= de_cmp_mode;
                rr_rD_addr <= de_rD_addr;
                rr_imm16 <= de_imm16;
                rr_rf_we <= de_rf_we;
                rr_pred_we <= de_pred_we;
                rr_pred_wr_sel <= de_pred_wr_sel;
                rr_pred_rd_sel <= de_pred_rd_sel;
                rr_wb_src <= de_wb_src;
                rr_use_imm <= de_use_imm;
                // Pipeline path: DE addresses directly (no override mux!)
                rr_rf_r0_addr <= de_rf_r0_addr;
                rr_rf_r1_addr <= de_rf_r1_addr;
                rr_rf_r2_addr <= de_rf_r2_addr;
                rr_rf_r3_addr <= de_rf_r3_addr;
            end else begin
                // Front stalled → RR drains, then goes empty
                rr_valid <= 1'b0;
                rr_rf_we <= 1'b0;
                rr_pred_we <= 1'b0;
            end
        end
        // else: sp_stall — RR holds
    end

    // Debug RF data (override port 0 with debug_rf_addr)
    assign debug_rf_data = {sp_ovr_rf_r0_data[3], sp_ovr_rf_r0_data[2],
                            sp_ovr_rf_r0_data[1], sp_ovr_rf_r0_data[0]};

    // ================================================================
    // External RF Write Mux (TC scatter / burst load — unchanged)
    // ================================================================
    reg [3:0] ext_w1_addr [0:3];
    reg [15:0] ext_w1_data [0:3];
    reg ext_w1_we [0:3];
    reg [3:0] ext_w2_addr [0:3];
    reg [15:0] ext_w2_data [0:3];
    reg ext_w2_we [0:3];
    reg [3:0] ext_w3_addr [0:3];
    reg [15:0] ext_w3_data [0:3];
    reg ext_w3_we [0:3];

    integer wi;
    always @(*) begin
        for (wi = 0; wi < 4; wi = wi + 1) begin
            ext_w1_addr[wi] = tc_w1_addr;
            ext_w1_data[wi] = tc_w1_data[wi*16 +: 16];
            ext_w1_we[wi] = tc_w1_we[wi];
            ext_w2_addr[wi] = tc_w2_addr;
            ext_w2_data[wi] = tc_w2_data[wi*16 +: 16];
            ext_w2_we[wi] = tc_w2_we[wi];
            ext_w3_addr[wi] = tc_w3_addr;
            ext_w3_data[wi] = tc_w3_data[wi*16 +: 16];
            ext_w3_we[wi] = tc_w3_we[wi];

            if (bu_w1_we) begin
                ext_w1_addr[wi] = bu_w1_addr;
                ext_w1_data[wi] = dmem_douta[wi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
                ext_w1_we[wi] = 1'b1;
            end
        end
    end

    // ================================================================
    // Generate: 4× SP Core — dual RF read ports (v1.7)
    // ================================================================
    // Pipeline reads: rr_rf_r*_addr → sp_core ppl ports → id_ex (internal)
    //   Critical path: rr_rf_r*_addr(FF) → RF 16:1 mux → id_ex_opB(FF) ≈ 5-6ns
    // Override reads: ovr_rf_r*_addr_mux → sp_core ovr ports → external data
    //   Used by: TC gather (a_hold), BU (bu_rf_data, bu_store_data), debug
    genvar t;
    generate
        for (t = 0; t < 4; t = t + 1) begin : SP_LANE
            sp_core #(.TID(t[1:0])) u_sp (
                .clk (clk), .rst_n (rst_n),
                .stall (sp_stall),
                .flush_id (1'b0),
                // Pipeline RF read (registered addresses, no mux in path)
                .ppl_rf_r0_addr (rr_rf_r0_addr),
                .ppl_rf_r1_addr (rr_rf_r1_addr),
                .ppl_rf_r2_addr (rr_rf_r2_addr),
                .ppl_rf_r3_addr (rr_rf_r3_addr),
                // Override RF read (for TC/BU/debug)
                .ovr_rf_r0_addr (ovr_rf_r0_addr_mux),
                .ovr_rf_r1_addr (ovr_rf_r1_addr_mux),
                .ovr_rf_r2_addr (ovr_rf_r2_addr_mux),
                .ovr_rf_r3_addr (ovr_rf_r3_addr_mux),
                .ovr_rf_r0_data (sp_ovr_rf_r0_data[t]),
                .ovr_rf_r1_data (sp_ovr_rf_r1_data[t]),
                .ovr_rf_r2_data (sp_ovr_rf_r2_data[t]),
                .ovr_rf_r3_data (sp_ovr_rf_r3_data[t]),
                // Predicate
                .pred_rd_sel (rr_pred_rd_sel),
                .pred_rd_val (sp_pred_rd_val[t]),
                // Control from RR stage
                .id_opcode (rr_opcode),
                .id_dt (rr_dt),
                .id_cmp_mode (rr_cmp_mode),
                .id_rf_we (rr_rf_we),
                .id_pred_we (rr_pred_we),
                .id_rD_addr (rr_rD_addr),
                .id_pred_wr_sel (rr_pred_wr_sel),
                .id_valid (rr_valid),
                .id_active (active_mask[t]),
                .id_wb_src (rr_wb_src),
                .id_use_imm (rr_use_imm),
                .id_imm16 (rr_imm16),
                // EX/MEM outputs
                .ex_mem_result_out (sp_ex_mem_result[t]),
                .ex_mem_store_out (sp_ex_mem_store[t]),
                .ex_mem_valid_out (sp_ex_mem_valid[t]),
                .ex_busy (sp_ex_busy[t]),
                .mem_rdata (dmem_dout_a[t]),
                // External writes
                .wb_ext_w1_addr (ext_w1_addr[t]),
                .wb_ext_w1_data (ext_w1_data[t]),
                .wb_ext_w1_we (ext_w1_we[t]),
                .wb_ext_w2_addr (ext_w2_addr[t]),
                .wb_ext_w2_data (ext_w2_data[t]),
                .wb_ext_w2_we (ext_w2_we[t]),
                .wb_ext_w3_addr (ext_w3_addr[t]),
                .wb_ext_w3_data (ext_w3_data[t]),
                .wb_ext_w3_we (ext_w3_we[t]),
                // MEM/WB outputs
                .mem_is_load (sp_mem_is_load[t]),
                .mem_is_store (sp_mem_is_store[t]),
                .wb_rD_addr (sp_wb_rD_addr[t]),
                .wb_rf_we (sp_wb_rf_we[t]),
                .wb_active (sp_wb_active[t]),
                .wb_valid (sp_wb_valid[t])
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

            assign dmem_addr_a[d] = bu_load_phase  ? bu_ld_addr :
                                    bu_store_phase  ? (bu_base_addr[d] + {{(`GPU_DMEM_ADDR_WIDTH-2){1'b0}}, bu_beat}) :
                                    sp_ex_mem_result[d][`GPU_DMEM_ADDR_WIDTH-1:0];

            assign dmem_din_a[d] = bu_store_phase ? bu_store_data[d][bu_beat] :
                                   sp_ex_mem_store[d];

            assign dmem_we_a[d] = bu_store_phase ? 1'b1 :
                                  (sp_mem_is_store[d] & sp_ex_mem_valid[d] & ~sp_stall);
        end
    endgenerate

endmodule

`endif // SM_CORE_V

