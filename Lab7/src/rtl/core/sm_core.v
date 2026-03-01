/* file: sm_core.v
 * SM core — all GPU pipeline logic.
 * Integrates: fetch unit, SM decoder, 4× SP cores,
 * scoreboard, TC top (tensor core controller), and
 * inline burst controller (WMMA.LOAD/STORE).
 *
 * BRAMs (IMEM + per-SP DMEM) are NOT instantiated here.
 * Port-A BRAM signals are exposed as module I/O.
 *
 * Pipeline: IF → ID → EX → MEM → WB
 *
 * Author: Jeremy Cai
 * Date: Mar. 1, 2026
 * Version: 2.0 — tc_top extracted, sm_decoder renamed
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

    // ── IMEM BRAM port-A (read-only by GPU) ──────────────
    output wire [`GPU_IMEM_ADDR_WIDTH-1:0] imem_addr,
    input wire [`GPU_IMEM_DATA_WIDTH-1:0] imem_rdata,

    // ── Per-SP DMEM BRAM port-A (GPU read/write) ─────────
    // Flat buses: bits [W*t +: W] = SP t
    output wire [4*`GPU_DMEM_ADDR_WIDTH-1:0] dmem_addra,
    output wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_dina,
    output wire [3:0] dmem_wea,
    input wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_douta,

    // ── Kernel control ───────────────────────────────────
    input wire kernel_start,
    input wire [`GPU_PC_WIDTH-1:0] kernel_entry_pc,
    output wire kernel_done,

    // ── Debug RF read (testbench observation) ────────────
    input wire [3:0] debug_rf_addr,
    output wire [4*`GPU_DMEM_DATA_WIDTH-1:0] debug_rf_data
);

    // ================================================================
    // 1. Internal signal declarations
    // ================================================================
    wire [`GPU_PC_WIDTH-1:0] if_id_pc;
    wire fetch_valid;
    wire fu_running;

    // Decoder outputs
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

    // Per-SP signals (arrays)
    wire [15:0] sp_rf_r0_data [0:3];
    wire [15:0] sp_rf_r1_data [0:3];
    wire [15:0] sp_rf_r2_data [0:3];
    wire [15:0] sp_rf_r3_data [0:3];
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

    // Flat bus unpack/pack
    genvar u;
    generate
        for (u = 0; u < 4; u = u + 1) begin : DMEM_IO
            assign dmem_dout_a[u] = dmem_douta[`GPU_DMEM_DATA_WIDTH*u +: `GPU_DMEM_DATA_WIDTH];
            assign dmem_addra[`GPU_DMEM_ADDR_WIDTH*u +: `GPU_DMEM_ADDR_WIDTH] = dmem_addr_a[u];
            assign dmem_dina[`GPU_DMEM_DATA_WIDTH*u +: `GPU_DMEM_DATA_WIDTH] = dmem_din_a[u];
            assign dmem_wea[u] = dmem_we_a[u];
        end
    endgenerate

    // RF read address bus (muxed: normal / TC gather / burst)
    reg [3:0] rf_r0_addr_mux, rf_r1_addr_mux, rf_r2_addr_mux, rf_r3_addr_mux;

    // Stall / flush
    wire sb_stall;
    wire any_ex_busy = sp_ex_busy[0] | sp_ex_busy[1] | sp_ex_busy[2] | sp_ex_busy[3];
    wire tc_busy;
    wire burst_busy;
    wire front_stall;
    wire sp_stall;
    wire sp_flush_id;

    // Active mask (Phase 1–3: all active)
    reg [3:0] active_mask;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) active_mask <= 4'b1111;
        else if (kernel_start) active_mask <= 4'b1111;
    end

    wire wmma_any = dec_is_wmma_mma | dec_is_wmma_load | dec_is_wmma_store;

    // TC top interface wires
    wire tc_rf_override;
    wire [3:0] tc_rf_r0, tc_rf_r1, tc_rf_r2, tc_rf_r3;
    wire [3:0] tc_w1_addr, tc_w2_addr, tc_w3_addr;
    wire [4*16-1:0] tc_w1_data, tc_w2_data, tc_w3_data;
    wire [3:0] tc_w1_we, tc_w2_we, tc_w3_we;

    // Flat RF read buses for tc_top
    wire [4*16-1:0] flat_rf_r0 = {sp_rf_r0_data[3], sp_rf_r0_data[2],
                                   sp_rf_r0_data[1], sp_rf_r0_data[0]};
    wire [4*16-1:0] flat_rf_r1 = {sp_rf_r1_data[3], sp_rf_r1_data[2],
                                   sp_rf_r1_data[1], sp_rf_r1_data[0]};
    wire [4*16-1:0] flat_rf_r2 = {sp_rf_r2_data[3], sp_rf_r2_data[2],
                                   sp_rf_r2_data[1], sp_rf_r2_data[0]};
    wire [4*16-1:0] flat_rf_r3 = {sp_rf_r3_data[3], sp_rf_r3_data[2],
                                   sp_rf_r3_data[1], sp_rf_r3_data[0]};

    // ================================================================
    // 2. Fetch Unit
    // ================================================================
    wire branch_taken;
    wire [`GPU_PC_WIDTH-1:0] branch_target;
    wire ret_detected;

    fetch_unit u_fetch (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr),
        .kernel_start(kernel_start),
        .kernel_entry_pc(kernel_entry_pc),
        .kernel_done(kernel_done),
        .running(fu_running),
        .branch_taken(branch_taken),
        .branch_target(branch_target),
        .front_stall(front_stall),
        .ret_detected(ret_detected),
        .if_id_pc(if_id_pc),
        .fetch_valid(fetch_valid)
    );

    // ================================================================
    // 2b. IR Latch — hold instruction during scoreboard stall
    //
    // Problem: sync-read BRAM delivers instruction one cycle after
    // PC presents the address. By the time a stall fires (combinational
    // from decoder output), PC has already advanced and the BRAM will
    // overwrite the stalled instruction next cycle.
    //
    // Fix: on the first cycle of a stall, latch imem_rdata. Feed the
    // latch to the decoder on subsequent stall cycles. When the stall
    // clears, the latched instruction issues and ir_latched resets.
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
    // 3. SM Decoder
    // ================================================================
    sm_decoder u_dec (
        .ir(dec_ir),
        .dec_opcode(dec_opcode),
        .dec_dt(dec_dt),
        .dec_cmp_mode(dec_cmp_mode),
        .dec_rD_addr(dec_rD_addr),
        .dec_rA_addr(dec_rA_addr),
        .dec_rB_addr(dec_rB_addr),
        .dec_rC_addr(dec_rC_addr),
        .dec_imm16(dec_imm16),
        .dec_rf_we(dec_rf_we),
        .dec_pred_we(dec_pred_we),
        .dec_pred_wr_sel(dec_pred_wr_sel),
        .dec_pred_rd_sel(dec_pred_rd_sel),
        .dec_wb_src(dec_wb_src),
        .dec_use_imm(dec_use_imm),
        .dec_uses_rA(dec_uses_rA),
        .dec_uses_rB(dec_uses_rB),
        .dec_is_fma(dec_is_fma),
        .dec_is_st(dec_is_st),
        .dec_is_branch(dec_is_branch),
        .dec_is_pbra(dec_is_pbra),
        .dec_is_ret(dec_is_ret),
        .dec_is_ld(dec_is_ld),
        .dec_is_store(dec_is_store),
        .dec_is_lds(dec_is_lds),
        .dec_is_sts(dec_is_sts),
        .dec_is_wmma_mma(dec_is_wmma_mma),
        .dec_is_wmma_load(dec_is_wmma_load),
        .dec_is_wmma_store(dec_is_wmma_store),
        .dec_wmma_sel(dec_wmma_sel),
        .dec_branch_target(dec_branch_target)
    );

    // ================================================================
    // 4. Stall / Flush / Issue Control
    // ================================================================
    // WMMA ops must wait for pipeline to drain before triggering.
    // Stall the front-end while waiting to prevent PC from advancing.
    wire wmma_drain_wait = fetch_valid & wmma_any & ~pipeline_drained
                         & ~tc_busy & ~burst_busy;
    assign front_stall = sb_stall | any_ex_busy | tc_busy | burst_busy
                       | wmma_drain_wait;
    assign sp_stall = any_ex_busy | tc_busy | burst_busy;
    assign sp_flush_id = sb_stall & ~sp_stall;

    wire id_can_issue = fetch_valid & ~front_stall & ~wmma_any;
    wire sp_id_valid = fetch_valid & ~wmma_any;

    wire id_issue_ctrl = fetch_valid & ~front_stall;
    assign branch_taken = id_issue_ctrl & dec_is_branch;
    assign branch_target = dec_branch_target;
    assign ret_detected = id_issue_ctrl & dec_is_ret;

    wire sb_issue = id_can_issue & dec_rf_we;

    // ================================================================
    // 5. Scoreboard
    // ================================================================
    wire [3:0] wb_active_mask_sb = {sp_wb_active[3], sp_wb_active[2],
                                    sp_wb_active[1], sp_wb_active[0]};

    scoreboard u_sb (
        .clk(clk), .rst_n(rst_n),
        .rA_addr(dec_rA_addr),
        .rB_addr(dec_rB_addr),
        .rD_addr(dec_rD_addr),
        .uses_rA(dec_uses_rA),
        .uses_rB(dec_uses_rB),
        .is_fma(dec_is_fma),
        .is_st(dec_is_st),
        .rf_we(dec_rf_we),
        .active_mask(active_mask),
        .issue(sb_issue),
        .wb_rD_addr(sp_wb_rD_addr[0]),
        .wb_rf_we(sp_wb_rf_we[0]),
        .wb_active_mask(wb_active_mask_sb),
        .stall(sb_stall),
        .any_pending(sb_any_pending)
    );

    wire sb_any_pending;

    // ================================================================
    // 6. Tensor Core Top
    // ================================================================
    // WMMA ops must wait for pipeline to drain — the TC gather reads RF
    // directly, bypassing the scoreboard hazard check.
    wire pipeline_drained = ~sb_any_pending & ~any_ex_busy;

    wire tc_trigger = fetch_valid & dec_is_wmma_mma & ~tc_busy & ~burst_busy
                    & pipeline_drained;

    tc_top u_tc_top (
        .clk(clk), .rst_n(rst_n),
        .trigger(tc_trigger),
        .dec_rA_addr(dec_rA_addr),
        .dec_rB_addr(dec_rB_addr),
        .dec_rC_addr(dec_rC_addr),
        .dec_rD_addr(dec_rD_addr),
        .sp_rf_r0_data(flat_rf_r0),
        .sp_rf_r1_data(flat_rf_r1),
        .sp_rf_r2_data(flat_rf_r2),
        .sp_rf_r3_data(flat_rf_r3),
        .busy(tc_busy),
        .rf_addr_override(tc_rf_override),
        .rf_r0_addr(tc_rf_r0),
        .rf_r1_addr(tc_rf_r1),
        .rf_r2_addr(tc_rf_r2),
        .rf_r3_addr(tc_rf_r3),
        .scat_w1_addr(tc_w1_addr),
        .scat_w1_data(tc_w1_data),
        .scat_w1_we(tc_w1_we),
        .scat_w2_addr(tc_w2_addr),
        .scat_w2_data(tc_w2_data),
        .scat_w2_we(tc_w2_we),
        .scat_w3_addr(tc_w3_addr),
        .scat_w3_data(tc_w3_data),
        .scat_w3_we(tc_w3_we)
    );

    // ================================================================
    // 7. Burst Controller — WMMA.LOAD / WMMA.STORE (inline)
    // ================================================================
    localparam [2:0] BU_IDLE       = 3'd0,
                     BU_LOAD_ADDR  = 3'd1,
                     BU_LOAD_BEAT  = 3'd2,
                     BU_STORE_READ = 3'd3,
                     BU_STORE_BEAT = 3'd4;

    reg [2:0] bu_state;
    reg [1:0] bu_beat;
    assign burst_busy = (bu_state != BU_IDLE);

    reg [3:0] bu_rD_base;
    reg [`GPU_DMEM_ADDR_WIDTH-1:0] bu_base_addr [0:3];
    reg [15:0] bu_store_data [0:3][0:3];

    wire bu_load_trigger = fetch_valid & dec_is_wmma_load & ~tc_busy
                         & ~burst_busy & pipeline_drained;
    wire bu_store_trigger = fetch_valid & dec_is_wmma_store & ~tc_busy
                          & ~burst_busy & pipeline_drained;

    // Burst RF addr override
    reg bu_rf_override;
    reg [3:0] bu_rf_r0, bu_rf_r1, bu_rf_r2, bu_rf_r3;

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
            for (bi = 0; bi < 4; bi = bi + 1) begin
                bu_base_addr[bi] <= {`GPU_DMEM_ADDR_WIDTH{1'b0}};
                bu_store_data[bi][0] <= 16'd0;
                bu_store_data[bi][1] <= 16'd0;
                bu_store_data[bi][2] <= 16'd0;
                bu_store_data[bi][3] <= 16'd0;
            end
        end else begin
            case (bu_state)
                BU_IDLE: begin
                    if (bu_load_trigger) begin
                        bu_rD_base <= dec_rD_addr;
                        for (bi = 0; bi < 4; bi = bi + 1)
                            bu_base_addr[bi] <= sp_rf_r0_data[bi][`GPU_DMEM_ADDR_WIDTH-1:0]
                                              + dec_imm16[`GPU_DMEM_ADDR_WIDTH-1:0];
                        bu_beat <= 2'd0;
                        bu_state <= BU_LOAD_ADDR;
                    end else if (bu_store_trigger) begin
                        bu_rD_base <= dec_rD_addr;
                        for (bi = 0; bi < 4; bi = bi + 1)
                            bu_base_addr[bi] <= sp_rf_r0_data[bi][`GPU_DMEM_ADDR_WIDTH-1:0]
                                              + dec_imm16[`GPU_DMEM_ADDR_WIDTH-1:0];
                        bu_beat <= 2'd0;
                        bu_state <= BU_STORE_READ;
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
                        bu_store_data[bi][0] <= sp_rf_r0_data[bi];
                        bu_store_data[bi][1] <= sp_rf_r1_data[bi];
                        bu_store_data[bi][2] <= sp_rf_r2_data[bi];
                        bu_store_data[bi][3] <= sp_rf_r3_data[bi];
                    end
                    bu_beat <= 2'd0;
                    bu_state <= BU_STORE_BEAT;
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
        bu_rf_override = 1'b0;
        bu_rf_r0 = 4'd0; bu_rf_r1 = 4'd0;
        bu_rf_r2 = 4'd0; bu_rf_r3 = 4'd0;
        bu_w1_addr = 4'd0;
        bu_w1_we = 1'b0;
        bu_dmem_override = 1'b0;

        case (bu_state)
            BU_STORE_READ: begin
                bu_rf_override = 1'b1;
                bu_rf_r0 = bu_rD_base;
                bu_rf_r1 = bu_rD_base + 4'd1;
                bu_rf_r2 = bu_rD_base + 4'd2;
                bu_rf_r3 = bu_rD_base + 4'd3;
            end
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
    // 8. RF Read Address Mux
    // ================================================================
    always @(*) begin
        // Default: decoder-driven
        rf_r0_addr_mux = dec_rA_addr;
        rf_r1_addr_mux = dec_rB_addr;
        rf_r2_addr_mux = (dec_is_fma | dec_is_st) ? dec_rD_addr : dec_rC_addr;
        rf_r3_addr_mux = 4'd0;

        if (tc_rf_override) begin
            rf_r0_addr_mux = tc_rf_r0;
            rf_r1_addr_mux = tc_rf_r1;
            rf_r2_addr_mux = tc_rf_r2;
            rf_r3_addr_mux = tc_rf_r3;
        end else if (bu_rf_override) begin
            rf_r0_addr_mux = bu_rf_r0;
            rf_r1_addr_mux = bu_rf_r1;
            rf_r2_addr_mux = bu_rf_r2;
            rf_r3_addr_mux = bu_rf_r3;
        end else if (!fu_running) begin
            // Debug: when idle, TB drives rf_r0 via debug_rf_addr
            rf_r0_addr_mux = debug_rf_addr;
        end
    end

    // Debug RF data: combinational read from all 4 SPs' port 0
    assign debug_rf_data = {sp_rf_r0_data[3], sp_rf_r0_data[2],
                            sp_rf_r0_data[1], sp_rf_r0_data[0]};

    // ================================================================
    // 9. External RF Write Mux (TC scatter / burst load)
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
            // TC scatter
            ext_w1_addr[wi] = tc_w1_addr;
            ext_w1_data[wi] = tc_w1_data[wi*16 +: 16];
            ext_w1_we[wi] = tc_w1_we[wi];
            ext_w2_addr[wi] = tc_w2_addr;
            ext_w2_data[wi] = tc_w2_data[wi*16 +: 16];
            ext_w2_we[wi] = tc_w2_we[wi];
            ext_w3_addr[wi] = tc_w3_addr;
            ext_w3_data[wi] = tc_w3_data[wi*16 +: 16];
            ext_w3_we[wi] = tc_w3_we[wi];

            // Burst load overrides W1 (mutually exclusive with TC scatter)
            if (bu_w1_we) begin
                ext_w1_addr[wi] = bu_w1_addr;
                ext_w1_data[wi] = dmem_dout_a[wi];
                ext_w1_we[wi] = 1'b1;
            end
        end
    end

    // ================================================================
    // 10. Generate: 4× SP Core
    // ================================================================
    genvar t;
    generate
        for (t = 0; t < 4; t = t + 1) begin : SP_LANE
            sp_core #(.TID(t[1:0])) u_sp (
                .clk(clk), .rst_n(rst_n),
                .stall(sp_stall),
                .flush_id(sp_flush_id),
                .rf_r0_addr(rf_r0_addr_mux),
                .rf_r1_addr(rf_r1_addr_mux),
                .rf_r2_addr(rf_r2_addr_mux),
                .rf_r3_addr(rf_r3_addr_mux),
                .rf_r0_data(sp_rf_r0_data[t]),
                .rf_r1_data(sp_rf_r1_data[t]),
                .rf_r2_data(sp_rf_r2_data[t]),
                .rf_r3_data(sp_rf_r3_data[t]),
                .pred_rd_sel(dec_pred_rd_sel),
                .pred_rd_val(sp_pred_rd_val[t]),
                .id_opcode(dec_opcode),
                .id_dt(dec_dt),
                .id_cmp_mode(dec_cmp_mode),
                .id_rf_we(dec_rf_we),
                .id_pred_we(dec_pred_we),
                .id_rD_addr(dec_rD_addr),
                .id_pred_wr_sel(dec_pred_wr_sel),
                .id_valid(sp_id_valid),
                .id_active(active_mask[t]),
                .id_wb_src(dec_wb_src),
                .id_use_imm(dec_use_imm),
                .id_imm16(dec_imm16),
                .ex_mem_result_out(sp_ex_mem_result[t]),
                .ex_mem_store_out(sp_ex_mem_store[t]),
                .ex_mem_valid_out(sp_ex_mem_valid[t]),
                .ex_busy(sp_ex_busy[t]),
                .mem_rdata(dmem_dout_a[t]),
                .wb_ext_w1_addr(ext_w1_addr[t]),
                .wb_ext_w1_data(ext_w1_data[t]),
                .wb_ext_w1_we(ext_w1_we[t]),
                .wb_ext_w2_addr(ext_w2_addr[t]),
                .wb_ext_w2_data(ext_w2_data[t]),
                .wb_ext_w2_we(ext_w2_we[t]),
                .wb_ext_w3_addr(ext_w3_addr[t]),
                .wb_ext_w3_data(ext_w3_data[t]),
                .wb_ext_w3_we(ext_w3_we[t]),
                .mem_is_load(sp_mem_is_load[t]),
                .mem_is_store(sp_mem_is_store[t]),
                .wb_rD_addr(sp_wb_rD_addr[t]),
                .wb_rf_we(sp_wb_rf_we[t]),
                .wb_active(sp_wb_active[t]),
                .wb_valid(sp_wb_valid[t])
            );
        end
    endgenerate

    // ================================================================
    // 11. DMEM Port-A Mux (normal vs burst)
    // ================================================================
    genvar d;
    generate
        for (d = 0; d < 4; d = d + 1) begin : DMEM_MUX
            wire bu_load_phase = (bu_state == BU_LOAD_ADDR) | (bu_state == BU_LOAD_BEAT);
            wire bu_store_phase = (bu_state == BU_STORE_BEAT);

            // Burst load: prefetch next addr (base+beat+1), first addr=base+0
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