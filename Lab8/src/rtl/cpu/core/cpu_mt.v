/* file: cpu_mt.v
 Description: Quad-threaded 7-stage pipeline Arm CPU module
 (IF1, IF2, ID, EX1, EX2, MEM, WB)
 IF split: IF1 = present address to BRAM
        IF2 = register instruction + pre-decode RF addrs
 EX split: EX1 = barrel shifter + operand B mux + WB bypass
        EX2 = ALU + MAC + branch
 Author: Jeremy Cai
 Date: Feb. 23, 2026
 Revision History:
   v2.8 (Mar. 5, 2026):
     - Per-thread halt detection: B . (0xEAFFFFFE) detected at IF2,
       2-sighting stability filter, sticky halted[tid] bits.
       Halted threads have PC frozen (no increment, no branch update).
     - cpu_start_i / entry_pc_i: external start pulse loads PCs,
       clears halted/halt_seen_once, sets running=1.
       CPU stalls until running=1.
     - cpu_done = &halted | (all PCs == CPU_DONE_PC) fallback.
   v2.7: MAC + CP10 coprocessor interface.
   v2.6: Barrel shifter input mux for imm rotation.
   v2.5: WB->EX1 registered bypass; raw RF data in ID.
*/

`ifndef CPU_MT_V
`define CPU_MT_V

`include "define.v"
`include "regfile.v"
`include "cu.v"
`include "alu.v"
`include "mac.v"
`include "cond_eval.v"
`include "bdtu.v"
`include "barrel_shifter.v"

module cpu_mt (
    input wire clk,
    input wire rst_n,
    // Execution control (from pkt_proc)
    input wire cpu_start_i,
    input wire [`PC_WIDTH-1:0] entry_pc_i,
    // Instruction memory interface
    input wire [`INSTR_WIDTH-1:0] i_mem_data_i,
    output wire [`PC_WIDTH-1:0] i_mem_addr_o,
    // Data memory interface
    input wire [`DATA_WIDTH-1:0] d_mem_data_i,
    output wire [`CPU_DMEM_ADDR_WIDTH-1:0] d_mem_addr_o,
    output wire [`DATA_WIDTH-1:0] d_mem_data_o,
    output wire d_mem_wen_o,
    output wire [1:0] d_mem_size_o,
    // Coprocessor interface (active at EX2 stage)
    output wire cp_wen_o,
    output wire cp_ren_o,
    output wire [3:0] cp_reg_o,
    output wire [31:0] cp_wr_data_o,
    input wire [31:0] cp_rd_data_i,

    output wire cpu_done
);

/* ================================================================
   GLOBAL CONTROL
   ================================================================ */
wire stall_all;
wire bdtu_busy;
wire branch_taken_ex2;
wire [`PC_WIDTH-1:0] branch_target_ex2;

/* v2.8: running flop — CPU stalled until cpu_start pulse */
reg running;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) running <= 1'b0;
    else if (cpu_start_i) running <= 1'b1;
end

/* ================================================================
   PER-THREAD HALT DETECTION (v2.8)

   B . (branch to self) = 0xEAFFFFFE, position-independent.
   Detected at IF2 where BRAM output is available.
   Require 2 consecutive sightings (8 cycles apart for same thread)
   to filter squashed fetches.

   Halted threads: PC frozen, pipeline slots become NOPs.
   cpu_start clears all halt state for restart.
   ================================================================ */
localparam [31:0] HALT_ENCODING = 32'hEAFF_FFFE;

reg [3:0] halted;
reg [3:0] halt_seen_once;

reg [`PC_WIDTH-1:0] pc_thread [0:3];

/* cpu_done: primary = all halted, fallback = all PCs at CPU_DONE_PC */
wire pc_done = (pc_thread[0] == `CPU_DONE_PC) &&
               (pc_thread[1] == `CPU_DONE_PC) &&
               (pc_thread[2] == `CPU_DONE_PC) &&
               (pc_thread[3] == `CPU_DONE_PC);

/* cpu_done gated by running: must not assert before cpu_start fires.
 * Without this gate, pc_done=1 on reset (all PCs at CPU_DONE_PC)
 * causes pkt_proc to exit P_CPU_RUN immediately. */
assign cpu_done = running & ((&halted) | pc_done);

/* ================================================================
   IF1 — present address to BRAM, round-robin thread scheduling
   ================================================================ */
reg [1:0] tid_if;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) tid_if <= 2'd0;
    else if (!stall_all) tid_if <= tid_if + 2'd1;
end

reg [1:0] tid_if2, tid_id, tid_ex1, tid_ex2, tid_mem, tid_wb;
reg valid_if2, valid_id, valid_ex1, valid_ex2, valid_mem, valid_wb;

wire [`PC_WIDTH-1:0] pc_if = pc_thread[tid_if];
wire [`PC_WIDTH-1:0] pc_plus4_if = pc_if + 32'd4;
assign i_mem_addr_o = pc_if;

/* v2.8: PC update with halt freeze + cpu_start reload */
integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (k = 0; k < 4; k = k + 1)
            pc_thread[k] <= `CPU_DONE_PC;
    end else if (cpu_start_i) begin
        /* cpu_start: all threads start at the same entry_pc.
         * Shared-code model: threads diverge via TID register,
         * not via staggered PCs. */
        for (k = 0; k < 4; k = k + 1)
            pc_thread[k] <= entry_pc_i;
    end else if (!stall_all) begin
        /* Normal PC update: skip halted threads */
        if (!halted[tid_if])
            pc_thread[tid_if] <= pc_plus4_if;
        if (branch_taken_ex2 && valid_ex2 && !halted[tid_ex2])
            pc_thread[tid_ex2] <= branch_target_ex2;
    end
end

/* v2.8b: Halt detection at IF2 (BRAM output valid here).
 * CRITICAL: Both set and clear gated on valid_if2.
 * In the 4T barrel, IF1 and EX2 are the same thread. When B .
 * hits EX2, the squashed IF1 fetch (at PC+4) produces a non-matching
 * BRAM output one cycle later. Without gating, that squashed fetch
 * clears halt_seen_once every redirect, preventing halted from
 * ever being set. */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        halted <= 4'b0;
        halt_seen_once <= 4'b0;
    end else if (cpu_start_i) begin
        halted <= 4'b0;
        halt_seen_once <= 4'b0;
    end else if (!stall_all && valid_if2) begin
        if (i_mem_data_i == HALT_ENCODING && !halted[tid_if2]) begin
            if (halt_seen_once[tid_if2])
                halted[tid_if2] <= 1'b1;
            else
                halt_seen_once[tid_if2] <= 1'b1;
        end else if (!halted[tid_if2]) begin
            halt_seen_once[tid_if2] <= 1'b0;
        end
    end
end

/* ================================================================
   IF1/IF2 PIPELINE REGISTER
   ================================================================ */
wire squash_if1 = branch_taken_ex2 && valid_ex2;
/* v2.8: invalidate halted thread fetches */
wire suppress_if1 = halted[tid_if];
reg [`PC_WIDTH-1:0] pc_plus4_if2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tid_if2 <= 2'd0;
        pc_plus4_if2 <= `PC_WIDTH'd0;
        valid_if2 <= 1'b0;
    end else if (!stall_all) begin
        tid_if2 <= tid_if;
        pc_plus4_if2 <= pc_plus4_if;
        valid_if2 <= !squash_if1 && !suppress_if1;
    end
end

/* ================================================================
   IF2 — register instruction + pre-decode RF addresses
   ================================================================ */
wire [3:0] rn_addr_pre = i_mem_data_i[19:16];
wire [3:0] rd_addr_pre = i_mem_data_i[15:12];
wire [3:0] rs_addr_pre = i_mem_data_i[11:8];
wire [3:0] rm_addr_pre = i_mem_data_i[3:0];

/* ================================================================
   IF2/ID PIPELINE REGISTER
   ================================================================ */
reg [`INSTR_WIDTH-1:0] instr_id_r;
reg [3:0] rn_addr_id, rd_addr_id, rs_addr_id, rm_addr_id;
reg [`PC_WIDTH-1:0] pc_plus4_id;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        instr_id_r <= {`INSTR_WIDTH{1'b0}};
        rn_addr_id <= 4'd0; rd_addr_id <= 4'd0;
        rs_addr_id <= 4'd0; rm_addr_id <= 4'd0;
        pc_plus4_id <= `PC_WIDTH'd0;
        tid_id <= 2'd0; valid_id <= 1'b0;
    end else if (!stall_all) begin
        instr_id_r <= i_mem_data_i;
        rn_addr_id <= rn_addr_pre; rd_addr_id <= rd_addr_pre;
        rs_addr_id <= rs_addr_pre; rm_addr_id <= rm_addr_pre;
        pc_plus4_id <= pc_plus4_if2;
        tid_id <= tid_if2; valid_id <= valid_if2;
    end
end

wire [`INSTR_WIDTH-1:0] instr_id = instr_id_r;

/* ================================================================
   ID — INSTRUCTION DECODE
   ================================================================ */
reg [3:0] cpsr_flags [0:3];

wire [3:0] cond_flags_id = cpsr_flags[tid_id];
wire cond_met_raw;

cond_eval u_cond_eval (
    .cond_code(instr_id[31:28]),
    .flags(cond_flags_id),
    .cond_met(cond_met_raw)
);

wire cond_met_id = cond_met_raw && valid_id;

wire t_dp_reg, t_dp_imm, t_swp, t_bx;
wire t_hdt_rego, t_hdt_immo, t_sdt_rego, t_sdt_immo;
wire t_bdt, t_br, t_mrs, t_msr_reg, t_msr_imm, t_swi, t_undef;
wire t_mcr, t_mrc;
wire [3:0] cu_rn_addr, cu_rd_addr, cu_rs_addr, cu_rm_addr;
wire [3:0] wr_addr1_id, wr_addr2_id;
wire wr_en1_id, wr_en2_id;
wire [3:0] alu_op_id;
wire alu_src_b_id, cpsr_wen_id;
wire [1:0] shift_type_id;
wire [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount_id;
wire shift_src_id;
wire [`DATA_WIDTH-1:0] imm32_id;
wire mem_read_id, mem_write_id;
wire [1:0] mem_size_id;
wire mem_signed_id;
wire addr_pre_idx_id, addr_up_id, addr_wb_id;
wire [2:0] wb_sel_id;
wire branch_en_id, branch_link_id, branch_exchange_id;
wire mul_en_id, mul_long_id, mul_signed_id, mul_accumulate_id;
wire psr_rd_id, psr_wr_id, psr_field_sel_id;
wire [3:0] psr_mask_id;
wire [15:0] bdt_list_id;
wire bdt_load_id, bdt_s_id, bdt_wb_id;
wire swap_byte_id, swi_en_id;
wire cp_wen_id, cp_ren_id;
wire use_rn_id, use_rd_id, use_rs_id, use_rm_id;
wire is_multi_cycle_id;

cu u_cu (
    .instr(instr_id), .cond_met(cond_met_id),
    .t_dp_reg(t_dp_reg), .t_dp_imm(t_dp_imm),
    .t_mul(), .t_mull(),
    .t_swp(t_swp), .t_bx(t_bx),
    .t_hdt_rego(t_hdt_rego), .t_hdt_immo(t_hdt_immo),
    .t_sdt_rego(t_sdt_rego), .t_sdt_immo(t_sdt_immo),
    .t_bdt(t_bdt), .t_br(t_br), .t_mrs(t_mrs),
    .t_msr_reg(t_msr_reg), .t_msr_imm(t_msr_imm),
    .t_swi(t_swi), .t_undef(t_undef),
    .t_mcr(t_mcr), .t_mrc(t_mrc),
    .rn_addr(cu_rn_addr), .rd_addr(cu_rd_addr),
    .rs_addr(cu_rs_addr), .rm_addr(cu_rm_addr),
    .wr_addr1(wr_addr1_id), .wr_en1(wr_en1_id),
    .wr_addr2(wr_addr2_id), .wr_en2(wr_en2_id),
    .alu_op(alu_op_id), .alu_src_b(alu_src_b_id), .cpsr_wen(cpsr_wen_id),
    .shift_type(shift_type_id), .shift_amount(shift_amount_id), .shift_src(shift_src_id),
    .imm32(imm32_id),
    .mem_read(mem_read_id), .mem_write(mem_write_id),
    .mem_size(mem_size_id), .mem_signed(mem_signed_id),
    .addr_pre_idx(addr_pre_idx_id), .addr_up(addr_up_id), .addr_wb(addr_wb_id),
    .wb_sel(wb_sel_id),
    .branch_en(branch_en_id), .branch_link(branch_link_id), .branch_exchange(branch_exchange_id),
    .mul_en(mul_en_id), .mul_long(mul_long_id),
    .mul_signed(mul_signed_id), .mul_accumulate(mul_accumulate_id),
    .psr_rd(psr_rd_id), .psr_wr(psr_wr_id),
    .psr_field_sel(psr_field_sel_id), .psr_mask(psr_mask_id),
    .bdt_list(bdt_list_id), .bdt_load(bdt_load_id),
    .bdt_s(bdt_s_id), .bdt_wb(bdt_wb_id),
    .swap_byte(swap_byte_id), .swi_en(swi_en_id),
    .cp_wen(cp_wen_id), .cp_ren(cp_ren_id),
    .use_rn(use_rn_id), .use_rd(use_rd_id),
    .use_rs(use_rs_id), .use_rm(use_rm_id),
    .is_multi_cycle(is_multi_cycle_id)
);

/* BDTU signals (forward-declared) */
wire [3:0] bdtu_rf_rd_addr;
wire [3:0] bdtu_wr_addr1, bdtu_wr_addr2;
wire [`DATA_WIDTH-1:0] bdtu_wr_data1, bdtu_wr_data2;
wire bdtu_wr_en1, bdtu_wr_en2;
wire bdtu_has_write = bdtu_wr_en1 | bdtu_wr_en2;

/* WB write signals (forward-declared) */
wire [3:0] wb_wr_addr1, wb_wr_addr2;
wire [`DATA_WIDTH-1:0] wb_wr_data1, wb_wr_data2;
wire wb_wr_en1, wb_wr_en2;

/* ================================================================
   REGISTER FILES (4 instances, one per thread)
   ================================================================ */
genvar g;
generate
    for (g = 0; g < 4; g = g + 1) begin : THREAD_RF
        wire is_wb_target = (tid_wb == g) && valid_wb;
        wire is_bdtu_target = (tid_mem == g) && bdtu_has_write;
        wire wena = (is_wb_target && (wb_wr_en1 || wb_wr_en2)) || is_bdtu_target;

        wire [3:0] wa1 = is_bdtu_target
            ? (bdtu_wr_en1 ? bdtu_wr_addr1 : bdtu_wr_addr2)
            : (wb_wr_en1 ? wb_wr_addr1 : wb_wr_addr2);
        wire [`DATA_WIDTH-1:0] wd1 = is_bdtu_target
            ? (bdtu_wr_en1 ? bdtu_wr_data1 : bdtu_wr_data2)
            : (wb_wr_en1 ? wb_wr_data1 : wb_wr_data2);
        wire [3:0] wa2 = is_bdtu_target
            ? ((bdtu_wr_en1 && bdtu_wr_en2) ? bdtu_wr_addr2 : wa1)
            : ((wb_wr_en1 && wb_wr_en2) ? wb_wr_addr2 : wa1);
        wire [`DATA_WIDTH-1:0] wd2 = is_bdtu_target
            ? ((bdtu_wr_en1 && bdtu_wr_en2) ? bdtu_wr_data2 : wd1)
            : ((wb_wr_en1 && wb_wr_en2) ? wb_wr_data2 : wd1);

        reg wena_r;
        reg [3:0] wa1_r, wa2_r;
        reg [`DATA_WIDTH-1:0] wd1_r, wd2_r;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wena_r <= 1'b0; wa1_r <= 4'd0; wa2_r <= 4'd0;
                wd1_r <= {`DATA_WIDTH{1'b0}}; wd2_r <= {`DATA_WIDTH{1'b0}};
            end else begin
                wena_r <= wena; wa1_r <= wa1; wa2_r <= wa2;
                wd1_r <= wd1; wd2_r <= wd2;
            end
        end

        wire [3:0] local_r3addr = (bdtu_busy && (tid_mem == g))
                                  ? bdtu_rf_rd_addr : rs_addr_id;
        wire [`DATA_WIDTH-1:0] rn_out, rm_out, r3_out, r4_out;

        regfile u_rf (
            .clk(clk),
            .r1addr(rn_addr_id), .r2addr(rm_addr_id),
            .r3addr(local_r3addr), .r4addr(rd_addr_id),
            .wena(wena_r),
            .wr_addr1(wa1_r), .wr_data1(wd1_r),
            .wr_addr2(wa2_r), .wr_data2(wd2_r),
            .r1data(rn_out), .r2data(rm_out),
            .r3data(r3_out), .r4data(r4_out)
        );
    end
endgenerate

/* RF read MUX */
reg [`DATA_WIDTH-1:0] rn_data_id, rm_data_id, r3_data_id, r4_data_id;

always @(*) begin
    case (tid_id)
        2'd0: begin rn_data_id = THREAD_RF[0].rn_out; rm_data_id = THREAD_RF[0].rm_out;
                    r3_data_id = THREAD_RF[0].r3_out; r4_data_id = THREAD_RF[0].r4_out; end
        2'd1: begin rn_data_id = THREAD_RF[1].rn_out; rm_data_id = THREAD_RF[1].rm_out;
                    r3_data_id = THREAD_RF[1].r3_out; r4_data_id = THREAD_RF[1].r4_out; end
        2'd2: begin rn_data_id = THREAD_RF[2].rn_out; rm_data_id = THREAD_RF[2].rm_out;
                    r3_data_id = THREAD_RF[2].r3_out; r4_data_id = THREAD_RF[2].r4_out; end
        default: begin rn_data_id = THREAD_RF[3].rn_out; rm_data_id = THREAD_RF[3].rm_out;
                       r3_data_id = THREAD_RF[3].r3_out; r4_data_id = THREAD_RF[3].r4_out; end
    endcase
end

/* BDTU read-data MUX */
reg [`DATA_WIDTH-1:0] bdtu_rf_rd_data;
always @(*) begin
    case (tid_mem)
        2'd0: bdtu_rf_rd_data = THREAD_RF[0].r3_out;
        2'd1: bdtu_rf_rd_data = THREAD_RF[1].r3_out;
        2'd2: bdtu_rf_rd_data = THREAD_RF[2].r3_out;
        default: bdtu_rf_rd_data = THREAD_RF[3].r3_out;
    endcase
end

/* ================================================================
   ID/EX1 PIPELINE REGISTER
   ================================================================ */
reg [3:0] alu_op_ex1;
reg alu_src_b_ex1, cpsr_wen_ex1;
reg [1:0] shift_type_ex1;
reg [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount_ex1;
reg shift_src_ex1;
reg [`DATA_WIDTH-1:0] imm32_ex1;
reg mem_read_ex1, mem_write_ex1;
reg [1:0] mem_size_ex1;
reg mem_signed_ex1, addr_pre_idx_ex1, addr_up_ex1, addr_wb_ex1;
reg [2:0] wb_sel_ex1;
reg [3:0] wr_addr1_ex1, wr_addr2_ex1;
reg wr_en1_ex1, wr_en2_ex1;
reg branch_en_ex1, branch_link_ex1, branch_exchange_ex1;
reg [`DATA_WIDTH-1:0] rn_data_ex1, rm_data_ex1, rs_data_ex1, rd_data_ex1;
reg [`PC_WIDTH-1:0] pc_plus4_ex1;
reg is_multi_cycle_ex1, t_bdt_ex1, t_swp_ex1;
reg [15:0] bdt_list_ex1;
reg bdt_load_ex1, bdt_s_ex1, bdt_wb_ex1;
reg addr_pre_idx_bdt_ex1, addr_up_bdt_ex1, swap_byte_ex1;
reg [3:0] base_reg_ex1, rm_addr_ex1;
reg psr_wr_ex1, psr_field_sel_ex1;
reg [3:0] psr_mask_ex1;
reg [3:0] rn_addr_ex1, rs_addr_ex1, rd_addr_ex1;
reg [3:0] fwd_addr1_ex1, fwd_addr2_ex1;
reg [`DATA_WIDTH-1:0] fwd_data1_ex1, fwd_data2_ex1;
reg fwd_en1_ex1, fwd_en2_ex1;
reg mul_en_ex1, mul_long_ex1, mul_signed_ex1, mul_accumulate_ex1;
reg cp_wen_ex1, cp_ren_ex1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_op_ex1 <= 4'd0; alu_src_b_ex1 <= 1'b0; cpsr_wen_ex1 <= 1'b0;
        shift_type_ex1 <= 2'd0; shift_amount_ex1 <= 5'd0; shift_src_ex1 <= 1'b0;
        imm32_ex1 <= 32'd0;
        mem_read_ex1 <= 1'b0; mem_write_ex1 <= 1'b0; mem_size_ex1 <= 2'd0;
        mem_signed_ex1 <= 1'b0; addr_pre_idx_ex1 <= 1'b0; addr_up_ex1 <= 1'b0; addr_wb_ex1 <= 1'b0;
        wb_sel_ex1 <= 3'd0;
        wr_addr1_ex1 <= 4'd0; wr_addr2_ex1 <= 4'd0; wr_en1_ex1 <= 1'b0; wr_en2_ex1 <= 1'b0;
        branch_en_ex1 <= 1'b0; branch_link_ex1 <= 1'b0; branch_exchange_ex1 <= 1'b0;
        rn_data_ex1 <= 32'd0; rm_data_ex1 <= 32'd0; rs_data_ex1 <= 32'd0; rd_data_ex1 <= 32'd0;
        pc_plus4_ex1 <= 32'd0;
        is_multi_cycle_ex1 <= 1'b0; t_bdt_ex1 <= 1'b0; t_swp_ex1 <= 1'b0;
        bdt_list_ex1 <= 16'd0; bdt_load_ex1 <= 1'b0; bdt_s_ex1 <= 1'b0; bdt_wb_ex1 <= 1'b0;
        addr_pre_idx_bdt_ex1 <= 1'b0; addr_up_bdt_ex1 <= 1'b0; swap_byte_ex1 <= 1'b0;
        base_reg_ex1 <= 4'd0; rm_addr_ex1 <= 4'd0;
        psr_wr_ex1 <= 1'b0; psr_mask_ex1 <= 4'd0; psr_field_sel_ex1 <= 1'b0;
        rn_addr_ex1 <= 4'd0; rs_addr_ex1 <= 4'd0; rd_addr_ex1 <= 4'd0;
        fwd_addr1_ex1 <= 4'd0; fwd_addr2_ex1 <= 4'd0;
        fwd_data1_ex1 <= 32'd0; fwd_data2_ex1 <= 32'd0;
        fwd_en1_ex1 <= 1'b0; fwd_en2_ex1 <= 1'b0;
        mul_en_ex1 <= 1'b0; mul_long_ex1 <= 1'b0; mul_signed_ex1 <= 1'b0; mul_accumulate_ex1 <= 1'b0;
        cp_wen_ex1 <= 1'b0; cp_ren_ex1 <= 1'b0;
        tid_ex1 <= 2'd0; valid_ex1 <= 1'b0;
    end else if (!stall_all) begin
        alu_op_ex1 <= alu_op_id; alu_src_b_ex1 <= alu_src_b_id; cpsr_wen_ex1 <= cpsr_wen_id;
        shift_type_ex1 <= shift_type_id; shift_amount_ex1 <= shift_amount_id; shift_src_ex1 <= shift_src_id;
        imm32_ex1 <= imm32_id;
        mem_read_ex1 <= mem_read_id; mem_write_ex1 <= mem_write_id; mem_size_ex1 <= mem_size_id;
        mem_signed_ex1 <= mem_signed_id; addr_pre_idx_ex1 <= addr_pre_idx_id;
        addr_up_ex1 <= addr_up_id; addr_wb_ex1 <= addr_wb_id;
        wb_sel_ex1 <= wb_sel_id;
        wr_addr1_ex1 <= wr_addr1_id; wr_addr2_ex1 <= wr_addr2_id;
        wr_en1_ex1 <= wr_en1_id; wr_en2_ex1 <= wr_en2_id;
        branch_en_ex1 <= branch_en_id; branch_link_ex1 <= branch_link_id; branch_exchange_ex1 <= branch_exchange_id;
        rn_data_ex1 <= rn_data_id; rm_data_ex1 <= rm_data_id;
        rs_data_ex1 <= r3_data_id; rd_data_ex1 <= r4_data_id;
        pc_plus4_ex1 <= pc_plus4_id;
        is_multi_cycle_ex1 <= is_multi_cycle_id; t_bdt_ex1 <= t_bdt; t_swp_ex1 <= t_swp;
        bdt_list_ex1 <= bdt_list_id; bdt_load_ex1 <= bdt_load_id;
        bdt_s_ex1 <= bdt_s_id; bdt_wb_ex1 <= bdt_wb_id;
        addr_pre_idx_bdt_ex1 <= addr_pre_idx_id; addr_up_bdt_ex1 <= addr_up_id; swap_byte_ex1 <= swap_byte_id;
        base_reg_ex1 <= rn_addr_id; rm_addr_ex1 <= rm_addr_id;
        psr_wr_ex1 <= psr_wr_id; psr_mask_ex1 <= psr_mask_id; psr_field_sel_ex1 <= psr_field_sel_id;
        rn_addr_ex1 <= rn_addr_id; rs_addr_ex1 <= rs_addr_id; rd_addr_ex1 <= rd_addr_id;
        fwd_addr1_ex1 <= wb_wr_addr1; fwd_addr2_ex1 <= wb_wr_addr2;
        fwd_data1_ex1 <= wb_wr_data1; fwd_data2_ex1 <= wb_wr_data2;
        fwd_en1_ex1 <= wb_wr_en1; fwd_en2_ex1 <= wb_wr_en2;
        mul_en_ex1 <= mul_en_id; mul_long_ex1 <= mul_long_id;
        mul_signed_ex1 <= mul_signed_id; mul_accumulate_ex1 <= mul_accumulate_id;
        cp_wen_ex1 <= cp_wen_id; cp_ren_ex1 <= cp_ren_id;
        tid_ex1 <= tid_id; valid_ex1 <= valid_id;
    end
end

/* ================================================================
   EX1 — SHIFT / OPERAND-PREPARE (barrel shifter + opB mux)
   ================================================================ */
wire fwd_rn_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rn_addr_ex1);
wire fwd_rn_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rn_addr_ex1);
wire [`DATA_WIDTH-1:0] rn_bypassed_ex1 = fwd_rn_p1 ? fwd_data1_ex1 :
                                          fwd_rn_p2 ? fwd_data2_ex1 : rn_data_ex1;

wire fwd_rm_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rm_addr_ex1);
wire fwd_rm_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rm_addr_ex1);
wire [`DATA_WIDTH-1:0] rm_bypassed_ex1 = fwd_rm_p1 ? fwd_data1_ex1 :
                                          fwd_rm_p2 ? fwd_data2_ex1 : rm_data_ex1;

wire fwd_rs_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rs_addr_ex1);
wire fwd_rs_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rs_addr_ex1);
wire [`DATA_WIDTH-1:0] rs_bypassed_ex1 = fwd_rs_p1 ? fwd_data1_ex1 :
                                          fwd_rs_p2 ? fwd_data2_ex1 : rs_data_ex1;

wire fwd_rd_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rd_addr_ex1);
wire fwd_rd_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rd_addr_ex1);
wire [`DATA_WIDTH-1:0] rd_bypassed_ex1 = fwd_rd_p1 ? fwd_data1_ex1 :
                                          fwd_rd_p2 ? fwd_data2_ex1 : rd_data_ex1;

wire [`DATA_WIDTH-1:0] pc_plus8_ex1 = pc_plus4_ex1 + 32'd4;
wire [`DATA_WIDTH-1:0] rn_val_ex1 = (rn_addr_ex1 == 4'd15) ? pc_plus8_ex1 : rn_bypassed_ex1;
wire [`DATA_WIDTH-1:0] rm_val_ex1 = (rm_addr_ex1 == 4'd15) ? pc_plus8_ex1 : rm_bypassed_ex1;
wire [`DATA_WIDTH-1:0] rs_val_ex1 = rs_bypassed_ex1;
wire [`DATA_WIDTH-1:0] rd_val_ex1 = rd_bypassed_ex1;

wire [3:0] cpsr_flags_ex1 = cpsr_flags[tid_ex1];
wire [`SHIFT_AMOUNT_WIDTH-1:0] actual_shamt_ex1 =
    shift_src_ex1 ? rs_val_ex1[`SHIFT_AMOUNT_WIDTH-1:0] : shift_amount_ex1;

wire [`DATA_WIDTH-1:0] bs_din_ex1 = alu_src_b_ex1 ? imm32_ex1 : rm_val_ex1;
wire [`DATA_WIDTH-1:0] bs_dout_ex1;
wire shifter_cout_ex1;

barrel_shifter u_barrel_shifter (
    .din(bs_din_ex1), .shamt(actual_shamt_ex1),
    .shift_type(shift_type_ex1), .is_imm_shift(~shift_src_ex1),
    .cin(cpsr_flags_ex1[`FLAG_C]),
    .dout(bs_dout_ex1), .cout(shifter_cout_ex1)
);

wire [`DATA_WIDTH-1:0] alu_src_b_val_ex1 = bs_dout_ex1;
wire [`PC_WIDTH-1:0] branch_target_br_ex1 = pc_plus4_ex1 + 32'd4 + imm32_ex1;

/* ================================================================
   EX1/EX2 PIPELINE REGISTER
   ================================================================ */
reg [`DATA_WIDTH-1:0] alu_src_b_val_ex2;
reg shifter_cout_ex2;
reg [`PC_WIDTH-1:0] branch_target_br_ex2;
reg carry_in_ex2;
reg [3:0] alu_op_ex2;
reg cpsr_wen_ex2, mem_read_ex2, mem_write_ex2;
reg [1:0] mem_size_ex2;
reg mem_signed_ex2, addr_pre_idx_ex2, addr_up_ex2, addr_wb_ex2;
reg [2:0] wb_sel_ex2;
reg [3:0] wr_addr1_ex2, wr_addr2_ex2;
reg wr_en1_ex2, wr_en2_ex2;
reg branch_en_ex2_r, branch_link_ex2, branch_exchange_ex2;
reg [`DATA_WIDTH-1:0] rn_data_ex2, rm_data_ex2, rs_data_ex2, rd_data_ex2;
reg [`PC_WIDTH-1:0] pc_plus4_ex2;
reg is_multi_cycle_ex2, t_bdt_ex2, t_swp_ex2;
reg [15:0] bdt_list_ex2;
reg bdt_load_ex2, bdt_s_ex2, bdt_wb_ex2;
reg addr_pre_idx_bdt_ex2, addr_up_bdt_ex2, swap_byte_ex2;
reg [3:0] base_reg_ex2, rm_addr_ex2;
reg psr_wr_ex2, psr_field_sel_ex2;
reg [3:0] psr_mask_ex2;
reg mul_en_ex2, mul_long_ex2, mul_signed_ex2, mul_accumulate_ex2;
reg cp_wen_ex2, cp_ren_ex2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_src_b_val_ex2 <= 32'd0; shifter_cout_ex2 <= 1'b0;
        branch_target_br_ex2 <= 32'd0; carry_in_ex2 <= 1'b0;
        alu_op_ex2 <= 4'd0; cpsr_wen_ex2 <= 1'b0;
        mem_read_ex2 <= 1'b0; mem_write_ex2 <= 1'b0; mem_size_ex2 <= 2'd0;
        mem_signed_ex2 <= 1'b0; addr_pre_idx_ex2 <= 1'b0; addr_up_ex2 <= 1'b0; addr_wb_ex2 <= 1'b0;
        wb_sel_ex2 <= 3'd0;
        wr_addr1_ex2 <= 4'd0; wr_addr2_ex2 <= 4'd0; wr_en1_ex2 <= 1'b0; wr_en2_ex2 <= 1'b0;
        branch_en_ex2_r <= 1'b0; branch_link_ex2 <= 1'b0; branch_exchange_ex2 <= 1'b0;
        rn_data_ex2 <= 32'd0; rm_data_ex2 <= 32'd0; rs_data_ex2 <= 32'd0; rd_data_ex2 <= 32'd0;
        pc_plus4_ex2 <= 32'd0;
        is_multi_cycle_ex2 <= 1'b0; t_bdt_ex2 <= 1'b0; t_swp_ex2 <= 1'b0;
        bdt_list_ex2 <= 16'd0; bdt_load_ex2 <= 1'b0; bdt_s_ex2 <= 1'b0; bdt_wb_ex2 <= 1'b0;
        addr_pre_idx_bdt_ex2 <= 1'b0; addr_up_bdt_ex2 <= 1'b0; swap_byte_ex2 <= 1'b0;
        base_reg_ex2 <= 4'd0; rm_addr_ex2 <= 4'd0;
        psr_wr_ex2 <= 1'b0; psr_mask_ex2 <= 4'd0; psr_field_sel_ex2 <= 1'b0;
        mul_en_ex2 <= 1'b0; mul_long_ex2 <= 1'b0; mul_signed_ex2 <= 1'b0; mul_accumulate_ex2 <= 1'b0;
        cp_wen_ex2 <= 1'b0; cp_ren_ex2 <= 1'b0;
        tid_ex2 <= 2'd0; valid_ex2 <= 1'b0;
    end else if (!stall_all) begin
        alu_src_b_val_ex2 <= alu_src_b_val_ex1; shifter_cout_ex2 <= shifter_cout_ex1;
        branch_target_br_ex2 <= branch_target_br_ex1; carry_in_ex2 <= cpsr_flags_ex1[`FLAG_C];
        alu_op_ex2 <= alu_op_ex1; cpsr_wen_ex2 <= cpsr_wen_ex1;
        mem_read_ex2 <= mem_read_ex1; mem_write_ex2 <= mem_write_ex1; mem_size_ex2 <= mem_size_ex1;
        mem_signed_ex2 <= mem_signed_ex1; addr_pre_idx_ex2 <= addr_pre_idx_ex1;
        addr_up_ex2 <= addr_up_ex1; addr_wb_ex2 <= addr_wb_ex1;
        wb_sel_ex2 <= wb_sel_ex1;
        wr_addr1_ex2 <= wr_addr1_ex1; wr_addr2_ex2 <= wr_addr2_ex1;
        wr_en1_ex2 <= wr_en1_ex1; wr_en2_ex2 <= wr_en2_ex1;
        branch_en_ex2_r <= branch_en_ex1; branch_link_ex2 <= branch_link_ex1; branch_exchange_ex2 <= branch_exchange_ex1;
        rn_data_ex2 <= rn_val_ex1; rm_data_ex2 <= rm_val_ex1;
        rs_data_ex2 <= rs_val_ex1; rd_data_ex2 <= rd_val_ex1;
        pc_plus4_ex2 <= pc_plus4_ex1;
        is_multi_cycle_ex2 <= is_multi_cycle_ex1; t_bdt_ex2 <= t_bdt_ex1; t_swp_ex2 <= t_swp_ex1;
        bdt_list_ex2 <= bdt_list_ex1; bdt_load_ex2 <= bdt_load_ex1;
        bdt_s_ex2 <= bdt_s_ex1; bdt_wb_ex2 <= bdt_wb_ex1;
        addr_pre_idx_bdt_ex2 <= addr_pre_idx_bdt_ex1; addr_up_bdt_ex2 <= addr_up_bdt_ex1; swap_byte_ex2 <= swap_byte_ex1;
        base_reg_ex2 <= base_reg_ex1; rm_addr_ex2 <= rm_addr_ex1;
        psr_wr_ex2 <= psr_wr_ex1; psr_mask_ex2 <= psr_mask_ex1; psr_field_sel_ex2 <= psr_field_sel_ex1;
        mul_en_ex2 <= mul_en_ex1; mul_long_ex2 <= mul_long_ex1;
        mul_signed_ex2 <= mul_signed_ex1; mul_accumulate_ex2 <= mul_accumulate_ex1;
        cp_wen_ex2 <= cp_wen_ex1; cp_ren_ex2 <= cp_ren_ex1;
        tid_ex2 <= tid_ex1; valid_ex2 <= valid_ex1;
    end
end

/* ================================================================
   EX2 — ALU / MAC / CP10 / BRANCH-RESOLVE
   ================================================================ */
wire [`DATA_WIDTH-1:0] rn_val_ex2 = rn_data_ex2;
wire [`DATA_WIDTH-1:0] rm_val_ex2 = rm_data_ex2;
wire [`DATA_WIDTH-1:0] rd_store_val_ex2 = rd_data_ex2;

wire [`DATA_WIDTH-1:0] alu_result_ex2;
wire [3:0] alu_flags_ex2;

alu u_alu (
    .operand_a(rn_val_ex2), .operand_b(alu_src_b_val_ex2),
    .alu_op(alu_op_ex2), .cin(carry_in_ex2),
    .shift_carry_out(shifter_cout_ex2),
    .result(alu_result_ex2), .alu_flags(alu_flags_ex2)
);

/* MAC */
wire [`DATA_WIDTH-1:0] mac_rn_acc_ex2 = mul_long_ex2 ? rn_data_ex2 : rd_data_ex2;
wire [`DATA_WIDTH-1:0] mac_result_lo_ex2, mac_result_hi_ex2;
wire [3:0] mac_flags_ex2;

mac u_mac (
    .rm(rm_data_ex2), .rs(rs_data_ex2),
    .rn_acc(mac_rn_acc_ex2), .rdlo_acc(rd_data_ex2),
    .mul_en(mul_en_ex2), .mul_long(mul_long_ex2),
    .mul_signed(mul_signed_ex2), .mul_accumulate(mul_accumulate_ex2),
    .result_lo(mac_result_lo_ex2), .result_hi(mac_result_hi_ex2),
    .mac_flags(mac_flags_ex2)
);

/* CP10 coprocessor interface */
assign cp_wen_o = cp_wen_ex2;
assign cp_ren_o = cp_ren_ex2;
assign cp_reg_o = base_reg_ex2;
assign cp_wr_data_o = rd_data_ex2;

/* Branch */
wire [`PC_WIDTH-1:0] branch_target_bx_ex2 = rm_val_ex2;
assign branch_taken_ex2 = branch_en_ex2_r;
assign branch_target_ex2 = branch_exchange_ex2 ? branch_target_bx_ex2 : branch_target_br_ex2;

wire psr_wr_flags_ex2 = psr_wr_ex2 && psr_mask_ex2[3] && !psr_field_sel_ex2;
wire [`CPU_DMEM_ADDR_WIDTH-1:0] mem_addr_ex2 = addr_pre_idx_ex2 ? alu_result_ex2 : rn_val_ex2;
wire [`DATA_WIDTH-1:0] store_data_ex2 = rd_store_val_ex2;

/* Flag source mux (ALU vs MAC) */
wire [3:0] flags_ex2 = mul_en_ex2 ? mac_flags_ex2 : alu_flags_ex2;

/* ================================================================
   EX3 — DEFERRED CPSR FLAG UPDATE (sidecar)
   ================================================================ */
reg alu_flag_n_ex3, alu_flag_c_ex3, alu_flag_v_ex3;
reg [3:0] alu_result_top4_ex3;
reg cpsr_wen_ex3, psr_wr_flags_ex3;
reg mul_en_ex3, mul_long_ex3;
reg [1:0] tid_ex3;
reg valid_ex3;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_flag_n_ex3 <= 1'b0; alu_flag_c_ex3 <= 1'b0; alu_flag_v_ex3 <= 1'b0;
        alu_result_top4_ex3 <= 4'b0;
        cpsr_wen_ex3 <= 1'b0; psr_wr_flags_ex3 <= 1'b0;
        mul_en_ex3 <= 1'b0; mul_long_ex3 <= 1'b0;
        tid_ex3 <= 2'd0; valid_ex3 <= 1'b0;
    end else if (!stall_all) begin
        alu_flag_n_ex3 <= flags_ex2[3]; alu_flag_c_ex3 <= flags_ex2[1]; alu_flag_v_ex3 <= flags_ex2[0];
        alu_result_top4_ex3 <= alu_result_ex2[31:28];
        cpsr_wen_ex3 <= cpsr_wen_ex2; psr_wr_flags_ex3 <= psr_wr_flags_ex2;
        mul_en_ex3 <= mul_en_ex2; mul_long_ex3 <= mul_long_ex2;
        tid_ex3 <= tid_ex2; valid_ex3 <= valid_ex2;
    end
end

/* Deferred Z from registered results in MEM stage */
reg [`DATA_WIDTH-1:0] alu_result_mem;
reg [`DATA_WIDTH-1:0] mac_result_lo_mem, mac_result_hi_mem;

wire flag_z_deferred =
    (mul_en_ex3 & mul_long_ex3) ? (mac_result_lo_mem == {`DATA_WIDTH{1'b0}} &&
                                   mac_result_hi_mem == {`DATA_WIDTH{1'b0}}) :
    mul_en_ex3                  ? (mac_result_lo_mem == {`DATA_WIDTH{1'b0}}) :
                                  (alu_result_mem == {`DATA_WIDTH{1'b0}});

integer f;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (f = 0; f < 4; f = f + 1)
            cpsr_flags[f] <= 4'b0;
    end else if (!stall_all && valid_ex3) begin
        if (psr_wr_flags_ex3)
            cpsr_flags[tid_ex3] <= alu_result_top4_ex3;
        else if (cpsr_wen_ex3)
            cpsr_flags[tid_ex3] <= {alu_flag_n_ex3, flag_z_deferred,
                                    alu_flag_c_ex3, alu_flag_v_ex3};
    end
end

/* ================================================================
   EX2/MEM PIPELINE REGISTER
   ================================================================ */
reg [`CPU_DMEM_ADDR_WIDTH-1:0] mem_addr_mem;
reg [`DATA_WIDTH-1:0] store_data_mem;
reg mem_read_mem, mem_write_mem;
reg [1:0] mem_size_mem;
reg mem_signed_mem;
reg [2:0] wb_sel_mem;
reg [3:0] wr_addr1_mem, wr_addr2_mem;
reg wr_en1_mem, wr_en2_mem;
reg [`PC_WIDTH-1:0] pc_plus4_mem;
reg is_multi_cycle_mem, t_bdt_mem, t_swp_mem;
reg [15:0] bdt_list_mem;
reg bdt_load_mem, bdt_s_mem, bdt_wb_mem;
reg addr_pre_idx_bdt_mem, addr_up_bdt_mem, swap_byte_mem;
reg [3:0] base_reg_mem;
reg [`DATA_WIDTH-1:0] base_value_mem;
reg [3:0] swp_rd_mem, swp_rm_mem;
reg [`DATA_WIDTH-1:0] cp_rd_data_mem;
reg mul_long_mem;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_mem <= {`DATA_WIDTH{1'b0}};
        mac_result_lo_mem <= {`DATA_WIDTH{1'b0}}; mac_result_hi_mem <= {`DATA_WIDTH{1'b0}};
        mem_addr_mem <= {`CPU_DMEM_ADDR_WIDTH{1'b0}}; store_data_mem <= {`DATA_WIDTH{1'b0}};
        mem_read_mem <= 1'b0; mem_write_mem <= 1'b0; mem_size_mem <= 2'd0; mem_signed_mem <= 1'b0;
        wb_sel_mem <= 3'd0;
        wr_addr1_mem <= 4'd0; wr_addr2_mem <= 4'd0; wr_en1_mem <= 1'b0; wr_en2_mem <= 1'b0;
        pc_plus4_mem <= {`PC_WIDTH{1'b0}};
        is_multi_cycle_mem <= 1'b0; t_bdt_mem <= 1'b0; t_swp_mem <= 1'b0;
        bdt_list_mem <= 16'd0; bdt_load_mem <= 1'b0; bdt_s_mem <= 1'b0; bdt_wb_mem <= 1'b0;
        addr_pre_idx_bdt_mem <= 1'b0; addr_up_bdt_mem <= 1'b0; swap_byte_mem <= 1'b0;
        base_reg_mem <= 4'd0; base_value_mem <= {`DATA_WIDTH{1'b0}};
        swp_rd_mem <= 4'd0; swp_rm_mem <= 4'd0;
        cp_rd_data_mem <= {`DATA_WIDTH{1'b0}}; mul_long_mem <= 1'b0;
        tid_mem <= 2'd0; valid_mem <= 1'b0;
    end else if (!stall_all) begin
        alu_result_mem <= alu_result_ex2;
        mac_result_lo_mem <= mac_result_lo_ex2; mac_result_hi_mem <= mac_result_hi_ex2;
        mem_addr_mem <= mem_addr_ex2; store_data_mem <= store_data_ex2;
        mem_read_mem <= mem_read_ex2; mem_write_mem <= mem_write_ex2;
        mem_size_mem <= mem_size_ex2; mem_signed_mem <= mem_signed_ex2;
        wb_sel_mem <= wb_sel_ex2;
        wr_addr1_mem <= wr_addr1_ex2; wr_addr2_mem <= wr_addr2_ex2;
        wr_en1_mem <= wr_en1_ex2; wr_en2_mem <= wr_en2_ex2;
        pc_plus4_mem <= pc_plus4_ex2;
        is_multi_cycle_mem <= is_multi_cycle_ex2; t_bdt_mem <= t_bdt_ex2; t_swp_mem <= t_swp_ex2;
        bdt_list_mem <= bdt_list_ex2; bdt_load_mem <= bdt_load_ex2;
        bdt_s_mem <= bdt_s_ex2; bdt_wb_mem <= bdt_wb_ex2;
        addr_pre_idx_bdt_mem <= addr_pre_idx_bdt_ex2; addr_up_bdt_mem <= addr_up_bdt_ex2; swap_byte_mem <= swap_byte_ex2;
        base_reg_mem <= base_reg_ex2; base_value_mem <= rn_val_ex2;
        swp_rd_mem <= wr_addr1_ex2; swp_rm_mem <= rm_addr_ex2;
        cp_rd_data_mem <= cp_rd_data_i; mul_long_mem <= mul_long_ex2;
        tid_mem <= tid_ex2; valid_mem <= valid_ex2;
    end
end

/* ================================================================
   MEM — DATA MEMORY ACCESS + BDTU
   ================================================================ */
wire [`DATA_WIDTH-1:0] bdtu_mem_addr, bdtu_mem_wdata;
wire bdtu_mem_rd, bdtu_mem_wr;
wire [1:0] bdtu_mem_size;

bdtu u_bdtu (
    .clk(clk), .rst_n(rst_n),
    .start(is_multi_cycle_mem),
    .op_bdt(t_bdt_mem), .op_swp(t_swp_mem),
    .reg_list(bdt_list_mem),
    .bdt_load(bdt_load_mem), .bdt_wb(bdt_wb_mem),
    .pre_index(addr_pre_idx_bdt_mem), .up_down(addr_up_bdt_mem),
    .bdt_s(bdt_s_mem), .swap_byte(swap_byte_mem),
    .swp_rd(swp_rd_mem), .swp_rm(swp_rm_mem),
    .base_reg(base_reg_mem), .base_value(base_value_mem),
    .rf_rd_addr(bdtu_rf_rd_addr), .rf_rd_data(bdtu_rf_rd_data),
    .wr_addr1(bdtu_wr_addr1), .wr_data1(bdtu_wr_data1), .wr_en1(bdtu_wr_en1),
    .wr_addr2(bdtu_wr_addr2), .wr_data2(bdtu_wr_data2), .wr_en2(bdtu_wr_en2),
    .mem_addr(bdtu_mem_addr), .mem_wdata(bdtu_mem_wdata),
    .mem_rd(bdtu_mem_rd), .mem_wr(bdtu_mem_wr),
    .mem_size(bdtu_mem_size), .mem_rdata(d_mem_data_i),
    .busy(bdtu_busy)
);

assign d_mem_addr_o = bdtu_busy ? bdtu_mem_addr : mem_addr_mem;
assign d_mem_data_o = bdtu_busy ? bdtu_mem_wdata : store_data_mem;
assign d_mem_wen_o = bdtu_busy ? bdtu_mem_wr : mem_write_mem;
assign d_mem_size_o = bdtu_busy ? bdtu_mem_size : mem_size_mem;

/* ================================================================
   MEM/WB PIPELINE REGISTER
   ================================================================ */
reg [`DATA_WIDTH-1:0] alu_result_wb;
reg [`PC_WIDTH-1:0] pc_plus4_wb;
reg [2:0] wb_sel_wb;
reg [3:0] wr_addr1_wb, wr_addr2_wb;
reg wr_en1_wb, wr_en2_wb;
reg [1:0] mem_size_wb;
reg mem_signed_wb;
reg [`DATA_WIDTH-1:0] mac_result_lo_wb, mac_result_hi_wb, cp_rd_data_wb;
reg mul_long_wb;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_wb <= {`DATA_WIDTH{1'b0}};
        mac_result_lo_wb <= {`DATA_WIDTH{1'b0}}; mac_result_hi_wb <= {`DATA_WIDTH{1'b0}};
        cp_rd_data_wb <= {`DATA_WIDTH{1'b0}};
        pc_plus4_wb <= {`PC_WIDTH{1'b0}}; wb_sel_wb <= 3'd0;
        wr_addr1_wb <= 4'd0; wr_addr2_wb <= 4'd0; wr_en1_wb <= 1'b0; wr_en2_wb <= 1'b0;
        mem_size_wb <= 2'd0; mem_signed_wb <= 1'b0; mul_long_wb <= 1'b0;
        tid_wb <= 2'd0; valid_wb <= 1'b0;
    end else begin
        alu_result_wb <= alu_result_mem;
        mac_result_lo_wb <= mac_result_lo_mem; mac_result_hi_wb <= mac_result_hi_mem;
        cp_rd_data_wb <= cp_rd_data_mem;
        pc_plus4_wb <= pc_plus4_mem; wb_sel_wb <= wb_sel_mem;
        wr_addr1_wb <= wr_addr1_mem; wr_addr2_wb <= wr_addr2_mem;
        wr_en1_wb <= wr_en1_mem; wr_en2_wb <= wr_en2_mem;
        mem_size_wb <= mem_size_mem; mem_signed_wb <= mem_signed_mem; mul_long_wb <= mul_long_mem;
        tid_wb <= tid_mem; valid_wb <= valid_mem;
    end
end

/* ================================================================
   WB — WRITE-BACK
   ================================================================ */
reg [`DATA_WIDTH-1:0] load_data_wb;

always @(*) begin
    case (mem_size_wb)
        2'b00: load_data_wb = mem_signed_wb
                     ? {{(`DATA_WIDTH-8){d_mem_data_i[7]}}, d_mem_data_i[7:0]}
                     : {{(`DATA_WIDTH-8){1'b0}}, d_mem_data_i[7:0]};
        2'b01: load_data_wb = mem_signed_wb
                     ? {{(`DATA_WIDTH-16){d_mem_data_i[15]}}, d_mem_data_i[15:0]}
                     : {{(`DATA_WIDTH-16){1'b0}}, d_mem_data_i[15:0]};
        default: load_data_wb = d_mem_data_i;
    endcase
end

wire [3:0] cpsr_flags_wb = cpsr_flags[tid_wb];

reg [`DATA_WIDTH-1:0] wb_data1;
always @(*) begin
    case (wb_sel_wb)
        `WB_ALU: wb_data1 = alu_result_wb;
        `WB_MEM: wb_data1 = load_data_wb;
        `WB_LINK: wb_data1 = pc_plus4_wb;
        `WB_PSR: wb_data1 = {cpsr_flags_wb, {(`DATA_WIDTH-4){1'b0}}};
        `WB_MUL: wb_data1 = mac_result_lo_wb;
        `WB_CP: wb_data1 = cp_rd_data_wb;
        default: wb_data1 = alu_result_wb;
    endcase
end

wire [`DATA_WIDTH-1:0] wb_data2 = mul_long_wb ? mac_result_hi_wb : alu_result_wb;

assign wb_wr_addr1 = wr_addr1_wb;
assign wb_wr_data1 = wb_data1;
assign wb_wr_en1 = wr_en1_wb && valid_wb;
assign wb_wr_addr2 = wr_addr2_wb;
assign wb_wr_data2 = wb_data2;
assign wb_wr_en2 = wr_en2_wb && valid_wb;

/* ================================================================
   STALL LOGIC
   v2.8: ~running freezes entire pipeline until cpu_start
   ================================================================ */
assign stall_all = bdtu_busy | ~running;

endmodule

`endif // CPU_MT_V