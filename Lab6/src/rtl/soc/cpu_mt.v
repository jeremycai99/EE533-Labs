/* file: cpu_mt.v
 Description: Quad-threaded 7-stage pipeline Arm CPU module
              (IF1, IF2, ID, EX1, EX2, MEM, WB)
              IF split: IF1 = present address to BRAM
                        IF2 = register instruction + pre-decode RF addrs
              EX split: EX1 = barrel shifter + operand B mux + WB bypass
                        EX2 = ALU + branch
              MAC and ILA debug removed.
 Author: Jeremy Cai
 Date: Feb. 23, 2026
 Version: 2.6
 Changes from 2.5:
   - Rotated-immediate timing fix: CU v1.1 no longer computes the
     rotation combinationally (~9ns path that violated 8ns budget).
     Instead, CU outputs the raw unrotated imm8 and sets shift_type=ROR,
     shift_amount=rot_amount.  In EX1, the barrel shifter input is muxed:
       bs_din = alu_src_b ? imm32 : Rm
     so dp_imm rotations go through the existing barrel shifter from
     registered inputs (~3-4ns, well within budget).  alu_src_b_val is
     now always bs_dout (the mux that bypassed the barrel shifter for
     immediates is removed).  For non-rotated immediates (rot=0), the
     barrel shifter is identity (LSL #0).
   - Bonus: shifter_carry_out is now correct for rotated immediates
     (previously always cin; now bit[31] of rotated result per ARM spec).
 Changes from 2.4:
   - Deferred Z flag: alu_flags_ex3 no longer captures ALU's combinational
     Z flag.  Instead, N/C/V are registered from the ALU (fast paths),
     and Z is computed from alu_result_mem (already registered in EX2/MEM
     pipe reg) one cycle later.  This breaks the 13-level critical path:
       alu_op → KSA → result_mux → zero_detect_wg → alu_flags_ex3
     into two shorter paths:
       EX2:  alu_op → KSA → result_mux → {alu_result_mem, N/C/V_ex3}
       MEM:  alu_result_mem → zero_detect → cpsr_flags  (~3ns)
   - WB→ID bypass relocated to EX1: In the 7-stage/4-thread design,
     tid_wb == tid_id, so WB must bypass to the same-thread ID read.
     The RF read mux (16:1 across 4 mux levels) plus bypass plus
     PC-adjust exceeded 8ns due to RF placement routing (~3.4ns net).
     Fix: ID now registers raw RF data (no bypass, no PC-adjust).
     WB forwarding info (addr, data, en) is registered in the ID/EX1
     pipe reg.  Bypass muxes and PC-adjust are applied in EX1 from
     registered values, well within the 8ns budget (~3ns path).
 Changes from 2.3:
   - Added deferred CPSR flag update ("EX3 registers") as sidecar.
 Changes from 2.2:
   - Added support for second write port in regfile.
   - Removed hold buffer.
   - Added branch squash.
 */

`ifndef CPU_MT_V
`define CPU_MT_V

`include "define.v"
`include "regfile.v"
`include "cu.v"
`include "alu.v"
`include "cond_eval.v"
`include "bdtu.v"
`include "barrel_shifter.v"

module cpu_mt (
    input  wire clk,
    input  wire rst_n,

    // Instruction memory interface (shared, single-ported)
    input  wire [`INSTR_WIDTH-1:0] i_mem_data_i,
    output wire [`PC_WIDTH-1:0] i_mem_addr_o,

    // Data memory interface (shared, single-ported)
    input  wire [`DATA_WIDTH-1:0] d_mem_data_i,
    output wire [`CPU_DMEM_ADDR_WIDTH-1:0] d_mem_addr_o,
    output wire [`DATA_WIDTH-1:0] d_mem_data_o,
    output wire d_mem_wen_o,
    output wire [1:0] d_mem_size_o,

    output wire cpu_done
);

/* ================================================================
   GLOBAL CONTROL
   ================================================================ */
wire stall_all;
wire bdtu_busy;
wire branch_taken_ex2;
wire [`PC_WIDTH-1:0] branch_target_ex2;

/* ================================================================
   IF1 — INSTRUCTION FETCH STAGE 1 (present address to BRAM)
   Round-robin thread scheduling.
   ================================================================ */
reg [1:0] tid_if;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        tid_if <= 2'd0;
    else if (!stall_all)
        tid_if <= tid_if + 2'd1;
end

/* Pipeline-stage thread IDs and validity bits (7 stages) */
reg [1:0] tid_if2, tid_id, tid_ex1, tid_ex2, tid_mem, tid_wb;
reg valid_if2, valid_id, valid_ex1, valid_ex2, valid_mem, valid_wb;

/* Per-thread program counters */
reg [`PC_WIDTH-1:0] pc_thread [0:3];

wire [`PC_WIDTH-1:0] pc_if = pc_thread[tid_if];
wire [`PC_WIDTH-1:0] pc_plus4_if = pc_if + 32'd4;

assign i_mem_addr_o = pc_if;

integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (k = 0; k < 4; k = k + 1)
            pc_thread[k] <= k * 4;
    end
    else if (!stall_all) begin
        pc_thread[tid_if] <= pc_plus4_if;
        if (branch_taken_ex2 && valid_ex2)
            pc_thread[tid_ex2] <= branch_target_ex2;
    end
end

assign cpu_done = (pc_thread[0] == `CPU_DONE_PC) &&
                  (pc_thread[1] == `CPU_DONE_PC) &&
                  (pc_thread[2] == `CPU_DONE_PC) &&
                  (pc_thread[3] == `CPU_DONE_PC);

/* ================================================================
   IF1/IF2 PIPELINE REGISTER
   Captures thread ID, PC+4, and validity.
   The instruction itself arrives from BRAM one cycle later (IF2).

   Branch squash: in the 7-stage / 4-thread design,
   tid_ex2 = tid_if (both = cycle%4).  When EX2 takes a branch,
   the IF1 fetch used the stale PC.  We invalidate it here.
   ================================================================ */
wire squash_if1 = branch_taken_ex2 && valid_ex2;

reg [`PC_WIDTH-1:0] pc_plus4_if2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tid_if2      <= 2'd0;
        pc_plus4_if2 <= `PC_WIDTH'd0;
        valid_if2    <= 1'b0;
    end
    else if (!stall_all) begin
        tid_if2      <= tid_if;
        pc_plus4_if2 <= pc_plus4_if;
        valid_if2    <= !squash_if1;
    end
end

/* ================================================================
   IF2 — INSTRUCTION FETCH STAGE 2
   BRAM output (i_mem_data_i) is available here; We register the
   instruction AND pre-decode the RF read addresses from raw instruction
   bits. This eliminates the CU-decode-to-RF-read critical path.
   ================================================================ */

/* Pre-decode RF addresses directly from BRAM output (0ns — wiring) */
wire [3:0] rn_addr_pre = i_mem_data_i[19:16];
wire [3:0] rd_addr_pre = i_mem_data_i[15:12];
wire [3:0] rs_addr_pre = i_mem_data_i[11:8];
wire [3:0] rm_addr_pre = i_mem_data_i[3:0];

/* ================================================================
   IF2/ID PIPELINE REGISTER
   Registers:  instruction, pre-decoded RF addresses, tid, pc+4,
               valid.
   ================================================================ */
reg [`INSTR_WIDTH-1:0] instr_id_r;
reg [3:0] rn_addr_id, rd_addr_id, rs_addr_id, rm_addr_id;
reg [`PC_WIDTH-1:0] pc_plus4_id;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        instr_id_r  <= {`INSTR_WIDTH{1'b0}};
        rn_addr_id  <= 4'd0;
        rd_addr_id  <= 4'd0;
        rs_addr_id  <= 4'd0;
        rm_addr_id  <= 4'd0;
        pc_plus4_id <= `PC_WIDTH'd0;
        tid_id      <= 2'd0;
        valid_id    <= 1'b0;
    end
    else if (!stall_all) begin
        instr_id_r  <= i_mem_data_i;
        rn_addr_id  <= rn_addr_pre;
        rd_addr_id  <= rd_addr_pre;
        rs_addr_id  <= rs_addr_pre;
        rm_addr_id  <= rm_addr_pre;
        pc_plus4_id <= pc_plus4_if2;
        tid_id      <= tid_if2;
        valid_id    <= valid_if2;
    end
end

/* The registered instruction feeds the CU and cond_eval */
wire [`INSTR_WIDTH-1:0] instr_id = instr_id_r;

/* ================================================================
   ID — INSTRUCTION DECODE
   CU decodes from the registered instruction (instr_id_r).
   RF reads use the pre-decoded registered addresses (rn_addr_id,
   rm_addr_id, rs_addr_id, rd_addr_id) which are ready at the
   START of the ID cycle, without waiting for CU decode.

   v2.5: Bypass muxes and PC-adjust REMOVED from ID.  Raw RF data
   is registered into the ID/EX1 pipe reg.  Bypass is applied in
   EX1 from registered forwarding info.
   ================================================================ */

/* Per-thread CPSR flags (declared here, updated in EX3 below) */
reg [3:0] cpsr_flags [0:3];

wire [3:0] cond_flags_id = cpsr_flags[tid_id];
wire cond_met_raw;

cond_eval u_cond_eval (
    .cond_code (instr_id[31:28]),
    .flags     (cond_flags_id),
    .cond_met  (cond_met_raw)
);

wire cond_met_id = cond_met_raw && valid_id;

/* Control Unit */
wire t_dp_reg, t_dp_imm, t_swp, t_bx;
wire t_hdt_rego, t_hdt_immo, t_sdt_rego, t_sdt_immo;
wire t_bdt, t_br, t_mrs, t_msr_reg, t_msr_imm, t_swi, t_undef;

wire [3:0] cu_rn_addr, cu_rd_addr, cu_rs_addr, cu_rm_addr;
wire [3:0] wr_addr1_id, wr_addr2_id;
wire wr_en1_id, wr_en2_id;

wire [3:0]  alu_op_id;
wire alu_src_b_id;
wire cpsr_wen_id;

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

wire psr_rd_id, psr_wr_id, psr_field_sel_id;
wire [3:0] psr_mask_id;

wire [15:0] bdt_list_id;
wire bdt_load_id, bdt_s_id, bdt_wb_id;
wire swap_byte_id;
wire swi_en_id;

wire use_rn_id, use_rd_id, use_rs_id, use_rm_id;
wire is_multi_cycle_id;

cu u_cu (
    .instr (instr_id),
    .cond_met (cond_met_id),
    .t_dp_reg(t_dp_reg), .t_dp_imm(t_dp_imm),
    .t_mul(), .t_mull(),
    .t_swp(t_swp), .t_bx(t_bx),
    .t_hdt_rego(t_hdt_rego), .t_hdt_immo(t_hdt_immo),
    .t_sdt_rego(t_sdt_rego), .t_sdt_immo(t_sdt_immo),
    .t_bdt(t_bdt), .t_br(t_br), .t_mrs(t_mrs),
    .t_msr_reg(t_msr_reg), .t_msr_imm(t_msr_imm),
    .t_swi(t_swi), .t_undef(t_undef),
    .rn_addr(cu_rn_addr), .rd_addr(cu_rd_addr),
    .rs_addr(cu_rs_addr), .rm_addr(cu_rm_addr),
    .wr_addr1(wr_addr1_id), .wr_en1(wr_en1_id),
    .wr_addr2(wr_addr2_id), .wr_en2(wr_en2_id),
    .alu_op(alu_op_id), .alu_src_b(alu_src_b_id),
    .cpsr_wen(cpsr_wen_id),
    .shift_type(shift_type_id), .shift_amount(shift_amount_id),
    .shift_src(shift_src_id),
    .imm32(imm32_id),
    .mem_read(mem_read_id), .mem_write(mem_write_id),
    .mem_size(mem_size_id), .mem_signed(mem_signed_id),
    .addr_pre_idx(addr_pre_idx_id), .addr_up(addr_up_id),
    .addr_wb(addr_wb_id),
    .wb_sel(wb_sel_id),
    .branch_en(branch_en_id), .branch_link(branch_link_id),
    .branch_exchange(branch_exchange_id),
    .mul_en(), .mul_long(),
    .mul_signed(), .mul_accumulate(),
    .psr_rd(psr_rd_id), .psr_wr(psr_wr_id),
    .psr_field_sel(psr_field_sel_id), .psr_mask(psr_mask_id),
    .bdt_list(bdt_list_id), .bdt_load(bdt_load_id),
    .bdt_s(bdt_s_id), .bdt_wb(bdt_wb_id),
    .swap_byte(swap_byte_id), .swi_en(swi_en_id),
    .use_rn(use_rn_id), .use_rd(use_rd_id),
    .use_rs(use_rs_id), .use_rm(use_rm_id),
    .is_multi_cycle(is_multi_cycle_id)
);

/* BDTU signals (forward-declared, driven by BDTU in MEM stage) */
wire [3:0] bdtu_rf_rd_addr;
wire [3:0] bdtu_wr_addr1, bdtu_wr_addr2;
wire [`DATA_WIDTH-1:0] bdtu_wr_data1, bdtu_wr_data2;
wire bdtu_wr_en1, bdtu_wr_en2;
wire bdtu_has_write = bdtu_wr_en1 | bdtu_wr_en2;

/* WB write signals (forward-declared, driven in WB stage) */
wire [3:0] wb_wr_addr1, wb_wr_addr2;
wire [`DATA_WIDTH-1:0] wb_wr_data1, wb_wr_data2;
wire wb_wr_en1, wb_wr_en2;

/* ================================================================
   REGISTER FILES (4 instances, one per thread)

   RF read addresses come from pre-decoded pipeline registers
   (rn_addr_id, rm_addr_id, rs_addr_id, rd_addr_id), which are
   available at the START of ID without waiting for CU decode.

   v2.5: Raw RF outputs go directly to pipe reg (no bypass in ID).
   ================================================================ */
genvar g;
generate
    for (g = 0; g < 4; g = g + 1) begin : THREAD_RF
        wire is_wb_target   = (tid_wb  == g) && valid_wb;
        wire is_bdtu_target = (tid_mem == g) && bdtu_has_write;

        wire wena = (is_wb_target && (wb_wr_en1 || wb_wr_en2))
                  || is_bdtu_target;

        wire [3:0] wa1 = is_bdtu_target
            ? (bdtu_wr_en1 ? bdtu_wr_addr1 : bdtu_wr_addr2)
            : (wb_wr_en1   ? wb_wr_addr1   : wb_wr_addr2);

        wire [`DATA_WIDTH-1:0] wd1 = is_bdtu_target
            ? (bdtu_wr_en1 ? bdtu_wr_data1 : bdtu_wr_data2)
            : (wb_wr_en1   ? wb_wr_data1   : wb_wr_data2);

        wire [3:0] wa2 = is_bdtu_target
            ? ((bdtu_wr_en1 && bdtu_wr_en2) ? bdtu_wr_addr2 : wa1)
            : ((wb_wr_en1   && wb_wr_en2)   ? wb_wr_addr2   : wa1);

        wire [`DATA_WIDTH-1:0] wd2 = is_bdtu_target
            ? ((bdtu_wr_en1 && bdtu_wr_en2) ? bdtu_wr_data2 : wd1)
            : ((wb_wr_en1   && wb_wr_en2)   ? wb_wr_data2   : wd1);

        /* ---- Write-port pipeline register (timing fix, v2.2) ---- */
        reg        wena_r;
        reg [3:0]  wa1_r, wa2_r;
        reg [`DATA_WIDTH-1:0] wd1_r, wd2_r;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wena_r <= 1'b0;
                wa1_r  <= 4'd0;
                wa2_r  <= 4'd0;
                wd1_r  <= {`DATA_WIDTH{1'b0}};
                wd2_r  <= {`DATA_WIDTH{1'b0}};
            end else begin
                wena_r <= wena;
                wa1_r  <= wa1;
                wa2_r  <= wa2;
                wd1_r  <= wd1;
                wd2_r  <= wd2;
            end
        end

        /* RF r3 port: rs_addr_id for normal ops, bdtu_rf_rd_addr
           when BDTU is active for this thread. */
        wire [3:0] local_r3addr = (bdtu_busy && (tid_mem == g))
                                  ? bdtu_rf_rd_addr : rs_addr_id;

        wire [`DATA_WIDTH-1:0] rn_out, rm_out, r3_out, r4_out;

        regfile u_rf (
            .clk     (clk),
            .r1addr  (rn_addr_id),
            .r2addr  (rm_addr_id),
            .r3addr  (local_r3addr),
            .r4addr  (rd_addr_id),
            .wena    (wena_r),
            .wr_addr1(wa1_r),   .wr_data1(wd1_r),
            .wr_addr2(wa2_r),   .wr_data2(wd2_r),
            .r1data  (rn_out),
            .r2data  (rm_out),
            .r3data  (r3_out),
            .r4data  (r4_out)
        );
    end
endgenerate

/* ── Read MUX: select outputs from RF[tid_id] ── */
reg [`DATA_WIDTH-1:0] rn_data_id, rm_data_id, r3_data_id, r4_data_id;

always @(*) begin
    case (tid_id)
        2'd0: begin
            rn_data_id = THREAD_RF[0].rn_out;
            rm_data_id = THREAD_RF[0].rm_out;
            r3_data_id = THREAD_RF[0].r3_out;
            r4_data_id = THREAD_RF[0].r4_out;
        end
        2'd1: begin
            rn_data_id = THREAD_RF[1].rn_out;
            rm_data_id = THREAD_RF[1].rm_out;
            r3_data_id = THREAD_RF[1].r3_out;
            r4_data_id = THREAD_RF[1].r4_out;
        end
        2'd2: begin
            rn_data_id = THREAD_RF[2].rn_out;
            rm_data_id = THREAD_RF[2].rm_out;
            r3_data_id = THREAD_RF[2].r3_out;
            r4_data_id = THREAD_RF[2].r4_out;
        end
        default: begin
            rn_data_id = THREAD_RF[3].rn_out;
            rm_data_id = THREAD_RF[3].rm_out;
            r3_data_id = THREAD_RF[3].r3_out;
            r4_data_id = THREAD_RF[3].r4_out;
        end
    endcase
end

/* BDTU read-data MUX: select from RF[tid_mem] port 3 */
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

   v2.5 changes:
   - Raw RF data registered (no bypass, no PC-adjust applied here).
   - RF read addresses propagated for EX1 bypass comparison.
   - WB forwarding info (addr, data, en) captured for EX1 bypass.
   ================================================================ */
reg [3:0] alu_op_ex1;
reg alu_src_b_ex1;
reg cpsr_wen_ex1;
reg [1:0] shift_type_ex1;
reg [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount_ex1;
reg shift_src_ex1;
reg [`DATA_WIDTH-1:0] imm32_ex1;
reg mem_read_ex1, mem_write_ex1;
reg [1:0] mem_size_ex1;
reg mem_signed_ex1;
reg addr_pre_idx_ex1, addr_up_ex1, addr_wb_ex1;
reg [2:0] wb_sel_ex1;
reg [3:0] wr_addr1_ex1, wr_addr2_ex1;
reg wr_en1_ex1, wr_en2_ex1;
reg branch_en_ex1, branch_link_ex1, branch_exchange_ex1;

reg [`DATA_WIDTH-1:0] rn_data_ex1, rm_data_ex1, rs_data_ex1, rd_data_ex1;
reg [`PC_WIDTH-1:0] pc_plus4_ex1;

reg is_multi_cycle_ex1;
reg t_bdt_ex1, t_swp_ex1;
reg [15:0] bdt_list_ex1;
reg bdt_load_ex1, bdt_s_ex1, bdt_wb_ex1;
reg addr_pre_idx_bdt_ex1, addr_up_bdt_ex1;
reg swap_byte_ex1;
reg [3:0] base_reg_ex1;
reg [3:0] rm_addr_ex1;

reg psr_wr_ex1;
reg [3:0] psr_mask_ex1;
reg psr_field_sel_ex1;

/* v2.5: RF read addresses for EX1 bypass comparison */
reg [3:0] rn_addr_ex1, rs_addr_ex1, rd_addr_ex1;

/* v2.5: WB forwarding info for EX1 bypass */
reg [3:0]              fwd_addr1_ex1, fwd_addr2_ex1;
reg [`DATA_WIDTH-1:0]  fwd_data1_ex1, fwd_data2_ex1;
reg                    fwd_en1_ex1,   fwd_en2_ex1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_op_ex1           <= 4'd0;
        alu_src_b_ex1        <= 1'b0;
        cpsr_wen_ex1         <= 1'b0;
        shift_type_ex1       <= 2'd0;
        shift_amount_ex1     <= 5'd0;
        shift_src_ex1        <= 1'b0;
        imm32_ex1            <= 32'd0;
        mem_read_ex1         <= 1'b0;
        mem_write_ex1        <= 1'b0;
        mem_size_ex1         <= 2'd0;
        mem_signed_ex1       <= 1'b0;
        addr_pre_idx_ex1     <= 1'b0;
        addr_up_ex1          <= 1'b0;
        addr_wb_ex1          <= 1'b0;
        wb_sel_ex1           <= 3'd0;
        wr_addr1_ex1         <= 4'd0;
        wr_addr2_ex1         <= 4'd0;
        wr_en1_ex1           <= 1'b0;
        wr_en2_ex1           <= 1'b0;
        branch_en_ex1        <= 1'b0;
        branch_link_ex1      <= 1'b0;
        branch_exchange_ex1  <= 1'b0;
        rn_data_ex1          <= 32'd0;
        rm_data_ex1          <= 32'd0;
        rs_data_ex1          <= 32'd0;
        rd_data_ex1          <= 32'd0;
        pc_plus4_ex1         <= 32'd0;
        is_multi_cycle_ex1   <= 1'b0;
        t_bdt_ex1            <= 1'b0;
        t_swp_ex1            <= 1'b0;
        bdt_list_ex1         <= 16'd0;
        bdt_load_ex1         <= 1'b0;
        bdt_s_ex1            <= 1'b0;
        bdt_wb_ex1           <= 1'b0;
        addr_pre_idx_bdt_ex1 <= 1'b0;
        addr_up_bdt_ex1      <= 1'b0;
        swap_byte_ex1        <= 1'b0;
        base_reg_ex1         <= 4'd0;
        rm_addr_ex1          <= 4'd0;
        psr_wr_ex1           <= 1'b0;
        psr_mask_ex1         <= 4'd0;
        psr_field_sel_ex1    <= 1'b0;
        rn_addr_ex1          <= 4'd0;
        rs_addr_ex1          <= 4'd0;
        rd_addr_ex1          <= 4'd0;
        fwd_addr1_ex1        <= 4'd0;
        fwd_addr2_ex1        <= 4'd0;
        fwd_data1_ex1        <= 32'd0;
        fwd_data2_ex1        <= 32'd0;
        fwd_en1_ex1          <= 1'b0;
        fwd_en2_ex1          <= 1'b0;
        tid_ex1              <= 2'd0;
        valid_ex1            <= 1'b0;
    end
    else if (!stall_all) begin
        alu_op_ex1           <= alu_op_id;
        alu_src_b_ex1        <= alu_src_b_id;
        cpsr_wen_ex1         <= cpsr_wen_id;
        shift_type_ex1       <= shift_type_id;
        shift_amount_ex1     <= shift_amount_id;
        shift_src_ex1        <= shift_src_id;
        imm32_ex1            <= imm32_id;
        mem_read_ex1         <= mem_read_id;
        mem_write_ex1        <= mem_write_id;
        mem_size_ex1         <= mem_size_id;
        mem_signed_ex1       <= mem_signed_id;
        addr_pre_idx_ex1     <= addr_pre_idx_id;
        addr_up_ex1          <= addr_up_id;
        addr_wb_ex1          <= addr_wb_id;
        wb_sel_ex1           <= wb_sel_id;
        wr_addr1_ex1         <= wr_addr1_id;
        wr_addr2_ex1         <= wr_addr2_id;
        wr_en1_ex1           <= wr_en1_id;
        wr_en2_ex1           <= wr_en2_id;
        branch_en_ex1        <= branch_en_id;
        branch_link_ex1      <= branch_link_id;
        branch_exchange_ex1  <= branch_exchange_id;
        /* v2.5: raw RF data — no bypass, no PC-adjust */
        rn_data_ex1          <= rn_data_id;
        rm_data_ex1          <= rm_data_id;
        rs_data_ex1          <= r3_data_id;
        rd_data_ex1          <= r4_data_id;
        pc_plus4_ex1         <= pc_plus4_id;
        is_multi_cycle_ex1   <= is_multi_cycle_id;
        t_bdt_ex1            <= t_bdt;
        t_swp_ex1            <= t_swp;
        bdt_list_ex1         <= bdt_list_id;
        bdt_load_ex1         <= bdt_load_id;
        bdt_s_ex1            <= bdt_s_id;
        bdt_wb_ex1           <= bdt_wb_id;
        addr_pre_idx_bdt_ex1 <= addr_pre_idx_id;
        addr_up_bdt_ex1      <= addr_up_id;
        swap_byte_ex1        <= swap_byte_id;
        base_reg_ex1         <= rn_addr_id;
        rm_addr_ex1          <= rm_addr_id;
        psr_wr_ex1           <= psr_wr_id;
        psr_mask_ex1         <= psr_mask_id;
        psr_field_sel_ex1    <= psr_field_sel_id;
        /* v2.5: RF read addresses for EX1 bypass */
        rn_addr_ex1          <= rn_addr_id;
        rs_addr_ex1          <= rs_addr_id;
        rd_addr_ex1          <= rd_addr_id;
        /* v2.5: WB forwarding info for EX1 bypass.
           In 7-stage/4-thread, tid_wb == tid_id at the same cycle.
           We capture the WB write signals here so that in EX1
           (one cycle later) we can apply the bypass from registered
           values.  wb_wr_en1/2 already incorporate valid_wb. */
        fwd_addr1_ex1        <= wb_wr_addr1;
        fwd_addr2_ex1        <= wb_wr_addr2;
        fwd_data1_ex1        <= wb_wr_data1;
        fwd_data2_ex1        <= wb_wr_data2;
        fwd_en1_ex1          <= wb_wr_en1;
        fwd_en2_ex1          <= wb_wr_en2;
        tid_ex1              <= tid_id;
        valid_ex1            <= valid_id;
    end
end

/* ================================================================
   EX1 — SHIFT / OPERAND-PREPARE  (barrel shifter + opB mux)

   v2.5: WB bypass and PC-adjust now applied here from registered
   values, instead of in ID.

   v2.6: Barrel shifter input muxed between Rm and imm32 based on
   alu_src_b.  For dp_imm with rotation, CU sets shift_type=ROR
   and shift_amount=rot_amount, so the barrel shifter performs the
   rotation from registered inputs.  alu_src_b_val is always
   bs_dout (no bypass mux).

   Bypass safety proof for 7-stage / 4-thread:
     At cycle C:  ID has thread T, WB has thread T (tid_wb==tid_id).
     At posedge C: raw RF data -> rn_data_ex1, WB info -> fwd_*_ex1.
     At cycle C+1: EX1 has thread T (tid_ex1 == T).
                   fwd_*_ex1 contains WB data for thread T from cycle C.
                   rn_data_ex1 contains stale RF data for thread T.
                   Bypass mux corrects rn_data_ex1 with fwd_data if
                   address matches.  Identical behavior to old ID bypass,
                   just deferred by one cycle with all inputs registered.
   ================================================================ */

/* ── EX1 Bypass from registered WB forwarding info ── */
wire fwd_rn_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rn_addr_ex1);
wire fwd_rn_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rn_addr_ex1);
wire [`DATA_WIDTH-1:0] rn_bypassed_ex1 = fwd_rn_p1 ? fwd_data1_ex1 :
                                          fwd_rn_p2 ? fwd_data2_ex1 :
                                                       rn_data_ex1;

wire fwd_rm_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rm_addr_ex1);
wire fwd_rm_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rm_addr_ex1);
wire [`DATA_WIDTH-1:0] rm_bypassed_ex1 = fwd_rm_p1 ? fwd_data1_ex1 :
                                          fwd_rm_p2 ? fwd_data2_ex1 :
                                                       rm_data_ex1;

wire fwd_rs_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rs_addr_ex1);
wire fwd_rs_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rs_addr_ex1);
wire [`DATA_WIDTH-1:0] rs_bypassed_ex1 = fwd_rs_p1 ? fwd_data1_ex1 :
                                          fwd_rs_p2 ? fwd_data2_ex1 :
                                                       rs_data_ex1;

wire fwd_rd_p1 = fwd_en1_ex1 && (fwd_addr1_ex1 == rd_addr_ex1);
wire fwd_rd_p2 = fwd_en2_ex1 && (fwd_addr2_ex1 == rd_addr_ex1);
wire [`DATA_WIDTH-1:0] rd_bypassed_ex1 = fwd_rd_p1 ? fwd_data1_ex1 :
                                          fwd_rd_p2 ? fwd_data2_ex1 :
                                                       rd_data_ex1;

/* ── PC+8 adjustment for reads of R15, applied after bypass ── */
wire [`DATA_WIDTH-1:0] pc_plus8_ex1 = pc_plus4_ex1 + 32'd4;

wire [`DATA_WIDTH-1:0] rn_val_ex1 =
    (rn_addr_ex1 == 4'd15) ? pc_plus8_ex1 : rn_bypassed_ex1;
wire [`DATA_WIDTH-1:0] rm_val_ex1 =
    (rm_addr_ex1 == 4'd15) ? pc_plus8_ex1 : rm_bypassed_ex1;
wire [`DATA_WIDTH-1:0] rs_val_ex1 = rs_bypassed_ex1;
wire [`DATA_WIDTH-1:0] rd_val_ex1 = rd_bypassed_ex1;

/* ── Barrel shifter + operand B mux ── */
wire [3:0] cpsr_flags_ex1 = cpsr_flags[tid_ex1];

wire [`SHIFT_AMOUNT_WIDTH-1:0] actual_shamt_ex1 =
    shift_src_ex1 ? rs_val_ex1[`SHIFT_AMOUNT_WIDTH-1:0] : shift_amount_ex1;

/* v2.6 FIX: Barrel shifter input mux.
 *
 * When alu_src_b=1 (dp_imm, msr_imm, sdt_immo, hdt_immo), feed
 * the immediate value into the barrel shifter instead of Rm.
 *   - dp_imm with rotation: CU sets shift_type=ROR, shift_amount=
 *     rot_amount → barrel shifter rotates the raw imm8 → correct result.
 *   - dp_imm without rotation: shift_amount=0, shift_type=LSL →
 *     barrel shifter is identity → imm32 passes through unchanged.
 *   - sdt_immo / hdt_immo: shift_amount=0, shift_type=LSL →
 *     barrel shifter is identity → imm32 passes through unchanged.
 *
 * When alu_src_b=0 (dp_reg, sdt_rego), feed Rm as before.
 */
wire [`DATA_WIDTH-1:0] bs_din_ex1 = alu_src_b_ex1 ? imm32_ex1 : rm_val_ex1;

wire [`DATA_WIDTH-1:0] bs_dout_ex1;
wire                   shifter_cout_ex1;

barrel_shifter u_barrel_shifter (
    .din        (bs_din_ex1),
    .shamt      (actual_shamt_ex1),
    .shift_type (shift_type_ex1),
    .is_imm_shift (~shift_src_ex1),
    .cin        (cpsr_flags_ex1[`FLAG_C]),
    .dout       (bs_dout_ex1),
    .cout       (shifter_cout_ex1)
);

/* v2.6 FIX: Always use barrel shifter output for operand B.
 *
 * OLD (v2.5): alu_src_b_val = alu_src_b ? imm32 : bs_dout
 *   This bypassed the barrel shifter for immediates, which was fine
 *   when CU pre-rotated imm_dp.  Now that CU outputs raw imm8,
 *   the barrel shifter must always be in the path.
 *
 * For all instruction types, bs_dout is correct:
 *   dp_imm (rot!=0): bs_dout = ROR(imm8, rot_amount) ✓
 *   dp_imm (rot==0): bs_dout = imm8 (identity) ✓
 *   dp_reg:          bs_dout = shifted Rm ✓
 *   sdt_immo:        bs_dout = imm12 (identity, shift=0) ✓
 *   sdt_rego:        bs_dout = shifted Rm ✓
 *   hdt_immo:        bs_dout = imm8 (identity, shift=0) ✓
 */
wire [`DATA_WIDTH-1:0] alu_src_b_val_ex1 = bs_dout_ex1;

wire [`PC_WIDTH-1:0] branch_target_br_ex1 =
    pc_plus4_ex1 + 32'd4 + imm32_ex1;

/* ================================================================
   EX1/EX2 PIPELINE REGISTER
   ================================================================ */

/* Computed results from EX1 */
reg [`DATA_WIDTH-1:0]  alu_src_b_val_ex2;
reg                    shifter_cout_ex2;
reg [`PC_WIDTH-1:0]    branch_target_br_ex2;
reg                    carry_in_ex2;

/* Pass-through control */
reg [3:0] alu_op_ex2;
reg cpsr_wen_ex2;
reg mem_read_ex2, mem_write_ex2;
reg [1:0] mem_size_ex2;
reg mem_signed_ex2;
reg addr_pre_idx_ex2, addr_up_ex2, addr_wb_ex2;
reg [2:0] wb_sel_ex2;
reg [3:0] wr_addr1_ex2, wr_addr2_ex2;
reg wr_en1_ex2, wr_en2_ex2;
reg branch_en_ex2_r, branch_link_ex2, branch_exchange_ex2;

/* Pass-through data */
reg [`DATA_WIDTH-1:0] rn_data_ex2, rm_data_ex2, rs_data_ex2, rd_data_ex2;
reg [`PC_WIDTH-1:0]   pc_plus4_ex2;

/* Multi-cycle / BDT / SWP */
reg is_multi_cycle_ex2;
reg t_bdt_ex2, t_swp_ex2;
reg [15:0] bdt_list_ex2;
reg bdt_load_ex2, bdt_s_ex2, bdt_wb_ex2;
reg addr_pre_idx_bdt_ex2, addr_up_bdt_ex2;
reg swap_byte_ex2;
reg [3:0] base_reg_ex2;
reg [3:0] rm_addr_ex2;

/* PSR */
reg psr_wr_ex2;
reg [3:0] psr_mask_ex2;
reg psr_field_sel_ex2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_src_b_val_ex2    <= 32'd0;
        shifter_cout_ex2     <= 1'b0;
        branch_target_br_ex2 <= 32'd0;
        carry_in_ex2         <= 1'b0;
        alu_op_ex2           <= 4'd0;
        cpsr_wen_ex2         <= 1'b0;
        mem_read_ex2         <= 1'b0;
        mem_write_ex2        <= 1'b0;
        mem_size_ex2         <= 2'd0;
        mem_signed_ex2       <= 1'b0;
        addr_pre_idx_ex2     <= 1'b0;
        addr_up_ex2          <= 1'b0;
        addr_wb_ex2          <= 1'b0;
        wb_sel_ex2           <= 3'd0;
        wr_addr1_ex2         <= 4'd0;
        wr_addr2_ex2         <= 4'd0;
        wr_en1_ex2           <= 1'b0;
        wr_en2_ex2           <= 1'b0;
        branch_en_ex2_r      <= 1'b0;
        branch_link_ex2      <= 1'b0;
        branch_exchange_ex2  <= 1'b0;
        rn_data_ex2          <= 32'd0;
        rm_data_ex2          <= 32'd0;
        rs_data_ex2          <= 32'd0;
        rd_data_ex2          <= 32'd0;
        pc_plus4_ex2         <= 32'd0;
        is_multi_cycle_ex2   <= 1'b0;
        t_bdt_ex2            <= 1'b0;
        t_swp_ex2            <= 1'b0;
        bdt_list_ex2         <= 16'd0;
        bdt_load_ex2         <= 1'b0;
        bdt_s_ex2            <= 1'b0;
        bdt_wb_ex2           <= 1'b0;
        addr_pre_idx_bdt_ex2 <= 1'b0;
        addr_up_bdt_ex2      <= 1'b0;
        swap_byte_ex2        <= 1'b0;
        base_reg_ex2         <= 4'd0;
        rm_addr_ex2          <= 4'd0;
        psr_wr_ex2           <= 1'b0;
        psr_mask_ex2         <= 4'd0;
        psr_field_sel_ex2    <= 1'b0;
        tid_ex2              <= 2'd0;
        valid_ex2            <= 1'b0;
    end
    else if (!stall_all) begin
        alu_src_b_val_ex2    <= alu_src_b_val_ex1;
        shifter_cout_ex2     <= shifter_cout_ex1;
        branch_target_br_ex2 <= branch_target_br_ex1;
        carry_in_ex2         <= cpsr_flags_ex1[`FLAG_C];
        alu_op_ex2           <= alu_op_ex1;
        cpsr_wen_ex2         <= cpsr_wen_ex1;
        mem_read_ex2         <= mem_read_ex1;
        mem_write_ex2        <= mem_write_ex1;
        mem_size_ex2         <= mem_size_ex1;
        mem_signed_ex2       <= mem_signed_ex1;
        addr_pre_idx_ex2     <= addr_pre_idx_ex1;
        addr_up_ex2          <= addr_up_ex1;
        addr_wb_ex2          <= addr_wb_ex1;
        wb_sel_ex2           <= wb_sel_ex1;
        wr_addr1_ex2         <= wr_addr1_ex1;
        wr_addr2_ex2         <= wr_addr2_ex1;
        wr_en1_ex2           <= wr_en1_ex1;
        wr_en2_ex2           <= wr_en2_ex1;
        branch_en_ex2_r      <= branch_en_ex1;
        branch_link_ex2      <= branch_link_ex1;
        branch_exchange_ex2  <= branch_exchange_ex1;
        rn_data_ex2          <= rn_val_ex1;
        rm_data_ex2          <= rm_val_ex1;
        rs_data_ex2          <= rs_val_ex1;
        rd_data_ex2          <= rd_val_ex1;
        pc_plus4_ex2         <= pc_plus4_ex1;
        is_multi_cycle_ex2   <= is_multi_cycle_ex1;
        t_bdt_ex2            <= t_bdt_ex1;
        t_swp_ex2            <= t_swp_ex1;
        bdt_list_ex2         <= bdt_list_ex1;
        bdt_load_ex2         <= bdt_load_ex1;
        bdt_s_ex2            <= bdt_s_ex1;
        bdt_wb_ex2           <= bdt_wb_ex1;
        addr_pre_idx_bdt_ex2 <= addr_pre_idx_bdt_ex1;
        addr_up_bdt_ex2      <= addr_up_bdt_ex1;
        swap_byte_ex2        <= swap_byte_ex1;
        base_reg_ex2         <= base_reg_ex1;
        rm_addr_ex2          <= rm_addr_ex1;
        psr_wr_ex2           <= psr_wr_ex1;
        psr_mask_ex2         <= psr_mask_ex1;
        psr_field_sel_ex2    <= psr_field_sel_ex1;
        tid_ex2              <= tid_ex1;
        valid_ex2            <= valid_ex1;
    end
end

/* ================================================================
   EX2 — ALU / BRANCH-RESOLVE
   ================================================================ */

wire [`DATA_WIDTH-1:0] rn_val_ex2       = rn_data_ex2;
wire [`DATA_WIDTH-1:0] rm_val_ex2       = rm_data_ex2;
wire [`DATA_WIDTH-1:0] rd_store_val_ex2 = rd_data_ex2;

wire [`DATA_WIDTH-1:0] alu_result_ex2;
wire [3:0] alu_flags_ex2;

alu u_alu (
    .operand_a       (rn_val_ex2),
    .operand_b       (alu_src_b_val_ex2),
    .alu_op          (alu_op_ex2),
    .cin             (carry_in_ex2),
    .shift_carry_out (shifter_cout_ex2),
    .result          (alu_result_ex2),
    .alu_flags       (alu_flags_ex2)
);

/* Branch target (final selection) */
wire [`PC_WIDTH-1:0] branch_target_bx_ex2 = rm_val_ex2;

assign branch_taken_ex2  = branch_en_ex2_r;
assign branch_target_ex2 = branch_exchange_ex2 ? branch_target_bx_ex2
                                                : branch_target_br_ex2;

/* PSR-write flag detection (combinational, feeds EX3 regs) */
wire psr_wr_flags_ex2 = psr_wr_ex2 && psr_mask_ex2[3] && !psr_field_sel_ex2;

/* Memory address / store data */
wire [`CPU_DMEM_ADDR_WIDTH-1:0] mem_addr_ex2 =
    addr_pre_idx_ex2 ? alu_result_ex2 : rn_val_ex2;
wire [`DATA_WIDTH-1:0] store_data_ex2 = rd_store_val_ex2;

/* ================================================================
   EX3 — DEFERRED CPSR FLAG UPDATE (sidecar registers)
   ================================================================
   v2.5 fix: The ALU's combinational Z flag (alu_flags_ex2[2]) depends
   on the full 32-bit result through a wide-gate zero-detect, creating
   a 13-level path from alu_op_ex2.  We do NOT register the Z flag
   here.  Instead, only N/C/V are captured (these are fast: N = bit[31],
   C = adder carry / shifter carry, V = overflow XOR).

   Z is computed in the next cycle from alu_result_mem, which is the
   same ALU result registered in the EX2/MEM pipe reg at the same
   clock edge.  The zero-detect on a registered input takes ~2-3ns
   (carry-chain wide-gate), well within the 8ns budget.

   Timing safety (4-thread interleave):
     Thread T, SUBS in EX2 at cycle C:
       posedge C: alu_result_ex2 → alu_result_mem (EX2/MEM pipe reg)
                  N/C/V          → alu_flag_{n,c,v}_ex3  (EX3 sidecar)
       cycle C+1: flag_z_deferred = (alu_result_mem == 0)  (combinational)
       posedge C+1: cpsr_flags[T] ← {N_ex3, Z_deferred, C_ex3, V_ex3}
     Thread T, next instr reads CPSR:
       ID at cycle C+4: reads cpsr_flags[T] — written 3 cycles earlier ✓
       EX1 at cycle C+5: reads carry_in — written 4 cycles earlier ✓
   ================================================================ */
reg        alu_flag_n_ex3;        /* N: result[31] — 1 bit, fast       */
reg        alu_flag_c_ex3;        /* C: adder carry / shifter cout     */
reg        alu_flag_v_ex3;        /* V: signed overflow                */
reg [3:0]  alu_result_top4_ex3;   /* result[31:28] for MSR instruction */
reg        cpsr_wen_ex3;
reg        psr_wr_flags_ex3;
reg [1:0]  tid_ex3;
reg        valid_ex3;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_flag_n_ex3      <= 1'b0;
        alu_flag_c_ex3      <= 1'b0;
        alu_flag_v_ex3      <= 1'b0;
        alu_result_top4_ex3 <= 4'b0;
        cpsr_wen_ex3        <= 1'b0;
        psr_wr_flags_ex3    <= 1'b0;
        tid_ex3             <= 2'd0;
        valid_ex3           <= 1'b0;
    end
    else if (!stall_all) begin
        alu_flag_n_ex3      <= alu_flags_ex2[3];  /* N */
        alu_flag_c_ex3      <= alu_flags_ex2[1];  /* C */
        alu_flag_v_ex3      <= alu_flags_ex2[0];  /* V */
        alu_result_top4_ex3 <= alu_result_ex2[31:28];
        cpsr_wen_ex3        <= cpsr_wen_ex2;
        psr_wr_flags_ex3    <= psr_wr_flags_ex2;
        tid_ex3             <= tid_ex2;
        valid_ex3           <= valid_ex2;
    end
end

/* Z flag from registered ALU result — breaks the critical path.
   alu_result_mem is captured at the same posedge as the EX3 regs.
   In the following cycle, this combinational zero-detect (~2-3ns
   with carry chain) feeds the cpsr_flags write — trivially within
   the 8ns budget from a registered source. */

reg [`DATA_WIDTH-1:0] alu_result_mem;
wire flag_z_deferred = (alu_result_mem == {`DATA_WIDTH{1'b0}});

/* Per-thread CPSR flags update — from EX3 registered values + deferred Z */
integer f;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (f = 0; f < 4; f = f + 1)
            cpsr_flags[f] <= 4'b0;
    end
    else if (!stall_all && valid_ex3) begin
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

reg is_multi_cycle_mem;
reg t_bdt_mem, t_swp_mem;
reg [15:0] bdt_list_mem;
reg bdt_load_mem, bdt_s_mem, bdt_wb_mem;
reg addr_pre_idx_bdt_mem, addr_up_bdt_mem;
reg swap_byte_mem;
reg [3:0] base_reg_mem;
reg [`DATA_WIDTH-1:0] base_value_mem;
reg [3:0] swp_rd_mem, swp_rm_mem;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_mem       <= {`DATA_WIDTH{1'b0}};
        mem_addr_mem         <= {`CPU_DMEM_ADDR_WIDTH{1'b0}};
        store_data_mem       <= {`DATA_WIDTH{1'b0}};
        mem_read_mem         <= 1'b0;
        mem_write_mem        <= 1'b0;
        mem_size_mem         <= 2'd0;
        mem_signed_mem       <= 1'b0;
        wb_sel_mem           <= 3'd0;
        wr_addr1_mem         <= 4'd0;
        wr_addr2_mem         <= 4'd0;
        wr_en1_mem           <= 1'b0;
        wr_en2_mem           <= 1'b0;
        pc_plus4_mem         <= {`PC_WIDTH{1'b0}};
        is_multi_cycle_mem   <= 1'b0;
        t_bdt_mem            <= 1'b0;
        t_swp_mem            <= 1'b0;
        bdt_list_mem         <= 16'd0;
        bdt_load_mem         <= 1'b0;
        bdt_s_mem            <= 1'b0;
        bdt_wb_mem           <= 1'b0;
        addr_pre_idx_bdt_mem <= 1'b0;
        addr_up_bdt_mem      <= 1'b0;
        swap_byte_mem        <= 1'b0;
        base_reg_mem         <= 4'd0;
        base_value_mem       <= {`DATA_WIDTH{1'b0}};
        swp_rd_mem           <= 4'd0;
        swp_rm_mem           <= 4'd0;
        tid_mem              <= 2'd0;
        valid_mem            <= 1'b0;
    end
    else if (!stall_all) begin
        alu_result_mem       <= alu_result_ex2;
        mem_addr_mem         <= mem_addr_ex2;
        store_data_mem       <= store_data_ex2;
        mem_read_mem         <= mem_read_ex2;
        mem_write_mem        <= mem_write_ex2;
        mem_size_mem         <= mem_size_ex2;
        mem_signed_mem       <= mem_signed_ex2;
        wb_sel_mem           <= wb_sel_ex2;
        wr_addr1_mem         <= wr_addr1_ex2;
        wr_addr2_mem         <= wr_addr2_ex2;
        wr_en1_mem           <= wr_en1_ex2;
        wr_en2_mem           <= wr_en2_ex2;
        pc_plus4_mem         <= pc_plus4_ex2;
        is_multi_cycle_mem   <= is_multi_cycle_ex2;
        t_bdt_mem            <= t_bdt_ex2;
        t_swp_mem            <= t_swp_ex2;
        bdt_list_mem         <= bdt_list_ex2;
        bdt_load_mem         <= bdt_load_ex2;
        bdt_s_mem            <= bdt_s_ex2;
        bdt_wb_mem           <= bdt_wb_ex2;
        addr_pre_idx_bdt_mem <= addr_pre_idx_bdt_ex2;
        addr_up_bdt_mem      <= addr_up_bdt_ex2;
        swap_byte_mem        <= swap_byte_ex2;
        base_reg_mem         <= base_reg_ex2;
        base_value_mem       <= rn_val_ex2;
        swp_rd_mem           <= wr_addr1_ex2;
        swp_rm_mem           <= rm_addr_ex2;
        tid_mem              <= tid_ex2;
        valid_mem            <= valid_ex2;
    end
end

/* ================================================================
   MEM — DATA MEMORY ACCESS + BDTU
   ================================================================ */
wire [`DATA_WIDTH-1:0] bdtu_mem_addr, bdtu_mem_wdata;
wire bdtu_mem_rd, bdtu_mem_wr;
wire [1:0] bdtu_mem_size;

bdtu u_bdtu (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (is_multi_cycle_mem),
    .op_bdt     (t_bdt_mem),
    .op_swp     (t_swp_mem),
    .reg_list   (bdt_list_mem),
    .bdt_load   (bdt_load_mem),
    .bdt_wb     (bdt_wb_mem),
    .pre_index  (addr_pre_idx_bdt_mem),
    .up_down    (addr_up_bdt_mem),
    .bdt_s      (bdt_s_mem),
    .swap_byte  (swap_byte_mem),
    .swp_rd     (swp_rd_mem),
    .swp_rm     (swp_rm_mem),
    .base_reg   (base_reg_mem),
    .base_value (base_value_mem),
    .rf_rd_addr (bdtu_rf_rd_addr),
    .rf_rd_data (bdtu_rf_rd_data),
    .wr_addr1   (bdtu_wr_addr1),
    .wr_data1   (bdtu_wr_data1),
    .wr_en1     (bdtu_wr_en1),
    .wr_addr2   (bdtu_wr_addr2),
    .wr_data2   (bdtu_wr_data2),
    .wr_en2     (bdtu_wr_en2),
    .mem_addr   (bdtu_mem_addr),
    .mem_wdata  (bdtu_mem_wdata),
    .mem_rd     (bdtu_mem_rd),
    .mem_wr     (bdtu_mem_wr),
    .mem_size   (bdtu_mem_size),
    .mem_rdata  (d_mem_data_i),
    .busy       (bdtu_busy)
);

assign d_mem_addr_o = bdtu_busy ? bdtu_mem_addr  : mem_addr_mem;
assign d_mem_data_o = bdtu_busy ? bdtu_mem_wdata : store_data_mem;
assign d_mem_wen_o  = bdtu_busy ? bdtu_mem_wr    : mem_write_mem;
assign d_mem_size_o = bdtu_busy ? bdtu_mem_size  : mem_size_mem;

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

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_wb <= {`DATA_WIDTH{1'b0}};
        pc_plus4_wb   <= {`PC_WIDTH{1'b0}};
        wb_sel_wb     <= 3'd0;
        wr_addr1_wb   <= 4'd0;
        wr_addr2_wb   <= 4'd0;
        wr_en1_wb     <= 1'b0;
        wr_en2_wb     <= 1'b0;
        mem_size_wb   <= 2'd0;
        mem_signed_wb <= 1'b0;
        tid_wb        <= 2'd0;
        valid_wb      <= 1'b0;
    end
    else begin
        alu_result_wb <= alu_result_mem;
        pc_plus4_wb   <= pc_plus4_mem;
        wb_sel_wb     <= wb_sel_mem;
        wr_addr1_wb   <= wr_addr1_mem;
        wr_addr2_wb   <= wr_addr2_mem;
        wr_en1_wb     <= wr_en1_mem;
        wr_en2_wb     <= wr_en2_mem;
        mem_size_wb   <= mem_size_mem;
        mem_signed_wb <= mem_signed_mem;
        tid_wb        <= tid_mem;
        valid_wb      <= valid_mem;
    end
end

/* ================================================================
   WB — WRITE-BACK
   ================================================================ */

/* Load data sign/zero extension */
reg [`DATA_WIDTH-1:0] load_data_wb;

always @(*) begin
    case (mem_size_wb)
        2'b00: load_data_wb = mem_signed_wb
                     ? {{(`DATA_WIDTH-8){d_mem_data_i[7]}},  d_mem_data_i[7:0]}
                     : {{(`DATA_WIDTH-8){1'b0}},             d_mem_data_i[7:0]};
        2'b01: load_data_wb = mem_signed_wb
                     ? {{(`DATA_WIDTH-16){d_mem_data_i[15]}}, d_mem_data_i[15:0]}
                     : {{(`DATA_WIDTH-16){1'b0}},             d_mem_data_i[15:0]};
        default: load_data_wb = d_mem_data_i;
    endcase
end

/* Port-1 mux */
wire [3:0] cpsr_flags_wb = cpsr_flags[tid_wb];

reg [`DATA_WIDTH-1:0] wb_data1;
always @(*) begin
    case (wb_sel_wb)
        `WB_ALU:  wb_data1 = alu_result_wb;
        `WB_MEM:  wb_data1 = load_data_wb;
        `WB_LINK: wb_data1 = pc_plus4_wb;
        `WB_PSR:  wb_data1 = {cpsr_flags_wb, {(`DATA_WIDTH-4){1'b0}}};
        default:  wb_data1 = alu_result_wb;
    endcase
end

/* Port-2: base writeback */
wire [`DATA_WIDTH-1:0] wb_data2 = alu_result_wb;

/* Route to register file write ports */
assign wb_wr_addr1 = wr_addr1_wb;
assign wb_wr_data1 = wb_data1;
assign wb_wr_en1   = wr_en1_wb && valid_wb;

assign wb_wr_addr2 = wr_addr2_wb;
assign wb_wr_data2 = wb_data2;
assign wb_wr_en2   = wr_en2_wb && valid_wb;

/* ================================================================
   STALL LOGIC
   ================================================================ */
assign stall_all = bdtu_busy;

endmodule

`endif // CPU_MT_V