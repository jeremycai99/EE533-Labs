/* file: cpu.v
 * Single thread 9-stage pipeline Arm CPU module
 * Pipeline: IF1 → IF2 → ID → EX1 → EX2 → EX3 → EX4 → MEM → WB
 *
 * Version: 2.5d
 * Revision history:
 *    - 2.0: Split EX into EX1/EX2.
 *    - 2.1: HDU module, mc_ex2_hazard.
 *    - 2.2: (BROKEN) Gated MEM/WB load_data_wb stale read.
 *    - 2.3: load_data_latch fix. wb_committed.
 *    - 2.4: CU v1.1 barrel shifter fix, BDTU v1.5.
 *    - 2.5: 8-stage pipeline for sync-read BRAM + NetFPGA timing.
 *
 *      IF1 / IF2 split:
 *        IF1 presents PC → IMEM.  BRAM latches address at posedge.
 *        IF2 reads i_mem_data_i combinationally (valid 1 cycle
 *        after address presented).  Same pattern as cpu_mt.v.
 *
 *      DMEM sync-read fix:
 *        MEM presents address → BRAM latches at posedge.
 *        WB reads d_mem_data_i combinationally (valid 1 cycle later).
 *        load_data_latch only used during BDTU stalls (wb_committed).
 *
 *      Forwarding: MEM (EX4/MEM reg) + WB (MEM/WB reg) + BDTU only.
 *        NO forwarding from EX1, EX2, EX3, or EX4.
 *        HDU stalls for all EX-stage hazards (3 cycles for EX2→EX1,
 *        2 cycles for EX3→EX1, 1 cycle for EX4→EX1,
 *        1 cycle for MEM-load→EX1).
 *        Branch penalty: 6 cycles (resolved in EX4).
 *
 *    - 2.5a: CRITICAL FIX — register instruction in IF2/ID pipe reg.
 *            (Feb. 24, 2026)
 *
 *    - 2.5b: CRITICAL FIX — post-BDTU pipeline flush.
 *            On BDTU completion (busy falls), flush IF1-EX4,
 *            redirect PC to BDT's PC+4.  Cost: 6 refill cycles.
 *            (Feb. 24, 2026)
 *
 *    - 2.5c: TIMING FIX — move barrel shifter from EX1 to EX2.
 *            (Feb. 24, 2026)
 *
 *    - 2.5d: TIMING FIX — 9-stage pipeline, split EX2 into EX2+EX3.
 *            Critical path (14.8 ns, -6.8 ns slack):
 *              bs_din_ex2 → barrel_shifter → ALU → Z_flag → FF
 *            Z flag recomputed in EX4 as (alu_result_ex4 == 0)
 *            to avoid the 8.5 ns ALU+Z path in EX3.
 *            HDU v2.1: added EX3 and EX4 hazard classes.
 *            Branch penalty: 6 cycles (resolved in EX4).
 *            (Feb. 24, 2026)
 */

`ifndef CPU_V
`define CPU_V

`include "define.v"
`include "pc.v"
`include "regfile.v"
`include "hdu.v"
`include "fu.v"
`include "cu.v"
`include "alu.v"
`include "cond_eval.v"
`include "bdtu.v"
`include "barrel_shifter.v"

module cpu (
    input wire clk,
    input wire rst_n,
    input wire [`INSTR_WIDTH-1:0] i_mem_data_i,
    output wire [`PC_WIDTH-1:0] i_mem_addr_o,
    input wire [`DATA_WIDTH-1:0] d_mem_data_i,
    output wire [`CPU_DMEM_ADDR_WIDTH-1:0] d_mem_addr_o,
    output wire [`DATA_WIDTH-1:0] d_mem_data_o,
    output wire d_mem_wen_o,
    output wire [1:0] d_mem_size_o,
    output wire cpu_done
);

/*=========================================================
 * ALL WIRE / REG DECLARATIONS
 *=========================================================*/

// ── Pipeline control ─────────────────────────────────────
wire stall_if1, stall_if2, stall_id, stall_ex1;
wire stall_ex2, stall_ex3, stall_ex4, stall_mem;
wire flush_if1if2, flush_if2id, flush_idex1;
wire flush_ex1ex2, flush_ex2ex3, flush_ex3ex4;

// ── IF1 ──────────────────────────────────────────────────
wire [`PC_WIDTH-1:0] pc_if1;
wire [`PC_WIDTH-1:0] pc_next_if1;
wire [`PC_WIDTH-1:0] pc_plus4_if1;
wire pc_en;

wire branch_taken_ex4;
wire [`PC_WIDTH-1:0] branch_target_ex4_wire;

// ── IF1/IF2 ──────────────────────────────────────────────
reg [`PC_WIDTH-1:0] pc_plus4_if2;
reg if1if2_valid;

// ── IF2 / instruction hold ───────────────────────────────
reg [`INSTR_WIDTH-1:0] instr_held;
reg held_valid;
wire [`INSTR_WIDTH-1:0] instr_id;

// ── IF2/ID ───────────────────────────────────────────────
reg [`INSTR_WIDTH-1:0] instr_reg_id;
reg [`PC_WIDTH-1:0] pc_plus4_id;
reg if2id_valid;

// ── ID ───────────────────────────────────────────────────
reg [3:0] cpsr_flags;

wire t_dp_reg, t_dp_imm, t_mul, t_mull, t_swp, t_bx;
wire t_hdt_rego, t_hdt_immo, t_sdt_rego, t_sdt_immo;
wire t_bdt, t_br, t_mrs, t_msr_reg, t_msr_imm, t_swi, t_undef;

wire [3:0] rn_addr_id, rd_addr_id, rs_addr_id, rm_addr_id;
wire [3:0] wr_addr1_id, wr_addr2_id;
wire wr_en1_id, wr_en2_id;

wire [3:0] alu_op_id;
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

wire mul_en_id, mul_long_id, mul_signed_id, mul_accumulate_id;

wire psr_rd_id, psr_wr_id, psr_field_sel_id;
wire [3:0] psr_mask_id;

wire [15:0] bdt_list_id;
wire bdt_load_id, bdt_s_id, bdt_wb_id;

wire swap_byte_id, swi_en_id;
wire use_rn_id, use_rd_id, use_rs_id, use_rm_id;
wire is_multi_cycle_id;

wire [3:0] bdtu_rf_rd_addr;
wire [3:0] r3addr_mux;

wire [`DATA_WIDTH-1:0] rn_data_id, rm_data_id, r3_data_id, r4_data_id;

wire [3:0] wb_wr_addr1, wb_wr_addr2;
wire [`DATA_WIDTH-1:0] wb_wr_data1, wb_wr_data2;
wire wb_wr_en1, wb_wr_en2;

wire [3:0] bdtu_wr_addr1, bdtu_wr_addr2;
wire [`DATA_WIDTH-1:0] bdtu_wr_data1, bdtu_wr_data2;
wire bdtu_wr_en1, bdtu_wr_en2;

wire bdtu_has_write;
wire rf_wr_en;
wire [3:0] rf_wr_addr1;
wire [`DATA_WIDTH-1:0] rf_wr_data1;
wire [3:0] rf_wr_addr2;
wire [`DATA_WIDTH-1:0] rf_wr_data2;

wire [`DATA_WIDTH-1:0] rn_data_pc_adj;
wire [`DATA_WIDTH-1:0] rm_data_pc_adj;

wire bdtu_busy;

// ── ID/EX1 ───────────────────────────────────────────────
reg [3:0]  alu_op_ex1;
reg        alu_src_b_ex1;
reg        cpsr_wen_ex1;
reg [1:0]  shift_type_ex1;
reg [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount_ex1;
reg        shift_src_ex1;
reg [`DATA_WIDTH-1:0] imm32_ex1;
reg        mem_read_ex1, mem_write_ex1;
reg [1:0]  mem_size_ex1;
reg        mem_signed_ex1;
reg        addr_pre_idx_ex1, addr_up_ex1, addr_wb_ex1;
reg [2:0]  wb_sel_ex1;
reg [3:0]  wr_addr1_ex1, wr_addr2_ex1;
reg        wr_en1_ex1, wr_en2_ex1;
reg        branch_en_ex1, branch_link_ex1, branch_exchange_ex1;
reg        use_rn_ex1, use_rm_ex1, use_rs_ex1, use_rd_ex1;
reg [3:0]  rn_addr_ex1, rm_addr_ex1, rs_addr_ex1, rd_addr_ex1;
reg [`DATA_WIDTH-1:0] rn_data_ex1, rm_data_ex1;
reg [`DATA_WIDTH-1:0] rs_data_ex1;
reg [`DATA_WIDTH-1:0] rd_data_ex1;
reg [`PC_WIDTH-1:0]   pc_plus4_ex1;
reg        is_multi_cycle_ex1;
reg        t_bdt_ex1, t_swp_ex1;
reg [15:0] bdt_list_ex1;
reg        bdt_load_ex1, bdt_s_ex1, bdt_wb_ex1;
reg        addr_pre_idx_bdt_ex1, addr_up_bdt_ex1;
reg        swap_byte_ex1;
reg [3:0]  base_reg_ex1;
reg        psr_wr_ex1;
reg [3:0]  psr_mask_ex1;
reg        psr_field_sel_ex1;
reg [3:0]  cond_code_ex1;
reg        valid_ex1;

// ── EX1 wires ────────────────────────────────────────────
wire [3:0] exmem_wr_addr1;
wire       exmem_wr_en1;
wire       exmem_is_load;

wire [3:0] memwb_wr_addr1;
wire       memwb_wr_en1;
wire [3:0] memwb_wr_addr2;
wire       memwb_wr_en2;

wire [2:0] fwd_a, fwd_b, fwd_s, fwd_d;

wire [`DATA_WIDTH-1:0] exmem_alu_result;
wire [`DATA_WIDTH-1:0] exmem_wb_data2;
wire [`DATA_WIDTH-1:0] wb_result_data;
wire [`DATA_WIDTH-1:0] wb_data2;

wire [`DATA_WIDTH-1:0] rn_fwd;
wire [`DATA_WIDTH-1:0] rm_fwd;
wire [`DATA_WIDTH-1:0] rs_fwd;
wire [`DATA_WIDTH-1:0] rd_store_fwd;

wire [`SHIFT_AMOUNT_WIDTH-1:0] actual_shamt;
wire [`DATA_WIDTH-1:0] bs_din;
wire [`DATA_WIDTH-1:0] bs_dout;
wire shifter_cout;

wire [`PC_WIDTH-1:0] branch_target_br_ex1;
wire [`PC_WIDTH-1:0] branch_target_bx_ex1;
wire [`PC_WIDTH-1:0] branch_target_ex1;

// ── EX1/EX2 ─────────────────────────────────────────────
reg [`DATA_WIDTH-1:0] rn_fwd_ex2;
reg [`DATA_WIDTH-1:0] bs_din_ex2;
reg [`SHIFT_AMOUNT_WIDTH-1:0] bs_shamt_ex2;
reg [1:0]  bs_shift_type_ex2;
reg        bs_imm_shift_ex2;
reg        bs_cin_ex2;
reg [3:0]  alu_op_ex2;
reg        cpsr_wen_ex2;
reg        mem_read_ex2, mem_write_ex2;
reg [1:0]  mem_size_ex2;
reg        mem_signed_ex2;
reg        addr_pre_idx_ex2;
reg [2:0]  wb_sel_ex2;
reg [3:0]  wr_addr1_ex2, wr_addr2_ex2;
reg        wr_en1_ex2, wr_en2_ex2;
reg        branch_en_ex2;
reg        branch_link_ex2;
reg [`PC_WIDTH-1:0]   branch_target_ex2;
reg [`DATA_WIDTH-1:0] store_data_ex2;
reg [`DATA_WIDTH-1:0] rn_fwd_for_addr_ex2;
reg [`PC_WIDTH-1:0]   pc_plus4_ex2;
reg [3:0]  cond_code_ex2;
reg        valid_ex2;
reg        is_multi_cycle_ex2;
reg        t_bdt_ex2, t_swp_ex2;
reg [15:0] bdt_list_ex2;
reg        bdt_load_ex2, bdt_s_ex2, bdt_wb_ex2;
reg        addr_pre_idx_bdt_ex2, addr_up_bdt_ex2;
reg        swap_byte_ex2;
reg [3:0]  base_reg_ex2;
reg [`DATA_WIDTH-1:0] base_value_ex2;
reg [3:0]  rd_addr_ex2, rm_addr_ex2;
reg        psr_wr_ex2;
reg [3:0]  psr_mask_ex2;
reg        psr_field_sel_ex2;

// ── EX2/EX3 (NEW in v2.5d) ─────────────────────────────
reg [`DATA_WIDTH-1:0] bs_dout_ex3;
reg        shifter_cout_ex3;
reg [`DATA_WIDTH-1:0] rn_fwd_ex3;
reg [3:0]  alu_op_ex3;
reg        cpsr_wen_ex3;
reg        mem_read_ex3, mem_write_ex3;
reg [1:0]  mem_size_ex3;
reg        mem_signed_ex3;
reg        addr_pre_idx_ex3;
reg [2:0]  wb_sel_ex3;
reg [3:0]  wr_addr1_ex3, wr_addr2_ex3;
reg        wr_en1_ex3, wr_en2_ex3;
reg        branch_en_ex3;
reg        branch_link_ex3;
reg [`PC_WIDTH-1:0]   branch_target_ex3;
reg [`DATA_WIDTH-1:0] store_data_ex3;
reg [`DATA_WIDTH-1:0] rn_fwd_for_addr_ex3;
reg [`PC_WIDTH-1:0]   pc_plus4_ex3;
reg [3:0]  cond_code_ex3;
reg        valid_ex3;
reg        is_multi_cycle_ex3;
reg        t_bdt_ex3, t_swp_ex3;
reg [15:0] bdt_list_ex3;
reg        bdt_load_ex3, bdt_s_ex3, bdt_wb_ex3;
reg        addr_pre_idx_bdt_ex3, addr_up_bdt_ex3;
reg        swap_byte_ex3;
reg [3:0]  base_reg_ex3;
reg [`DATA_WIDTH-1:0] base_value_ex3;
reg [3:0]  rd_addr_ex3, rm_addr_ex3;
reg        psr_wr_ex3;
reg [3:0]  psr_mask_ex3;
reg        psr_field_sel_ex3;

// ── EX3 (ALU) wires ────────────────────────────────────
wire [`DATA_WIDTH-1:0] alu_result_ex3;
wire [3:0]             alu_flags_ex3_w;

// ── EX3/EX4 (was EX2/EX3 in v2.5c) ────────────────────
reg [`DATA_WIDTH-1:0] alu_result_ex4;
reg [3:0]  alu_flags_ex4;          // Z bit unreliable — recomputed in EX4
reg        shifter_cout_ex4;
reg        cpsr_wen_ex4;
reg        mem_read_ex4, mem_write_ex4;
reg [1:0]  mem_size_ex4;
reg        mem_signed_ex4;
reg        addr_pre_idx_ex4;
reg [2:0]  wb_sel_ex4;
reg [3:0]  wr_addr1_ex4, wr_addr2_ex4;
reg        wr_en1_ex4, wr_en2_ex4;
reg        branch_en_ex4;
reg        branch_link_ex4;
reg [`PC_WIDTH-1:0]   branch_target_ex4_r;
reg [`DATA_WIDTH-1:0] store_data_ex4;
reg [`DATA_WIDTH-1:0] rn_fwd_for_addr_ex4;
reg [`PC_WIDTH-1:0]   pc_plus4_ex4;
reg [3:0]  cond_code_ex4;
reg        valid_ex4;
reg        is_multi_cycle_ex4;
reg        t_bdt_ex4, t_swp_ex4;
reg [15:0] bdt_list_ex4;
reg        bdt_load_ex4, bdt_s_ex4, bdt_wb_ex4;
reg        addr_pre_idx_bdt_ex4, addr_up_bdt_ex4;
reg        swap_byte_ex4;
reg [3:0]  base_reg_ex4;
reg [`DATA_WIDTH-1:0] base_value_ex4;
reg [3:0]  rd_addr_ex4, rm_addr_ex4;
reg        psr_wr_ex4;
reg [3:0]  psr_mask_ex4;
reg        psr_field_sel_ex4;

// ── EX4 (cond eval, branch, Z recompute, gating) ───────
wire cond_met_raw_ex4;
wire cond_met_ex4;
wire [`CPU_DMEM_ADDR_WIDTH-1:0] mem_addr_ex4;
wire z_flag_recomputed_ex4;
wire [3:0] alu_flags_final_ex4;

wire wr_en1_gated_ex4;
wire wr_en2_gated_ex4;
wire mem_read_gated_ex4;
wire mem_write_gated_ex4;
wire cpsr_wen_gated_ex4;
wire is_multi_cycle_gated_ex4;

// ── EX4/MEM (was EX3/MEM in v2.5c) ─────────────────────
reg [`DATA_WIDTH-1:0]          alu_result_mem;
reg [`CPU_DMEM_ADDR_WIDTH-1:0] mem_addr_mem;
reg [`DATA_WIDTH-1:0]          store_data_mem;
reg        mem_read_mem, mem_write_mem;
reg [1:0]  mem_size_mem;
reg        mem_signed_mem;
reg [2:0]  wb_sel_mem;
reg [3:0]  wr_addr1_mem, wr_addr2_mem;
reg        wr_en1_mem, wr_en2_mem;
reg [`PC_WIDTH-1:0] pc_plus4_mem;
reg        is_multi_cycle_mem;
reg        t_bdt_mem, t_swp_mem;
reg [15:0] bdt_list_mem;
reg        bdt_load_mem, bdt_s_mem, bdt_wb_mem;
reg        addr_pre_idx_bdt_mem, addr_up_bdt_mem;
reg        swap_byte_mem;
reg [3:0]  base_reg_mem;
reg [`DATA_WIDTH-1:0] base_value_mem;
reg [3:0]  swp_rd_mem, swp_rm_mem;

// ── MEM (BDTU, DMEM mux) ────────────────────────────────
wire [`CPU_DMEM_ADDR_WIDTH-1:0] bdtu_mem_addr;
wire [`DATA_WIDTH-1:0]          bdtu_mem_wdata;
wire       bdtu_mem_rd, bdtu_mem_wr;
wire [1:0] bdtu_mem_size;
wire [`DATA_WIDTH-1:0] bdtu_rf_rd_data;

// ── MEM/WB ───────────────────────────────────────────────
reg [`DATA_WIDTH-1:0] alu_result_wb;
reg [`PC_WIDTH-1:0]   pc_plus4_wb;
reg [2:0]  wb_sel_wb;
reg [3:0]  wr_addr1_wb, wr_addr2_wb;
reg        wr_en1_wb, wr_en2_wb;
reg [1:0]  mem_size_wb;
reg        mem_signed_wb;
reg [`DATA_WIDTH-1:0] load_data_latch;
reg        wb_committed;

// ── WB ───────────────────────────────────────────────────
wire [`DATA_WIDTH-1:0] load_data_src;
reg  [`DATA_WIDTH-1:0] load_data_wb;
reg  [`DATA_WIDTH-1:0] wb_data1;


/*=========================================================
 * FORWARDING MUX FUNCTION
 *=========================================================*/

function [`DATA_WIDTH-1:0] fwd_mux;
    input [2:0] sel;
    input [`DATA_WIDTH-1:0] reg_val;
    input [`DATA_WIDTH-1:0] exmem_p1;
    input [`DATA_WIDTH-1:0] exmem_p2;
    input [`DATA_WIDTH-1:0] memwb_p1;
    input [`DATA_WIDTH-1:0] memwb_p2;
    input [`DATA_WIDTH-1:0] bdtu_p1;
    input [`DATA_WIDTH-1:0] bdtu_p2;
    begin
        case (sel)
            `FWD_NONE:     fwd_mux = reg_val;
            `FWD_EXMEM:    fwd_mux = exmem_p1;
            `FWD_EXMEM_P2: fwd_mux = exmem_p2;
            `FWD_MEMWB:    fwd_mux = memwb_p1;
            `FWD_MEMWB_P2: fwd_mux = memwb_p2;
            `FWD_BDTU_P1:  fwd_mux = bdtu_p1;
            `FWD_BDTU_P2:  fwd_mux = bdtu_p2;
            default:       fwd_mux = reg_val;
        endcase
    end
endfunction


/*=========================================================
 * HAZARD DETECTION UNIT (9-stage, HDU v2.1)
 *=========================================================*/

hdu u_hdu (
    // EX2 destinations (from EX1/EX2 reg)
    .ex1ex2_wd1            (wr_addr1_ex2),
    .ex1ex2_we1            (wr_en1_ex2),
    .ex1ex2_wd2            (wr_addr2_ex2),
    .ex1ex2_we2            (wr_en2_ex2),

    // EX3 destinations (from EX2/EX3 reg) — NEW in v2.5d
    .ex2ex3_wd1            (wr_addr1_ex3),
    .ex2ex3_we1            (wr_en1_ex3),
    .ex2ex3_wd2            (wr_addr2_ex3),
    .ex2ex3_we2            (wr_en2_ex3),

    // EX4 destinations (from EX3/EX4 reg)
    .ex3ex4_wd1            (wr_addr1_ex4),
    .ex3ex4_we1            (wr_en1_ex4),
    .ex3ex4_wd2            (wr_addr2_ex4),
    .ex3ex4_we2            (wr_en2_ex4),
    .ex3ex4_is_multi_cycle (is_multi_cycle_ex4),
    .ex3ex4_valid          (valid_ex4),

    // MEM load info (from EX4/MEM reg)
    .ex4mem_is_load        (mem_read_mem),
    .ex4mem_wd1            (wr_addr1_mem),
    .ex4mem_we1            (wr_en1_mem),

    // EX1 sources
    .ex1_rn                (rn_addr_ex1),
    .ex1_rm                (rm_addr_ex1),
    .ex1_rs                (rs_addr_ex1),
    .ex1_rd_store          (rd_addr_ex1),
    .ex1_use_rn            (use_rn_ex1),
    .ex1_use_rm            (use_rm_ex1),
    .ex1_use_rs            (use_rs_ex1),
    .ex1_use_rd_st         (use_rd_ex1),

    // Branch (resolved in EX4), BDTU
    .branch_taken          (branch_taken_ex4),
    .bdtu_busy             (bdtu_busy),

    // Stall outputs
    .stall_if1             (stall_if1),
    .stall_if2             (stall_if2),
    .stall_id              (stall_id),
    .stall_ex1             (stall_ex1),
    .stall_ex2             (stall_ex2),
    .stall_ex3             (stall_ex3),
    .stall_ex4             (stall_ex4),
    .stall_mem             (stall_mem),

    // Flush outputs
    .flush_if1if2          (flush_if1if2),
    .flush_if2id           (flush_if2id),
    .flush_idex1           (flush_idex1),
    .flush_ex1ex2          (flush_ex1ex2),
    .flush_ex2ex3          (flush_ex2ex3),
    .flush_ex3ex4          (flush_ex3ex4)
);


/*********************************************************
 ************ IF1 Stage ************
 *********************************************************/

wire bdtu_done_flush;

assign pc_plus4_if1 = pc_if1 + 32'd4;
assign pc_next_if1  = bdtu_done_flush  ? pc_plus4_mem
                    : branch_taken_ex4 ? branch_target_ex4_wire
                    : pc_plus4_if1;
assign pc_en        = ~stall_if1 | bdtu_done_flush;

pc u_pc (
    .clk    (clk),
    .rst_n  (rst_n),
    .pc_in  (pc_next_if1),
    .en     (pc_en),
    .pc_out (pc_if1)
);

assign i_mem_addr_o = pc_if1;
assign cpu_done     = (pc_if1 == `CPU_DONE_PC);


/*********************************************************
 ******** IF1/IF2 Pipeline Register ********
 *********************************************************/

wire flush_if1if2_f, flush_if2id_f, flush_idex1_f;
wire flush_ex1ex2_f, flush_ex2ex3_f, flush_ex3ex4_f;
wire flush_ex4mem;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pc_plus4_if2 <= `PC_WIDTH'd0;
        if1if2_valid <= 1'b0;
    end else if (flush_if1if2_f) begin
        if1if2_valid <= 1'b0;
    end else if (!stall_if2) begin
        pc_plus4_if2 <= pc_plus4_if1;
        if1if2_valid <= 1'b1;
    end
end


/*********************************************************
 ************ IF2 Stage ************
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        held_valid <= 1'b0;
    else if (flush_if2id_f)
        held_valid <= 1'b0;
    else if (stall_id && !held_valid) begin
        instr_held <= i_mem_data_i;
        held_valid <= 1'b1;
    end
    else if (!stall_id)
        held_valid <= 1'b0;
end

assign instr_id = held_valid ? instr_held : i_mem_data_i;


/*********************************************************
 ******** IF2/ID Pipeline Register ********
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        instr_reg_id <= {`INSTR_WIDTH{1'b0}};
        pc_plus4_id  <= `PC_WIDTH'd0;
        if2id_valid  <= 1'b0;
    end else if (flush_if2id_f) begin
        instr_reg_id <= {`INSTR_WIDTH{1'b0}};
        if2id_valid  <= 1'b0;
    end else if (!stall_id) begin
        instr_reg_id <= instr_id;
        pc_plus4_id  <= pc_plus4_if2;
        if2id_valid  <= if1if2_valid;
    end
end


/*********************************************************
 ************ ID Stage ************
 *********************************************************/

cu u_cu (
    .instr       (instr_reg_id),
    .cond_met    (if2id_valid),
    .t_dp_reg    (t_dp_reg),
    .t_dp_imm    (t_dp_imm),
    .t_mul       (t_mul),
    .t_mull      (t_mull),
    .t_swp       (t_swp),
    .t_bx        (t_bx),
    .t_hdt_rego  (t_hdt_rego),
    .t_hdt_immo  (t_hdt_immo),
    .t_sdt_rego  (t_sdt_rego),
    .t_sdt_immo  (t_sdt_immo),
    .t_bdt       (t_bdt),
    .t_br        (t_br),
    .t_mrs       (t_mrs),
    .t_msr_reg   (t_msr_reg),
    .t_msr_imm   (t_msr_imm),
    .t_swi       (t_swi),
    .t_undef     (t_undef),
    .rn_addr     (rn_addr_id),
    .rd_addr     (rd_addr_id),
    .rs_addr     (rs_addr_id),
    .rm_addr     (rm_addr_id),
    .wr_addr1    (wr_addr1_id),
    .wr_en1      (wr_en1_id),
    .wr_addr2    (wr_addr2_id),
    .wr_en2      (wr_en2_id),
    .alu_op      (alu_op_id),
    .alu_src_b   (alu_src_b_id),
    .cpsr_wen    (cpsr_wen_id),
    .shift_type  (shift_type_id),
    .shift_amount(shift_amount_id),
    .shift_src   (shift_src_id),
    .imm32       (imm32_id),
    .mem_read    (mem_read_id),
    .mem_write   (mem_write_id),
    .mem_size    (mem_size_id),
    .mem_signed  (mem_signed_id),
    .addr_pre_idx(addr_pre_idx_id),
    .addr_up     (addr_up_id),
    .addr_wb     (addr_wb_id),
    .wb_sel      (wb_sel_id),
    .branch_en   (branch_en_id),
    .branch_link (branch_link_id),
    .branch_exchange(branch_exchange_id),
    .mul_en      (mul_en_id),
    .mul_long    (mul_long_id),
    .mul_signed  (mul_signed_id),
    .mul_accumulate(mul_accumulate_id),
    .psr_rd      (psr_rd_id),
    .psr_wr      (psr_wr_id),
    .psr_field_sel(psr_field_sel_id),
    .psr_mask    (psr_mask_id),
    .bdt_list    (bdt_list_id),
    .bdt_load    (bdt_load_id),
    .bdt_s       (bdt_s_id),
    .bdt_wb      (bdt_wb_id),
    .swap_byte   (swap_byte_id),
    .swi_en      (swi_en_id),
    .use_rn      (use_rn_id),
    .use_rd      (use_rd_id),
    .use_rs      (use_rs_id),
    .use_rm      (use_rm_id),
    .is_multi_cycle(is_multi_cycle_id)
);

assign r3addr_mux = bdtu_busy ? bdtu_rf_rd_addr : rs_addr_id;

assign bdtu_has_write = bdtu_wr_en1 | bdtu_wr_en2;

assign rf_wr_en = bdtu_has_write ? 1'b1
                                 : (wb_wr_en1 | wb_wr_en2);

assign rf_wr_addr1 = bdtu_has_write
    ? (bdtu_wr_en1 ? bdtu_wr_addr1 : bdtu_wr_addr2)
    : (wb_wr_en1   ? wb_wr_addr1   : wb_wr_addr2);

assign rf_wr_data1 = bdtu_has_write
    ? (bdtu_wr_en1 ? bdtu_wr_data1 : bdtu_wr_data2)
    : (wb_wr_en1   ? wb_wr_data1   : wb_wr_data2);

assign rf_wr_addr2 = (bdtu_wr_en1 & bdtu_wr_en2)
    ? bdtu_wr_addr2
    : bdtu_has_write
        ? (wb_wr_en1 ? wb_wr_addr1 : rf_wr_addr1)
        : (wb_wr_en2 ? wb_wr_addr2 : rf_wr_addr1);

assign rf_wr_data2 = (bdtu_wr_en1 & bdtu_wr_en2)
    ? bdtu_wr_data2
    : bdtu_has_write
        ? (wb_wr_en1 ? wb_wr_data1 : rf_wr_data1)
        : (wb_wr_en2 ? wb_wr_data2 : rf_wr_data1);

regfile u_regfile (
    .clk      (clk),
    .r1addr   (rn_addr_id),
    .r2addr   (rm_addr_id),
    .r3addr   (r3addr_mux),
    .r4addr   (rd_addr_id),
    .wena     (rf_wr_en),
    .wr_addr1 (rf_wr_addr1),
    .wr_data1 (rf_wr_data1),
    .wr_addr2 (rf_wr_addr2),
    .wr_data2 (rf_wr_data2),
    .r1data   (rn_data_id),
    .r2data   (rm_data_id),
    .r3data   (r3_data_id),
    .r4data   (r4_data_id)
);

assign rn_data_pc_adj = (rn_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rn_data_id;
assign rm_data_pc_adj = (rm_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rm_data_id;


/*********************************************************
 ******** ID/EX1 Pipeline Register ********
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_op_ex1          <= 4'd0;
        alu_src_b_ex1       <= 1'b0;
        cpsr_wen_ex1        <= 1'b0;
        shift_type_ex1      <= 2'd0;
        shift_amount_ex1    <= 5'd0;
        shift_src_ex1       <= 1'b0;
        imm32_ex1           <= 32'd0;
        mem_read_ex1        <= 1'b0;
        mem_write_ex1       <= 1'b0;
        mem_size_ex1        <= 2'd0;
        mem_signed_ex1      <= 1'b0;
        addr_pre_idx_ex1    <= 1'b0;
        addr_up_ex1         <= 1'b0;
        addr_wb_ex1         <= 1'b0;
        wb_sel_ex1          <= 3'd0;
        wr_addr1_ex1        <= 4'd0;
        wr_addr2_ex1        <= 4'd0;
        wr_en1_ex1          <= 1'b0;
        wr_en2_ex1          <= 1'b0;
        branch_en_ex1       <= 1'b0;
        branch_link_ex1     <= 1'b0;
        branch_exchange_ex1 <= 1'b0;
        use_rn_ex1          <= 1'b0;
        use_rm_ex1          <= 1'b0;
        use_rs_ex1          <= 1'b0;
        use_rd_ex1          <= 1'b0;
        rn_addr_ex1         <= 4'd0;
        rm_addr_ex1         <= 4'd0;
        rs_addr_ex1         <= 4'd0;
        rd_addr_ex1         <= 4'd0;
        rn_data_ex1         <= 32'd0;
        rm_data_ex1         <= 32'd0;
        rs_data_ex1         <= 32'd0;
        rd_data_ex1         <= 32'd0;
        pc_plus4_ex1        <= 32'd0;
        is_multi_cycle_ex1  <= 1'b0;
        t_bdt_ex1           <= 1'b0;
        t_swp_ex1           <= 1'b0;
        bdt_list_ex1        <= 16'd0;
        bdt_load_ex1        <= 1'b0;
        bdt_s_ex1           <= 1'b0;
        bdt_wb_ex1          <= 1'b0;
        addr_pre_idx_bdt_ex1<= 1'b0;
        addr_up_bdt_ex1     <= 1'b0;
        swap_byte_ex1       <= 1'b0;
        base_reg_ex1        <= 4'd0;
        psr_wr_ex1          <= 1'b0;
        psr_mask_ex1        <= 4'd0;
        psr_field_sel_ex1   <= 1'b0;
        cond_code_ex1       <= 4'b1110;
        valid_ex1           <= 1'b0;
    end
    else if (flush_idex1_f) begin
        alu_op_ex1          <= 4'd0;
        alu_src_b_ex1       <= 1'b0;
        cpsr_wen_ex1        <= 1'b0;
        shift_type_ex1      <= 2'd0;
        shift_amount_ex1    <= 5'd0;
        shift_src_ex1       <= 1'b0;
        imm32_ex1           <= 32'd0;
        mem_read_ex1        <= 1'b0;
        mem_write_ex1       <= 1'b0;
        mem_size_ex1        <= 2'd0;
        mem_signed_ex1      <= 1'b0;
        addr_pre_idx_ex1    <= 1'b0;
        addr_up_ex1         <= 1'b0;
        addr_wb_ex1         <= 1'b0;
        wb_sel_ex1          <= 3'd0;
        wr_addr1_ex1        <= 4'd0;
        wr_addr2_ex1        <= 4'd0;
        wr_en1_ex1          <= 1'b0;
        wr_en2_ex1          <= 1'b0;
        branch_en_ex1       <= 1'b0;
        branch_link_ex1     <= 1'b0;
        branch_exchange_ex1 <= 1'b0;
        use_rn_ex1          <= 1'b0;
        use_rm_ex1          <= 1'b0;
        use_rs_ex1          <= 1'b0;
        use_rd_ex1          <= 1'b0;
        rn_addr_ex1         <= 4'd0;
        rm_addr_ex1         <= 4'd0;
        rs_addr_ex1         <= 4'd0;
        rd_addr_ex1         <= 4'd0;
        rn_data_ex1         <= 32'd0;
        rm_data_ex1         <= 32'd0;
        rs_data_ex1         <= 32'd0;
        rd_data_ex1         <= 32'd0;
        pc_plus4_ex1        <= 32'd0;
        is_multi_cycle_ex1  <= 1'b0;
        t_bdt_ex1           <= 1'b0;
        t_swp_ex1           <= 1'b0;
        bdt_list_ex1        <= 16'd0;
        bdt_load_ex1        <= 1'b0;
        bdt_s_ex1           <= 1'b0;
        bdt_wb_ex1          <= 1'b0;
        addr_pre_idx_bdt_ex1<= 1'b0;
        addr_up_bdt_ex1     <= 1'b0;
        swap_byte_ex1       <= 1'b0;
        base_reg_ex1        <= 4'd0;
        psr_wr_ex1          <= 1'b0;
        psr_mask_ex1        <= 4'd0;
        psr_field_sel_ex1   <= 1'b0;
        cond_code_ex1       <= 4'b1110;
        valid_ex1           <= 1'b0;
    end else if (!stall_ex1) begin
        alu_op_ex1          <= alu_op_id;
        alu_src_b_ex1       <= alu_src_b_id;
        cpsr_wen_ex1        <= cpsr_wen_id;
        shift_type_ex1      <= shift_type_id;
        shift_amount_ex1    <= shift_amount_id;
        shift_src_ex1       <= shift_src_id;
        imm32_ex1           <= imm32_id;
        mem_read_ex1        <= mem_read_id;
        mem_write_ex1       <= mem_write_id;
        mem_size_ex1        <= mem_size_id;
        mem_signed_ex1      <= mem_signed_id;
        addr_pre_idx_ex1    <= addr_pre_idx_id;
        addr_up_ex1         <= addr_up_id;
        addr_wb_ex1         <= addr_wb_id;
        wb_sel_ex1          <= wb_sel_id;
        wr_addr1_ex1        <= wr_addr1_id;
        wr_addr2_ex1        <= wr_addr2_id;
        wr_en1_ex1          <= wr_en1_id;
        wr_en2_ex1          <= wr_en2_id;
        branch_en_ex1       <= branch_en_id;
        branch_link_ex1     <= branch_link_id;
        branch_exchange_ex1 <= branch_exchange_id;
        use_rn_ex1          <= use_rn_id;
        use_rm_ex1          <= use_rm_id;
        use_rs_ex1          <= use_rs_id;
        use_rd_ex1          <= use_rd_id;
        rn_addr_ex1         <= rn_addr_id;
        rm_addr_ex1         <= rm_addr_id;
        rs_addr_ex1         <= rs_addr_id;
        rd_addr_ex1         <= rd_addr_id;
        rn_data_ex1         <= rn_data_pc_adj;
        rm_data_ex1         <= rm_data_pc_adj;
        rs_data_ex1         <= r3_data_id;
        rd_data_ex1         <= r4_data_id;
        pc_plus4_ex1        <= pc_plus4_id;
        is_multi_cycle_ex1  <= is_multi_cycle_id;
        t_bdt_ex1           <= t_bdt;
        t_swp_ex1           <= t_swp;
        bdt_list_ex1        <= bdt_list_id;
        bdt_load_ex1        <= bdt_load_id;
        bdt_s_ex1           <= bdt_s_id;
        bdt_wb_ex1          <= bdt_wb_id;
        addr_pre_idx_bdt_ex1<= addr_pre_idx_id;
        addr_up_bdt_ex1     <= addr_up_id;
        swap_byte_ex1       <= swap_byte_id;
        base_reg_ex1        <= rn_addr_id;
        psr_wr_ex1          <= psr_wr_id;
        psr_mask_ex1        <= psr_mask_id;
        psr_field_sel_ex1   <= psr_field_sel_id;
        cond_code_ex1       <= instr_reg_id[31:28];
        valid_ex1           <= if2id_valid;
    end else begin
        if (fwd_a != `FWD_NONE) rn_data_ex1 <= rn_fwd;
        if (fwd_b != `FWD_NONE) rm_data_ex1 <= rm_fwd;
        if (fwd_s != `FWD_NONE) rs_data_ex1 <= rs_fwd;
        if (fwd_d != `FWD_NONE) rd_data_ex1 <= rd_store_fwd;
    end
end


/*********************************************************
 ************ EX1 Stage ************
 *  Forwarding unit, branch target, BS input resolution.
 *********************************************************/

fu u_fu (
    .ex_rn          (rn_addr_ex1),
    .ex_rm          (rm_addr_ex1),
    .ex_rs          (rs_addr_ex1),
    .ex_rd_store    (rd_addr_ex1),
    .ex_use_rn      (use_rn_ex1),
    .ex_use_rm      (use_rm_ex1),
    .ex_use_rs      (use_rs_ex1),
    .ex_use_rd_st   (use_rd_ex1),
    .exmem_wd1      (exmem_wr_addr1),
    .exmem_we1      (exmem_wr_en1),
    .exmem_is_load  (exmem_is_load),
    .exmem_wd2      (wr_addr2_mem),
    .exmem_we2      (wr_en2_mem),
    .memwb_wd1      (memwb_wr_addr1),
    .memwb_we1      (memwb_wr_en1),
    .memwb_wd2      (memwb_wr_addr2),
    .memwb_we2      (memwb_wr_en2),
    .bdtu_wd1       (bdtu_wr_addr1),
    .bdtu_we1       (bdtu_wr_en1),
    .bdtu_wd2       (bdtu_wr_addr2),
    .bdtu_we2       (bdtu_wr_en2),
    .fwd_a          (fwd_a),
    .fwd_b          (fwd_b),
    .fwd_s          (fwd_s),
    .fwd_d          (fwd_d)
);

assign rn_fwd = fwd_mux(fwd_a, rn_data_ex1,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data, wb_data2,
    bdtu_wr_data1, bdtu_wr_data2);

assign rm_fwd = fwd_mux(fwd_b, rm_data_ex1,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data, wb_data2,
    bdtu_wr_data1, bdtu_wr_data2);

assign rs_fwd = fwd_mux(fwd_s, rs_data_ex1,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data, wb_data2,
    bdtu_wr_data1, bdtu_wr_data2);

assign rd_store_fwd = fwd_mux(fwd_d, rd_data_ex1,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data, wb_data2,
    bdtu_wr_data1, bdtu_wr_data2);

assign actual_shamt = shift_src_ex1 ? rs_fwd[`SHIFT_AMOUNT_WIDTH-1:0]
                                    : shift_amount_ex1;

assign bs_din = alu_src_b_ex1 ? imm32_ex1 : rm_fwd;

assign branch_target_br_ex1 = pc_plus4_ex1 + 32'd4 + imm32_ex1;
assign branch_target_bx_ex1 = rm_fwd;
assign branch_target_ex1    = branch_exchange_ex1 ? branch_target_bx_ex1
                                                  : branch_target_br_ex1;


/*********************************************************
 ******** EX1/EX2 Pipeline Register ********
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rn_fwd_ex2          <= 32'd0;
        bs_din_ex2          <= 32'd0;
        bs_shamt_ex2        <= 5'd0;
        bs_shift_type_ex2   <= 2'd0;
        bs_imm_shift_ex2    <= 1'b0;
        bs_cin_ex2          <= 1'b0;
        alu_op_ex2          <= 4'd0;
        cpsr_wen_ex2        <= 1'b0;
        mem_read_ex2        <= 1'b0;
        mem_write_ex2       <= 1'b0;
        mem_size_ex2        <= 2'd0;
        mem_signed_ex2      <= 1'b0;
        addr_pre_idx_ex2    <= 1'b0;
        wb_sel_ex2          <= 3'd0;
        wr_addr1_ex2        <= 4'd0;
        wr_addr2_ex2        <= 4'd0;
        wr_en1_ex2          <= 1'b0;
        wr_en2_ex2          <= 1'b0;
        branch_en_ex2       <= 1'b0;
        branch_link_ex2     <= 1'b0;
        branch_target_ex2   <= 32'd0;
        store_data_ex2      <= 32'd0;
        rn_fwd_for_addr_ex2 <= 32'd0;
        pc_plus4_ex2        <= 32'd0;
        cond_code_ex2       <= 4'b1110;
        valid_ex2           <= 1'b0;
        is_multi_cycle_ex2  <= 1'b0;
        t_bdt_ex2           <= 1'b0;
        t_swp_ex2           <= 1'b0;
        bdt_list_ex2        <= 16'd0;
        bdt_load_ex2        <= 1'b0;
        bdt_s_ex2           <= 1'b0;
        bdt_wb_ex2          <= 1'b0;
        addr_pre_idx_bdt_ex2<= 1'b0;
        addr_up_bdt_ex2     <= 1'b0;
        swap_byte_ex2       <= 1'b0;
        base_reg_ex2        <= 4'd0;
        base_value_ex2      <= 32'd0;
        rd_addr_ex2         <= 4'd0;
        rm_addr_ex2         <= 4'd0;
        psr_wr_ex2          <= 1'b0;
        psr_mask_ex2        <= 4'd0;
        psr_field_sel_ex2   <= 1'b0;
    end
    else if (flush_ex1ex2_f) begin
        rn_fwd_ex2          <= 32'd0;
        bs_din_ex2          <= 32'd0;
        bs_shamt_ex2        <= 5'd0;
        bs_shift_type_ex2   <= 2'd0;
        bs_imm_shift_ex2    <= 1'b0;
        bs_cin_ex2          <= 1'b0;
        alu_op_ex2          <= 4'd0;
        cpsr_wen_ex2        <= 1'b0;
        mem_read_ex2        <= 1'b0;
        mem_write_ex2       <= 1'b0;
        mem_size_ex2        <= 2'd0;
        mem_signed_ex2      <= 1'b0;
        addr_pre_idx_ex2    <= 1'b0;
        wb_sel_ex2          <= 3'd0;
        wr_addr1_ex2        <= 4'd0;
        wr_addr2_ex2        <= 4'd0;
        wr_en1_ex2          <= 1'b0;
        wr_en2_ex2          <= 1'b0;
        branch_en_ex2       <= 1'b0;
        branch_link_ex2     <= 1'b0;
        branch_target_ex2   <= 32'd0;
        store_data_ex2      <= 32'd0;
        rn_fwd_for_addr_ex2 <= 32'd0;
        pc_plus4_ex2        <= 32'd0;
        cond_code_ex2       <= 4'b1110;
        valid_ex2           <= 1'b0;
        is_multi_cycle_ex2  <= 1'b0;
        t_bdt_ex2           <= 1'b0;
        t_swp_ex2           <= 1'b0;
        bdt_list_ex2        <= 16'd0;
        bdt_load_ex2        <= 1'b0;
        bdt_s_ex2           <= 1'b0;
        bdt_wb_ex2          <= 1'b0;
        addr_pre_idx_bdt_ex2<= 1'b0;
        addr_up_bdt_ex2     <= 1'b0;
        swap_byte_ex2       <= 1'b0;
        base_reg_ex2        <= 4'd0;
        base_value_ex2      <= 32'd0;
        rd_addr_ex2         <= 4'd0;
        rm_addr_ex2         <= 4'd0;
        psr_wr_ex2          <= 1'b0;
        psr_mask_ex2        <= 4'd0;
        psr_field_sel_ex2   <= 1'b0;
    end else if (!stall_ex2) begin
        rn_fwd_ex2          <= rn_fwd;
        bs_din_ex2          <= bs_din;
        bs_shamt_ex2        <= actual_shamt;
        bs_shift_type_ex2   <= shift_type_ex1;
        bs_imm_shift_ex2    <= ~shift_src_ex1;
        bs_cin_ex2          <= cpsr_flags[`FLAG_C];
        alu_op_ex2          <= alu_op_ex1;
        cpsr_wen_ex2        <= cpsr_wen_ex1;
        mem_read_ex2        <= mem_read_ex1;
        mem_write_ex2       <= mem_write_ex1;
        mem_size_ex2        <= mem_size_ex1;
        mem_signed_ex2      <= mem_signed_ex1;
        addr_pre_idx_ex2    <= addr_pre_idx_ex1;
        wb_sel_ex2          <= wb_sel_ex1;
        wr_addr1_ex2        <= wr_addr1_ex1;
        wr_addr2_ex2        <= wr_addr2_ex1;
        wr_en1_ex2          <= wr_en1_ex1;
        wr_en2_ex2          <= wr_en2_ex1;
        branch_en_ex2       <= branch_en_ex1;
        branch_link_ex2     <= branch_link_ex1;
        branch_target_ex2   <= branch_target_ex1;
        store_data_ex2      <= rd_store_fwd;
        rn_fwd_for_addr_ex2 <= rn_fwd;
        pc_plus4_ex2        <= pc_plus4_ex1;
        cond_code_ex2       <= cond_code_ex1;
        valid_ex2           <= valid_ex1;
        is_multi_cycle_ex2  <= is_multi_cycle_ex1;
        t_bdt_ex2           <= t_bdt_ex1;
        t_swp_ex2           <= t_swp_ex1;
        bdt_list_ex2        <= bdt_list_ex1;
        bdt_load_ex2        <= bdt_load_ex1;
        bdt_s_ex2           <= bdt_s_ex1;
        bdt_wb_ex2          <= bdt_wb_ex1;
        addr_pre_idx_bdt_ex2<= addr_pre_idx_bdt_ex1;
        addr_up_bdt_ex2     <= addr_up_bdt_ex1;
        swap_byte_ex2       <= swap_byte_ex1;
        base_reg_ex2        <= base_reg_ex1;
        base_value_ex2      <= rn_fwd;
        rd_addr_ex2         <= rd_addr_ex1;
        rm_addr_ex2         <= rm_addr_ex1;
        psr_wr_ex2          <= psr_wr_ex1;
        psr_mask_ex2        <= psr_mask_ex1;
        psr_field_sel_ex2   <= psr_field_sel_ex1;
    end
end


/*********************************************************
 ************ EX2 Stage ************
 *  Barrel shifter ONLY.
 *
 *  *** v2.5d: ALU removed from EX2.  Only the barrel
 *  shifter runs here.  Its outputs (bs_dout, shifter_cout)
 *  are registered into the EX2/EX3 pipe reg.
 *
 *  Timing budget (EX2):
 *    barrel_shifter logic + routing  ≈ 6.3 ns  ✓
 *********************************************************/

barrel_shifter u_barrel_shifter (
    .din        (bs_din_ex2),
    .shamt      (bs_shamt_ex2),
    .shift_type (bs_shift_type_ex2),
    .is_imm_shift (bs_imm_shift_ex2),
    .cin        (bs_cin_ex2),
    .dout       (bs_dout),
    .cout       (shifter_cout)
);


/*********************************************************
 ******** EX2/EX3 Pipeline Register ********
 *  NEW in v2.5d: Registers barrel shifter outputs
 *  (bs_dout, shifter_cout) and operand_a (rn_fwd_ex2)
 *  for the ALU in EX3.  Also passes through all control
 *  signals.
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bs_dout_ex3         <= 32'd0;
        shifter_cout_ex3    <= 1'b0;
        rn_fwd_ex3          <= 32'd0;
        alu_op_ex3          <= 4'd0;
        cpsr_wen_ex3        <= 1'b0;
        mem_read_ex3        <= 1'b0;
        mem_write_ex3       <= 1'b0;
        mem_size_ex3        <= 2'd0;
        mem_signed_ex3      <= 1'b0;
        addr_pre_idx_ex3    <= 1'b0;
        wb_sel_ex3          <= 3'd0;
        wr_addr1_ex3        <= 4'd0;
        wr_addr2_ex3        <= 4'd0;
        wr_en1_ex3          <= 1'b0;
        wr_en2_ex3          <= 1'b0;
        branch_en_ex3       <= 1'b0;
        branch_link_ex3     <= 1'b0;
        branch_target_ex3   <= 32'd0;
        store_data_ex3      <= 32'd0;
        rn_fwd_for_addr_ex3 <= 32'd0;
        pc_plus4_ex3        <= 32'd0;
        cond_code_ex3       <= 4'b1110;
        valid_ex3           <= 1'b0;
        is_multi_cycle_ex3  <= 1'b0;
        t_bdt_ex3           <= 1'b0;
        t_swp_ex3           <= 1'b0;
        bdt_list_ex3        <= 16'd0;
        bdt_load_ex3        <= 1'b0;
        bdt_s_ex3           <= 1'b0;
        bdt_wb_ex3          <= 1'b0;
        addr_pre_idx_bdt_ex3<= 1'b0;
        addr_up_bdt_ex3     <= 1'b0;
        swap_byte_ex3       <= 1'b0;
        base_reg_ex3        <= 4'd0;
        base_value_ex3      <= 32'd0;
        rd_addr_ex3         <= 4'd0;
        rm_addr_ex3         <= 4'd0;
        psr_wr_ex3          <= 1'b0;
        psr_mask_ex3        <= 4'd0;
        psr_field_sel_ex3   <= 1'b0;
    end
    else if (flush_ex2ex3_f) begin
        bs_dout_ex3         <= 32'd0;
        shifter_cout_ex3    <= 1'b0;
        rn_fwd_ex3          <= 32'd0;
        alu_op_ex3          <= 4'd0;
        cpsr_wen_ex3        <= 1'b0;
        mem_read_ex3        <= 1'b0;
        mem_write_ex3       <= 1'b0;
        mem_size_ex3        <= 2'd0;
        mem_signed_ex3      <= 1'b0;
        addr_pre_idx_ex3    <= 1'b0;
        wb_sel_ex3          <= 3'd0;
        wr_addr1_ex3        <= 4'd0;
        wr_addr2_ex3        <= 4'd0;
        wr_en1_ex3          <= 1'b0;
        wr_en2_ex3          <= 1'b0;
        branch_en_ex3       <= 1'b0;
        branch_link_ex3     <= 1'b0;
        branch_target_ex3   <= 32'd0;
        store_data_ex3      <= 32'd0;
        rn_fwd_for_addr_ex3 <= 32'd0;
        pc_plus4_ex3        <= 32'd0;
        cond_code_ex3       <= 4'b1110;
        valid_ex3           <= 1'b0;
        is_multi_cycle_ex3  <= 1'b0;
        t_bdt_ex3           <= 1'b0;
        t_swp_ex3           <= 1'b0;
        bdt_list_ex3        <= 16'd0;
        bdt_load_ex3        <= 1'b0;
        bdt_s_ex3           <= 1'b0;
        bdt_wb_ex3          <= 1'b0;
        addr_pre_idx_bdt_ex3<= 1'b0;
        addr_up_bdt_ex3     <= 1'b0;
        swap_byte_ex3       <= 1'b0;
        base_reg_ex3        <= 4'd0;
        base_value_ex3      <= 32'd0;
        rd_addr_ex3         <= 4'd0;
        rm_addr_ex3         <= 4'd0;
        psr_wr_ex3          <= 1'b0;
        psr_mask_ex3        <= 4'd0;
        psr_field_sel_ex3   <= 1'b0;
    end else if (!stall_ex3) begin
        bs_dout_ex3         <= bs_dout;
        shifter_cout_ex3    <= shifter_cout;
        rn_fwd_ex3          <= rn_fwd_ex2;
        alu_op_ex3          <= alu_op_ex2;
        cpsr_wen_ex3        <= cpsr_wen_ex2;
        mem_read_ex3        <= mem_read_ex2;
        mem_write_ex3       <= mem_write_ex2;
        mem_size_ex3        <= mem_size_ex2;
        mem_signed_ex3      <= mem_signed_ex2;
        addr_pre_idx_ex3    <= addr_pre_idx_ex2;
        wb_sel_ex3          <= wb_sel_ex2;
        wr_addr1_ex3        <= wr_addr1_ex2;
        wr_addr2_ex3        <= wr_addr2_ex2;
        wr_en1_ex3          <= wr_en1_ex2;
        wr_en2_ex3          <= wr_en2_ex2;
        branch_en_ex3       <= branch_en_ex2;
        branch_link_ex3     <= branch_link_ex2;
        branch_target_ex3   <= branch_target_ex2;
        store_data_ex3      <= store_data_ex2;
        rn_fwd_for_addr_ex3 <= rn_fwd_for_addr_ex2;
        pc_plus4_ex3        <= pc_plus4_ex2;
        cond_code_ex3       <= cond_code_ex2;
        valid_ex3           <= valid_ex2;
        is_multi_cycle_ex3  <= is_multi_cycle_ex2;
        t_bdt_ex3           <= t_bdt_ex2;
        t_swp_ex3           <= t_swp_ex2;
        bdt_list_ex3        <= bdt_list_ex2;
        bdt_load_ex3        <= bdt_load_ex2;
        bdt_s_ex3           <= bdt_s_ex2;
        bdt_wb_ex3          <= bdt_wb_ex2;
        addr_pre_idx_bdt_ex3<= addr_pre_idx_bdt_ex2;
        addr_up_bdt_ex3     <= addr_up_bdt_ex2;
        swap_byte_ex3       <= swap_byte_ex2;
        base_reg_ex3        <= base_reg_ex2;
        base_value_ex3      <= base_value_ex2;
        rd_addr_ex3         <= rd_addr_ex2;
        rm_addr_ex3         <= rm_addr_ex2;
        psr_wr_ex3          <= psr_wr_ex2;
        psr_mask_ex3        <= psr_mask_ex2;
        psr_field_sel_ex3   <= psr_field_sel_ex2;
    end
end


/*********************************************************
 ************ EX3 Stage ************
 *  ALU ONLY.  No flag Z wide-gate here.
 *
 *  *** v2.5d: The ALU runs on registered barrel shifter
 *  outputs (bs_dout_ex3) and registered operand_a
 *  (rn_fwd_ex3).  The ALU produces alu_result and
 *  alu_flags (N, Z, C, V).  However, the Z flag from
 *  alu_flags is NOT used — it is recomputed in EX4
 *  from the registered alu_result to break the timing
 *  path.  N, C, V are captured in EX3/EX4 pipe reg.
 *
 *  Timing budget (EX3):
 *    ALU (KSA adder + result mux) ≈ 6.4 ns
 *    + N/C/V flag extraction      ≈ 0.5 ns
 *                                 ≈ 6.9 ns  ✓
 *********************************************************/

alu u_alu (
    .operand_a       (rn_fwd_ex3),
    .operand_b       (bs_dout_ex3),
    .alu_op          (alu_op_ex3),
    .cin             (cpsr_flags[`FLAG_C]),
    .shift_carry_out (shifter_cout_ex3),
    .result          (alu_result_ex3),
    .alu_flags       (alu_flags_ex3_w)
);


/*********************************************************
 ******** EX3/EX4 Pipeline Register ********
 *  Registers ALU result and flags {N, ?, C, V}.
 *  The Z bit in alu_flags_ex4 is stale — EX4 recomputes
 *  it from alu_result_ex4.
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_ex4      <= 32'd0;
        alu_flags_ex4       <= 4'd0;
        shifter_cout_ex4    <= 1'b0;
        cpsr_wen_ex4        <= 1'b0;
        mem_read_ex4        <= 1'b0;
        mem_write_ex4       <= 1'b0;
        mem_size_ex4        <= 2'd0;
        mem_signed_ex4      <= 1'b0;
        addr_pre_idx_ex4    <= 1'b0;
        wb_sel_ex4          <= 3'd0;
        wr_addr1_ex4        <= 4'd0;
        wr_addr2_ex4        <= 4'd0;
        wr_en1_ex4          <= 1'b0;
        wr_en2_ex4          <= 1'b0;
        branch_en_ex4       <= 1'b0;
        branch_link_ex4     <= 1'b0;
        branch_target_ex4_r <= 32'd0;
        store_data_ex4      <= 32'd0;
        rn_fwd_for_addr_ex4 <= 32'd0;
        pc_plus4_ex4        <= 32'd0;
        cond_code_ex4       <= 4'b1110;
        valid_ex4           <= 1'b0;
        is_multi_cycle_ex4  <= 1'b0;
        t_bdt_ex4           <= 1'b0;
        t_swp_ex4           <= 1'b0;
        bdt_list_ex4        <= 16'd0;
        bdt_load_ex4        <= 1'b0;
        bdt_s_ex4           <= 1'b0;
        bdt_wb_ex4          <= 1'b0;
        addr_pre_idx_bdt_ex4<= 1'b0;
        addr_up_bdt_ex4     <= 1'b0;
        swap_byte_ex4       <= 1'b0;
        base_reg_ex4        <= 4'd0;
        base_value_ex4      <= 32'd0;
        rd_addr_ex4         <= 4'd0;
        rm_addr_ex4         <= 4'd0;
        psr_wr_ex4          <= 1'b0;
        psr_mask_ex4        <= 4'd0;
        psr_field_sel_ex4   <= 1'b0;
    end
    else if (flush_ex3ex4_f) begin
        alu_result_ex4      <= 32'd0;
        alu_flags_ex4       <= 4'd0;
        shifter_cout_ex4    <= 1'b0;
        cpsr_wen_ex4        <= 1'b0;
        mem_read_ex4        <= 1'b0;
        mem_write_ex4       <= 1'b0;
        mem_size_ex4        <= 2'd0;
        mem_signed_ex4      <= 1'b0;
        addr_pre_idx_ex4    <= 1'b0;
        wb_sel_ex4          <= 3'd0;
        wr_addr1_ex4        <= 4'd0;
        wr_addr2_ex4        <= 4'd0;
        wr_en1_ex4          <= 1'b0;
        wr_en2_ex4          <= 1'b0;
        branch_en_ex4       <= 1'b0;
        branch_link_ex4     <= 1'b0;
        branch_target_ex4_r <= 32'd0;
        store_data_ex4      <= 32'd0;
        rn_fwd_for_addr_ex4 <= 32'd0;
        pc_plus4_ex4        <= 32'd0;
        cond_code_ex4       <= 4'b1110;
        valid_ex4           <= 1'b0;
        is_multi_cycle_ex4  <= 1'b0;
        t_bdt_ex4           <= 1'b0;
        t_swp_ex4           <= 1'b0;
        bdt_list_ex4        <= 16'd0;
        bdt_load_ex4        <= 1'b0;
        bdt_s_ex4           <= 1'b0;
        bdt_wb_ex4          <= 1'b0;
        addr_pre_idx_bdt_ex4<= 1'b0;
        addr_up_bdt_ex4     <= 1'b0;
        swap_byte_ex4       <= 1'b0;
        base_reg_ex4        <= 4'd0;
        base_value_ex4      <= 32'd0;
        rd_addr_ex4         <= 4'd0;
        rm_addr_ex4         <= 4'd0;
        psr_wr_ex4          <= 1'b0;
        psr_mask_ex4        <= 4'd0;
        psr_field_sel_ex4   <= 1'b0;
    end else if (!stall_ex4) begin
        alu_result_ex4      <= alu_result_ex3;
        alu_flags_ex4       <= alu_flags_ex3_w;
        shifter_cout_ex4    <= shifter_cout_ex3;
        cpsr_wen_ex4        <= cpsr_wen_ex3;
        mem_read_ex4        <= mem_read_ex3;
        mem_write_ex4       <= mem_write_ex3;
        mem_size_ex4        <= mem_size_ex3;
        mem_signed_ex4      <= mem_signed_ex3;
        addr_pre_idx_ex4    <= addr_pre_idx_ex3;
        wb_sel_ex4          <= wb_sel_ex3;
        wr_addr1_ex4        <= wr_addr1_ex3;
        wr_addr2_ex4        <= wr_addr2_ex3;
        wr_en1_ex4          <= wr_en1_ex3;
        wr_en2_ex4          <= wr_en2_ex3;
        branch_en_ex4       <= branch_en_ex3;
        branch_link_ex4     <= branch_link_ex3;
        branch_target_ex4_r <= branch_target_ex3;
        store_data_ex4      <= store_data_ex3;
        rn_fwd_for_addr_ex4 <= rn_fwd_for_addr_ex3;
        pc_plus4_ex4        <= pc_plus4_ex3;
        cond_code_ex4       <= cond_code_ex3;
        valid_ex4           <= valid_ex3;
        is_multi_cycle_ex4  <= is_multi_cycle_ex3;
        t_bdt_ex4           <= t_bdt_ex3;
        t_swp_ex4           <= t_swp_ex3;
        bdt_list_ex4        <= bdt_list_ex3;
        bdt_load_ex4        <= bdt_load_ex3;
        bdt_s_ex4           <= bdt_s_ex3;
        bdt_wb_ex4          <= bdt_wb_ex3;
        addr_pre_idx_bdt_ex4<= addr_pre_idx_bdt_ex3;
        addr_up_bdt_ex4     <= addr_up_bdt_ex3;
        swap_byte_ex4       <= swap_byte_ex3;
        base_reg_ex4        <= base_reg_ex3;
        base_value_ex4      <= base_value_ex3;
        rd_addr_ex4         <= rd_addr_ex3;
        rm_addr_ex4         <= rm_addr_ex3;
        psr_wr_ex4          <= psr_wr_ex3;
        psr_mask_ex4        <= psr_mask_ex3;
        psr_field_sel_ex4   <= psr_field_sel_ex3;
    end
end


/*********************************************************
 ************ EX4 Stage ************
 *  Z flag recompute, condition evaluation, branch
 *  resolution, CPSR update.
 *
 *  *** v2.5d: Z flag is recomputed here from the
 *  registered alu_result_ex4, replacing the stale Z bit
 *  in alu_flags_ex4.  This breaks the critical path:
 *    EX3 had: ALU + Z_wide_NOR = 8.5 ns (too slow)
 *    EX4 now: Z_wide_NOR (~2 ns) + cond_eval (~1 ns)
 *             = ~3 ns  ✓
 *
 *  The final flags vector used for CPSR update is
 *  alu_flags_final_ex4 = {N, Z_recomputed, C, V}.
 *********************************************************/

/* Z flag recomputed from registered result */
assign z_flag_recomputed_ex4 = (alu_result_ex4 == 32'd0);

/* Final flags: replace Z bit with recomputed value.
 * alu_flags layout: [3]=N, [2]=Z, [1]=C, [0]=V */
assign alu_flags_final_ex4 = {alu_flags_ex4[3],
                               z_flag_recomputed_ex4,
                               alu_flags_ex4[1:0]};

cond_eval u_cond_eval (
    .cond_code (cond_code_ex4),
    .flags     (cpsr_flags),
    .cond_met  (cond_met_raw_ex4)
);

assign cond_met_ex4 = cond_met_raw_ex4 && valid_ex4;

/* Condition-gated control signals */
assign wr_en1_gated_ex4         = wr_en1_ex4         & cond_met_ex4;
assign wr_en2_gated_ex4         = wr_en2_ex4         & cond_met_ex4;
assign mem_read_gated_ex4       = mem_read_ex4       & cond_met_ex4;
assign mem_write_gated_ex4      = mem_write_ex4      & cond_met_ex4;
assign cpsr_wen_gated_ex4       = cpsr_wen_ex4       & cond_met_ex4;
assign is_multi_cycle_gated_ex4 = is_multi_cycle_ex4 & cond_met_ex4;

/* Branch resolution — in EX4 (was EX3 in v2.5c) */
assign branch_taken_ex4       = branch_en_ex4 & cond_met_ex4;
assign branch_target_ex4_wire = branch_target_ex4_r;

/* Memory address (pre-indexed uses ALU result, else base register) */
assign mem_addr_ex4 = addr_pre_idx_ex4 ? alu_result_ex4 : rn_fwd_for_addr_ex4;

/* CPSR update — uses alu_flags_final_ex4 (with recomputed Z) */
wire psr_wr_flags_ex4 = psr_wr_ex4 && psr_mask_ex4[3] && !psr_field_sel_ex4
                         && cond_met_ex4;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        cpsr_flags <= 4'b0;
    else if (!stall_ex4) begin
        if (psr_wr_flags_ex4)
            cpsr_flags <= alu_result_ex4[31:28];
        else if (cpsr_wen_gated_ex4)
            cpsr_flags <= alu_flags_final_ex4;
    end
end


/*********************************************************
 ******** EX4/MEM Pipeline Register ********
 *  Receives gated signals from EX4 condition evaluation.
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_mem      <= {`DATA_WIDTH{1'b0}};
        mem_addr_mem        <= {`CPU_DMEM_ADDR_WIDTH{1'b0}};
        store_data_mem      <= {`DATA_WIDTH{1'b0}};
        mem_read_mem        <= 1'b0;
        mem_write_mem       <= 1'b0;
        mem_size_mem        <= 2'd0;
        mem_signed_mem      <= 1'b0;
        wb_sel_mem          <= 3'd0;
        wr_addr1_mem        <= 4'd0;
        wr_addr2_mem        <= 4'd0;
        wr_en1_mem          <= 1'b0;
        wr_en2_mem          <= 1'b0;
        pc_plus4_mem        <= {`PC_WIDTH{1'b0}};
        is_multi_cycle_mem  <= 1'b0;
        t_bdt_mem           <= 1'b0;
        t_swp_mem           <= 1'b0;
        bdt_list_mem        <= 16'd0;
        bdt_load_mem        <= 1'b0;
        bdt_s_mem           <= 1'b0;
        bdt_wb_mem          <= 1'b0;
        addr_pre_idx_bdt_mem<= 1'b0;
        addr_up_bdt_mem     <= 1'b0;
        swap_byte_mem       <= 1'b0;
        base_reg_mem        <= 4'd0;
        base_value_mem      <= {`DATA_WIDTH{1'b0}};
        swp_rd_mem          <= 4'd0;
        swp_rm_mem          <= 4'd0;
    end else if (flush_ex4mem) begin
        mem_read_mem        <= 1'b0;
        mem_write_mem       <= 1'b0;
        wr_en1_mem          <= 1'b0;
        wr_en2_mem          <= 1'b0;
        is_multi_cycle_mem  <= 1'b0;
        t_bdt_mem           <= 1'b0;
        t_swp_mem           <= 1'b0;
    end else if (!stall_mem) begin
        alu_result_mem      <= alu_result_ex4;
        mem_addr_mem        <= mem_addr_ex4;
        store_data_mem      <= store_data_ex4;
        mem_read_mem        <= mem_read_gated_ex4;
        mem_write_mem       <= mem_write_gated_ex4;
        mem_size_mem        <= mem_size_ex4;
        mem_signed_mem      <= mem_signed_ex4;
        wb_sel_mem          <= wb_sel_ex4;
        wr_addr1_mem        <= wr_addr1_ex4;
        wr_addr2_mem        <= wr_addr2_ex4;
        wr_en1_mem          <= wr_en1_gated_ex4;
        wr_en2_mem          <= wr_en2_gated_ex4;
        pc_plus4_mem        <= pc_plus4_ex4;
        is_multi_cycle_mem  <= is_multi_cycle_gated_ex4;
        t_bdt_mem           <= t_bdt_ex4 & cond_met_ex4;
        t_swp_mem           <= t_swp_ex4 & cond_met_ex4;
        bdt_list_mem        <= bdt_list_ex4;
        bdt_load_mem        <= bdt_load_ex4;
        bdt_s_mem           <= bdt_s_ex4;
        bdt_wb_mem          <= bdt_wb_ex4;
        addr_pre_idx_bdt_mem<= addr_pre_idx_bdt_ex4;
        addr_up_bdt_mem     <= addr_up_bdt_ex4;
        swap_byte_mem       <= swap_byte_ex4;
        base_reg_mem        <= base_reg_ex4;
        base_value_mem      <= base_value_ex4;
        swp_rd_mem          <= rd_addr_ex4;
        swp_rm_mem          <= rm_addr_ex4;
    end
end

/* Forwarding aliases (MEM stage → FU) */
assign exmem_wr_addr1   = wr_addr1_mem;
assign exmem_wr_en1     = wr_en1_mem;
assign exmem_is_load    = mem_read_mem;
assign exmem_alu_result = alu_result_mem;
assign exmem_wb_data2   = alu_result_mem;


/*********************************************************
 ************ MEM Stage ************
 *  Presents d_mem_addr_o to DMEM.  BRAM latches address
 *  at posedge.  d_mem_data_i valid in WB (next cycle).
 *********************************************************/

assign bdtu_rf_rd_data = r3_data_id;

bdtu u_bdtu (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (is_multi_cycle_mem),
    .op_bdt      (t_bdt_mem),
    .op_swp      (t_swp_mem),
    .reg_list    (bdt_list_mem),
    .bdt_load    (bdt_load_mem),
    .bdt_wb      (bdt_wb_mem),
    .pre_index   (addr_pre_idx_bdt_mem),
    .up_down     (addr_up_bdt_mem),
    .bdt_s       (bdt_s_mem),
    .swap_byte   (swap_byte_mem),
    .swp_rd      (swp_rd_mem),
    .swp_rm      (swp_rm_mem),
    .base_reg    (base_reg_mem),
    .base_value  (base_value_mem),
    .rf_rd_addr  (bdtu_rf_rd_addr),
    .rf_rd_data  (bdtu_rf_rd_data),
    .wr_addr1    (bdtu_wr_addr1),
    .wr_data1    (bdtu_wr_data1),
    .wr_en1      (bdtu_wr_en1),
    .wr_addr2    (bdtu_wr_addr2),
    .wr_data2    (bdtu_wr_data2),
    .wr_en2      (bdtu_wr_en2),
    .mem_addr    (bdtu_mem_addr),
    .mem_wdata   (bdtu_mem_wdata),
    .mem_rd      (bdtu_mem_rd),
    .mem_wr      (bdtu_mem_wr),
    .mem_size    (bdtu_mem_size),
    .mem_rdata   (d_mem_data_i),
    .busy        (bdtu_busy)
);

assign d_mem_addr_o = bdtu_busy ? bdtu_mem_addr  : mem_addr_mem;
assign d_mem_data_o = bdtu_busy ? bdtu_mem_wdata : store_data_mem;
assign d_mem_wen_o  = bdtu_busy ? bdtu_mem_wr    : mem_write_mem;
assign d_mem_size_o = bdtu_busy ? bdtu_mem_size  : mem_size_mem;


/*********************************************************
 * *** FIX v2.5b: Post-BDTU pipeline flush ***
 *
 * When BDTU finishes (busy falls), flush IF1-EX4 and
 * redirect PC to the BDT's PC+4 (next instruction).
 * Cost: 6 refill cycles (9-stage: IF1-EX4 = 6 stages).
 *********************************************************/

reg bdtu_busy_prev;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) bdtu_busy_prev <= 1'b0;
    else        bdtu_busy_prev <= bdtu_busy;
end

assign bdtu_done_flush = bdtu_busy_prev & ~bdtu_busy;

// Combined flush signals: HDU flushes OR post-BDTU flush
assign flush_if1if2_f = flush_if1if2 | bdtu_done_flush;
assign flush_if2id_f  = flush_if2id  | bdtu_done_flush;
assign flush_idex1_f  = flush_idex1  | bdtu_done_flush;
assign flush_ex1ex2_f = flush_ex1ex2 | bdtu_done_flush;
assign flush_ex2ex3_f = flush_ex2ex3 | bdtu_done_flush;
assign flush_ex3ex4_f = flush_ex3ex4 | bdtu_done_flush;
assign flush_ex4mem   = bdtu_done_flush;  // EX4/MEM only flushed by BDTU


/*********************************************************
 ******** MEM/WB Pipeline Register ********
 *********************************************************/

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        load_data_latch <= {`DATA_WIDTH{1'b0}};
    else if (!wb_committed)
        load_data_latch <= d_mem_data_i;
end

assign load_data_src = wb_committed ? load_data_latch : d_mem_data_i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_wb <= `DATA_WIDTH'd0;
        pc_plus4_wb   <= `DATA_WIDTH'd0;
        wb_sel_wb     <= 3'd0;
        wr_addr1_wb   <= 4'd0;
        wr_addr2_wb   <= 4'd0;
        wr_en1_wb     <= 1'b0;
        wr_en2_wb     <= 1'b0;
        mem_size_wb   <= 2'd0;
        mem_signed_wb <= 1'b0;
        wb_committed  <= 1'b0;
    end
    else if (!stall_mem) begin
        alu_result_wb <= alu_result_mem;
        pc_plus4_wb   <= pc_plus4_mem;
        wb_sel_wb     <= wb_sel_mem;
        wr_addr1_wb   <= wr_addr1_mem;
        wr_addr2_wb   <= wr_addr2_mem;
        wr_en1_wb     <= wr_en1_mem;
        wr_en2_wb     <= wr_en2_mem;
        mem_size_wb   <= mem_size_mem;
        mem_signed_wb <= mem_signed_mem;
        wb_committed  <= 1'b0;
    end
    else if (!wb_committed) begin
        wb_committed  <= 1'b1;
    end
end

assign memwb_wr_addr1 = wr_addr1_wb;
assign memwb_wr_en1   = wr_en1_wb;
assign memwb_wr_addr2 = wr_addr2_wb;
assign memwb_wr_en2   = wr_en2_wb;


/*********************************************************
 ************ WB Stage ************
 *********************************************************/

always @(*) begin
    case (mem_size_wb)
        2'b00:
            load_data_wb = mem_signed_wb
                ? {{(`DATA_WIDTH-8){load_data_src[7]}},   load_data_src[7:0]}
                : {{(`DATA_WIDTH-8){1'b0}},               load_data_src[7:0]};
        2'b01:
            load_data_wb = mem_signed_wb
                ? {{(`DATA_WIDTH-16){load_data_src[15]}},  load_data_src[15:0]}
                : {{(`DATA_WIDTH-16){1'b0}},               load_data_src[15:0]};
        default:
            load_data_wb = load_data_src;
    endcase
end

always @(*) begin
    case (wb_sel_wb)
        `WB_ALU:  wb_data1 = alu_result_wb;
        `WB_MEM:  wb_data1 = load_data_wb;
        `WB_LINK: wb_data1 = pc_plus4_wb;
        `WB_PSR:  wb_data1 = {cpsr_flags, {(`DATA_WIDTH-4){1'b0}}};
        default:  wb_data1 = alu_result_wb;
    endcase
end

assign wb_data2 = alu_result_wb;

assign wb_wr_addr1 = wr_addr1_wb;
assign wb_wr_data1 = wb_data1;
assign wb_wr_en1   = wr_en1_wb & ~wb_committed;

assign wb_wr_addr2 = wr_addr2_wb;
assign wb_wr_data2 = wb_data2;
assign wb_wr_en2   = wr_en2_wb & ~wb_committed;

assign wb_result_data = wb_data1;

endmodule

`endif // CPU_V