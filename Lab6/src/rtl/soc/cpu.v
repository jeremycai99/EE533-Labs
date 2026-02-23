/* file: cpu.v
 * Single thread 6-stage pipeline Arm CPU module
 * Pipeline: IF → ID → EX1 → EX2 → MEM → WB
 *
 * Version: 2.3
 * Revision history:
 *    - 2.0: Split EX into EX1/EX2. Condition evaluation in EX2.
 *    - 2.1: Replaced inline HDU with hdu.v module. Added
 *            multi-cycle-in-EX2 hazard (mc_ex2_hazard).
 *    - 2.2: (BROKEN) Gated MEM/WB with stall_mem + wb_committed,
 *            but combinational load_data_wb read stale d_mem_data_i
 *            during BDTU stalls, corrupting register writes.
 *    - 2.3: Fixed by adding load_data_latch to capture d_mem_data_i
 *            at MEM→WB transition, isolating WB from BDTU bus
 *            activity. MEM/WB stall-guard + committed suppression
 *            now correct. (Feb. 22, 2026)
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

wire stall_if, stall_id, stall_ex1, stall_ex2, stall_mem;
wire flush_ifid, flush_idex1, flush_ex1ex2;

wire [`PC_WIDTH-1:0] pc_if;
wire [`PC_WIDTH-1:0] pc_next_if;
wire [`PC_WIDTH-1:0] pc_plus4_if;
wire pc_en;

wire branch_taken_ex2;
wire [`PC_WIDTH-1:0] branch_target_ex2_wire;

reg [`INSTR_WIDTH-1:0] instr_held;
reg held_valid;
wire [`INSTR_WIDTH-1:0] instr_id;

reg [`PC_WIDTH-1:0] pc_plus4_id;
reg ifid_valid;

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
reg use_rn_ex1, use_rm_ex1, use_rs_ex1, use_rd_ex1;
reg [3:0] rn_addr_ex1, rm_addr_ex1, rs_addr_ex1, rd_addr_ex1;
reg [`DATA_WIDTH-1:0] rn_data_ex1, rm_data_ex1;
reg [`DATA_WIDTH-1:0] rs_data_ex1;
reg [`DATA_WIDTH-1:0] rd_data_ex1;
reg [`PC_WIDTH-1:0] pc_plus4_ex1;
reg is_multi_cycle_ex1;
reg t_bdt_ex1, t_swp_ex1;
reg [15:0] bdt_list_ex1;
reg bdt_load_ex1, bdt_s_ex1, bdt_wb_ex1;
reg addr_pre_idx_bdt_ex1, addr_up_bdt_ex1;
reg swap_byte_ex1;
reg [3:0] base_reg_ex1;
reg psr_wr_ex1;
reg [3:0] psr_mask_ex1;
reg psr_field_sel_ex1;
reg [3:0] cond_code_ex1;
reg valid_ex1;

wire [3:0] exmem_wr_addr1;
wire exmem_wr_en1;
wire exmem_is_load;

wire [3:0] memwb_wr_addr1;
wire memwb_wr_en1;
wire [3:0] memwb_wr_addr2;
wire memwb_wr_en2;

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

wire [`DATA_WIDTH-1:0] alu_src_b_val_ex1;

wire [`PC_WIDTH-1:0] branch_target_br_ex1;
wire [`PC_WIDTH-1:0] branch_target_bx_ex1;
wire [`PC_WIDTH-1:0] branch_target_ex1;

reg [`DATA_WIDTH-1:0] rn_fwd_ex2;
reg [`DATA_WIDTH-1:0] alu_b_ex2;
reg shifter_cout_ex2;
reg [3:0] alu_op_ex2;
reg cpsr_wen_ex2;
reg mem_read_ex2, mem_write_ex2;
reg [1:0] mem_size_ex2;
reg mem_signed_ex2;
reg addr_pre_idx_ex2;
reg [2:0] wb_sel_ex2;
reg [3:0] wr_addr1_ex2, wr_addr2_ex2;
reg wr_en1_ex2, wr_en2_ex2;
reg branch_en_ex2;
reg [`PC_WIDTH-1:0] branch_target_ex2_r;
reg [`DATA_WIDTH-1:0] store_data_ex2;
reg [`DATA_WIDTH-1:0] rn_fwd_for_addr_ex2;
reg [`PC_WIDTH-1:0] pc_plus4_ex2;
reg [3:0] cond_code_ex2;
reg valid_ex2;
reg is_multi_cycle_ex2;
reg t_bdt_ex2, t_swp_ex2;
reg [15:0] bdt_list_ex2;
reg bdt_load_ex2, bdt_s_ex2, bdt_wb_ex2;
reg addr_pre_idx_bdt_ex2, addr_up_bdt_ex2;
reg swap_byte_ex2;
reg [3:0] base_reg_ex2;
reg [`DATA_WIDTH-1:0] base_value_ex2;
reg [3:0] rd_addr_ex2, rm_addr_ex2;
reg psr_wr_ex2;
reg [3:0] psr_mask_ex2;
reg psr_field_sel_ex2;
reg branch_link_ex2;

wire [`DATA_WIDTH-1:0] alu_result_ex2;
wire [3:0] alu_flags_ex2;

wire cond_met_raw_ex2;
wire cond_met_ex2;

wire [`CPU_DMEM_ADDR_WIDTH-1:0] mem_addr_ex2;

wire wr_en1_gated_ex2;
wire wr_en2_gated_ex2;
wire mem_read_gated_ex2;
wire mem_write_gated_ex2;
wire cpsr_wen_gated_ex2;
wire is_multi_cycle_gated_ex2;

reg [`DATA_WIDTH-1:0] alu_result_mem;
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

wire [`CPU_DMEM_ADDR_WIDTH-1:0] bdtu_mem_addr;
wire [`DATA_WIDTH-1:0] bdtu_mem_wdata;
wire bdtu_mem_rd, bdtu_mem_wr;
wire [1:0] bdtu_mem_size;
wire [`DATA_WIDTH-1:0] bdtu_rf_rd_data;

reg [`DATA_WIDTH-1:0] alu_result_wb;
reg [`DATA_WIDTH-1:0] pc_plus4_wb;
reg [2:0] wb_sel_wb;
reg [3:0] wr_addr1_wb, wr_addr2_wb;
reg wr_en1_wb, wr_en2_wb;
reg [1:0] mem_size_wb;
reg mem_signed_wb;

// *** FIX v2.3: captured load data, isolated from BDTU bus ***
reg [`DATA_WIDTH-1:0] load_data_latch;

reg [`DATA_WIDTH-1:0] load_data_wb;
reg [`DATA_WIDTH-1:0] wb_data1;

// *** FIX v2.3: committed flag for WB write-once semantics ***
reg wb_committed;

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
 * HAZARD DETECTION UNIT
 *=========================================================*/

hdu u_hdu (
    .ex1ex2_wd1            (wr_addr1_ex2),
    .ex1ex2_we1            (wr_en1_ex2),
    .ex1ex2_wd2            (wr_addr2_ex2),
    .ex1ex2_we2            (wr_en2_ex2),
    .ex1ex2_is_multi_cycle (is_multi_cycle_ex2),
    .ex1ex2_valid          (valid_ex2),
    .ex2mem_is_load        (mem_read_mem),
    .ex2mem_wd1            (wr_addr1_mem),
    .ex2mem_we1            (wr_en1_mem),
    .ex1_rn                (rn_addr_ex1),
    .ex1_rm                (rm_addr_ex1),
    .ex1_rs                (rs_addr_ex1),
    .ex1_rd_store          (rd_addr_ex1),
    .ex1_use_rn            (use_rn_ex1),
    .ex1_use_rm            (use_rm_ex1),
    .ex1_use_rs            (use_rs_ex1),
    .ex1_use_rd_st         (use_rd_ex1),
    .branch_taken          (branch_taken_ex2),
    .bdtu_busy             (bdtu_busy),
    .stall_if              (stall_if),
    .stall_id              (stall_id),
    .stall_ex1             (stall_ex1),
    .stall_ex2             (stall_ex2),
    .stall_mem             (stall_mem),
    .flush_ifid            (flush_ifid),
    .flush_idex1           (flush_idex1),
    .flush_ex1ex2          (flush_ex1ex2)
);


/*********************************************************
 ************ IF Stage ************
 *********************************************************/

assign pc_plus4_if = pc_if + 32'd4;
assign pc_next_if  = branch_taken_ex2 ? branch_target_ex2_wire : pc_plus4_if;
assign pc_en       = ~stall_if;

pc u_pc (
    .clk    (clk),
    .rst_n  (rst_n),
    .pc_in  (pc_next_if),
    .en     (pc_en),
    .pc_out (pc_if)
);

assign i_mem_addr_o = pc_if;
assign cpu_done     = (pc_if == `CPU_DONE_PC);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        held_valid <= 1'b0;
    else if (flush_ifid)
        held_valid <= 1'b0;
    else if (stall_id && !held_valid) begin
        instr_held <= i_mem_data_i;
        held_valid <= 1'b1;
    end
    else if (!stall_id)
        held_valid <= 1'b0;
end

assign instr_id = held_valid ? instr_held : i_mem_data_i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pc_plus4_id <= `PC_WIDTH'd0;
        ifid_valid  <= 1'b0;
    end else if (flush_ifid) begin
        ifid_valid <= 1'b0;
    end else if (!stall_id) begin
        pc_plus4_id <= pc_plus4_if;
        ifid_valid  <= 1'b1;
    end
end


/*********************************************************
 ************ ID Stage ************
 *********************************************************/

cu u_cu (
    .instr       (instr_id),
    .cond_met    (ifid_valid),
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
    else if (flush_idex1) begin
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
        cond_code_ex1       <= instr_id[31:28];
        valid_ex1           <= ifid_valid;
    end
end


/*********************************************************
 ************ EX1 Stage ************
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
assign bs_din = rm_fwd;

barrel_shifter u_barrel_shifter (
    .din        (bs_din),
    .shamt      (actual_shamt),
    .shift_type (shift_type_ex1),
    .cin        (cpsr_flags[`FLAG_C]),
    .dout       (bs_dout),
    .cout       (shifter_cout)
);

assign alu_src_b_val_ex1 = alu_src_b_ex1 ? imm32_ex1 : bs_dout;

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
        alu_b_ex2           <= 32'd0;
        shifter_cout_ex2    <= 1'b0;
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
        branch_target_ex2_r <= 32'd0;
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
    else if (flush_ex1ex2) begin
        rn_fwd_ex2          <= 32'd0;
        alu_b_ex2           <= 32'd0;
        shifter_cout_ex2    <= 1'b0;
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
        branch_target_ex2_r <= 32'd0;
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
        alu_b_ex2           <= alu_src_b_val_ex1;
        shifter_cout_ex2    <= shifter_cout;
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
        branch_target_ex2_r <= branch_target_ex1;
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
 *********************************************************/

alu u_alu (
    .operand_a       (rn_fwd_ex2),
    .operand_b       (alu_b_ex2),
    .alu_op          (alu_op_ex2),
    .cin             (cpsr_flags[`FLAG_C]),
    .shift_carry_out (shifter_cout_ex2),
    .result          (alu_result_ex2),
    .alu_flags       (alu_flags_ex2)
);

cond_eval u_cond_eval (
    .cond_code (cond_code_ex2),
    .flags     (cpsr_flags),
    .cond_met  (cond_met_raw_ex2)
);

assign cond_met_ex2 = cond_met_raw_ex2 && valid_ex2;

assign wr_en1_gated_ex2          = wr_en1_ex2          & cond_met_ex2;
assign wr_en2_gated_ex2          = wr_en2_ex2          & cond_met_ex2;
assign mem_read_gated_ex2        = mem_read_ex2        & cond_met_ex2;
assign mem_write_gated_ex2       = mem_write_ex2       & cond_met_ex2;
assign cpsr_wen_gated_ex2        = cpsr_wen_ex2        & cond_met_ex2;
assign is_multi_cycle_gated_ex2  = is_multi_cycle_ex2  & cond_met_ex2;

assign branch_taken_ex2       = branch_en_ex2 & cond_met_ex2;
assign branch_target_ex2_wire = branch_target_ex2_r;

assign mem_addr_ex2 = addr_pre_idx_ex2 ? alu_result_ex2 : rn_fwd_for_addr_ex2;

wire psr_wr_flags_ex2 = psr_wr_ex2 && psr_mask_ex2[3] && !psr_field_sel_ex2
                         && cond_met_ex2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        cpsr_flags <= 4'b0;
    else if (!stall_ex2) begin
        if (psr_wr_flags_ex2)
            cpsr_flags <= alu_result_ex2[31:28];
        else if (cpsr_wen_gated_ex2)
            cpsr_flags <= alu_flags_ex2;
    end
end


/*********************************************************
 ******** EX2/MEM Pipeline Register ********
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
    end else if (!stall_mem) begin
        alu_result_mem      <= alu_result_ex2;
        mem_addr_mem        <= mem_addr_ex2;
        store_data_mem      <= store_data_ex2;
        mem_read_mem        <= mem_read_gated_ex2;
        mem_write_mem       <= mem_write_gated_ex2;
        mem_size_mem        <= mem_size_ex2;
        mem_signed_mem      <= mem_signed_ex2;
        wb_sel_mem          <= wb_sel_ex2;
        wr_addr1_mem        <= wr_addr1_ex2;
        wr_addr2_mem        <= wr_addr2_ex2;
        wr_en1_mem          <= wr_en1_gated_ex2;
        wr_en2_mem          <= wr_en2_gated_ex2;
        pc_plus4_mem        <= pc_plus4_ex2;
        is_multi_cycle_mem  <= is_multi_cycle_gated_ex2;
        t_bdt_mem           <= t_bdt_ex2 & cond_met_ex2;
        t_swp_mem           <= t_swp_ex2 & cond_met_ex2;
        bdt_list_mem        <= bdt_list_ex2;
        bdt_load_mem        <= bdt_load_ex2;
        bdt_s_mem           <= bdt_s_ex2;
        bdt_wb_mem          <= bdt_wb_ex2;
        addr_pre_idx_bdt_mem<= addr_pre_idx_bdt_ex2;
        addr_up_bdt_mem     <= addr_up_bdt_ex2;
        swap_byte_mem       <= swap_byte_ex2;
        base_reg_mem        <= base_reg_ex2;
        base_value_mem      <= base_value_ex2;
        swp_rd_mem          <= rd_addr_ex2;
        swp_rm_mem          <= rm_addr_ex2;
    end
end

assign exmem_wr_addr1   = wr_addr1_mem;
assign exmem_wr_en1     = wr_en1_mem;
assign exmem_is_load    = mem_read_mem;
assign exmem_alu_result = alu_result_mem;
assign exmem_wb_data2   = alu_result_mem;


/*********************************************************
 ************ MEM Stage ************
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
 ******** MEM/WB Pipeline Register ********
 *********************************************************/

// *** FIX v2.3: Capture d_mem_data_i into a latch at the
//   MEM→WB boundary.  During a BDTU stall the memory bus
//   carries BDTU traffic; without this latch the WB load
//   path would pick up BDTU read-data instead of the
//   preceding LDR's data.
//
//   load_data_latch is sampled on every cycle that MEM/WB
//   advances (!stall_mem).  When the pipeline is stalled
//   the latch holds the value that was valid at transition
//   time, isolating WB from further bus changes.

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        load_data_latch <= {`DATA_WIDTH{1'b0}};
    else if (!stall_mem)
        load_data_latch <= d_mem_data_i;
    // else: hold — d_mem_data_i now carries BDTU traffic
end

// *** FIX v2.3: Gate MEM/WB with stall_mem so control
//   signals (wr_en, wb_sel, addrs) are preserved across
//   the BDTU stall for correct forwarding.
//   wb_committed ensures the register-file write fires
//   exactly once during the stall window.

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

// *** FIX v2.3: Use load_data_latch instead of raw
//   d_mem_data_i so the load result is stable during stalls.

always @(*) begin
    case (mem_size_wb)
        2'b00:
            load_data_wb = mem_signed_wb
                ? {{(`DATA_WIDTH-8){load_data_latch[7]}}, load_data_latch[7:0]}
                : {{(`DATA_WIDTH-8){1'b0}},               load_data_latch[7:0]};
        2'b01:
            load_data_wb = mem_signed_wb
                ? {{(`DATA_WIDTH-16){load_data_latch[15]}}, load_data_latch[15:0]}
                : {{(`DATA_WIDTH-16){1'b0}},                load_data_latch[15:0]};
        default:
            load_data_wb = load_data_latch;
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
assign wb_wr_en1   = wr_en1_wb & ~wb_committed;    // *** FIX v2.3

assign wb_wr_addr2 = wr_addr2_wb;
assign wb_wr_data2 = wb_data2;
assign wb_wr_en2   = wr_en2_wb & ~wb_committed;    // *** FIX v2.3

assign wb_result_data = wb_data1;

endmodule

`endif // CPU_V