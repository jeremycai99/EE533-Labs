/* file: cpu.v
 Description: Single thread 5-stage pipeline Arm CPU module
 Author: Jeremy Cai
 Date: Feb. 20, 2026
 Version: 1.2
 Revision history:
    - 1.0: Initial version (Feb. 18, 2026)
    - 1.1: Added 4th register-file read port to fix port-3 conflict
            (Feb. 20, 2026).  Rs and Rd now have dedicated read ports,
            fixing UMLAL/MLA/SMLAL which need both simultaneously.
    - 1.2: Added secondary write-port (port 2) forwarding through
            FU and HDU.  EX/MEM and MEM/WB port-2 writeback values
            (base-writeback / RdHi) are now forwarded, fixing hazards
            such as LDR Rd,[Rn,#off]! followed by STR Rt,[Rn,#x].
            (Feb. 20, 2026)
 */

/*  define.v ADDITIONS REQUIRED for this revision:

 */

`ifndef CPU_V
`define CPU_V

`include "define.v"
`include "pc.v"
`include "regfile.v"
`include "hdu.v"
`include "fu.v"
`include "mac.v"
`include "cu.v"
`include "alu.v"
`include "cond_eval.v"
`include "bdtu.v"
`include "barrel_shifter.v"

module cpu (
    input wire clk,
    input wire rst_n,
    // Instruction memory interface
    input wire [`INSTR_WIDTH-1:0] i_mem_data_i, // Instruction memory data input
    output wire [`PC_WIDTH-1:0] i_mem_addr_o,   // Instruction memory address output

    // Data memory interface
    input wire [`DATA_WIDTH-1:0] d_mem_data_i,  // Data memory data input (32-bit)
    output wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o, // Data memory address output
    output wire [`DATA_WIDTH-1:0] d_mem_data_o, // Data memory data output (32-bit)
    output wire d_mem_wen_o,                    // Data memory write enable
    output wire [1:0] d_mem_size_o,             // Data memory access size (00=byte, 01=halfword, 10=word)
    output wire cpu_done,                       // Signal to indicate CPU completion

    // ILA Debug Interface
    input wire [4:0] ila_debug_sel,
    output reg [`DATA_WIDTH-1:0] ila_debug_data
);

/*********************************************************
 ************   IF Stage Signals and Logic    ************
 *********************************************************/

// Major signal declarations
// Pipeline control signals declaration from HDU
wire stall_if, stall_id, stall_ex, stall_mem;
wire flush_ifid, flush_idex, flush_exmem;
wire bdtu_busy;
wire bdtu_starting;

// Branch control signals
wire branch_taken_ex; // Branch evaluation result in EX stage
wire [`PC_WIDTH-1:0] branch_target_ex; // Target address calculated in EX stage

// CPU pipeline stage internal signal definitions
// IF stage signals
wire [`PC_WIDTH-1:0] pc_if;
wire [`PC_WIDTH-1:0] pc_next_if;
wire [`PC_WIDTH-1:0] pc_plus4_if;
wire pc_en = ~stall_if;

assign pc_plus4_if = pc_if + 32'd4;
assign pc_next_if = branch_taken_ex ? branch_target_ex : pc_plus4_if;

pc u_pc (
    .clk(clk),
    .rst_n(rst_n),
    .pc_in(pc_next_if),
    .en(pc_en),
    .pc_out(pc_if)
);

assign i_mem_addr_o = pc_if;

// CPU Done signal: Active when PC reaches max value (all 1s)
assign cpu_done = (pc_if == `CPU_DONE_PC);

/* SPECIAL CONSIDERATION FOR SYNC-READ BEHAVIOR OF INSTRUCTION MEMORY */
reg [`INSTR_WIDTH-1:0] instr_held;
reg held_valid;

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

wire [`INSTR_WIDTH-1:0] instr_id = held_valid ? instr_held : i_mem_data_i;

reg [`PC_WIDTH-1:0] pc_plus4_id;
reg ifid_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pc_plus4_id <= `PC_WIDTH'd0;
        ifid_valid <= 1'b0;
    end else if (flush_ifid) begin
        ifid_valid <= 1'b0;
    end else if (!stall_id) begin
        pc_plus4_id <= pc_plus4_if;
        ifid_valid <= 1'b1;
    end else begin
        // Stalled: hold
    end
end


/*********************************************************
 ************   ID Stage Signals and Logic    ************
 *********************************************************/

reg [3:0] cpsr_flags;
wire cond_met_raw;
wire cond_met_id;

wire [3:0] effective_flags = psr_wr_flags_ex ? alu_result_ex[31:28] :
                             cpsr_wen_ex ? new_flags : cpsr_flags;

cond_eval u_cond_eval (
    .cond_code(instr_id[31:28]),
    .flags(effective_flags),
    .cond_met(cond_met_raw)
);

assign cond_met_id = cond_met_raw && ifid_valid;

/* Start Control Unit (CU) Signal Declaration*/
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

wire mem_read_id;
wire mem_write_id;
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

wire swap_byte_id;
wire swi_en_id;

wire use_rn_id, use_rd_id, use_rs_id, use_rm_id;

wire is_multi_cycle_id;

/* End Control Unit (CU) Signal Declaration*/

cu u_cu (
    .instr(instr_id),
    .cond_met(cond_met_id),
    .t_dp_reg(t_dp_reg),
    .t_dp_imm(t_dp_imm),
    .t_mul(t_mul),
    .t_mull(t_mull),
    .t_swp(t_swp),
    .t_bx(t_bx),
    .t_hdt_rego(t_hdt_rego),
    .t_hdt_immo(t_hdt_immo),
    .t_sdt_rego(t_sdt_rego),
    .t_sdt_immo(t_sdt_immo),
    .t_bdt(t_bdt),
    .t_br(t_br),
    .t_mrs(t_mrs),
    .t_msr_reg(t_msr_reg),
    .t_msr_imm(t_msr_imm),
    .t_swi(t_swi),
    .t_undef(t_undef),
    .rn_addr(rn_addr_id),
    .rd_addr(rd_addr_id),
    .rs_addr(rs_addr_id),
    .rm_addr(rm_addr_id),
    .wr_addr1(wr_addr1_id),
    .wr_en1(wr_en1_id),
    .wr_addr2(wr_addr2_id),
    .wr_en2(wr_en2_id),
    .alu_op(alu_op_id),
    .alu_src_b(alu_src_b_id),
    .cpsr_wen(cpsr_wen_id),
    .shift_type(shift_type_id),
    .shift_amount(shift_amount_id),
    .shift_src(shift_src_id),
    .imm32(imm32_id),
    .mem_read(mem_read_id),
    .mem_write(mem_write_id),
    .mem_size(mem_size_id),
    .mem_signed(mem_signed_id),
    .addr_pre_idx(addr_pre_idx_id),
    .addr_up(addr_up_id),
    .addr_wb(addr_wb_id),
    .wb_sel(wb_sel_id),
    .branch_en(branch_en_id),
    .branch_link(branch_link_id),
    .branch_exchange(branch_exchange_id),
    .mul_en(mul_en_id),
    .mul_long(mul_long_id),
    .mul_signed(mul_signed_id),
    .mul_accumulate(mul_accumulate_id),
    .psr_rd(psr_rd_id),
    .psr_wr(psr_wr_id),
    .psr_field_sel(psr_field_sel_id),
    .psr_mask(psr_mask_id),
    .bdt_list(bdt_list_id),
    .bdt_load(bdt_load_id),
    .bdt_s(bdt_s_id),
    .bdt_wb(bdt_wb_id),
    .swap_byte(swap_byte_id),
    .swi_en(swi_en_id),
    .use_rn(use_rn_id),
    .use_rd(use_rd_id),
    .use_rs(use_rs_id),
    .use_rm(use_rm_id),
    .is_multi_cycle(is_multi_cycle_id)
);

/* Start Register File (RF) Signals */

// ── Read port address routing ──────────────────────────────────────
//
// Port 1: Rn                   (always rn_addr_id)
// Port 2: Rm                   (always rm_addr_id)
// Port 3: Rs                   (rs_addr_id; shared with BDTU when busy)
// Port 4: Rd (store / accum)   (always rd_addr_id)

wire [3:0] bdtu_rf_rd_addr;
wire [3:0] r3addr_mux = bdtu_busy ? bdtu_rf_rd_addr : rs_addr_id;

wire [`DATA_WIDTH-1:0] rn_data_id, rm_data_id, r3_data_id, r4_data_id;

// Write-back signals from WB stage
wire [3:0]  wb_wr_addr1, wb_wr_addr2;
wire [`DATA_WIDTH-1:0] wb_wr_data1, wb_wr_data2;
wire wb_wr_en1,   wb_wr_en2;

// BDTU write-back signals
wire [3:0]  bdtu_wr_addr1, bdtu_wr_addr2;
wire [`DATA_WIDTH-1:0] bdtu_wr_data1, bdtu_wr_data2;
wire bdtu_wr_en1,   bdtu_wr_en2;

wire bdtu_has_write = bdtu_wr_en1 | bdtu_wr_en2;

wire rf_wr_en = bdtu_has_write ? 1'b1 // BDTU writing → enable
    : (wb_wr_en1 | wb_wr_en2); // else WB controls

wire [3:0] rf_wr_addr1 = bdtu_has_write
    ? (bdtu_wr_en1 ? bdtu_wr_addr1 : bdtu_wr_addr2)
    : (wb_wr_en1   ? wb_wr_addr1   : wb_wr_addr2);

wire [`DATA_WIDTH-1:0] rf_wr_data1 = bdtu_has_write
    ? (bdtu_wr_en1 ? bdtu_wr_data1 : bdtu_wr_data2)
    : (wb_wr_en1   ? wb_wr_data1   : wb_wr_data2);

wire [3:0] rf_wr_addr2 = (bdtu_wr_en1 & bdtu_wr_en2)
    ? bdtu_wr_addr2                     // BDTU needs both ports
    : bdtu_has_write
        ? (wb_wr_en1 ? wb_wr_addr1 : rf_wr_addr1) // spare port → WB
        : (wb_wr_en2 ? wb_wr_addr2 : rf_wr_addr1);

wire [`DATA_WIDTH-1:0] rf_wr_data2 = (bdtu_wr_en1 & bdtu_wr_en2)
    ? bdtu_wr_data2
    : bdtu_has_write
        ? (wb_wr_en1 ? wb_wr_data1 : rf_wr_data1)
        : (wb_wr_en2 ? wb_wr_data2 : rf_wr_data1);

// Debug
wire [`DATA_WIDTH-1:0] debug_reg_out;
/* End Register File (RF) Signals */

regfile u_regfile (
    .clk (clk),
    .r1addr (rn_addr_id),
    .r2addr (rm_addr_id),
    .r3addr (r3addr_mux),
    .r4addr (rd_addr_id),
    .wena (rf_wr_en),
    .wr_addr1 (rf_wr_addr1),
    .wr_data1 (rf_wr_data1),
    .wr_addr2 (rf_wr_addr2),
    .wr_data2 (rf_wr_data2),
    .r1data (rn_data_id),
    .r2data (rm_data_id),
    .r3data (r3_data_id),
    .r4data (r4_data_id),
    .ila_cpu_reg_addr (ila_debug_sel[`REG_ADDR_WIDTH-1:0]),
    .ila_cpu_reg_data (debug_reg_out)
);

// When executing an ARM instruction, PC reads as the address of the
// current instruction plus 8.
// Ref: https://developer.arm.com/documentation/ddi0406/c/Application-Level-Architecture/Application-Level-Programmers--Model/ARM-core-registers
wire [`DATA_WIDTH-1:0] rn_data_pc_adj = (rn_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rn_data_id;
wire [`DATA_WIDTH-1:0] rm_data_pc_adj = (rm_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rm_data_id;

/*********************************************************
 ********   ID/EX PIPELINE Register Definition    ********
 *********************************************************/
reg [3:0]  alu_op_ex;
reg alu_src_b_ex;
reg cpsr_wen_ex;
reg [1:0] shift_type_ex;
reg [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount_ex;
reg shift_src_ex;
reg [`DATA_WIDTH-1:0] imm32_ex;
reg mem_read_ex, mem_write_ex;
reg [1:0] mem_size_ex;
reg mem_signed_ex;
reg addr_pre_idx_ex, addr_up_ex, addr_wb_ex;
reg [2:0] wb_sel_ex;
reg [3:0] wr_addr1_ex, wr_addr2_ex;
reg wr_en1_ex, wr_en2_ex;
reg branch_en_ex, branch_link_ex, branch_exchange_ex;
reg mul_en_ex, mul_long_ex, mul_signed_ex, mul_accumulate_ex;
reg use_rn_ex, use_rm_ex, use_rs_ex, use_rd_ex;

// Register addresses (for forwarding)
reg [3:0]  rn_addr_ex, rm_addr_ex, rs_addr_ex, rd_addr_ex;

// Register data — separate Rs and Rd pipeline registers (v1.1)
reg [`DATA_WIDTH-1:0] rn_data_ex, rm_data_ex;
reg [`DATA_WIDTH-1:0] rs_data_ex;
reg [`DATA_WIDTH-1:0] rd_data_ex;
reg [`DATA_WIDTH-1:0] pc_plus4_ex;

// BDT / multi-cycle
reg is_multi_cycle_ex;
reg t_bdt_ex, t_swp_ex;
reg [15:0] bdt_list_ex;
reg bdt_load_ex, bdt_s_ex, bdt_wb_ex;
reg addr_pre_idx_bdt_ex, addr_up_bdt_ex;
reg swap_byte_ex;
reg [3:0] base_reg_ex;

reg psr_wr_ex;
reg [3:0] psr_mask_ex;
reg psr_field_sel_ex;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n || flush_idex) begin
        alu_op_ex <= 4'd0;
        alu_src_b_ex <= 1'b0;
        cpsr_wen_ex <= 1'b0;
        shift_type_ex <= 2'd0;
        shift_amount_ex <= 5'd0;
        shift_src_ex <= 1'b0;
        imm32_ex <= 32'd0;
        mem_read_ex <= 1'b0;
        mem_write_ex <= 1'b0;
        mem_size_ex <= 2'd0;
        mem_signed_ex <= 1'b0;
        addr_pre_idx_ex <= 1'b0;
        addr_up_ex <= 1'b0;
        addr_wb_ex <= 1'b0;
        wb_sel_ex <= 3'd0;
        wr_addr1_ex <= 4'd0;
        wr_addr2_ex <= 4'd0;
        wr_en1_ex <= 1'b0;
        wr_en2_ex <= 1'b0;
        branch_en_ex <= 1'b0;
        branch_link_ex <= 1'b0;
        branch_exchange_ex <= 1'b0;
        mul_en_ex <= 1'b0;
        mul_long_ex <= 1'b0;
        mul_signed_ex <= 1'b0;
        mul_accumulate_ex <= 1'b0;
        use_rn_ex <= 1'b0;
        use_rm_ex <= 1'b0;
        use_rs_ex <= 1'b0;
        use_rd_ex <= 1'b0;
        rn_addr_ex <= 4'd0;
        rm_addr_ex <= 4'd0;
        rs_addr_ex <= 4'd0;
        rd_addr_ex <= 4'd0;
        rn_data_ex <= 32'd0;
        rm_data_ex <= 32'd0;
        rs_data_ex <= 32'd0;
        rd_data_ex <= 32'd0;
        pc_plus4_ex <= 32'd0;
        is_multi_cycle_ex <= 1'b0;
        t_bdt_ex <= 1'b0;
        t_swp_ex <= 1'b0;
        bdt_list_ex <= 16'd0;
        bdt_load_ex <= 1'b0;
        bdt_s_ex <= 1'b0;
        bdt_wb_ex <= 1'b0;
        addr_pre_idx_bdt_ex <= 1'b0;
        addr_up_bdt_ex <= 1'b0;
        swap_byte_ex <= 1'b0;
        base_reg_ex <= 4'd0;
        psr_wr_ex <= 1'b0;
        psr_mask_ex <= 4'd0;
        psr_field_sel_ex <= 1'b0;
    end
    else if (!stall_ex) begin
        alu_op_ex <= alu_op_id;
        alu_src_b_ex <= alu_src_b_id;
        cpsr_wen_ex <= cpsr_wen_id;
        shift_type_ex <= shift_type_id;
        shift_amount_ex <= shift_amount_id;
        shift_src_ex <= shift_src_id;
        imm32_ex <= imm32_id;
        mem_read_ex <= mem_read_id;
        mem_write_ex <= mem_write_id;
        mem_size_ex <= mem_size_id;
        mem_signed_ex <= mem_signed_id;
        addr_pre_idx_ex <= addr_pre_idx_id;
        addr_up_ex <= addr_up_id;
        addr_wb_ex <= addr_wb_id;
        wb_sel_ex <= wb_sel_id;
        wr_addr1_ex <= wr_addr1_id;
        wr_addr2_ex <= wr_addr2_id;
        wr_en1_ex <= wr_en1_id;
        wr_en2_ex <= wr_en2_id;
        branch_en_ex <= branch_en_id;
        branch_link_ex <= branch_link_id;
        branch_exchange_ex <= branch_exchange_id;
        mul_en_ex <= mul_en_id;
        mul_long_ex <= mul_long_id;
        mul_signed_ex <= mul_signed_id;
        mul_accumulate_ex <= mul_accumulate_id;
        use_rn_ex <= use_rn_id;
        use_rm_ex <= use_rm_id;
        use_rs_ex <= use_rs_id;
        use_rd_ex <= use_rd_id;
        rn_addr_ex <= rn_addr_id;
        rm_addr_ex <= rm_addr_id;
        rs_addr_ex <= rs_addr_id;
        rd_addr_ex <= rd_addr_id;
        rn_data_ex <= rn_data_pc_adj;
        rm_data_ex <= rm_data_pc_adj;
        rs_data_ex <= r3_data_id;  // Port 3: Rs data
        rd_data_ex <= r4_data_id;  // Port 4: Rd data (store / accumulator)
        pc_plus4_ex <= pc_plus4_id;
        is_multi_cycle_ex <= is_multi_cycle_id;
        t_bdt_ex <= t_bdt;
        t_swp_ex <= t_swp;
        bdt_list_ex <= bdt_list_id;
        bdt_load_ex <= bdt_load_id;
        bdt_s_ex <= bdt_s_id;
        bdt_wb_ex <= bdt_wb_id;
        addr_pre_idx_bdt_ex <= addr_pre_idx_id;
        addr_up_bdt_ex <= addr_up_id;
        swap_byte_ex <= swap_byte_id;
        base_reg_ex <= rn_addr_id;
        psr_wr_ex <= psr_wr_id;
        psr_mask_ex <= psr_mask_id;
        psr_field_sel_ex <= psr_field_sel_id;
    end
end

/*********************************************************
 ************   EX Stage Signals and Logic    ************
 *********************************************************/

// ── Forwarding-unit interface wires (EX/MEM) ──
wire [3:0] exmem_wr_addr1;
wire exmem_wr_en1;
wire exmem_is_load;

// ── Forwarding-unit interface wires (MEM/WB) ──
wire [3:0] memwb_wr_addr1;
wire memwb_wr_en1;
wire [3:0] memwb_wr_addr2;       // v1.2: port-2 address exposed
wire       memwb_wr_en2;         // v1.2: port-2 enable  exposed

// ── Forwarding select outputs ──
wire [2:0] fwd_a, fwd_b, fwd_s, fwd_d;

// ── Forwarding data wires (declared here, assigned after EX/MEM & WB) ──
wire [`DATA_WIDTH-1:0] exmem_alu_result;  // EX/MEM port-1 data (ALU result)
wire [`DATA_WIDTH-1:0] exmem_wb_data2;    // v1.2: EX/MEM port-2 data (base WB / RdHi)
wire [`DATA_WIDTH-1:0] wb_result_data;    // MEM/WB port-1 data (final WB mux output)
// wb_data2 (MEM/WB port-2 data) is declared in the WB section below.

fu u_fu (
    .ex_rn (rn_addr_ex),
    .ex_rm (rm_addr_ex),
    .ex_rs (rs_addr_ex),
    .ex_rd_store (rd_addr_ex),
    .ex_use_rn (use_rn_ex),
    .ex_use_rm (use_rm_ex),
    .ex_use_rs (use_rs_ex),
    .ex_use_rd_st (use_rd_ex),

    // EX/MEM — port 1
    .exmem_wd1 (exmem_wr_addr1),
    .exmem_we1 (exmem_wr_en1),
    .exmem_is_load  (exmem_is_load),

    // EX/MEM — port 2  (v1.2)
    .exmem_wd2 (wr_addr2_mem),
    .exmem_we2 (wr_en2_mem),

    // MEM/WB — port 1
    .memwb_wd1 (memwb_wr_addr1),
    .memwb_we1 (memwb_wr_en1),

    // MEM/WB — port 2  (v1.2)
    .memwb_wd2 (memwb_wr_addr2),
    .memwb_we2 (memwb_wr_en2),

    // BDTU
    .bdtu_wd1 (bdtu_wr_addr1),
    .bdtu_we1 (bdtu_wr_en1),
    .bdtu_wd2 (bdtu_wr_addr2),
    .bdtu_we2 (bdtu_wr_en2),

    // Outputs
    .fwd_a (fwd_a),
    .fwd_b (fwd_b),
    .fwd_s (fwd_s),
    .fwd_d (fwd_d)
);

// ── 7-to-1 forwarding mux (v1.2: added EXMEM_P2 and MEMWB_P2) ──
//
// NOTE: If `FWD_EXMEM_P2 / `FWD_MEMWB_P2 are not yet in define.v,
//       add them:  `define FWD_EXMEM_P2  3'b101
//                  `define FWD_MEMWB_P2  3'b110

function [`DATA_WIDTH-1:0] fwd_mux;
    input [2:0]              sel;
    input [`DATA_WIDTH-1:0]  reg_val;      // register-file value (default)
    input [`DATA_WIDTH-1:0]  exmem_p1;     // EX/MEM port-1
    input [`DATA_WIDTH-1:0]  exmem_p2;     // EX/MEM port-2
    input [`DATA_WIDTH-1:0]  memwb_p1;     // MEM/WB port-1
    input [`DATA_WIDTH-1:0]  memwb_p2;     // MEM/WB port-2
    input [`DATA_WIDTH-1:0]  bdtu_p1;      // BDTU  port-1
    input [`DATA_WIDTH-1:0]  bdtu_p2;      // BDTU  port-2
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

// wb_data2 is a wire declared in the WB section; Verilog allows
// referencing it here because all wires are module-scoped.
wire [`DATA_WIDTH-1:0] rn_fwd = fwd_mux(fwd_a, rn_data_ex,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data,   wb_data2,
    bdtu_wr_data1,    bdtu_wr_data2);

wire [`DATA_WIDTH-1:0] rm_fwd = fwd_mux(fwd_b, rm_data_ex,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data,   wb_data2,
    bdtu_wr_data1,    bdtu_wr_data2);

wire [`DATA_WIDTH-1:0] rs_fwd = fwd_mux(fwd_s, rs_data_ex,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data,   wb_data2,
    bdtu_wr_data1,    bdtu_wr_data2);

wire [`DATA_WIDTH-1:0] rd_store_fwd = fwd_mux(fwd_d, rd_data_ex,
    exmem_alu_result, exmem_wb_data2,
    wb_result_data,   wb_data2,
    bdtu_wr_data1,    bdtu_wr_data2);

// Barrel Shifter
wire [`SHIFT_AMOUNT_WIDTH-1:0] actual_shamt = shift_src_ex ? rs_fwd[`SHIFT_AMOUNT_WIDTH-1:0] : shift_amount_ex;

wire [`DATA_WIDTH-1:0] bs_din = rm_fwd;

wire [`DATA_WIDTH-1:0] bs_dout;
wire shifter_cout;

barrel_shifter u_barrel_shifter (
    .din (bs_din),
    .shamt (actual_shamt),
    .shift_type (shift_type_ex),
    .cin (cpsr_flags[`FLAG_C]),
    .dout (bs_dout),
    .cout (shifter_cout)
);

wire [`DATA_WIDTH-1:0] shifted_rm = bs_dout;

// ALU signals and instance
wire [`DATA_WIDTH-1:0] alu_src_b_val = alu_src_b_ex ? imm32_ex : shifted_rm;

wire [`DATA_WIDTH-1:0] alu_result_ex;
wire [3:0] alu_flags_ex;

alu u_alu (
    .operand_a (rn_fwd),
    .operand_b (alu_src_b_val),
    .alu_op (alu_op_ex),
    .cin (cpsr_flags[`FLAG_C]),
    .shift_carry_out (shifter_cout),
    .result (alu_result_ex),
    .alu_flags (alu_flags_ex)
);

// MAC Unit
wire [`DATA_WIDTH-1:0] mac_result_lo, mac_result_hi;
wire [3:0]  mac_flags;

mac u_mac (
    .rm (rm_fwd),
    .rs (rs_fwd),
    .rn_acc (rn_fwd),
    .rdlo_acc (rd_store_fwd),
    .mul_en (mul_en_ex),
    .mul_long (mul_long_ex),
    .mul_signed (mul_signed_ex),
    .mul_accumulate (mul_accumulate_ex),
    .result_lo (mac_result_lo),
    .result_hi (mac_result_hi),
    .mac_flags (mac_flags)
);

// Branch target calculation
wire [`PC_WIDTH-1:0] branch_target_br = pc_plus4_ex + 32'd4 + imm32_ex;
wire [`PC_WIDTH-1:0] branch_target_bx = rm_fwd;

assign branch_taken_ex  = branch_en_ex;
assign branch_target_ex = branch_exchange_ex ? branch_target_bx : branch_target_br;

// Condition flags update logic
wire [3:0] new_flags = mul_en_ex ? mac_flags : alu_flags_ex;

wire psr_wr_flags_ex = psr_wr_ex && psr_mask_ex[3] && !psr_field_sel_ex;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        cpsr_flags <= 4'b0;
    else if (!stall_ex) begin
        if (psr_wr_flags_ex)
            cpsr_flags <= alu_result_ex[31:28];   // MSR: flags from operand
        else if (cpsr_wen_ex)
            cpsr_flags <= new_flags;              // DP/MUL: flags from ALU/MAC
    end
end

wire [`DMEM_ADDR_WIDTH-1:0] mem_addr_ex = addr_pre_idx_ex ? alu_result_ex : rn_fwd;
wire [`DATA_WIDTH-1:0] store_data_ex = rd_store_fwd;

/*********************************************************
 ********   EX/MEM PIPELINE Register Definition   ********
 *********************************************************/

reg [`DATA_WIDTH-1:0] alu_result_mem;
reg [`DMEM_ADDR_WIDTH-1:0] mem_addr_mem;
reg [`DATA_WIDTH-1:0] store_data_mem;
reg mem_read_mem,  mem_write_mem;
reg [1:0] mem_size_mem;
reg mem_signed_mem;
reg [2:0] wb_sel_mem;
reg [3:0] wr_addr1_mem, wr_addr2_mem;
reg wr_en1_mem, wr_en2_mem;
reg [`DATA_WIDTH-1:0] mac_result_lo_mem, mac_result_hi_mem;
reg [`PC_WIDTH-1:0] pc_plus4_mem;

// BDTU fields
reg is_multi_cycle_mem;
reg t_bdt_mem, t_swp_mem;
reg [15:0] bdt_list_mem;
reg bdt_load_mem, bdt_s_mem, bdt_wb_mem;
reg addr_pre_idx_bdt_mem, addr_up_bdt_mem;
reg swap_byte_mem;
reg [3:0]  base_reg_mem;
reg [`DATA_WIDTH-1:0] base_value_mem;
reg [3:0]  swp_rd_mem, swp_rm_mem;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n || flush_exmem) begin
        alu_result_mem <= {`DATA_WIDTH{1'b0}};
        mem_addr_mem <= {`DMEM_ADDR_WIDTH{1'b0}};
        store_data_mem <= {`DATA_WIDTH{1'b0}};
        mem_read_mem <= 1'b0;
        mem_write_mem <= 1'b0;
        mem_size_mem <= 2'd0;
        mem_signed_mem <= 1'b0;
        wb_sel_mem <= 3'd0;
        wr_addr1_mem <= 4'd0;
        wr_addr2_mem <= 4'd0;
        wr_en1_mem <= 1'b0;
        wr_en2_mem <= 1'b0;
        mac_result_lo_mem <= {`DATA_WIDTH{1'b0}};
        mac_result_hi_mem <= {`DATA_WIDTH{1'b0}};
        pc_plus4_mem <= {`PC_WIDTH{1'b0}};
        is_multi_cycle_mem <= 1'b0;
        t_bdt_mem <= 1'b0;
        t_swp_mem <= 1'b0;
        bdt_list_mem <= 16'd0;
        bdt_load_mem <= 1'b0;
        bdt_s_mem <= 1'b0;
        bdt_wb_mem <= 1'b0;
        addr_pre_idx_bdt_mem <= 1'b0;
        addr_up_bdt_mem <= 1'b0;
        swap_byte_mem <= 1'b0;
        base_reg_mem <= 4'd0;
        base_value_mem <= {`DATA_WIDTH{1'b0}};
        swp_rd_mem <= 4'd0;
        swp_rm_mem <= 4'd0;
    end
    else if (!stall_mem) begin
        alu_result_mem <= alu_result_ex;
        mem_addr_mem <= mem_addr_ex;
        store_data_mem <= store_data_ex;
        mem_read_mem <= mem_read_ex;
        mem_write_mem <= mem_write_ex;
        mem_size_mem <= mem_size_ex;
        mem_signed_mem <= mem_signed_ex;
        wb_sel_mem <= wb_sel_ex;
        wr_addr1_mem <= wr_addr1_ex;
        wr_addr2_mem <= wr_addr2_ex;
        wr_en1_mem <= wr_en1_ex;
        wr_en2_mem <= wr_en2_ex;
        mac_result_lo_mem <= mac_result_lo;
        mac_result_hi_mem <= mac_result_hi;
        pc_plus4_mem <= pc_plus4_ex;
        is_multi_cycle_mem <= is_multi_cycle_ex;
        t_bdt_mem <= t_bdt_ex;
        t_swp_mem <= t_swp_ex;
        bdt_list_mem <= bdt_list_ex;
        bdt_load_mem <= bdt_load_ex;
        bdt_s_mem <= bdt_s_ex;
        bdt_wb_mem <= bdt_wb_ex;
        addr_pre_idx_bdt_mem <= addr_pre_idx_bdt_ex;
        addr_up_bdt_mem <= addr_up_bdt_ex;
        swap_byte_mem <= swap_byte_ex;
        base_reg_mem <= base_reg_ex;
        base_value_mem <= rn_fwd;
        swp_rd_mem <= rd_addr_ex;
        swp_rm_mem <= rm_addr_ex;
    end
end

// ── Expose EX/MEM for forwarding ──
assign exmem_wr_addr1  = wr_addr1_mem;
assign exmem_wr_en1    = wr_en1_mem;
assign exmem_is_load   = mem_read_mem;
assign exmem_alu_result = alu_result_mem;

// v1.2: Port-2 forwarding data from EX/MEM.
//   • Long multiply (WB_MUL) -> mac_result_hi (RdHi)
//   • Everything else        -> alu_result    (base writeback address)
//
// The FU's exmem_valid2 has no exmem_is_load guard because port-2
// always carries a value computed in EX (never memory-read data).
assign exmem_wb_data2 = (wb_sel_mem == `WB_MUL) ? mac_result_hi_mem
                                                  : alu_result_mem;

/*********************************************************
 ************   MEM Stage Signals and Logic    ***********
 *********************************************************/
wire [`DATA_WIDTH-1:0] bdtu_mem_addr, bdtu_mem_wdata;
wire        bdtu_mem_rd, bdtu_mem_wr;
wire [1:0]  bdtu_mem_size;

// BDTU register reads via port 3 (free during BDTU stall)
wire [`DATA_WIDTH-1:0] bdtu_rf_rd_data = r3_data_id;

bdtu u_bdtu (
    .clk (clk),
    .rst_n (rst_n),
    .start (is_multi_cycle_mem),
    .op_bdt (t_bdt_mem),
    .op_swp (t_swp_mem),
    .reg_list (bdt_list_mem),
    .bdt_load (bdt_load_mem),
    .bdt_wb (bdt_wb_mem),
    .pre_index (addr_pre_idx_bdt_mem),
    .up_down (addr_up_bdt_mem),
    .bdt_s (bdt_s_mem),
    .swap_byte (swap_byte_mem),
    .swp_rd (swp_rd_mem),
    .swp_rm (swp_rm_mem),
    .base_reg (base_reg_mem),
    .base_value (base_value_mem),
    .rf_rd_addr (bdtu_rf_rd_addr),
    .rf_rd_data (bdtu_rf_rd_data),
    .wr_addr1 (bdtu_wr_addr1),
    .wr_data1 (bdtu_wr_data1),
    .wr_en1 (bdtu_wr_en1),
    .wr_addr2 (bdtu_wr_addr2),
    .wr_data2 (bdtu_wr_data2),
    .wr_en2 (bdtu_wr_en2),
    .mem_addr (bdtu_mem_addr),
    .mem_wdata (bdtu_mem_wdata),
    .mem_rd (bdtu_mem_rd),
    .mem_wr (bdtu_mem_wr),
    .mem_size (bdtu_mem_size),
    .mem_rdata (d_mem_data_i),
    .busy (bdtu_busy)
);

// Data Memory Interface Mux — BDTU has priority
assign d_mem_addr_o = bdtu_busy ? bdtu_mem_addr  : mem_addr_mem;
assign d_mem_data_o = bdtu_busy ? bdtu_mem_wdata : store_data_mem;
assign d_mem_wen_o  = bdtu_busy ? bdtu_mem_wr    : mem_write_mem;
assign d_mem_size_o = bdtu_busy ? bdtu_mem_size  : mem_size_mem;

/*********************************************************
 ********   MEM/WB PIPELINE Register Definition   ********
 *********************************************************/

reg [`DATA_WIDTH-1:0] alu_result_wb;
reg [`DATA_WIDTH-1:0] mac_result_lo_wb, mac_result_hi_wb;
reg [`DATA_WIDTH-1:0] pc_plus4_wb;
reg [2:0]  wb_sel_wb;
reg [3:0]  wr_addr1_wb, wr_addr2_wb;
reg        wr_en1_wb,   wr_en2_wb;
reg [1:0]  mem_size_wb;
reg        mem_signed_wb;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_wb <= `DATA_WIDTH'd0;
        mac_result_lo_wb <= `DATA_WIDTH'd0;
        mac_result_hi_wb <= `DATA_WIDTH'd0;
        pc_plus4_wb <= `DATA_WIDTH'd0;
        wb_sel_wb <= 3'd0;
        wr_addr1_wb <= 4'd0;
        wr_addr2_wb <= 4'd0;
        wr_en1_wb <= 1'b0;
        wr_en2_wb <= 1'b0;
        mem_size_wb <= 2'd0;
        mem_signed_wb <= 1'b0;
    end
    else begin
        alu_result_wb <= alu_result_mem;
        mac_result_lo_wb <= mac_result_lo_mem;
        mac_result_hi_wb <= mac_result_hi_mem;
        pc_plus4_wb <= pc_plus4_mem;
        wb_sel_wb <= wb_sel_mem;
        wr_addr1_wb <= wr_addr1_mem;
        wr_addr2_wb <= wr_addr2_mem;
        wr_en1_wb <= wr_en1_mem;
        wr_en2_wb <= wr_en2_mem;
        mem_size_wb <= mem_size_mem;
        mem_signed_wb <= mem_signed_mem;
    end
end

// ── Expose MEM/WB for forwarding ──
assign memwb_wr_addr1 = wr_addr1_wb;
assign memwb_wr_en1   = wr_en1_wb;
assign memwb_wr_addr2 = wr_addr2_wb;   // v1.2: port-2
assign memwb_wr_en2   = wr_en2_wb;     // v1.2: port-2


/*********************************************************
 ************   WB Stage Signals and Logic    ************
 *********************************************************/

reg [`DATA_WIDTH-1:0] load_data_wb;

always @(*) begin
    case (mem_size_wb)
        2'b00: // Byte
            load_data_wb = mem_signed_wb ? {{(`DATA_WIDTH-8){d_mem_data_i[7]}},  d_mem_data_i[7:0]}
                                         : {{(`DATA_WIDTH-8){1'b0}}, d_mem_data_i[7:0]};
        2'b01: // Halfword
            load_data_wb = mem_signed_wb ? {{(`DATA_WIDTH-16){d_mem_data_i[15]}}, d_mem_data_i[15:0]}
                                         : {{(`DATA_WIDTH-16){1'b0}}, d_mem_data_i[15:0]};
        default: // Word
            load_data_wb = d_mem_data_i;
    endcase
end

// ── WB Data Mux (port 1) ──────────────────────────────────────────
reg [`DATA_WIDTH-1:0] wb_data1;

always @(*) begin
    case (wb_sel_wb)
        `WB_ALU: wb_data1 = alu_result_wb;
        `WB_MEM: wb_data1 = load_data_wb;
        `WB_LINK: wb_data1 = pc_plus4_wb;
        // Bug fix for CPSR flags write back for MRS
        `WB_PSR: wb_data1 = {{cpsr_flags, {(`DATA_WIDTH-4){1'b0}}}};
        `WB_MUL: wb_data1 = mac_result_lo_wb;
        default: wb_data1 = alu_result_wb;
    endcase
end

// ── WB Data (port 2): long multiply RdHi or base writeback ──
wire [`DATA_WIDTH-1:0] wb_data2 = (wb_sel_wb == `WB_MUL) ? mac_result_hi_wb : alu_result_wb;

// Route to register file write ports
assign wb_wr_addr1 = wr_addr1_wb;
assign wb_wr_data1 = wb_data1;
assign wb_wr_en1   = wr_en1_wb;

assign wb_wr_addr2 = wr_addr2_wb;
assign wb_wr_data2 = wb_data2;
assign wb_wr_en2   = wr_en2_wb;

// WB result for forwarding (MEM/WB port-1 path)
assign wb_result_data = wb_data1;
// wb_data2 serves as MEM/WB port-2 forwarding data (FWD_MEMWB_P2)

// ── HDU (placed after all signals it references are declared) ──
hdu u_hdu (
    .idex_is_load (mem_read_ex),

    // Port 1
    .idex_wd1 (wr_addr1_ex),
    .idex_we1 (wr_en1_ex),

    // Port 2  (v1.2)
    .idex_wd2 (wr_addr2_ex),
    .idex_we2 (wr_en2_ex),

    // IF/ID source registers
    .ifid_rn (rn_addr_id),
    .ifid_rm (rm_addr_id),
    .ifid_rs (rs_addr_id),
    .ifid_rd_store (rd_addr_id),
    .ifid_use_rn (use_rn_id),
    .ifid_use_rm (use_rm_id),
    .ifid_use_rs (use_rs_id),
    .ifid_use_rd_st (use_rd_id),

    .branch_taken (branch_taken_ex),
    .bdtu_busy (bdtu_busy),

    // Outputs
    .stall_if (stall_if),
    .stall_id (stall_id),
    .stall_ex (stall_ex),
    .stall_mem (stall_mem),

    .flush_ifid (flush_ifid),
    .flush_idex (flush_idex),
    .flush_exmem (flush_exmem)
);

// ── ILA Debug Interface ──
always @(*) begin
    if (ila_debug_sel[4]) begin
        // Mode 1: Register File Debug (MSB = 1)
        ila_debug_data = debug_reg_out;
    end else begin
        // Mode 0: System Debug (MSB = 0)
        case (ila_debug_sel[3:0])
            4'd0:  ila_debug_data = pc_if;
            4'd1:  ila_debug_data = instr_id;
            4'd2:  ila_debug_data = rn_data_id;
            4'd3:  ila_debug_data = rm_data_id;
            4'd4:  ila_debug_data = alu_result_ex;
            4'd5:  ila_debug_data = store_data_ex;
            4'd6:  ila_debug_data = wb_data1;
            4'd7:  ila_debug_data = {{(`DATA_WIDTH-9){1'b0}},
                                     ifid_valid, bdtu_busy,
                                     branch_taken_ex, stall_if,
                                     wr_en1_wb, mem_write_mem,
                                     mem_write_ex, wr_en1_id,
                                     mem_write_id};
            4'd8:  ila_debug_data = {{(`DATA_WIDTH-4){1'b0}}, cpsr_flags};
            4'd9:  ila_debug_data = {{(`DATA_WIDTH-4){1'b0}}, wr_addr1_wb};
            4'd10: ila_debug_data = {{(`DATA_WIDTH-4){1'b0}}, wr_addr1_ex};
            4'd11: ila_debug_data = mac_result_lo_wb;
            4'd12: ila_debug_data = mac_result_hi_wb;
            4'd13: ila_debug_data = d_mem_data_i;
            4'd14: ila_debug_data = d_mem_addr_o;
            4'd15: ila_debug_data = {{(`DATA_WIDTH-16){1'b0}}, bdt_list_mem};
            default: ila_debug_data = {`DATA_WIDTH{1'b1}};
        endcase
    end
end

endmodule

`endif // CPU_V