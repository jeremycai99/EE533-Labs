/* file: cpu_mt.v
 Description: Quad-threaded 5-stage pipeline Arm CPU module
 Author: Jeremy Cai
 Date: Feb. 18, 2026
 Version: 1.0
 */

`ifndef CPU_MT_V
`define CPU_MT_V

`include "define.v"
`include "regfile.v"
`include "mac.v"
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
    output wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o,
    output wire [`DATA_WIDTH-1:0] d_mem_data_o,
    output wire d_mem_wen_o,
    output wire [1:0] d_mem_size_o,

    output wire cpu_done,

    // ILA Debug Interface
    input  wire [1:0] ila_thread_sel,
    input  wire [4:0] ila_debug_sel,
    output reg [`DATA_WIDTH-1:0] ila_debug_data
);

wire stall_all; // global stall (BDTU only)
wire bdtu_busy;
wire branch_taken_ex;
wire [`PC_WIDTH-1:0] branch_target_ex;

reg [1:0] tid_if;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        tid_if <= 2'd0;
    else if (!stall_all)
        tid_if <= tid_if + 2'd1;    // wraps 0→1→2→3→0…
end

/* Pipeline-stage thread IDs and validity bits */
reg [1:0] tid_id,  tid_ex,  tid_mem, tid_wb;
reg valid_id, valid_ex, valid_mem, valid_wb;

reg [`PC_WIDTH-1:0] pc_thread [0:3];

/* Read MUX  */
wire [`PC_WIDTH-1:0] pc_if = pc_thread[tid_if];   // MUX
wire [`PC_WIDTH-1:0] pc_plus4_if = pc_if + 32'd4;

assign i_mem_addr_o = pc_if;

/* Write logic (sequential advance + branch override) */
integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (k = 0; k < 4; k = k + 1)
            pc_thread[k] <= `PC_WIDTH'd0;
    end
    else if (!stall_all) begin
        /* Default: advance the thread currently in IF */
        pc_thread[tid_if] <= pc_plus4_if;
        /* Branch override from EX (always a DIFFERENT thread) */
        if (branch_taken_ex && valid_ex)
            pc_thread[tid_ex] <= branch_target_ex;
    end
end

assign cpu_done = (pc_thread[0] == `CPU_DONE_PC) &&
                  (pc_thread[1] == `CPU_DONE_PC) &&
                  (pc_thread[2] == `CPU_DONE_PC) &&
                  (pc_thread[3] == `CPU_DONE_PC);

/* Hold buffer — captures i_mem_data_i on the first stall cycle
   so it is not lost if the memory bus changes.              */
reg [`INSTR_WIDTH-1:0] instr_held;
reg held_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        held_valid <= 1'b0;
    else if (stall_all && !held_valid) begin
        instr_held <= i_mem_data_i;
        held_valid <= 1'b1;
    end
    else if (!stall_all)
        held_valid <= 1'b0;
end

wire [`INSTR_WIDTH-1:0] instr_id = held_valid ? instr_held : i_mem_data_i;

/* IF/ID pipeline register */
reg [`PC_WIDTH-1:0] pc_plus4_id;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pc_plus4_id <= `PC_WIDTH'd0;
        tid_id <= 2'd0;
        valid_id <= 1'b0;
    end
    else if (!stall_all) begin
        pc_plus4_id <= pc_plus4_if;
        tid_id <= tid_if;
        valid_id <= 1'b1;
    end
end

/* Per-thread CPSR flags (declared here, updated in EX below) */
reg [3:0] cpsr_flags [0:3];

/* Condition evaluation (reads own thread's flags — no bypass) */
wire [3:0] cond_flags_id = cpsr_flags[tid_id];
wire cond_met_raw;

cond_eval u_cond_eval (
    .cond_code (instr_id[31:28]),
    .flags     (cond_flags_id),
    .cond_met  (cond_met_raw)
);

wire cond_met_id = cond_met_raw && valid_id;

/* Control Unit (purely combinational — unchanged) */
wire t_dp_reg, t_dp_imm, t_mul, t_mull, t_swp, t_bx;
wire t_hdt_rego, t_hdt_immo, t_sdt_rego, t_sdt_immo;
wire t_bdt, t_br, t_mrs, t_msr_reg, t_msr_imm, t_swi, t_undef;

wire [3:0] rn_addr_id, rd_addr_id, rs_addr_id, rm_addr_id;
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
wire mul_en_id, mul_long_id, mul_signed_id, mul_accumulate_id;

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
    .t_mul(t_mul), .t_mull(t_mull), .t_swp(t_swp), .t_bx(t_bx),
    .t_hdt_rego(t_hdt_rego), .t_hdt_immo(t_hdt_immo),
    .t_sdt_rego(t_sdt_rego), .t_sdt_immo(t_sdt_immo),
    .t_bdt(t_bdt), .t_br(t_br), .t_mrs(t_mrs),
    .t_msr_reg(t_msr_reg), .t_msr_imm(t_msr_imm),
    .t_swi(t_swi), .t_undef(t_undef),
    .rn_addr(rn_addr_id), .rd_addr(rd_addr_id),
    .rs_addr(rs_addr_id), .rm_addr(rm_addr_id),
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
    .mul_en(mul_en_id), .mul_long(mul_long_id),
    .mul_signed(mul_signed_id), .mul_accumulate(mul_accumulate_id),
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

/* Generate 4 register-file instances */
genvar g;
generate
    for (g = 0; g < 4; g = g + 1) begin : THREAD_RF

        /* === Write DEMUX: enable only the targeted instance === */
        wire is_wb_target   = (tid_wb  == g[1:0]) && valid_wb;
        wire is_bdtu_target = (tid_mem == g[1:0]) && bdtu_has_write;

        // Write enable — only ONE of WB/BDTU can target this instance
        // (because tid_wb ≠ tid_mem always)
        wire wena = (is_wb_target && (wb_wr_en1 || wb_wr_en2))
                  || is_bdtu_target;

        /* Write-port address/data routing */
        // BDTU has priority when this instance is BDTU's target
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

        /* === Read: addresses broadcast, port-3 steered for BDTU === */
        wire [3:0] local_r3addr = (bdtu_busy && (tid_mem == g[1:0]))
                                  ? bdtu_rf_rd_addr : rs_addr_id;

        /* Read-data outputs (accessible via THREAD_RF[g].xxx) */
        wire [`DATA_WIDTH-1:0] rn_out, rm_out, r3_out, r4_out, dbg_out;

        regfile u_rf (
            .clk (clk),
            /* Read addresses — broadcast to all instances (Fig 2: R0,R1) */
            .r1addr (rn_addr_id),
            .r2addr (rm_addr_id),
            .r3addr (local_r3addr),
            .r4addr (rd_addr_id),
            /* Write ports — DEMUX enables only this instance */
            .wena (wena),
            .wr_addr1 (wa1),   .wr_data1 (wd1),
            .wr_addr2 (wa2),   .wr_data2 (wd2),
            /* Read data outputs */
            .r1data (rn_out),
            .r2data (rm_out),
            .r3data (r3_out),
            .r4data (r4_out),
            /* Debug */
            .ila_cpu_reg_addr (ila_debug_sel[`REG_ADDR_WIDTH-1:0]),
            .ila_cpu_reg_data (dbg_out)
        );
    end
endgenerate

/* ── Read MUX: select outputs from RF[tid_id] (Fig 2: R0D,R1D) ── */
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

/* Debug read MUX (by ila_thread_sel) */
reg [`DATA_WIDTH-1:0] debug_reg_out;

always @(*) begin
    case (ila_thread_sel)
        2'd0: debug_reg_out = THREAD_RF[0].dbg_out;
        2'd1: debug_reg_out = THREAD_RF[1].dbg_out;
        2'd2: debug_reg_out = THREAD_RF[2].dbg_out;
        default: debug_reg_out = THREAD_RF[3].dbg_out;
    endcase
end

/* PC+8 adjustment for reads of R15 (ARM DDI0406C) */
wire [`DATA_WIDTH-1:0] rn_data_pc_adj =
    (rn_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rn_data_id;
wire [`DATA_WIDTH-1:0] rm_data_pc_adj =
    (rm_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rm_data_id;

reg [3:0] alu_op_ex;
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

reg [`DATA_WIDTH-1:0] rn_data_ex, rm_data_ex, rs_data_ex, rd_data_ex;
reg [`PC_WIDTH-1:0] pc_plus4_ex;

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
    if (!rst_n) begin
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
        tid_ex <= 2'd0;
        valid_ex <= 1'b0;
    end
    else if (!stall_all) begin
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
        rn_data_ex <= rn_data_pc_adj;
        rm_data_ex <= rm_data_pc_adj;
        rs_data_ex <= r3_data_id;
        rd_data_ex <= r4_data_id;
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
        tid_ex <= tid_id;
        valid_ex <= valid_id;
    end
end


/* Operand wires — straight from pipeline register, no forwarding */
wire [`DATA_WIDTH-1:0] rn_val = rn_data_ex;
wire [`DATA_WIDTH-1:0] rm_val = rm_data_ex;
wire [`DATA_WIDTH-1:0] rs_val = rs_data_ex;
wire [`DATA_WIDTH-1:0] rd_store_val = rd_data_ex;

/* Per-thread carry for this EX instruction */
wire [3:0] cpsr_flags_ex = cpsr_flags[tid_ex];

/* Barrel Shifter */
wire [`SHIFT_AMOUNT_WIDTH-1:0] actual_shamt =
    shift_src_ex ? rs_val[`SHIFT_AMOUNT_WIDTH-1:0] : shift_amount_ex;

wire [`DATA_WIDTH-1:0] bs_dout;
wire                   shifter_cout;

barrel_shifter u_barrel_shifter (
    .din (rm_val),
    .shamt (actual_shamt),
    .shift_type (shift_type_ex),
    .cin (cpsr_flags_ex[`FLAG_C]),
    .dout (bs_dout),
    .cout (shifter_cout)
);

/*  ALU  */
wire [`DATA_WIDTH-1:0] alu_src_b_val =
    alu_src_b_ex ? imm32_ex : bs_dout;

wire [`DATA_WIDTH-1:0] alu_result_ex;
wire [3:0] alu_flags_ex;

alu u_alu (
    .operand_a (rn_val),
    .operand_b (alu_src_b_val),
    .alu_op (alu_op_ex),
    .cin (cpsr_flags_ex[`FLAG_C]),
    .shift_carry_out (shifter_cout),
    .result (alu_result_ex),
    .alu_flags (alu_flags_ex)
);

/*  MAC  */
wire [`DATA_WIDTH-1:0] mac_result_lo, mac_result_hi;
wire [3:0] mac_flags;

mac u_mac (
    .rm (rm_val),
    .rs (rs_val),
    .rn_acc (rn_val),
    .rdlo_acc (rd_store_val),
    .mul_en (mul_en_ex),
    .mul_long (mul_long_ex),
    .mul_signed (mul_signed_ex),
    .mul_accumulate (mul_accumulate_ex),
    .result_lo (mac_result_lo),
    .result_hi (mac_result_hi),
    .mac_flags (mac_flags)
);

/*  Branch Target */
wire [`PC_WIDTH-1:0] branch_target_br = pc_plus4_ex + 32'd4 + imm32_ex;
wire [`PC_WIDTH-1:0] branch_target_bx = rm_val;

assign branch_taken_ex = branch_en_ex;
assign branch_target_ex = branch_exchange_ex ? branch_target_bx
                                             : branch_target_br;

/*  Per-thread CPSR flags update  */
wire [3:0] new_flags = mul_en_ex ? mac_flags : alu_flags_ex;
wire psr_wr_flags_ex = psr_wr_ex && psr_mask_ex[3] && !psr_field_sel_ex;

integer f;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (f = 0; f < 4; f = f + 1)
            cpsr_flags[f] <= 4'b0;
    end
    else if (!stall_all && valid_ex) begin
        if (psr_wr_flags_ex)
            cpsr_flags[tid_ex] <= alu_result_ex[31:28];
        else if (cpsr_wen_ex)
            cpsr_flags[tid_ex] <= new_flags;
    end
end

/*  Memory address / store data  */
wire [`DMEM_ADDR_WIDTH-1:0] mem_addr_ex = addr_pre_idx_ex ? alu_result_ex : rn_val;
wire [`DATA_WIDTH-1:0] store_data_ex = rd_store_val;

reg [`DATA_WIDTH-1:0] alu_result_mem;
reg [`DMEM_ADDR_WIDTH-1:0] mem_addr_mem;
reg [`DATA_WIDTH-1:0] store_data_mem;
reg mem_read_mem, mem_write_mem;
reg [1:0] mem_size_mem;
reg mem_signed_mem;
reg [2:0] wb_sel_mem;
reg [3:0] wr_addr1_mem, wr_addr2_mem;
reg wr_en1_mem, wr_en2_mem;
reg [`DATA_WIDTH-1:0] mac_result_lo_mem, mac_result_hi_mem;
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
        tid_mem <= 2'd0;
        valid_mem <= 1'b0;
    end
    else if (!stall_all) begin
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
        base_value_mem <= rn_val;
        swp_rd_mem <= wr_addr1_ex;
        swp_rm_mem <= rm_data_ex;
        tid_mem <= tid_ex;
        valid_mem <= valid_ex;
    end
end

wire [`DATA_WIDTH-1:0] bdtu_mem_addr, bdtu_mem_wdata;
wire bdtu_mem_rd, bdtu_mem_wr;
wire [1:0] bdtu_mem_size;

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

/* Data memory bus mux — BDTU has priority */
assign d_mem_addr_o = bdtu_busy ? bdtu_mem_addr  : mem_addr_mem;
assign d_mem_data_o = bdtu_busy ? bdtu_mem_wdata : store_data_mem;
assign d_mem_wen_o  = bdtu_busy ? bdtu_mem_wr    : mem_write_mem;
assign d_mem_size_o = bdtu_busy ? bdtu_mem_size  : mem_size_mem;

reg [`DATA_WIDTH-1:0] alu_result_wb;
reg [`DATA_WIDTH-1:0] mac_result_lo_wb, mac_result_hi_wb;
reg [`PC_WIDTH-1:0] pc_plus4_wb;
reg [2:0] wb_sel_wb;
reg [3:0] wr_addr1_wb, wr_addr2_wb;
reg wr_en1_wb, wr_en2_wb;
reg [1:0] mem_size_wb;
reg mem_signed_wb;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        alu_result_wb <= {`DATA_WIDTH{1'b0}};
        mac_result_lo_wb <= {`DATA_WIDTH{1'b0}};
        mac_result_hi_wb <= {`DATA_WIDTH{1'b0}};
        pc_plus4_wb <= {`PC_WIDTH{1'b0}};
        wb_sel_wb <= 3'd0;
        wr_addr1_wb <= 4'd0;
        wr_addr2_wb <= 4'd0;
        wr_en1_wb <= 1'b0;
        wr_en2_wb <= 1'b0;
        mem_size_wb <= 2'd0;
        mem_signed_wb <= 1'b0;
        tid_wb <= 2'd0;
        valid_wb <= 1'b0;
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
        tid_wb <= tid_mem;
        valid_wb <= valid_mem;
    end
end

/*  Load data sign/zero extension  */
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

/*  Port-1 mux  */
wire [3:0] cpsr_flags_wb = cpsr_flags[tid_wb];  // for MRS

reg [`DATA_WIDTH-1:0] wb_data1;
always @(*) begin
    case (wb_sel_wb)
        `WB_ALU:  wb_data1 = alu_result_wb;
        `WB_MEM:  wb_data1 = load_data_wb;
        `WB_LINK: wb_data1 = pc_plus4_wb;
        `WB_PSR:  wb_data1 = {cpsr_flags_wb, {(`DATA_WIDTH-4){1'b0}}};
        `WB_MUL:  wb_data1 = mac_result_lo_wb;
        default:  wb_data1 = alu_result_wb;
    endcase
end

/*  Port-2: long-multiply RdHi or base writeback  */
wire [`DATA_WIDTH-1:0] wb_data2 = (wb_sel_wb == `WB_MUL)
                                  ? mac_result_hi_wb : alu_result_wb;

/*  Route to register file write ports (DEMUX input data)  */
assign wb_wr_addr1 = wr_addr1_wb;
assign wb_wr_data1 = wb_data1;
assign wb_wr_en1   = wr_en1_wb && valid_wb;

assign wb_wr_addr2 = wr_addr2_wb;
assign wb_wr_data2 = wb_data2;
assign wb_wr_en2   = wr_en2_wb && valid_wb;

assign stall_all = bdtu_busy;

always @(*) begin
    if (ila_debug_sel[4]) begin
        ila_debug_data = debug_reg_out;     // RF debug (per ila_thread_sel)
    end else begin
        case (ila_debug_sel[3:0])
            4'd0:  ila_debug_data = pc_thread[ila_thread_sel];
            4'd1:  ila_debug_data = instr_id;
            4'd2:  ila_debug_data = rn_data_id;
            4'd3:  ila_debug_data = rm_data_id;
            4'd4:  ila_debug_data = alu_result_ex;
            4'd5:  ila_debug_data = store_data_ex;
            4'd6:  ila_debug_data = wb_data1;
            4'd7:  ila_debug_data = {{(`DATA_WIDTH-10){1'b0}},
                                     tid_if, valid_id, bdtu_busy,
                                     branch_taken_ex, stall_all,
                                     wr_en1_wb, mem_write_mem};
            4'd8:  ila_debug_data = {{(`DATA_WIDTH-4){1'b0}},
                                     cpsr_flags[ila_thread_sel]};
            4'd9:  ila_debug_data = {{(`DATA_WIDTH-4){1'b0}}, wr_addr1_wb};
            4'd10: ila_debug_data = {{(`DATA_WIDTH-4){1'b0}}, wr_addr1_ex};
            4'd11: ila_debug_data = mac_result_lo_wb;
            4'd12: ila_debug_data = mac_result_hi_wb;
            4'd13: ila_debug_data = d_mem_data_i;
            4'd14: ila_debug_data = d_mem_addr_o;
            4'd15: ila_debug_data = {{(`DATA_WIDTH-8){1'b0}},
                                     tid_id, tid_ex, tid_mem, tid_wb};
            default: ila_debug_data = {`DATA_WIDTH{1'b1}};
        endcase
    end
end

endmodule

`endif // CPU_MT_V