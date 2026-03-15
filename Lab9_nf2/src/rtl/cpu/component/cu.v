/* file: cu.v
 Description: Control unit module for the Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Mar. 4, 2026
 Version: 1.2
    Revision history:
        - v1.0 — Initial (Feb. 2026).
        - v1.1 — Deferred imm rotation to EX1 barrel shifter.
        - v1.2 — CP10 coprocessor support (Mar. 4, 2026).
        - v1.3 — Remove multiply execution support (Mar. 6, 2026).
                  MUL/MULL/MLAL decode retained for instruction classification
                  (prevents mis-decode as DP register). All mul_* outputs removed.
 */

`ifndef CU_V
`define CU_V

`include "define.v"
`include "cond_eval.v"

module cu (
    input wire [`INSTR_WIDTH-1:0] instr,
    input wire cond_met,

    // Instruction type decoding
    output wire t_dp_reg,
    output wire t_dp_imm,
    output wire t_mul,      // kept for classification only
    output wire t_mull,     // kept for classification only
    output wire t_swp,
    output wire t_bx,
    output wire t_hdt_rego,
    output wire t_hdt_immo,
    output wire t_sdt_rego,
    output wire t_sdt_immo,
    output wire t_bdt,
    output wire t_br,
    output wire t_mrs,
    output wire t_msr_reg,
    output wire t_msr_imm,
    output wire t_swi,
    output wire t_undef,
    output wire t_mcr,
    output wire t_mrc,

    // Register file addresses
    output wire [3:0] rn_addr,
    output wire [3:0] rd_addr,
    output wire [3:0] rs_addr,
    output wire [3:0] rm_addr,

    // Register file write control
    output wire [3:0] wr_addr1,
    output wire wr_en1,
    output wire [3:0] wr_addr2,
    output wire wr_en2,

    // ALU control
    output wire [3:0] alu_op,
    output wire alu_src_b,
    output wire cpsr_wen,

    // Barrel shifter control
    output wire [1:0] shift_type,
    output wire [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount,
    output wire shift_src,

    // Immediate value
    output reg [31:0] imm32,

    // Memory access control
    output wire mem_read,
    output wire mem_write,
    output reg [1:0] mem_size,
    output wire mem_signed,

    // Address mode control
    output wire addr_pre_idx,
    output wire addr_up,
    output wire addr_wb,

    // Write back source select
    output reg [2:0] wb_sel,

    // Branch control
    output wire branch_en,
    output wire branch_link,
    output wire branch_exchange,

    // Multiply outputs — stubbed to zero (v1.3: MAC removed)
    output wire mul_en,
    output wire mul_long,
    output wire mul_signed,
    output wire mul_accumulate,

    // PSR transfer control
    output wire psr_rd,
    output wire psr_wr,
    output wire psr_field_sel,
    output wire [3:0] psr_mask,

    // Block data transfer control
    output wire [15:0] bdt_list,
    output wire bdt_load,
    output wire bdt_s,
    output wire bdt_wb,

    // Swap and SWI control
    output wire swap_byte,
    output wire swi_en,

    // Coprocessor control
    output wire cp_wen,
    output wire cp_ren,

    // Register usage flags
    output wire use_rn,
    output wire use_rd,
    output wire use_rs,
    output wire use_rm,

    // Multi-cycle instruction indication
    output wire is_multi_cycle
);

// ================================================================
// Step 1: Instruction field extraction
// ================================================================
wire [2:0] f_primary_opcode = instr[27:25];
wire [3:0] f_opcode = instr[24:21];
wire f_l = instr[20];
wire f_s = instr[20];
wire [3:0] f_rn = instr[19:16];
wire [3:0] f_rd = instr[15:12];
wire [3:0] f_rs = instr[11:8];
wire [4:0] f_shamt = instr[11:7];
wire [1:0] f_sh = instr[6:5];
wire f_bit7 = instr[7];
wire f_bit4 = instr[4];
wire [3:0] f_rm = instr[3:0];
wire [7:0] f_imm8 = instr[7:0];
wire [3:0] f_rot = instr[11:8];
wire [11:0] f_off12 = instr[11:0];
wire [23:0] f_off24 = instr[23:0];

assign rn_addr = f_rn;
assign rd_addr = f_rd;
assign rs_addr = f_rs;
assign rm_addr = f_rm;

// ================================================================
// Step 2: Instruction type decoding
// ================================================================
wire o000 = (f_primary_opcode == 3'b000);
wire o001 = (f_primary_opcode == 3'b001);
wire o010 = (f_primary_opcode == 3'b010);
wire o011 = (f_primary_opcode == 3'b011);
wire o100 = (f_primary_opcode == 3'b100);
wire o101 = (f_primary_opcode == 3'b101);
wire o111 = (f_primary_opcode == 3'b111);

wire b7_4_eq_1001 = (instr[7:4] == 4'b1001);
wire b7_and_b4 = f_bit7 & f_bit4;
wire sh_nonzero = (f_sh != 0);

// Multiply decode — kept for classification only
wire dec_mul = o000 & b7_4_eq_1001 & ~instr[24] & ~instr[23] & ~instr[22];
wire dec_mull = o000 & b7_4_eq_1001 & ~instr[24] & instr[23];

wire dec_swp = o000 & b7_4_eq_1001 & instr[24] & ~instr[23] & ~instr[21] & ~instr[20] & (f_rs == 4'd0);
wire dec_bx = (instr[27:4] == 24'b000100101111111111110001);

wire dec_mrs = o000 & (instr[24:23] == 2'b10) &
               ~instr[21] & ~instr[20] &
               (f_rn == 4'hF) & (f_off12 == 12'd0);

wire dec_msr_reg = o000 & (instr[24:23] == 2'b10) &
                   instr[21] & ~instr[20] &
                   (f_rd == 4'hF) & (instr[11:4] == 8'd0);

wire dec_msr_imm = o001 & (instr[24:23] == 2'b10) &
                   instr[21] & ~instr[20] & (f_rd == 4'hF);

wire hdt_pat = o000 & b7_and_b4 & sh_nonzero;
wire dec_hdt_rego = hdt_pat & ~instr[22];
wire dec_hdt_immo = hdt_pat & instr[22];

wire dp_reg_ok = o000 & (~f_bit4 | (f_bit4 & ~f_bit7));
wire dec_dp_reg = dp_reg_ok & ~dec_bx & ~dec_mrs & ~dec_msr_reg;
wire dec_dp_imm = o001 & ~dec_msr_imm;

wire dec_sdt_immo = o010;
wire dec_sdt_rego = o011 & ~f_bit4;

wire dec_bdt = o100;
wire dec_br = o101;
wire dec_swi = o111 & instr[24];

wire cp_num_match = (f_rs == 4'b1010);
wire dec_cp_base = o111 & ~instr[24] & f_bit4 & cp_num_match;
wire dec_mcr = dec_cp_base & ~f_l;
wire dec_mrc = dec_cp_base & f_l;

wire dec_valid = dec_dp_reg | dec_dp_imm | dec_mul | dec_mull |
                 dec_swp | dec_bx | dec_hdt_rego | dec_hdt_immo |
                 dec_sdt_immo | dec_sdt_rego | dec_bdt | dec_br |
                 dec_mrs | dec_msr_reg | dec_msr_imm | dec_swi |
                 dec_mcr | dec_mrc;

wire dec_undef = ~dec_valid;

assign t_dp_reg = dec_dp_reg;
assign t_dp_imm = dec_dp_imm;
assign t_mul = dec_mul;
assign t_mull = dec_mull;
assign t_swp = dec_swp;
assign t_bx = dec_bx;
assign t_hdt_rego = dec_hdt_rego;
assign t_hdt_immo = dec_hdt_immo;
assign t_sdt_immo = dec_sdt_immo;
assign t_sdt_rego = dec_sdt_rego;
assign t_bdt = dec_bdt;
assign t_br = dec_br;
assign t_mrs = dec_mrs;
assign t_msr_reg = dec_msr_reg;
assign t_msr_imm = dec_msr_imm;
assign t_swi = dec_swi;
assign t_undef = dec_undef;
assign t_mcr = dec_mcr;
assign t_mrc = dec_mrc;

// ================================================================
// Step 3: Convenience groups
// ================================================================
wire is_dp = dec_dp_reg | dec_dp_imm;
wire is_sdt = dec_sdt_immo | dec_sdt_rego;
wire is_hdt = dec_hdt_rego | dec_hdt_immo;
wire is_load = f_l;
wire dp_test = (f_opcode[3:2] == 2'b10);

// ================================================================
// Step 4: Immediate value generation
// ================================================================
wire [4:0] rot_amount = {f_rot, 1'b0};
wire [31:0] imm8_ext = {24'b0, f_imm8};
wire [31:0] imm_sdt = {20'b0, f_off12};
wire [31:0] imm_hdt = {24'b0, f_rs, f_rm};
wire [31:0] imm_dp = imm8_ext; // rotation deferred to EX1 barrel shifter
wire [31:0] imm_br = {{6{f_off24[23]}}, f_off24, 2'b00};

always @(*) begin
    if (dec_dp_imm | dec_msr_imm) imm32 = imm_dp;
    else if (dec_br) imm32 = imm_br;
    else if (dec_sdt_immo) imm32 = imm_sdt;
    else if (dec_hdt_immo) imm32 = imm_hdt;
    else imm32 = 32'b0;
end

// ================================================================
// Step 5: ALU control
// ================================================================
wire [3:0] alu_op_mem = addr_up ? 4'b0100 : 4'b0010;
assign alu_op = (is_dp) ? f_opcode :
                (is_sdt | is_hdt) ? alu_op_mem :
                (dec_msr_reg | dec_msr_imm) ? 4'b1101 :
                4'b0100;

assign alu_src_b = dec_dp_imm | dec_msr_imm | dec_sdt_immo | dec_hdt_immo;

/* v1.3: multiply removed from cpsr_wen */
assign cpsr_wen = cond_met & (is_dp & f_s);

// ================================================================
// Step 6: Barrel shifter control
// ================================================================
wire sh_active = dec_dp_reg | dec_sdt_rego;
wire imm_rot_active = (dec_dp_imm | dec_msr_imm) & (rot_amount != 5'd0);

assign shift_type = sh_active ? f_sh :
                    imm_rot_active ? `SHIFT_ROR : 2'b00;

assign shift_amount = sh_active ? f_shamt :
                      (dec_dp_imm | dec_msr_imm) ? rot_amount : 5'b0;

assign shift_src = dec_dp_reg & f_bit4;

// ================================================================
// Step 7: Memory control
// ================================================================
assign mem_read = cond_met & (is_sdt | is_hdt) & is_load;
assign mem_write = cond_met & (is_sdt | is_hdt) & ~is_load;

always @(*) begin
    if (is_sdt) mem_size = instr[22] ? 2'b00 : 2'b10;
    else if (is_hdt) begin
        case (f_sh)
            2'b01: mem_size = 2'b01;
            2'b10: mem_size = 2'b00;
            2'b11: mem_size = 2'b01;
            default: mem_size = 2'b10;
        endcase
    end
    else mem_size = 2'b10;
end

assign mem_signed = is_hdt & f_sh[1];

// ================================================================
// Step 8: Address mode control
// ================================================================
assign addr_pre_idx = instr[24];
assign addr_up = instr[23];
assign addr_wb = ~instr[24] | instr[21];

// ================================================================
// Step 9: Write-back source select
// v1.3: WB_MUL removed
// ================================================================
always @(*) begin
    if ((is_sdt | is_hdt) & is_load) wb_sel = `WB_MEM;
    else if (dec_br & instr[24]) wb_sel = `WB_LINK;
    else if (dec_mrs) wb_sel = `WB_PSR;
    else if (dec_mrc) wb_sel = `WB_CP;
    else wb_sel = `WB_ALU;
end

// ================================================================
// Step 10: Register write control
// v1.3: multiply writes removed
// ================================================================
assign wr_addr1 = (dec_br & instr[24]) ? 4'd14 : f_rd;

wire raw_we1 = (is_dp & ~dp_test) |
               ((is_sdt | is_hdt) & is_load) |
               dec_mrs |
               (dec_br & instr[24]) |
               dec_mrc;
assign wr_en1 = cond_met & raw_we1;

assign wr_addr2 = f_rn;
wire raw_we2 = (is_sdt | is_hdt) & addr_wb;
assign wr_en2 = cond_met & raw_we2;

// ================================================================
// Step 11: Branch control
// ================================================================
assign branch_en = cond_met & (dec_br | dec_bx);
assign branch_link = cond_met & dec_br & instr[24];
assign branch_exchange = cond_met & dec_bx;

// ================================================================
// Step 12: Multiply outputs — stubbed (v1.3: MAC removed)
// ================================================================
assign mul_en = 1'b0;
assign mul_long = 1'b0;
assign mul_signed = 1'b0;
assign mul_accumulate = 1'b0;

// ================================================================
// Step 13: PSR transfer control
// ================================================================
assign psr_rd = dec_mrs;
assign psr_wr = cond_met & (dec_msr_reg | dec_msr_imm);
assign psr_field_sel = instr[22];
assign psr_mask = f_rn;

// ================================================================
// Step 14: Block data transfer control
// ================================================================
assign bdt_list = instr[15:0];
assign bdt_load = f_l;
assign bdt_s = instr[22];
assign bdt_wb = instr[21];

// ================================================================
// Step 15: Swap and SWI control
// ================================================================
assign swap_byte = instr[22];
assign swi_en = cond_met & dec_swi;

// ================================================================
// Step 16: Coprocessor control
// ================================================================
assign cp_wen = cond_met & dec_mcr;
assign cp_ren = cond_met & dec_mrc;

// ================================================================
// Step 17: Register usage flags
// v1.3: multiply references removed
// ================================================================
assign use_rn = is_dp | is_sdt | is_hdt | dec_swp | dec_bdt;
assign use_rm = dec_dp_reg | dec_hdt_rego | dec_sdt_rego |
                dec_swp | dec_bx | dec_msr_reg;
assign use_rs = dec_dp_reg & f_bit4;
assign use_rd = ((is_sdt | is_hdt) & ~is_load) | dec_mcr;

// ================================================================
// Step 18: Multi-cycle instruction indication
// ================================================================
assign is_multi_cycle = cond_met & ((dec_bdt & |bdt_list) | dec_swp);

endmodule

`endif // CU_V