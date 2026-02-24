/* file: cu.v
 Description: Control unit module for the Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 23, 2026
 Version: 1.1
 Changes from 1.0:
   - Rotated immediate fix: imm_dp no longer computes the rotation
     combinationally.  The raw 8-bit immediate (zero-extended) is
     output as imm32, and the rotation amount is routed to the barrel
     shifter control signals (shift_type = ROR, shift_amount = rot_amount).
     The EX1 barrel shifter performs the rotation from registered inputs,
     eliminating the ~9ns combinational path that violated 8ns timing.
   - This also corrects the shifter_carry_out for MOVS/ANDS/etc with
     rotated immediates (previously always cin; now correct per ARM spec).
 */

`ifndef CU_V
`define CU_V

`include "define.v"
`include "cond_eval.v"

// Pure combinational control unit and offload the sequential control for multi-cycle instruction to 
// BDTU to simplify this design.
module cu (
    // Keep input of control signals simply the instruction and condition evaluation results. The condition evaluation is done in cond_eval module, and the output control signals are generated based on the instruction decoding and condition evaluation results.
    input wire [`INSTR_WIDTH-1:0] instr, // Input instruction from IF/ID pipeline register
    input wire cond_met, // Condition evaluation result from cond_eval module
    /* Large fanout control signals*/
    // Instruction type decoding outputs
    output wire t_dp_reg, // Data processing register operand
    output wire t_dp_imm, // Data processing immediate operand
    output wire t_mul, // Multiply instruction
    output wire t_mull, // Multiply long instruction
    output wire t_swp, // Single data swap instruction
    output wire t_bx, // Branch and exchange instruction
    output wire t_hdt_rego, // Halfword data transfer with register offset
    output wire t_hdt_immo, // Halfword data transfer with immediate offset
    output wire t_sdt_rego, // Single data transfer with register offset
    output wire t_sdt_immo, // Single data transfer with immediate offset
    output wire t_bdt, // Block data transfer instruction
    output wire t_br, // Branch instruction
    output wire t_mrs, // Move from PSR instruction
    output wire t_msr_reg, // Move to PSR instruction with register operand
    output wire t_msr_imm, // Move to PSR instruction with immediate operand
    output wire t_swi, // Software interrupt instruction
    output wire t_undef, // Undefined instruction

    // Register file addresses
    output wire [3:0] rn_addr, // Rn register address for reading
    output wire [3:0] rd_addr, // Rd register address for reading/writing
    output wire [3:0] rs_addr, // Rs register address for reading (for multiply instructions)
    output wire [3:0] rm_addr, // Rm register address for reading (for data processing instructions with register operand, and multiply instructions)

    // Register file write control signals for single-cycle instructions
    // Please be aware that for instructions that have W flag set, the write back to register file
    // happens so eventually we end up with two write ports in the register file and a wider pipeline register to accommodate this complex datapath design
    output wire [3:0] wr_addr1,
    output wire wr_en1,
    output wire [3:0] wr_addr2,
    output wire wr_en2,

    // ALU control signals
    output wire [3:0] alu_op, // ALU operation code for data processing
    output wire alu_src_b, // ALU source B select: 0 for register operand, 1 for immediate operand
    output wire cpsr_wen, // Control signal to indicate whether the instruction updates CPSR flags

    // Barrel shifter control signals
    output wire [1:0] shift_type, // Barrel shifter type: 00 for LSL, 01 for LSR, 10 for ASR, 11 for ROR
    output wire [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount, // Barrel shifter amount (5 bits to support shifts up to 31)
    output wire shift_src, // Barrel shifter source select: 0 for immediate shift amount, 1 for register specified shift amount

    // Immediate Value
    output reg [31:0] imm32, // 32-bit immediate value extracted from instruction for data processing immediate instructions and branch instructions (after sign extension for branch offset)

    //Memory access control signals
    output wire mem_read, // Memory read enable signal for load instructions
    output wire mem_write, // Memory write enable signal for store instructions
    output reg [1:0] mem_size, // Memory access size: 00 for byte, 01 for halfword, 10 for word (for single data transfer instructions)
    output wire mem_signed, // Memory access signed/unsigned control: 0 for unsigned, 1 for signed (for single data transfer instructions)

    //Address mode control signals
    output wire addr_pre_idx, // Addressing mode control signal for pre-indexing (1) vs post-indexing (0) for single data transfer and block data transfer instructions
    output wire addr_up, // Addressing mode control signal for up (1) vs down (0) for single data transfer and block data transfer instructions
    output wire addr_wb, // Addressing mode control signal for write-back (1) vs no write-back (0) for single data transfer and block data transfer instructions
    
    // Write back source select signal
    output reg [2:0] wb_sel,

    // Branch control signals
    output wire branch_en,
    output wire branch_link,
    output wire branch_exchange,

    // Multiply control signals (not used in this lab) 
    output wire mul_en,
    output wire mul_long,
    output wire mul_signed,
    output wire mul_accumulate,

    //PSR transfer control signals (not used in this lab)
    output wire psr_rd,
    output wire psr_wr,
    output wire psr_field_sel,
    output wire [3:0] psr_mask,

    // Block data transfer control signals output to BDTU
    output wire [15:0] bdt_list, // Register list for block data transfer instructions (LDM/STM)
    output wire bdt_load, // Load/store control signal for block data transfer instructions: 1 for load (LDM), 0 for store (STM)
    output wire bdt_s, // PSR transfer control signal for block data transfer instructions: 1 to update CPSR with the value from the last loaded register in LDM, 0 to not update CPSR
    output wire bdt_wb, // Write-back control signal for block data transfer instructions: 1 to write back the updated base address to Rn, 0 to not write back

    //Swap and software interrupt control signals (not used in this lab)
    output wire swap_byte, // Byte/word control signal for SWP instruction: 1 for byte swap (SWPB), 0 for word swap (SWP)
    output wire swi_en, // Enable signal for software interrupt instruction (SWI)

    //Register usage flags for hazard detection and forwarding unit
    output wire use_rn,
    output wire use_rd,
    output wire use_rs,
    output wire use_rm,

    // Multi-cycle instruction indication
    output wire is_multi_cycle
);

// Step 1: Instruction field extraction
wire [2:0] f_primary_opcode = instr[27:25]; // Primary opcode field for instruction type decoding. Naming of this field is not official
wire [3:0] f_opcode = instr[24:21]; // Opcode field for data processing instruction decoding
wire f_l = instr[20]; // L bit for load
wire f_s = instr[20]; // S bit for seting condition codes
wire [3:0] f_rn = instr[19:16]; // Rn register address field
wire [3:0] f_rd = instr[15:12]; // Rd register address field
wire [3:0] f_rs = instr[11:8]; // Rs register address field (for multiply instructions)
wire [4:0] f_shamt = instr[11:7]; // Shift amount field for data processing instructions with immediate shift amount
wire [1:0] f_sh = instr[6:5]; // Shift type field for data processing instructions with immediate shift amount.  See encoding definition in define.v
wire f_bit7 = instr[7]; // Bit 7 for halfword data transfer instruction decoding
wire f_bit4 = instr[4]; // Bit 4 for halfword data transfer instruction decoding
wire [3:0] f_rm = instr[3:0]; // Rm register address field for data processing instructions with register operand, and for multiply instructions
wire [7:0] f_imm8 = instr[7:0]; // Immediate value field for data processing immediate instructions
wire [3:0] f_rot = instr[11:8]; // Rotate field for data processing immediate instructions (the actual rotate amount is this field multiplied by 2)
wire [11:0] f_off12 = instr[11:0]; // Offset value field for single data transfer instructions (the actual immediate value is this field after processing the I bit for immediate offset or register offset)
wire [23:0] f_off24 = instr[23:0]; // Offset value field for branch instructions (the actual immediate value is this field sign-extended and multiplied by 4)

// Connect register address to differnt fields
assign rn_addr = f_rn;
assign rd_addr = f_rd;
assign rs_addr = f_rs;
assign rm_addr = f_rm;

// Step 2: Instruction type decoding
wire o000 = (f_primary_opcode == 3'b000);
wire o001 = (f_primary_opcode == 3'b001);
wire o010 = (f_primary_opcode == 3'b010);
wire o011 = (f_primary_opcode == 3'b011);
wire o100 = (f_primary_opcode == 3'b100);
wire o101 = (f_primary_opcode == 3'b101);
wire o111 = (f_primary_opcode == 3'b111);

wire b7_4_eq_1001 = (instr[7:4] == 4'b1001); //Bit 7 to 4 equal to 1001 for multiply instructions
wire b7_and_b4 = f_bit7 & f_bit4; // Bit 7 and bit 4

wire sh_nonzero = (f_sh != 0); // Shift type not equal to zero for data processing instructions.

// Determine multipy insructions based on current encoding scheme.
wire dec_mul = o000 & b7_4_eq_1001 & ~instr[24] & ~instr[23] & ~instr[22];
wire dec_mull = o000 & b7_4_eq_1001 & ~instr[24] & instr[23];

// Determine single data swap instruction based on current encoding scheme
wire dec_swp = o000 & b7_4_eq_1001 & instr[24] & ~instr[23] & ~instr[21] & ~instr[20] & (f_rs == 4'd0); //Skip instr[22] B field

// Determine branch exchange instruction
wire dec_bx = (instr[27:4] == 24'b000100101111111111110001); // Fixed encoding for BX instruction based on Armv4T encoding. 24'h 12FFF1 in hexadecimal

wire dec_mrs = o000 & (instr[24:23] == 2'b10) &
                ~instr[21] & ~instr[20] &
                (f_rn == 4'hF) & (f_off12 == 12'd0);

wire dec_msr_reg = o000 & (instr[24:23] == 2'b10) &
                    instr[21] & ~instr[20] &
                    (f_rd == 4'hF) & (instr[11:4] == 8'd0);

wire dec_msr_imm = o001 & (instr[24:23] == 2'b10) &
                    instr[21] & ~instr[20] & (f_rd == 4'hF);

// Halfword Data Transfer
wire hdt_pat    = o000 & b7_and_b4 & sh_nonzero;
wire dec_hdt_rego = hdt_pat & ~instr[22];
wire dec_hdt_immo = hdt_pat &  instr[22];

// Data Processing (register)
wire dp_reg_ok = o000 & (~f_bit4 | (f_bit4 & ~f_bit7));
wire dec_dp_reg  = dp_reg_ok & ~dec_bx & ~dec_mrs & ~dec_msr_reg;

// Data Processing (immediate)
wire dec_dp_imm = o001 & ~dec_msr_imm;

// Single Data Transfer
wire dec_sdt_immo = o010;
wire dec_sdt_rego = o011 & ~f_bit4;

// Block Data Transfer / Branch / SWI
wire dec_bdt = o100;
wire dec_br  = o101;
wire dec_swi = o111 & instr[24];

// Undefined instruction identification
wire dec_valid = dec_dp_reg  | dec_dp_imm  | dec_mul    | dec_mull   |
                    dec_swp   | dec_bx    | dec_hdt_rego | dec_hdt_immo |
                    dec_sdt_immo | dec_sdt_rego | dec_bdt  | dec_br     |
                    dec_mrs   | dec_msr_reg | dec_msr_imm  | dec_swi;

wire dec_undef = ~dec_valid;

assign t_dp_reg  = dec_dp_reg;
assign t_dp_imm  = dec_dp_imm;
assign t_mul     = dec_mul;
assign t_mull    = dec_mull;
assign t_swp     = dec_swp;
assign t_bx      = dec_bx;
assign t_hdt_rego  = dec_hdt_rego;
assign t_hdt_immo  = dec_hdt_immo;
assign t_sdt_immo  = dec_sdt_immo;
assign t_sdt_rego  = dec_sdt_rego;
assign t_bdt     = dec_bdt;
assign t_br      = dec_br;
assign t_mrs     = dec_mrs;
assign t_msr_reg = dec_msr_reg;
assign t_msr_imm = dec_msr_imm;
assign t_swi     = dec_swi;
assign t_undef   = dec_undef;

// Step 3: Identify convenience groups
wire is_dp = dec_dp_reg | dec_dp_imm;
wire is_sdt = dec_sdt_immo | dec_sdt_rego;
wire is_hdt = dec_hdt_rego | dec_hdt_immo;
wire is_load = f_l;
wire dp_test = (f_opcode[3:2] == 2'b10);

// Step 4: Immediate value generation
wire [4:0] rot_amount = {f_rot, 1'b0};
wire [31:0] imm8_ext = {24'b0, f_imm8}; // Zero-extend the 8-bit immediate value to 32 bits
wire [31:0] imm_sdt = {20'b0, f_off12};
wire [31:0] imm_hdt = {24'b0, f_rs, f_rm}; 

/* v1.1 FIX: Rotation deferred to EX1 barrel shifter.
 *
 * OLD (timing violation — ~9ns combinational path in ID stage):
 *   wire [31:0] imm_dp = (rot_amount == 0) ? imm8_ext
 *       : (imm8_ext >> rot_amount) | (imm8_ext << (32 - rot_amount));
 *
 * NEW: Output the raw unrotated immediate.  The rotation is performed
 * by the EX1 barrel shifter using shift_type=ROR, shift_amount=rot_amount
 * (see Step 6 below).  The barrel shifter operates on registered inputs
 * from the ID/EX1 pipe reg, well within the 8ns timing budget (~3-4ns).
 */
wire [31:0] imm_dp = imm8_ext;

wire [31:0] imm_br = {{6{f_off24[23]}}, f_off24, 2'b00}; // Sign-extend the 24-bit branch offset to 32 bits and shift left by 2 (multiply by 4)

always @(*) begin
    if (dec_dp_imm | dec_msr_imm) begin
        imm32 = imm_dp;
    end else if (dec_br) begin
        imm32 = imm_br;
    end else if (dec_sdt_immo) begin
        imm32 = imm_sdt; // For single data transfer instructions with immediate offset, the immediate value is the zero-extended 12-bit offset field (the I bit is already processed in the instruction type decoding to determine whether it's immediate offset or register offset)
    end else if (dec_hdt_immo) begin
        imm32 = imm_hdt; // For halfword data transfer instructions, the immediate value is constructed from the shift type field and bit 7 and bit 4 based on the current encoding scheme for halfword data transfer instructions
    end else begin
         imm32 = 32'b0; // Default value when immediate is not used
    end
end

// Step 5: Generate ALU control signals
wire [3:0] alu_op_mem = addr_up ? 4'b0100 : 4'b0010; // ADD : SUB
assign alu_op = (is_dp) ? f_opcode :
                (is_sdt | is_hdt) ? alu_op_mem :
                (dec_msr_reg | dec_msr_imm) ? 4'b1101 : // MOV: pass operand_b through
                                              4'b0100; // Default ADD for others

assign alu_src_b = dec_dp_imm | dec_msr_imm | dec_sdt_immo | dec_hdt_immo; // Use immediate value for data processing immediate instructions, move to PSR immediate instructions, single data transfer with immediate offset, and halfword data transfer with immediate offset
assign cpsr_wen = cond_met & ((is_dp | dec_mul | dec_mull) & f_s);

// Step 6: Generate barrel shifter control signals
/* v1.1 FIX: For dp_imm / msr_imm with non-zero rotation, route the
 * rotation through the EX1 barrel shifter instead of computing it
 * combinationally in imm_dp.
 *
 * Priority: sh_active (dp_reg, sdt_rego) > imm_rot_active (dp_imm, msr_imm)
 * These are mutually exclusive instruction types, so no conflict.
 *
 * When imm_rot_active:
 *   shift_type   = ROR  (rotate right)
 *   shift_amount = rot_amount = {f_rot, 1'b0}  (0-30 in steps of 2)
 *   shift_src    = 0  (immediate shift amount, not register)
 *
 * cpu_mt.v EX1 must feed imm32 (instead of Rm) into the barrel shifter
 * when alu_src_b=1.  See cpu_mt.v v2.6 changes.
 */
wire sh_active = dec_dp_reg | dec_sdt_rego;
wire imm_rot_active = (dec_dp_imm | dec_msr_imm) & (rot_amount != 5'd0);

assign shift_type = sh_active      ? f_sh       :
                    imm_rot_active  ? `SHIFT_ROR : 2'b00;

assign shift_amount = sh_active             ? f_shamt    :
                      (dec_dp_imm | dec_msr_imm) ? rot_amount : 5'b0;

assign shift_src = dec_dp_reg & f_bit4; // For data processing instructions with register operand, if bit 4 is 1 then the shift amount is specified in Rs, otherwise it's specified in the immediate shift amount field

// Step 7: Single-cycle memory control signal generation
assign mem_read = cond_met & (is_sdt | is_hdt) & is_load;
assign mem_write = cond_met & (is_sdt | is_hdt) & ~is_load;

always @(*) begin
    if (is_sdt)
        mem_size = instr[22] ? 2'b00 : 2'b10; // LDRB/STRB : LDR/STR
    else if (is_hdt) begin
        case (f_sh)
            2'b01:   mem_size = 2'b01; // Unsigned half
            2'b10:   mem_size = 2'b00; // Signed byte
            2'b11:   mem_size = 2'b01; // Signed half
            default: mem_size = 2'b10;
        endcase
    end
    else
        mem_size = 2'b10; // default word
end

assign mem_signed = is_hdt & f_sh[1];

// Step 8: Addressing mode control signal generation
assign addr_pre_idx = instr[24]; // P bit
assign addr_up  = instr[23]; // U bit
assign addr_wb  = ~instr[24] | instr[21]; // post-ix or W

// Step 9: Write-back source select signal generation
always @(*) begin
    if ((is_sdt | is_hdt) & is_load)
                                        wb_sel = 3'b001;  // memory data
    else if (dec_br & instr[24])     wb_sel = 3'b010;  // BL return addr
    else if (dec_mrs)                wb_sel = 3'b011;  // PSR value
    else if (dec_mul | dec_mull)     wb_sel = 3'b100;  // multiplier
    else                             wb_sel = 3'b000;  // ALU result
end

// Step 10: Register write signal generation (Also see BDTU unit for multi-cycle instruction register write control)
assign wr_addr1 = dec_mul ? f_rn  :    // MUL/MLA dest
                 (dec_br & instr[24]) ? 4'd14 :    // BL -> R14
                 f_rd;
// --- Primary write enable ---
wire raw_we1 = (is_dp & ~dp_test)            | // DP (non-test)
                ((is_sdt | is_hdt) & is_load) | // LDR / LDRH / …
                dec_mul                        | // MUL / MLA
                dec_mull                       | // MULL / MLAL
                dec_mrs                        | // MRS
                (dec_br & instr[24]); // BL (link)
assign wr_en1 = cond_met & raw_we1;

assign wr_addr2 = f_rn;
// --- Secondary write enable ---
wire raw_we2 = ((is_sdt | is_hdt) & addr_wb) | // SDT/HDT writeback
                dec_mull;
assign wr_en2 = cond_met & raw_we2;

// Step 11: Branch control signal generation
assign branch_en   = cond_met & (dec_br | dec_bx);
assign branch_link = cond_met & dec_br & instr[24];
assign branch_exchange = cond_met & dec_bx;

// Step 12: Multiply control signal generation (not used in this lab)
assign mul_en  = cond_met & (dec_mul | dec_mull);
assign mul_long = dec_mull;
assign mul_signed = dec_mull & instr[22];
assign mul_accumulate = (dec_mul | dec_mull) & instr[21];

// Step 13: PSR transfer control signal generation
assign psr_rd = dec_mrs;
assign psr_wr = cond_met & (dec_msr_reg | dec_msr_imm);
assign psr_field_sel = instr[22]; 
assign psr_mask = f_rn; // field mask

// Step 14: Block data transfer control signal generation
assign bdt_list  = instr[15:0];
assign bdt_load = f_l;  // bit 20 = L
assign bdt_s = instr[22]; // S bit
assign bdt_wb = instr[21]; // W bit

// Step 15: Swap and software interrupt control signal generation (not used in this lab)
assign swap_byte = instr[22]; // 1 = SWPB
assign swi_en = cond_met & dec_swi;

// Step 16: Register usage flag generation for hazard detection and forwarding unit
assign use_rn = is_dp | is_sdt | is_hdt | dec_swp | dec_bdt |
                (dec_mull & instr[21]);              // MLAL acc RdHi

assign use_rm = dec_dp_reg | dec_hdt_rego | dec_sdt_rego |
                dec_mul  | dec_mull   | dec_swp     | dec_bx |
                dec_msr_reg;

assign use_rs = (dec_dp_reg & f_bit4) | dec_mul | dec_mull;

assign use_rd = ((is_sdt | is_hdt) & ~is_load)  |  // STR data
                (dec_mul  & instr[21])            |  // MLA accumulate
                (dec_mull & instr[21]);              // MLAL acc RdLo

// Step 17: Multi-cycle instruction indication
assign is_multi_cycle = cond_met & ((dec_bdt & |bdt_list) | dec_swp);

endmodule

`endif // CU_V