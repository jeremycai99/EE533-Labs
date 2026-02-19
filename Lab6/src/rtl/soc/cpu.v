/* file: cpu.v
 Description: Single thread 5-stage pipeline Arm CPU module
 Author: Jeremy Cai
 Date: Feb. 18, 2026
 Version: 1.0
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
    input wire [`DATA_WIDTH-1:0] d_mem_data_i,  // Data memory data input (64-bit)
    output wire [`DMEM_ADDR_WIDTH-1:0] d_mem_addr_o, // Data memory address output
    output wire [`DATA_WIDTH-1:0] d_mem_data_o, // Data memory data output (64-bit)
    output wire d_mem_wen_o,                    // Data memory write enable
    output wire cpu_done,                       // Signal to indicate CPU completion

    // ILA Debug Interface
    // Multiplexed Debug Port (Full 64-bit width)
    // [4] = 0: System Debug (Selects via [3:0])
    // [4] = 1: Register File Debug (Address via [2:0])
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

// Branch control signals
wire branch_taken_ex; // Branch effective in EX stage
wire [`PC_WIDTH-1:0] branch_target_ex; // Target address calculated in EX stage

// CPU pipeline stage internal signal definitions
// IF stage signals
wire [`PC_WIDTH-1:0] pc_if;
wire [`PC_WIDTH-1:0] pc_next_if;
wire [`PC_WIDTH-1:0] pc_plus4_if; // Use PC+4 here to match with arm 32bit instruction behavior and assembly
                                 // Will be truncated to smaller width for our small imem design
wire pc_en = ~stall_if; // PC enable is inverse of IF stall (stalling IF means holding PC)

assign pc_plus4_if = pc_if + 32'd4; // Increment PC by 4 for next instruction (32-bit aligned)
assign pc_next_if = branch_taken_ex ? branch_target_ex : pc_plus4_if; // Muxing logic for next PC: branch target or PC+4

pc u_pc (
    .clk(clk),
    .rst_n(rst_n),
    .pc_next(pc_next_if),
    .pc_en(pc_en),
    .pc_out(pc_if)
);

assign i_mem_addr_o = pc_if; // Instruction memory address is current PC (truncated as needed)

// CPU Done signal: Active when PC reaches max value (all 1s)
assign cpu_done = (pc_if == {`PC_WIDTH{1'b1}});

/* SPECIAL CONSIDERATION FOR SYNC-READ BEHAVIOR OF INSTRUCTION MEMORY */
wire [`INSTR_WIDTH-1:0] instr_id = i_mem_data_i; // Instruction fetched in IF stage, available in ID stage due to synchronous read

reg [`PC_WIDTH-1:0] pc_plus4_id; // Register to hold PC+4 for ID stage (for branch target calculation and debugging)
reg ifid_valid; // This valid bit willl be regarded as keeper logic to verify if the instruction output in the consequent clock should
                //be considered as flushed instruction or not.

// Keeper logic to keep 5 stages of pipeline design
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pc_plus4_id <= `PC_WIDTH'd0;
        ifid_valid <= 1'b0;
    end else if (flush_ifid) begin
        // Hold the current value (do not update)
        // On flush, the pc_plus4_id is not determined by the IF stage logic
        ifid_valid <= 1'b0;   // No change
    end else if (!stall_id) begin
        // On flush, we can choose to clear the valid bit to indicate no valid instruction in ID stage
        pc_plus4_id <= pc_plus4_if; // Optional: Clear PC+4 on flush (not strictly necessary)
        ifid_valid <= 1'b1;   // Mark as invalid instruction due to flush
    end else begin
        // Normal operation: hold pc_plus4_id and ifid_valid unchanged. Latch infer but no harm for now
    end
end


/*********************************************************
 ************   ID Stage Signals and Logic    ************
 *********************************************************/

reg [3:0] cpsr_flags; // Current Program Status Register flags (N, Z, C, V)
wire cond_met_raw; // Condition code evaluation result before considering instruction validity
wire cond_met_id; // Condition code evaluation result for ID stage (considering flush and valid bit)

cond_eval u_cond_eval (
    .cond_code(instr_id[31:28]), // Condition code from instruction
    .cpsr_flags(cpsr_flags), // Current CPSR flags
    .cond_met(cond_met_raw) // Condition met output
);

assign cond_met_id = cond_met_raw && ifid_valid; // Condition is met only if raw condition is true, instruction is valid, and not being flushed

/* Start Control Unit (CU) Signal Declaration*/
wire t_dp_reg, t_dp_imm, t_mul, t_mull, t_swp, t_bx;
wire t_hdt_rego, t_hdt_immo, t_sdt_rego, t_sdt_immo;
wire t_bdt, t_br, t_mrs, t_msr_reg, t_msr_imm, t_swi, t_undef;

wire [3:0] rn_addr_id, rd_addr_id, rs_addr_id, rm_addr_id;

wire [3:0] wr_addr1_id, wr_addr2_id;
wire wr_en1_id, wr_en2_id;

wire [3:0] alu_op_id;
wire alu_src_b_id; // 0 = register, 1 = immediate
wire cpsr_wen_id; // CPSR write enable

wire [1:0] shift_type_id;
wire [`SHIFT_AMOUNT_WIDTH-1:0] shift_amount_id;
wire shift_src_id; // 0 = immediate shift, 1 = register shift

wire [31:0] imm32_id; // 32-bit immediate value after decoding

wire mem_read_id; // Memory read enable
wire mem_write_id; // Memory write enable
wire [1:0] mem_size_id; // Memory access size (00=byte,
wire mem_signed_id; // Memory access signed (for loads)

wire addr_pre_idx_id, addr_up_id, addr_wb_id; // Addressing mode control signals for load/store

wire [2:0] wb_sel_id; // Writeback select signal for choosing what goes to register write data

wire branch_en_id, branch_link_id, branch_exchange_id; // Branch control signals

wire mul_en_id, mul_long_id, mul_signed_id, mul_accumulate_id; // Multiply unit control signals

wire psr_rd_id, psr_wr_id, psr_field_sel_id; // PSR access control signals

wire [3:0] psr_mask_id; // Mask for which PSR flags to update (N, Z, C, V)

wire [15:0] bdt_list_id; // Block data transfer register list
wire bdt_load_id, bdt_s_id, bdt_wb_id; // Block data transfer control signals


wire swap_byte_id; // Byte/word control signal for SWP instruction: 1 for byte swap (SWPB), 0 for word swap (SWP)
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
    .wr_en1(wr_en1_id), // Primary writeback register and enable
    .wr_addr2(wr_addr2_id),
    .wr_en2(wr_en2_id), // Secondary writeback register and enable (for long multiply)
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

// Port 3 is shared: Rd (store data / MLA accum) or Rs (shift reg /
// MUL multiplier).  When BDTU is busy the ID stage is stalled,
// so port 3 is free for BDTU register reads (STM data).
wire [3:0] r3addr_id = use_rd_id ? rd_addr_id : rs_addr_id;

wire [3:0] bdtu_rf_rd_addr; // Driven by BDTU instance below
wire [3:0] r3addr_mux = bdtu_busy ? bdtu_rf_rd_addr : r3addr_id;

wire [31:0] rn_data_id, rm_data_id, r3_data_id;

// Write-back signals from WB stage
wire [3:0]  wb_wr_addr1, wb_wr_addr2;
wire [31:0] wb_wr_data1, wb_wr_data2;
wire wb_wr_en1,   wb_wr_en2;

// BDTU write-back signals
wire [3:0]  bdtu_wr_addr1, bdtu_wr_addr2;
wire [31:0] bdtu_wr_data1, bdtu_wr_data2;
wire bdtu_wr_en1,   bdtu_wr_en2;

// Merged regfile write ports: BDTU has priority when busy.
// The regfile has a single wena that gates BOTH write ports, so
// when only one port is active we mirror it to avoid garbage writes.
wire rf_wr_en = bdtu_busy ? (bdtu_wr_en1 | bdtu_wr_en2) : (wb_wr_en1  | wb_wr_en2);

wire [3:0] rf_wr_addr1 = bdtu_busy ? bdtu_wr_addr1 : (wb_wr_en1 ? wb_wr_addr1 : wb_wr_addr2);
wire [31:0] rf_wr_data1 = bdtu_busy ? bdtu_wr_data1 : (wb_wr_en1 ? wb_wr_data1 : wb_wr_data2);

wire [3:0] rf_wr_addr2 = bdtu_busy ? bdtu_wr_addr2 : (wb_wr_en2 ? wb_wr_addr2 : rf_wr_addr1);
wire [31:0] rf_wr_data2 = bdtu_busy ? bdtu_wr_data2 : (wb_wr_en2 ? wb_wr_data2 : rf_wr_data1);

// Debug
wire [31:0] debug_reg_out;
/* End Register File (RF) Signals */

regfile u_regfile (
    .clk (clk),
    .r1addr (rn_addr_id),
    .r2addr (rm_addr_id),
    .r3addr (r3addr_mux),
    .wena (rf_wr_en),
    .wr_addr1 (rf_wr_addr1),
    .wr_data1 (rf_wr_data1),
    .wr_addr2 (rf_wr_addr2),
    .wr_data2 (rf_wr_data2),
    .r1data (rn_data_id),
    .r2data (rm_data_id),
    .r3data (r3_data_id),
    .ila_cpu_reg_addr (ila_debug_sel[`REG_ADDR_WIDTH-1:0]),
    .ila_cpu_reg_data (debug_reg_out)
);

//When executing an ARM instruction, PC reads as the address of the current instruction plus 8.
// Reference link: https://developer.arm.com/documentation/ddi0406/c/Application-Level-Architecture/Application-Level-Programmers--Model/ARM-core-registers
// Provide PC+8 when reading R15 (ARM convention: R15 = PC+8)
wire [31:0] rn_data_pc_adj = (rn_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rn_data_id;
wire [31:0] rm_data_pc_adj = (rm_addr_id == 4'd15) ? (pc_plus4_id + 32'd4) : rm_data_id;




// Newly added ILA Debug Interface logic
// Now using full `DATA_WIDTH` (64 bits) for output.
// Signals smaller than 64 bits are zero-padded.
always @(*) begin
    if (ila_debug_sel[4]) begin
        // Mode 1: Register File Debug (MSB = 1)
        // Address is taken from ila_debug_sel[2:0] which is already wired to regfile
        ila_debug_data = debug_reg_out;
    end else begin
        // Mode 0: System Debug (MSB = 0)
        case (ila_debug_sel[3:0])
            // 0: Program Counter (Fetch) - 9 bits
            4'd0: ila_debug_data = { {`DATA_WIDTH-`PC_WIDTH{1'b0}}, pc_if };
            
            // 1: Instruction (Decode) - 32 bits
            4'd1: ila_debug_data = { {`DATA_WIDTH-`INSTR_WIDTH{1'b0}}, instr_id };
            
            // 2: Register Read Data A (Decode) - Full 64 bits
            4'd2: ila_debug_data = r0_out_id;
            
            // 3: Register Read Data B (Decode) - Full 64 bits
            4'd3: ila_debug_data = r1_out_id;
            
            // 4: EX Stage Result / Address (Execute) - Full 64 bits
            4'd4: ila_debug_data = r0_out_ex;
            
            // 5: EX Stage Write Data (Execute) - Full 64 bits
            4'd5: ila_debug_data = r1_out_ex;
            
            // 6: Writeback Data / Memory Read Data (Writeback) - Full 64 bits
            4'd6: ila_debug_data = d_mem_data_i;
            
            // 7: Control Signals Vector
            // [0]: Reg Write ID, [1]: Mem Write ID, [2]: Mem Write Mem, [3]: Reg Write WB
            4'd7: ila_debug_data = { {`DATA_WIDTH-4{1'b0}}, reg_write_wb, mem_write_mem, mem_write_id, reg_write_id };

            // 8: Destination Register Address (Decode) - 3 bits
            4'd8: ila_debug_data = { {`DATA_WIDTH-`REG_ADDR_WIDTH{1'b0}}, rdaddr_id };

            // 9: Destination Register Address (Writeback) - 3 bits
            4'd9: ila_debug_data = { {`DATA_WIDTH-`REG_ADDR_WIDTH{1'b0}}, rdaddr_wb };

            default: ila_debug_data = {`DATA_WIDTH{1'b1}}; // All 1s for invalid selection
        endcase
    end
end

endmodule

`endif // CPU_V