/* file: sm_decoder.v
 Description: This file implements the instruction decoder for the CUDA-like SM core.
 The decoder takes a 32-bit instruction from the instruction memory and decodes it into control signals for the rest of the pipeline.
 Author: Jeremy Cai
 Date: Feb. 28, 2026
 Version: 1.1
 Revision history:
    - Feb. 28, 2026: v1.0 — Initial implementation.
    - Mar. 04, 2026: v1.1 — Fix OP_SET decoder signals
*/

`ifndef SM_DECODER_V
`define SM_DECODER_V

`include "gpu_define.v"

module sm_decoder (
    // Input: 32-bit instruction word (from IMEM BRAM)
    input wire [31:0] ir,

    // Field extraction
    output wire [4:0] dec_opcode,
    output wire dec_dt,
    output wire [1:0] dec_cmp_mode,
    output wire [3:0] dec_rD_addr,
    output wire [3:0] dec_rA_addr,
    output wire [3:0] dec_rB_addr,
    output wire [3:0] dec_rC_addr,
    output wire [15:0] dec_imm16,

    // SP core control (broadcast to all 4 SPs)
    output wire dec_rf_we,
    output wire dec_pred_we,
    output wire [1:0] dec_pred_wr_sel,
    output wire [1:0] dec_pred_rd_sel,
    output wire [2:0] dec_wb_src,
    output wire dec_use_imm,

    // Scoreboard interface
    output wire dec_uses_rA,
    output wire dec_uses_rB,
    output wire dec_is_fma,
    output wire dec_is_st,

    // Branch / control flow
    output wire dec_is_branch,
    output wire dec_is_pbra,
    output wire dec_is_ret,

    // Memory classification
    output wire dec_is_ld,
    output wire dec_is_store,
    output wire dec_is_lds,
    output wire dec_is_sts,

    // WMMA classification (Phase 3)
    output wire dec_is_wmma_mma,
    output wire dec_is_wmma_load,
    output wire dec_is_wmma_store,
    output wire [1:0] dec_wmma_sel,

    // Branch target
    output wire [`GPU_PC_WIDTH-1:0] dec_branch_target
);

    // ================================================================
    // Field extraction
    // ================================================================
    wire [4:0] opcode = ir[31:27];
    wire dt = ir[26];
    wire [1:0] res = ir[25:24];
    wire [3:0] rD = ir[23:20];
    wire [3:0] rA = ir[19:16];
    wire [3:0] rB = ir[15:12];
    wire [3:0] rC = ir[11:8];

    assign dec_opcode = opcode;
    assign dec_dt = dt;
    assign dec_cmp_mode = res;
    assign dec_rD_addr = rD;
    assign dec_rA_addr = rA;
    assign dec_rB_addr = rB;
    assign dec_rC_addr = rC;
    assign dec_wmma_sel = res;

    assign dec_imm16 = ir[15:0];

    // ================================================================
    // Opcode classification
    // ================================================================

    // Memory
    assign dec_is_ld  = (opcode == `OP_LD);
    assign dec_is_store = (opcode == `OP_ST);
    assign dec_is_lds = (opcode == `OP_LDS);
    assign dec_is_sts = (opcode == `OP_STS);

    // Control flow
    assign dec_is_branch = (opcode == `OP_BRA);
    assign dec_is_pbra = (opcode == `OP_PBRA);
    assign dec_is_ret = (opcode == `OP_RET);

    // WMMA (Phase 3)
    assign dec_is_wmma_mma = (opcode == `WMMA_MMA);
    assign dec_is_wmma_load = (opcode == `WMMA_LOAD);
    assign dec_is_wmma_store = (opcode == `WMMA_STORE);

    // FMA: rD is also a read source (accumulator)
    assign dec_is_fma = (opcode == `OP_FMA);

    // ST/STS: rD field is store data source (read), not write dest
    assign dec_is_st = dec_is_store | dec_is_sts | dec_is_wmma_store;

    // ================================================================
    // Register file write enable
    // ================================================================
    // Most ALU/FPU ops write rD. Exceptions: NOP, ST, STS, BRA, PBRA,
    // RET, SETP/SET (write pred RF not GPR), WMMA (handled separately).
    reg rf_we_r;
    always @(*) begin
        case (opcode)
            `OP_NOP:  rf_we_r = 1'b0;
            `OP_ST:   rf_we_r = 1'b0;
            `OP_STS:  rf_we_r = 1'b0;
            `OP_SETP: rf_we_r = 1'b0;
            `OP_SET:  rf_we_r = 1'b0;  // v1.1: SET writes pred, not GPR
            `OP_BRA:  rf_we_r = 1'b0;
            `OP_PBRA: rf_we_r = 1'b0;
            `OP_RET:  rf_we_r = 1'b0;
            `WMMA_MMA:   rf_we_r = 1'b0;
            `WMMA_LOAD:  rf_we_r = 1'b0;
            `WMMA_STORE: rf_we_r = 1'b0;
            default:  rf_we_r = 1'b1;
        endcase
    end
    assign dec_rf_we = rf_we_r;

    // ================================================================
    // Predicate RF write (SETP + SET)
    // ================================================================
    // v1.1: include OP_SET — writes immediate val to pred register
    assign dec_pred_we = (opcode == `OP_SETP) | (opcode == `OP_SET);
    assign dec_pred_wr_sel = rD[1:0];

    // ================================================================
    // Predicate RF read select
    // ================================================================
    assign dec_pred_rd_sel = dec_is_pbra ? ir[26:25] : res;

    // ================================================================
    // Source register usage (for scoreboard hazard check)
    // ================================================================
    reg uses_rA_r;
    always @(*) begin
        case (opcode)
            `OP_NOP, `OP_MOVI, `OP_SET, `OP_BRA, `OP_PBRA, `OP_RET:
                uses_rA_r = 1'b0;
            `WMMA_MMA:
                uses_rA_r = 1'b0;
            default:
                uses_rA_r = 1'b1;
        endcase
    end
    assign dec_uses_rA = uses_rA_r;

    reg uses_rB_r;
    always @(*) begin
        case (opcode)
            `OP_ADD, `OP_SUB, `OP_MUL, `OP_FMA,
            `OP_MAX, `OP_MIN,
            `OP_AND, `OP_OR, `OP_XOR,
            `OP_SETP, `OP_SELP:
                uses_rB_r = 1'b1;
            default:
                uses_rB_r = 1'b0;
        endcase
    end
    assign dec_uses_rB = uses_rB_r;

    // ================================================================
    // Immediate select (opB = imm16 instead of rB)
    // ================================================================
    reg use_imm_r;
    always @(*) begin
        case (opcode)
            `OP_MOVI: use_imm_r = 1'b1;
            `OP_SHL, `OP_SHR: use_imm_r = 1'b1;
            `OP_ADDI, `OP_MULI: use_imm_r = 1'b1;
            `OP_LD, `OP_LDS: use_imm_r = 1'b1;
            `OP_ST, `OP_STS: use_imm_r = 1'b1;
            `OP_SET: use_imm_r = 1'b1;
            `WMMA_LOAD: use_imm_r = 1'b1;
            `WMMA_STORE: use_imm_r = 1'b1;
            default: use_imm_r = 1'b0;
        endcase
    end
    assign dec_use_imm = use_imm_r;

    // ================================================================
    // Writeback source mux selector
    // ================================================================
    reg [2:0] wb_src_r;
    always @(*) begin
        case (opcode)
            `OP_LD, `OP_LDS: wb_src_r = 3'd1;
            `OP_ST, `OP_STS: wb_src_r = 3'd2;
            `WMMA_LOAD: wb_src_r = 3'd1;
            `WMMA_STORE: wb_src_r = 3'd2;
            default: wb_src_r = 3'd0;
        endcase
    end
    assign dec_wb_src = wb_src_r;

    // ================================================================
    // Branch target extraction
    // ================================================================
    wire [`GPU_PC_WIDTH-1:0] bra_target  = {{(`GPU_PC_WIDTH-27){1'b0}}, ir[26:0]};
    wire [`GPU_PC_WIDTH-1:0] pbra_target = {{(`GPU_PC_WIDTH-13){ir[24]}}, ir[24:12]};

    assign dec_branch_target = dec_is_pbra ? pbra_target : bra_target;

endmodule

`endif // SM_DECODER_V