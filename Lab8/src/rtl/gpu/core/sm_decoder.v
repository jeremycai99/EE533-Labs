/* file: sm_decoder.v
 Description: This file implements the instruction decoder for the CUDA-like SM core.
 The decoder takes a 32-bit instruction from the instruction memory and decodes it into control signals for the rest of the pipeline.
 Author: Jeremy Cai
 Date: Feb. 28, 2026
 Version: 1.0
 Revision history:
    - Feb. 28, 2026: Initial implementation of the CUDA-like SM core instruction decoder.
*/

`ifndef SM_DECODER_V
`define SM_DECODER_V

`include "gpu_define.v"

module sm_decoder (
    // Input: 32-bit instruction word (from IMEM BRAM)
    input wire [31:0] ir,

    // ── Field extraction ────────────────────────────────
    output wire [4:0] dec_opcode,
    output wire dec_dt,
    output wire [1:0] dec_cmp_mode,    // ir[25:24], CMP for SETP
    output wire [3:0] dec_rD_addr,
    output wire [3:0] dec_rA_addr,
    output wire [3:0] dec_rB_addr,
    output wire [3:0] dec_rC_addr,     // WMMA.MMA only: ir[11:8]
    output wire [15:0] dec_imm16,

    // ── SP core control (broadcast to all 4 SPs) ───────
    output wire dec_rf_we,       // writes GPR
    output wire dec_pred_we,     // writes predicate RF (SETP)
    output wire [1:0] dec_pred_wr_sel, // which P register for SETP
    output wire [1:0] dec_pred_rd_sel, // which P register to read (SELP/PBRA)
    output wire [2:0] dec_wb_src,      // WB mux: 0=ALU, 1=MEM, 2=STORE(no WB)
    output wire dec_use_imm,     // opB source = immediate

    // ── Scoreboard interface ────────────────────────────
    output wire dec_uses_rA,     // instruction reads rA
    output wire dec_uses_rB,     // instruction reads rB
    output wire dec_is_fma,      // FMA: rD is also a read source (accum)
    output wire dec_is_st,       // ST/STS: rD is store data (read, not write)

    // ── Branch / control flow ───────────────────────────
    output wire dec_is_branch,   // BRA (unconditional)
    output wire dec_is_pbra,     // PBRA (predicated, Phase 4)
    output wire dec_is_ret,      // RET → kernel_done

    // ── Memory classification ───────────────────────────
    output wire dec_is_ld,       // LD
    output wire dec_is_store,    // ST
    output wire dec_is_lds,      // LDS (shared memory load)
    output wire dec_is_sts,      // STS (shared memory store)

    // ── WMMA classification (Phase 3) ───────────────────
    output wire dec_is_wmma_mma,
    output wire dec_is_wmma_load,
    output wire dec_is_wmma_store,
    output wire [1:0] dec_wmma_sel,    // ir[25:24]: 00=A, 01=B for WMMA.LOAD

    // ── Branch target ───────────────────────────────────
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
    assign dec_cmp_mode = res;       // reused as CMP mode for SETP
    assign dec_rD_addr = rD;
    assign dec_rA_addr = rA;
    assign dec_rB_addr = rB;
    assign dec_rC_addr = rC;
    assign dec_wmma_sel = res;       // reused as WMMA.LOAD selector

    // Immediate: low 16 bits for I/M-type
    // For P-type (MOVI): ir[15:0] carries the 16-bit value
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
    // RET, SETP (writes pred RF not GPR), WMMA (handled separately).
    reg rf_we_r;
    always @(*) begin
        case (opcode)
            `OP_NOP:  rf_we_r = 1'b0;
            `OP_ST:   rf_we_r = 1'b0;
            `OP_STS:  rf_we_r = 1'b0;
            `OP_SETP: rf_we_r = 1'b0;
            `OP_BRA:  rf_we_r = 1'b0;
            `OP_PBRA: rf_we_r = 1'b0;
            `OP_RET:  rf_we_r = 1'b0;
            `WMMA_MMA:   rf_we_r = 1'b0;  // tensor core writes via scatter
            `WMMA_LOAD:  rf_we_r = 1'b0;  // burst controller writes
            `WMMA_STORE: rf_we_r = 1'b0;  // no RF write
            default:  rf_we_r = 1'b1;     // LD, MOV, MOVI, CVT, ADD..., SET, SELP, etc.
        endcase
    end
    assign dec_rf_we = rf_we_r;

    // ================================================================
    // Predicate RF write (SETP only)
    // ================================================================
    assign dec_pred_we = (opcode == `OP_SETP);
    assign dec_pred_wr_sel = rD[1:0]; // P0–P3 from low 2 bits of rD field

    // ================================================================
    // Predicate RF read select
    // ================================================================
    // SELP: reads predicate to choose between rA and rB → ir[25:24]
    // PBRA: reads predicate for branch condition → ir[26:25]
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
                uses_rA_r = 1'b0;  // WMMA.MMA gather bypasses normal path
            default:
                uses_rA_r = 1'b1;  // most ops read rA
        endcase
    end
    assign dec_uses_rA = uses_rA_r;

    reg uses_rB_r;
    always @(*) begin
        case (opcode)
            // R-type binary ops that use rB
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
            `OP_MOVI: use_imm_r = 1'b1;  // P-type: imm16 to opB
            `OP_SHL, `OP_SHR: use_imm_r = 1'b1;  // I-type: shift amount
            `OP_ADDI, `OP_MULI: use_imm_r = 1'b1;  // I-type: immediate operand
            `OP_LD, `OP_LDS: use_imm_r = 1'b1;  // M-type: offset for addr calc
            `OP_ST, `OP_STS: use_imm_r = 1'b1;  // M-type: offset for addr calc
            `OP_SET: use_imm_r = 1'b1;  // immediate 0/1
            `WMMA_LOAD: use_imm_r = 1'b1;  // M-type: offset
            `WMMA_STORE: use_imm_r = 1'b1;  // M-type: offset
            default: use_imm_r = 1'b0;
        endcase
    end
    assign dec_use_imm = use_imm_r;

    // ================================================================
    // Writeback source mux selector
    // ================================================================
    // 3'd0 = ALU/FPU/CVT result
    // 3'd1 = MEM load data (LD/LDS)
    // 3'd2 = Store marker (ST/STS: rf_we=0, but drives mem_is_store)
    // 3'd3 = reserved (WMMA)
    reg [2:0] wb_src_r;
    always @(*) begin
        case (opcode)
            `OP_LD, `OP_LDS: wb_src_r = 3'd1;  // load from memory
            `OP_ST, `OP_STS: wb_src_r = 3'd2;  // store marker
            `WMMA_LOAD: wb_src_r = 3'd1;  // burst load
            `WMMA_STORE: wb_src_r = 3'd2;  // burst store
            default: wb_src_r = 3'd0;  // ALU/FPU result
        endcase
    end
    assign dec_wb_src = wb_src_r;

    // ================================================================
    // Branch target extraction
    // ================================================================
    // BRA: absolute 27-bit target from ir[26:0], zero-extended to PC width
    // PBRA: target from ir[24:12] (Phase 4, may be relative or absolute)
    wire [`GPU_PC_WIDTH-1:0] bra_target  = {{(`GPU_PC_WIDTH-27){1'b0}}, ir[26:0]};
    wire [`GPU_PC_WIDTH-1:0] pbra_target = {{(`GPU_PC_WIDTH-13){ir[24]}}, ir[24:12]};

    assign dec_branch_target = dec_is_pbra ? pbra_target : bra_target;

endmodule

`endif // SM_DECODER_V