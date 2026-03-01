/* file: sp_core.v
 Description: This file implements the CUDA-like SP (streaming processor) core pipeline design.
 This implementation focuses on the per-thread behavior and the core should output the register content to
    tensor core for tensor operations for 4 x 4 matrix multiplication and accumulation. This process is a multi-cycle
    operation.
 Author: Jeremy Cai
 Date: Feb. 28, 2026
 Version: 1.0
 Revision history:
    - Feb. 28, 2026: Initial implementation of the CUDA-like SP core pipeline.
*/

`ifndef SP_CORE_V
`define SP_CORE_V

`include "gpu_define.v"
`include "gpr_regfile.v"
`include "pred_regfile.v"
`include "int16alu.v"
`include "bf16fpu.v"
`include "pplbfintcvt.v"

module sp_core #(
    parameter [1:0] TID = 2'b00
)(
    input wire clk,
    input wire rst_n,

    // Pipeline control (from SM stall controller)
    input wire stall,
    input wire flush_id,

    // ID Stage: RF Read
    input wire [3:0] rf_r0_addr,
    input wire [3:0] rf_r1_addr,
    input wire [3:0] rf_r2_addr,
    input wire [3:0] rf_r3_addr,
    output wire [15:0] rf_r0_data,
    output wire [15:0] rf_r1_data,
    output wire [15:0] rf_r2_data,
    output wire [15:0] rf_r3_data,

    // ID Stage: Predicate Read
    input wire [1:0] pred_rd_sel,
    output wire pred_rd_val,

    // ID -> EX: Control from SM decoder
    input wire [4:0] id_opcode,
    input wire id_dt,
    input wire [1:0] id_cmp_mode,
    input wire id_rf_we,
    input wire id_pred_we,
    input wire [3:0] id_rD_addr,
    input wire [1:0] id_pred_wr_sel,
    input wire id_valid,
    input wire id_active,
    input wire [2:0] id_wb_src,
    input wire id_use_imm,
    input wire [15:0] id_imm16,

    // EX -> SM: results for memory + stall
    output wire [15:0] ex_mem_result_out,
    output wire [15:0] ex_mem_store_out,
    output wire ex_mem_valid_out,
    output wire ex_busy,

    // MEM -> SP: memory read data from SM
    input wire [15:0] mem_rdata,

    // WB: WMMA Scatter — external W1-W3 ports from SM
    input wire [3:0] wb_ext_w1_addr,
    input wire [15:0] wb_ext_w1_data,
    input wire wb_ext_w1_we,
    input wire [3:0] wb_ext_w2_addr,
    input wire [15:0] wb_ext_w2_data,
    input wire wb_ext_w2_we,
    input wire [3:0] wb_ext_w3_addr,
    input wire [15:0] wb_ext_w3_data,
    input wire wb_ext_w3_we,

    // MEM -> SM: BRAM control (active during MEM stage)
    output wire mem_is_load,
    output wire mem_is_store,

    // WB -> SM: scoreboard feedback
    output wire [3:0] wb_rD_addr,
    output wire wb_rf_we,
    output wire wb_active,
    output wire wb_valid
);

    // TID constant
    wire [15:0] tid_val = {14'b0, TID};

    // Forward declarations — WB write signals (needed for RF forwarding)
    wire w0_we;
    wire [3:0] w0_addr;
    wire [15:0] w0_data;
    wire pred_wr_we;
    wire [1:0] pred_wr_sel;
    wire pred_wr_data;

    // ====================================================================
    // GPR Register File — 16x16b, 4R4W
    // ====================================================================
    gpr_regfile u_gpr_rf (
        .clk(clk), .rst_n(rst_n),
        .read_addr1(rf_r0_addr), .read_addr2(rf_r1_addr),
        .read_addr3(rf_r2_addr), .read_addr4(rf_r3_addr),
        .read_data1(rf_r0_data), .read_data2(rf_r1_data),
        .read_data3(rf_r2_data), .read_data4(rf_r3_data),
        .write_addr1(w0_addr), .write_data1(w0_data), .write_en1(w0_we),
        .write_addr2(wb_ext_w1_addr), .write_data2(wb_ext_w1_data), .write_en2(wb_ext_w1_we),
        .write_addr3(wb_ext_w2_addr), .write_data3(wb_ext_w2_data), .write_en3(wb_ext_w2_we),
        .write_addr4(wb_ext_w3_addr), .write_data4(wb_ext_w3_data), .write_en4(wb_ext_w3_we)
    );

    // ====================================================================
    // Predicate Register File — 4x1b, 1R1W
    // ====================================================================
    pred_regfile u_pred_rf (
        .clk(clk), .rst_n(rst_n),
        .read_sel(pred_rd_sel), .read_val(pred_rd_val),
        .write_sel(pred_wr_sel), .write_data(pred_wr_data), .write_en(pred_wr_we)
    );

    // ====================================================================
    // ID/EX Pipeline Register
    // ====================================================================
    reg [15:0] id_ex_opA;
    reg [15:0] id_ex_opB;
    reg [15:0] id_ex_opC;
    reg [4:0] id_ex_opcode;
    reg id_ex_dt;
    reg [1:0] id_ex_cmp_mode;
    reg id_ex_pred_val;
    reg id_ex_rf_we;
    reg id_ex_pred_we;
    reg [3:0] id_ex_rD_addr;
    reg [1:0] id_ex_pred_wr_sel;
    reg id_ex_valid;
    reg id_ex_active;
    reg [2:0] id_ex_wb_src;

    // MOV.TID detection: MOV opcode with DT=1
    wire sel_tid = (id_opcode == `OP_MOV) && id_dt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            id_ex_valid <= 1'b0;
            id_ex_rf_we <= 1'b0;
            id_ex_pred_we <= 1'b0;
            id_ex_active <= 1'b0;
            id_ex_pred_val <= 1'b0;
            id_ex_opcode <= 5'd0;
            id_ex_dt <= 1'b0;
            id_ex_cmp_mode <= 2'd0;
            id_ex_rD_addr <= 4'd0;
            id_ex_pred_wr_sel <= 2'd0;
            id_ex_wb_src <= 3'd0;
            id_ex_opA <= 16'd0;
            id_ex_opB <= 16'd0;
            id_ex_opC <= 16'd0;
        end else if (!stall) begin
            if (flush_id) begin
                id_ex_valid <= 1'b0;
                id_ex_rf_we <= 1'b0;
                id_ex_pred_we <= 1'b0;
            end else begin
                id_ex_opA <= sel_tid ? tid_val : rf_r0_data;
                id_ex_opB <= id_use_imm ? id_imm16 : rf_r1_data;
                id_ex_opC <= rf_r2_data;
                id_ex_pred_val <= pred_rd_val;
                id_ex_opcode <= id_opcode;
                id_ex_dt <= sel_tid ? 1'b0 : id_dt;
                id_ex_cmp_mode <= id_cmp_mode;
                id_ex_rf_we <= id_rf_we;
                id_ex_pred_we <= id_pred_we;
                id_ex_rD_addr <= id_rD_addr;
                id_ex_pred_wr_sel <= id_pred_wr_sel;
                id_ex_valid <= id_valid;
                id_ex_active <= id_active;
                id_ex_wb_src <= id_wb_src;
            end
        end
    end

    // ====================================================================
    // EX Stage: INT16 ALU + BF16 FPU + DT MUX
    // ====================================================================

    // Launched flag: prevents re-triggering ALU/FPU when stall drops
    reg id_ex_launched;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            id_ex_launched <= 1'b0;
        else if (!stall)
            id_ex_launched <= 1'b0;
        else if (!id_ex_launched)
            id_ex_launched <= id_ex_valid;
    end

    // CVT instruction bypasses ALU/FPU
    wire is_cvt = (id_ex_opcode == `OP_CVT);

    // Gate valid_in: one-shot pulse on first EX cycle only
    wire alu_valid_in = id_ex_valid & ~id_ex_dt & ~id_ex_launched & ~is_cvt;
    wire fpu_valid_in = id_ex_valid & id_ex_dt & ~id_ex_launched & ~is_cvt;
    wire cvt_valid_in = id_ex_valid & is_cvt & ~id_ex_launched;

    // --- INT16 ALU ---
    wire [15:0] alu_result;
    wire alu_valid_out, alu_busy;
    wire alu_cmp_eq, alu_cmp_ne, alu_cmp_lt, alu_cmp_le;

    int16alu u_alu (
        .clk(clk), .rst_n(rst_n),
        .alu_op(id_ex_opcode), .valid_in(alu_valid_in),
        .cmp_mode(id_ex_cmp_mode), .pred_val(id_ex_pred_val),
        .op_a(id_ex_opA), .op_b(id_ex_opB), .op_c(id_ex_opC),
        .result(alu_result), .valid_out(alu_valid_out), .busy(alu_busy),
        .cmp_eq(alu_cmp_eq), .cmp_ne(alu_cmp_ne),
        .cmp_lt(alu_cmp_lt), .cmp_le(alu_cmp_le)
    );

    // --- BF16 FPU ---
    wire [15:0] fpu_result;
    wire fpu_valid_out, fpu_busy;
    wire fpu_cmp_eq, fpu_cmp_ne, fpu_cmp_lt, fpu_cmp_le;

    bf16fpu u_fpu (
        .clk(clk), .rst_n(rst_n),
        .alu_op(id_ex_opcode), .valid_in(fpu_valid_in),
        .cmp_mode(id_ex_cmp_mode), .pred_val(id_ex_pred_val),
        .op_a(id_ex_opA), .op_b(id_ex_opB), .op_c(id_ex_opC),
        .result(fpu_result), .valid_out(fpu_valid_out), .busy(fpu_busy),
        .cmp_eq(fpu_cmp_eq), .cmp_ne(fpu_cmp_ne),
        .cmp_lt(fpu_cmp_lt), .cmp_le(fpu_cmp_le)
    );

    // --- CVT: INT16<->BF16 converter (2-cycle pipeline) ---
    wire [15:0] cvt_result;

    pplbfintcvt u_cvt (
        .clk(clk), .rst_n(rst_n),
        .dt(id_ex_dt), .in(id_ex_opA), .out(cvt_result)
    );

    reg [1:0] cvt_pipe_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cvt_pipe_valid <= 2'b00;
        else begin
            cvt_pipe_valid[0] <= cvt_valid_in;
            cvt_pipe_valid[1] <= cvt_pipe_valid[0];
        end
    end
    wire cvt_done = cvt_pipe_valid[1];
    wire cvt_busy = cvt_pipe_valid[0];

    // --- DT MUX ---
    wire [15:0] alu_fpu_result = id_ex_dt ? fpu_result : alu_result;
    wire [15:0] ex_result_muxed = cvt_done ? cvt_result : alu_fpu_result;
    wire alu_fpu_valid = id_ex_dt ? fpu_valid_out : alu_valid_out;
    wire ex_valid_out = cvt_done | alu_fpu_valid;
    wire alu_fpu_busy = id_ex_dt ? fpu_busy : alu_busy;
    assign ex_busy = cvt_busy | alu_fpu_busy;

    // --- Comparator output MUX ---
    wire alu_cmp_selected = (id_ex_cmp_mode == 2'd0) ? alu_cmp_eq :
                            (id_ex_cmp_mode == 2'd1) ? alu_cmp_ne :
                            (id_ex_cmp_mode == 2'd2) ? alu_cmp_lt : alu_cmp_le;
    wire fpu_cmp_selected = (id_ex_cmp_mode == 2'd0) ? fpu_cmp_eq :
                            (id_ex_cmp_mode == 2'd1) ? fpu_cmp_ne :
                            (id_ex_cmp_mode == 2'd2) ? fpu_cmp_lt : fpu_cmp_le;
    wire cmp_out_muxed = id_ex_dt ? fpu_cmp_selected : alu_cmp_selected;

    // ====================================================================
    // EX/MEM Pipeline Register
    // ====================================================================
    reg [15:0] ex_mem_result;
    reg [15:0] ex_mem_store_data;
    reg ex_mem_cmp_out;
    reg ex_mem_rf_we;
    reg ex_mem_pred_we;
    reg [3:0] ex_mem_rD_addr;
    reg [1:0] ex_mem_pred_wr_sel;
    reg ex_mem_valid;
    reg ex_mem_active;
    reg [2:0] ex_mem_wb_src;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ex_mem_valid <= 1'b0;
            ex_mem_rf_we <= 1'b0;
            ex_mem_pred_we <= 1'b0;
            ex_mem_active <= 1'b0;
            ex_mem_result <= 16'd0;
            ex_mem_store_data <= 16'd0;
            ex_mem_cmp_out <= 1'b0;
            ex_mem_rD_addr <= 4'd0;
            ex_mem_pred_wr_sel <= 2'd0;
            ex_mem_wb_src <= 3'd0;
        end else if (ex_valid_out) begin
            // Result ready — capture immediately (valid_out is 1-cycle pulse)
            ex_mem_valid <= 1'b1;
            ex_mem_result <= ex_result_muxed;
            ex_mem_store_data <= id_ex_opC;
            ex_mem_cmp_out <= cmp_out_muxed;
            ex_mem_rf_we <= id_ex_rf_we;
            ex_mem_pred_we <= id_ex_pred_we;
            ex_mem_rD_addr <= id_ex_rD_addr;
            ex_mem_pred_wr_sel <= id_ex_pred_wr_sel;
            ex_mem_active <= id_ex_active;
            ex_mem_wb_src <= id_ex_wb_src;
        end else if (!stall) begin
            ex_mem_valid <= 1'b0;
            ex_mem_rf_we <= 1'b0;
            ex_mem_pred_we <= 1'b0;
        end
    end

    // Outputs to SM memory unit
    assign ex_mem_result_out = ex_mem_result;
    assign ex_mem_store_out = ex_mem_store_data;
    assign ex_mem_valid_out = ex_mem_valid;

    // BRAM control: SM uses these to drive dmem read/write enables
    assign mem_is_load = ex_mem_valid & (ex_mem_wb_src == 3'd1);
    assign mem_is_store = ex_mem_valid & (ex_mem_wb_src == 3'd2);

    // ====================================================================
    // MEM/WB Pipeline Register
    // ====================================================================
    // NOTE: mem_rdata (BRAM load data) is NOT captured here.
    // It bypasses mem_wb and feeds directly into the WB-stage mux,
    // because sync-read BRAM output arrives 1 cycle after address
    // presentation — the same posedge that mem_wb captures control.
    reg [15:0] mem_wb_data;
    reg mem_wb_cmp_out;
    reg mem_wb_rf_we;
    reg mem_wb_pred_we;
    reg [3:0] mem_wb_rD_addr;
    reg [1:0] mem_wb_pred_wr_sel;
    reg mem_wb_valid;
    reg mem_wb_active;
    reg [2:0] mem_wb_wb_src;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_wb_valid <= 1'b0;
            mem_wb_rf_we <= 1'b0;
            mem_wb_pred_we <= 1'b0;
            mem_wb_active <= 1'b0;
            mem_wb_data <= 16'd0;
            mem_wb_cmp_out <= 1'b0;
            mem_wb_rD_addr <= 4'd0;
            mem_wb_pred_wr_sel <= 2'd0;
            mem_wb_wb_src <= 3'd0;
        end else if (!stall) begin
            mem_wb_data <= ex_mem_result;
            mem_wb_cmp_out <= ex_mem_cmp_out;
            mem_wb_rf_we <= ex_mem_rf_we;
            mem_wb_pred_we <= ex_mem_pred_we;
            mem_wb_rD_addr <= ex_mem_rD_addr;
            mem_wb_pred_wr_sel <= ex_mem_pred_wr_sel;
            mem_wb_valid <= ex_mem_valid;
            mem_wb_active <= ex_mem_active;
            mem_wb_wb_src <= ex_mem_wb_src;
        end
    end

    // ====================================================================
    // WB Stage: RF Write + Predicate Write + Scoreboard Feedback
    // ====================================================================

    // WB source mux (AFTER mem_wb) — BRAM bypass for loads
    wire [15:0] wb_data_final = (mem_wb_wb_src == 3'd1) ? mem_rdata : mem_wb_data;

    // Scalar GPR writeback (W0)
    assign w0_we = mem_wb_valid & mem_wb_rf_we & mem_wb_active & ~stall;
    assign w0_addr = mem_wb_rD_addr;
    assign w0_data = wb_data_final;

    // Predicate RF writeback (SETP)
    assign pred_wr_we = mem_wb_valid & mem_wb_pred_we & mem_wb_active & ~stall;
    assign pred_wr_sel = mem_wb_pred_wr_sel;
    assign pred_wr_data = mem_wb_cmp_out;

    // Scoreboard feedback to SM
    assign wb_rD_addr = mem_wb_rD_addr;
    assign wb_rf_we = mem_wb_rf_we & mem_wb_valid;
    assign wb_active = mem_wb_active;
    assign wb_valid = mem_wb_valid;

endmodule

`endif // SP_CORE_V
