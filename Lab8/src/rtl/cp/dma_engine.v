/* file: dma_engine.v
Description: DMA engine interfacing CP10 register file and GPU.
   Three transfer modes:
     D_IMEM:   CPU DMEM → GPU IMEM (32→32, 1:1)
     D_UNPACK: CPU DMEM → GPU DMEM (32→2×16, burst_all: cpu_ptr advances across banks)
     D_PACK:   GPU DMEM → CPU DMEM (2×16→32, burst_all: cpu_ptr advances across banks)
Author: Jeremy Cai
Date: Mar. 5, 2026
Version: 1.1
Revision history:
    - Mar. 4, 2026: v1.0 — Initial implementation.
    - Mar. 5, 2026: v1.1 — Cosmetic: clear gpu_dmem_sel on D_IMEM entry to avoid stale values in traces.
*/

`ifndef DMA_ENGINE_V
`define DMA_ENGINE_V

module dma_engine (
    input wire clk,
    input wire rst_n,
    // CP10 control interface
    input wire [31:0] dma_src_addr,
    input wire [31:0] dma_dst_addr,
    input wire [15:0] dma_xfer_len,
    input wire dma_start,
    input wire dma_dir,               // 0=CPU→GPU, 1=GPU→CPU
    input wire dma_tgt,               // 0=DMEM, 1=IMEM
    input wire [1:0] dma_bank,
    input wire dma_auto_inc,
    input wire dma_burst_all,

    output reg dma_busy,
    output reg dma_error,
    output wire [31:0] dma_cur_addr,

    // CPU DMEM Port B
    output reg [11:0] cpu_dmem_addr,
    output reg [31:0] cpu_dmem_din,
    output reg cpu_dmem_we,
    input wire [31:0] cpu_dmem_dout,

    // GPU IMEM ext Port B
    output reg [7:0] gpu_imem_addr,
    output reg [31:0] gpu_imem_din,
    output reg gpu_imem_we,

    // GPU DMEM ext Port B (bank-selected externally)
    output reg [1:0] gpu_dmem_sel,
    output reg [9:0] gpu_dmem_addr,
    output reg [15:0] gpu_dmem_din,
    output reg gpu_dmem_we,
    input wire [15:0] gpu_dmem_dout
);

    // ================================================================
    // FSM States
    // ================================================================
    localparam S_IDLE         = 4'd0;
    localparam S_IMEM_RD      = 4'd1;
    localparam S_IMEM_WR      = 4'd2;
    localparam S_UNPACK_RD    = 4'd3;
    localparam S_UNPACK_WR_LO = 4'd4;
    localparam S_UNPACK_WR_HI = 4'd5;
    localparam S_PACK_RD_LO   = 4'd6;
    localparam S_PACK_RD_HI   = 4'd7;
    localparam S_PACK_WAIT    = 4'd8;
    localparam S_PACK_WR      = 4'd9;
    localparam S_BANK_NEXT    = 4'd10;
    localparam S_DONE         = 4'd11;

    reg [3:0] state;

    // ================================================================
    // Transfer parameters (latched on dma_start)
    // ================================================================
    reg dir_r, tgt_r, auto_inc_r, burst_all_r;

    reg [31:0] cpu_ptr, gpu_ptr;
    reg [15:0] words_left, xfer_len_r;
    reg [31:0] gpu_base_r;

    reg [1:0] cur_bank, banks_done;

    reg [31:0] cpu_data_r;
    reg [15:0] gpu_lo_r;

    assign dma_cur_addr = cpu_ptr;

    // ================================================================
    // FSM
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            dma_busy <= 1'b0;
            dma_error <= 1'b0;
            cpu_dmem_addr <= 12'd0;
            cpu_dmem_din <= 32'd0;
            cpu_dmem_we <= 1'b0;
            gpu_imem_addr <= 8'd0;
            gpu_imem_din <= 32'd0;
            gpu_imem_we <= 1'b0;
            gpu_dmem_sel <= 2'd0;
            gpu_dmem_addr <= 10'd0;
            gpu_dmem_din <= 16'd0;
            gpu_dmem_we <= 1'b0;
            cpu_ptr <= 32'd0;
            gpu_ptr <= 32'd0;
            words_left <= 16'd0;
            xfer_len_r <= 16'd0;
            gpu_base_r <= 32'd0;
            dir_r <= 1'b0;
            tgt_r <= 1'b0;
            auto_inc_r <= 1'b0;
            burst_all_r <= 1'b0;
            cur_bank <= 2'd0;
            banks_done <= 2'd0;
            cpu_data_r <= 32'd0;
            gpu_lo_r <= 16'd0;
        end else begin
            // Default: deassert write enables each cycle
            cpu_dmem_we <= 1'b0;
            gpu_imem_we <= 1'b0;
            gpu_dmem_we <= 1'b0;

            case (state)

            // ────────────────────────────────────────────
            S_IDLE: begin
                if (dma_start) begin
                    dir_r <= dma_dir;
                    tgt_r <= dma_tgt;
                    auto_inc_r <= dma_auto_inc;
                    burst_all_r <= dma_burst_all;
                    xfer_len_r <= dma_xfer_len;
                    words_left <= dma_xfer_len;
                    cur_bank <= dma_bank;
                    banks_done <= 2'd0;
                    dma_busy <= 1'b1;
                    dma_error <= 1'b0;

                    if (dma_dir & dma_tgt) begin
                        // Invalid: GPU IMEM → CPU not supported
                        dma_error <= 1'b1;
                        state <= S_DONE;
                    end
                    else if (dma_xfer_len == 16'd0) begin
                        state <= S_DONE;
                    end
                    // D_IMEM: CPU → GPU IMEM (32→32)
                    else if (~dma_dir & dma_tgt) begin
                        cpu_ptr <= dma_src_addr;
                        gpu_ptr <= dma_dst_addr;
                        gpu_base_r <= dma_dst_addr;
                        cpu_dmem_addr <= dma_src_addr[11:0];
                        // v1.1: clear DMEM sel to avoid stale values in traces
                        gpu_dmem_sel <= 2'd0;
                        state <= S_IMEM_RD;
                    end
                    // D_UNPACK: CPU → GPU DMEM (32→2×16)
                    // burst_all: cpu_ptr advances across banks (each bank gets different data)
                    else if (~dma_dir & ~dma_tgt) begin
                        cpu_ptr <= dma_src_addr;
                        gpu_ptr <= dma_dst_addr;
                        gpu_base_r <= dma_dst_addr;
                        cpu_dmem_addr <= dma_src_addr[11:0];
                        gpu_dmem_sel <= dma_bank;
                        state <= S_UNPACK_RD;
                    end
                    // D_PACK: GPU DMEM → CPU (2×16→32)
                    // burst_all: cpu_ptr advances across banks (each bank fills different CPU words)
                    else begin
                        cpu_ptr <= dma_dst_addr;
                        gpu_ptr <= dma_src_addr;
                        gpu_base_r <= dma_src_addr;
                        gpu_dmem_sel <= dma_bank;
                        gpu_dmem_addr <= dma_src_addr[9:0];
                        state <= S_PACK_RD_LO;
                    end
                end
            end

            // ════════════════════════════════════════════
            // D_IMEM: CPU DMEM → GPU IMEM (32→32)
            // ════════════════════════════════════════════

            S_IMEM_RD: begin
                state <= S_IMEM_WR;
            end

            S_IMEM_WR: begin
                gpu_imem_addr <= gpu_ptr[7:0];
                gpu_imem_din <= cpu_dmem_dout;
                gpu_imem_we <= 1'b1;
                cpu_ptr <= cpu_ptr + 32'd1;
                gpu_ptr <= gpu_ptr + 32'd1;
                words_left <= words_left - 16'd1;
                if (words_left == 16'd1) begin
                    state <= S_BANK_NEXT;
                end else begin
                    cpu_dmem_addr <= cpu_ptr[11:0] + 12'd1;
                    state <= S_IMEM_RD;
                end
            end

            // ════════════════════════════════════════════
            // D_UNPACK: CPU DMEM → GPU DMEM (32→2×16)
            // ════════════════════════════════════════════

            S_UNPACK_RD: begin
                state <= S_UNPACK_WR_LO;
            end

            S_UNPACK_WR_LO: begin
                cpu_data_r <= cpu_dmem_dout;
                gpu_dmem_addr <= gpu_ptr[9:0];
                gpu_dmem_din <= cpu_dmem_dout[15:0];
                gpu_dmem_we <= 1'b1;
                state <= S_UNPACK_WR_HI;
            end

            S_UNPACK_WR_HI: begin
                gpu_dmem_addr <= gpu_ptr[9:0] + 10'd1;
                gpu_dmem_din <= cpu_data_r[31:16];
                gpu_dmem_we <= 1'b1;
                cpu_ptr <= cpu_ptr + 32'd1;
                gpu_ptr <= gpu_ptr + 32'd2;
                words_left <= words_left - 16'd1;
                if (words_left == 16'd1) begin
                    state <= S_BANK_NEXT;
                end else begin
                    cpu_dmem_addr <= cpu_ptr[11:0] + 12'd1;
                    state <= S_UNPACK_RD;
                end
            end

            // ════════════════════════════════════════════
            // D_PACK: GPU DMEM → CPU DMEM (2×16→32)
            // ════════════════════════════════════════════

            S_PACK_RD_LO: begin
                state <= S_PACK_RD_HI;
            end

            S_PACK_RD_HI: begin
                gpu_lo_r <= gpu_dmem_dout;
                gpu_dmem_addr <= gpu_ptr[9:0] + 10'd1;
                state <= S_PACK_WAIT;
            end

            S_PACK_WAIT: begin
                state <= S_PACK_WR;
            end

            S_PACK_WR: begin
                cpu_dmem_addr <= cpu_ptr[11:0];
                cpu_dmem_din <= {gpu_dmem_dout, gpu_lo_r};
                cpu_dmem_we <= 1'b1;
                cpu_ptr <= cpu_ptr + 32'd1;
                gpu_ptr <= gpu_ptr + 32'd2;
                words_left <= words_left - 16'd1;
                if (words_left == 16'd1) begin
                    state <= S_BANK_NEXT;
                end else begin
                    gpu_dmem_addr <= gpu_ptr[9:0] + 10'd2;
                    state <= S_PACK_RD_LO;
                end
            end

            // ════════════════════════════════════════════
            // Bank iteration + completion
            //
            // burst_all behavior:
            //   - gpu_ptr resets to gpu_base_r (each bank reads/writes same GPU addresses)
            //   - cpu_ptr keeps advancing (each bank maps to consecutive CPU address range)
            //   - gpu_dmem_sel increments to select next bank
            //   - Total CPU words consumed/produced = xfer_len × 4 (for 4 banks)
            // ════════════════════════════════════════════

            S_BANK_NEXT: begin
                if (burst_all_r && banks_done < 2'd3) begin
                    banks_done <= banks_done + 2'd1;
                    cur_bank <= cur_bank + 2'd1;
                    gpu_dmem_sel <= cur_bank + 2'd1;
                    gpu_ptr <= gpu_base_r;
                    words_left <= xfer_len_r;
                    if (~dir_r & tgt_r) begin
                        cpu_dmem_addr <= cpu_ptr[11:0];
                        state <= S_IMEM_RD;
                    end else if (~dir_r & ~tgt_r) begin
                        cpu_dmem_addr <= cpu_ptr[11:0];
                        state <= S_UNPACK_RD;
                    end else begin
                        gpu_dmem_addr <= gpu_base_r[9:0];
                        state <= S_PACK_RD_LO;
                    end
                end else begin
                    if (auto_inc_r)
                        cur_bank <= cur_bank + 2'd1;
                    state <= S_DONE;
                end
            end

            S_DONE: begin
                dma_busy <= 1'b0;
                state <= S_IDLE;
            end

            default: begin
                state <= S_IDLE;
                dma_busy <= 1'b0;
            end

            endcase
        end
    end

endmodule

`endif // DMA_ENGINE_V