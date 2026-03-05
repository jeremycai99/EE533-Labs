/* file: dma_engine.v
Description: This file contains the implementation of the DMA engine, which interfaces with the CP10 register file and GPU.
Author: Jeremy Cai
Date: Mar. 4, 2026
Version: 1.0
Revision history:
    - Mar. 4, 2026: v1.0 — Initial implementation of the DMA engine.
*/

`ifndef DMA_ENGINE_V
`define DMA_ENGINE_V

module dma_engine (
    input wire clk,
    input wire rst_n,
    // CP10 control interface
    input wire [31:0] dma_src_addr,   // source base address
    input wire [31:0] dma_dst_addr,   // destination base address
    input wire [15:0] dma_xfer_len,   // transfer length (CPU 32-bit words)
    input wire dma_start,             // 1-cycle pulse
    input wire dma_dir,               // 0=CPU->GPU, 1=GPU->CPU
    input wire dma_tgt,               // 0=DMEM, 1=IMEM
    input wire [1:0] dma_bank,        // initial GPU DMEM bank select
    input wire dma_auto_inc,          // auto-increment bank after done
    input wire dma_burst_all,         // repeat for all 4 banks

    output reg dma_busy,
    output reg dma_error,
    output wire [31:0] dma_cur_addr,  // current address (for CP10 CR9)

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
    output reg [1:0] gpu_dmem_sel,    // bank select
    output reg [9:0] gpu_dmem_addr,
    output reg [15:0] gpu_dmem_din,
    output reg gpu_dmem_we,
    input wire [15:0] gpu_dmem_dout
);

    // ================================================================
    // FSM States
    // ================================================================
    localparam S_IDLE         = 4'd0;
    // D_IMEM: CPU DMEM -> GPU IMEM (32->32)
    localparam S_IMEM_RD      = 4'd1;  // present CPU DMEM addr
    localparam S_IMEM_WR      = 4'd2;  // write to GPU IMEM
    // D_UNPACK: CPU DMEM -> GPU DMEM (32->16)
    localparam S_UNPACK_RD    = 4'd3;  // present CPU DMEM addr
    localparam S_UNPACK_WR_LO = 4'd4;  // write low 16b
    localparam S_UNPACK_WR_HI = 4'd5;  // write high 16b
    // D_PACK: GPU DMEM -> CPU DMEM (16->32)
    localparam S_PACK_RD_LO   = 4'd6;  // present GPU DMEM addr for low
    localparam S_PACK_RD_HI   = 4'd7;  // latch low, present addr for high
    localparam S_PACK_WAIT    = 4'd8;  // wait 1 cyc for BRAM high-half read
    localparam S_PACK_WR      = 4'd9;  // write packed word to CPU DMEM
    // Bank iteration
    localparam S_BANK_NEXT    = 4'd10; // advance to next bank
    localparam S_DONE         = 4'd11;

    reg [3:0] state;

    // ================================================================
    // Transfer parameters (latched on dma_start)
    // ================================================================
    reg dir_r;             // latched direction
    reg tgt_r;             // latched target
    reg auto_inc_r;
    reg burst_all_r;

    // Address counters
    reg [31:0] cpu_ptr;    // CPU DMEM address pointer
    reg [31:0] gpu_ptr;    // GPU-side address pointer (IMEM or DMEM)
    reg [15:0] words_left; // words remaining in current bank transfer
    reg [15:0] xfer_len_r; // latched transfer length (for bank resets)
    reg [31:0] gpu_base_r; // latched GPU base addr (for bank resets)

    // Bank tracking
    reg [1:0] cur_bank;    // current bank being serviced
    reg [1:0] banks_done;  // number of banks completed (for burst_all)

    // Data capture registers
    reg [31:0] cpu_data_r; // captured CPU DMEM read data
    reg [15:0] gpu_lo_r;   // captured GPU DMEM low half

    // ================================================================
    // cur_addr output (CPU-side pointer for status readback)
    // ================================================================
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
                    // Latch parameters
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

                    // Validate: dir=1, tgt=1 is invalid
                    if (dma_dir & dma_tgt) begin
                        dma_error <= 1'b1;
                        state <= S_DONE;
                    end
                    // Validate: zero length
                    else if (dma_xfer_len == 16'd0) begin
                        state <= S_DONE;
                    end
                    // D_IMEM: CPU->GPU IMEM
                    else if (~dma_dir & dma_tgt) begin
                        cpu_ptr <= dma_src_addr;
                        gpu_ptr <= dma_dst_addr;
                        gpu_base_r <= dma_dst_addr;
                        cpu_dmem_addr <= dma_src_addr[11:0];
                        state <= S_IMEM_RD;
                    end
                    // D_UNPACK: CPU->GPU DMEM
                    else if (~dma_dir & ~dma_tgt) begin
                        cpu_ptr <= dma_src_addr;
                        gpu_ptr <= dma_dst_addr;
                        gpu_base_r <= dma_dst_addr;
                        cpu_dmem_addr <= dma_src_addr[11:0];
                        gpu_dmem_sel <= dma_bank;
                        state <= S_UNPACK_RD;
                    end
                    // D_PACK: GPU->CPU DMEM
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
            // D_IMEM: CPU DMEM -> GPU IMEM (32->32)
            // ════════════════════════════════════════════

            S_IMEM_RD: begin
                // CPU DMEM addr already presented; wait 1 cycle for data
                state <= S_IMEM_WR;
            end

            S_IMEM_WR: begin
                // CPU DMEM dout is valid — write to GPU IMEM
                gpu_imem_addr <= gpu_ptr[7:0];
                gpu_imem_din <= cpu_dmem_dout;
                gpu_imem_we <= 1'b1;
                // Advance pointers
                cpu_ptr <= cpu_ptr + 32'd1;
                gpu_ptr <= gpu_ptr + 32'd1;
                words_left <= words_left - 16'd1;
                if (words_left == 16'd1) begin
                    // Last word — check for bank iteration
                    state <= S_BANK_NEXT;
                end else begin
                    // More words: present next CPU DMEM addr
                    cpu_dmem_addr <= cpu_ptr[11:0] + 12'd1;
                    state <= S_IMEM_RD;
                end
            end

            // ════════════════════════════════════════════
            // D_UNPACK: CPU DMEM -> GPU DMEM (32->16)
            // ════════════════════════════════════════════

            S_UNPACK_RD: begin
                // CPU DMEM addr already presented; wait 1 cycle for data
                state <= S_UNPACK_WR_LO;
            end

            S_UNPACK_WR_LO: begin
                // Capture CPU data, write low 16b
                cpu_data_r <= cpu_dmem_dout;
                gpu_dmem_addr <= gpu_ptr[9:0];
                gpu_dmem_din <= cpu_dmem_dout[15:0];
                gpu_dmem_we <= 1'b1;
                state <= S_UNPACK_WR_HI;
            end

            S_UNPACK_WR_HI: begin
                // Write high 16b to next GPU DMEM address
                gpu_dmem_addr <= gpu_ptr[9:0] + 10'd1;
                gpu_dmem_din <= cpu_data_r[31:16];
                gpu_dmem_we <= 1'b1;
                // Advance pointers
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
            // D_PACK: GPU DMEM -> CPU DMEM (16->32)
            // ════════════════════════════════════════════

            S_PACK_RD_LO: begin
                // GPU DMEM addr already presented; wait 1 cycle for data
                state <= S_PACK_RD_HI;
            end

            S_PACK_RD_HI: begin
                // Capture low half, present address for high half
                gpu_lo_r <= gpu_dmem_dout;
                gpu_dmem_addr <= gpu_ptr[9:0] + 10'd1;
                state <= S_PACK_WAIT;
            end

            S_PACK_WAIT: begin
                // Wait 1 cycle for BRAM to produce high-half data.
                // Address was presented in S_PACK_RD_HI (NBA), so
                // BRAM captured it at the S_PACK_WAIT posedge.
                // Data will be valid at the S_PACK_WR posedge.
                state <= S_PACK_WR;
            end

            S_PACK_WR: begin
                // GPU DMEM dout is now high half — write packed to CPU
                cpu_dmem_addr <= cpu_ptr[11:0];
                cpu_dmem_din <= {gpu_dmem_dout, gpu_lo_r};
                cpu_dmem_we <= 1'b1;
                // Advance pointers
                cpu_ptr <= cpu_ptr + 32'd1;
                gpu_ptr <= gpu_ptr + 32'd2;
                words_left <= words_left - 16'd1;
                if (words_left == 16'd1) begin
                    state <= S_BANK_NEXT;
                end else begin
                    // Present next GPU read addr
                    gpu_dmem_addr <= gpu_ptr[9:0] + 10'd2;
                    state <= S_PACK_RD_LO;
                end
            end

            // ════════════════════════════════════════════
            // Bank iteration + completion
            // ════════════════════════════════════════════

            S_BANK_NEXT: begin
                if (burst_all_r && banks_done < 2'd3) begin
                    // More banks to service
                    banks_done <= banks_done + 2'd1;
                    cur_bank <= cur_bank + 2'd1;
                    gpu_dmem_sel <= cur_bank + 2'd1;
                    // Reset GPU-side pointer, keep CPU-side advancing
                    gpu_ptr <= gpu_base_r;
                    words_left <= xfer_len_r;
                    // Branch to appropriate first state
                    if (~dir_r & tgt_r) begin
                        // D_IMEM: re-present CPU addr
                        cpu_dmem_addr <= cpu_ptr[11:0];
                        state <= S_IMEM_RD;
                    end else if (~dir_r & ~tgt_r) begin
                        // D_UNPACK
                        cpu_dmem_addr <= cpu_ptr[11:0];
                        state <= S_UNPACK_RD;
                    end else begin
                        // D_PACK
                        gpu_dmem_addr <= gpu_base_r[9:0];
                        state <= S_PACK_RD_LO;
                    end
                end else begin
                    // All banks done (or single-bank mode)
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