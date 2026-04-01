/* file: cp10_regfile.v
Description: This file contains the implementation of the CP10 register file, which interfaces with the DMA engine and GPU.
Author: Jeremy Cai
Date: Mar. 4, 2026
Version: 1.0
Revision history:
    - Mar. 4, 2026: v1.0 — Initial implementation of the CP10 register file.
*/

`ifndef CP10_REGFILE_V
`define CP10_REGFILE_V

module cp10_regfile (
    input wire clk,
    input wire rst_n,

    // Coprocessor interface from ARM CPU (EX2 stage)
    input wire cp_wen, // MCR: write enable (1 cycle)
    input wire cp_ren, // MRC: read enable (1 cycle)
    input wire [3:0] cp_reg, // CRn: register select (0–15)
    input wire [31:0] cp_wdata, // write data from CPU Rd
    output reg [31:0] cp_rdata, // read data to CPU Rd

    // DMA Engine interface
    output wire [31:0] dma_src_addr,
    output wire [31:0] dma_dst_addr,
    output wire [15:0] dma_xfer_len,
    output wire dma_start, // 1-cycle pulse
    output wire dma_dir, // 0=CPU→GPU, 1=GPU→CPU
    output wire dma_tgt, // 0=DMEM, 1=IMEM
    output wire [1:0] dma_bank,
    output wire dma_auto_inc,
    output wire dma_burst_all,
    input wire dma_busy,
    input wire dma_error,
    input wire [31:0] dma_cur_addr,

    // GPU interface (to/from sm_top)
    output wire gpu_kernel_start, // 1-cycle pulse
    output wire gpu_reset_n, // active-low reset to sm_core
    output wire [31:0] gpu_entry_pc,
    output wire [3:0] gpu_thread_mask,
    output wire [31:0] gpu_scratch,
    input wire gpu_kernel_done, // 1-cycle pulse from sm_core
    input wire gpu_active // high while kernel running (from sm_top)
);

    // ================================================================
    // Register storage
    // ================================================================

    // DMA Control (CR0–CR3)
    reg [31:0] cr0_dma_src; // CR0: DMA source address
    reg [31:0] cr1_dma_dst; // CR1: DMA destination address
    reg [15:0] cr2_dma_len; // CR2: DMA transfer length
    reg [6:0] cr3_dma_ctrl; // CR3: {burst_all, auto_inc, bank[1:0], tgt, dir, start}

    // GPU Kernel (CR4–CR6)
    reg [31:0] cr4_gpu_pc; // CR4: GPU entry PC
    reg [2:0] cr5_gpu_ctrl; // CR5: {auto_dma, reset, start}
    // CR6: GPU_STATUS — read-only, derived combinationally

    // Advanced (CR7–CR8)
    reg [3:0] cr7_thread_mask; // CR7: active thread mask
    reg [31:0] cr8_gpu_scratch; // CR8: scratch register

    // CR9: DMA_CUR_ADDR — read-only from DMA engine

    // ================================================================
    // Status tracking — sticky done flag
    // ================================================================
    // kernel_done is a 1-cycle pulse. Latch it into a sticky flag
    // so the CPU can poll CR6 at any time after completion.
    // Cleared when a new kernel is started (CR5[0] write).
    reg gpu_done_flag;

    // ================================================================
    // Write-trigger pulse generation
    // ================================================================
    // CR3[0] and CR5[0] are "start" bits. When the CPU writes 1,
    // we generate a 1-cycle pulse on the output, then auto-clear
    // the bit in the register.

    // Detect MCR writes to specific CRs
    wire wr_cr0 = cp_wen & (cp_reg == 4'd0);
    wire wr_cr1 = cp_wen & (cp_reg == 4'd1);
    wire wr_cr2 = cp_wen & (cp_reg == 4'd2);
    wire wr_cr3 = cp_wen & (cp_reg == 4'd3);
    wire wr_cr4 = cp_wen & (cp_reg == 4'd4);
    wire wr_cr5 = cp_wen & (cp_reg == 4'd5);
    wire wr_cr7 = cp_wen & (cp_reg == 4'd7);
    wire wr_cr8 = cp_wen & (cp_reg == 4'd8);

    // DMA start: pulse when CR3[0] written as 1, gated by ~dma_busy
    reg dma_start_r;
    // GPU start: pulse when CR5[0] written as 1
    reg gpu_start_r;
    // auto_dma trigger: pulse when kernel_done fires and auto_dma is set
    reg auto_dma_trigger;

    // ================================================================
    // Register write logic
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cr0_dma_src <= 32'd0;
            cr1_dma_dst <= 32'd0;
            cr2_dma_len <= 16'd0;
            cr3_dma_ctrl <= 7'd0;
            cr4_gpu_pc <= 32'd0;
            cr5_gpu_ctrl <= 3'd0;
            cr7_thread_mask <= 4'b1111;  // default: all threads active
            cr8_gpu_scratch <= 32'd0;
            gpu_done_flag <= 1'b0;
            dma_start_r <= 1'b0;
            gpu_start_r <= 1'b0;
            auto_dma_trigger <= 1'b0;
        end else begin
            // Pulse auto-clear (default: deassert after 1 cycle)
            dma_start_r <= 1'b0;
            gpu_start_r <= 1'b0;
            auto_dma_trigger <= 1'b0;

            // CR3[0] auto-clear
            if (cr3_dma_ctrl[0])
                cr3_dma_ctrl[0] <= 1'b0;

            // CR5[0] auto-clear
            if (cr5_gpu_ctrl[0])
                cr5_gpu_ctrl[0] <= 1'b0;

            // MCR writes
            if (wr_cr0) cr0_dma_src <= cp_wdata;
            if (wr_cr1) cr1_dma_dst <= cp_wdata;
            if (wr_cr2) cr2_dma_len <= cp_wdata[15:0];

            if (wr_cr3) begin
                cr3_dma_ctrl <= cp_wdata[6:0];
                // Generate DMA start pulse if bit[0]=1 and DMA not busy
                if (cp_wdata[0] & ~dma_busy)
                    dma_start_r <= 1'b1;
            end

            if (wr_cr4) cr4_gpu_pc <= cp_wdata;

            if (wr_cr5) begin
                cr5_gpu_ctrl <= cp_wdata[2:0];
                // Generate GPU start pulse if bit[0]=1
                if (cp_wdata[0]) begin
                    gpu_start_r <= 1'b1;
                    gpu_done_flag <= 1'b0;  // clear done on new start
                end
            end

            // CR6: read-only (no write)
            if (wr_cr7) cr7_thread_mask <= cp_wdata[3:0];
            if (wr_cr8) cr8_gpu_scratch <= cp_wdata;
            // CR9: read-only

            // Sticky done flag capture
            if (gpu_kernel_done)
                gpu_done_flag <= 1'b1;

            // auto_dma: trigger DMA readback on kernel_done
            if (gpu_kernel_done & cr5_gpu_ctrl[2] & ~dma_busy)
                auto_dma_trigger <= 1'b1;

        end
    end

    // ================================================================
    // MRC read mux
    // ================================================================
    // CR6 status: {idle, active, done}
    wire gpu_idle = ~gpu_active & gpu_done_flag;
    wire [31:0] cr6_gpu_status = {29'd0, gpu_idle, gpu_active, gpu_done_flag};

    always @(*) begin
        case (cp_reg)
            4'd0:  cp_rdata = cr0_dma_src;
            4'd1:  cp_rdata = cr1_dma_dst;
            4'd2:  cp_rdata = {16'd0, cr2_dma_len};
            4'd3:  cp_rdata = {25'd0, cr3_dma_ctrl};
            4'd4:  cp_rdata = cr4_gpu_pc;
            4'd5:  cp_rdata = {29'd0, cr5_gpu_ctrl};
            4'd6:  cp_rdata = cr6_gpu_status;
            4'd7:  cp_rdata = {28'd0, cr7_thread_mask};
            4'd8:  cp_rdata = cr8_gpu_scratch;
            4'd9:  cp_rdata = dma_cur_addr;
            default: cp_rdata = 32'd0;
        endcase
    end

    // ================================================================
    // Output assignments
    // ================================================================

    // DMA outputs
    assign dma_src_addr = cr0_dma_src;
    assign dma_dst_addr = cr1_dma_dst;
    assign dma_xfer_len = cr2_dma_len;
    assign dma_dir = auto_dma_trigger ? 1'b1         // auto_dma forces GPU→CPU
                                      : cr3_dma_ctrl[1];
    assign dma_tgt = cr3_dma_ctrl[2];
    assign dma_bank = cr3_dma_ctrl[4:3];
    assign dma_auto_inc = cr3_dma_ctrl[5];
    assign dma_burst_all = cr3_dma_ctrl[6];
    assign dma_start = dma_start_r | auto_dma_trigger;

    // GPU outputs
    assign gpu_kernel_start = gpu_start_r;
    assign gpu_reset_n = ~cr5_gpu_ctrl[1]; // bit[1]=1 → reset active (rst_n=0)
    assign gpu_entry_pc = cr4_gpu_pc;
    assign gpu_thread_mask = cr7_thread_mask;
    assign gpu_scratch = cr8_gpu_scratch;

endmodule

`endif // CP10_REGFILE_V