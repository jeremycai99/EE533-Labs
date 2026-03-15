/* file: soc.v
 Description: SoC top-level wrapper — v8 FGMT architecture.
   4-thread barrel ARM + 4-thread SIMT GPU + pkt_proc + DMA + conv FIFO.
   pkt_proc is the sole FIFO master. No CP0 — CPU never touches FIFO.
   CP10 bridges CPU to DMA engine and GPU.
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 Version: 1.0
*/

`ifndef SOC_V
`define SOC_V

`include "define.v"
`include "gpu_define.v"
`include "cpu_mt.v"
`include "cp10_regfile.v"
`include "dma_engine.v"
`include "pkt_proc.v"
`include "conv_fifo.v"
`include "sm_core.v"
`include "test_i_mem.v"
`include "test_d_mem.v"
`include "test_gpu_imem.v"
`include "test_gpu_dmem.v"

module soc (
    input wire clk,
    input wire rst_n,

    // NetFPGA RX (from output_port_lookup)
    input wire [63:0] in_data,
    input wire [7:0] in_ctrl,
    input wire in_wr,
    output wire in_rdy,

    // NetFPGA TX (to output_queues)
    output wire [63:0] out_data,
    output wire [7:0] out_ctrl,
    output wire out_wr,
    input wire out_rdy
);

// ================================================================
//   INTERNAL WIRES — pkt_proc <-> conv_fifo
// ================================================================
wire [11:0] pp_fifo_addr;
wire [63:0] pp_fifo_wdata;
wire pp_fifo_we;
wire [63:0] pp_fifo_rdata;
wire [1:0] pp_fifo_mode;
wire [11:0] pp_fifo_head_wr_data, pp_fifo_tail_wr_data;
wire pp_fifo_head_wr, pp_fifo_tail_wr;
wire pp_fifo_tx_start, pp_fifo_pkt_ack;
wire [11:0] fifo_head_ptr, fifo_tail_ptr, fifo_pkt_end;
wire fifo_pkt_ready, fifo_tx_done;
wire fifo_nearly_full, fifo_empty, fifo_full;

// ================================================================
//   INTERNAL WIRES — pkt_proc <-> CPU memories
// ================================================================
wire [`IMEM_ADDR_WIDTH-1:0] pp_imem_addr;
wire [31:0] pp_imem_din;
wire pp_imem_we;

wire [`DMEM_ADDR_WIDTH-1:0] pp_dmem_addr;
wire [31:0] pp_dmem_din;
wire pp_dmem_we;
wire [31:0] pp_dmem_dout;

// ================================================================
//   INTERNAL WIRES — pkt_proc <-> CPU control
// ================================================================
wire pp_cpu_rst_n;
wire pp_cpu_start;
wire [31:0] pp_entry_pc;
wire pp_active, pp_owns_port_b;

// ================================================================
//   INTERNAL WIRES — CPU <-> memories
// ================================================================
wire [`PC_WIDTH-1:0] cpu_imem_byte_addr;
wire [`INSTR_WIDTH-1:0] cpu_imem_rdata;

wire [`CPU_DMEM_ADDR_WIDTH-1:0] cpu_dmem_byte_addr;
wire [`DATA_WIDTH-1:0] cpu_dmem_rdata;
wire [`DATA_WIDTH-1:0] cpu_dmem_wdata;
wire cpu_dmem_wen;
wire [1:0] cpu_dmem_size;
wire cpu_done_w;

// ================================================================
//   INTERNAL WIRES — CPU <-> CP10
// ================================================================
wire cp_wen, cp_ren;
wire [3:0] cp_reg;
wire [31:0] cp_wr_data, cp_rd_data;

// ================================================================
//   INTERNAL WIRES — CP10 <-> DMA
// ================================================================
wire [31:0] dma_src_addr, dma_dst_addr;
wire [15:0] dma_xfer_len;
wire dma_start, dma_dir, dma_tgt, dma_auto_inc, dma_burst_all;
wire [1:0] dma_bank;
wire dma_busy, dma_error;
wire [31:0] dma_cur_addr;

// ================================================================
//   INTERNAL WIRES — CP10 <-> GPU
// ================================================================
wire gpu_kernel_start_w, gpu_reset_n_w;
wire [31:0] gpu_entry_pc_w, gpu_scratch_w;
wire [3:0] gpu_thread_mask_w;
wire gpu_kernel_done_w;

/* gpu_active: flop set on kernel_start, cleared on kernel_done */
reg gpu_active_r;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) gpu_active_r <= 1'b0;
    else if (gpu_kernel_start_w) gpu_active_r <= 1'b1;
    else if (gpu_kernel_done_w) gpu_active_r <= 1'b0;
end

// ================================================================
//   INTERNAL WIRES — DMA <-> CPU DMEM Port B
// ================================================================
wire [11:0] dma_cpu_dmem_addr;
wire [31:0] dma_cpu_dmem_din;
wire dma_cpu_dmem_we;
wire [31:0] dma_cpu_dmem_dout;

// ================================================================
//   INTERNAL WIRES — DMA <-> GPU IMEM Port B
// ================================================================
wire [7:0] dma_gpu_imem_addr;
wire [31:0] dma_gpu_imem_din;
wire dma_gpu_imem_we;

// ================================================================
//   INTERNAL WIRES — DMA <-> GPU DMEM Port B
// ================================================================
wire [1:0] dma_gpu_dmem_sel;
wire [9:0] dma_gpu_dmem_addr;
wire [15:0] dma_gpu_dmem_din;
wire dma_gpu_dmem_we;
wire [15:0] dma_gpu_dmem_dout;

// ================================================================
//   INTERNAL WIRES — sm_core <-> GPU memories
// ================================================================
wire [`GPU_IMEM_ADDR_WIDTH-1:0] core_imem_addr;
wire [`GPU_IMEM_DATA_WIDTH-1:0] core_imem_rdata;

wire [4*`GPU_DMEM_ADDR_WIDTH-1:0] core_dmem_addr;
wire [4*`GPU_DMEM_DATA_WIDTH-1:0] core_dmem_din;
wire [3:0] core_dmem_we;
wire [4*`GPU_DMEM_DATA_WIDTH-1:0] core_dmem_dout;

// ================================================================
//   CONVERTIBLE FIFO
// ================================================================
conv_fifo #(
    .ADDR_WIDTH(12), .DATA_WIDTH(64), .CTRL_WIDTH(8)
) u_conv_fifo (
    .clk(clk), .rst_n(rst_n),
    .mode(pp_fifo_mode),
    // RX (Port A)
    .in_data(in_data), .in_ctrl(in_ctrl), .in_wr(in_wr), .in_rdy(in_rdy),
    // TX
    .out_data(out_data), .out_ctrl(out_ctrl), .out_wr(out_wr), .out_rdy(out_rdy),
    // TX drain control
    .tx_start(pp_fifo_tx_start), .pkt_ack(pp_fifo_pkt_ack), .tx_done(fifo_tx_done),
    // SRAM Port B (pkt_proc)
    .sram_addr(pp_fifo_addr), .sram_wdata(pp_fifo_wdata),
    .sram_we(pp_fifo_we), .sram_rdata(pp_fifo_rdata),
    // Pointer I/O
    .head_ptr_in(pp_fifo_head_wr_data), .head_ptr_wr(pp_fifo_head_wr),
    .tail_ptr_in(pp_fifo_tail_wr_data), .tail_ptr_wr(pp_fifo_tail_wr),
    .head_ptr_out(fifo_head_ptr), .tail_ptr_out(fifo_tail_ptr),
    .pkt_end_ptr(fifo_pkt_end),
    // Status
    .pkt_ready(fifo_pkt_ready), .nearly_full(fifo_nearly_full),
    .fifo_empty(fifo_empty), .fifo_full(fifo_full)
);

// ================================================================
//   PACKET PROCESSOR (sole FIFO master)
// ================================================================
pkt_proc #(
    .FIFO_ADDR_WIDTH(12),
    .IMEM_ADDR_WIDTH(`IMEM_ADDR_WIDTH),
    .DMEM_ADDR_WIDTH(`DMEM_ADDR_WIDTH)
) u_pkt_proc (
    .clk(clk), .rst_n(rst_n),
    // FIFO interface
    .fifo_addr(pp_fifo_addr), .fifo_wdata(pp_fifo_wdata),
    .fifo_we(pp_fifo_we), .fifo_rdata(pp_fifo_rdata),
    .fifo_mode(pp_fifo_mode),
    .fifo_head_wr_data(pp_fifo_head_wr_data), .fifo_head_wr(pp_fifo_head_wr),
    .fifo_tail_wr_data(pp_fifo_tail_wr_data), .fifo_tail_wr(pp_fifo_tail_wr),
    .fifo_tx_start(pp_fifo_tx_start),
    .fifo_head_ptr(fifo_head_ptr), .fifo_pkt_end(fifo_pkt_end),
    .fifo_pkt_ready(fifo_pkt_ready), .fifo_pkt_ack(pp_fifo_pkt_ack),
    .fifo_tx_done(fifo_tx_done),
    // CPU IMEM Port B
    .imem_addr(pp_imem_addr), .imem_din(pp_imem_din), .imem_we(pp_imem_we),
    // CPU DMEM Port B (direct — muxed externally)
    .dmem_addr(pp_dmem_addr), .dmem_din(pp_dmem_din),
    .dmem_we(pp_dmem_we), .dmem_dout(pp_dmem_dout),
    // CPU control
    .cpu_rst_n(pp_cpu_rst_n), .cpu_start(pp_cpu_start),
    .entry_pc(pp_entry_pc), .cpu_done(cpu_done_w),
    // Status
    .active(pp_active), .owns_port_b(pp_owns_port_b)
);

// ================================================================
//   CPU IMEM — 4096×32b Dual-Port
//     Port A: CPU fetch (read-only, byte addr → word addr)
//     Port B: pkt_proc write (LOAD_IMEM, word-addressed)
// ================================================================
test_i_mem u_cpu_imem (
    .clka(clk),
    .addra(cpu_imem_byte_addr[`IMEM_ADDR_WIDTH+1:2]),
    .dina({`IMEM_DATA_WIDTH{1'b0}}),
    .wea(1'b0),
    .douta(cpu_imem_rdata),
    .clkb(clk),
    .addrb(pp_imem_addr),
    .dinb(pp_imem_din),
    .web(pp_imem_we),
    .doutb()
);

// ================================================================
//   CPU DMEM — 4096×32b Dual-Port
//     Port A: CPU R/W (byte addr → word addr)
//     Port B: muxed (pkt_proc | DMA, word-addressed)
// ================================================================
wire [`DMEM_ADDR_WIDTH-1:0] dmem_portb_addr;
wire [31:0] dmem_portb_din;
wire dmem_portb_we;
wire [31:0] dmem_portb_dout;

/* Port B Mux: pkt_proc vs DMA, selected by pp_owns_port_b */
assign dmem_portb_addr = pp_owns_port_b ? pp_dmem_addr : dma_cpu_dmem_addr;
assign dmem_portb_din = pp_owns_port_b ? pp_dmem_din : dma_cpu_dmem_din;
assign dmem_portb_we = pp_owns_port_b ? pp_dmem_we : dma_cpu_dmem_we;

/* Route Port B read data back to both consumers */
assign pp_dmem_dout = dmem_portb_dout;
assign dma_cpu_dmem_dout = dmem_portb_dout;

test_d_mem u_cpu_dmem (
    .clka(clk),
    .addra(cpu_dmem_byte_addr[`DMEM_ADDR_WIDTH+1:2]),
    .dina(cpu_dmem_wdata),
    .wea(cpu_dmem_wen),
    .douta(cpu_dmem_rdata),
    .clkb(clk),
    .addrb(dmem_portb_addr),
    .dinb(dmem_portb_din),
    .web(dmem_portb_we),
    .doutb(dmem_portb_dout)
);

// ================================================================
//   ARM CPU — 7-stage FGMT barrel (cpu_mt v2.8)
//     rst_n gated by pkt_proc's cpu_rst_n
// ================================================================
wire cpu_rst_gated = rst_n & pp_cpu_rst_n;

cpu_mt u_cpu_mt (
    .clk(clk), .rst_n(cpu_rst_gated),
    .cpu_start_i(pp_cpu_start), .entry_pc_i(pp_entry_pc),
    // IMEM
    .i_mem_data_i(cpu_imem_rdata), .i_mem_addr_o(cpu_imem_byte_addr),
    // DMEM
    .d_mem_data_i(cpu_dmem_rdata), .d_mem_addr_o(cpu_dmem_byte_addr),
    .d_mem_data_o(cpu_dmem_wdata), .d_mem_wen_o(cpu_dmem_wen),
    .d_mem_size_o(cpu_dmem_size),
    // CP10
    .cp_wen_o(cp_wen), .cp_ren_o(cp_ren), .cp_reg_o(cp_reg),
    .cp_wr_data_o(cp_wr_data), .cp_rd_data_i(cp_rd_data),
    .cpu_done(cpu_done_w)
);

// ================================================================
//   CP10 — Compute Coprocessor Register File
// ================================================================
cp10_regfile u_cp10 (
    .clk(clk), .rst_n(rst_n),
    .cp_wen(cp_wen), .cp_ren(cp_ren), .cp_reg(cp_reg),
    .cp_wdata(cp_wr_data), .cp_rdata(cp_rd_data),
    // DMA
    .dma_src_addr(dma_src_addr), .dma_dst_addr(dma_dst_addr),
    .dma_xfer_len(dma_xfer_len), .dma_start(dma_start),
    .dma_dir(dma_dir), .dma_tgt(dma_tgt), .dma_bank(dma_bank),
    .dma_auto_inc(dma_auto_inc), .dma_burst_all(dma_burst_all),
    .dma_busy(dma_busy), .dma_error(dma_error), .dma_cur_addr(dma_cur_addr),
    // GPU
    .gpu_kernel_start(gpu_kernel_start_w), .gpu_reset_n(gpu_reset_n_w),
    .gpu_entry_pc(gpu_entry_pc_w), .gpu_thread_mask(gpu_thread_mask_w),
    .gpu_scratch(gpu_scratch_w),
    .gpu_kernel_done(gpu_kernel_done_w), .gpu_active(gpu_active_r)
);

// ================================================================
//   DMA ENGINE
// ================================================================ */
dma_engine u_dma (
    .clk(clk), .rst_n(rst_n),
    // CP10 control
    .dma_src_addr(dma_src_addr), .dma_dst_addr(dma_dst_addr),
    .dma_xfer_len(dma_xfer_len), .dma_start(dma_start),
    .dma_dir(dma_dir), .dma_tgt(dma_tgt), .dma_bank(dma_bank),
    .dma_auto_inc(dma_auto_inc), .dma_burst_all(dma_burst_all),
    .dma_busy(dma_busy), .dma_error(dma_error), .dma_cur_addr(dma_cur_addr),
    // CPU DMEM Port B
    .cpu_dmem_addr(dma_cpu_dmem_addr), .cpu_dmem_din(dma_cpu_dmem_din),
    .cpu_dmem_we(dma_cpu_dmem_we), .cpu_dmem_dout(dma_cpu_dmem_dout),
    // GPU IMEM Port B
    .gpu_imem_addr(dma_gpu_imem_addr), .gpu_imem_din(dma_gpu_imem_din),
    .gpu_imem_we(dma_gpu_imem_we),
    // GPU DMEM Port B
    .gpu_dmem_sel(dma_gpu_dmem_sel), .gpu_dmem_addr(dma_gpu_dmem_addr),
    .gpu_dmem_din(dma_gpu_dmem_din), .gpu_dmem_we(dma_gpu_dmem_we),
    .gpu_dmem_dout(dma_gpu_dmem_dout)
);

// ================================================================
//   GPU IMEM — 256×32b Dual-Port
//     Port A: sm_core fetch (read-only)
//     Port B: DMA ext write
// ================================================================
test_gpu_imem u_gpu_imem (
    .clka(clk),
    .addra(core_imem_addr),
    .dina({`GPU_IMEM_DATA_WIDTH{1'b0}}),
    .wea(1'b0),
    .douta(core_imem_rdata),
    .clkb(clk),
    .addrb(dma_gpu_imem_addr),
    .dinb(dma_gpu_imem_din),
    .web(dma_gpu_imem_we),
    .doutb()
);

// ================================================================
//   GPU DMEM — 4× 1024×16b Dual-Port Banks
//     Port A: sm_core R/W (flattened bus)
//     Port B: DMA ext R/W (muxed by dma_gpu_dmem_sel)
// ================================================================
wire [3:0] gpu_dmem_web;
assign gpu_dmem_web[0] = dma_gpu_dmem_we & (dma_gpu_dmem_sel == 2'd0);
assign gpu_dmem_web[1] = dma_gpu_dmem_we & (dma_gpu_dmem_sel == 2'd1);
assign gpu_dmem_web[2] = dma_gpu_dmem_we & (dma_gpu_dmem_sel == 2'd2);
assign gpu_dmem_web[3] = dma_gpu_dmem_we & (dma_gpu_dmem_sel == 2'd3);

wire [4*`GPU_DMEM_DATA_WIDTH-1:0] gpu_dmem_doutb_flat;

genvar gi;
generate
    for (gi = 0; gi < 4; gi = gi + 1) begin : GPU_DMEM_BANK
        test_gpu_dmem u_gpu_dmem (
            // Port A: sm_core
            .clka(clk),
            .addra(core_dmem_addr[gi*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]),
            .dina(core_dmem_din[gi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]),
            .wea(core_dmem_we[gi]),
            .douta(core_dmem_dout[gi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]),
            // Port B: DMA ext
            .clkb(clk),
            .addrb(dma_gpu_dmem_addr),
            .dinb(dma_gpu_dmem_din),
            .web(gpu_dmem_web[gi]),
            .doutb(gpu_dmem_doutb_flat[gi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH])
        );
    end
endgenerate

/* Port B read mux: select bank based on dma_gpu_dmem_sel */
reg [`GPU_DMEM_DATA_WIDTH-1:0] gpu_dmem_doutb_mux;
always @(*) begin
    case (dma_gpu_dmem_sel)
        2'd0: gpu_dmem_doutb_mux = gpu_dmem_doutb_flat[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        2'd1: gpu_dmem_doutb_mux = gpu_dmem_doutb_flat[1*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        2'd2: gpu_dmem_doutb_mux = gpu_dmem_doutb_flat[2*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        2'd3: gpu_dmem_doutb_mux = gpu_dmem_doutb_flat[3*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
    endcase
end
assign dma_gpu_dmem_dout = gpu_dmem_doutb_mux;

// ================================================================
//   SM CORE — 4-thread SIMT GPU
//     rst_n gated by CP10's gpu_reset_n (active-low from CR5[1])
// ================================================================
wire gpu_rst_gated = rst_n & gpu_reset_n_w;

sm_core u_sm_core (
    .clk(clk), .rst_n(gpu_rst_gated),
    // IMEM
    .imem_addr(core_imem_addr), .imem_rdata(core_imem_rdata),
    // DMEM (flattened 4-bank)
    .dmem_addra(core_dmem_addr), .dmem_dina(core_dmem_din),
    .dmem_wea(core_dmem_we), .dmem_douta(core_dmem_dout),
    // Kernel control
    .kernel_start(gpu_kernel_start_w),
    .kernel_entry_pc(gpu_entry_pc_w[`GPU_PC_WIDTH-1:0]),
    .thread_mask(gpu_thread_mask_w),
    .kernel_done(gpu_kernel_done_w)
);

endmodule

`endif // SOC_V