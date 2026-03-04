/* file: sm_top.v
 Description: This file implements the Streaming Multiprocessor (SM) top module, which includes the SM core,
    test IMEM, and test DMEM. 
 Date: Feb. 28, 2026
 Version: 1.0
 Revision history:
    - Feb. 28, 2026: Initial implementation of the SM core.
*/

`ifndef SM_TOP_V
`define SM_TOP_V

`include "gpu_define.v"
`include "sm_core.v"
`include "test_gpu_imem.v"
`include "test_gpu_dmem.v"

module sm_top (
    input wire clk,
    input wire rst_n,

    // Kernel control
    input wire kernel_start,
    input wire [`GPU_PC_WIDTH-1:0] kernel_entry_pc,
    output wire kernel_done,

    // External IMEM port B (program load)
    input wire [`GPU_IMEM_ADDR_WIDTH-1:0] ext_imem_addr,
    input wire [`GPU_IMEM_DATA_WIDTH-1:0] ext_imem_din,
    input wire ext_imem_we,
    output wire [`GPU_IMEM_DATA_WIDTH-1:0] ext_imem_dout,

    // External DMEM port B (data preload / readback)
    //    2-bit sel picks which thread's DMEM is accessed
    input wire [1:0] ext_dmem_sel,
    input wire [`GPU_DMEM_ADDR_WIDTH-1:0] ext_dmem_addr,
    input wire [`GPU_DMEM_DATA_WIDTH-1:0] ext_dmem_din,
    input wire ext_dmem_we,
    output wire [`GPU_DMEM_DATA_WIDTH-1:0] ext_dmem_dout
);

    // ================================================================
    // Internal wires — sm_core ↔ memories
    // ================================================================

    // IMEM interface (core side)
    wire [`GPU_IMEM_ADDR_WIDTH-1:0] core_imem_addr;
    wire [`GPU_IMEM_DATA_WIDTH-1:0] core_imem_rdata;

    // DMEM interface (core side, concatenated across 4 threads)
    wire [4*`GPU_DMEM_ADDR_WIDTH-1:0] core_dmem_addr;
    wire [4*`GPU_DMEM_DATA_WIDTH-1:0] core_dmem_din;
    wire [3:0] core_dmem_we;
    wire [4*`GPU_DMEM_DATA_WIDTH-1:0] core_dmem_dout;

    // Per-thread DMEM port B read data
    // Flaten due to XST synthesis issue with 2D arrays of outputs from generate
    wire [4*`GPU_DMEM_DATA_WIDTH-1:0] dmem_doutb_flat;

    // Per-thread DMEM port B write enable (decoded from ext_dmem_sel)
    wire [3:0] dmem_web;
    assign dmem_web[0] = ext_dmem_we & (ext_dmem_sel == 2'd0);
    assign dmem_web[1] = ext_dmem_we & (ext_dmem_sel == 2'd1);
    assign dmem_web[2] = ext_dmem_we & (ext_dmem_sel == 2'd2);
    assign dmem_web[3] = ext_dmem_we & (ext_dmem_sel == 2'd3);

    // Mux port B read data based on ext_dmem_sel
    reg [`GPU_DMEM_DATA_WIDTH-1:0] ext_dmem_dout_r;

    always @(*) begin
        case (ext_dmem_sel)
            2'd0: ext_dmem_dout_r = dmem_doutb_flat[0*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
            2'd1: ext_dmem_dout_r = dmem_doutb_flat[1*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
            2'd2: ext_dmem_dout_r = dmem_doutb_flat[2*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
            2'd3: ext_dmem_dout_r = dmem_doutb_flat[3*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH];
        endcase
    end

    assign ext_dmem_dout = ext_dmem_dout_r;

    // ================================================================
    // SM Core
    // ================================================================
    sm_core u_sm_core (
        .clk (clk),
        .rst_n (rst_n),
        .imem_addr (core_imem_addr),
        .imem_rdata (core_imem_rdata),
        .dmem_addra (core_dmem_addr),
        .dmem_dina (core_dmem_din),
        .dmem_wea (core_dmem_we),
        .dmem_douta (core_dmem_dout),
        .kernel_start (kernel_start),
        .kernel_entry_pc (kernel_entry_pc),
        .kernel_done (kernel_done)
    );

    // ================================================================
    // Instruction Memory (1× dual-port BRAM)
    //   Port A: sm_core fetch (read-only)
    //   Port B: external program load
    // ================================================================
    test_gpu_imem u_imem (
        // Port A — core fetch (read-only)
        .clka (clk),
        .addra (core_imem_addr),
        .dina ({`GPU_IMEM_DATA_WIDTH{1'b0}}),
        .wea (1'b0),
        .douta (core_imem_rdata),
        // Port B — external load / readback
        .clkb (clk),
        .addrb (ext_imem_addr),
        .dinb (ext_imem_din),
        .web (ext_imem_we),
        .doutb (ext_imem_dout)
    );

    // ================================================================
    // Data Memories (4× dual-port BRAM, one per thread)
    //   Port A: sm_core load/store
    //   Port B: external data preload / readback (muxed by ext_dmem_sel)
    // ================================================================
    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : DMEM_LANE
            test_gpu_dmem u_dmem (
                // Port A — core load/store
                .clka (clk),
                .addra (core_dmem_addr[gi*`GPU_DMEM_ADDR_WIDTH +: `GPU_DMEM_ADDR_WIDTH]),
                .dina (core_dmem_din[gi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]),
                .wea (core_dmem_we[gi]),
                .douta (core_dmem_dout[gi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH]),
                // Port B — external access
                .clkb (clk),
                .addrb (ext_dmem_addr),
                .dinb (ext_dmem_din),
                .web (dmem_web[gi]),
                .doutb (dmem_doutb_flat[gi*`GPU_DMEM_DATA_WIDTH +: `GPU_DMEM_DATA_WIDTH])
            );
        end
    endgenerate

endmodule

`endif // SM_TOP_V