/* file: top.v
 * Description: Top module — CPU SoC + ILA + SoC Driver
 * Author: Jeremy Cai
 * Date: Feb. 14, 2026
 * Version: 1.1
 */

`include "define.v"
`include "ila.v"
`include "soc.v"
`include "soc_driver.v"

module top (
    input  wire clk,
    input  wire rst_n,

    // System Control
    input  wire debug_mode,

    // User Transaction Interface (To SoC Driver)
    input  wire user_valid,
    input  wire user_cmd,       // 0=Read, 1=Write
    input  wire [`MMIO_ADDR_WIDTH-1:0] user_addr,
    input  wire [`MMIO_DATA_WIDTH-1:0] user_wdata,
    output wire user_ready,
    output wire [`MMIO_DATA_WIDTH-1:0] user_rdata,

    // Driver Status / Quality
    output wire [`MMIO_ADDR_WIDTH-1:0] status,
    output wire [7:0] conn_status,
    output wire [`MMIO_DATA_WIDTH-1:0] txn_quality,
    output wire [`MMIO_DATA_WIDTH-1:0] txn_counters,
    input  wire clear_stats,

    // ILA Control Interface (Direct Debug Access)
    input  wire [2:0] ila_addr,
    input  wire [`MMIO_DATA_WIDTH-1:0] ila_din,
    input  wire ila_we,
    output wire [`MMIO_DATA_WIDTH-1:0] ila_dout
);

    // Clock Gating
    wire soc_clk_en;
    wire soc_gated_clk;

    // MMIO Bus: Driver → SoC
    wire                         req_val;
    wire                         req_rdy;
    wire                         req_cmd;
    wire [`MMIO_ADDR_WIDTH-1:0]  req_addr;
    wire [`MMIO_DATA_WIDTH-1:0]  req_data;

    // MMIO Bus: SoC → Driver
    wire                         resp_val;
    wire                         resp_rdy;
    wire                         resp_cmd;
    wire [`MMIO_ADDR_WIDTH-1:0]  resp_addr;
    wire [`MMIO_DATA_WIDTH-1:0]  resp_data;

    // Debug Interface: ILA ↔ SoC
    wire [4:0]                   cpu_debug_sel;
    wire [`DATA_WIDTH-1:0]       cpu_debug_data;
    wire                         ila_soc_start;

    // Busy Signal: Driver → ILA  (keep SoC clock alive during MMIO)
    wire driver_busy;
    assign driver_busy = conn_status[5]; // bit 5 = (state != IDLE)

    reg soc_clk_en_latch;

    always @(negedge clk or negedge rst_n) begin
        if (!rst_n) soc_clk_en_latch <= 1'b1;   // FIX: was 1'b0
        else        soc_clk_en_latch <= soc_clk_en;
    end

    assign soc_gated_clk = clk & soc_clk_en_latch;

    soc_driver #(
        .TIMEOUT_THRESHOLD(16'd1000)
    ) u_driver (
        .clk          (clk),
        .rst_n        (rst_n),

        .user_valid   (user_valid),
        .user_ready   (user_ready),
        .user_cmd     (user_cmd),
        .user_addr    (user_addr),
        .user_wdata   (user_wdata),
        .user_rdata   (user_rdata),

        .status       (status),
        .conn_status  (conn_status),
        .txn_quality  (txn_quality),
        .txn_counters (txn_counters),
        .clear_stats  (clear_stats),

        .soc_req_val  (req_val),
        .soc_req_rdy  (req_rdy),
        .soc_req_cmd  (req_cmd),
        .soc_req_addr (req_addr),
        .soc_req_data (req_data),

        .soc_resp_val (resp_val),
        .soc_resp_rdy (resp_rdy),
        .soc_resp_cmd (resp_cmd),
        .soc_resp_addr(resp_addr),
        .soc_resp_data(resp_data)
    );

    ila u_ila (
        .clk            (clk),
        .rst_n          (rst_n),

        .ila_addr       (ila_addr),
        .ila_din        (ila_din),
        .ila_we         (ila_we),
        .ila_dout       (ila_dout),

        .cpu_debug_sel  (cpu_debug_sel),
        .cpu_debug_data (cpu_debug_data),

        .soc_start      (ila_soc_start),
        .debug_mode     (debug_mode),
        .soc_clk_en     (soc_clk_en),

        .soc_busy       (driver_busy)
    );

    soc u_soc (
        .clk            (soc_gated_clk),
        .rst_n          (rst_n),

        .req_cmd        (req_cmd),
        .req_addr       (req_addr),
        .req_data       (req_data),
        .req_val        (req_val),
        .req_rdy        (req_rdy),

        .resp_cmd       (resp_cmd),
        .resp_addr      (resp_addr),
        .resp_data      (resp_data),
        .resp_val       (resp_val),
        .resp_rdy       (resp_rdy),

        .start          (ila_soc_start),
        .ila_debug_sel  (cpu_debug_sel),
        .ila_debug_data (cpu_debug_data)
    );

endmodule