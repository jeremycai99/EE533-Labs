/* file: conv_fifo.v
Description: This file contains the implementation of the convertible FIFO
Author: Jeremy Cai
Date: Mar. 4, 2026
Version: 1.0
Revision history:
    - Mar. 4, 2026: v1.0 — Initial implementation of the convertible FIFO.
*/

`ifndef CONV_FIFO_V
`define CONV_FIFO_V

`include "test_dpfifo.v"

module conv_fifo #(
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 64,
    parameter CTRL_WIDTH = 8,
    parameter NEARLY_FULL_THRESH = 4
)(
    input wire clk,
    input wire rst_n,

    // Mode control
    input wire [1:0] mode, // 0=RX_FIFO, 1=SRAM, 2=TX_DRAIN

    // NetFPGA RX interface (Port A write side)
    input wire [DATA_WIDTH-1:0] in_data,
    input wire [CTRL_WIDTH-1:0] in_ctrl,
    input wire in_wr,
    output wire in_rdy,

    // NetFPGA TX interface (Port B read side)
    output wire [DATA_WIDTH-1:0] out_data,
    output reg [CTRL_WIDTH-1:0] out_ctrl,
    output wire out_wr,
    input wire out_rdy,

    // TX drain control
    input wire tx_start,
    input wire pkt_ack, // clears pkt_ready (from pkt_proc)
    output reg tx_done,

    // External SRAM Port B access (pkt_proc)
    input wire [ADDR_WIDTH-1:0] sram_addr,
    input wire [DATA_WIDTH-1:0] sram_wdata,
    input wire sram_we,
    output wire [DATA_WIDTH-1:0] sram_rdata,

    // Pointer I/O
    input wire [ADDR_WIDTH-1:0] head_ptr_in,
    input wire head_ptr_wr,
    input wire [ADDR_WIDTH-1:0] tail_ptr_in,
    input wire tail_ptr_wr,
    output wire [ADDR_WIDTH-1:0] head_ptr_out,
    output wire [ADDR_WIDTH-1:0] tail_ptr_out,
    output wire [ADDR_WIDTH-1:0] pkt_end_ptr,

    // Status
    output wire pkt_ready,
    output wire nearly_full,
    output wire fifo_empty,
    output wire fifo_full
);

    localparam DEPTH = 1 << ADDR_WIDTH;

    localparam [1:0] MODE_RX_FIFO  = 2'd0;
    localparam [1:0] MODE_SRAM     = 2'd1;
    localparam [1:0] MODE_TX_DRAIN = 2'd2;

    // ================================================================
    // Dual-Port BRAM (64b data only, no ctrl)
    // ================================================================
    reg [ADDR_WIDTH-1:0] porta_addr;
    reg [DATA_WIDTH-1:0] porta_din;
    reg porta_we;

    reg [ADDR_WIDTH-1:0] portb_addr;
    reg [DATA_WIDTH-1:0] portb_din;
    reg portb_we;
    wire [DATA_WIDTH-1:0] portb_dout;

    test_dpfifo #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_bram (
        .clka(clk),
        .addra(porta_addr),
        .dina(porta_din),
        .wea(porta_we),
        .douta(),
        .clkb(clk),
        .addrb(portb_addr),
        .dinb(portb_din),
        .web(portb_we),
        .doutb(portb_dout)
    );

    // ================================================================
    // Head / Tail Pointer Management
    // ================================================================
    reg [ADDR_WIDTH-1:0] head_ptr;
    reg [ADDR_WIDTH-1:0] tail_ptr;
    reg [ADDR_WIDTH-1:0] pkt_end_r;
    reg pkt_ready_r;

    assign head_ptr_out = head_ptr;
    assign tail_ptr_out = tail_ptr;
    assign pkt_end_ptr = pkt_end_r;
    assign pkt_ready = pkt_ready_r;

    wire [ADDR_WIDTH-1:0] word_count = tail_ptr - head_ptr;

    assign fifo_empty = (word_count == {ADDR_WIDTH{1'b0}});
    assign fifo_full = (word_count == {ADDR_WIDTH{1'b1}});
    assign nearly_full = (word_count >= ({ADDR_WIDTH{1'b1}} - NEARLY_FULL_THRESH));
    assign in_rdy = (mode == MODE_RX_FIFO) & ~nearly_full;

    // ================================================================
    // RX: Capture in_ctrl on module header (word 0)
    // ================================================================
    // NetFPGA convention: ctrl != 0 on word 0 (module header), 0 on payload.
    // We capture it — this is the output port routing from output_port_lookup.
    reg [CTRL_WIDTH-1:0] rx_ctrl_r;

    // Packet boundary: in_wr falling edge
    reg in_wr_prev;
    wire rx_pkt_done = in_wr_prev & ~in_wr & (mode == MODE_RX_FIFO);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            in_wr_prev <= 1'b0;
        else
            in_wr_prev <= in_wr & (mode == MODE_RX_FIFO);
    end

    // ================================================================
    // TX Drain FSM
    // ================================================================
    localparam TX_IDLE = 2'd0;
    localparam TX_READ = 2'd1;
    localparam TX_SEND = 2'd2;
    localparam TX_DONE = 2'd3;

    reg [1:0] tx_state;
    reg [ADDR_WIDTH-1:0] tx_ptr;
    reg [ADDR_WIDTH-1:0] tx_end;
    reg tx_first_word;

    // TX output
    assign out_data = portb_dout;
    assign out_wr = (tx_state == TX_SEND);

    // out_ctrl: replay captured rx_ctrl_r on word 0, zero otherwise.
    // This mirrors the IDS transparent passthrough — the port routing
    // set by output_port_lookup comes through unchanged.
    always @(*) begin
        if ((tx_state == TX_SEND) && tx_first_word)
            out_ctrl = rx_ctrl_r;
        else
            out_ctrl = {CTRL_WIDTH{1'b0}};
    end

    // SRAM read output (Port B in non-TX modes)
    assign sram_rdata = portb_dout;

    // ================================================================
    // Port A Mux (RX write)
    // ================================================================
    always @(*) begin
        if (mode == MODE_RX_FIFO && in_wr && !nearly_full) begin
            porta_addr = tail_ptr;
            porta_din = in_data;
            porta_we = 1'b1;
        end else begin
            porta_addr = {ADDR_WIDTH{1'b0}};
            porta_din = {DATA_WIDTH{1'b0}};
            porta_we = 1'b0;
        end
    end

    // ================================================================
    // Port B Mux (TX drain vs external SRAM)
    // ================================================================
    always @(*) begin
        if (mode == MODE_TX_DRAIN && tx_state != TX_IDLE) begin
            portb_addr = tx_ptr;
            portb_din = {DATA_WIDTH{1'b0}};
            portb_we = 1'b0;
        end else begin
            portb_addr = sram_addr;
            portb_din = sram_wdata;
            portb_we = sram_we;
        end
    end

    // ================================================================
    // Main Sequential Logic
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head_ptr <= {ADDR_WIDTH{1'b0}};
            tail_ptr <= {ADDR_WIDTH{1'b0}};
            pkt_end_r <= {ADDR_WIDTH{1'b0}};
            pkt_ready_r <= 1'b0;
            rx_ctrl_r <= {CTRL_WIDTH{1'b0}};
            tx_state <= TX_IDLE;
            tx_ptr <= {ADDR_WIDTH{1'b0}};
            tx_end <= {ADDR_WIDTH{1'b0}};
            tx_first_word <= 1'b0;
            tx_done <= 1'b0;
        end else begin
            tx_done <= 1'b0;

            // Pointer writes from pkt_proc
            if (head_ptr_wr) head_ptr <= head_ptr_in;
            if (tail_ptr_wr) tail_ptr <= tail_ptr_in;

            // RX_FIFO: auto-advance tail, capture ctrl
            if (mode == MODE_RX_FIFO && in_wr && !nearly_full) begin
                tail_ptr <= tail_ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                // Capture ctrl on module header (ctrl != 0)
                if (|in_ctrl)
                    rx_ctrl_r <= in_ctrl;
            end

            // RX packet boundary
            if (rx_pkt_done) begin
                pkt_end_r <= tail_ptr - {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                pkt_ready_r <= 1'b1;
            end

            // pkt_proc acknowledges — clear pkt_ready
            if (pkt_ack)
                pkt_ready_r <= 1'b0;

            // TX Drain FSM
            case (tx_state)
            TX_IDLE: begin
                if (tx_start && mode == MODE_TX_DRAIN) begin
                    tx_ptr <= head_ptr;
                    tx_end <= tail_ptr - {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                    tx_first_word <= 1'b1;
                    tx_state <= TX_READ;
                end
            end

            TX_READ: begin
                tx_state <= TX_SEND;
            end

            TX_SEND: begin
                if (out_rdy) begin
                    if (tx_ptr == tx_end) begin
                        head_ptr <= tx_ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                        pkt_ready_r <= 1'b0;
                        tx_done <= 1'b1;
                        tx_state <= TX_DONE;
                    end else begin
                        tx_ptr <= tx_ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                        tx_first_word <= 1'b0;
                        tx_state <= TX_READ;
                    end
                end
            end

            TX_DONE: begin
                tx_state <= TX_IDLE;
            end
            endcase
        end
    end

endmodule

`endif // CONV_FIFO_V