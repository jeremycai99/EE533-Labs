/* file: pkt_proc.v
Description: This file contains the implementation of the packet processor, which interfaces with the CP0 register file and the convertible FIFO
Author: Jeremy Cai
Date: Mar. 4, 2026
Version: 1.0
Revision history:
    - Mar. 4, 2026: v1.0 — Initial implementation of the packet processor.
*/

`ifndef PKT_PROC_V
`define PKT_PROC_V

//  Packet processor command and format definitions
//  Commands: LOAD_IMEM, LOAD_DMEM, CPU_START, READBACK, SEND_PKT.

//  Command word encoding (64 bits):
//  [63:60]  cmd[3:0]     — command opcode
//  [59:48]  addr[11:0]   — base address for IMEM/DMEM operations
//  [47:32]  count[15:0]  — number of 64bit data words following
//  [31:0]   param[31:0]  — command-specific (entry_pc, etc. semi-reserved)

//  Opcodes:
//  4'h0  NOP          — skip (no data words follow)
//  4'h1  LOAD_IMEM    — unpack count 64bit words -> 2*count 32bit instrs at addr
//  4'h2  LOAD_DMEM    — unpack count 64bit words -> 2*count 32bit data at addr
//  4'h3  CPU_START    — param[31:0] = entry_pc, wait for cpu_done

//  Data word (follows LOAD_IMEM / LOAD_DMEM commands):
//  fifo_rdata[63:32]  — written to target[addr+1]
//  fifo_rdata[31:0]   — written to target[addr]

//  Packet flow: FIFO stores commands + data sequentially.
module pkt_proc #(
    parameter FIFO_ADDR_WIDTH = 12,
    parameter IMEM_ADDR_WIDTH = 10,
    parameter DMEM_ADDR_WIDTH = 12
)(
    input wire clk,
    input wire rst_n,

    // Convertible FIFO interface (Port B master)
    output reg [FIFO_ADDR_WIDTH-1:0] fifo_addr,
    output reg [63:0] fifo_wdata,
    output reg fifo_we,
    input wire [63:0] fifo_rdata,

    output reg [1:0] fifo_mode,        // 0=RX_FIFO, 1=SRAM, 2=TX_DRAIN
    output reg [FIFO_ADDR_WIDTH-1:0] fifo_head_wr_data,
    output reg fifo_head_wr,
    output reg [FIFO_ADDR_WIDTH-1:0] fifo_tail_wr_data,
    output reg fifo_tail_wr,
    output reg fifo_tx_start,

    input wire [FIFO_ADDR_WIDTH-1:0] fifo_head_ptr,
    input wire [FIFO_ADDR_WIDTH-1:0] fifo_pkt_end,
    input wire fifo_pkt_ready,
    output reg fifo_pkt_ack,           // clears pkt_ready when processing starts
    input wire fifo_tx_done,

    // CPU IMEM Port B
    output reg [IMEM_ADDR_WIDTH-1:0] imem_addr,
    output reg [31:0] imem_din,
    output reg imem_we,

    // CPU DMEM Port B
    output reg [DMEM_ADDR_WIDTH-1:0] dmem_addr,
    output reg [31:0] dmem_din,
    output reg dmem_we,
    input wire [31:0] dmem_dout,

    // CPU control
    output reg cpu_rst_n,
    output reg cpu_start,
    output reg [31:0] entry_pc,
    input wire cpu_done,

    // Status
    output wire active,
    output wire owns_port_b
);

    // ================================================================
    // Command opcodes
    // ================================================================
    localparam [3:0] CMD_NOP       = 4'h0;
    localparam [3:0] CMD_LOAD_IMEM = 4'h1;
    localparam [3:0] CMD_LOAD_DMEM = 4'h2;
    localparam [3:0] CMD_CPU_START = 4'h3;
    localparam [3:0] CMD_READBACK  = 4'h4;
    localparam [3:0] CMD_SEND_PKT  = 4'h5;

    // ================================================================
    // FSM states (5-bit, 17 states)
    // ================================================================
    localparam [4:0] P_IDLE       = 5'd0;
    localparam [4:0] P_FETCH_CMD  = 5'd1;
    localparam [4:0] P_FETCH_WAIT = 5'd2;   // FIFO BRAM latency
    localparam [4:0] P_DECODE_CMD = 5'd3;
    localparam [4:0] P_LOAD_RD    = 5'd4;
    localparam [4:0] P_LOAD_WAIT  = 5'd5;   // FIFO BRAM latency
    localparam [4:0] P_LOAD_UNPACK= 5'd6;   // write lo half
    localparam [4:0] P_LOAD_WR_HI = 5'd7;   // write hi half
    localparam [4:0] P_CPU_START  = 5'd8;
    localparam [4:0] P_CPU_RUN    = 5'd9;
    localparam [4:0] P_RB_RD_LO   = 5'd10;  // present DMEM addr (lo)
    localparam [4:0] P_RB_RD_HI   = 5'd11;  // present addr+1, wait lo
    localparam [4:0] P_RB_WAIT    = 5'd12;  // capture lo, wait hi
    localparam [4:0] P_RB_WR      = 5'd13;  // pack + write to FIFO
    localparam [4:0] P_SEND_SETUP = 5'd14;  // set ptrs + mode
    localparam [4:0] P_SEND_START = 5'd15;  // pulse tx_start (ptrs settled)
    localparam [4:0] P_SEND_WAIT  = 5'd16;  // wait tx_done
    localparam [4:0] P_SEND_DONE  = 5'd17;  // return to RX_FIFO

    reg [4:0] state;

    // ================================================================
    // Working registers
    // ================================================================
    reg [FIFO_ADDR_WIDTH-1:0] rd_ptr;
    reg [FIFO_ADDR_WIDTH-1:0] pkt_end_r;
    reg [15:0] count;
    reg [DMEM_ADDR_WIDTH-1:0] base_addr;
    reg target;                              // 0=IMEM, 1=DMEM
    reg [63:0] data_r;
    reg [31:0] rb_lo_r;

    // Readback write pointer into FIFO
    reg [FIFO_ADDR_WIDTH-1:0] rb_wr_ptr;
    reg [FIFO_ADDR_WIDTH-1:0] rb_base;

    // ================================================================
    // Status
    // ================================================================
    assign active = (state != P_IDLE);
    assign owns_port_b = active && (state != P_CPU_RUN)
                                && (state != P_SEND_WAIT);

    // ================================================================
    // FSM
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= P_IDLE;
            rd_ptr <= {FIFO_ADDR_WIDTH{1'b0}};
            pkt_end_r <= {FIFO_ADDR_WIDTH{1'b0}};
            count <= 16'd0;
            base_addr <= {DMEM_ADDR_WIDTH{1'b0}};
            target <= 1'b0;
            data_r <= 64'd0;
            rb_lo_r <= 32'd0;
            rb_wr_ptr <= {FIFO_ADDR_WIDTH{1'b0}};
            rb_base <= {FIFO_ADDR_WIDTH{1'b0}};
            fifo_addr <= {FIFO_ADDR_WIDTH{1'b0}};
            fifo_wdata <= 64'd0;
            fifo_we <= 1'b0;
            fifo_mode <= 2'd0;
            fifo_head_wr_data <= {FIFO_ADDR_WIDTH{1'b0}};
            fifo_head_wr <= 1'b0;
            fifo_tail_wr_data <= {FIFO_ADDR_WIDTH{1'b0}};
            fifo_tail_wr <= 1'b0;
            fifo_tx_start <= 1'b0;
            fifo_pkt_ack <= 1'b0;
            imem_addr <= {IMEM_ADDR_WIDTH{1'b0}};
            imem_din <= 32'd0;
            imem_we <= 1'b0;
            dmem_addr <= {DMEM_ADDR_WIDTH{1'b0}};
            dmem_din <= 32'd0;
            dmem_we <= 1'b0;
            cpu_rst_n <= 1'b0;
            cpu_start <= 1'b0;
            entry_pc <= 32'd0;
        end else begin
            // Default: deassert pulses
            fifo_we <= 1'b0;
            fifo_head_wr <= 1'b0;
            fifo_tail_wr <= 1'b0;
            fifo_tx_start <= 1'b0;
            fifo_pkt_ack <= 1'b0;
            imem_we <= 1'b0;
            dmem_we <= 1'b0;
            cpu_start <= 1'b0;

            case (state)

            // IDLE: wait for packet
            P_IDLE: begin
                cpu_rst_n <= 1'b0;
                if (fifo_pkt_ready) begin
                    fifo_mode <= 2'd1;        // SRAM mode
                    fifo_pkt_ack <= 1'b1;     // clear pkt_ready
                    rd_ptr <= fifo_head_ptr;
                    pkt_end_r <= fifo_pkt_end;
                    rb_base <= {FIFO_ADDR_WIDTH{1'b0}};
                    rb_wr_ptr <= {FIFO_ADDR_WIDTH{1'b0}};
                    state <= P_FETCH_CMD;
                end
            end

            // FETCH_CMD
            P_FETCH_CMD: begin
                if (rd_ptr > pkt_end_r) begin
                    // Consumed all commands — advance head past packet
                    fifo_head_wr_data <= rd_ptr;
                    fifo_head_wr <= 1'b1;
                    fifo_mode <= 2'd0;
                    state <= P_IDLE;
                end else begin
                    fifo_addr <= rd_ptr;
                    state <= P_FETCH_WAIT;
                end
            end

            P_FETCH_WAIT: begin
                state <= P_DECODE_CMD;
            end

            // DECODE_CMD
            P_DECODE_CMD: begin
                rd_ptr <= rd_ptr + {{(FIFO_ADDR_WIDTH-1){1'b0}}, 1'b1};

                case (fifo_rdata[63:60])
                CMD_NOP: begin
                    state <= P_FETCH_CMD;
                end

                CMD_LOAD_IMEM: begin
                    base_addr <= {{(DMEM_ADDR_WIDTH-12){1'b0}}, fifo_rdata[59:48]};
                    count <= fifo_rdata[47:32];
                    target <= 1'b0;
                    state <= P_LOAD_RD;
                end

                CMD_LOAD_DMEM: begin
                    base_addr <= fifo_rdata[59:48];
                    count <= fifo_rdata[47:32];
                    target <= 1'b1;
                    state <= P_LOAD_RD;
                end

                CMD_CPU_START: begin
                    entry_pc <= fifo_rdata[31:0];
                    state <= P_CPU_START;
                end

                CMD_READBACK: begin
                    base_addr <= fifo_rdata[59:48];
                    count <= fifo_rdata[47:32];
                    rb_base <= rb_wr_ptr;
                    state <= P_RB_RD_LO;
                end

                CMD_SEND_PKT: begin
                    state <= P_SEND_SETUP;
                end

                default: begin
                    state <= P_FETCH_CMD;
                end
                endcase
            end

            // LOAD: FIFO -> IMEM/DMEM (64b -> 2×32b)
            P_LOAD_RD: begin
                fifo_addr <= rd_ptr;
                state <= P_LOAD_WAIT;
            end

            P_LOAD_WAIT: begin
                state <= P_LOAD_UNPACK;
            end

            P_LOAD_UNPACK: begin
                data_r <= fifo_rdata;
                rd_ptr <= rd_ptr + {{(FIFO_ADDR_WIDTH-1){1'b0}}, 1'b1};
                if (target == 1'b0) begin
                    imem_addr <= base_addr[IMEM_ADDR_WIDTH-1:0];
                    imem_din <= fifo_rdata[31:0];
                    imem_we <= 1'b1;
                end else begin
                    dmem_addr <= base_addr;
                    dmem_din <= fifo_rdata[31:0];
                    dmem_we <= 1'b1;
                end
                state <= P_LOAD_WR_HI;
            end

            P_LOAD_WR_HI: begin
                if (target == 1'b0) begin
                    imem_addr <= base_addr[IMEM_ADDR_WIDTH-1:0] + {{(IMEM_ADDR_WIDTH-1){1'b0}}, 1'b1};
                    imem_din <= data_r[63:32];
                    imem_we <= 1'b1;
                end else begin
                    dmem_addr <= base_addr + {{(DMEM_ADDR_WIDTH-1){1'b0}}, 1'b1};
                    dmem_din <= data_r[63:32];
                    dmem_we <= 1'b1;
                end
                base_addr <= base_addr + {{(DMEM_ADDR_WIDTH-2){1'b0}}, 2'd2};
                count <= count - 16'd1;
                if (count == 16'd1)
                    state <= P_FETCH_CMD;
                else
                    state <= P_LOAD_RD;
            end

            // CPU_START
            P_CPU_START: begin
                cpu_rst_n <= 1'b1;
                cpu_start <= 1'b1;
                state <= P_CPU_RUN;
            end

            P_CPU_RUN: begin
                if (cpu_done) begin
                    cpu_rst_n <= 1'b0;
                    fifo_mode <= 2'd1; // back to SRAM for readback
                    state <= P_FETCH_CMD;
                end
            end

            // READBACK: DMEM -> FIFO (2×32b -> 64b)
            P_RB_RD_LO: begin
                dmem_addr <= base_addr;
                state <= P_RB_RD_HI;
            end

            P_RB_RD_HI: begin
                // lo not valid yet — present hi addr
                dmem_addr <= base_addr + {{(DMEM_ADDR_WIDTH-1){1'b0}}, 1'b1};
                state <= P_RB_WAIT;
            end

            P_RB_WAIT: begin
                // lo now valid from BRAM
                rb_lo_r <= dmem_dout;
                state <= P_RB_WR;
            end

            P_RB_WR: begin
                // hi now valid — pack and write to FIFO
                fifo_addr <= rb_wr_ptr;
                fifo_wdata <= {dmem_dout, rb_lo_r};
                fifo_we <= 1'b1;
                rb_wr_ptr <= rb_wr_ptr + {{(FIFO_ADDR_WIDTH-1){1'b0}}, 1'b1};
                base_addr <= base_addr + {{(DMEM_ADDR_WIDTH-2){1'b0}}, 2'd2};
                count <= count - 16'd1;
                if (count == 16'd1)
                    state <= P_FETCH_CMD;
                else
                    state <= P_RB_RD_LO;
            end

            // SEND_PKT: set ptrs, then TX drain
            P_SEND_SETUP: begin
                // Set head = readback base, tail = readback end
                fifo_head_wr_data <= rb_base;
                fifo_head_wr <= 1'b1;
                fifo_tail_wr_data <= rb_wr_ptr;
                fifo_tail_wr <= 1'b1;
                fifo_mode <= 2'd2;            // TX_DRAIN
                state <= P_SEND_START;
            end

            P_SEND_START: begin
                // Pointers settled (NBA from prev cycle took effect).
                // Now safe to start TX drain.
                fifo_tx_start <= 1'b1;
                state <= P_SEND_WAIT;
            end

            P_SEND_WAIT: begin
                if (fifo_tx_done) begin
                    state <= P_SEND_DONE;
                end
            end

            P_SEND_DONE: begin
                fifo_mode <= 2'd0;            // back to RX_FIFO
                state <= P_IDLE;
            end

            endcase
        end
    end

endmodule

`endif // PKT_PROC_V