/* file: pkt_proc.v
Description: This file contains the implementation of the packet processor, which interfaces with the CP0 register file and the convertible FIFO
Author: Jeremy Cai
Date: Mar. 4, 2026
Version: 2.0
Revision history:
    - Mar. 4, 2026: v1.0 — Initial implementation of the packet processor.
    - Mar. 7, 2026: v2.1 — Auto-detect network packet, no external port.
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

    output reg [1:0] fifo_mode,
    output reg [FIFO_ADDR_WIDTH-1:0] fifo_head_wr_data,
    output reg fifo_head_wr,
    output reg [FIFO_ADDR_WIDTH-1:0] fifo_tail_wr_data,
    output reg fifo_tail_wr,
    output reg fifo_tx_start,

    input wire [FIFO_ADDR_WIDTH-1:0] fifo_head_ptr,
    input wire [FIFO_ADDR_WIDTH-1:0] fifo_pkt_end,
    input wire fifo_pkt_ready,
    output reg fifo_pkt_ack,
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
    // Constants
    // ================================================================
    localparam [3:0] CMD_NOP       = 4'h0;
    localparam [3:0] CMD_LOAD_IMEM = 4'h1;
    localparam [3:0] CMD_LOAD_DMEM = 4'h2;
    localparam [3:0] CMD_CPU_START = 4'h3;
    localparam [3:0] CMD_READBACK  = 4'h4;
    localparam [3:0] CMD_SEND_PKT  = 4'h5;

    localparam [15:0] SOC_ETHERTYPE = 16'h88B5;

    // ================================================================
    // FSM states
    // ================================================================
    localparam [4:0] P_IDLE        = 5'd0;
    localparam [4:0] P_FETCH_CMD   = 5'd1;
    localparam [4:0] P_FETCH_WAIT  = 5'd2;
    localparam [4:0] P_DECODE_CMD  = 5'd3;
    localparam [4:0] P_LOAD_RD     = 5'd4;
    localparam [4:0] P_LOAD_WAIT   = 5'd5;
    localparam [4:0] P_LOAD_UNPACK = 5'd6;
    localparam [4:0] P_LOAD_WR_HI  = 5'd7;
    localparam [4:0] P_CPU_START   = 5'd8;
    localparam [4:0] P_CPU_RUN     = 5'd9;
    localparam [4:0] P_RB_RD_LO    = 5'd10;
    localparam [4:0] P_RB_RD_HI    = 5'd11;
    localparam [4:0] P_RB_WAIT     = 5'd12;
    localparam [4:0] P_RB_WR       = 5'd13;
    localparam [4:0] P_SEND_SETUP  = 5'd14;
    localparam [4:0] P_SEND_START  = 5'd15;
    localparam [4:0] P_SEND_WAIT   = 5'd16;
    localparam [4:0] P_SEND_DONE   = 5'd17;
    // v2.1: network auto-detect states
    localparam [4:0] P_HDR_RD      = 5'd18;
    localparam [4:0] P_HDR_WAIT    = 5'd19;
    localparam [4:0] P_HDR_SAVE    = 5'd20;
    localparam [4:0] P_PASSTHRU    = 5'd21;
    localparam [4:0] P_SEND_HDR    = 5'd22;

    reg [4:0] state;

    // ================================================================
    // Working registers (v1.0)
    // ================================================================
    reg [FIFO_ADDR_WIDTH-1:0] rd_ptr;
    reg [FIFO_ADDR_WIDTH-1:0] pkt_end_r;
    reg [15:0] count;
    reg [DMEM_ADDR_WIDTH-1:0] base_addr;
    reg target;
    reg [63:0] data_r;
    reg [31:0] rb_lo_r;
    reg [FIFO_ADDR_WIDTH-1:0] rb_wr_ptr;
    reg [FIFO_ADDR_WIDTH-1:0] rb_base;

    // ================================================================
    // auto-detect registers
    // ================================================================
    reg first_decode;
    reg pkt_is_network;
    reg [63:0] hdr_word0, hdr_word1, hdr_word2;
    reg [1:0] hdr_cnt;
    reg [FIFO_ADDR_WIDTH-1:0] pkt_head_r;

    // First word opcode check: valid command = {1,2,3,4,5}
    wire [3:0] first_opcode = fifo_rdata[63:60];
    wire first_is_command = (first_opcode >= 4'd1) && (first_opcode <= 4'd5);

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
            first_decode <= 1'b0;
            pkt_is_network <= 1'b0;
            hdr_word0 <= 64'd0; hdr_word1 <= 64'd0; hdr_word2 <= 64'd0;
            hdr_cnt <= 2'd0;
            pkt_head_r <= {FIFO_ADDR_WIDTH{1'b0}};
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

            // ========================================================
            // IDLE: wait for packet
            // ========================================================
            P_IDLE: begin
                cpu_rst_n <= 1'b0;
                if (fifo_pkt_ready) begin
                    fifo_mode <= 2'd1;
                    fifo_pkt_ack <= 1'b1;
                    rd_ptr <= fifo_head_ptr;
                    pkt_end_r <= fifo_pkt_end;
                    pkt_head_r <= fifo_head_ptr;
                    rb_base <= {FIFO_ADDR_WIDTH{1'b0}};
                    rb_wr_ptr <= {FIFO_ADDR_WIDTH{1'b0}};
                    first_decode <= 1'b1;
                    pkt_is_network <= 1'b0;
                    state <= P_FETCH_CMD;
                end
            end

            // ========================================================
            // FETCH_CMD (unchanged from v1.0)
            // ========================================================
            P_FETCH_CMD: begin
                if (rd_ptr > pkt_end_r) begin
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

            // ========================================================
            // DECODE_CMD
            // ========================================================
            P_DECODE_CMD: begin
                if (first_decode && !first_is_command) begin
                    // ── Network packet detected ──
                    first_decode <= 1'b0;
                    pkt_is_network <= 1'b1;
                    hdr_word0 <= fifo_rdata;
                    hdr_cnt <= 2'd1;
                    rd_ptr <= rd_ptr + {{(FIFO_ADDR_WIDTH-1){1'b0}}, 1'b1};
                    state <= P_HDR_RD;
                end else begin
                    // ── Injection or post-header command parsing ──
                    first_decode <= 1'b0;
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
            end

            // ========================================================
            // Read remaining header words (1 and 2)
            // ========================================================
            P_HDR_RD: begin
                fifo_addr <= rd_ptr;
                state <= P_HDR_WAIT;
            end

            P_HDR_WAIT: begin
                state <= P_HDR_SAVE;
            end

            P_HDR_SAVE: begin
                case (hdr_cnt)
                    2'd1: hdr_word1 <= fifo_rdata;
                    2'd2: hdr_word2 <= fifo_rdata;
                    default: ;
                endcase
                rd_ptr <= rd_ptr + {{(FIFO_ADDR_WIDTH-1){1'b0}}, 1'b1};

                if (hdr_cnt == 2'd2) begin
                    // Word 2: {SRC_MAC[31:0], EtherType[15:0], Pad[15:0]}
                    if (fifo_rdata[31:16] == SOC_ETHERTYPE) begin
                        // SoC packet: reserve slots 0-2 for TX header
                        rb_wr_ptr <= {{(FIFO_ADDR_WIDTH-2){1'b0}}, 2'd3};
                        state <= P_FETCH_CMD;
                    end else begin
                        // Not SoC: passthrough unchanged
                        state <= P_PASSTHRU;
                    end
                end else begin
                    hdr_cnt <= hdr_cnt + 2'd1;
                    state <= P_HDR_RD;
                end
            end

            // ========================================================
            // Passthrough — drain packet unchanged
            // ========================================================
            P_PASSTHRU: begin
                fifo_head_wr_data <= pkt_head_r;
                fifo_head_wr <= 1'b1;
                fifo_tail_wr_data <= pkt_end_r + {{(FIFO_ADDR_WIDTH-1){1'b0}}, 1'b1};
                fifo_tail_wr <= 1'b1;
                fifo_mode <= 2'd2;
                state <= P_SEND_START;
            end

            // ========================================================
            // LOAD: FIFO -> IMEM/DMEM (unchanged from v1.0)
            // ========================================================
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

            // ========================================================
            // CPU_START (unchanged from v1.0)
            // ========================================================
            P_CPU_START: begin
                cpu_rst_n <= 1'b1;
                cpu_start <= 1'b1;
                state <= P_CPU_RUN;
            end

            P_CPU_RUN: begin
                if (cpu_done) begin
                    cpu_rst_n <= 1'b0;
                    fifo_mode <= 2'd1;
                    state <= P_FETCH_CMD;
                end
            end

            // ========================================================
            // READBACK: DMEM -> FIFO (unchanged from v1.0)
            // ========================================================
            P_RB_RD_LO: begin
                dmem_addr <= base_addr;
                state <= P_RB_RD_HI;
            end

            P_RB_RD_HI: begin
                dmem_addr <= base_addr + {{(DMEM_ADDR_WIDTH-1){1'b0}}, 1'b1};
                state <= P_RB_WAIT;
            end

            P_RB_WAIT: begin
                rb_lo_r <= dmem_dout;
                state <= P_RB_WR;
            end

            P_RB_WR: begin
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

            // ========================================================
            // SEND_PKT
            // v2.1: network SoC packets prepend saved headers
            // ========================================================
            P_SEND_SETUP: begin
                if (pkt_is_network) begin
                    hdr_cnt <= 2'd0;
                    state <= P_SEND_HDR;
                end else begin
                    fifo_head_wr_data <= rb_base;
                    fifo_head_wr <= 1'b1;
                    fifo_tail_wr_data <= rb_wr_ptr;
                    fifo_tail_wr <= 1'b1;
                    fifo_mode <= 2'd2;
                    state <= P_SEND_START;
                end
            end

            // ========================================================
            // Write saved headers to FIFO slots 0,1,2
            // ========================================================
            P_SEND_HDR: begin
                fifo_addr <= {{(FIFO_ADDR_WIDTH-2){1'b0}}, hdr_cnt};
                case (hdr_cnt)
                    2'd0: fifo_wdata <= hdr_word0;
                    2'd1: fifo_wdata <= hdr_word1;
                    2'd2: fifo_wdata <= hdr_word2;
                    default: fifo_wdata <= 64'd0;
                endcase
                fifo_we <= 1'b1;

                if (hdr_cnt == 2'd2) begin
                    fifo_head_wr_data <= {FIFO_ADDR_WIDTH{1'b0}};
                    fifo_head_wr <= 1'b1;
                    fifo_tail_wr_data <= rb_wr_ptr;
                    fifo_tail_wr <= 1'b1;
                    fifo_mode <= 2'd2;
                    state <= P_SEND_START;
                end else begin
                    hdr_cnt <= hdr_cnt + 2'd1;
                end
            end

            P_SEND_START: begin
                fifo_tx_start <= 1'b1;
                state <= P_SEND_WAIT;
            end

            P_SEND_WAIT: begin
                if (fifo_tx_done) begin
                    state <= P_SEND_DONE;
                end
            end

            P_SEND_DONE: begin
                fifo_mode <= 2'd0;
                state <= P_IDLE;
            end

            endcase
        end
    end

endmodule

`endif // PKT_PROC_V