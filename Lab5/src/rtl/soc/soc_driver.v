/* file: soc_driver.v
 * Description: SoC driver module for controlling and interfacing with the SoC.
 *   Now includes transaction quality monitoring (timeout detection, latency
 *   tracking, transaction counting, protocol error detection).
 *
 * Status register encoding:
 *   0xAAAA_AAAA — integrity pass
 *   0xFFFF_FFFF — integrity fail (addr/cmd mismatch)
 *   0xDEAD_DEAD — timeout (request or response phase)
 *
 * Author: Jeremy Cai
 * Date: Feb. 14, 2026
 * Version: 1.6
 */

`ifndef SOC_DRIVER_V
`define SOC_DRIVER_V

`include "define.v"

module soc_driver #(
    parameter TIMEOUT_THRESHOLD = 16'd1000
)(
    input wire clk,
    input wire rst_n,
    // input wire start, // Removed
    input wire user_valid,
    output wire user_ready,
    input wire user_cmd,
    input wire [`MMIO_ADDR_WIDTH-1:0] user_addr,
    input wire [`MMIO_DATA_WIDTH-1:0] user_wdata,
    output reg [`MMIO_DATA_WIDTH-1:0] user_rdata,
    output reg [`MMIO_ADDR_WIDTH-1:0] status,
    output reg soc_req_val,
    input wire soc_req_rdy,
    output reg soc_req_cmd,
    output reg [`MMIO_ADDR_WIDTH-1:0] soc_req_addr,
    output reg [`MMIO_DATA_WIDTH-1:0] soc_req_data,
    input wire soc_resp_val,
    output reg soc_resp_rdy,
    input wire soc_resp_cmd,
    input wire [`MMIO_ADDR_WIDTH-1:0] soc_resp_addr,
    input wire [`MMIO_DATA_WIDTH-1:0] soc_resp_data,
    output wire [7:0]                   conn_status,
    output wire [`MMIO_DATA_WIDTH-1:0]  txn_quality,
    output wire [`MMIO_DATA_WIDTH-1:0]  txn_counters,
    input  wire                         clear_stats
);

    localparam DW = `MMIO_DATA_WIDTH;
    localparam FIFO_DEPTH = 16;
    localparam FIFO_WIDTH = 1 + `MMIO_ADDR_WIDTH + `MMIO_DATA_WIDTH; 

    wire fifo_full;
    wire fifo_empty;
    wire fifo_wr_en;
    reg  fifo_rd_en; 
    wire [FIFO_WIDTH-1:0] fifo_din;
    wire [FIFO_WIDTH-1:0] fifo_dout;

    assign fifo_din   = {user_cmd, user_addr, user_wdata}; 
    assign fifo_wr_en = user_valid && !fifo_full; 
    assign user_ready = !fifo_full; 

    wire current_cmd;
    wire [`MMIO_ADDR_WIDTH-1:0] current_addr;
    wire [`MMIO_DATA_WIDTH-1:0] current_data;

    assign {current_cmd, current_addr, current_data} = fifo_dout; 

    // FSM States
    localparam STATE_IDLE      = 2'b00;
    localparam STATE_SEND_REQ  = 2'b01;
    localparam STATE_WAIT_RESP = 2'b10;
    localparam STATE_CHECK     = 2'b11;

    reg [1:0] current_state, next_state;

    // Internal registers
    reg [`MMIO_ADDR_WIDTH-1:0] active_addr;
    reg active_cmd;
    reg captured_resp_cmd;
    reg [`MMIO_ADDR_WIDTH-1:0] captured_resp_addr;

    // Timer & Quality Logic
    reg [15:0] txn_timer;
    reg        txn_timed_out;         
    reg        txn_req_phase_timeout; 

    wire req_timeout  = (current_state == STATE_SEND_REQ) && (txn_timer >= TIMEOUT_THRESHOLD);
    wire resp_timeout = (current_state == STATE_WAIT_RESP) && (txn_timer >= TIMEOUT_THRESHOLD);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) txn_timer <= 16'd0;
        else if (current_state == STATE_IDLE) txn_timer <= 16'd0;
        else txn_timer <= txn_timer + 16'd1;
    end

    // Quality Registers
    reg        link_active_r;
    reg        req_timeout_flag_r;
    reg        resp_timeout_flag_r;
    reg        protocol_error_r;
    reg [15:0] max_latency_r;
    reg [15:0] timeout_count_r;
    reg [15:0] total_txn_count_r;
    reg [15:0] read_txn_count_r;
    reg [15:0] write_txn_count_r;

    // State Transition
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) current_state <= STATE_IDLE;
        else        current_state <= next_state;
    end

    always @(*) begin
        next_state = current_state;
        case (current_state)
            STATE_IDLE: begin
                if (!fifo_empty) 
                    next_state = STATE_SEND_REQ;
            end
            STATE_SEND_REQ: begin
                if (soc_req_rdy)       
                    next_state = STATE_WAIT_RESP;
                else if (req_timeout)
                    next_state = STATE_CHECK;
            end
            STATE_WAIT_RESP: begin
                if (soc_resp_val)      
                    next_state = STATE_CHECK;
                else if (resp_timeout)
                    next_state = STATE_CHECK;
            end
            STATE_CHECK: begin
                next_state = STATE_IDLE;
            end
            default: next_state = STATE_IDLE;
        endcase
    end

    // Output Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_rd_en            <= 1'b0;
            soc_req_val           <= 1'b0;
            soc_req_cmd           <= 1'b0;
            soc_req_addr          <= {`MMIO_ADDR_WIDTH{1'b0}};
            soc_req_data          <= {DW{1'b0}};
            soc_resp_rdy          <= 1'b0;
            status                <= {`MMIO_ADDR_WIDTH{1'b0}};
            user_rdata            <= {DW{1'b0}};
            active_cmd            <= 1'b0;
            active_addr           <= {`MMIO_ADDR_WIDTH{1'b0}};
            captured_resp_cmd     <= 1'b0;
            captured_resp_addr    <= {`MMIO_ADDR_WIDTH{1'b0}};
            txn_timed_out         <= 1'b0;
            txn_req_phase_timeout <= 1'b0;
        end else begin
            // Default de-assertions
            fifo_rd_en   <= 1'b0;
            soc_req_val  <= 1'b0;
            soc_resp_rdy <= 1'b0;

            case (current_state)
                STATE_IDLE: begin
                    txn_timed_out         <= 1'b0;
                    txn_req_phase_timeout <= 1'b0;
                    // Modified: Removed '&& start'
                    if (!fifo_empty) begin
                        fifo_rd_en   <= 1'b1;
                        soc_req_cmd  <= current_cmd;
                        soc_req_addr <= current_addr;
                        soc_req_data <= current_data;
                        active_cmd   <= current_cmd;
                        active_addr  <= current_addr;
                    end
                end

                STATE_SEND_REQ: begin
                    // If soc_req_rdy is high, state changes to WAIT_RESP next cycle,
                    // where soc_req_val will default to 0.
                    soc_req_val <= 1'b1;            
                    
                    if (req_timeout && !soc_req_rdy) begin
                        soc_req_val           <= 1'b0;
                        txn_timed_out         <= 1'b1;
                        txn_req_phase_timeout <= 1'b1;
                    end
                end

                STATE_WAIT_RESP: begin
                    soc_resp_rdy <= 1'b1;
                    if (soc_resp_val) begin
                        captured_resp_cmd  <= soc_resp_cmd;
                        captured_resp_addr <= soc_resp_addr;
                        if (soc_resp_cmd == 1'b0)
                            user_rdata <= soc_resp_data;
                    end else if (resp_timeout) begin
                        soc_resp_rdy  <= 1'b0;
                        txn_timed_out <= 1'b1;
                    end
                end

                STATE_CHECK: begin
                    if (txn_timed_out) begin
                        status <= 32'hDEAD_DEAD;
                    end else if ((captured_resp_addr == active_addr) &&
                                 (captured_resp_cmd  == active_cmd)) begin
                        status <= 32'hAAAA_AAAA;
                    end else begin
                        status <= 32'hFFFF_FFFF;
                    end
                end
            endcase
        end
    end

    // Quality Monitor Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            link_active_r       <= 1'b0;
            req_timeout_flag_r  <= 1'b0;
            resp_timeout_flag_r <= 1'b0;
            protocol_error_r    <= 1'b0;
            max_latency_r       <= 16'd0;
            timeout_count_r     <= 16'd0;
            total_txn_count_r   <= 16'd0;
            read_txn_count_r    <= 16'd0;
            write_txn_count_r   <= 16'd0;
        end else if (clear_stats) begin
            link_active_r       <= 1'b0;
            req_timeout_flag_r  <= 1'b0;
            resp_timeout_flag_r <= 1'b0;
            protocol_error_r    <= 1'b0;
            max_latency_r       <= 16'd0;
            timeout_count_r     <= 16'd0;
            total_txn_count_r   <= 16'd0;
            read_txn_count_r    <= 16'd0;
            write_txn_count_r   <= 16'd0;
        end else begin
            if (soc_resp_val && (current_state == STATE_IDLE || current_state == STATE_SEND_REQ))
                protocol_error_r <= 1'b1;

            if (current_state == STATE_CHECK) begin
                if (txn_timed_out) begin
                    timeout_count_r <= timeout_count_r + 16'd1;
                    if (txn_req_phase_timeout) req_timeout_flag_r  <= 1'b1;
                    else                       resp_timeout_flag_r <= 1'b1;
                end else begin
                    link_active_r     <= 1'b1;
                    total_txn_count_r <= total_txn_count_r + 16'd1;
                    if (!active_cmd) read_txn_count_r  <= read_txn_count_r  + 16'd1;
                    else             write_txn_count_r <= write_txn_count_r + 16'd1;
                    if (txn_timer > max_latency_r) max_latency_r <= txn_timer;
                end
            end
        end
    end

    assign conn_status = {
        1'b1, txn_timed_out, (current_state != STATE_IDLE), ~fifo_empty,
        protocol_error_r, resp_timeout_flag_r, req_timeout_flag_r, link_active_r
    };

    assign txn_quality = {
        {(DW-48){1'b0}}, timeout_count_r, max_latency_r, 12'd0,
        link_active_r, protocol_error_r, resp_timeout_flag_r, req_timeout_flag_r
    };

    assign txn_counters = {
        {(DW-48){1'b0}}, write_txn_count_r, read_txn_count_r, total_txn_count_r
    };

    // FIFO Instance
    sync_fifo #(
        .DATA_WIDTH(FIFO_WIDTH), 
        .DEPTH(FIFO_DEPTH)
    ) u_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(fifo_wr_en),
        .din(fifo_din),
        .rd_en(fifo_rd_en),
        .dout(fifo_dout),
        .full(fifo_full),
        .empty(fifo_empty)
    );

endmodule

// FIFO Module
module sync_fifo #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire [DATA_WIDTH-1:0] din,
    input wire rd_en,
    output wire [DATA_WIDTH-1:0] dout, 
    output wire full,
    output wire empty
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH):0] count;
    reg [$clog2(DEPTH)-1:0] wr_ptr;
    reg [$clog2(DEPTH)-1:0] rd_ptr;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);
    
    assign dout = mem[rd_ptr];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (wr_en && !full) begin
            mem[wr_ptr] <= din;
            wr_ptr <= (wr_ptr == DEPTH-1) ? 0 : wr_ptr + 1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= 0;
        end else if (rd_en && !empty) begin
            rd_ptr <= (rd_ptr == DEPTH-1) ? 0 : rd_ptr + 1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10:   count <= count + 1;
                2'b01:   count <= count - 1;
                default: count <= count;
            endcase
        end
    end

endmodule

`endif //SOC_DRIVER_V