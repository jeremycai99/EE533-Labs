/* file: soc_driver.v
 Description: This file contains the SoC driver module for controlling and interfacing with the SoC.
 Author: Jeremy Cai
 Date: Feb. 14, 2026
 Version: 1.2
 */

`ifndef SOC_DRIVER_V
`define SOC_DRIVER_V

// IMPORTANT: Include definitions
`include "define.v"

module soc_driver (
    input wire clk,
    input wire rst_n,
    
    // Control signal
    input wire start, // New Start Port

    // User interface signals
    input wire user_valid,
    output wire user_ready,
    input wire user_cmd,
    input wire [`MMIO_ADDR_WIDTH-1:0] user_addr,
    input wire [`MMIO_DATA_WIDTH-1:0] user_wdata,
    output reg [`MMIO_DATA_WIDTH-1:0] user_rdata,
    output reg [`MMIO_ADDR_WIDTH-1:0] status,
    
    // SoC interface signals
    output reg soc_req_val,
    input wire soc_req_rdy,
    output reg soc_req_cmd,
    output reg [`MMIO_ADDR_WIDTH-1:0] soc_req_addr,
    output reg [`MMIO_DATA_WIDTH-1:0] soc_req_data,

    input wire soc_resp_val,
    output reg soc_resp_rdy,
    input wire soc_resp_cmd,
    input wire [`MMIO_ADDR_WIDTH-1:0] soc_resp_addr,
    input wire [`MMIO_DATA_WIDTH-1:0] soc_resp_data
);

    // FIFO Configuration
    localparam FIFO_DEPTH = 16;
    // Width = 1 (cmd) + Addr + Data
    localparam FIFO_WIDTH = 1 + `MMIO_ADDR_WIDTH + `MMIO_DATA_WIDTH; 

    wire fifo_full;
    wire fifo_empty;
    wire fifo_wr_en;
    reg  fifo_rd_en; 
    wire [FIFO_WIDTH-1:0] fifo_din;
    wire [FIFO_WIDTH-1:0] fifo_dout;

    assign fifo_din = {user_cmd, user_addr, user_wdata}; 
    assign fifo_wr_en = user_valid && !fifo_full; 
    assign user_ready = !fifo_full; 

    wire current_cmd;
    wire [`MMIO_ADDR_WIDTH-1:0] current_addr;
    wire [`MMIO_DATA_WIDTH-1:0] current_data;

    assign {current_cmd, current_addr, current_data} = fifo_dout; 

    localparam STATE_IDLE = 2'b00;
    localparam STATE_SEND_REQ = 2'b01;
    localparam STATE_WAIT_RESP = 2'b10;
    localparam STATE_CHECK = 2'b11;

    reg [1:0] current_state, next_state;

    // Internal registers to hold current transaction details
    reg [`MMIO_ADDR_WIDTH-1:0] active_addr;
    reg active_cmd; // <--- ADDED THIS LINE TO FIX THE ERROR
    
    // Registers to capture response for integrity check
    reg captured_resp_cmd;
    reg [`MMIO_ADDR_WIDTH-1:0] captured_resp_addr;

    // State Transition
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) current_state <= STATE_IDLE;
        else        current_state <= next_state;
    end

    // Next State Logic
    always @(*) begin
        next_state = current_state;
        case (current_state)
            STATE_IDLE: begin
                // Only transition if FIFO has data AND start is high
                if (!fifo_empty && start) 
                    next_state = STATE_SEND_REQ;
            end
            
            STATE_SEND_REQ: begin
                if (soc_req_rdy) 
                    next_state = STATE_WAIT_RESP;
            end

            STATE_WAIT_RESP: begin
                if (soc_resp_val) 
                    next_state = STATE_CHECK;
            end

            STATE_CHECK: begin
                next_state = STATE_IDLE;
            end
            default: next_state = STATE_IDLE;
        endcase
    end

    // Output and Control Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_rd_en <= 1'b0;
            soc_req_val <= 1'b0;
            soc_req_cmd <= 1'b0;
            soc_req_addr <= 0;
            soc_req_data <= 0;
            soc_resp_rdy <= 1'b0;
            status <= 32'b0;
            user_rdata <= 0;
            active_cmd <= 0;
            active_addr <= 0;
            captured_resp_cmd <= 0;
            captured_resp_addr <= 0;
        end else begin
            // Default signals
            fifo_rd_en <= 1'b0;
            soc_resp_rdy <= 1'b0; 

            case (current_state)
                STATE_IDLE: begin
                    // Only pop if start is high
                    if (!fifo_empty && start) begin
                        fifo_rd_en <= 1'b1; // Pop from FIFO
                        
                        // Latch data from FIFO
                        soc_req_cmd  <= current_cmd;
                        soc_req_addr <= current_addr;
                        soc_req_data <= current_data;
                        
                        // Save for check
                        active_cmd   <= current_cmd;
                        active_addr  <= current_addr;
                    end
                end

                STATE_SEND_REQ: begin
                    soc_req_val <= 1'b1;
                    if (soc_req_rdy) begin
                        soc_req_val <= 1'b0; 
                    end
                end

                STATE_WAIT_RESP: begin
                    soc_resp_rdy <= 1'b1; 
                    if (soc_resp_val) begin
                        captured_resp_cmd <= soc_resp_cmd;
                        captured_resp_addr <= soc_resp_addr;

                        if (soc_resp_cmd == 0) begin
                            user_rdata <= soc_resp_data;
                        end
                    end
                end

                STATE_CHECK: begin
                    soc_resp_rdy <= 1'b0;
                    if ((captured_resp_addr == active_addr) && (captured_resp_cmd == active_cmd)) begin
                        status <= 32'hAAAAAAAA; 
                    end else begin
                        status <= 32'hFFFFFFFF; 
                    end
                end
            endcase
        end
    end

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

// sync_fifo module definition
module sync_fifo #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire [DATA_WIDTH-1:0] din,
    input wire rd_en,
    output reg [DATA_WIDTH-1:0] dout,
    output wire full,
    output wire empty
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH):0] count;
    reg [$clog2(DEPTH)-1:0] wr_ptr;
    reg [$clog2(DEPTH)-1:0] rd_ptr;

    assign full = (count == DEPTH);
    assign empty = (count == 0);

    // Write Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (wr_en && !full) begin
            mem[wr_ptr] <= din;
            wr_ptr <= (wr_ptr == DEPTH-1) ? 0 : wr_ptr + 1;
        end
    end

    // Read Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= 0;
            dout <= 0;
        end else if (rd_en && !empty) begin
            dout <= mem[rd_ptr]; 
            rd_ptr <= (rd_ptr == DEPTH-1) ? 0 : rd_ptr + 1;
        end else if (!empty) begin
             // FWFT-like behavior: show next data
             dout <= mem[rd_ptr];
        end
    end

    // Count Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end

endmodule

`endif //SOC_DRIVER_V