/* file: bdtu.v
 Description: Block data transfer unit module for multi-cycle
              instructions in Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 23, 2026
 Version: 1.5
 Changes from 1.4:
   - Added 2-stage store-path pipeline to break the critical path:
       remaining (FF) → priority encoder → rf_rd_addr → RF read →
       thread mux → mem_wdata byte mux → d_mem_data_o → BRAM
     into three stages each under 8 ns:
       Stage 1: remaining → rf_rd_addr_p1        (~4.2 ns)
       Stage 2: rf_rd_addr_p1 → RF → pipe2_wdata (~6.8 ns)
       Stage 3: pipe2_wdata → mem_wdata → BRAM    (~2.7 ns)
   - Pipeline Reg 1 (rf_rd_addr_p1, pipe1_*) registers the RF read
     address and store control signals (address, valid, size).
   - Pipeline Reg 2 (pipe2_*) registers the RF read data returned
     through cpu_mt's bdtu_rf_rd_data mux, plus forwarded control.
   - mem_wr is now driven by pipe2_wr_valid (pipelined).
   - mem_wdata is now driven by pipe2_wdata  (pipelined).
   - mem_addr muxes pipe2_addr (writes) vs cur_addr (reads).
   - mem_rd remains combinational (load path is not pipelined).
   - Added S_MEM_DRAIN state for store pipeline drain.
     State flow for BDT stores:
       S_BDT_XFER → S_MEM_DRAIN → [S_BDT_WB] → S_DONE → S_DRAIN
     State flow for SWP:
       S_SWP_WR   → S_MEM_DRAIN → S_DONE → S_DRAIN
   - pipe1_wr_valid_in asserts during S_BDT_XFER (!r_load,
     remaining!=0) and S_SWP_RD_WAIT, one cycle before the RF
     read data is needed.  This aligns with the rf_rd_addr
     captured in pipe1 on the same posedge.
   - State encoding widened to accommodate S_MEM_DRAIN (still 4 bits).
 */

`ifndef BDTU_V
`define BDTU_V
`include "define.v"

module bdtu (
    input wire clk,
    input wire rst_n,

    input wire start,
    input wire op_bdt,
    input wire op_swp,

    input wire [15:0] reg_list,
    input wire bdt_load,
    input wire bdt_wb,
    input wire pre_index,
    input wire up_down,
    input wire bdt_s,

    input wire swap_byte,
    input wire [3:0] swp_rd,
    input wire [3:0] swp_rm,

    input wire [3:0] base_reg,
    input wire [`DATA_WIDTH-1:0] base_value,

    output wire [3:0] rf_rd_addr,
    input wire [`DATA_WIDTH-1:0] rf_rd_data,

    output wire [3:0] wr_addr1,
    output wire [`DATA_WIDTH-1:0] wr_data1,
    output wire wr_en1,
    output wire [3:0] wr_addr2,
    output wire [`DATA_WIDTH-1:0] wr_data2,
    output wire wr_en2,

    output wire [`CPU_DMEM_ADDR_WIDTH-1:0] mem_addr,
    output wire [`DATA_WIDTH-1:0] mem_wdata,
    output wire mem_rd,
    output wire mem_wr,
    output wire [1:0] mem_size,
    input wire [`DATA_WIDTH-1:0] mem_rdata,

    output wire busy
);

/* -----------------------------------------------------------
   State encoding — 4 bits
   ----------------------------------------------------------- */
localparam [3:0]
    S_IDLE        = 4'd0,
    S_BDT_XFER    = 4'd1,
    S_BDT_LAST    = 4'd2,
    S_BDT_WB      = 4'd3,
    S_SWP_RD      = 4'd4,
    S_SWP_RD_WAIT = 4'd5,
    S_SWP_WR      = 4'd6,
    S_DONE        = 4'd7,
    S_DRAIN       = 4'd8,
    S_MEM_DRAIN   = 4'd9;   // store pipeline drain

reg [3:0] state;

reg [15:0] remaining;
reg [`CPU_DMEM_ADDR_WIDTH-1:0] cur_addr;
reg [`DATA_WIDTH-1:0] r_new_base;
reg r_load;
reg r_wb;
reg r_is_swp;
reg r_byte;
reg [3:0] r_base_reg;
reg [3:0] r_swp_rd;
reg [3:0] r_swp_rm;
reg [`DATA_WIDTH-1:0] swp_temp;

reg [3:0] prev_reg;
reg rd_pending;

/* ---- popcount ---- */
wire [1:0] pc_l1_0 = {1'b0, reg_list[0]}  + {1'b0, reg_list[1]};
wire [1:0] pc_l1_1 = {1'b0, reg_list[2]}  + {1'b0, reg_list[3]};
wire [1:0] pc_l1_2 = {1'b0, reg_list[4]}  + {1'b0, reg_list[5]};
wire [1:0] pc_l1_3 = {1'b0, reg_list[6]}  + {1'b0, reg_list[7]};
wire [1:0] pc_l1_4 = {1'b0, reg_list[8]}  + {1'b0, reg_list[9]};
wire [1:0] pc_l1_5 = {1'b0, reg_list[10]} + {1'b0, reg_list[11]};
wire [1:0] pc_l1_6 = {1'b0, reg_list[12]} + {1'b0, reg_list[13]};
wire [1:0] pc_l1_7 = {1'b0, reg_list[14]} + {1'b0, reg_list[15]};

wire [2:0] pc_l2_0 = {1'b0, pc_l1_0} + {1'b0, pc_l1_1};
wire [2:0] pc_l2_1 = {1'b0, pc_l1_2} + {1'b0, pc_l1_3};
wire [2:0] pc_l2_2 = {1'b0, pc_l1_4} + {1'b0, pc_l1_5};
wire [2:0] pc_l2_3 = {1'b0, pc_l1_6} + {1'b0, pc_l1_7};

wire [3:0] pc_l3_0 = {1'b0, pc_l2_0} + {1'b0, pc_l2_1};
wire [3:0] pc_l3_1 = {1'b0, pc_l2_2} + {1'b0, pc_l2_3};

wire [4:0] num_regs = {1'b0, pc_l3_0} + {1'b0, pc_l3_1};

/* ---- priority encoder ---- */
wire [3:0] cur_reg = remaining[0]  ? 4'd0  :
                     remaining[1]  ? 4'd1  :
                     remaining[2]  ? 4'd2  :
                     remaining[3]  ? 4'd3  :
                     remaining[4]  ? 4'd4  :
                     remaining[5]  ? 4'd5  :
                     remaining[6]  ? 4'd6  :
                     remaining[7]  ? 4'd7  :
                     remaining[8]  ? 4'd8  :
                     remaining[9]  ? 4'd9  :
                     remaining[10] ? 4'd10 :
                     remaining[11] ? 4'd11 :
                     remaining[12] ? 4'd12 :
                     remaining[13] ? 4'd13 :
                     remaining[14] ? 4'd14 :
                     remaining[15] ? 4'd15 :
                                     4'd0;

/* ---- address / base calculation ---- */
wire [`DATA_WIDTH-1:0] total_off  = {{(`DATA_WIDTH-5){1'b0}}, num_regs, 2'b00};
wire [`DATA_WIDTH-1:0] base_up    = base_value + total_off;
wire [`DATA_WIDTH-1:0] base_dn    = base_value - total_off;

wire [`DATA_WIDTH-1:0] start_addr = up_down
    ? (pre_index ? base_value + 32'd4 : base_value)
    : (pre_index ? base_dn            : base_dn + 32'd4);

wire [`DATA_WIDTH-1:0] calc_new_base = up_down ? base_up : base_dn;

wire is_last = (remaining != 16'd0) &&
               ((remaining & (remaining - 16'd1)) == 16'd0);

/* ==============================================================
   Store-path pipeline registers (2 stages)

   Pipeline Reg 1 (pipe1): captures the RF read address and
     store control signals.  rf_rd_addr_p1 drives the RF's r3
     port via cpu_mt; the RF returns data combinationally.

   Pipeline Reg 2 (pipe2): captures the RF read data (returned
     as rf_rd_data) plus forwarded control from pipe1.  pipe2
     drives the actual memory write port.

   The read path (mem_rd for LDM / SWP read) is NOT pipelined;
   mem_rd and the read address use cur_addr directly.
   ============================================================== */

/* Combinational rf_rd_addr — feeds pipe1, NOT output directly */
wire [3:0] rf_rd_addr_comb =
    (state == S_BDT_XFER && !r_load)              ? cur_reg  :
    (state == S_SWP_RD_WAIT || state == S_SWP_WR)  ? r_swp_rm :
                                                      4'd0;

/* pipe1 write-valid input: asserted the cycle BEFORE the RF
   read data is needed by pipe2.
   - BDT store: during S_BDT_XFER when there are registers left
   - SWP:       during S_SWP_RD_WAIT (so the RF read of Rm
                happens during S_SWP_WR and pipe2 captures it)  */
wire pipe1_wr_valid_in =
    (state == S_BDT_XFER && !r_load && remaining != 16'd0)
  | (state == S_SWP_RD_WAIT);

/* Pipeline stage 1 registers */
reg [3:0]                       rf_rd_addr_p1;
reg [`CPU_DMEM_ADDR_WIDTH-1:0]  pipe1_addr;
reg                             pipe1_wr_valid;
reg [1:0]                       pipe1_size;

/* Pipeline stage 2 registers */
reg [`DATA_WIDTH-1:0]           pipe2_wdata;
reg [`CPU_DMEM_ADDR_WIDTH-1:0]  pipe2_addr;
reg                             pipe2_wr_valid;
reg [1:0]                       pipe2_size;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rf_rd_addr_p1  <= 4'd0;
        pipe1_addr     <= {`CPU_DMEM_ADDR_WIDTH{1'b0}};
        pipe1_wr_valid <= 1'b0;
        pipe1_size     <= 2'b10;
        pipe2_wdata    <= {`DATA_WIDTH{1'b0}};
        pipe2_addr     <= {`CPU_DMEM_ADDR_WIDTH{1'b0}};
        pipe2_wr_valid <= 1'b0;
        pipe2_size     <= 2'b10;
    end
    else begin
        /* Stage 1: capture from FSM combinational outputs */
        rf_rd_addr_p1  <= rf_rd_addr_comb;
        pipe1_addr     <= cur_addr;
        pipe1_wr_valid <= pipe1_wr_valid_in;
        pipe1_size     <= (r_is_swp && r_byte) ? 2'b00 : 2'b10;

        /* Stage 2: capture RF read data + forwarded control */
        pipe2_wdata    <= rf_rd_data;
        pipe2_addr     <= pipe1_addr;
        pipe2_wr_valid <= pipe1_wr_valid;
        pipe2_size     <= pipe1_size;
    end
end

/* ---- FSM ---- */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state      <= S_IDLE;
        remaining  <= 16'd0;
        cur_addr   <= {`CPU_DMEM_ADDR_WIDTH{1'b0}};
        r_new_base <= {`DATA_WIDTH{1'b0}};
        r_load     <= 1'b0;
        r_wb       <= 1'b0;
        r_is_swp   <= 1'b0;
        r_byte     <= 1'b0;
        r_base_reg <= 4'd0;
        r_swp_rd   <= 4'd0;
        r_swp_rm   <= 4'd0;
        swp_temp   <= {`DATA_WIDTH{1'b0}};
        prev_reg   <= 4'd0;
        rd_pending <= 1'b0;
    end
    else begin
        case (state)

        S_IDLE: begin
            rd_pending <= 1'b0;
            if (start) begin
                r_base_reg <= base_reg;
                if (op_swp) begin
                    r_is_swp <= 1'b1;
                    r_byte   <= swap_byte;
                    r_swp_rd <= swp_rd;
                    r_swp_rm <= swp_rm;
                    r_load   <= 1'b0;
                    r_wb     <= 1'b0;
                    cur_addr <= base_value;
                    state    <= S_SWP_RD;
                end
                else if (op_bdt) begin
                    r_is_swp   <= 1'b0;
                    r_load     <= bdt_load;
                    r_wb       <= bdt_wb;
                    r_new_base <= calc_new_base;
                    remaining  <= reg_list;
                    cur_addr   <= start_addr;
                    state      <= S_BDT_XFER;
                end
            end
        end

        /* ---- BDT transfer loop ----
           Loads: unchanged from v1.4 (read path not pipelined).
           Stores: FSM still iterates at 1 register/cycle, but
             the actual memory writes happen 2 cycles later via
             the store pipeline.  Transition to S_MEM_DRAIN
             (instead of S_DONE) to drain the pipe.           */
        S_BDT_XFER: begin
            if (remaining == 16'd0) begin
                rd_pending <= 1'b0;
                if (r_load)
                    state <= r_wb ? S_BDT_WB : S_DONE;
                else
                    state <= S_MEM_DRAIN;
            end
            else begin
                cur_addr  <= cur_addr + 32'd4;
                remaining <= remaining & (remaining - 16'd1);
                if (r_load) begin
                    prev_reg   <= cur_reg;
                    rd_pending <= 1'b1;
                end
                if (is_last) begin
                    if (r_load)
                        state <= S_BDT_LAST;
                    else
                        state <= S_MEM_DRAIN;
                end
            end
        end

        S_BDT_LAST: begin
            rd_pending <= 1'b0;
            state <= r_wb ? S_BDT_WB : S_DONE;
        end

        S_BDT_WB: begin
            state <= S_DONE;
        end

        S_SWP_RD: begin
            state <= S_SWP_RD_WAIT;
        end

        S_SWP_RD_WAIT: begin
            swp_temp <= r_byte ? {{(`DATA_WIDTH-8){1'b0}}, mem_rdata[7:0]}
                               : mem_rdata;
            state <= S_SWP_WR;
        end

        /* S_SWP_WR: wr_en1 writes swp_temp → Rd (RF write path).
           The Rm → memory store is handled by the store pipeline:
           pipe1 was loaded during S_SWP_RD_WAIT, RF reads Rm this
           cycle, pipe2 captures data at this posedge, and pipe2
           drives the memory write during S_MEM_DRAIN.            */
        S_SWP_WR: begin
            state <= S_MEM_DRAIN;
        end

        /* S_MEM_DRAIN: busy=1, store pipeline draining.
           pipe1 → RF read in progress (last store data).
           pipe2 → may be writing the second-to-last store.
           Transition:
             SWP or BDT store without WB → S_DONE
             BDT store with WB           → S_BDT_WB             */
        S_MEM_DRAIN: begin
            state <= (!r_is_swp && r_wb) ? S_BDT_WB : S_DONE;
        end

        /* S_DONE: busy=1 — RF write-port drain cycle.
           pipe2 may still be writing the last store to memory.
           The write-port pipeline register in cpu_mt commits
           the BDTU's last wr_en assertion on this cycle.        */
        S_DONE: begin
            rd_pending <= 1'b0;
            state <= S_DRAIN;
        end

        /* S_DRAIN: busy=0 — pipeline resumes on next posedge.
           Store pipeline is fully drained (pipe2_wr_valid=0).   */
        S_DRAIN: begin
            state <= S_IDLE;
        end

        default: state <= S_IDLE;

        endcase
    end
end

/* ---- last-write data latch (unchanged) ---- */
reg [`DATA_WIDTH-1:0] last_wr_data1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        last_wr_data1 <= {`DATA_WIDTH{1'b0}};
    else if (rd_pending && (state == S_BDT_XFER || state == S_BDT_LAST))
        last_wr_data1 <= mem_rdata;
    else if (state == S_SWP_WR)
        last_wr_data1 <= swp_temp;
end

/* ==============================================================
   Combinational output logic
   ============================================================== */

/* Busy / pipeline stall.
   S_MEM_DRAIN keeps busy=1 (store pipe + RF drain).
   S_DONE keeps busy=1 (RF write-port drain).
   S_DRAIN releases busy=0 so the pipeline advances.            */
assign busy = (state == S_IDLE)  ? start  :
              (state == S_DRAIN) ? 1'b0   : 1'b1;

/* Register file read address — PIPELINED (stage 1 output).
   Drives the RF's r3 port via cpu_mt one cycle after the FSM
   computes rf_rd_addr_comb.                                     */
assign rf_rd_addr = rf_rd_addr_p1;

/* Memory address — mux between pipelined (writes) and direct (reads).
   When pipe2 has a valid store, use the pipelined address.
   Otherwise use cur_addr for reads (LDM, SWP_RD).              */
assign mem_addr = pipe2_wr_valid ? pipe2_addr : cur_addr;

/* Memory read enable — NOT pipelined (load path is direct).     */
assign mem_rd = (state == S_BDT_XFER && r_load && remaining != 16'd0)
              | (state == S_SWP_RD);

/* Memory write enable — PIPELINED (driven by pipe2).            */
assign mem_wr = pipe2_wr_valid;

/* Memory write data — PIPELINED (from pipe2 RF read capture).
   Byte replication for SWP byte mode.                           */
assign mem_wdata = (r_is_swp && r_byte) ? {4{pipe2_wdata[7:0]}}
                                        : pipe2_wdata;

/* Memory access size.
   r_is_swp and r_byte are stable throughout the operation,
   so the pipelined and direct values are identical.             */
assign mem_size = (r_is_swp && r_byte) ? 2'b00 : 2'b10;

/* ---- Register write port 1 (LDM data / SWP Rd) ----
   NOT pipelined — these go through cpu_mt's RF write-port
   pipeline register (wena_r) which provides 1-cycle delay.
   S_DONE drains that register.                                  */
assign wr_addr1 = r_is_swp ? r_swp_rd : prev_reg;
assign wr_data1 = (state == S_SWP_WR) ? swp_temp : mem_rdata;
assign wr_en1   = (rd_pending && (state == S_BDT_XFER || state == S_BDT_LAST))
                | (state == S_SWP_WR);

/* ---- Register write port 2 (base writeback) ----              */
assign wr_addr2 = r_base_reg;
assign wr_data2 = r_new_base;
assign wr_en2   = (state == S_BDT_WB);

endmodule

`endif // BDTU_V