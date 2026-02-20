/* file: bdtu.v
 Description: Block data transfer unit module for multi-cycle instructions in Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 19, 2026
 Version: 1.3
 */

`ifndef BDTU_V
`define BDTU_V
`include "define.v"

module bdtu (
    input wire clk, // Clock signal
    input wire rst_n, // Active-low reset signal
    
    // Control signals from CU
    input wire start, // Connect to CU is_multi_cycle signal.
    input wire op_bdt, // The BDTU takes control of BDT instructions
    input wire op_swp, // The BDTU takes control of SWP instruction
    
    // BDT instruction field configurations from CU
    input wire [15:0] reg_list, // Register list for block data transfer instructions
    input wire bdt_load, // 1 = LDM, 0 = STM
    input wire bdt_wb, // W bt: Write-back flag for block data transfer instructions
    input wire pre_index, // P bit: Pre/Post indexing flag 1 = pre-indexing, 0 = post-indexing
    input wire up_down, // U bit: Up/Down flag 1 = increment, 0 = decrement
    input wire bdt_s, // S bit: PSR & force user bit for block data transfer instructions
    
    // SWP instruction field configurations from CU
    input wire swap_byte, // B bit: 1 = byte swap, 0 = word swap for SWP instruction
    input wire [3:0] swp_rd, // Rd register address for SWP instruction
    input wire [3:0] swp_rm, // Rm register address for SWP instruction

    // Common signals
    input wire [3:0] base_reg, // Base register address for BDT instructions
    input wire [`DATA_WIDTH-1:0] base_value, // Base register value for BDT instructions

    output wire [3:0] rf_rd_addr, // Register file read address (STM / SWP Rm)
    input wire [`DATA_WIDTH-1:0] rf_rd_data, // Register file read data

    output wire [3:0] wr_addr1, // Register file write address for BDT instruction
    output wire [`DATA_WIDTH-1:0] wr_data1, // Register file write data for BDT instruction
    output wire wr_en1, // Register file write enable for BDT instruction
    output wire [3:0] wr_addr2, // Register file write address for second write port (for multiply instructions)
    output wire [`DATA_WIDTH-1:0] wr_data2, // Register file write data for second write port (for multiply instructions)
    output wire wr_en2, // Register file write enable for second write port (for multiply instructions)

    // Memory interface signals
    output wire [`DMEM_ADDR_WIDTH-1:0] mem_addr, // Memory address for load/store operations
    output wire [`DATA_WIDTH-1:0] mem_wdata, // Memory write data for store operations
    output wire mem_rd, // Memory read enable signal not needed for our block memory design. Act as placeholder
    output wire mem_wr, // Memory write enable signal
    output wire [1:0] mem_size, // Memory access size: 00 = byte, 01 = halfword, 10 = word
    input wire [`DATA_WIDTH-1:0] mem_rdata, // Memory read data for load operations

    output wire busy // Signal to indicate BDTU is processing an instruction, used for stalling the pipeline in CU
);

localparam [2:0]
    S_IDLE = 3'd0,  // Waiting for trigger
    S_BDT_XFER = 3'd1,  // BDT: assert address; for LDM also write prev data
    S_BDT_LAST = 3'd2,  // BDT LDM: drain – write final mem_rdata, no new read
    S_BDT_WB = 3'd3,  // BDT: write updated base back to Rn
    S_SWP_RD = 3'd4,  // SWP: assert read address to memory
    S_SWP_RD_WAIT = 3'd5,  // SWP: wait for sync-read data, latch into swp_temp
    S_SWP_WR = 3'd6,  // SWP: write Rm -> memory, swp_temp -> Rd
    S_DONE = 3'd7;  // 1-cycle drain so pipeline can advance

reg [2:0] state;

reg [15:0] remaining; // BDT: unprocessed register bits
reg [`DMEM_ADDR_WIDTH-1:0] cur_addr; // Current memory address
reg [`DATA_WIDTH-1:0] r_new_base; // Pre-computed new base for writeback
reg r_load; // Latched bdt_load
reg r_wb; // Latched bdt_wb
reg r_is_swp; // 1 = SWP operation in progress
reg r_byte; // Latched swap_byte
reg [3:0] r_base_reg; // Latched Rn address
reg [3:0] r_swp_rd; // Latched SWP Rd
reg [3:0] r_swp_rm; // Latched SWP Rm
reg [`DATA_WIDTH-1:0] swp_temp; // SWP: value read from memory

// Sync-read pipeline registers (LDM only)
reg [3:0] prev_reg; // Register index whose read was issued last cycle
reg rd_pending; // 1 = a sync-read response is due this cycle

// Number of set bits in the register list, used for calculating address offsets and writeback value
// Level 1: pair-wise addition -> 8 × 2-bit sums 
wire [1:0] pc_l1_0 = {1'b0, reg_list[0]}  + {1'b0, reg_list[1]};
wire [1:0] pc_l1_1 = {1'b0, reg_list[2]}  + {1'b0, reg_list[3]};
wire [1:0] pc_l1_2 = {1'b0, reg_list[4]}  + {1'b0, reg_list[5]};
wire [1:0] pc_l1_3 = {1'b0, reg_list[6]}  + {1'b0, reg_list[7]};
wire [1:0] pc_l1_4 = {1'b0, reg_list[8]}  + {1'b0, reg_list[9]};
wire [1:0] pc_l1_5 = {1'b0, reg_list[10]} + {1'b0, reg_list[11]};
wire [1:0] pc_l1_6 = {1'b0, reg_list[12]} + {1'b0, reg_list[13]};
wire [1:0] pc_l1_7 = {1'b0, reg_list[14]} + {1'b0, reg_list[15]};

// Level 2: quad-wise addition -> 4 × 3-bit sums
wire [2:0] pc_l2_0 = {1'b0, pc_l1_0} + {1'b0, pc_l1_1};
wire [2:0] pc_l2_1 = {1'b0, pc_l1_2} + {1'b0, pc_l1_3};
wire [2:0] pc_l2_2 = {1'b0, pc_l1_4} + {1'b0, pc_l1_5};
wire [2:0] pc_l2_3 = {1'b0, pc_l1_6} + {1'b0, pc_l1_7};

// Level 3: octet-wise addition -> 2 × 4-bit sums
wire [3:0] pc_l3_0 = {1'b0, pc_l2_0} + {1'b0, pc_l2_1};
wire [3:0] pc_l3_1 = {1'b0, pc_l2_2} + {1'b0, pc_l2_3};

// Level 4: final 5-bit popcount
wire [4:0] num_regs = {1'b0, pc_l3_0} + {1'b0, pc_l3_1};

// Priority encoder
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

// BDT start address and base calculation
wire [`DATA_WIDTH-1:0] total_off = {{(`DATA_WIDTH-5){1'b0}}, num_regs, 2'b00};    // N × 4
wire [`DATA_WIDTH-1:0] base_up = base_value + total_off;          // Rn + 4N
wire [`DATA_WIDTH-1:0] base_dn = base_value - total_off;          // Rn − 4N

wire [`DATA_WIDTH-1:0] start_addr = up_down
    ? (pre_index ? base_value + 32'd4 : base_value)          // IB / IA
    : (pre_index ? base_dn : base_dn + 32'd4);  // DB / DA

wire [`DATA_WIDTH-1:0] calc_new_base = up_down ? base_up : base_dn;

// Detect the last transfer for the multi-cycle instructions
wire is_last = (remaining != 16'd0) && ((remaining & (remaining - 16'd1)) == 16'd0);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        remaining <= 16'd0;
        cur_addr <= {`DMEM_ADDR_WIDTH{1'b0}};
        r_new_base <= {`DATA_WIDTH{1'b0}};
        r_load <= 1'b0;
        r_wb <= 1'b0;
        r_is_swp <= 1'b0;
        r_byte <= 1'b0;
        r_base_reg <= 4'd0;
        r_swp_rd <= 4'd0;
        r_swp_rm <= 4'd0;
        swp_temp <= {`DATA_WIDTH{1'b0}};
        prev_reg <= 4'd0;
        rd_pending <= 1'b0;
    end
    else begin
        case (state)
        // IDLE state behavior
        S_IDLE: begin
            rd_pending <= 1'b0;
            if (start) begin
                r_base_reg <= base_reg;

                if (op_swp) begin
                    r_is_swp <= 1'b1;
                    r_byte <= swap_byte;
                    r_swp_rd <= swp_rd;
                    r_swp_rm <= swp_rm;
                    r_load <= 1'b0;
                    r_wb <= 1'b0;
                    cur_addr <= base_value; // address = [Rn]
                    state <= S_SWP_RD;
                end
                else if (op_bdt) begin
                    r_is_swp <= 1'b0;
                    r_load <= bdt_load;
                    r_wb <= bdt_wb;
                    r_new_base <= calc_new_base;
                    remaining <= reg_list;
                    cur_addr <= start_addr;
                    state <= S_BDT_XFER;
                end
            end
        end

        // BDT: transfer one register per cycle
        S_BDT_XFER: begin
            if (remaining == 16'd0) begin
                // Safety: empty register list (UNPREDICTABLE per ARM).
                // Skip transfers, proceed to WB or DONE.
                rd_pending <= 1'b0;
                state <= r_wb ? S_BDT_WB : S_DONE;
            end
            else begin
                cur_addr <= cur_addr + 32'd4;
                remaining <= remaining & (remaining - 16'd1);  // clear lowest bit

                if (r_load) begin
                    // LDM: track which register this cycle's address belongs to.
                    // The data for THIS address arrives next cycle (sync read).
                    prev_reg <= cur_reg;
                    rd_pending <= 1'b1;
                end

                if (is_last) begin
                    if (r_load)
                        state <= S_BDT_LAST;   // need drain cycle for final read
                    else
                        state <= r_wb ? S_BDT_WB : S_DONE;
                end
                // else: stay in S_BDT_XFER
            end
        end

        //  BDT LDM: drain the read pipeline
        S_BDT_LAST: begin
            rd_pending <= 1'b0;
            state <= r_wb ? S_BDT_WB : S_DONE;
        end

        //  BDT: base writeback
        S_BDT_WB: begin
            state <= S_DONE;
        end

        //  SWP: assert read address
        //  mem_addr = Rn, mem_rd = 1.  Data will be available on
        //  mem_rdata NEXT cycle.
        S_SWP_RD: begin
            state <= S_SWP_RD_WAIT;
        end

        //  SWP: wait for sync-read data
        //  mem_rdata now holds mem[Rn].  Latch it into swp_temp.
        S_SWP_RD_WAIT: begin
            swp_temp <= r_byte ? {{(`DATA_WIDTH-8){1'b0}}, mem_rdata[7:0]}
                               : mem_rdata;
            state <= S_SWP_WR;
        end

        //  SWP: write Rm -> memory, swp_temp -> Rd
        //  Both commits occur at the posedge leaving this state.
        S_SWP_WR: begin
            state <= S_DONE;
        end

        //  DONE: 1-cycle buffer
        //  busy = 0 here, so the pipeline advances on the next
        //  posedge.  The BDTU simultaneously returns to IDLE.
        S_DONE: begin
            rd_pending <= 1'b0;
            state <= S_IDLE;
        end

        default: state <= S_IDLE;

        endcase
    end
end

reg [`DATA_WIDTH-1:0] last_wr_data1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        last_wr_data1 <= {`DATA_WIDTH{1'b0}};
    else if (rd_pending && (state == S_BDT_XFER || state == S_BDT_LAST))
        last_wr_data1 <= mem_rdata;
    else if (state == S_SWP_WR)
        last_wr_data1 <= swp_temp;
end

// Combinational Output Logic
// Busy / pipeline stall
assign busy = (state == S_IDLE) ? start  :
              (state == S_DONE) ? 1'b0   : 1'b1;

// Register file read address
//  STM : read the register being stored to memory.
//  SWP_WR : read Rm (value to write to memory).
//  SWP_RD_WAIT: present Rm address one cycle early for setup time.
assign rf_rd_addr = (state == S_BDT_XFER && !r_load)              ? cur_reg   :
                    (state == S_SWP_RD_WAIT || state == S_SWP_WR)  ? r_swp_rm  :
                                                                     4'd0;

// Memory address
assign mem_addr = cur_addr;

// Memory read enable
assign mem_rd = (state == S_BDT_XFER && r_load && remaining != 16'd0)
              | (state == S_SWP_RD);

// Memory write enable
assign mem_wr = (state == S_BDT_XFER && !r_load && remaining != 16'd0)
              | (state == S_SWP_WR);

// Memory write data
assign mem_wdata = (r_is_swp && r_byte) ? {4{rf_rd_data[7:0]}} : rf_rd_data;

// Memory access size
assign mem_size = (r_is_swp && r_byte) ? 2'b00 : 2'b10;

// Register write port 1 (data)
//  LDM : write mem_rdata (from PREVIOUS cycle's address) into prev_reg.
//  SWP_WR : write swp_temp (latched in S_SWP_RD_WAIT) -> Rd.
//  S_DONE : defensive repeat of the last write (same data, same register).
//
assign wr_addr1 = r_is_swp ? r_swp_rd : prev_reg;
assign wr_data1 = (state == S_DONE)    ? last_wr_data1 :
                  (state == S_SWP_WR)  ? swp_temp      : mem_rdata;
assign wr_en1 = (rd_pending && (state == S_BDT_XFER || state == S_BDT_LAST))
                | (state == S_SWP_WR)
                | (state == S_DONE && (r_load || r_is_swp));

// Register write port 2 (base writeback)
assign wr_addr2 = r_base_reg;
assign wr_data2 = r_new_base;
assign wr_en2   = (state == S_BDT_WB)
                | (state == S_DONE && r_wb);

endmodule

`endif // BDTU_V