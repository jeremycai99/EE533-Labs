/* file: soc_mt.v
 Description: SoC top module for the Quad-thread CPU design with memory access interface
 Author: Jeremy Cai
 Date: Feb. 14, 2026
 Version: 1.6
 Changes:
   - IMEM CPU path: byte-addressed PC converted to word index (>> 2)
   - DMEM CPU path: byte-addressed data addr converted to word index (>> 2)
   - CPU outputs gated with system_active to prevent X propagation
   - Added d_mem_size_o port from updated cpu interface
   - DMEM response explicitly zero-extended to MMIO_DATA_WIDTH
 */

`ifndef SOC_MT_V
`define SOC_MT_V

`include "define.v"

// CPU and test memory modules
// Remove from soc.v when synthesis
// ---------------------------
`include "test_i_mem.v"
`include "test_d_mem.v"
// ---------------------------

// `include "i_mem.v"
// `include "d_mem.v"

`include "cpu_mt.v"

module soc_mt (
    input wire clk,
    input wire rst_n,

    input wire req_cmd, //Request command 0 for read and 1 for write
    input wire [`MMIO_ADDR_WIDTH-1:0] req_addr, //Fixed request address width
    input wire [`MMIO_DATA_WIDTH-1:0] req_data, //Data to write for write requests
    input wire req_val,
    output wire req_rdy,

    output wire resp_cmd, //Response command 0 for read response and 1 for write response
    output wire [`MMIO_ADDR_WIDTH-1:0] resp_addr, //Response address
    output wire [`MMIO_DATA_WIDTH-1:0] resp_data, //Response data
    output wire resp_val,
    input wire resp_rdy,

    // External start signal (Active High, Level Sensitive)
<<<<<<< HEAD
<<<<<<< HEAD
    input wire start
=======
=======
>>>>>>> refs/remotes/origin/timing_opt
    input wire start,
    
    // Expanded Debug Interface
    // Bit 4: 0 = System Debug, 1 = Register File Debug
    // Bits 3-0: Selection index or Register Address
    input wire [1:0] ila_thread_sel,
    input wire [4:0] ila_debug_sel,       
    output wire [`DATA_WIDTH-1:0] ila_debug_data // Full 64-bit debug data output
<<<<<<< HEAD
>>>>>>> refs/remotes/origin/timing_opt
=======
>>>>>>> refs/remotes/origin/timing_opt
);

//  Address Region Decode — req_addr[31:30]
//    2'b00  (0x0000_0000)  →  IMEM   (blocked while CPU active)
//    2'b01  (0x4000_0000)  →  CTRL   write: start CPU (internal latch)
//                                     read : {31'b0, system_active}
//    2'b10  (0x8000_0000)  →  DMEM   (Port B — safe any time)
//    2'b11                 →  reserved (returns 0)

localparam REGION_IMEM = 2'b00;
localparam REGION_CTRL = 2'b01;
localparam REGION_DMEM = 2'b10;
localparam REGION_RESERVED = 2'b11;

//MMIO interface FSM states
localparam STATE_IDLE = 2'b00;
localparam STATE_ACCESS = 2'b01;
localparam STATE_RESP = 2'b10;

reg [1:0] current_state, next_state;

//MMIO interface registers
reg req_cmd_reg;
reg [`MMIO_ADDR_WIDTH-1:0] req_addr_reg;
reg [`MMIO_DATA_WIDTH-1:0] req_data_reg;
reg [1:0] req_region_reg;

reg resp_pending;
reg resp_val_reg;
reg resp_cmd_reg;
reg [`MMIO_ADDR_WIDTH-1:0] resp_addr_reg;
reg [`MMIO_DATA_WIDTH-1:0] resp_data_reg;

//CPU control signals
reg cpu_active; // Internal register for MMIO-triggered start
wire system_active; // Combined active signal (MMIO latch OR external start pin)
wire cpu_done; // Indicates whether the CPU has completed execution

// Combine external start pin with internal MMIO active latch
assign system_active = cpu_active | start;

//MMIO interface handshaking signals
wire req_fire = req_val && req_rdy; // Indicates a valid request handshake
wire req_is_ctrl = (req_region_reg == REGION_CTRL); // Indicates if the request is targeting the control region
wire req_is_imem = (req_region_reg == REGION_IMEM); // Indicates if the request is targeting the instruction memory region
wire req_is_dmem = (req_region_reg == REGION_DMEM); // Indicates if the request is targeting the data memory region

// The two-port data memory has the advantage to access data memory without stalling the CPU 
// However, we block access if the CPU is running (via MMIO or external Start pin) to prevent IMEM conflicts
assign req_rdy = (current_state == STATE_IDLE) && (~system_active);

assign resp_val = resp_val_reg;
assign resp_cmd = resp_cmd_reg;
assign resp_addr = resp_addr_reg;
assign resp_data = resp_data_reg;

//Host interface active flags
wire host_active = (current_state == STATE_ACCESS) || (current_state == STATE_RESP); //Qualify only when in access or response state

wire imem_host_active = host_active && req_is_imem;
wire dmem_host_active = host_active && req_is_dmem;
wire ctrl_host_active = host_active && req_is_ctrl;

//Instruction memory muxing
wire [`PC_WIDTH-1:0] cpu_imem_addr;     // byte address from CPU (PC += 4)
wire [`INSTR_WIDTH-1:0] imem_dout;

reg [`PC_WIDTH-1:0] imem_addr_mux;
reg [`INSTR_WIDTH-1:0] imem_din_mux;
reg imem_we_mux;

// Data memory address and data muxing logic
wire [`CPU_DMEM_ADDR_WIDTH-1:0] cpu_dmem_addr;  // byte address from CPU
wire [`DATA_WIDTH-1:0]      cpu_dmem_wdata;
wire                        cpu_dmem_wen;
wire [1:0]                  cpu_dmem_size;   // access size from CPU (not used by word-only BRAM)
wire [`DATA_WIDTH-1:0]      dmem_douta;      // Port A read  → CPU
wire [`DATA_WIDTH-1:0]      dmem_doutb;

reg [`CPU_DMEM_ADDR_WIDTH-1:0] dmem_addr_mux;
reg [`DATA_WIDTH-1:0] dmem_din_mux;
reg dmem_we_mux;

// Instruction memory address muxing logic
// MMIO path: req_addr_reg carries a word index directly.
// CPU  path: cpu_imem_addr is a byte address (PC += 4),
//            so we right-shift by 2 to obtain the word index.
//            When the CPU is inactive (system_active == 0) the
//            address is parked at 0 to avoid X-propagation.
always @(*) begin
    if (imem_host_active) begin
        imem_addr_mux = req_addr_reg[(`PC_WIDTH-1):0]; 
        imem_din_mux = req_data_reg[(`INSTR_WIDTH-1):0]; 
        imem_we_mux = req_cmd_reg && ((current_state == STATE_ACCESS)); 
    end else begin
        imem_addr_mux = system_active ? (cpu_imem_addr >> 2) : {`PC_WIDTH{1'b0}};
        imem_din_mux = `INSTR_WIDTH'b0; 
        imem_we_mux = 1'b0; 
    end
end

// Data memory muxing logic
// MMIO path: req_addr_reg carries a word index directly.
// CPU  path: cpu_dmem_addr is a byte address produced by the
//            CPU's load/store unit, so we right-shift by 2 to
//            obtain the word index for the word-only BRAM.
//            When the CPU is inactive, all signals are driven to
//            safe values (addr=0, data=0, we=0) to prevent
//            X-propagation from uninitialized CPU outputs.
always @(*) begin
    if (dmem_host_active) begin
        dmem_addr_mux = req_addr_reg[(`CPU_DMEM_ADDR_WIDTH-1):0];
        dmem_din_mux = req_data_reg[(`DATA_WIDTH-1):0];
        dmem_we_mux = req_cmd_reg && ((current_state == STATE_ACCESS));
    end else begin
        dmem_addr_mux = system_active ? (cpu_dmem_addr >> 2) : {`CPU_DMEM_ADDR_WIDTH{1'b0}};
        dmem_din_mux  = system_active ? cpu_dmem_wdata       : {`DATA_WIDTH{1'b0}};
        dmem_we_mux   = system_active & cpu_dmem_wen;
    end
end


// CPU start and done logic (MMIO based)
// Note: This logic only controls the internal latch 'cpu_active'.
// The actual CPU reset is controlled by 'system_active' which includes the 'start' pin.
wire start_pulse = (current_state == STATE_ACCESS) && req_is_ctrl && req_cmd_reg && !system_active; 

always @ (posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cpu_active <= 1'b0;
    end else begin
        if (start_pulse) begin
            cpu_active <= 1'b1; // Latch active on MMIO command
        end else if (cpu_done) begin
            cpu_active <= 1'b0; // Clear latch when CPU finishes
        end
    end
end

//MMIO interface FSM design
always @(*) begin
    next_state = current_state;
    case (current_state)
        STATE_IDLE: if (req_fire) next_state = STATE_ACCESS;
        STATE_ACCESS: next_state = STATE_RESP;
        STATE_RESP: if (resp_val_reg && resp_rdy) next_state = STATE_IDLE;
        default: next_state = STATE_IDLE;
    endcase
end

// MMIO interface sequential logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_state <= STATE_IDLE;
        req_cmd_reg <= 1'b0;
        req_addr_reg <= {`MMIO_ADDR_WIDTH{1'b0}};
        req_data_reg <= {`MMIO_DATA_WIDTH{1'b0}};
        req_region_reg <= 2'b00;
        resp_pending <= 1'b0;
        resp_val_reg <= 1'b0;
        resp_cmd_reg <= 1'b0;
        resp_addr_reg <= {`MMIO_ADDR_WIDTH{1'b0}};
        resp_data_reg <= {`MMIO_DATA_WIDTH{1'b0}};
    end else begin
        current_state <= next_state;
        if (req_fire) begin
            req_cmd_reg <= req_cmd;
            req_addr_reg <= req_addr;
            req_data_reg <= req_data;
            req_region_reg <= req_addr[31:30];
            resp_pending <= 1'b1; 
            resp_val_reg <= 1'b0; 
        end

        if (resp_val_reg && resp_rdy) begin
            resp_val_reg <= 1'b0; 
            resp_pending <= 1'b0; 
        end

        if (current_state == STATE_RESP && resp_pending && ~resp_val_reg) begin
            resp_val_reg <= 1'b1; 
            resp_cmd_reg <= req_cmd_reg; 
            resp_addr_reg <= req_addr_reg; 

            // Generate response based on the request region
            case (req_region_reg)
                REGION_IMEM: resp_data_reg <= {{(`MMIO_DATA_WIDTH-`INSTR_WIDTH){1'b0}}, imem_dout}; 
                REGION_DMEM: resp_data_reg <= {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, dmem_douta}; 
                // Return system_active (includes external start pin status)
                REGION_CTRL: resp_data_reg <= {{(`MMIO_DATA_WIDTH-1){1'b0}}, system_active}; 
                default: resp_data_reg <= {`MMIO_DATA_WIDTH{1'b0}}; 
            endcase
        end
    end
end

// CPU Reset Logic:
// The CPU is active (out of reset) if global reset is high AND (MMIO started it OR external start pin is high)
wire cpu_rst_n = rst_n & system_active; 

cpu_mt u_cpu_mt (
    .clk(clk),
    .rst_n(cpu_rst_n),
    .i_mem_data_i(imem_dout),
    .i_mem_addr_o(cpu_imem_addr),
    .d_mem_addr_o(cpu_dmem_addr),
    .d_mem_data_i(dmem_douta),
    .d_mem_data_o(cpu_dmem_wdata),
    .d_mem_wen_o(cpu_dmem_wen),
    .d_mem_size_o(cpu_dmem_size),
<<<<<<< HEAD
<<<<<<< HEAD
    .cpu_done(cpu_done)
=======
=======
>>>>>>> refs/remotes/origin/timing_opt
    .cpu_done(cpu_done),
    
    .ila_thread_sel(ila_thread_sel),
    .ila_debug_sel(ila_debug_sel),       
    .ila_debug_data(ila_debug_data)      
<<<<<<< HEAD
>>>>>>> refs/remotes/origin/timing_opt
=======
>>>>>>> refs/remotes/origin/timing_opt
);

//Make sure that the address space doesn't exceed the size of imem and dmem

test_i_mem u_i_mem (
    .clk(clk),
    .din(imem_din_mux), 
    .addr(imem_addr_mux[(`IMEM_ADDR_WIDTH-1):0]), 
    .we(imem_we_mux), 
    .dout(imem_dout)
);

test_d_mem u_d_mem (
    .clka(clk),
    .dina(dmem_din_mux), 
    .addra(dmem_addr_mux[(`DMEM_ADDR_WIDTH-1):0]), 
    .wea(dmem_we_mux), 
    .douta(dmem_douta), 
    // Port B reserved for GPU access in future labs
    .clkb(clk),
    .dinb({`DMEM_DATA_WIDTH{1'b0}}),
    .addrb({`DMEM_ADDR_WIDTH{1'b0}}),
    .web(1'b0),
    .doutb()
);

endmodule

`endif // SOC_MT_V