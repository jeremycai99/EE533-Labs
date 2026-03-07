/* file: regfile.v
 Description: Register file module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Mar. 6, 2026
 Version: 2.0
 Revision history:
    - 1.0: Initial version with basic functionality for Lab 5 only (Feb. 9, 2026)
    - 1.1: Updated version with register forwarding support (Feb. 17, 2026)
    - 1.2: Updated version with support for second write port for multiply instructions (Feb. 18, 2026)
    - 1.3: Updated version with 4 read ports and forwarding removed for Arm pipeline design (Mar. 6, 2026)
    - 1.4: Updated version with 4R1W, 2-copy distributed RAM (Mar. 6, 2026)
    - 2.0: Updated version with BRAM-based shared regfile for 4 threads, single instance, 3 sync read ports + 1 async read port (Mar. 6, 2026)
 */

`ifndef REGFILE_V
`define REGFILE_V

`include "define.v"

module regfile (
    input wire clk,

    // Synchronous read ports (BRAM) — present addr cycle N, data at N+1
    input wire [5:0] r1addr,
    output reg [`REG_DATA_WIDTH-1:0] r1data,
    input wire [5:0] r2addr,
    output reg [`REG_DATA_WIDTH-1:0] r2data,
    input wire [5:0] r4addr,
    output reg [`REG_DATA_WIDTH-1:0] r4data,

    // Asynchronous read port (distributed RAM) — same-cycle data
    input wire [5:0] r3addr,
    output wire [`REG_DATA_WIDTH-1:0] r3data,

    // Write port (mirrored to all storage)
    input wire wr_en,
    input wire [5:0] wr_addr,
    input wire [`REG_DATA_WIDTH-1:0] wr_data
);

    // ================================================================
    //  BRAM storage: 3 copies, one per sync read port.
    //  Each: 64×32, Port A = sync read, Port B = sync write.
    //  XST infers RAMB16 in true dual-port mode.
    // ================================================================
    (* ram_style = "block" *) reg [`REG_DATA_WIDTH-1:0] mem_r1 [0:63];
    (* ram_style = "block" *) reg [`REG_DATA_WIDTH-1:0] mem_r2 [0:63];
    (* ram_style = "block" *) reg [`REG_DATA_WIDTH-1:0] mem_r4 [0:63];

    // Sync reads (Port A behavior)
    always @(posedge clk) begin
        r1data <= mem_r1[r1addr];
        r2data <= mem_r2[r2addr];
        r4data <= mem_r4[r4addr];
    end

    // Sync writes mirrored to all BRAM copies (Port B behavior)
    always @(posedge clk) begin
        if (wr_en) begin
            mem_r1[wr_addr] <= wr_data;
            mem_r2[wr_addr] <= wr_data;
            mem_r4[wr_addr] <= wr_data;
        end
    end

    // ================================================================
    //  Distributed RAM shadow: r3 async read port.
    //  Required for BDTU same-cycle register reads.
    //  64×32 = ~128 LUTs (2 LUTs per RAM16X1D × 64 entries... 
    //  actually 32 RAM16X1D per bit × 1 bit... let me just annotate).
    // ================================================================
    (* ram_style = "distributed" *) reg [`REG_DATA_WIDTH-1:0] mem_r3 [0:63];

    // Async read
    assign r3data = mem_r3[r3addr];

    // Sync write mirrored
    always @(posedge clk) begin
        if (wr_en)
            mem_r3[wr_addr] <= wr_data;
    end

endmodule

`endif // REGFILE_V