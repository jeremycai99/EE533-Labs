/* file: gpr_regfile.v
 Description: GPU general-purpose register file — 16 × 16-bit, 8R4W.
 Two sets of read ports:
   Ports 1-4 (pipeline): driven by registered rr_rf_r*_addr, feed sp_core id_ex.
   Ports 5-8 (override): driven by bypass mux (TC/BU/debug), feed external consumers.
 Both sets are pure async reads without forwarding.
 Forwarding is unnecessary because:
   - Scoreboard ensures dependent reads happen ≥1 cycle after producer WB write.
   - External writes (TC scatter, BU load) occur when pipeline is drained.
 Author: Jeremy Cai
 Date: Mar. 6, 2026
 Version: 2.0
 Revision history:
    - Feb. 26, 2026: v1.0 — 4R4W with forwarding.
    - Mar. 02, 2026: v1.1 — 8R4W, forwarding removed, dual read ports
      for timing isolation (pipeline vs TC/BU override).
    - Mar. 6, 2026: v2.0 — 4R1W, dual-copy distributed RAM, read/write port merging.
*/

`ifndef GPR_REGFILE_V
`define GPR_REGFILE_V

`include "gpu_define.v"

module gpr_regfile (
    input wire clk,
    input wire rst_n,           // unused for storage; kept for interface compat

    // Pipeline read addresses (used when ovr_sel == 0)
    input wire [3:0] read_addr1,
    input wire [3:0] read_addr2,
    input wire [3:0] read_addr3,
    input wire [3:0] read_addr4,

    // Override read addresses (used when ovr_sel == 1)
    input wire [3:0] ovr_read_addr1,
    input wire [3:0] ovr_read_addr2,
    input wire [3:0] ovr_read_addr3,
    input wire [3:0] ovr_read_addr4,

    // Override select: 0 = pipeline reads, 1 = override reads
    input wire ovr_sel,

    // Read data outputs (reflects whichever address set is active)
    output wire [15:0] read_data1,
    output wire [15:0] read_data2,
    output wire [15:0] read_data3,
    output wire [15:0] read_data4,

    // Single write port
    input wire [3:0] write_addr,
    input wire [15:0] write_data,
    input wire write_en
);

    // ================================================================
    //  Read address mux: pipeline vs override
    // ================================================================
    wire [3:0] ra1 = ovr_sel ? ovr_read_addr1 : read_addr1;
    wire [3:0] ra2 = ovr_sel ? ovr_read_addr2 : read_addr2;
    wire [3:0] ra3 = ovr_sel ? ovr_read_addr3 : read_addr3;
    wire [3:0] ra4 = ovr_sel ? ovr_read_addr4 : read_addr4;

    // ================================================================
    //  Storage: 2 copies of 16×16b unpacked arrays.
    //  Each copy → 1 sync write + 2 async reads (RAM16X1D).
    //  Copy 0 serves read ports 1 & 2.
    //  Copy 1 serves read ports 3 & 4.
    //
    //  NO synchronous reset — essential for distributed RAM inference.
    //  XST attribute ensures the tool doesn't promote to block RAM.
    // ================================================================

    (* ram_style = "distributed" *) reg [15:0] rf_copy0 [0:15];
    (* ram_style = "distributed" *) reg [15:0] rf_copy1 [0:15];

    // ================================================================
    //  Async read — maps to dist RAM read ports (SPO + DPO)
    // ================================================================
    assign read_data1 = rf_copy0[ra1];
    assign read_data2 = rf_copy0[ra2];
    assign read_data3 = rf_copy1[ra3];
    assign read_data4 = rf_copy1[ra4];

    // ================================================================
    //  Synchronous write — single port, mirrored to both copies.
    //  No reset: dist RAM contents undefined at power-up; kernel
    //  software must initialize registers before use.
    // ================================================================
    always @(posedge clk) begin
        if (write_en) begin
            rf_copy0[write_addr] <= write_data;
            rf_copy1[write_addr] <= write_data;
        end
    end

endmodule

`endif // GPR_REGFILE_V