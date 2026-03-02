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
 Date: Feb. 26, 2026
 Version: 1.1
 Revision history:
    - Feb. 26, 2026: v1.0 — 4R4W with forwarding.
    - Mar. 02, 2026: v1.1 — 8R4W, forwarding removed, dual read ports
      for timing isolation (pipeline vs TC/BU override).
*/

`ifndef GPR_REGFILE_V
`define GPR_REGFILE_V

`include "gpu_define.v"

module gpr_regfile (
    input wire clk,
    input wire rst_n,

    // Pipeline read ports (for sp_core id_ex capture)
    input wire [3:0] read_addr1,
    input wire [3:0] read_addr2,
    input wire [3:0] read_addr3,
    input wire [3:0] read_addr4,
    output wire [15:0] read_data1,
    output wire [15:0] read_data2,
    output wire [15:0] read_data3,
    output wire [15:0] read_data4,

    // Override read ports (for TC gather / BU data / debug)
    input wire [3:0] ovr_read_addr1,
    input wire [3:0] ovr_read_addr2,
    input wire [3:0] ovr_read_addr3,
    input wire [3:0] ovr_read_addr4,
    output wire [15:0] ovr_read_data1,
    output wire [15:0] ovr_read_data2,
    output wire [15:0] ovr_read_data3,
    output wire [15:0] ovr_read_data4,

    // Write ports (unchanged)
    input wire [3:0] write_addr1,
    input wire [3:0] write_addr2,
    input wire [3:0] write_addr3,
    input wire [3:0] write_addr4,
    input wire [15:0] write_data1,
    input wire [15:0] write_data2,
    input wire [15:0] write_data3,
    input wire [15:0] write_data4,
    input wire write_en1,
    input wire write_en2,
    input wire write_en3,
    input wire write_en4
);

    reg [16*16-1:0] gpr_regs; // 16 registers × 16 bits, packed flat

    // Pipeline reads: pure async, no forwarding
    assign read_data1 = gpr_regs[read_addr1*16 +: 16];
    assign read_data2 = gpr_regs[read_addr2*16 +: 16];
    assign read_data3 = gpr_regs[read_addr3*16 +: 16];
    assign read_data4 = gpr_regs[read_addr4*16 +: 16];

    // Override reads: pure async, no forwarding
    assign ovr_read_data1 = gpr_regs[ovr_read_addr1*16 +: 16];
    assign ovr_read_data2 = gpr_regs[ovr_read_addr2*16 +: 16];
    assign ovr_read_data3 = gpr_regs[ovr_read_addr3*16 +: 16];
    assign ovr_read_data4 = gpr_regs[ovr_read_addr4*16 +: 16];

    // Synchronous write (priority: w1 > w2 > w3 > w4)
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gpr_regs <= {16*16{1'b0}};
        end else begin
            for (i = 0; i < 16; i = i + 1) begin
                if (write_en4 && write_addr4 == i[3:0]) gpr_regs[i*16 +: 16] <= write_data4;
                if (write_en3 && write_addr3 == i[3:0]) gpr_regs[i*16 +: 16] <= write_data3;
                if (write_en2 && write_addr2 == i[3:0]) gpr_regs[i*16 +: 16] <= write_data2;
                if (write_en1 && write_addr1 == i[3:0]) gpr_regs[i*16 +: 16] <= write_data1;
            end
        end
    end

endmodule

`endif // GPR_REGFILE_V