/* file: gpr_regfile.v
 Description: This file implements the GPU general-purpose register regfile for the CUDA-like SM core.
 This design is a 4R4W register file design for one thread of an SM core, supporting 16 16-bit registers.
 Register file support register forwarding to avoid data hazards in the pipeline.
 Author: Jeremy Cai
 Date: Feb. 26, 2026
 Version: 1.0
 Revision history:
    - Feb. 26, 2026: Initial implementation of the CUDA-like SM core general-purpose register file.
*/

`ifndef GPR_REGFILE_V
`define GPR_REGFILE_V

`include "gpu_define.v"

module gpr_regfile (
    input wire clk,
    input wire rst_n,
    input wire [3:0] read_addr1,
    input wire [3:0] read_addr2,
    input wire [3:0] read_addr3,
    input wire [3:0] read_addr4,
    output reg [15:0] read_data1,
    output reg [15:0] read_data2,
    output reg [15:0] read_data3,
    output reg [15:0] read_data4,
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
    reg [15:0] gpr_regs [15:0]; // 16 registers of 16 bits each
    
    // Register file read logic: combinational read with register forwarding
    always @(*) begin
        read_data1 = (read_addr1 == write_addr1 && write_en1) ? write_data1 :
                     (read_addr1 == write_addr2 && write_en2) ? write_data2 :
                     (read_addr1 == write_addr3 && write_en3) ? write_data3 :
                     (read_addr1 == write_addr4 && write_en4) ? write_data4 :
                     gpr_regs[read_addr1];

        read_data2 = (read_addr2 == write_addr1 && write_en1) ? write_data1 :
                     (read_addr2 == write_addr2 && write_en2) ? write_data2 :
                     (read_addr2 == write_addr3 && write_en3) ? write_data3 :
                     (read_addr2 == write_addr4 && write_en4) ? write_data4 :
                     gpr_regs[read_addr2];

        read_data3 = (read_addr3 == write_addr1 && write_en1) ? write_data1 :
                     (read_addr3 == write_addr2 && write_en2) ? write_data2 :
                     (read_addr3 == write_addr3 && write_en3) ? write_data3 :
                     (read_addr3 == write_addr4 && write_en4) ? write_data4 :
                     gpr_regs[read_addr3];

        read_data4 = (read_addr4 == write_addr1 && write_en1) ? write_data1 :
                     (read_addr4 == write_addr2 && write_en2) ? write_data2 :
                     (read_addr4 == write_addr3 && write_en3) ? write_data3 :
                     (read_addr4 == write_addr4 && write_en4) ? write_data4 :
                     gpr_regs[read_addr4];
    end

    integer i;
    // Register file write logic: synchronous write
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 16; i = i + 1) begin
                gpr_regs[i] <= 16'b0;
            end
        end else begin
            if (write_en1) gpr_regs[write_addr1] <= write_data1;
            if (write_en2) gpr_regs[write_addr2] <= write_data2;
            if (write_en3) gpr_regs[write_addr3] <= write_data3;
            if (write_en4) gpr_regs[write_addr4] <= write_data4;
        end
    end

endmodule

`endif // GPR_REGFILE_V
