/* file: pred_regfile.v
 Description: This file implements the GPU predicate register regfile for the CUDA-like SM core.
 This design is a 4R4W register file design for one thread of an SM core, supporting 16 1-bit predicate registers.
 Register file support register forwarding to avoid data hazards in the pipeline.
 Author: Jeremy Cai
 Date: Feb. 28, 2026
 Version: 1.0
 Revision history:
    - Feb. 28, 2026: Initial implementation of the CUDA-like SM core predicate register file.
*/

`ifndef PRED_REGFILE_V
`define PRED_REGFILE_V

`include "gpu_define.v"

module pred_regfile (
    input wire clk,
    input wire rst_n,
    // Read port (combinational, for SELP/PBRA in ID stage)
    input wire [1:0] read_sel,    // P0-P3 select
    output wire read_val,    // predicate value
    // Write port (synchronous, from SETP in WB stage)
    input wire [1:0] write_sel,   // P0-P3 select
    input wire write_data,  // compare result bit
    input wire write_en     // from WB: valid & pred_we & active
);

    reg [3:0] pred;  // P0=pred[0], P1=pred[1], P2=pred[2], P3=pred[3]

    // Read with write-through forwarding
    assign read_val = (read_sel == write_sel && write_en) ? write_data
                                                          : pred[read_sel];

    // Synchronous write
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pred <= 4'b0;
        else if (write_en)
            pred[write_sel] <= write_data;
    end

endmodule

`endif // PRED_REGFILE_V