`ifndef XILINX_PRIM_MIN_V
`define XILINX_PRIM_MIN_V

`timescale 1 ns / 1 ps

module VCC(output P);
    assign P = 1'b1;
endmodule

module GND(output G);
    assign G = 1'b0;
endmodule

module INV(input I, output O);
    assign O = ~I;
endmodule

module LUT1 #(parameter [1:0] INIT = 2'h0)(input I0, output O);
    assign O = INIT[I0];
endmodule

module LUT2 #(parameter [3:0] INIT = 4'h0)(input I0, input I1, output O);
    assign O = INIT[{I1, I0}];
endmodule

module LUT3 #(parameter [7:0] INIT = 8'h00)(input I0, input I1, input I2, output O);
    assign O = INIT[{I2, I1, I0}];
endmodule

module LUT4 #(parameter [15:0] INIT = 16'h0000)(input I0, input I1, input I2, input I3, output O);
    assign O = INIT[{I3, I2, I1, I0}];
endmodule

module LUT2_L #(parameter [3:0] INIT = 4'h0)(input I0, input I1, output LO);
    assign LO = INIT[{I1, I0}];
endmodule

module LUT3_L #(parameter [7:0] INIT = 8'h00)(input I0, input I1, input I2, output LO);
    assign LO = INIT[{I2, I1, I0}];
endmodule

module LUT4_L #(parameter [15:0] INIT = 16'h0000)(input I0, input I1, input I2, input I3, output LO);
    assign LO = INIT[{I3, I2, I1, I0}];
endmodule

module LUT2_D #(parameter [3:0] INIT = 4'h0)(input I0, input I1, output LO, output O);
    wire y = INIT[{I1, I0}];
    assign LO = y;
    assign O = y;
endmodule

module LUT4_D #(parameter [15:0] INIT = 16'h0000)(input I0, input I1, input I2, input I3, output LO, output O);
    wire y = INIT[{I3, I2, I1, I0}];
    assign LO = y;
    assign O = y;
endmodule

module MUXCY(input CI, input DI, input S, output O);
    assign O = S ? CI : DI;
endmodule

module XORCY(input CI, input LI, output O);
    assign O = CI ^ LI;
endmodule

module MUXF5(input I0, input I1, input S, output O);
    assign O = S ? I1 : I0;
endmodule

module MULT18X18(input [17:0] A, input [17:0] B, output [35:0] P);
    assign P = A * B;
endmodule

module FD #(parameter INIT = 1'b0)(input C, input D, output reg Q);
    initial Q = INIT;
    always @(posedge C)
        Q <= D;
endmodule

module FDE #(parameter INIT = 1'b0)(input C, input CE, input D, output reg Q);
    initial Q = INIT;
    always @(posedge C)
        if (CE)
            Q <= D;
endmodule

module FDRS(input C, input D, input R, input S, output reg Q);
    initial Q = 1'b0;
    always @(posedge C) begin
        if (R)
            Q <= 1'b0;
        else if (S)
            Q <= 1'b1;
        else
            Q <= D;
    end
endmodule

`endif // XILINX_PRIM_MIN_V
