////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 1995-2008 Xilinx, Inc.  All rights reserved.
////////////////////////////////////////////////////////////////////////////////
//   ____  ____
//  /   /\/   /
// /___/  \  /    Vendor: Xilinx
// \   \   \/     Version: K.39
//  \   \         Application: netgen
//  /   /         Filename: bf16mult.v
// /___/   /\     Timestamp: Thu Mar 05 14:54:12 2026
// \   \  /  \ 
//  \___\/\___\
//             
// Command	: -intstyle ise -w -sim -ofmt verilog "C:\Documents and Settings\student\Desktop\EE533\lab8\tmp\_cg\bf16mult.ngc" "C:\Documents and Settings\student\Desktop\EE533\lab8\tmp\_cg\bf16mult.v" 
// Device	: 2vp30ff896-6
// Input file	: C:/Documents and Settings/student/Desktop/EE533/lab8/tmp/_cg/bf16mult.ngc
// Output file	: C:/Documents and Settings/student/Desktop/EE533/lab8/tmp/_cg/bf16mult.v
// # of Modules	: 1
// Design Name	: bf16mult
// Xilinx        : C:\Xilinx\10.1\ISE
//             
// Purpose:    
//     This verilog netlist is a verification model and uses simulation 
//     primitives which may not represent the true implementation of the 
//     device, however the netlist is functionally correct and should not 
//     be modified. This file cannot be synthesized and should only be used 
//     with supported simulation tools.
//             
// Reference:  
//     Development System Reference Guide, Chapter 23 and Synthesis and Simulation Design Guide, Chapter 6
//             
////////////////////////////////////////////////////////////////////////////////

`timescale 1 ns/1 ps

module bf16mult (
  clk, a, b, result
);
  input clk;
  input [15 : 0] a;
  input [15 : 0] b;
  output [15 : 0] result;
  
  // synthesis translate_off
  
  wire sig00000001;
  wire sig00000002;
  wire sig00000003;
  wire sig00000004;
  wire sig00000005;
  wire sig00000006;
  wire sig00000007;
  wire sig00000008;
  wire sig00000009;
  wire sig0000000a;
  wire sig0000000b;
  wire sig0000000c;
  wire sig0000000d;
  wire sig0000000e;
  wire sig0000000f;
  wire sig00000010;
  wire sig00000011;
  wire sig00000012;
  wire sig00000013;
  wire sig00000014;
  wire sig00000015;
  wire sig00000016;
  wire sig00000017;
  wire sig00000018;
  wire sig00000019;
  wire sig0000001a;
  wire sig0000001b;
  wire sig0000001c;
  wire sig0000001d;
  wire sig0000001e;
  wire sig0000001f;
  wire sig00000020;
  wire sig00000021;
  wire sig00000022;
  wire sig00000023;
  wire sig00000024;
  wire sig00000025;
  wire sig00000026;
  wire sig00000027;
  wire sig00000028;
  wire sig00000029;
  wire sig0000002a;
  wire sig0000002b;
  wire sig0000002c;
  wire sig0000002d;
  wire sig0000002e;
  wire sig0000002f;
  wire sig00000030;
  wire sig00000031;
  wire \blk00000003/sig000000f8 ;
  wire \blk00000003/sig000000f7 ;
  wire \blk00000003/sig000000f6 ;
  wire \blk00000003/sig000000f5 ;
  wire \blk00000003/sig000000f4 ;
  wire \blk00000003/sig000000f3 ;
  wire \blk00000003/sig000000f2 ;
  wire \blk00000003/sig000000f1 ;
  wire \blk00000003/sig000000f0 ;
  wire \blk00000003/sig000000ef ;
  wire \blk00000003/sig000000ee ;
  wire \blk00000003/sig000000ed ;
  wire \blk00000003/sig000000ec ;
  wire \blk00000003/sig000000eb ;
  wire \blk00000003/sig000000ea ;
  wire \blk00000003/sig000000e9 ;
  wire \blk00000003/sig000000e8 ;
  wire \blk00000003/sig000000e7 ;
  wire \blk00000003/sig000000e6 ;
  wire \blk00000003/sig000000e5 ;
  wire \blk00000003/sig000000e4 ;
  wire \blk00000003/sig000000e3 ;
  wire \blk00000003/sig000000e2 ;
  wire \blk00000003/sig000000e1 ;
  wire \blk00000003/sig000000e0 ;
  wire \blk00000003/sig000000df ;
  wire \blk00000003/sig000000de ;
  wire \blk00000003/sig000000dd ;
  wire \blk00000003/sig000000dc ;
  wire \blk00000003/sig000000db ;
  wire \blk00000003/sig000000da ;
  wire \blk00000003/sig000000d9 ;
  wire \blk00000003/sig000000d8 ;
  wire \blk00000003/sig000000d7 ;
  wire \blk00000003/sig000000d6 ;
  wire \blk00000003/sig000000d5 ;
  wire \blk00000003/sig000000d4 ;
  wire \blk00000003/sig000000d3 ;
  wire \blk00000003/sig000000d2 ;
  wire \blk00000003/sig000000d1 ;
  wire \blk00000003/sig000000d0 ;
  wire \blk00000003/sig000000cf ;
  wire \blk00000003/sig000000ce ;
  wire \blk00000003/sig000000cd ;
  wire \blk00000003/sig000000cc ;
  wire \blk00000003/sig000000cb ;
  wire \blk00000003/sig000000ca ;
  wire \blk00000003/sig000000c9 ;
  wire \blk00000003/sig000000c8 ;
  wire \blk00000003/sig000000c7 ;
  wire \blk00000003/sig000000c6 ;
  wire \blk00000003/sig000000c5 ;
  wire \blk00000003/sig000000c4 ;
  wire \blk00000003/sig000000c3 ;
  wire \blk00000003/sig000000c2 ;
  wire \blk00000003/sig000000c1 ;
  wire \blk00000003/sig000000c0 ;
  wire \blk00000003/sig000000bf ;
  wire \blk00000003/sig000000be ;
  wire \blk00000003/sig000000bd ;
  wire \blk00000003/sig000000bc ;
  wire \blk00000003/sig000000bb ;
  wire \blk00000003/sig000000ba ;
  wire \blk00000003/sig000000b9 ;
  wire \blk00000003/sig000000b8 ;
  wire \blk00000003/sig000000b7 ;
  wire \blk00000003/sig000000b6 ;
  wire \blk00000003/sig000000b5 ;
  wire \blk00000003/sig000000b4 ;
  wire \blk00000003/sig000000b3 ;
  wire \blk00000003/sig000000b2 ;
  wire \blk00000003/sig000000b1 ;
  wire \blk00000003/sig000000b0 ;
  wire \blk00000003/sig000000af ;
  wire \blk00000003/sig000000ae ;
  wire \blk00000003/sig000000ad ;
  wire \blk00000003/sig000000ac ;
  wire \blk00000003/sig000000ab ;
  wire \blk00000003/sig000000aa ;
  wire \blk00000003/sig000000a9 ;
  wire \blk00000003/sig000000a8 ;
  wire \blk00000003/sig000000a7 ;
  wire \blk00000003/sig000000a6 ;
  wire \blk00000003/sig000000a5 ;
  wire \blk00000003/sig000000a4 ;
  wire \blk00000003/sig000000a3 ;
  wire \blk00000003/sig000000a2 ;
  wire \blk00000003/sig000000a1 ;
  wire \blk00000003/sig000000a0 ;
  wire \blk00000003/sig0000009f ;
  wire \blk00000003/sig0000009e ;
  wire \blk00000003/sig0000009d ;
  wire \blk00000003/sig0000009c ;
  wire \blk00000003/sig0000009b ;
  wire \blk00000003/sig0000009a ;
  wire \blk00000003/sig00000099 ;
  wire \blk00000003/sig00000098 ;
  wire \blk00000003/sig00000097 ;
  wire \blk00000003/sig00000096 ;
  wire \blk00000003/sig00000095 ;
  wire \blk00000003/sig00000094 ;
  wire \blk00000003/sig00000093 ;
  wire \blk00000003/sig00000092 ;
  wire \blk00000003/sig00000091 ;
  wire \blk00000003/sig00000090 ;
  wire \blk00000003/sig0000008f ;
  wire \blk00000003/sig0000008e ;
  wire \blk00000003/sig0000008d ;
  wire \blk00000003/sig0000008c ;
  wire \blk00000003/sig0000008b ;
  wire \blk00000003/sig0000008a ;
  wire \blk00000003/sig00000089 ;
  wire \blk00000003/sig00000088 ;
  wire \blk00000003/sig00000087 ;
  wire \blk00000003/sig00000086 ;
  wire \blk00000003/sig00000085 ;
  wire \blk00000003/sig00000084 ;
  wire \blk00000003/sig00000083 ;
  wire \blk00000003/sig00000082 ;
  wire \blk00000003/sig00000081 ;
  wire \blk00000003/sig00000080 ;
  wire \blk00000003/sig0000007f ;
  wire \blk00000003/sig0000007e ;
  wire \blk00000003/sig0000007d ;
  wire \blk00000003/sig0000007c ;
  wire \blk00000003/sig0000007b ;
  wire \blk00000003/sig0000007a ;
  wire \blk00000003/sig00000079 ;
  wire \blk00000003/sig00000078 ;
  wire \blk00000003/sig00000077 ;
  wire \blk00000003/sig00000076 ;
  wire \blk00000003/sig00000075 ;
  wire \blk00000003/sig00000074 ;
  wire \blk00000003/sig00000073 ;
  wire \blk00000003/sig00000072 ;
  wire \blk00000003/sig00000071 ;
  wire \blk00000003/sig00000070 ;
  wire \blk00000003/sig0000006f ;
  wire \blk00000003/sig0000006e ;
  wire \blk00000003/sig0000006d ;
  wire \blk00000003/sig0000006c ;
  wire \blk00000003/sig0000006b ;
  wire \blk00000003/sig0000006a ;
  wire \blk00000003/sig00000069 ;
  wire \blk00000003/sig00000068 ;
  wire \blk00000003/sig00000067 ;
  wire \blk00000003/sig00000066 ;
  wire \blk00000003/sig00000065 ;
  wire \blk00000003/sig00000034 ;
  wire \blk00000003/sig00000033 ;
  wire NLW_blk00000001_P_UNCONNECTED;
  wire NLW_blk00000002_G_UNCONNECTED;
  wire \NLW_blk00000003/blk0000003e_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk0000002d_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<35>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<34>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<33>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<32>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<31>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<30>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<29>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<28>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<27>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<26>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<25>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<24>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<23>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<22>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<21>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<20>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<19>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<18>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<17>_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_P<16>_UNCONNECTED ;
  assign
    sig00000001 = a[15],
    sig00000002 = a[14],
    sig00000003 = a[13],
    sig00000004 = a[12],
    sig00000005 = a[11],
    sig00000006 = a[10],
    sig00000007 = a[9],
    sig00000008 = a[8],
    sig00000009 = a[7],
    sig0000000a = a[6],
    sig0000000b = a[5],
    sig0000000c = a[4],
    sig0000000d = a[3],
    sig0000000e = a[2],
    sig0000000f = a[1],
    sig00000010 = a[0],
    sig00000011 = b[15],
    sig00000012 = b[14],
    sig00000013 = b[13],
    sig00000014 = b[12],
    sig00000015 = b[11],
    sig00000016 = b[10],
    sig00000017 = b[9],
    sig00000018 = b[8],
    sig00000019 = b[7],
    sig0000001a = b[6],
    sig0000001b = b[5],
    sig0000001c = b[4],
    sig0000001d = b[3],
    sig0000001e = b[2],
    sig0000001f = b[1],
    sig00000020 = b[0],
    result[15] = sig00000022,
    result[14] = sig00000023,
    result[13] = sig00000024,
    result[12] = sig00000025,
    result[11] = sig00000026,
    result[10] = sig00000027,
    result[9] = sig00000028,
    result[8] = sig00000029,
    result[7] = sig0000002a,
    result[6] = sig0000002b,
    result[5] = sig0000002c,
    result[4] = sig0000002d,
    result[3] = sig0000002e,
    result[2] = sig0000002f,
    result[1] = sig00000030,
    result[0] = sig00000031,
    sig00000021 = clk;
  VCC   blk00000001 (
    .P(NLW_blk00000001_P_UNCONNECTED)
  );
  GND   blk00000002 (
    .G(NLW_blk00000002_G_UNCONNECTED)
  );
  MUXF5   \blk00000003/blk0000009c  (
    .I0(\blk00000003/sig00000033 ),
    .I1(\blk00000003/sig000000f8 ),
    .S(\blk00000003/sig0000007b ),
    .O(\blk00000003/sig00000086 )
  );
  LUT4 #(
    .INIT ( 16'h3332 ))
  \blk00000003/blk0000009b  (
    .I0(\blk00000003/sig000000ea ),
    .I1(\blk00000003/sig00000087 ),
    .I2(\blk00000003/sig000000eb ),
    .I3(\blk00000003/sig00000065 ),
    .O(\blk00000003/sig000000f8 )
  );
  INV   \blk00000003/blk0000009a  (
    .I(\blk00000003/sig00000065 ),
    .O(\blk00000003/sig000000b4 )
  );
  INV   \blk00000003/blk00000099  (
    .I(\blk00000003/sig000000cf ),
    .O(\blk00000003/sig000000a4 )
  );
  LUT3 #(
    .INIT ( 8'h08 ))
  \blk00000003/blk00000098  (
    .I0(\blk00000003/sig000000e6 ),
    .I1(\blk00000003/sig000000c0 ),
    .I2(\blk00000003/sig000000c4 ),
    .O(\blk00000003/sig000000f7 )
  );
  LUT4 #(
    .INIT ( 16'hFF08 ))
  \blk00000003/blk00000097  (
    .I0(\blk00000003/sig000000c8 ),
    .I1(\blk00000003/sig000000cc ),
    .I2(\blk00000003/sig000000b8 ),
    .I3(\blk00000003/sig000000f3 ),
    .O(\blk00000003/sig000000f6 )
  );
  MUXF5   \blk00000003/blk00000096  (
    .I0(\blk00000003/sig000000f6 ),
    .I1(\blk00000003/sig000000f7 ),
    .S(\blk00000003/sig000000bc ),
    .O(\blk00000003/sig0000007b )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000095  (
    .I0(\blk00000003/sig000000d0 ),
    .I1(\blk00000003/sig000000e4 ),
    .I2(\blk00000003/sig000000e2 ),
    .I3(\blk00000003/sig000000d3 ),
    .O(\blk00000003/sig000000ef )
  );
  LUT4 #(
    .INIT ( 16'hFFFE ))
  \blk00000003/blk00000094  (
    .I0(\blk00000003/sig000000e2 ),
    .I1(\blk00000003/sig000000e4 ),
    .I2(\blk00000003/sig000000d0 ),
    .I3(\blk00000003/sig000000f5 ),
    .O(\blk00000003/sig000000e9 )
  );
  LUT4 #(
    .INIT ( 16'hFFFE ))
  \blk00000003/blk00000093  (
    .I0(\blk00000003/sig000000c4 ),
    .I1(\blk00000003/sig000000c8 ),
    .I2(\blk00000003/sig000000b8 ),
    .I3(\blk00000003/sig000000bc ),
    .O(\blk00000003/sig000000f5 )
  );
  LUT4 #(
    .INIT ( 16'hFFEA ))
  \blk00000003/blk00000092  (
    .I0(\blk00000003/sig000000bc ),
    .I1(\blk00000003/sig000000d0 ),
    .I2(\blk00000003/sig000000f4 ),
    .I3(\blk00000003/sig000000c8 ),
    .O(\blk00000003/sig00000087 )
  );
  LUT4 #(
    .INIT ( 16'h1110 ))
  \blk00000003/blk00000091  (
    .I0(\blk00000003/sig000000c4 ),
    .I1(\blk00000003/sig000000b8 ),
    .I2(\blk00000003/sig000000cf ),
    .I3(\blk00000003/sig000000e8 ),
    .O(\blk00000003/sig000000f4 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000090  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig0000006d ),
    .I2(\blk00000003/sig0000006c ),
    .O(\blk00000003/sig000000a7 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000008f  (
    .I0(\blk00000003/sig000000d3 ),
    .O(\blk00000003/sig000000a2 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000008e  (
    .I0(\blk00000003/sig000000d6 ),
    .O(\blk00000003/sig000000a0 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000008d  (
    .I0(\blk00000003/sig000000d9 ),
    .O(\blk00000003/sig0000009e )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000008c  (
    .I0(\blk00000003/sig000000dc ),
    .O(\blk00000003/sig0000009c )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000008b  (
    .I0(\blk00000003/sig000000df ),
    .O(\blk00000003/sig0000009a )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000008a  (
    .I0(\blk00000003/sig000000e2 ),
    .O(\blk00000003/sig00000098 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk00000089  (
    .I0(\blk00000003/sig000000e4 ),
    .O(\blk00000003/sig00000096 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000088  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig0000006d ),
    .I2(\blk00000003/sig0000006c ),
    .O(\blk00000003/sig000000a5 )
  );
  LUT3 #(
    .INIT ( 8'h15 ))
  \blk00000003/blk00000087  (
    .I0(\blk00000003/sig00000078 ),
    .I1(\blk00000003/sig00000065 ),
    .I2(\blk00000003/sig0000006e ),
    .O(\blk00000003/sig00000093 )
  );
  LUT4 #(
    .INIT ( 16'h3332 ))
  \blk00000003/blk00000086  (
    .I0(\blk00000003/sig000000f1 ),
    .I1(\blk00000003/sig000000c8 ),
    .I2(\blk00000003/sig000000f2 ),
    .I3(\blk00000003/sig000000f0 ),
    .O(\blk00000003/sig000000f3 )
  );
  LUT4 #(
    .INIT ( 16'h3111 ))
  \blk00000003/blk00000085  (
    .I0(\blk00000003/sig000000d0 ),
    .I1(\blk00000003/sig000000cf ),
    .I2(\blk00000003/sig000000e7 ),
    .I3(\blk00000003/sig00000065 ),
    .O(\blk00000003/sig000000f2 )
  );
  LUT4 #(
    .INIT ( 16'hAA80 ))
  \blk00000003/blk00000084  (
    .I0(\blk00000003/sig000000d0 ),
    .I1(\blk00000003/sig000000e4 ),
    .I2(\blk00000003/sig000000e7 ),
    .I3(\blk00000003/sig000000cf ),
    .O(\blk00000003/sig000000f1 )
  );
  LUT4 #(
    .INIT ( 16'hFF08 ))
  \blk00000003/blk00000083  (
    .I0(\blk00000003/sig000000ef ),
    .I1(\blk00000003/sig000000ee ),
    .I2(\blk00000003/sig00000065 ),
    .I3(\blk00000003/sig000000ed ),
    .O(\blk00000003/sig000000f0 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000082  (
    .I0(\blk00000003/sig000000d6 ),
    .I1(\blk00000003/sig000000d9 ),
    .I2(\blk00000003/sig000000dc ),
    .I3(\blk00000003/sig000000df ),
    .O(\blk00000003/sig000000ee )
  );
  LUT2 #(
    .INIT ( 4'hE ))
  \blk00000003/blk00000081  (
    .I0(\blk00000003/sig000000b8 ),
    .I1(\blk00000003/sig000000c4 ),
    .O(\blk00000003/sig000000ed )
  );
  LUT2 #(
    .INIT ( 4'hE ))
  \blk00000003/blk00000080  (
    .I0(\blk00000003/sig0000007b ),
    .I1(\blk00000003/sig00000087 ),
    .O(\blk00000003/sig0000007e )
  );
  LUT2 #(
    .INIT ( 4'h2 ))
  \blk00000003/blk0000007f  (
    .I0(\blk00000003/sig00000087 ),
    .I1(\blk00000003/sig0000007b ),
    .O(\blk00000003/sig0000007c )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk0000007e  (
    .I0(\blk00000003/sig000000d3 ),
    .I1(\blk00000003/sig000000d6 ),
    .I2(\blk00000003/sig000000d9 ),
    .I3(\blk00000003/sig000000ec ),
    .O(\blk00000003/sig000000e7 )
  );
  LUT3 #(
    .INIT ( 8'h80 ))
  \blk00000003/blk0000007d  (
    .I0(\blk00000003/sig000000dc ),
    .I1(\blk00000003/sig000000df ),
    .I2(\blk00000003/sig000000e2 ),
    .O(\blk00000003/sig000000ec )
  );
  LUT3 #(
    .INIT ( 8'hFB ))
  \blk00000003/blk0000007c  (
    .I0(\blk00000003/sig000000df ),
    .I1(\blk00000003/sig000000cf ),
    .I2(\blk00000003/sig000000dc ),
    .O(\blk00000003/sig000000eb )
  );
  LUT4 #(
    .INIT ( 16'hFFFE ))
  \blk00000003/blk0000007b  (
    .I0(\blk00000003/sig000000d6 ),
    .I1(\blk00000003/sig000000d9 ),
    .I2(\blk00000003/sig000000d3 ),
    .I3(\blk00000003/sig000000e9 ),
    .O(\blk00000003/sig000000ea )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk0000007a  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig0000006d ),
    .I2(\blk00000003/sig0000006c ),
    .O(\blk00000003/sig00000094 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000079  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig0000006e ),
    .I2(\blk00000003/sig0000006d ),
    .O(\blk00000003/sig00000090 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000078  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig0000006c ),
    .I2(\blk00000003/sig0000006b ),
    .O(\blk00000003/sig000000a8 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000077  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig0000006b ),
    .I2(\blk00000003/sig0000006a ),
    .O(\blk00000003/sig000000aa )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000076  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig0000006a ),
    .I2(\blk00000003/sig00000069 ),
    .O(\blk00000003/sig000000ac )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000075  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig00000069 ),
    .I2(\blk00000003/sig00000068 ),
    .O(\blk00000003/sig000000ae )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000074  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig00000068 ),
    .I2(\blk00000003/sig00000067 ),
    .O(\blk00000003/sig000000b0 )
  );
  LUT3 #(
    .INIT ( 8'hA8 ))
  \blk00000003/blk00000073  (
    .I0(\blk00000003/sig000000e7 ),
    .I1(\blk00000003/sig000000e4 ),
    .I2(\blk00000003/sig00000065 ),
    .O(\blk00000003/sig000000e8 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000072  (
    .I0(\blk00000003/sig00000065 ),
    .I1(\blk00000003/sig00000067 ),
    .I2(\blk00000003/sig00000066 ),
    .O(\blk00000003/sig000000b2 )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk00000071  (
    .I0(sig00000008),
    .I1(sig00000009),
    .I2(sig00000006),
    .I3(sig00000007),
    .O(\blk00000003/sig000000b9 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000070  (
    .I0(sig00000008),
    .I1(sig00000009),
    .I2(sig00000006),
    .I3(sig00000007),
    .O(\blk00000003/sig000000b5 )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk0000006f  (
    .I0(sig00000018),
    .I1(sig00000019),
    .I2(sig00000016),
    .I3(sig00000017),
    .O(\blk00000003/sig000000c5 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk0000006e  (
    .I0(sig00000018),
    .I1(sig00000019),
    .I2(sig00000016),
    .I3(sig00000017),
    .O(\blk00000003/sig000000c1 )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk0000006d  (
    .I0(sig00000004),
    .I1(sig00000005),
    .I2(sig00000002),
    .I3(sig00000003),
    .O(\blk00000003/sig000000bb )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk0000006c  (
    .I0(sig00000004),
    .I1(sig00000005),
    .I2(sig00000002),
    .I3(sig00000003),
    .O(\blk00000003/sig000000b7 )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk0000006b  (
    .I0(sig00000014),
    .I1(sig00000015),
    .I2(sig00000012),
    .I3(sig00000013),
    .O(\blk00000003/sig000000c7 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk0000006a  (
    .I0(sig00000014),
    .I1(sig00000015),
    .I2(sig00000012),
    .I3(sig00000013),
    .O(\blk00000003/sig000000c3 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000069  (
    .I0(sig0000000f),
    .I1(sig00000010),
    .I2(sig0000000d),
    .I3(sig0000000e),
    .O(\blk00000003/sig000000bd )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000068  (
    .I0(sig0000001f),
    .I1(sig00000020),
    .I2(sig0000001d),
    .I3(sig0000001e),
    .O(\blk00000003/sig000000c9 )
  );
  LUT3 #(
    .INIT ( 8'h4F ))
  \blk00000003/blk00000067  (
    .I0(\blk00000003/sig000000b8 ),
    .I1(\blk00000003/sig000000cc ),
    .I2(\blk00000003/sig000000c8 ),
    .O(\blk00000003/sig000000e6 )
  );
  LUT3 #(
    .INIT ( 8'h01 ))
  \blk00000003/blk00000066  (
    .I0(sig0000000a),
    .I1(sig0000000b),
    .I2(sig0000000c),
    .O(\blk00000003/sig000000bf )
  );
  LUT3 #(
    .INIT ( 8'h01 ))
  \blk00000003/blk00000065  (
    .I0(sig0000001a),
    .I1(sig0000001b),
    .I2(sig0000001c),
    .O(\blk00000003/sig000000cb )
  );
  LUT4 #(
    .INIT ( 16'h0028 ))
  \blk00000003/blk00000064  (
    .I0(\blk00000003/sig000000e6 ),
    .I1(sig00000011),
    .I2(sig00000001),
    .I3(\blk00000003/sig000000e5 ),
    .O(\blk00000003/sig0000007f )
  );
  LUT3 #(
    .INIT ( 8'hA2 ))
  \blk00000003/blk00000063  (
    .I0(\blk00000003/sig000000bc ),
    .I1(\blk00000003/sig000000c0 ),
    .I2(\blk00000003/sig000000c4 ),
    .O(\blk00000003/sig000000e5 )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk00000062  (
    .I0(sig00000009),
    .I1(sig00000019),
    .O(\blk00000003/sig000000e3 )
  );
  MUXCY   \blk00000003/blk00000061  (
    .CI(\blk00000003/sig00000034 ),
    .DI(sig00000009),
    .S(\blk00000003/sig000000e3 ),
    .O(\blk00000003/sig000000e0 )
  );
  XORCY   \blk00000003/blk00000060  (
    .CI(\blk00000003/sig00000034 ),
    .LI(\blk00000003/sig000000e3 ),
    .O(\blk00000003/sig000000e4 )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk0000005f  (
    .I0(sig00000008),
    .I1(sig00000018),
    .O(\blk00000003/sig000000e1 )
  );
  MUXCY   \blk00000003/blk0000005e  (
    .CI(\blk00000003/sig000000e0 ),
    .DI(sig00000008),
    .S(\blk00000003/sig000000e1 ),
    .O(\blk00000003/sig000000dd )
  );
  XORCY   \blk00000003/blk0000005d  (
    .CI(\blk00000003/sig000000e0 ),
    .LI(\blk00000003/sig000000e1 ),
    .O(\blk00000003/sig000000e2 )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk0000005c  (
    .I0(sig00000007),
    .I1(sig00000017),
    .O(\blk00000003/sig000000de )
  );
  MUXCY   \blk00000003/blk0000005b  (
    .CI(\blk00000003/sig000000dd ),
    .DI(sig00000007),
    .S(\blk00000003/sig000000de ),
    .O(\blk00000003/sig000000da )
  );
  XORCY   \blk00000003/blk0000005a  (
    .CI(\blk00000003/sig000000dd ),
    .LI(\blk00000003/sig000000de ),
    .O(\blk00000003/sig000000df )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk00000059  (
    .I0(sig00000006),
    .I1(sig00000016),
    .O(\blk00000003/sig000000db )
  );
  MUXCY   \blk00000003/blk00000058  (
    .CI(\blk00000003/sig000000da ),
    .DI(sig00000006),
    .S(\blk00000003/sig000000db ),
    .O(\blk00000003/sig000000d7 )
  );
  XORCY   \blk00000003/blk00000057  (
    .CI(\blk00000003/sig000000da ),
    .LI(\blk00000003/sig000000db ),
    .O(\blk00000003/sig000000dc )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk00000056  (
    .I0(sig00000005),
    .I1(sig00000015),
    .O(\blk00000003/sig000000d8 )
  );
  MUXCY   \blk00000003/blk00000055  (
    .CI(\blk00000003/sig000000d7 ),
    .DI(sig00000005),
    .S(\blk00000003/sig000000d8 ),
    .O(\blk00000003/sig000000d4 )
  );
  XORCY   \blk00000003/blk00000054  (
    .CI(\blk00000003/sig000000d7 ),
    .LI(\blk00000003/sig000000d8 ),
    .O(\blk00000003/sig000000d9 )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk00000053  (
    .I0(sig00000004),
    .I1(sig00000014),
    .O(\blk00000003/sig000000d5 )
  );
  MUXCY   \blk00000003/blk00000052  (
    .CI(\blk00000003/sig000000d4 ),
    .DI(sig00000004),
    .S(\blk00000003/sig000000d5 ),
    .O(\blk00000003/sig000000d1 )
  );
  XORCY   \blk00000003/blk00000051  (
    .CI(\blk00000003/sig000000d4 ),
    .LI(\blk00000003/sig000000d5 ),
    .O(\blk00000003/sig000000d6 )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk00000050  (
    .I0(sig00000003),
    .I1(sig00000013),
    .O(\blk00000003/sig000000d2 )
  );
  MUXCY   \blk00000003/blk0000004f  (
    .CI(\blk00000003/sig000000d1 ),
    .DI(sig00000003),
    .S(\blk00000003/sig000000d2 ),
    .O(\blk00000003/sig000000cd )
  );
  XORCY   \blk00000003/blk0000004e  (
    .CI(\blk00000003/sig000000d1 ),
    .LI(\blk00000003/sig000000d2 ),
    .O(\blk00000003/sig000000d3 )
  );
  LUT2 #(
    .INIT ( 4'h6 ))
  \blk00000003/blk0000004d  (
    .I0(sig00000002),
    .I1(sig00000012),
    .O(\blk00000003/sig000000ce )
  );
  MUXCY   \blk00000003/blk0000004c  (
    .CI(\blk00000003/sig000000cd ),
    .DI(sig00000002),
    .S(\blk00000003/sig000000ce ),
    .O(\blk00000003/sig000000d0 )
  );
  XORCY   \blk00000003/blk0000004b  (
    .CI(\blk00000003/sig000000cd ),
    .LI(\blk00000003/sig000000ce ),
    .O(\blk00000003/sig000000cf )
  );
  MUXCY   \blk00000003/blk0000004a  (
    .CI(\blk00000003/sig000000ca ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000cb ),
    .O(\blk00000003/sig000000cc )
  );
  MUXCY   \blk00000003/blk00000049  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000c9 ),
    .O(\blk00000003/sig000000ca )
  );
  MUXCY   \blk00000003/blk00000048  (
    .CI(\blk00000003/sig000000c6 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000c7 ),
    .O(\blk00000003/sig000000c8 )
  );
  MUXCY   \blk00000003/blk00000047  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000c5 ),
    .O(\blk00000003/sig000000c6 )
  );
  MUXCY   \blk00000003/blk00000046  (
    .CI(\blk00000003/sig000000c2 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000c3 ),
    .O(\blk00000003/sig000000c4 )
  );
  MUXCY   \blk00000003/blk00000045  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000c1 ),
    .O(\blk00000003/sig000000c2 )
  );
  MUXCY   \blk00000003/blk00000044  (
    .CI(\blk00000003/sig000000be ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000bf ),
    .O(\blk00000003/sig000000c0 )
  );
  MUXCY   \blk00000003/blk00000043  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000bd ),
    .O(\blk00000003/sig000000be )
  );
  MUXCY   \blk00000003/blk00000042  (
    .CI(\blk00000003/sig000000ba ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000bb ),
    .O(\blk00000003/sig000000bc )
  );
  MUXCY   \blk00000003/blk00000041  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000b9 ),
    .O(\blk00000003/sig000000ba )
  );
  MUXCY   \blk00000003/blk00000040  (
    .CI(\blk00000003/sig000000b6 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000b7 ),
    .O(\blk00000003/sig000000b8 )
  );
  MUXCY   \blk00000003/blk0000003f  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000b5 ),
    .O(\blk00000003/sig000000b6 )
  );
  XORCY   \blk00000003/blk0000003e  (
    .CI(\blk00000003/sig000000b3 ),
    .LI(\blk00000003/sig000000b4 ),
    .O(\NLW_blk00000003/blk0000003e_O_UNCONNECTED )
  );
  MUXCY   \blk00000003/blk0000003d  (
    .CI(\blk00000003/sig000000b3 ),
    .DI(\blk00000003/sig00000034 ),
    .S(\blk00000003/sig000000b4 ),
    .O(\blk00000003/sig00000095 )
  );
  XORCY   \blk00000003/blk0000003c  (
    .CI(\blk00000003/sig000000b1 ),
    .LI(\blk00000003/sig000000b2 ),
    .O(\blk00000003/sig0000007a )
  );
  MUXCY   \blk00000003/blk0000003b  (
    .CI(\blk00000003/sig000000b1 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000b2 ),
    .O(\blk00000003/sig000000b3 )
  );
  XORCY   \blk00000003/blk0000003a  (
    .CI(\blk00000003/sig000000af ),
    .LI(\blk00000003/sig000000b0 ),
    .O(\blk00000003/sig0000007d )
  );
  MUXCY   \blk00000003/blk00000039  (
    .CI(\blk00000003/sig000000af ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000b0 ),
    .O(\blk00000003/sig000000b1 )
  );
  XORCY   \blk00000003/blk00000038  (
    .CI(\blk00000003/sig000000ad ),
    .LI(\blk00000003/sig000000ae ),
    .O(\blk00000003/sig00000082 )
  );
  MUXCY   \blk00000003/blk00000037  (
    .CI(\blk00000003/sig000000ad ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000ae ),
    .O(\blk00000003/sig000000af )
  );
  XORCY   \blk00000003/blk00000036  (
    .CI(\blk00000003/sig000000ab ),
    .LI(\blk00000003/sig000000ac ),
    .O(\blk00000003/sig00000080 )
  );
  MUXCY   \blk00000003/blk00000035  (
    .CI(\blk00000003/sig000000ab ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000ac ),
    .O(\blk00000003/sig000000ad )
  );
  XORCY   \blk00000003/blk00000034  (
    .CI(\blk00000003/sig000000a9 ),
    .LI(\blk00000003/sig000000aa ),
    .O(\blk00000003/sig00000081 )
  );
  MUXCY   \blk00000003/blk00000033  (
    .CI(\blk00000003/sig000000a9 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000aa ),
    .O(\blk00000003/sig000000ab )
  );
  XORCY   \blk00000003/blk00000032  (
    .CI(\blk00000003/sig000000a6 ),
    .LI(\blk00000003/sig000000a8 ),
    .O(\blk00000003/sig00000084 )
  );
  MUXCY   \blk00000003/blk00000031  (
    .CI(\blk00000003/sig000000a6 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000a8 ),
    .O(\blk00000003/sig000000a9 )
  );
  XORCY   \blk00000003/blk00000030  (
    .CI(\blk00000003/sig00000091 ),
    .LI(\blk00000003/sig000000a7 ),
    .O(\blk00000003/sig00000083 )
  );
  MUXCY   \blk00000003/blk0000002f  (
    .CI(\blk00000003/sig00000091 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000a5 ),
    .O(\blk00000003/sig000000a6 )
  );
  XORCY   \blk00000003/blk0000002e  (
    .CI(\blk00000003/sig000000a3 ),
    .LI(\blk00000003/sig000000a4 ),
    .O(\blk00000003/sig00000085 )
  );
  MUXCY   \blk00000003/blk0000002d  (
    .CI(\blk00000003/sig000000a3 ),
    .DI(\blk00000003/sig00000034 ),
    .S(\blk00000003/sig000000a4 ),
    .O(\NLW_blk00000003/blk0000002d_O_UNCONNECTED )
  );
  XORCY   \blk00000003/blk0000002c  (
    .CI(\blk00000003/sig000000a1 ),
    .LI(\blk00000003/sig000000a2 ),
    .O(\blk00000003/sig00000088 )
  );
  MUXCY   \blk00000003/blk0000002b  (
    .CI(\blk00000003/sig000000a1 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000a2 ),
    .O(\blk00000003/sig000000a3 )
  );
  XORCY   \blk00000003/blk0000002a  (
    .CI(\blk00000003/sig0000009f ),
    .LI(\blk00000003/sig000000a0 ),
    .O(\blk00000003/sig00000089 )
  );
  MUXCY   \blk00000003/blk00000029  (
    .CI(\blk00000003/sig0000009f ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig000000a0 ),
    .O(\blk00000003/sig000000a1 )
  );
  XORCY   \blk00000003/blk00000028  (
    .CI(\blk00000003/sig0000009d ),
    .LI(\blk00000003/sig0000009e ),
    .O(\blk00000003/sig0000008a )
  );
  MUXCY   \blk00000003/blk00000027  (
    .CI(\blk00000003/sig0000009d ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig0000009e ),
    .O(\blk00000003/sig0000009f )
  );
  XORCY   \blk00000003/blk00000026  (
    .CI(\blk00000003/sig0000009b ),
    .LI(\blk00000003/sig0000009c ),
    .O(\blk00000003/sig0000008b )
  );
  MUXCY   \blk00000003/blk00000025  (
    .CI(\blk00000003/sig0000009b ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig0000009c ),
    .O(\blk00000003/sig0000009d )
  );
  XORCY   \blk00000003/blk00000024  (
    .CI(\blk00000003/sig00000099 ),
    .LI(\blk00000003/sig0000009a ),
    .O(\blk00000003/sig0000008c )
  );
  MUXCY   \blk00000003/blk00000023  (
    .CI(\blk00000003/sig00000099 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig0000009a ),
    .O(\blk00000003/sig0000009b )
  );
  XORCY   \blk00000003/blk00000022  (
    .CI(\blk00000003/sig00000097 ),
    .LI(\blk00000003/sig00000098 ),
    .O(\blk00000003/sig0000008d )
  );
  MUXCY   \blk00000003/blk00000021  (
    .CI(\blk00000003/sig00000097 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig00000098 ),
    .O(\blk00000003/sig00000099 )
  );
  XORCY   \blk00000003/blk00000020  (
    .CI(\blk00000003/sig00000095 ),
    .LI(\blk00000003/sig00000096 ),
    .O(\blk00000003/sig0000008e )
  );
  MUXCY   \blk00000003/blk0000001f  (
    .CI(\blk00000003/sig00000095 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig00000096 ),
    .O(\blk00000003/sig00000097 )
  );
  MUXCY   \blk00000003/blk0000001e  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig00000094 ),
    .O(\blk00000003/sig00000092 )
  );
  MUXCY   \blk00000003/blk0000001d  (
    .CI(\blk00000003/sig00000092 ),
    .DI(\blk00000003/sig00000034 ),
    .S(\blk00000003/sig00000093 ),
    .O(\blk00000003/sig0000008f )
  );
  MUXCY   \blk00000003/blk0000001c  (
    .CI(\blk00000003/sig0000008f ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig00000090 ),
    .O(\blk00000003/sig00000091 )
  );
  FDRS   \blk00000003/blk0000001b  (
    .C(sig00000021),
    .D(\blk00000003/sig0000008e ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig0000002a)
  );
  FDRS   \blk00000003/blk0000001a  (
    .C(sig00000021),
    .D(\blk00000003/sig0000008d ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig00000029)
  );
  FDRS   \blk00000003/blk00000019  (
    .C(sig00000021),
    .D(\blk00000003/sig0000008c ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig00000028)
  );
  FDRS   \blk00000003/blk00000018  (
    .C(sig00000021),
    .D(\blk00000003/sig0000008b ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig00000027)
  );
  FDRS   \blk00000003/blk00000017  (
    .C(sig00000021),
    .D(\blk00000003/sig0000008a ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig00000026)
  );
  FDRS   \blk00000003/blk00000016  (
    .C(sig00000021),
    .D(\blk00000003/sig00000089 ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig00000025)
  );
  FDRS   \blk00000003/blk00000015  (
    .C(sig00000021),
    .D(\blk00000003/sig00000088 ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig00000024)
  );
  FDRS   \blk00000003/blk00000014  (
    .C(sig00000021),
    .D(\blk00000003/sig00000085 ),
    .R(\blk00000003/sig00000086 ),
    .S(\blk00000003/sig00000087 ),
    .Q(sig00000023)
  );
  FDRS   \blk00000003/blk00000013  (
    .C(sig00000021),
    .D(\blk00000003/sig00000084 ),
    .R(\blk00000003/sig0000007e ),
    .S(\blk00000003/sig00000033 ),
    .Q(sig00000030)
  );
  FDRS   \blk00000003/blk00000012  (
    .C(sig00000021),
    .D(\blk00000003/sig00000083 ),
    .R(\blk00000003/sig0000007e ),
    .S(\blk00000003/sig00000033 ),
    .Q(sig00000031)
  );
  FDRS   \blk00000003/blk00000011  (
    .C(sig00000021),
    .D(\blk00000003/sig00000082 ),
    .R(\blk00000003/sig0000007e ),
    .S(\blk00000003/sig00000033 ),
    .Q(sig0000002d)
  );
  FDRS   \blk00000003/blk00000010  (
    .C(sig00000021),
    .D(\blk00000003/sig00000081 ),
    .R(\blk00000003/sig0000007e ),
    .S(\blk00000003/sig00000033 ),
    .Q(sig0000002f)
  );
  FDRS   \blk00000003/blk0000000f  (
    .C(sig00000021),
    .D(\blk00000003/sig00000080 ),
    .R(\blk00000003/sig0000007e ),
    .S(\blk00000003/sig00000033 ),
    .Q(sig0000002e)
  );
  FDRS   \blk00000003/blk0000000e  (
    .C(sig00000021),
    .D(\blk00000003/sig0000007f ),
    .R(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig00000033 ),
    .Q(sig00000022)
  );
  FDRS   \blk00000003/blk0000000d  (
    .C(sig00000021),
    .D(\blk00000003/sig0000007d ),
    .R(\blk00000003/sig0000007e ),
    .S(\blk00000003/sig00000033 ),
    .Q(sig0000002c)
  );
  FDRS   \blk00000003/blk0000000c  (
    .C(sig00000021),
    .D(\blk00000003/sig0000007a ),
    .R(\blk00000003/sig0000007b ),
    .S(\blk00000003/sig0000007c ),
    .Q(sig0000002b)
  );
  MUXCY   \blk00000003/blk0000000b  (
    .CI(\blk00000003/sig00000034 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig00000076 ),
    .O(\blk00000003/sig00000079 )
  );
  MUXCY   \blk00000003/blk0000000a  (
    .CI(\blk00000003/sig00000079 ),
    .DI(\blk00000003/sig00000033 ),
    .S(\blk00000003/sig00000075 ),
    .O(\blk00000003/sig00000077 )
  );
  XORCY   \blk00000003/blk00000009  (
    .CI(\blk00000003/sig00000077 ),
    .LI(\blk00000003/sig00000034 ),
    .O(\blk00000003/sig00000078 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000008  (
    .I0(\blk00000003/sig00000074 ),
    .I1(\blk00000003/sig00000073 ),
    .I2(\blk00000003/sig00000072 ),
    .I3(\blk00000003/sig00000071 ),
    .O(\blk00000003/sig00000076 )
  );
  LUT2 #(
    .INIT ( 4'h1 ))
  \blk00000003/blk00000007  (
    .I0(\blk00000003/sig00000070 ),
    .I1(\blk00000003/sig0000006f ),
    .O(\blk00000003/sig00000075 )
  );
  MULT18X18   \blk00000003/blk00000006  (
    .A({\blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , 
\blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , 
\blk00000003/sig00000034 , sig0000000a, sig0000000b, sig0000000c, sig0000000d, sig0000000e, sig0000000f, sig00000010}),
    .B({\blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , 
\blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , \blk00000003/sig00000033 , 
\blk00000003/sig00000034 , sig0000001a, sig0000001b, sig0000001c, sig0000001d, sig0000001e, sig0000001f, sig00000020}),
    .P({\NLW_blk00000003/blk00000006_P<35>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<34>_UNCONNECTED , 
\NLW_blk00000003/blk00000006_P<33>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<32>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<31>_UNCONNECTED , 
\NLW_blk00000003/blk00000006_P<30>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<29>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<28>_UNCONNECTED , 
\NLW_blk00000003/blk00000006_P<27>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<26>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<25>_UNCONNECTED , 
\NLW_blk00000003/blk00000006_P<24>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<23>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<22>_UNCONNECTED , 
\NLW_blk00000003/blk00000006_P<21>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<20>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<19>_UNCONNECTED , 
\NLW_blk00000003/blk00000006_P<18>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<17>_UNCONNECTED , \NLW_blk00000003/blk00000006_P<16>_UNCONNECTED , 
\blk00000003/sig00000065 , \blk00000003/sig00000066 , \blk00000003/sig00000067 , \blk00000003/sig00000068 , \blk00000003/sig00000069 , 
\blk00000003/sig0000006a , \blk00000003/sig0000006b , \blk00000003/sig0000006c , \blk00000003/sig0000006d , \blk00000003/sig0000006e , 
\blk00000003/sig0000006f , \blk00000003/sig00000070 , \blk00000003/sig00000071 , \blk00000003/sig00000072 , \blk00000003/sig00000073 , 
\blk00000003/sig00000074 })
  );
  VCC   \blk00000003/blk00000005  (
    .P(\blk00000003/sig00000034 )
  );
  GND   \blk00000003/blk00000004  (
    .G(\blk00000003/sig00000033 )
  );

// synthesis translate_on

endmodule

// synthesis translate_off

`timescale  1 ps / 1 ps
