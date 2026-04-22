////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 1995-2008 Xilinx, Inc.  All rights reserved.
////////////////////////////////////////////////////////////////////////////////
//   ____  ____
//  /   /\/   /
// /___/  \  /    Vendor: Xilinx
// \   \   \/     Version: K.39
//  \   \         Application: netgen
//  /   /         Filename: bf16addsub.v
// /___/   /\     Timestamp: Thu Mar 05 14:48:27 2026
// \   \  /  \ 
//  \___\/\___\
//             
// Command	: -intstyle ise -w -sim -ofmt verilog "C:\Documents and Settings\student\Desktop\EE533\lab8\tmp\_cg\bf16addsub.ngc" "C:\Documents and Settings\student\Desktop\EE533\lab8\tmp\_cg\bf16addsub.v" 
// Device	: 2vp50ff1152-7
// Input file	: C:/Documents and Settings/student/Desktop/EE533/lab8/tmp/_cg/bf16addsub.ngc
// Output file	: C:/Documents and Settings/student/Desktop/EE533/lab8/tmp/_cg/bf16addsub.v
// # of Modules	: 1
// Design Name	: bf16addsub
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

module bf16addsub (
  clk, operation, a, b, result
);
  input clk;
  input [5 : 0] operation;
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
  wire sig00000032;
  wire sig00000033;
  wire sig00000034;
  wire sig00000035;
  wire sig00000036;
  wire sig00000037;
  wire \blk00000003/sig00000214 ;
  wire \blk00000003/sig00000213 ;
  wire \blk00000003/sig00000212 ;
  wire \blk00000003/sig00000211 ;
  wire \blk00000003/sig00000210 ;
  wire \blk00000003/sig0000020f ;
  wire \blk00000003/sig0000020e ;
  wire \blk00000003/sig0000020d ;
  wire \blk00000003/sig0000020c ;
  wire \blk00000003/sig0000020b ;
  wire \blk00000003/sig0000020a ;
  wire \blk00000003/sig00000209 ;
  wire \blk00000003/sig00000208 ;
  wire \blk00000003/sig00000207 ;
  wire \blk00000003/sig00000206 ;
  wire \blk00000003/sig00000205 ;
  wire \blk00000003/sig00000204 ;
  wire \blk00000003/sig00000203 ;
  wire \blk00000003/sig00000202 ;
  wire \blk00000003/sig00000201 ;
  wire \blk00000003/sig00000200 ;
  wire \blk00000003/sig000001ff ;
  wire \blk00000003/sig000001fe ;
  wire \blk00000003/sig000001fd ;
  wire \blk00000003/sig000001fc ;
  wire \blk00000003/sig000001fb ;
  wire \blk00000003/sig000001fa ;
  wire \blk00000003/sig000001f9 ;
  wire \blk00000003/sig000001f8 ;
  wire \blk00000003/sig000001f7 ;
  wire \blk00000003/sig000001f6 ;
  wire \blk00000003/sig000001f5 ;
  wire \blk00000003/sig000001f4 ;
  wire \blk00000003/sig000001f3 ;
  wire \blk00000003/sig000001f2 ;
  wire \blk00000003/sig000001f1 ;
  wire \blk00000003/sig000001f0 ;
  wire \blk00000003/sig000001ef ;
  wire \blk00000003/sig000001ee ;
  wire \blk00000003/sig000001ed ;
  wire \blk00000003/sig000001ec ;
  wire \blk00000003/sig000001eb ;
  wire \blk00000003/sig000001ea ;
  wire \blk00000003/sig000001e9 ;
  wire \blk00000003/sig000001e8 ;
  wire \blk00000003/sig000001e7 ;
  wire \blk00000003/sig000001e6 ;
  wire \blk00000003/sig000001e5 ;
  wire \blk00000003/sig000001e4 ;
  wire \blk00000003/sig000001e3 ;
  wire \blk00000003/sig000001e2 ;
  wire \blk00000003/sig000001e1 ;
  wire \blk00000003/sig000001e0 ;
  wire \blk00000003/sig000001df ;
  wire \blk00000003/sig000001de ;
  wire \blk00000003/sig000001dd ;
  wire \blk00000003/sig000001dc ;
  wire \blk00000003/sig000001db ;
  wire \blk00000003/sig000001da ;
  wire \blk00000003/sig000001d9 ;
  wire \blk00000003/sig000001d8 ;
  wire \blk00000003/sig000001d7 ;
  wire \blk00000003/sig000001d6 ;
  wire \blk00000003/sig000001d5 ;
  wire \blk00000003/sig000001d4 ;
  wire \blk00000003/sig000001d3 ;
  wire \blk00000003/sig000001d2 ;
  wire \blk00000003/sig000001d1 ;
  wire \blk00000003/sig000001d0 ;
  wire \blk00000003/sig000001cf ;
  wire \blk00000003/sig000001ce ;
  wire \blk00000003/sig000001cd ;
  wire \blk00000003/sig000001cc ;
  wire \blk00000003/sig000001cb ;
  wire \blk00000003/sig000001ca ;
  wire \blk00000003/sig000001c9 ;
  wire \blk00000003/sig000001c8 ;
  wire \blk00000003/sig000001c7 ;
  wire \blk00000003/sig000001c6 ;
  wire \blk00000003/sig000001c5 ;
  wire \blk00000003/sig000001c4 ;
  wire \blk00000003/sig000001c3 ;
  wire \blk00000003/sig000001c2 ;
  wire \blk00000003/sig000001c1 ;
  wire \blk00000003/sig000001c0 ;
  wire \blk00000003/sig000001bf ;
  wire \blk00000003/sig000001be ;
  wire \blk00000003/sig000001bd ;
  wire \blk00000003/sig000001bc ;
  wire \blk00000003/sig000001bb ;
  wire \blk00000003/sig000001ba ;
  wire \blk00000003/sig000001b9 ;
  wire \blk00000003/sig000001b8 ;
  wire \blk00000003/sig000001b7 ;
  wire \blk00000003/sig000001b6 ;
  wire \blk00000003/sig000001b5 ;
  wire \blk00000003/sig000001b4 ;
  wire \blk00000003/sig000001b3 ;
  wire \blk00000003/sig000001b2 ;
  wire \blk00000003/sig000001b1 ;
  wire \blk00000003/sig000001b0 ;
  wire \blk00000003/sig000001af ;
  wire \blk00000003/sig000001ae ;
  wire \blk00000003/sig000001ad ;
  wire \blk00000003/sig000001ac ;
  wire \blk00000003/sig000001ab ;
  wire \blk00000003/sig000001aa ;
  wire \blk00000003/sig000001a9 ;
  wire \blk00000003/sig000001a8 ;
  wire \blk00000003/sig000001a7 ;
  wire \blk00000003/sig000001a6 ;
  wire \blk00000003/sig000001a5 ;
  wire \blk00000003/sig000001a4 ;
  wire \blk00000003/sig000001a3 ;
  wire \blk00000003/sig000001a2 ;
  wire \blk00000003/sig000001a1 ;
  wire \blk00000003/sig000001a0 ;
  wire \blk00000003/sig0000019f ;
  wire \blk00000003/sig0000019e ;
  wire \blk00000003/sig0000019d ;
  wire \blk00000003/sig0000019c ;
  wire \blk00000003/sig0000019b ;
  wire \blk00000003/sig0000019a ;
  wire \blk00000003/sig00000199 ;
  wire \blk00000003/sig00000198 ;
  wire \blk00000003/sig00000197 ;
  wire \blk00000003/sig00000196 ;
  wire \blk00000003/sig00000195 ;
  wire \blk00000003/sig00000191 ;
  wire \blk00000003/sig00000190 ;
  wire \blk00000003/sig0000018f ;
  wire \blk00000003/sig0000018e ;
  wire \blk00000003/sig0000018d ;
  wire \blk00000003/sig0000018c ;
  wire \blk00000003/sig0000018b ;
  wire \blk00000003/sig0000018a ;
  wire \blk00000003/sig00000189 ;
  wire \blk00000003/sig00000188 ;
  wire \blk00000003/sig00000187 ;
  wire \blk00000003/sig00000186 ;
  wire \blk00000003/sig00000185 ;
  wire \blk00000003/sig00000184 ;
  wire \blk00000003/sig00000183 ;
  wire \blk00000003/sig00000182 ;
  wire \blk00000003/sig00000181 ;
  wire \blk00000003/sig00000180 ;
  wire \blk00000003/sig0000017f ;
  wire \blk00000003/sig0000017e ;
  wire \blk00000003/sig0000017d ;
  wire \blk00000003/sig0000017c ;
  wire \blk00000003/sig0000017b ;
  wire \blk00000003/sig0000017a ;
  wire \blk00000003/sig00000179 ;
  wire \blk00000003/sig00000178 ;
  wire \blk00000003/sig00000177 ;
  wire \blk00000003/sig00000176 ;
  wire \blk00000003/sig00000175 ;
  wire \blk00000003/sig00000174 ;
  wire \blk00000003/sig00000173 ;
  wire \blk00000003/sig00000172 ;
  wire \blk00000003/sig00000171 ;
  wire \blk00000003/sig00000170 ;
  wire \blk00000003/sig0000016f ;
  wire \blk00000003/sig0000016e ;
  wire \blk00000003/sig0000016d ;
  wire \blk00000003/sig0000016c ;
  wire \blk00000003/sig0000016b ;
  wire \blk00000003/sig0000016a ;
  wire \blk00000003/sig00000169 ;
  wire \blk00000003/sig00000168 ;
  wire \blk00000003/sig00000167 ;
  wire \blk00000003/sig00000166 ;
  wire \blk00000003/sig00000165 ;
  wire \blk00000003/sig00000164 ;
  wire \blk00000003/sig00000163 ;
  wire \blk00000003/sig00000162 ;
  wire \blk00000003/sig00000161 ;
  wire \blk00000003/sig00000160 ;
  wire \blk00000003/sig0000015f ;
  wire \blk00000003/sig0000015e ;
  wire \blk00000003/sig0000015d ;
  wire \blk00000003/sig0000015c ;
  wire \blk00000003/sig0000015b ;
  wire \blk00000003/sig0000015a ;
  wire \blk00000003/sig00000159 ;
  wire \blk00000003/sig00000158 ;
  wire \blk00000003/sig00000157 ;
  wire \blk00000003/sig00000156 ;
  wire \blk00000003/sig00000155 ;
  wire \blk00000003/sig00000154 ;
  wire \blk00000003/sig00000153 ;
  wire \blk00000003/sig00000152 ;
  wire \blk00000003/sig00000151 ;
  wire \blk00000003/sig00000150 ;
  wire \blk00000003/sig0000014f ;
  wire \blk00000003/sig0000014e ;
  wire \blk00000003/sig0000014d ;
  wire \blk00000003/sig0000014c ;
  wire \blk00000003/sig0000014b ;
  wire \blk00000003/sig0000014a ;
  wire \blk00000003/sig00000149 ;
  wire \blk00000003/sig00000148 ;
  wire \blk00000003/sig00000147 ;
  wire \blk00000003/sig00000146 ;
  wire \blk00000003/sig00000145 ;
  wire \blk00000003/sig00000144 ;
  wire \blk00000003/sig00000143 ;
  wire \blk00000003/sig00000142 ;
  wire \blk00000003/sig00000141 ;
  wire \blk00000003/sig00000140 ;
  wire \blk00000003/sig0000013f ;
  wire \blk00000003/sig0000013e ;
  wire \blk00000003/sig0000013d ;
  wire \blk00000003/sig0000013c ;
  wire \blk00000003/sig0000013b ;
  wire \blk00000003/sig0000013a ;
  wire \blk00000003/sig00000139 ;
  wire \blk00000003/sig00000138 ;
  wire \blk00000003/sig00000137 ;
  wire \blk00000003/sig00000136 ;
  wire \blk00000003/sig00000135 ;
  wire \blk00000003/sig00000134 ;
  wire \blk00000003/sig00000133 ;
  wire \blk00000003/sig00000132 ;
  wire \blk00000003/sig00000131 ;
  wire \blk00000003/sig00000130 ;
  wire \blk00000003/sig0000012f ;
  wire \blk00000003/sig0000012e ;
  wire \blk00000003/sig0000012d ;
  wire \blk00000003/sig0000012c ;
  wire \blk00000003/sig0000012b ;
  wire \blk00000003/sig0000012a ;
  wire \blk00000003/sig00000129 ;
  wire \blk00000003/sig00000128 ;
  wire \blk00000003/sig00000127 ;
  wire \blk00000003/sig00000126 ;
  wire \blk00000003/sig00000125 ;
  wire \blk00000003/sig00000124 ;
  wire \blk00000003/sig00000123 ;
  wire \blk00000003/sig00000122 ;
  wire \blk00000003/sig00000121 ;
  wire \blk00000003/sig00000120 ;
  wire \blk00000003/sig0000011f ;
  wire \blk00000003/sig0000011e ;
  wire \blk00000003/sig0000011d ;
  wire \blk00000003/sig0000011c ;
  wire \blk00000003/sig0000011b ;
  wire \blk00000003/sig0000011a ;
  wire \blk00000003/sig00000119 ;
  wire \blk00000003/sig00000118 ;
  wire \blk00000003/sig00000117 ;
  wire \blk00000003/sig00000116 ;
  wire \blk00000003/sig00000115 ;
  wire \blk00000003/sig00000114 ;
  wire \blk00000003/sig00000113 ;
  wire \blk00000003/sig00000112 ;
  wire \blk00000003/sig00000111 ;
  wire \blk00000003/sig00000110 ;
  wire \blk00000003/sig0000010f ;
  wire \blk00000003/sig0000010e ;
  wire \blk00000003/sig0000010d ;
  wire \blk00000003/sig0000010c ;
  wire \blk00000003/sig0000010b ;
  wire \blk00000003/sig0000010a ;
  wire \blk00000003/sig00000109 ;
  wire \blk00000003/sig00000108 ;
  wire \blk00000003/sig00000107 ;
  wire \blk00000003/sig00000106 ;
  wire \blk00000003/sig00000105 ;
  wire \blk00000003/sig00000104 ;
  wire \blk00000003/sig00000103 ;
  wire \blk00000003/sig00000102 ;
  wire \blk00000003/sig00000101 ;
  wire \blk00000003/sig00000100 ;
  wire \blk00000003/sig000000ff ;
  wire \blk00000003/sig000000fe ;
  wire \blk00000003/sig000000fd ;
  wire \blk00000003/sig000000fc ;
  wire \blk00000003/sig000000fb ;
  wire \blk00000003/sig000000fa ;
  wire \blk00000003/sig000000f9 ;
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
  wire \blk00000003/sig0000003a ;
  wire \blk00000003/sig00000039 ;
  wire NLW_blk00000001_P_UNCONNECTED;
  wire NLW_blk00000002_G_UNCONNECTED;
  wire \NLW_blk00000003/blk000000d2_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk000000ac_Q_UNCONNECTED ;
  wire \NLW_blk00000003/blk000000a8_Q_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000065_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000044_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000033_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk0000001c_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000009_O_UNCONNECTED ;
  wire \NLW_blk00000003/blk00000006_O_UNCONNECTED ;
  assign
    sig00000021 = operation[5],
    sig00000022 = operation[4],
    sig00000023 = operation[3],
    sig00000024 = operation[2],
    sig00000025 = operation[1],
    sig00000026 = operation[0],
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
    result[15] = sig00000028,
    result[14] = sig00000029,
    result[13] = sig0000002a,
    result[12] = sig0000002b,
    result[11] = sig0000002c,
    result[10] = sig0000002d,
    result[9] = sig0000002e,
    result[8] = sig0000002f,
    result[7] = sig00000030,
    result[6] = sig00000031,
    result[5] = sig00000032,
    result[4] = sig00000033,
    result[3] = sig00000034,
    result[2] = sig00000035,
    result[1] = sig00000036,
    result[0] = sig00000037,
    sig00000027 = clk;
  VCC   blk00000001 (
    .P(NLW_blk00000001_P_UNCONNECTED)
  );
  GND   blk00000002 (
    .G(NLW_blk00000002_G_UNCONNECTED)
  );
  LUT4_L #(
    .INIT ( 16'h0105 ))
  \blk00000003/blk000001be  (
    .I0(\blk00000003/sig00000198 ),
    .I1(\blk00000003/sig00000094 ),
    .I2(\blk00000003/sig000001a7 ),
    .I3(\blk00000003/sig0000008c ),
    .LO(\blk00000003/sig000001ee )
  );
  LUT4_L #(
    .INIT ( 16'h040C ))
  \blk00000003/blk000001bd  (
    .I0(\blk00000003/sig0000008c ),
    .I1(\blk00000003/sig00000198 ),
    .I2(\blk00000003/sig0000014d ),
    .I3(\blk00000003/sig00000094 ),
    .LO(\blk00000003/sig000001a6 )
  );
  LUT3_L #(
    .INIT ( 8'h13 ))
  \blk00000003/blk000001bc  (
    .I0(\blk00000003/sig00000094 ),
    .I1(\blk00000003/sig0000014d ),
    .I2(\blk00000003/sig0000008c ),
    .LO(\blk00000003/sig000001ed )
  );
  LUT4_L #(
    .INIT ( 16'hFCFA ))
  \blk00000003/blk000001bb  (
    .I0(\blk00000003/sig00000169 ),
    .I1(\blk00000003/sig00000171 ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .LO(\blk00000003/sig000001e7 )
  );
  LUT4_D #(
    .INIT ( 16'hF3F5 ))
  \blk00000003/blk000001ba  (
    .I0(\blk00000003/sig00000171 ),
    .I1(\blk00000003/sig00000163 ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .LO(\blk00000003/sig000001ab ),
    .O(\blk00000003/sig000001a4 )
  );
  LUT4_D #(
    .INIT ( 16'hF5F3 ))
  \blk00000003/blk000001b9  (
    .I0(\blk00000003/sig00000165 ),
    .I1(\blk00000003/sig0000015d ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .LO(\blk00000003/sig000001b0 ),
    .O(\blk00000003/sig000001a8 )
  );
  LUT3_L #(
    .INIT ( 8'hD0 ))
  \blk00000003/blk000001b8  (
    .I0(\blk00000003/sig00000165 ),
    .I1(\blk00000003/sig00000094 ),
    .I2(\blk00000003/sig0000008c ),
    .LO(\blk00000003/sig000001eb )
  );
  LUT4_D #(
    .INIT ( 16'hDFD5 ))
  \blk00000003/blk000001b7  (
    .I0(\blk00000003/sig00000165 ),
    .I1(\blk00000003/sig00000094 ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .LO(\blk00000003/sig000001f4 ),
    .O(\blk00000003/sig000001be )
  );
  LUT2_L #(
    .INIT ( 4'hE ))
  \blk00000003/blk000001b6  (
    .I0(\blk00000003/sig00000096 ),
    .I1(\blk00000003/sig0000008c ),
    .LO(\blk00000003/sig000001b2 )
  );
  LUT4_D #(
    .INIT ( 16'hDFD5 ))
  \blk00000003/blk000001b5  (
    .I0(\blk00000003/sig00000161 ),
    .I1(\blk00000003/sig00000094 ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .LO(\blk00000003/sig000001bd ),
    .O(\blk00000003/sig000001b1 )
  );
  LUT4_D #(
    .INIT ( 16'hF3F5 ))
  \blk00000003/blk000001b4  (
    .I0(\blk00000003/sig0000016d ),
    .I1(\blk00000003/sig0000015f ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .LO(\blk00000003/sig000001a5 ),
    .O(\blk00000003/sig000001e8 )
  );
  LUT4_D #(
    .INIT ( 16'hF3F5 ))
  \blk00000003/blk000001b3  (
    .I0(\blk00000003/sig0000016f ),
    .I1(\blk00000003/sig00000161 ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .LO(\blk00000003/sig000001a9 ),
    .O(\blk00000003/sig000001ea )
  );
  LUT4_L #(
    .INIT ( 16'h8400 ))
  \blk00000003/blk000001b2  (
    .I0(\blk00000003/sig0000012c ),
    .I1(\blk00000003/sig000001a1 ),
    .I2(\blk00000003/sig00000092 ),
    .I3(\blk00000003/sig000001f2 ),
    .LO(\blk00000003/sig000001f1 )
  );
  LUT4_L #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk000001b1  (
    .I0(\blk00000003/sig00000114 ),
    .I1(\blk00000003/sig00000118 ),
    .I2(\blk00000003/sig0000011c ),
    .I3(\blk00000003/sig00000120 ),
    .LO(\blk00000003/sig000001a0 )
  );
  LUT2_D #(
    .INIT ( 4'h8 ))
  \blk00000003/blk000001b0  (
    .I0(\blk00000003/sig0000008c ),
    .I1(\blk00000003/sig00000094 ),
    .LO(\blk00000003/sig000001f0 ),
    .O(\blk00000003/sig00000197 )
  );
  MUXF5   \blk00000003/blk000001af  (
    .I0(\blk00000003/sig00000214 ),
    .I1(\blk00000003/sig0000003a ),
    .S(\blk00000003/sig00000213 ),
    .O(\blk00000003/sig0000014c )
  );
  LUT4 #(
    .INIT ( 16'h40C0 ))
  \blk00000003/blk000001ae  (
    .I0(\blk00000003/sig000000fe ),
    .I1(\blk00000003/sig0000010a ),
    .I2(\blk00000003/sig0000010e ),
    .I3(\blk00000003/sig00000206 ),
    .O(\blk00000003/sig00000214 )
  );
  MUXF5   \blk00000003/blk000001ad  (
    .I0(\blk00000003/sig00000212 ),
    .I1(\blk00000003/sig00000211 ),
    .S(\blk00000003/sig00000106 ),
    .O(\blk00000003/sig00000213 )
  );
  LUT3 #(
    .INIT ( 8'h08 ))
  \blk00000003/blk000001ac  (
    .I0(\blk00000003/sig00000102 ),
    .I1(\blk00000003/sig000000fe ),
    .I2(\blk00000003/sig0000010a ),
    .O(\blk00000003/sig00000212 )
  );
  LUT4 #(
    .INIT ( 16'hAAEA ))
  \blk00000003/blk000001ab  (
    .I0(\blk00000003/sig000000fa ),
    .I1(\blk00000003/sig00000102 ),
    .I2(\blk00000003/sig000000fe ),
    .I3(\blk00000003/sig0000010a ),
    .O(\blk00000003/sig00000211 )
  );
  LUT3 #(
    .INIT ( 8'hB1 ))
  \blk00000003/blk000001aa  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig000001e4 ),
    .I2(\blk00000003/sig0000015b ),
    .O(\blk00000003/sig00000154 )
  );
  LUT3 #(
    .INIT ( 8'h1D ))
  \blk00000003/blk000001a9  (
    .I0(\blk00000003/sig000001fb ),
    .I1(\blk00000003/sig000001b9 ),
    .I2(\blk00000003/sig000001e2 ),
    .O(\blk00000003/sig00000153 )
  );
  INV   \blk00000003/blk000001a8  (
    .I(\blk00000003/sig00000114 ),
    .O(\blk00000003/sig00000112 )
  );
  INV   \blk00000003/blk000001a7  (
    .I(\blk00000003/sig00000118 ),
    .O(\blk00000003/sig00000116 )
  );
  INV   \blk00000003/blk000001a6  (
    .I(\blk00000003/sig0000011c ),
    .O(\blk00000003/sig0000011a )
  );
  INV   \blk00000003/blk000001a5  (
    .I(\blk00000003/sig00000120 ),
    .O(\blk00000003/sig0000011e )
  );
  INV   \blk00000003/blk000001a4  (
    .I(\blk00000003/sig00000131 ),
    .O(\blk00000003/sig000000f6 )
  );
  INV   \blk00000003/blk000001a3  (
    .I(sig00000010),
    .O(\blk00000003/sig00000070 )
  );
  INV   \blk00000003/blk000001a2  (
    .I(sig00000020),
    .O(\blk00000003/sig00000075 )
  );
  LUT4 #(
    .INIT ( 16'h0093 ))
  \blk00000003/blk000001a1  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig00000146 ),
    .I2(\blk00000003/sig00000131 ),
    .I3(\blk00000003/sig000001b8 ),
    .O(\blk00000003/sig00000210 )
  );
  LUT3 #(
    .INIT ( 8'h1D ))
  \blk00000003/blk000001a0  (
    .I0(\blk00000003/sig000001d5 ),
    .I1(\blk00000003/sig000001db ),
    .I2(\blk00000003/sig000001d8 ),
    .O(\blk00000003/sig0000020f )
  );
  MUXF5   \blk00000003/blk0000019f  (
    .I0(\blk00000003/sig0000020f ),
    .I1(\blk00000003/sig00000210 ),
    .S(\blk00000003/sig000001b9 ),
    .O(\blk00000003/sig00000204 )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk0000019e  (
    .I0(\blk00000003/sig00000208 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig000001c0 ),
    .I3(\blk00000003/sig000001b8 ),
    .O(\blk00000003/sig0000020e )
  );
  LUT4 #(
    .INIT ( 16'h820A ))
  \blk00000003/blk0000019d  (
    .I0(\blk00000003/sig000001fa ),
    .I1(\blk00000003/sig00000131 ),
    .I2(\blk00000003/sig00000140 ),
    .I3(\blk00000003/sig000001ce ),
    .O(\blk00000003/sig0000020d )
  );
  MUXF5   \blk00000003/blk0000019c  (
    .I0(\blk00000003/sig0000020d ),
    .I1(\blk00000003/sig0000020e ),
    .S(\blk00000003/sig000001db ),
    .O(\blk00000003/sig000001df )
  );
  LUT4 #(
    .INIT ( 16'hFAD8 ))
  \blk00000003/blk0000019b  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig000001d9 ),
    .I3(\blk00000003/sig000001c1 ),
    .O(\blk00000003/sig0000020c )
  );
  LUT4 #(
    .INIT ( 16'hE444 ))
  \blk00000003/blk0000019a  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig000001de ),
    .I2(\blk00000003/sig000001b9 ),
    .I3(\blk00000003/sig000001b8 ),
    .O(\blk00000003/sig0000020b )
  );
  MUXF5   \blk00000003/blk00000199  (
    .I0(\blk00000003/sig0000020b ),
    .I1(\blk00000003/sig0000020c ),
    .S(\blk00000003/sig000001db ),
    .O(\blk00000003/sig000001fd )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk00000198  (
    .I0(\blk00000003/sig00000155 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig000001de ),
    .I3(\blk00000003/sig000001d6 ),
    .O(\blk00000003/sig0000020a )
  );
  LUT4 #(
    .INIT ( 16'h6240 ))
  \blk00000003/blk00000197  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig00000155 ),
    .I2(\blk00000003/sig000001c9 ),
    .I3(\blk00000003/sig000001d9 ),
    .O(\blk00000003/sig00000209 )
  );
  MUXF5   \blk00000003/blk00000196  (
    .I0(\blk00000003/sig00000209 ),
    .I1(\blk00000003/sig0000020a ),
    .S(\blk00000003/sig000001db ),
    .O(\blk00000003/sig000001fc )
  );
  LUT4 #(
    .INIT ( 16'hFF6A ))
  \blk00000003/blk00000195  (
    .I0(\blk00000003/sig00000140 ),
    .I1(\blk00000003/sig00000131 ),
    .I2(\blk00000003/sig000001ce ),
    .I3(\blk00000003/sig000001b9 ),
    .O(\blk00000003/sig00000208 )
  );
  LUT4 #(
    .INIT ( 16'h0090 ))
  \blk00000003/blk00000194  (
    .I0(\blk00000003/sig00000143 ),
    .I1(\blk00000003/sig00000131 ),
    .I2(\blk00000003/sig00000148 ),
    .I3(\blk00000003/sig00000155 ),
    .O(\blk00000003/sig000001d2 )
  );
  LUT4 #(
    .INIT ( 16'h999C ))
  \blk00000003/blk00000193  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig00000207 ),
    .I2(\blk00000003/sig000001e3 ),
    .I3(\blk00000003/sig00000205 ),
    .O(\blk00000003/sig0000017a )
  );
  LUT4 #(
    .INIT ( 16'h6996 ))
  \blk00000003/blk00000192  (
    .I0(\blk00000003/sig0000017b ),
    .I1(sig00000026),
    .I2(sig00000011),
    .I3(sig00000001),
    .O(\blk00000003/sig00000207 )
  );
  LUT4 #(
    .INIT ( 16'h96FF ))
  \blk00000003/blk00000191  (
    .I0(sig00000001),
    .I1(sig00000011),
    .I2(sig00000026),
    .I3(\blk00000003/sig00000102 ),
    .O(\blk00000003/sig00000206 )
  );
  LUT3 #(
    .INIT ( 8'h96 ))
  \blk00000003/blk00000190  (
    .I0(sig00000001),
    .I1(sig00000011),
    .I2(sig00000026),
    .O(\blk00000003/sig00000172 )
  );
  LUT4 #(
    .INIT ( 16'h0213 ))
  \blk00000003/blk0000018f  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig00000155 ),
    .I2(\blk00000003/sig00000203 ),
    .I3(\blk00000003/sig00000204 ),
    .O(\blk00000003/sig00000205 )
  );
  LUT4 #(
    .INIT ( 16'hFBBF ))
  \blk00000003/blk0000018e  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig000001c0 ),
    .I2(\blk00000003/sig00000146 ),
    .I3(\blk00000003/sig00000131 ),
    .O(\blk00000003/sig00000203 )
  );
  LUT3 #(
    .INIT ( 8'h96 ))
  \blk00000003/blk0000018d  (
    .I0(sig00000001),
    .I1(sig00000011),
    .I2(sig00000026),
    .O(\blk00000003/sig00000183 )
  );
  LUT4 #(
    .INIT ( 16'h0213 ))
  \blk00000003/blk0000018c  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig00000202 ),
    .I3(\blk00000003/sig00000201 ),
    .O(\blk00000003/sig000001ff )
  );
  LUT4 #(
    .INIT ( 16'hFF1B ))
  \blk00000003/blk0000018b  (
    .I0(\blk00000003/sig000001db ),
    .I1(\blk00000003/sig000001d8 ),
    .I2(\blk00000003/sig000001b8 ),
    .I3(\blk00000003/sig00000155 ),
    .O(\blk00000003/sig00000202 )
  );
  LUT3 #(
    .INIT ( 8'hF7 ))
  \blk00000003/blk0000018a  (
    .I0(\blk00000003/sig000001d5 ),
    .I1(\blk00000003/sig000001db ),
    .I2(\blk00000003/sig00000155 ),
    .O(\blk00000003/sig00000201 )
  );
  LUT4 #(
    .INIT ( 16'hA596 ))
  \blk00000003/blk00000189  (
    .I0(\blk00000003/sig0000019e ),
    .I1(\blk00000003/sig000001b4 ),
    .I2(sig00000001),
    .I3(\blk00000003/sig00000200 ),
    .O(\blk00000003/sig00000180 )
  );
  LUT4 #(
    .INIT ( 16'h0213 ))
  \blk00000003/blk00000188  (
    .I0(\blk00000003/sig000001db ),
    .I1(\blk00000003/sig000001ff ),
    .I2(\blk00000003/sig000001e6 ),
    .I3(\blk00000003/sig000001e5 ),
    .O(\blk00000003/sig00000200 )
  );
  LUT4 #(
    .INIT ( 16'h0213 ))
  \blk00000003/blk00000187  (
    .I0(\blk00000003/sig000000be ),
    .I1(\blk00000003/sig000001d5 ),
    .I2(sig00000020),
    .I3(sig00000010),
    .O(\blk00000003/sig00000157 )
  );
  LUT4 #(
    .INIT ( 16'hA820 ))
  \blk00000003/blk00000186  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig0000000e),
    .I3(sig0000001e),
    .O(\blk00000003/sig000001e1 )
  );
  LUT4 #(
    .INIT ( 16'h9669 ))
  \blk00000003/blk00000185  (
    .I0(\blk00000003/sig0000019e ),
    .I1(\blk00000003/sig0000017e ),
    .I2(sig00000001),
    .I3(\blk00000003/sig000001fe ),
    .O(\blk00000003/sig0000017d )
  );
  LUT4 #(
    .INIT ( 16'h0F02 ))
  \blk00000003/blk00000184  (
    .I0(\blk00000003/sig000001fd ),
    .I1(\blk00000003/sig00000155 ),
    .I2(\blk00000003/sig000001b4 ),
    .I3(\blk00000003/sig000001dc ),
    .O(\blk00000003/sig000001fe )
  );
  LUT4 #(
    .INIT ( 16'hB44B ))
  \blk00000003/blk00000183  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig000001fc ),
    .I2(\blk00000003/sig0000019e ),
    .I3(sig00000001),
    .O(\blk00000003/sig00000181 )
  );
  LUT2 #(
    .INIT ( 4'h7 ))
  \blk00000003/blk00000182  (
    .I0(\blk00000003/sig000000fa ),
    .I1(\blk00000003/sig00000106 ),
    .O(\blk00000003/sig00000186 )
  );
  LUT4 #(
    .INIT ( 16'h0880 ))
  \blk00000003/blk00000181  (
    .I0(\blk00000003/sig00000106 ),
    .I1(\blk00000003/sig000000fa ),
    .I2(sig00000011),
    .I3(sig00000026),
    .O(\blk00000003/sig0000019b )
  );
  LUT3 #(
    .INIT ( 8'h28 ))
  \blk00000003/blk00000180  (
    .I0(\blk00000003/sig0000010e ),
    .I1(sig00000011),
    .I2(sig00000026),
    .O(\blk00000003/sig0000019a )
  );
  LUT4 #(
    .INIT ( 16'hBE14 ))
  \blk00000003/blk0000017f  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig00000011),
    .I2(sig00000026),
    .I3(sig00000001),
    .O(\blk00000003/sig00000150 )
  );
  LUT4 #(
    .INIT ( 16'h0880 ))
  \blk00000003/blk0000017e  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig000001dd ),
    .I2(\blk00000003/sig00000146 ),
    .I3(\blk00000003/sig00000131 ),
    .O(\blk00000003/sig000001fb )
  );
  LUT4 #(
    .INIT ( 16'hFAD8 ))
  \blk00000003/blk0000017d  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig000001d9 ),
    .I3(\blk00000003/sig000001c1 ),
    .O(\blk00000003/sig000001fa )
  );
  LUT4 #(
    .INIT ( 16'h0213 ))
  \blk00000003/blk0000017c  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig00000155 ),
    .I2(\blk00000003/sig000001f9 ),
    .I3(\blk00000003/sig000001f8 ),
    .O(\blk00000003/sig000001d4 )
  );
  LUT4 #(
    .INIT ( 16'hFF1B ))
  \blk00000003/blk0000017b  (
    .I0(\blk00000003/sig000001c4 ),
    .I1(\blk00000003/sig000001c1 ),
    .I2(\blk00000003/sig000001c0 ),
    .I3(\blk00000003/sig000001b9 ),
    .O(\blk00000003/sig000001f9 )
  );
  LUT4 #(
    .INIT ( 16'hA2A7 ))
  \blk00000003/blk0000017a  (
    .I0(\blk00000003/sig00000146 ),
    .I1(\blk00000003/sig000001b8 ),
    .I2(\blk00000003/sig000001b9 ),
    .I3(\blk00000003/sig000001d8 ),
    .O(\blk00000003/sig000001f8 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000179  (
    .I0(\blk00000003/sig00000143 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig00000146 ),
    .I3(\blk00000003/sig00000155 ),
    .O(\blk00000003/sig000001b5 )
  );
  LUT4 #(
    .INIT ( 16'h7FEA ))
  \blk00000003/blk00000178  (
    .I0(\blk00000003/sig00000146 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig00000131 ),
    .I3(\blk00000003/sig00000143 ),
    .O(\blk00000003/sig000001d1 )
  );
  LUT4 #(
    .INIT ( 16'h7FFD ))
  \blk00000003/blk00000177  (
    .I0(\blk00000003/sig000001c1 ),
    .I1(\blk00000003/sig00000131 ),
    .I2(\blk00000003/sig00000146 ),
    .I3(\blk00000003/sig00000143 ),
    .O(\blk00000003/sig000001bb )
  );
  LUT4 #(
    .INIT ( 16'hFFEA ))
  \blk00000003/blk00000176  (
    .I0(\blk00000003/sig0000014d ),
    .I1(\blk00000003/sig00000152 ),
    .I2(\blk00000003/sig00000167 ),
    .I3(\blk00000003/sig0000014b ),
    .O(\blk00000003/sig000001ef )
  );
  LUT4 #(
    .INIT ( 16'h0080 ))
  \blk00000003/blk00000175  (
    .I0(\blk00000003/sig000001f7 ),
    .I1(\blk00000003/sig000001b3 ),
    .I2(\blk00000003/sig000000f3 ),
    .I3(\blk00000003/sig000000f5 ),
    .O(\blk00000003/sig00000151 )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk00000174  (
    .I0(\blk00000003/sig000000ed ),
    .I1(\blk00000003/sig000000ea ),
    .I2(\blk00000003/sig000000e7 ),
    .I3(\blk00000003/sig000000e4 ),
    .O(\blk00000003/sig000001f7 )
  );
  LUT4 #(
    .INIT ( 16'hF888 ))
  \blk00000003/blk00000173  (
    .I0(sig00000001),
    .I1(\blk00000003/sig0000019c ),
    .I2(\blk00000003/sig000001f6 ),
    .I3(\blk00000003/sig0000019d ),
    .O(\blk00000003/sig0000014e )
  );
  LUT4 #(
    .INIT ( 16'h44E4 ))
  \blk00000003/blk00000172  (
    .I0(\blk00000003/sig0000010a ),
    .I1(\blk00000003/sig00000150 ),
    .I2(\blk00000003/sig0000010e ),
    .I3(\blk00000003/sig0000019e ),
    .O(\blk00000003/sig000001f6 )
  );
  LUT4 #(
    .INIT ( 16'hFFFE ))
  \blk00000003/blk00000171  (
    .I0(\blk00000003/sig00000106 ),
    .I1(\blk00000003/sig000000fa ),
    .I2(\blk00000003/sig000001cc ),
    .I3(\blk00000003/sig000001f5 ),
    .O(\blk00000003/sig000001d0 )
  );
  LUT4 #(
    .INIT ( 16'h40EA ))
  \blk00000003/blk00000170  (
    .I0(\blk00000003/sig0000013d ),
    .I1(\blk00000003/sig000001cd ),
    .I2(\blk00000003/sig00000131 ),
    .I3(\blk00000003/sig0000013a ),
    .O(\blk00000003/sig000001f5 )
  );
  LUT4 #(
    .INIT ( 16'hA8AA ))
  \blk00000003/blk0000016f  (
    .I0(\blk00000003/sig000001f4 ),
    .I1(\blk00000003/sig00000096 ),
    .I2(\blk00000003/sig0000008f ),
    .I3(\blk00000003/sig00000163 ),
    .O(\blk00000003/sig000001f3 )
  );
  LUT4 #(
    .INIT ( 16'hAAA8 ))
  \blk00000003/blk0000016e  (
    .I0(\blk00000003/sig00000195 ),
    .I1(\blk00000003/sig0000008c ),
    .I2(\blk00000003/sig000001f3 ),
    .I3(\blk00000003/sig00000092 ),
    .O(\blk00000003/sig0000009c )
  );
  LUT3 #(
    .INIT ( 8'hFB ))
  \blk00000003/blk0000016d  (
    .I0(\blk00000003/sig0000008c ),
    .I1(\blk00000003/sig00000163 ),
    .I2(\blk00000003/sig00000096 ),
    .O(\blk00000003/sig000001b6 )
  );
  LUT3 #(
    .INIT ( 8'h2F ))
  \blk00000003/blk0000016c  (
    .I0(\blk00000003/sig00000163 ),
    .I1(\blk00000003/sig00000094 ),
    .I2(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig000001e9 )
  );
  LUT4 #(
    .INIT ( 16'h0305 ))
  \blk00000003/blk0000016b  (
    .I0(\blk00000003/sig0000016b ),
    .I1(\blk00000003/sig0000015d ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .O(\blk00000003/sig000001ec )
  );
  LUT4 #(
    .INIT ( 16'h9A95 ))
  \blk00000003/blk0000016a  (
    .I0(\blk00000003/sig00000128 ),
    .I1(\blk00000003/sig00000094 ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .O(\blk00000003/sig000001f2 )
  );
  LUT3 #(
    .INIT ( 8'h8D ))
  \blk00000003/blk00000169  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001ad ),
    .I2(\blk00000003/sig000001a2 ),
    .O(\blk00000003/sig000000b5 )
  );
  LUT3 #(
    .INIT ( 8'h90 ))
  \blk00000003/blk00000168  (
    .I0(\blk00000003/sig0000012f ),
    .I1(\blk00000003/sig0000008f ),
    .I2(\blk00000003/sig000001f1 ),
    .O(\blk00000003/sig000001a7 )
  );
  LUT4 #(
    .INIT ( 16'hFFFE ))
  \blk00000003/blk00000167  (
    .I0(\blk00000003/sig000001ef ),
    .I1(\blk00000003/sig000001f0 ),
    .I2(\blk00000003/sig00000110 ),
    .I3(\blk00000003/sig000001a7 ),
    .O(\blk00000003/sig0000007a )
  );
  LUT4 #(
    .INIT ( 16'hBABB ))
  \blk00000003/blk00000166  (
    .I0(\blk00000003/sig0000014d ),
    .I1(\blk00000003/sig0000014b ),
    .I2(\blk00000003/sig00000110 ),
    .I3(\blk00000003/sig000001ee ),
    .O(\blk00000003/sig00000077 )
  );
  LUT4 #(
    .INIT ( 16'h5551 ))
  \blk00000003/blk00000165  (
    .I0(\blk00000003/sig0000014b ),
    .I1(\blk00000003/sig000001ed ),
    .I2(\blk00000003/sig00000110 ),
    .I3(\blk00000003/sig000001a7 ),
    .O(\blk00000003/sig00000082 )
  );
  LUT4 #(
    .INIT ( 16'h9A95 ))
  \blk00000003/blk00000164  (
    .I0(\blk00000003/sig00000128 ),
    .I1(\blk00000003/sig00000094 ),
    .I2(\blk00000003/sig0000008c ),
    .I3(\blk00000003/sig00000096 ),
    .O(\blk00000003/sig00000126 )
  );
  LUT4 #(
    .INIT ( 16'hAFAC ))
  \blk00000003/blk00000163  (
    .I0(\blk00000003/sig000001ea ),
    .I1(\blk00000003/sig000001eb ),
    .I2(\blk00000003/sig00000092 ),
    .I3(\blk00000003/sig000001ec ),
    .O(\blk00000003/sig000001c6 )
  );
  LUT4 #(
    .INIT ( 16'h3A30 ))
  \blk00000003/blk00000162  (
    .I0(\blk00000003/sig000001e7 ),
    .I1(\blk00000003/sig000001e8 ),
    .I2(\blk00000003/sig00000092 ),
    .I3(\blk00000003/sig000001e9 ),
    .O(\blk00000003/sig000001c5 )
  );
  LUT3 #(
    .INIT ( 8'h1D ))
  \blk00000003/blk00000161  (
    .I0(\blk00000003/sig000001af ),
    .I1(\blk00000003/sig0000008f ),
    .I2(\blk00000003/sig000001b7 ),
    .O(\blk00000003/sig000000b0 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000160  (
    .I0(\blk00000003/sig0000012f ),
    .I1(\blk00000003/sig0000008f ),
    .O(\blk00000003/sig0000012d )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000015f  (
    .I0(\blk00000003/sig00000113 ),
    .O(\blk00000003/sig000000ad )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000015e  (
    .I0(\blk00000003/sig00000117 ),
    .O(\blk00000003/sig000000ab )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000015d  (
    .I0(\blk00000003/sig0000011b ),
    .O(\blk00000003/sig000000a9 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000015c  (
    .I0(\blk00000003/sig0000011f ),
    .O(\blk00000003/sig000000a7 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000015b  (
    .I0(\blk00000003/sig00000123 ),
    .O(\blk00000003/sig000000a5 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk0000015a  (
    .I0(\blk00000003/sig00000127 ),
    .O(\blk00000003/sig000000a3 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk00000159  (
    .I0(\blk00000003/sig0000012b ),
    .O(\blk00000003/sig000000a1 )
  );
  LUT1 #(
    .INIT ( 2'h2 ))
  \blk00000003/blk00000158  (
    .I0(\blk00000003/sig0000012e ),
    .O(\blk00000003/sig0000009f )
  );
  LUT3 #(
    .INIT ( 8'h27 ))
  \blk00000003/blk00000157  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001b7 ),
    .I2(\blk00000003/sig000001af ),
    .O(\blk00000003/sig000000ae )
  );
  LUT3 #(
    .INIT ( 8'h6C ))
  \blk00000003/blk00000156  (
    .I0(\blk00000003/sig00000131 ),
    .I1(\blk00000003/sig00000140 ),
    .I2(\blk00000003/sig000001ce ),
    .O(\blk00000003/sig00000155 )
  );
  LUT3 #(
    .INIT ( 8'hEA ))
  \blk00000003/blk00000155  (
    .I0(\blk00000003/sig000001da ),
    .I1(\blk00000003/sig000001d2 ),
    .I2(\blk00000003/sig000001d7 ),
    .O(\blk00000003/sig000001e6 )
  );
  LUT4 #(
    .INIT ( 16'h4062 ))
  \blk00000003/blk00000154  (
    .I0(\blk00000003/sig00000155 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig000001de ),
    .I3(\blk00000003/sig000001b9 ),
    .O(\blk00000003/sig000001e5 )
  );
  LUT4 #(
    .INIT ( 16'hFBBB ))
  \blk00000003/blk00000153  (
    .I0(\blk00000003/sig000001db ),
    .I1(\blk00000003/sig0000015a ),
    .I2(\blk00000003/sig000001c1 ),
    .I3(\blk00000003/sig00000148 ),
    .O(\blk00000003/sig000001e4 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000152  (
    .I0(sig00000010),
    .I1(sig00000020),
    .O(\blk00000003/sig000000db )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000151  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig00000010),
    .I2(sig00000020),
    .O(\blk00000003/sig000001dd )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000150  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig0000000f),
    .I2(sig0000001f),
    .O(\blk00000003/sig000001d5 )
  );
  LUT4 #(
    .INIT ( 16'h3222 ))
  \blk00000003/blk0000014f  (
    .I0(\blk00000003/sig000001da ),
    .I1(\blk00000003/sig000001db ),
    .I2(\blk00000003/sig000001d7 ),
    .I3(\blk00000003/sig000001d2 ),
    .O(\blk00000003/sig000001e3 )
  );
  LUT3 #(
    .INIT ( 8'h6C ))
  \blk00000003/blk0000014e  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig00000146 ),
    .I2(\blk00000003/sig00000131 ),
    .O(\blk00000003/sig000001db )
  );
  LUT4 #(
    .INIT ( 16'hEF4F ))
  \blk00000003/blk0000014d  (
    .I0(\blk00000003/sig000001db ),
    .I1(\blk00000003/sig000001e1 ),
    .I2(\blk00000003/sig00000158 ),
    .I3(\blk00000003/sig000001e0 ),
    .O(\blk00000003/sig000001e2 )
  );
  LUT4 #(
    .INIT ( 16'hFFEA ))
  \blk00000003/blk0000014c  (
    .I0(\blk00000003/sig000001d8 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig000001c0 ),
    .I3(\blk00000003/sig000001d7 ),
    .O(\blk00000003/sig000001e0 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000014b  (
    .I0(sig0000000f),
    .I1(sig0000001f),
    .O(\blk00000003/sig000000da )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk0000014a  (
    .I0(\blk00000003/sig000001d7 ),
    .I1(\blk00000003/sig000001d8 ),
    .I2(\blk00000003/sig000001c0 ),
    .I3(\blk00000003/sig000001b8 ),
    .O(\blk00000003/sig00000159 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000149  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig0000000e),
    .I2(sig0000001e),
    .O(\blk00000003/sig000001d7 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000148  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig0000000d),
    .I2(sig0000001d),
    .O(\blk00000003/sig000001d8 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000147  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig0000000c),
    .I2(sig0000001c),
    .O(\blk00000003/sig000001c0 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000146  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig0000000b),
    .I2(sig0000001b),
    .O(\blk00000003/sig000001b8 )
  );
  LUT4 #(
    .INIT ( 16'h4BB4 ))
  \blk00000003/blk00000145  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig000001df ),
    .I2(\blk00000003/sig00000177 ),
    .I3(\blk00000003/sig00000172 ),
    .O(\blk00000003/sig00000176 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000144  (
    .I0(sig0000000e),
    .I1(sig0000001e),
    .O(\blk00000003/sig000000d8 )
  );
  LUT4 #(
    .INIT ( 16'h666A ))
  \blk00000003/blk00000143  (
    .I0(\blk00000003/sig00000143 ),
    .I1(\blk00000003/sig00000131 ),
    .I2(\blk00000003/sig00000148 ),
    .I3(\blk00000003/sig00000146 ),
    .O(\blk00000003/sig000001b9 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000142  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig000001dd ),
    .I2(\blk00000003/sig000001c0 ),
    .O(\blk00000003/sig000001de )
  );
  LUT3 #(
    .INIT ( 8'hFE ))
  \blk00000003/blk00000141  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig00000146 ),
    .I2(\blk00000003/sig00000143 ),
    .O(\blk00000003/sig000001ce )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000140  (
    .I0(sig0000000d),
    .I1(sig0000001d),
    .O(\blk00000003/sig000000d6 )
  );
  LUT3 #(
    .INIT ( 8'h08 ))
  \blk00000003/blk0000013f  (
    .I0(\blk00000003/sig000001d2 ),
    .I1(\blk00000003/sig000001d5 ),
    .I2(\blk00000003/sig000001db ),
    .O(\blk00000003/sig000001dc )
  );
  LUT4 #(
    .INIT ( 16'h0080 ))
  \blk00000003/blk0000013e  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig000001c1 ),
    .I2(\blk00000003/sig00000148 ),
    .I3(\blk00000003/sig00000155 ),
    .O(\blk00000003/sig000001da )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000013d  (
    .I0(sig0000000c),
    .I1(sig0000001c),
    .O(\blk00000003/sig000000d4 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000013c  (
    .I0(sig0000000b),
    .I1(sig0000001b),
    .O(\blk00000003/sig000000d2 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000013b  (
    .I0(sig0000000a),
    .I1(sig0000001a),
    .O(\blk00000003/sig000000d0 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000013a  (
    .I0(sig00000009),
    .I1(sig00000019),
    .O(\blk00000003/sig000000ce )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000139  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig000001d7 ),
    .I2(\blk00000003/sig000001d8 ),
    .O(\blk00000003/sig000001d9 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk00000138  (
    .I0(\blk00000003/sig000001b9 ),
    .I1(\blk00000003/sig000001d5 ),
    .I2(\blk00000003/sig000001b8 ),
    .O(\blk00000003/sig000001d6 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000137  (
    .I0(sig00000008),
    .I1(sig00000018),
    .O(\blk00000003/sig000000cc )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000136  (
    .I0(sig00000007),
    .I1(sig00000017),
    .O(\blk00000003/sig000000ca )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000135  (
    .I0(sig00000006),
    .I1(sig00000016),
    .O(\blk00000003/sig000000c8 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000134  (
    .I0(sig00000005),
    .I1(sig00000015),
    .O(\blk00000003/sig000000c6 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000133  (
    .I0(sig00000004),
    .I1(sig00000014),
    .O(\blk00000003/sig000000c4 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000132  (
    .I0(sig00000003),
    .I1(sig00000013),
    .O(\blk00000003/sig000000c2 )
  );
  LUT4 #(
    .INIT ( 16'h4BB4 ))
  \blk00000003/blk00000131  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig000001d4 ),
    .I2(\blk00000003/sig00000191 ),
    .I3(\blk00000003/sig00000172 ),
    .O(\blk00000003/sig00000190 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000130  (
    .I0(sig00000002),
    .I1(sig00000012),
    .O(\blk00000003/sig000000c0 )
  );
  LUT4 #(
    .INIT ( 16'h4BB4 ))
  \blk00000003/blk0000012f  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig000001d3 ),
    .I2(\blk00000003/sig00000189 ),
    .I3(\blk00000003/sig00000172 ),
    .O(\blk00000003/sig00000188 )
  );
  LUT4 #(
    .INIT ( 16'hF888 ))
  \blk00000003/blk0000012e  (
    .I0(\blk00000003/sig000001d2 ),
    .I1(\blk00000003/sig000001c4 ),
    .I2(\blk00000003/sig000001c1 ),
    .I3(\blk00000003/sig000001b5 ),
    .O(\blk00000003/sig000001d3 )
  );
  LUT4 #(
    .INIT ( 16'hFFEA ))
  \blk00000003/blk0000012d  (
    .I0(\blk00000003/sig000001d0 ),
    .I1(\blk00000003/sig00000155 ),
    .I2(\blk00000003/sig000001d1 ),
    .I3(\blk00000003/sig000001cf ),
    .O(\blk00000003/sig000001b4 )
  );
  LUT4 #(
    .INIT ( 16'h222A ))
  \blk00000003/blk0000012c  (
    .I0(\blk00000003/sig00000134 ),
    .I1(\blk00000003/sig00000131 ),
    .I2(\blk00000003/sig000001ce ),
    .I3(\blk00000003/sig00000140 ),
    .O(\blk00000003/sig000001cf )
  );
  LUT4 #(
    .INIT ( 16'hFFFE ))
  \blk00000003/blk0000012b  (
    .I0(\blk00000003/sig00000146 ),
    .I1(\blk00000003/sig00000148 ),
    .I2(\blk00000003/sig00000140 ),
    .I3(\blk00000003/sig00000143 ),
    .O(\blk00000003/sig000001cd )
  );
  LUT3 #(
    .INIT ( 8'h4E ))
  \blk00000003/blk0000012a  (
    .I0(\blk00000003/sig00000137 ),
    .I1(\blk00000003/sig0000013a ),
    .I2(\blk00000003/sig00000134 ),
    .O(\blk00000003/sig000001cc )
  );
  LUT4 #(
    .INIT ( 16'h01F1 ))
  \blk00000003/blk00000129  (
    .I0(\blk00000003/sig00000167 ),
    .I1(\blk00000003/sig000001ca ),
    .I2(\blk00000003/sig00000096 ),
    .I3(\blk00000003/sig000001cb ),
    .O(\blk00000003/sig0000008d )
  );
  LUT4 #(
    .INIT ( 16'hBBAB ))
  \blk00000003/blk00000128  (
    .I0(\blk00000003/sig0000016f ),
    .I1(\blk00000003/sig00000171 ),
    .I2(\blk00000003/sig0000015f ),
    .I3(\blk00000003/sig0000015d ),
    .O(\blk00000003/sig000001cb )
  );
  LUT3 #(
    .INIT ( 8'h31 ))
  \blk00000003/blk00000127  (
    .I0(\blk00000003/sig0000016d ),
    .I1(\blk00000003/sig00000169 ),
    .I2(\blk00000003/sig0000016b ),
    .O(\blk00000003/sig000001ca )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000126  (
    .I0(\blk00000003/sig0000016d ),
    .I1(\blk00000003/sig0000016b ),
    .I2(\blk00000003/sig00000167 ),
    .I3(\blk00000003/sig00000169 ),
    .O(\blk00000003/sig00000095 )
  );
  LUT3 #(
    .INIT ( 8'h04 ))
  \blk00000003/blk00000125  (
    .I0(\blk00000003/sig00000161 ),
    .I1(\blk00000003/sig00000163 ),
    .I2(\blk00000003/sig00000094 ),
    .O(\blk00000003/sig0000008e )
  );
  LUT3 #(
    .INIT ( 8'h01 ))
  \blk00000003/blk00000124  (
    .I0(\blk00000003/sig00000165 ),
    .I1(\blk00000003/sig00000161 ),
    .I2(\blk00000003/sig00000163 ),
    .O(\blk00000003/sig00000093 )
  );
  LUT4 #(
    .INIT ( 16'hFFD8 ))
  \blk00000003/blk00000123  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig0000001a),
    .I2(sig0000000a),
    .I3(\blk00000003/sig00000148 ),
    .O(\blk00000003/sig000001c9 )
  );
  LUT4 #(
    .INIT ( 16'h01F1 ))
  \blk00000003/blk00000122  (
    .I0(\blk00000003/sig00000169 ),
    .I1(\blk00000003/sig000001c7 ),
    .I2(\blk00000003/sig00000096 ),
    .I3(\blk00000003/sig000001c8 ),
    .O(\blk00000003/sig00000090 )
  );
  LUT4 #(
    .INIT ( 16'hFFAB ))
  \blk00000003/blk00000121  (
    .I0(\blk00000003/sig0000016f ),
    .I1(\blk00000003/sig0000015d ),
    .I2(\blk00000003/sig0000015f ),
    .I3(\blk00000003/sig00000171 ),
    .O(\blk00000003/sig000001c8 )
  );
  LUT3 #(
    .INIT ( 8'hF1 ))
  \blk00000003/blk00000120  (
    .I0(\blk00000003/sig0000016b ),
    .I1(\blk00000003/sig0000016d ),
    .I2(\blk00000003/sig00000167 ),
    .O(\blk00000003/sig000001c7 )
  );
  LUT4 #(
    .INIT ( 16'h0004 ))
  \blk00000003/blk0000011f  (
    .I0(\blk00000003/sig00000161 ),
    .I1(\blk00000003/sig00000165 ),
    .I2(\blk00000003/sig00000094 ),
    .I3(\blk00000003/sig00000163 ),
    .O(\blk00000003/sig00000091 )
  );
  LUT3 #(
    .INIT ( 8'hE4 ))
  \blk00000003/blk0000011e  (
    .I0(\blk00000003/sig000000be ),
    .I1(sig0000000a),
    .I2(sig0000001a),
    .O(\blk00000003/sig000001c1 )
  );
  LUT3 #(
    .INIT ( 8'h27 ))
  \blk00000003/blk0000011d  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001a3 ),
    .I2(\blk00000003/sig000001c6 ),
    .O(\blk00000003/sig000000b9 )
  );
  LUT3 #(
    .INIT ( 8'hAC ))
  \blk00000003/blk0000011c  (
    .I0(\blk00000003/sig00000094 ),
    .I1(\blk00000003/sig00000096 ),
    .I2(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig000001ac )
  );
  LUT3 #(
    .INIT ( 8'h4E ))
  \blk00000003/blk0000011b  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001c5 ),
    .I2(\blk00000003/sig000001c6 ),
    .O(\blk00000003/sig000000bb )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk0000011a  (
    .I0(\blk00000003/sig0000016f ),
    .I1(\blk00000003/sig0000015d ),
    .I2(\blk00000003/sig0000015f ),
    .I3(\blk00000003/sig00000171 ),
    .O(\blk00000003/sig00000097 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000119  (
    .I0(\blk00000003/sig00000146 ),
    .I1(\blk00000003/sig00000131 ),
    .O(\blk00000003/sig000001c4 )
  );
  LUT4 #(
    .INIT ( 16'h0213 ))
  \blk00000003/blk00000118  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig00000155 ),
    .I2(\blk00000003/sig000001c3 ),
    .I3(\blk00000003/sig000001c2 ),
    .O(\blk00000003/sig000001bf )
  );
  LUT4 #(
    .INIT ( 16'hFF41 ))
  \blk00000003/blk00000117  (
    .I0(\blk00000003/sig000001b8 ),
    .I1(\blk00000003/sig00000131 ),
    .I2(\blk00000003/sig00000146 ),
    .I3(\blk00000003/sig000001b9 ),
    .O(\blk00000003/sig000001c3 )
  );
  LUT4 #(
    .INIT ( 16'hFF1B ))
  \blk00000003/blk00000116  (
    .I0(\blk00000003/sig00000146 ),
    .I1(\blk00000003/sig000001c0 ),
    .I2(\blk00000003/sig000001c1 ),
    .I3(\blk00000003/sig000001b9 ),
    .O(\blk00000003/sig000001c2 )
  );
  LUT4 #(
    .INIT ( 16'h4BB4 ))
  \blk00000003/blk00000115  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig000001bf ),
    .I2(\blk00000003/sig0000018f ),
    .I3(\blk00000003/sig00000172 ),
    .O(\blk00000003/sig0000018e )
  );
  LUT4 #(
    .INIT ( 16'hFEAE ))
  \blk00000003/blk00000114  (
    .I0(\blk00000003/sig0000008c ),
    .I1(\blk00000003/sig000001bd ),
    .I2(\blk00000003/sig00000092 ),
    .I3(\blk00000003/sig000001be ),
    .O(\blk00000003/sig000001b7 )
  );
  LUT3 #(
    .INIT ( 8'h27 ))
  \blk00000003/blk00000113  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001b7 ),
    .I2(\blk00000003/sig000001af ),
    .O(\blk00000003/sig0000009d )
  );
  LUT4 #(
    .INIT ( 16'h4BB4 ))
  \blk00000003/blk00000112  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig000001bc ),
    .I2(\blk00000003/sig0000018c ),
    .I3(\blk00000003/sig00000172 ),
    .O(\blk00000003/sig0000018b )
  );
  LUT4 #(
    .INIT ( 16'h0213 ))
  \blk00000003/blk00000111  (
    .I0(\blk00000003/sig00000148 ),
    .I1(\blk00000003/sig00000155 ),
    .I2(\blk00000003/sig000001bb ),
    .I3(\blk00000003/sig000001ba ),
    .O(\blk00000003/sig000001bc )
  );
  LUT3 #(
    .INIT ( 8'hF1 ))
  \blk00000003/blk00000110  (
    .I0(\blk00000003/sig000001b8 ),
    .I1(\blk00000003/sig00000146 ),
    .I2(\blk00000003/sig000001b9 ),
    .O(\blk00000003/sig000001ba )
  );
  LUT4 #(
    .INIT ( 16'h110F ))
  \blk00000003/blk0000010f  (
    .I0(\blk00000003/sig000001b6 ),
    .I1(\blk00000003/sig00000092 ),
    .I2(\blk00000003/sig000001b7 ),
    .I3(\blk00000003/sig0000008f ),
    .O(\blk00000003/sig00000099 )
  );
  LUT4 #(
    .INIT ( 16'hB44B ))
  \blk00000003/blk0000010e  (
    .I0(\blk00000003/sig000001b4 ),
    .I1(\blk00000003/sig000001b5 ),
    .I2(\blk00000003/sig0000019f ),
    .I3(\blk00000003/sig00000172 ),
    .O(\blk00000003/sig00000185 )
  );
  LUT3 #(
    .INIT ( 8'h80 ))
  \blk00000003/blk0000010d  (
    .I0(\blk00000003/sig000000f0 ),
    .I1(\blk00000003/sig000000e1 ),
    .I2(\blk00000003/sig000000de ),
    .O(\blk00000003/sig000001b3 )
  );
  LUT4 #(
    .INIT ( 16'hFF35 ))
  \blk00000003/blk0000010c  (
    .I0(\blk00000003/sig0000015f ),
    .I1(\blk00000003/sig00000163 ),
    .I2(\blk00000003/sig00000092 ),
    .I3(\blk00000003/sig000001b2 ),
    .O(\blk00000003/sig000001af )
  );
  LUT4 #(
    .INIT ( 16'hFCAC ))
  \blk00000003/blk0000010b  (
    .I0(\blk00000003/sig0000008c ),
    .I1(\blk00000003/sig000001b0 ),
    .I2(\blk00000003/sig00000092 ),
    .I3(\blk00000003/sig000001b1 ),
    .O(\blk00000003/sig000001ae )
  );
  LUT3 #(
    .INIT ( 8'h1B ))
  \blk00000003/blk0000010a  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001ae ),
    .I2(\blk00000003/sig000001af ),
    .O(\blk00000003/sig000000b1 )
  );
  LUT3 #(
    .INIT ( 8'h4E ))
  \blk00000003/blk00000109  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001ad ),
    .I2(\blk00000003/sig000001ae ),
    .O(\blk00000003/sig000000b3 )
  );
  LUT4 #(
    .INIT ( 16'h05C5 ))
  \blk00000003/blk00000108  (
    .I0(\blk00000003/sig000001ab ),
    .I1(\blk00000003/sig000001aa ),
    .I2(\blk00000003/sig00000092 ),
    .I3(\blk00000003/sig000001ac ),
    .O(\blk00000003/sig000001ad )
  );
  LUT2 #(
    .INIT ( 4'h2 ))
  \blk00000003/blk00000107  (
    .I0(\blk00000003/sig0000015f ),
    .I1(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig000001aa )
  );
  LUT3 #(
    .INIT ( 8'hAC ))
  \blk00000003/blk00000106  (
    .I0(\blk00000003/sig000001a8 ),
    .I1(\blk00000003/sig000001a9 ),
    .I2(\blk00000003/sig00000092 ),
    .O(\blk00000003/sig000001a2 )
  );
  LUT4 #(
    .INIT ( 16'hAAAE ))
  \blk00000003/blk00000105  (
    .I0(\blk00000003/sig0000014b ),
    .I1(\blk00000003/sig000001a6 ),
    .I2(\blk00000003/sig00000110 ),
    .I3(\blk00000003/sig000001a7 ),
    .O(\blk00000003/sig00000083 )
  );
  LUT3 #(
    .INIT ( 8'hAC ))
  \blk00000003/blk00000104  (
    .I0(\blk00000003/sig000001a4 ),
    .I1(\blk00000003/sig000001a5 ),
    .I2(\blk00000003/sig00000092 ),
    .O(\blk00000003/sig000001a3 )
  );
  LUT3 #(
    .INIT ( 8'h27 ))
  \blk00000003/blk00000103  (
    .I0(\blk00000003/sig0000008f ),
    .I1(\blk00000003/sig000001a2 ),
    .I2(\blk00000003/sig000001a3 ),
    .O(\blk00000003/sig000000b7 )
  );
  LUT3 #(
    .INIT ( 8'h84 ))
  \blk00000003/blk00000102  (
    .I0(\blk00000003/sig00000124 ),
    .I1(\blk00000003/sig000001a0 ),
    .I2(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig000001a1 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000101  (
    .I0(sig00000008),
    .I1(sig00000009),
    .I2(sig00000006),
    .I3(sig00000007),
    .O(\blk00000003/sig000000f7 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk00000100  (
    .I0(sig00000018),
    .I1(sig00000019),
    .I2(sig00000016),
    .I3(sig00000017),
    .O(\blk00000003/sig00000103 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk000000ff  (
    .I0(sig00000004),
    .I1(sig00000005),
    .I2(sig00000002),
    .I3(sig00000003),
    .O(\blk00000003/sig000000f9 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk000000fe  (
    .I0(sig00000014),
    .I1(sig00000015),
    .I2(sig00000012),
    .I3(sig00000013),
    .O(\blk00000003/sig00000105 )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk000000fd  (
    .I0(\blk00000003/sig0000019f ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig00000020),
    .I3(sig00000010),
    .O(\blk00000003/sig0000017e )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk000000fc  (
    .I0(\blk00000003/sig0000019f ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig0000001f),
    .I3(sig0000000f),
    .O(\blk00000003/sig0000017b )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk000000fb  (
    .I0(\blk00000003/sig0000019f ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig0000001e),
    .I3(sig0000000e),
    .O(\blk00000003/sig00000177 )
  );
  LUT2 #(
    .INIT ( 4'h8 ))
  \blk00000003/blk000000fa  (
    .I0(\blk00000003/sig000000fa ),
    .I1(\blk00000003/sig00000106 ),
    .O(\blk00000003/sig0000019f )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk000000f9  (
    .I0(\blk00000003/sig0000019f ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig0000001d),
    .I3(sig0000000d),
    .O(\blk00000003/sig00000191 )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk000000f8  (
    .I0(\blk00000003/sig0000019f ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig0000001c),
    .I3(sig0000000c),
    .O(\blk00000003/sig0000018f )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk000000f7  (
    .I0(\blk00000003/sig0000019f ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig0000001b),
    .I3(sig0000000b),
    .O(\blk00000003/sig0000018c )
  );
  LUT4 #(
    .INIT ( 16'h5410 ))
  \blk00000003/blk000000f6  (
    .I0(\blk00000003/sig0000019f ),
    .I1(\blk00000003/sig000000be ),
    .I2(sig0000001a),
    .I3(sig0000000a),
    .O(\blk00000003/sig00000189 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk000000f5  (
    .I0(sig00000011),
    .I1(sig00000026),
    .O(\blk00000003/sig0000019e )
  );
  LUT3 #(
    .INIT ( 8'h13 ))
  \blk00000003/blk000000f4  (
    .I0(\blk00000003/sig00000106 ),
    .I1(\blk00000003/sig000000fe ),
    .I2(\blk00000003/sig000000fa ),
    .O(\blk00000003/sig0000019d )
  );
  LUT4 #(
    .INIT ( 16'hFFC8 ))
  \blk00000003/blk000000f3  (
    .I0(\blk00000003/sig0000019a ),
    .I1(\blk00000003/sig00000102 ),
    .I2(\blk00000003/sig00000199 ),
    .I3(\blk00000003/sig0000019b ),
    .O(\blk00000003/sig0000019c )
  );
  LUT4 #(
    .INIT ( 16'h040C ))
  \blk00000003/blk000000f2  (
    .I0(\blk00000003/sig00000106 ),
    .I1(\blk00000003/sig000000fe ),
    .I2(\blk00000003/sig0000010a ),
    .I3(\blk00000003/sig000000fa ),
    .O(\blk00000003/sig00000199 )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk000000f1  (
    .I0(sig00000008),
    .I1(sig00000009),
    .I2(sig00000006),
    .I3(sig00000007),
    .O(\blk00000003/sig000000fb )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk000000f0  (
    .I0(sig00000018),
    .I1(sig00000019),
    .I2(sig00000016),
    .I3(sig00000017),
    .O(\blk00000003/sig00000107 )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk000000ef  (
    .I0(sig00000004),
    .I1(sig00000005),
    .I2(sig00000002),
    .I3(sig00000003),
    .O(\blk00000003/sig000000fd )
  );
  LUT4 #(
    .INIT ( 16'h8000 ))
  \blk00000003/blk000000ee  (
    .I0(sig00000014),
    .I1(sig00000015),
    .I2(sig00000012),
    .I3(sig00000013),
    .O(\blk00000003/sig00000109 )
  );
  LUT2 #(
    .INIT ( 4'h8 ))
  \blk00000003/blk000000ed  (
    .I0(\blk00000003/sig00000152 ),
    .I1(\blk00000003/sig00000167 ),
    .O(\blk00000003/sig00000198 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk000000ec  (
    .I0(sig0000001f),
    .I1(sig00000020),
    .I2(sig0000001d),
    .I3(sig0000001e),
    .O(\blk00000003/sig0000010b )
  );
  LUT3 #(
    .INIT ( 8'h01 ))
  \blk00000003/blk000000eb  (
    .I0(sig0000001a),
    .I1(sig0000001b),
    .I2(sig0000001c),
    .O(\blk00000003/sig0000010d )
  );
  LUT4 #(
    .INIT ( 16'hA8AA ))
  \blk00000003/blk000000ea  (
    .I0(\blk00000003/sig0000014f ),
    .I1(\blk00000003/sig0000014b ),
    .I2(\blk00000003/sig0000014d ),
    .I3(\blk00000003/sig00000197 ),
    .O(\blk00000003/sig0000007b )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk000000e9  (
    .I0(sig0000000f),
    .I1(sig00000010),
    .I2(sig0000000d),
    .I3(sig0000000e),
    .O(\blk00000003/sig000000ff )
  );
  LUT3 #(
    .INIT ( 8'h01 ))
  \blk00000003/blk000000e8  (
    .I0(sig0000000a),
    .I1(sig0000000b),
    .I2(sig0000000c),
    .O(\blk00000003/sig00000101 )
  );
  LUT4 #(
    .INIT ( 16'h040C ))
  \blk00000003/blk000000e7  (
    .I0(\blk00000003/sig000000fa ),
    .I1(\blk00000003/sig00000172 ),
    .I2(\blk00000003/sig00000196 ),
    .I3(\blk00000003/sig00000106 ),
    .O(\blk00000003/sig00000149 )
  );
  LUT4 #(
    .INIT ( 16'h7FFF ))
  \blk00000003/blk000000e6  (
    .I0(\blk00000003/sig0000010a ),
    .I1(\blk00000003/sig000000fe ),
    .I2(\blk00000003/sig0000010e ),
    .I3(\blk00000003/sig00000102 ),
    .O(\blk00000003/sig00000196 )
  );
  LUT4 #(
    .INIT ( 16'h5F4C ))
  \blk00000003/blk000000e5  (
    .I0(\blk00000003/sig000000fa ),
    .I1(\blk00000003/sig000000fe ),
    .I2(\blk00000003/sig00000106 ),
    .I3(\blk00000003/sig0000010a ),
    .O(\blk00000003/sig0000014a )
  );
  LUT2 #(
    .INIT ( 4'h2 ))
  \blk00000003/blk000000e4  (
    .I0(\blk00000003/sig0000014b ),
    .I1(\blk00000003/sig0000014d ),
    .O(\blk00000003/sig00000078 )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk000000e3  (
    .I0(sig0000000e),
    .I1(sig0000000f),
    .I2(sig0000000c),
    .I3(sig0000000d),
    .O(\blk00000003/sig0000006f )
  );
  LUT4 #(
    .INIT ( 16'h0001 ))
  \blk00000003/blk000000e2  (
    .I0(sig0000001e),
    .I1(sig0000001f),
    .I2(sig0000001c),
    .I3(sig0000001d),
    .O(\blk00000003/sig00000074 )
  );
  LUT2 #(
    .INIT ( 4'h8 ))
  \blk00000003/blk000000e1  (
    .I0(\blk00000003/sig00000096 ),
    .I1(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig0000008b )
  );
  LUT2 #(
    .INIT ( 4'h1 ))
  \blk00000003/blk000000e0  (
    .I0(sig0000000b),
    .I1(sig0000000a),
    .O(\blk00000003/sig0000006d )
  );
  LUT2 #(
    .INIT ( 4'h1 ))
  \blk00000003/blk000000df  (
    .I0(sig0000001b),
    .I1(sig0000001a),
    .O(\blk00000003/sig00000072 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000de  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000156 ),
    .Q(\blk00000003/sig00000195 )
  );
  MUXCY   \blk00000003/blk000000dc  (
    .CI(\blk00000003/sig00000178 ),
    .DI(\blk00000003/sig00000191 ),
    .S(\blk00000003/sig00000190 ),
    .O(\blk00000003/sig0000018d )
  );
  XORCY   \blk00000003/blk000000db  (
    .CI(\blk00000003/sig00000178 ),
    .LI(\blk00000003/sig00000190 ),
    .O(\blk00000003/sig00000170 )
  );
  MUXCY   \blk00000003/blk000000da  (
    .CI(\blk00000003/sig0000018d ),
    .DI(\blk00000003/sig0000018f ),
    .S(\blk00000003/sig0000018e ),
    .O(\blk00000003/sig0000018a )
  );
  XORCY   \blk00000003/blk000000d9  (
    .CI(\blk00000003/sig0000018d ),
    .LI(\blk00000003/sig0000018e ),
    .O(\blk00000003/sig0000016e )
  );
  MUXCY   \blk00000003/blk000000d8  (
    .CI(\blk00000003/sig0000018a ),
    .DI(\blk00000003/sig0000018c ),
    .S(\blk00000003/sig0000018b ),
    .O(\blk00000003/sig00000187 )
  );
  XORCY   \blk00000003/blk000000d7  (
    .CI(\blk00000003/sig0000018a ),
    .LI(\blk00000003/sig0000018b ),
    .O(\blk00000003/sig0000016c )
  );
  MUXCY   \blk00000003/blk000000d6  (
    .CI(\blk00000003/sig00000187 ),
    .DI(\blk00000003/sig00000189 ),
    .S(\blk00000003/sig00000188 ),
    .O(\blk00000003/sig00000184 )
  );
  XORCY   \blk00000003/blk000000d5  (
    .CI(\blk00000003/sig00000187 ),
    .LI(\blk00000003/sig00000188 ),
    .O(\blk00000003/sig0000016a )
  );
  MUXCY   \blk00000003/blk000000d4  (
    .CI(\blk00000003/sig00000184 ),
    .DI(\blk00000003/sig00000186 ),
    .S(\blk00000003/sig00000185 ),
    .O(\blk00000003/sig00000182 )
  );
  XORCY   \blk00000003/blk000000d3  (
    .CI(\blk00000003/sig00000184 ),
    .LI(\blk00000003/sig00000185 ),
    .O(\blk00000003/sig00000168 )
  );
  MUXCY   \blk00000003/blk000000d2  (
    .CI(\blk00000003/sig00000182 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000183 ),
    .O(\NLW_blk00000003/blk000000d2_O_UNCONNECTED )
  );
  XORCY   \blk00000003/blk000000d1  (
    .CI(\blk00000003/sig00000182 ),
    .LI(\blk00000003/sig00000183 ),
    .O(\blk00000003/sig00000166 )
  );
  MUXCY   \blk00000003/blk000000d0  (
    .CI(\blk00000003/sig00000174 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000181 ),
    .O(\blk00000003/sig0000017f )
  );
  XORCY   \blk00000003/blk000000cf  (
    .CI(\blk00000003/sig00000174 ),
    .LI(\blk00000003/sig00000181 ),
    .O(\blk00000003/sig00000164 )
  );
  MUXCY   \blk00000003/blk000000ce  (
    .CI(\blk00000003/sig0000017f ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000180 ),
    .O(\blk00000003/sig0000017c )
  );
  XORCY   \blk00000003/blk000000cd  (
    .CI(\blk00000003/sig0000017f ),
    .LI(\blk00000003/sig00000180 ),
    .O(\blk00000003/sig00000162 )
  );
  MUXCY   \blk00000003/blk000000cc  (
    .CI(\blk00000003/sig0000017c ),
    .DI(\blk00000003/sig0000017e ),
    .S(\blk00000003/sig0000017d ),
    .O(\blk00000003/sig00000179 )
  );
  XORCY   \blk00000003/blk000000cb  (
    .CI(\blk00000003/sig0000017c ),
    .LI(\blk00000003/sig0000017d ),
    .O(\blk00000003/sig00000160 )
  );
  MUXCY   \blk00000003/blk000000ca  (
    .CI(\blk00000003/sig00000179 ),
    .DI(\blk00000003/sig0000017b ),
    .S(\blk00000003/sig0000017a ),
    .O(\blk00000003/sig00000175 )
  );
  XORCY   \blk00000003/blk000000c9  (
    .CI(\blk00000003/sig00000179 ),
    .LI(\blk00000003/sig0000017a ),
    .O(\blk00000003/sig0000015e )
  );
  MUXCY   \blk00000003/blk000000c8  (
    .CI(\blk00000003/sig00000175 ),
    .DI(\blk00000003/sig00000177 ),
    .S(\blk00000003/sig00000176 ),
    .O(\blk00000003/sig00000178 )
  );
  XORCY   \blk00000003/blk000000c7  (
    .CI(\blk00000003/sig00000175 ),
    .LI(\blk00000003/sig00000176 ),
    .O(\blk00000003/sig0000015c )
  );
  MUXCY   \blk00000003/blk000000c6  (
    .CI(\blk00000003/sig00000173 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000156 ),
    .O(\blk00000003/sig00000174 )
  );
  MUXCY   \blk00000003/blk000000c5  (
    .CI(\blk00000003/sig00000172 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000003a ),
    .O(\blk00000003/sig00000173 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000c4  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000170 ),
    .Q(\blk00000003/sig00000171 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000c3  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000016e ),
    .Q(\blk00000003/sig0000016f )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000c2  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000016c ),
    .Q(\blk00000003/sig0000016d )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000c1  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000016a ),
    .Q(\blk00000003/sig0000016b )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000c0  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000168 ),
    .Q(\blk00000003/sig00000169 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000bf  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000166 ),
    .Q(\blk00000003/sig00000167 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000be  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000164 ),
    .Q(\blk00000003/sig00000165 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000bd  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000162 ),
    .Q(\blk00000003/sig00000163 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000bc  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000160 ),
    .Q(\blk00000003/sig00000161 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000bb  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000015e ),
    .Q(\blk00000003/sig0000015f )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000ba  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000015c ),
    .Q(\blk00000003/sig0000015d )
  );
  MUXCY   \blk00000003/blk000000b9  (
    .CI(\blk00000003/sig0000015a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig0000015b )
  );
  MUXCY   \blk00000003/blk000000b8  (
    .CI(\blk00000003/sig00000158 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000159 ),
    .O(\blk00000003/sig0000015a )
  );
  MUXCY   \blk00000003/blk000000b7  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000157 ),
    .O(\blk00000003/sig00000158 )
  );
  MUXF5   \blk00000003/blk000000b6  (
    .I0(\blk00000003/sig00000153 ),
    .I1(\blk00000003/sig00000154 ),
    .S(\blk00000003/sig00000155 ),
    .O(\blk00000003/sig00000156 )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000b5  (
    .C(sig00000027),
    .D(\blk00000003/sig000000f3 ),
    .Q(\blk00000003/sig00000114 )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000b4  (
    .C(sig00000027),
    .D(\blk00000003/sig000000f0 ),
    .Q(\blk00000003/sig00000118 )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000b3  (
    .C(sig00000027),
    .D(\blk00000003/sig000000ed ),
    .Q(\blk00000003/sig0000011c )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000b2  (
    .C(sig00000027),
    .D(\blk00000003/sig000000ea ),
    .Q(\blk00000003/sig00000120 )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000b1  (
    .C(sig00000027),
    .D(\blk00000003/sig000000e7 ),
    .Q(\blk00000003/sig00000124 )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000b0  (
    .C(sig00000027),
    .D(\blk00000003/sig000000e4 ),
    .Q(\blk00000003/sig00000128 )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000af  (
    .C(sig00000027),
    .D(\blk00000003/sig000000e1 ),
    .Q(\blk00000003/sig0000012c )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000ae  (
    .C(sig00000027),
    .D(\blk00000003/sig000000de ),
    .Q(\blk00000003/sig0000012f )
  );
  FD #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000ad  (
    .C(sig00000027),
    .D(\blk00000003/sig00000151 ),
    .Q(\blk00000003/sig00000152 )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000ac  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000150 ),
    .Q(\NLW_blk00000003/blk000000ac_Q_UNCONNECTED )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000ab  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000014e ),
    .Q(\blk00000003/sig0000014f )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000aa  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000014c ),
    .Q(\blk00000003/sig0000014d )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000a9  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig0000014a ),
    .Q(\blk00000003/sig0000014b )
  );
  FDE #(
    .INIT ( 1'b0 ))
  \blk00000003/blk000000a8  (
    .C(sig00000027),
    .CE(\blk00000003/sig0000003a ),
    .D(\blk00000003/sig00000149 ),
    .Q(\NLW_blk00000003/blk000000a8_Q_UNCONNECTED )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk000000a7  (
    .I0(sig00000019),
    .I1(sig00000009),
    .O(\blk00000003/sig00000147 )
  );
  MUXCY   \blk00000003/blk000000a6  (
    .CI(\blk00000003/sig0000003a ),
    .DI(sig00000019),
    .S(\blk00000003/sig00000147 ),
    .O(\blk00000003/sig00000144 )
  );
  XORCY   \blk00000003/blk000000a5  (
    .CI(\blk00000003/sig0000003a ),
    .LI(\blk00000003/sig00000147 ),
    .O(\blk00000003/sig00000148 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk000000a4  (
    .I0(sig00000018),
    .I1(sig00000008),
    .O(\blk00000003/sig00000145 )
  );
  MUXCY   \blk00000003/blk000000a3  (
    .CI(\blk00000003/sig00000144 ),
    .DI(sig00000018),
    .S(\blk00000003/sig00000145 ),
    .O(\blk00000003/sig00000141 )
  );
  XORCY   \blk00000003/blk000000a2  (
    .CI(\blk00000003/sig00000144 ),
    .LI(\blk00000003/sig00000145 ),
    .O(\blk00000003/sig00000146 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk000000a1  (
    .I0(sig00000017),
    .I1(sig00000007),
    .O(\blk00000003/sig00000142 )
  );
  MUXCY   \blk00000003/blk000000a0  (
    .CI(\blk00000003/sig00000141 ),
    .DI(sig00000017),
    .S(\blk00000003/sig00000142 ),
    .O(\blk00000003/sig0000013e )
  );
  XORCY   \blk00000003/blk0000009f  (
    .CI(\blk00000003/sig00000141 ),
    .LI(\blk00000003/sig00000142 ),
    .O(\blk00000003/sig00000143 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000009e  (
    .I0(sig00000016),
    .I1(sig00000006),
    .O(\blk00000003/sig0000013f )
  );
  MUXCY   \blk00000003/blk0000009d  (
    .CI(\blk00000003/sig0000013e ),
    .DI(sig00000016),
    .S(\blk00000003/sig0000013f ),
    .O(\blk00000003/sig0000013b )
  );
  XORCY   \blk00000003/blk0000009c  (
    .CI(\blk00000003/sig0000013e ),
    .LI(\blk00000003/sig0000013f ),
    .O(\blk00000003/sig00000140 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000009b  (
    .I0(sig00000015),
    .I1(sig00000005),
    .O(\blk00000003/sig0000013c )
  );
  MUXCY   \blk00000003/blk0000009a  (
    .CI(\blk00000003/sig0000013b ),
    .DI(sig00000015),
    .S(\blk00000003/sig0000013c ),
    .O(\blk00000003/sig00000138 )
  );
  XORCY   \blk00000003/blk00000099  (
    .CI(\blk00000003/sig0000013b ),
    .LI(\blk00000003/sig0000013c ),
    .O(\blk00000003/sig0000013d )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000098  (
    .I0(sig00000014),
    .I1(sig00000004),
    .O(\blk00000003/sig00000139 )
  );
  MUXCY   \blk00000003/blk00000097  (
    .CI(\blk00000003/sig00000138 ),
    .DI(sig00000014),
    .S(\blk00000003/sig00000139 ),
    .O(\blk00000003/sig00000135 )
  );
  XORCY   \blk00000003/blk00000096  (
    .CI(\blk00000003/sig00000138 ),
    .LI(\blk00000003/sig00000139 ),
    .O(\blk00000003/sig0000013a )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000095  (
    .I0(sig00000013),
    .I1(sig00000003),
    .O(\blk00000003/sig00000136 )
  );
  MUXCY   \blk00000003/blk00000094  (
    .CI(\blk00000003/sig00000135 ),
    .DI(sig00000013),
    .S(\blk00000003/sig00000136 ),
    .O(\blk00000003/sig00000132 )
  );
  XORCY   \blk00000003/blk00000093  (
    .CI(\blk00000003/sig00000135 ),
    .LI(\blk00000003/sig00000136 ),
    .O(\blk00000003/sig00000137 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000092  (
    .I0(sig00000012),
    .I1(sig00000002),
    .O(\blk00000003/sig00000133 )
  );
  MUXCY   \blk00000003/blk00000091  (
    .CI(\blk00000003/sig00000132 ),
    .DI(sig00000012),
    .S(\blk00000003/sig00000133 ),
    .O(\blk00000003/sig00000130 )
  );
  XORCY   \blk00000003/blk00000090  (
    .CI(\blk00000003/sig00000132 ),
    .LI(\blk00000003/sig00000133 ),
    .O(\blk00000003/sig00000134 )
  );
  XORCY   \blk00000003/blk0000008f  (
    .CI(\blk00000003/sig00000130 ),
    .LI(\blk00000003/sig0000003a ),
    .O(\blk00000003/sig00000131 )
  );
  MUXCY   \blk00000003/blk0000008e  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig0000012f ),
    .S(\blk00000003/sig0000012d ),
    .O(\blk00000003/sig00000129 )
  );
  XORCY   \blk00000003/blk0000008d  (
    .CI(\blk00000003/sig0000003a ),
    .LI(\blk00000003/sig0000012d ),
    .O(\blk00000003/sig0000012e )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk0000008c  (
    .I0(\blk00000003/sig0000012c ),
    .I1(\blk00000003/sig00000092 ),
    .O(\blk00000003/sig0000012a )
  );
  MUXCY   \blk00000003/blk0000008b  (
    .CI(\blk00000003/sig00000129 ),
    .DI(\blk00000003/sig0000012c ),
    .S(\blk00000003/sig0000012a ),
    .O(\blk00000003/sig00000125 )
  );
  XORCY   \blk00000003/blk0000008a  (
    .CI(\blk00000003/sig00000129 ),
    .LI(\blk00000003/sig0000012a ),
    .O(\blk00000003/sig0000012b )
  );
  MUXCY   \blk00000003/blk00000089  (
    .CI(\blk00000003/sig00000125 ),
    .DI(\blk00000003/sig00000128 ),
    .S(\blk00000003/sig00000126 ),
    .O(\blk00000003/sig00000121 )
  );
  XORCY   \blk00000003/blk00000088  (
    .CI(\blk00000003/sig00000125 ),
    .LI(\blk00000003/sig00000126 ),
    .O(\blk00000003/sig00000127 )
  );
  LUT2 #(
    .INIT ( 4'h9 ))
  \blk00000003/blk00000087  (
    .I0(\blk00000003/sig00000124 ),
    .I1(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig00000122 )
  );
  MUXCY   \blk00000003/blk00000086  (
    .CI(\blk00000003/sig00000121 ),
    .DI(\blk00000003/sig00000124 ),
    .S(\blk00000003/sig00000122 ),
    .O(\blk00000003/sig0000011d )
  );
  XORCY   \blk00000003/blk00000085  (
    .CI(\blk00000003/sig00000121 ),
    .LI(\blk00000003/sig00000122 ),
    .O(\blk00000003/sig00000123 )
  );
  MUXCY   \blk00000003/blk00000084  (
    .CI(\blk00000003/sig0000011d ),
    .DI(\blk00000003/sig00000120 ),
    .S(\blk00000003/sig0000011e ),
    .O(\blk00000003/sig00000119 )
  );
  XORCY   \blk00000003/blk00000083  (
    .CI(\blk00000003/sig0000011d ),
    .LI(\blk00000003/sig0000011e ),
    .O(\blk00000003/sig0000011f )
  );
  MUXCY   \blk00000003/blk00000082  (
    .CI(\blk00000003/sig00000119 ),
    .DI(\blk00000003/sig0000011c ),
    .S(\blk00000003/sig0000011a ),
    .O(\blk00000003/sig00000115 )
  );
  XORCY   \blk00000003/blk00000081  (
    .CI(\blk00000003/sig00000119 ),
    .LI(\blk00000003/sig0000011a ),
    .O(\blk00000003/sig0000011b )
  );
  MUXCY   \blk00000003/blk00000080  (
    .CI(\blk00000003/sig00000115 ),
    .DI(\blk00000003/sig00000118 ),
    .S(\blk00000003/sig00000116 ),
    .O(\blk00000003/sig00000111 )
  );
  XORCY   \blk00000003/blk0000007f  (
    .CI(\blk00000003/sig00000115 ),
    .LI(\blk00000003/sig00000116 ),
    .O(\blk00000003/sig00000117 )
  );
  MUXCY   \blk00000003/blk0000007e  (
    .CI(\blk00000003/sig00000111 ),
    .DI(\blk00000003/sig00000114 ),
    .S(\blk00000003/sig00000112 ),
    .O(\blk00000003/sig0000010f )
  );
  XORCY   \blk00000003/blk0000007d  (
    .CI(\blk00000003/sig00000111 ),
    .LI(\blk00000003/sig00000112 ),
    .O(\blk00000003/sig00000113 )
  );
  XORCY   \blk00000003/blk0000007c  (
    .CI(\blk00000003/sig0000010f ),
    .LI(\blk00000003/sig0000003a ),
    .O(\blk00000003/sig00000110 )
  );
  MUXCY   \blk00000003/blk0000007b  (
    .CI(\blk00000003/sig0000010c ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000010d ),
    .O(\blk00000003/sig0000010e )
  );
  MUXCY   \blk00000003/blk0000007a  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000010b ),
    .O(\blk00000003/sig0000010c )
  );
  MUXCY   \blk00000003/blk00000079  (
    .CI(\blk00000003/sig00000108 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000109 ),
    .O(\blk00000003/sig0000010a )
  );
  MUXCY   \blk00000003/blk00000078  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000107 ),
    .O(\blk00000003/sig00000108 )
  );
  MUXCY   \blk00000003/blk00000077  (
    .CI(\blk00000003/sig00000104 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000105 ),
    .O(\blk00000003/sig00000106 )
  );
  MUXCY   \blk00000003/blk00000076  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000103 ),
    .O(\blk00000003/sig00000104 )
  );
  MUXCY   \blk00000003/blk00000075  (
    .CI(\blk00000003/sig00000100 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000101 ),
    .O(\blk00000003/sig00000102 )
  );
  MUXCY   \blk00000003/blk00000074  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000ff ),
    .O(\blk00000003/sig00000100 )
  );
  MUXCY   \blk00000003/blk00000073  (
    .CI(\blk00000003/sig000000fc ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000fd ),
    .O(\blk00000003/sig000000fe )
  );
  MUXCY   \blk00000003/blk00000072  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000fb ),
    .O(\blk00000003/sig000000fc )
  );
  MUXCY   \blk00000003/blk00000071  (
    .CI(\blk00000003/sig000000f8 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000f9 ),
    .O(\blk00000003/sig000000fa )
  );
  MUXCY   \blk00000003/blk00000070  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000f7 ),
    .O(\blk00000003/sig000000f8 )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk0000006f  (
    .I0(sig00000009),
    .I1(sig00000019),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig0000003a ),
    .O(\blk00000003/sig000000dc )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk0000006e  (
    .I0(sig00000008),
    .I1(sig00000018),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000df )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk0000006d  (
    .I0(sig00000007),
    .I1(sig00000017),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000e2 )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk0000006c  (
    .I0(sig00000006),
    .I1(sig00000016),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000e5 )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk0000006b  (
    .I0(sig00000005),
    .I1(sig00000015),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000e8 )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk0000006a  (
    .I0(sig00000004),
    .I1(sig00000014),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000eb )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk00000069  (
    .I0(sig00000003),
    .I1(sig00000013),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000ee )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk00000068  (
    .I0(sig00000002),
    .I1(sig00000012),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000f1 )
  );
  LUT4 #(
    .INIT ( 16'h35CA ))
  \blk00000003/blk00000067  (
    .I0(\blk00000003/sig00000039 ),
    .I1(\blk00000003/sig00000039 ),
    .I2(\blk00000003/sig000000f6 ),
    .I3(\blk00000003/sig00000039 ),
    .O(\blk00000003/sig000000f4 )
  );
  XORCY   \blk00000003/blk00000066  (
    .CI(\blk00000003/sig000000f2 ),
    .LI(\blk00000003/sig000000f4 ),
    .O(\blk00000003/sig000000f5 )
  );
  MUXCY   \blk00000003/blk00000065  (
    .CI(\blk00000003/sig000000f2 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000f4 ),
    .O(\NLW_blk00000003/blk00000065_O_UNCONNECTED )
  );
  XORCY   \blk00000003/blk00000064  (
    .CI(\blk00000003/sig000000ef ),
    .LI(\blk00000003/sig000000f1 ),
    .O(\blk00000003/sig000000f3 )
  );
  MUXCY   \blk00000003/blk00000063  (
    .CI(\blk00000003/sig000000ef ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000f1 ),
    .O(\blk00000003/sig000000f2 )
  );
  XORCY   \blk00000003/blk00000062  (
    .CI(\blk00000003/sig000000ec ),
    .LI(\blk00000003/sig000000ee ),
    .O(\blk00000003/sig000000f0 )
  );
  MUXCY   \blk00000003/blk00000061  (
    .CI(\blk00000003/sig000000ec ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000ee ),
    .O(\blk00000003/sig000000ef )
  );
  XORCY   \blk00000003/blk00000060  (
    .CI(\blk00000003/sig000000e9 ),
    .LI(\blk00000003/sig000000eb ),
    .O(\blk00000003/sig000000ed )
  );
  MUXCY   \blk00000003/blk0000005f  (
    .CI(\blk00000003/sig000000e9 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000eb ),
    .O(\blk00000003/sig000000ec )
  );
  XORCY   \blk00000003/blk0000005e  (
    .CI(\blk00000003/sig000000e6 ),
    .LI(\blk00000003/sig000000e8 ),
    .O(\blk00000003/sig000000ea )
  );
  MUXCY   \blk00000003/blk0000005d  (
    .CI(\blk00000003/sig000000e6 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000e8 ),
    .O(\blk00000003/sig000000e9 )
  );
  XORCY   \blk00000003/blk0000005c  (
    .CI(\blk00000003/sig000000e3 ),
    .LI(\blk00000003/sig000000e5 ),
    .O(\blk00000003/sig000000e7 )
  );
  MUXCY   \blk00000003/blk0000005b  (
    .CI(\blk00000003/sig000000e3 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000e5 ),
    .O(\blk00000003/sig000000e6 )
  );
  XORCY   \blk00000003/blk0000005a  (
    .CI(\blk00000003/sig000000e0 ),
    .LI(\blk00000003/sig000000e2 ),
    .O(\blk00000003/sig000000e4 )
  );
  MUXCY   \blk00000003/blk00000059  (
    .CI(\blk00000003/sig000000e0 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000e2 ),
    .O(\blk00000003/sig000000e3 )
  );
  XORCY   \blk00000003/blk00000058  (
    .CI(\blk00000003/sig000000dd ),
    .LI(\blk00000003/sig000000df ),
    .O(\blk00000003/sig000000e1 )
  );
  MUXCY   \blk00000003/blk00000057  (
    .CI(\blk00000003/sig000000dd ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000df ),
    .O(\blk00000003/sig000000e0 )
  );
  XORCY   \blk00000003/blk00000056  (
    .CI(\blk00000003/sig00000039 ),
    .LI(\blk00000003/sig000000dc ),
    .O(\blk00000003/sig000000de )
  );
  MUXCY   \blk00000003/blk00000055  (
    .CI(\blk00000003/sig00000039 ),
    .DI(\blk00000003/sig0000003a ),
    .S(\blk00000003/sig000000dc ),
    .O(\blk00000003/sig000000dd )
  );
  MUXCY   \blk00000003/blk00000054  (
    .CI(\blk00000003/sig0000003a ),
    .DI(sig00000010),
    .S(\blk00000003/sig000000db ),
    .O(\blk00000003/sig000000d9 )
  );
  MUXCY   \blk00000003/blk00000053  (
    .CI(\blk00000003/sig000000d9 ),
    .DI(sig0000000f),
    .S(\blk00000003/sig000000da ),
    .O(\blk00000003/sig000000d7 )
  );
  MUXCY   \blk00000003/blk00000052  (
    .CI(\blk00000003/sig000000d7 ),
    .DI(sig0000000e),
    .S(\blk00000003/sig000000d8 ),
    .O(\blk00000003/sig000000d5 )
  );
  MUXCY   \blk00000003/blk00000051  (
    .CI(\blk00000003/sig000000d5 ),
    .DI(sig0000000d),
    .S(\blk00000003/sig000000d6 ),
    .O(\blk00000003/sig000000d3 )
  );
  MUXCY   \blk00000003/blk00000050  (
    .CI(\blk00000003/sig000000d3 ),
    .DI(sig0000000c),
    .S(\blk00000003/sig000000d4 ),
    .O(\blk00000003/sig000000d1 )
  );
  MUXCY   \blk00000003/blk0000004f  (
    .CI(\blk00000003/sig000000d1 ),
    .DI(sig0000000b),
    .S(\blk00000003/sig000000d2 ),
    .O(\blk00000003/sig000000cf )
  );
  MUXCY   \blk00000003/blk0000004e  (
    .CI(\blk00000003/sig000000cf ),
    .DI(sig0000000a),
    .S(\blk00000003/sig000000d0 ),
    .O(\blk00000003/sig000000cd )
  );
  MUXCY   \blk00000003/blk0000004d  (
    .CI(\blk00000003/sig000000cd ),
    .DI(sig00000009),
    .S(\blk00000003/sig000000ce ),
    .O(\blk00000003/sig000000cb )
  );
  MUXCY   \blk00000003/blk0000004c  (
    .CI(\blk00000003/sig000000cb ),
    .DI(sig00000008),
    .S(\blk00000003/sig000000cc ),
    .O(\blk00000003/sig000000c9 )
  );
  MUXCY   \blk00000003/blk0000004b  (
    .CI(\blk00000003/sig000000c9 ),
    .DI(sig00000007),
    .S(\blk00000003/sig000000ca ),
    .O(\blk00000003/sig000000c7 )
  );
  MUXCY   \blk00000003/blk0000004a  (
    .CI(\blk00000003/sig000000c7 ),
    .DI(sig00000006),
    .S(\blk00000003/sig000000c8 ),
    .O(\blk00000003/sig000000c5 )
  );
  MUXCY   \blk00000003/blk00000049  (
    .CI(\blk00000003/sig000000c5 ),
    .DI(sig00000005),
    .S(\blk00000003/sig000000c6 ),
    .O(\blk00000003/sig000000c3 )
  );
  MUXCY   \blk00000003/blk00000048  (
    .CI(\blk00000003/sig000000c3 ),
    .DI(sig00000004),
    .S(\blk00000003/sig000000c4 ),
    .O(\blk00000003/sig000000c1 )
  );
  MUXCY   \blk00000003/blk00000047  (
    .CI(\blk00000003/sig000000c1 ),
    .DI(sig00000003),
    .S(\blk00000003/sig000000c2 ),
    .O(\blk00000003/sig000000bf )
  );
  MUXCY   \blk00000003/blk00000046  (
    .CI(\blk00000003/sig000000bf ),
    .DI(sig00000002),
    .S(\blk00000003/sig000000c0 ),
    .O(\blk00000003/sig000000bd )
  );
  MUXCY   \blk00000003/blk00000045  (
    .CI(\blk00000003/sig000000bd ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000003a ),
    .O(\blk00000003/sig000000be )
  );
  XORCY   \blk00000003/blk00000044  (
    .CI(\blk00000003/sig000000bc ),
    .LI(\blk00000003/sig0000003a ),
    .O(\NLW_blk00000003/blk00000044_O_UNCONNECTED )
  );
  MUXCY   \blk00000003/blk00000043  (
    .CI(\blk00000003/sig000000bc ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000003a ),
    .O(\blk00000003/sig0000009e )
  );
  XORCY   \blk00000003/blk00000042  (
    .CI(\blk00000003/sig000000ba ),
    .LI(\blk00000003/sig000000bb ),
    .O(\blk00000003/sig00000076 )
  );
  MUXCY   \blk00000003/blk00000041  (
    .CI(\blk00000003/sig000000ba ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000bb ),
    .O(\blk00000003/sig000000bc )
  );
  XORCY   \blk00000003/blk00000040  (
    .CI(\blk00000003/sig000000b8 ),
    .LI(\blk00000003/sig000000b9 ),
    .O(\blk00000003/sig00000079 )
  );
  MUXCY   \blk00000003/blk0000003f  (
    .CI(\blk00000003/sig000000b8 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000b9 ),
    .O(\blk00000003/sig000000ba )
  );
  XORCY   \blk00000003/blk0000003e  (
    .CI(\blk00000003/sig000000b6 ),
    .LI(\blk00000003/sig000000b7 ),
    .O(\blk00000003/sig0000007e )
  );
  MUXCY   \blk00000003/blk0000003d  (
    .CI(\blk00000003/sig000000b6 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000b7 ),
    .O(\blk00000003/sig000000b8 )
  );
  XORCY   \blk00000003/blk0000003c  (
    .CI(\blk00000003/sig000000b4 ),
    .LI(\blk00000003/sig000000b5 ),
    .O(\blk00000003/sig0000007c )
  );
  MUXCY   \blk00000003/blk0000003b  (
    .CI(\blk00000003/sig000000b4 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000b5 ),
    .O(\blk00000003/sig000000b6 )
  );
  XORCY   \blk00000003/blk0000003a  (
    .CI(\blk00000003/sig000000b2 ),
    .LI(\blk00000003/sig000000b3 ),
    .O(\blk00000003/sig0000007d )
  );
  MUXCY   \blk00000003/blk00000039  (
    .CI(\blk00000003/sig000000b2 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000b3 ),
    .O(\blk00000003/sig000000b4 )
  );
  XORCY   \blk00000003/blk00000038  (
    .CI(\blk00000003/sig000000af ),
    .LI(\blk00000003/sig000000b1 ),
    .O(\blk00000003/sig00000080 )
  );
  MUXCY   \blk00000003/blk00000037  (
    .CI(\blk00000003/sig000000af ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000b1 ),
    .O(\blk00000003/sig000000b2 )
  );
  XORCY   \blk00000003/blk00000036  (
    .CI(\blk00000003/sig0000009a ),
    .LI(\blk00000003/sig000000b0 ),
    .O(\blk00000003/sig0000007f )
  );
  MUXCY   \blk00000003/blk00000035  (
    .CI(\blk00000003/sig0000009a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000ae ),
    .O(\blk00000003/sig000000af )
  );
  XORCY   \blk00000003/blk00000034  (
    .CI(\blk00000003/sig000000ac ),
    .LI(\blk00000003/sig000000ad ),
    .O(\blk00000003/sig00000081 )
  );
  MUXCY   \blk00000003/blk00000033  (
    .CI(\blk00000003/sig000000ac ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000ad ),
    .O(\NLW_blk00000003/blk00000033_O_UNCONNECTED )
  );
  XORCY   \blk00000003/blk00000032  (
    .CI(\blk00000003/sig000000aa ),
    .LI(\blk00000003/sig000000ab ),
    .O(\blk00000003/sig00000084 )
  );
  MUXCY   \blk00000003/blk00000031  (
    .CI(\blk00000003/sig000000aa ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000ab ),
    .O(\blk00000003/sig000000ac )
  );
  XORCY   \blk00000003/blk00000030  (
    .CI(\blk00000003/sig000000a8 ),
    .LI(\blk00000003/sig000000a9 ),
    .O(\blk00000003/sig00000085 )
  );
  MUXCY   \blk00000003/blk0000002f  (
    .CI(\blk00000003/sig000000a8 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000a9 ),
    .O(\blk00000003/sig000000aa )
  );
  XORCY   \blk00000003/blk0000002e  (
    .CI(\blk00000003/sig000000a6 ),
    .LI(\blk00000003/sig000000a7 ),
    .O(\blk00000003/sig00000086 )
  );
  MUXCY   \blk00000003/blk0000002d  (
    .CI(\blk00000003/sig000000a6 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000a7 ),
    .O(\blk00000003/sig000000a8 )
  );
  XORCY   \blk00000003/blk0000002c  (
    .CI(\blk00000003/sig000000a4 ),
    .LI(\blk00000003/sig000000a5 ),
    .O(\blk00000003/sig00000087 )
  );
  MUXCY   \blk00000003/blk0000002b  (
    .CI(\blk00000003/sig000000a4 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000a5 ),
    .O(\blk00000003/sig000000a6 )
  );
  XORCY   \blk00000003/blk0000002a  (
    .CI(\blk00000003/sig000000a2 ),
    .LI(\blk00000003/sig000000a3 ),
    .O(\blk00000003/sig00000088 )
  );
  MUXCY   \blk00000003/blk00000029  (
    .CI(\blk00000003/sig000000a2 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000a3 ),
    .O(\blk00000003/sig000000a4 )
  );
  XORCY   \blk00000003/blk00000028  (
    .CI(\blk00000003/sig000000a0 ),
    .LI(\blk00000003/sig000000a1 ),
    .O(\blk00000003/sig00000089 )
  );
  MUXCY   \blk00000003/blk00000027  (
    .CI(\blk00000003/sig000000a0 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig000000a1 ),
    .O(\blk00000003/sig000000a2 )
  );
  XORCY   \blk00000003/blk00000026  (
    .CI(\blk00000003/sig0000009e ),
    .LI(\blk00000003/sig0000009f ),
    .O(\blk00000003/sig0000008a )
  );
  MUXCY   \blk00000003/blk00000025  (
    .CI(\blk00000003/sig0000009e ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000009f ),
    .O(\blk00000003/sig000000a0 )
  );
  MUXCY   \blk00000003/blk00000024  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000009d ),
    .O(\blk00000003/sig0000009b )
  );
  MUXCY   \blk00000003/blk00000023  (
    .CI(\blk00000003/sig0000009b ),
    .DI(\blk00000003/sig0000003a ),
    .S(\blk00000003/sig0000009c ),
    .O(\blk00000003/sig00000098 )
  );
  MUXCY   \blk00000003/blk00000022  (
    .CI(\blk00000003/sig00000098 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000099 ),
    .O(\blk00000003/sig0000009a )
  );
  MUXCY   \blk00000003/blk00000021  (
    .CI(\blk00000003/sig00000096 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000097 ),
    .O(\blk00000003/sig0000008c )
  );
  MUXCY   \blk00000003/blk00000020  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000095 ),
    .O(\blk00000003/sig00000096 )
  );
  MUXCY   \blk00000003/blk0000001f  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000093 ),
    .O(\blk00000003/sig00000094 )
  );
  MUXF5   \blk00000003/blk0000001e  (
    .I0(\blk00000003/sig00000090 ),
    .I1(\blk00000003/sig00000091 ),
    .S(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig00000092 )
  );
  MUXF5   \blk00000003/blk0000001d  (
    .I0(\blk00000003/sig0000008d ),
    .I1(\blk00000003/sig0000008e ),
    .S(\blk00000003/sig0000008c ),
    .O(\blk00000003/sig0000008f )
  );
  MUXF5   \blk00000003/blk0000001c  (
    .I0(\blk00000003/sig0000008b ),
    .I1(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000008c ),
    .O(\NLW_blk00000003/blk0000001c_O_UNCONNECTED )
  );
  FDRS   \blk00000003/blk0000001b  (
    .C(sig00000027),
    .D(\blk00000003/sig0000008a ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig00000030)
  );
  FDRS   \blk00000003/blk0000001a  (
    .C(sig00000027),
    .D(\blk00000003/sig00000089 ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig0000002f)
  );
  FDRS   \blk00000003/blk00000019  (
    .C(sig00000027),
    .D(\blk00000003/sig00000088 ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig0000002e)
  );
  FDRS   \blk00000003/blk00000018  (
    .C(sig00000027),
    .D(\blk00000003/sig00000087 ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig0000002d)
  );
  FDRS   \blk00000003/blk00000017  (
    .C(sig00000027),
    .D(\blk00000003/sig00000086 ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig0000002c)
  );
  FDRS   \blk00000003/blk00000016  (
    .C(sig00000027),
    .D(\blk00000003/sig00000085 ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig0000002b)
  );
  FDRS   \blk00000003/blk00000015  (
    .C(sig00000027),
    .D(\blk00000003/sig00000084 ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig0000002a)
  );
  FDRS   \blk00000003/blk00000014  (
    .C(sig00000027),
    .D(\blk00000003/sig00000081 ),
    .R(\blk00000003/sig00000082 ),
    .S(\blk00000003/sig00000083 ),
    .Q(sig00000029)
  );
  FDRS   \blk00000003/blk00000013  (
    .C(sig00000027),
    .D(\blk00000003/sig00000080 ),
    .R(\blk00000003/sig0000007a ),
    .S(\blk00000003/sig00000039 ),
    .Q(sig00000036)
  );
  FDRS   \blk00000003/blk00000012  (
    .C(sig00000027),
    .D(\blk00000003/sig0000007f ),
    .R(\blk00000003/sig0000007a ),
    .S(\blk00000003/sig00000039 ),
    .Q(sig00000037)
  );
  FDRS   \blk00000003/blk00000011  (
    .C(sig00000027),
    .D(\blk00000003/sig0000007e ),
    .R(\blk00000003/sig0000007a ),
    .S(\blk00000003/sig00000039 ),
    .Q(sig00000033)
  );
  FDRS   \blk00000003/blk00000010  (
    .C(sig00000027),
    .D(\blk00000003/sig0000007d ),
    .R(\blk00000003/sig0000007a ),
    .S(\blk00000003/sig00000039 ),
    .Q(sig00000035)
  );
  FDRS   \blk00000003/blk0000000f  (
    .C(sig00000027),
    .D(\blk00000003/sig0000007c ),
    .R(\blk00000003/sig0000007a ),
    .S(\blk00000003/sig00000039 ),
    .Q(sig00000034)
  );
  FDRS   \blk00000003/blk0000000e  (
    .C(sig00000027),
    .D(\blk00000003/sig0000007b ),
    .R(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000039 ),
    .Q(sig00000028)
  );
  FDRS   \blk00000003/blk0000000d  (
    .C(sig00000027),
    .D(\blk00000003/sig00000079 ),
    .R(\blk00000003/sig0000007a ),
    .S(\blk00000003/sig00000039 ),
    .Q(sig00000032)
  );
  FDRS   \blk00000003/blk0000000c  (
    .C(sig00000027),
    .D(\blk00000003/sig00000076 ),
    .R(\blk00000003/sig00000077 ),
    .S(\blk00000003/sig00000078 ),
    .Q(sig00000031)
  );
  MUXCY   \blk00000003/blk0000000b  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000075 ),
    .O(\blk00000003/sig00000073 )
  );
  MUXCY   \blk00000003/blk0000000a  (
    .CI(\blk00000003/sig00000073 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000074 ),
    .O(\blk00000003/sig00000071 )
  );
  MUXCY   \blk00000003/blk00000009  (
    .CI(\blk00000003/sig00000071 ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000072 ),
    .O(\NLW_blk00000003/blk00000009_O_UNCONNECTED )
  );
  MUXCY   \blk00000003/blk00000008  (
    .CI(\blk00000003/sig0000003a ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig00000070 ),
    .O(\blk00000003/sig0000006e )
  );
  MUXCY   \blk00000003/blk00000007  (
    .CI(\blk00000003/sig0000006e ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000006f ),
    .O(\blk00000003/sig0000006c )
  );
  MUXCY   \blk00000003/blk00000006  (
    .CI(\blk00000003/sig0000006c ),
    .DI(\blk00000003/sig00000039 ),
    .S(\blk00000003/sig0000006d ),
    .O(\NLW_blk00000003/blk00000006_O_UNCONNECTED )
  );
  VCC   \blk00000003/blk00000005  (
    .P(\blk00000003/sig0000003a )
  );
  GND   \blk00000003/blk00000004  (
    .G(\blk00000003/sig00000039 )
  );

// synthesis translate_on

endmodule

// synthesis translate_off

`timescale  1 ps / 1 ps
