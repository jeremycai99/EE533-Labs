/* file: define.v
 Description: This file contains global definitions and parameters for the project.
 Author: Jeremy Cai
 Date: Feb. 9, 2026
 Version: 1.0
 */

//Data width definition
`define DATA_WIDTH 64 //Maximum 64 due to KSA design
`define ALU_OP_WIDTH 4

//ALU operation codes (4 bits)
//Support list: ADD SUB AND OR XNOR CMP LSL LSR SBCMP LSTC RSTC
`define ALU_OP_ADD 4'b0000
`define ALU_OP_SUB 4'b0001
`define ALU_OP_AND 4'b0010
`define ALU_OP_OR  4'b0011
`define ALU_OP_XNOR 4'b0100
`define ALU_OP_CMP 4'b0101
`define ALU_OP_LSL 4'b0110
`define ALU_OP_LSR 4'b0111
// Not standard ALU operation and not used by Arm ISA
`define ALU_OP_SBCMP 4'b1000
`define ALU_OP_LSTC 4'b1001
`define ALU_OP_RSTC 4'b1010



