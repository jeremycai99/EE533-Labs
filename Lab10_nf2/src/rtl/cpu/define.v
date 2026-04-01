/* file: define.v
 Description: This file contains global definitions and parameters for the project.
 Author: Jeremy Cai
 Date: Feb. 21, 2026
 Version: 1.3
 Revision History:
    - 1.0: Initial version with basic definitions for Lab5 (Feb. 9, 2026)
    - 1.1: Updated version with more Arm-based definitions for Lab6 (Feb. 17, 2026)
    - 1.2: Updated data memory parameters (Feb. 21, 2026)
    - 1.3: CPU_DONE_PC -> 0x3FFC (top of 4096-word IMEM), added WB_CP (Mar. 5, 2026)
 */

`ifndef DEFINE_V
`define DEFINE_V

//Data width definition
`define DATA_WIDTH 32
`define ALU_OP_WIDTH 4

//Instruction width definition
`define INSTR_WIDTH 32

// CPU data memory parameters
`define CPU_DMEM_ADDR_WIDTH 32

//ALU operation codes (4 bits)
`define ALU_OP_AND  4'b0000
`define ALU_OP_EOR  4'b0001
`define ALU_OP_SUB  4'b0010
`define ALU_OP_RSB  4'b0011
`define ALU_OP_ADD  4'b0100
`define ALU_OP_ADC  4'b0101
`define ALU_OP_SBC  4'b0110
`define ALU_OP_RSC  4'b0111
`define ALU_OP_TST  4'b1000
`define ALU_OP_TEQ  4'b1001
`define ALU_OP_CMP  4'b1010
`define ALU_OP_CMN  4'b1011
`define ALU_OP_ORR  4'b1100
`define ALU_OP_MOV  4'b1101
`define ALU_OP_BIC  4'b1110
`define ALU_OP_MVN  4'b1111

//Condition code definitions (4 bits)
`define COND_WIDTH 4
`define COND_EQ 4'b0000
`define COND_NE 4'b0001
`define COND_CS 4'b0010
`define COND_CC 4'b0011
`define COND_MI 4'b0100
`define COND_PL 4'b0101
`define COND_VS 4'b0110
`define COND_VC 4'b0111
`define COND_HI 4'b1000
`define COND_LS 4'b1001
`define COND_GE 4'b1010
`define COND_LT 4'b1011
`define COND_GT 4'b1100
`define COND_LE 4'b1101
`define COND_AL 4'b1110
`define COND_NV 4'b1111

//Barrel shifter operation codes (2 bits)
`define SHIFT_TYPE_WIDTH 2
`define SHIFT_AMOUNT_WIDTH 5
`define SHIFT_LSL 2'b00
`define SHIFT_LSR 2'b01
`define SHIFT_ASR 2'b10
`define SHIFT_ROR 2'b11

// CPSR flag bit positions
`define FLAG_N 3
`define FLAG_Z 2
`define FLAG_C 1
`define FLAG_V 0

// Instruction type encodings (for control logic)
`define ITYPE2_DP 2'b00
`define ITYPE2_SDT 2'b01
`define ITYPE3_BDT 3'b100
`define ITYPE3_BR  3'b101

// Forwarding unit parameters
`define FWD_NONE 2'b00
`define FWD_MEM_WB   2'b01
`define FWD_EX_MEM    2'b10

//Register file parameters
`define REG_ADDR_WIDTH 4
`define REG_DATA_WIDTH 32
`define REG_DEPTH 16

//Program counter parameters
`define PC_WIDTH 32

//MMIO interface parameters
`define MMIO_ADDR_WIDTH 32
`define MMIO_DATA_WIDTH 32

// Write-Back Source Select (wb_sel encoding)
`define WB_ALU   3'b000     // ALU / barrel-shifter result
`define WB_MEM   3'b001     // Memory load data
`define WB_LINK  3'b010     // PC+4 for BL return address
`define WB_PSR   3'b011     // CPSR / SPSR value (MRS)
`define WB_MUL   3'b100     // MAC unit result
`define WB_CP    3'b101     // Coprocessor read data (MRC)

// Forward Mux Encoding
`define FWD_NONE     3'b000
`define FWD_EXMEM    3'b001
`define FWD_MEMWB    3'b010
`define FWD_BDTU_P1  3'b011
`define FWD_BDTU_P2  3'b100
`define FWD_EXMEM_P2  3'b101
`define FWD_MEMWB_P2  3'b110

// CPU Done PC Value — top of 4096-word IMEM (fallback halt address)
`define CPU_DONE_PC 32'h0000_3FFC

// Instruction memory and data memory configuration
`define IMEM_ADDR_WIDTH 12
`define IMEM_DATA_WIDTH 32
`define IMEM_DEPTH 4096

`define DMEM_ADDR_WIDTH 12
`define DMEM_DATA_WIDTH 32
`define DMEM_DEPTH 4096

`endif // DEFINE_V