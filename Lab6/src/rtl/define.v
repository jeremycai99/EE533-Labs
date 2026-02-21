/* file: define.v
 Description: This file contains global definitions and parameters for the project.
 Author: Jeremy Cai
 Date: Feb. 21, 2026
 Version: 1.2
 Revision History:
    - 1.0: Initial version with basic definitions for Lab5 (Feb. 9, 2026)
    - 1.1: Updated version with more Arm-based definitions for Lab6 (Feb. 17, 2026)
    - 1.2: Updated data memory parameters (Feb. 21, 2026)
 */

//Data width definition
`define DATA_WIDTH 32
`define ALU_OP_WIDTH 4

//Instruction width definition
`define INSTR_WIDTH 32

// CPU data memory parameters
`define CPU_DMEM_ADDR_WIDTH 32 //32 bits for CPU data memory address space (

//ALU operation codes (4 bits)
//Support list: ADD EOR, SUB RSB, AND ADC, SBC RSC, TST TEQ, CMP CMN, ORR MOV, BIC MVN
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
`define COND_EQ 4'b0000 // Equal
`define COND_NE 4'b0001 // Not equal
`define COND_CS 4'b0010 // Carry set (unsigned higher or same)
`define COND_CC 4'b0011 // Carry clear (unsigned lower)
`define COND_MI 4'b0100 // Minus (negative)
`define COND_PL 4'b0101 // Plus (positive or zero)
`define COND_VS 4'b0110 // Overflow
`define COND_VC 4'b0111 // No overflow
`define COND_HI 4'b1000 // Unsigned higher
`define COND_LS 4'b1001 // Unsigned lower or same
`define COND_GE 4'b1010 // Signed greater than or equal
`define COND_LT 4'b1011 // Signed less than
`define COND_GT 4'b1100 // Signed greater than
`define COND_LE 4'b1101 // Signed less than or equal
`define COND_AL 4'b1110 // Always (unconditional)
`define COND_NV 4'b1111 // Never (reserved)

//Barrel shifter operation codes (2 bits)
`define SHIFT_TYPE_WIDTH 2
`define SHIFT_AMOUNT_WIDTH 5 //Shift amount width (5 bits to support shifts up to 31)
`define SHIFT_LSL 2'b00 // Logical Shift Left
`define SHIFT_LSR 2'b01 // Logical Shift Right
`define SHIFT_ASR 2'b10 // Arithmetic Shift Right
`define SHIFT_ROR 2'b11 // Rotate Right

// CPSR flag bit positions
`define FLAG_N 3 // Negative flag
`define FLAG_Z 2 // Zero flag
`define FLAG_C 1 // Carry flag
`define FLAG_V 0 // Overflow flag

// Instruction type encodings (for control logic), can be 2b or 3b
`define ITYPE2_DP 2'b00 // Data Processing
`define ITYPE2_SDT 2'b01 // Memory Access
`define ITYPE3_BDT 3'b100 // Block data transfer (LDM/STM/PUSH/POP)
`define ITYPE3_BR  3'b101 // Branch (B/BL)

// Forwarding unit parameters
`define FWD_NONE 2'b00 // No forwarding
`define FWD_MEM_WB   2'b01 // Forward from MEM/WB stage
`define FWD_EX_MEM    2'b10 // Forward from EX/MEM stage


//Register file parameters
`define REG_ADDR_WIDTH 4 //4 bits for 16 registers (lab 6 will use 16 registers to align with Arm's R0-R15)
`define REG_DATA_WIDTH 32 //32-bit registers
`define REG_DEPTH 16 //16 registers in total

//Program counter parameters
`define PC_WIDTH 32 //32-bit program counter (will be truncated to smaller width for our small instruction memory)

//MMIO interface parameters
`define MMIO_ADDR_WIDTH 32 //32 bits for MMIO address space
`define MMIO_DATA_WIDTH 32 //32 bits for MMIO data width

// Write-Back Source Select (wb_sel encoding)
`define WB_ALU   3'b000     // ALU / barrel-shifter result
`define WB_MEM   3'b001     // Memory load data
`define WB_LINK  3'b010     // PC+4 for BL return address
`define WB_PSR   3'b011     // CPSR / SPSR value (MRS)
`define WB_MUL   3'b100     // MAC unit result

// Forward Mux Encoding
`define FWD_NONE     3'b000
`define FWD_EXMEM    3'b001
`define FWD_MEMWB    3'b010
`define FWD_BDTU_P1  3'b011
`define FWD_BDTU_P2  3'b100
`define FWD_EXMEM_P2  3'b101
`define FWD_MEMWB_P2  3'b110

// CPU Done PC Value (used for testbench to detect when CPU has finished executing)
`define CPU_DONE_PC 32'h0000_0200

// Instruction memory and data memory configuration
`define IMEM_ADDR_WIDTH 10 //10 bits for instruction memory address space. Will be truncated to smaller width for our small instruction memory.
`define IMEM_DATA_WIDTH 32 //32 bits for instruction width
`define IMEM_DEPTH 1024 //1024 instructions in instruction memory (32KB)

`define DMEM_ADDR_WIDTH 12 //12 bits for data memory address space. Will be truncated to smaller width for our small data memory.
`define DMEM_DATA_WIDTH 32 //32 bits for data width
`define DMEM_DEPTH 4096 //4096 words in data memory (16KB)