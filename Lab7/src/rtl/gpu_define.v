/* file: gpu_define.v
 Description: This file contains global definitions for the GPU design.
 Author: Jeremy Cai
 Date: Feb. 24, 2026
 Version: 1.0
 Revision history:
    - Feb. 24, 2026: Initial implementation of global definitions for the GPU design.
 */

`ifndef GPU_DEFINE_V
`define GPU_DEFINE_V

// Instruction Opcode Definitions
`define OP_NOP  5'b00000
// Memory Operations
`define OP_ST   5'b00001
`define OP_LD   5'b00010
// Move instructions, DT irrelevant
`define OP_MOV  5'b00011
`define OP_MOVI 5'b00100
// DT conversion operations
`define OP_CVT  5'b00101
// Arithmetic Operations. DT relevant
`define OP_ADD  5'b00110
`define OP_SUB  5'b00111
`define OP_MUL  5'b01000
`define OP_FMA  5'b01001
`define OP_MAX  5'b01010
`define OP_MIN  5'b01011
`define OP_ABS  5'b01100
`define OP_NEG  5'b01101
// Logical and Shift Operations. DT irrelevant
`define OP_AND  5'b01110
`define OP_OR   5'b01111
`define OP_XOR  5'b10000
`define OP_SHL  5'b10001 // Shift left is always logical shift
`define OP_SHR  5'b10010 // In our design, shift right is a arithmetic shift for INT16
                         //mathcing INT16 encoding behavior. We can use instruction pairs 
                         //to perform logical shift if needed.
// Immediate Operations
`define OP_ADDI 5'b10011
`define OP_MULI 5'b10100
// Predicate and Selection Operations
`define OP_SETP 5'b10101
`define OP_SELP 5'b10110
// Branch and Control Operations
`define OP_BRA  5'b10111
`define OP_PBRA 5'b11000
`define OP_RET  5'b11001
// Set instruction with immediate value 0/1
`define OP_SET  5'b11010
// Load/Store instructions for shared memory (LDS/STS). May not used.
`define OP_LDS  5'b11011
`define OP_STS  5'b11100
// WMMA Opcode Definitions for tensor core
`define WMMA_MMA   5'b11101
`define WMMA_LOAD  5'b11110
`define WMMA_STORE 5'b11111

// Comparison Operation Definitions
`define COMP_EQ 2'b00 // equal
`define COMP_NE 2'b01 // not equal
`define COMP_LT 2'b10 // less than
`define COMP_LE 2'b11 // less than or equal

// Data Type Definitions
`define DT_INT16 1'b0
`define DT_BF16  1'b1

// GPU IMEM and DMEM width and depth definitions
`define GPU_IMEM_ADDR_WIDTH 8 // 256 instructions
`define GPU_IMEM_DATA_WIDTH 32 // 32-bit instruction width

`define GPU_DMEM_ADDR_WIDTH 10 // 1024 entries of data memory
`define GPU_DMEM_DATA_WIDTH 16 // 16-bit data width for GPU DMEM per thread

//GPU PC width definition
`define GPU_PC_WIDTH 32

`endif // GPU_DEFINE_V
