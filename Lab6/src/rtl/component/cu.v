/* file: cu.v
 Description: Control unit module for the Arm pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 17, 2026
 Version: 1.0
 */

/* ========================================================================
 *  Figure 4-1 (ASCII) : ARM 32-bit instruction set formats (bit[31:0])
 *
 *  Bit positions:
 *    31                                                        0
 *    3 3 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 9 8 7 6 5 4 3 2 1 0
 *    1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
 *
 *  ----------------------------------------------------------------------
 *  Data Processing / PSR Transfer (DP/PSR)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | [27:26]=00 | I | Opcode | S | Rn | Rd | Operand2
 *
 *  ----------------------------------------------------------------------
 *  Multiply
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 000000 | A | S | Rd | Rn | Rs | 1001 | Rm
 *
 *  ----------------------------------------------------------------------
 *  Multiply Long
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 00001 | U | A | S | RdHi | RdLo | Rs | 1001 | Rm
 *
 *  ----------------------------------------------------------------------
 *  Single Data Swap (SWP/SWPB)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 00010 | B | 00 | Rn | Rd | 00001001 | Rm
 *
 *  ----------------------------------------------------------------------
 *  Branch and Exchange (BX)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 000100101111111111110001 | Rn
 *
 *  ----------------------------------------------------------------------
 *  Halfword Data Transfer (register offset)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 000 | P | U | 0 | W | L | Rn | Rd | 0000 | 1 | S | H | 1 | Rm
 *
 *  ----------------------------------------------------------------------
 *  Halfword Data Transfer (immediate offset)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 000 | P | U | 1 | W | L | Rn | Rd | Offset | 1 | S | H | 1 | Offset
 *
 *  ----------------------------------------------------------------------
 *  Single Data Transfer (LDR/STR)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 01 | I | P | U | B | W | L | Rn | Rd | Offset
 *
 *  ----------------------------------------------------------------------
 *  Undefined
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 011 | <undefined/implementation-defined pattern>
 *
 *  ----------------------------------------------------------------------
 *  Block Data Transfer (LDM/STM)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 100 | P | U | S | W | L | Rn | Register List(16 bits)
 *
 *  ----------------------------------------------------------------------
 *  Branch (B / BL)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 101 | L | Signed 24-bit Offset
 *
 *  ----------------------------------------------------------------------
 *  Coprocessor Data Transfer
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 110 | P | U | N | W | L | Rn | CRd | CP# | Offset
 *
 *  ----------------------------------------------------------------------
 *  Coprocessor Data Operation
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 1110 | CP Opc | CRn | CRd | CP# | CP | 0 | CRm
 *
 *  ----------------------------------------------------------------------
 *  Coprocessor Register Transfer
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 1110 | CP Opc | L | CRn | Rd | CP# | CP | 1 | CRm
 *
 *  ----------------------------------------------------------------------
 *  Software Interrupt (SWI)
 *  ----------------------------------------------------------------------
 *   [31:28] Cond | 1111 | (ignored by processor / immediate field)
 * ======================================================================== */


`ifndef CU_V
`define CU_V

`include "define.v"
`include "cond_eval.v"
module cu (
    // Keep input of control signals simply the instruction. The condition evaluation is done in cond_eval module, and the output control signals are generated based on the instruction decoding and condition evaluation results.
    input wire [`INSTR_WIDTH-1:0] instr, // Input instruction from IF/ID pipeline register
    /* Large fanout control signals*/
    // Program condition evaluations
    output wire [COND_WIDTH-1:0] cond_code, // Condition code extracted from instruction for condition evaluation
    // Instruction-type flags
    output wire is_dp, // Flag indicating if the instruction is a data processing instruction
    output wire is_mul, // Flag indicating if the instruction is a multiply instruction (not used in Lab 6 but can be extended for future labs)
    output wire is_mul_long, // Flag indicating if the instruction is a long multiply instruction (not used in Lab 6 but can be extended for future labs)
    output wire is_mrs, // Flag indicating if the instruction is an MRS instruction
    output wire is_msr, // Flag indicating if the instruction is an MSR instruction

);

endmodule

`endif // CU_V