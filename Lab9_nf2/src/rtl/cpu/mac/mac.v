/* file: mac.v
 Description: Dummy Multiply-Accumulate (MAC) unit for ARM multiply instructions.
              Supports MUL, MLA, UMULL, SMULL/SMLAL, UMLAL.
 Author: Jeremy Cai
 Date: Mar. 5, 2026
 Version: 2.0
    Revision History:
    - 1.0: Initial implementation with basic multiply and accumulate logic.
    - 2.0: Updated to match ARM behavior more closely, including flag generation and control signals.
 */
 
`ifndef MAC_V
`define MAC_V

`include "define.v"
`include "test_int32mult.v"
`include "addsub.v"

module mac (
    // Operand inputs
    input wire [`REG_DATA_WIDTH-1:0] rm,        // Multiplicand (Rm field)
    input wire [`REG_DATA_WIDTH-1:0] rs,        // Multiplier   (Rs field)
    input wire [`REG_DATA_WIDTH-1:0] rn_acc,    // MLA: Rn accumulate / Long: RdHi accumulate
    input wire [`REG_DATA_WIDTH-1:0] rdlo_acc,  // Long: RdLo accumulate (UMLAL/SMLAL)

    // Control inputs from CU (latched in ID/EX)
    input wire mul_en,
    input wire mul_long,        // 1 = 64-bit long multiply
    input wire mul_signed,      // 1 = signed (SMULL/SMLAL)
    input wire mul_accumulate,  // 1 = accumulate variant (MLA/UMLAL/SMLAL)

    // Result outputs
    output wire [`REG_DATA_WIDTH-1:0] result_lo, // MUL/MLA result, or RdLo for long
    output wire [`REG_DATA_WIDTH-1:0] result_hi, // RdHi for long multiply; 0 for short
    output wire [3:0] mac_flags                  // {N, Z, C(0), V(0)}
);

// ================================================================
// Stage 1: 32×32 → 64-bit multiply
// ================================================================
wire [63:0] product;

test_int32mult u_mult (
    .a(rm),
    .b(rs),
    .p(product)
);

// ================================================================
// Stage 2: 64-bit accumulate via two chained 32-bit addsub units
// ================================================================

// Low accumulate operand: MLA uses rn_acc, long uses rdlo_acc
wire [`REG_DATA_WIDTH-1:0] acc_lo_b = mul_long ? rdlo_acc : rn_acc;

wire [`REG_DATA_WIDTH-1:0] acc_lo_result;
wire acc_lo_carry;

addsub u_acc_lo (
    .operand_a(product[31:0]),
    .operand_b(acc_lo_b),
    .sub(1'b0),
    .carry_in(1'b0),
    .result(acc_lo_result),
    .overflow(),              // unused — ARM MUL V flag is unpredictable
    .carry_out(acc_lo_carry)
);

// High accumulate: product[63:32] + rn_acc + carry from low
wire [`REG_DATA_WIDTH-1:0] acc_hi_result;

addsub u_acc_hi (
    .operand_a(product[63:32]),
    .operand_b(rn_acc),
    .sub(1'b0),
    .carry_in(acc_lo_carry),
    .result(acc_hi_result),
    .overflow(),
    .carry_out()
);

// ================================================================
// Result mux
// ================================================================
assign result_lo = mul_accumulate ? acc_lo_result : product[31:0];
assign result_hi = mul_long ? (mul_accumulate ? acc_hi_result : product[63:32])
                            : {`REG_DATA_WIDTH{1'b0}};

// ================================================================
// Flag generation
// ARM MUL/MULL: N and Z are meaningful; C and V are unpredictable (0)
// ================================================================
wire n_flag = mul_long ? result_hi[`REG_DATA_WIDTH-1] : result_lo[`REG_DATA_WIDTH-1];
wire z_flag = mul_long ? ({result_hi, result_lo} == 64'd0) : (result_lo == {`REG_DATA_WIDTH{1'b0}});

assign mac_flags = mul_en ? {n_flag, z_flag, 1'b0, 1'b0} : 4'b0;

endmodule

`endif // MAC_V