/* file: mac.v
 Description: Dummy Multiply-Accumulate (MAC) unit for ARM multiply instructions.
              Supports MUL, MLA, UMULL, SMULL/SMLAL, UMLAL.
 Author: Jeremy Cai
 Date: Feb. 18, 2026
 Version: 1.0
 */

`ifndef MAC_V
`define MAC_V

`include "define.v"

module mac (
    // Operand inputs
    input  wire [`REG_DATA_WIDTH-1:0] rm, // Multiplicand (Rm field)
    input  wire [`REG_DATA_WIDTH-1:0] rs, // Multiplier   (Rs field)
    input  wire [`REG_DATA_WIDTH-1:0] rn_acc, // Accumulate value for MLA  (Rn field / RdHi for long)
    input  wire [`REG_DATA_WIDTH-1:0] rdlo_acc, // Accumulate low for SMLAL/UMLAL (Rd field = RdLo)

    // Control inputs from CU (latched in ID/EX)
    input  wire mul_en, // Multiply enable
    input  wire mul_long, // 1 = 64-bit long multiply (UMULL/SMULL/UMLAL/SMLAL)
    input  wire mul_signed, // 1 = signed multiply (SMULL/SMLAL), 0 = unsigned
    input  wire mul_accumulate, // 1 = multiply-accumulate variant (MLA/UMLAL/SMLAL)

    // Result outputs
    output wire [`REG_DATA_WIDTH-1:0] result_lo, // Lower 32 bits  (MUL/MLA result, or RdLo for long)
    output wire [`REG_DATA_WIDTH-1:0] result_hi, // Upper 32 bits  (RdHi for long multiply; 0 for short)
    output wire [3:0]  mac_flags // {N, Z, C(0), V(0)} — MUL/MULL flag outputs
);

// Core Multiply
// Signed 64-bit product (covers both signed and unsigned via mux)
wire signed [2*`REG_DATA_WIDTH-1:0] s_product = $signed(rm) * $signed(rs);
wire        [2*`REG_DATA_WIDTH-1:0] u_product = rm * rs;

wire [2*`REG_DATA_WIDTH-1:0] product = mul_signed ? s_product : u_product;

// Accumulate
// Short multiply accumulate: result = Rm * Rs + Rn
wire [`REG_DATA_WIDTH-1:0] short_acc = product + rn_acc;

// Long multiply accumulate: result = Rm * Rs + {RdHi, RdLo}
wire [2*`REG_DATA_WIDTH-1:0] long_acc_val = {rn_acc, rdlo_acc};
wire [2*`REG_DATA_WIDTH-1:0] long_acc     = product + long_acc_val;

// Result Mux
wire [2*`REG_DATA_WIDTH-1:0] long_result  = mul_accumulate ? long_acc  : product;
wire [`REG_DATA_WIDTH-1:0] short_result = mul_accumulate ? short_acc : product;

assign result_lo = mul_long ? long_result[`REG_DATA_WIDTH-1:0]  : short_result;
assign result_hi = mul_long ? long_result[2*`REG_DATA_WIDTH-1:`REG_DATA_WIDTH] : {`REG_DATA_WIDTH{1'b0}};

// Flag Generation
// ARM MUL/MULL sets N and Z; C and V are UNPREDICTABLE (we set to 0)
// wire [`REG_DATA_WIDTH-1:0] flag_check = mul_long ? long_result[2*`REG_DATA_WIDTH-1:`REG_DATA_WIDTH] : short_result;
wire        n_flag = mul_long ? long_result[2*`REG_DATA_WIDTH-1] : short_result[`REG_DATA_WIDTH-1];
wire        z_flag = mul_long ? (long_result == {2*`REG_DATA_WIDTH{1'b0}}) : (short_result == {`REG_DATA_WIDTH{1'b0}});

assign mac_flags = mul_en ? {n_flag, z_flag, 1'b0, 1'b0} : 4'b0;

endmodule

`endif // MAC_V