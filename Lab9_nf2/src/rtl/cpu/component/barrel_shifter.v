/* file: barrel_shifter.v
 Description: Barrel shifter module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 23, 2026
 Version: 1.1
 Changes from 1.0:
   - ROR fix: replaced (din >> shamt) | (din << (DATA_WIDTH - shamt))
     with {din, din} >> shamt (concatenation-based rotation).
     The old expression synthesized incorrectly on Virtex/ISE, producing
     zero for all non-zero shift amounts.  The concatenation idiom
     uses a single right-shift on a 64-bit value, which maps cleanly
     to a mux tree in synthesis.
   - cout fix: replaced variable-index expressions din[DATA_WIDTH-shamt]
     and din[shamt-1] with intermediate wires computed from the full
     shift result, avoiding synthesis issues with variable bit selects
     involving subtraction.
 */
 
`ifndef BARREL_SHIFTER_V
`define BARREL_SHIFTER_V

`include "define.v"

module barrel_shifter (
    input wire [`DATA_WIDTH-1:0] din, // Input value to be shifted
    input wire [`SHIFT_AMOUNT_WIDTH-1:0] shamt, // Shift amount (0-31)
    input wire [1:0] shift_type, // 00: LSL, 01: LSR, 10: ASR, 11: ROR
    input wire cin, // Carry in (current C flag)
    input wire is_imm_shift, // 1 = immediate shift encoding, 0 = register shift
    output reg [`DATA_WIDTH-1:0] dout, // Shifted output value
    output reg cout // Carry out (last bit shifted out)
);

/* Precompute all shift results from wires — helps synthesis map to
   efficient mux trees rather than inferring complex combinational
   logic inside the always block. */

wire [`DATA_WIDTH-1:0] lsl_result = din << shamt;
wire [`DATA_WIDTH-1:0] lsr_result = din >> shamt;
wire [`DATA_WIDTH-1:0] asr_result = $signed(din) >>> shamt;

/* ROR via concatenation: {din, din} is 64 bits.  Right-shifting by
   shamt and truncating to 32 bits gives the rotate-right result.
   This idiom synthesizes to a single mux tree (no subtract, no OR).
   
   Proof for shamt=24, din=1:
     {1, 1} = 64'h00000001_00000001
     >> 24  = 64'h00000000_00000100...
     lower 32 bits = 0x00000100  ✓ (= ROR(1, 24))
*/
wire [63:0] ror_concat = {din, din};
wire [`DATA_WIDTH-1:0] ror_result = ror_concat >> shamt;

/* Carry-out helpers.
   LSL: last bit shifted out = din[32 - shamt] = din bit that moves past MSB.
   LSR/ASR/ROR: last bit shifted out = din[shamt - 1].
   We compute these with full-width mux to avoid synthesis issues with
   variable bit indexing involving subtraction. */

/* For LSL cout: the bit at position (DATA_WIDTH - shamt).
   Equivalent to checking if lsl_result would have overflowed. */
reg lsl_cout_r;
always @(*) begin
    case (shamt)
        5'd1:  lsl_cout_r = din[31];
        5'd2:  lsl_cout_r = din[30];
        5'd3:  lsl_cout_r = din[29];
        5'd4:  lsl_cout_r = din[28];
        5'd5:  lsl_cout_r = din[27];
        5'd6:  lsl_cout_r = din[26];
        5'd7:  lsl_cout_r = din[25];
        5'd8:  lsl_cout_r = din[24];
        5'd9:  lsl_cout_r = din[23];
        5'd10: lsl_cout_r = din[22];
        5'd11: lsl_cout_r = din[21];
        5'd12: lsl_cout_r = din[20];
        5'd13: lsl_cout_r = din[19];
        5'd14: lsl_cout_r = din[18];
        5'd15: lsl_cout_r = din[17];
        5'd16: lsl_cout_r = din[16];
        5'd17: lsl_cout_r = din[15];
        5'd18: lsl_cout_r = din[14];
        5'd19: lsl_cout_r = din[13];
        5'd20: lsl_cout_r = din[12];
        5'd21: lsl_cout_r = din[11];
        5'd22: lsl_cout_r = din[10];
        5'd23: lsl_cout_r = din[9];
        5'd24: lsl_cout_r = din[8];
        5'd25: lsl_cout_r = din[7];
        5'd26: lsl_cout_r = din[6];
        5'd27: lsl_cout_r = din[5];
        5'd28: lsl_cout_r = din[4];
        5'd29: lsl_cout_r = din[3];
        5'd30: lsl_cout_r = din[2];
        5'd31: lsl_cout_r = din[1];
        default: lsl_cout_r = cin; // shamt==0 handled separately
    endcase
end

/* For LSR/ASR/ROR cout: the bit at position (shamt - 1). */
reg rshift_cout_r;
always @(*) begin
    case (shamt)
        5'd1:  rshift_cout_r = din[0];
        5'd2:  rshift_cout_r = din[1];
        5'd3:  rshift_cout_r = din[2];
        5'd4:  rshift_cout_r = din[3];
        5'd5:  rshift_cout_r = din[4];
        5'd6:  rshift_cout_r = din[5];
        5'd7:  rshift_cout_r = din[6];
        5'd8:  rshift_cout_r = din[7];
        5'd9:  rshift_cout_r = din[8];
        5'd10: rshift_cout_r = din[9];
        5'd11: rshift_cout_r = din[10];
        5'd12: rshift_cout_r = din[11];
        5'd13: rshift_cout_r = din[12];
        5'd14: rshift_cout_r = din[13];
        5'd15: rshift_cout_r = din[14];
        5'd16: rshift_cout_r = din[15];
        5'd17: rshift_cout_r = din[16];
        5'd18: rshift_cout_r = din[17];
        5'd19: rshift_cout_r = din[18];
        5'd20: rshift_cout_r = din[19];
        5'd21: rshift_cout_r = din[20];
        5'd22: rshift_cout_r = din[21];
        5'd23: rshift_cout_r = din[22];
        5'd24: rshift_cout_r = din[23];
        5'd25: rshift_cout_r = din[24];
        5'd26: rshift_cout_r = din[25];
        5'd27: rshift_cout_r = din[26];
        5'd28: rshift_cout_r = din[27];
        5'd29: rshift_cout_r = din[28];
        5'd30: rshift_cout_r = din[29];
        5'd31: rshift_cout_r = din[30];
        default: rshift_cout_r = cin; // shamt==0 handled separately
    endcase
end

/* Main shift/rotate output mux */
always @(*) begin
    if (shamt == 0) begin
        if (!is_imm_shift) begin
            // Register shift by 0: identity for all types
            dout = din;
            cout = cin;
        end else begin
            // Immediate shift with shift_imm == 0: ARM special encodings
            case (shift_type)
                `SHIFT_LSL: begin
                    // LSL #0: identity
                    dout = din;
                    cout = cin;
                end
                `SHIFT_LSR: begin
                    // LSR #0 encodes LSR #32: result is zero, cout is din[31]
                    dout = {`DATA_WIDTH{1'b0}};
                    cout = din[`DATA_WIDTH-1];
                end
                `SHIFT_ASR: begin
                    // ASR #0 encodes ASR #32: result is all sign bits
                    dout = {`DATA_WIDTH{din[`DATA_WIDTH-1]}};
                    cout = din[`DATA_WIDTH-1];
                end
                `SHIFT_ROR: begin
                    // ROR #0 encodes RRX: {Cin, din[31:1]}, cout = din[0]
                    dout = {cin, din[`DATA_WIDTH-1:1]};
                    cout = din[0];
                end
                default: begin
                    dout = din;
                    cout = cin;
                end
            endcase
        end
    end else begin
        case(shift_type)
            `SHIFT_LSL: begin
                dout = lsl_result;
                cout = lsl_cout_r;
            end
            `SHIFT_LSR: begin
                dout = lsr_result;
                cout = rshift_cout_r;
            end
            `SHIFT_ASR: begin
                dout = asr_result;
                cout = rshift_cout_r;
            end
            `SHIFT_ROR: begin
                /* v1.1 FIX: concatenation-based rotation.
                 * {din, din} >> shamt: the 64-bit concatenation right-shifted
                 * by shamt, truncated to 32 bits, gives the correct ROR result.
                 * This synthesizes to a clean mux tree with no subtract/OR. */
                dout = ror_result;
                cout = rshift_cout_r;
            end
            default: begin
                dout = din;
                cout = cin;
            end
        endcase
    end
end

endmodule

`endif // BARREL_SHIFTER_V