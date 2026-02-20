/* file: barrel_shifter.v
 Description: Barrel shifter module for the pipeline CPU design
 Author: Jeremy Cai
 Date: Feb. 17, 2026
 Version: 1.0
 */
 
`ifndef BARREL_SHIFTER_V
`define BARREL_SHIFTER_V

`include "define.v"

module barrel_shifter (
    input wire [`DATA_WIDTH-1:0] din, // Input value to be shifted
    input wire [`SHIFT_AMOUNT_WIDTH-1:0] shamt, // Shift amount (0-63)
    input wire [1:0] shift_type, // 00: LSL, 01: LSR, 10: ASR, 11: ROR
    input wire cin, // Carry in for shift operations (e.g., for ROR, the last bit shifted out becomes the new MSB)
    input wire is_imm_shift, // Indicates if this is an immediate shift (for shift by register, the shift amount is in a register and can be 0-31)
    output reg [`DATA_WIDTH-1:0] dout, // Shifted output value
    output reg cout // Carry out for shift operations (e.g., for LSL, the last bit shifted out)
);

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
                dout = din << shamt; // Logical Shift Left
                cout = din[`DATA_WIDTH-shamt]; // Last bit shifted out becomes carry out
            end
            `SHIFT_LSR: begin
                dout = din >> shamt; // Logical Shift Right
                cout = din[shamt-1]; // Last bit shifted out becomes carry out
            end
            `SHIFT_ASR: begin
                dout = $signed(din) >>> shamt; // Arithmetic Shift Right (preserves sign bit)
                cout = din[shamt-1]; // Last bit shifted out becomes carry out
            end
            `SHIFT_ROR: begin
                dout = (din >> shamt) | (din << (`DATA_WIDTH - shamt)); // Rotate Right
                cout = din[shamt-1]; // Last bit shifted out becomes carry out
            end
            default: begin
                dout = din; // Default to no shift
                cout = cin; // Carry out is just the carry in for default case
            end
        endcase
    end
end

endmodule

`endif // BARREL_SHIFTER_V