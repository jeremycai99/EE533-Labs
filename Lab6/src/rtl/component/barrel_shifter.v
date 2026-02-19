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
    output reg [`DATA_WIDTH-1:0] dout, // Shifted output value
    output reg cout // Carry out for shift operations (e.g., for LSL, the last bit shifted out)
);

always @(*) begin
    if (shamt == 0) begin
        dout = din; // No shift
        cout = cin; // Carry out is just the carry in when no shift
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