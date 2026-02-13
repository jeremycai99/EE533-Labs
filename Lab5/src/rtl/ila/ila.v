/* file: ila.v
 Description: ILA module for SoC debugging and verification
 Author: Jeremy Cai
 Date: Feb. 13, 2026
 Version: 1.0
 */

`ifndef ILA_V
`define ILA_V

`include "define.v"

// ILA module behavior explanation:
// This ILA module contains probes to monitor the key registers and signals in the CPU and (future) GPU module.
// The ILA is connected to the soc top (soc.v) module with the MMIO interface, allowing it to access the instruction
// memory, data memory, CPU core register file, and pipeline registers for debugging purposes. This module also contains
// a single-cycle clock generator to drive the soc top, enabling step-by-step execution for detailed observation of the
// CPU and GPU behavior during simulation in debug mode.

// This ILA is connected to the top module, where it can expose the signals or register to the netFPGA SW/NW interface
// and automatically get the commands or feedback the requested data.

// The collected metadata is stored in the BRAM of the ILA module with memory mapping:


// ILA module definition
module ila (

);


endmodule

`endif // ILA_V