/* file: sp_core.v
 Description: This file implements the CUDA-like SP (streaming processor) core pipeline design.
 This implementation focuses on the per-thread behavior and the core should output the register content to
    tensor core for tensor operations for 4 x 4 matrix multiplication and accumulation. This process is a multi-cycle
    operation.
 Author: Jeremy Cai
 Date: Feb. 25, 2026
 Version: 1.0
 Revision history:
    - Feb. 25, 2026: Initial implementation of the CUDA-like SP core pipeline.
*/

`ifndef SP_CORE_V
`define SP_CORE_V

`include "gpu_define.v"

module sp_core


`endif // SP_CORE_V
