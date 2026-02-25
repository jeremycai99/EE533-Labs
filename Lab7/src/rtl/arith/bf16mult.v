/* file: bf16mult.v
 Description: This file implements the multiplication operation for BF16 format.
 BF16 format: [15] sign, [14:7] exponent (bias=127), [6:0] fraction
 Author: Jeremy Cai
 Date: Feb. 24, 2026
 Version: 1.0
 Revision history:
    - Feb. 24, 2026: Initial implementation of BF16 multiplication.
 */

`ifndef BF16MULT_V
`define BF16MULT_V

`include "gpu_define.v"

