#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda::wmma;

// ============================================================
// WMMA Tensor Core ANN Kernel
// Network: 16 inputs → 16 hidden (ReLU) → 16 output
// Uses 16×16×16 WMMA tiles (perfect fit)
//
// Launch: 1 block, 1 warp (32 threads)
// Each wmma.mma does: C[16×16] += A[16×16] × B[16×16]
//   - 4,096 multiply-accumulate ops in ONE instruction
//   - All 32 threads cooperate via hardware crossbar
// ============================================================

// Memory layout (all pre-packed by host):
//   input:    16 × BF16  (treated as 16×1, but padded to 16×16 for WMMA)
//   weights1: 16 × 16 × BF16  (hidden weights, row-major)
//   bias1:    16 × FP32        (hidden bias)
//   weights2: 16 × 16 × BF16  (output weights, row-major)
//   bias2:    16 × FP32        (output bias)
//   output:   16 × FP32        (final result)

// ============================================================
// Layer 1: hidden[16] = ReLU(weights1[16×16] × input[16] + bias1[16])
// ============================================================
//
// WMMA requires 16×16 matrices, but input is a vector (16×1).
// Solution: pad input to 16×16 (first column = input, rest = 0)
//
//   weights1[16×16]  ×  input_padded[16×16]  =  result[16×16]
//                                                  ↑
//                                        only column 0 matters
//                                        result[i][0] = dot(W_row_i, input)

__global__ void ann_wmma_layer1_relu(
    const __nv_bfloat16 *input_padded,  // 16×16 (col0 = input, rest = 0)
    const __nv_bfloat16 *weights1,      // 16×16 row-major
    const float *bias1,                 // 16
    float *hidden_result                // 16×16 (only col0 used)
) {
    // Declare fragments — these live in the SHARED register file
    // Each thread holds 8 elements of each fragment
    //
    // fragment layout across 32 threads:
    //   Thread 0:  %r1=[A[0][0]|A[0][1]], %r2=[A[2][0]|A[2][1]], ...
    //   Thread 1:  %r1=[A[0][2]|A[0][3]], %r2=[A[2][2]|A[2][3]], ...
    //   ...
    //   Thread 31: %r1=[A[9][14]|A[9][15]], %r2=[A[11][14]|A[11][15]], ...

    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> w_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> x_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;

    // Step 1: Initialize accumulator to zero
    //   PTX: mov.f32 %f1, 0f00000000;  (× 8 regs per thread)
    //   All 32 threads set their 8 accumulator registers to 0
    fill_fragment(acc_frag, 0.0f);

    // Step 2: Load weight matrix (16×16) from global memory into fragment
    //   PTX: wmma.load.a.sync.aligned.row.m16n16k16.global.bf16
    //         {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, [%rd1], 16;
    //
    //   What happens inside (hardware):
    //     Each of 32 threads loads 8 BF16 values (= 4 regs × 2 packed)
    //     Thread 0 loads: W[0][0:1], W[2][0:1], W[4][0:1], W[6][0:1]
    //     Thread 1 loads: W[0][2:3], W[2][2:3], W[4][2:3], W[6][2:3]
    //     ...
    //     32 threads × 8 values = 256 values = full 16×16 matrix
    load_matrix_sync(w_frag, weights1, 16);

    // Step 3: Load input matrix (16×16 padded) into fragment
    //   PTX: wmma.load.b.sync.aligned.col.m16n16k16.global.bf16
    //         {%r9, %r10, %r11, %r12, %r13, %r14, %r15, %r16}, [%rd2], 16;
    //
    //   Same distribution: 32 threads × 8 values = 256 values
    //   Since input is padded, most values are 0.0
    //   Only column 0 has real input data
    load_matrix_sync(x_frag, input_padded, 16);

    // Step 4: THE TENSOR CORE OPERATION
    //   PTX: wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16
    //         {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8},       ← C out (8 FP32)
    //         {%r1,%r2,%r3,%r4,%r5,%r6,%r7,%r8},       ← A in  (8 BF16)
    //         {%r9,%r10,%r11,%r12,%r13,%r14,%r15,%r16}, ← B in  (8 BF16)
    //         {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8};       ← C acc (8 FP32)
    //
    //   What happens inside the tensor core (ONE cycle):
    //
    //     32 threads' registers → CROSSBAR → gather rows/cols
    //     → 256 PARALLEL MAC UNITS
    //     → C[i][j] += Σ_k A[i][k] × B[k][j]  for all i,j
    //     → 4,096 multiply-accumulate operations
    //     → scatter 256 results back to 32 threads
    //
    mma_sync(acc_frag, w_frag, x_frag, acc_frag);

    // Step 5: Add bias and apply ReLU
    //   Runs on CUDA cores (not tensor cores)
    //   Each thread processes its 8 fragment elements
    //   PTX: add.f32, max.f32 (standard CUDA core ops)
    for (int i = 0; i < acc_frag.num_elements; i++) {
        acc_frag.x[i] += bias1[i % 16];
        acc_frag.x[i] = fmaxf(acc_frag.x[i], 0.0f);
    }

    // Step 6: Store result back to global memory
    //   PTX: wmma.store.d.sync.aligned.row.m16n16k16.global.f32
    //         [%rd3], {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8}, 16;
    //   Each thread writes its 8 FP32 values to correct positions
    store_matrix_sync(hidden_result, acc_frag, 16, mem_row_major);
}


// ============================================================
// Layer 2: output[16] = weights2[16×16] × hidden[16] + bias2[16]
// Same as layer 1 but no ReLU
// ============================================================
__global__ void ann_wmma_layer2_linear(
    const __nv_bfloat16 *hidden_padded,
    const __nv_bfloat16 *weights2,
    const float *bias2,
    float *output
) {
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> w_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> x_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;

    fill_fragment(acc_frag, 0.0f);
    load_matrix_sync(w_frag, weights2, 16);
    load_matrix_sync(x_frag, hidden_padded, 16);
    mma_sync(acc_frag, w_frag, x_frag, acc_frag);

    for (int i = 0; i < acc_frag.num_elements; i++) {
        acc_frag.x[i] += bias2[i % 16];
    }

    store_matrix_sync(output, acc_frag, 16, mem_row_major);
}


// ============================================================
// Argmax: find which of 16 outputs is largest
// Pure CUDA core operation (no tensor core needed)
// ============================================================
__global__ void ann_wmma_argmax(
    const float *output,
    int *result
) {
    if (threadIdx.x == 0) {
        float max_val = output[0];
        int max_idx = 0;
        for (int i = 1; i < 16; i++) {
            if (output[i * 16] > max_val) {
                max_val = output[i * 16];
                max_idx = i;
            }
        }
        *result = max_idx;
    }
}


// ============================================================
// COMPARISON: Same operation WITHOUT tensor cores (scalar)
// ============================================================
__global__ void ann_scalar_layer1_relu(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weights,
    const __nv_bfloat16 *bias,
    __nv_bfloat16 *output
) {
    int j = threadIdx.x;
    __nv_bfloat16 sum = bias[j];

    // 16 scalar FMAs — nvcc emits: ld.global.u16 + fma.rn.bf16 × 16
    for (int i = 0; i < 16; i++) {
        sum = __hfma(weights[j * 16 + i], input[i], sum);
    }

    __nv_bfloat16 zero = __float2bfloat16(0.0f);
    output[j] = __hgt(sum, zero) ? sum : zero;
}