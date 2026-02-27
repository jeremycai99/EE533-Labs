#include <cuda_bf16.h>

// Kernel 1: Vector Addition (INT16)
__global__ void vec_add(short *a, short *b, short *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

// Kernel 2: Vector Subtraction (INT16)
__global__ void vec_sub(short *a, short *b, short *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] - b[idx];
}

// Kernel 3: BFloat16 Vector Multiply
__global__ void bf16_vector_mul(__nv_bfloat16 *a, __nv_bfloat16 *b, __nv_bfloat16 *c) {
    int idx = threadIdx.x;
    c[idx] = __hmul(a[idx], b[idx]);
}

// Kernel 4: BFloat16 FMA
__global__ void bf16_fma(__nv_bfloat16 *a, __nv_bfloat16 *b,
                         __nv_bfloat16 *c, __nv_bfloat16 *d) {
    int idx = threadIdx.x;
    d[idx] = __hfma(a[idx], b[idx], c[idx]);
}

// Kernel 5: ReLU
__global__ void relu(__nv_bfloat16 *in, __nv_bfloat16 *out) {
    int idx = threadIdx.x;
    out[idx] = __hmax(in[idx], (__nv_bfloat16)0.0f);
}