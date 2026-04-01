/*
 * file: ann_inference.cu
 * Description: ANN inference kernel for custom SIMT GPU with 4×4 bf16 tensor core.
 * Target: NetFPGA SmartNIC — packet-driven power quality anomaly detection.
 *
 * Compilation (PTX only — no GPU execution required):
 *   nvcc -arch=sm_80 --ptx ann_inference.cu -o ann_inference.ptx
 *
 * Author: Jeremy Cai
 * Date: Mar. 2026
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════════════
 *  Constants — match hardware constraints
 * ═══════════════════════════════════════════════════════════════ */

#define N           4       /* matrix/vector dimension (4×4 tensor core) */
#define NUM_LAYERS  2       /* 2-layer FC network */
#define BATCH_SIZE  4       /* process 4 packets = 4×4 matrix */
#define DMEM_DEPTH  1024    /* GPU DMEM words per bank */

#define ADDR_X   0
#define ADDR_W1  4
#define ADDR_B1  8
#define ADDR_W2  12
#define ADDR_B2  16
#define ADDR_H   20
#define ADDR_Y   24

/* ═══════════════════════════════════════════════════════════════
 *  bf16 helper — compile-time conversion for weight constants
 * ═══════════════════════════════════════════════════════════════ */

__device__ __forceinline__ __nv_bfloat16 bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

/* ═══════════════════════════════════════════════════════════════
 *  Kernel 1: ANN inference — explicit bf16 FMA
 * ═══════════════════════════════════════════════════════════════ */

__global__ void ann_infer_bf16(__nv_bfloat16* dmem) {
    /* ── Thread → bank mapping ─────────────────────────────
     * Parser: threadIdx.x → MOV.TID R_tid
     * Each thread accesses its private bank region.
     * Stride = DMEM_DEPTH (bank size).
     */
    const int tid = threadIdx.x;            /* ISA: MOV.TID R0 */
    const int bank_offset = tid * DMEM_DEPTH;

    /* ── Pointers into bank-local DMEM ─────────────────────
     * Parser: base + offset → LD.f Rn, [R_base + imm]
     * On hardware, each thread naturally accesses its own bank,
     * so bank_offset disappears — just use DMEM[addr] directly.
     */
    __nv_bfloat16* X  = &dmem[bank_offset + ADDR_X];
    __nv_bfloat16* W1 = &dmem[bank_offset + ADDR_W1];
    __nv_bfloat16* B1 = &dmem[bank_offset + ADDR_B1];
    __nv_bfloat16* W2 = &dmem[bank_offset + ADDR_W2];
    __nv_bfloat16* B2 = &dmem[bank_offset + ADDR_B2];
    __nv_bfloat16* H  = &dmem[bank_offset + ADDR_H];
    __nv_bfloat16* Y  = &dmem[bank_offset + ADDR_Y];

    /* ═════════════════════════════════════════════════════════
     * LAYER 1: H = ReLU(W1 × X + B1)
     * ═════════════════════════════════════════════════════════ */


    #pragma unroll
    for (int j = 0; j < N; j++) {
        /* PTX: mul.rn.bf16 + 3× fma.rn.bf16 (4-element dot product)
         * Parser target: recognize as one column of W1×X
         */
        __nv_bfloat16 acc = B1[j];  /* ISA: LD.f R_acc, [R_b1 + j] */
        #pragma unroll
        for (int k = 0; k < N; k++) {
            /* PTX: fma.rn.bf16 %acc, %w1k, %xkj, %acc
             * Parser: part of 4-element dot product → WMMA.MMA candidate
             */
            acc = __hfma(W1[k], X[k * N + j], acc);
        }
        /* ReLU: max(acc, 0)
         * PTX: max.bf16 %h, %acc, 0x0000
         * ISA: MAX.f R_h, R_acc, R_zero  (or dedicated ReLU)
         */
        __nv_bfloat16 zero = bf16(0.0f);
        H[j] = __hmax(acc, zero);   /* ISA: MAX.f Rn, Rn, R_zero */
    }

    /* ═════════════════════════════════════════════════════════
     * LAYER 2: Y = W2 × H + B2   (no activation — raw scores)
     * ═════════════════════════════════════════════════════════ */

    #pragma unroll
    for (int j = 0; j < N; j++) {
        __nv_bfloat16 acc = B2[j];
        #pragma unroll
        for (int k = 0; k < N; k++) {
            acc = __hfma(W2[k], H[k * N + j], acc);
        }
        Y[j] = acc;             /* ISA: ST.f Rn, [R_base + j] */
    }
}


/* ═══════════════════════════════════════════════════════════════
 *  Kernel 2: Direct ISA-mapped version
 * ═══════════════════════════════════════════════════════════════ */

__global__ void ann_infer_direct(__nv_bfloat16* dmem) {
    const int tid = threadIdx.x;
    const int base = tid * DMEM_DEPTH;

    /* ── Load W1 row (4 bf16) ──────────────────────────────
     * ISA: MOVI R0, ADDR_W1
     *      WMMA.LOAD R0, R0     ; R0=W1[t][0], R1=W1[t][1], R2=W1[t][2], R3=W1[t][3]
     */
    __nv_bfloat16 w0 = dmem[base + ADDR_W1 + 0];
    __nv_bfloat16 w1 = dmem[base + ADDR_W1 + 1];
    __nv_bfloat16 w2 = dmem[base + ADDR_W1 + 2];
    __nv_bfloat16 w3 = dmem[base + ADDR_W1 + 3];

    /* ── Load X row (4 bf16) ───────────────────────────────
     * ISA: MOVI R4, ADDR_X
     *      WMMA.LOAD R4, R4     ; R4=X[t][0], R5=X[t][1], R6=X[t][2], R7=X[t][3]
     */
    __nv_bfloat16 x0 = dmem[base + ADDR_X + 0];
    __nv_bfloat16 x1 = dmem[base + ADDR_X + 1];
    __nv_bfloat16 x2 = dmem[base + ADDR_X + 2];
    __nv_bfloat16 x3 = dmem[base + ADDR_X + 3];

    /* ── Load B1 row (4 bf16) ──────────────────────────────
     * ISA: MOVI R8,  B1[t][0]   (or LD.f from DMEM)
     *      MOVI R9,  B1[t][1]
     *      MOVI R10, B1[t][2]
     *      MOVI R11, B1[t][3]
     */
    __nv_bfloat16 b0 = dmem[base + ADDR_B1 + 0];
    __nv_bfloat16 b1_val = dmem[base + ADDR_B1 + 1];
    __nv_bfloat16 b2 = dmem[base + ADDR_B1 + 2];
    __nv_bfloat16 b3 = dmem[base + ADDR_B1 + 3];

    /* Compute 4 output elements of row t */
    __nv_bfloat16 h0 = __hfma(w0, x0, __hfma(w1, x1, __hfma(w2, x2, __hfma(w3, x3, b0))));
    __nv_bfloat16 h1 = __hfma(w0, x0, __hfma(w1, x1, __hfma(w2, x2, __hfma(w3, x3, b1_val))));
    __nv_bfloat16 h2 = __hfma(w0, x0, __hfma(w1, x1, __hfma(w2, x2, __hfma(w3, x3, b2))));
    __nv_bfloat16 h3 = __hfma(w0, x0, __hfma(w1, x1, __hfma(w2, x2, __hfma(w3, x3, b3))));

    /* ── ReLU activation ───────────────────────────────────
     * ISA: MOVI R_zero, 0
     *      MAX.f R12, R12, R_zero
     *      MAX.f R13, R13, R_zero
     *      MAX.f R14, R14, R_zero
     *      MAX.f R15, R15, R_zero
     */
    __nv_bfloat16 zero = bf16(0.0f);
    h0 = __hmax(h0, zero);
    h1 = __hmax(h1, zero);
    h2 = __hmax(h2, zero);
    h3 = __hmax(h3, zero);

    /* ── Store H to DMEM ───────────────────────────────────
     * ISA: MOVI R0, ADDR_H
     *      WMMA.STORE R12, R0, 0
     */
    dmem[base + ADDR_H + 0] = h0;
    dmem[base + ADDR_H + 1] = h1;
    dmem[base + ADDR_H + 2] = h2;
    dmem[base + ADDR_H + 3] = h3;

    /* ── Load W2, B2, reload H ─────────────────────────────
     * ISA: MOVI R0, ADDR_W2
     *      WMMA.LOAD R0, R0
     *      MOVI R4, ADDR_H
     *      WMMA.LOAD R4, R4     ; reload H as "B" matrix for layer 2
     *      MOVI R8..R11, B2 row
     */
    w0 = dmem[base + ADDR_W2 + 0];
    w1 = dmem[base + ADDR_W2 + 1];
    w2 = dmem[base + ADDR_W2 + 2];
    w3 = dmem[base + ADDR_W2 + 3];

    b0      = dmem[base + ADDR_B2 + 0];
    b1_val  = dmem[base + ADDR_B2 + 1];
    b2      = dmem[base + ADDR_B2 + 2];
    b3      = dmem[base + ADDR_B2 + 3];

    /* ── Layer 2: Y = W2 × H + B2 ─────────────────────────
     * ISA: WMMA.MMA R12, R0, R4, R8
     */
    __nv_bfloat16 y0 = __hfma(w0, h0, __hfma(w1, h1, __hfma(w2, h2, __hfma(w3, h3, b0))));
    __nv_bfloat16 y1 = __hfma(w0, h0, __hfma(w1, h1, __hfma(w2, h2, __hfma(w3, h3, b1_val))));
    __nv_bfloat16 y2 = __hfma(w0, h0, __hfma(w1, h1, __hfma(w2, h2, __hfma(w3, h3, b2))));
    __nv_bfloat16 y3 = __hfma(w0, h0, __hfma(w1, h1, __hfma(w2, h2, __hfma(w3, h3, b3))));

    /* ── Store Y to DMEM ───────────────────────────────────
     * ISA: MOVI R0, ADDR_Y
     *      WMMA.STORE R12, R0, 0
     *      RET
     */
    dmem[base + ADDR_Y + 0] = y0;
    dmem[base + ADDR_Y + 1] = y1;
    dmem[base + ADDR_Y + 2] = y2;
    dmem[base + ADDR_Y + 3] = y3;
}


/* ═══════════════════════════════════════════════════════════════
 *  Kernel 3: Hand-optimized target ISA (as CUDA for validation)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef VALIDATE
#include <cstdio>
#include <cstring>
#include <cmath>

/* Example weights for power quality ANN (pre-trained, quantized to bf16) */
static float W1_float[4][4] = {
    { 0.5f,  0.3f, -0.2f,  0.1f},   /* row 0: voltage feature emphasis */
    { 0.1f,  0.6f,  0.2f, -0.1f},   /* row 1: current feature emphasis */
    {-0.1f,  0.2f,  0.7f,  0.0f},   /* row 2: THD feature emphasis */
    { 0.0f, -0.1f,  0.1f,  0.8f}    /* row 3: frequency feature emphasis */
};

static float B1_float[4][4] = {
    { 0.1f,  0.1f,  0.1f,  0.1f},
    { 0.0f,  0.0f,  0.0f,  0.0f},
    {-0.05f,-0.05f,-0.05f,-0.05f},
    { 0.0f,  0.0f,  0.0f,  0.0f}
};

static float W2_float[4][4] = {
    { 0.8f, -0.3f, -0.2f, -0.1f},   /* row 0: normal */
    {-0.2f,  0.7f,  0.1f,  0.0f},   /* row 1: sag/swell */
    {-0.1f,  0.1f,  0.8f, -0.1f},   /* row 2: harmonic */
    { 0.0f, -0.1f, -0.1f,  0.9f}    /* row 3: frequency */
};

static float B2_float[4][4] = {
    { 0.1f,  0.1f,  0.1f,  0.1f},
    {-0.1f, -0.1f, -0.1f, -0.1f},
    { 0.0f,  0.0f,  0.0f,  0.0f},
    { 0.0f,  0.0f,  0.0f,  0.0f}
};

/* 4 test packets (columns of X) */
static float X_float[4][4] = {
    /* Packet 0: normal */
    { 0.02f,  0.01f,  0.03f, 0.001f},
    /* Packet 1: voltage sag */
    { 0.15f,  0.05f,  0.04f, 0.002f},
    /* Packet 2: harmonic distortion */
    { 0.03f,  0.02f,  0.25f, 0.001f},
    /* Packet 3: frequency event */
    { 0.01f,  0.01f,  0.02f, 0.10f}
};

void host_reference(float Y_ref[4][4]) {
    float H[4][4];
    /* Layer 1 */
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float sum = B1_float[i][j];
            for (int k = 0; k < 4; k++)
                sum += W1_float[i][k] * X_float[k][j];
            H[i][j] = fmaxf(sum, 0.0f);  /* ReLU */
        }
    /* Layer 2 */
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float sum = B2_float[i][j];
            for (int k = 0; k < 4; k++)
                sum += W2_float[i][k] * H[k][j];
            Y_ref[i][j] = sum;
        }
}

int main() {
    /* Allocate device DMEM: 4 banks × DMEM_DEPTH */
    __nv_bfloat16* d_dmem;
    size_t dmem_bytes = 4 * DMEM_DEPTH * sizeof(__nv_bfloat16);
    cudaMalloc(&d_dmem, dmem_bytes);
    cudaMemset(d_dmem, 0, dmem_bytes);

    /* Pack data into per-bank layout */
    __nv_bfloat16 h_dmem[4 * DMEM_DEPTH] = {};
    for (int t = 0; t < 4; t++) {
        int base = t * DMEM_DEPTH;
        for (int j = 0; j < 4; j++) {
            h_dmem[base + ADDR_X  + j] = __float2bfloat16(X_float[t][j]);
            h_dmem[base + ADDR_W1 + j] = __float2bfloat16(W1_float[t][j]);
            h_dmem[base + ADDR_B1 + j] = __float2bfloat16(B1_float[t][j]);
            h_dmem[base + ADDR_W2 + j] = __float2bfloat16(W2_float[t][j]);
            h_dmem[base + ADDR_B2 + j] = __float2bfloat16(B2_float[t][j]);
        }
    }
    cudaMemcpy(d_dmem, h_dmem, dmem_bytes, cudaMemcpyHostToDevice);

    /* Launch: 1 block, 4 threads (= 4 SP lanes) */
    ann_infer_bf16<<<1, 4>>>(d_dmem);
    cudaDeviceSynchronize();

    /* Read back results */
    cudaMemcpy(h_dmem, d_dmem, dmem_bytes, cudaMemcpyDeviceToHost);

    /* Compare with host reference */
    float Y_ref[4][4];
    host_reference(Y_ref);

    printf("ANN Inference Results (4 packets × 4 classes):\n");
    printf("%-8s %-10s %-10s %-10s %-10s\n", "Packet", "Normal", "Sag/Swell", "Harmonic", "Freq");
    for (int j = 0; j < 4; j++) {
        printf("  P%d:    ", j);
        for (int i = 0; i < 4; i++) {
            int base = i * DMEM_DEPTH;
            float gpu_val = __bfloat162float(h_dmem[base + ADDR_Y + j]);
            float ref_val = Y_ref[i][j];
            printf("%.4f%s  ", gpu_val, fabsf(gpu_val - ref_val) > 0.1f ? "!" : " ");
        }
        printf("\n");
    }

    /* Find max class per packet */
    printf("\nClassification:\n");
    const char* classes[] = {"Normal", "Sag/Swell", "Harmonic", "Frequency"};
    for (int j = 0; j < 4; j++) {
        int best = 0;
        float best_val = -1e9f;
        for (int i = 0; i < 4; i++) {
            float v = Y_ref[i][j];
            if (v > best_val) { best_val = v; best = i; }
        }
        printf("  Packet %d: %s (score=%.3f)\n", j, classes[best], best_val);
    }

    cudaFree(d_dmem);
    return 0;
}
#endif /* VALIDATE */
