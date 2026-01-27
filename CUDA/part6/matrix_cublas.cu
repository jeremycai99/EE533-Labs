// matrix_cublas_sgemm.cu
// Minimal cuBLAS SGEMM (float) + kernel timing (ms)
// Compile: nvcc -O2 matrix_cublas_sgemm.cu -o matrix_cublas_sgemm.exe -lcublas
// Run:     matrix_cublas_sgemm.exe 1024

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char** argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    // Host memory
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    if (!hA || !hB || !hC) {
        printf("Host malloc failed\n");
        return 1;
    }

    // Initialize A and B
    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    // Device memory
    float *dA = NULL, *dB = NULL, *dC = NULL;
    if (cudaMalloc((void**)&dA, bytes) != cudaSuccess ||
        cudaMalloc((void**)&dB, bytes) != cudaSuccess ||
        cudaMalloc((void**)&dC, bytes) != cudaSuccess) {
        printf("cudaMalloc failed\n");
        return 1;
    }

    // Copy to device
    if (cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cudaMemcpy H2D failed\n");
        return 1;
    }

    // cuBLAS handle
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate failed\n");
        return 1;
    }

    // SGEMM params: C = alpha * op(A) * op(B) + beta * C
    // NOTE: cuBLAS assumes column-major storage by default.
    // To treat our row-major arrays as-is, we compute:
    //   C^T = B^T * A^T
    // which corresponds to swapping A/B in the call below.
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Timing (GPU time for the cuBLAS call)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                dB, N,   // B first
                dA, N,   // A second
                &beta,
                dC, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                dB, N,
                dA, N,
                &beta,
                dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy back
    if (cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cudaMemcpy D2H failed\n");
        return 1;
    }

    // Print time + one value
    printf("N=%d, time_ms=%f, C0=%f\n", N, ms, hC[0]);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}