#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

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

    // Launch config
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // ---- Timing: CUDA events (kernel time only) ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Optional warmup (keeps timing more stable)
    matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Minimal error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back (also syncs)
    if (cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cudaMemcpy D2H failed\n");
        return 1;
    }

    // Output in a CPU-like way: report size + time
    printf("GPU (naive cuda) execution time N=%d, time_ms=%f, C0=%f\n", N, ms, hC[0]);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;

}
