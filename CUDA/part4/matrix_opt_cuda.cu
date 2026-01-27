// matrix_tiled_cuda.cu
// Tiled CUDA matmul using shared memory (TILE_WIDTH=16) + kernel timing (ms)
// Compile: nvcc -O2 matrix_tiled_cuda.cu -o matrix_tiled_cuda.exe
// Run:     matrix_tiled_cuda.exe 1024

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int m = 0; m < numTiles; ++m) {
        // Load A tile
        int aCol = m * TILE_WIDTH + tx;
        if (Row < N && aCol < N)
            ds_A[ty][tx] = A[Row * N + aCol];
        else
            ds_A[ty][tx] = 0.0f;

        // Load B tile
        int bRow = m * TILE_WIDTH + ty;
        if (Col < N && bRow < N)
            ds_B[ty][tx] = B[bRow * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
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

    // Initialize A and B (simple)
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

    // Launch config (matches TILE_WIDTH)
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing with CUDA events (kernel time only)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
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

    // Copy result back
    if (cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cudaMemcpy D2H failed\n");
        return 1;
    }

    // Output (time + one value)
    printf("N=%d, time_ms=%f, C0=%f\n", N, ms, hC[0]);

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