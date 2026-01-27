#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

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

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// Exposed C function for Python (ctypes/cffi)
// h_A, h_B, h_C are HOST pointers to NxN float arrays (row-major)
extern "C" __declspec(dllexport)

void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = (size_t)N * (size_t)N * sizeof(float);
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}