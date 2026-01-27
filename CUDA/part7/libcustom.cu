#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#ifdef _WIN32
  #define EXPORT extern "C" __declspec(dllexport)
#else
  #define EXPORT extern "C"
#endif


// Simple CUDA error check (minimal)
static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
    }
}


__global__ void conv2d_u8_same_kernel(const unsigned char* img,
                                      const float* ker,
                                      float* out,
                                      int M, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (x >= M || y >= M) return;

    int r = N / 2;
    float sum = 0.0f;

    for (int ky = 0; ky < N; ky++) {
        for (int kx = 0; kx < N; kx++) {
            int iy = y + ky - r;
            int ix = x + kx - r;
            if (iy >= 0 && iy < M && ix >= 0 && ix < M) {
                unsigned char pix = img[iy * M + ix];
                sum += ((float)pix) * ker[ky * N + kx];
            }
        }
    }
    out[y * M + x] = sum;
}

//Non-CUDA reference implementation: CPU convolution
EXPORT float cpu_convolve_u8(const unsigned char* image, const float* kernel,
                           float* out, int M, int N) {
#ifdef _WIN32
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
#endif

    int r = N / 2;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < N; ky++) {
                for (int kx = 0; kx < N; kx++) {
                    int iy = y + ky - r;
                    int ix = x + kx - r;
                    if (iy >= 0 && iy < M && ix >= 0 && ix < M) {
                        unsigned char pix = image[iy * M + ix];
                        sum += ((float)pix) * kernel[ky * N + kx];
                    }
                }
            }
            out[y * M + x] = sum;
        }
    }
#ifdef _WIN32
    QueryPerformanceCounter(&t1);
    double ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
    return (float)ms;
#else
    return -1.0f;
#endif
}

EXPORT float gpu_convolve_u8(const unsigned char* h_img,
                            const float* h_ker,
                            float* h_out,
                            int M, int N)
{
    if (!h_img || !h_ker || !h_out || M <= 0 || N <= 0) return -1.0f;

    size_t imgBytes = (size_t)M * (size_t)M * sizeof(unsigned char);
    size_t kerBytes = (size_t)N * (size_t)N * sizeof(float);
    size_t outBytes = (size_t)M * (size_t)M * sizeof(float);

    unsigned char* d_img = NULL;
    float* d_ker = NULL;
    float* d_out = NULL;

    cudaError_t e;

    e = cudaMalloc((void**)&d_img, imgBytes); checkCuda(e, "cudaMalloc d_img");
    e = cudaMalloc((void**)&d_ker, kerBytes); checkCuda(e, "cudaMalloc d_ker");
    e = cudaMalloc((void**)&d_out, outBytes); checkCuda(e, "cudaMalloc d_out");

    e = cudaMemcpy(d_img, h_img, imgBytes, cudaMemcpyHostToDevice); checkCuda(e, "H2D img");
    e = cudaMemcpy(d_ker, h_ker, kerBytes, cudaMemcpyHostToDevice); checkCuda(e, "H2D ker");

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // Timing: kernel only
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Optional warm-up (helps stabilize measurements)
    conv2d_u8_same_kernel<<<grid, block>>>(d_img, d_ker, d_out, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    conv2d_u8_same_kernel<<<grid, block>>>(d_img, d_ker, d_out, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Catch kernel launch/runtime errors
    e = cudaGetLastError(); checkCuda(e, "kernel launch");
    e = cudaDeviceSynchronize(); checkCuda(e, "kernel sync");

    e = cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost); checkCuda(e, "D2H out");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_img);
    cudaFree(d_ker);
    cudaFree(d_out);

    return ms;
}