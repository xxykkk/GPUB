#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <chrono>

#define N 4096  // 假设矩阵大小是 1024x1024

using namespace nvcuda;

// Tensor Core 内核函数 (TF32 版本)：计算 D = A * B + C
__global__ void tensorCoreMatrixMultiplyAdd_TF32(float *A, float *B, float *C, float *D, int n) {
    // TF32 推荐的 WMMA tile 尺寸配置是 16x16x8
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 8;

    int warpM = blockIdx.y; 
    int warpN = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);

    if (warpM >= n / WMMA_M || warpN >= n / WMMA_N) return;

    // 声明 WMMA 数据片段 (注意：这里使用的是 wmma::precision::tf32)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    // 初始化累加器
    wmma::load_matrix_sync(acc_frag, C + cRow * n + cCol, n, wmma::mem_row_major);

    // 遍历 K 维度进行乘加运算，步长变为 WMMA_K (即 8)
    for (int i = 0; i < n; i += WMMA_K) {
        // 全局内存直接是 float 类型
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * n + i, n);
        wmma::load_matrix_sync(b_frag, B + i * n + warpN * WMMA_N, n);
        
        // Tensor Core 执行 TF32 乘加
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 存回结果
    wmma::store_matrix_sync(D + cRow * n + cCol, acc_frag, n, wmma::mem_row_major);
}

int main() {
    // 所有矩阵都直接使用标准单精度浮点数 (float)
    int size = N * N * sizeof(float);
    
    float *A, *B, *C, *D;
    float *d_A, *d_B, *d_C, *d_D;

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);
    D = (float*)malloc(size);

    // 不需要麻烦的类型转换，直接赋 float 值
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(128); // 4个 Warp
    dim3 numBlocks((N / 16) / 4, N / 16); // 切块逻辑不变

    // 预热 GPU
    tensorCoreMatrixMultiplyAdd_TF32<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int times = 1000;
    for (int i = 0; i < times; i++) {
        tensorCoreMatrixMultiplyAdd_TF32<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for " << times << " iterations: (TF32 Tensor Core) " << milliseconds / 1000 << " s" << std::endl;

    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);
    std::cout << "D[0][0] = " << D[0] << " (Expected: 2051)" << std::endl;

    free(A); free(B); free(C); free(D);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}