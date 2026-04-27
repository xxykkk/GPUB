#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>

#define N 1024  // 假设矩阵大小是 1024x1024

using namespace nvcuda;

// Tensor Core 内核函数：计算 D = A * B + C
__global__ void tensorCoreMatrixMultiplyAdd(__half *A, __half *B, float *C, float *D, int n) {
    // WMMA 的 tile 尺寸配置 (16x16x16 是一种标准配置)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // 计算当前 Warp 负责的输出块 (Tile) 坐标
    // blockDim.x 是 128 (4个warp)，所以用 threadIdx.x / 32 计算当前 block 内的 warp ID
    int warpM = blockIdx.y; 
    int warpN = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);

    // 边界检查
    if (warpM >= n / WMMA_M || warpN >= n / WMMA_N) return;

    // 声明 WMMA 数据片段 (Fragments)
    // 注意：我们将 A 和 B 设置为 row_major (行优先)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    // 将矩阵 C 的数据加载到累加器中，作为初始值 (即完成 + C 的操作)
    wmma::load_matrix_sync(acc_frag, C + cRow * n + cCol, n, wmma::mem_row_major);

    // 遍历 K 维度进行乘加运算
    for (int i = 0; i < n; i += WMMA_K) {
        // 从全局内存加载 A 和 B 到寄存器片段
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * n + i, n);
        wmma::load_matrix_sync(b_frag, B + i * n + warpN * WMMA_N, n);
        
        // Tensor Core 核心计算指令：acc_frag = a_frag * b_frag + acc_frag
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 将结果存回全局内存的矩阵 D
    wmma::store_matrix_sync(D + cRow * n + cCol, acc_frag, n, wmma::mem_row_major);
}

int main() {
    // A 和 B 使用半精度 (FP16)，C 和 D 使用单精度 (FP32)
    int size_fp16 = N * N * sizeof(__half);
    int size_fp32 = N * N * sizeof(float);
    
    __half *A, *B;
    float *C, *D;
    __half *d_A, *d_B;
    float *d_C, *d_D;

    // 分配主机内存
    A = (__half*)malloc(size_fp16);
    B = (__half*)malloc(size_fp16);
    C = (float*)malloc(size_fp32);
    D = (float*)malloc(size_fp32);

    // 初始化矩阵 (由于 A 和 B 是 __half 类型，需要用 __float2half 进行转换)
    for (int i = 0; i < N * N; i++) {
        A[i] = __float2half(1.0f); // 假设 A 为全 1 矩阵
        B[i] = __float2half(2.0f); // 假设 B 为全 2 矩阵
        C[i] = 3.0f;               // 假设 C 为全 3 矩阵
    }

    // 分配设备内存
    cudaMalloc((void**)&d_A, size_fp16);
    cudaMalloc((void**)&d_B, size_fp16);
    cudaMalloc((void**)&d_C, size_fp32);
    cudaMalloc((void**)&d_D, size_fp32);

    // 将数据拷贝到设备
    cudaMemcpy(d_A, A, size_fp16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_fp16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_fp32, cudaMemcpyHostToDevice);

    // 设置线程块和网格尺寸
    // 每个线程块设置 128 个线程 (即 4 个 Warp)
    dim3 threadsPerBlock(128);
    // N / 16 计算出总共需要多少个 tile。X 维度除以 4 是因为一个 block 处理了 4 个 tile。
    dim3 numBlocks((N / 16) / 4, N / 16);

    // 预热 GPU (去掉第一次启动的开销，让时间测量更准确)
    tensorCoreMatrixMultiplyAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    cudaDeviceSynchronize();

    // 创建 CUDA 事件测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 启动 Tensor Core 内核
    for (int i = 0; i < 10000; i++) {
        tensorCoreMatrixMultiplyAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for 10000 iterations: (Tensor Core) " << milliseconds / 1000 << " s" << std::endl;

    // 拷贝回结果
    cudaMemcpy(D, d_D, size_fp32, cudaMemcpyDeviceToHost);

    // 输出结果验证： 1.0 * 2.0 * 1024 + 3.0 = 2051
    std::cout << "D[0][0] = " << D[0] << " (Expected: 2051)" << std::endl;

    // 释放内存
    free(A); free(B); free(C); free(D);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}