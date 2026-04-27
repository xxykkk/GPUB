#include <iostream>
#include <cuda_runtime.h>
#include <chrono>


#define N 1024  // 假设矩阵大小是 1024x1024

// CUDA内核函数：计算 C = A * B + C
__global__ void matrixMultiplyAdd(float *A, float *B, float *C, float *D, int n) {
    // 每个线程负责计算 D(i, j)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        // 计算 A * B
        for (int k = 0; k < n; k++) {
            value += A[row * n + k] * B[k * n + col];
        }
        // 加上 C
        D[row * n + col] = value + C[row * n + col];
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *A, *B, *C, *D;
    float *d_A, *d_B, *d_C, *d_D;

    // 分配主机内存
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);
    D = (float*)malloc(size);

    // 初始化矩阵A, B, C
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f; // 假设 A 为全 1 矩阵
        B[i] = 2.0f; // 假设 B 为全 2 矩阵
        C[i] = 3.0f; // 假设 C 为全 3 矩阵
    }

    // 分配设备内存
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);

    // 将数据从主机内存拷贝到设备内存
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    // 设置线程块和网格尺寸
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    matrixMultiplyAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N); //预热
    
    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // auto start = std::chrono::high_resolution_clock::now();  // 记录开始时间

    // 启动 CUDA 内核
    for (int i=0;i<10000;i++){
        matrixMultiplyAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
        // 将计算结果从设备内存拷贝回主机内存
    }
    

    // auto end = std::chrono::high_resolution_clock::now();  // 记录结束时间
    // std::chrono::duration<double> duration = end - start;
    // std::cout << "Time for 1000 iterations: (C++) " << duration.count() * 1000 << " ms" << std::endl;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for 1000 iterations: (CUDA) " << milliseconds / 1000 << " s" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);

    // 输出一部分结果
    std::cout << "D[0][0] = " << D[0] << std::endl;

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);



    return 0;
}