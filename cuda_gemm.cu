#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cuda_fp16.h> // 引入 FP16 支持
#include <string.h>

#define N 4096  // 假设矩阵大小是 NxN

// ---------------------------------------------------------
// FP32 内核函数：计算 C = A * B + C
// ---------------------------------------------------------
__global__ void matrixMultiplyAdd(float *A, float *B, float *C, float *D, int n) {
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

// ---------------------------------------------------------
// FP16 内核函数：计算 C = A * B + C (不使用 Tensor Core)
// ---------------------------------------------------------
__global__ void matrixMultiplyAddFP16(half *A, half *B, half *C, half *D, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        half value = __float2half(0.0f);
        for (int k = 0; k < n; k++) {
            // 使用重载的 FP16 乘法和加法运算符，调用普通的 FP16 CUDA Core
            value = value + A[row * n + k] * B[k * n + col];
        }
        D[row * n + col] = value + C[row * n + col];
    }
}

// ---------------------------------------------------------
// 数据格式转换内核：float -> half 和 half -> float
// ---------------------------------------------------------
__global__ void float2half_kernel(float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void half2float_kernel(half* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __half2float(in[idx]);
    }
}


int main(int argc, char** argv) {
    // 解析命令行参数
    bool use_fp16 = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fp16") == 0) {
            use_fp16 = true;
            break;
        }
    }

    if (use_fp16) {
        std::cout << "Mode: FP16 (Half Precision)" << std::endl;
    } else {
        std::cout << "Mode: FP32 (Single Precision)" << std::endl;
    }

    int size_fp32 = N * N * sizeof(float);
    int size_fp16 = N * N * sizeof(half);

    float *A, *B, *C, *D;
    float *d_A, *d_B, *d_C, *d_D;
    half *d_A_h = nullptr, *d_B_h = nullptr, *d_C_h = nullptr, *d_D_h = nullptr;

    // 分配主机内存
    A = (float*)malloc(size_fp32);
    B = (float*)malloc(size_fp32);
    C = (float*)malloc(size_fp32);
    D = (float*)malloc(size_fp32);

    // 初始化矩阵A, B, C
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f; // 假设 A 为全 1 矩阵
        B[i] = 2.0f; // 假设 B 为全 2 矩阵
        C[i] = 3.0f; // 假设 C 为全 3 矩阵
    }

    // 分配设备内存 (FP32)
    cudaMalloc((void**)&d_A, size_fp32);
    cudaMalloc((void**)&d_B, size_fp32);
    cudaMalloc((void**)&d_C, size_fp32);
    cudaMalloc((void**)&d_D, size_fp32);

    // 将数据从主机内存拷贝到设备内存
    cudaMemcpy(d_A, A, size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_fp32, cudaMemcpyHostToDevice);

    // 如果启用 FP16，在显存中申请 half 空间并转换格式
    if (use_fp16) {
        cudaMalloc((void**)&d_A_h, size_fp16);
        cudaMalloc((void**)&d_B_h, size_fp16);
        cudaMalloc((void**)&d_C_h, size_fp16);
        cudaMalloc((void**)&d_D_h, size_fp16);

        int threads = 256;
        int blocks = (N * N + threads - 1) / threads;
        float2half_kernel<<<blocks, threads>>>(d_A, d_A_h, N * N);
        float2half_kernel<<<blocks, threads>>>(d_B, d_B_h, N * N);
        float2half_kernel<<<blocks, threads>>>(d_C, d_C_h, N * N);
        cudaDeviceSynchronize();
    }

    // 设置线程块和网格尺寸
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 预热 (Warm-up)
    if (use_fp16) {
        matrixMultiplyAddFP16<<<numBlocks, threadsPerBlock>>>(d_A_h, d_B_h, d_C_h, d_D_h, N);
    } else {
        matrixMultiplyAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    }

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 启动 CUDA 内核
    int times = 1000;
    for (int i = 0; i < times; i++) {
        if (use_fp16) {
            matrixMultiplyAddFP16<<<numBlocks, threadsPerBlock>>>(d_A_h, d_B_h, d_C_h, d_D_h, N);
        } else {
            matrixMultiplyAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for " << times << " iterations: (CUDA) " << milliseconds / 1000 << " s" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 如果启用 FP16，将结果从 half 转回 float
    if (use_fp16) {
        int threads = 256;
        int blocks = (N * N + threads - 1) / threads;
        half2float_kernel<<<blocks, threads>>>(d_D_h, d_D, N * N);
        cudaDeviceSynchronize();
    }

    // 将计算结果从设备内存拷贝回主机内存
    cudaMemcpy(D, d_D, size_fp32, cudaMemcpyDeviceToHost);

    // 输出一部分结果验证正确性 (1024 * 1*2 + 3 = 2051)
    std::cout << "D[0][0] = " << D[0] << std::endl;

    // 释放内存
    free(A); free(B); free(C); free(D);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    
    if (use_fp16) {
        cudaFree(d_A_h); cudaFree(d_B_h); cudaFree(d_C_h); cudaFree(d_D_h);
    }

    return 0;
}