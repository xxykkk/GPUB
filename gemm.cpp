#include <iostream>
#include <ctime>  // 用于 C 语言的时间计时

#define N 1024  // 矩阵大小为 1024x1024

void matrixMultiplyCPU(float *A, float *B, float *C, float *D, int n) {
    // CPU 计算矩阵 D = A * B + C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float value = 0.0f;
            for (int k = 0; k < n; k++) {
                value += A[i * n + k] * B[k * n + j];
            }
            D[i * n + j] = value + C[i * n + j];
        }
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *A, *B, *C, *D;

    // 分配主机内存
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);
    D = (float*)malloc(size);

    // 初始化矩阵 A, B, C
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;  // 假设 A 为全 1 矩阵
        B[i] = 2.0f;  // 假设 B 为全 2 矩阵
        C[i] = 3.0f;  // 假设 C 为全 3 矩阵
    }

    // 计时开始（CPU）
    clock_t startCPU = clock();

    // 执行 CPU 矩阵乘法 1000 次
    for (int i = 0; i < 10; i++) {
        matrixMultiplyCPU(A, B, C, D, N);
    }

    // 计时结束（CPU）
    clock_t endCPU = clock();
    double cpu_time = (double)(endCPU - startCPU) / CLOCKS_PER_SEC;

    // 输出 CPU 执行时间
    std::cout << "CPU time for 1000 iterations: " << cpu_time << " s" << std::endl;

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}