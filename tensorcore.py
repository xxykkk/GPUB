import torch
import time

if not torch.cuda.is_available():
    raise SystemError("未检测到 GPU")

def benchmark(size, label, use_fp16=True, iterations=100):
    # 根据 size 生成矩阵
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    
    if use_fp16:
        a = a.half()
        b = b.half()
        
    # 预热
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    tflops = (2 * (size ** 3)) / avg_time / 1e12
    print(f"[{label}] 矩阵大小: {size}x{size} | 平均耗时: {avg_time*1000:.3f} ms | 吞吐量: {tflops:.2f} TFLOPS")

print("--- V100 算力单元解剖测试 ---")

# 1. 基准：FP32 正常 CUDA Core
benchmark(8192, "FP32 (普通 CUDA Core)", use_fp16=False)

# 2. 控制变量：FP16 正常 CUDA Core (故意用 8191 破坏对齐，禁用 Tensor Core)
benchmark(8191, "FP16 (普通 CUDA Core - 未对齐)", use_fp16=True)

# 3. 火力全开：FP16 Tensor Core (使用 8192 完美对齐，触发 Tensor Core)
benchmark(8192, "FP16 (Tensor Core 满血版 - 对齐)", use_fp16=True)