# GPUB
The benchmark test for GPU, including Tensor Floats. 

## Simple Test
command: ```python tensorcore.py```

## CPU Test
command: ```g++ gemm.cpp -o gemm && ./gemm```

## CUDA Test (FP32/FP16)
command: 

```

### FP32: 
nvcc -arch=sm_70 cuda_gemm.cu -o cuda_gemm && ./cuda_gemm
### FP16:
nvcc -arch=sm_70 cuda_gemm.cu -o cuda_gemm && ./cuda_gemm --fp16
```


## TF32 Test

command: 
```
nvcc tf32_gemm.cu -arch=sm_80 -lcublas -o tf32_gemm && ./tf32_gemm
```

## TF16 Test
command:
```
nvcc tf16_gemm.cu -arch=sm_70 -lcublas -o tf16_gemm && ./tf16_gemm
```






