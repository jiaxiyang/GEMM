# /usr/local/cuda/bin/nvcc --std=c++17 -I../../cutlass/include/ -I../../cutlass/tools/util/include/ --ptxas-options=-v -O3 -o cutlass_gemm cutlass_gemm.cu -arch=sm_86
# ./cutlass_gemm
/usr/local/cuda/bin/nvcc --std=c++17 -I../../cutlass/include/ -I../../cutlass/tools/util/include/ --ptxas-options=-v -O3 -o ampere_tf32_tensorop_gemm ampere_tf32_tensorop_gemm.cu  -arch=sm_86
./ampere_tf32_tensorop_gemm