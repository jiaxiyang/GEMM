/usr/local/cuda/bin/nvcc --std=c++17 -I../../cutlass/include/ -I../../cutlass/tools/util/include/ --ptxas-options=-v -O3 -o cutlass_gemm cutlass_gemm.cu 
