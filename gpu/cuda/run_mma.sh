/usr/local/cuda/bin/nvcc  --std=c++17 -I../../cutlass/include/ -I../../cutlass/tools/util/include/ --ptxas-options=-v -O3 -o mma_gemm mma_gemm.cu -arch=sm_86
./mma_gemm