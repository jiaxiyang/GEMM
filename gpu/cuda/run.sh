set -ex
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
# /usr/local/cuda/bin/nvcc --ptxas-options=-v -O3 -o cuda_sgemm cuda_sgemm.cu -lcublas -arch=sm_86
# ./cuda_sgemm
# python3 plot_bar.py cuda_sgemm.csv cuda_sgemm.png

# /usr/local/cuda/bin/nvcc --resource-usage -O3 -o my_sgemm my_sgemm.cu -lcublas  2>&1 | c++filt 
/usr/local/cuda/bin/nvcc --std=c++17 -I../../cutlass/include/ -I../../cutlass/tools/util/include/ --ptxas-options=-v -O3 -o my_gemm my_gemm.cu cutlass_gemm.cu ampere_tf32_tensorop_gemm.cu  ampere_fp16_tensorop_gemm.cu -lcublas -arch=sm_86
./my_gemm
python3 plot_bar.py my_gemm.csv my_gemm.png