set -ex
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=7
/usr/local/cuda/bin/nvcc -O3 -o cuda_sgemm cuda_sgemm.cu -lcublas
./cuda_sgemm
python3 plot_bar.py cuda_sgemm.csv cuda_sgemm.png
# /usr/local/cuda/bin/nvcc -O3 -o matmul matmul.cu -lcublas
# ./matmul