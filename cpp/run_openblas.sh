g++ -Ofast -o openblas_matmul_example openblas.cpp -std=c++11 -lopenblas -I/mnt/data-2/home/xiyang.jia/pixi/.pixi/env/include/openblas/ -L/mnt/data-2/home/xiyang.jia/pixi/.pixi/env/lib64/ -pthread
./openblas_matmul_example
OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 ./openblas_matmul_example
rm openblas_matmul_example

