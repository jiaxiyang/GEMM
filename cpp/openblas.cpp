#include <cblas.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>

// 初始化矩阵的函数
void initialize_matrix(std::vector<float>& matrix, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dis(gen);
    }
}

int main() {
    std::vector<int> sizes;
    for (int i = 7; i <= 14; ++i) {
        sizes.push_back(std::pow(2, i));
    }

    // 打印表头
    std::cout << "+-------+---------+----------+---------+\n";
    std::cout << "|   num |  GFlops | time(ms) |  GFLOPS |\n";
    std::cout << "+-------+---------+----------+---------+\n";

    for (int N : sizes) {
        std::vector<float> A(N * N), B(N * N), C(N * N);

        initialize_matrix(A, N);
        initialize_matrix(B, N);

        auto start = std::chrono::steady_clock::now();

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;

        double gflops = 2.0 * N * N * N / (1e9);
        double time_ms = diff.count() * 1000;
        double gflops_rate = gflops / time_ms * 1000;

        // 打印每行结果
        std::cout << "| " << std::right << std::setw(5) << N << " | "
                  << std::fixed << std::setprecision(2) << std::setw(7) << gflops << " | "
                  << std::setw(8) << time_ms << " | "
                  << std::setw(7) << gflops_rate << " |\n";
    }

    std::cout << "+-------+---------+----------+---------+\n";

    return 0;
}

/*
./openblas_matmul_example
+-------+---------+----------+---------+
|   num |  GFlops | time(ms) |  GFLOPS |
+-------+---------+----------+---------+
|   128 |    0.00 |     0.32 |   12.93 |
|   256 |    0.03 |     0.64 |   52.53 |
|   512 |    0.27 |     1.45 |  185.50 |
|  1024 |    2.15 |     3.77 |  570.13 |
|  2048 |   17.18 |    21.07 |  815.39 |
|  4096 |  137.44 |   156.53 |  878.04 |
|  8192 | 1099.51 |  1135.06 |  968.68 |
| 16384 | 8796.09 |  6381.39 | 1378.40 |
+-------+---------+----------+---------+

OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 ./openblas_matmul_example
+-------+---------+----------+---------+
|   num |  GFlops | time(ms) |  GFLOPS |
+-------+---------+----------+---------+
|   128 |    0.00 |     0.56 |    7.55 |
|   256 |    0.03 |     0.32 |  103.71 |
|   512 |    0.27 |     0.51 |  523.20 |
|  1024 |    2.15 |     1.92 | 1117.83 |
|  2048 |   17.18 |    12.23 | 1405.11 |
|  4096 |  137.44 |    85.48 | 1607.80 |
|  8192 | 1099.51 |   512.29 | 2146.28 |
| 16384 | 8796.09 |  4236.85 | 2076.09 |
+-------+---------+----------+---------+

1. 当计算量较小时, 多线程加速会影响执行速度
*/