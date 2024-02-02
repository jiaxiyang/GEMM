#include <stdint.h>
#include "cuda_fp16.h"
#include <stdio.h>
#include <cuda_runtime.h>

inline __device__ __host__ size_t div_ceil(size_t a, size_t b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32
const int TESTNUM = 13;
const int M_list[TESTNUM] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
const int N_list[TESTNUM] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
const int K_list[TESTNUM] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
const int outer_repeat = 1, inner_repeat = 10;

__global__ void mmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K)
{
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N)
    {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i)
    {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2)
        {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < MMA_M)
    {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}

void mmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K)
{
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

float testPerformance(
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat)
{

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        mmaNaive(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

void test_mma_fp16()
{
    {
        printf("\nKernal = mySgemmV1Aligned\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;

        {

            for (int i = 0; i < TESTNUM; i++)
            {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = 0.0;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++)
                {
                    double this_sec = testPerformance(gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
                // performance[M] = avg_Gflops;
                // data[i].mySgemmV1Aligned = avg_Gflops;
                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }
}
int main(int argc, char const *argv[])
{
    /* code */
    test_mma_fp16();
    return 0;
}
