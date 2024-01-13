#!/usr/bin/env python3
import numpy as np
import time
import argparse
from prettytable import PrettyTable

def profiling(num):
    N = num
    # memory: N * N 
    A = np.random.randn(N, N).astype(np.float32)

    # memory: N * N 
    B = np.random.randn(N, N).astype(np.float32)
    
    # memory: N * N; comput: N * N * 2 N
    flops = N * N * 2 * N / 1e9
    # print(f"{flops / 1e9:.2f} GFlops")

    # warmup
    C = A @ B
    
    tic = time.monotonic()
    C = A @ B
    toc = time.monotonic()
    s = toc - tic
    flops_rate = flops / s
    return flops, s, flops_rate
    # print(f"{flops / (toc - tic) / 1e9} GFLOPS")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process matmul.')
    # parser.add_argument("n", type=int)
    parser.add_argument('-n', '--num', type=int) 
    args = parser.parse_args()
    # N = args.num

    table = PrettyTable(['num', 'GFlops', 'time(ms)', 'GFLOPS'])
    for N in [2**n for n in range(7, 15)]:
        flops, s, flops_rate = profiling(N)
        table.add_row([N, f"{flops:.2f}", f"{s*1000:.2f}", f"{flops_rate:.2f}"])

    print(table)

'''
+-------+---------+----------+---------+
|  num  |  GFlops | time(ms) |  GFLOPS |
+-------+---------+----------+---------+
|  128  |   0.00  |   0.95   |   4.43  |
|  256  |   0.03  |   0.80   |  41.82  |
|  512  |   0.27  |   0.61   |  436.77 |
|  1024 |   2.15  |   1.99   | 1077.64 |
|  2048 |  17.18  |  123.69  |  138.90 |
|  4096 |  137.44 |  183.71  |  748.13 |
|  8192 | 1099.51 |  457.52  | 2403.21 |
| 16384 | 8796.09 | 3107.55  | 2830.56 |
+-------+---------+----------+---------+
从表中可以看出 2048 执行效率比较差
'''