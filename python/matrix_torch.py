#!/usr/bin/env python3
import numpy as np
import time
import argparse
from sympy import false
import torch
from rich.progress import track
from rich.console import Console
from rich.table import Table
from rich import print
from rich import box

def profiling(num, use_cuda=False):
    N = num
    # memory: N * N 
    A = np.random.randn(N, N).astype(np.float32)
    A_tensor = torch.tensor(A)
    if use_cuda:
        A_tensor = A_tensor.cuda()

    # memory: N * N 
    B = np.random.randn(N, N).astype(np.float32)
    B_tensor = torch.tensor(B)
    if use_cuda:
        B_tensor = B_tensor.cuda()
    
    # memory: N * N; comput: N * N * 2 N
    flops = N * N * 2 * N / 1e9

    # warmup
    C = A_tensor @ B_tensor

    tic = time.monotonic()
    C = A_tensor @ B_tensor
    toc = time.monotonic()
    s = toc - tic
    flops_rate = flops / s
    return flops, s, flops_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process matmul.')
    parser.add_argument('-n', '--num', type=int) 
    args = parser.parse_args()
    # N = args.num

    console = Console()
    table = Table(show_header=True, header_style="bold blue", box=box.ASCII2)
    for col in ['num', 'GFlops', 'time(ms)', 'GFLOPS', 'GFlops_cuda', 'time_cuda(ms)', 'GFLOPS_cuda']:
        table.add_column(col, justify='right')
    
    # for N in track([2**n for n in range(7, 15)]):
    tasks = [2 ** n for n in range(7, 15)]
    with console.status("[bold green]Working on tasks...") as status:
        while tasks:
            N = tasks.pop(0)
            flops, s, flops_rate = profiling(N)
            flops_cuda, s_cuda, flops_rate_cuda = profiling(N, True)
            table.add_row(f"{N}", f"{flops:.2f}", f"{s*1000:.2f}", f"{flops_rate:.2f}", f"{flops_cuda:.2f}", f"{s_cuda*1000:.2f}",f"{flops_rate_cuda:.2f}")
            console.log(f"{N} matmul complete")

    print(table)

'''
+-------+---------+----------+---------+-------------+---------------+-------------+
|   num |  GFlops | time(ms) |  GFLOPS | GFlops_cuda | time_cuda(ms) | GFLOPS_cuda |
+-------+---------+----------+---------+-------------+---------------+-------------+
|   128 |    0.00 |     0.58 |    7.24 |        0.00 |          0.08 |       54.66 |
|   256 |    0.03 |     0.15 |  226.08 |        0.03 |          0.02 |     1431.14 |
|   512 |    0.27 |     0.40 |  671.73 |        0.27 |          0.03 |     8656.42 |
|  1024 |    2.15 |     1.89 | 1134.48 |        2.15 |          0.03 |    83358.54 |
|  2048 |   17.18 |     9.30 | 1847.60 |       17.18 |          0.28 |    61651.06 |
|  4096 |  137.44 |    83.13 | 1653.34 |      137.44 |          0.29 |   466111.20 |
|  8192 | 1099.51 |   431.38 | 2548.80 |     1099.51 |          0.50 |  2212025.53 |
| 16384 | 8796.09 |  3244.75 | 2710.87 |     8796.09 |          1.27 |  6953284.93 |
+-------+---------+----------+---------+-------------+---------------+-------------+

1. 大矩阵运算 cuda 比 cpu 快的多， 1000多倍的加速比
'''