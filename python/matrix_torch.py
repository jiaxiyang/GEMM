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
|  num  |  GFlops | time(ms) |  GFLOPS | GFlops_cuda | time_cuda(ms) | GFLOPS_cuda |
+-------+---------+----------+---------+-------------+---------------+-------------+
|  128  |   0.00  |   0.13   |  32.15  |     0.00    |      0.07     |    59.00    |
|  256  |   0.03  |   0.12   |  281.88 |     0.03    |      0.02     |   1405.54   |
|  512  |   0.27  |   0.43   |  627.04 |     0.27    |      0.03     |   9208.76   |
|  1024 |   2.15  |   1.66   | 1294.45 |     2.15    |      0.02     |   87313.81  |
|  2048 |  17.18  |  11.02   | 1559.09 |    17.18    |      0.24     |   70786.15  |
|  4096 |  137.44 |  69.84   | 1967.88 |    137.44   |      0.27     |  514876.06  |
|  8192 | 1099.51 |  439.56  | 2501.41 |   1099.51   |      0.47     |  2322874.25 |
| 16384 | 8796.09 | 2916.23  | 3016.25 |   8796.09   |      1.22     |  7219641.14 |
+-------+---------+----------+---------+-------------+---------------+-------------+

1. 大矩阵运算 cuda 比 cpu 快的多， 1000多倍的加速比
'''