#!/usr/bin/env python3
import numpy as np
import time
import argparse
from prettytable import PrettyTable
from rich.progress import track
from rich.console import Console
from rich.table import Table
from rich import print
from rich import box


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

    tic = time.perf_counter()
    C = A @ B
    toc = time.perf_counter()
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

    console = Console()
    table = Table(show_header=True, header_style="bold blue", box=box.ASCII2)
    for col in ['num', 'GFlops', 'time(ms)', 'GFLOPS']:
        table.add_column(col, justify='right')
    
    tasks = [2 ** n for n in range(7, 15)]
    with console.status("[bold green]Working on diff matmul...") as status: 
        while tasks:
            N = tasks.pop(0)
            flops, s, flops_rate = profiling(N)
            table.add_row(f"{N}", f"{flops:.2f}", f"{s*1000:.2f}", f"{flops_rate:.2f}")
            console.log(f"{N} matmul complete")

    print(table)

'''
+-------+---------+----------+---------+
|   num |  GFlops | time(ms) |  GFLOPS |
+-------+---------+----------+---------+
|   128 |    0.00 |     1.66 |    2.53 |
|   256 |    0.03 |     2.72 |   12.32 |
|   512 |    0.27 |    17.04 |   15.75 |
|  1024 |    2.15 |    70.78 |   30.34 |
|  2048 |   17.18 |   152.17 |  112.90 |
|  4096 |  137.44 |   225.83 |  608.59 |
|  8192 | 1099.51 |   547.01 | 2010.04 |
| 16384 | 8796.09 |  3373.71 | 2607.24 |
+-------+---------+----------+---------+
从表中可以看出 2048 执行效率比较差
'''