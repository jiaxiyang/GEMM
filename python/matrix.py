#!/usr/bin/env python3
import numpy as np
import time
import argparse

N = 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process matmul.')
    # parser.add_argument("n", type=int)
    parser.add_argument('-n', '--num', type=int) 
    args = parser.parse_args()
    N = args.num
    print(f"N: {N}")

    # memory: N * N 
    A = np.random.randn(N, N).astype(np.float32)

    # memory: N * N 
    B = np.random.randn(N, N).astype(np.float32)
    
    # memory: N * N; comput: N * N * 2 N
    flops = N * N * 2 * N
    print(f"{flops / 1e9:.2f} GFlops")

    tic = time.monotonic()
    C = A @ B
    toc = time.monotonic()
    print(f"{flops / (toc - tic) / 1e9} GFLOPS")
    
# (prediction) [xiyang.jia@tcloud-3090-005 python]$ python matrix.py -n 1024
# N: 1024
# 2.15 GFlops
# 836.0912458796764 GFLOPS
# (prediction) [xiyang.jia@tcloud-3090-005 python]$ python matrix.py -n 512
# N: 512
# 0.27 GFlops
# 368.5406903517908 GFLOPS
# (prediction) [xiyang.jia@tcloud-3090-005 python]$ python matrix.py -n 1024
# N: 1024
# 2.15 GFlops
# 705.4175640324194 GFLOPS
# (prediction) [xiyang.jia@tcloud-3090-005 python]$ python matrix.py -n 2048
# N: 2048
# 17.18 GFlops
# 116.15083572419698 GFLOPS
# (prediction) [xiyang.jia@tcloud-3090-005 python]$ python matrix.py -n 4096
# N: 4096
# 137.44 GFlops
# 712.1434856945628 GFLOPS

# 2048 比 1024 和 4096 效率都低