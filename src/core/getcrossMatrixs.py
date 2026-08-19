import numpy as np
from typing import List


def getcrossMatrixs(MM: np.ndarray, num_folds: int = 10, rng: np.random.Generator = None) -> List[np.ndarray]:
    """
    对应 MATLAB getcrossMatrixs.m
    等份划分非零位置，生成 num_folds 个留出矩阵。
    """
    if rng is None:
        rng = np.random.default_rng()
    nz_rows, nz_cols = np.nonzero(MM)
    N = len(nz_rows)
    D = rng.permutation(N)
    first = N // num_folds
    zeroM = np.zeros_like(MM)
    crossdata = [zeroM.copy() for _ in range(num_folds)]
    for i in range(num_folds):
        start = i * first
        end = (i + 1) * first
        for j in range(start, end):
            r = nz_rows[D[j]]
            c = nz_cols[D[j]]
            crossdata[i][r, c] = MM[r, c]
    k = N - num_folds * first
    idx = num_folds * first
    for j in range(k):
        r = nz_rows[D[idx]]
        c = nz_cols[D[idx]]
        crossdata[j][r, c] = MM[r, c]
        idx += 1
    return crossdata

