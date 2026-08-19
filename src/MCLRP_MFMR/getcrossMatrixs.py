from __future__ import annotations

import numpy as np


def getcrossMatrixs(MM: np.ndarray, num_folds: int = 10, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    num_folds = int(num_folds)
    if num_folds < 1:
        raise ValueError("num_folds must be >= 1")

    nz_rows, nz_cols = np.nonzero(MM)
    n_obs = len(nz_rows)
    order = rng.permutation(n_obs)
    fold_size = n_obs // num_folds
    folds = [np.zeros_like(MM) for _ in range(num_folds)]

    for fold_idx in range(num_folds):
        start = fold_idx * fold_size
        end = (fold_idx + 1) * fold_size
        for pos in range(start, end):
            row = nz_rows[order[pos]]
            col = nz_cols[order[pos]]
            folds[fold_idx][row, col] = MM[row, col]

    remainder = n_obs - num_folds * fold_size
    pos = num_folds * fold_size
    for fold_idx in range(remainder):
        row = nz_rows[order[pos]]
        col = nz_cols[order[pos]]
        folds[fold_idx][row, col] = MM[row, col]
        pos += 1
    return folds
