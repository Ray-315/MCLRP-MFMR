from __future__ import annotations

import numpy as np


def _pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def paired_cluster_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    comparator: np.ndarray,
    test_mask: np.ndarray,
    *,
    iterations: int = 20_000,
    seed: int = 20260906,
) -> dict[str, float]:
    """Bootstrap paired metric differences by resampling drug columns."""
    truth = np.asarray(truth, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    mask = np.asarray(test_mask, dtype=bool)
    if not (truth.shape == reference.shape == comparator.shape == mask.shape):
        raise ValueError("truth, predictions, and test_mask must have the same shape")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")

    def differences(columns: np.ndarray) -> tuple[float, float, float]:
        ys: list[np.ndarray] = []
        refs: list[np.ndarray] = []
        comps: list[np.ndarray] = []
        for column in columns:
            selected = mask[:, column]
            ys.append(truth[selected, column])
            refs.append(reference[selected, column])
            comps.append(comparator[selected, column])
        y = np.concatenate(ys)
        ref = np.concatenate(refs)
        comp = np.concatenate(comps)
        delta_pcc = _pcc(y, ref) - _pcc(y, comp)
        delta_rmse = float(np.sqrt(np.mean((y - comp) ** 2)) - np.sqrt(np.mean((y - ref) ** 2)))
        delta_mae = float(np.mean(np.abs(y - comp)) - np.mean(np.abs(y - ref)))
        return delta_pcc, delta_rmse, delta_mae

    columns = np.arange(truth.shape[1], dtype=np.int64)
    point = differences(columns)
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(iterations), 3), dtype=np.float64)
    for index in range(int(iterations)):
        samples[index] = differences(rng.choice(columns, size=len(columns), replace=True))

    output: dict[str, float] = {}
    for metric_index, metric in enumerate(("pcc", "rmse", "mae")):
        output[f"delta_{metric}"] = float(point[metric_index])
        output[f"delta_{metric}_ci_low"] = float(np.quantile(samples[:, metric_index], 0.025))
        output[f"delta_{metric}_ci_high"] = float(np.quantile(samples[:, metric_index], 0.975))
    return output


def make_locked_t0_test_mask(
    matrix: np.ndarray,
    *,
    test_fraction: float,
    seed: int,
    min_train_per_row: int,
    min_train_per_col: int,
    min_test_per_col: int,
) -> np.ndarray:
    """Create one fixed entry-wise test mask while preserving T0 context."""
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between zero and one")

    observed = np.isfinite(values) & (values != 0)
    row_train_counts = observed.sum(axis=1).astype(np.int64)
    col_train_counts = observed.sum(axis=0).astype(np.int64)
    test_mask = np.zeros(observed.shape, dtype=bool)
    rng = np.random.default_rng(int(seed))

    for col_idx in range(observed.shape[1]):
        observed_rows = np.flatnonzero(observed[:, col_idx])
        maximum = int(len(observed_rows) - int(min_train_per_col))
        target = max(int(min_test_per_col), int(np.floor(len(observed_rows) * float(test_fraction))))
        if maximum < int(min_test_per_col):
            raise ValueError(
                f"Column {col_idx} cannot supply {min_test_per_col} independent test entries "
                f"while retaining {min_train_per_col} training entries"
            )
        target = min(target, maximum)

        selected = 0
        for row_idx in rng.permutation(observed_rows):
            if row_train_counts[row_idx] - 1 < int(min_train_per_row):
                continue
            if col_train_counts[col_idx] - 1 < int(min_train_per_col):
                continue
            test_mask[row_idx, col_idx] = True
            row_train_counts[row_idx] -= 1
            col_train_counts[col_idx] -= 1
            selected += 1
            if selected == target:
                break

        if selected < int(min_test_per_col):
            raise ValueError(
                f"Column {col_idx} cannot supply {min_test_per_col} independent test entries "
                "under the row-context constraint"
            )

    return test_mask
