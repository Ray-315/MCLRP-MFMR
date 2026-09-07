from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


DEFAULT_ALPHA_CANDIDATES = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))


def load_existing_records(path: Path | str, *, enabled: bool) -> list[dict[str, object]]:
    if not enabled:
        return []
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(source).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def resolve_cached_truth(payload: Mapping[str, np.ndarray], fallback: np.ndarray) -> np.ndarray:
    """Use the self-contained cached truth matrix when present."""
    if "truth" in payload:
        return np.asarray(payload["truth"], dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32)


def endpoint_ratios(values: Iterable[float]) -> tuple[float, float]:
    ratios = sorted({float(value) for value in values})
    if not ratios:
        raise ValueError("at least one mask ratio is required")
    return ratios[0], ratios[-1]


def mask_outer_responses(matrix: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    mask = np.asarray(test_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("matrix and test_mask must have matching shapes")
    return np.where(mask, 0.0, values).astype(np.float32)


def _finite_vectors(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    values = tuple(np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays)
    if not values:
        return values
    if len({value.shape for value in values}) != 1:
        raise ValueError("all arrays must have the same shape")
    keep = np.logical_and.reduce([np.isfinite(value) for value in values])
    return tuple(value[keep] for value in values)


def safe_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _finite_vectors(y_true, y_pred)
    if truth.size < 2 or np.std(truth) == 0.0 or np.std(prediction) == 0.0:
        return 0.0
    value = float(np.corrcoef(truth, prediction)[0, 1])
    return value if np.isfinite(value) else 0.0


def safe_scc(first: np.ndarray, second: np.ndarray) -> float:
    left, right = _finite_vectors(first, second)
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _finite_vectors(y_true, y_pred)
    return float(np.sqrt(np.mean((truth - prediction) ** 2))) if truth.size else 0.0


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _finite_vectors(y_true, y_pred)
    return float(np.mean(np.abs(truth - prediction))) if truth.size else 0.0


def constrained_t0_mask(
    matrix: np.ndarray,
    *,
    fraction: float,
    seed: int,
    min_train_per_row: int = 1,
    min_train_per_col: int = 5,
) -> np.ndarray:
    """Select a reproducible fraction of observed entries while preserving T0 context."""
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be between zero and one")
    if min_train_per_row < 1 or min_train_per_col < 1:
        raise ValueError("minimum training counts must be positive")

    observed = np.isfinite(values) & (values != 0)
    rows, cols = np.where(observed)
    target = int(round(float(fraction) * len(rows)))
    if target < 1:
        raise ValueError("fraction selects no observed entries")

    row_remaining = observed.sum(axis=1).astype(np.int64)
    col_remaining = observed.sum(axis=0).astype(np.int64)
    mask = np.zeros_like(observed, dtype=bool)
    rng = np.random.default_rng(int(seed))
    for position in rng.permutation(len(rows)):
        row = int(rows[position])
        col = int(cols[position])
        if row_remaining[row] <= int(min_train_per_row):
            continue
        if col_remaining[col] <= int(min_train_per_col):
            continue
        mask[row, col] = True
        row_remaining[row] -= 1
        col_remaining[col] -= 1
        if int(mask.sum()) == target:
            break

    if int(mask.sum()) != target:
        raise ValueError(
            f"unable to select {target} test entries under T0 constraints; selected {int(mask.sum())}"
        )
    return mask


def select_adaptive_alpha(
    truth: np.ndarray,
    imputer: np.ndarray,
    ridge: np.ndarray,
    *,
    fold_ids: np.ndarray,
    candidates: Sequence[float] = DEFAULT_ALPHA_CANDIDATES,
) -> tuple[float, dict[float, dict[str, float]]]:
    """Choose alpha by mean inner-fold PCC, resolving ties toward 0.5."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    imputer = np.asarray(imputer, dtype=np.float64).reshape(-1)
    ridge = np.asarray(ridge, dtype=np.float64).reshape(-1)
    fold_ids = np.asarray(fold_ids).reshape(-1)
    if not (truth.shape == imputer.shape == ridge.shape == fold_ids.shape):
        raise ValueError("truth, branch predictions, and fold_ids must have matching shapes")
    candidate_values = tuple(float(alpha) for alpha in candidates)
    if not candidate_values or any(alpha < 0.0 or alpha > 1.0 for alpha in candidate_values):
        raise ValueError("alpha candidates must lie in [0, 1]")

    scores: dict[float, dict[str, float]] = {}
    unique_folds = np.unique(fold_ids)
    for alpha in candidate_values:
        prediction = alpha * imputer + (1.0 - alpha) * ridge
        fold_pcc = [safe_pcc(truth[fold_ids == fold], prediction[fold_ids == fold]) for fold in unique_folds]
        fold_rmse = [rmse(truth[fold_ids == fold], prediction[fold_ids == fold]) for fold in unique_folds]
        fold_mae = [mae(truth[fold_ids == fold], prediction[fold_ids == fold]) for fold in unique_folds]
        scores[alpha] = {
            "mean_pcc": float(np.mean(fold_pcc)),
            "mean_rmse": float(np.mean(fold_rmse)),
            "mean_mae": float(np.mean(fold_mae)),
        }

    best_pcc = max(item["mean_pcc"] for item in scores.values())
    tied = [alpha for alpha in candidate_values if np.isclose(scores[alpha]["mean_pcc"], best_pcc, rtol=0.0, atol=1e-12)]
    selected = min(tied, key=lambda alpha: (abs(alpha - 0.5), alpha))
    return float(selected), scores


def _quartile_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 4:
        return np.ones(values.size, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    quartiles = np.empty(values.size, dtype=np.int64)
    quartiles[order] = np.minimum(4, (np.arange(values.size) * 4 // values.size) + 1)
    return quartiles


def branch_complementarity_metrics(
    truth: np.ndarray,
    imputer: np.ndarray,
    ridge: np.ndarray,
    fusion: np.ndarray,
    *,
    tolerance: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    truth, imputer, ridge, fusion = _finite_vectors(truth, imputer, ridge, fusion)
    if truth.size == 0:
        raise ValueError("at least one finite held-out entry is required")

    error_imputer = truth - imputer
    error_ridge = truth - ridge
    abs_imputer = np.abs(error_imputer)
    abs_ridge = np.abs(error_ridge)
    difference = abs_imputer - abs_ridge
    tied = np.abs(difference) < float(tolerance)
    imputer_better = difference < -float(tolerance)
    ridge_better = difference > float(tolerance)

    overall = {
        "prediction_pearson": safe_pcc(imputer, ridge),
        "prediction_spearman": safe_scc(imputer, ridge),
        "error_pearson": safe_pcc(error_imputer, error_ridge),
        "error_spearman": safe_scc(error_imputer, error_ridge),
        "imputer_better_share": float(np.mean(imputer_better)),
        "ridge_better_share": float(np.mean(ridge_better)),
        "tie_share": float(np.mean(tied)),
        "fusion_delta_pcc_vs_imputer": safe_pcc(truth, fusion) - safe_pcc(truth, imputer),
        "fusion_delta_pcc_vs_ridge": safe_pcc(truth, fusion) - safe_pcc(truth, ridge),
        "fusion_rmse_gain_vs_imputer": rmse(truth, imputer) - rmse(truth, fusion),
        "fusion_rmse_gain_vs_ridge": rmse(truth, ridge) - rmse(truth, fusion),
        "fusion_mae_gain_vs_imputer": mae(truth, imputer) - mae(truth, fusion),
        "fusion_mae_gain_vs_ridge": mae(truth, ridge) - mae(truth, fusion),
        "n_test": int(truth.size),
    }

    disagreement = np.abs(imputer - ridge)
    quartile_ids = _quartile_ids(disagreement)
    rows: list[dict[str, float | int | str]] = []
    for quartile in np.unique(quartile_ids):
        selected = quartile_ids == quartile
        for method, prediction in (
            ("imputer_only", imputer),
            ("ridge_only", ridge),
            ("fixed_fusion", fusion),
        ):
            rows.append(
                {
                    "quartile": int(quartile),
                    "method": method,
                    "pcc": safe_pcc(truth[selected], prediction[selected]),
                    "rmse": rmse(truth[selected], prediction[selected]),
                    "mae": mae(truth[selected], prediction[selected]),
                    "mean_disagreement": float(np.mean(disagreement[selected])),
                    "n_test": int(selected.sum()),
                }
            )
    return overall, pd.DataFrame(rows)


def mean_sd_ci(values: Iterable[float], confidence: float = 0.95) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    half_width = float(t.ppf((1.0 + confidence) / 2.0, array.size - 1) * sd / np.sqrt(array.size)) if array.size > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci_low": mean - half_width, "ci_high": mean + half_width, "n": int(array.size)}


def summarize_numeric_by_group(frame: pd.DataFrame, groups: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, group in frame.groupby(list(groups), dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base: dict[str, float | int | str] = dict(zip(groups, key_values))
        for metric in metrics:
            stats = mean_sd_ci(group[metric].to_numpy(dtype=float))
            rows.append({**base, "metric": metric, **stats})
    return pd.DataFrame(rows)
