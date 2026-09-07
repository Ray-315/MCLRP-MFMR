from __future__ import annotations

import numpy as np
import pandas as pd

from MCLRP_MFMR.additional_experiments import (
    branch_complementarity_metrics,
    constrained_t0_mask,
    endpoint_ratios,
    mask_outer_responses,
    load_existing_records,
    resolve_cached_truth,
    select_adaptive_alpha,
)


def test_select_adaptive_alpha_uses_mean_pcc_and_ties_toward_half() -> None:
    truth = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
    imputer = truth.copy()
    ridge = truth.copy()
    selected, scores = select_adaptive_alpha(
        truth,
        imputer,
        ridge,
        fold_ids=np.array([0, 0, 1, 1]),
        candidates=(0.0, 0.5, 1.0),
    )

    assert selected == 0.5
    assert set(scores) == {0.0, 0.5, 1.0}


def test_constrained_t0_mask_is_reproducible_and_preserves_context() -> None:
    matrix = np.arange(1, 49, dtype=float).reshape(8, 6)
    first = constrained_t0_mask(
        matrix,
        fraction=0.30,
        seed=7,
        min_train_per_row=1,
        min_train_per_col=3,
    )
    second = constrained_t0_mask(
        matrix,
        fraction=0.30,
        seed=7,
        min_train_per_row=1,
        min_train_per_col=3,
    )

    assert np.array_equal(first, second)
    train = (matrix != 0) & ~first
    assert np.all(train.sum(axis=1) >= 1)
    assert np.all(train.sum(axis=0) >= 3)
    assert abs(first.sum() / np.count_nonzero(matrix) - 0.30) <= 1 / matrix.size


def test_branch_complementarity_metrics_reports_entry_shares_and_quartiles() -> None:
    truth = np.arange(1, 13, dtype=float)
    imputer = truth + np.array([0.0, 0.2, -0.1, 0.7, -0.8, 0.1, 0.3, -0.4, 1.0, -1.1, 0.5, -0.2])
    ridge = truth + np.array([0.3, 0.0, -0.4, 0.1, -0.2, 0.6, -0.7, 0.2, 0.4, -0.3, 0.8, -0.9])
    fusion = 0.5 * imputer + 0.5 * ridge

    overall, quartiles = branch_complementarity_metrics(
        truth,
        imputer,
        ridge,
        fusion,
        tolerance=1e-12,
    )

    shares = overall["imputer_better_share"] + overall["ridge_better_share"] + overall["tie_share"]
    assert np.isclose(shares, 1.0)
    assert set(quartiles["quartile"].tolist()) == {1, 2, 3, 4}
    assert set(quartiles["method"].tolist()) == {"imputer_only", "ridge_only", "fixed_fusion"}


def test_branch_metrics_reject_negative_tolerance() -> None:
    values = np.arange(4, dtype=float)
    try:
        branch_complementarity_metrics(values, values, values, values, tolerance=-1.0)
    except ValueError as exc:
        assert "tolerance" in str(exc)
    else:
        raise AssertionError("negative tolerance must be rejected")


def test_resolve_cached_truth_prefers_self_contained_cache() -> None:
    cached = np.arange(6, dtype=float).reshape(2, 3)
    fallback = np.zeros((8, 9), dtype=float)

    resolved = resolve_cached_truth({"truth": cached}, fallback)

    assert np.array_equal(resolved, cached)


def test_endpoint_ratios_returns_sorted_extremes() -> None:
    assert endpoint_ratios([0.30, 0.10, 0.20]) == (0.10, 0.30)


def test_outer_test_perturbation_cannot_change_inner_training_matrix() -> None:
    matrix = np.arange(1, 13, dtype=float).reshape(3, 4)
    test_mask = np.zeros_like(matrix, dtype=bool)
    test_mask[0, 1] = True
    test_mask[2, 3] = True
    changed = matrix.copy()
    changed[test_mask] = [1e6, -1e6]

    assert np.array_equal(
        mask_outer_responses(matrix, test_mask),
        mask_outer_responses(changed, test_mask),
    )


def test_resume_loads_existing_nonempty_records(tmp_path) -> None:
    path = tmp_path / "scores.csv"
    pd.DataFrame([{"alpha": 0.5, "mean_pcc": 0.7}]).to_csv(path, index=False)

    assert load_existing_records(path, enabled=True) == [{"alpha": 0.5, "mean_pcc": 0.7}]
    assert load_existing_records(path, enabled=False) == []
