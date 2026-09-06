from __future__ import annotations

import numpy as np

from MCLRP_MFMR.independent_t0 import make_locked_t0_test_mask, paired_cluster_bootstrap


def test_locked_mask_is_deterministic_and_uses_only_observed_entries() -> None:
    matrix = np.arange(1, 25, dtype=np.float32).reshape(8, 3)

    first = make_locked_t0_test_mask(
        matrix,
        test_fraction=0.25,
        seed=20260906,
        min_train_per_row=1,
        min_train_per_col=4,
        min_test_per_col=2,
    )
    second = make_locked_t0_test_mask(
        matrix * 10.0,
        test_fraction=0.25,
        seed=20260906,
        min_train_per_row=1,
        min_train_per_col=4,
        min_test_per_col=2,
    )

    assert np.array_equal(first, second)
    assert np.all(first <= (matrix != 0))
    assert np.all(first.sum(axis=0) == 2)


def test_locked_mask_preserves_t0_training_context() -> None:
    matrix = np.ones((10, 4), dtype=np.float32)
    matrix[0, 1:] = 0.0

    test_mask = make_locked_t0_test_mask(
        matrix,
        test_fraction=0.30,
        seed=7,
        min_train_per_row=1,
        min_train_per_col=5,
        min_test_per_col=2,
    )
    train_mask = (matrix != 0) & ~test_mask

    test_rows, test_cols = np.where(test_mask)
    assert np.all(train_mask.sum(axis=1)[test_rows] >= 1)
    assert np.all(train_mask.sum(axis=0)[test_cols] >= 5)
    assert not test_mask[0, 0]


def test_locked_mask_rejects_impossible_per_drug_requirement() -> None:
    matrix = np.ones((5, 2), dtype=np.float32)

    try:
        make_locked_t0_test_mask(
            matrix,
            test_fraction=0.20,
            seed=1,
            min_train_per_row=1,
            min_train_per_col=4,
            min_test_per_col=2,
        )
    except ValueError as exc:
        assert "independent test entries" in str(exc)
    else:
        raise AssertionError("Expected an impossible locked split to raise ValueError")


def test_paired_cluster_bootstrap_is_deterministic_and_uses_reference_sign() -> None:
    truth = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    reference = truth.copy()
    comparator = np.asarray([[1.5, 1.0], [1.5, 3.0], [1.5, 5.0]], dtype=np.float64)
    mask = np.ones_like(truth, dtype=bool)

    first = paired_cluster_bootstrap(truth, reference, comparator, mask, iterations=200, seed=9)
    second = paired_cluster_bootstrap(truth, reference, comparator, mask, iterations=200, seed=9)

    assert first == second
    assert first["delta_rmse"] > 0
    assert first["delta_mae"] > 0
    assert first["delta_pcc"] > 0
