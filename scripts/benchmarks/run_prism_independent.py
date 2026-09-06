from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MCLRP_MFMR.independent_t0 import make_locked_t0_test_mask
from MCLRP_MFMR.t0_mfmr_protocol import (
    MFMRConfig,
    MCLRPConfig,
    MutationHeadConfig,
    T0DatasetBundle,
    build_ccle_mutation_features,
    evaluate_prediction,
    predict_mfmr_t0_seed,
)


DEFAULT_METHODS = ("original_mclrp", "imputer_only", "ridge_only", "mfmr_base", "mfmr_mutation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a locked PRISM 19Q4 independent T0 test.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--split-seed", type=int, default=20260906)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--min-train-per-row", type=int, default=1)
    parser.add_argument("--min-train-per-drug", type=int, default=30)
    parser.add_argument("--min-test-per-drug", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _load_bundle(bundle_dir: Path) -> tuple[T0DatasetBundle, np.ndarray]:
    path = bundle_dir / "bundle.npz"
    payload = np.load(path, allow_pickle=True)
    bundle = T0DatasetBundle(
        name="PRISM19Q4_independent",
        X=payload["X"].astype(np.float32),
        M=payload["M"].astype(np.float32),
        cell_labels=payload["cell_ids"].astype(object),
        drug_labels=payload["drug_labels"].astype(object),
        response_path=str(path),
        expression_path=str(path),
    )
    return bundle, payload["observed_mask"].astype(bool)


def main() -> None:
    args = parse_args()
    bundle, observed = _load_bundle(args.bundle_dir)
    if not np.array_equal(observed, bundle.M != 0):
        raise ValueError("The current T0 implementation requires observed responses to be non-zero")

    test_mask = make_locked_t0_test_mask(
        bundle.M,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
        min_train_per_row=args.min_train_per_row,
        min_train_per_col=args.min_train_per_drug,
        min_test_per_col=args.min_test_per_drug,
    )
    fold = np.where(test_mask, bundle.M, 0.0).astype(np.float32)
    folds = [fold]

    output_dir = args.output_dir
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "locked_split.npz",
        test_mask=test_mask,
        train_mask=observed & ~test_mask,
        cell_ids=bundle.cell_labels,
        drug_labels=bundle.drug_labels,
    )

    base_config = MFMRConfig(num_folds=1, seeds=tuple(args.seeds), min_train_per_drug=args.min_train_per_drug)
    mutation_config = MutationHeadConfig(enabled=True, residual_inner_cv=5)
    mclrp_config = MCLRPConfig()
    mutation = None
    if "mfmr_mutation" in args.methods:
        mutation = build_ccle_mutation_features(
            expected_rows=bundle.M.shape[0],
            mutation_path=args.bundle_dir / "mutation_features.csv",
        )

    seed_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    drug_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        config = replace(base_config, random_state=int(seed))
        for method in args.methods:
            prediction_path = prediction_dir / f"{method}__seed_{seed}.npz"
            if args.resume and prediction_path.exists():
                pred = np.load(prediction_path)["prediction"].astype(np.float32)
            else:
                result = predict_mfmr_t0_seed(
                    bundle.name,
                    bundle.X,
                    bundle.M,
                    folds,
                    method,
                    config,
                    mutation_config=mutation_config,
                    mutation_features=mutation,
                    mclrp_config=mclrp_config,
                )
                pred = result.prediction.astype(np.float32)
                np.savez_compressed(prediction_path, prediction=pred)
            seed_row, per_fold, per_drug, per_cell = evaluate_prediction(bundle, method, seed, pred, folds)
            seed_rows.append(seed_row)
            fold_rows.extend(per_fold)
            drug_rows.extend(per_drug)
            cell_rows.extend(per_cell)
            print(json.dumps(seed_row, ensure_ascii=False), flush=True)

    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "split_metrics.csv", index=False)
    pd.DataFrame(drug_rows).to_csv(output_dir / "per_drug_metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(output_dir / "per_cell_metrics.csv", index=False)
    summary = (
        pd.DataFrame(seed_rows)
        .groupby("method", sort=False)[["overall_pcc", "rmse", "mae"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output_dir / "method_summary.csv", index=False)

    run_config = {
        "bundle_dir": str(args.bundle_dir.resolve()),
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "test_mask_sha256": _sha256_array(test_mask),
        "n_train": int((observed & ~test_mask).sum()),
        "n_test": int(test_mask.sum()),
        "mfmr_config": asdict(base_config),
        "mutation_config": asdict(mutation_config),
        "mclrp_config": asdict(mclrp_config),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
