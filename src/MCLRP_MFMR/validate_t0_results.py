from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCLRP_MFMR.paths import RESULTS_DIR
from MCLRP_MFMR.t0_mfmr_protocol import load_t0_dataset, subset_dataset


RESULT_TABLES = (
    "method_summary.csv",
    "seed_summary.csv",
    "fold_summary.csv",
    "per_drug_pcc.csv",
    "per_cell_line_pcc.csv",
    "ablation_summary.csv",
    "statistical_tests.csv",
    "diagnostics.csv",
    "protocol_audit.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate strict T0 MCLRP-MFMR benchmark outputs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "t0_mfmr")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_config(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.json in {results_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_shape(dataset: str, config: dict[str, Any]) -> tuple[int, int]:
    bundle = load_t0_dataset(dataset)
    subset, _, _ = subset_dataset(
        bundle,
        max_cell_lines=config.get("max_cell_lines"),
        max_drugs=config.get("max_drugs"),
    )
    return tuple(int(x) for x in subset.M.shape)


def _check_no_unexpected_nan_or_inf(
    tables: dict[str, pd.DataFrame],
    errors: list[str],
    warnings: list[str],
) -> None:
    for name, df in tables.items():
        if df.empty:
            continue
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            continue
        for col in numeric.columns:
            values = numeric[col].to_numpy(dtype=float)
            nan_mask = np.isnan(values)
            inf_mask = np.isinf(values)
            if name == "statistical_tests.csv" and col in {"paired_t_pvalue", "wilcoxon_pvalue"}:
                allowed = (df.get("n_pairs", pd.Series(0, index=df.index)).to_numpy() < 2) & nan_mask
                unexpected_nan = nan_mask & (~allowed)
                if np.any(allowed):
                    warnings.append(
                        f"{name}:{col} has NaN for {int(np.sum(allowed))} row(s) with n_pairs < 2; p-values are not computable."
                    )
            else:
                unexpected_nan = nan_mask
            if np.any(unexpected_nan):
                errors.append(f"{name}:{col} contains {int(np.sum(unexpected_nan))} unexpected NaN value(s).")
            if np.any(inf_mask):
                errors.append(f"{name}:{col} contains {int(np.sum(inf_mask))} infinite value(s).")


def _check_requested_results(
    config: dict[str, Any],
    seed_df: pd.DataFrame,
    errors: list[str],
) -> None:
    required_cols = {"dataset", "method", "seed"}
    if seed_df.empty or not required_cols.issubset(seed_df.columns):
        errors.append("seed_summary.csv is missing or lacks dataset/method/seed columns.")
        return
    present = {
        (str(row.dataset), str(row.method), int(row.seed))
        for row in seed_df[["dataset", "method", "seed"]].itertuples(index=False)
    }
    missing: list[str] = []
    for dataset in config.get("datasets", []):
        for method in config.get("methods", []):
            for seed in config.get("mfmr_config", {}).get("seeds", []):
                key = (str(dataset), str(method), int(seed))
                if key not in present:
                    missing.append(f"{dataset}/seed{seed}/{method}")
    if missing:
        errors.append("Missing requested method result(s): " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))


def _check_pcc_ranges(tables: dict[str, pd.DataFrame], errors: list[str]) -> None:
    for name, df in tables.items():
        if df.empty:
            continue
        pcc_cols = [col for col in df.columns if "pcc" in col.lower()]
        for col in pcc_cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            bad = finite & ((values < -1.000001) | (values > 1.000001))
            if np.any(bad):
                errors.append(f"{name}:{col} contains {int(np.sum(bad))} impossible PCC value(s) outside [-1, 1].")


def _check_protocol_audit(audit_df: pd.DataFrame, config: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    required = {
        "dataset",
        "seed",
        "fold",
        "method",
        "drug_index",
        "n_train_entries_for_drug",
        "n_test_entries_for_drug",
        "n_total_observed_entries_for_drug",
        "expression_pc_mode",
        "expression_pc_train_rows",
        "imputer_test_entries_masked",
        "ridge_train_rows_only",
        "mutation_residual_mode",
        "residual_inner_cv",
        "skipped_reason",
    }
    if audit_df.empty:
        errors.append("protocol_audit.csv is missing or empty.")
        return
    missing_cols = sorted(required - set(audit_df.columns))
    if missing_cols:
        errors.append("protocol_audit.csv missing required column(s): " + ", ".join(missing_cols))
        return

    if "fold_mask_sha256" not in audit_df.columns:
        warnings.append("protocol_audit.csv has no fold_mask_sha256 column; exact fold identity cannot be validated.")
    else:
        for key, group in audit_df.groupby(["dataset", "seed", "fold"], dropna=False):
            by_method = group.groupby("method")["fold_mask_sha256"].first()
            if by_method.nunique(dropna=False) != 1:
                errors.append(f"Methods used non-identical fold masks for dataset/seed/fold={key}.")

    for key, group in audit_df.groupby(["dataset", "seed", "fold", "drug_index"], dropna=False):
        if group["n_test_entries_for_drug"].nunique(dropna=False) != 1:
            errors.append(f"Held-out test counts differ across methods for dataset/seed/fold/drug={key}.")
        if group["n_total_observed_entries_for_drug"].nunique(dropna=False) != 1:
            errors.append(f"Observed totals differ across methods for dataset/seed/fold/drug={key}.")

    expected_methods = set(str(m) for m in config.get("methods", []))
    actual_methods = set(str(m) for m in audit_df["method"].dropna().unique())
    missing_methods = sorted(expected_methods - actual_methods)
    if missing_methods:
        errors.append("protocol_audit.csv missing method(s): " + ", ".join(missing_methods))


def _check_prediction_npz(
    results_dir: Path,
    config: dict[str, Any],
    audit_df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:
    pred_dir = results_dir / "predictions"
    seeds = [int(s) for s in config.get("mfmr_config", {}).get("seeds", [])]
    if 0 not in seeds:
        warnings.append("Seed 0 was not requested; prediction NPZ files are not expected from the current runner.")
        return
    if not pred_dir.exists():
        warnings.append("predictions/ directory is missing; shape checks for saved arrays were skipped.")
        return

    shape_cache = {str(dataset): _expected_shape(str(dataset), config) for dataset in config.get("datasets", [])}
    for dataset in config.get("datasets", []):
        dataset = str(dataset)
        expected_shape = shape_cache[dataset]
        expected_folds = int(config.get("mfmr_config", {}).get("num_folds", 10))
        audit_hashes = None
        if not audit_df.empty and "fold_mask_sha256" in audit_df.columns:
            audit_hashes = (
                audit_df[(audit_df["dataset"] == dataset) & (audit_df["seed"] == 0)]
                .groupby("fold")["fold_mask_sha256"]
                .first()
                .sort_index()
                .tolist()
            )
        for method in config.get("methods", []):
            method = str(method)
            path = pred_dir / f"{dataset}_{method}_seed0.npz"
            if not path.exists():
                warnings.append(f"Missing saved prediction file for {dataset}/seed0/{method}: {path.name}")
                continue
            with np.load(path, allow_pickle=False) as payload:
                for key in ("prediction", "imputer", "ridge", "base"):
                    if key not in payload:
                        errors.append(f"{path.name} missing array {key!r}.")
                        continue
                    arr = payload[key]
                    if tuple(arr.shape) != expected_shape:
                        errors.append(f"{path.name}:{key} has shape {arr.shape}, expected {expected_shape}.")
                if "folds" not in payload:
                    errors.append(f"{path.name} missing array 'folds'.")
                    continue
                folds = payload["folds"]
                if tuple(folds.shape) != (expected_folds, *expected_shape):
                    errors.append(f"{path.name}:folds has shape {folds.shape}, expected {(expected_folds, *expected_shape)}.")
                    continue
                heldout_mask = np.any(folds != 0, axis=0)
                prediction = payload["prediction"]
                if not np.all(np.isfinite(prediction[heldout_mask])):
                    errors.append(f"{path.name}:prediction has non-finite value(s) on held-out entries.")
                if audit_hashes and len(audit_hashes) == expected_folds:
                    from MCLRP_MFMR.t0_mfmr_protocol import _mask_sha256

                    npz_hashes = [_mask_sha256(fold != 0) for fold in folds]
                    if npz_hashes != audit_hashes:
                        errors.append(f"{path.name}:fold masks do not match protocol_audit.csv hashes.")

            if method == "mfmr_base":
                with np.load(path, allow_pickle=False) as payload:
                    for key in ("imputer", "ridge"):
                        if key not in payload or tuple(payload[key].shape) != expected_shape:
                            errors.append(f"mfmr_base saved predictions missing valid {key!r} array for {dataset}/seed0.")


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        config = _load_config(results_dir)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    tables = {name: _read_csv(results_dir / name) for name in RESULT_TABLES}
    for required in ("method_summary.csv", "seed_summary.csv", "fold_summary.csv", "protocol_audit.csv"):
        if tables[required].empty:
            errors.append(f"{required} is missing or empty.")

    _check_requested_results(config, tables["seed_summary.csv"], errors)
    _check_protocol_audit(tables["protocol_audit.csv"], config, errors, warnings)
    _check_pcc_ranges(tables, errors)
    _check_no_unexpected_nan_or_inf(tables, errors, warnings)
    _check_prediction_npz(results_dir, config, tables["protocol_audit.csv"], errors, warnings)

    print("Validation summary")
    print(f"Results directory: {results_dir}")
    print(f"Errors: {len(errors)}")
    for item in errors:
        print(f"ERROR: {item}")
    print(f"Warnings: {len(warnings)}")
    for item in warnings:
        print(f"WARNING: {item}")
    if not errors:
        print("Status: PASS")
        return 0
    print("Status: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
