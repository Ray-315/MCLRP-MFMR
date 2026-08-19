from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCLRP_MFMR.paths import RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready CSV tables from strict T0 benchmark outputs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "t0_mfmr")
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(mean: object, std: object) -> str:
    mean_val = pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0]
    std_val = pd.to_numeric(pd.Series([std]), errors="coerce").iloc[0]
    if not np.isfinite(mean_val):
        return ""
    if not np.isfinite(std_val):
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f} ± {std_val:.4f}"


def _metric_table(method_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if method_summary.empty:
        return pd.DataFrame(columns=["dataset", "method", metric, "n_seeds", "n_test_total"])
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    rows: list[dict[str, object]] = []
    for row in method_summary.sort_values(["dataset", "method"]).itertuples(index=False):
        payload = row._asdict()
        rows.append(
            {
                "dataset": payload.get("dataset"),
                "method": payload.get("method"),
                metric: _fmt(payload.get(mean_col), payload.get(std_col)),
                "n_seeds": payload.get("n_seeds"),
                "n_test_total": payload.get("n_test_total"),
            }
        )
    return pd.DataFrame(rows)


def _ablation_table(ablation: pd.DataFrame) -> pd.DataFrame:
    if ablation.empty:
        return pd.DataFrame(columns=["dataset", "method", "reference_method", "n_pairs", "delta_pcc", "delta_rmse", "delta_mae"])
    out = ablation.sort_values(["dataset", "method"]).copy()
    rename = {
        "delta_pcc_mean": "delta_pcc",
        "delta_rmse_mean": "delta_rmse",
        "delta_mae_mean": "delta_mae",
    }
    out = out.rename(columns=rename)
    keep = ["dataset", "method", "reference_method", "n_pairs", "delta_pcc", "delta_rmse", "delta_mae"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    for col in ("delta_pcc", "delta_rmse", "delta_mae"):
        out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: f"{x:.4f}" if np.isfinite(x) else "")
    return out[keep]


def _mutation_table(method_summary: pd.DataFrame, mutation_comparison: pd.DataFrame | None = None) -> pd.DataFrame:
    if mutation_comparison is not None and not mutation_comparison.empty:
        df = mutation_comparison.copy()
        for col in (
            "overall_pcc_mean",
            "overall_pcc_std",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "delta_pcc_vs_base",
            "delta_rmse_vs_base",
            "delta_mae_vs_base",
            "paired_t_pvalue_pcc",
            "wilcoxon_pvalue_pcc",
        ):
            if col not in df.columns:
                df[col] = np.nan
        rows: list[dict[str, object]] = []
        for row in df.sort_values(["dataset", "residual_mode", "method"]).itertuples(index=False):
            payload = row._asdict()
            rows.append(
                {
                    "dataset": payload.get("dataset"),
                    "residual_mode": payload.get("residual_mode"),
                    "method": payload.get("method"),
                    "overall_pcc": _fmt(payload.get("overall_pcc_mean"), payload.get("overall_pcc_std")),
                    "rmse": _fmt(payload.get("rmse_mean"), payload.get("rmse_std")),
                    "mae": _fmt(payload.get("mae_mean"), payload.get("mae_std")),
                    "delta_pcc_vs_base": f"{float(payload.get('delta_pcc_vs_base')):.4f}"
                    if np.isfinite(pd.to_numeric(pd.Series([payload.get("delta_pcc_vs_base")]), errors="coerce").iloc[0])
                    else "",
                    "delta_rmse_vs_base": f"{float(payload.get('delta_rmse_vs_base')):.4f}"
                    if np.isfinite(pd.to_numeric(pd.Series([payload.get("delta_rmse_vs_base")]), errors="coerce").iloc[0])
                    else "",
                    "delta_mae_vs_base": f"{float(payload.get('delta_mae_vs_base')):.4f}"
                    if np.isfinite(pd.to_numeric(pd.Series([payload.get("delta_mae_vs_base")]), errors="coerce").iloc[0])
                    else "",
                    "paired_t_pvalue_pcc": f"{float(payload.get('paired_t_pvalue_pcc')):.4g}"
                    if np.isfinite(pd.to_numeric(pd.Series([payload.get("paired_t_pvalue_pcc")]), errors="coerce").iloc[0])
                    else "",
                    "wilcoxon_pvalue_pcc": f"{float(payload.get('wilcoxon_pvalue_pcc')):.4g}"
                    if np.isfinite(pd.to_numeric(pd.Series([payload.get("wilcoxon_pvalue_pcc")]), errors="coerce").iloc[0])
                    else "",
                    "n_seeds": payload.get("n_seeds"),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "dataset",
                "residual_mode",
                "method",
                "overall_pcc",
                "rmse",
                "mae",
                "delta_pcc_vs_base",
                "delta_rmse_vs_base",
                "delta_mae_vs_base",
                "paired_t_pvalue_pcc",
                "wilcoxon_pvalue_pcc",
                "n_seeds",
            ],
        )
    if method_summary.empty:
        return pd.DataFrame(columns=["dataset", "method", "overall_pcc", "rmse", "mae", "n_seeds"])
    mask = method_summary["method"].astype(str).str.contains("mutation", case=False, na=False)
    rows: list[dict[str, object]] = []
    for row in method_summary[mask].sort_values(["dataset", "method"]).itertuples(index=False):
        payload = row._asdict()
        rows.append(
            {
                "dataset": payload.get("dataset"),
                "method": payload.get("method"),
                "overall_pcc": _fmt(payload.get("overall_pcc_mean"), payload.get("overall_pcc_std")),
                "rmse": _fmt(payload.get("rmse_mean"), payload.get("rmse_std")),
                "mae": _fmt(payload.get("mae_mean"), payload.get("mae_std")),
                "n_seeds": payload.get("n_seeds"),
            }
        )
    return pd.DataFrame(rows, columns=["dataset", "method", "overall_pcc", "rmse", "mae", "n_seeds"])


def _tests_table(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "reference_method",
                "metric",
                "n_pairs",
                "mean_delta",
                "paired_t_pvalue",
                "wilcoxon_pvalue",
            ]
        )
    out = tests.sort_values(["dataset", "reference_method", "method"]).copy()
    for col in ("mean_delta", "paired_t_pvalue", "wilcoxon_pvalue"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: f"{x:.4g}" if np.isfinite(x) else "")
    return out


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    method_summary = _read(results_dir / "method_summary.csv")
    _read(results_dir / "fold_summary.csv")
    ablation = _read(results_dir / "ablation_summary.csv")
    tests = _read(results_dir / "statistical_tests.csv")
    mutation_comparison = _read(results_dir / "mutation_residual_mode_comparison.csv")

    _metric_table(method_summary, "overall_pcc").to_csv(output_dir / "table_main_pcc.csv", index=False, encoding="utf-8-sig")
    _metric_table(method_summary, "rmse").to_csv(output_dir / "table_main_rmse.csv", index=False, encoding="utf-8-sig")
    _ablation_table(ablation).to_csv(output_dir / "table_ablation.csv", index=False, encoding="utf-8-sig")
    _mutation_table(method_summary, mutation_comparison).to_csv(output_dir / "table_mutation_head.csv", index=False, encoding="utf-8-sig")
    _tests_table(tests).to_csv(output_dir / "table_statistical_tests.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote paper tables to {output_dir}")


if __name__ == "__main__":
    main()
