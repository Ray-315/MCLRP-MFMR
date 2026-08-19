from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCLRP_MFMR.paths import RESULTS_DIR
from MCLRP_MFMR.t0_mfmr_protocol import _ablation_summary, _statistical_tests, _summary_from_seed


DETAIL_TABLES = (
    "seed_summary.csv",
    "fold_summary.csv",
    "per_drug_pcc.csv",
    "per_cell_line_pcc.csv",
    "diagnostics.csv",
    "protocol_audit.csv",
)
METRICS = ("overall_pcc", "rmse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict T0 next-stage MFMR outputs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "t0_mfmr")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--mutation-quick-dir", type=Path, default=None)
    parser.add_argument("--mutation-inner-dir", type=Path, default=None)
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _sort_by_order(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "method" in out.columns:
        out["_method_order"] = out["method"].astype(str).map({m: i for i, m in enumerate(methods)}).fillna(len(methods))
    sort_cols = [c for c in ("dataset", "_method_order", "method", "seed", "fold", "drug_idx", "cell_idx", "drug_index") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out.drop(columns=[c for c in ("_method_order",) if c in out.columns])


def _copy_predictions(src_dir: Path, dst_dir: Path, datasets: Iterable[str], methods: Iterable[str]) -> None:
    pred_src = src_dir / "predictions"
    pred_dst = dst_dir / "predictions"
    pred_dst.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        for method in methods:
            src = pred_src / f"{dataset}_{method}_seed0.npz"
            if src.exists():
                shutil.copy2(src, pred_dst / src.name)


def merge_base_and_ablation(results_dir: Path, base_dir: Path, ablation_dir: Path) -> list[str]:
    base_config = _read_json(base_dir / "config.json")
    ablation_config = _read_json(ablation_dir / "config.json")
    base_methods = [str(x) for x in base_config.get("methods", [])]
    ablation_methods = [str(x) for x in ablation_config.get("methods", [])]
    extra_methods = [m for m in ablation_methods if m not in base_methods]
    combined_methods = base_methods + extra_methods
    datasets = [str(x) for x in base_config.get("datasets", [])]

    merged_tables: dict[str, pd.DataFrame] = {}
    for table in DETAIL_TABLES:
        base_df = _read(base_dir / table)
        ablation_df = _read(ablation_dir / table)
        if "method" in base_df.columns:
            base_df = base_df[base_df["method"].astype(str).isin(base_methods)].copy()
        if "method" in ablation_df.columns:
            ablation_df = ablation_df[ablation_df["method"].astype(str).isin(extra_methods)].copy()
        merged = pd.concat([base_df, ablation_df], ignore_index=True)
        merged_tables[table] = _sort_by_order(merged, combined_methods)
        _write(results_dir / table, merged_tables[table])

    seed_df = merged_tables.get("seed_summary.csv", pd.DataFrame())
    method_df = _summary_from_seed(seed_df)
    ablation_df = _ablation_summary(seed_df)
    tests_df = _statistical_tests(seed_df)
    _write(results_dir / "method_summary.csv", _sort_by_order(method_df, combined_methods))
    _write(results_dir / "ablation_summary.csv", _sort_by_order(ablation_df, combined_methods))
    _write(results_dir / "statistical_tests.csv", _sort_by_order(tests_df, combined_methods))

    config = dict(base_config)
    config["methods"] = combined_methods
    config["output_dir"] = str(results_dir.resolve())
    config["next_stage_sources"] = {
        "base_dir": str(base_dir.resolve()),
        "ablation_dir": str(ablation_dir.resolve()),
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    _copy_predictions(base_dir, results_dir, datasets, base_methods)
    _copy_predictions(ablation_dir, results_dir, datasets, extra_methods)
    return combined_methods


def _paired_stats(left: pd.Series, right: pd.Series) -> tuple[float, float, float]:
    delta = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float) - pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    finite = delta[np.isfinite(delta)]
    mean_delta = float(np.mean(finite)) if finite.size else np.nan
    if finite.size < 2:
        return mean_delta, np.nan, np.nan
    if np.allclose(finite, 0.0, atol=1e-15, rtol=0.0):
        return mean_delta, 1.0, 1.0
    try:
        t_p = float(ttest_rel(left, right).pvalue)
    except Exception:
        t_p = np.nan
    try:
        w_p = float(wilcoxon(left, right, zero_method="wilcox").pvalue)
    except Exception:
        w_p = np.nan
    return mean_delta, t_p, w_p


def _mode_label(config: dict[str, Any], fallback: str) -> str:
    cv = int(config.get("mutation_head_config", {}).get("residual_inner_cv", 0))
    return f"inner_cv{cv}" if cv > 1 else fallback


def _mutation_rows_for_mode(run_dir: Path, residual_mode: str, base_seed_fallback: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    seed = _read(run_dir / "seed_summary.csv")
    if seed.empty:
        return []
    base_seed_fallback = base_seed_fallback if base_seed_fallback is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for dataset, ddf in seed.groupby("dataset", sort=True):
        base = ddf[ddf["method"].astype(str) == "mfmr_base"].copy()
        if base.empty and not base_seed_fallback.empty:
            base = base_seed_fallback[
                (base_seed_fallback["dataset"].astype(str) == str(dataset))
                & (base_seed_fallback["method"].astype(str) == "mfmr_base")
            ].copy()
        for method, mdf in ddf.groupby("method", sort=True):
            method = str(method)
            out: dict[str, Any] = {
                "dataset": dataset,
                "residual_mode": residual_mode,
                "method": method,
                "n_seeds": int(mdf["seed"].nunique()),
            }
            for metric in METRICS:
                values = pd.to_numeric(mdf[metric], errors="coerce")
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_std"] = float(values.std(ddof=0))
                if method == "mfmr_mutation" and not base.empty:
                    merged = mdf[["seed", metric]].merge(base[["seed", metric]], on="seed", suffixes=("", "_base"))
                    delta, t_p, w_p = _paired_stats(merged[metric], merged[f"{metric}_base"])
                    suffix = "pcc" if metric == "overall_pcc" else metric
                    out[f"delta_{suffix}_vs_base"] = delta
                    out[f"paired_t_pvalue_{suffix}"] = t_p
                    out[f"wilcoxon_pvalue_{suffix}"] = w_p
                elif method == "mfmr_base":
                    suffix = "pcc" if metric == "overall_pcc" else metric
                    out[f"delta_{suffix}_vs_base"] = 0.0
                    out[f"paired_t_pvalue_{suffix}"] = 1.0
                    out[f"wilcoxon_pvalue_{suffix}"] = 1.0
            rows.append(out)
        if "mfmr_base" not in set(ddf["method"].astype(str)) and not base.empty:
            out = {
                "dataset": dataset,
                "residual_mode": residual_mode,
                "method": "mfmr_base",
                "n_seeds": int(base["seed"].nunique()),
            }
            for metric in METRICS:
                values = pd.to_numeric(base[metric], errors="coerce")
                suffix = "pcc" if metric == "overall_pcc" else metric
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_std"] = float(values.std(ddof=0))
                out[f"delta_{suffix}_vs_base"] = 0.0
                out[f"paired_t_pvalue_{suffix}"] = 1.0
                out[f"wilcoxon_pvalue_{suffix}"] = 1.0
            rows.append(out)
    return rows


def make_mutation_comparison(results_dir: Path, quick_dir: Path | None, inner_dir: Path | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_seed = _read(results_dir / "seed_summary.csv")
    quick_config = _read_json(quick_dir / "config.json") if quick_dir else {}
    inner_config = _read_json(inner_dir / "config.json") if inner_dir else {}
    quick_label = _mode_label(quick_config, "quick_cv0")
    inner_label = _mode_label(inner_config, "inner_cv5")
    if quick_dir:
        rows.extend(_mutation_rows_for_mode(quick_dir, quick_label, base_seed))
    if inner_dir:
        rows.extend(_mutation_rows_for_mode(inner_dir, inner_label, base_seed))
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        _write(results_dir / "mutation_residual_mode_comparison.csv", comparison)
        return comparison

    # Keep one base row per dataset and both mutation-mode rows.
    base_rows = comparison[comparison["method"].astype(str) == "mfmr_base"].copy()
    base_rows = base_rows.sort_values(["dataset", "residual_mode"]).drop_duplicates("dataset", keep="last")
    base_rows["residual_mode"] = "base"
    mutation_rows = comparison[comparison["method"].astype(str) == "mfmr_mutation"].copy()

    if quick_dir and inner_dir and not mutation_rows.empty:
        quick_seed = _read(quick_dir / "seed_summary.csv")
        inner_seed = _read(inner_dir / "seed_summary.csv")
        quick_mut = quick_seed[quick_seed["method"].astype(str) == "mfmr_mutation"]
        inner_mut = inner_seed[inner_seed["method"].astype(str) == "mfmr_mutation"]
        for dataset in sorted(set(inner_mut["dataset"].astype(str))):
            q = quick_mut[quick_mut["dataset"].astype(str) == dataset]
            i = inner_mut[inner_mut["dataset"].astype(str) == dataset]
            for metric in METRICS:
                merged = i[["seed", metric]].merge(q[["seed", metric]], on="seed", suffixes=("_inner", "_quick"))
                delta, t_p, w_p = _paired_stats(merged[f"{metric}_inner"], merged[f"{metric}_quick"])
                suffix = "pcc" if metric == "overall_pcc" else metric
                mask = (mutation_rows["dataset"].astype(str) == dataset) & (mutation_rows["residual_mode"].astype(str) == inner_label)
                mutation_rows.loc[mask, f"delta_{suffix}_vs_quick_mutation"] = delta
                mutation_rows.loc[mask, f"paired_t_pvalue_{suffix}_vs_quick"] = t_p
                mutation_rows.loc[mask, f"wilcoxon_pvalue_{suffix}_vs_quick"] = w_p
                qmask = (mutation_rows["dataset"].astype(str) == dataset) & (mutation_rows["residual_mode"].astype(str) == quick_label)
                mutation_rows.loc[qmask, f"delta_{suffix}_vs_quick_mutation"] = 0.0

    out = pd.concat([base_rows, mutation_rows], ignore_index=True)
    out = out.sort_values(["dataset", "residual_mode", "method"]).reset_index(drop=True)
    _write(results_dir / "mutation_residual_mode_comparison.csv", out)
    return out


def _avg_per_drug(path: Path, method: str, prefix: str) -> pd.DataFrame:
    df = _read(path / "per_drug_pcc.csv")
    if df.empty:
        return pd.DataFrame()
    df = df[df["method"].astype(str) == method].copy()
    if df.empty:
        return df
    grouped = (
        df.groupby(["dataset", "drug_idx", "drug"], as_index=False)
        .agg(pcc=("pcc", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"), n_test=("n_test", "sum"), n_seeds=("seed", "nunique"))
        .rename(columns={"pcc": f"{prefix}_pcc", "rmse": f"{prefix}_rmse", "mae": f"{prefix}_mae"})
    )
    return grouped


def _top_rows(df: pd.DataFrame, column: str, label: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for dataset, ddf in df.groupby("dataset", sort=True):
        keep = ddf[np.isfinite(pd.to_numeric(ddf[column], errors="coerce"))].copy()
        if keep.empty:
            continue
        keep = keep.sort_values(column, ascending=ascending).head(n)
        keep.insert(1, "comparison", label)
        rows.append(keep)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "No matched rows.\n"
    view = df[columns].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{float(x):.4f}" if np.isfinite(float(x)) else "")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in view.iterrows()]
    return "\n".join([header, sep, *body]) + "\n"


def make_biology_outputs(
    results_dir: Path,
    mutation_quick_dir: Path | None,
    mutation_inner_dir: Path | None,
) -> pd.DataFrame:
    biology_dir = results_dir / "biology"
    biology_dir.mkdir(parents=True, exist_ok=True)
    base = _avg_per_drug(results_dir, "mfmr_base", "mfmr_base")
    original = _avg_per_drug(results_dir, "original_mclrp", "original_mclrp")
    ridge = _avg_per_drug(results_dir, "ridge_only", "ridge_only")
    if base.empty:
        gains = pd.DataFrame()
        _write(biology_dir / "per_drug_gain_by_dataset.csv", gains)
        return gains
    gains = base.merge(original, on=["dataset", "drug_idx", "drug"], how="left").merge(ridge, on=["dataset", "drug_idx", "drug"], how="left")
    gains["gain_vs_original_mclrp_pcc"] = gains["mfmr_base_pcc"] - gains["original_mclrp_pcc"]
    gains["gain_vs_ridge_only_pcc"] = gains["mfmr_base_pcc"] - gains["ridge_only_pcc"]

    if mutation_quick_dir:
        quick = _avg_per_drug(mutation_quick_dir, "mfmr_mutation", "mfmr_mutation_quick")
        if not quick.empty:
            gains = gains.merge(quick[["dataset", "drug_idx", "drug", "mfmr_mutation_quick_pcc"]], on=["dataset", "drug_idx", "drug"], how="left")
            gains["mutation_quick_gain_vs_base_pcc"] = gains["mfmr_mutation_quick_pcc"] - gains["mfmr_base_pcc"]
    if mutation_inner_dir:
        inner = _avg_per_drug(mutation_inner_dir, "mfmr_mutation", "mfmr_mutation_inner")
        if not inner.empty:
            gains = gains.merge(inner[["dataset", "drug_idx", "drug", "mfmr_mutation_inner_pcc"]], on=["dataset", "drug_idx", "drug"], how="left")
            gains["mutation_inner_gain_vs_base_pcc"] = gains["mfmr_mutation_inner_pcc"] - gains["mfmr_base_pcc"]

    gains = gains.sort_values(["dataset", "drug_idx"]).reset_index(drop=True)
    _write(biology_dir / "per_drug_gain_by_dataset.csv", gains)

    top = pd.concat(
        [
            _top_rows(gains, "gain_vs_original_mclrp_pcc", "mfmr_base_vs_original_mclrp", 10, ascending=False),
            _top_rows(gains, "gain_vs_ridge_only_pcc", "mfmr_base_vs_ridge_only", 10, ascending=False),
        ],
        ignore_index=True,
    )
    _write(biology_dir / "top_improved_drugs.csv", top)

    degraded_parts = [
        _top_rows(gains[gains["gain_vs_original_mclrp_pcc"] < 0], "gain_vs_original_mclrp_pcc", "mfmr_base_vs_original_mclrp", 10, ascending=True),
        _top_rows(gains[gains["gain_vs_ridge_only_pcc"] < 0], "gain_vs_ridge_only_pcc", "mfmr_base_vs_ridge_only", 10, ascending=True),
    ]
    if "mutation_inner_gain_vs_base_pcc" in gains.columns:
        degraded_parts.append(_top_rows(gains[gains["mutation_inner_gain_vs_base_pcc"] < 0], "mutation_inner_gain_vs_base_pcc", "mfmr_mutation_inner_vs_base", 10, ascending=True))
    degraded = pd.concat(degraded_parts, ignore_index=True) if degraded_parts else pd.DataFrame()
    _write(biology_dir / "worst_degraded_drugs.csv", degraded)

    lines = ["# Representative Drug Cases", ""]
    for dataset, ddf in gains.groupby("dataset", sort=True):
        lines.extend([f"## {dataset}", ""])
        lines.extend(["Top mfmr_base gains over original_mclrp:", ""])
        lines.append(_markdown_table(ddf.sort_values("gain_vs_original_mclrp_pcc", ascending=False), ["drug", "mfmr_base_pcc", "original_mclrp_pcc", "gain_vs_original_mclrp_pcc"]))
        lines.extend(["Top mfmr_base gains over ridge_only:", ""])
        lines.append(_markdown_table(ddf.sort_values("gain_vs_ridge_only_pcc", ascending=False), ["drug", "mfmr_base_pcc", "ridge_only_pcc", "gain_vs_ridge_only_pcc"]))
        under = ddf[(ddf["gain_vs_original_mclrp_pcc"] < 0) | (ddf["gain_vs_ridge_only_pcc"] < 0)].copy()
        lines.extend(["Underperforming cases:", ""])
        lines.append(_markdown_table(under.sort_values(["gain_vs_original_mclrp_pcc", "gain_vs_ridge_only_pcc"]), ["drug", "mfmr_base_pcc", "gain_vs_original_mclrp_pcc", "gain_vs_ridge_only_pcc"]))
        if "mutation_inner_gain_vs_base_pcc" in ddf.columns:
            lines.extend(["Mutation residual contribution cases:", ""])
            lines.append(_markdown_table(ddf.sort_values("mutation_inner_gain_vs_base_pcc", ascending=False), ["drug", "mfmr_base_pcc", "mfmr_mutation_inner_pcc", "mutation_inner_gain_vs_base_pcc"]))
        lines.append("")
    (biology_dir / "representative_drug_cases.md").write_text("\n".join(lines), encoding="utf-8")
    return gains


def _fmt_num(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    return f"{x:.{digits}f}" if np.isfinite(x) else ""


def write_ablation_report(results_dir: Path) -> None:
    ablation = _read(results_dir / "ablation_summary.csv")
    tests = _read(results_dir / "statistical_tests.csv")
    lines = ["# Ablation Benchmark Report", ""]
    lines.append("All rows use the strict T0 random-entry masking protocol and the same fold masks as mfmr_base for each dataset/seed/fold.")
    lines.append("")
    if ablation.empty:
        lines.append("No ablation rows were available.")
    else:
        for dataset, ddf in ablation.groupby("dataset", sort=True):
            lines.extend([f"## {dataset}", ""])
            table = ddf[["method", "reference_method", "n_pairs", "delta_pcc_mean", "delta_rmse_mean", "delta_mae_mean"]].copy()
            lines.append(_markdown_table(table, ["method", "reference_method", "n_pairs", "delta_pcc_mean", "delta_rmse_mean", "delta_mae_mean"], max_rows=50))
            pcc_tests = tests[(tests["dataset"].astype(str) == str(dataset)) & (tests["metric"].astype(str) == "overall_pcc")]
            if not pcc_tests.empty:
                lines.append("Paired seed-level PCC tests:")
                lines.append(_markdown_table(pcc_tests[["method", "reference_method", "mean_delta", "paired_t_pvalue", "wilcoxon_pvalue"]], ["method", "reference_method", "mean_delta", "paired_t_pvalue", "wilcoxon_pvalue"], max_rows=50))
            contribution = ddf.copy()
            contribution["branch_contribution_signal"] = np.where(
                (contribution["delta_pcc_mean"] < 0) & (contribution["delta_rmse_mean"] > 0) & (contribution["delta_mae_mean"] > 0),
                "supports positive contribution",
                "mixed or neutral",
            )
            lines.append("Interpretation:")
            lines.append(_markdown_table(contribution[["method", "branch_contribution_signal"]], ["method", "branch_contribution_signal"], max_rows=50))
    (results_dir / "ABLATION_BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_mutation_report(results_dir: Path) -> None:
    comparison = _read(results_dir / "mutation_residual_mode_comparison.csv")
    lines = ["# Mutation Head Benchmark Report", ""]
    lines.append("Mutation-head runs compare mfmr_base, mfmr_mutation quick residual fitting, and mfmr_mutation with inner cross-fitting.")
    lines.append("")
    if comparison.empty:
        lines.append("No mutation comparison rows were available.")
    else:
        for dataset, ddf in comparison.groupby("dataset", sort=True):
            lines.extend([f"## {dataset}", ""])
            cols = ["residual_mode", "method", "overall_pcc_mean", "rmse_mean", "mae_mean", "delta_pcc_vs_base", "delta_rmse_vs_base", "delta_mae_vs_base"]
            lines.append(_markdown_table(ddf, cols, max_rows=20))
            improved = ddf[(ddf["method"].astype(str) == "mfmr_mutation") & (pd.to_numeric(ddf["delta_pcc_vs_base"], errors="coerce") > 0)]
            status = "improved over mfmr_base by PCC" if not improved.empty else "did not improve over mfmr_base by PCC"
            lines.append(f"Interpretation: mutation residual head {status} in the available mode rows.")
            lines.append("")
    (results_dir / "MUTATION_HEAD_BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_stage_summary(results_dir: Path) -> None:
    ablation = _read(results_dir / "ablation_summary.csv")
    mutation = _read(results_dir / "mutation_residual_mode_comparison.csv")
    biology = _read(results_dir / "biology" / "per_drug_gain_by_dataset.csv")
    lines = [
        "# Next Stage Summary",
        "",
        "## Status",
        "",
        f"- Ablation benchmark: {'complete' if not ablation.empty else 'not complete'}; rows={len(ablation)}.",
        f"- Mutation-head benchmark: {'complete' if not mutation.empty else 'not complete'}; rows={len(mutation)}.",
        f"- Biological interpretation outputs: {'complete' if not biology.empty else 'not complete'}; rows={len(biology)}.",
        "",
        "## Remaining Limitations",
        "",
        "- Mutation-head interpretation is associative and should not be presented as causal biology.",
        "- Per-drug PCC can be unstable for drugs with narrow response variance or limited effective test support.",
        "- The strict T0 setting evaluates random held-out entries, not unseen-cell-line or unseen-drug generalization.",
        "",
        "## Manuscript Recommendation",
        "",
        "- Main results: keep mfmr_base as the primary strict T0 model unless mutation inner cross-fit shows consistent positive deltas across datasets.",
        "- Supplementary results: include ablation deltas, paired seed-level tests, mutation quick-vs-inner comparison, and per-drug gain/degradation tables.",
    ]
    (results_dir / "NEXT_STAGE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    base_dir = args.base_dir.resolve()
    ablation_dir = args.ablation_dir.resolve()
    quick_dir = args.mutation_quick_dir.resolve() if args.mutation_quick_dir else None
    inner_dir = args.mutation_inner_dir.resolve() if args.mutation_inner_dir else None
    results_dir.mkdir(parents=True, exist_ok=True)

    merge_base_and_ablation(results_dir, base_dir, ablation_dir)
    make_mutation_comparison(results_dir, quick_dir, inner_dir)
    make_biology_outputs(results_dir, quick_dir, inner_dir)
    write_ablation_report(results_dir)
    write_mutation_report(results_dir)
    write_next_stage_summary(results_dir)
    print(f"Wrote next-stage outputs to {results_dir}")


if __name__ == "__main__":
    main()
