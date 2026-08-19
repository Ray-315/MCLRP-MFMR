from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import RESULTS_DIR


DEFAULT_INPUT_DIR = RESULTS_DIR / "reviewer_mclrp_ablation_5task_10x10_v1"
DEFAULT_EIGHT_MODEL_SOURCE = RESULTS_DIR / "paper_figures" / "fig4_8model" / "fig4_8model_per_drug_metrics.csv"
FULL_MODEL = "MCLRP"
VARIANTS = ("MCLRP_NoPCA_reconstructed", "MCLRP_NoTrace_reconstructed")
MODEL_ORDER = (FULL_MODEL, *VARIANTS)
METRICS = ("pcc", "scc", "rmse", "mae")
CORRELATION_METRICS = ("pcc", "scc")
BOOTSTRAP_N = 20_000
BOOTSTRAP_SEED = 731
EIGHT_MODEL_PROPOSED = "MCLRP_MFMR"
EIGHT_MODEL_BASELINES = ("MCLRP", "MC", "MCRR", "SRMF", "DeepIC50", "GeneVAE", "RR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the paired five-task reconstructed MCLRP ablations")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eight-model-source", type=Path, default=DEFAULT_EIGHT_MODEL_SOURCE)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def bh_adjust(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.where(np.isfinite(values))[0]
    if finite_indices.size == 0:
        return result
    ordered_indices = finite_indices[np.argsort(values[finite_indices])]
    ordered = values[ordered_indices]
    adjusted = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered_indices] = np.minimum(adjusted, 1.0)
    return result


def safe_ttest_rel(full: np.ndarray, variant: np.ndarray) -> float:
    full = np.asarray(full, dtype=float)
    variant = np.asarray(variant, dtype=float)
    valid = np.isfinite(full) & np.isfinite(variant)
    if int(valid.sum()) < 2:
        return float("nan")
    if np.allclose(full[valid], variant[valid], atol=0.0, rtol=0.0):
        return 1.0
    return float(ttest_rel(variant[valid], full[valid]).pvalue)


def safe_wilcoxon(full: np.ndarray, variant: np.ndarray) -> float:
    full = np.asarray(full, dtype=float)
    variant = np.asarray(variant, dtype=float)
    valid = np.isfinite(full) & np.isfinite(variant)
    delta = variant[valid] - full[valid]
    if delta.size == 0:
        return float("nan")
    if np.allclose(delta, 0.0, atol=0.0, rtol=0.0):
        return 1.0
    return float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)


def paired_seed_tests(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full = seed_metrics.loc[seed_metrics["model"].eq(FULL_MODEL)].copy()
    for variant in VARIANTS:
        other = seed_metrics.loc[seed_metrics["model"].eq(variant)].copy()
        merged = full.merge(other, on=["dataset", "seed"], suffixes=("_full", "_variant"), validate="one_to_one")
        for dataset, group in merged.groupby("dataset", sort=False):
            for metric in METRICS:
                full_values = group[f"{metric}_full"].to_numpy(dtype=float)
                variant_values = group[f"{metric}_variant"].to_numpy(dtype=float)
                delta = variant_values - full_values
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "reference": FULL_MODEL,
                        "metric": metric,
                        "delta_definition": "variant_minus_full_mclrp",
                        "n_pairs": int(np.isfinite(delta).sum()),
                        "mean_delta": float(np.nanmean(delta)),
                        "sd_delta": float(np.nanstd(delta, ddof=0)),
                        "paired_t_pvalue": safe_ttest_rel(full_values, variant_values),
                        "wilcoxon_pvalue": safe_wilcoxon(full_values, variant_values),
                    }
                )
    out = pd.DataFrame(rows)
    out["wilcoxon_q_bh_within_metric"] = np.nan
    for metric, indices in out.groupby("metric", sort=False).groups.items():
        out.loc[indices, "wilcoxon_q_bh_within_metric"] = bh_adjust(out.loc[indices, "wilcoxon_pvalue"])
    return out


def build_task_deltas(task_summary: pd.DataFrame) -> pd.DataFrame:
    full = task_summary.loc[task_summary["model"].eq(FULL_MODEL)].copy()
    rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        other = task_summary.loc[task_summary["model"].eq(variant)].copy()
        merged = other.merge(full, on="dataset", suffixes=("_variant", "_full"), validate="one_to_one")
        record = merged[["dataset"]].copy()
        record["variant"] = variant
        record["reference"] = FULL_MODEL
        record["delta_definition"] = "variant_minus_full_mclrp"
        for metric in ("macro_pcc", "macro_scc", "entry_pcc", "entry_scc", "entry_rmse", "entry_mae"):
            record[f"delta_{metric}"] = merged[f"{metric}_variant"] - merged[f"{metric}_full"]
            record[f"{metric}_variant"] = merged[f"{metric}_variant"]
            record[f"{metric}_full"] = merged[f"{metric}_full"]
        rows.append(record)
    return pd.concat(rows, ignore_index=True)


def build_overall_summary(task_summary: pd.DataFrame) -> pd.DataFrame:
    overall = (
        task_summary.groupby(["model", "display_name"], as_index=False)
        .agg(
            macro_pcc=("macro_pcc", "mean"),
            macro_scc=("macro_scc", "mean"),
            mean_entry_pcc=("entry_pcc", "mean"),
            mean_entry_scc=("entry_scc", "mean"),
            n_tasks=("dataset", "nunique"),
            total_valid_drugs=("n_valid_pcc", "sum"),
        )
    )
    base = overall.loc[overall["model"].eq(FULL_MODEL)].iloc[0]
    overall["delta_macro_pcc_vs_mclrp"] = overall["macro_pcc"] - float(base["macro_pcc"])
    overall["delta_macro_scc_vs_mclrp"] = overall["macro_scc"] - float(base["macro_scc"])
    overall["delta_definition"] = "variant_minus_full_mclrp"
    overall["model_order"] = overall["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
    return overall.sort_values("model_order").drop(columns="model_order").reset_index(drop=True)


def build_per_drug_deltas(mean_per_drug: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "drug_idx", "drug"]
    full = mean_per_drug.loc[mean_per_drug["model"].eq(FULL_MODEL), keys + ["n", "pcc", "scc"]].copy()
    rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        other = mean_per_drug.loc[mean_per_drug["model"].eq(variant), keys + ["n", "pcc", "scc"]].copy()
        merged = other.merge(full, on=keys, suffixes=("_variant", "_full"), validate="one_to_one")
        merged["variant"] = variant
        merged["reference"] = FULL_MODEL
        merged["delta_definition"] = "variant_minus_full_mclrp"
        merged["delta_pcc"] = merged["pcc_variant"] - merged["pcc_full"]
        merged["delta_scc"] = merged["scc_variant"] - merged["scc_full"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def hierarchical_bootstrap(
    values: pd.DataFrame,
    *,
    task_col: str,
    value_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, int, int]:
    grouped = {
        str(task): group[value_col].to_numpy(dtype=float)
        for task, group in values.groupby(task_col, sort=False)
    }
    grouped = {task: vals[np.isfinite(vals)] for task, vals in grouped.items() if np.isfinite(vals).any()}
    tasks = list(grouped)
    if not tasks:
        return np.nan, np.nan, np.nan, 0, 0
    estimate = float(np.mean([np.mean(grouped[task]) for task in tasks]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in sampled_tasks:
            task_values = grouped[str(task)]
            task_means.append(float(np.mean(rng.choice(task_values, size=len(task_values), replace=True))))
        boot[iteration] = float(np.mean(task_means))
    return estimate, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(tasks), int(sum(len(v) for v in grouped.values()))


def summarize_ablation_bootstrap(per_drug_delta: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_index, variant in enumerate(VARIANTS):
        variant_df = per_drug_delta.loc[per_drug_delta["variant"].eq(variant)]
        for metric_index, metric in enumerate(CORRELATION_METRICS):
            estimate, low, high, n_tasks, n_units = hierarchical_bootstrap(
                variant_df,
                task_col="dataset",
                value_col=f"delta_{metric}",
                n_boot=n_boot,
                seed=seed + variant_index * 101 + metric_index,
            )
            rows.append(
                {
                    "scope": "all_five_tasks",
                    "dataset": "ALL",
                    "variant": variant,
                    "reference": FULL_MODEL,
                    "metric": metric,
                    "delta_definition": "variant_minus_full_mclrp",
                    "mean_delta": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_tasks": n_tasks,
                    "n_drug_task_units": n_units,
                    "bootstrap_resamples": n_boot,
                    "bootstrap_seed": seed + variant_index * 101 + metric_index,
                    "resampling": "response matrices, then drugs within matrix",
                }
            )
            for dataset_index, (dataset, group) in enumerate(variant_df.groupby("dataset", sort=False)):
                estimate, low, high, n_tasks, n_units = hierarchical_bootstrap(
                    group,
                    task_col="dataset",
                    value_col=f"delta_{metric}",
                    n_boot=n_boot,
                    seed=seed + 1000 + variant_index * 101 + metric_index * 17 + dataset_index,
                )
                rows.append(
                    {
                        "scope": "single_task",
                        "dataset": dataset,
                        "variant": variant,
                        "reference": FULL_MODEL,
                        "metric": metric,
                        "delta_definition": "variant_minus_full_mclrp",
                        "mean_delta": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_tasks": n_tasks,
                        "n_drug_task_units": n_units,
                        "bootstrap_resamples": n_boot,
                        "bootstrap_seed": seed + 1000 + variant_index * 101 + metric_index * 17 + dataset_index,
                        "resampling": "drugs within one response matrix",
                    }
                )
    return pd.DataFrame(rows)


def summarize_eight_model_bootstrap(source: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    keys = ["dataset_label", "drug_idx", "drug"]
    proposed = source.loc[source["model"].eq(EIGHT_MODEL_PROPOSED), keys + ["pcc", "scc"]].copy()
    rows: list[dict[str, object]] = []
    for baseline_index, baseline in enumerate(EIGHT_MODEL_BASELINES):
        other = source.loc[source["model"].eq(baseline), keys + ["pcc", "scc"]].copy()
        merged = other.merge(proposed, on=keys, suffixes=("_baseline", "_proposed"), validate="one_to_one")
        for metric_index, metric in enumerate(CORRELATION_METRICS):
            merged[f"delta_{metric}"] = merged[f"{metric}_proposed"] - merged[f"{metric}_baseline"]
            estimate, low, high, n_tasks, n_units = hierarchical_bootstrap(
                merged,
                task_col="dataset_label",
                value_col=f"delta_{metric}",
                n_boot=n_boot,
                seed=seed + 5000 + baseline_index * 101 + metric_index,
            )
            rows.append(
                {
                    "baseline_model": baseline,
                    "reference_model": EIGHT_MODEL_PROPOSED,
                    "metric": metric,
                    "delta_definition": "mfmr_mutation_v2_minus_comparator",
                    "mean_delta": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_tasks": n_tasks,
                    "n_drug_task_units": n_units,
                    "bootstrap_resamples": n_boot,
                    "bootstrap_seed": seed + 5000 + baseline_index * 101 + metric_index,
                    "resampling": "response matrices, then drugs within matrix",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(require_file(input_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {manifest.get('status')}")
    if manifest.get("completed_seed_model_units") != 150 or manifest.get("completed_fold_model_units") != 1500:
        raise RuntimeError("Expected exactly 150 seed-model units and 1500 fold-model units")

    seed_metrics = pd.read_csv(require_file(input_dir / "seed_metrics.csv"))
    task_summary = pd.read_csv(require_file(input_dir / "task_model_summary.csv"))
    mean_per_drug = pd.read_csv(require_file(input_dir / "mean_per_drug_metrics.csv"))
    if seed_metrics[list(METRICS)].isna().any().any():
        raise RuntimeError("Seed metrics contain missing values")

    tests = paired_seed_tests(seed_metrics)
    task_deltas = build_task_deltas(task_summary)
    overall = build_overall_summary(task_summary)
    per_drug_delta = build_per_drug_deltas(mean_per_drug)
    ablation_bootstrap = summarize_ablation_bootstrap(per_drug_delta, args.bootstrap, args.bootstrap_seed)
    eight_source = pd.read_csv(require_file(args.eight_model_source.resolve()))
    eight_bootstrap = summarize_eight_model_bootstrap(eight_source, args.bootstrap, args.bootstrap_seed)

    outputs = {
        "mclrp_ablation_paired_seed_tests.csv": tests,
        "mclrp_ablation_task_deltas.csv": task_deltas,
        "mclrp_ablation_overall_summary.csv": overall,
        "mclrp_ablation_per_drug_deltas.csv": per_drug_delta,
        "mclrp_ablation_hierarchical_bootstrap.csv": ablation_bootstrap,
        "eight_model_hierarchical_deltas.csv": eight_bootstrap,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    analysis_manifest = {
        "source_run": str(input_dir),
        "source_config_hash": manifest["config_hash"],
        "delta_definitions": {
            "mclrp_ablation": "variant - full MCLRP",
            "eight_model": "MFMR-mutation_v2 - comparator",
            "mfmr_ablation_table5": "ablated MFMR - full MFMR (unchanged source convention)",
        },
        "bootstrap_resamples": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_resampling": "response matrices, then drugs within matrix",
        "bh_family": "ten dataset-by-variant comparisons within each metric",
        "outputs": list(outputs),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(analysis_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
