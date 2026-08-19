from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCLRP_MFMR.paths import RESULTS_DIR
from MCLRP_MFMR.t0_mfmr_protocol import load_t0_dataset, subset_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready figures from strict T0 benchmark outputs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "t0_mfmr")
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_config(results_dir: Path) -> dict:
    path = results_dir / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)
    _finish(fig, path)


def _method_comparison(method_summary: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "method_comparison_pcc.png"
    if method_summary.empty or "overall_pcc_mean" not in method_summary.columns:
        _placeholder(path, "Method comparison", "method_summary.csv is missing or empty.")
        return
    df = method_summary.sort_values(["dataset", "method"]).copy()
    datasets = list(df["dataset"].astype(str).unique())
    methods = list(df["method"].astype(str).unique())
    x = np.arange(len(datasets), dtype=float)
    width = min(0.8 / max(len(methods), 1), 0.16)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.4), 4.8))
    for idx, method in enumerate(methods):
        y: list[float] = []
        err: list[float] = []
        for dataset in datasets:
            row = df[(df["dataset"].astype(str) == dataset) & (df["method"].astype(str) == method)]
            y.append(float(row["overall_pcc_mean"].iloc[0]) if not row.empty else np.nan)
            err.append(float(row["overall_pcc_std"].iloc[0]) if not row.empty and "overall_pcc_std" in row else 0.0)
        ax.bar(x + (idx - (len(methods) - 1) / 2) * width, y, width=width, yerr=err, label=method, capsize=2)
    ax.set_ylabel("Overall PCC")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=min(3, len(methods)))
    _finish(fig, path)


def _ablation_plot(ablation: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "ablation_comparison.png"
    if ablation.empty or "delta_pcc_mean" not in ablation.columns:
        _placeholder(path, "Ablation comparison", "No ablation rows are available for this run.")
        return
    df = ablation.sort_values(["dataset", "method"]).copy()
    labels = (df["dataset"].astype(str) + "\n" + df["method"].astype(str)).tolist()
    values = pd.to_numeric(df["delta_pcc_mean"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.5))
    ax.bar(np.arange(len(labels)), values, color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta PCC vs reference")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    _finish(fig, path)


def _per_drug_gain(per_drug: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "per_drug_pcc_gain.png"
    needed = {"dataset", "method", "drug_idx", "pcc"}
    if per_drug.empty or not needed.issubset(per_drug.columns):
        _placeholder(path, "Per-drug PCC gain", "per_drug_pcc.csv is missing required columns.")
        return
    methods = set(per_drug["method"].astype(str))
    if not {"mfmr_base", "original_mclrp", "ridge_only"}.issubset(methods):
        _placeholder(path, "Per-drug PCC gain", "mfmr_base, original_mclrp, and ridge_only are required.")
        return
    avg = per_drug.groupby(["dataset", "method", "drug_idx"], as_index=False)["pcc"].mean()
    base = avg[avg["method"] == "mfmr_base"][["dataset", "drug_idx", "pcc"]].rename(columns={"pcc": "mfmr_base"})
    original = avg[avg["method"] == "original_mclrp"][["dataset", "drug_idx", "pcc"]].rename(columns={"pcc": "original_mclrp"})
    ridge = avg[avg["method"] == "ridge_only"][["dataset", "drug_idx", "pcc"]].rename(columns={"pcc": "ridge_only"})
    merged = base.merge(original, on=["dataset", "drug_idx"]).merge(ridge, on=["dataset", "drug_idx"])
    if merged.empty:
        _placeholder(path, "Per-drug PCC gain", "No matched per-drug rows are available.")
        return
    merged["gain_vs_original_mclrp"] = merged["mfmr_base"] - merged["original_mclrp"]
    merged["gain_vs_ridge_only"] = merged["mfmr_base"] - merged["ridge_only"]
    merged = merged.sort_values(["dataset", "drug_idx"])
    x = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(max(8, len(merged) * 0.32), 4.6))
    ax.plot(x, merged["gain_vs_original_mclrp"], marker="o", linewidth=1.2, label="mfmr_base - original_mclrp")
    ax.plot(x, merged["gain_vs_ridge_only"], marker="s", linewidth=1.2, label="mfmr_base - ridge_only")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("PCC gain")
    ax.set_xlabel("Dataset/drug index")
    ax.set_xticks(x)
    ax.set_xticklabels((merged["dataset"].astype(str) + "/" + merged["drug_idx"].astype(str)).tolist(), rotation=90, fontsize=7)
    ax.legend(frameon=False)
    _finish(fig, path)


def _scatter_prediction(results_dir: Path, config: dict, output_dir: Path) -> None:
    path = output_dir / "predicted_vs_observed_scatter.png"
    pred_dir = results_dir / "predictions"
    if not pred_dir.exists():
        _placeholder(path, "Predicted vs observed", "No saved prediction arrays are available.")
        return
    datasets = [str(x) for x in config.get("datasets", [])]
    methods = [str(x) for x in config.get("methods", [])]
    candidates: list[tuple[str, str, Path]] = []
    preferred_methods = [m for m in ("mfmr_base", "mfmr_mutation", "ridge_only", "original_mclrp") if m in methods] + methods
    for dataset in datasets:
        for method in preferred_methods:
            npz = pred_dir / f"{dataset}_{method}_seed0.npz"
            if npz.exists():
                candidates.append((dataset, method, npz))
                break
        if candidates:
            break
    if not candidates:
        _placeholder(path, "Predicted vs observed", "No seed0 prediction NPZ matched config.json.")
        return
    dataset, method, npz = candidates[0]
    bundle = load_t0_dataset(dataset)
    subset, _, _ = subset_dataset(bundle, max_cell_lines=config.get("max_cell_lines"), max_drugs=config.get("max_drugs"))
    with np.load(npz, allow_pickle=False) as payload:
        pred = payload["prediction"]
        folds = payload["folds"]
    mask = np.any(folds != 0, axis=0)
    y_true = subset.M[mask].astype(float)
    y_pred = pred[mask].astype(float)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[good]
    y_pred = y_pred[good]
    if y_true.size > 5000:
        rng = np.random.default_rng(0)
        keep = rng.choice(y_true.size, size=5000, replace=False)
        y_true = y_true[keep]
        y_pred = y_pred[keep]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_true, y_pred, s=10, alpha=0.45, edgecolors="none")
    lo = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
    hi = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9)
    ax.set_xlabel("Observed response")
    ax.set_ylabel("Predicted response")
    ax.set_title(f"{dataset} / {method} / seed 0")
    _finish(fig, path)


def _mutation_improvement(
    per_drug: pd.DataFrame,
    seed_summary: pd.DataFrame,
    mutation_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    path = output_dir / "mutation_residual_improvement.png"
    if not per_drug.empty and {"dataset", "method", "drug_idx", "pcc"}.issubset(per_drug.columns):
        methods = set(per_drug["method"].astype(str))
        if {"mfmr_mutation", "mfmr_base"}.issubset(methods):
            avg = per_drug.groupby(["dataset", "method", "drug_idx"], as_index=False)["pcc"].mean()
            base = avg[avg["method"] == "mfmr_base"][["dataset", "drug_idx", "pcc"]].rename(columns={"pcc": "mfmr_base"})
            mut = avg[avg["method"] == "mfmr_mutation"][["dataset", "drug_idx", "pcc"]].rename(columns={"pcc": "mfmr_mutation"})
            merged = mut.merge(base, on=["dataset", "drug_idx"])
            merged["gain"] = merged["mfmr_mutation"] - merged["mfmr_base"]
            labels = (merged["dataset"].astype(str) + "/" + merged["drug_idx"].astype(str)).tolist()
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.32), 4.4))
            ax.bar(np.arange(len(labels)), merged["gain"], color="#59a14f")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("PCC gain")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            _finish(fig, path)
            return
    if not seed_summary.empty and {"dataset", "method", "seed", "overall_pcc"}.issubset(seed_summary.columns):
        methods = set(seed_summary["method"].astype(str))
        if {"mfmr_mutation", "mfmr_base"}.issubset(methods):
            base = seed_summary[seed_summary["method"] == "mfmr_base"][["dataset", "seed", "overall_pcc"]]
            mut = seed_summary[seed_summary["method"] == "mfmr_mutation"][["dataset", "seed", "overall_pcc"]]
            merged = mut.merge(base, on=["dataset", "seed"], suffixes=("_mutation", "_base"))
            if not merged.empty:
                grouped = (merged["overall_pcc_mutation"] - merged["overall_pcc_base"]).groupby(merged["dataset"]).mean()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(np.arange(len(grouped)), grouped.to_numpy(), color="#59a14f")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel("Mean PCC gain")
                ax.set_xticks(np.arange(len(grouped)))
                ax.set_xticklabels(grouped.index.astype(str), rotation=25, ha="right")
                _finish(fig, path)
                return
    if not mutation_comparison.empty and {"dataset", "method", "residual_mode", "delta_pcc_vs_base"}.issubset(mutation_comparison.columns):
        df = mutation_comparison[mutation_comparison["method"].astype(str) == "mfmr_mutation"].copy()
        if not df.empty:
            df = df.sort_values(["dataset", "residual_mode"])
            labels = (df["dataset"].astype(str) + "\n" + df["residual_mode"].astype(str)).tolist()
            values = pd.to_numeric(df["delta_pcc_vs_base"], errors="coerce").to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4.4))
            ax.bar(np.arange(len(labels)), values, color="#59a14f")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Mean PCC gain vs mfmr_base")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            _finish(fig, path)
            return
    _placeholder(path, "Mutation residual improvement", "mfmr_mutation and mfmr_base matched rows are required.")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(results_dir)

    method_summary = _read(results_dir / "method_summary.csv")
    ablation = _read(results_dir / "ablation_summary.csv")
    per_drug = _read(results_dir / "per_drug_pcc.csv")
    seed_summary = _read(results_dir / "seed_summary.csv")
    mutation_comparison = _read(results_dir / "mutation_residual_mode_comparison.csv")

    _method_comparison(method_summary, output_dir)
    _ablation_plot(ablation, output_dir)
    _per_drug_gain(per_drug, output_dir)
    _scatter_prediction(results_dir, config, output_dir)
    _mutation_improvement(per_drug, seed_summary, mutation_comparison, output_dir)
    print(f"Wrote paper figures to {output_dir}")


if __name__ == "__main__":
    main()
