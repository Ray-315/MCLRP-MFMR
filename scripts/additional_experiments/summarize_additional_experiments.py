from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from MCLRP_MFMR.additional_experiments import endpoint_ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a data-derived summary of the additional experiments")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/additional_experiments")
    return parser.parse_args()


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    adaptive = pd.read_csv(args.results_dir / "adaptive_fusion/raw_seed_metrics.csv")
    alpha = pd.read_csv(args.results_dir / "adaptive_fusion/alpha_summary.csv")
    sensitivity = pd.read_csv(args.results_dir / "mask_sensitivity/raw_seed_metrics.csv")
    slopes = pd.read_csv(args.results_dir / "mask_sensitivity/degradation_slopes.csv")
    branches = pd.read_csv(args.results_dir / "branch_complementarity/raw_seed_metrics.csv")
    quartiles = pd.read_csv(args.results_dir / "branch_complementarity/raw_quartile_metrics.csv")

    paired = adaptive.pivot(index=["dataset", "seed"], columns="method", values="pcc").reset_index()
    paired["delta"] = paired["adaptive_fusion"] - paired["fixed_fusion"]
    adaptive_by_dataset = paired.groupby("dataset")["delta"].mean()
    stable_better = bool((adaptive_by_dataset > 0).all())

    endpoints = sensitivity[sensitivity["mask_ratio"].isin([sensitivity["mask_ratio"].min(), sensitivity["mask_ratio"].max()])]
    endpoint_mean = endpoints.groupby(["dataset", "method", "mask_ratio"], as_index=False)["pcc"].mean()
    endpoint_pivot = endpoint_mean.pivot(index=["dataset", "method"], columns="mask_ratio", values="pcc")
    low_ratio, high_ratio = endpoint_ratios(sensitivity["mask_ratio"].unique())
    endpoint_pivot["change"] = endpoint_pivot[high_ratio] - endpoint_pivot[low_ratio]

    slope_pivot = slopes.pivot(index="dataset", columns="method", values="pcc_slope")
    comparable = slope_pivot.dropna(subset=["mfmr_base", "original_mclrp"])
    slower = int((comparable["mfmr_base"].abs() < comparable["original_mclrp"].abs()).sum())

    branch_means = branches.groupby("dataset").mean(numeric_only=True)
    fusion_both = (
        (branch_means["fusion_delta_pcc_vs_imputer"] > 0)
        & (branch_means["fusion_delta_pcc_vs_ridge"] > 0)
    )
    qmeans = quartiles.groupby(["quartile", "method"])["pcc"].mean().unstack("method")
    qbenefit = qmeans["fixed_fusion"] - qmeans[["imputer_only", "ridge_only"]].max(axis=1)
    disagreement_pattern = bool(qbenefit.loc[qbenefit.index.max()] > qbenefit.loc[qbenefit.index.min()])

    lines = [
        "# Additional experiments summary",
        "",
        "All statements below are generated from the experiment CSV files.",
        "",
        f"1. Adaptive fusion was {'consistently better' if stable_better else 'not consistently better'} than fixed 0.5 across all evaluated datasets by mean seed-level PCC.",
        "2. Mean selected alpha by dataset: " + "; ".join(f"{row.dataset}={_fmt(row['mean'])}" for _, row in alpha.iterrows()) + ".",
        "3. Mean PCC change from 10% to 30% masking: " + "; ".join(
            f"{dataset}/{method}={_fmt(row['change'])}" for (dataset, method), row in endpoint_pivot.iterrows()
        ) + ".",
        f"4. MFMR had a smaller absolute PCC degradation slope than Original MCLRP in {slower}/{len(comparable)} comparable datasets.",
        "5. Mean imputer/ridge prediction Pearson correlation: " + "; ".join(f"{dataset}={_fmt(row.prediction_pearson)}" for dataset, row in branch_means.iterrows()) + ".",
        "6. Mean imputer/ridge error Pearson correlation: " + "; ".join(f"{dataset}={_fmt(row.error_pearson)}" for dataset, row in branch_means.iterrows()) + ".",
        f"7. Fixed fusion exceeded both individual branches in mean PCC in {int(fusion_both.sum())}/{len(fusion_both)} datasets.",
        f"8. Fusion benefit relative to the better branch was {'larger' if disagreement_pattern else 'not larger'} in the highest than the lowest disagreement quartile.",
        "",
    ]
    (args.results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
