from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from MCLRP_MFMR.additional_experiments import branch_complementarity_metrics, resolve_cached_truth, summarize_numeric_by_group
from MCLRP_MFMR.t0_mfmr_protocol import load_t0_dataset


NAME_PATTERN = re.compile(r"^(?P<dataset>.+)__seed_(?P<seed>\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MFMR imputer/ridge complementarity from cached OOF predictions")
    parser.add_argument("--prediction-dir", type=Path, default=PROJECT_ROOT / "results/additional_experiments/adaptive_fusion/predictions")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/additional_experiments/branch_complementarity")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--prism-bundle-dir", type=Path)
    return parser.parse_args()


def _truth_matrix(dataset: str, prism_bundle_dir: Path | None) -> np.ndarray:
    if dataset == "PRISM19Q4_independent":
        if prism_bundle_dir is None:
            raise ValueError("--prism-bundle-dir is required when PRISM predictions are present")
        return np.load(prism_bundle_dir / "bundle.npz")["M"].astype(np.float32)
    return load_t0_dataset(dataset).M


def _figures(raw: pd.DataFrame, quartiles: pd.DataFrame, output_dir: Path) -> None:
    raw.to_csv(output_dir / "figure_data_branch_metrics.csv", index=False)
    quartiles.to_csv(output_dir / "figure_data_disagreement_quartiles.csv", index=False)
    datasets = list(dict.fromkeys(raw["dataset"].tolist()))
    aggregate = raw.groupby("dataset").mean(numeric_only=True).reindex(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(aggregate))
    axes[0].bar(x - 0.18, aggregate["prediction_pearson"], 0.36, label="Prediction")
    axes[0].bar(x + 0.18, aggregate["error_pearson"], 0.36, label="Error")
    axes[0].set_xticks(x, aggregate.index, rotation=35, ha="right")
    axes[0].set_ylabel("Pearson correlation")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.25, aggregate["imputer_better_share"], 0.25, label="Imputer better")
    axes[1].bar(x, aggregate["ridge_better_share"], 0.25, label="Ridge better")
    axes[1].bar(x + 0.25, aggregate["tie_share"], 0.25, label="Tie")
    axes[1].set_xticks(x, aggregate.index, rotation=35, ha="right")
    axes[1].set_ylabel("Held-out entry share")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"branch_correlations_and_shares.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    qsummary = quartiles.groupby(["dataset", "quartile", "method"], as_index=False)[["pcc", "rmse", "mae", "mean_disagreement"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for axis, metric in zip(axes, ("pcc", "rmse", "mae")):
        for method, group in qsummary.groupby("method"):
            line = group.groupby("quartile", as_index=False)[metric].mean()
            axis.plot(line["quartile"], line[metric], marker="o", label=method)
        axis.set_xlabel("Disagreement quartile")
        axis.set_ylabel(metric.upper())
        axis.set_xticks([1, 2, 3, 4])
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"disagreement_quartile_performance.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    quartile_frames: list[pd.DataFrame] = []
    caches = sorted(args.prediction_dir.glob("*__seed_*.npz"))
    if not caches:
        raise FileNotFoundError(f"No adaptive-fusion prediction caches in {args.prediction_dir}")

    truth_cache: dict[str, np.ndarray] = {}
    for cache in caches:
        match = NAME_PATTERN.match(cache.name)
        if match is None:
            continue
        dataset = match.group("dataset")
        seed = int(match.group("seed"))
        if dataset not in truth_cache:
            truth_cache[dataset] = _truth_matrix(dataset, args.prism_bundle_dir)
        payload = np.load(cache)
        matrix = resolve_cached_truth(payload, truth_cache[dataset])
        test_mask = payload["test_mask"].astype(bool)
        if test_mask.shape != matrix.shape:
            raise ValueError(f"Prediction/matrix shape mismatch for {cache.name}")
        overall, quartiles = branch_complementarity_metrics(
            matrix[test_mask],
            payload["imputer"][test_mask],
            payload["ridge"][test_mask],
            payload["fixed"][test_mask],
            tolerance=args.tolerance,
        )
        rows.append({"dataset": dataset, "seed": seed, **overall})
        quartiles.insert(0, "seed", seed)
        quartiles.insert(0, "dataset", dataset)
        quartile_frames.append(quartiles)

    raw = pd.DataFrame(rows)
    quartile_raw = pd.concat(quartile_frames, ignore_index=True)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    quartile_raw.to_csv(args.output_dir / "raw_quartile_metrics.csv", index=False)
    metric_columns = [column for column in raw.columns if column not in {"dataset", "seed", "n_test"}]
    summarize_numeric_by_group(raw, ("dataset",), metric_columns).to_csv(args.output_dir / "dataset_summary.csv", index=False)
    _figures(raw, quartile_raw, args.output_dir)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": "branch_complementarity",
                "prediction_dir": str(args.prediction_dir.resolve()),
                "tolerance": args.tolerance,
                "prediction_source": "cached held-out arrays from adaptive_fusion; no model retraining",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        f"""# Imputer/ridge complementarity

Purpose: test whether the MFMR imputer and ridge branches contain complementary held-out prediction information.

The analysis reads the exact aligned truth, T0 test mask, imputer, ridge, and fixed-fusion arrays cached by adaptive fusion; no model is retrained. It covers the five main datasets and PRISM when present, using seeds 0-9. An entry is tied when the absolute-error difference is below `{args.tolerance:g}`. Prediction/error Pearson and Spearman correlations, branch winner shares, fusion gains, and disagreement-quartile PCC/RMSE/MAE are computed per seed. `dataset_summary.csv` reports mean, SD, and 95% t intervals.

Run `python scripts/additional_experiments/analyze_branch_complementarity.py` after adaptive fusion. `raw_seed_metrics.csv` and `raw_quartile_metrics.csv` contain analysis units; `dataset_summary.csv` contains aggregate statistics; `figure_data_*.csv` backs the PNG/PDF figures; `config.json` records the prediction source and tolerance.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
