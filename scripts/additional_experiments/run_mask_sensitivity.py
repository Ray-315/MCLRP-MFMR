from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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

from MCLRP_MFMR.additional_experiments import constrained_t0_mask, mae, rmse, safe_pcc, summarize_numeric_by_group
from MCLRP_MFMR.t0_mfmr_protocol import MFMRConfig, load_t0_dataset, predict_mfmr_t0_seed


DEFAULT_DATASETS = ("CCLE", "ERKAUC30", "ERKIC50", "PI3KAUC", "PI3KIC50")
DEFAULT_RATIOS = (0.10, 0.15, 0.20, 0.25, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T0 missing-ratio sensitivity experiment")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--methods", nargs="+", default=["original_mclrp", "mfmr_base"])
    parser.add_argument("--min-train-per-row", type=int, default=1)
    parser.add_argument("--min-train-per-drug", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/additional_experiments/mask_sensitivity")
    parser.add_argument("--max-cell-lines", type=int)
    parser.add_argument("--max-drugs", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _metric_row(dataset: str, ratio: float, seed: int, method: str, truth: np.ndarray, prediction: np.ndarray, n_train: int, n_test: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "mask_ratio": float(ratio),
        "seed": int(seed),
        "method": method,
        "pcc": safe_pcc(truth, prediction),
        "rmse": rmse(truth, prediction),
        "mae": mae(truth, prediction),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def _make_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    summary.to_csv(output_dir / "figure_data_metrics_by_ratio.csv", index=False)
    datasets = list(dict.fromkeys(summary["dataset"].tolist()))
    methods = list(dict.fromkeys(summary["method"].tolist()))
    colors = {method: color for method, color in zip(methods, ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"))}
    for metric in ("pcc", "rmse", "mae"):
        fig, axes = plt.subplots(1, len(datasets), figsize=(max(7, 2.5 * len(datasets)), 3.3), squeeze=False, sharey=False)
        for axis, dataset in zip(axes[0], datasets):
            subset = summary[(summary["dataset"] == dataset) & (summary["metric"] == metric)]
            for method in methods:
                line = subset[subset["method"] == method].sort_values("mask_ratio")
                if line.empty:
                    continue
                axis.errorbar(line["mask_ratio"] * 100, line["mean"], yerr=line["sd"], marker="o", capsize=2, label=method, color=colors[method])
            axis.set_title(dataset)
            axis.set_xlabel("Masked observed entries (%)")
        axes[0][0].set_ylabel(metric.upper())
        handles, labels = axes[0][-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"mask_sensitivity_{metric}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    mask_dir = args.output_dir / "masks"
    prediction_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    config = MFMRConfig(num_folds=1, seeds=tuple(args.seeds), min_train_per_drug=args.min_train_per_drug, weight_imp=0.5, weight_ridge=0.5)
    rows: list[dict[str, object]] = []

    for dataset_name in args.datasets:
        bundle = load_t0_dataset(dataset_name)
        if args.max_cell_lines is not None:
            bundle = replace(bundle, X=bundle.X[: args.max_cell_lines], M=bundle.M[: args.max_cell_lines], cell_labels=bundle.cell_labels[: args.max_cell_lines])
        if args.max_drugs is not None:
            bundle = replace(bundle, M=bundle.M[:, : args.max_drugs], drug_labels=bundle.drug_labels[: args.max_drugs])
        observed = bundle.M != 0
        for ratio in args.ratios:
            for seed in args.seeds:
                mask_path = mask_dir / f"{dataset_name}__ratio_{ratio:.2f}__seed_{seed}.npz"
                if args.resume and mask_path.exists():
                    test_mask = np.load(mask_path)["test_mask"].astype(bool)
                else:
                    test_mask = constrained_t0_mask(
                        bundle.M,
                        fraction=ratio,
                        seed=seed,
                        min_train_per_row=args.min_train_per_row,
                        min_train_per_col=args.min_train_per_drug,
                    )
                    np.savez_compressed(mask_path, test_mask=test_mask, train_mask=observed & ~test_mask)
                folds = [np.where(test_mask, bundle.M, 0.0).astype(np.float32)]
                current_config = replace(config, random_state=int(seed))
                for method in args.methods:
                    cache = prediction_dir / f"{dataset_name}__ratio_{ratio:.2f}__seed_{seed}__{method}.npz"
                    if args.resume and cache.exists():
                        prediction = np.load(cache)["prediction"].astype(np.float32)
                    else:
                        result = predict_mfmr_t0_seed(bundle.name, bundle.X, bundle.M, folds, method, current_config)
                        prediction = result.prediction.astype(np.float32)
                        np.savez_compressed(cache, prediction=prediction)
                    rows.append(
                        _metric_row(
                            dataset_name,
                            ratio,
                            seed,
                            method,
                            bundle.M[test_mask],
                            prediction[test_mask],
                            int((observed & ~test_mask).sum()),
                            int(test_mask.sum()),
                        )
                    )
                print(json.dumps({"dataset": dataset_name, "ratio": ratio, "seed": seed, "status": "complete"}), flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    summary = summarize_numeric_by_group(raw, ("dataset", "mask_ratio", "method"), ("pcc", "rmse", "mae"))
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    reference = raw[raw["mask_ratio"] == min(args.ratios)][["dataset", "seed", "method", "pcc", "rmse", "mae"]].rename(
        columns={"pcc": "pcc_reference", "rmse": "rmse_reference", "mae": "mae_reference"}
    )
    degradation = raw.merge(reference, on=["dataset", "seed", "method"], how="left")
    for metric in ("pcc", "rmse", "mae"):
        degradation[f"delta_{metric}"] = degradation[metric] - degradation[f"{metric}_reference"]
    degradation.to_csv(args.output_dir / "degradation_from_10pct.csv", index=False)

    slope_rows: list[dict[str, object]] = []
    for (dataset, method), group in raw.groupby(["dataset", "method"]):
        row: dict[str, object] = {"dataset": dataset, "method": method}
        for metric in ("pcc", "rmse", "mae"):
            row[f"{metric}_slope"] = float(np.polyfit(group["mask_ratio"], group[metric], 1)[0])
        slope_rows.append(row)
    pd.DataFrame(slope_rows).to_csv(args.output_dir / "degradation_slopes.csv", index=False)
    _make_plots(summary, args.output_dir)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": "mask_sensitivity",
                "datasets": args.datasets,
                "ratios": args.ratios,
                "seeds": args.seeds,
                "methods": args.methods,
                "mask_definition": "fraction of originally observed nonzero response entries",
                "min_train_per_row": args.min_train_per_row,
                "min_train_per_drug": args.min_train_per_drug,
                "mfmr_config": asdict(config),
                "leakage_control": "one constrained mask is shared by every method for each dataset-ratio-seed unit",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        """# Missing-ratio sensitivity

This experiment masks 10%, 15%, 20%, 25%, or 30% of the originally observed responses under T0 row/drug context constraints. Original MCLRP and fixed-fusion MFMR use exactly the same mask for every dataset-ratio-seed unit. All preprocessing and fitting are repeated from the visible training responses only. Run `python scripts/additional_experiments/run_mask_sensitivity.py`. Outputs contain masks, cached predictions, raw seed metrics, mean/SD/95% CI summaries, degradation relative to 10%, fitted slopes, figure-data CSV, and PNG/PDF figures.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
