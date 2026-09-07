from __future__ import annotations

import argparse
import json
import shutil
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

from MCLRP_MFMR.additional_experiments import summarize_numeric_by_group
from scripts.additional_experiments.run_adaptive_fusion import _save_figures as save_adaptive_figures
from scripts.additional_experiments.run_mask_sensitivity import _make_plots as save_sensitivity_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-dataset additional-experiment shards")
    parser.add_argument("experiment", choices=("adaptive_fusion", "mask_sensitivity"))
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _frames(shards: list[Path], name: str) -> pd.DataFrame:
    paths = [shard / name for shard in shards if (shard / name).exists()]
    if not paths:
        raise FileNotFoundError(f"No {name} files found under shards")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _copy_tree_files(shards: list[Path], subdir: str, output_dir: Path) -> None:
    target = output_dir / subdir
    target.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        source = shard / subdir
        if not source.exists():
            continue
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)


def merge_adaptive(shards: list[Path], output_dir: Path) -> None:
    fold = _frames(shards, "raw_fold_metrics.csv").drop_duplicates(["dataset", "seed", "fold", "method"])
    seed = _frames(shards, "raw_seed_metrics.csv").drop_duplicates(["dataset", "seed", "method"])
    scores = _frames(shards, "inner_alpha_scores.csv").drop_duplicates(["dataset", "seed", "fold", "alpha"])
    fold.to_csv(output_dir / "raw_fold_metrics.csv", index=False)
    seed.to_csv(output_dir / "raw_seed_metrics.csv", index=False)
    scores.to_csv(output_dir / "inner_alpha_scores.csv", index=False)
    summarize_numeric_by_group(seed, ("dataset", "method"), ("pcc", "rmse", "mae")).to_csv(output_dir / "dataset_summary.csv", index=False)
    selected = fold[fold["method"] == "adaptive_fusion"]
    alpha = selected.groupby("dataset")["selected_alpha"].agg(["mean", "std", "count"]).reset_index()
    share = selected.groupby("dataset")["selected_alpha"].apply(lambda values: float(np.mean(np.isclose(values, 0.5)))).rename("alpha_0_5_share").reset_index()
    alpha.merge(share, on="dataset").to_csv(output_dir / "alpha_summary.csv", index=False)
    _copy_tree_files(shards, "predictions", output_dir)
    save_adaptive_figures(fold, seed, output_dir)


def merge_sensitivity(shards: list[Path], output_dir: Path) -> None:
    raw = _frames(shards, "raw_seed_metrics.csv").drop_duplicates(["dataset", "mask_ratio", "seed", "method"])
    raw.to_csv(output_dir / "raw_seed_metrics.csv", index=False)
    summary = summarize_numeric_by_group(raw, ("dataset", "mask_ratio", "method"), ("pcc", "rmse", "mae"))
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    low = float(raw["mask_ratio"].min())
    reference = raw[raw["mask_ratio"] == low][["dataset", "seed", "method", "pcc", "rmse", "mae"]].rename(columns={metric: f"{metric}_reference" for metric in ("pcc", "rmse", "mae")})
    degradation = raw.merge(reference, on=["dataset", "seed", "method"], how="left")
    for metric in ("pcc", "rmse", "mae"):
        degradation[f"delta_{metric}"] = degradation[metric] - degradation[f"{metric}_reference"]
    degradation.to_csv(output_dir / "degradation_from_10pct.csv", index=False)
    slope_rows: list[dict[str, object]] = []
    for (dataset, method), group in raw.groupby(["dataset", "method"]):
        row: dict[str, object] = {"dataset": dataset, "method": method}
        for metric in ("pcc", "rmse", "mae"):
            row[f"{metric}_slope"] = float(np.polyfit(group["mask_ratio"], group[metric], 1)[0])
        slope_rows.append(row)
    pd.DataFrame(slope_rows).to_csv(output_dir / "degradation_slopes.csv", index=False)
    _copy_tree_files(shards, "predictions", output_dir)
    _copy_tree_files(shards, "masks", output_dir)
    save_sensitivity_figures(summary, output_dir)


def main() -> None:
    args = parse_args()
    shards = sorted(path for path in args.shard_dir.iterdir() if path.is_dir())
    if not shards:
        raise FileNotFoundError(f"No shard directories in {args.shard_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment == "adaptive_fusion":
        merge_adaptive(shards, args.output_dir)
    else:
        merge_sensitivity(shards, args.output_dir)
    configs = [json.loads((shard / "config.json").read_text(encoding="utf-8")) for shard in shards]
    (args.output_dir / "config.json").write_text(json.dumps({"experiment": args.experiment, "shards": configs}, indent=2), encoding="utf-8")
    if args.experiment == "adaptive_fusion":
        readme_text = """# Adaptive fusion

Purpose: compare imputer-only, ridge-only, fixed alpha=0.5 fusion, and fold-adaptive fusion in T0 random-entry interpolation.

Datasets: CCLE, four legacy CGP tasks, and PRISM 19Q4 when its locked bundle is available. Seeds are 0-9. Main tasks use 10 outer folds and five inner folds; PRISM retains its single locked outer test and uses five inner folds within the locked training portion. Candidate alpha values are 0.0-1.0 in increments of 0.1. All response statistics, gene selection, scaling, SVD, imputation, and regression are refit from the relevant training partition. Outer responses never select alpha. Ties are resolved toward 0.5.

Run `pwsh scripts/additional_experiments/run_all_additional_experiments.ps1` for the complete workflow. `raw_fold_metrics.csv` and `raw_seed_metrics.csv` contain performance units; `inner_alpha_scores.csv` contains every candidate score; `dataset_summary.csv` and `alpha_summary.csv` contain aggregate statistics; `predictions/` contains aligned truth, masks, and branch/fusion arrays; `figure_data_*.csv` back the PNG/PDF figures; `config.json` records all frozen settings.
"""
    else:
        readme_text = """# Missing-ratio sensitivity

Purpose: quantify T0 interpolation performance as 10%, 15%, 20%, 25%, or 30% of originally observed response entries are hidden.

Datasets: CCLE and four legacy CGP tasks. Seeds are 0-9. Original MCLRP and fixed-fusion MFMR share the exact constrained mask for every dataset-ratio-seed unit. Each retained training row has at least one response and each retained drug column has at least five responses. All response-dependent preprocessing and fitting use visible training entries only.

Run `pwsh scripts/additional_experiments/run_all_additional_experiments.ps1` for the complete workflow. `raw_seed_metrics.csv` contains PCC/RMSE/MAE and train/test counts; `dataset_summary.csv` contains mean/SD/95% CI; `degradation_from_10pct.csv` contains paired changes from 10%; `degradation_slopes.csv` contains fitted PCC/RMSE/MAE slopes; `masks/` and `predictions/` retain exact reproducibility artifacts; `figure_data_metrics_by_ratio.csv` backs the PNG/PDF figures; `config.json` records settings.
"""
    (args.output_dir / "README.md").write_text(readme_text, encoding="utf-8")


if __name__ == "__main__":
    main()
