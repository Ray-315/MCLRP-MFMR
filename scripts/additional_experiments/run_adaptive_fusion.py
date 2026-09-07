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

from MCLRP_MFMR.additional_experiments import (
    DEFAULT_ALPHA_CANDIDATES,
    load_existing_records,
    mae,
    mask_outer_responses,
    rmse,
    safe_pcc,
    select_adaptive_alpha,
    summarize_numeric_by_group,
)
from MCLRP_MFMR.getcrossMatrixs import getcrossMatrixs
from MCLRP_MFMR.t0_mfmr_protocol import (
    MFMRConfig,
    T0DatasetBundle,
    load_t0_dataset,
    predict_mfmr_t0_seed,
)


DEFAULT_DATASETS = ("CCLE", "ERKAUC30", "ERKIC50", "PI3KAUC", "PI3KIC50")
METHODS = ("imputer_only", "ridge_only", "fixed_fusion", "adaptive_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free nested-CV adaptive MFMR fusion")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/additional_experiments/adaptive_fusion")
    parser.add_argument("--prism-bundle-dir", type=Path)
    parser.add_argument("--prism-split", type=Path)
    parser.add_argument("--max-cell-lines", type=int)
    parser.add_argument("--max-drugs", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_prism(bundle_dir: Path) -> T0DatasetBundle:
    payload = np.load(bundle_dir / "bundle.npz", allow_pickle=True)
    return T0DatasetBundle(
        name="PRISM19Q4_independent",
        X=payload["X"].astype(np.float32),
        M=payload["M"].astype(np.float32),
        cell_labels=payload["cell_ids"].astype(object),
        drug_labels=payload["drug_labels"].astype(object),
        response_path=str(bundle_dir / "bundle.npz"),
        expression_path=str(bundle_dir / "bundle.npz"),
    )


def _subset(bundle: T0DatasetBundle, max_rows: int | None, max_cols: int | None) -> T0DatasetBundle:
    rows = np.arange(bundle.M.shape[0])[:max_rows]
    cols = np.arange(bundle.M.shape[1])[:max_cols]
    return T0DatasetBundle(
        name=bundle.name,
        X=bundle.X[rows],
        M=bundle.M[np.ix_(rows, cols)],
        cell_labels=bundle.cell_labels[rows],
        drug_labels=bundle.drug_labels[cols],
        response_path=bundle.response_path,
        expression_path=bundle.expression_path,
    )


def _fold_vectors(matrix: np.ndarray, folds: list[np.ndarray], *predictions: np.ndarray) -> tuple[np.ndarray, ...]:
    truth_parts: list[np.ndarray] = []
    prediction_parts: list[list[np.ndarray]] = [[] for _ in predictions]
    fold_parts: list[np.ndarray] = []
    for fold_index, fold in enumerate(folds):
        mask = fold != 0
        truth_parts.append(matrix[mask])
        fold_parts.append(np.full(int(mask.sum()), fold_index, dtype=np.int64))
        for target, prediction in zip(prediction_parts, predictions):
            target.append(prediction[mask])
    return (
        np.concatenate(truth_parts),
        *(np.concatenate(parts) for parts in prediction_parts),
        np.concatenate(fold_parts),
    )


def _outer_folds(bundle: T0DatasetBundle, seed: int, count: int, prism_split: Path | None) -> list[np.ndarray]:
    if bundle.name == "PRISM19Q4_independent":
        if prism_split is None:
            raise ValueError("--prism-split is required for PRISM")
        test_mask = np.load(prism_split)["test_mask"].astype(bool)
        if test_mask.shape != bundle.M.shape:
            raise ValueError("PRISM locked split shape does not match bundle")
        return [np.where(test_mask, bundle.M, 0.0).astype(np.float32)]
    return getcrossMatrixs(bundle.M, num_folds=count, rng=np.random.default_rng(int(seed)))


def _metric_row(dataset: str, seed: int, fold: int, method: str, truth: np.ndarray, pred: np.ndarray, **extra: float) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "fold": int(fold),
        "method": method,
        "pcc": safe_pcc(truth, pred),
        "rmse": rmse(truth, pred),
        "mae": mae(truth, pred),
        "n_test": int(len(truth)),
        **extra,
    }


def run_dataset(
    bundle: T0DatasetBundle,
    seeds: list[int],
    config: MFMRConfig,
    inner_fold_count: int,
    output_dir: Path,
    prism_split: Path | None,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    score_rows = load_existing_records(output_dir / "inner_alpha_scores.csv", enabled=resume)

    for seed in seeds:
        cache = prediction_dir / f"{bundle.name}__seed_{seed}.npz"
        outer_folds = _outer_folds(bundle, seed, config.num_folds, prism_split)
        test_mask_all = np.logical_or.reduce([fold != 0 for fold in outer_folds])
        if resume and cache.exists():
            saved = np.load(cache)
            imputer = saved["imputer"]
            ridge = saved["ridge"]
            fixed = saved["fixed"]
            adaptive = saved["adaptive"]
            selected_alphas = saved["selected_alphas"]
            inner_pccs = saved["inner_pccs"]
        else:
            fixed_config = replace(config, random_state=int(seed), weight_imp=0.5, weight_ridge=0.5)
            outer = predict_mfmr_t0_seed(bundle.name, bundle.X, bundle.M, outer_folds, "mfmr_base", fixed_config)
            imputer = outer.imputer.astype(np.float32)
            ridge = outer.ridge.astype(np.float32)
            fixed = (0.5 * imputer + 0.5 * ridge).astype(np.float32)
            adaptive = np.zeros_like(bundle.M, dtype=np.float32)
            selected_alphas = np.zeros(len(outer_folds), dtype=np.float64)
            inner_pccs = np.zeros(len(outer_folds), dtype=np.float64)

            for outer_index, outer_fold in enumerate(outer_folds):
                outer_test = outer_fold != 0
                outer_training_matrix = mask_outer_responses(bundle.M, outer_test)
                inner_folds = getcrossMatrixs(
                    outer_training_matrix,
                    num_folds=inner_fold_count,
                    rng=np.random.default_rng(1_000_003 * int(seed) + outer_index + 17),
                )
                inner = predict_mfmr_t0_seed(
                    bundle.name,
                    bundle.X,
                    outer_training_matrix,
                    inner_folds,
                    "mfmr_base",
                    fixed_config,
                )
                truth_inner, imp_inner, ridge_inner, fold_ids = _fold_vectors(
                    outer_training_matrix, inner_folds, inner.imputer, inner.ridge
                )
                alpha, scores = select_adaptive_alpha(
                    truth_inner,
                    imp_inner,
                    ridge_inner,
                    fold_ids=fold_ids,
                    candidates=DEFAULT_ALPHA_CANDIDATES,
                )
                selected_alphas[outer_index] = alpha
                inner_pccs[outer_index] = scores[alpha]["mean_pcc"]
                adaptive[outer_test] = (alpha * imputer[outer_test] + (1.0 - alpha) * ridge[outer_test]).astype(np.float32)
                for candidate, metrics in scores.items():
                    score_rows.append(
                        {
                            "dataset": bundle.name,
                            "seed": int(seed),
                            "fold": outer_index + 1,
                            "alpha": candidate,
                            **metrics,
                            "selected": bool(candidate == alpha),
                        }
                    )
            np.savez_compressed(
                cache,
                truth=bundle.M,
                test_mask=test_mask_all,
                imputer=imputer,
                ridge=ridge,
                fixed=fixed,
                adaptive=adaptive,
                selected_alphas=selected_alphas,
                inner_pccs=inner_pccs,
            )

        for outer_index, outer_fold in enumerate(outer_folds):
            mask = outer_fold != 0
            truth = bundle.M[mask]
            selected = float(selected_alphas[outer_index])
            inner_pcc = float(inner_pccs[outer_index])
            for method, prediction in (
                ("imputer_only", imputer),
                ("ridge_only", ridge),
                ("fixed_fusion", fixed),
                ("adaptive_fusion", adaptive),
            ):
                fold_rows.append(
                    _metric_row(
                        bundle.name,
                        seed,
                        outer_index + 1,
                        method,
                        truth,
                        prediction[mask],
                        selected_alpha=selected if method == "adaptive_fusion" else np.nan,
                        inner_pcc=inner_pcc if method == "adaptive_fusion" else np.nan,
                    )
                )
        truth_all = bundle.M[test_mask_all]
        for method, prediction in (
            ("imputer_only", imputer),
            ("ridge_only", ridge),
            ("fixed_fusion", fixed),
            ("adaptive_fusion", adaptive),
        ):
            seed_rows.append(_metric_row(bundle.name, seed, 0, method, truth_all, prediction[test_mask_all]))
        print(json.dumps({"dataset": bundle.name, "seed": seed, "status": "complete"}), flush=True)
    score_frame = pd.DataFrame(score_rows)
    if not score_frame.empty:
        score_frame = score_frame.drop_duplicates(["dataset", "seed", "fold", "alpha"], keep="last")
    return pd.DataFrame(fold_rows), pd.DataFrame(seed_rows), score_frame


def _save_figures(fold_df: pd.DataFrame, seed_df: pd.DataFrame, output_dir: Path) -> None:
    figure_data = fold_df[fold_df["method"] == "adaptive_fusion"][
        ["dataset", "seed", "fold", "selected_alpha", "inner_pcc"]
    ].copy()
    figure_data.to_csv(output_dir / "figure_data_alpha.csv", index=False)
    datasets = list(dict.fromkeys(fold_df["dataset"].tolist()))

    fig, axes = plt.subplots(1, len(datasets), figsize=(max(6, 2.4 * len(datasets)), 3.2), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        values = figure_data.loc[figure_data["dataset"] == dataset, "selected_alpha"].to_numpy()
        axis.hist(values, bins=np.linspace(-0.05, 1.05, 12), color="#4472C4", edgecolor="white")
        axis.set_title(dataset)
        axis.set_xlabel("Selected alpha")
    axes[0][0].set_ylabel("Outer folds")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"selected_alpha_distribution.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    paired = seed_df.pivot(index=["dataset", "seed"], columns="method", values="pcc").reset_index()
    paired["delta_pcc"] = paired["adaptive_fusion"] - paired["fixed_fusion"]
    paired.to_csv(output_dir / "figure_data_fixed_vs_adaptive.csv", index=False)
    summary = paired.groupby("dataset").agg(
        fixed_mean=("fixed_fusion", "mean"), adaptive_mean=("adaptive_fusion", "mean"), delta_mean=("delta_pcc", "mean"), delta_sd=("delta_pcc", "std")
    ).reset_index()

    x = np.arange(len(summary))
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(5, 1.2 * len(summary)), 3.6))
    axis.bar(x - width / 2, summary["fixed_mean"], width, label="Fixed 0.5")
    axis.bar(x + width / 2, summary["adaptive_mean"], width, label="Adaptive")
    axis.set_xticks(x, summary["dataset"], rotation=35, ha="right")
    axis.set_ylabel("PCC")
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fixed_vs_adaptive_pcc.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(5, 1.2 * len(summary)), 3.6))
    axis.bar(x, summary["delta_mean"], yerr=summary["delta_sd"].fillna(0), color="#70AD47", capsize=3)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, summary["dataset"], rotation=35, ha="right")
    axis.set_ylabel("Adaptive - fixed ΔPCC")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"adaptive_minus_fixed_delta_pcc.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = MFMRConfig(num_folds=args.outer_folds, seeds=tuple(args.seeds), weight_imp=0.5, weight_ridge=0.5)
    all_fold: list[pd.DataFrame] = []
    all_seed: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    for name in args.datasets:
        if name == "PRISM19Q4_independent":
            if args.prism_bundle_dir is None:
                raise ValueError("--prism-bundle-dir is required for PRISM")
            bundle = _load_prism(args.prism_bundle_dir)
        else:
            bundle = load_t0_dataset(name)
        bundle = _subset(bundle, args.max_cell_lines, args.max_drugs)
        fold_df, seed_df, score_df = run_dataset(
            bundle, args.seeds, config, args.inner_folds, args.output_dir, args.prism_split, args.resume
        )
        all_fold.append(fold_df)
        all_seed.append(seed_df)
        all_scores.append(score_df)

    fold_df = pd.concat(all_fold, ignore_index=True)
    seed_df = pd.concat(all_seed, ignore_index=True)
    score_df = pd.concat(all_scores, ignore_index=True) if any(not frame.empty for frame in all_scores) else pd.DataFrame()
    fold_df.to_csv(args.output_dir / "raw_fold_metrics.csv", index=False)
    seed_df.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    score_df.to_csv(args.output_dir / "inner_alpha_scores.csv", index=False)
    summarize_numeric_by_group(seed_df, ("dataset", "method"), ("pcc", "rmse", "mae")).to_csv(
        args.output_dir / "dataset_summary.csv", index=False
    )
    alpha_rows = fold_df[fold_df["method"] == "adaptive_fusion"].groupby("dataset")["selected_alpha"].agg(["mean", "std", "count"]).reset_index()
    half_share = fold_df[fold_df["method"] == "adaptive_fusion"].groupby("dataset")["selected_alpha"].apply(lambda values: float(np.mean(np.isclose(values, 0.5)))).rename("alpha_0_5_share").reset_index()
    alpha_rows.merge(half_share, on="dataset").to_csv(args.output_dir / "alpha_summary.csv", index=False)
    _save_figures(fold_df, seed_df, args.output_dir)
    config_payload = {
        "experiment": "adaptive_fusion",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "alpha_candidates": DEFAULT_ALPHA_CANDIDATES,
        "selection_metric": "mean inner-fold PCC",
        "tie_break": "closest to 0.5, then lower alpha",
        "mfmr_config": asdict(config),
        "leakage_control": "alpha selected only from inner OOF predictions within each outer-training partition",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        """# Adaptive fusion

This experiment compares the imputer branch, ridge branch, fixed 0.5 fusion, and nested-CV adaptive fusion under strict T0 random-entry interpolation. For every outer fold, alpha is selected from five-fold OOF predictions generated entirely inside the outer-training partition. The outer test responses are never used for selection. Seeds, loaders, preprocessing, branch models, and metrics are inherited from the primary pipeline. Run `python scripts/additional_experiments/run_adaptive_fusion.py`; outputs include raw fold/seed metrics, all inner-alpha scores, summaries, cached predictions, figure-data CSV files, and PNG/PDF figures.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
