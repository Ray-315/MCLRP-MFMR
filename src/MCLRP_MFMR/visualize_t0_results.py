from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MCLRP_MFMR.paths import RESULTS_DIR


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _bar_by_method(summary: pd.DataFrame, figures_dir: Path) -> list[str]:
    if summary.empty:
        return []
    outputs: list[str] = []
    metrics = [
        ("overall_pcc_mean", "Overall PCC", "method_overall_pcc.png"),
        ("rmse_mean", "RMSE", "method_rmse.png"),
        ("mae_mean", "MAE", "method_mae.png"),
    ]
    for metric, ylabel, filename in metrics:
        if metric not in summary.columns:
            continue
        pivot = summary.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(pivot.index) + 3), 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("Method")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by method")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Dataset", fontsize=8)
        ax.tick_params(axis="x", rotation=35)
        outputs.append(_save(fig, figures_dir / filename))
    return outputs


def _boxplot_metric(frame: pd.DataFrame, metric: str, title: str, filename: str, figures_dir: Path) -> list[str]:
    if frame.empty or metric not in frame.columns:
        return []
    methods = sorted(frame["method"].dropna().unique().tolist())
    if not methods:
        return []
    values = [frame.loc[frame["method"] == method, metric].dropna().to_numpy(dtype=float) for method in methods]
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(methods) + 3), 4.5))
    try:
        ax.boxplot(values, tick_labels=methods, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=methods, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    return [_save(fig, figures_dir / filename)]


def _per_drug_heatmap(per_drug: pd.DataFrame, figures_dir: Path) -> list[str]:
    if per_drug.empty or "pcc" not in per_drug.columns:
        return []
    grouped = per_drug.groupby(["method", "drug_idx"], sort=True)["pcc"].mean().reset_index()
    if grouped.empty:
        return []
    pivot = grouped.pivot(index="method", columns="drug_idx", values="pcc").sort_index()
    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * pivot.shape[1] + 3), max(3.5, 0.35 * pivot.shape[0] + 2)))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Mean per-drug PCC")
    ax.set_xlabel("Drug index")
    ax.set_ylabel("Method")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, label="PCC")
    return [_save(fig, figures_dir / "per_drug_pcc_heatmap.png")]


def _ablation_plot(ablation: pd.DataFrame, figures_dir: Path) -> list[str]:
    if ablation.empty or "delta_pcc_mean" not in ablation.columns:
        return []
    labels = ablation["dataset"].astype(str) + " / " + ablation["method"].astype(str)
    values = ablation["delta_pcc_mean"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels) + 3), 4.5))
    colors = np.where(values >= 0, "#2E7D32", "#C62828")
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Ablation delta vs reference")
    ax.set_ylabel("Delta overall PCC")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return [_save(fig, figures_dir / "ablation_delta_pcc.png")]


def _write_index(figures: list[str], figures_dir: Path, results_dir: Path) -> str:
    rows = []
    for figure in figures:
        rel = Path(figure).relative_to(figures_dir)
        rows.append(f'<li><a href="{rel.as_posix()}">{rel.as_posix()}</a><br><img src="{rel.as_posix()}" style="max-width: 960px; width: 100%;"></li>')
    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset=\"utf-8\">",
            "<title>MCLRP-MFMR T0 Figures</title>",
            "<h1>MCLRP-MFMR T0 Figures</h1>",
            f"<p>Source results: {results_dir}</p>",
            "<ul>",
            *rows,
            "</ul>",
        ]
    )
    index_path = figures_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return str(index_path)


def visualize_t0_results(results_dir: Path | str | None = None, output_dir: Path | str | None = None) -> dict[str, Any]:
    results_root = Path(results_dir) if results_dir is not None else RESULTS_DIR / "t0_mfmr"
    figures_dir = Path(output_dir) if output_dir is not None else results_root / "figures"
    summary = _read_csv(results_root / "method_summary.csv")
    seed = _read_csv(results_root / "seed_summary.csv")
    fold = _read_csv(results_root / "fold_summary.csv")
    per_drug = _read_csv(results_root / "per_drug_pcc.csv")
    ablation = _read_csv(results_root / "ablation_summary.csv")

    figures: list[str] = []
    figures.extend(_bar_by_method(summary, figures_dir))
    figures.extend(_boxplot_metric(seed, "overall_pcc", "Seed-level overall PCC", "seed_overall_pcc_boxplot.png", figures_dir))
    figures.extend(_boxplot_metric(fold, "pcc", "Fold-level PCC", "fold_pcc_boxplot.png", figures_dir))
    figures.extend(_per_drug_heatmap(per_drug, figures_dir))
    figures.extend(_ablation_plot(ablation, figures_dir))
    index_path = _write_index(figures, figures_dir, results_root)
    return {"results_dir": str(results_root), "figures_dir": str(figures_dir), "figures": figures, "index": index_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures for strict T0 MCLRP-MFMR benchmark results.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "t0_mfmr")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = visualize_t0_results(results_dir=args.results_dir, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
