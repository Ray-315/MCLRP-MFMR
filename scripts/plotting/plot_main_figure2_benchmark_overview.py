from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root from plotting script")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import RESULTS_DIR
from core.standardized_dataset_loaders import load_ccle_standardized_bundle
from scripts.benchmarks.paper.cgp.run_cgp_benchmark import load_dataset as load_cgp_dataset
from scripts.benchmarks.paper.gdsc.run_gdsc_benchmark import (
    BenchmarkConfig as GDSCBenchmarkConfig,
    load_or_prepare_dataset,
)


FIG_DPI = 600
FONT_FAMILY = "DejaVu Sans"
AXIS_LABEL_SIZE = 10.5
TICK_LABEL_SIZE = 8.8
LEGEND_SIZE = 8.6
FRAME_COLOR = "#334155"
GRID_COLOR = "#E6EDF4"
LIGHT_GRAY = "#CBD5E1"
TEXT_DARK = "#1F2937"

NATURE_BLUE = "#4A7CFF"
NATURE_ORANGE = "#E8923F"
SKY_BLUE = "#6CB4EE"
PALE_ORANGE = "#F5C78E"
CELL_DARK = "#2D3E50"
MFMR_PURPLE = "#7B61FF"

MCLRP_COLOR = NATURE_ORANGE
MFMR_COLOR = NATURE_BLUE
ARROW_BLUE = NATURE_BLUE
CIRCLE_ORANGE = NATURE_ORANGE
IDENTITY_LINE_COLOR = "#A5C8E1"
SCATTER_COLOR = "#3D6DFF"
DATASET_ROOT_COLORS = {
    "CCLE": "#8E6BBE",
    "GDSC": "#4A7CFF",
    "CGP": "#E8923F",
}
DATASET_ROOT_DISPLAY = {
    "CCLE": "CCLE",
    "GDSC": "GDSC",
    "CGP": "CGP",
}
UPLIFT_TASK_COLORS = {
    "GDSC-ERK_AUC": NATURE_BLUE,
    "GDSC-ERK_IC50": SKY_BLUE,
    "GDSC-PI3K_AUC": NATURE_BLUE,
    "GDSC-PI3K_IC50": SKY_BLUE,
}
UPLIFT_TASK_DISPLAY = {
    "GDSC-ERK_AUC": "GDSC ERK AUC",
    "GDSC-ERK_IC50": "GDSC ERK IC50",
    "GDSC-PI3K_AUC": "GDSC PI3K AUC",
    "GDSC-PI3K_IC50": "GDSC PI3K IC50",
}

SUBTASK_ORDER = [
    "CCLE",
    "GDSC-ERK_AUC",
    "GDSC-ERK_IC50",
    "GDSC-PI3K_AUC",
    "GDSC-PI3K_IC50",
    "CGP-ERK_AUC",
    "CGP-ERK_IC50",
    "CGP-PI3K_AUC",
    "CGP-PI3K_IC50",
]
SUBTASK_SHORT_LABELS = {
    "CCLE": "CCLE",
    "GDSC-ERK_AUC": "GDSC\nERK AUC",
    "GDSC-ERK_IC50": "GDSC\nERK IC50",
    "GDSC-PI3K_AUC": "GDSC\nPI3K AUC",
    "GDSC-PI3K_IC50": "GDSC\nPI3K IC50",
    "CGP-ERK_AUC": "CGP\nERK AUC",
    "CGP-ERK_IC50": "CGP\nERK IC50",
    "CGP-PI3K_AUC": "CGP\nPI3K AUC",
    "CGP-PI3K_IC50": "CGP\nPI3K IC50",
}
ABLATION_SHARED_VARIANT_NAME = "A5_MFMR_Full_mean_prediction.npz"
ABLATION_CGP_VARIANT_NAME = "C6_CGP_Full_mean_prediction.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render composite Figure 2 benchmark overview.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR), help="Project results root.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "plotting" / "outputs" / "main_figures" / "figure2_benchmark_overview"),
        help="Output directory for the composite figure and panel assets.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Clear the target output directory before rendering.")
    return parser.parse_args()


def set_publication_style() -> None:
    sns.set_theme(style="white")
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": AXIS_LABEL_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": FRAME_COLOR,
            "axes.labelcolor": TEXT_DARK,
            "xtick.color": TEXT_DARK,
            "ytick.color": TEXT_DARK,
            "text.color": TEXT_DARK,
            "axes.linewidth": 1.1,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    if overwrite and path.exists():
        for child in path.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(path.glob("*"), reverse=True):
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)
    (path / "panels").mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in (".png", ".pdf", ".svg"):
        out_path = output_base.with_suffix(suffix)
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def frame_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.1)
    ax.tick_params(colors=TEXT_DARK, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.75, alpha=0.95)
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-10, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=15.5,
        fontweight="normal",
        fontfamily="DejaVu Serif",
        color="#111111",
        zorder=20,
        clip_on=False,
    )


def compact_label(label: str, width: int = 16) -> str:
    return textwrap.fill(str(label), width=width, break_long_words=False, break_on_hyphens=False)


def load_sources(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_root = results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv"
    task_summary = pd.read_csv(require_file(benchmark_root / "GLOBAL" / "dataset_task_model_summary.csv"))
    uplift = pd.read_csv(require_file(benchmark_root / "UPLIFT_MFMR_vs_MCLRP" / "all_task_per_drug_uplift.csv"))
    return task_summary, uplift


def prepare_top_row(task_df: pd.DataFrame) -> pd.DataFrame:
    work = task_df.loc[task_df["model"].isin(["MCLRP", "MCLRP_MFMR"])].copy()
    work["dataset_label"] = pd.Categorical(work["dataset_label"], categories=SUBTASK_ORDER, ordered=True)
    work = work.sort_values(["dataset_label", "model"]).reset_index(drop=True)
    work["mean_pcc"] = numeric_series(work["mean_pcc"])
    work["mean_scc"] = numeric_series(work["mean_scc"])
    return work


def prepare_uplift(uplift_df: pd.DataFrame) -> pd.DataFrame:
    work = uplift_df.copy()
    for col in ("pcc_MCLRP", "pcc_MCLRP_MFMR", "delta_pcc", "scc_MCLRP", "scc_MCLRP_MFMR", "delta_scc", "n_obs"):
        if col in work.columns:
            work[col] = numeric_series(work[col])
    return work.rename(columns={"group": "dataset_root"})


def select_top10_uplift(work: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in {"pcc", "scc"}:
        raise ValueError(f"Unsupported metric: {metric}")
    cols = [
        "dataset_root",
        "dataset",
        "dataset_label",
        "drug",
        "drug_idx",
        "n_obs",
        f"{metric}_MCLRP",
        f"{metric}_MCLRP_MFMR",
        f"delta_{metric}",
    ]
    top = work.loc[:, cols].sort_values(f"delta_{metric}", ascending=False).head(10).copy()
    top["plot_label"] = [compact_label(drug, width=14) for drug in top["drug"]]
    return top.reset_index(drop=True)


def select_case_studies(work: pd.DataFrame) -> pd.DataFrame:
    out = (
        work.sort_values("pcc_MCLRP_MFMR", ascending=False)
        .drop_duplicates(subset=["drug"], keep="first")
        .head(4)
        .copy()
        .reset_index(drop=True)
    )
    if out.empty:
        raise RuntimeError("No rows available for case-study scatter plots")
    out["plot_label"] = [compact_label(drug, width=18) for drug in out["drug"]]
    return out


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    return safe_corr(rx, ry)


def safe_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def load_case_prediction(results_root: Path, dataset_root: str, dataset_name: str, drug_name: str) -> dict[str, object]:
    if dataset_root == "CCLE":
        ablation_path = results_root / "latest" / "ablation_mclrp_mfmr" / "shared_main" / "CCLE" / "predictions" / ABLATION_SHARED_VARIANT_NAME
        pred = np.load(require_file(ablation_path))["prediction"].astype(np.float32)
        bundle = load_ccle_standardized_bundle()
        true_mat = bundle.M.astype(np.float32)
        labels = [str(x) for x in bundle.drug_labels.tolist()]
    elif dataset_root == "GDSC":
        ablation_path = results_root / "latest" / "ablation_mclrp_mfmr" / "shared_main" / dataset_name / "predictions" / ABLATION_SHARED_VARIANT_NAME
        pred = np.load(require_file(ablation_path))["prediction"].astype(np.float32)
        bundle = load_or_prepare_dataset(dataset_name, GDSCBenchmarkConfig())
        true_mat = bundle.M.astype(np.float32)
        labels = [str(x) for x in bundle.drug_labels.tolist()]
    elif dataset_root == "CGP":
        ablation_path = results_root / "latest" / "ablation_mclrp_mfmr" / "cgp_main" / dataset_name / "predictions" / ABLATION_CGP_VARIANT_NAME
        pred = np.load(require_file(ablation_path))["prediction"].astype(np.float32)
        bundle = load_cgp_dataset(dataset_name)
        true_mat = bundle.M.astype(np.float32)
        labels = [str(x) for x in bundle.drug_labels.tolist()]
    else:
        raise ValueError(f"Unsupported dataset root: {dataset_root}")

    if drug_name not in labels:
        raise KeyError(f"{drug_name} not found in {dataset_root}::{dataset_name}")
    j = labels.index(drug_name)
    true_values = true_mat[:, j].astype(np.float32)
    pred_values = pred[:, j].astype(np.float32)
    mask = true_values != 0
    x = true_values[mask]
    y = pred_values[mask]
    if x.size == 0:
        raise RuntimeError(f"No observed entries for {dataset_name}::{drug_name}")
    return {
        "dataset_name": dataset_name,
        "drug": drug_name,
        "x": x,
        "y": y,
        "n_obs": int(mask.sum()),
        "pcc": safe_corr(x, y),
        "scc": safe_spearman(x, y),
        "r2": safe_r2(x, y),
        "prediction_source": str(ablation_path),
    }


def draw_grouped_metric_panel(ax: plt.Axes, work: pd.DataFrame, metric: str) -> None:
    pivot = (
        work.pivot(index="dataset_label", columns="model", values=metric)
        .reindex(SUBTASK_ORDER)
        .loc[:, ["MCLRP", "MCLRP_MFMR"]]
    )
    xpos = np.arange(len(SUBTASK_ORDER), dtype=float)
    width = 0.34
    ax.bar(xpos - width / 2, pivot["MCLRP"].to_numpy(dtype=float), width=width, color=MCLRP_COLOR, edgecolor="white", linewidth=0.9, label="MCLRP")
    ax.bar(xpos + width / 2, pivot["MCLRP_MFMR"].to_numpy(dtype=float), width=width, color=MFMR_COLOR, edgecolor="white", linewidth=0.9, label="MCLRP-MFMR")
    values = pivot.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    lower = max(0.20, float(np.floor((values.min() - 0.04) * 10) / 10))
    upper = min(1.02, float(np.ceil((values.max() + 0.04) * 20) / 20))
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Mean PCC" if metric == "mean_pcc" else "Mean SCC")
    ax.set_xticks(xpos)
    ax.set_xticklabels([SUBTASK_SHORT_LABELS[x] for x in SUBTASK_ORDER], rotation=0, ha="center")
    frame_axis(ax, grid_axis="y")
    legend = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=LIGHT_GRAY, fancybox=False, borderpad=0.35, handlelength=1.6)
    legend.get_frame().set_linewidth(0.9)


def draw_uplift_dumbbell(ax: plt.Axes, top10: pd.DataFrame) -> None:
    xpos = np.arange(len(top10), dtype=float)
    base = top10["pcc_MCLRP"].to_numpy(dtype=float)
    lift = top10["pcc_MCLRP_MFMR"].to_numpy(dtype=float)
    task_labels = top10["dataset_label"].tolist()
    present_tasks = [key for key in ("GDSC-PI3K_AUC", "GDSC-PI3K_IC50") if key in set(task_labels)]

    for x, y0, y1, delta, task_label in zip(xpos, base, lift, top10["delta_pcc"].to_numpy(dtype=float), task_labels):
        dataset_color = UPLIFT_TASK_COLORS.get(str(task_label), ARROW_BLUE)
        arrow = FancyArrowPatch(
            (x, y0),
            (x, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=2.0,
            color=dataset_color,
            alpha=0.95,
            shrinkA=2,
            shrinkB=2,
            clip_on=False,
        )
        ax.add_patch(arrow)
        ax.scatter([x], [y0], s=56, facecolor="white", edgecolor=dataset_color, linewidth=1.6, zorder=3)
        ax.scatter([x], [y1], s=66, facecolor=dataset_color, edgecolor="white", linewidth=0.9, zorder=4)
        ax.text(x, y1 + 0.018, f"+{delta:.2f}", ha="center", va="bottom", fontsize=9.6, color="black", clip_on=False)

    lower = max(0.00, float(np.floor((base.min() - 0.08) * 20) / 20))
    upper = min(1.08, float(np.ceil((lift.max() + 0.09) * 20) / 20))
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.75, len(top10) - 0.25)
    ax.set_ylabel("PCC")
    ax.set_xticks(xpos)
    ax.set_xticklabels(top10["plot_label"].tolist(), rotation=28, ha="right")
    frame_axis(ax, grid_axis="y")

    handles = [
        Line2D([0], [0], marker="o", markersize=6.8, linestyle="", markerfacecolor="white", markeredgecolor=ARROW_BLUE, label="MCLRP"),
    ]
    if len(present_tasks) >= 2:
        handles.extend(
            [
                Line2D([0, 1], [0, 0], color=UPLIFT_TASK_COLORS[present_tasks[0]], linewidth=2.2, label=UPLIFT_TASK_DISPLAY[present_tasks[0]]),
                Line2D([0], [0], marker="o", markersize=7.2, linestyle="", markerfacecolor=ARROW_BLUE, markeredgecolor="white", label="MCLRP-MFMR"),
                Line2D([0, 1], [0, 0], color=UPLIFT_TASK_COLORS[present_tasks[1]], linewidth=2.2, label=UPLIFT_TASK_DISPLAY[present_tasks[1]]),
            ]
        )
    else:
        handles.append(Line2D([0], [0], marker="o", markersize=7.2, linestyle="", markerfacecolor=ARROW_BLUE, markeredgecolor="white", label="MCLRP-MFMR"))
        handles.extend(
            [
                Line2D([0, 1], [0, 0], color=UPLIFT_TASK_COLORS[key], linewidth=2.2, label=UPLIFT_TASK_DISPLAY[key])
                for key in present_tasks
            ]
        )
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRAY,
        fancybox=False,
        borderpad=0.30,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.9)


def draw_case_scatter(ax: plt.Axes, case_meta: pd.Series, case_data: dict[str, object]) -> None:
    x = np.asarray(case_data["x"], dtype=float)
    y = np.asarray(case_data["y"], dtype=float)
    ax.scatter(x, y, s=18, c=SCATTER_COLOR, alpha=0.82, edgecolors="white", linewidths=0.28)

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    margin = 0.06 * (hi - lo + 1e-6)
    lo -= margin
    hi += margin
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=IDENTITY_LINE_COLOR, linewidth=1.1)
    if np.nanstd(x) > 0 and np.nanstd(y) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        xx = np.linspace(lo, hi, 200)
        ax.plot(xx, slope * xx + intercept, color=NATURE_ORANGE, linewidth=1.7)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    frame_axis(ax, grid_axis="both")

    task_text = str(case_meta["dataset_label"]).replace("GDSC-", "GDSC ").replace("CGP-", "CGP ")
    title_box = f"{task_text}\n{case_meta['plot_label']}"
    stat_box = f"R² = {case_data['r2']:.3f}\nPCC = {case_data['pcc']:.3f}\nSCC = {case_data['scc']:.3f}"
    ax.text(
        0.03,
        0.97,
        title_box,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.1,
        bbox=dict(boxstyle="square,pad=0.22", facecolor="white", edgecolor=LIGHT_GRAY, linewidth=0.8),
    )
    ax.text(
        0.97,
        0.06,
        stat_box,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.0,
        bbox=dict(boxstyle="square,pad=0.20", facecolor="white", edgecolor=LIGHT_GRAY, linewidth=0.8),
    )


def render_panel_bar(work: pd.DataFrame, metric: str, output_base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    draw_grouped_metric_panel(ax, work, metric)
    fig.tight_layout()
    return save_figure(fig, output_base)


def render_panel_dumbbell(top10: pd.DataFrame, output_base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(16.8, 4.8))
    draw_uplift_dumbbell(ax, top10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return save_figure(fig, output_base)


def render_panel_case_row(cases: pd.DataFrame, case_payloads: list[dict[str, object]], output_base: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.3))
    for ax, (_, case_meta), case_data in zip(axes, cases.iterrows(), case_payloads):
        draw_case_scatter(ax, case_meta, case_data)
    fig.tight_layout(w_pad=1.0)
    return save_figure(fig, output_base)


def render_main_figure(
    work: pd.DataFrame,
    top10_pcc: pd.DataFrame,
    cases: pd.DataFrame,
    case_payloads: list[dict[str, object]],
    output_base: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(16.8, 13.8))
    grid = GridSpec(nrows=3, ncols=4, figure=fig, height_ratios=[1.0, 1.0, 1.0], hspace=0.34, wspace=0.32)

    ax_pcc = fig.add_subplot(grid[0, 0:2])
    ax_scc = fig.add_subplot(grid[0, 2:4])
    ax_dumbbell = fig.add_subplot(grid[1, :])
    scatter_axes = [fig.add_subplot(grid[2, idx]) for idx in range(4)]

    draw_grouped_metric_panel(ax_pcc, work, "mean_pcc")
    draw_grouped_metric_panel(ax_scc, work, "mean_scc")
    draw_uplift_dumbbell(ax_dumbbell, top10_pcc)
    for ax, (_, case_meta), case_data in zip(scatter_axes, cases.iterrows(), case_payloads):
        draw_case_scatter(ax, case_meta, case_data)

    panel_axes = [ax_pcc, ax_scc, ax_dumbbell, *scatter_axes]
    for label, ax in zip(list("ABCDEFG"), panel_axes):
        add_panel_label(ax, label)

    fig.subplots_adjust(top=0.985, bottom=0.055, left=0.055, right=0.985)
    return save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    set_publication_style()

    results_root = Path(args.results_root).resolve()
    output_dir = prepare_output_dir(Path(args.output_dir).resolve(), overwrite=bool(args.overwrite))
    panel_dir = output_dir / "panels"

    task_df, uplift_df = load_sources(results_root)
    top_row = prepare_top_row(task_df)
    uplift = prepare_uplift(uplift_df)
    top10_pcc = select_top10_uplift(uplift, "pcc")
    cases = select_case_studies(uplift)
    case_payloads = [
        load_case_prediction(results_root, str(row["dataset_root"]), str(row["dataset"]), str(row["drug"]))
        for _, row in cases.iterrows()
    ]

    rendered_paths: list[Path] = []
    rendered_paths.extend(render_panel_bar(top_row, "mean_pcc", panel_dir / "panel_top_mean_pcc"))
    rendered_paths.extend(render_panel_bar(top_row, "mean_scc", panel_dir / "panel_top_mean_scc"))
    rendered_paths.extend(render_panel_dumbbell(top10_pcc, panel_dir / "panel_top10_uplift_dumbbell"))
    rendered_paths.extend(render_panel_case_row(cases, case_payloads, panel_dir / "panel_case_studies_row"))
    rendered_paths.extend(render_main_figure(top_row, top10_pcc, cases, case_payloads, output_dir / "figure2_main_composite"))

    manifest = {
        "figure": "figure2_benchmark_overview",
        "plan_source": "Plot plan docx / Figure 2 benchmark overview composite",
        "rendered_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "top_row_source": str(results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "GLOBAL" / "dataset_task_model_summary.csv"),
        "uplift_source": str(results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "UPLIFT_MFMR_vs_MCLRP" / "all_task_per_drug_uplift.csv"),
        "top10_rule": "Middle row uses the top 10 drugs ranked by delta_pcc; callout arrows and points are colored by GDSC subtasks (ERK/PI3K with AUC/IC50 legend).",
        "case_rule": "Top 4 unique drugs ranked by pcc_MCLRP_MFMR across the full all-task drug table, deduplicated by drug name.",
        "top10_pcc_drugs": top10_pcc.loc[:, ["dataset_label", "drug", "delta_pcc"]].to_dict(orient="records"),
        "case_studies": [
            {
                "dataset_label": str(case_meta["dataset_label"]),
                "drug": str(case_meta["drug"]),
                "delta_pcc_from_benchmark": float(case_meta.get("delta_pcc", np.nan)),
                "delta_scc_from_benchmark": float(case_meta.get("delta_scc", np.nan)),
                "scatter_pcc_from_prediction_cache": float(case_payload["pcc"]),
                "scatter_scc_from_prediction_cache": float(case_payload["scc"]),
                "prediction_source": str(case_payload["prediction_source"]),
            }
            for (_, case_meta), case_payload in zip(cases.iterrows(), case_payloads)
        ],
        "rendered_files": [str(path) for path in rendered_paths],
        "note": "Bottom-row scatter panels reuse preserved ablation mean_prediction caches because the current full 8-model benchmark tree does not retain mean_prediction files.",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rendered_files": [str(path) for path in rendered_paths]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
