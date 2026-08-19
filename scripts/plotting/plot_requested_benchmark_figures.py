from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


PRIMARY_BLUE = "#5B9BD5"
DARK_BLUE = "#2D3E50"
MID_BLUE = "#7FAFD9"
LIGHT_BLUE = "#A5C8E1"
PALE_BLUE = "#D4E4F1"
SKY_BLUE = "#6CB4EE"
SOFT_BLUE = "#BDD3EC"
LIGHT_TEAL = "#8FC6C1"
DARK_TEAL = "#2A6672"
SAND_ORANGE = "#FAE6CC"
PALE_ORANGE = "#F5D7AE"
LIGHT_ORANGE = "#F0B46C"
MID_ORANGE = "#E8923F"
PRIMARY_ORANGE = "#E67E22"
DARK_ORANGE = "#C97A1B"
SLATE = "#64748B"
DARK_GRAY = "#334155"
GRID_BLUE = "#E6EDF4"
LINE_BLUE = "#C4D3E0"

FIG_DPI = 600
FONT_FAMILY = "DejaVu Sans"
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 9
TITLE_SIZE = 13
ANNOT_SIZE = 8
BAR_EDGE_WIDTH = 0.7

TASK_ORDER = ["ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"]
TASK_DISPLAY = {
    "ERK_AUC": "ERK AUC",
    "ERK_IC50": "ERK IC50",
    "PI3K_AUC": "PI3K AUC",
    "PI3K_IC50": "PI3K IC50",
}
DATASET_ROOT_ORDER = ["CCLE", "GDSC", "CGP"]
SUBTASK_LABEL_ORDER = [
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
SUBTASK_DISPLAY = {
    "CCLE": "CCLE",
    "GDSC-ERK_AUC": "ERK AUC",
    "GDSC-ERK_IC50": "ERK IC50",
    "GDSC-PI3K_AUC": "PI3K AUC",
    "GDSC-PI3K_IC50": "PI3K IC50",
    "CGP-ERK_AUC": "ERK AUC",
    "CGP-ERK_IC50": "ERK IC50",
    "CGP-PI3K_AUC": "PI3K AUC",
    "CGP-PI3K_IC50": "PI3K IC50",
}

MODEL_DISPLAY_NAMES = {
    "MC": "MC",
    "RR": "RR",
    "MCRR": "MCRR",
    "SRMF": "SRMF",
    "DeepIC50": "DeepIC50",
    "GeneVAE": "GeneVAE",
    "MCLRP": "MCLRP",
    "MCLRP_MFMR": "MCLRP-MFMR",
}

MODEL_PAPER_ORDER = ["MC", "RR", "MCRR", "SRMF", "DeepIC50", "GeneVAE", "MCLRP", "MCLRP_MFMR"]

SHARED_VARIANT_ORDER = [
    "A0_Original_MCLRP",
    "A1_ImpOnly",
    "A2_RidgeOnly",
    "A3_MFMR_Equal",
    "A4_MFMR_SingleView",
    "A5_MFMR_Full",
]
SHARED_VARIANT_LABELS = {
    "A0_Original_MCLRP": "Original\nbackbone",
    "A1_ImpOnly": "Imputer\nonly",
    "A2_RidgeOnly": "Ridge\nonly",
    "A3_MFMR_Equal": "Equal\nfusion",
    "A4_MFMR_SingleView": "Single\nview",
    "A5_MFMR_Full": "Full\nshared",
}

CGP_VARIANT_ORDER = [
    "C0_CGP_Base",
    "C1_CGP_TissueLatent",
    "C2_CGP_FullMinusPathway",
    "C4_CGP_FullMinusLatent",
    "C5_CGP_FullMinusInteraction",
    "C6_CGP_Full",
]
CGP_VARIANT_LABELS = {
    "C0_CGP_Base": "Base\nhead",
    "C1_CGP_TissueLatent": "Tissue+\nlatent",
    "C2_CGP_FullMinusPathway": "- Pathway",
    "C4_CGP_FullMinusLatent": "- Latent",
    "C5_CGP_FullMinusInteraction": "- Interaction",
    "C6_CGP_Full": "Full\nmutation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate requested benchmark, uplift, and ablation figures for the MCLRP-MFMR paper."
    )
    parser.add_argument("--input_dir", type=str, default=str(RESULTS_DIR), help="Root results directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "plotting" / "outputs" / "requested_summary_figures"),
        help="Directory for exported figures.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing figures in the output directory.")
    return parser.parse_args()


def set_publication_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": DARK_GRAY,
            "axes.labelcolor": DARK_GRAY,
            "xtick.color": DARK_GRAY,
            "ytick.color": DARK_GRAY,
            "text.color": DARK_GRAY,
            "grid.color": GRID_BLUE,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.grid": False,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )


def make_bluepurple_cmap() -> LinearSegmentedColormap:
    return sns.color_palette("viridis", as_cmap=True)


def save_figure(fig: plt.Figure, output_base: Path) -> Path:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    out_path = output_base.with_suffix(".png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_path


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_color(DARK_GRAY)
    ax.spines["bottom"].set_color(DARK_GRAY)
    ax.tick_params(colors=DARK_GRAY)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_BLUE, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def discover_sources(results_root: Path) -> dict[str, Path]:
    benchmark_root = results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv"
    ablation_root = results_root / "latest" / "ablation_mclrp_mfmr" / "aggregate"
    return {
        "overall": benchmark_root / "GLOBAL" / "all_drug_model_mean.csv",
        "task_summary": benchmark_root / "GLOBAL" / "dataset_task_model_summary.csv",
        "all_per_drug": benchmark_root / "GLOBAL" / "all_per_drug_metrics.csv",
        "ccle_summary": benchmark_root / "CCLE" / "summary_metrics.csv",
        "gdsc_agg": benchmark_root / "GDSC" / "aggregate" / "aggregate_model_average.csv",
        "cgp_agg": benchmark_root / "CGP" / "aggregate" / "aggregate_model_average.csv",
        "uplift": benchmark_root / "UPLIFT_MFMR_vs_MCLRP" / "all_task_per_drug_uplift.csv",
        "uplift_summary": benchmark_root / "UPLIFT_MFMR_vs_MCLRP" / "task_uplift_summary.csv",
        "ablation_summary": ablation_root / "all_variant_summary.csv",
        "ablation_seed": ablation_root / "all_seed_results.csv",
        "ablation_fold": ablation_root / "all_fold_results.csv",
    }


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def truncate_label(text: str, max_len: int = 28) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def dataset_root_from_label(label: str) -> str:
    if label == "CCLE":
        return "CCLE"
    return label.split("-", 1)[0]


def task_from_label(label: str) -> str:
    if label == "CCLE":
        return "CCLE"
    return label.split("-", 1)[1]


def load_dataset_average_tables(paths: dict[str, Path]) -> pd.DataFrame:
    ccle = pd.read_csv(paths["ccle_summary"]).loc[:, ["model", "mean_pcc", "mean_scc"]].copy()
    ccle["dataset_label"] = "CCLE"

    gdsc = pd.read_csv(paths["gdsc_agg"]).loc[:, ["model", "mean_pcc", "mean_scc"]].copy()
    gdsc["dataset_label"] = "GDSC"

    cgp = pd.read_csv(paths["cgp_agg"]).loc[:, ["model", "mean_pcc", "mean_scc"]].copy()
    cgp["dataset_label"] = "CGP"

    return pd.concat([ccle, gdsc, cgp], ignore_index=True)


def build_model_order(overall_df: pd.DataFrame) -> list[str]:
    available = set(overall_df["model"].tolist())
    return [model for model in MODEL_PAPER_ORDER if model in available]


def build_model_palette(model_order: list[str]) -> dict[str, str]:
    fixed = {
        "MC": PALE_BLUE,
        "RR": SOFT_BLUE,
        "MCRR": LIGHT_BLUE,
        "SRMF": PRIMARY_BLUE,
        "DeepIC50": PALE_ORANGE,
        "GeneVAE": MID_ORANGE,
        "MCLRP": LIGHT_TEAL,
        "MCLRP_MFMR": DARK_TEAL,
    }
    return {model: fixed.get(model, LIGHT_BLUE) for model in model_order}


def build_two_model_palette() -> dict[str, str]:
    return {"MCLRP": LIGHT_TEAL, "MCLRP_MFMR": DARK_TEAL}


def add_value_labels_vertical(ax: plt.Axes, bars, dy: float = 0.012, fontsize: int = ANNOT_SIZE) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=DARK_GRAY,
            rotation=0,
        )


def format_task_label(task_name: str) -> str:
    return TASK_DISPLAY.get(task_name, task_name.replace("_", " "))


def format_dataset_label_short(label: str) -> str:
    if label == "CCLE":
        return "CCLE"
    root = dataset_root_from_label(label)
    task = format_task_label(task_from_label(label))
    return f"{root}\n{task}"


def add_group_headers(ax: plt.Axes, group_centers: dict[str, float], y: float = 1.02) -> None:
    for label, center in group_centers.items():
        ax.text(center, y, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)


def prepare_uplift_df(uplift_df: pd.DataFrame) -> pd.DataFrame:
    work = uplift_df.copy()
    work["dataset_root"] = work["dataset_label"].map(dataset_root_from_label)
    work["task_label"] = work["dataset_label"].map(task_from_label).map(format_task_label)
    work["display_drug"] = work["drug"].astype(str)
    mask = work["dataset_root"].isin(["GDSC", "CGP"])
    work.loc[mask, "display_drug"] = work.loc[mask, "drug"].astype(str) + " (" + work.loc[mask, "task_label"] + ")"
    return work


def prepare_all_per_drug_df(per_drug_df: pd.DataFrame) -> pd.DataFrame:
    work = per_drug_df.copy()
    work["dataset_root"] = work["dataset_label"].map(dataset_root_from_label)
    work["task_label"] = work["dataset_label"].map(task_from_label).map(format_task_label)
    work["display_drug"] = work["drug"].astype(str)
    mask = work["dataset_root"].isin(["GDSC", "CGP"])
    work.loc[mask, "display_drug"] = work.loc[mask, "drug"].astype(str) + " (" + work.loc[mask, "task_label"] + ")"
    return work


def prepare_ablation_summary(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = summary_df.copy()
    work["task_display"] = work["dataset"].map(lambda x: "CCLE" if x == "CCLE" else format_task_label(x))
    work["variant_label"] = work["variant_id"]
    work.loc[work["group"] == "shared_main", "variant_label"] = work.loc[work["group"] == "shared_main", "variant_id"].map(
        SHARED_VARIANT_LABELS
    )
    work.loc[work["group"] == "cgp_main", "variant_label"] = work.loc[work["group"] == "cgp_main", "variant_id"].map(
        CGP_VARIANT_LABELS
    )

    aggregated = (
        work.groupby(["group", "variant_id", "variant_label"], as_index=False)
        .agg(mean_pcc=("mean_pcc", "mean"), mean_scc=("mean_scc", "mean"))
    )
    full_ids = {"shared_main": "A5_MFMR_Full", "cgp_main": "C6_CGP_Full"}
    frames = []
    for group_name, full_id in full_ids.items():
        panel = aggregated.loc[aggregated["group"] == group_name].copy()
        full_mean = float(panel.loc[panel["variant_id"] == full_id, "mean_pcc"].iloc[0])
        panel["delta_vs_full"] = panel["mean_pcc"] - full_mean
        frames.append(panel)
    aggregated = pd.concat(frames, ignore_index=True)
    return work, aggregated


def get_variant_palette(group_name: str) -> dict[str, str]:
    if group_name == "shared_main":
        return {
            "A0_Original_MCLRP": PALE_BLUE,
            "A1_ImpOnly": SOFT_BLUE,
            "A2_RidgeOnly": LIGHT_BLUE,
            "A3_MFMR_Equal": SKY_BLUE,
            "A4_MFMR_SingleView": MID_BLUE,
            "A5_MFMR_Full": DARK_BLUE,
        }
    return {
        "C0_CGP_Base": SAND_ORANGE,
        "C1_CGP_TissueLatent": PALE_ORANGE,
        "C2_CGP_FullMinusPathway": LIGHT_ORANGE,
        "C4_CGP_FullMinusLatent": MID_ORANGE,
        "C5_CGP_FullMinusInteraction": PRIMARY_ORANGE,
        "C6_CGP_Full": DARK_ORANGE,
    }


def plot_overall_model_comparison(overall_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    work = overall_df.set_index("model").reindex(model_order).reset_index()
    palette = build_model_palette(model_order)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    for ax, metric_col, metric_name in zip(axes, ["mean_pcc", "mean_scc"], ["Mean PCC", "Mean SCC"]):
        ax.bar(
            np.arange(len(work)),
            work[metric_col].to_numpy(dtype=float),
            color=[palette[m] for m in work["model"]],
            edgecolor=DARK_GRAY,
            linewidth=BAR_EDGE_WIDTH,
            width=0.72,
        )
        ax.set_xticks(np.arange(len(work)))
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(model, model) for model in work["model"]], rotation=35, ha="right")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_ylim(0.0, max(0.82, float(work[metric_col].max()) + 0.08))
        style_axis(ax, grid_axis="y")

    fig.suptitle("Overall 8-Model Mean Performance", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.80, bottom=0.22, wspace=0.12)
    return save_figure(fig, output_dir / "figure_overall_8model_mean_comparison")


def plot_dataset_model_comparison(dataset_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = build_model_palette(model_order)
    datasets = DATASET_ROOT_ORDER
    x = np.arange(len(datasets))
    width = 0.095

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6), sharey=True)
    for ax, metric_col, metric_name in zip(axes, ["mean_pcc", "mean_scc"], ["Mean PCC", "Mean SCC"]):
        for idx, model in enumerate(model_order):
            values = []
            for dataset_label in datasets:
                value = dataset_df.loc[
                    (dataset_df["dataset_label"] == dataset_label) & (dataset_df["model"] == model), metric_col
                ].iloc[0]
                values.append(float(value))
            offset = (idx - (len(model_order) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=palette[model],
                edgecolor=DARK_GRAY,
                linewidth=0.45,
                label=MODEL_DISPLAY_NAMES.get(model, model),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_ylim(0.0, max(0.86, float(dataset_df[metric_col].max()) + 0.08))
        style_axis(ax, grid_axis="y")

    handles = [Patch(facecolor=palette[m], edgecolor=DARK_GRAY, label=MODEL_DISPLAY_NAMES.get(m, m)) for m in model_order]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Dataset-Level 8-Model Mean Performance", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.12)
    return save_figure(fig, output_dir / "figure_dataset_8model_mean_comparison")


def plot_task_model_comparison(task_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = build_model_palette(model_order)
    width = 0.095
    task_positions = np.arange(len(TASK_ORDER))

    fig, axes = plt.subplots(2, 2, figsize=(16.6, 9.0), sharey=False)
    panel_specs = [
        (axes[0, 0], "GDSC", "mean_pcc", "GDSC Mean PCC"),
        (axes[0, 1], "GDSC", "mean_scc", "GDSC Mean SCC"),
        (axes[1, 0], "CGP", "mean_pcc", "CGP Mean PCC"),
        (axes[1, 1], "CGP", "mean_scc", "CGP Mean SCC"),
    ]

    for ax, group_name, metric_col, title in panel_specs:
        panel = task_df.loc[task_df["group"] == group_name].copy()
        for idx, model in enumerate(model_order):
            values = []
            for task_name in TASK_ORDER:
                value = panel.loc[(panel["dataset"] == task_name) & (panel["model"] == model), metric_col].iloc[0]
                values.append(float(value))
            offset = (idx - (len(model_order) - 1) / 2) * width
            ax.bar(
                task_positions + offset,
                values,
                width=width,
                color=palette[model],
                edgecolor=DARK_GRAY,
                linewidth=0.35,
            )
        ax.set_xticks(task_positions)
        ax.set_xticklabels([format_task_label(task_name) for task_name in TASK_ORDER], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(metric_col.replace("mean_", "Mean ").upper())
        ax.set_ylim(0.0, max(0.96, float(panel[metric_col].max()) + 0.08))
        style_axis(ax, grid_axis="y")

    handles = [Patch(facecolor=palette[m], edgecolor=DARK_GRAY, label=MODEL_DISPLAY_NAMES.get(m, m)) for m in model_order]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("GDSC / CGP Molecular-Task Mean Performance", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.88, bottom=0.18, wspace=0.12, hspace=0.28)
    return save_figure(fig, output_dir / "figure_task_8model_mean_comparison")


def plot_mclrp_vs_mfmr_subtask_mean(task_df: pd.DataFrame, output_dir: Path) -> Path:
    palette = build_two_model_palette()
    work = task_df.loc[task_df["model"].isin(["MCLRP", "MCLRP_MFMR"])].copy()
    work = work.set_index(["dataset_label", "model"]).reindex(
        pd.MultiIndex.from_product([SUBTASK_LABEL_ORDER, ["MCLRP", "MCLRP_MFMR"]], names=["dataset_label", "model"])
    ).reset_index()
    x = np.arange(len(SUBTASK_LABEL_ORDER))
    width = 0.34
    group_centers = {"CCLE": 0.0, "GDSC": 2.5, "CGP": 6.5}

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.6), sharey=True)
    for ax, metric_col, metric_name in zip(axes, ["mean_pcc", "mean_scc"], ["Mean PCC", "Mean SCC"]):
        for idx, model in enumerate(["MCLRP", "MCLRP_MFMR"]):
            sub = work.loc[work["model"] == model].copy()
            offset = (-0.5 + idx) * width
            ax.bar(
                x + offset,
                sub[metric_col].to_numpy(dtype=float),
                width=width,
                color=palette[model],
                edgecolor=DARK_GRAY,
                linewidth=BAR_EDGE_WIDTH,
                label=MODEL_DISPLAY_NAMES[model],
            )
        ax.axvline(0.5, color=LINE_BLUE, linewidth=1.0)
        ax.axvline(4.5, color=LINE_BLUE, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([SUBTASK_DISPLAY[label] for label in SUBTASK_LABEL_ORDER], rotation=30, ha="right")
        ax.set_ylim(0.0, max(0.95, float(work[metric_col].max()) + 0.08))
        ax.set_ylabel(metric_name)
        ax.set_xlabel(metric_name, labelpad=12)
        add_group_headers(ax, group_centers)
        style_axis(ax, grid_axis="y")

    handles = [Patch(facecolor=palette[m], edgecolor=DARK_GRAY, label=MODEL_DISPLAY_NAMES[m]) for m in ["MCLRP", "MCLRP_MFMR"]]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("MCLRP vs MCLRP-MFMR Across Datasets and Molecular Subtasks", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.82, bottom=0.24, wspace=0.08)
    return save_figure(fig, output_dir / "figure_mclrp_vs_mfmr_subtask_mean_comparison")


def plot_per_drug_pairwise(uplift_df: pd.DataFrame, dataset_root: str, output_dir: Path) -> Path:
    palette = build_two_model_palette()
    panel_df = uplift_df.loc[uplift_df["dataset_root"] == dataset_root].copy()
    panel_df = panel_df.sort_values(["delta_pcc", "delta_scc"], ascending=[False, False]).reset_index(drop=True)
    n_rows = len(panel_df)
    fig_height = max(7.4, 2.9 + n_rows * 0.23)
    label_fontsize = 7.3 if n_rows <= 30 else 5.8 if n_rows <= 80 else 4.6

    fig, axes = plt.subplots(1, 2, figsize=(17.2, fig_height), sharey=True)
    metric_specs = [
        (axes[0], "pcc_MCLRP", "pcc_MCLRP_MFMR", "Mean PCC"),
        (axes[1], "scc_MCLRP", "scc_MCLRP_MFMR", "Mean SCC"),
    ]
    y = np.arange(n_rows)
    bar_h = 0.36
    labels = [truncate_label(text, max_len=22 if dataset_root == "CCLE" else 34) for text in panel_df["display_drug"]]

    for ax, start_col, end_col, metric_name in metric_specs:
        ax.barh(
            y + bar_h / 2,
            panel_df[start_col].to_numpy(dtype=float),
            height=bar_h,
            color=palette["MCLRP"],
            edgecolor=DARK_GRAY,
            linewidth=0.3,
            label="MCLRP",
        )
        ax.barh(
            y - bar_h / 2,
            panel_df[end_col].to_numpy(dtype=float),
            height=bar_h,
            color=palette["MCLRP_MFMR"],
            edgecolor=DARK_GRAY,
            linewidth=0.3,
            label="MCLRP-MFMR",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(metric_name)
        style_axis(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=label_fontsize)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()

    left_margin = 0.19 if dataset_root == "CCLE" else 0.31
    right_margin = 0.985
    title_x = (left_margin + right_margin) / 2.0

    handles = [
        Patch(facecolor=palette["MCLRP"], edgecolor=DARK_GRAY, label="MCLRP"),
        Patch(facecolor=palette["MCLRP_MFMR"], edgecolor=DARK_GRAY, label="MCLRP-MFMR"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(title_x, 0.955))
    fig.suptitle(
        f"{dataset_root} Per-Drug MCLRP vs MCLRP-MFMR Comparison",
        x=title_x,
        y=0.99,
        fontsize=TITLE_SIZE + 1,
    )
    fig.subplots_adjust(left=left_margin, right=right_margin, top=0.92, bottom=0.07, wspace=0.10)
    return save_figure(fig, output_dir / f"figure_per_drug_mclrp_vs_mfmr_{dataset_root.lower()}")


def plot_top5_uplift(uplift_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 8.0))
    axes = np.atleast_1d(axes)
    palette = build_two_model_palette()

    for ax, dataset_root in zip(axes, DATASET_ROOT_ORDER):
        panel_df = uplift_df.loc[uplift_df["dataset_root"] == dataset_root].copy()
        panel_df = panel_df.sort_values("delta_pcc", ascending=False).head(10).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(panel_df))
        start_values = panel_df["pcc_MCLRP"].to_numpy(dtype=float)
        end_values = panel_df["pcc_MCLRP_MFMR"].to_numpy(dtype=float)
        delta_values = panel_df["delta_pcc"].to_numpy(dtype=float)

        for ypos, start, end, delta in zip(y, start_values, end_values, delta_values):
            ax.annotate(
                "",
                xy=(end, ypos),
                xytext=(start, ypos),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": LINE_BLUE,
                    "lw": 2.8,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 16,
                },
            )
            ax.scatter(start, ypos, s=76, color=palette["MCLRP"], edgecolor=DARK_GRAY, linewidth=0.55, zorder=3)
            ax.scatter(end, ypos, s=94, color=palette["MCLRP_MFMR"], edgecolor=DARK_GRAY, linewidth=0.55, zorder=4)
            ax.text(end + 0.012, ypos, f"{delta:+.3f}", va="center", ha="left", fontsize=9, color=DARK_GRAY)

        ax.set_yticks(y)
        ax.set_yticklabels([truncate_label(drug, max_len=26) for drug in panel_df["display_drug"].tolist()], fontsize=7.6)
        ax.set_title(dataset_root)
        xmin = max(0.0, float(np.nanmin(start_values)) - 0.06)
        xmax = min(1.0, float(np.nanmax(end_values)) + 0.16)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("PCC")
        style_axis(ax, grid_axis="x")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=palette["MCLRP"], markerfacecolor=palette["MCLRP"], markeredgecolor=DARK_GRAY, markersize=7.5, label="MCLRP"),
        Line2D([0], [0], marker="o", linestyle="", color=palette["MCLRP_MFMR"], markerfacecolor=palette["MCLRP_MFMR"], markeredgecolor=DARK_GRAY, markersize=8.5, label="MCLRP-MFMR"),
        Line2D([0, 1], [0, 0], color=LINE_BLUE, linewidth=2.8, label="Improvement arrow"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Top-10 PCC Uplift of MCLRP-MFMR over MCLRP", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.text(
        0.5,
        0.01,
        "Each panel shows the top-10 drugs ranked by Delta PCC within one dataset; GDSC/CGP labels include the source molecular task.",
        ha="center",
        fontsize=ANNOT_SIZE,
        color=DARK_GRAY,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.82, bottom=0.14, wspace=0.48)
    return save_figure(fig, output_dir / "figure_mfmr_vs_mclrp_top5_pcc_uplift")


def plot_per_drug_heatmap(all_per_drug_df: pd.DataFrame, dataset_root: str, model_order: list[str], output_dir: Path) -> Path:
    panel_df = all_per_drug_df.loc[all_per_drug_df["dataset_root"] == dataset_root].copy()
    pcc_pivot = panel_df.pivot(index="display_drug", columns="model", values="pcc").reindex(columns=model_order)
    scc_pivot = panel_df.pivot(index="display_drug", columns="model", values="scc").reindex(columns=model_order)

    if {"MCLRP", "MCLRP_MFMR"}.issubset(pcc_pivot.columns):
        sort_key = (pcc_pivot["MCLRP_MFMR"] - pcc_pivot["MCLRP"]).sort_values(ascending=False)
        ordered_rows = sort_key.index.tolist()
        pcc_pivot = pcc_pivot.reindex(index=ordered_rows)
        scc_pivot = scc_pivot.reindex(index=ordered_rows)

    n_rows = len(pcc_pivot)
    fig_height = max(6.8, 2.8 + n_rows * 0.10)
    left_margin = 0.23 if dataset_root == "CCLE" else 0.31
    label_fontsize = 7.0 if n_rows <= 30 else 5.6 if n_rows <= 90 else 4.6

    fig, axes = plt.subplots(1, 2, figsize=(14.6, fig_height), sharey=True)
    cmap = make_bluepurple_cmap()
    for idx, (ax, pivot, metric_name) in enumerate(zip(axes, [pcc_pivot, scc_pivot], ["PCC", "SCC"])):
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.15,
            linecolor=GRID_BLUE,
            cbar=idx == 1,
            cbar_kws={"label": metric_name, "shrink": 0.78},
        )
        ax.set_title(metric_name, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(model, model) for model in model_order], rotation=35, ha="right")
        if idx == 0:
            ax.set_yticklabels([truncate_label(text.get_text(), max_len=32) for text in ax.get_yticklabels()], rotation=0, fontsize=label_fontsize)
        else:
            ax.tick_params(axis="y", labelleft=False)
        sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle(f"{dataset_root} 8-Model Per-Drug Heatmap", x=0.5, y=0.998, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(left=left_margin, right=0.98, top=0.93, bottom=0.05, wspace=0.12)
    return save_figure(fig, output_dir / f"figure_per_drug_heatmap_{dataset_root.lower()}")


def plot_delta_distribution(uplift_df: pd.DataFrame, uplift_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    task_order_pcc = uplift_summary_df.sort_values("mean_delta_pcc", ascending=False)["dataset_label"].tolist()
    task_order_scc = uplift_summary_df.sort_values("mean_delta_scc", ascending=False)["dataset_label"].tolist()
    root_palette = {"CCLE": DARK_BLUE, "GDSC": PRIMARY_BLUE, "CGP": PRIMARY_ORANGE}

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.0))

    violin_specs = [
        (axes[0, 0], "delta_pcc", "ΔPCC by Task", task_order_pcc, "positive_pcc_drugs", "num_drugs"),
        (axes[0, 1], "delta_scc", "ΔSCC by Task", task_order_scc, "positive_scc_drugs", "num_drugs"),
    ]
    for ax, metric_col, title, order, positive_col, count_col in violin_specs:
        sns.violinplot(
            data=uplift_df,
            x="dataset_label",
            y=metric_col,
            order=order,
            ax=ax,
            color=SOFT_BLUE,
            inner=None,
            cut=0,
            linewidth=0.8,
        )
        sns.boxplot(
            data=uplift_df,
            x="dataset_label",
            y=metric_col,
            order=order,
            ax=ax,
            width=0.18,
            color=DARK_BLUE,
            showfliers=False,
        )
        ax.axhline(0.0, color=DARK_GRAY, linewidth=0.9, linestyle="--")
        ax.set_title(title, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric_col.replace("delta_", ""))
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([format_dataset_label_short(label) for label in order], rotation=0)
        ymin, ymax = ax.get_ylim()
        text_y = ymax - (ymax - ymin) * 0.02
        for xpos, label in enumerate(order):
            row = uplift_summary_df.loc[uplift_summary_df["dataset_label"] == label].iloc[0]
            ratio = float(row[positive_col]) / float(row[count_col])
            ax.text(xpos, text_y, f"{ratio:.0%}", ha="center", va="top", fontsize=ANNOT_SIZE, color=DARK_GRAY)
        style_axis(ax, grid_axis="y")

    cdf_specs = [
        (axes[1, 0], "delta_pcc", "ΔPCC CDF by Dataset"),
        (axes[1, 1], "delta_scc", "ΔSCC CDF by Dataset"),
    ]
    for ax, metric_col, title in cdf_specs:
        for dataset_root in DATASET_ROOT_ORDER:
            values = np.sort(uplift_df.loc[uplift_df["dataset_root"] == dataset_root, metric_col].to_numpy(dtype=float))
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, color=root_palette[dataset_root], linewidth=2.0, label=dataset_root)
        ax.axvline(0.0, color=DARK_GRAY, linewidth=0.9, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(metric_col.replace("delta_", "Δ").upper())
        ax.set_ylabel("Cumulative fraction")
        style_axis(ax, grid_axis="both")

    handles = [Line2D([0], [0], color=root_palette[name], linewidth=2.2, label=name) for name in DATASET_ROOT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Drug-Level Uplift Distribution of MCLRP-MFMR over MCLRP", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.20, hspace=0.28)
    return save_figure(fig, output_dir / "figure_uplift_distribution")


def plot_delta_distribution(uplift_df: pd.DataFrame, uplift_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    task_order_pcc = uplift_summary_df.sort_values("mean_delta_pcc", ascending=False)["dataset_label"].tolist()
    task_order_scc = uplift_summary_df.sort_values("mean_delta_scc", ascending=False)["dataset_label"].tolist()
    root_palette = {"CCLE": DARK_BLUE, "GDSC": PRIMARY_BLUE, "CGP": PRIMARY_ORANGE}

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.0))

    violin_specs = [
        (axes[0, 0], "delta_pcc", r"$\Delta$PCC by Task", task_order_pcc, "positive_pcc_drugs", "num_drugs"),
        (axes[0, 1], "delta_scc", r"$\Delta$SCC by Task", task_order_scc, "positive_scc_drugs", "num_drugs"),
    ]
    for ax, metric_col, title, order, positive_col, count_col in violin_specs:
        sns.violinplot(
            data=uplift_df,
            x="dataset_label",
            y=metric_col,
            order=order,
            ax=ax,
            color=SOFT_BLUE,
            inner=None,
            cut=0,
            linewidth=0.8,
        )
        sns.boxplot(
            data=uplift_df,
            x="dataset_label",
            y=metric_col,
            order=order,
            ax=ax,
            width=0.18,
            color=DARK_BLUE,
            showfliers=False,
        )
        ax.axhline(0.0, color=DARK_GRAY, linewidth=0.9, linestyle="--")
        ax.set_title(title, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric_col.replace("delta_", "").upper())
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([format_dataset_label_short(label) for label in order], rotation=0)
        ymin, ymax = ax.get_ylim()
        text_y = ymax - (ymax - ymin) * 0.02
        for xpos, label in enumerate(order):
            row = uplift_summary_df.loc[uplift_summary_df["dataset_label"] == label].iloc[0]
            ratio = float(row[positive_col]) / float(row[count_col])
            ax.text(xpos, text_y, f"{ratio:.0%}", ha="center", va="top", fontsize=ANNOT_SIZE, color=DARK_GRAY)
        style_axis(ax, grid_axis="y")

    cdf_specs = [
        (axes[1, 0], "delta_pcc", r"$\Delta$PCC CDF by Dataset"),
        (axes[1, 1], "delta_scc", r"$\Delta$SCC CDF by Dataset"),
    ]
    for ax, metric_col, title in cdf_specs:
        for dataset_root in DATASET_ROOT_ORDER:
            values = np.sort(uplift_df.loc[uplift_df["dataset_root"] == dataset_root, metric_col].to_numpy(dtype=float))
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, color=root_palette[dataset_root], linewidth=2.0, label=dataset_root)
        ax.axvline(0.0, color=DARK_GRAY, linewidth=0.9, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(rf"$\Delta${metric_col.replace('delta_', '').upper()}")
        ax.set_ylabel("Cumulative fraction")
        style_axis(ax, grid_axis="both")

    handles = [Line2D([0], [0], color=root_palette[name], linewidth=2.2, label=name) for name in DATASET_ROOT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Drug-Level Uplift Distribution of MCLRP-MFMR over MCLRP", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.20, hspace=0.28)
    return save_figure(fig, output_dir / "figure_uplift_distribution")


def plot_nobs_vs_pcc(all_per_drug_df: pd.DataFrame, output_dir: Path) -> Path:
    palette = build_two_model_palette()
    work = all_per_drug_df.loc[all_per_drug_df["model"].isin(["MCLRP", "MCLRP_MFMR"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharey=True)
    for ax, dataset_root in zip(axes, DATASET_ROOT_ORDER):
        panel = work.loc[work["dataset_root"] == dataset_root].copy()
        for model in ["MCLRP", "MCLRP_MFMR"]:
            sub = panel.loc[panel["model"] == model].copy()
            ax.scatter(sub["n_obs"], sub["pcc"], s=30, alpha=0.72, color=palette[model], edgecolor="none", label=MODEL_DISPLAY_NAMES[model])
            if sub["n_obs"].nunique() > 1:
                coef = np.polyfit(sub["n_obs"].to_numpy(dtype=float), sub["pcc"].to_numpy(dtype=float), deg=1)
                x_line = np.linspace(sub["n_obs"].min(), sub["n_obs"].max(), 100)
                y_line = coef[0] * x_line + coef[1]
                ax.plot(x_line, y_line, color=palette[model], linewidth=1.8)
        ax.set_title(dataset_root)
        ax.set_xlabel("n_obs")
        ax.set_ylabel("PCC")
        style_axis(ax, grid_axis="both")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=palette["MCLRP"], markerfacecolor=palette["MCLRP"], markersize=7, label="MCLRP"),
        Line2D([0], [0], marker="o", linestyle="", color=palette["MCLRP_MFMR"], markerfacecolor=palette["MCLRP_MFMR"], markersize=7, label="MCLRP-MFMR"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Drug-Level Sample Size vs PCC", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.80, bottom=0.20, wspace=0.12)
    return save_figure(fig, output_dir / "figure_nobs_vs_pcc_scatter")


def plot_num_best(task_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = build_model_palette(model_order)
    summary = (
        task_df.groupby("model", as_index=False)[["num_best_pcc", "num_best_scc"]]
        .sum()
        .set_index("model")
        .reindex(model_order)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    for ax, metric_col, title in zip(axes, ["num_best_pcc", "num_best_scc"], ["Total num_best_pcc", "Total num_best_scc"]):
        bars = ax.bar(
            np.arange(len(summary)),
            summary[metric_col].to_numpy(dtype=float),
            color=[palette[m] for m in summary["model"]],
            edgecolor=DARK_GRAY,
            linewidth=BAR_EDGE_WIDTH,
            width=0.72,
        )
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(model, model) for model in summary["model"]], rotation=35, ha="right")
        ax.set_ylabel("Number of best-performing drugs")
        ax.set_title(title)
        for bar in bars:
            value = int(round(bar.get_height()))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{value}", ha="center", va="bottom", fontsize=ANNOT_SIZE)
        style_axis(ax, grid_axis="y")

    fig.suptitle("How Often Each Model Achieves the Best Drug-Level Performance", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.12)
    return save_figure(fig, output_dir / "figure_num_best_counts")


def plot_dataset_best_pcc_donuts(task_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = build_model_palette(model_order)
    dataset_order = SUBTASK_LABEL_ORDER

    fig, axes = plt.subplots(3, 3, figsize=(16.2, 14.2))
    axes = np.asarray(axes).reshape(-1)

    legend_handles = [
        Patch(facecolor=palette[model], edgecolor="white", label=MODEL_DISPLAY_NAMES.get(model, model))
        for model in model_order
    ]

    for ax, dataset_label in zip(axes, dataset_order):
        panel = (
            task_df.loc[task_df["dataset_label"] == dataset_label, ["model", "num_best_pcc"]]
            .set_index("model")
            .reindex(model_order)
            .fillna(0.0)
            .reset_index()
        )
        values = panel["num_best_pcc"].to_numpy(dtype=float)
        total = int(round(values.sum()))

        def _autopct(pct: float) -> str:
            count = int(round(pct * total / 100.0))
            return f"{count}" if count > 0 else ""

        ax.pie(
            values,
            startangle=90,
            counterclock=False,
            colors=[palette[model] for model in panel["model"]],
            wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 1.0},
            autopct=_autopct,
            pctdistance=0.82,
            textprops={"fontsize": 8, "color": DARK_GRAY},
        )
        ax.text(
            0.0,
            0.02,
            format_dataset_label_short(dataset_label),
            ha="center",
            va="center",
            fontsize=10,
            color=DARK_GRAY,
            linespacing=1.1,
            fontweight="bold",
        )
        ax.text(0.0, -0.28, f"n = {total}", ha="center", va="center", fontsize=8.5, color=SLATE)
        ax.set_aspect("equal")
        ax.set_title(f"{format_dataset_label_short(dataset_label)}", pad=10)

    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Best-PCC Drug Counts by Model for Each Dataset", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.15, hspace=0.28)
    return save_figure(fig, output_dir / "figure_dataset_best_pcc_donuts")


def plot_ablation_seed_stability(seed_df: pd.DataFrame, output_dir: Path) -> Path:
    seed_mean = (
        seed_df.groupby(["group", "variant_id", "seed"], as_index=False)
        .agg(seed_mean_pcc=("overall_pcc", "mean"))
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharey=False)
    group_specs = [
        (axes[0], "shared_main", SHARED_VARIANT_ORDER, SHARED_VARIANT_LABELS, "Shared Backbone Seed-Level PCC"),
        (axes[1], "cgp_main", CGP_VARIANT_ORDER, CGP_VARIANT_LABELS, "Mutation Head Seed-Level PCC"),
    ]

    for ax, group_name, order, label_map, title in group_specs:
        palette = get_variant_palette(group_name)
        panel = seed_mean.loc[seed_mean["group"] == group_name].copy()
        sns.violinplot(
            data=panel,
            x="variant_id",
            y="seed_mean_pcc",
            order=order,
            ax=ax,
            hue="variant_id",
            dodge=False,
            inner=None,
            cut=0,
            linewidth=0.8,
            palette=[palette[variant_id] for variant_id in order],
        )
        sns.boxplot(
            data=panel,
            x="variant_id",
            y="seed_mean_pcc",
            order=order,
            ax=ax,
            width=0.18,
            showfliers=False,
            color=PRIMARY_BLUE,
            boxprops={"facecolor": LIGHT_BLUE, "edgecolor": DARK_BLUE, "alpha": 0.95},
            whiskerprops={"linewidth": 0.8, "color": DARK_BLUE},
            capprops={"linewidth": 0.8, "color": DARK_BLUE},
            medianprops={"color": DARK_BLUE, "linewidth": 1.1},
        )
        sns.stripplot(
            data=panel,
            x="variant_id",
            y="seed_mean_pcc",
            order=order,
            ax=ax,
            color=DARK_GRAY,
            size=3.2,
            alpha=0.70,
            jitter=0.12,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([label_map[variant_id] for variant_id in order], rotation=0)
        ax.set_ylabel("Seed-level mean PCC")
        ax.set_xlabel("")
        ax.set_title(title)
        style_axis(ax, grid_axis="y")

    fig.suptitle("Seed-Level Ablation Stability", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.18)
    return save_figure(fig, output_dir / "figure_ablation_seed_stability")


def plot_ablation_bar(aggregated_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), sharey=False)
    group_specs = [
        (axes[0], "shared_main", SHARED_VARIANT_ORDER, SHARED_VARIANT_LABELS, "Shared Backbone Mean PCC"),
        (axes[1], "cgp_main", CGP_VARIANT_ORDER, CGP_VARIANT_LABELS, "Mutation Head Mean PCC"),
    ]

    for ax, group_name, order, label_map, title in group_specs:
        palette = get_variant_palette(group_name)
        panel = aggregated_df.loc[aggregated_df["group"] == group_name].set_index("variant_id").reindex(order).reset_index()
        bars = ax.bar(
            np.arange(len(panel)),
            panel["mean_pcc"].to_numpy(dtype=float),
            color=[palette[variant_id] for variant_id in panel["variant_id"]],
            edgecolor=DARK_GRAY,
            linewidth=BAR_EDGE_WIDTH,
            width=0.72,
        )
        ax.set_xticks(np.arange(len(panel)))
        ax.set_xticklabels([label_map[variant_id] for variant_id in panel["variant_id"]], rotation=0)
        ax.set_ylabel("Mean PCC")
        ax.set_title(title)
        ax.set_ylim(0.0, max(0.96, float(panel["mean_pcc"].max()) + 0.08))
        add_value_labels_vertical(ax, bars, dy=0.006, fontsize=7)
        style_axis(ax, grid_axis="y")

    fig.suptitle("Ablation: Mean PCC by Model Variant", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.16)
    return save_figure(fig, output_dir / "figure_ablation_mean_pcc_bar")


def plot_ablation_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    cmap = make_bluepurple_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))
    vmin = float(summary_df["mean_pcc"].min())
    vmax = float(summary_df["mean_pcc"].max())
    group_specs = [
        (axes[0], "shared_main", SHARED_VARIANT_ORDER, SHARED_VARIANT_LABELS, ["CCLE"] + TASK_ORDER, "Shared Backbone Heatmap"),
        (axes[1], "cgp_main", CGP_VARIANT_ORDER, CGP_VARIANT_LABELS, TASK_ORDER, "Mutation Head Heatmap"),
    ]

    for idx, (ax, group_name, order, label_map, tasks, title) in enumerate(group_specs):
        panel = summary_df.loc[(summary_df["group"] == group_name) & (summary_df["dataset"].isin(tasks))].copy()
        pivot = panel.pivot(index="variant_id", columns="dataset", values="mean_pcc").reindex(index=order, columns=tasks)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.7,
            linecolor=LINE_BLUE,
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": ANNOT_SIZE},
            cbar=idx == 1,
            cbar_kws={"label": "Mean PCC", "shrink": 0.86},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(["CCLE" if task == "CCLE" else format_task_label(task) for task in tasks], rotation=20, ha="right")
        ax.set_yticklabels([label_map[variant_id] for variant_id in order], rotation=0)
        sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle("Ablation Heatmap", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.16)
    return save_figure(fig, output_dir / "figure_ablation_heatmap")


def plot_ablation_waterfall(aggregated_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), sharex=False)
    group_specs = [
        (axes[0], "shared_main", SHARED_VARIANT_ORDER, SHARED_VARIANT_LABELS, "Shared Backbone PCC Loss vs Full"),
        (axes[1], "cgp_main", CGP_VARIANT_ORDER, CGP_VARIANT_LABELS, "Mutation Head PCC Loss vs Full"),
    ]

    for ax, group_name, order, label_map, title in group_specs:
        palette = get_variant_palette(group_name)
        panel = aggregated_df.loc[aggregated_df["group"] == group_name].set_index("variant_id").reindex(order).reset_index()
        panel["loss_vs_full"] = -panel["delta_vs_full"]
        work = panel.loc[panel["loss_vs_full"] != 0].copy()
        work = work.sort_values("loss_vs_full", ascending=False).reset_index(drop=True)
        y = np.arange(len(work))
        bars = ax.barh(
            y,
            work["loss_vs_full"].to_numpy(dtype=float),
            color=[palette[variant_id] for variant_id in work["variant_id"]],
            edgecolor=DARK_GRAY,
            linewidth=BAR_EDGE_WIDTH,
            height=0.68,
        )
        ax.axvline(0.0, color=DARK_GRAY, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([label_map[variant_id] for variant_id in work["variant_id"]])
        ax.invert_yaxis()
        ax.set_xlabel("PCC Loss vs Full")
        ax.set_title(title)
        max_loss = max(0.02, float(work["loss_vs_full"].max()) + 0.04)
        min_loss = min(-0.02, float(work["loss_vs_full"].min()) - 0.02)
        ax.set_xlim(min_loss, max_loss)
        for bar, value in zip(bars, work["loss_vs_full"].tolist()):
            ha = "left" if value >= 0 else "right"
            x_text = value + 0.004 if value >= 0 else value - 0.004
            ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha=ha, fontsize=ANNOT_SIZE)
        style_axis(ax, grid_axis="x")

    fig.suptitle("Ablation Waterfall: Contribution Relative to the Full Model", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.84, bottom=0.10, wspace=0.24)
    return save_figure(fig, output_dir / "figure_ablation_waterfall")


def plot_ablation_boxplot(fold_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharey=False)
    group_specs = [
        (axes[0], "shared_main", SHARED_VARIANT_ORDER, SHARED_VARIANT_LABELS, "Shared Backbone 10x10-Fold PCC"),
        (axes[1], "cgp_main", CGP_VARIANT_ORDER, CGP_VARIANT_LABELS, "Mutation Head 10x10-Fold PCC"),
    ]

    for ax, group_name, order, label_map, title in group_specs:
        palette = get_variant_palette(group_name)
        panel = fold_df.loc[fold_df["group"] == group_name].copy()
        sns.boxplot(
            data=panel,
            x="variant_id",
            y="pcc",
            hue="variant_id",
            order=order,
            ax=ax,
            palette=[palette[variant_id] for variant_id in order],
            dodge=False,
            showfliers=False,
            linewidth=0.8,
            width=0.68,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([label_map[variant_id] for variant_id in order], rotation=0)
        ax.set_ylabel("Fold-level PCC")
        ax.set_xlabel("")
        ax.set_title(title)
        style_axis(ax, grid_axis="y")

    fig.suptitle("Ablation Stability Across 10x10-Fold Cross-Validation", x=0.5, y=0.98, fontsize=TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.18)
    return save_figure(fig, output_dir / "figure_ablation_boxplot")


def write_manifest(paths: dict[str, Path], exported_files: list[Path], output_dir: Path) -> None:
    payload = {
        "data_sources": {key: str(value) for key, value in paths.items()},
        "exported_files": [str(path) for path in exported_files],
    }
    (output_dir / "requested_figure_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    set_publication_style()
    source_paths = discover_sources(input_dir)
    source_paths = {key: require_file(path) for key, path in source_paths.items()}

    overall_df = pd.read_csv(source_paths["overall"])
    task_df = pd.read_csv(source_paths["task_summary"])
    all_per_drug_df = prepare_all_per_drug_df(pd.read_csv(source_paths["all_per_drug"]))
    dataset_df = load_dataset_average_tables(source_paths)
    uplift_df = prepare_uplift_df(pd.read_csv(source_paths["uplift"]))
    uplift_summary_df = pd.read_csv(source_paths["uplift_summary"])
    ablation_summary_raw = pd.read_csv(source_paths["ablation_summary"])
    ablation_summary_df, ablation_aggregated_df = prepare_ablation_summary(ablation_summary_raw)
    ablation_seed_df = pd.read_csv(source_paths["ablation_seed"])
    ablation_fold_df = pd.read_csv(source_paths["ablation_fold"])

    model_order = build_model_order(overall_df)

    exported_files: list[Path] = []
    exported_files.append(plot_overall_model_comparison(overall_df, model_order, output_dir))
    exported_files.append(plot_dataset_model_comparison(dataset_df, model_order, output_dir))
    exported_files.append(plot_task_model_comparison(task_df, model_order, output_dir))
    exported_files.append(plot_mclrp_vs_mfmr_subtask_mean(task_df, output_dir))
    for dataset_root in DATASET_ROOT_ORDER:
        exported_files.append(plot_per_drug_pairwise(uplift_df, dataset_root, output_dir))
        exported_files.append(plot_per_drug_heatmap(all_per_drug_df, dataset_root, model_order, output_dir))
    exported_files.append(plot_top5_uplift(uplift_df, output_dir))
    exported_files.append(plot_delta_distribution(uplift_df, uplift_summary_df, output_dir))
    exported_files.append(plot_nobs_vs_pcc(all_per_drug_df, output_dir))
    exported_files.append(plot_num_best(task_df, model_order, output_dir))
    exported_files.append(plot_dataset_best_pcc_donuts(task_df, model_order, output_dir))
    exported_files.append(plot_ablation_bar(ablation_aggregated_df, output_dir))
    exported_files.append(plot_ablation_heatmap(ablation_summary_df, output_dir))
    exported_files.append(plot_ablation_waterfall(ablation_aggregated_df, output_dir))
    exported_files.append(plot_ablation_seed_stability(ablation_seed_df, output_dir))
    exported_files.append(plot_ablation_boxplot(ablation_fold_df, output_dir))

    write_manifest(source_paths, exported_files, output_dir)
    print(json.dumps({"exported_files": [str(path) for path in exported_files]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Example:
    # conda run -n torch310 python plotting/scripts/plot_requested_benchmark_figures.py --overwrite
    main()
