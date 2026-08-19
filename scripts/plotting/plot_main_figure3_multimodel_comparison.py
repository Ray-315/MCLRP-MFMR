from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
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


FIG_DPI = 600
FONT_FAMILY = "DejaVu Sans"
AXIS_LABEL_SIZE = 10.5
TICK_LABEL_SIZE = 8.8
LEGEND_SIZE = 8.4
FRAME_COLOR = "#334155"
GRID_COLOR = "#E6EDF4"
LIGHT_GRAY = "#CBD5E1"
TEXT_DARK = "#1F2937"

MODEL_ORDER = ["MC", "RR", "MCRR", "SRMF", "DeepIC50", "GeneVAE", "MCLRP", "MCLRP_MFMR"]
MODEL_DISPLAY = {
    "MC": "MC",
    "RR": "RR",
    "MCRR": "MCRR",
    "SRMF": "SRMF",
    "DeepIC50": "DeepIC50",
    "GeneVAE": "GeneVAE",
    "MCLRP": "MCLRP",
    "MCLRP_MFMR": "MCLRP-MFMR",
}
MODEL_COLORS = {
    "MC": "#7ED957",
    "RR": "#C9D6DC",
    "MCRR": "#F1C40F",
    "SRMF": "#6CB4EE",
    "DeepIC50": "#C97A63",
    "GeneVAE": "#8E6BBE",
    "MCLRP": "#F4A261",
    "MCLRP_MFMR": "#4A7CFF",
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
ROOT_ORDER = ["CCLE", "GDSC", "CGP"]
DATASET_NOTE_TEXT = {
    "ALL": "All drugs",
    "CCLE": "CCLE set",
    "GDSC": "GDSC tasks",
    "CGP": "CGP tasks",
}
DATASET_NOTE_COLORS = {
    "ALL": "#475569",
    "CCLE": "#8E6BBE",
    "GDSC": "#4A7CFF",
    "CGP": "#E8923F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render composite Figure 3 multi-model comparison.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR), help="Project results root.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "plotting" / "outputs" / "main_figures" / "figure3_multimodel_comparison"),
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
    saved_paths: list[Path] = []
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


def load_tables(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_root = results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "GLOBAL"
    task_df = pd.read_csv(require_file(global_root / "dataset_task_model_summary.csv"))
    per_drug_df = pd.read_csv(require_file(global_root / "all_per_drug_metrics.csv"))
    return task_df, per_drug_df


def prepare_task_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    work = task_df.copy()
    work["dataset_label"] = pd.Categorical(work["dataset_label"], categories=SUBTASK_ORDER, ordered=True)
    work["mean_pcc"] = numeric_series(work["mean_pcc"])
    work["mean_scc"] = numeric_series(work["mean_scc"])
    work = work.loc[work["model"].isin(MODEL_ORDER)].copy()
    return work.sort_values(["dataset_label", "model"]).reset_index(drop=True)


def prepare_per_drug(per_drug_df: pd.DataFrame) -> pd.DataFrame:
    work = per_drug_df.copy()
    work["pcc"] = numeric_series(work["pcc"])
    work["scc"] = numeric_series(work["scc"])
    work = work.loc[work["model"].isin(MODEL_ORDER)].copy()
    return work


def count_best_hits(work: pd.DataFrame, group_cols: list[str], metric: str = "pcc") -> pd.DataFrame:
    best = work.groupby(group_cols)[metric].transform("max")
    hits = work.loc[work[metric].eq(best)].copy()
    counts = hits.groupby("model").size().reindex(MODEL_ORDER, fill_value=0).reset_index(name="count")
    counts["prop"] = counts["count"] / max(1, counts["count"].sum())
    return counts


def draw_grouped_bar_panel(ax: plt.Axes, work: pd.DataFrame, metric: str) -> None:
    pivot = (
        work.pivot(index="dataset_label", columns="model", values=metric)
        .reindex(index=SUBTASK_ORDER, columns=MODEL_ORDER)
        .astype(float)
    )
    n_models = len(MODEL_ORDER)
    xpos = np.arange(len(SUBTASK_ORDER), dtype=float)
    total_width = 0.86
    bar_w = total_width / n_models
    left = xpos - total_width / 2 + bar_w / 2
    for idx, model in enumerate(MODEL_ORDER):
        ax.bar(
            left + idx * bar_w,
            pivot[model].to_numpy(dtype=float),
            width=bar_w * 0.92,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.55,
            label=MODEL_DISPLAY[model],
        )
    values = pivot.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    lower = max(0.15, float(np.floor((values.min() - 0.05) * 10) / 10))
    upper = min(1.02, float(np.ceil((values.max() + 0.05) * 20) / 20))
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Mean PCC" if metric == "mean_pcc" else "Mean SCC")
    ax.set_xticks(xpos)
    ax.set_xticklabels([SUBTASK_SHORT_LABELS[x] for x in SUBTASK_ORDER], rotation=0, ha="center")
    frame_axis(ax, grid_axis="y")
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=8.0, markerfacecolor=MODEL_COLORS[m], markeredgecolor="none", label=MODEL_DISPLAY[m]) for m in MODEL_ORDER]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRAY,
        fancybox=False,
        borderpad=0.35,
        ncol=4,
        columnspacing=0.75,
        handletextpad=0.35,
    )
    legend.get_frame().set_linewidth(0.9)


# ─── Donut annotation configuration ──────────────────────────────────────
DONUT_LABEL_THRESHOLD = 0.08
MAX_LABELS_PER_RING = 2
MAX_LABELS_PER_DONUT = 3
DONUT_TARGET_MODELS = ["MCLRP", "MCLRP_MFMR"]
DONUT_CENTER = (-0.52, 0.0)
DONUT_OUTER_RADIUS = 1.12
DONUT_INNER_RADIUS = 0.76


def select_major_segments(counts_df: pd.DataFrame, threshold: float = DONUT_LABEL_THRESHOLD,
                          max_k: int = MAX_LABELS_PER_RING) -> pd.DataFrame:
    """Return rows eligible for external labeling: above *threshold*, capped at *max_k*."""
    eligible = counts_df.loc[counts_df["prop"] >= threshold].copy()
    if len(eligible) > max_k:
        eligible = eligible.nlargest(max_k, "prop")
    return eligible


def _distribute_y(targets: list[float], lower: float, upper: float, gap: float) -> list[float]:
    """Space label y-positions into a neat non-overlapping column."""
    if not targets:
        return []
    n = len(targets)
    order = sorted(range(n), key=lambda i: targets[i])
    placed = [0.0] * n
    cursor = lower - gap
    for idx in order:
        placed[idx] = min(upper, max(targets[idx], cursor + gap))
        cursor = placed[idx]
    if n > 0 and placed[order[-1]] > upper:
        shift = placed[order[-1]] - upper
        placed = [max(lower, y - shift) for y in placed]
    for rank in range(1, len(order)):
        prev_i, curr_i = order[rank - 1], order[rank]
        if placed[curr_i] < placed[prev_i] + gap:
            placed[curr_i] = placed[prev_i] + gap
    return placed


def _segment_anchor(
    wedges,
    counts_df: pd.DataFrame,
    model: str,
    radius: float,
    fallback_angle_deg: float,
) -> tuple[float, float]:
    row = counts_df.loc[counts_df["model"].eq(model)]
    theta_deg = fallback_angle_deg
    if not row.empty:
        row_idx = int(row.index[0])
        if row_idx < len(wedges):
            wedge = wedges[row_idx]
            if float(row.iloc[0]["prop"]) > 0.0 and abs(float(wedge.theta2) - float(wedge.theta1)) > 1e-6:
                theta_deg = 0.5 * (float(wedge.theta1) + float(wedge.theta2))
    theta = math.radians(theta_deg)
    outer_r = radius + 0.05
    return math.cos(theta) * outer_r, math.sin(theta) * outer_r


def annotate_target_model_shares(
    ax: plt.Axes,
    wedges_outer,
    wedges_inner,
    pcc_counts_df: pd.DataFrame,
    scc_counts_df: pd.DataFrame,
) -> None:
    label_specs = [
        {
            "counts_df": pcc_counts_df,
            "wedges": wedges_outer,
            "radius": 1.0,
            "model": "MCLRP",
            "metric": "PCC",
            "side": "right",
            "text_x": 1.62,
            "text_y": 0.82,
            "elbow_x": 1.45,
            "fallback_angle_deg": 25.0,
        },
        {
            "counts_df": pcc_counts_df,
            "wedges": wedges_outer,
            "radius": 1.0,
            "model": "MCLRP_MFMR",
            "metric": "PCC",
            "side": "right",
            "text_x": 1.62,
            "text_y": -0.84,
            "elbow_x": 1.45,
            "fallback_angle_deg": -25.0,
        },
        {
            "counts_df": scc_counts_df,
            "wedges": wedges_inner,
            "radius": 0.68,
            "model": "MCLRP",
            "metric": "SCC",
            "side": "left",
            "text_x": -1.62,
            "text_y": 0.62,
            "elbow_x": -1.45,
            "fallback_angle_deg": 155.0,
        },
        {
            "counts_df": scc_counts_df,
            "wedges": wedges_inner,
            "radius": 0.68,
            "model": "MCLRP_MFMR",
            "metric": "SCC",
            "side": "left",
            "text_x": -1.62,
            "text_y": -0.68,
            "elbow_x": -1.45,
            "fallback_angle_deg": 205.0,
        },
    ]
    for spec in label_specs:
        row = spec["counts_df"].loc[spec["counts_df"]["model"].eq(spec["model"])]
        if row.empty:
            continue
        share = float(row.iloc[0]["prop"]) * 100.0
        x_start, y_start = _segment_anchor(
            spec["wedges"],
            spec["counts_df"],
            spec["model"],
            radius=spec["radius"],
            fallback_angle_deg=spec["fallback_angle_deg"],
        )
        ax.plot(
            [x_start, spec["elbow_x"], spec["text_x"]],
            [y_start, spec["text_y"], spec["text_y"]],
            color=FRAME_COLOR,
            linewidth=0.62,
            solid_capstyle="round",
            clip_on=False,
        )
        ha = "left" if spec["side"] == "right" else "right"
        text_pad = 0.04 if spec["side"] == "right" else -0.04
        ax.text(
            spec["text_x"] + text_pad,
            spec["text_y"],
            f"{MODEL_DISPLAY[spec['model']]} ({spec['metric']}) {share:.1f}%",
            ha=ha,
            va="center",
            fontsize=8.2,
            color=TEXT_DARK,
            clip_on=False,
        )


def draw_donut_model_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=6.6,
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="none",
            label=MODEL_DISPLAY[m],
        )
        for m in MODEL_ORDER
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRAY,
        fancybox=False,
        borderpad=0.28,
        ncol=1,
        columnspacing=0.55,
        handletextpad=0.32,
        borderaxespad=0.18,
        fontsize=8.0,
    )
    legend.get_frame().set_linewidth(0.85)


def annotate_major_segments(
    ax: plt.Axes,
    wedges,
    counts_df: pd.DataFrame,
    radius: float,
    text_col_x: float = 1.22,
    y_bounds: tuple[float, float] = (-0.88, 0.88),
    gap: float = 0.18,
    threshold: float = DONUT_LABEL_THRESHOLD,
    max_k: int = MAX_LABELS_PER_DONUT,
) -> int:
    """Annotate only major segments with short leader lines to a fixed text column.

    Returns the number of labels drawn.
    """
    major = select_major_segments(counts_df, threshold=threshold, max_k=max_k)
    if major.empty:
        return 0
    items: list[dict] = []
    for row_idx, row in major.iterrows():
        if int(row_idx) >= len(wedges):
            continue
        wedge = wedges[int(row_idx)]
        theta = math.radians(0.5 * (wedge.theta1 + wedge.theta2))
        outer_r = radius + 0.05
        items.append({
            "x_start": math.cos(theta) * outer_r,
            "y_start": math.sin(theta) * outer_r,
            "target_y": math.sin(theta) * outer_r,
            "label": f"{MODEL_DISPLAY.get(str(row['model']), str(row['model']))} {row['prop'] * 100:.1f}%",
        })
    if not items:
        return 0
    placed_ys = _distribute_y([it["target_y"] for it in items], y_bounds[0], y_bounds[1], gap)
    elbow_x = text_col_x - 0.06
    for item, text_y in zip(items, placed_ys):
        ax.plot(
            [item["x_start"], elbow_x, text_col_x],
            [item["y_start"], text_y, text_y],
            color=FRAME_COLOR, linewidth=0.60, solid_capstyle="round", clip_on=False,
        )
        ax.text(
            text_col_x + 0.03, text_y, item["label"],
            ha="left", va="center", fontsize=8.0, color=TEXT_DARK, clip_on=False,
        )
    return len(items)


def draw_multiring_donut(
    ax: plt.Axes,
    pcc_counts_df: pd.DataFrame,
    scc_counts_df: pd.DataFrame,
    center_label: str,
    note_label: str | None = None,
) -> None:
    pcc_colors = [MODEL_COLORS[m] for m in pcc_counts_df["model"]]
    scc_colors = [MODEL_COLORS[m] for m in scc_counts_df["model"]]
    pcc_values = pcc_counts_df["count"].to_numpy(dtype=float)
    scc_values = scc_counts_df["count"].to_numpy(dtype=float)
    wedges_outer, _ = ax.pie(
        pcc_values, radius=DONUT_OUTER_RADIUS, center=DONUT_CENTER, colors=pcc_colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=1.0),
    )
    wedges_inner, _ = ax.pie(
        scc_values, radius=DONUT_INNER_RADIUS, center=DONUT_CENTER, colors=scc_colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=1.0),
    )
    ax.text(DONUT_CENTER[0], DONUT_CENTER[1], center_label, ha="center", va="center",
            fontsize=10.0, color=TEXT_DARK, fontweight="bold")
    draw_donut_model_legend(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.92, 2.18)
    ax.set_ylim(-1.26, 1.26)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.1)


def render_panel_bar(work: pd.DataFrame, metric: str, output_base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(16.6, 3.8))
    draw_grouped_bar_panel(ax, work, metric)
    fig.tight_layout()
    return save_figure(fig, output_base)


def render_panel_donuts(
    global_counts_pcc: pd.DataFrame,
    global_counts_scc: pd.DataFrame,
    root_counts_pcc: dict[str, pd.DataFrame],
    root_counts_scc: dict[str, pd.DataFrame],
    output_base: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(16.6, 3.8))
    draw_multiring_donut(axes[0], global_counts_pcc, global_counts_scc, "ALL", note_label="ALL")
    draw_multiring_donut(axes[1], root_counts_pcc["CCLE"], root_counts_scc["CCLE"], "CCLE", note_label="CCLE")
    draw_multiring_donut(axes[2], root_counts_pcc["GDSC"], root_counts_scc["GDSC"], "GDSC", note_label="GDSC")
    draw_multiring_donut(axes[3], root_counts_pcc["CGP"], root_counts_scc["CGP"], "CGP", note_label="CGP")
    fig.tight_layout(w_pad=1.1)
    return save_figure(fig, output_base)


def render_main_figure(
    work: pd.DataFrame,
    global_counts_pcc: pd.DataFrame,
    global_counts_scc: pd.DataFrame,
    root_counts_pcc: dict[str, pd.DataFrame],
    root_counts_scc: dict[str, pd.DataFrame],
    output_base: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(16.8, 14.0))
    grid = GridSpec(nrows=3, ncols=4, figure=fig, height_ratios=[1.0, 1.0, 1.0], hspace=0.36, wspace=0.38)
    ax_top = fig.add_subplot(grid[0, :])
    ax_mid = fig.add_subplot(grid[1, :])
    donut_axes = [fig.add_subplot(grid[2, i]) for i in range(4)]

    draw_grouped_bar_panel(ax_top, work, "mean_pcc")
    draw_grouped_bar_panel(ax_mid, work, "mean_scc")
    draw_multiring_donut(donut_axes[0], global_counts_pcc, global_counts_scc, "ALL", note_label="ALL")
    draw_multiring_donut(donut_axes[1], root_counts_pcc["CCLE"], root_counts_scc["CCLE"], "CCLE", note_label="CCLE")
    draw_multiring_donut(donut_axes[2], root_counts_pcc["GDSC"], root_counts_scc["GDSC"], "GDSC", note_label="GDSC")
    draw_multiring_donut(donut_axes[3], root_counts_pcc["CGP"], root_counts_scc["CGP"], "CGP", note_label="CGP")

    panel_axes = [ax_top, ax_mid, *donut_axes]
    for label, ax in zip(list("ABCDEF"), panel_axes):
        add_panel_label(ax, label)

    fig.subplots_adjust(top=0.985, bottom=0.045, left=0.045, right=0.985)
    return save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    set_publication_style()

    results_root = Path(args.results_root).resolve()
    output_dir = prepare_output_dir(Path(args.output_dir).resolve(), overwrite=bool(args.overwrite))
    panel_dir = output_dir / "panels"

    task_df, per_drug_df = load_tables(results_root)
    task_df = prepare_task_summary(task_df)
    per_drug_df = prepare_per_drug(per_drug_df)

    global_counts_pcc = count_best_hits(per_drug_df, ["dataset_label", "drug"], metric="pcc")
    global_counts_scc = count_best_hits(per_drug_df, ["dataset_label", "drug"], metric="scc")
    root_counts_pcc = {root: count_best_hits(per_drug_df.loc[per_drug_df["group"] == root].copy(), ["dataset_label", "drug"], metric="pcc") for root in ROOT_ORDER}
    root_counts_scc = {root: count_best_hits(per_drug_df.loc[per_drug_df["group"] == root].copy(), ["dataset_label", "drug"], metric="scc") for root in ROOT_ORDER}

    rendered_paths: list[Path] = []
    rendered_paths.extend(render_panel_bar(task_df, "mean_pcc", panel_dir / "panel_global_8model_mean_pcc"))
    rendered_paths.extend(render_panel_bar(task_df, "mean_scc", panel_dir / "panel_global_8model_mean_scc"))
    rendered_paths.extend(render_panel_donuts(global_counts_pcc, global_counts_scc, root_counts_pcc, root_counts_scc, panel_dir / "panel_best_share_donuts"))
    rendered_paths.extend(render_main_figure(task_df, global_counts_pcc, global_counts_scc, root_counts_pcc, root_counts_scc, output_dir / "figure3_main_composite"))

    manifest = {
        "figure": "figure3_multimodel_comparison",
        "plan_source": "Plot_Plan_and_Requests.docx / Figure 3",
        "rendered_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "task_summary_source": str(results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "GLOBAL" / "dataset_task_model_summary.csv"),
        "per_drug_source": str(results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "GLOBAL" / "all_per_drug_metrics.csv"),
        "global_best_pcc_counts": global_counts_pcc.to_dict(orient="records"),
        "global_best_scc_counts": global_counts_scc.to_dict(orient="records"),
        "root_best_pcc_counts": {root: root_counts_pcc[root].to_dict(orient="records") for root in ROOT_ORDER},
        "root_best_scc_counts": {root: root_counts_scc[root].to_dict(orient="records") for root in ROOT_ORDER},
        "rendered_files": [str(path) for path in rendered_paths],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rendered_files": [str(path) for path in rendered_paths]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
