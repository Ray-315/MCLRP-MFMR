from __future__ import annotations

import argparse
import json
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
TICK_LABEL_SIZE = 8.6
LEGEND_SIZE = 8.2
FRAME_COLOR = "#334155"
GRID_COLOR = "#E6EDF4"
LIGHT_GRAY = "#CBD5E1"
TEXT_DARK = "#1F2937"

MODEL_ORDER = ["MCLRP", "MCLRP_MFMR"]
MODEL_DISPLAY = {
    "MCLRP": "MCLRP",
    "MCLRP_MFMR": "MCLRP-MFMR",
}
MODEL_COLORS = {
    "MCLRP": "#E8923F",
    "MCLRP_MFMR": "#4A7CFF",
}
ROOT_ORDER = ["CCLE", "GDSC", "CGP"]
SUBTASK_DISPLAY = {
    "CCLE": "CCLE",
    "GDSC-ERK_AUC": "GDSC ERK AUC",
    "GDSC-ERK_IC50": "GDSC ERK IC50",
    "GDSC-PI3K_AUC": "GDSC PI3K AUC",
    "GDSC-PI3K_IC50": "GDSC PI3K IC50",
    "CGP-ERK_AUC": "CGP ERK AUC",
    "CGP-ERK_IC50": "CGP ERK IC50",
    "CGP-PI3K_AUC": "CGP PI3K AUC",
    "CGP-PI3K_IC50": "CGP PI3K IC50",
}
SUBTASK_COLORS = {
    "CCLE": "#8E6BBE",
    "GDSC-ERK_AUC": "#2AA198",
    "GDSC-ERK_IC50": "#88D3C7",
    "GDSC-PI3K_AUC": "#E76F51",
    "GDSC-PI3K_IC50": "#F4A261",
    "CGP-ERK_AUC": "#2CA02C",
    "CGP-ERK_IC50": "#98DF8A",
    "CGP-PI3K_AUC": "#D4A017",
    "CGP-PI3K_IC50": "#F1C40F",
}
UPLIFT_AXIS_BASE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render composite Figure 4 per-drug comparison.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR), help="Project results root.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "plotting" / "outputs" / "main_figures" / "figure4_per_drug_comparison"),
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


def uplift_forward(values: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(values, dtype=float)
    transformed = np.sign(arr) * np.log1p(np.abs(arr) / UPLIFT_AXIS_BASE)
    if np.isscalar(values):
        return float(transformed)
    return transformed


def uplift_inverse(values: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(values, dtype=float)
    restored = np.sign(arr) * UPLIFT_AXIS_BASE * np.expm1(np.abs(arr))
    if np.isscalar(values):
        return float(restored)
    return restored


def uplift_ticks(lower: float, upper: float) -> list[float]:
    candidates = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    return [tick for tick in candidates if lower <= tick <= upper]


def load_ccle_bar_tables(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], str]:
    per_drug_path = results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "GLOBAL" / "all_per_drug_metrics.csv"
    work = pd.read_csv(require_file(per_drug_path))
    work = work.loc[(work["dataset_label"] == "CCLE") & (work["model"].isin(MODEL_ORDER))].copy()
    work["pcc"] = numeric_series(work["pcc"])
    work["scc"] = numeric_series(work["scc"])

    map_path = results_root / "tuning" / "mclrp_global_search" / "ccle_paper_drug_map.csv"
    if map_path.exists():
        drug_map = pd.read_csv(map_path)
        drug_order = drug_map.sort_values("drug_idx")["benchmark_drug"].tolist()
        label_lookup = dict(zip(drug_map["benchmark_drug"], drug_map["paper_drug"]))
        labels = [str(label_lookup.get(drug, drug)) for drug in drug_order]
        label_source = str(map_path)
    else:
        drug_order = sorted(work["drug"].drop_duplicates().tolist(), key=lambda x: int(str(x).split("_")[-1]))
        labels = [f"Drug{i}" for i in range(1, len(drug_order) + 1)]
        label_source = "fallback: Drug1~24 order"

    pcc_pivot = (
        work.pivot(index="drug", columns="model", values="pcc")
        .reindex(index=drug_order, columns=MODEL_ORDER)
        .astype(float)
    )
    scc_pivot = (
        work.pivot(index="drug", columns="model", values="scc")
        .reindex(index=drug_order, columns=MODEL_ORDER)
        .astype(float)
    )
    return pcc_pivot, scc_pivot, drug_order, labels, label_source


def load_uplift_table(results_root: Path) -> pd.DataFrame:
    uplift_path = results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "UPLIFT_MFMR_vs_MCLRP" / "all_task_per_drug_uplift.csv"
    work = pd.read_csv(require_file(uplift_path))
    work["delta_pcc"] = numeric_series(work["delta_pcc"])
    work["delta_scc"] = numeric_series(work["delta_scc"])
    return work


def draw_ccle_bar_panel(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    labels: list[str],
    metric_name: str,
    show_legend: bool,
) -> None:
    xpos = np.arange(len(labels), dtype=float)
    width = 0.38
    ax.bar(
        xpos - width / 2,
        pivot["MCLRP"].to_numpy(dtype=float),
        width=width,
        color=MODEL_COLORS["MCLRP"],
        edgecolor="white",
        linewidth=0.65,
        label=MODEL_DISPLAY["MCLRP"],
    )
    ax.bar(
        xpos + width / 2,
        pivot["MCLRP_MFMR"].to_numpy(dtype=float),
        width=width,
        color=MODEL_COLORS["MCLRP_MFMR"],
        edgecolor="white",
        linewidth=0.65,
        label=MODEL_DISPLAY["MCLRP_MFMR"],
    )
    values = pivot.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    lower = max(0.0, float(np.floor((values.min() - 0.05) * 10) / 10))
    upper = min(1.0, float(np.ceil((values.max() + 0.03) * 20) / 20))
    ax.set_ylim(lower, upper)
    ax.set_ylabel(metric_name)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=62, ha="right")
    ax.tick_params(axis="x", labelsize=8.0, pad=1.2)
    frame_axis(ax, grid_axis="y")
    if show_legend:
        handles = [
            Line2D([0], [0], marker="s", linestyle="", markersize=7.5, markerfacecolor=MODEL_COLORS[m], markeredgecolor="none", label=MODEL_DISPLAY[m])
            for m in MODEL_ORDER
        ]
        legend = ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor=LIGHT_GRAY,
            fancybox=False,
            borderpad=0.35,
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.35,
        )
        legend.get_frame().set_linewidth(0.9)


def draw_uplift_violin(
    ax: plt.Axes,
    uplift_df: pd.DataFrame,
    metric: str,
    show_legend: bool,
) -> None:
    y_col = f"delta_{metric}"
    box_color = "#C96C1A" if metric == "pcc" else "#305BCE"
    violin_fill = "#FAE4CF" if metric == "pcc" else "#DCE8FF"
    rng = np.random.default_rng(20260327)

    data_list = [
        uplift_df.loc[uplift_df["group"] == root, y_col].dropna().to_numpy(dtype=float)
        for root in ROOT_ORDER
    ]
    positions = np.arange(len(ROOT_ORDER), dtype=float)
    widths = [0.92, 0.62, 0.92]

    violins = ax.violinplot(
        data_list,
        positions=positions,
        widths=widths,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor(violin_fill)
        body.set_edgecolor(box_color)
        body.set_linewidth(1.0)
        body.set_alpha(0.65)

    ax.boxplot(
        data_list,
        positions=positions,
        widths=0.17,
        vert=True,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=box_color, linewidth=1.6),
        boxprops=dict(facecolor="white", edgecolor=box_color, linewidth=1.35),
        whiskerprops=dict(color=box_color, linewidth=1.15),
        capprops=dict(color=box_color, linewidth=1.15),
    )

    for pos, root in enumerate(ROOT_ORDER):
        block = uplift_df.loc[uplift_df["group"] == root].copy()
        jitter = rng.uniform(-0.15, 0.15, size=len(block))
        colors = [SUBTASK_COLORS[str(label)] for label in block["dataset_label"].tolist()]
        ax.scatter(
            np.full(len(block), pos, dtype=float) + jitter,
            block[y_col].to_numpy(dtype=float),
            s=19,
            c=colors,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.28,
            zorder=5,
        )

    ymin = min(float(np.min(vals)) for vals in data_list)
    ymax = max(float(np.max(vals)) for vals in data_list)
    pad = 0.08
    lower = float(np.floor((ymin - pad) * 20) / 20)
    upper = float(np.ceil((ymax + pad) * 20) / 20)
    ax.set_yscale("function", functions=(uplift_forward, uplift_inverse))
    ax.set_ylim(lower, upper)
    ticks = uplift_ticks(lower, upper)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])
    ax.axhline(0.0, color="black", linewidth=0.95, linestyle="--", zorder=1)
    ax.set_ylabel(f"Δ{metric.upper()}")
    ax.set_xticks(positions)
    ax.set_xticklabels(ROOT_ORDER)
    frame_axis(ax, grid_axis="y")

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5.6,
                markerfacecolor=SUBTASK_COLORS[key],
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=SUBTASK_DISPLAY[key],
            )
            for key in SUBTASK_DISPLAY
        ]
        legend = ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor=LIGHT_GRAY,
            fancybox=False,
            borderpad=0.28,
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.32,
        )
        legend.get_frame().set_linewidth(0.9)


def render_ccle_bar_panel(pivot: pd.DataFrame, labels: list[str], metric_name: str, output_base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(16.8, 4.1))
    draw_ccle_bar_panel(ax, pivot, labels, metric_name, show_legend=True)
    fig.tight_layout()
    return save_figure(fig, output_base)


def render_uplift_panel(uplift_df: pd.DataFrame, metric: str, output_base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.15, 4.1))
    draw_uplift_violin(ax, uplift_df, metric, show_legend=True)
    fig.tight_layout()
    return save_figure(fig, output_base)


def render_main_figure(
    ccle_pcc: pd.DataFrame,
    ccle_scc: pd.DataFrame,
    ccle_labels: list[str],
    uplift_df: pd.DataFrame,
    output_base: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(16.8, 13.8))
    grid = GridSpec(nrows=3, ncols=2, figure=fig, height_ratios=[1.0, 1.0, 1.0], hspace=0.36, wspace=0.22)

    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :])
    ax_c = fig.add_subplot(grid[2, 0])
    ax_d = fig.add_subplot(grid[2, 1])

    draw_ccle_bar_panel(ax_a, ccle_pcc, ccle_labels, "PCC", show_legend=True)
    draw_ccle_bar_panel(ax_b, ccle_scc, ccle_labels, "SCC", show_legend=True)
    draw_uplift_violin(ax_c, uplift_df, "pcc", show_legend=True)
    draw_uplift_violin(ax_d, uplift_df, "scc", show_legend=True)

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.subplots_adjust(top=0.988, bottom=0.05, left=0.055, right=0.985)
    return save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    set_publication_style()

    results_root = Path(args.results_root).resolve()
    output_dir = prepare_output_dir(Path(args.output_dir).resolve(), overwrite=bool(args.overwrite))
    panel_dir = output_dir / "panels"

    ccle_pcc, ccle_scc, ccle_order, ccle_labels, label_source = load_ccle_bar_tables(results_root)
    uplift_df = load_uplift_table(results_root)

    rendered_paths: list[Path] = []
    rendered_paths.extend(render_ccle_bar_panel(ccle_pcc, ccle_labels, "PCC", panel_dir / "panel_ccle_per_drug_pcc"))
    rendered_paths.extend(render_ccle_bar_panel(ccle_scc, ccle_labels, "SCC", panel_dir / "panel_ccle_per_drug_scc"))
    rendered_paths.extend(render_uplift_panel(uplift_df, "pcc", panel_dir / "panel_uplift_violin_pcc"))
    rendered_paths.extend(render_uplift_panel(uplift_df, "scc", panel_dir / "panel_uplift_violin_scc"))
    rendered_paths.extend(render_main_figure(ccle_pcc, ccle_scc, ccle_labels, uplift_df, output_dir / "figure4_main_composite"))

    manifest = {
        "figure": "figure4_per_drug_comparison",
        "plan_source": "Plot_Plan_and_Requests.docx / updated Figure 4 requirements",
        "rendered_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "ccle_source": str(results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "GLOBAL" / "all_per_drug_metrics.csv"),
        "uplift_source": str(results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv" / "UPLIFT_MFMR_vs_MCLRP" / "all_task_per_drug_uplift.csv"),
        "ccle_label_source": label_source,
        "ccle_drug_order": ccle_order,
        "ccle_display_labels": ccle_labels,
        "bottom_rule": "Violin panels use per task-drug uplift rows grouped by CCLE/GDSC/CGP root sets; scatter colors encode dataset_label subtasks; y-axis uses a log-like function scale around zero.",
        "rendered_files": [str(path) for path in rendered_paths],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rendered_files": [str(path) for path in rendered_paths]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
