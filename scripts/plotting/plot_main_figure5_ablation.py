from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


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
LEGEND_SIZE = 8.2
FRAME_COLOR = "#334155"
GRID_COLOR = "#E6EDF4"
LIGHT_GRAY = "#CBD5E1"
TEXT_DARK = "#1F2937"

PCC_BOX = "#C97A1B"
PCC_TREND = "#F0B46C"
PCC_FILL = "#FBE6D0"
PCC_SCATTER = "#4A7CFF"

SCC_BOX = "#335FC6"
SCC_TREND = "#A5C8E1"
SCC_FILL = "#DCE8FF"
SCC_SCATTER = "#E8923F"

SHARED_VARIANTS = [
    "A0_Original_MCLRP",
    "A1_ImpOnly",
    "A2_RidgeOnly",
    "A3_MFMR_Equal",
    "A4_MFMR_SingleView",
    "A5_MFMR_Full",
]
CGP_VARIANTS = [
    "C0_CGP_Base",
    "C1_CGP_TissueLatent",
    "C2_CGP_FullMinusPathway",
    "C4_CGP_FullMinusLatent",
    "C5_CGP_FullMinusInteraction",
    "C6_CGP_Full",
]
SHARED_GAIN_VARIANTS = [
    "A1_ImpOnly",
    "A2_RidgeOnly",
    "A3_MFMR_Equal",
    "A4_MFMR_SingleView",
    "A5_MFMR_Full",
]
MUTATION_GAIN_VARIANTS = [
    "C1_CGP_TissueLatent",
    "C2_CGP_FullMinusPathway",
    "C4_CGP_FullMinusLatent",
    "C5_CGP_FullMinusInteraction",
    "C6_CGP_Full",
]
GAIN_LABELS = {
    "A1_ImpOnly": "ImpOnly",
    "A2_RidgeOnly": "RidgeOnly",
    "A3_MFMR_Equal": "Equal",
    "A4_MFMR_SingleView": "Single\nView",
    "A5_MFMR_Full": "Shared\nFull",
    "C1_CGP_TissueLatent": "Tissue+\nLatent",
    "C2_CGP_FullMinusPathway": "Full-\nPathway",
    "C4_CGP_FullMinusLatent": "Full-\nLatent",
    "C5_CGP_FullMinusInteraction": "Full-\nInteract",
    "C6_CGP_Full": "CGP\nFull",
}
DONUT_LABELS = {
    "A0_Original_MCLRP": "Orig",
    "A1_ImpOnly": "Imp",
    "A2_RidgeOnly": "Ridge",
    "A3_MFMR_Equal": "Equal",
    "A4_MFMR_SingleView": "1View",
    "A5_MFMR_Full": "SFull",
    "C0_CGP_Base": "Base",
    "C1_CGP_TissueLatent": "T+L",
    "C2_CGP_FullMinusPathway": "-Path",
    "C4_CGP_FullMinusLatent": "-Lat",
    "C5_CGP_FullMinusInteraction": "-Int",
    "C6_CGP_Full": "CFull",
}
VARIANT_COLORS = {
    "A0_Original_MCLRP": "#B8C4CC",
    "A1_ImpOnly": "#6CC7BE",
    "A2_RidgeOnly": "#5B9BD5",
    "A3_MFMR_Equal": "#F1C40F",
    "A4_MFMR_SingleView": "#D989A3",
    "A5_MFMR_Full": "#E8923F",
    "C0_CGP_Base": "#98A6B3",
    "C1_CGP_TissueLatent": "#7B61FF",
    "C2_CGP_FullMinusPathway": "#75C46B",
    "C4_CGP_FullMinusLatent": "#6C91BF",
    "C5_CGP_FullMinusInteraction": "#C97A63",
    "C6_CGP_Full": "#4A7CFF",
}
ROOT_ORDER = ["CCLE", "GDSC", "CGP"]
ROOT_RING_RADII = {"CCLE": 1.00, "GDSC": 0.73, "CGP": 0.46}
RING_WIDTH = 0.20
HEATMAP_CMAP = sns.diverging_palette(15, 245, s=90, l=55, as_cmap=True)
SHARED_ROW_ORDER = ["CCLE", "ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"]
CGP_ROW_ORDER = ["ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"]
ROW_DISPLAY = {
    "CCLE": "CCLE",
    "ERK_AUC": "ERK AUC",
    "ERK_IC50": "ERK IC50",
    "PI3K_AUC": "PI3K AUC",
    "PI3K_IC50": "PI3K IC50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render composite Figure 5 ablation overview.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR), help="Project results root.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "plotting" / "outputs" / "main_figures" / "figure5_ablation"),
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


def compute_violin_ylim(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return (-0.05, 0.05)
    lower_data = float(finite_values.min())
    upper_data = float(finite_values.max())
    span = max(upper_data - lower_data, 1e-6)
    lower_pad = max(span * 0.07, 0.008)
    upper_pad = max(span * 0.18, 0.014 if upper_data < 0.10 else 0.040)
    lower = min(0.0, lower_data) - lower_pad
    upper = upper_data + upper_pad
    return (lower, upper)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    corr = spearmanr(y_true, y_pred).correlation
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def load_gain_table(results_root: Path) -> pd.DataFrame:
    fold_df = pd.read_csv(require_file(results_root / "latest" / "ablation_mclrp_mfmr" / "aggregate" / "all_fold_results.csv"))
    for col in ("pcc", "scc", "seed", "fold"):
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")

    baseline_ids = {
        "shared_main": "A0_Original_MCLRP",
        "cgp_main": "C0_CGP_Base",
    }
    base_rows = []
    for group, baseline_id in baseline_ids.items():
        base = fold_df.loc[fold_df["group"].eq(group) & fold_df["variant_id"].eq(baseline_id), ["group", "dataset", "seed", "fold", "pcc", "scc"]].copy()
        base = base.rename(columns={"pcc": "baseline_pcc", "scc": "baseline_scc"})
        base_rows.append(base)
    base_df = pd.concat(base_rows, ignore_index=True)

    work = fold_df.merge(base_df, on=["group", "dataset", "seed", "fold"], how="left")
    work = work.loc[work["variant_id"].isin(SHARED_GAIN_VARIANTS + MUTATION_GAIN_VARIANTS)].copy()
    work["gain_pcc"] = work["pcc"] - work["baseline_pcc"]
    work["gain_scc"] = work["scc"] - work["baseline_scc"]
    work["variant_display"] = work["variant_id"].map(GAIN_LABELS)
    return work.sort_values(["group", "variant_id", "dataset", "seed", "fold"]).reset_index(drop=True)


def load_ablation_summary_table(results_root: Path) -> pd.DataFrame:
    summary = pd.read_csv(require_file(results_root / "latest" / "ablation_mclrp_mfmr" / "aggregate" / "all_variant_summary.csv"))
    for col in ("mean_pcc", "mean_scc", "abs_gain_vs_full_pcc", "abs_gain_vs_full_scc"):
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary["dataset_key"] = summary["dataset"].astype(str)
    return summary


def build_heatmap_panel_table(summary_df: pd.DataFrame, *, group: str, metric: str) -> pd.DataFrame:
    if group == "shared_main":
        variant_order = SHARED_VARIANTS
        row_order = SHARED_ROW_ORDER
    else:
        variant_order = CGP_VARIANTS
        row_order = CGP_ROW_ORDER

    metric_col = f"mean_{metric}"
    delta_col = f"abs_gain_vs_full_{metric}"
    work = summary_df.loc[summary_df["group"].eq(group) & summary_df["variant_id"].isin(variant_order)].copy()
    work = work[["dataset_key", "variant_id", metric_col, delta_col]].copy()
    work.rename(columns={metric_col: "metric_value", delta_col: "delta_value"}, inplace=True)
    work["dataset_key"] = pd.Categorical(work["dataset_key"], categories=row_order, ordered=True)
    work["variant_id"] = pd.Categorical(work["variant_id"], categories=variant_order, ordered=True)
    work.sort_values(["dataset_key", "variant_id"], inplace=True)
    return work


def abbreviate_variant(variant_id: str) -> str:
    return DONUT_LABELS.get(variant_id, variant_id)


def draw_ablation_heatmap(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    *,
    group: str,
    title: str,
) -> None:
    value_matrix = panel_df.pivot(index="dataset_key", columns="variant_id", values="delta_value")
    metric_matrix = panel_df.pivot(index="dataset_key", columns="variant_id", values="metric_value")

    vmax = float(np.nanmax(np.abs(value_matrix.to_numpy(dtype=float))))
    vmax = max(vmax, 1e-4)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    annot = metric_matrix.astype(object).copy()
    for row_name in annot.index:
        for col_name in annot.columns:
            metric_value = metric_matrix.loc[row_name, col_name]
            delta_value = value_matrix.loc[row_name, col_name]
            if pd.isna(metric_value):
                annot.loc[row_name, col_name] = ""
            else:
                annot.loc[row_name, col_name] = f"{float(metric_value):.3f}\nd{float(delta_value):+.3f}"

    sns.heatmap(
        value_matrix,
        ax=ax,
        cmap=HEATMAP_CMAP,
        norm=norm,
        annot=annot.to_numpy(dtype=object),
        fmt="",
        linewidths=1.0,
        linecolor="white",
        cbar=False,
        annot_kws={"fontsize": 8.2, "color": TEXT_DARK, "ha": "center", "va": "center"},
    )

    ax.set_title(title, fontsize=11.1, pad=7.0)
    ax.set_xlabel("Ablation Variant")
    ax.set_ylabel("Task")
    ax.set_xticklabels([abbreviate_variant(str(label)) for label in value_matrix.columns], rotation=0, ha="center")
    ax.set_yticklabels([ROW_DISPLAY.get(str(label), str(label)) for label in value_matrix.index], rotation=0)

    full_variant = SHARED_VARIANTS[-1] if group == "shared_main" else CGP_VARIANTS[-1]
    if full_variant in value_matrix.columns:
        j = list(value_matrix.columns).index(full_variant)
        ax.add_patch(
            plt.Rectangle(
                (j, 0),
                1,
                value_matrix.shape[0],
                fill=False,
                edgecolor=FRAME_COLOR,
                linewidth=1.5,
                linestyle="-",
                clip_on=False,
            )
        )

    frame_axis(ax, grid_axis=None)
    ax.tick_params(length=0)
    ax.text(
        1.0,
        1.05,
        "Cell text: mean metric and d vs full",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        color="#475569",
    )


def load_ablation_per_drug_table(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ablation_root = results_root / "latest" / "ablation_mclrp_mfmr"

    ccle_bundle = load_ccle_standardized_bundle()
    ccle_truth = ccle_bundle.M.astype(np.float32)
    ccle_drugs = [str(label) for label in ccle_bundle.drug_labels.tolist()]
    for variant_id in SHARED_VARIANTS:
        pred_path = ablation_root / "shared_main" / "CCLE" / "predictions" / f"{variant_id}_mean_prediction.npz"
        pred = np.load(require_file(pred_path))["prediction"].astype(np.float32)
        for j, drug in enumerate(ccle_drugs):
            mask = ccle_truth[:, j] != 0
            rows.append(
                {
                    "root": "CCLE",
                    "dataset": "CCLE",
                    "drug_key": f"CCLE::{drug}",
                    "drug": drug,
                    "variant_id": variant_id,
                    "pcc": safe_corr(ccle_truth[mask, j], pred[mask, j]),
                    "scc": safe_spearman(ccle_truth[mask, j], pred[mask, j]),
                }
            )

    gdsc_cfg = GDSCBenchmarkConfig()
    for dataset_name in ("ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"):
        bundle = load_or_prepare_dataset(dataset_name, gdsc_cfg)
        truth = bundle.M.astype(np.float32)
        drugs = [str(label) for label in bundle.drug_labels.tolist()]
        for variant_id in SHARED_VARIANTS:
            pred_path = ablation_root / "shared_main" / dataset_name / "predictions" / f"{variant_id}_mean_prediction.npz"
            pred = np.load(require_file(pred_path))["prediction"].astype(np.float32)
            for j, drug in enumerate(drugs):
                mask = truth[:, j] != 0
                rows.append(
                    {
                        "root": "GDSC",
                        "dataset": dataset_name,
                        "drug_key": f"{dataset_name}::{drug}",
                        "drug": drug,
                        "variant_id": variant_id,
                        "pcc": safe_corr(truth[mask, j], pred[mask, j]),
                        "scc": safe_spearman(truth[mask, j], pred[mask, j]),
                    }
                )

    for dataset_name in ("ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"):
        bundle = load_cgp_dataset(dataset_name)
        truth = bundle.M.astype(np.float32)
        drugs = [str(label) for label in bundle.drug_labels.tolist()]
        for variant_id in CGP_VARIANTS:
            pred_path = ablation_root / "cgp_main" / dataset_name / "predictions" / f"{variant_id}_mean_prediction.npz"
            pred = np.load(require_file(pred_path))["prediction"].astype(np.float32)
            for j, drug in enumerate(drugs):
                mask = truth[:, j] != 0
                rows.append(
                    {
                        "root": "CGP",
                        "dataset": dataset_name,
                        "drug_key": f"{dataset_name}::{drug}",
                        "drug": drug,
                        "variant_id": variant_id,
                        "pcc": safe_corr(truth[mask, j], pred[mask, j]),
                        "scc": safe_spearman(truth[mask, j], pred[mask, j]),
                    }
                )

    return pd.DataFrame(rows)


def summarize_besthit_counts(per_drug_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for root in ROOT_ORDER:
        sdf = per_drug_df.loc[per_drug_df["root"].eq(root)].copy()
        order = SHARED_VARIANTS if root in {"CCLE", "GDSC"} else CGP_VARIANTS
        best = sdf.groupby("drug_key")[metric].transform("max")
        hits = sdf.loc[np.isclose(sdf[metric], best, atol=1e-10)].copy()
        counts = hits.groupby("variant_id").size().reindex(order, fill_value=0)
        total = max(1, int(counts.sum()))
        for variant_id, count in counts.items():
            records.append(
                {
                    "root": root,
                    "metric": metric,
                    "variant_id": variant_id,
                    "count": int(count),
                    "prop": float(count) / float(total),
                    "short_label": DONUT_LABELS[variant_id],
                }
            )
    return pd.DataFrame(records)


def draw_gain_violin(
    ax: plt.Axes,
    gain_df: pd.DataFrame,
    metric: str,
    variant_order: list[str],
    legend_loc: str = "upper right",
) -> None:
    y_col = "gain_pcc" if metric == "pcc" else "gain_scc"
    box_color = PCC_BOX if metric == "pcc" else SCC_BOX
    trend_color = PCC_TREND if metric == "pcc" else SCC_TREND
    violin_fill = PCC_FILL if metric == "pcc" else SCC_FILL
    scatter_color = PCC_SCATTER if metric == "pcc" else SCC_SCATTER

    positions = np.arange(len(variant_order), dtype=float)
    data_list = [
        gain_df.loc[gain_df["variant_id"].eq(variant_id), y_col].dropna().to_numpy(dtype=float)
        for variant_id in variant_order
    ]

    violins = ax.violinplot(
        data_list,
        positions=positions,
        widths=0.84,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor(violin_fill)
        body.set_edgecolor(box_color)
        body.set_linewidth(1.0)
        body.set_alpha(0.68)

    ax.boxplot(
        data_list,
        positions=positions,
        widths=0.15,
        vert=True,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=box_color, linewidth=1.55),
        boxprops=dict(facecolor="white", edgecolor=box_color, linewidth=1.25),
        whiskerprops=dict(color=box_color, linewidth=1.05),
        capprops=dict(color=box_color, linewidth=1.05),
    )

    rng = np.random.default_rng(20260327 if metric == "pcc" else 20260328)
    medians = []
    for pos, values in zip(positions, data_list):
        if values.size == 0:
            medians.append(np.nan)
            continue
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        ax.scatter(
            np.full(values.size, pos) + jitter,
            values,
            s=12,
            c=scatter_color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.22,
            zorder=5,
        )
        medians.append(float(np.median(values)))

    ax.plot(positions, medians, color=trend_color, linewidth=2.1, marker="o", markersize=4.2, zorder=6)
    ax.axhline(0.0, color="black", linewidth=0.95, linestyle="--", zorder=1)
    ax.set_ylabel("PCC Gain" if metric == "pcc" else "SCC Gain")
    ax.set_xticks(positions)
    ax.set_xticklabels([GAIN_LABELS[variant_id] for variant_id in variant_order], rotation=0, ha="center")
    ax.set_xlim(-0.55, len(variant_order) - 0.45)

    values = gain_df[y_col].dropna().to_numpy(dtype=float)
    lower, upper = compute_violin_ylim(values)
    ax.set_ylim(lower, upper)
    frame_axis(ax, grid_axis="y")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5.2, markerfacecolor=scatter_color, markeredgecolor="white", markeredgewidth=0.4, label="Fold points"),
        Line2D([0, 1], [0, 0], color=box_color, linewidth=1.8, label="Box / median"),
        Line2D([0, 1], [0, 0], color=trend_color, linewidth=2.1, marker="o", markersize=4.0, label="Median trend"),
    ]
    legend = ax.legend(
        handles=handles,
        loc=legend_loc,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRAY,
        fancybox=False,
        borderpad=0.32,
    )
    legend.get_frame().set_linewidth(0.9)
    ax.set_title("")


def format_percent(value: float) -> str:
    pct = value * 100.0
    if abs(pct - round(pct)) < 0.05:
        return f"{int(round(pct))}%"
    return f"{pct:.1f}%"


# ─── Ablation donut annotation configuration ──────────────────────────────
ABLATION_LABEL_THRESHOLD = 0.10
ABLATION_MAX_RING_LABELS = 1
ABLATION_SIDE_TOP_N = 3


def _distribute_y_fig5(targets: list[float], lower: float, upper: float, gap: float) -> list[float]:
    """Space y-positions vertically without overlaps."""
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


def annotate_ring_top_only(
    ax: plt.Axes, wedges, values: np.ndarray, labels: list[str],
    radius: float, text_col_x: float = 1.18,
    threshold: float = ABLATION_LABEL_THRESHOLD,
    max_k: int = ABLATION_MAX_RING_LABELS,
) -> None:
    """Annotate only the single largest segment of a ring with a short leader line."""
    total = float(values.sum())
    if total <= 0:
        return
    props = values / total
    eligible = [(i, float(props[i]), labels[i]) for i in range(len(values))
                if float(props[i]) >= threshold and values[i] > 0]
    eligible.sort(key=lambda x: -x[1])
    eligible = eligible[:max_k]
    if not eligible:
        return
    for idx, prop, label in eligible:
        if idx >= len(wedges):
            continue
        wedge = wedges[idx]
        theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        outer_r = radius + 0.04
        x_start = np.cos(theta) * outer_r
        y_start = np.sin(theta) * outer_r
        elbow_x = text_col_x - 0.05
        ax.plot(
            [x_start, elbow_x, text_col_x],
            [y_start, y_start, y_start],
            color=FRAME_COLOR, linewidth=0.55, solid_capstyle="round", clip_on=False,
        )
        ax.text(
            text_col_x + 0.02, y_start,
            f"{label} {prop * 100:.1f}%",
            ha="left", va="center", fontsize=8.1, color=TEXT_DARK, clip_on=False,
        )


def draw_side_annotation_list(
    ax: plt.Axes, besthit_df: pd.DataFrame, metric: str,
    x_start: float = 0.62, y_start: float = 0.97,
    top_n: int = ABLATION_SIDE_TOP_N,
) -> None:
    """Draw grouped side annotation list to the right of the donut chart.

    Format:
        CCLE
          Orig   12.5%
          SFull  16.7%
        GDSC
          Equal  41.0%
          ...
    """
    y = y_start
    line_h = 0.058
    group_gap = 0.030
    root_colors = {"CCLE": "#8E6BBE", "GDSC": "#4A7CFF", "CGP": "#E8923F"}

    for root in ROOT_ORDER:
        ring_df = besthit_df.loc[
            (besthit_df["metric"].eq(metric)) & (besthit_df["root"].eq(root))
        ].copy()
        if ring_df.empty:
            continue
        total = max(1, int(ring_df["count"].sum()))
        top = ring_df.nlargest(top_n, "prop")
        # Group header
        ax.text(
            x_start, y, root, transform=ax.transAxes,
            ha="left", va="top", fontsize=8.2, fontweight="bold",
            color=root_colors.get(root, TEXT_DARK),
        )
        y -= line_h
        for _, row in top.iterrows():
            if row["count"] <= 0:
                continue
            pct = row["prop"] * 100.0
            ax.text(
                x_start + 0.02, y,
                f"{row['short_label']:>6s}  {pct:5.1f}%",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.0, color=TEXT_DARK, family="monospace",
            )
            y -= line_h * 0.88
        y -= group_gap


def draw_root_callouts(ax: plt.Axes) -> None:
    """Simplified left-side ring identification – short lines, bold labels."""
    callout_specs = [
        ("CCLE", [-0.10, -0.34, -1.08], [0.98, 1.08, 1.08]),
        ("GDSC", [-0.04, -0.30, -1.08], [0.72, 0.78, 0.78]),
        ("CGP",  [0.02, -0.24, -1.08], [0.45, 0.46, 0.46]),
    ]
    for label, xs, ys in callout_specs:
        ax.plot(xs, ys, color="black", linewidth=0.85, clip_on=False)
        ax.text(
            -1.14, ys[-1], label,
            ha="right", va="center", fontsize=8.2,
            color="black", fontweight="bold", clip_on=False,
        )


def draw_besthit_donut(ax: plt.Axes, besthit_df: pd.DataFrame, metric: str) -> None:
    """Clean triple-ring donut: rings show structure, side annotation shows numbers."""
    for root in ROOT_ORDER:
        ring_df = besthit_df.loc[
            (besthit_df["metric"].eq(metric)) & (besthit_df["root"].eq(root))
        ].copy()
        values = ring_df["count"].to_numpy(dtype=float)
        labels = ring_df["short_label"].tolist()
        colors = [VARIANT_COLORS[vid] for vid in ring_df["variant_id"].tolist()]
        radius = ROOT_RING_RADII[root]
        wedges, _ = ax.pie(
            values, radius=radius, startangle=90, counterclock=False,
            colors=colors,
            wedgeprops=dict(width=RING_WIDTH, edgecolor="white", linewidth=1.05),
        )
        # Only annotate top-1 segment per ring directly on chart
        annotate_ring_top_only(ax, wedges, values, labels, radius=radius, text_col_x=1.18)

    ax.text(0.0, 0.0, f"Best\n{metric.upper()}", ha="center", va="center",
            fontsize=11.0, fontweight="bold")
    draw_root_callouts(ax)
    # Right-side grouped annotation list replaces dense leader lines
    draw_side_annotation_list(ax, besthit_df, metric, x_start=0.62, y_start=0.97)

    ax.set_aspect("equal")
    ax.set_xlim(-1.82, 1.82)
    ax.set_ylim(-1.40, 1.40)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.1)


def render_gain_panel(
    gain_df: pd.DataFrame,
    metric: str,
    variant_order: list[str],
    output_base: Path,
    legend_loc: str = "upper right",
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.15, 4.1))
    draw_gain_violin(ax, gain_df, metric, variant_order=variant_order, legend_loc=legend_loc)
    fig.tight_layout()
    return save_figure(fig, output_base)


def render_heatmap_panel(
    panel_df: pd.DataFrame,
    *,
    group: str,
    title: str,
    output_base: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.35, 4.65))
    draw_ablation_heatmap(ax, panel_df, group=group, title=title)
    fig.tight_layout()
    return save_figure(fig, output_base)


def render_donut_panel(besthit_df: pd.DataFrame, metric: str, output_base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.15, 4.1))
    draw_besthit_donut(ax, besthit_df, metric)
    fig.tight_layout()
    return save_figure(fig, output_base)


def render_main_figure(summary_df: pd.DataFrame, output_base: Path) -> list[Path]:
    fig = plt.figure(figsize=(17.2, 10.2))
    grid = GridSpec(nrows=2, ncols=2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.24)

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_ablation_heatmap(
        ax_a,
        build_heatmap_panel_table(summary_df, group="shared_main", metric="pcc"),
        group="shared_main",
        title="Shared Backbone Ablation | Mean PCC",
    )
    draw_ablation_heatmap(
        ax_b,
        build_heatmap_panel_table(summary_df, group="shared_main", metric="scc"),
        group="shared_main",
        title="Shared Backbone Ablation | Mean SCC",
    )
    draw_ablation_heatmap(
        ax_c,
        build_heatmap_panel_table(summary_df, group="cgp_main", metric="pcc"),
        group="cgp_main",
        title="Mutation Head Ablation | Mean PCC",
    )
    draw_ablation_heatmap(
        ax_d,
        build_heatmap_panel_table(summary_df, group="cgp_main", metric="scc"),
        group="cgp_main",
        title="Mutation Head Ablation | Mean SCC",
    )

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle("Ablation Heatmap Overview", fontsize=12.0, y=0.985, fontweight="normal")
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.06, right=0.985)
    return save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    set_publication_style()

    results_root = Path(args.results_root).resolve()
    output_dir = prepare_output_dir(Path(args.output_dir).resolve(), overwrite=bool(args.overwrite))
    panel_dir = output_dir / "panels"

    gain_df = load_gain_table(results_root)
    summary_df = load_ablation_summary_table(results_root)
    gain_df.to_csv(output_dir / "figure5_gain_fold_table.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "figure5_variant_summary_table.csv", index=False, encoding="utf-8-sig")

    shared_pcc_df = build_heatmap_panel_table(summary_df, group="shared_main", metric="pcc")
    shared_scc_df = build_heatmap_panel_table(summary_df, group="shared_main", metric="scc")
    mutation_pcc_df = build_heatmap_panel_table(summary_df, group="cgp_main", metric="pcc")
    mutation_scc_df = build_heatmap_panel_table(summary_df, group="cgp_main", metric="scc")
    shared_pcc_df.to_csv(output_dir / "figure5_shared_pcc_heatmap_table.csv", index=False, encoding="utf-8-sig")
    shared_scc_df.to_csv(output_dir / "figure5_shared_scc_heatmap_table.csv", index=False, encoding="utf-8-sig")
    mutation_pcc_df.to_csv(output_dir / "figure5_mutation_pcc_heatmap_table.csv", index=False, encoding="utf-8-sig")
    mutation_scc_df.to_csv(output_dir / "figure5_mutation_scc_heatmap_table.csv", index=False, encoding="utf-8-sig")

    rendered_paths: list[Path] = []
    rendered_paths.extend(render_heatmap_panel(shared_pcc_df, group="shared_main", title="Shared Backbone Ablation | Mean PCC", output_base=panel_dir / "panel_ablation_shared_pcc"))
    rendered_paths.extend(render_heatmap_panel(shared_scc_df, group="shared_main", title="Shared Backbone Ablation | Mean SCC", output_base=panel_dir / "panel_ablation_shared_scc"))
    rendered_paths.extend(render_heatmap_panel(mutation_pcc_df, group="cgp_main", title="Mutation Head Ablation | Mean PCC", output_base=panel_dir / "panel_ablation_mutation_pcc"))
    rendered_paths.extend(render_heatmap_panel(mutation_scc_df, group="cgp_main", title="Mutation Head Ablation | Mean SCC", output_base=panel_dir / "panel_ablation_mutation_scc"))
    rendered_paths.extend(render_main_figure(summary_df, output_dir / "figure5_main_composite"))

    manifest = {
        "figure": "figure5_ablation",
        "plan_source": "Plot_Plan_and_Requests.docx / Figure 5 ablation requirements",
        "rendered_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "gain_source": str(results_root / "latest" / "ablation_mclrp_mfmr" / "aggregate" / "all_fold_results.csv"),
        "summary_source": str(results_root / "latest" / "ablation_mclrp_mfmr" / "aggregate" / "all_variant_summary.csv"),
        "gain_rule": "Heatmap color encodes delta versus the full model within each ablation family; cell text shows mean metric and delta versus the full model.",
        "layout_rule": "2x2 heatmap layout with shared backbone PCC/SCC on the top row and mutation-head PCC/SCC on the bottom row; rows are tasks and columns are ablation variants.",
        "shared_gain_variants": SHARED_GAIN_VARIANTS,
        "mutation_gain_variants": MUTATION_GAIN_VARIANTS,
        "rendered_files": [str(path) for path in rendered_paths],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rendered_files": [str(path) for path in rendered_paths]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
