from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
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


PRIMARY_BLUE = "#1D4ED8"
DARK_BLUE = "#1E3A8A"
MID_BLUE = "#3B82F6"
LIGHT_BLUE = "#93C5FD"
PALE_BLUE = "#DBEAFE"
SLATE = "#64748B"
LIGHT_SLATE = "#CBD5E1"
LIGHT_GRAY = "#E5E7EB"
DARK_GRAY = "#334155"

FIG_DPI = 600
FONT_FAMILY = "DejaVu Sans"
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 9
TITLE_SIZE = 13
ANNOT_SIZE = 8
LINE_WIDTH = 1.8
BAR_EDGE_WIDTH = 0.7
MARKER_SIZE = 34

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

ABLATION_DISPLAY_NAMES = {
    "A0_Original_MCLRP": "Original",
    "A1_ImpOnly": "ImpOnly",
    "A2_RidgeOnly": "RidgeOnly",
    "A3_MFMR_Equal": "Equal",
    "A4_MFMR_SingleView": "SingleView",
    "A5_MFMR_Full": "Full",
    "C0_CGP_Base": "Base",
    "C1_CGP_TissueLatent": "Tissue+Latent",
    "C2_CGP_FullMinusPathway": "Full-Pathway",
    "C4_CGP_FullMinusLatent": "Full-Latent",
    "C5_CGP_FullMinusInteraction": "Full-Interact",
    "C6_CGP_Full": "Full",
}

SHARED_ABLATION_ORDER = [
    "A0_Original_MCLRP",
    "A1_ImpOnly",
    "A2_RidgeOnly",
    "A3_MFMR_Equal",
    "A4_MFMR_SingleView",
    "A5_MFMR_Full",
]

CGP_ABLATION_ORDER = [
    "C0_CGP_Base",
    "C1_CGP_TissueLatent",
    "C2_CGP_FullMinusPathway",
    "C4_CGP_FullMinusLatent",
    "C5_CGP_FullMinusInteraction",
    "C6_CGP_Full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready result figures for the MCLRP bioinformatics manuscript."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Root directory containing experiment result files. Defaults to project results/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "plotting" / "outputs" / "result_figures"),
        help="Directory for exported figure files.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=("both", "pdf", "png"),
        help="Export format for figures.",
    )
    parser.add_argument(
        "--per_drug_top_n",
        type=int,
        default=24,
        help="Number of per-drug entries to show in the grouped bar and heatmap figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing figure files in the output directory.",
    )
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
            "grid.color": LIGHT_GRAY,
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
            "lines.linewidth": LINE_WIDTH,
        }
    )


def make_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_blues",
        [LIGHT_GRAY, PALE_BLUE, LIGHT_BLUE, MID_BLUE, DARK_BLUE],
    )


def save_figure(fig: plt.Figure, output_base: Path, fmt: str) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    suffixes = ("pdf", "png") if fmt == "both" else (fmt,)
    exported: list[Path] = []
    for suffix in suffixes:
        out_path = output_base.with_suffix(f".{suffix}")
        fig.savefig(
            out_path,
            dpi=FIG_DPI if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        exported.append(out_path)
    plt.close(fig)
    return exported


def find_first_file(input_dir: Path, target_name: str, must_contain: Iterable[str] | None = None) -> Path | None:
    must_contain = tuple(must_contain or ())
    candidates = []
    for path in input_dir.rglob(target_name):
        rel_text = str(path.relative_to(input_dir)).lower()
        if all(token.lower() in rel_text for token in must_contain):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def discover_data_sources(input_dir: Path) -> dict[str, Path | None]:
    return {
        "overall_model_mean": find_first_file(
            input_dir, "all_drug_model_mean.csv", must_contain=("benchmark_10x10_mfmr_best_fullcv", "global")
        ),
        "task_model_summary": find_first_file(
            input_dir, "dataset_task_model_summary.csv", must_contain=("benchmark_10x10_mfmr_best_fullcv", "global")
        ),
        "all_per_drug_metrics": find_first_file(
            input_dir, "all_per_drug_metrics.csv", must_contain=("benchmark_10x10_mfmr_best_fullcv", "global")
        ),
        "all_task_per_drug_uplift": find_first_file(
            input_dir, "all_task_per_drug_uplift.csv", must_contain=("uplift_mfmr_vs_mclrp",)
        ),
        "stage2_shared": find_first_file(input_dir, "stage2_shared.csv", must_contain=("global_cuda_mfmr_search",)),
        "stage3_joint": find_first_file(input_dir, "stage3_joint.csv", must_contain=("global_cuda_mfmr_search",)),
        "stage5_final": find_first_file(input_dir, "stage5_final.csv", must_contain=("global_cuda_mfmr_search",)),
        "best_candidate": find_first_file(
            input_dir, "best_candidate.json", must_contain=("global_cuda_mfmr_search",)
        ),
        "all_variant_summary": find_first_file(
            input_dir, "all_variant_summary.csv", must_contain=("ablation_mclrp_mfmr", "aggregate")
        ),
        "all_seed_results": find_first_file(
            input_dir, "all_seed_results.csv", must_contain=("ablation_mclrp_mfmr", "aggregate")
        ),
    }


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")


def load_csv(path: Path | None, required_cols: Iterable[str] | None = None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if required_cols is not None:
        require_columns(df, required_cols, path)
    return df


def load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prettify_task_label(row: pd.Series) -> str:
    dataset_label = str(row.get("dataset_label", "")).strip()
    if dataset_label:
        return dataset_label
    group = str(row.get("group", "")).strip()
    dataset = str(row.get("dataset", "")).strip()
    if group == "CCLE":
        return "CCLE"
    if group and dataset:
        return f"{group}-{dataset}"
    return dataset or group


def compute_model_order(overall_df: pd.DataFrame, preferred_metric: str = "mean_pcc") -> list[str]:
    work = overall_df.sort_values(preferred_metric, ascending=False).copy()
    order = work["model"].tolist()
    if "MCLRP_MFMR" in order:
        order.insert(0, order.pop(order.index("MCLRP_MFMR")))
    return order


def strongest_baseline(overall_df: pd.DataFrame) -> str | None:
    baseline_df = overall_df.loc[overall_df["model"] != "MCLRP_MFMR"].copy()
    if baseline_df.empty:
        return None
    return baseline_df.sort_values("mean_pcc", ascending=False)["model"].iloc[0]


def build_model_palette(model_order: list[str], strongest_model: str | None) -> dict[str, str]:
    palette: dict[str, str] = {}
    gray_cycle = [LIGHT_GRAY, LIGHT_SLATE, SLATE, "#94A3B8", "#CBD5E1", "#9CA3AF"]
    gray_idx = 0
    for model in model_order:
        if model == "MCLRP_MFMR":
            palette[model] = PRIMARY_BLUE
        elif strongest_model is not None and model == strongest_model:
            palette[model] = MID_BLUE
        elif model == "MCLRP":
            palette[model] = SLATE
        else:
            palette[model] = gray_cycle[min(gray_idx, len(gray_cycle) - 1)]
            gray_idx += 1
    return palette


def annotate_bar_values(ax: plt.Axes, values: list[float], positions: list[float], horizontal: bool = True) -> None:
    for value, pos in zip(values, positions):
        if not np.isfinite(value):
            continue
        text = f"{value:.3f}"
        if horizontal:
            ax.text(value + 0.006, pos, text, va="center", ha="left", fontsize=ANNOT_SIZE, color=DARK_GRAY)
        else:
            ax.text(pos, value + 0.006, text, va="bottom", ha="center", fontsize=ANNOT_SIZE, color=DARK_GRAY)


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_color(DARK_GRAY)
    ax.spines["bottom"].set_color(DARK_GRAY)
    ax.tick_params(colors=DARK_GRAY)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def plot_model_comparison(
    overall_df: pd.DataFrame | None,
    task_df: pd.DataFrame | None,
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    if overall_df is None:
        return []

    order = compute_model_order(overall_df, preferred_metric="mean_pcc")
    best_baseline = strongest_baseline(overall_df)
    palette = build_model_palette(order, best_baseline)
    work = overall_df.set_index("model").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), sharey=True)
    metrics = [("mean_pcc", "PCC"), ("mean_scc", "SCC")]

    for ax, (metric_col, metric_label) in zip(axes, metrics):
        y = np.arange(len(work))
        values = work[metric_col].to_numpy(dtype=float)
        colors = [palette.get(model, LIGHT_GRAY) for model in work["model"]]
        ax.barh(
            y,
            values,
            color=colors,
            edgecolor=DARK_GRAY,
            linewidth=BAR_EDGE_WIDTH,
            height=0.7,
        )

        if task_df is not None and metric_col in task_df.columns:
            for idx, model in enumerate(work["model"]):
                points = task_df.loc[task_df["model"] == model, metric_col].to_numpy(dtype=float)
                if points.size == 0:
                    continue
                jitter = np.linspace(-0.17, 0.17, num=points.size)
                ax.scatter(
                    points,
                    np.full_like(points, idx, dtype=float) + jitter,
                    s=MARKER_SIZE,
                    color=PALE_BLUE if model == "MCLRP_MFMR" else LIGHT_SLATE,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                    alpha=0.95,
                )

        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(model, model) for model in work["model"]])
        ax.invert_yaxis()
        ax.set_xlabel(f"Overall {metric_label}")
        ax.set_title(f"{metric_label} Comparison")
        ax.set_xlim(0.0, max(1.0, float(np.nanmax(values)) + 0.08))
        style_axis(ax, grid_axis="x")
        annotate_bar_values(ax, values.tolist(), y.tolist(), horizontal=True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette["MCLRP_MFMR"])]
    labels = [MODEL_DISPLAY_NAMES.get("MCLRP_MFMR", "MCLRP_MFMR")]
    if best_baseline is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=palette[best_baseline]))
        labels.append(f"Strongest baseline ({MODEL_DISPLAY_NAMES.get(best_baseline, best_baseline)})")
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color="white", markerfacecolor=LIGHT_SLATE))
    labels.append("Task-level results")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Overall Model Comparison", y=1.05, fontsize=TITLE_SIZE + 1)

    return save_figure(fig, output_dir / "figure_model_comparison_overall", fmt)


def select_per_drug_rows(
    per_drug_df: pd.DataFrame,
    uplift_df: pd.DataFrame | None,
    max_rows: int,
    model_order: list[str],
) -> list[str]:
    work = per_drug_df.copy()
    work["dataset_label"] = work.apply(prettify_task_label, axis=1)
    work["item_label"] = work["dataset_label"].astype(str) + " | " + work["drug"].astype(str)

    if uplift_df is not None and {"dataset_label", "drug", "delta_pcc"}.issubset(uplift_df.columns):
        uplift = uplift_df.copy()
        uplift["item_label"] = uplift["dataset_label"].astype(str) + " | " + uplift["drug"].astype(str)
        selected = uplift.sort_values("delta_pcc", ascending=False)["item_label"].drop_duplicates().tolist()
        return selected[: max_rows if max_rows > 0 else len(selected)]

    pivot = work.pivot_table(index="item_label", columns="model", values="pcc", aggfunc="first").reindex(columns=model_order)
    if "MCLRP_MFMR" in pivot.columns:
        selected = pivot.sort_values("MCLRP_MFMR", ascending=False).index.tolist()
    else:
        selected = pivot.index.tolist()
    return selected[: max_rows if max_rows > 0 else len(selected)]


def plot_per_drug_comparison(
    per_drug_df: pd.DataFrame | None,
    uplift_df: pd.DataFrame | None,
    overall_df: pd.DataFrame | None,
    output_dir: Path,
    fmt: str,
    per_drug_top_n: int,
) -> list[Path]:
    if per_drug_df is None or overall_df is None:
        return []

    # Current repository tables expose PCC/SCC at the per-drug level.
    # If future result tables add RMSE/MAE columns, extend the `metrics`
    # tuples below and the same plotting scaffold can be reused directly.
    model_order = compute_model_order(overall_df, preferred_metric="mean_pcc")
    best_baseline = strongest_baseline(overall_df) or "MCLRP"
    focus_models = [model for model in ["MCLRP_MFMR", best_baseline, "MCLRP"] if model in model_order]
    palette = build_model_palette(model_order, best_baseline)

    work = per_drug_df.copy()
    work["dataset_label"] = work.apply(prettify_task_label, axis=1)
    work["item_label"] = work["dataset_label"].astype(str) + " | " + work["drug"].astype(str)
    selected_rows = select_per_drug_rows(work, uplift_df, per_drug_top_n, model_order)
    work = work.loc[work["item_label"].isin(selected_rows)].copy()
    selected_order = [label for label in selected_rows if label in set(work["item_label"])]

    exports: list[Path] = []

    focus_work = work.loc[work["model"].isin(focus_models)].copy()
    if not focus_work.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15.5, max(7.5, len(selected_order) * 0.28)))
        metrics = [("pcc", "PCC"), ("scc", "SCC")]
        y = np.arange(len(selected_order))
        bar_group_width = 0.76
        bar_height = bar_group_width / max(1, len(focus_models))

        for ax, (metric_col, metric_label) in zip(axes, metrics):
            for idx, model in enumerate(focus_models):
                subset = (
                    focus_work.loc[focus_work["model"] == model, ["item_label", metric_col]]
                    .set_index("item_label")
                    .reindex(selected_order)
                )
                offset = (idx - (len(focus_models) - 1) / 2.0) * bar_height
                values = subset[metric_col].to_numpy(dtype=float)
                ax.barh(
                    y + offset,
                    values,
                    height=bar_height * 0.92,
                    color=palette.get(model, LIGHT_GRAY),
                    edgecolor=DARK_GRAY,
                    linewidth=BAR_EDGE_WIDTH,
                    label=MODEL_DISPLAY_NAMES.get(model, model),
                )
            ax.set_yticks(y)
            if ax is axes[0]:
                ax.set_yticklabels(selected_order)
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel(metric_label)
            ax.set_title(f"{metric_label} per drug")
            style_axis(ax, grid_axis="x")

        fig.suptitle("Per-Drug Comparison (selected drugs)", y=1.02, fontsize=TITLE_SIZE + 1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 1.01))
        exports.extend(save_figure(fig, output_dir / "figure_per_drug_comparison_bar", fmt))

    fig_height = max(8.0, len(selected_order) * 0.28)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, fig_height))
    cmap = make_blue_cmap()
    for ax, metric_col, title in zip(axes, ("pcc", "scc"), ("PCC Heatmap", "SCC Heatmap")):
        pivot = (
            work.pivot_table(index="item_label", columns="model", values=metric_col, aggfunc="first")
            .reindex(index=selected_order, columns=model_order)
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.4,
            linecolor="white",
            cbar=True,
            cbar_kws={"shrink": 0.8, "label": title.split()[0]},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Drug")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(str(text.get_text()), text.get_text()) for text in ax.get_xticklabels()],
            rotation=35,
            ha="right",
        )
        sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle("Per-Drug Multi-Model Heatmap (selected drugs)", y=1.02, fontsize=TITLE_SIZE + 1)
    exports.extend(save_figure(fig, output_dir / "figure_per_drug_heatmap", fmt))
    return exports


def filter_stage_frame_to_best(frame: pd.DataFrame, best_candidate: dict | None, varying_cols: Iterable[str]) -> pd.DataFrame:
    if frame.empty or not best_candidate:
        return frame.copy()
    varying_cols = set(varying_cols)
    filter_cols = [
        "topg_imp",
        "comp_imp",
        "topg_ridge",
        "comp_ridge",
        "ridge_alpha",
        "weight_imp",
        "weight_ridge",
        "final_alpha",
        "mutation_latent_dim",
        "imputer_max_iter",
    ]
    work = frame.copy()
    for col in filter_cols:
        if col in varying_cols or col not in work.columns or col not in best_candidate:
            continue
        candidate_value = best_candidate[col]
        if np.issubdtype(work[col].dtype, np.number):
            subset = work.loc[np.isclose(work[col], candidate_value)]
        else:
            subset = work.loc[work[col] == candidate_value]
        if not subset.empty:
            work = subset.copy()
    return work


def aggregate_for_line(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(x_col, as_index=False)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "std": "std_value", "count": "n"})
        .sort_values(x_col)
    )
    grouped["std_value"] = grouped["std_value"].fillna(0.0)
    return grouped


def plot_hparam_tuning(
    stage2_df: pd.DataFrame | None,
    stage3_df: pd.DataFrame | None,
    best_candidate: dict | None,
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    exports: list[Path] = []

    if stage2_df is not None and not stage2_df.empty:
        stage2_focus = filter_stage_frame_to_best(stage2_df, best_candidate, varying_cols=("weight_imp", "ridge_alpha"))
        if stage2_focus["weight_imp"].nunique() < 2:
            stage2_focus = stage2_df.copy()

        weight_curve = aggregate_for_line(stage2_focus, "weight_imp", "global_mean_pcc")
        alpha_curve = aggregate_for_line(stage2_focus, "ridge_alpha", "global_mean_pcc")

        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
        for ax, curve_df, x_col, x_label in (
            (axes[0], weight_curve, "weight_imp", "Weight of imputer branch"),
            (axes[1], alpha_curve, "ridge_alpha", "Ridge alpha"),
        ):
            x = curve_df[x_col].to_numpy(dtype=float)
            y = curve_df["mean_value"].to_numpy(dtype=float)
            yerr = curve_df["std_value"].to_numpy(dtype=float)
            ax.plot(x, y, color=PRIMARY_BLUE, marker="o", markersize=5.5, linewidth=LINE_WIDTH)
            ax.fill_between(x, y - yerr, y + yerr, color=PALE_BLUE, alpha=0.75, linewidth=0)
            best_idx = int(np.argmax(y))
            ax.scatter(x[best_idx], y[best_idx], color=DARK_BLUE, s=60, zorder=4)
            ax.annotate(
                f"best={x[best_idx]:g}\nscore={y[best_idx]:.3f}",
                xy=(x[best_idx], y[best_idx]),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=ANNOT_SIZE,
                color=DARK_GRAY,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": LIGHT_SLATE},
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Global mean PCC")
            ax.set_title(f"{x_label} vs performance")
            style_axis(ax, grid_axis="both")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

        fig.suptitle("Hyperparameter Tuning Curves", y=1.02, fontsize=TITLE_SIZE + 1)
        exports.extend(save_figure(fig, output_dir / "figure_hparam_tuning_line", fmt))

    if stage3_df is not None and not stage3_df.empty:
        stage3_focus = filter_stage_frame_to_best(
            stage3_df, best_candidate, varying_cols=("final_alpha", "mutation_latent_dim")
        )
        if best_candidate and "imputer_max_iter" in stage3_focus.columns and "imputer_max_iter" in best_candidate:
            filtered = stage3_focus.loc[np.isclose(stage3_focus["imputer_max_iter"], best_candidate["imputer_max_iter"])]
            if not filtered.empty:
                stage3_focus = filtered.copy()

        heatmap_df = (
            stage3_focus.groupby(["final_alpha", "mutation_latent_dim"], as_index=False)["cgp_mean_pcc"]
            .mean()
            .pivot(index="final_alpha", columns="mutation_latent_dim", values="cgp_mean_pcc")
            .sort_index()
        )
        if not heatmap_df.empty:
            fig, ax = plt.subplots(figsize=(6.6, 5.2))
            sns.heatmap(
                heatmap_df,
                ax=ax,
                cmap=make_blue_cmap(),
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".3f",
                annot_kws={"fontsize": ANNOT_SIZE},
                cbar_kws={"label": "CGP mean PCC", "shrink": 0.85},
            )
            ax.set_title("Mutation-head tuning heatmap")
            ax.set_xlabel("Mutation latent dimension")
            ax.set_ylabel("Final alpha")
            sns.despine(ax=ax, left=True, bottom=True)
            exports.extend(save_figure(fig, output_dir / "figure_hparam_tuning_heatmap", fmt))

    return exports


def ablation_palette(order: list[str], full_variant: str) -> list[str]:
    palette = []
    for variant in order:
        if variant == full_variant:
            palette.append(DARK_BLUE)
        elif "SingleView" in variant or "TissueLatent" in variant:
            palette.append(MID_BLUE)
        elif "RidgeOnly" in variant or "FullMinusPathway" in variant:
            palette.append(LIGHT_BLUE)
        elif "FullMinusLatent" in variant or "FullMinusInteraction" in variant:
            palette.append(PALE_BLUE)
        elif "Original" in variant or variant.endswith("_Base"):
            palette.append(LIGHT_GRAY)
        else:
            palette.append(LIGHT_SLATE)
    return palette


def summarize_ablation_seed(
    seed_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None,
    group: str,
    order: list[str],
) -> pd.DataFrame | None:
    if seed_df is not None and not seed_df.empty:
        work = seed_df.loc[seed_df["group"] == group].copy()
        if not work.empty:
            seed_level = (
                work.groupby(["variant_id", "seed"], as_index=False)["overall_pcc"].mean()
                .groupby("variant_id", as_index=False)["overall_pcc"]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "mean_pcc", "std": "std_pcc"})
            )
            seed_level["variant_id"] = pd.Categorical(seed_level["variant_id"], categories=order, ordered=True)
            return seed_level.sort_values("variant_id").reset_index(drop=True)

    if summary_df is not None and not summary_df.empty:
        work = summary_df.loc[summary_df["group"] == group].copy()
        if work.empty:
            return None
        agg = (
            work.groupby("variant_id", as_index=False)
            .agg(mean_pcc=("mean_pcc", "mean"), std_pcc=("mean_pcc", "std"))
        )
        agg["std_pcc"] = agg["std_pcc"].fillna(0.0)
        agg["variant_id"] = pd.Categorical(agg["variant_id"], categories=order, ordered=True)
        return agg.sort_values("variant_id").reset_index(drop=True)
    return None


def plot_ablation(
    summary_df: pd.DataFrame | None,
    seed_df: pd.DataFrame | None,
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    if summary_df is None and seed_df is None:
        return []

    shared_summary = summarize_ablation_seed(seed_df, summary_df, "shared_main", SHARED_ABLATION_ORDER)
    cgp_summary = summarize_ablation_seed(seed_df, summary_df, "cgp_main", CGP_ABLATION_ORDER)
    if shared_summary is None or cgp_summary is None:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    panels = [
        (axes[0], shared_summary, SHARED_ABLATION_ORDER, "A5_MFMR_Full", "Shared backbone ablation"),
        (axes[1], cgp_summary, CGP_ABLATION_ORDER, "C6_CGP_Full", "CGP mutation-head ablation"),
    ]

    for ax, panel_df, order, full_variant, title in panels:
        panel_df = panel_df.set_index("variant_id").reindex(order).reset_index()
        x = np.arange(len(order))
        values = panel_df["mean_pcc"].to_numpy(dtype=float)
        errors = panel_df["std_pcc"].fillna(0.0).to_numpy(dtype=float)
        colors = ablation_palette(order, full_variant)
        ax.bar(
            x,
            values,
            yerr=errors,
            color=colors,
            edgecolor=DARK_GRAY,
            linewidth=BAR_EDGE_WIDTH,
            capsize=3,
        )
        annotate_bar_values(ax, values.tolist(), x.tolist(), horizontal=False)
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_DISPLAY_NAMES.get(variant, variant) for variant in order], rotation=25, ha="right")
        ax.set_ylabel("Mean PCC")
        ax.set_title(title)
        lower = max(0.0, float(np.nanmin(values - errors)) - 0.04)
        upper = min(1.0, float(np.nanmax(values + errors)) + 0.06)
        ax.set_ylim(lower, upper)
        style_axis(ax, grid_axis="y")

    fig.suptitle("Ablation Study Summary", y=1.02, fontsize=TITLE_SIZE + 1)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=DARK_BLUE, label="Full model"),
        plt.Rectangle((0, 0), 1, 1, color=MID_BLUE, label="Core reduced variants"),
        plt.Rectangle((0, 0), 1, 1, color=LIGHT_BLUE, label="Additional module removal"),
        plt.Rectangle((0, 0), 1, 1, color=LIGHT_GRAY, label="Base / original"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    return save_figure(fig, output_dir / "figure_ablation", fmt)


def write_data_manifest(data_sources: dict[str, Path | None], output_dir: Path) -> Path:
    manifest = {key: (str(path) if path is not None else None) for key, path in data_sources.items()}
    manifest_path = output_dir / "data_sources.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def build_runtime_summary(exported_paths: list[Path], manifest_path: Path, output_dir: Path) -> Path:
    payload = {
        "output_dir": str(output_dir),
        "num_figures": len({path.stem for path in exported_paths}),
        "exported_files": [str(path) for path in exported_paths],
        "data_manifest": str(manifest_path),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        for pattern in ("*.pdf", "*.png", "*.json"):
            for path in output_dir.glob(pattern):
                path.unlink()

    set_publication_style()
    data_sources = discover_data_sources(input_dir)

    overall_df = load_csv(data_sources["overall_model_mean"], required_cols=("model", "mean_pcc", "mean_scc"))
    task_df = load_csv(
        data_sources["task_model_summary"],
        required_cols=("model", "mean_pcc", "mean_scc", "group", "dataset"),
    )
    per_drug_df = load_csv(
        data_sources["all_per_drug_metrics"],
        required_cols=("model", "drug", "pcc", "scc"),
    )
    uplift_df = load_csv(data_sources["all_task_per_drug_uplift"])
    stage2_df = load_csv(data_sources["stage2_shared"])
    stage3_df = load_csv(data_sources["stage3_joint"])
    best_candidate = load_json(data_sources["best_candidate"])
    ablation_summary = load_csv(data_sources["all_variant_summary"])
    ablation_seed = load_csv(data_sources["all_seed_results"])

    exported_paths: list[Path] = []
    exported_paths.extend(plot_model_comparison(overall_df, task_df, output_dir, args.format))
    exported_paths.extend(
        plot_per_drug_comparison(per_drug_df, uplift_df, overall_df, output_dir, args.format, args.per_drug_top_n)
    )
    exported_paths.extend(plot_hparam_tuning(stage2_df, stage3_df, best_candidate, output_dir, args.format))
    exported_paths.extend(plot_ablation(ablation_summary, ablation_seed, output_dir, args.format))

    manifest_path = write_data_manifest(data_sources, output_dir)
    summary_path = build_runtime_summary(exported_paths, manifest_path, output_dir)

    print(
        json.dumps(
            {
                "exported_files": [str(path) for path in exported_paths],
                "data_manifest": str(manifest_path),
                "run_summary": str(summary_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    # Example:
    # conda run -n torch310 python plotting/scripts/plot_result_figures.py --overwrite
    main()
