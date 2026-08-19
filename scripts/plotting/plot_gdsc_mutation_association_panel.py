from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu


CURRENT_FILE = Path(__file__).resolve()
ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Cannot locate project root")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.standardized_dataset_loaders import load_gdsc_standardized_bundle
from project_paths import GDSC_STANDARDIZED_DIR, LATEST_RESULTS_DIR


DEFAULT_VARIANTS = {
    "ERK_AUC": "A5_MFMR_Full",
    "ERK_IC50": "A5_MFMR_Full",
    "PI3K_AUC": "A5_MFMR_Full",
    "PI3K_IC50": "A5_MFMR_Full",
}
MAPK_GENES = {
    "ABL2", "EGFR", "FGFR3", "JAK2", "ALK", "BRAF", "EGFR.1", "ERBB2", "FGFR2", "FGFR3.1",
    "FLT3", "HRAS", "KDR", "KIT", "KRAS", "MAP2K4", "MET", "NF1", "NF2", "NRAS", "PDGFRA",
}
PI3K_GENES = {
    "AKT2", "PIK3CA", "PIK3R1", "PTEN", "STK11", "TSC1", "CCND1", "CCND2", "CCND3", "CDK4", "CDK6", "CDKN2A", "CDKN2C",
}
GENE_COLORS = {
    "MAPK": "#ef4444",
    "PI3K": "#6366f1",
    "Other": "#f59e0b",
}
FRAME_COLOR = "#334155"
GRID_COLOR = "#E6EDF4"
TEXT_DARK = "#1F2937"
FONT_FAMILY = "DejaVu Sans"
AXIS_LABEL_SIZE = 10.5
TICK_LABEL_SIZE = 8.8
LEGEND_SIZE = 8.6


def set_publication_style() -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a paper-style mutation association panel for one GDSC subtask.")
    parser.add_argument("--dataset", choices=sorted(DEFAULT_VARIANTS), default="ERK_AUC")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--min-mut", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-drugs-per-network", type=int, default=7)
    parser.add_argument("--top-boxplots", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "plotting" / "outputs" / "requested_summary_figures" / "gdsc_mutation_panel"),
    )
    return parser.parse_args()


def gene_category(gene: str) -> str:
    if gene in MAPK_GENES:
        return "MAPK"
    if gene in PI3K_GENES:
        return "PI3K"
    return "Other"


def parse_gene_state(series: pd.Series) -> np.ndarray:
    s = series.fillna("").astype(str)
    mut = s.str.split("::").str[0].fillna("na")
    return ((~mut.str.startswith("wt")) & (~mut.str.startswith("na")) & (mut != "")).to_numpy()


def load_prediction_matrix(dataset: str, variant: str) -> np.ndarray:
    path = (
        LATEST_RESULTS_DIR
        / "ablation_mclrp_mfmr"
        / "shared_main"
        / dataset
        / "predictions"
        / f"{variant}_mean_prediction.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)["prediction"].astype(np.float32)


def load_mutation_table(cell_ids: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    mut = pd.read_csv(GDSC_STANDARDIZED_DIR / "mutation_features.csv")
    mut = mut.set_index("model_id").loc[cell_ids].reset_index()
    gene_cols = [c for c in mut.columns if c not in {"model_id", "model_name", "Cancer Type", "Tissue"}]
    return mut, gene_cols


def calc_associations(
    observed: np.ndarray,
    predicted: np.ndarray,
    drug_labels: np.ndarray,
    mutations: pd.DataFrame,
    gene_cols: list[str],
    min_mut: int,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gene_masks = {gene: parse_gene_state(mutations[gene]) for gene in gene_cols}
    for drug_idx, drug_label in enumerate(drug_labels):
        valid = observed[:, drug_idx] != 0
        if valid.sum() < 30:
            continue
        y_obs = observed[:, drug_idx]
        y_pred = predicted[:, drug_idx]
        for gene in gene_cols:
            gmask = gene_masks[gene]
            mut_mask = gmask & valid
            wt_mask = (~gmask) & valid
            n_mut = int(mut_mask.sum())
            n_wt = int(wt_mask.sum())
            if n_mut < min_mut or n_wt < min_mut:
                continue
            obs_mut = y_obs[mut_mask]
            obs_wt = y_obs[wt_mask]
            pred_mut = y_pred[mut_mask]
            pred_wt = y_pred[wt_mask]
            try:
                p_obs = float(mannwhitneyu(obs_mut, obs_wt, alternative="two-sided").pvalue)
                p_pred = float(mannwhitneyu(pred_mut, pred_wt, alternative="two-sided").pvalue)
            except ValueError:
                continue
            if p_obs >= alpha or p_pred >= alpha:
                continue
            delta_obs = float(np.median(obs_mut) - np.median(obs_wt))
            delta_pred = float(np.median(pred_mut) - np.median(pred_wt))
            score = (-math.log10(max(p_obs * p_pred, 1e-300))) * (abs(delta_obs) + abs(delta_pred))
            rows.append(
                {
                    "drug_idx": int(drug_idx),
                    "drug_label": str(drug_label),
                    "gene": gene,
                    "category": gene_category(gene),
                    "n_mut": n_mut,
                    "n_wt": n_wt,
                    "p_obs": p_obs,
                    "p_pred": p_pred,
                    "delta_obs": delta_obs,
                    "delta_pred": delta_pred,
                    "score": score,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "p_pred", "p_obs"], ascending=[False, True, True]).reset_index(drop=True)


def choose_hub_genes(assoc: pd.DataFrame, n_hubs: int = 2) -> list[str]:
    hub_score = assoc.groupby("gene").agg(
        edges=("drug_label", "count"),
        total_score=("score", "sum"),
        best_score=("score", "max"),
    )
    hub_score = hub_score.sort_values(["edges", "total_score", "best_score"], ascending=[False, False, False])
    return hub_score.head(n_hubs).index.tolist()


def spread_positions(points: list[tuple[float, float]], min_gap: float = 0.09, low: float = 0.08, high: float = 0.92) -> list[tuple[float, float]]:
    if not points:
        return []
    ordered = sorted(enumerate(points), key=lambda item: item[1][1])
    ys = [y for _, (_, y) in ordered]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - high
    if overflow > 0:
        ys = [y - overflow for y in ys]
    for i in range(len(ys) - 2, -1, -1):
        ys[i] = min(ys[i], ys[i + 1] - min_gap)
    underflow = low - ys[0]
    if underflow > 0:
        ys = [y + underflow for y in ys]
    out = [None] * len(points)
    for (orig_idx, (x, _)), y in zip(ordered, ys):
        out[orig_idx] = (x, y)
    return out


def draw_network(ax: plt.Axes, assoc: pd.DataFrame, hub_gene: str, top_drugs: int, panel_label: str) -> None:
    sub = assoc[assoc["gene"] == hub_gene].head(top_drugs).copy()
    if sub.empty:
        ax.axis("off")
        return
    hub_cat = sub["category"].iloc[0]
    hub_pos = (0.5, 0.5)
    angles = np.linspace(-0.85 * math.pi, 0.85 * math.pi, len(sub), endpoint=True)
    max_score = max(sub["score"].max(), 1.0)
    raw_positions = []
    for angle in angles:
        x = 0.5 + 0.34 * math.cos(angle)
        y = 0.5 + 0.34 * math.sin(angle)
        raw_positions.append((x, y))
    right_points = [(x, y) for x, y in raw_positions if x >= hub_pos[0]]
    left_points = [(x, y) for x, y in raw_positions if x < hub_pos[0]]
    spread_right = spread_positions(right_points, min_gap=0.11)
    spread_left = spread_positions(left_points, min_gap=0.10)
    positions = []
    r_i = 0
    l_i = 0
    for x, y in raw_positions:
        if x >= hub_pos[0]:
            positions.append(spread_right[r_i])
            r_i += 1
        else:
            positions.append(spread_left[l_i])
            l_i += 1
    for (x, y), row in zip(positions, sub.itertuples(index=False)):
        rad = 0.22 * np.sign(y - hub_pos[1]) if abs(y - hub_pos[1]) > 0.04 else 0.16
        edge = FancyArrowPatch(
            hub_pos,
            (x, y),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=1.2 + 3.8 * (row.score / max_score),
            color="#93c5fd",
            alpha=0.85,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(edge)
        ax.scatter(x, y, s=80, color="#22d3ee", edgecolor="#0f172a", linewidth=1.0, zorder=3)
        ha = "left" if x >= 0.5 else "right"
        dx = 0.045 if ha == "left" else -0.045
        ax.text(x + dx, y, row.drug_label, ha=ha, va="center", fontsize=9)
    ax.scatter(
        hub_pos[0],
        hub_pos[1],
        s=280,
        color=GENE_COLORS[hub_cat],
        edgecolor="#0f172a",
        linewidth=1.1,
        zorder=4,
    )
    ax.text(hub_pos[0], hub_pos[1] - 0.085, hub_gene, ha="center", va="top", fontsize=11, weight="bold")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#22d3ee", markeredgecolor="#0f172a", markersize=8, label="Drug"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENE_COLORS["MAPK"], markeredgecolor="#0f172a", markersize=8, label="MAPK gene"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENE_COLORS["PI3K"], markeredgecolor="#0f172a", markersize=8, label="PI3K gene"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize=8.2)
    add_panel_label(ax, panel_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_boxplot_panel(
    ax: plt.Axes,
    row: pd.Series,
    observed: np.ndarray,
    predicted: np.ndarray,
    mutations: pd.DataFrame,
    panel_label: str,
) -> None:
    gmask = parse_gene_state(mutations[row["gene"]])
    valid = observed[:, int(row["drug_idx"])] != 0
    mut_mask = gmask & valid
    wt_mask = (~gmask) & valid
    obs_wt = observed[wt_mask, int(row["drug_idx"])]
    obs_mut = observed[mut_mask, int(row["drug_idx"])]
    pred_wt = predicted[wt_mask, int(row["drug_idx"])]
    pred_mut = predicted[mut_mask, int(row["drug_idx"])]
    xs = [0.85, 1.85, 3.35, 4.35]
    data = [obs_wt, obs_mut, pred_wt, pred_mut]
    colors = ["#38bdf8", "#b91c1c", "#38bdf8", "#b91c1c"]
    rng = np.random.default_rng(42)
    for x, arr, c in zip(xs, data, colors):
        jitter = rng.normal(0, 0.08, size=len(arr))
        ax.scatter(np.full(len(arr), x) + jitter, arr, s=8, color=c, alpha=0.72, linewidths=0)
    bp = ax.boxplot(
        data,
        positions=xs,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.2},
        boxprops={"linewidth": 1.8},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
    for line, color in zip(bp["whiskers"], [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2], colors[3], colors[3]]):
        line.set_color(color)
    for line, color in zip(bp["caps"], [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2], colors[3], colors[3]]):
        line.set_color(color)
    ax.set_xticks(xs, ["Wild Type", "Mutation", "Wild Type", "Mutation"], fontsize=8.2)
    ax.set_title(f"{row['drug_label']} - {row['gene']}", fontsize=10.5, weight="bold")
    ax.text(0.08, 0.98, f"Observed\np={row['p_obs']:.2e}", transform=ax.transAxes, ha="left", va="top", fontsize=8.2)
    ax.text(0.63, 0.98, f"Predicted\np={row['p_pred']:.2e}", transform=ax.transAxes, ha="left", va="top", fontsize=8.2)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    add_panel_label(ax, panel_label)
    legend_items = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#38bdf8", markersize=7, label=f"WT n={int(row['n_wt'])}"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#b91c1c", markersize=7, label=f"Mut n={int(row['n_mut'])}"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False, fontsize=8.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.0)
    ax.tick_params(axis="y", labelsize=8.2)


def main() -> None:
    args = parse_args()
    variant = args.variant or DEFAULT_VARIANTS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    bundle = load_gdsc_standardized_bundle(args.dataset)
    predicted = load_prediction_matrix(args.dataset, variant)
    observed = bundle.M.astype(np.float32)
    mutations, gene_cols = load_mutation_table(bundle.cell_ids)
    assoc = calc_associations(observed, predicted, bundle.drug_labels, mutations, gene_cols, args.min_mut, args.alpha)
    assoc_path = output_dir / f"{args.dataset}_{variant}_mutation_associations.csv"
    assoc.to_csv(assoc_path, index=False, encoding="utf-8-sig")
    if assoc.empty:
        raise RuntimeError("No significant mutation-drug associations found.")

    hub_genes = choose_hub_genes(assoc, n_hubs=2)
    box_df = assoc.head(args.top_boxplots).copy()

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :1])
    ax_b = fig.add_subplot(gs[0, 1:])
    draw_network(ax_a, assoc, hub_genes[0], args.top_drugs_per_network, "A")
    if len(hub_genes) > 1:
        draw_network(ax_b, assoc, hub_genes[1], args.top_drugs_per_network + 1, "B")
    else:
        ax_b.axis("off")

    panel_labels = ["C", "D", "E", "F", "G", "H"]
    axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]
    for ax, (_, row), label in zip(axes, box_df.iterrows(), panel_labels):
        draw_boxplot_panel(ax, row, observed, predicted, mutations, label)
    for ax in axes[len(box_df):]:
        ax.axis("off")

    fig.suptitle(f"{args.dataset} | {variant} | Mutation Association Panel", fontsize=17, weight="bold")
    out_path = output_dir / f"{args.dataset}_{variant}_mutation_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")
    print(f"saved associations: {assoc_path}")


if __name__ == "__main__":
    main()
