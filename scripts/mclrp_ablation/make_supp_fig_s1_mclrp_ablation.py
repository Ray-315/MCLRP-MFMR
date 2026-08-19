from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import RESULTS_DIR


DEFAULT_INPUT_DIR = RESULTS_DIR / "reviewer_mclrp_ablation_5task_10x10_v1"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "paper_figures" / "figS1_mclrp_ablation"
MODELS = ("MCLRP", "MCLRP_NoPCA_reconstructed", "MCLRP_NoTrace_reconstructed")
DISPLAY = {
    "MCLRP": "MCLRP",
    "MCLRP_NoPCA_reconstructed": "NoPCA (reconstructed)",
    "MCLRP_NoTrace_reconstructed": "NoTrace (reconstructed)",
}
COLORS = {
    "MCLRP": "#1B4B7A",
    "MCLRP_NoPCA_reconstructed": "#6CA6A6",
    "MCLRP_NoTrace_reconstructed": "#C97A1B",
}
GDSC_TASKS = ("GDSC-ERK_AUC", "GDSC-ERK_IC50", "GDSC-PI3K_AUC", "GDSC-PI3K_IC50")
TASK_DISPLAY = {
    "GDSC-ERK_AUC": "ERK-AUC",
    "GDSC-ERK_IC50": "ERK-IC50",
    "GDSC-PI3K_AUC": "PI3K-AUC",
    "GDSC-PI3K_IC50": "PI3K-IC50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Supplementary Figure S1 for reconstructed MCLRP ablations")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#E7EDF3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def draw_ccle_per_drug(ax: plt.Axes, data: pd.DataFrame, metric: str, panel: str) -> None:
    ccle = data.loc[data["dataset"].eq("CCLE")].copy()
    drugs = ccle.loc[ccle["model"].eq("MCLRP")].sort_values("drug_idx")[["drug_idx", "drug"]]
    x = np.arange(len(drugs))
    offsets = np.linspace(-0.22, 0.22, len(MODELS))
    for offset, model in zip(offsets, MODELS):
        values = ccle.loc[ccle["model"].eq(model)].sort_values("drug_idx")[metric].to_numpy(dtype=float)
        ax.scatter(x + offset, values, s=17, color=COLORS[model], edgecolor="white", linewidth=0.35, label=DISPLAY[model])
    ax.axhline(0, color="#AEB8C4", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in drugs["drug"].tolist()], rotation=55, ha="right", fontsize=6.3)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("CCLE drug")
    ax.set_title(f"CCLE per-drug {metric.upper()}", fontsize=10, loc="left")
    style_axis(ax)
    panel_label(ax, panel)


def draw_gdsc_summary(ax: plt.Axes, summary: pd.DataFrame, metric: str, panel: str) -> None:
    data = summary.loc[summary["dataset"].isin(GDSC_TASKS)].copy()
    x = np.arange(len(GDSC_TASKS))
    width = 0.24
    for model_index, model in enumerate(MODELS):
        values = (
            data.loc[data["model"].eq(model)]
            .set_index("dataset")
            .reindex(GDSC_TASKS)[metric]
            .to_numpy(dtype=float)
        )
        ax.bar(x + (model_index - 1) * width, values, width=width, color=COLORS[model], label=DISPLAY[model])
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[task] for task in GDSC_TASKS], rotation=20, ha="right")
    ax.set_ylabel("Drug-macro " + metric.split("_")[-1].upper())
    ax.set_xlabel("GDSC response task")
    ax.set_title("GDSC task-level " + metric.split("_")[-1].upper(), fontsize=10, loc="left")
    style_axis(ax)
    panel_label(ax, panel)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_drug = pd.read_csv(input_dir / "mean_per_drug_metrics.csv")
    summary = pd.read_csv(input_dir / "task_model_summary.csv")
    if set(MODELS) - set(per_drug["model"].unique()):
        raise RuntimeError("Missing reconstructed-ablation model rows")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelcolor": "#334155",
            "text.color": "#263442",
            "xtick.color": "#526173",
            "ytick.color": "#526173",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    draw_ccle_per_drug(axes[0, 0], per_drug, "pcc", "A")
    draw_ccle_per_drug(axes[0, 1], per_drug, "scc", "B")
    draw_gdsc_summary(axes[1, 0], summary, "macro_pcc", "C")
    draw_gdsc_summary(axes[1, 1], summary, "macro_scc", "D")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=3, frameon=False)

    prefix = output_dir / "figS1_mclrp_reconstructed_ablation"
    paths = []
    for suffix in (".pdf", ".png", ".svg"):
        path = prefix.with_suffix(suffix)
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    plt.close(fig)
    manifest = {
        "figure": "Supplementary Figure S1",
        "source_run": str(input_dir),
        "panels": {
            "A": "CCLE per-drug PCC",
            "B": "CCLE per-drug SCC",
            "C": "four GDSC task-level macro PCC values",
            "D": "four GDSC task-level macro SCC values",
        },
        "models": list(MODELS),
        "outputs": [str(path) for path in paths],
    }
    (output_dir / "figS1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
