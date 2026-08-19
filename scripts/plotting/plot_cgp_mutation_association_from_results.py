from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu


CURRENT_FILE = Path(__file__).resolve()
ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Cannot locate project root")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_paths import CGP_RAW_DATA_DIR, LATEST_RESULTS_DIR


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HOTSPOT_PATTERNS = {
    "BRAF": r"V600",
    "KRAS": r"G12|G13|Q61|A146",
    "NRAS": r"G12|G13|Q61",
    "PIK3CA": r"H1047|E545|E542|Q546|P539",
    "PTEN": r"\*|fs|\?",
}
MAPK_GENES = {
    "ABL2", "EGFR", "FGFR3", "JAK2", "ALK", "BRAF", "EGFR.1", "ERBB2", "FGFR2", "FGFR3.1",
    "FLT3", "HRAS", "KDR", "KIT", "KRAS", "MAP2K4", "MET", "NF1", "NF2", "NRAS", "PDGFRA",
}
PI3K_GENES = {
    "AKT2", "EGFR", "ERBB2", "HRAS", "KRAS", "NRAS", "PIK3CA", "PIK3R1", "PTEN", "STK11", "TSC1",
    "CCND1", "CCND2", "CCND3", "CDK4", "CDK6", "CDKN2A", "CDKN2C",
}
DATASET_TO_SIZE = {
    "ERK_AUC": 30,
    "ERK_IC50": 32,
    "PI3K_AUC": 28,
    "PI3K_IC50": 29,
}
GENE_COLORS = {
    "MAPK": "#ef4444",
    "PI3K": "#6366f1",
    "Other": "#f59e0b",
}


@dataclass(frozen=True)
class AssocRow:
    drug_idx: int
    drug_label: str
    gene: str
    category: str
    n_mut: int
    n_wt: int
    p_obs: float
    p_pred: float
    delta_obs: float
    delta_pred: float
    score: float


def col_letters_to_idx(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def load_shared_strings(zf: ZipFile) -> list[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall("x:si", NS):
        texts = [node.text or "" for node in si.findall(".//x:t", NS)]
        strings.append("".join(texts))
    return strings


def parse_sheet_subset(xlsx_path: Path, sheet_name: str, wanted_headers: Iterable[str] | None = None) -> pd.DataFrame:
    with ZipFile(xlsx_path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        }
        target = None
        for sheet in workbook.findall("x:sheets/x:sheet", NS):
            if sheet.attrib["name"] == sheet_name:
                target = rel_map[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
                break
        if target is None:
            raise ValueError(f"Sheet not found: {sheet_name}")
        shared = load_shared_strings(zf)
        root = ET.fromstring(zf.read(f"xl/{target}"))
    rows = root.findall("x:sheetData/x:row", NS)
    header_cells = rows[0].findall("x:c", NS)
    headers_by_idx: dict[int, str] = {}
    for cell in header_cells:
        idx = col_letters_to_idx(cell.attrib["r"])
        if cell.attrib.get("t") == "s":
            headers_by_idx[idx] = shared[int(cell.findtext("x:v", default="0", namespaces=NS))]
        else:
            headers_by_idx[idx] = cell.findtext("x:v", default="", namespaces=NS)
    if wanted_headers is None:
        selected = headers_by_idx
    else:
        wanted = set(wanted_headers)
        selected = {idx: name for idx, name in headers_by_idx.items() if name in wanted}
    records = []
    for row in rows[1:]:
        rec = {name: "" for name in selected.values()}
        for cell in row.findall("x:c", NS):
            idx = col_letters_to_idx(cell.attrib["r"])
            if idx not in selected:
                continue
            raw = cell.findtext("x:v", default="", namespaces=NS)
            if cell.attrib.get("t") == "s" and raw != "":
                val = shared[int(raw)]
            else:
                val = raw
            rec[selected[idx]] = val
        records.append(rec)
    return pd.DataFrame.from_records(records)


def parse_gene_state(series: pd.Series) -> np.ndarray:
    s = series.fillna("").astype(str)
    mut = s.str.split("::").str[0].fillna("na")
    return ((~mut.str.startswith("wt")) & (~mut.str.startswith("na")) & (mut != "")).to_numpy()


def gene_category(gene: str) -> str:
    if gene in PI3K_GENES:
        return "PI3K"
    if gene in MAPK_GENES:
        return "MAPK"
    return "Other"


def build_mutation_table(xlsx_path: Path) -> tuple[pd.DataFrame, list[str]]:
    headers = ["Cell Line", "Cancer Type", "Tissue"]
    # Parse header row first to discover genes without requiring openpyxl.
    sheet1_full = parse_sheet_subset(xlsx_path, "Sheet1")
    sheet2_full = parse_sheet_subset(xlsx_path, "Sheet2")
    gene_cols = [c for c in sheet1_full.columns if c not in headers and c != "Cosmic_ID"]
    out = sheet1_full[["Cell Line", "Cancer Type", "Tissue"]].copy()
    for gene in gene_cols:
        m1 = parse_gene_state(sheet1_full[gene]) if gene in sheet1_full.columns else np.zeros(len(out), dtype=bool)
        m2 = parse_gene_state(sheet2_full[gene]) if gene in sheet2_full.columns else np.zeros(len(out), dtype=bool)
        out[gene] = m1 | m2
    return out, gene_cols


def load_prediction_matrix(dataset: str, variant: str) -> np.ndarray:
    path = (
        LATEST_RESULTS_DIR
        / "ablation_mclrp_mfmr"
        / "cgp_main"
        / dataset
        / "predictions"
        / f"{variant}_mean_prediction.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)["prediction"].astype(np.float32)


def load_observed_matrix(dataset: str) -> np.ndarray:
    mapping = {
        "ERK_AUC": "ERKAUC30",
        "ERK_IC50": "ERKIC50",
        "PI3K_AUC": "PI3KAUC",
        "PI3K_IC50": "PI3KIC50",
    }
    name = mapping[dataset]
    return np.load(CGP_RAW_DATA_DIR / f"{name}.npz", allow_pickle=True)[name].astype(np.float32)


def calc_assoc_rows(
    observed: np.ndarray,
    predicted: np.ndarray,
    mutations: pd.DataFrame,
    genes: list[str],
    min_mut: int,
    alpha: float,
) -> pd.DataFrame:
    rows: list[AssocRow] = []
    for drug_idx in range(observed.shape[1]):
        y_obs = observed[:, drug_idx]
        y_pred = predicted[:, drug_idx]
        valid = y_obs != 0
        if valid.sum() < 30:
            continue
        for gene in genes:
            mut_mask = mutations[gene].to_numpy(dtype=bool) & valid
            wt_mask = (~mutations[gene].to_numpy(dtype=bool)) & valid
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
                AssocRow(
                    drug_idx=drug_idx,
                    drug_label=f"Drug_{drug_idx + 1:02d}",
                    gene=gene,
                    category=gene_category(gene),
                    n_mut=n_mut,
                    n_wt=n_wt,
                    p_obs=p_obs,
                    p_pred=p_pred,
                    delta_obs=delta_obs,
                    delta_pred=delta_pred,
                    score=score,
                )
            )
    return pd.DataFrame([r.__dict__ for r in rows]).sort_values(["score", "p_pred", "p_obs"], ascending=[False, True, True])


def choose_panel_rows(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    chosen = []
    used = set()
    for row in df.itertuples(index=False):
        key = (row.drug_label, row.gene)
        if key in used:
            continue
        used.add(key)
        chosen.append(row._asdict())
        if len(chosen) >= top_n:
            break
    return pd.DataFrame(chosen)


def draw_network(ax: plt.Axes, assoc: pd.DataFrame, title: str) -> None:
    drugs = list(dict.fromkeys(assoc["drug_label"].tolist()))
    genes = list(dict.fromkeys(assoc["gene"].tolist()))
    drug_y = np.linspace(0.9, 0.1, max(len(drugs), 1))
    gene_y = np.linspace(0.9, 0.1, max(len(genes), 1))
    drug_pos = {d: (0.16, y) for d, y in zip(drugs, drug_y)}
    gene_pos = {g: (0.84, y) for g, y in zip(genes, gene_y)}
    max_score = assoc["score"].max() if not assoc.empty else 1.0
    for row in assoc.itertuples(index=False):
        x1, y1 = drug_pos[row.drug_label]
        x2, y2 = gene_pos[row.gene]
        width = 0.8 + 2.8 * (row.score / max_score)
        color = "#93c5fd" if row.delta_pred >= 0 else "#fca5a5"
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.55, solid_capstyle="round")
    for drug, (x, y) in drug_pos.items():
        ax.scatter(x, y, s=170, color="#22d3ee", edgecolor="#0f172a", linewidth=1.0, zorder=3)
        ax.text(x - 0.025, y, drug, ha="right", va="center", fontsize=9)
    for gene, (x, y) in gene_pos.items():
        cat = assoc.loc[assoc["gene"] == gene, "category"].iloc[0]
        ax.scatter(x, y, s=180, color=GENE_COLORS[cat], edgecolor="#0f172a", linewidth=1.0, zorder=3)
        ax.text(x + 0.025, y, gene, ha="left", va="center", fontsize=9)
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#22d3ee", markeredgecolor="#0f172a", markersize=8, label="Drug"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENE_COLORS["MAPK"], markeredgecolor="#0f172a", markersize=8, label="MAPK gene"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENE_COLORS["PI3K"], markeredgecolor="#0f172a", markersize=8, label="PI3K gene"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENE_COLORS["Other"], markeredgecolor="#0f172a", markersize=8, label="Other gene"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=8)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_box_panels(
    axes: list[plt.Axes],
    assoc: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    mutations: pd.DataFrame,
) -> None:
    for ax, row in zip(axes, assoc.itertuples(index=False)):
        valid = observed[:, row.drug_idx] != 0
        mut_mask = mutations[row.gene].to_numpy(dtype=bool) & valid
        wt_mask = (~mutations[row.gene].to_numpy(dtype=bool)) & valid
        obs_wt = observed[wt_mask, row.drug_idx]
        obs_mut = observed[mut_mask, row.drug_idx]
        pred_wt = predicted[wt_mask, row.drug_idx]
        pred_mut = predicted[mut_mask, row.drug_idx]
        xs = [0.9, 1.9, 3.4, 4.4]
        data = [obs_wt, obs_mut, pred_wt, pred_mut]
        colors = ["#38bdf8", "#b91c1c", "#38bdf8", "#b91c1c"]
        for x, arr, c in zip(xs, data, colors):
            jitter = np.random.default_rng(42).normal(0, 0.08, size=len(arr))
            ax.scatter(np.full(len(arr), x) + jitter, arr, s=8, color=c, alpha=0.75, linewidths=0)
        bp = ax.boxplot(
            data,
            positions=xs,
            widths=0.5,
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
        for whisker, color in zip(bp["whiskers"], [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2], colors[3], colors[3]]):
            whisker.set_color(color)
        for cap, color in zip(bp["caps"], [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2], colors[3], colors[3]]):
            cap.set_color(color)
        ax.set_xticks(xs, ["Obs WT", "Obs Mut", "Pred WT", "Pred Mut"], rotation=0, fontsize=8)
        ax.set_title(f"{row.drug_label} - {row.gene}", fontsize=11, weight="bold")
        ax.text(
            0.03,
            0.97,
            f"p(obs)={row.p_obs:.2e}\np(pred)={row.p_pred:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cbd5e1"},
        )
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    for ax in axes[len(assoc):]:
        ax.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CGP mutation-association network and consistency panels from MCLRP-MFMR results")
    parser.add_argument("--dataset", choices=sorted(DATASET_TO_SIZE), default="PI3K_AUC")
    parser.add_argument("--variant", default="C6_CGP_Full")
    parser.add_argument("--top-edges", type=int, default=14)
    parser.add_argument("--top-panels", type=int, default=6)
    parser.add_argument("--min-mut", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "plotting" / "outputs" / "requested_summary_figures" / "mutation_association"
    output_dir.mkdir(parents=True, exist_ok=True)

    mutation_table, gene_cols = build_mutation_table(CGP_RAW_DATA_DIR / "Mutation.xlsx")
    observed = load_observed_matrix(args.dataset)
    predicted = load_prediction_matrix(args.dataset, args.variant)
    if observed.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: observed={observed.shape}, predicted={predicted.shape}")

    assoc = calc_assoc_rows(observed, predicted, mutation_table, gene_cols, min_mut=args.min_mut, alpha=args.alpha)
    assoc_csv = output_dir / f"{args.dataset}_{args.variant}_mutation_associations.csv"
    assoc.to_csv(assoc_csv, index=False, encoding="utf-8-sig")
    if assoc.empty:
        raise RuntimeError("No mutation-drug associations passed the current filters.")

    net_df = choose_panel_rows(assoc, args.top_edges)
    panel_df = choose_panel_rows(assoc, args.top_panels)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])
    ax_net = fig.add_subplot(gs[0, :])
    draw_network(ax_net, net_df, f"{args.dataset} | {args.variant} | Mutation-Drug Association Network")
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    if len(panel_df) > 3:
        fig2 = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs2 = fig2.add_gridspec(1, len(panel_df))
        axes2 = [fig2.add_subplot(gs2[0, i]) for i in range(len(panel_df))]
        draw_box_panels(axes2, panel_df, observed, predicted, mutation_table)
        fig2.suptitle(f"{args.dataset} | Observed vs Predicted Mutation Consistency", fontsize=14, weight="bold")
        fig2.savefig(output_dir / f"{args.dataset}_{args.variant}_mutation_boxpanels.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
    else:
        draw_box_panels(axes, panel_df, observed, predicted, mutation_table)

    fig.savefig(output_dir / f"{args.dataset}_{args.variant}_mutation_network.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved associations to {assoc_csv}")


if __name__ == "__main__":
    main()
