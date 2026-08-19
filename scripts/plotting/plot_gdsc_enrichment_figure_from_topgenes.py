from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
import requests


CURRENT_FILE = Path(__file__).resolve()
ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Cannot locate project root")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
LIBRARIES = {
    "go_mf": "GO_Molecular_Function_2023",
    "kegg": "KEGG_2021_Human",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a figure-7-style enrichment overview from GDSC top genes.")
    parser.add_argument("--dataset", default="PI3K_AUC", choices=["ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"])
    parser.add_argument("--variant", default="A5_MFMR_Full")
    parser.add_argument("--top-genes", type=int, default=300)
    parser.add_argument("--top-drugs", type=int, default=12)
    parser.add_argument("--terms-per-drug", type=int, default=2)
    parser.add_argument("--top-terms-panel", type=int, default=18)
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(ROOT / "plotting" / "outputs" / "requested_summary_figures" / "gdsc_functional_summary"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "plotting" / "outputs" / "requested_summary_figures" / "gdsc_enrichment"),
    )
    return parser.parse_args()


def add_gene_list(genes: list[str], description: str) -> int:
    payload = {"list": (None, "\n".join(genes)), "description": (None, description)}
    resp = requests.post(f"{ENRICHR_BASE}/addList", files=payload, timeout=60)
    resp.raise_for_status()
    return int(resp.json()["userListId"])


def enrich_gene_list(genes: list[str], library: str, description: str) -> pd.DataFrame:
    user_list_id = add_gene_list(genes, description)
    resp = requests.get(
        f"{ENRICHR_BASE}/enrich",
        params={"userListId": user_list_id, "backgroundType": library},
        timeout=60,
    )
    resp.raise_for_status()
    records = resp.json().get(library, [])
    rows = []
    for rec in records:
        rows.append(
            {
                "rank": int(rec[0]),
                "term_name": str(rec[1]),
                "p_value": float(rec[2]),
                "combined_score": float(rec[4]),
                "overlap_genes": ",".join(rec[5]) if isinstance(rec[5], list) else str(rec[5]),
                "adjusted_p_value": float(rec[6]),
                "legacy_score": float(rec[3]),
            }
        )
    return pd.DataFrame(rows)


def bezier_edge(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str, lw: float, alpha: float) -> None:
    dx = x1 - x0
    verts = [
        (x0, y0),
        (x0 + dx * 0.35, y0),
        (x1 - dx * 0.35, y1),
        (x1, y1),
    ]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round")
    ax.add_patch(patch)


def normalize_term_name(name: str) -> str:
    return name.split(" (")[0].strip()


def load_inputs(input_dir: Path, dataset: str, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_path = input_dir / f"{dataset}_{variant}_top1000_genes_per_drug.csv"
    tissue_path = input_dir / f"{dataset}_{variant}_tissue_mean_response.csv"
    if not top_path.exists():
        raise FileNotFoundError(top_path)
    if not tissue_path.exists():
        raise FileNotFoundError(tissue_path)
    return pd.read_csv(top_path), pd.read_csv(tissue_path)


def select_drugs(top_df: pd.DataFrame, top_drugs: int) -> list[str]:
    score = (
        top_df.groupby("drug_label")["abs_sum_score"]
        .mean()
        .sort_values(ascending=False)
    )
    return score.head(top_drugs).index.tolist()


def run_enrichment_for_drugs(
    top_df: pd.DataFrame,
    drugs: list[str],
    top_genes: int,
    cache_dir: Path,
    dataset: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    go_rows = []
    kegg_rows = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        genes = (
            top_df.loc[top_df["drug_label"] == drug]
            .sort_values("rank_within_drug")
            .head(top_genes)["gene_symbol"]
            .astype(str)
            .tolist()
        )
        for key, lib in LIBRARIES.items():
            cache_path = cache_dir / f"{dataset}_{variant}_{drug.replace('/', '_').replace(' ', '_')}_{key}.json"
            if cache_path.exists():
                enrich_df = pd.read_json(cache_path)
            else:
                enrich_df = enrich_gene_list(genes, lib, f"{dataset}-{variant}-{drug}-{key}")
                enrich_df.to_json(cache_path, orient="records", force_ascii=False)
            enrich_df = enrich_df.copy()
            enrich_df["drug_label"] = drug
            enrich_df["term_name"] = enrich_df["term_name"].map(normalize_term_name)
            enrich_df["neglog10_fdr"] = -np.log10(np.maximum(enrich_df["adjusted_p_value"].astype(float), 1e-300))
            if key == "go_mf":
                go_rows.append(enrich_df)
            else:
                kegg_rows.append(enrich_df)
    return pd.concat(go_rows, ignore_index=True), pd.concat(kegg_rows, ignore_index=True)


def choose_panel_terms(enrich_df: pd.DataFrame, terms_per_drug: int, top_terms_panel: int) -> pd.DataFrame:
    picked = enrich_df.groupby("drug_label", group_keys=False).head(terms_per_drug).copy()
    agg = (
        picked.groupby("term_name")
        .agg(total_score=("neglog10_fdr", "sum"), hits=("drug_label", "nunique"))
        .sort_values(["hits", "total_score"], ascending=[False, False])
        .head(top_terms_panel)
    )
    panel = picked[picked["term_name"].isin(agg.index)].copy()
    panel = panel.merge(agg, left_on="term_name", right_index=True, how="left")
    return panel.sort_values(["hits", "total_score", "term_name"], ascending=[False, False, True])


def draw_alluvial(ax: plt.Axes, panel_df: pd.DataFrame, title: str, drug_colors: dict[str, str]) -> None:
    drugs = list(dict.fromkeys(panel_df["drug_label"].tolist()))
    terms = (
        panel_df.groupby("term_name")["neglog10_fdr"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not drugs or not terms:
        ax.axis("off")
        return
    drug_y = np.linspace(0.94, 0.06, len(drugs))
    term_y = np.linspace(0.94, 0.06, len(terms))
    drug_pos = {drug: y for drug, y in zip(drugs, drug_y)}
    term_pos = {term: y for term, y in zip(terms, term_y)}
    max_score = max(panel_df["neglog10_fdr"].max(), 1.0)
    for row in panel_df.itertuples(index=False):
        lw = 1.0 + 5.0 * (row.neglog10_fdr / max_score)
        bezier_edge(ax, 0.14, drug_pos[row.drug_label], 0.86, term_pos[row.term_name], drug_colors[row.drug_label], lw, 0.55)
    for drug, y in drug_pos.items():
        ax.add_patch(Rectangle((0.04, y - 0.014), 0.06, 0.028, facecolor=drug_colors[drug], edgecolor="white", lw=0.8))
        ax.text(0.03, y, drug, ha="right", va="center", fontsize=8)
    for term, y in term_pos.items():
        ax.add_patch(Rectangle((0.90, y - 0.014), 0.06, 0.028, facecolor="#f3d1d1", edgecolor="white", lw=0.8))
        ax.text(0.97, y, term, ha="left", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold")


def draw_heatmap(ax: plt.Axes, tissue_df: pd.DataFrame, drugs: list[str]) -> None:
    heat = tissue_df[tissue_df["drug_label"].isin(drugs)].pivot(index="Tissue", columns="drug_label", values="predicted_mean")
    heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]
    arr = heat.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(np.arange(heat.shape[1]), labels=heat.columns)
    ax.set_yticks(np.arange(heat.shape[0]), labels=heat.index)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    ax.set_title("Mean Predicted Response Across Tissues", fontsize=13, weight="bold")
    ax.set_xlabel("Drug")
    ax.set_ylabel("Tissue")
    return im


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    top_df, tissue_df = load_inputs(input_dir, args.dataset, args.variant)
    drugs = select_drugs(top_df, args.top_drugs)

    go_df, kegg_df = run_enrichment_for_drugs(top_df, drugs, args.top_genes, cache_dir, args.dataset, args.variant)
    go_df.to_csv(output_dir / f"{args.dataset}_{args.variant}_go_mf_enrichment.csv", index=False, encoding="utf-8-sig")
    kegg_df.to_csv(output_dir / f"{args.dataset}_{args.variant}_kegg_enrichment.csv", index=False, encoding="utf-8-sig")

    go_panel = choose_panel_terms(go_df, args.terms_per_drug, args.top_terms_panel)
    kegg_panel = choose_panel_terms(kegg_df, args.terms_per_drug, args.top_terms_panel)

    palette = plt.get_cmap("tab20", len(drugs))
    drug_colors = {drug: palette(i) for i, drug in enumerate(drugs)}

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    ax_go = fig.add_subplot(gs[0, 0])
    ax_kegg = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    draw_alluvial(ax_go, go_panel, "GO Molecular Function", drug_colors)
    draw_alluvial(ax_kegg, kegg_panel, "KEGG Pathways", drug_colors)
    im = draw_heatmap(ax_heat, tissue_df, drugs)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.022, pad=0.01)
    cbar.set_label("Mean predicted response")
    fig.suptitle(f"{args.dataset} | {args.variant} | First-pass Enrichment Figure", fontsize=16, weight="bold")

    out_base = output_dir / f"{args.dataset}_{args.variant}_enrichment_overview_v1"
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset": args.dataset,
        "variant": args.variant,
        "top_drugs": drugs,
        "go_rows": int(len(go_df)),
        "kegg_rows": int(len(kegg_df)),
        "figure": str(out_base.with_suffix(".png")),
    }
    (output_dir / f"{args.dataset}_{args.variant}_enrichment_overview_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
