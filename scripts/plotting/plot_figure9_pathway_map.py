from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root from plot_figure9_pathway_map")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import RESULTS_DIR
from plotting.scripts.plot_public_figure_utils import clean_dir, default_explanation_root, default_plots_root, run_python_script, write_publish_manifest
from scripts.main_figure.plot_style_bioinfo import (
    PANEL_LABEL_SIZE,
    SMALL_TEXT_SIZE,
    SUBTITLE_SIZE,
    TITLE_SIZE,
    TEXT_DARK,
    add_panel_label,
    save_publication_figure,
    set_bioinfo_style,
    wrap_text,
)


TASKS = ["ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"]
PANEL_LABELS = ["A", "B", "C", "D"]
DRUG_COLOR = "#44C1D4"
GENE_COLOR = "#E8923F"
PATHWAY_COLOR = "#9B8CF2"
EDGE_DRUG_GENE = "#89AFFF"
EDGE_GENE_PATH = "#B7A9FF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Figure 9 pathway map from existing Figure 8 enrichment outputs.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(default_plots_root(9)))
    return parser.parse_args()


def parse_overlap_genes(value: str) -> list[str]:
    return [gene.strip() for gene in str(value).split(",") if gene.strip()]


def task_file_stem(task_name: str) -> str:
    return task_name.lower()


def ensure_inputs(results_root: Path) -> Path:
    explanation_root = default_explanation_root("fig7")
    required = [explanation_root / "tables" / f"fig7_{task_file_stem(task)}_kegg_panel_terms.csv" for task in TASKS]
    if all(path.exists() for path in required):
        return explanation_root
    render_script = PROJECT_ROOT / "plotting" / "scripts" / "plot_figure8_enrichment_summary.py"
    run_python_script(render_script, "--results-root", str(results_root), "--output-dir", str(default_plots_root(8)))
    return explanation_root


def load_task_inputs(explanation_root: Path, task_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = task_file_stem(task_name)
    tables = explanation_root / "tables"
    kegg_path = tables / f"fig7_{stem}_kegg_panel_terms.csv"
    genes_path = tables / f"fig7_{stem}_top_genes_per_drug.csv"
    if not kegg_path.exists() or not genes_path.exists():
        raise FileNotFoundError(f"Missing task inputs for {task_name}")
    return pd.read_csv(kegg_path), pd.read_csv(genes_path)


def choose_tripartite_subset(kegg_df: pd.DataFrame, top_genes: pd.DataFrame) -> dict[str, object]:
    pathway_rank = (
        kegg_df.groupby("term_name")
        .agg(hits=("hits", "max"), total_score=("total_score", "max"))
        .sort_values(["hits", "total_score", "term_name"], ascending=[False, False, True])
    )
    selected_pathways = pathway_rank.head(4).index.tolist()
    reduced = kegg_df.loc[kegg_df["term_name"].isin(selected_pathways)].copy()
    drug_rank = (
        reduced.groupby("drug_label")
        .agg(total_score=("total_score", "max"), terms=("term_name", "nunique"))
        .sort_values(["terms", "total_score", "drug_label"], ascending=[False, False, True])
    )
    selected_drugs = drug_rank.head(4).index.tolist()
    reduced = reduced.loc[reduced["drug_label"].isin(selected_drugs)].copy()

    overlap_counter: Counter[str] = Counter()
    for genes in reduced["overlap_genes"].fillna("").astype(str):
        overlap_counter.update(parse_overlap_genes(genes))

    gene_rank_df = (
        top_genes.loc[top_genes["drug_label"].isin(selected_drugs)]
        .groupby("gene_symbol")
        .agg(abs_sum_score=("abs_sum_score", "max"), min_rank=("rank_within_drug", "min"))
        .reset_index()
    )
    gene_rank_df["overlap_hits"] = gene_rank_df["gene_symbol"].map(lambda gene: overlap_counter.get(str(gene), 0))
    gene_rank_df = gene_rank_df.loc[gene_rank_df["overlap_hits"] > 0].copy()
    gene_rank_df.sort_values(["overlap_hits", "abs_sum_score", "min_rank", "gene_symbol"], ascending=[False, False, True, True], inplace=True)
    selected_genes = gene_rank_df.head(6)["gene_symbol"].astype(str).tolist()

    drug_gene_edges: list[dict[str, object]] = []
    for drug in selected_drugs:
        sub = top_genes.loc[(top_genes["drug_label"] == drug) & (top_genes["gene_symbol"].isin(selected_genes))].copy()
        sub.sort_values(["rank_within_drug", "abs_sum_score"], ascending=[True, False], inplace=True)
        for row in sub.head(4).itertuples(index=False):
            drug_gene_edges.append(
                {
                    "drug_label": row.drug_label,
                    "gene_symbol": row.gene_symbol,
                    "weight": float(row.abs_sum_score),
                }
            )

    gene_pathway_edges: list[dict[str, object]] = []
    for row in reduced.itertuples(index=False):
        for gene in parse_overlap_genes(row.overlap_genes):
            if gene in selected_genes:
                gene_pathway_edges.append(
                    {
                        "gene_symbol": gene,
                        "term_name": row.term_name,
                        "drug_label": row.drug_label,
                        "weight": float(row.neglog10_fdr),
                    }
                )

    if not drug_gene_edges:
        for gene in selected_genes[:2]:
            drug_gene_edges.append({"drug_label": selected_drugs[0], "gene_symbol": gene, "weight": 1.0})
    if not gene_pathway_edges:
        for gene in selected_genes[:2]:
            gene_pathway_edges.append({"gene_symbol": gene, "term_name": selected_pathways[0], "drug_label": selected_drugs[0], "weight": 1.0})

    return {
        "selected_drugs": selected_drugs,
        "selected_genes": selected_genes,
        "selected_pathways": selected_pathways,
        "drug_gene_edges": pd.DataFrame(drug_gene_edges),
        "gene_pathway_edges": pd.DataFrame(gene_pathway_edges),
    }


def positions_for(items: list[str], *, x: float, top: float = 0.88, bottom: float = 0.12) -> dict[str, tuple[float, float]]:
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]: (x, 0.50)}
    ys = [top - idx * ((top - bottom) / (len(items) - 1)) for idx in range(len(items))]
    return {item: (x, y) for item, y in zip(items, ys)}


def curved_edge(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str, lw: float, alpha: float) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    verts = [(x0, y0), (x0 + dx * 0.35, y0), (x1 - dx * 0.35, y1), (x1, y1)]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round", zorder=1))


def draw_panel(ax: plt.Axes, *, task_name: str, panel_label: str, panel_data: dict[str, object]) -> None:
    ax.set_axis_off()
    drugs = panel_data["selected_drugs"]
    genes = panel_data["selected_genes"]
    pathways = panel_data["selected_pathways"]
    dg_edges = panel_data["drug_gene_edges"]
    gp_edges = panel_data["gene_pathway_edges"]

    drug_pos = positions_for(drugs, x=0.15, top=0.84, bottom=0.16)
    gene_pos = positions_for(genes, x=0.46, top=0.80, bottom=0.20)
    path_pos = positions_for(pathways, x=0.74, top=0.82, bottom=0.18)

    dg_max = max(float(dg_edges["weight"].max()), 1.0)
    gp_max = max(float(gp_edges["weight"].max()), 1.0)
    for row in dg_edges.itertuples(index=False):
        curved_edge(ax, drug_pos[row.drug_label], gene_pos[row.gene_symbol], color=EDGE_DRUG_GENE, lw=0.8 + 1.2 * (float(row.weight) / dg_max), alpha=0.32)
    for row in gp_edges.itertuples(index=False):
        curved_edge(ax, gene_pos[row.gene_symbol], path_pos[row.term_name], color=EDGE_GENE_PATH, lw=0.8 + 1.1 * (float(row.weight) / gp_max), alpha=0.30)

    ax.add_patch(Rectangle((0.03, 0.06), 0.94, 0.86, fill=False, linewidth=1.05, edgecolor="#334155", zorder=10))

    for drug, (x, y) in drug_pos.items():
        ax.add_patch(Circle((x, y), radius=0.017, facecolor=DRUG_COLOR, edgecolor="white", linewidth=0.9, zorder=3))
        ax.text(x - 0.035, y, wrap_text(drug, width=13, max_lines=2), ha="right", va="center", fontsize=8.4, color=TEXT_DARK)
    for gene, (x, y) in gene_pos.items():
        ax.add_patch(Circle((x, y), radius=0.0185, facecolor=GENE_COLOR, edgecolor="white", linewidth=0.95, zorder=4))
        ax.text(x, y - 0.036, gene, ha="center", va="top", fontsize=8.4, color=TEXT_DARK)
    for pathway, (x, y) in path_pos.items():
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.050, y - 0.022),
                0.10,
                0.040,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                facecolor="#F2EFFF",
                edgecolor=PATHWAY_COLOR,
                linewidth=0.9,
                zorder=3,
            )
        )
        ax.text(x + 0.062, y, wrap_text(pathway, width=15, max_lines=3), ha="left", va="center", fontsize=8.0, color=TEXT_DARK)

    ax.text(0.10, 0.95, "Drug", ha="center", va="bottom", fontsize=SUBTITLE_SIZE, color="#0F6C7D")
    ax.text(0.48, 0.95, "Gene", ha="center", va="bottom", fontsize=SUBTITLE_SIZE, color="#A25714")
    ax.text(0.74, 0.95, "Pathway", ha="center", va="bottom", fontsize=SUBTITLE_SIZE, color="#6653C6")
    ax.text(0.5, 1.02, task_name.replace("_", " "), ha="center", va="bottom", fontsize=TITLE_SIZE, color=TEXT_DARK, transform=ax.transAxes)
    add_panel_label(ax, panel_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def export_tables(output_dir: Path, task_name: str, panel_data: dict[str, object]) -> list[str]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stem = task_name.lower()
    nodes = []
    for label, node_type in (
        *[(item, "drug") for item in panel_data["selected_drugs"]],
        *[(item, "gene") for item in panel_data["selected_genes"]],
        *[(item, "pathway") for item in panel_data["selected_pathways"]],
    ):
        nodes.append({"node_label": label, "node_type": node_type})
    node_path = tables_dir / f"fig9_{stem}_nodes.csv"
    pd.DataFrame(nodes).to_csv(node_path, index=False, encoding="utf-8-sig")
    dg_path = tables_dir / f"fig9_{stem}_drug_gene_edges.csv"
    gp_path = tables_dir / f"fig9_{stem}_gene_pathway_edges.csv"
    panel_data["drug_gene_edges"].to_csv(dg_path, index=False, encoding="utf-8-sig")
    panel_data["gene_pathway_edges"].to_csv(gp_path, index=False, encoding="utf-8-sig")
    return [str(node_path), str(dg_path), str(gp_path)]


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    clean_dir(output_dir)
    panels_dir = output_dir / "panels"
    composite_dir = output_dir / "composite"
    panels_dir.mkdir(parents=True, exist_ok=True)
    composite_dir.mkdir(parents=True, exist_ok=True)
    set_bioinfo_style()

    explanation_root = ensure_inputs(results_root)
    panel_records: list[dict[str, object]] = []
    panel_files: list[str] = []
    fig = plt.figure(figsize=(15.6, 16.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.16, wspace=0.12)

    for idx, task_name in enumerate(TASKS):
        kegg_df, top_genes = load_task_inputs(explanation_root, task_name)
        panel_data = choose_tripartite_subset(kegg_df, top_genes)
        export_paths = export_tables(output_dir, task_name, panel_data)

        fig_panel, ax_panel = plt.subplots(figsize=(8.4, 6.8))
        draw_panel(ax_panel, task_name=task_name, panel_label=PANEL_LABELS[idx], panel_data=panel_data)
        panel_prefix = panels_dir / f"fig9_{task_name.lower()}_pathway_panel"
        rendered_panel = [str(path) for path in save_publication_figure(fig_panel, panel_prefix)]
        panel_files.extend(rendered_panel)

        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        draw_panel(ax, task_name=task_name, panel_label=PANEL_LABELS[idx], panel_data=panel_data)
        panel_records.append({"task": task_name, "panel_label": PANEL_LABELS[idx], "files": rendered_panel, "tables": export_paths})

    fig.suptitle("Drug-Gene-Pathway Activation Overview", fontsize=11.2, y=0.988, fontweight="normal", color=TEXT_DARK)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.955, bottom=0.035)
    composite_files = [str(path) for path in save_publication_figure(fig, composite_dir / "fig9_pathway_overview")]

    manifest = write_publish_manifest(
        output_dir,
        {
            "figure": "fig9",
            "source": str(explanation_root),
            "panels": panel_records,
            "composite_files": composite_files,
        },
        stem="fig9_pathway_overview",
    )
    print(json.dumps({"figure": "fig9", "output_dir": str(output_dir), "manifest": str(manifest), "composite_files": composite_files}, ensure_ascii=False))


if __name__ == "__main__":
    main()
