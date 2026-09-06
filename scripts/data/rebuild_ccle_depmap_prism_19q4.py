from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import CCLE_EXTERNAL_DATA_DIR, CCLE_STANDARDIZED_DIR
from core.strict_random_mutation_cv import HOTSPOT_PATTERNS, MAPK_GENES, PI3K_GENES


DEP_MAP_RELEASE = "DepMap Public 19Q4"
PRISM_RELEASE = "PRISM Repurposing 19Q4"
DOWNLOADS = {
    "CCLE_expression.csv": "https://ndownloader.figshare.com/files/20234346",
    "CCLE_mutations.csv": "https://ndownloader.figshare.com/files/20274747",
    "secondary-screen-dose-response-curve-parameters.csv": "https://ndownloader.figshare.com/files/20237739",
    "secondary-screen-cell-line-info.csv": "https://ndownloader.figshare.com/files/20237769",
}
DEP_MAP_DIR = CCLE_EXTERNAL_DATA_DIR / "depmap_public_19Q4"
PRISM_DIR = CCLE_EXTERNAL_DATA_DIR / "prism_repurposing_19Q4"
MANIFEST_PATH = CCLE_STANDARDIZED_DIR / "manifest.json"
BUNDLE_PATH = CCLE_STANDARDIZED_DIR / "bundle.npz"
MUTATION_FEATURES_PATH = CCLE_STANDARDIZED_DIR / "mutation_features.csv"
SELECTED_DRUGS_PATH = CCLE_STANDARDIZED_DIR / "selected_drugs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild standardized CCLE freeze from DepMap/PRISM 19Q4.")
    parser.add_argument("--num-drugs", type=int, default=24)
    parser.add_argument("--metric", choices=("auc", "ic50"), default="auc")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def _selected_gene_columns() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for gene in list(MAPK_GENES) + list(PI3K_GENES):
        if gene not in seen:
            seen.add(gene)
            ordered.append(gene)
    return ordered


def download_file(session: requests.Session, url: str, out_path: Path, timeout_sec: int, force: bool) -> None:
    if out_path.exists() and not force:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, timeout=timeout_sec, stream=True) as response:
        response.raise_for_status()
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def parse_expression_columns(columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gene_symbols: list[str] = []
    gene_ids: list[str] = []
    for column in columns:
        text = str(column)
        match = re.match(r"^(?P<symbol>.+?)\s+\((?P<gene_id>[^)]+)\)$", text)
        if match:
            gene_symbols.append(match.group("symbol"))
            gene_ids.append(match.group("gene_id"))
        else:
            gene_symbols.append(text)
            gene_ids.append(text)
    return np.asarray(gene_symbols, dtype=object), np.asarray(gene_ids, dtype=object)


def load_expression_table(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    expr = pd.read_csv(path, index_col=0, low_memory=False)
    expr.index = expr.index.astype(str)
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    gene_symbols, gene_ids = parse_expression_columns(expr.columns.tolist())
    return expr, gene_symbols, gene_ids


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("TRUE")


def _state_priority(df: pd.DataFrame) -> str:
    if bool(df["is_hotspot"].any()):
        return "hotspot::nci"
    if bool(df["is_damaging"].any()):
        return "damaging::nci"
    if bool(df["is_nonsilent"].any()):
        return "mut::nci"
    return "wt::nci"


def build_mutation_feature_table(
    mutation_path: Path,
    cell_info: pd.DataFrame,
    ordered_depmap_ids: list[str],
) -> pd.DataFrame:
    selected_genes = _selected_gene_columns()
    mutation = pd.read_csv(
        mutation_path,
        usecols=[
            "DepMap_ID",
            "Hugo_Symbol",
            "Variant_annotation",
            "Variant_Classification",
            "isDeleterious",
            "isTCGAhotspot",
            "isCOSMIChotspot",
            "Tumor_Sample_Barcode",
        ],
        low_memory=False,
    )
    mutation["DepMap_ID"] = mutation["DepMap_ID"].astype(str)
    mutation["Hugo_Symbol"] = mutation["Hugo_Symbol"].astype(str)
    mutation = mutation[mutation["Hugo_Symbol"].isin(selected_genes)].copy()
    mutation["Variant_annotation"] = mutation["Variant_annotation"].astype(str).str.lower()
    mutation["is_damaging"] = _bool_series(mutation["isDeleterious"]) | mutation["Variant_annotation"].eq("damaging")
    mutation["is_hotspot"] = _bool_series(mutation["isTCGAhotspot"]) | _bool_series(mutation["isCOSMIChotspot"])
    mutation["is_nonsilent"] = ~mutation["Variant_annotation"].eq("silent")
    grouped = (
        mutation.groupby(["DepMap_ID", "Hugo_Symbol"], sort=False)[["is_damaging", "is_hotspot", "is_nonsilent"]]
        .apply(_state_priority)
        .rename("state")
        .reset_index()
    )
    pivot = grouped.pivot(index="DepMap_ID", columns="Hugo_Symbol", values="state")
    cell_meta = cell_info.drop_duplicates("depmap_id").set_index("depmap_id")

    rows: list[dict[str, object]] = []
    for row_index, depmap_id in enumerate(ordered_depmap_ids):
        base = {
            "row_index": row_index,
            "DepMap_ID": depmap_id,
            "CCLE_ID": str(cell_meta.at[depmap_id, "ccle_name"]) if depmap_id in cell_meta.index else depmap_id,
            "Cancer Type": str(cell_meta.at[depmap_id, "secondary_tissue"]) if depmap_id in cell_meta.index else "Unknown",
            "Tissue": str(cell_meta.at[depmap_id, "primary_tissue"]) if depmap_id in cell_meta.index else "Unknown",
        }
        if base["Cancer Type"] in {"", "nan", "None"}:
            base["Cancer Type"] = base["Tissue"]
        for gene in selected_genes:
            if depmap_id in pivot.index and gene in pivot.columns:
                base[gene] = str(pivot.at[depmap_id, gene]) if pd.notna(pivot.at[depmap_id, gene]) else "wt::nci"
            else:
                base[gene] = "wt::nci"
        rows.append(base)
    return pd.DataFrame(rows)


def select_compounds(response: pd.DataFrame, num_drugs: int) -> pd.DataFrame:
    selected_genes = set(_selected_gene_columns())

    def target_hit(text: str) -> bool:
        if pd.isna(text):
            return False
        tokens = [token.strip() for token in str(text).replace(";", ",").split(",")]
        return any(token in selected_genes for token in tokens)

    response = response.copy()
    response["passed_str_profiling"] = response["passed_str_profiling"].astype(str).str.upper().eq("TRUE")
    response = response[response["passed_str_profiling"]].copy()
    meta = response[["broad_id", "name", "target"]].drop_duplicates("broad_id").copy()
    meta["target_hit"] = meta["target"].map(target_hit)
    coverage = (
        response.groupby("broad_id", sort=False)["auc"]
        .apply(lambda series: int(series.notna().sum()))
        .rename("coverage")
        .reset_index()
    )
    meta = meta.merge(coverage, on="broad_id", how="left").fillna({"coverage": 0})
    chosen = meta[meta["target_hit"]].sort_values(["coverage", "name", "broad_id"], ascending=[False, True, True])
    if len(chosen) < num_drugs:
        filler = meta[~meta["broad_id"].isin(chosen["broad_id"])].sort_values(
            ["coverage", "name", "broad_id"],
            ascending=[False, True, True],
        )
        chosen = pd.concat([chosen, filler], ignore_index=True)
    return chosen.head(num_drugs).reset_index(drop=True)


def build_response_matrix(
    response_path: Path,
    selected_compounds: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["depmap_id", "ccle_name", "broad_id", "auc", "ic50", "passed_str_profiling"]
    response = pd.read_csv(response_path, usecols=usecols, low_memory=False)
    response["depmap_id"] = response["depmap_id"].astype(str)
    response["passed_str_profiling"] = response["passed_str_profiling"].astype(str).str.upper().eq("TRUE")
    response = response[response["passed_str_profiling"]].copy()
    metric_col = metric.lower()
    response = response[response["broad_id"].isin(selected_compounds["broad_id"])].copy()
    response["drug_label"] = response["broad_id"].map(
        {
            row.broad_id: f"{row.name}::{row.broad_id}"
            for row in selected_compounds.itertuples(index=False)
        }
    )
    response_mat = response.pivot_table(index="depmap_id", columns="drug_label", values=metric_col, aggfunc="mean")
    ordered_labels = [f"{row.name}::{row.broad_id}" for row in selected_compounds.itertuples(index=False)]
    response_mat = response_mat.reindex(columns=ordered_labels)
    return response, response_mat


def main() -> None:
    args = parse_args()
    session = requests.Session()
    DEP_MAP_DIR.mkdir(parents=True, exist_ok=True)
    PRISM_DIR.mkdir(parents=True, exist_ok=True)
    CCLE_STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DOWNLOADS.items():
        target_dir = DEP_MAP_DIR if filename.startswith("CCLE_") else PRISM_DIR
        download_file(session, url, target_dir / filename, args.timeout_sec, args.force_download)

    expression_path = DEP_MAP_DIR / "CCLE_expression.csv"
    mutation_path = DEP_MAP_DIR / "CCLE_mutations.csv"
    response_path = PRISM_DIR / "secondary-screen-dose-response-curve-parameters.csv"
    cell_info_path = PRISM_DIR / "secondary-screen-cell-line-info.csv"

    expr_df, gene_symbols, gene_ids = load_expression_table(expression_path)
    response_probe = pd.read_csv(
        response_path,
        usecols=["depmap_id", "broad_id", "name", "target", "auc", "passed_str_profiling"],
        low_memory=False,
    )
    selected_compounds = select_compounds(response_probe, args.num_drugs)
    _, response_mat = build_response_matrix(response_path, selected_compounds, args.metric)
    cell_info = pd.read_csv(
        cell_info_path,
        usecols=["depmap_id", "ccle_name", "primary_tissue", "secondary_tissue", "passed_str_profiling"],
        low_memory=False,
    )
    cell_info["depmap_id"] = cell_info["depmap_id"].astype(str)
    cell_info["passed_str_profiling"] = cell_info["passed_str_profiling"].astype(str).str.upper().eq("TRUE")
    cell_info = cell_info[cell_info["passed_str_profiling"]].copy()

    common_ids = [depmap_id for depmap_id in response_mat.index.astype(str).tolist() if depmap_id in expr_df.index]
    common_ids = [depmap_id for depmap_id in common_ids if depmap_id in set(cell_info["depmap_id"].tolist())]
    if not common_ids:
        raise RuntimeError("No common CCLE cell lines after aligning expression, PRISM response, and cell metadata.")

    mutation_features = build_mutation_feature_table(mutation_path, cell_info, common_ids)
    common_ids = [depmap_id for depmap_id in common_ids if depmap_id in set(mutation_features["DepMap_ID"].tolist())]
    mutation_features = mutation_features.set_index("DepMap_ID").loc[common_ids].reset_index()

    X = expr_df.loc[common_ids].to_numpy(dtype=np.float32)
    M = np.nan_to_num(response_mat.reindex(common_ids).to_numpy(dtype=np.float32), nan=0.0).astype(np.float32)
    drug_labels = np.asarray(response_mat.columns.astype(str).tolist(), dtype=object)
    cell_ids = np.asarray(common_ids, dtype=object)

    np.savez_compressed(
        BUNDLE_PATH,
        X=X,
        M=M,
        cell_ids=cell_ids,
        drug_labels=drug_labels,
        gene_symbols=gene_symbols,
        gene_ids=gene_ids,
        metric=np.asarray(args.metric, dtype=object),
        release=np.asarray(f"{DEP_MAP_RELEASE} + {PRISM_RELEASE}", dtype=object),
    )
    mutation_features.to_csv(MUTATION_FEATURES_PATH, index=False, encoding="utf-8-sig")
    selected_compounds.to_csv(SELECTED_DRUGS_PATH, index=False, encoding="utf-8-sig")

    manifest = {
        "freeze_name": "ccle_depmap_prism_19q4",
        "depmap_release": DEP_MAP_RELEASE,
        "prism_release": PRISM_RELEASE,
        "metric": args.metric,
        "num_selected_drugs": int(len(selected_compounds)),
        "num_cells": int(X.shape[0]),
        "num_genes": int(X.shape[1]),
        "num_observed": int(np.count_nonzero(M)),
        "bundle_path": str(BUNDLE_PATH),
        "mutation_features_path": str(MUTATION_FEATURES_PATH),
        "selected_drugs_path": str(SELECTED_DRUGS_PATH),
        "downloads": DOWNLOADS,
        "selection_policy": (
            "Select compounds whose annotated targets intersect the MAPK/PI3K gene panel; "
            "rank by non-null AUC coverage among STR-profiled lines; fill remaining slots by highest overall coverage."
        ),
        "notes": [
            "This freeze replaces legacy CCLE_X/MMnormal inputs for active benchmarks.",
            "The response matrix is derived from PRISM secondary-screen dose-response parameters.",
            "Mutation features are aligned directly by DepMap_ID without external row-order recovery.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
