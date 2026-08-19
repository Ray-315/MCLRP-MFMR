from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from project_paths import GDSC_STANDARDIZED_DIR

from .strict_random_mutation_cv import HOTSPOT_PATTERNS, MAPK_GENES, PI3K_GENES


GDSC_MUTATION_FEATURES_FILE = GDSC_STANDARDIZED_DIR / "mutation_features.csv"
GDSC_MUTATION_FEATURES_SUMMARY_FILE = GDSC_STANDARDIZED_DIR / "manifest.json"
GDSC_MUTATION_METADATA_COLUMNS = ("model_id", "model_name", "Cancer Type", "Tissue")
GENE_ALIASES = {
    "EGFR.1": "EGFR",
    "FGFR3.1": "FGFR3",
}


@dataclass(frozen=True)
class GDSCMutationFeatureParts:
    tissue: np.ndarray
    mapk: np.ndarray
    pi3k: np.ndarray
    overall: np.ndarray
    binary: np.ndarray
    model_ids: np.ndarray


def _selected_gene_columns() -> list[str]:
    wanted = list(MAPK_GENES) + list(PI3K_GENES)
    ordered: list[str] = []
    seen: set[str] = set()
    for gene in wanted:
        if gene not in seen:
            seen.add(gene)
            ordered.append(gene)
    return ordered


def _parse_gene_state(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = series.fillna("wt::nci").astype(str)
    mut = s.str.split("::").str[0].fillna("wt")
    cn = s.str.split("::").str[-1].fillna("nci")
    mutated = ((~mut.str.startswith("wt")) & (~mut.str.startswith("na"))).astype(np.float32).to_numpy()
    amp = cn.str.contains(">=8").astype(np.float32).to_numpy()
    loss = cn.eq("0").astype(np.float32).to_numpy()
    return mutated, amp, loss, mut.to_numpy()


def _stack_mean(arrays: list[np.ndarray], length: int) -> np.ndarray:
    if not arrays:
        return np.zeros(length, dtype=np.float32)
    return np.mean(np.vstack(arrays), axis=0).astype(np.float32)


def _pathway_summary(gene_set: Sequence[str], df: pd.DataFrame) -> np.ndarray:
    muts, amps, losses, hots = [], [], [], []
    for gene in gene_set:
        if gene not in df.columns:
            fallback = GENE_ALIASES.get(gene, gene)
            if fallback not in df.columns:
                continue
            gene = fallback
        m, a, l, tok = _parse_gene_state(df[gene])
        muts.append(m)
        amps.append(a)
        losses.append(l)
        base_gene = GENE_ALIASES.get(gene, gene)
        if base_gene in HOTSPOT_PATTERNS:
            hotspot = (
                pd.Series(tok)
                .str.contains(HOTSPOT_PATTERNS[base_gene], regex=True, na=False)
                .astype(np.float32)
                .to_numpy()
            )
            hots.append(hotspot)
    return np.vstack(
        [
            _stack_mean(muts, len(df)),
            _stack_mean(amps, len(df)),
            _stack_mean(losses, len(df)),
            _stack_mean(hots, len(df)),
        ]
    ).T.astype(np.float32)


def load_gdsc_mutation_table(feature_path: Path | None = None) -> pd.DataFrame:
    path = Path(feature_path) if feature_path is not None else GDSC_MUTATION_FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"GDSC mutation/CNV feature table not found: {path}. "
            "Run scripts/data/prepare_gdsc_mutation_features.py first, or pass --mfmr-mut-feature-file."
        )
    df = pd.read_csv(path, low_memory=False)
    if "model_id" not in df.columns:
        raise ValueError(f"Invalid GDSC mutation feature table: missing 'model_id' column in {path}")
    df["model_id"] = df["model_id"].astype(str)
    return df


def build_gdsc_mutation_feature_parts(
    cell_ids: Sequence[str],
    *,
    feature_path: Path | None = None,
) -> GDSCMutationFeatureParts:
    feature_table = load_gdsc_mutation_table(feature_path=feature_path)
    feature_table = feature_table.drop_duplicates("model_id", keep="first").set_index("model_id")
    target_index = pd.Index([str(cell_id) for cell_id in cell_ids], name="model_id")
    aligned = feature_table.reindex(target_index).copy()
    for col in GDSC_MUTATION_METADATA_COLUMNS[1:]:
        if col not in aligned.columns:
            aligned[col] = "Unknown"
        aligned[col] = aligned[col].fillna("Unknown").astype(str)

    selected_gene_columns = _selected_gene_columns()
    for gene in selected_gene_columns:
        if gene in aligned.columns:
            aligned[gene] = aligned[gene].fillna("wt::nci").astype(str)
            continue
        canonical = GENE_ALIASES.get(gene, gene)
        if canonical in aligned.columns:
            aligned[gene] = aligned[canonical].fillna("wt::nci").astype(str)
        else:
            aligned[gene] = "wt::nci"

    tissue = (
        pd.get_dummies(aligned[["Cancer Type", "Tissue"]].astype(str), dummy_na=False)
        .astype(np.float32)
        .to_numpy()
    )
    binary_feats: list[np.ndarray] = []
    mut_feats: list[np.ndarray] = []
    amp_feats: list[np.ndarray] = []
    loss_feats: list[np.ndarray] = []
    for gene in selected_gene_columns:
        mut, amp, loss, tok = _parse_gene_state(aligned[gene])
        mut_feats.append(mut)
        amp_feats.append(amp)
        loss_feats.append(loss)
        binary_feats.extend([mut, amp, loss])
        canonical = GENE_ALIASES.get(gene, gene)
        if canonical in HOTSPOT_PATTERNS:
            hotspot = (
                pd.Series(tok)
                .str.contains(HOTSPOT_PATTERNS[canonical], regex=True, na=False)
                .astype(np.float32)
                .to_numpy()
            )
            binary_feats.append(hotspot)
    if binary_feats:
        binary = np.vstack(binary_feats).T.astype(np.float32)
    else:
        binary = np.zeros((len(aligned), 0), dtype=np.float32)
    overall = np.vstack(
        [
            _stack_mean(mut_feats, len(aligned)),
            _stack_mean(amp_feats, len(aligned)),
            _stack_mean(loss_feats, len(aligned)),
        ]
    ).T.astype(np.float32)
    return GDSCMutationFeatureParts(
        tissue=tissue,
        mapk=_pathway_summary(MAPK_GENES, aligned),
        pi3k=_pathway_summary(PI3K_GENES, aligned),
        overall=overall,
        binary=binary,
        model_ids=target_index.to_numpy(dtype=object),
    )
