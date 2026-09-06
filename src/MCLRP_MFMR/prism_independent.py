from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.rebuild_ccle_depmap_prism_19q4 import (
    build_mutation_feature_table,
    build_response_matrix,
    load_expression_table,
    select_compounds,
)


RAW_FILENAMES = {
    "expression": "CCLE_expression.csv",
    "mutation": "CCLE_mutations.csv",
    "response": "secondary-screen-dose-response-curve-parameters.csv",
    "cell_info": "secondary-screen-cell-line-info.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_prism_independent_bundle(
    *,
    raw_dir: Path,
    output_dir: Path,
    num_drugs: int = 24,
    metric: str = "auc",
) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    paths = {name: raw_dir / filename for name, filename in RAW_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PRISM/DepMap inputs: {missing}")

    expression, gene_symbols, gene_ids = load_expression_table(paths["expression"])
    response_probe = pd.read_csv(
        paths["response"],
        usecols=["depmap_id", "broad_id", "name", "target", "auc", "passed_str_profiling"],
        low_memory=False,
    )
    selected = select_compounds(response_probe, int(num_drugs))
    _, response_matrix = build_response_matrix(paths["response"], selected, metric)

    cell_info = pd.read_csv(
        paths["cell_info"],
        usecols=["depmap_id", "ccle_name", "primary_tissue", "secondary_tissue", "passed_str_profiling"],
        low_memory=False,
    )
    cell_info["depmap_id"] = cell_info["depmap_id"].astype(str)
    cell_info = cell_info[cell_info["passed_str_profiling"].astype(str).str.upper().eq("TRUE")].copy()
    allowed_ids = set(cell_info["depmap_id"])
    common_ids = [
        depmap_id
        for depmap_id in response_matrix.index.astype(str)
        if depmap_id in expression.index and depmap_id in allowed_ids
    ]
    if not common_ids:
        raise RuntimeError("No common cell lines across PRISM response, DepMap expression, and metadata")

    mutation_ids = set(
        pd.read_csv(paths["mutation"], usecols=["DepMap_ID"], low_memory=False)["DepMap_ID"]
        .dropna()
        .astype(str)
    )
    mutation_features = build_mutation_feature_table(paths["mutation"], cell_info, common_ids)
    mutation_features["mutation_available"] = mutation_features["DepMap_ID"].isin(mutation_ids)
    metadata_columns = {
        "row_index",
        "DepMap_ID",
        "CCLE_ID",
        "Cancer Type",
        "Tissue",
        "mutation_available",
    }
    gene_columns = [column for column in mutation_features.columns if column not in metadata_columns]
    unavailable = ~mutation_features["mutation_available"]
    mutation_features.loc[unavailable, gene_columns] = "na::nci"

    raw_response = response_matrix.reindex(common_ids).to_numpy(dtype=np.float32)
    observed_mask = np.isfinite(raw_response)
    matrix = np.nan_to_num(raw_response, nan=0.0).astype(np.float32)
    values = expression.loc[common_ids].to_numpy(dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "bundle.npz",
        X=values,
        M=matrix,
        observed_mask=observed_mask,
        cell_ids=np.asarray(common_ids, dtype=object),
        drug_labels=np.asarray(response_matrix.columns.astype(str), dtype=object),
        gene_symbols=gene_symbols,
        gene_ids=gene_ids,
        metric=np.asarray(metric, dtype=object),
        release=np.asarray("DepMap Public 19Q4 + PRISM Repurposing 19Q4", dtype=object),
    )
    mutation_features.to_csv(output_dir / "mutation_features.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "selected_drugs.csv", index=False, encoding="utf-8-sig")

    manifest: dict[str, object] = {
        "freeze_name": "prism19q4_independent_locked_test",
        "depmap_release": "DepMap Public 19Q4",
        "prism_release": "PRISM Repurposing 19Q4",
        "metric": metric,
        "num_selected_drugs": int(len(selected)),
        "num_cells": int(values.shape[0]),
        "num_genes": int(values.shape[1]),
        "num_observed": int(observed_mask.sum()),
        "num_mutation_available_cells": int(mutation_features["mutation_available"].sum()),
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "selection_policy": (
            "Select compounds whose annotated targets intersect the frozen MAPK/PI3K panel, "
            "rank by non-null AUC coverage among STR-profiled lines, and fill remaining slots by coverage."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
