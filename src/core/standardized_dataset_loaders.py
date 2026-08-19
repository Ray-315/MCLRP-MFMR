from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project_paths import CCLE_STANDARDIZED_DIR, GDSC_STANDARDIZED_DIR


@dataclass(frozen=True)
class StandardizedCCLEBundle:
    X: np.ndarray
    M: np.ndarray
    cell_ids: np.ndarray
    drug_labels: np.ndarray
    gene_symbols: np.ndarray
    manifest: dict[str, object]


@dataclass(frozen=True)
class StandardizedGDSCBundle:
    X: np.ndarray
    M: np.ndarray
    cell_ids: np.ndarray
    drug_labels: np.ndarray
    gene_ids: np.ndarray
    gene_symbols: np.ndarray
    pathway: str
    metric: str
    manifest: dict[str, object]


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_ccle_standardized_bundle(bundle_path: Path | None = None) -> StandardizedCCLEBundle:
    bundle_path = bundle_path or (CCLE_STANDARDIZED_DIR / "bundle.npz")
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing standardized CCLE bundle: {bundle_path}. "
            "Run scripts/data/rebuild_ccle_depmap_prism_19q4.py first."
        )
    payload = np.load(bundle_path, allow_pickle=True)
    manifest = _load_manifest(bundle_path.with_name("manifest.json"))
    return StandardizedCCLEBundle(
        X=payload["X"].astype(np.float32),
        M=payload["M"].astype(np.float32),
        cell_ids=payload["cell_ids"].astype(object),
        drug_labels=payload["drug_labels"].astype(object),
        gene_symbols=payload["gene_symbols"].astype(object),
        manifest=manifest,
    )


def load_gdsc_standardized_bundle(
    dataset_name: str,
    bundle_dir: Path | None = None,
) -> StandardizedGDSCBundle:
    bundle_dir = bundle_dir or GDSC_STANDARDIZED_DIR
    bundle_path = bundle_dir / f"{dataset_name}_bundle.npz"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing standardized GDSC bundle: {bundle_path}. "
            "Run scripts/data/rebuild_gdsc_cmp_snapshot.py first."
        )
    payload = np.load(bundle_path, allow_pickle=True)
    manifest = _load_manifest(bundle_dir / "manifest.json")
    return StandardizedGDSCBundle(
        X=payload["X"].astype(np.float32),
        M=payload["M"].astype(np.float32),
        cell_ids=payload["cell_ids"].astype(object),
        drug_labels=payload["drug_labels"].astype(object),
        gene_ids=payload["gene_ids"].astype(object),
        gene_symbols=payload["gene_symbols"].astype(object),
        pathway=str(payload["pathway"].item() if getattr(payload["pathway"], "shape", ()) == () else payload["pathway"]),
        metric=str(payload["metric"].item() if getattr(payload["metric"], "shape", ()) == () else payload["metric"]),
        manifest=manifest,
    )
