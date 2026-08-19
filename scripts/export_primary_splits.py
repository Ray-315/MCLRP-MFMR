from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("CCLE", "ERKAUC30", "ERKIC50", "PI3KAUC", "PI3KIC50")


def mask_sha256(mask: np.ndarray) -> str:
    arr = np.asarray(mask, dtype=np.uint8)
    shape = np.asarray(arr.shape, dtype=np.int64)
    return hashlib.sha256(shape.tobytes() + np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the frozen primary 10-seed x 10-fold assignments.")
    parser.add_argument("--project-root", type=Path, required=True, help="Original MCLRP_Python project containing data/")
    parser.add_argument("--audit", type=Path, required=True, help="Frozen protocol_audit.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from MCLRP_MFMR.t0_mfmr_protocol import load_t0_dataset
    from core.getcrossMatrixs import getcrossMatrixs

    audit = pd.read_csv(args.audit, usecols=["dataset", "seed", "fold", "fold_mask_sha256"])
    expected = audit.drop_duplicates().set_index(["dataset", "seed", "fold"])["fold_mask_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        bundle = load_t0_dataset(dataset)
        observed = bundle.M != 0
        for seed in range(10):
            folds = getcrossMatrixs(bundle.M, num_folds=10, rng=np.random.default_rng(seed))
            assignment = np.zeros(bundle.M.shape, dtype=np.int8)
            hashes: list[str] = []
            for fold_index, fold_matrix in enumerate(folds, start=1):
                mask = fold_matrix != 0
                assignment[mask] = fold_index
                actual_hash = mask_sha256(mask)
                expected_hash = str(expected.loc[(dataset, seed, fold_index)])
                if actual_hash != expected_hash:
                    raise RuntimeError(f"Fold hash mismatch: {dataset} seed={seed} fold={fold_index}")
                hashes.append(actual_hash)
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "fold": fold_index,
                    "n_test": int(mask.sum()),
                    "fold_mask_sha256": actual_hash,
                })
            if not np.array_equal(assignment > 0, observed):
                raise RuntimeError(f"Assignments do not cover observed entries exactly: {dataset} seed={seed}")
            np.savez_compressed(
                args.output_dir / f"{dataset}_seed{seed}.npz",
                fold_id=assignment,
                observed_mask=observed,
                fold_mask_sha256=np.asarray(hashes),
            )

    pd.DataFrame(rows).to_csv(args.output_dir / "fold_manifest.csv", index=False)
    metadata = {
        "protocol": "strict_t0_random_entry_masking_mfmr_v1",
        "datasets": list(DATASETS),
        "seeds": list(range(10)),
        "folds_per_seed": 10,
        "encoding": "fold_id is 0 for unobserved entries and 1..10 for held-out fold membership",
        "verification": "All 500 fold masks matched fold_mask_sha256 in the frozen protocol audit.",
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
