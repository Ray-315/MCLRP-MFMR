from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from MCLRP_MFMR.prism_independent import build_prism_independent_bundle


def test_builder_aligns_prism_expression_and_mutation_availability(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "built"
    raw_dir.mkdir()

    expression = pd.DataFrame(
        {
            "TSPAN6 (7105)": [1.0, 2.0, 3.0],
            "BRAF (673)": [4.0, 5.0, 6.0],
        },
        index=["ACH-000001", "ACH-000002", "ACH-000003"],
    )
    expression.to_csv(raw_dir / "CCLE_expression.csv")

    response_rows = []
    for depmap_id in expression.index:
        for broad_id, name, target, auc in (
            ("BRD-A", "drug-a", "BRAF", 0.3),
            ("BRD-B", "drug-b", "PIK3CA", 0.6),
        ):
            response_rows.append(
                {
                    "depmap_id": depmap_id,
                    "ccle_name": depmap_id,
                    "broad_id": broad_id,
                    "name": name,
                    "target": target,
                    "auc": auc,
                    "ic50": 1.0,
                    "passed_str_profiling": True,
                }
            )
    pd.DataFrame(response_rows).to_csv(
        raw_dir / "secondary-screen-dose-response-curve-parameters.csv", index=False
    )
    pd.DataFrame(
        {
            "depmap_id": expression.index,
            "ccle_name": expression.index,
            "primary_tissue": ["lung", "lung", "skin"],
            "secondary_tissue": ["NSCLC", "NSCLC", "melanoma"],
            "passed_str_profiling": [True, True, True],
        }
    ).to_csv(raw_dir / "secondary-screen-cell-line-info.csv", index=False)
    pd.DataFrame(
        {
            "DepMap_ID": ["ACH-000001", "ACH-000002"],
            "Hugo_Symbol": ["BRAF", "PIK3CA"],
            "Variant_annotation": ["damaging", "damaging"],
            "Variant_Classification": ["Missense_Mutation", "Missense_Mutation"],
            "isDeleterious": [True, True],
            "isTCGAhotspot": [True, False],
            "isCOSMIChotspot": [True, False],
            "Tumor_Sample_Barcode": ["ACH-000001", "ACH-000002"],
        }
    ).to_csv(raw_dir / "CCLE_mutations.csv", index=False)

    manifest = build_prism_independent_bundle(
        raw_dir=raw_dir,
        output_dir=output_dir,
        num_drugs=2,
        metric="auc",
    )

    payload = np.load(output_dir / "bundle.npz", allow_pickle=True)
    mutation = pd.read_csv(output_dir / "mutation_features.csv")
    saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert payload["X"].shape == (3, 2)
    assert payload["M"].shape == (3, 2)
    assert payload["cell_ids"].tolist() == expression.index.tolist()
    assert mutation["mutation_available"].tolist() == [True, True, False]
    assert mutation.loc[2, "BRAF"].startswith("na::")
    assert manifest["num_cells"] == saved_manifest["num_cells"] == 3
    assert manifest["num_mutation_available_cells"] == 2
