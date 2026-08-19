# Standardized Benchmark Data

This directory contains the active benchmark inputs used by the current runtime.

## CCLE

Files:

- `CCLE/bundle.npz`
- `CCLE/mutation_features.csv`
- `CCLE/selected_drugs.csv`
- `CCLE/manifest.json`

Freeze:

- expression: `DepMap Public 19Q4`
- mutations: `DepMap Public 19Q4`
- response: `PRISM Repurposing 19Q4 secondary screen`

Builder:

- `python scripts/data/rebuild_ccle_depmap_prism_19q4.py --num-drugs 24 --metric auc`

## GDSC

Files:

- `GDSC/ERK_AUC_bundle.npz`
- `GDSC/ERK_IC50_bundle.npz`
- `GDSC/PI3K_AUC_bundle.npz`
- `GDSC/PI3K_IC50_bundle.npz`
- `GDSC/mutation_features.csv`
- `GDSC/selected_drugs.csv`
- `GDSC/model_manifest.csv`
- `GDSC/response_curves.csv`
- `GDSC/manifest.json`

Freeze:

- drug response: Cell Model Passports API snapshot
- RNA-seq: processed per-model RNASeq files from the same snapshot
- mutation/CNV: Cell Model Passports API snapshot

Builder:

- `python scripts/data/rebuild_gdsc_cmp_snapshot.py --workers 4`

## Notes

- These standardized bundles are the only active `CCLE/GDSC` runtime inputs.
- Legacy `raw/CCLE` and `raw/GDSC` files are preserved only as archive material.
