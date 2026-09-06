# PRISM 19Q4 locked independent test

This repository extension contains the code, locked split, predictions, metrics, bootstrap output, and manifests used for the PRISM 19Q4 independent T0 test in the Bioinformatics manuscript. It does not redistribute upstream pharmacogenomic data.

## Upstream files

Obtain the following files from [DepMap Public 19Q4](https://figshare.com/articles/dataset/DepMap_19Q4_Public/11384241) and [PRISM Repurposing 19Q4](https://depmap.org/portal/data_page/?release=PRISM+Repurposing+19Q4&tab=allData):

- `CCLE_expression.csv`
- `CCLE_mutations.csv`
- `secondary-screen-dose-response-curve-parameters.csv`
- `secondary-screen-cell-line-info.csv`

Place all four files in one raw-data directory. Their expected SHA-256 values are recorded in `data/manifests/PRISM19Q4-independent-manifest.json` and Supplementary Section S12.

## Reproduction

Run the commands from the root of the full MCLRP-MFMR repository with its documented Python environment:

```powershell
python scripts/data/build_prism_independent_bundle.py --raw-dir data/external/PRISM19Q4_independent --output-dir data/standardized/PRISM19Q4_independent --num-drugs 24 --metric auc

python scripts/benchmarks/run_prism_independent.py --bundle-dir data/standardized/PRISM19Q4_independent --output-dir results/prism19q4_independent_locked --methods original_mclrp imputer_only ridge_only mfmr_base mfmr_mutation --seeds 0 1 2 3 4 5 6 7 8 9 --split-seed 20260906 --test-fraction 0.20 --min-train-per-row 1 --min-train-per-drug 30 --min-test-per-drug 10

python scripts/benchmarks/summarize_prism_independent.py --bundle-dir data/standardized/PRISM19Q4_independent --results-dir results/prism19q4_independent_locked --reference mfmr_base --iterations 20000 --seed 20260906
```

The locked test mask is stored at `results/prism19q4_independent_locked/locked_split.npz`. It contains 2,265 entries and has SHA-256 `9596c32b7929534560d5e530f6309d49452fbeb6cd8538af52d7d61d633436e4`. The result directory contains 50 prediction matrices: five methods times ten algorithm seeds.

## Mutation scope

All 477 aligned PRISM cell lines have DepMap 19Q4 mutation records. Copy-number values were not included in this resource-specific test; the corresponding state remains `nci` and is not inferred. Upstream raw data and the expression/response bundle are excluded from this archive.
