# Reproduction guide

## Primary MFMR experiment

The frozen protocol is `strict_t0_random_entry_masking_mfmr_v1`. It evaluates
CCLE, ERKAUC30, ERKIC50, PI3KAUC, and PI3KIC50 with seeds 0-9 and ten
random-entry folds. Each observed response belongs to exactly one fold for each
seed. Methods within a dataset and seed share the same mask.

The exact assignments are stored as:

`splits/primary_10x10/<dataset>_seed<seed>.npz`

In the actual repository the directory is:

`splits/primary_10x10/`

`fold_id` is zero for unobserved entries and 1-10 for observed entries. The
archived release results contain the corresponding protocol-audit files and
fold-mask hashes.

After reconstructing the upstream inputs according to `data/README.md`, run
the MFMR methods from the repository root:

```bash
python src/MCLRP_MFMR/run_t0_pipeline.py \
  --datasets CCLE ERKAUC30 ERKIC50 PI3KAUC PI3KIC50 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --methods global_mean row_col_mean imputer_only ridge_only mfmr_base \
            mfmr_no_expression mfmr_no_row_stats mfmr_no_imputer mfmr_no_ridge \
  --num-folds 10 --save-predictions \
  --output-dir results/reproduced_primary
```

Validate reproduced outputs with:

```bash
python src/MCLRP_MFMR/validate_t0_results.py \
  --results-dir results/reproduced_primary
```

## Frozen manuscript results

The manuscript's frozen numerical outputs are not tracked in the main Git
history. Download the `MCLRP-MFMR-results-v1.0.zip` asset from the corresponding
GitHub Release and extract it separately when you need to inspect the archived
paper results.

The archive preserves the `results/...` directory layout used by the analysis
and plotting scripts.

## Original MCLRP comparator boundary

The manuscript's primary comparison includes the packaged original-MCLRP
implementation and a train-only calibration diagnostic.

The upstream MCLRP source and local translated copies are intentionally omitted
from this public repository because a clear redistribution licence was not
identified in the upstream software locations checked for this release. See
`THIRD_PARTY.md`.

Consequently, the public repository can rerun the MFMR branch and inspect the
archived comparator-derived outputs in the release asset. Exact re-execution of
the packaged Python-translated MCLRP comparator requires separately authorized
source.

## Reconstructed MCLRP component ablation

The fixed configuration is recorded in:

`configs/mclrp_ablation_5task_10x10.json`

Analysis and plotting scripts that operate on the archived derived outputs are
retained in the repository. The completed derived outputs are provided in the
separate result archive.

## Statistical reporting

Primary fold-, seed-, and method-level summaries, paired statistical tests,
protocol audits, reconstructed-ablation seed tests, and hierarchical-bootstrap
outputs are included in the separate frozen result archive.

Drugs sharing a response matrix should not be treated as independent biological
samples.
