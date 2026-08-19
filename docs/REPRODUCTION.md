# Reproduction guide

## Primary MFMR experiment

The frozen protocol is `strict_t0_random_entry_masking_mfmr_v1`. It evaluates CCLE, ERKAUC30, ERKIC50, PI3KAUC, and PI3KIC50 with seeds 0–9 and ten random-entry folds. Each observed response belongs to exactly one fold for each seed. Methods within a dataset and seed share the same mask.

The exact assignments are stored as `splits/primary_10x10/<dataset>_seed<seed>.npz`. `fold_id` is zero for unobserved entries and 1–10 for observed entries. Every fold mask was verified against `fold_mask_sha256` in `results/primary_final_lock/protocol_audit.csv`.

After reconstructing the upstream inputs according to `data/README.md`, run the MFMR methods from the repository root:

```bash
python src/MCLRP_MFMR/run_t0_pipeline.py \
  --datasets CCLE ERKAUC30 ERKIC50 PI3KAUC PI3KIC50 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --methods global_mean row_col_mean imputer_only ridge_only mfmr_base \
            mfmr_no_expression mfmr_no_row_stats mfmr_no_imputer mfmr_no_ridge \
  --num-folds 10 --save-predictions \
  --output-dir results/reproduced_primary
```

Validate outputs with:

```bash
python src/MCLRP_MFMR/validate_t0_results.py --results-dir results/reproduced_primary
```

## Original MCLRP comparator boundary

The manuscript's primary comparison includes the packaged original-MCLRP implementation and a train-only calibration diagnostic. Frozen outputs and statistical comparisons are included under `results/primary_final_lock/`.

The upstream MCLRP source and local translated copies are intentionally omitted from this public release because a clear redistribution license was not identified in the upstream public software locations checked for this release. See `THIRD_PARTY.md`.

Consequently, this public package can rerun the MFMR branch and inspect the archived original-MCLRP comparison, but exact re-execution of the packaged Python-translated MCLRP comparator requires separately authorized source.

## Reconstructed MCLRP component ablation

The fixed configuration is recorded in `configs/mclrp_ablation_5task_10x10.json`. The completed derived outputs are retained in `results/mclrp_ablation_5task_10x10/`, including OOF predictions, summary tables, and analysis products.

The executable runner that depended on the locally translated MCLRP source is omitted from the public release. Analysis/plotting scripts that operate on the archived derived outputs remain available.

## Statistical reporting

Primary fold-, seed-, and method-level summaries are in `results/primary_final_lock`. The reconstructed-ablation seed tests and hierarchical-bootstrap outputs are in `results/mclrp_ablation_5task_10x10/analysis`. Drugs sharing a response matrix should not be treated as independent biological samples.
