# Public release packaging note

- Removed embedded upstream/legacy pharmacogenomic input files.
- Removed upstream MCLRP source and local translated copies pending a clear redistribution license.
- Retained frozen derived comparator outputs and statistical/audit artifacts.
- Sanitized machine-specific absolute paths from public manifests.

# Changelog

## Strict T0 MCLRP-MFMR Protocol

- Relocated the strict T0 implementation and CLIs into the dedicated `MCLRP_MFMR/` folder.
- Added a leak-safe T0 random-entry masking protocol in `MCLRP_MFMR/t0_mfmr_protocol.py`.
- Added train-only expression PC fitting via `fit_expr_pcs_train_only`, with train-row-only variance selection, scaling, SVD fitting, and zero padding for small folds.
- Added MFMR base prediction branches, baselines, ablations, and mutation/CNV residual-head support.
- Added canonical CLIs:
  - `run_mfmr_t0.py`
  - `run_full_benchmark_mfmr_t0.py`
- Added leakage-safety checks in `MCLRP_MFMR/tests/test_t0_strict_protocol.py`.
- Added root and `core/` compatibility exports for the new strict T0 protocol.
- Added packaged minimal input data under `MCLRP_MFMR/data/raw/` for CCLE and CGP T0 experiments.
- Added packaged standardized GDSC bundles under `MCLRP_MFMR/data/standardized/GDSC/`.
- Added GDSC dataset choices: `GDSC_ERK_AUC`, `GDSC_ERK_IC50`, `GDSC_PI3K_AUC`, and `GDSC_PI3K_IC50`.
- Added local path resolution in `MCLRP_MFMR/paths.py`; package scripts now default to project-level `results/`.
- Added packaged legacy MCLRP Python baseline support under `MCLRP_MFMR/legacy_mclrp/`.
- Added visualization and one-command pipeline CLIs:
  - `visualize_t0_results.py`
  - `run_t0_pipeline.py`

## Validation and Benchmark-Hardening Stage

- Integrated `original_mclrp` into the strict T0 benchmark method list and verified it runs under the same generated fold masks as MFMR methods.
- Hardened the original MCLRP path against leakage:
  - held-out entries are masked before solver fitting;
  - legacy expression PCs are fitted per outer fold on rows with training entries only;
  - solver epsilon is computed from fold training entries only, not the full response matrix.
- Added `results/t0_mfmr/protocol_audit.csv` with per dataset/seed/fold/method/drug train counts, held-out counts, expression-PC mode, train-row counts, masking status, ridge/mutation mode, skip reason, and fold hash.
- Added `MCLRP_MFMR/validate_t0_results.py` for result completeness, fold identity, held-out count consistency, PCC range, finite-value, prediction-shape, held-out-prediction, and MFMR base component-array checks.
- Added paper table generation in `MCLRP_MFMR/make_paper_tables.py`, writing:
  - `table_main_pcc.csv`
  - `table_main_rmse.csv`
  - `table_ablation.csv`
  - `table_mutation_head.csv`
  - `table_statistical_tests.csv`
- Added paper figure generation in `MCLRP_MFMR/make_paper_figures.py`, including method comparison, ablation, per-drug gain, predicted-vs-observed, and mutation residual improvement figures.
- Hardened PCC calculation for constant predictors by coercing non-finite correlations to `0.0`.
- Standardized diagnostics output so non-applicable fields are explicit rather than blank/NaN.
- Project-level benchmark outputs now default to `results/t0_mfmr/` via `MCLRP_MFMR/paths.py`.

Notes:
- The default mutation residual mode is quick mode (`residual_inner_cv=0`): imputer train meta-predictions are extracted after masking the target drug column at transform time, and ridge train meta-predictions are in-sample. Set `--residual-inner-cv` above zero to cross-fit base predictions inside each outer training fold.
- Single-seed runs cannot produce paired t-test or Wilcoxon p-values; the validator reports those p-value NaNs as warnings when `n_pairs < 2`.
- Paper figure scripts write placeholder panels for mutation or ablation views when the required methods were not included in the run.
- Existing legacy scripts and result files were left intact.

## Full Base Benchmark and Calibration Diagnostic

- Added `original_mclrp_calibrated` as a separate diagnostic method under the strict T0 protocol.
- The diagnostic fits per-fold/per-drug affine calibration using only observed training entries and applies the fitted transform to held-out original MCLRP predictions.
- Preserved the uncalibrated `original_mclrp` method and added calibration audit fields for fitted slope/intercept, fit counts, and skipped reasons.
- Extended strict protocol validation checks to cover calibration train-mask usage, test-entry exclusion from coefficient fitting, fold preservation, held-out count preservation, and positive-affine PCC behavior.
- Completed the full base benchmark for CCLE, ERKAUC30, ERKIC50, PI3KAUC, and PI3KIC50 across seeds 0-9 with methods:
  - `global_mean`
  - `row_col_mean`
  - `original_mclrp`
  - `original_mclrp_calibrated`
  - `imputer_only`
  - `ridge_only`
  - `mfmr_base`
- Final `validate_t0_results.py --results-dir results\t0_mfmr` status: PASS with 0 errors and 0 warnings.
- Generated paper-ready tables in `results/t0_mfmr/paper_tables/` and figures in `results/t0_mfmr/figures/`.
- Wrote the full report to `results/t0_mfmr/FULL_BASE_BENCHMARK_REPORT.md`.
- Calibration partially supports the original MCLRP scale-mismatch hypothesis: fitted slopes were consistently above 1, and RMSE/MAE improved on every dataset, but PCC did not improve and decreased modestly on CGP datasets.
- `mfmr_base` is recommended for manuscript main base-model results because it remained best across all five datasets by PCC, RMSE, and MAE.

## Ablation, Mutation-Head, and Biological Interpretation Stage

- Archived the validated full base benchmark before next-stage generation under `results/t0_mfmr_archives/base_strict_t0_20260524_152406`.
- Completed the full strict T0 ablation benchmark for CCLE, ERKAUC30, ERKIC50, PI3KAUC, and PI3KIC50 across seeds 0-9 with:
  - `mfmr_base`
  - `imputer_only`
  - `ridge_only`
  - `mfmr_no_expression`
  - `mfmr_no_row_stats`
  - `mfmr_no_imputer`
  - `mfmr_no_ridge`
- Extended ablation summaries to include `imputer_only` and `ridge_only` versus `mfmr_base`, and extended paired seed-level tests to PCC, RMSE, and MAE.
- Added exact-spec prediction caching in the benchmark loop so alias ablations can reuse identical predictions while preserving separate audit, diagnostic, and prediction outputs.
- Completed mutation-head benchmarking for ERKAUC30, ERKIC50, PI3KAUC, and PI3KIC50 in both quick residual mode and inner 5-fold cross-fit mode.
- Added `MCLRP_MFMR/make_next_stage_outputs.py` to merge archived base outputs with validated ablation outputs, build mutation residual mode comparisons, and generate biological interpretation files.
- Generated:
  - `results/t0_mfmr/ablation_summary.csv`
  - `results/t0_mfmr/mutation_residual_mode_comparison.csv`
  - `results/t0_mfmr/biology/per_drug_gain_by_dataset.csv`
  - `results/t0_mfmr/biology/top_improved_drugs.csv`
  - `results/t0_mfmr/biology/worst_degraded_drugs.csv`
  - `results/t0_mfmr/biology/representative_drug_cases.md`
  - `results/t0_mfmr/ABLATION_BENCHMARK_REPORT.md`
  - `results/t0_mfmr/MUTATION_HEAD_BENCHMARK_REPORT.md`
  - `results/t0_mfmr/NEXT_STAGE_SUMMARY.md`
- Regenerated paper tables and figures, including `table_ablation.csv`, `table_mutation_head.csv`, `ablation_comparison.png`, and `mutation_residual_improvement.png`.
- Final merged `validate_t0_results.py --results-dir results\t0_mfmr` status: PASS with 0 errors and 0 warnings.
