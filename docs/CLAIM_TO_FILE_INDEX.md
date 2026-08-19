# Manuscript result index

The paths below refer to files inside the separate
`MCLRP-MFMR-results-v1.0.zip` GitHub Release / Zenodo archive unless stated
otherwise.

- Primary mean plus/minus standard deviations:
  `results/primary_final_lock/method_summary.csv` and related `paper_tables/`.
- Fold-level metrics:
  `results/primary_final_lock/fold_summary.csv`.
- Leakage and fold-mask audit:
  `results/primary_final_lock/protocol_audit.csv`.
- Primary paired tests:
  `results/primary_final_lock/statistical_tests.csv`.
- MFMR ablations:
  `results/primary_final_lock/ablation_summary.csv`.
- Mutation residual-mode comparison:
  `results/primary_final_lock/mutation_residual_mode_comparison.csv`.
- Full reconstructed MCLRP-ablation metrics:
  `results/mclrp_ablation_5task_10x10/fold_metrics.csv`,
  `seed_metrics.csv`, and `seed_per_drug_metrics.csv`.
- Reconstructed-ablation bootstrap and paired tests:
  `results/mclrp_ablation_5task_10x10/analysis/`.
- Exact primary fold assignments are tracked directly in the Git repository:
  `splits/primary_10x10/`.
- Frozen run-integrity files:
  `results/primary_final_lock/FINAL_RESULT_MANIFEST.json` and
  `results/mclrp_ablation_5task_10x10/run_manifest.json`.
