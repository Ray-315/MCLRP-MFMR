# Reconstructed MCLRP ablation benchmark

This directory retains analysis and plotting utilities for the completed strict train-only comparison of full MCLRP with two reconstructed ablations (`MCLRP_NoPCA_reconstructed` and `MCLRP_NoTrace_reconstructed`). The frozen run used five tasks, ten seeds, and ten random-entry folds.

The completed derived outputs are in `results/mclrp_ablation_5task_10x10/`.

The public release intentionally omits the runner and locally translated MCLRP solver source because a clear redistribution license was not identified for the upstream software release. See `THIRD_PARTY.md`.

You can still rerun the downstream analysis/figure generation against the archived outputs, for example:

```bash
python scripts/mclrp_ablation/analyze_mclrp_ablation_5task.py \
  --input-dir results/mclrp_ablation_5task_10x10 \
  --bootstrap 20000
```

The `reconstructed` suffix is part of the claim boundary: the original MCLRP authors did not publish executable NoPCA or NoTrace branches in the public source release used for this study.
