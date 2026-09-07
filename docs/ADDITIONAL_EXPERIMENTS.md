# Additional fusion and masking experiments

This extension adds nested fusion-weight selection, response-masking sensitivity,
and held-out branch complementarity. The primary model and preprocessing remain
unchanged. The source scripts support this repository's `src/` layout.

## Frozen design and numerical outputs

The five primary tasks are CCLE, ERKAUC30, ERKIC50, PI3KAUC, and PI3KIC50.
The latter four identifiers refer to legacy CGP response matrices. PRISM 19Q4
is additionally included in adaptive fusion and branch complementarity.

- Seeds: 0 through 9.
- Adaptive fusion: ten outer folds for primary tasks; one locked PRISM test;
  five inner folds; imputer weight 0.0 through 1.0 in increments of 0.1.
  Select by mean inner-fold PCC, breaking ties towards 0.5, then lower weight.
- Mask sensitivity: 10%, 15%, 20%, 25%, and 30% of observed entries; one
  shared holdout per task/ratio/seed; at least one training observation per row
  and five per drug. Models are Original MCLRP and fixed-fusion MFMR.
- Complementarity: aligned held-out branch predictions, signed residual
  correlations, absolute-error wins (tie tolerance 1e-6), and disagreement quartiles.
- Additional-experiment summaries use sample SD and descriptive Student-t
  intervals across ten seeds. Previously released primary tables retain their
  original population-SD convention. PRISM uses one fixed test across seeds.

`results/additional_experiments/` contains frozen raw metrics, candidate scores,
summary tables, plot-data CSV files, and configurations. Counts are 240 adaptive
seed/method rows, 2,040 outer-fold/method rows, 5,610 inner candidate-score rows,
500 masking evaluations, 60 branch summaries, and 720 quartile/method rows.
These files contain derived metrics, not upstream response or expression values.

`splits/additional_experiments/mask_sensitivity/` contains the 250 exact boolean
training/test masks. Primary adaptive outer assignments reuse
`splits/primary_10x10/`; the PRISM mask is
`splits/additional_experiments/PRISM19Q4_locked.npz` (2,265 test entries).
PRISM label arrays are stored as strings, with no pickle required.

Full prediction caches are not tracked by this update. The archived CSV files
can be inspected without upstream data; recomputation requires local datasets.
Software version and DOI metadata continue to refer to the previous published
release until a new release is created.

## Run from a source checkout

Activate the documented environment (for example, `conda activate base`) and
install dependencies from `environment/requirements.txt`. Obtain and arrange
upstream datasets using `docs/REPRODUCTION.md` and
`docs/PRISM19Q4_INDEPENDENT.md`. The PRISM bundle builder supplies the aligned
local expression/response bundle. Run commands at the repository root.

Use a separate directory for reruns to preserve the checked-in metric snapshot:

```powershell
python scripts/additional_experiments/run_adaptive_fusion.py --output-dir runs/additional/adaptive_fusion --resume

python scripts/additional_experiments/run_adaptive_fusion.py --datasets PRISM19Q4_independent --outer-folds 1 --prism-bundle-dir data/standardized/PRISM19Q4_independent --prism-split splits/additional_experiments/PRISM19Q4_locked.npz --output-dir runs/additional/prism_adaptive --resume

python scripts/additional_experiments/run_mask_sensitivity.py --methods mfmr_base --output-dir runs/additional/mask_sensitivity --resume

python scripts/additional_experiments/analyze_branch_complementarity.py --prediction-dir runs/additional/adaptive_fusion/predictions --output-dir runs/additional/branch_complementarity

python scripts/additional_experiments/summarize_additional_experiments.py --results-dir results/additional_experiments
```

The first adaptive command processes five primary datasets. To combine primary
and PRISM reruns, use separate dataset subdirectories and `aggregate_shards.py`
before complementarity analysis. Its `--help` documents the shard interface.
Do not merge a new rerun into the frozen snapshot.

The all-dataset PowerShell orchestrator is also provided. Its Python executable
can be selected using `-Python`; it uses the activated `python` by default.
It writes to `results/additional_experiments`, so run it in a separate checkout
if retaining the tracked snapshot. It includes Original MCLRP and consequently
requires separately authorized comparator source as described in `THIRD_PARTY.md`.
That source is not redistributed here. The `--methods mfmr_base` command above
runs the public MFMR implementation without the comparator. Frozen MCLRP metrics
and exact comparison masks are included for inspection.

## Validation

To regenerate Supplementary Tables S13--S18 and the two branch figure panels
directly from the tracked CSV files (no upstream datasets required):

```powershell
python scripts/plotting/make_additional_supplementary.py --output-dir outputs/additional_supplementary
```

Other experiment figure panels are produced by their respective run scripts;
the table script also copies those panels when they exist in its results input.
The disagreement figure shows PCC separately for each dataset rather than
pooling absolute errors across response scales.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest src/MCLRP_MFMR/tests/test_additional_experiments.py src/MCLRP_MFMR/tests/test_additional_cli.py
```

Tests cover inner-training masking, alpha selection and tie rules, constrained
sampling, branch statistics, resume metadata, and all five CLI entry points.
The public export does not include raw data, local prediction caches, temporary
shards, upstream MCLRP code, credentials, or manuscript source files.
