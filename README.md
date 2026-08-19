# MCLRP-MFMR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22008296.svg)](https://doi.org/10.5281/zenodo.22008296)

Official reproducibility repository for:

**MCLRP–MFMR: Fold-Specific Low-Rank Completion and Transcriptomic Fusion for Anticancer Drug-Response Interpolation**

MFMR is a fold-specific extension of MCLRP for anticancer drug-response interpolation. It combines transcriptome-informed iterative imputation with target-wise ridge prediction while enforcing a strict training/test information boundary during response-matrix reconstruction.

## Overview

The main objective of this repository is to provide the code, frozen configurations, cross-validation assignments, and analysis procedures required to reproduce the MFMR experiments described in the manuscript.

The primary evaluation uses strict T0 random-entry interpolation. For every outer fold, held-out drug-response entries are masked **before** any response-derived preprocessing, feature construction, imputation, dimensionality reduction, or regression.

The primary benchmark contains:

- CCLE
- CGP ERK-AUC
- CGP ERK-IC50
- CGP PI3K-AUC
- CGP PI3K-IC50

Each task is evaluated using:

- 10 random seeds
- 10 folds per seed
- identical held-out masks across compared methods

## Repository structure

```text
MCLRP-MFMR/
├── CITATION.cff
├── LICENSE
├── README.md
├── THIRD_PARTY.md
├── PUBLIC_RELEASE_NOTES.md
├── pyproject.toml
├── project_paths.py
│
├── configs/
│   ├── primary_final_lock.json
│   └── mclrp_ablation_5task_10x10.json
│
├── data/
│   ├── README.md
│   └── manifests/
│
├── docs/
│   ├── DATA_AVAILABILITY.md
│   ├── REPRODUCTION.md
│   └── CLAIM_TO_FILE_INDEX.md
│
├── environment/
│   └── requirements.txt
│
├── scripts/
│   ├── plotting/
│   └── mclrp_ablation/
│
├── splits/
│   └── primary_10x10/
│
└── src/
    └── MCLRP_MFMR/
```

## What is included

This repository contains:

- MFMR source code
- frozen model configurations
- preprocessing and evaluation code
- exact 10-seed × 10-fold primary evaluation assignments
- statistical-analysis scripts
- plotting scripts
- dataset manifests
- data-reconstruction documentation
- integrity and reproducibility documentation

The frozen split files under `splits/primary_10x10/` are retained in the repository so that the exact evaluation partitions used in the manuscript can be reconstructed.

## Results

The complete experimental result archive is distributed separately rather than tracked directly in the Git repository.

The result archive contains materials such as:

- primary final-lock predictions
- seed-level and fold-level metrics
- protocol-audit files
- paired statistical tests
- reconstructed MCLRP ablation outputs
- out-of-fold prediction matrices
- bootstrap results
- derived tables used in the manuscript and supplementary material

The archived results are provided through the corresponding GitHub Release and/or Zenodo record.

**Result archive:** https://github.com/Ray-315/MCLRP-MFMR/releases/tag/v1.0

**Zenodo DOI:** to be added after archival.

This separation keeps the Git repository focused on source code and reproducibility metadata while allowing the complete frozen numerical output to remain publicly accessible.

## Data availability

The upstream pharmacogenomic datasets are not redistributed in this repository.

The study uses data derived from resources including:

- Cancer Cell Line Encyclopedia (CCLE)
- Genomics of Drug Sensitivity in Cancer (GDSC)
- CGP / legacy drug-response benchmark resources
- Cell Model Passports-derived molecular features where applicable

Users should obtain the upstream datasets from their original providers under the applicable access and licensing terms.

Dataset manifests and reconstruction instructions are provided in:

```text
data/README.md
data/manifests/
```

After reconstruction, local data should be placed according to the directory structure documented in `data/README.md`.

## Installation

Python dependencies are listed in:

```text
environment/requirements.txt
```

A typical installation is:

```bash
git clone https://github.com/Ray-315/MCLRP-MFMR.git
cd MCLRP-MFMR

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r environment/requirements.txt
```

## Reproducing the primary MFMR experiment

After reconstructing the required upstream inputs, run the primary MFMR evaluation from the repository root.

```bash
python src/MCLRP_MFMR/run_t0_pipeline.py \
  --datasets CCLE ERKAUC30 ERKIC50 PI3KAUC PI3KIC50 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --methods global_mean row_col_mean imputer_only ridge_only mfmr_base \
            mfmr_no_expression mfmr_no_row_stats mfmr_no_imputer mfmr_no_ridge \
  --num-folds 10 \
  --save-predictions \
  --output-dir results/reproduced_primary
```

The generated outputs can then be checked with:

```bash
python src/MCLRP_MFMR/validate_t0_results.py \
  --results-dir results/reproduced_primary
```

More detailed instructions are available in:

```text
docs/REPRODUCTION.md
```

## Evaluation protocol

For each dataset and random seed, observed response entries are partitioned into ten folds.

For an outer fold:

1. held-out response entries are masked;
2. all response-dependent quantities are recomputed using training-visible responses only;
3. expression preprocessing is fitted within the permitted training boundary;
4. model fitting is performed;
5. predictions are generated only for held-out entries;
6. the ten folds are reassembled into a complete out-of-fold prediction surface for that seed.

Fold-level scores are computational partitions and are not treated as independent biological replicates.

The exact assignments used by the primary benchmark are available under:

```text
splits/primary_10x10/
```

## Original MCLRP implementation

The manuscript compares MFMR with the packaged implementation of the previously published MCLRP method.

The upstream MCLRP source code and locally translated copies are **not redistributed in this repository**, because a sufficiently clear software redistribution licence was not identified for the upstream source used during preparation of this release.

The present repository therefore contains MFMR code and reproducibility materials developed for this study while keeping the upstream MCLRP software outside the public package.

Further details are provided in:

```text
THIRD_PARTY.md
```

## Reconstructed MCLRP ablations

The manuscript also reports reconstructed MCLRP component analyses.

The corresponding frozen configuration is stored in:

```text
configs/mclrp_ablation_5task_10x10.json
```

Analysis and plotting utilities that operate on the archived derived outputs are retained in this repository.

The complete numerical outputs for these analyses are distributed with the separate result archive.

## Reproducibility

The project was designed around an explicit information boundary.

In particular, held-out responses are excluded before fold-dependent operations including:

- response summaries
- target-conditioned gene selection
- scaling
- dimensionality reduction
- iterative imputation
- regression fitting

The same outer masks are used when methods are compared within a dataset and seed.

See `docs/REPRODUCTION.md` for the full protocol description.

## Citation

If you use this repository, please cite the accompanying manuscript.

Citation metadata for the authors and software is provided in:

```text
CITATION.cff
```

Authors:

- Shengrui Han
- Zimo Li
- Zhaorui Cui
- Chunxin Yuan
- Jialiang Yang

## Contact

For questions regarding the repository or reproduction of the experiments, please open a GitHub issue.

Correspondence regarding the manuscript may also be directed to the corresponding authors listed in the article.

## License

Original MCLRP-MFMR code and documentation in this repository are released under the **MIT License**, unless otherwise noted. See `LICENSE`.

Upstream datasets, third-party software, and excluded upstream MCLRP materials remain subject to their respective licences and terms of use and are not covered by this repository's MIT License.
