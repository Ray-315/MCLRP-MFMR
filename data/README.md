# Data access and reconstruction

Raw/standardized upstream pharmacogenomic inputs are intentionally not included in this public repository.

## Expected layout

The runtime expects the following project-level locations:

```text
data/
├── raw/
│   ├── CCLE/
│   │   ├── CCLE_X.npz
│   │   ├── MMnormal.npz
│   │   └── CCLE_Mutation_19Q1_aligned_to_CCLE_X.csv   # optional mutation analysis
│   └── CGP/
│       ├── CGP_X.npz
│       ├── ERKAUC30.npz
│       ├── ERKIC50.npz
│       ├── PI3KAUC.npz
│       ├── PI3KIC50.npz
│       └── Mutation.xlsx                               # optional mutation analysis
└── standardized/
    └── GDSC/
        ├── ERK_AUC_bundle.npz
        ├── ERK_IC50_bundle.npz
        ├── PI3K_AUC_bundle.npz
        ├── PI3K_IC50_bundle.npz
        └── mutation_features.csv                       # optional mutation analysis
```

## Source notes

- **Legacy MCLRP/CCLE/CGP benchmark bundle:** upstream MCLRP GitHub `https://github.com/kunwangouc/MCLRP` and Zenodo `https://doi.org/10.5281/zenodo.8127448` contain the prior-work release used as the historical benchmark source.
- **GDSC:** the active manifest records the GDSC2 fitted-dose-response workbook dated 27 October 2023, an RNA-seq file dated 22 September 2025, a gene-map file dated 12 December 2024, and a Cell Model Passports-derived mutation/CNV table.
- **CCLE:** the primary final-lock benchmark uses the legacy CCLE expression/response bundle and the aligned 19Q1 mutation table described in `data/manifests/CCLE-manifest.json`.
- **CGP:** the manuscript deliberately reports the available legacy benchmark bundle without asserting a more specific upstream release identifier that is absent from the retained project records.

## Verification

After reconstructing the inputs, compare matrix dimensions and available hashes with:

- `data/manifests/`
- `configs/primary_final_lock.json`
- `configs/mclrp_ablation_5task_10x10.json`
- `results/primary_final_lock/FINAL_RESULT_MANIFEST.json`

The explicit outer-fold assignments are already included in `splits/primary_10x10/`; no upstream response values are embedded in those split files.
