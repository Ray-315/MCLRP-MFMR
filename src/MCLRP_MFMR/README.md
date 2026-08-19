# MCLRP-MFMR strict T0 implementation

This package implements the strict T0 random-entry MFMR workflow used in the manuscript. All response-derived preprocessing is fitted after masking the outer-test entries.

Input data are not bundled in the public repository. See `../../data/README.md` from the repository root (or the top-level `data/README.md`) for the expected layout.

Recommended leakage-safety check:

```bash
python src/MCLRP_MFMR/tests/test_t0_strict_protocol.py
```

Example MFMR smoke run after data reconstruction:

```bash
python src/MCLRP_MFMR/run_mfmr_t0.py --dataset CCLE --seed 0 --method mfmr_base --num-folds 2 --max-drugs 5 --max-cell-lines 100 --save-predictions
```

The `original_mclrp` and `original_mclrp_calibrated` method labels are retained for compatibility with the frozen result schema, but the locally translated upstream MCLRP source is not redistributed in the public repository. Requesting those methods will raise an explanatory error. See `THIRD_PARTY.md`.
