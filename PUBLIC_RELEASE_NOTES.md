# Public release notes

This repository contains the public reproducibility code for the MCLRP-MFMR study.

The main Git repository contains:

- MFMR source code;
- frozen configurations;
- exact primary cross-validation split assignments;
- data manifests and reconstruction instructions;
- statistical and plotting scripts;
- reproducibility documentation.

Large frozen experimental outputs are distributed separately through the
corresponding GitHub Release / Zenodo archive and are intentionally not tracked
in the main Git history.

For the public release, embedded upstream pharmacogenomic data, upstream MCLRP
source code, local translated MCLRP copies, machine-specific logs, absolute
local paths, and Python cache files were removed.

The upstream MCLRP source is not redistributed because a sufficiently clear
software redistribution licence was not identified for the source used during
preparation of this release. See `THIRD_PARTY.md`.

Original MCLRP-MFMR code and documentation in this repository are released
under the MIT License unless otherwise noted. Upstream datasets, third-party
software, and excluded upstream MCLRP materials are not covered by this licence.
See `LICENSE` and `THIRD_PARTY.md`.
