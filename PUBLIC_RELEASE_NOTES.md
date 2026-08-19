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

A software licence for the authors' MFMR implementation still requires final
author approval; see `LICENSE_PENDING.md`.
