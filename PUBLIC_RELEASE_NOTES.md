# Public release sanitization

This directory was generated from the manuscript reproducibility package for public GitHub deposition. The scientific result tables, prediction archives, explicit fold assignments, statistical outputs, and MFMR implementation were retained.

Changes made for the public release:

- removed embedded upstream/legacy raw input files;
- removed upstream MCLRP source and local translated copies pending a clear redistribution license;
- removed the MCLRP ablation runner that depended on those omitted source files;
- removed machine-specific console logs;
- replaced machine-specific absolute paths in manifests with stable repository-relative paths;
- added `.gitignore`, `CITATION.cff`, `THIRD_PARTY.md`, and public-facing data/reproduction instructions;
- retained archived comparator-derived results so the manuscript's numerical comparisons remain inspectable.

A code license for the authors' MFMR implementation still requires author approval; see `LICENSE_PENDING.md`.
