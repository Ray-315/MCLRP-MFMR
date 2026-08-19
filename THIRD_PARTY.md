# Third-party and upstream-material notice

## Original MCLRP

MCLRP is prior work and is cited by the manuscript as:

- K. Wang et al., “MCLRP: enhanced prediction of anticancer drug response through low-rank matrix completion and transcriptomic profiling,” BMC Biology (2026), DOI: `10.1186/s12915-025-02457-8`.
- Upstream GitHub: `https://github.com/kunwangouc/MCLRP`
- Upstream Zenodo record: `https://doi.org/10.5281/zenodo.8127448`

The public GitHub repository and Zenodo record checked for this release do not provide an explicit standard software license that clearly authorizes republishing modified/translated source under this repository's future license. Therefore this public package does **not** redistribute:

- the upstream MATLAB MCLRP source;
- local Python translations of that source;
- MCLRP solver copies used only to execute the comparator or reconstructed MCLRP ablations.

The manuscript's frozen comparator-derived result files are retained under `results/` so the reported numerical comparisons, statistical summaries, and audits remain inspectable.

To make the exact comparator implementation independently rerunnable from this repository, the authors should obtain/confirm redistribution permission or a clear software license from the relevant upstream rights holder(s), then add the authorized implementation with attribution and its original license/notice.

## Upstream datasets

CCLE/DepMap, GDSC, CGP/legacy benchmark files, and Cell Model Passports-derived inputs are not redistributed here. Users should obtain them from the original providers/upstream releases and comply with their terms. The repository keeps only manifests, hashes/dimensions where available, fold assignments, and derived results needed to identify the evaluated freeze.
