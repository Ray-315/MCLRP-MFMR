from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STANDARDIZED_DATA_DIR = DATA_DIR / "standardized"
PROJECT_DATA_DIR = DATA_DIR


def _prefer_existing(primary: Path, shared: Path) -> Path:
    """Use packaged MFMR data when present, otherwise reuse project-level data."""
    return primary if primary.exists() else shared


CCLE_RAW_DATA_DIR = _prefer_existing(
    RAW_DATA_DIR / "CCLE",
    PROJECT_DATA_DIR / "raw" / "CCLE",
)
CGP_RAW_DATA_DIR = _prefer_existing(
    RAW_DATA_DIR / "CGP",
    PROJECT_DATA_DIR / "raw" / "CGP",
)
GDSC_STANDARDIZED_DIR = _prefer_existing(
    STANDARDIZED_DATA_DIR / "GDSC",
    PROJECT_DATA_DIR / "standardized" / "GDSC",
)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


REQUIRED_DATA_FILES = (
    CCLE_RAW_DATA_DIR / "MMnormal.npz",
    CCLE_RAW_DATA_DIR / "CCLE_X.npz",
    CGP_RAW_DATA_DIR / "CGP_X.npz",
    CGP_RAW_DATA_DIR / "ERKAUC30.npz",
    CGP_RAW_DATA_DIR / "ERKIC50.npz",
    CGP_RAW_DATA_DIR / "PI3KAUC.npz",
    CGP_RAW_DATA_DIR / "PI3KIC50.npz",
    CGP_RAW_DATA_DIR / "Mutation.xlsx",
    GDSC_STANDARDIZED_DIR / "ERK_AUC_bundle.npz",
    GDSC_STANDARDIZED_DIR / "ERK_IC50_bundle.npz",
    GDSC_STANDARDIZED_DIR / "PI3K_AUC_bundle.npz",
    GDSC_STANDARDIZED_DIR / "PI3K_IC50_bundle.npz",
    GDSC_STANDARDIZED_DIR / "mutation_features.csv",
)


def assert_required_data() -> None:
    missing = [str(path) for path in REQUIRED_DATA_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing packaged MCLRP-MFMR data files:\n" + "\n".join(missing))


def data_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not DATA_DIR.exists():
        return rows
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(PACKAGE_ROOT)),
                    "bytes": int(path.stat().st_size),
                }
            )
    return rows
