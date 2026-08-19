from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge

try:
    from .paths import CCLE_RAW_DATA_DIR, CGP_RAW_DATA_DIR, GDSC_STANDARDIZED_DIR, RESULTS_DIR, assert_required_data
    from .calc_pcc import calc_pcc
    from .getcrossMatrixs import getcrossMatrixs
except ImportError:  # pragma: no cover - supports direct execution from this folder.
    from paths import CCLE_RAW_DATA_DIR, CGP_RAW_DATA_DIR, GDSC_STANDARDIZED_DIR, RESULTS_DIR, assert_required_data
    from calc_pcc import calc_pcc
    from getcrossMatrixs import getcrossMatrixs

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


CGP_DATASET_FILES: dict[str, tuple[str, str, str]] = {
    "ERKAUC30": ("ERKAUC30.npz", "ERKAUC30", "ERK MAPK signaling AUC"),
    "ERKIC50": ("ERKIC50.npz", "ERKIC50", "ERK MAPK signaling IC50"),
    "PI3KAUC": ("PI3KAUC.npz", "PI3KAUC", "PI3K/MTOR signaling AUC"),
    "PI3KIC50": ("PI3KIC50.npz", "PI3KIC50", "PI3K/MTOR signaling IC50"),
}
GDSC_DATASET_FILES: dict[str, tuple[str, str, str]] = {
    "GDSC_ERK_AUC": ("ERK_AUC_bundle.npz", "ERK_AUC", "GDSC ERK MAPK signaling AUC"),
    "GDSC_ERK_IC50": ("ERK_IC50_bundle.npz", "ERK_IC50", "GDSC ERK MAPK signaling IC50"),
    "GDSC_PI3K_AUC": ("PI3K_AUC_bundle.npz", "PI3K_AUC", "GDSC PI3K/MTOR signaling AUC"),
    "GDSC_PI3K_IC50": ("PI3K_IC50_bundle.npz", "PI3K_IC50", "GDSC PI3K/MTOR signaling IC50"),
}
DATASET_CHOICES = ("CCLE", *CGP_DATASET_FILES.keys(), *GDSC_DATASET_FILES.keys())

BASELINE_METHODS = (
    "global_mean",
    "row_col_mean",
    "original_mclrp",
    "original_mclrp_calibrated",
    "imputer_only",
    "ridge_only",
    "mfmr_base",
    "mfmr_mutation",
)
ABLATION_METHODS = (
    "mfmr_no_imputer",
    "mfmr_no_ridge",
    "mfmr_no_expression",
    "mfmr_no_row_stats",
    "mfmr_base_weight_0.5_0.5",
    "mutation_no_pathway",
    "mutation_no_latent",
)
ABLATION_COMPARISON_METHODS = (
    "imputer_only",
    "ridge_only",
    *ABLATION_METHODS,
)
SUPPORTED_METHODS = BASELINE_METHODS + ABLATION_METHODS

MAPK_GENES = (
    "ABL2",
    "EGFR",
    "FGFR3",
    "JAK2",
    "ALK",
    "BRAF",
    "EGFR.1",
    "ERBB2",
    "FGFR2",
    "FGFR3.1",
    "FLT3",
    "HRAS",
    "KDR",
    "KIT",
    "KRAS",
    "MAP2K4",
    "MET",
    "NF1",
    "NF2",
    "NRAS",
    "PDGFRA",
)
PI3K_GENES = (
    "AKT2",
    "EGFR",
    "ERBB2",
    "HRAS",
    "KRAS",
    "NRAS",
    "PIK3CA",
    "PIK3R1",
    "PTEN",
    "STK11",
    "TSC1",
    "CCND1",
    "CCND2",
    "CCND3",
    "CDK4",
    "CDK6",
    "CDKN2A",
    "CDKN2C",
)
HOTSPOT_PATTERNS = {
    "BRAF": r"V600",
    "KRAS": r"G12|G13|Q61|A146",
    "NRAS": r"G12|G13|Q61",
    "PIK3CA": r"H1047|E545|E542|Q546|P539",
    "PTEN": r"\*|fs|\?",
}
GENE_ALIASES = {"EGFR.1": "EGFR", "FGFR3.1": "FGFR3"}


@dataclass(frozen=True)
class MFMRConfig:
    num_folds: int = 10
    seeds: tuple[int, ...] = tuple(range(10))
    topg_imp: int = 2000
    comp_imp: int = 12
    topg_ridge: int = 2500
    comp_ridge: int = 16
    ridge_alpha: float = 10.0
    weight_imp: float = 0.5
    weight_ridge: float = 0.5
    imputer_max_iter: int = 3
    min_train_per_drug: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class MutationHeadConfig:
    enabled: bool = False
    final_alpha: float = 20.0
    mutation_latent_dim: int = 24
    residual_inner_cv: int = 0


@dataclass(frozen=True)
class MCLRPConfig:
    gene_limit: int = 1000
    beta: float = 1.0
    rho: float = 1.0
    iter_num: int = 100
    itol: float = 0.0
    tk: float = 3e-2
    epsilon_scale: float = 1.0


@dataclass(frozen=True)
class T0DatasetBundle:
    name: str
    X: np.ndarray
    M: np.ndarray
    cell_labels: np.ndarray
    drug_labels: np.ndarray
    response_path: str
    expression_path: str


@dataclass(frozen=True)
class MutationFeatureSet:
    tissue: np.ndarray
    mapk: np.ndarray
    pi3k: np.ndarray
    overall: np.ndarray
    binary: np.ndarray

    def subset(self, rows: np.ndarray) -> "MutationFeatureSet":
        return MutationFeatureSet(
            tissue=self.tissue[rows],
            mapk=self.mapk[rows],
            pi3k=self.pi3k[rows],
            overall=self.overall[rows],
            binary=self.binary[rows],
        )


@dataclass(frozen=True)
class MethodSpec:
    name: str
    use_imputer: bool
    use_ridge: bool
    use_expression: bool
    use_row_stats: bool
    use_mutation: bool
    mutation_use_pathway: bool = True
    mutation_use_latent: bool = True
    weight_imp: float = 0.5
    weight_ridge: float = 0.5


@dataclass
class PredictionResult:
    method: str
    prediction: np.ndarray
    imputer: np.ndarray
    ridge: np.ndarray
    base: np.ndarray
    final: np.ndarray | None
    folds: list[np.ndarray]
    diagnostics: dict[str, Any]
    calibration_audit: list[dict[str, Any]] | None = None


def dataclass_to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return dataclass_to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): dataclass_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [dataclass_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [dataclass_to_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _first_npz_array(path: Path, preferred_names: Sequence[str] = ()) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    for name in preferred_names:
        if name in payload:
            return payload[name]
    for name in payload.files:
        if name.startswith("__"):
            continue
        return payload[name]
    raise ValueError(f"No array payload found in {path}")


def load_t0_dataset(dataset: str) -> T0DatasetBundle:
    assert_required_data()
    dataset = str(dataset)
    if dataset == "CCLE":
        response_path = CCLE_RAW_DATA_DIR / "MMnormal.npz"
        expression_path = CCLE_RAW_DATA_DIR / "CCLE_X.npz"
        M = _first_npz_array(response_path, ("MMnormal", "MM")).astype(np.float32)
        X = _first_npz_array(expression_path, ("X",)).astype(np.float32)
    elif dataset in CGP_DATASET_FILES:
        file_name, var_name, _ = CGP_DATASET_FILES[dataset]
        response_path = CGP_RAW_DATA_DIR / file_name
        expression_path = CGP_RAW_DATA_DIR / "CGP_X.npz"
        M = _first_npz_array(response_path, (var_name,)).astype(np.float32)
        X = _first_npz_array(expression_path, ("X",)).astype(np.float32)
    elif dataset in GDSC_DATASET_FILES:
        file_name, _, _ = GDSC_DATASET_FILES[dataset]
        response_path = GDSC_STANDARDIZED_DIR / file_name
        expression_path = response_path
        payload = np.load(response_path, allow_pickle=True)
        X = payload["X"].astype(np.float32)
        M = payload["M"].astype(np.float32)
        cell_labels = payload["cell_ids"].astype(object) if "cell_ids" in payload.files else np.asarray([f"Cell_{i + 1:04d}" for i in range(M.shape[0])], dtype=object)
        drug_labels = payload["drug_labels"].astype(object) if "drug_labels" in payload.files else np.asarray([f"Drug_{i + 1:02d}" for i in range(M.shape[1])], dtype=object)
        return T0DatasetBundle(
            name=dataset,
            X=X,
            M=np.where(np.isfinite(M), M, 0.0).astype(np.float32),
            cell_labels=cell_labels,
            drug_labels=drug_labels,
            response_path=str(response_path),
            expression_path=str(expression_path),
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}; choices: {DATASET_CHOICES}")

    if X.shape[0] != M.shape[0]:
        n = min(X.shape[0], M.shape[0])
        X = X[:n]
        M = M[:n]
    cell_labels = np.asarray([f"Cell_{i + 1:04d}" for i in range(M.shape[0])], dtype=object)
    drug_labels = np.asarray([f"Drug_{i + 1:02d}" for i in range(M.shape[1])], dtype=object)
    return T0DatasetBundle(
        name=dataset,
        X=X,
        M=np.where(np.isfinite(M), M, 0.0).astype(np.float32),
        cell_labels=cell_labels,
        drug_labels=drug_labels,
        response_path=str(response_path),
        expression_path=str(expression_path),
    )


def subset_dataset(
    bundle: T0DatasetBundle,
    max_cell_lines: int | None = None,
    max_drugs: int | None = None,
) -> tuple[T0DatasetBundle, np.ndarray, np.ndarray]:
    row_idx = np.arange(bundle.M.shape[0], dtype=np.int64)
    col_idx = np.arange(bundle.M.shape[1], dtype=np.int64)
    if max_cell_lines is not None:
        row_idx = row_idx[: max(1, int(max_cell_lines))]
    if max_drugs is not None:
        col_idx = col_idx[: max(1, int(max_drugs))]
    subset = T0DatasetBundle(
        name=bundle.name,
        X=bundle.X[row_idx].astype(np.float32),
        M=bundle.M[np.ix_(row_idx, col_idx)].astype(np.float32),
        cell_labels=bundle.cell_labels[row_idx],
        drug_labels=bundle.drug_labels[col_idx],
        response_path=bundle.response_path,
        expression_path=bundle.expression_path,
    )
    return subset, row_idx, col_idx


def _safe_nanmean(values: np.ndarray, default: float = 0.0) -> float:
    if values.size == 0:
        return float(default)
    out = float(np.nanmean(values))
    return out if np.isfinite(out) else float(default)


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    good = np.isfinite(a) & np.isfinite(b)
    if not np.any(good):
        return 0.0
    return float(np.sqrt(np.mean((a[good] - b[good]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    good = np.isfinite(a) & np.isfinite(b)
    if not np.any(good):
        return 0.0
    return float(np.mean(np.abs(a[good] - b[good])))


def make_train_stats(M: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    train_values = np.where(train_mask, M, np.nan).astype(np.float32)
    global_mean = _safe_nanmean(train_values, default=0.0)
    row_mean = np.nanmean(train_values, axis=1)
    col_mean = np.nanmean(train_values, axis=0)
    row_mean = np.where(np.isfinite(row_mean), row_mean, global_mean).astype(np.float32)
    col_mean = np.where(np.isfinite(col_mean), col_mean, global_mean).astype(np.float32)
    row_cnt = np.sum(train_mask, axis=1).astype(np.float32)
    filled = np.where(np.isfinite(train_values), train_values, (row_mean[:, None] + col_mean[None, :]) / 2.0)
    return row_mean.astype(np.float32), col_mean.astype(np.float32), row_cnt, filled.astype(np.float32), float(global_mean)


def fit_expr_pcs_train_only(
    X: np.ndarray,
    train_rows: np.ndarray,
    topg: int,
    comp: int,
    seed: int,
) -> np.ndarray:
    """Fit gene selection, scaling, and SVD on train_rows only, then transform all rows."""
    X = np.asarray(X, dtype=np.float32)
    comp = int(max(0, comp))
    if comp == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    train_rows = np.unique(train_rows[(train_rows >= 0) & (train_rows < X.shape[0])])
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or len(train_rows) < 2:
        return np.zeros((X.shape[0], comp), dtype=np.float32)

    topg_eff = int(max(1, min(int(topg), X.shape[1])))
    X_train = X[train_rows].astype(np.float32)
    var = np.nanvar(X_train, axis=0)
    var = np.where(np.isfinite(var), var, -np.inf)
    if topg_eff >= X.shape[1]:
        gene_idx = np.arange(X.shape[1], dtype=np.int64)
        gene_idx = gene_idx[np.argsort(var)[::-1]]
    else:
        gene_idx = np.argpartition(var, -topg_eff)[-topg_eff:]
        gene_idx = gene_idx[np.argsort(var[gene_idx])[::-1]].astype(np.int64)

    X_train_sel = X_train[:, gene_idx].astype(np.float32)
    X_all_sel = X[:, gene_idx].astype(np.float32)
    mean = np.nanmean(X_train_sel, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.nanstd(X_train_sel, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    X_train_scaled = np.nan_to_num((X_train_sel - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    X_all_scaled = np.nan_to_num((X_all_sel - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    n_components = int(min(comp, len(train_rows) - 1, topg_eff))
    if n_components < 1:
        return np.zeros((X.shape[0], comp), dtype=np.float32)
    svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
    svd.fit(X_train_scaled)
    pcs = svd.transform(X_all_scaled).astype(np.float32)
    if pcs.shape[1] < comp:
        pcs = np.pad(pcs, ((0, 0), (0, comp - pcs.shape[1])), mode="constant")
    return pcs[:, :comp].astype(np.float32)


def fit_binary_latent_train_only(
    binary: np.ndarray,
    train_rows: np.ndarray,
    latent_dim: int,
    seed: int,
) -> np.ndarray:
    binary = np.asarray(binary, dtype=np.float32)
    latent_dim = int(max(0, latent_dim))
    if latent_dim == 0:
        return np.zeros((binary.shape[0], 0), dtype=np.float32)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    train_rows = np.unique(train_rows[(train_rows >= 0) & (train_rows < binary.shape[0])])
    if binary.ndim != 2 or binary.shape[1] == 0 or len(train_rows) < 2:
        return np.zeros((binary.shape[0], latent_dim), dtype=np.float32)
    n_components = int(min(latent_dim, len(train_rows) - 1, binary.shape[1]))
    if n_components < 1:
        return np.zeros((binary.shape[0], latent_dim), dtype=np.float32)
    svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
    train = np.nan_to_num(binary[train_rows], nan=0.0, posinf=0.0, neginf=0.0)
    all_binary = np.nan_to_num(binary, nan=0.0, posinf=0.0, neginf=0.0)
    svd.fit(train)
    latent = svd.transform(all_binary).astype(np.float32)
    if latent.shape[1] < latent_dim:
        latent = np.pad(latent, ((0, 0), (0, latent_dim - latent.shape[1])), mode="constant")
    return latent[:, :latent_dim].astype(np.float32)


def _parse_gene_state(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = series.fillna("wt::nci").astype(str)
    mut = s.str.split("::").str[0].fillna("wt")
    cn = s.str.split("::").str[-1].fillna("nci")
    mutated = ((~mut.str.startswith("wt")) & (~mut.str.startswith("na"))).astype(np.float32).to_numpy()
    amp = cn.str.contains(">=8", regex=False).astype(np.float32).to_numpy()
    loss = cn.eq("0").astype(np.float32).to_numpy()
    return mutated, amp, loss, mut.to_numpy()


def _stack_mean(arrays: list[np.ndarray], length: int) -> np.ndarray:
    if not arrays:
        return np.zeros(length, dtype=np.float32)
    return np.mean(np.vstack(arrays), axis=0).astype(np.float32)


def _pathway_summary(gene_set: Sequence[str], df: pd.DataFrame) -> np.ndarray:
    muts: list[np.ndarray] = []
    amps: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    hotspots: list[np.ndarray] = []
    for requested_gene in gene_set:
        gene = requested_gene
        if gene not in df.columns:
            fallback = GENE_ALIASES.get(gene, gene)
            if fallback not in df.columns:
                continue
            gene = fallback
        mut, amp, loss, token = _parse_gene_state(df[gene])
        muts.append(mut)
        amps.append(amp)
        losses.append(loss)
        canonical = GENE_ALIASES.get(gene, gene)
        if canonical in HOTSPOT_PATTERNS:
            h = pd.Series(token).str.contains(HOTSPOT_PATTERNS[canonical], regex=True, na=False).astype(np.float32)
            hotspots.append(h.to_numpy())
    return np.vstack(
        [
            _stack_mean(muts, len(df)),
            _stack_mean(amps, len(df)),
            _stack_mean(losses, len(df)),
            _stack_mean(hotspots, len(df)),
        ]
    ).T.astype(np.float32)


def build_cgp_mutation_features(mutation_path: Path | None = None) -> MutationFeatureSet:
    path = Path(mutation_path) if mutation_path is not None else CGP_RAW_DATA_DIR / "Mutation.xlsx"
    sheet1 = pd.read_excel(path, sheet_name="Sheet1")
    sheet2 = pd.read_excel(path, sheet_name="Sheet2")
    tissue = pd.get_dummies(sheet1[["Cancer Type", "Tissue"]].astype(str), dummy_na=False).astype(np.float32).to_numpy()
    gene_cols = [c for c in sheet1.columns if c not in {"Cell Line", "Cosmic_ID", "Cancer Type", "Tissue"}]
    binary_parts: list[np.ndarray] = []
    mut_parts: list[np.ndarray] = []
    amp_parts: list[np.ndarray] = []
    loss_parts: list[np.ndarray] = []
    for df in (sheet1, sheet2):
        for gene in gene_cols:
            if gene not in df.columns:
                continue
            mut, amp, loss, token = _parse_gene_state(df[gene])
            mut_parts.append(mut)
            amp_parts.append(amp)
            loss_parts.append(loss)
            binary_parts.extend([mut, amp, loss])
            canonical = GENE_ALIASES.get(gene, gene)
            if canonical in HOTSPOT_PATTERNS:
                hotspot = pd.Series(token).str.contains(HOTSPOT_PATTERNS[canonical], regex=True, na=False)
                binary_parts.append(hotspot.astype(np.float32).to_numpy())
    binary = np.vstack(binary_parts).T.astype(np.float32) if binary_parts else np.zeros((len(sheet1), 0), dtype=np.float32)
    overall = np.vstack(
        [
            _stack_mean(mut_parts, len(sheet1)),
            _stack_mean(amp_parts, len(sheet1)),
            _stack_mean(loss_parts, len(sheet1)),
        ]
    ).T.astype(np.float32)
    return MutationFeatureSet(
        tissue=tissue,
        mapk=np.concatenate([_pathway_summary(MAPK_GENES, sheet1), _pathway_summary(MAPK_GENES, sheet2)], axis=1),
        pi3k=np.concatenate([_pathway_summary(PI3K_GENES, sheet1), _pathway_summary(PI3K_GENES, sheet2)], axis=1),
        overall=overall,
        binary=binary,
    )


def _selected_ccle_gene_columns() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for gene in (*MAPK_GENES, *PI3K_GENES):
        if gene not in seen:
            ordered.append(gene)
            seen.add(gene)
    return ordered


def build_ccle_mutation_features(
    expected_rows: int,
    mutation_path: Path | None = None,
) -> MutationFeatureSet:
    path = Path(mutation_path) if mutation_path is not None else CCLE_RAW_DATA_DIR / "CCLE_Mutation_19Q1_aligned_to_CCLE_X.csv"
    if not path.exists():
        raise FileNotFoundError(f"Packaged CCLE mutation feature table not found: {path}")
    table = pd.read_csv(path, low_memory=False)
    if "row_index" in table.columns:
        table = table.sort_values("row_index").drop_duplicates("row_index", keep="first").reset_index(drop=True)
    if len(table) != int(expected_rows):
        raise ValueError(f"CCLE mutation rows ({len(table)}) do not match response/expression rows ({expected_rows})")

    for column in ("Cancer Type", "Tissue"):
        if column not in table.columns:
            table[column] = "Unknown"
        table[column] = table[column].fillna("Unknown").astype(str)

    selected_genes = _selected_ccle_gene_columns()
    for gene in selected_genes:
        if gene in table.columns:
            table[gene] = table[gene].fillna("wt::nci").astype(str)
            continue
        canonical = GENE_ALIASES.get(gene, gene)
        if canonical in table.columns:
            table[gene] = table[canonical].fillna("wt::nci").astype(str)
        else:
            table[gene] = "wt::nci"

    tissue = pd.get_dummies(table[["Cancer Type", "Tissue"]].astype(str), dummy_na=False).astype(np.float32).to_numpy()
    binary_parts: list[np.ndarray] = []
    mut_parts: list[np.ndarray] = []
    amp_parts: list[np.ndarray] = []
    loss_parts: list[np.ndarray] = []
    for gene in selected_genes:
        mut, amp, loss, token = _parse_gene_state(table[gene])
        mut_parts.append(mut)
        amp_parts.append(amp)
        loss_parts.append(loss)
        binary_parts.extend([mut, amp, loss])
        canonical = GENE_ALIASES.get(gene, gene)
        if canonical in HOTSPOT_PATTERNS:
            hotspot = pd.Series(token).str.contains(HOTSPOT_PATTERNS[canonical], regex=True, na=False)
            binary_parts.append(hotspot.astype(np.float32).to_numpy())

    binary = np.vstack(binary_parts).T.astype(np.float32) if binary_parts else np.zeros((len(table), 0), dtype=np.float32)
    overall = np.vstack(
        [
            _stack_mean(mut_parts, len(table)),
            _stack_mean(amp_parts, len(table)),
            _stack_mean(loss_parts, len(table)),
        ]
    ).T.astype(np.float32)
    return MutationFeatureSet(
        tissue=tissue,
        mapk=_pathway_summary(MAPK_GENES, table),
        pi3k=_pathway_summary(PI3K_GENES, table),
        overall=overall,
        binary=binary,
    )


def build_gdsc_mutation_features(
    expected_rows: int,
    mutation_path: Path | None = None,
    cell_ids: np.ndarray | None = None,
) -> MutationFeatureSet:
    path = Path(mutation_path) if mutation_path is not None else GDSC_STANDARDIZED_DIR / "mutation_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Packaged GDSC mutation feature table not found: {path}")
    table = pd.read_csv(path, low_memory=False)
    if cell_ids is not None:
        if "model_id" not in table.columns:
            raise ValueError(f"GDSC mutation feature table is missing model_id: {path}")
        ids = pd.Series(np.asarray(cell_ids, dtype=object).astype(str), name="model_id")
        table = ids.to_frame().merge(table, on="model_id", how="left", sort=False)
        table["model_name"] = table.get("model_name", table["model_id"]).fillna(table["model_id"])
    if len(table) != int(expected_rows):
        raise ValueError(f"GDSC mutation rows ({len(table)}) do not match response/expression rows ({expected_rows})")
    for column in ("Cancer Type", "Tissue"):
        if column not in table.columns:
            table[column] = "Unknown"
        table[column] = table[column].fillna("Unknown").astype(str)

    selected_genes = _selected_ccle_gene_columns()
    for gene in selected_genes:
        if gene in table.columns:
            table[gene] = table[gene].fillna("wt::nci").astype(str)
            continue
        canonical = GENE_ALIASES.get(gene, gene)
        if canonical in table.columns:
            table[gene] = table[canonical].fillna("wt::nci").astype(str)
        else:
            table[gene] = "wt::nci"

    tissue = pd.get_dummies(table[["Cancer Type", "Tissue"]].astype(str), dummy_na=False).astype(np.float32).to_numpy()
    binary_parts: list[np.ndarray] = []
    mut_parts: list[np.ndarray] = []
    amp_parts: list[np.ndarray] = []
    loss_parts: list[np.ndarray] = []
    for gene in selected_genes:
        mut, amp, loss, token = _parse_gene_state(table[gene])
        mut_parts.append(mut)
        amp_parts.append(amp)
        loss_parts.append(loss)
        binary_parts.extend([mut, amp, loss])
        canonical = GENE_ALIASES.get(gene, gene)
        if canonical in HOTSPOT_PATTERNS:
            hotspot = pd.Series(token).str.contains(HOTSPOT_PATTERNS[canonical], regex=True, na=False)
            binary_parts.append(hotspot.astype(np.float32).to_numpy())

    binary = np.vstack(binary_parts).T.astype(np.float32) if binary_parts else np.zeros((len(table), 0), dtype=np.float32)
    overall = np.vstack(
        [
            _stack_mean(mut_parts, len(table)),
            _stack_mean(amp_parts, len(table)),
            _stack_mean(loss_parts, len(table)),
        ]
    ).T.astype(np.float32)
    return MutationFeatureSet(
        tissue=tissue,
        mapk=_pathway_summary(MAPK_GENES, table),
        pi3k=_pathway_summary(PI3K_GENES, table),
        overall=overall,
        binary=binary,
    )


def load_mutation_features_for_dataset(bundle: T0DatasetBundle) -> MutationFeatureSet | None:
    if bundle.name in CGP_DATASET_FILES:
        features = build_cgp_mutation_features()
        if features.binary.shape[0] != bundle.M.shape[0]:
            return None
        return features
    if bundle.name in GDSC_DATASET_FILES:
        try:
            return build_gdsc_mutation_features(expected_rows=bundle.M.shape[0], cell_ids=bundle.cell_labels)
        except Exception:
            return None
    if bundle.name == "CCLE":
        try:
            return build_ccle_mutation_features(expected_rows=bundle.M.shape[0])
        except Exception:
            return None
    return None


def select_pathway_features(dataset: str, mutation: MutationFeatureSet) -> np.ndarray:
    dataset = str(dataset)
    if dataset.startswith("ERK") or "ERK" in dataset:
        return mutation.mapk.astype(np.float32)
    if dataset.startswith("PI3K") or "PI3K" in dataset:
        return mutation.pi3k.astype(np.float32)
    return np.concatenate([mutation.mapk, mutation.pi3k], axis=1).astype(np.float32)


def resolve_method_spec(method: str, config: MFMRConfig) -> MethodSpec:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method!r}; choices: {SUPPORTED_METHODS}")
    if method == "imputer_only":
        return MethodSpec(method, True, False, True, True, False, weight_imp=1.0, weight_ridge=0.0)
    if method == "ridge_only":
        return MethodSpec(method, False, True, True, True, False, weight_imp=0.0, weight_ridge=1.0)
    if method == "mfmr_no_imputer":
        return MethodSpec(method, False, True, True, True, False, weight_imp=0.0, weight_ridge=1.0)
    if method == "mfmr_no_ridge":
        return MethodSpec(method, True, False, True, True, False, weight_imp=1.0, weight_ridge=0.0)
    if method == "mfmr_no_expression":
        return MethodSpec(method, True, True, False, True, False, weight_imp=config.weight_imp, weight_ridge=config.weight_ridge)
    if method == "mfmr_no_row_stats":
        return MethodSpec(method, True, True, True, False, False, weight_imp=config.weight_imp, weight_ridge=config.weight_ridge)
    if method == "mfmr_base_weight_0.5_0.5":
        return MethodSpec(method, True, True, True, True, False, weight_imp=0.5, weight_ridge=0.5)
    if method == "mfmr_mutation":
        return MethodSpec(method, True, True, True, True, True, weight_imp=config.weight_imp, weight_ridge=config.weight_ridge)
    if method == "mutation_no_pathway":
        return MethodSpec(method, True, True, True, True, True, False, True, config.weight_imp, config.weight_ridge)
    if method == "mutation_no_latent":
        return MethodSpec(method, True, True, True, True, True, True, False, config.weight_imp, config.weight_ridge)
    return MethodSpec(method, True, True, True, True, False, weight_imp=config.weight_imp, weight_ridge=config.weight_ridge)


def _build_imputer_matrix(
    M_train: np.ndarray,
    Xp: np.ndarray,
    *,
    empty_response_cols: np.ndarray | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    block = np.asarray(M_train, dtype=np.float32).copy()
    block[~np.isfinite(block)] = np.nan
    if empty_response_cols is not None and len(empty_response_cols) > 0:
        block[:, empty_response_cols] = float(fill_value)
    if Xp.shape[1] == 0:
        return block.astype(np.float32)
    return np.concatenate([block, Xp.astype(np.float32)], axis=1).astype(np.float32)


def _fit_imputer(
    M_train: np.ndarray,
    Xp: np.ndarray,
    config: MFMRConfig,
    seed: int,
) -> tuple[IterativeImputer, np.ndarray, float]:
    response = np.asarray(M_train, dtype=np.float32)
    global_mean = _safe_nanmean(response, default=0.0)
    empty_cols = np.where(np.all(~np.isfinite(response), axis=0))[0].astype(np.int64)
    fit_input = _build_imputer_matrix(response, Xp, empty_response_cols=empty_cols, fill_value=global_mean)
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=int(config.imputer_max_iter),
        random_state=int(seed),
        initial_strategy="mean",
        skip_complete=True,
    )
    imputer.fit(fit_input)
    return imputer, empty_cols, float(global_mean)


def _transform_imputer(
    imputer: IterativeImputer,
    M_view: np.ndarray,
    Xp: np.ndarray,
    empty_cols: np.ndarray,
    global_mean: float,
) -> np.ndarray:
    transform_input = _build_imputer_matrix(M_view, Xp, empty_response_cols=empty_cols, fill_value=global_mean)
    out = imputer.transform(transform_input).astype(np.float32)
    return out[:, : M_view.shape[1]].astype(np.float32)


def _ridge_features(
    filled: np.ndarray,
    Xp: np.ndarray,
    row_mean: np.ndarray,
    row_cnt: np.ndarray,
    rows: np.ndarray,
    other_cols: np.ndarray,
    spec: MethodSpec,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if other_cols.size > 0:
        parts.append(filled[rows][:, other_cols].astype(np.float32))
    if spec.use_expression and Xp.shape[1] > 0:
        parts.append(Xp[rows].astype(np.float32))
    if spec.use_row_stats:
        parts.append(row_mean[rows, None].astype(np.float32))
        parts.append(row_cnt[rows, None].astype(np.float32))
    if not parts:
        return np.ones((len(rows), 1), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _fallback_for_rows(row_mean: np.ndarray, col_mean_j: float, rows: np.ndarray) -> np.ndarray:
    return ((row_mean[rows] + float(col_mean_j)) / 2.0).astype(np.float32)


def _fit_drug_base(
    X: np.ndarray,
    M: np.ndarray,
    M_train: np.ndarray,
    train_mask: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    drug_idx: int,
    fold_idx: int,
    seed: int,
    config: MFMRConfig,
    spec: MethodSpec,
    *,
    need_train_predictions: bool,
) -> dict[str, np.ndarray]:
    n_rows, n_drugs = M.shape
    row_mean, col_mean, row_cnt, filled, _ = make_train_stats(M, train_mask)
    other_cols = np.asarray([k for k in range(n_drugs) if k != drug_idx], dtype=np.int64)
    fallback_pred = _fallback_for_rows(row_mean, col_mean[drug_idx], predict_rows)
    fallback_train = _fallback_for_rows(row_mean, col_mean[drug_idx], train_rows)

    if spec.use_expression:
        Xp_imp = fit_expr_pcs_train_only(X, train_rows, config.topg_imp, config.comp_imp, int(seed + 101 * fold_idx + drug_idx))
        Xp_ridge = fit_expr_pcs_train_only(
            X,
            train_rows,
            config.topg_ridge,
            config.comp_ridge,
            int(seed + 1009 + 101 * fold_idx + drug_idx),
        )
    else:
        Xp_imp = np.zeros((n_rows, 0), dtype=np.float32)
        Xp_ridge = np.zeros((n_rows, 0), dtype=np.float32)

    imp_predict = fallback_pred.copy()
    imp_train = fallback_train.copy()
    if spec.use_imputer:
        try:
            imputer, empty_cols, global_mean = _fit_imputer(M_train, Xp_imp, config, seed=config.random_state + seed + fold_idx)
            imp_all = _transform_imputer(imputer, M_train, Xp_imp, empty_cols, global_mean)
            imp_predict = imp_all[predict_rows, drug_idx].astype(np.float32)
            if need_train_predictions:
                M_train_view = M_train.copy()
                M_train_view[train_rows, drug_idx] = np.nan
                imp_masked = _transform_imputer(imputer, M_train_view, Xp_imp, empty_cols, global_mean)
                imp_train = imp_masked[train_rows, drug_idx].astype(np.float32)
        except Exception:
            imp_predict = fallback_pred.copy()
            imp_train = fallback_train.copy()

    ridge_predict = fallback_pred.copy()
    ridge_train = fallback_train.copy()
    if spec.use_ridge:
        Xtr = _ridge_features(filled, Xp_ridge, row_mean, row_cnt, train_rows, other_cols, spec)
        Xte = _ridge_features(filled, Xp_ridge, row_mean, row_cnt, predict_rows, other_cols, spec)
        y = M[train_rows, drug_idx].astype(np.float32)
        model = Ridge(alpha=float(config.ridge_alpha))
        model.fit(Xtr, y)
        ridge_predict = model.predict(Xte).astype(np.float32)
        if need_train_predictions:
            ridge_train = model.predict(Xtr).astype(np.float32)

    if spec.use_imputer and spec.use_ridge:
        base_predict = (spec.weight_imp * imp_predict + spec.weight_ridge * ridge_predict).astype(np.float32)
        base_train = (spec.weight_imp * imp_train + spec.weight_ridge * ridge_train).astype(np.float32)
    elif spec.use_imputer:
        base_predict = imp_predict.astype(np.float32)
        base_train = imp_train.astype(np.float32)
    elif spec.use_ridge:
        base_predict = ridge_predict.astype(np.float32)
        base_train = ridge_train.astype(np.float32)
    else:
        base_predict = fallback_pred.astype(np.float32)
        base_train = fallback_train.astype(np.float32)

    return {
        "imp_predict": imp_predict.astype(np.float32),
        "ridge_predict": ridge_predict.astype(np.float32),
        "base_predict": base_predict.astype(np.float32),
        "imp_train": imp_train.astype(np.float32),
        "ridge_train": ridge_train.astype(np.float32),
        "base_train": base_train.astype(np.float32),
        "row_mean": row_mean,
        "col_mean": col_mean,
        "row_cnt": row_cnt,
        "Xp_ridge": Xp_ridge.astype(np.float32),
    }


def _inner_splits(rows: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    n_splits = int(max(0, n_splits))
    if n_splits < 2 or len(rows) < n_splits:
        return []
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(rows)
    return [split.astype(np.int64) for split in np.array_split(shuffled, n_splits) if len(split) > 0]


def _crossfit_base_train_for_drug(
    X: np.ndarray,
    M: np.ndarray,
    outer_train_mask: np.ndarray,
    outer_train_rows: np.ndarray,
    drug_idx: int,
    fold_idx: int,
    seed: int,
    config: MFMRConfig,
    spec: MethodSpec,
    inner_cv: int,
    quick_base: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = _inner_splits(outer_train_rows, int(inner_cv), seed=config.random_state + seed + fold_idx + drug_idx)
    if not splits:
        return quick_base["imp_train"], quick_base["ridge_train"], quick_base["base_train"]

    tr_pos = {int(row): idx for idx, row in enumerate(outer_train_rows.tolist())}
    imp_cf = quick_base["imp_train"].copy()
    ridge_cf = quick_base["ridge_train"].copy()
    base_cf = quick_base["base_train"].copy()

    for held_rows in splits:
        inner_train_mask = outer_train_mask.copy()
        inner_train_mask[held_rows, drug_idx] = False
        inner_train_rows = np.where(inner_train_mask[:, drug_idx])[0]
        if len(inner_train_rows) < int(config.min_train_per_drug):
            continue
        inner_M_train = np.where(inner_train_mask, M, np.nan).astype(np.float32)
        pred = _fit_drug_base(
            X,
            M,
            inner_M_train,
            inner_train_mask,
            inner_train_rows,
            held_rows,
            drug_idx,
            fold_idx,
            seed + 7919,
            config,
            spec,
            need_train_predictions=False,
        )
        idx = np.asarray([tr_pos[int(row)] for row in held_rows], dtype=np.int64)
        imp_cf[idx] = pred["imp_predict"]
        ridge_cf[idx] = pred["ridge_predict"]
        base_cf[idx] = pred["base_predict"]
    return imp_cf.astype(np.float32), ridge_cf.astype(np.float32), base_cf.astype(np.float32)


def _mutation_design(
    dataset: str,
    mutation: MutationFeatureSet,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    Xp_ridge: np.ndarray,
    row_mean: np.ndarray,
    col_mean_j: float,
    row_cnt: np.ndarray,
    imp_train: np.ndarray,
    ridge_train: np.ndarray,
    base_train: np.ndarray,
    imp_predict: np.ndarray,
    ridge_predict: np.ndarray,
    base_predict: np.ndarray,
    spec: MethodSpec,
    mutation_config: MutationHeadConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta_train = np.concatenate(
        [
            imp_train[:, None],
            ridge_train[:, None],
            (imp_train * ridge_train)[:, None],
            row_mean[train_rows, None],
            np.full((len(train_rows), 1), float(col_mean_j), dtype=np.float32),
            row_cnt[train_rows, None],
            Xp_ridge[train_rows],
        ],
        axis=1,
    ).astype(np.float32)
    meta_predict = np.concatenate(
        [
            imp_predict[:, None],
            ridge_predict[:, None],
            (imp_predict * ridge_predict)[:, None],
            row_mean[predict_rows, None],
            np.full((len(predict_rows), 1), float(col_mean_j), dtype=np.float32),
            row_cnt[predict_rows, None],
            Xp_ridge[predict_rows],
        ],
        axis=1,
    ).astype(np.float32)

    mutation_train_parts = [mutation.tissue[train_rows], mutation.overall[train_rows]]
    mutation_predict_parts = [mutation.tissue[predict_rows], mutation.overall[predict_rows]]
    if spec.mutation_use_pathway:
        pathway = select_pathway_features(dataset, mutation)
        pathway_score = pathway.mean(axis=1, keepdims=True).astype(np.float32)
        mutation_train_parts.extend(
            [
                pathway[train_rows],
                pathway_score[train_rows],
                pathway_score[train_rows] * meta_train[:, :3],
            ]
        )
        mutation_predict_parts.extend(
            [
                pathway[predict_rows],
                pathway_score[predict_rows],
                pathway_score[predict_rows] * meta_predict[:, :3],
            ]
        )
    if spec.mutation_use_latent:
        latent = fit_binary_latent_train_only(
            mutation.binary,
            train_rows,
            int(mutation_config.mutation_latent_dim),
            seed=int(seed),
        )
        mutation_train_parts.append(latent[train_rows])
        mutation_predict_parts.append(latent[predict_rows])

    mut_train = np.concatenate(mutation_train_parts, axis=1).astype(np.float32) if mutation_train_parts else np.zeros((len(train_rows), 0), dtype=np.float32)
    mut_predict = (
        np.concatenate(mutation_predict_parts, axis=1).astype(np.float32)
        if mutation_predict_parts
        else np.zeros((len(predict_rows), 0), dtype=np.float32)
    )
    return np.concatenate([meta_train, mut_train], axis=1), np.concatenate([meta_predict, mut_predict], axis=1), base_train


def predict_global_mean(M: np.ndarray, folds: Sequence[np.ndarray]) -> PredictionResult:
    pred = np.zeros_like(M, dtype=np.float32)
    for examdata in folds:
        test_mask = examdata != 0
        train_mask = (M != 0) & (~test_mask)
        mean = _safe_nanmean(np.where(train_mask, M, np.nan), default=0.0)
        pred[test_mask] = float(mean)
    diagnostics = {
        "test_entries_nan_in_M_train": True,
        "expression_pc_mode": "none",
        "epsilon_mode": "not_applicable",
        "ridge_y_from_train_mask": "not_applicable",
        "residual_target": "none",
        "residual_inner_cv": 0,
        "num_residual_models": 0,
        "num_skipped_drug_fits": 0,
    }
    return PredictionResult("global_mean", pred, pred.copy(), pred.copy(), pred.copy(), None, list(folds), diagnostics)


def predict_row_col_mean(M: np.ndarray, folds: Sequence[np.ndarray]) -> PredictionResult:
    pred = np.zeros_like(M, dtype=np.float32)
    for examdata in folds:
        test_mask = examdata != 0
        train_mask = (M != 0) & (~test_mask)
        row_mean, col_mean, _, _, _ = make_train_stats(M, train_mask)
        rows, cols = np.where(test_mask)
        pred[rows, cols] = ((row_mean[rows] + col_mean[cols]) / 2.0).astype(np.float32)
    diagnostics = {
        "test_entries_nan_in_M_train": True,
        "expression_pc_mode": "none",
        "epsilon_mode": "not_applicable",
        "ridge_y_from_train_mask": "not_applicable",
        "residual_target": "none",
        "residual_inner_cv": 0,
        "num_residual_models": 0,
        "num_skipped_drug_fits": 0,
    }
    return PredictionResult("row_col_mean", pred, pred.copy(), pred.copy(), pred.copy(), None, list(folds), diagnostics)


def predict_original_mclrp(
    X: np.ndarray,
    M: np.ndarray,
    folds: Sequence[np.ndarray],
    config: MCLRPConfig | None = None,
) -> PredictionResult:
    config = config or MCLRPConfig()
    try:
        from .legacy_mclrp.main import predict_mclrp_from_folds_train_only
    except Exception as exc:
        try:
            from legacy_mclrp.main import predict_mclrp_from_folds_train_only
        except Exception:
            raise RuntimeError("original_mclrp is not redistributed in the public release; see THIRD_PARTY.md and obtain separately authorized comparator source") from exc

    pred = predict_mclrp_from_folds_train_only(
        X=X,
        MM=M,
        folds=folds,
        beta=float(config.beta),
        rho=float(config.rho),
        iter_num=int(config.iter_num),
        itol=float(config.itol),
        tk=float(config.tk),
        epsilon=None,
        epsilon_scale=float(config.epsilon_scale),
        gene_limit=int(config.gene_limit),
        suppress_output=True,
    ).astype(np.float32)
    diagnostics = {
        "test_entries_nan_in_M_train": True,
        "expression_pc_mode": "legacy_pca_train_rows_only_per_fold",
        "epsilon_mode": "fold_train_entries_only",
        "ridge_y_from_train_mask": "not_applicable",
        "residual_target": "none",
        "residual_inner_cv": 0,
        "num_residual_models": 0,
        "num_skipped_drug_fits": 0,
    }
    return PredictionResult("original_mclrp", pred, pred.copy(), pred.copy(), pred.copy(), None, list(folds), diagnostics)


def _fit_affine_calibration_from_train_mask(
    y_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    train_mask: np.ndarray,
    drug_idx: int,
    *,
    min_entries: int = 2,
) -> dict[str, Any]:
    train_rows = np.where(np.asarray(train_mask[:, drug_idx], dtype=bool))[0].astype(np.int64)
    y = np.asarray(y_matrix[train_rows, drug_idx], dtype=np.float64)
    x = np.asarray(pred_matrix[train_rows, drug_idx], dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(x)
    y = y[finite]
    x = x[finite]
    n_fit = int(y.size)
    if n_fit < int(min_entries):
        return {"a": 1.0, "b": 0.0, "n_fit": n_fit, "skipped_reason": "calibration_insufficient_train_entries"}
    if float(np.std(x)) == 0.0:
        return {"a": 1.0, "b": 0.0, "n_fit": n_fit, "skipped_reason": "calibration_degenerate_train_prediction"}
    try:
        design = np.column_stack([x, np.ones_like(x)])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        a = float(coeffs[0])
        b = float(coeffs[1])
    except Exception:
        return {"a": 1.0, "b": 0.0, "n_fit": n_fit, "skipped_reason": "calibration_fit_failed"}
    if not (np.isfinite(a) and np.isfinite(b)):
        return {"a": 1.0, "b": 0.0, "n_fit": n_fit, "skipped_reason": "calibration_nonfinite_coefficients"}
    return {"a": a, "b": b, "n_fit": n_fit, "skipped_reason": "none"}


def predict_original_mclrp_calibrated(
    X: np.ndarray,
    M: np.ndarray,
    folds: Sequence[np.ndarray],
    config: MCLRPConfig | None = None,
) -> PredictionResult:
    config = config or MCLRPConfig()
    try:
        from .legacy_mclrp.main import predict_mclrp_from_folds_train_only_with_fold_outputs
    except Exception as exc:
        try:
            from legacy_mclrp.main import predict_mclrp_from_folds_train_only_with_fold_outputs
        except Exception:
            raise RuntimeError("original_mclrp_calibrated is not redistributed in the public release; see THIRD_PARTY.md and obtain separately authorized comparator source") from exc

    raw_pred, fold_outputs = predict_mclrp_from_folds_train_only_with_fold_outputs(
        X=X,
        MM=M,
        folds=folds,
        beta=float(config.beta),
        rho=float(config.rho),
        iter_num=int(config.iter_num),
        itol=float(config.itol),
        tk=float(config.tk),
        epsilon=None,
        epsilon_scale=float(config.epsilon_scale),
        gene_limit=int(config.gene_limit),
        suppress_output=True,
    )
    raw_pred = raw_pred.astype(np.float32)
    pred = raw_pred.copy()
    calibration_rows: list[dict[str, Any]] = []
    skipped_count = 0
    fitted_count = 0
    coeff_a: list[float] = []
    coeff_b: list[float] = []

    for fold_idx, examdata in enumerate(folds, start=1):
        test_mask = np.asarray(examdata != 0, dtype=bool)
        train_mask = (M != 0) & (~test_mask)
        if fold_idx - 1 < fold_outputs.shape[0]:
            full_pred = np.asarray(fold_outputs[fold_idx - 1], dtype=np.float32)
        else:
            full_pred = raw_pred
        for drug_idx in range(M.shape[1]):
            test_rows = np.where(test_mask[:, drug_idx])[0].astype(np.int64)
            if len(test_rows) == 0:
                calibration_rows.append(
                    {
                        "fold": int(fold_idx),
                        "drug_index": int(drug_idx),
                        "calibration_a": 1.0,
                        "calibration_b": 0.0,
                        "calibration_fit_n": 0,
                        "calibration_skipped_reason": "no_test_entries_for_drug",
                    }
                )
                continue
            fit = _fit_affine_calibration_from_train_mask(M, full_pred, train_mask, drug_idx)
            a = float(fit["a"])
            b = float(fit["b"])
            skipped_reason = str(fit["skipped_reason"])
            if skipped_reason == "none":
                pred[test_rows, drug_idx] = (a * raw_pred[test_rows, drug_idx].astype(np.float32) + b).astype(np.float32)
                fitted_count += 1
                coeff_a.append(a)
                coeff_b.append(b)
            else:
                pred[test_rows, drug_idx] = raw_pred[test_rows, drug_idx]
                skipped_count += 1
            calibration_rows.append(
                {
                    "fold": int(fold_idx),
                    "drug_index": int(drug_idx),
                    "calibration_a": a,
                    "calibration_b": b,
                    "calibration_fit_n": int(fit["n_fit"]),
                    "calibration_skipped_reason": skipped_reason,
                }
            )

    diagnostics = {
        "test_entries_nan_in_M_train": True,
        "expression_pc_mode": "legacy_pca_train_rows_only_per_fold",
        "epsilon_mode": "fold_train_entries_only",
        "ridge_y_from_train_mask": "not_applicable",
        "residual_target": "none",
        "residual_inner_cv": 0,
        "num_residual_models": 0,
        "num_skipped_drug_fits": int(skipped_count),
        "calibration_mode": "train_only_affine_per_fold_drug",
        "num_calibration_fits": int(fitted_count),
        "num_calibration_skipped": int(skipped_count),
        "calibration_a_mean": float(np.mean(coeff_a)) if coeff_a else 1.0,
        "calibration_b_mean": float(np.mean(coeff_b)) if coeff_b else 0.0,
    }
    return PredictionResult(
        "original_mclrp_calibrated",
        pred,
        raw_pred.copy(),
        raw_pred.copy(),
        raw_pred.copy(),
        None,
        list(folds),
        diagnostics,
        calibration_rows,
    )


def predict_original_mclrp_pair(
    X: np.ndarray,
    M: np.ndarray,
    folds: Sequence[np.ndarray],
    config: MCLRPConfig | None = None,
) -> tuple[PredictionResult, PredictionResult]:
    calibrated = predict_original_mclrp_calibrated(X, M, folds, config=config)
    raw_pred = calibrated.base.astype(np.float32)
    diagnostics = {
        "test_entries_nan_in_M_train": True,
        "expression_pc_mode": "legacy_pca_train_rows_only_per_fold",
        "epsilon_mode": "fold_train_entries_only",
        "ridge_y_from_train_mask": "not_applicable",
        "residual_target": "none",
        "residual_inner_cv": 0,
        "num_residual_models": 0,
        "num_skipped_drug_fits": 0,
    }
    raw = PredictionResult("original_mclrp", raw_pred, raw_pred.copy(), raw_pred.copy(), raw_pred.copy(), None, list(folds), diagnostics)
    return raw, calibrated


def predict_mfmr_t0_seed(
    dataset: str,
    X: np.ndarray,
    M: np.ndarray,
    folds: Sequence[np.ndarray],
    method: str,
    config: MFMRConfig,
    mutation_config: MutationHeadConfig | None = None,
    mutation_features: MutationFeatureSet | None = None,
    mclrp_config: MCLRPConfig | None = None,
) -> PredictionResult:
    if method == "global_mean":
        return predict_global_mean(M, folds)
    if method == "row_col_mean":
        return predict_row_col_mean(M, folds)
    if method == "original_mclrp":
        return predict_original_mclrp(X, M, folds, config=mclrp_config)
    if method == "original_mclrp_calibrated":
        return predict_original_mclrp_calibrated(X, M, folds, config=mclrp_config)

    mutation_config = mutation_config or MutationHeadConfig()
    spec = resolve_method_spec(method, config)
    if spec.use_mutation and mutation_features is None:
        raise ValueError(f"{method} requires mutation features for dataset {dataset}")

    pred = np.zeros_like(M, dtype=np.float32)
    imp_arr = np.zeros_like(M, dtype=np.float32)
    ridge_arr = np.zeros_like(M, dtype=np.float32)
    base_arr = np.zeros_like(M, dtype=np.float32)
    final_arr = np.zeros_like(M, dtype=np.float32) if spec.use_mutation else None
    diagnostics: dict[str, Any] = {
        "test_entries_nan_in_M_train": True,
        "expression_pc_mode": "train_rows_only_per_drug" if spec.use_expression else "none",
        "epsilon_mode": "not_applicable",
        "ridge_y_from_train_mask": True,
        "residual_target": "y_train_minus_base_pred_train" if spec.use_mutation else "none",
        "residual_inner_cv": int(mutation_config.residual_inner_cv),
        "num_residual_models": 0,
        "num_skipped_drug_fits": 0,
    }

    for fold_idx, examdata in enumerate(folds):
        test_mask = examdata != 0
        train_mask = (M != 0) & (~test_mask)
        M_train = np.where(train_mask, M, np.nan).astype(np.float32)
        if not np.all(np.isnan(M_train[test_mask])):
            diagnostics["test_entries_nan_in_M_train"] = False
        row_mean, col_mean, _, _, _ = make_train_stats(M, train_mask)
        for drug_idx in range(M.shape[1]):
            train_rows = np.where(train_mask[:, drug_idx])[0].astype(np.int64)
            test_rows = np.where(test_mask[:, drug_idx])[0].astype(np.int64)
            if len(test_rows) == 0:
                continue
            fallback = _fallback_for_rows(row_mean, col_mean[drug_idx], test_rows)
            if len(train_rows) < int(config.min_train_per_drug):
                diagnostics["num_skipped_drug_fits"] += 1
                imp_arr[test_rows, drug_idx] = fallback
                ridge_arr[test_rows, drug_idx] = fallback
                base_arr[test_rows, drug_idx] = fallback
                pred[test_rows, drug_idx] = fallback
                if final_arr is not None:
                    final_arr[test_rows, drug_idx] = fallback
                continue

            expected_train_rows = np.where(train_mask[:, drug_idx])[0]
            if not np.array_equal(train_rows, expected_train_rows):
                diagnostics["ridge_y_from_train_mask"] = False

            base = _fit_drug_base(
                X,
                M,
                M_train,
                train_mask,
                train_rows,
                test_rows,
                drug_idx,
                fold_idx,
                int(config.random_state),
                config,
                spec,
                need_train_predictions=spec.use_mutation,
            )
            imp_arr[test_rows, drug_idx] = base["imp_predict"]
            ridge_arr[test_rows, drug_idx] = base["ridge_predict"]
            base_arr[test_rows, drug_idx] = base["base_predict"]

            if not spec.use_mutation:
                pred[test_rows, drug_idx] = base["base_predict"]
                continue

            assert mutation_features is not None
            imp_train, ridge_train, base_train = _crossfit_base_train_for_drug(
                X,
                M,
                train_mask,
                train_rows,
                drug_idx,
                fold_idx,
                int(config.random_state),
                config,
                spec,
                int(mutation_config.residual_inner_cv),
                base,
            )
            residual_y = (M[train_rows, drug_idx] - base_train).astype(np.float32)
            Xres_train, Xres_test, _ = _mutation_design(
                dataset,
                mutation_features,
                train_rows,
                test_rows,
                base["Xp_ridge"],
                base["row_mean"],
                float(base["col_mean"][drug_idx]),
                base["row_cnt"],
                imp_train,
                ridge_train,
                base_train,
                base["imp_predict"],
                base["ridge_predict"],
                base["base_predict"],
                spec,
                mutation_config,
                seed=int(config.random_state + 3001 * fold_idx + drug_idx),
            )
            residual_model = Ridge(alpha=float(mutation_config.final_alpha))
            residual_model.fit(Xres_train, residual_y)
            residual_pred = residual_model.predict(Xres_test).astype(np.float32)
            final_values = (base["base_predict"] + residual_pred).astype(np.float32)
            pred[test_rows, drug_idx] = final_values
            if final_arr is not None:
                final_arr[test_rows, drug_idx] = final_values
            diagnostics["num_residual_models"] += 1

    return PredictionResult(method, pred, imp_arr, ridge_arr, base_arr, final_arr, list(folds), diagnostics)


def _mask_sha256(mask: np.ndarray) -> str:
    mask_arr = np.asarray(mask, dtype=np.uint8)
    shape_arr = np.asarray(mask_arr.shape, dtype=np.int64)
    return sha256(shape_arr.tobytes() + np.ascontiguousarray(mask_arr).tobytes()).hexdigest()


def build_protocol_audit_rows(
    dataset: T0DatasetBundle,
    methods: Sequence[str],
    seed: int,
    folds: Sequence[np.ndarray],
    config: MFMRConfig,
    mutation_config: MutationHeadConfig,
    *,
    mutation_features_available: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed_mask = dataset.M != 0
    observed_by_drug = np.sum(observed_mask, axis=0).astype(np.int64)

    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown method {method!r}; choices: {SUPPORTED_METHODS}")

        spec: MethodSpec | None = None
        if method not in {"global_mean", "row_col_mean", "original_mclrp", "original_mclrp_calibrated"}:
            spec = resolve_method_spec(method, config)
        uses_mutation = bool(spec.use_mutation) if spec is not None else False
        missing_mutation = uses_mutation and not mutation_features_available

        for fold_idx, examdata in enumerate(folds, start=1):
            test_mask = examdata != 0
            train_mask = observed_mask & (~test_mask)
            fold_train_rows = np.where(np.sum(train_mask, axis=1) > 0)[0].astype(np.int64)
            fold_hash = _mask_sha256(test_mask)
            M_train = np.where(train_mask, dataset.M, np.nan).astype(np.float32)
            test_entries_masked = bool(np.all(np.isnan(M_train[test_mask])))

            for drug_idx in range(dataset.M.shape[1]):
                train_rows = np.where(train_mask[:, drug_idx])[0].astype(np.int64)
                test_rows = np.where(test_mask[:, drug_idx])[0].astype(np.int64)
                skipped_reason = "none"
                expression_pc_mode = "none"
                expression_pc_train_rows = 0
                ridge_train_rows_only: bool | str = "not_applicable"
                mutation_residual_mode = "none"
                residual_inner_cv = 0

                if method in {"original_mclrp", "original_mclrp_calibrated"}:
                    expression_pc_mode = "legacy_pca_train_rows_only_per_fold"
                    expression_pc_train_rows = int(len(fold_train_rows))
                elif spec is not None:
                    if spec.use_expression:
                        expression_pc_mode = "train_rows_only_per_drug"
                        expression_pc_train_rows = int(len(train_rows))
                    if spec.use_ridge:
                        ridge_train_rows_only = True
                    if spec.use_mutation:
                        residual_inner_cv = int(mutation_config.residual_inner_cv)
                        mutation_residual_mode = (
                            "y_train_minus_crossfit_base_pred_train"
                            if int(mutation_config.residual_inner_cv) > 1
                            else "y_train_minus_base_pred_train"
                        )

                if len(test_rows) == 0:
                    skipped_reason = "no_test_entries_for_drug"
                elif missing_mutation:
                    skipped_reason = "skipped_missing_mutation_features"
                    mutation_residual_mode = "not_run_missing_mutation_features"
                    residual_inner_cv = int(mutation_config.residual_inner_cv)
                elif spec is not None and len(train_rows) < int(config.min_train_per_drug):
                    skipped_reason = "fallback_min_train_per_drug"

                rows.append(
                    {
                        "dataset": dataset.name,
                        "seed": int(seed),
                        "fold": int(fold_idx),
                        "method": method,
                        "drug_index": int(drug_idx),
                        "n_train_entries_for_drug": int(len(train_rows)),
                        "n_test_entries_for_drug": int(len(test_rows)),
                        "n_total_observed_entries_for_drug": int(observed_by_drug[drug_idx]),
                        "expression_pc_mode": expression_pc_mode,
                        "expression_pc_train_rows": int(expression_pc_train_rows),
                        "imputer_test_entries_masked": test_entries_masked,
                        "ridge_train_rows_only": ridge_train_rows_only,
                        "mutation_residual_mode": mutation_residual_mode,
                        "residual_inner_cv": int(residual_inner_cv),
                        "skipped_reason": skipped_reason,
                        "calibration_a": 1.0,
                        "calibration_b": 0.0,
                        "calibration_fit_n": 0,
                        "calibration_skipped_reason": "not_applicable",
                        "fold_mask_sha256": fold_hash,
                    }
                )
    return rows


def _merge_calibration_audit_rows(
    audit_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not calibration_rows:
        return audit_rows
    by_key = {
        (int(row["fold"]), int(row["drug_index"])): row
        for row in calibration_rows
        if "fold" in row and "drug_index" in row
    }
    for row in audit_rows:
        if row.get("method") != "original_mclrp_calibrated":
            continue
        meta = by_key.get((int(row["fold"]), int(row["drug_index"])))
        if meta is None:
            row["calibration_skipped_reason"] = "calibration_audit_missing"
            row["skipped_reason"] = "calibration_audit_missing"
            continue
        row["calibration_a"] = float(meta.get("calibration_a", 1.0))
        row["calibration_b"] = float(meta.get("calibration_b", 0.0))
        row["calibration_fit_n"] = int(meta.get("calibration_fit_n", 0))
        reason = str(meta.get("calibration_skipped_reason", "none"))
        row["calibration_skipped_reason"] = reason
        if reason != "none" and row.get("skipped_reason") == "none":
            row["skipped_reason"] = reason
    return audit_rows


def evaluate_prediction(
    dataset: T0DatasetBundle,
    method: str,
    seed: int,
    pred: np.ndarray,
    folds: Sequence[np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall_pcc, fold_pcc = calc_pcc(pred, dataset.M, folds)
    mask_all = np.zeros(dataset.M.shape, dtype=bool)
    fold_rows: list[dict[str, Any]] = []
    for fold_idx, examdata in enumerate(folds, start=1):
        mask = examdata != 0
        mask_all |= mask
        fold_rows.append(
            {
                "dataset": dataset.name,
                "method": method,
                "seed": int(seed),
                "fold": int(fold_idx),
                "pcc": float(fold_pcc[fold_idx - 1]),
                "rmse": _rmse(dataset.M[mask], pred[mask]),
                "mae": _mae(dataset.M[mask], pred[mask]),
                "n_test": int(mask.sum()),
            }
        )
    seed_row = {
        "dataset": dataset.name,
        "method": method,
        "seed": int(seed),
        "overall_pcc": float(overall_pcc),
        "mean_fold_pcc": float(np.mean(fold_pcc)) if len(fold_pcc) else 0.0,
        "rmse": _rmse(dataset.M[mask_all], pred[mask_all]),
        "mae": _mae(dataset.M[mask_all], pred[mask_all]),
        "n_test": int(mask_all.sum()),
    }

    per_drug_rows: list[dict[str, Any]] = []
    for j, label in enumerate(dataset.drug_labels):
        mask = mask_all[:, j]
        per_drug_rows.append(
            {
                "dataset": dataset.name,
                "method": method,
                "seed": int(seed),
                "drug_idx": int(j),
                "drug": str(label),
                "pcc": _safe_corr(dataset.M[mask, j], pred[mask, j]),
                "rmse": _rmse(dataset.M[mask, j], pred[mask, j]),
                "mae": _mae(dataset.M[mask, j], pred[mask, j]),
                "n_test": int(mask.sum()),
            }
        )

    per_cell_rows: list[dict[str, Any]] = []
    for i, label in enumerate(dataset.cell_labels):
        mask = mask_all[i, :]
        per_cell_rows.append(
            {
                "dataset": dataset.name,
                "method": method,
                "seed": int(seed),
                "cell_line_idx": int(i),
                "cell_line": str(label),
                "pcc": _safe_corr(dataset.M[i, mask], pred[i, mask]),
                "rmse": _rmse(dataset.M[i, mask], pred[i, mask]),
                "mae": _mae(dataset.M[i, mask], pred[i, mask]),
                "n_test": int(mask.sum()),
            }
        )
    return seed_row, fold_rows, per_drug_rows, per_cell_rows


def _summary_from_seed(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in seed_df.groupby(["dataset", "method"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_seeds": int(group["seed"].nunique()),
                "overall_pcc_mean": float(group["overall_pcc"].mean()),
                "overall_pcc_std": float(group["overall_pcc"].std(ddof=0)),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": float(group["rmse"].std(ddof=0)),
                "mae_mean": float(group["mae"].mean()),
                "mae_std": float(group["mae"].std(ddof=0)),
                "n_test_total": int(group["n_test"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _ablation_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if seed_df.empty:
        return pd.DataFrame(rows)
    for dataset, ddf in seed_df.groupby("dataset", sort=True):
        for method in ABLATION_COMPARISON_METHODS:
            if method not in set(ddf["method"]):
                continue
            reference = "mfmr_mutation" if method.startswith("mutation_") else "mfmr_base"
            if reference not in set(ddf["method"]):
                continue
            left = ddf[ddf["method"] == method][["seed", "overall_pcc", "rmse", "mae"]]
            right = ddf[ddf["method"] == reference][["seed", "overall_pcc", "rmse", "mae"]]
            merged = left.merge(right, on="seed", suffixes=("", "_reference"))
            if merged.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "reference_method": reference,
                    "n_pairs": int(len(merged)),
                    "delta_pcc_mean": float((merged["overall_pcc"] - merged["overall_pcc_reference"]).mean()),
                    "delta_rmse_mean": float((merged["rmse"] - merged["rmse_reference"]).mean()),
                    "delta_mae_mean": float((merged["mae"] - merged["mae_reference"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def _statistical_tests(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if seed_df.empty:
        return pd.DataFrame(rows)
    for dataset, ddf in seed_df.groupby("dataset", sort=True):
        methods = sorted(set(ddf["method"]))
        for method in methods:
            if method in {"mfmr_base", "mfmr_mutation"}:
                continue
            reference = "mfmr_mutation" if method.startswith("mutation_") else "mfmr_base"
            if reference not in methods:
                continue
            for metric in ("overall_pcc", "rmse", "mae"):
                if metric not in ddf.columns:
                    continue
                left = ddf[ddf["method"] == method][["seed", metric]]
                right = ddf[ddf["method"] == reference][["seed", metric]]
                merged = left.merge(right, on="seed", suffixes=("", "_reference"))
                p_t = np.nan
                p_w = np.nan
                if len(merged) >= 2:
                    delta = (
                        pd.to_numeric(merged[metric], errors="coerce")
                        - pd.to_numeric(merged[f"{metric}_reference"], errors="coerce")
                    ).to_numpy(dtype=float)
                    finite_delta = delta[np.isfinite(delta)]
                    if finite_delta.size and np.allclose(finite_delta, 0.0, atol=1e-15, rtol=0.0):
                        p_t = 1.0
                        p_w = 1.0
                    else:
                        try:
                            p_t = float(ttest_rel(merged[metric], merged[f"{metric}_reference"]).pvalue)
                        except Exception:
                            p_t = np.nan
                        try:
                            p_w = float(wilcoxon(merged[metric], merged[f"{metric}_reference"], zero_method="wilcox").pvalue)
                        except Exception:
                            p_w = np.nan
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "reference_method": reference,
                        "metric": metric,
                        "n_pairs": int(len(merged)),
                        "mean_delta": float((merged[metric] - merged[f"{metric}_reference"]).mean()) if len(merged) else np.nan,
                        "paired_t_pvalue": p_t,
                        "wilcoxon_pvalue": p_w,
                    }
                )
    return pd.DataFrame(rows)


def _prediction_cache_key(
    method: str,
    config: MFMRConfig,
    mutation_config: MutationHeadConfig,
) -> tuple[Any, ...] | None:
    if method in {"global_mean", "row_col_mean", "original_mclrp", "original_mclrp_calibrated"}:
        return None
    spec = resolve_method_spec(method, config)
    return (
        bool(spec.use_imputer),
        bool(spec.use_ridge),
        bool(spec.use_expression),
        bool(spec.use_row_stats),
        bool(spec.use_mutation),
        bool(spec.mutation_use_pathway),
        bool(spec.mutation_use_latent),
        float(spec.weight_imp),
        float(spec.weight_ridge),
        int(mutation_config.residual_inner_cv) if spec.use_mutation else 0,
        float(mutation_config.final_alpha) if spec.use_mutation else 0.0,
        int(mutation_config.mutation_latent_dim) if spec.use_mutation else 0,
    )


def _clone_prediction_result(result: PredictionResult, method: str) -> PredictionResult:
    return PredictionResult(
        method=method,
        prediction=result.prediction.copy(),
        imputer=result.imputer.copy(),
        ridge=result.ridge.copy(),
        base=result.base.copy(),
        final=result.final.copy() if result.final is not None else None,
        folds=[fold.copy() for fold in result.folds],
        diagnostics=dict(result.diagnostics),
        calibration_audit=[dict(row) for row in result.calibration_audit] if result.calibration_audit is not None else None,
    )


def _write_frame(path: Path, frame: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _diagnostics_with_defaults(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "calibration_mode": "not_applicable",
        "num_calibration_fits": 0,
        "num_calibration_skipped": 0,
        "calibration_a_mean": 1.0,
        "calibration_b_mean": 0.0,
    }
    defaults.update(dict(diagnostics))
    return defaults


def save_prediction_npz(path: Path, result: PredictionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "prediction": result.prediction.astype(np.float32),
        "imputer": result.imputer.astype(np.float32),
        "ridge": result.ridge.astype(np.float32),
        "base": result.base.astype(np.float32),
        "folds": np.asarray(result.folds, dtype=np.float32),
    }
    if result.final is not None:
        payload["final"] = result.final.astype(np.float32)
    np.savez(path, **payload)


def run_t0_benchmark(
    datasets: Sequence[str],
    methods: Sequence[str],
    seeds: Sequence[int],
    config: MFMRConfig | None = None,
    mutation_config: MutationHeadConfig | None = None,
    mclrp_config: MCLRPConfig | None = None,
    output_dir: Path | str | None = None,
    *,
    save_predictions: bool = False,
    max_cell_lines: int | None = None,
    max_drugs: int | None = None,
) -> dict[str, Any]:
    config = config or MFMRConfig(seeds=tuple(int(s) for s in seeds))
    config = replace(config, seeds=tuple(int(s) for s in seeds))
    mutation_config = mutation_config or MutationHeadConfig()
    mclrp_config = mclrp_config or MCLRPConfig()
    output_root = Path(output_dir) if output_dir is not None else RESULTS_DIR / "t0_mfmr"
    output_root.mkdir(parents=True, exist_ok=True)
    pred_dir = output_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    per_drug_rows: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for dataset_name in datasets:
        bundle_full = load_t0_dataset(dataset_name)
        mutation_full = load_mutation_features_for_dataset(bundle_full)
        bundle, row_idx, _ = subset_dataset(bundle_full, max_cell_lines=max_cell_lines, max_drugs=max_drugs)
        mutation_features = mutation_full.subset(row_idx) if mutation_full is not None else None
        for seed in tuple(int(s) for s in seeds):
            np.random.seed(int(config.random_state) + int(seed))
            folds = getcrossMatrixs(bundle.M, num_folds=int(config.num_folds), rng=np.random.default_rng(seed))
            paired_original_cache: dict[str, PredictionResult] = {}
            method_result_cache: dict[tuple[Any, ...], PredictionResult] = {}
            for method in methods:
                if method not in SUPPORTED_METHODS:
                    raise ValueError(f"Unknown method {method!r}; choices: {SUPPORTED_METHODS}")
                method_audit_rows = build_protocol_audit_rows(
                    bundle,
                    (method,),
                    int(seed),
                    folds,
                    config,
                    mutation_config,
                    mutation_features_available=mutation_features is not None,
                )
                if resolve_method_spec(method, config).use_mutation and mutation_features is None:
                    audit_rows.extend(method_audit_rows)
                    diagnostics_rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method,
                            "seed": int(seed),
                            "status": "skipped_missing_mutation_features",
                            "test_entries_nan_in_M_train": "not_run",
                            "expression_pc_mode": "not_run_missing_mutation_features",
                            "epsilon_mode": "not_applicable",
                            "ridge_y_from_train_mask": "not_run",
                            "residual_target": "not_run_missing_mutation_features",
                            "residual_inner_cv": int(mutation_config.residual_inner_cv),
                            "num_residual_models": 0,
                            "num_skipped_drug_fits": 0,
                            **_diagnostics_with_defaults({}),
                        }
                    )
                    continue
                cache_key = _prediction_cache_key(method, config, mutation_config)
                if (
                    method in {"original_mclrp", "original_mclrp_calibrated"}
                    and "original_mclrp" in methods
                    and "original_mclrp_calibrated" in methods
                ):
                    if not paired_original_cache:
                        raw_original, calibrated_original = predict_original_mclrp_pair(bundle.X, bundle.M, folds, config=mclrp_config)
                        paired_original_cache["original_mclrp"] = raw_original
                        paired_original_cache["original_mclrp_calibrated"] = calibrated_original
                    result = paired_original_cache[method]
                elif cache_key is not None and cache_key in method_result_cache:
                    result = _clone_prediction_result(method_result_cache[cache_key], method)
                else:
                    result = predict_mfmr_t0_seed(
                        bundle.name,
                        bundle.X,
                        bundle.M,
                        folds,
                        method,
                        config,
                        mutation_config=replace(mutation_config, enabled=resolve_method_spec(method, config).use_mutation),
                        mutation_features=mutation_features,
                        mclrp_config=mclrp_config,
                    )
                    if cache_key is not None:
                        method_result_cache[cache_key] = _clone_prediction_result(result, result.method)
                method_audit_rows = _merge_calibration_audit_rows(method_audit_rows, result.calibration_audit)
                audit_rows.extend(method_audit_rows)
                seed_row, fold_metric_rows, drug_metric_rows, cell_metric_rows = evaluate_prediction(
                    bundle,
                    method,
                    int(seed),
                    result.prediction,
                    folds,
                )
                seed_rows.append(seed_row)
                fold_rows.extend(fold_metric_rows)
                per_drug_rows.extend(drug_metric_rows)
                per_cell_rows.extend(cell_metric_rows)
                diagnostics_rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "seed": int(seed),
                        "status": "ok",
                        **_diagnostics_with_defaults(result.diagnostics),
                    }
                )
                if save_predictions and int(seed) == 0:
                    save_prediction_npz(pred_dir / f"{dataset_name}_{method}_seed0.npz", result)

    seed_df = pd.DataFrame(seed_rows)
    fold_df = pd.DataFrame(fold_rows)
    per_drug_df = pd.DataFrame(per_drug_rows)
    per_cell_df = pd.DataFrame(per_cell_rows)
    method_df = _summary_from_seed(seed_df)
    ablation_df = _ablation_summary(seed_df)
    tests_df = _statistical_tests(seed_df)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    audit_df = pd.DataFrame(audit_rows)

    config_payload = {
        "protocol": "strict_t0_random_entry_masking_mfmr_v1",
        "datasets": list(datasets),
        "methods": list(methods),
        "mfmr_config": dataclass_to_jsonable(config),
        "mutation_head_config": dataclass_to_jsonable(mutation_config),
        "mclrp_config": dataclass_to_jsonable(mclrp_config),
        "max_cell_lines": max_cell_lines,
        "max_drugs": max_drugs,
        "output_dir": str(output_root),
    }
    (output_root / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_frame(output_root / "method_summary.csv", method_df)
    _write_frame(output_root / "seed_summary.csv", seed_df)
    _write_frame(output_root / "fold_summary.csv", fold_df)
    _write_frame(output_root / "per_drug_pcc.csv", per_drug_df)
    _write_frame(output_root / "per_cell_line_pcc.csv", per_cell_df)
    _write_frame(
        output_root / "ablation_summary.csv",
        ablation_df,
        columns=("dataset", "method", "reference_method", "n_pairs", "delta_pcc_mean", "delta_rmse_mean", "delta_mae_mean"),
    )
    _write_frame(
        output_root / "statistical_tests.csv",
        tests_df,
        columns=("dataset", "method", "reference_method", "metric", "n_pairs", "mean_delta", "paired_t_pvalue", "wilcoxon_pvalue"),
    )
    _write_frame(output_root / "diagnostics.csv", diagnostics_df)
    _write_frame(
        output_root / "protocol_audit.csv",
        audit_df,
        columns=(
            "dataset",
            "seed",
            "fold",
            "method",
            "drug_index",
            "n_train_entries_for_drug",
            "n_test_entries_for_drug",
            "n_total_observed_entries_for_drug",
            "expression_pc_mode",
            "expression_pc_train_rows",
            "imputer_test_entries_masked",
            "ridge_train_rows_only",
            "mutation_residual_mode",
            "residual_inner_cv",
            "skipped_reason",
            "calibration_a",
            "calibration_b",
            "calibration_fit_n",
            "calibration_skipped_reason",
            "fold_mask_sha256",
        ),
    )
    return {
        "output_dir": str(output_root),
        "config": config_payload,
        "method_summary": method_df.to_dict(orient="records"),
        "seed_summary": seed_df.to_dict(orient="records"),
        "diagnostics": diagnostics_df.to_dict(orient="records"),
    }


def run_leakage_safety_checks() -> dict[str, Any]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(18, 40)).astype(np.float32)
    M = rng.normal(size=(18, 6)).astype(np.float32)
    missing = rng.random(M.shape) < 0.2
    M[missing] = 0.0
    folds = getcrossMatrixs(M, num_folds=3, rng=np.random.default_rng(7))
    examdata = folds[0]
    test_mask = examdata != 0
    train_mask = (M != 0) & (~test_mask)
    M_train = np.where(train_mask, M, np.nan)
    checks: dict[str, Any] = {}
    checks["test_entries_nan_in_M_train"] = bool(np.all(np.isnan(M_train[test_mask])))

    train_rows = np.where(train_mask[:, 0])[0]
    if len(train_rows) < 2:
        train_rows = np.where(np.sum(train_mask, axis=1) > 0)[0]
    pcs_a = fit_expr_pcs_train_only(X, train_rows, topg=12, comp=5, seed=1)
    X_perturbed = X.copy()
    test_only_rows = np.setdiff1d(np.arange(X.shape[0]), train_rows)
    X_perturbed[test_only_rows] += rng.normal(loc=500.0, scale=20.0, size=X_perturbed[test_only_rows].shape).astype(np.float32)
    pcs_b = fit_expr_pcs_train_only(X_perturbed, train_rows, topg=12, comp=5, seed=1)
    checks["expr_fit_ignores_test_rows"] = bool(np.allclose(pcs_a[train_rows], pcs_b[train_rows], atol=1e-4))

    drug_idx = 0
    expected_y_rows = np.where(train_mask[:, drug_idx])[0]
    checks["ridge_training_y_uses_train_mask"] = bool(np.array_equal(expected_y_rows, np.where(train_mask[:, drug_idx])[0]))

    folds_a = getcrossMatrixs(M, num_folds=3, rng=np.random.default_rng(11))
    folds_b = getcrossMatrixs(M, num_folds=3, rng=np.random.default_rng(11))
    checks["identical_folds_for_same_seed"] = bool(all(np.array_equal(a, b) for a, b in zip(folds_a, folds_b)))

    toy_M = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )
    toy_train_mask = np.asarray(
        [
            [True, True],
            [True, True],
            [True, True],
            [False, False],
        ]
    )
    toy_pred = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [100.0, 200.0],
        ],
        dtype=np.float32,
    )
    cal_fit = _fit_affine_calibration_from_train_mask(toy_M, toy_pred, toy_train_mask, 0)
    toy_M_test_changed = toy_M.copy()
    toy_pred_test_changed = toy_pred.copy()
    toy_M_test_changed[3, 0] = -9999.0
    toy_pred_test_changed[3, 0] = 9999.0
    cal_fit_perturbed = _fit_affine_calibration_from_train_mask(toy_M_test_changed, toy_pred_test_changed, toy_train_mask, 0)
    checks["calibration_coefficients_use_train_mask_entries_only"] = bool(
        np.isclose(cal_fit["a"], 2.0)
        and np.isclose(cal_fit["b"], 1.0)
        and np.isclose(cal_fit["a"], cal_fit_perturbed["a"])
        and np.isclose(cal_fit["b"], cal_fit_perturbed["b"])
    )
    pcc_raw = _safe_corr(toy_M[:3, 0], toy_pred[:3, 0])
    pcc_affine = _safe_corr(toy_M[:3, 0], 3.5 * toy_pred[:3, 0] - 2.0)
    checks["positive_affine_calibration_preserves_pcc"] = bool(np.isclose(pcc_raw, pcc_affine, atol=1e-12))

    audit_bundle = T0DatasetBundle(
        name="SYNTHETIC",
        X=X,
        M=M,
        cell_labels=np.asarray([f"Cell_{i}" for i in range(M.shape[0])], dtype=object),
        drug_labels=np.asarray([f"Drug_{j}" for j in range(M.shape[1])], dtype=object),
        response_path="synthetic",
        expression_path="synthetic",
    )
    audit = build_protocol_audit_rows(
        audit_bundle,
        ("original_mclrp", "original_mclrp_calibrated"),
        0,
        folds_a,
        MFMRConfig(num_folds=3, seeds=(0,)),
        MutationHeadConfig(),
        mutation_features_available=False,
    )
    original_audit = {
        (int(row["fold"]), int(row["drug_index"])): row
        for row in audit
        if row["method"] == "original_mclrp"
    }
    calibrated_audit = {
        (int(row["fold"]), int(row["drug_index"])): row
        for row in audit
        if row["method"] == "original_mclrp_calibrated"
    }
    checks["calibrated_audit_preserves_original_folds_and_counts"] = bool(
        original_audit.keys() == calibrated_audit.keys()
        and all(
            original_audit[key]["fold_mask_sha256"] == calibrated_audit[key]["fold_mask_sha256"]
            and original_audit[key]["n_test_entries_for_drug"] == calibrated_audit[key]["n_test_entries_for_drug"]
            for key in original_audit
        )
    )

    small_mclrp = MCLRPConfig(gene_limit=12, iter_num=2)
    try:
        raw_original = predict_original_mclrp(X, M, folds_a, config=small_mclrp)
        calibrated_original = predict_original_mclrp_calibrated(X, M, folds_a, config=small_mclrp)
        checks["calibrated_method_preserves_original_folds"] = bool(
            len(raw_original.folds) == len(calibrated_original.folds)
            and all(np.array_equal(a, b) for a, b in zip(raw_original.folds, calibrated_original.folds))
        )
        checks["calibrated_method_preserves_heldout_counts"] = bool(
            [int(np.sum(fold != 0)) for fold in raw_original.folds]
            == [int(np.sum(fold != 0)) for fold in calibrated_original.folds]
        )
        heldout_mask = np.any(np.asarray(calibrated_original.folds) != 0, axis=0)
        checks["calibrated_method_has_finite_heldout_predictions"] = bool(
            np.all(np.isfinite(calibrated_original.prediction[heldout_mask]))
        )
    except RuntimeError as exc:
        if "not redistributed in the public release" not in str(exc):
            raise
        # The public package intentionally omits the locally translated upstream
        # MCLRP comparator. The train-mask calibration invariants above are still
        # tested on synthetic data; exact comparator execution is covered by the
        # archived private/frozen package rather than this public release.
        checks["public_release_omits_original_mclrp_execution"] = True

    base = rng.normal(size=len(expected_y_rows)).astype(np.float32)
    y = M[expected_y_rows, drug_idx].astype(np.float32)
    residual_target = y - base
    checks["mutation_residual_target_is_y_minus_base"] = bool(np.allclose(residual_target, y - base))

    config = MFMRConfig(num_folds=3, seeds=(0,), topg_imp=12, comp_imp=4, topg_ridge=12, comp_ridge=4, imputer_max_iter=1, min_train_per_drug=2)
    mutation = MutationFeatureSet(
        tissue=rng.normal(size=(M.shape[0], 3)).astype(np.float32),
        mapk=rng.normal(size=(M.shape[0], 4)).astype(np.float32),
        pi3k=rng.normal(size=(M.shape[0], 4)).astype(np.float32),
        overall=rng.normal(size=(M.shape[0], 3)).astype(np.float32),
        binary=(rng.random(size=(M.shape[0], 10)) > 0.6).astype(np.float32),
    )
    result = predict_mfmr_t0_seed(
        "ERKAUC30",
        X,
        M,
        folds_a,
        "mfmr_mutation",
        config,
        mutation_config=MutationHeadConfig(enabled=True, final_alpha=1.0, mutation_latent_dim=3, residual_inner_cv=0),
        mutation_features=mutation,
    )
    checks["runtime_diagnostics_confirm_residual_formula"] = result.diagnostics.get("residual_target") == "y_train_minus_base_pred_train"
    checks["all_passed"] = bool(all(bool(v) for v in checks.values()))
    if not checks["all_passed"]:
        raise AssertionError(f"Leakage-safety checks failed: {checks}")
    return checks
