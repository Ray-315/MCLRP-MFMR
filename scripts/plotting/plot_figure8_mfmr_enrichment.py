from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
import requests
from scipy.stats import t as student_t


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate the MCLRP project root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import GDSC_STANDARDIZED_DIR, LATEST_RESULTS_DIR, RESULTS_DIR  # noqa: E402
from scripts.main_figure.plot_style_bioinfo import set_bioinfo_style  # noqa: E402


ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
DEFAULT_LIBRARIES = {
    "go_mf": "GO_Molecular_Function_2026",
    "kegg": "KEGG_2021_Human",
}
SUPPORTED_DATASETS = ("ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50")
CACHE_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create MCLRP-Fig.8-style GO-MF and KEGG Sankey panels from "
            "MCLRP-MFMR prediction-derived drug-gene associations."
        )
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="PI3K_AUC")
    parser.add_argument("--variant", default="A5_MFMR_Full")
    parser.add_argument("--prediction-file", type=str, default=None)
    parser.add_argument("--bundle-file", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "paper_figures" / "fig8_mfmr_enrichment"),
    )
    parser.add_argument("--model-label", default="MCLRP-MFMR")
    parser.add_argument("--top-genes", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--min-drugs", type=int, default=2)
    parser.add_argument("--max-go-terms", type=int, default=16)
    parser.add_argument("--max-kegg-terms", type=int, default=22)
    parser.add_argument("--main-go-terms", type=int, default=10)
    parser.add_argument("--main-kegg-terms", type=int, default=12)
    parser.add_argument("--go-library", default=DEFAULT_LIBRARIES["go_mf"])
    parser.add_argument("--kegg-library", default=DEFAULT_LIBRARIES["kegg"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use exact-hash cache entries only; do not call Enrichr.")
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gene_list_hash(genes: Iterable[str]) -> str:
    payload = "\n".join(str(gene).strip() for gene in genes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "unnamed"


def default_prediction_path(dataset: str, variant: str) -> Path:
    return (
        LATEST_RESULTS_DIR
        / "ablation_mclrp_mfmr"
        / "shared_main"
        / dataset
        / "predictions"
        / f"{variant}_mean_prediction.npz"
    )


def default_bundle_path(dataset: str) -> Path:
    return GDSC_STANDARDIZED_DIR / f"{dataset}_bundle.npz"


def load_model_inputs(
    dataset: str,
    variant: str,
    prediction_file: str | None,
    bundle_file: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, Path]:
    prediction_path = Path(prediction_file) if prediction_file else default_prediction_path(dataset, variant)
    bundle_path = Path(bundle_file) if bundle_file else default_bundle_path(dataset)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {bundle_path}")

    prediction_payload = np.load(prediction_path, allow_pickle=True)
    if "prediction" not in prediction_payload.files:
        raise KeyError(f"{prediction_path} does not contain a 'prediction' array")
    prediction = prediction_payload["prediction"].astype(np.float32)

    bundle = np.load(bundle_path, allow_pickle=True)
    required = {"X", "M", "gene_symbols", "drug_labels"}
    missing = sorted(required.difference(bundle.files))
    if missing:
        raise KeyError(f"{bundle_path} is missing required arrays: {missing}")
    expression = bundle["X"].astype(np.float32)
    observed = bundle["M"].astype(np.float32)
    gene_symbols = bundle["gene_symbols"].astype(object)
    drug_labels = bundle["drug_labels"].astype(object)

    if prediction.shape != observed.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match response shape {observed.shape}")
    if expression.shape[0] != observed.shape[0]:
        raise ValueError(f"Expression rows {expression.shape[0]} do not match response rows {observed.shape[0]}")
    if expression.shape[1] != len(gene_symbols):
        raise ValueError(f"Expression columns {expression.shape[1]} do not match gene symbols {len(gene_symbols)}")
    if observed.shape[1] != len(drug_labels):
        raise ValueError(f"Response columns {observed.shape[1]} do not match drug labels {len(drug_labels)}")
    if not np.isfinite(expression).all():
        raise ValueError("Expression matrix contains non-finite values; explicit preprocessing is required")
    if not np.isfinite(prediction[observed != 0]).all():
        raise ValueError("Prediction matrix contains non-finite values at evaluated response entries")

    return expression, observed, prediction, gene_symbols, drug_labels, prediction_path, bundle_path


def prediction_correlation_vector(
    expression: np.ndarray,
    prediction: np.ndarray,
    *,
    block_size: int = 2048,
) -> np.ndarray:
    """Pearson correlation of every expression column with one prediction vector.

    Columns are processed in bounded blocks.  The GDSC expression matrix has more
    than 40,000 columns, and centering/casting the complete matrix at once can make
    NumPy allocate several temporary copies on some BLAS builds.
    """
    if expression.ndim != 2 or prediction.ndim != 1 or expression.shape[0] != prediction.shape[0]:
        raise ValueError("Correlation inputs have incompatible shapes")
    if expression.shape[0] < 3:
        return np.full(expression.shape[1], np.nan, dtype=np.float64)
    if block_size < 1:
        raise ValueError("block_size must be positive")

    y = np.asarray(prediction, dtype=np.float64)
    y_centered = y - y.mean(dtype=np.float64)
    y_ss = float(np.dot(y_centered, y_centered))
    corr = np.full(expression.shape[1], np.nan, dtype=np.float64)
    for start in range(0, expression.shape[1], block_size):
        stop = min(start + block_size, expression.shape[1])
        block = np.array(expression[:, start:stop], dtype=np.float64, copy=True, order="C")
        block -= block.mean(axis=0, dtype=np.float64)
        numerator = np.asarray(y_centered @ block, dtype=np.float64)
        x_ss = np.einsum("ij,ij->j", block, block, optimize=False)
        denominator = np.sqrt(np.maximum(x_ss, 0.0) * max(y_ss, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr[start:stop] = numerator / denominator
    corr = np.clip(corr, -1.0, 1.0)
    corr[~np.isfinite(corr)] = np.nan
    return corr


def correlation_p_values(correlations: np.ndarray, n_samples: int) -> np.ndarray:
    correlations = np.asarray(correlations, dtype=np.float64)
    if n_samples < 3:
        return np.full(correlations.shape, np.nan, dtype=np.float64)
    degrees_freedom = n_samples - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = correlations * np.sqrt(degrees_freedom / np.maximum(1e-15, 1.0 - correlations**2))
    values = 2.0 * student_t.sf(np.abs(statistic), degrees_freedom)
    values[~np.isfinite(values)] = np.nan
    return values


def rank_prediction_associated_genes(
    expression: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    gene_symbols: np.ndarray,
    drug_labels: np.ndarray,
    *,
    top_n: int,
    min_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank genes using prediction-only correlations; observed values define coverage only."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    normalized_symbols = [str(value).strip() for value in gene_symbols]

    for drug_idx, drug_value in enumerate(drug_labels):
        drug_label = str(drug_value)
        valid_rows = (observed[:, drug_idx] != 0) & np.isfinite(prediction[:, drug_idx])
        n_samples = int(valid_rows.sum())
        coverage_rows.append(
            {
                "drug_idx": int(drug_idx),
                "drug_label": drug_label,
                "n_prediction_pairs": n_samples,
                "eligible": bool(n_samples >= min_samples),
            }
        )
        if n_samples < min_samples:
            print(f"skip {drug_label}: only {n_samples} prediction pairs", flush=True)
            continue

        correlations = prediction_correlation_vector(
            expression[valid_rows],
            prediction[valid_rows, drug_idx],
        )
        order = np.argsort(-np.nan_to_num(np.abs(correlations), nan=-np.inf), kind="mergesort")
        selected_indices: list[int] = []
        seen_symbols: set[str] = set()
        for gene_idx in order:
            gene_symbol = normalized_symbols[int(gene_idx)]
            canonical = gene_symbol.upper()
            if not gene_symbol or canonical in {"NAN", "NONE"} or canonical in seen_symbols:
                continue
            if not np.isfinite(correlations[int(gene_idx)]):
                continue
            seen_symbols.add(canonical)
            selected_indices.append(int(gene_idx))
            if len(selected_indices) >= top_n:
                break
        if len(selected_indices) < top_n:
            raise ValueError(f"{drug_label} yielded only {len(selected_indices)} finite unique genes")

        selected_corr = correlations[selected_indices]
        selected_p = correlation_p_values(selected_corr, n_samples)
        for rank, (gene_idx, corr, p_value) in enumerate(
            zip(selected_indices, selected_corr, selected_p),
            start=1,
        ):
            rows.append(
                {
                    "drug_idx": int(drug_idx),
                    "drug_label": drug_label,
                    "gene_idx": int(gene_idx),
                    "gene_symbol": normalized_symbols[gene_idx],
                    "r_predicted": float(corr),
                    "abs_r_predicted": float(abs(corr)),
                    "p_predicted": float(p_value),
                    "direction": "positive" if corr >= 0 else "negative",
                    "rank_within_drug": int(rank),
                    "n_prediction_pairs": n_samples,
                }
            )
        print(f"ranked {drug_label}: {n_samples} pairs, top {len(selected_indices)} genes", flush=True)

    ranking = pd.DataFrame.from_records(rows)
    coverage = pd.DataFrame.from_records(coverage_rows)
    if ranking.empty:
        raise ValueError("No drugs had enough prediction pairs to build gene rankings")
    return ranking, coverage


def normalize_enrichr_term(raw_term: str) -> tuple[str, str]:
    text = " ".join(str(raw_term).split())
    go_match = re.match(r"^(.*?)\s*\((GO:\d+)\)$", text, flags=re.IGNORECASE)
    if go_match:
        return go_match.group(2).upper(), go_match.group(1).strip()
    kegg_match = re.match(r"^(hsa\d+)[:~]\s*(.*)$", text, flags=re.IGNORECASE)
    if kegg_match:
        return kegg_match.group(1).lower(), kegg_match.group(2).strip()
    return "", text


def _request_json(method: str, url: str, *, timeout: float, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    headers = {"User-Agent": "MCLRP-MFMR-Figure8/1.0 (functional-enrichment)"}
    for attempt in range(4):
        try:
            response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def query_enrichr(genes: list[str], library: str, description: str, *, timeout: float) -> list[dict[str, Any]]:
    add_payload = _request_json(
        "POST",
        f"{ENRICHR_BASE}/addList",
        timeout=timeout,
        files={"list": (None, "\n".join(genes)), "description": (None, description)},
    )
    user_list_id = int(add_payload["userListId"])
    enrichment_payload = _request_json(
        "GET",
        f"{ENRICHR_BASE}/enrich",
        timeout=timeout,
        params={"userListId": user_list_id, "backgroundType": library},
    )

    rows: list[dict[str, Any]] = []
    for record in enrichment_payload.get(library, []):
        term_id, term_name = normalize_enrichr_term(str(record[1]))
        overlap_genes = record[5] if isinstance(record[5], list) else re.split(r"[,;]", str(record[5]))
        overlap_genes = [str(gene).strip() for gene in overlap_genes if str(gene).strip()]
        adjusted = float(record[6])
        neglog = -math.log10(max(adjusted, np.nextafter(0.0, 1.0)))
        rows.append(
            {
                "term_rank": int(record[0]),
                "term_id": term_id,
                "term_name": term_name,
                "p_value": float(record[2]),
                "legacy_score": float(record[3]),
                "combined_score": float(record[4]),
                "overlap_genes": ";".join(overlap_genes),
                "overlap_count": int(len(overlap_genes)),
                "adjusted_p_value": adjusted,
                "neglog10_fdr": float(neglog),
                "neglog10_fdr_capped": float(min(neglog, 50.0)),
            }
        )
    return rows


def cache_path_for(
    cache_dir: Path,
    *,
    drug_idx: int,
    drug_label: str,
    database_key: str,
    library: str,
    genes: list[str],
) -> Path:
    digest = gene_list_hash(genes)
    return (
        cache_dir
        / database_key
        / f"{drug_idx:02d}_{slugify(drug_label)}__{slugify(library)}__n{len(genes)}__{digest[:16]}.json"
    )


def enrich_one_gene_set(
    *,
    drug_idx: int,
    drug_label: str,
    database_key: str,
    library: str,
    genes: list[str],
    cache_dir: Path,
    timeout: float,
    refresh_cache: bool,
    offline: bool,
) -> tuple[pd.DataFrame, str, Path]:
    cache_path = cache_path_for(
        cache_dir,
        drug_idx=drug_idx,
        drug_label=drug_label,
        database_key=database_key,
        library=library,
        genes=genes,
    )
    digest = gene_list_hash(genes)
    source = "cache"
    payload: dict[str, Any]
    if cache_path.exists() and not refresh_cache:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "library": library,
            "gene_hash": digest,
            "gene_count": len(genes),
            "drug_label": drug_label,
        }
        actual = {key: payload.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Invalid enrichment cache metadata in {cache_path}: {actual} != {expected}")
    else:
        if offline:
            raise FileNotFoundError(f"Offline mode: exact enrichment cache is missing: {cache_path}")
        records = query_enrichr(
            genes,
            library,
            f"MCLRP-MFMR-{drug_label}-{database_key}-{digest[:12]}",
            timeout=timeout,
        )
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at": now_iso(),
            "enrichment_service": ENRICHR_BASE,
            "database_key": database_key,
            "library": library,
            "drug_idx": int(drug_idx),
            "drug_label": drug_label,
            "gene_count": len(genes),
            "gene_hash": digest,
            "records": records,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        source = "api"

    frame = pd.DataFrame.from_records(payload.get("records", []))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "term_rank",
                "term_id",
                "term_name",
                "p_value",
                "legacy_score",
                "combined_score",
                "overlap_genes",
                "overlap_count",
                "adjusted_p_value",
                "neglog10_fdr",
                "neglog10_fdr_capped",
            ]
        )
    frame.insert(0, "drug_label", drug_label)
    frame.insert(0, "drug_idx", int(drug_idx))
    frame.insert(0, "library", library)
    frame.insert(0, "database_key", database_key)
    frame["input_gene_count"] = int(len(genes))
    frame["input_gene_hash"] = digest
    frame["cache_file"] = str(cache_path)
    return frame, source, cache_path


def run_enrichment(
    ranking: pd.DataFrame,
    *,
    libraries: dict[str, str],
    cache_dir: Path,
    workers: int,
    timeout: float,
    refresh_cache: bool,
    offline: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    for (drug_idx, drug_label), group in ranking.groupby(["drug_idx", "drug_label"], sort=True):
        genes = group.sort_values("rank_within_drug")["gene_symbol"].astype(str).tolist()
        for database_key, library in libraries.items():
            tasks.append(
                {
                    "drug_idx": int(drug_idx),
                    "drug_label": str(drug_label),
                    "database_key": database_key,
                    "library": library,
                    "genes": genes,
                    "cache_dir": cache_dir,
                    "timeout": timeout,
                    "refresh_cache": refresh_cache,
                    "offline": offline,
                }
            )

    frames: list[pd.DataFrame] = []
    cache_audit: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_map = {executor.submit(enrich_one_gene_set, **task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            frame, source, cache_path = future.result()
            frames.append(frame)
            cache_audit.append(
                {
                    "drug_idx": task["drug_idx"],
                    "drug_label": task["drug_label"],
                    "database_key": task["database_key"],
                    "library": task["library"],
                    "source": source,
                    "cache_file": str(cache_path),
                    "result_rows": int(len(frame)),
                }
            )
            print(
                f"enriched {task['drug_label']} / {task['database_key']}: "
                f"{len(frame)} rows ({source})",
                flush=True,
            )

    if not frames:
        raise ValueError("No enrichment tasks were produced")
    enrichment = pd.concat(frames, ignore_index=True)
    enrichment.sort_values(["database_key", "drug_idx", "term_rank"], inplace=True)
    return enrichment, sorted(cache_audit, key=lambda row: (row["database_key"], row["drug_idx"]))


def select_shared_terms(
    enrichment: pd.DataFrame,
    *,
    fdr_threshold: float,
    min_drugs: int,
    max_terms: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "drug_idx",
        "drug_label",
        "term_id",
        "term_name",
        "overlap_count",
        "adjusted_p_value",
        "neglog10_fdr_capped",
    }
    missing = required.difference(enrichment.columns)
    if missing:
        raise KeyError(f"Enrichment table is missing columns: {sorted(missing)}")
    if not 0.0 < fdr_threshold <= 1.0:
        raise ValueError("fdr_threshold must be in (0, 1]")
    if min_drugs < 1:
        raise ValueError("min_drugs must be positive")

    significant = enrichment.loc[
        np.isfinite(pd.to_numeric(enrichment["adjusted_p_value"], errors="coerce"))
        & (pd.to_numeric(enrichment["adjusted_p_value"], errors="coerce") < fdr_threshold)
    ].copy()
    significant["term_key"] = np.where(
        significant["term_id"].fillna("").astype(str).str.len() > 0,
        significant["term_id"].astype(str),
        significant["term_name"].astype(str).str.casefold(),
    )
    significant.sort_values("adjusted_p_value", inplace=True)
    significant.drop_duplicates(["drug_label", "term_key"], keep="first", inplace=True)
    if significant.empty:
        return significant, pd.DataFrame()

    summary = (
        significant.groupby(["term_key", "term_id", "term_name"], dropna=False, as_index=False)
        .agg(
            drug_count=("drug_label", "nunique"),
            best_fdr=("adjusted_p_value", "min"),
            median_fdr=("adjusted_p_value", "median"),
            total_neglog10_fdr=("neglog10_fdr_capped", "sum"),
            total_overlap_genes=("overlap_count", "sum"),
        )
    )
    summary = summary.loc[summary["drug_count"] >= min_drugs].copy()
    summary.sort_values(
        ["drug_count", "total_neglog10_fdr", "best_fdr", "term_name"],
        ascending=[False, False, True, True],
        inplace=True,
    )
    if max_terms > 0:
        summary = summary.head(max_terms).copy()
    summary["term_order"] = np.arange(len(summary), dtype=int)
    summary["term_base_label"] = np.where(
        summary["term_id"].fillna("").astype(str).str.len() > 0,
        summary["term_id"].astype(str) + "~" + summary["term_name"].astype(str),
        summary["term_name"].astype(str),
    )
    summary["term_label"] = (
        summary["term_base_label"].astype(str)
        + " ("
        + summary["drug_count"].astype(int).astype(str)
        + ")"
    )

    selected = significant.loc[significant["term_key"].isin(summary["term_key"])].copy()
    selected = selected.merge(
        summary[
            [
                "term_key",
                "drug_count",
                "best_fdr",
                "median_fdr",
                "total_neglog10_fdr",
                "term_order",
                "term_base_label",
                "term_label",
            ]
        ],
        on="term_key",
        how="inner",
    )
    selected.sort_values(["drug_idx", "term_order"], inplace=True)
    selected.reset_index(drop=True, inplace=True)
    return selected, summary.reset_index(drop=True)


def annotate_term_direction(edges: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    """Annotate each enriched drug-term edge with the direction of its overlap genes.

    Enrichr over-representation is unsigned.  Direction is added only after enrichment by
    mapping the genes reported in ``overlap_genes`` back to their drug-specific Pearson
    correlations with the out-of-fold prediction.  The mean correlation is used for color;
    the median and sign balance are retained in the exported edge tables for auditability.
    """
    required_edges = {"drug_label", "overlap_genes"}
    required_ranking = {"drug_label", "gene_symbol", "r_predicted"}
    missing_edges = required_edges.difference(edges.columns)
    missing_ranking = required_ranking.difference(ranking.columns)
    if missing_edges:
        raise KeyError(f"Enrichment edges are missing columns: {sorted(missing_edges)}")
    if missing_ranking:
        raise KeyError(f"Gene ranking is missing columns: {sorted(missing_ranking)}")

    correlation_lookup = {
        (str(row.drug_label), str(row.gene_symbol).strip().upper()): float(row.r_predicted)
        for row in ranking.itertuples(index=False)
        if str(row.gene_symbol).strip() and np.isfinite(float(row.r_predicted))
    }
    annotations: list[dict[str, Any]] = []
    for row in edges.itertuples(index=False):
        overlap_symbols = [
            symbol.strip().upper()
            for symbol in str(row.overlap_genes).split(";")
            if symbol.strip()
        ]
        correlations = np.asarray(
            [
                correlation_lookup[(str(row.drug_label), symbol)]
                for symbol in overlap_symbols
                if (str(row.drug_label), symbol) in correlation_lookup
            ],
            dtype=np.float64,
        )
        if correlations.size:
            mean_r = float(np.mean(correlations))
            median_r = float(np.median(correlations))
            positive_fraction = float(np.mean(correlations > 0))
            negative_fraction = float(np.mean(correlations < 0))
            direction_label = "higher-prediction-associated" if mean_r > 0 else "lower-prediction-associated"
        else:
            mean_r = median_r = positive_fraction = negative_fraction = float("nan")
            direction_label = "unavailable"
        annotations.append(
            {
                "direction_gene_count": int(correlations.size),
                "mean_overlap_r": mean_r,
                "median_overlap_r": median_r,
                "positive_gene_fraction": positive_fraction,
                "negative_gene_fraction": negative_fraction,
                "direction_label": direction_label,
            }
        )
    annotated = pd.concat([edges.reset_index(drop=True), pd.DataFrame.from_records(annotations)], axis=1)
    if annotated["direction_gene_count"].eq(0).any():
        missing_count = int(annotated["direction_gene_count"].eq(0).sum())
        raise ValueError(f"Could not map overlap genes back to ranked correlations for {missing_count} edges")
    return annotated


def _combined_palette(count: int) -> list[str]:
    if count <= 0:
        return []
    colors: list[str] = []
    for name in ("tab20", "tab20b", "tab20c", "Set3"):
        cmap = plt.get_cmap(name)
        if hasattr(cmap, "colors"):
            colors.extend(to_hex(color) for color in cmap.colors)
        else:
            colors.extend(to_hex(cmap(index / max(count - 1, 1))) for index in range(count))
        if len(colors) >= count:
            break
    return colors[:count]


def _layout_nodes(
    labels: list[str],
    counts: dict[str, int],
    *,
    link_height: float,
    gap: float,
) -> dict[str, tuple[float, float]]:
    total_height = sum(counts[label] * link_height for label in labels) + gap * max(len(labels) - 1, 0)
    cursor = 0.5 + total_height / 2.0
    bounds: dict[str, tuple[float, float]] = {}
    for label in labels:
        height = counts[label] * link_height
        upper = cursor
        lower = upper - height
        bounds[label] = (lower, upper)
        cursor = lower - gap
    return bounds


def _spread_label_positions(
    anchors: dict[str, float],
    *,
    min_gap: float,
    low: float = 0.025,
    high: float = 0.975,
) -> dict[str, float]:
    """Separate direct-label baselines while preserving their vertical order."""
    if not anchors:
        return {}
    ordered = sorted(anchors.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    targets = np.asarray([item[1] for item in ordered], dtype=np.float64)
    if len(targets) == 1:
        return {labels[0]: float(np.clip(targets[0], low, high))}

    gap = min(float(min_gap), (high - low) / (len(targets) - 1))
    positions = targets.copy()
    positions[0] = max(positions[0], low)
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    positions -= max(0.0, positions[-1] - high)
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - gap)
    positions += max(0.0, low - positions[0])
    if positions[-1] > high + 1e-12:
        positions = np.linspace(low, high, len(positions), dtype=np.float64)
    return {label: float(position) for label, position in zip(labels, positions)}


def _ribbon_patch(
    x0: float,
    y0_low: float,
    y0_high: float,
    x1: float,
    y1_low: float,
    y1_high: float,
    *,
    color: str,
    alpha: float,
) -> PathPatch:
    dx = x1 - x0
    vertices = [
        (x0, y0_low),
        (x0 + 0.36 * dx, y0_low),
        (x1 - 0.36 * dx, y1_low),
        (x1, y1_low),
        (x1, y1_high),
        (x1 - 0.36 * dx, y1_high),
        (x0 + 0.36 * dx, y0_high),
        (x0, y0_high),
        (x0, y0_low),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(
        MplPath(vertices, codes),
        facecolor=color,
        edgecolor=color,
        linewidth=0.18,
        alpha=alpha,
        antialiased=True,
    )


def _wrap_label(value: str, *, width: int, max_lines: int) -> str:
    chunks = textwrap.wrap(
        " ".join(str(value).split()),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(chunks) <= max_lines:
        return "\n".join(chunks)
    kept = chunks[:max_lines]
    kept[-1] = kept[-1].rstrip(" .") + "…"
    return "\n".join(kept)


def draw_sankey_panel(
    ax: plt.Axes,
    edges: pd.DataFrame,
    term_summary: pd.DataFrame,
    *,
    panel_label: str,
    title: str,
    subtitle: str,
) -> None:
    ax.set_axis_off()
    if edges.empty or term_summary.empty:
        ax.text(0.5, 0.5, "No shared significant terms", ha="center", va="center", fontsize=12)
        ax.text(-0.015, 1.015, panel_label, transform=ax.transAxes, fontsize=16, fontfamily="DejaVu Serif")
        ax.set_title(title, fontsize=12, pad=10)
        return

    edges = edges.copy().reset_index(drop=True)
    edges["edge_id"] = np.arange(len(edges), dtype=int)
    term_order = term_summary.sort_values("term_order")["term_key"].astype(str).tolist()
    term_labels = term_summary.set_index("term_key")["term_label"].astype(str).to_dict()
    drugs = (
        edges[["drug_idx", "drug_label"]]
        .drop_duplicates()
        .sort_values(["drug_idx", "drug_label"])["drug_label"]
        .astype(str)
        .tolist()
    )
    drug_counts = edges.groupby("drug_label")["edge_id"].count().astype(int).to_dict()
    term_counts = edges.groupby("term_key")["edge_id"].count().astype(int).to_dict()
    link_count = int(len(edges))
    left_gap = min(0.010, 0.18 / max(len(drugs) - 1, 1))
    right_gap = min(0.014, 0.18 / max(len(term_order) - 1, 1))
    available_left = 0.90 - left_gap * max(len(drugs) - 1, 0)
    available_right = 0.90 - right_gap * max(len(term_order) - 1, 0)
    link_height = min(available_left, available_right) / max(link_count, 1)
    drug_bounds = _layout_nodes(drugs, drug_counts, link_height=link_height, gap=left_gap)
    term_bounds = _layout_nodes(term_order, term_counts, link_height=link_height, gap=right_gap)

    drug_rank = {label: index for index, label in enumerate(drugs)}
    term_rank = {label: index for index, label in enumerate(term_order)}
    source_slots: dict[int, tuple[float, float]] = {}
    target_slots: dict[int, tuple[float, float]] = {}
    for drug in drugs:
        cursor = drug_bounds[drug][0]
        subset = edges.loc[edges["drug_label"].eq(drug)].sort_values("term_key", key=lambda s: s.map(term_rank))
        for edge_id in subset["edge_id"].astype(int):
            source_slots[edge_id] = (cursor, cursor + link_height)
            cursor += link_height
    for term in term_order:
        cursor = term_bounds[term][0]
        subset = edges.loc[edges["term_key"].eq(term)].sort_values("drug_label", key=lambda s: s.map(drug_rank))
        for edge_id in subset["edge_id"].astype(int):
            target_slots[edge_id] = (cursor, cursor + link_height)
            cursor += link_height

    drug_colors = {label: color for label, color in zip(drugs, _combined_palette(len(drugs)))}
    term_colors = {label: color for label, color in zip(term_order, _combined_palette(len(term_order)))}
    x_left0, x_left1 = 0.125, 0.152
    x_right0, x_right1 = 0.705, 0.732
    for row in edges.itertuples(index=False):
        y0_low, y0_high = source_slots[int(row.edge_id)]
        y1_low, y1_high = target_slots[int(row.edge_id)]
        ax.add_patch(
            _ribbon_patch(
                x_left1,
                y0_low,
                y0_high,
                x_right0,
                y1_low,
                y1_high,
                color=drug_colors[str(row.drug_label)],
                alpha=0.62,
            )
        )

    drug_anchors = {drug: float(np.mean(drug_bounds[drug])) for drug in drugs}
    drug_label_y = _spread_label_positions(drug_anchors, min_gap=0.024)
    for drug in drugs:
        lower, upper = drug_bounds[drug]
        ax.add_patch(
            Rectangle(
                (x_left0, lower),
                x_left1 - x_left0,
                upper - lower,
                facecolor=drug_colors[drug],
                edgecolor="white",
                linewidth=0.45,
                zorder=10,
            )
        )
        anchor_y = drug_anchors[drug]
        text_y = drug_label_y[drug]
        if abs(text_y - anchor_y) > 0.003:
            ax.plot(
                [x_left0, x_left0 - 0.006],
                [anchor_y, text_y],
                color="#94A3B8",
                linewidth=0.38,
                solid_capstyle="round",
                zorder=11,
            )
        ax.text(
            x_left0 - 0.008,
            text_y,
            _wrap_label(drug, width=18, max_lines=2),
            ha="right",
            va="center",
            fontsize=8.1,
            linespacing=0.98,
        )

    term_anchors = {term: float(np.mean(term_bounds[term])) for term in term_order}
    term_label_y = _spread_label_positions(term_anchors, min_gap=0.034)
    for term in term_order:
        lower, upper = term_bounds[term]
        ax.add_patch(
            Rectangle(
                (x_right0, lower),
                x_right1 - x_right0,
                upper - lower,
                facecolor=term_colors[term],
                edgecolor="white",
                linewidth=0.45,
                zorder=10,
            )
        )
        anchor_y = term_anchors[term]
        text_y = term_label_y[term]
        if abs(text_y - anchor_y) > 0.003:
            ax.plot(
                [x_right1, x_right1 + 0.006],
                [anchor_y, text_y],
                color="#94A3B8",
                linewidth=0.38,
                solid_capstyle="round",
                zorder=11,
            )
        ax.text(
            x_right1 + 0.008,
            text_y,
            _wrap_label(term_labels[term], width=54, max_lines=2),
            ha="left",
            va="center",
            fontsize=7.8,
            linespacing=0.96,
        )

    ax.text(0.5, 1.028, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12.2, weight="bold")
    ax.text(0.5, 1.004, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.7, color="#475569")
    ax.text(-0.015, 1.015, panel_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=16, fontfamily="DejaVu Serif")
    ax.text(
        0.50,
        -0.024,
        "Each ribbon denotes one significant drug-term association; parentheses give the number of drugs.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.8,
        color="#64748B",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


ENRICHMENT_DIRECTION_CMAP = LinearSegmentedColormap.from_list(
    "prediction_direction",
    ["#2F6BFF", "#F8FAFC", "#E67E22"],
    N=256,
)
ENRICHMENT_DIRECTION_NORM = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.50)


def enrichment_bubble_size(neglog10_fdr: np.ndarray | float) -> np.ndarray:
    values = np.asarray(neglog10_fdr, dtype=np.float64)
    scaled = np.clip(values, 0.0, 10.0) / 10.0
    return 18.0 + 112.0 * np.power(scaled, 0.78)


def draw_directional_enrichment_matrix(
    ax: plt.Axes,
    count_ax: plt.Axes,
    edges: pd.DataFrame,
    term_summary: pd.DataFrame,
    *,
    drug_order: list[str],
    panel_label: str,
    title: str,
    show_drug_labels: bool,
) -> Any:
    """Draw an enrichment bubble matrix with an adjacent recurrence bar chart."""
    if edges.empty or term_summary.empty:
        ax.text(0.5, 0.5, "No shared significant terms", ha="center", va="center", fontsize=11)
        count_ax.set_axis_off()
        return None

    terms = term_summary.sort_values("term_order").reset_index(drop=True)
    term_order = terms["term_key"].astype(str).tolist()
    term_index = {term: idx for idx, term in enumerate(term_order)}
    drug_index = {drug: idx for idx, drug in enumerate(drug_order)}
    plotted = edges.loc[
        edges["term_key"].astype(str).isin(term_index)
        & edges["drug_label"].astype(str).isin(drug_index)
    ].copy()
    plotted["x"] = plotted["drug_label"].astype(str).map(drug_index)
    plotted["y"] = plotted["term_key"].astype(str).map(term_index)

    for row_idx in range(len(terms)):
        if row_idx % 2 == 0:
            ax.axhspan(row_idx - 0.5, row_idx + 0.5, color="#F8FAFC", zorder=0)
            count_ax.axhspan(row_idx - 0.5, row_idx + 0.5, color="#F8FAFC", zorder=0)

    collection = ax.scatter(
        plotted["x"].to_numpy(dtype=float),
        plotted["y"].to_numpy(dtype=float),
        s=enrichment_bubble_size(plotted["neglog10_fdr_capped"].to_numpy(dtype=float)),
        c=plotted["mean_overlap_r"].to_numpy(dtype=float),
        cmap=ENRICHMENT_DIRECTION_CMAP,
        norm=ENRICHMENT_DIRECTION_NORM,
        edgecolors="#334155",
        linewidths=0.45,
        alpha=0.94,
        zorder=3,
    )

    y_labels = [
        _wrap_label(f"{row.term_name}  [{int(row.drug_count)}]", width=43, max_lines=2)
        for row in terms.itertuples(index=False)
    ]
    ax.set_xlim(-0.6, len(drug_order) - 0.4)
    ax.set_ylim(len(terms) - 0.45, -0.55)
    ax.set_xticks(np.arange(len(drug_order)))
    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(y_labels, fontsize=8.2)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0, pad=3)
    if show_drug_labels:
        ax.set_xticklabels(drug_order, rotation=58, ha="right", va="top", fontsize=7.6)
    else:
        ax.set_xticklabels([])
    ax.set_xticks(np.arange(-0.5, len(drug_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(terms), 1), minor=True)
    ax.grid(which="minor", color="#E2E8F0", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11.5, weight="bold", pad=8)
    ax.text(
        -0.095,
        1.015,
        panel_label,
        transform=ax.transAxes,
        fontsize=15,
        fontfamily="DejaVu Serif",
        weight="bold",
        va="bottom",
    )

    recurrence = terms["drug_count"].to_numpy(dtype=float)
    y = np.arange(len(terms))
    count_ax.barh(y, recurrence, height=0.56, color="#64748B", edgecolor="#334155", linewidth=0.4, zorder=2)
    for y_value, count in zip(y, recurrence):
        count_ax.text(count + 0.35, y_value, f"{int(count)}", va="center", ha="left", fontsize=7.4, color="#334155")
    count_ax.set_xlim(0, max(len(drug_order), float(np.max(recurrence)) + 3.0))
    count_ax.set_ylim(len(terms) - 0.45, -0.55)
    count_ax.set_yticks([])
    count_ax.set_xticks([0, 10, 20, len(drug_order)])
    count_ax.set_xticklabels(["0", "10", "20", str(len(drug_order))], fontsize=7.3)
    count_ax.tick_params(axis="x", length=2.5, color="#94A3B8", pad=2)
    count_ax.set_title("Drug count", fontsize=8.0, color="#475569", pad=8)
    count_ax.grid(axis="x", color="#E2E8F0", linewidth=0.45)
    for spine in count_ax.spines.values():
        spine.set_visible(False)
    return collection


def add_matrix_legends(fig: plt.Figure, collection: Any, *, bottom: float) -> None:
    color_ax = fig.add_axes([0.34, bottom, 0.30, 0.012])
    colorbar = fig.colorbar(collection, cax=color_ax, orientation="horizontal")
    colorbar.set_ticks([-0.5, 0.0, 0.5])
    colorbar.set_ticklabels(["lower predicted response", "mixed", "higher predicted response"])
    colorbar.ax.tick_params(labelsize=7.7, length=2.5, pad=2)
    colorbar.outline.set_edgecolor("#CBD5E1")
    colorbar.set_label("Mean Pearson r among overlap genes (directional annotation)", fontsize=8.0, labelpad=3)

    legend_ax = fig.add_axes([0.69, bottom - 0.006, 0.27, 0.04])
    legend_ax.set_axis_off()
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=math.sqrt(float(enrichment_bubble_size(value))) * 0.78,
            markerfacecolor="#CBD5E1",
            markeredgecolor="#334155",
            markeredgewidth=0.45,
            label=f"{value:g}",
        )
        for value in (2.0, 5.0, 10.0)
    ]
    legend_ax.legend(
        handles=handles,
        title="−log10(FDR)",
        loc="center left",
        frameon=False,
        ncol=3,
        fontsize=7.6,
        title_fontsize=7.8,
        handletextpad=0.4,
        columnspacing=0.9,
    )


def save_figure(fig: plt.Figure, prefix: Path, *, dpi: int) -> list[Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = prefix.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        saved.append(path)
    plt.close(fig)
    return saved


def render_outputs(
    go_edges: pd.DataFrame,
    go_summary: pd.DataFrame,
    kegg_edges: pd.DataFrame,
    kegg_summary: pd.DataFrame,
    *,
    drug_order: list[str],
    model_label: str,
    dataset: str,
    top_genes: int,
    fdr: float,
    min_drugs: int,
    main_go_terms: int,
    main_kegg_terms: int,
    output_dir: Path,
    dpi: int,
) -> dict[str, list[str]]:
    set_bioinfo_style()
    outputs: dict[str, list[str]] = {}
    dataset_display = dataset.replace("_", " ")
    main_go_summary = go_summary.head(max(main_go_terms, 1)).copy()
    main_kegg_summary = kegg_summary.head(max(main_kegg_terms, 1)).copy()
    main_go_edges = go_edges.loc[go_edges["term_key"].isin(main_go_summary["term_key"])].copy()
    main_kegg_edges = kegg_edges.loc[ kegg_edges["term_key"].isin(main_kegg_summary["term_key"])].copy()

    # Main panels: a matrix makes the drug-by-term pattern, within-query FDR,
    # recurrence, and post-enrichment direction visible at the same time.
    for database_key, edges, summary, panel_label, panel_title, file_stem in (
        ("go_mf", main_go_edges, main_go_summary, "A", "GO molecular function", "fig8A_go_mf_mfmr"),
        ("kegg", main_kegg_edges, main_kegg_summary, "B", "KEGG pathways", "fig8B_kegg_mfmr"),
    ):
        height = max(8.0, 0.34 * len(summary) + 4.0)
        fig_panel, (matrix_ax, count_ax) = plt.subplots(
            1,
            2,
            figsize=(16.8, height),
            gridspec_kw={"width_ratios": [12.0, 1.55], "wspace": 0.035},
        )
        fig_panel.subplots_adjust(left=0.24, right=0.985, top=0.88, bottom=0.21)
        collection = draw_directional_enrichment_matrix(
            matrix_ax,
            count_ax,
            edges,
            summary,
            drug_order=drug_order,
            panel_label=panel_label,
            title=panel_title,
            show_drug_labels=True,
        )
        fig_panel.suptitle(
            f"Prediction-associated functional enrichment in {dataset_display}",
            x=0.24,
            y=0.965,
            ha="left",
            fontsize=13.2,
            weight="bold",
        )
        fig_panel.text(
            0.24,
            0.925,
            f"{model_label}; top {top_genes:,} genes per drug; within-query FDR < {fdr:g}; terms recurring in ≥{min_drugs} drugs",
            ha="left",
            va="center",
            fontsize=8.6,
            color="#475569",
        )
        if collection is not None:
            add_matrix_legends(fig_panel, collection, bottom=0.055)
        outputs[f"{database_key}_matrix"] = [
            str(path) for path in save_figure(fig_panel, output_dir / file_stem, dpi=dpi)
        ]

    composite_height = max(15.5, 0.30 * (len(main_go_summary) + len(main_kegg_summary)) + 5.0)
    fig = plt.figure(figsize=(17.4, composite_height))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[12.0, 1.55],
        height_ratios=[max(len(main_go_summary), 1), max(len(main_kegg_summary), 1)],
        left=0.24,
        right=0.985,
        top=0.91,
        bottom=0.14,
        hspace=0.18,
        wspace=0.035,
    )
    go_ax = fig.add_subplot(grid[0, 0])
    go_count_ax = fig.add_subplot(grid[0, 1])
    kegg_ax = fig.add_subplot(grid[1, 0])
    kegg_count_ax = fig.add_subplot(grid[1, 1])
    go_collection = draw_directional_enrichment_matrix(
        go_ax,
        go_count_ax,
        main_go_edges,
        main_go_summary,
        drug_order=drug_order,
        panel_label="A",
        title="GO molecular function",
        show_drug_labels=False,
    )
    draw_directional_enrichment_matrix(
        kegg_ax,
        kegg_count_ax,
        main_kegg_edges,
        main_kegg_summary,
        drug_order=drug_order,
        panel_label="B",
        title="KEGG pathways",
        show_drug_labels=True,
    )
    fig.suptitle(
        f"Prediction-associated functional enrichment in {dataset_display}",
        x=0.24,
        y=0.975,
        ha="left",
        fontsize=14.0,
        weight="bold",
    )
    fig.text(
        0.24,
        0.945,
        f"{model_label}; top {top_genes:,} genes per drug; within-query FDR < {fdr:g}; terms recurring in ≥{min_drugs} drugs",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#475569",
    )
    if go_collection is not None:
        add_matrix_legends(fig, go_collection, bottom=0.048)
    outputs["composite"] = [
        str(path) for path in save_figure(fig, output_dir / "fig8_mfmr_biological_enrichment", dpi=dpi)
    ]

    # Supplementary Sankey version: retained for continuity with the reference MCLRP
    # article, but no longer used as the information-dense main figure.
    sankey_subtitle = (
        f"{model_label} prediction-derived | top {top_genes:,} genes/drug | "
        f"FDR < {fdr:g} | ≥{min_drugs} drugs"
    )
    go_height = max(8.6, 0.32 * max(go_edges["drug_label"].nunique(), len(go_summary)))
    fig_go, ax_go = plt.subplots(figsize=(14.0, go_height))
    fig_go.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.045)
    draw_sankey_panel(
        ax_go,
        go_edges,
        go_summary,
        panel_label="A",
        title="GO molecular function",
        subtitle=sankey_subtitle,
    )
    outputs["supplementary_go_sankey"] = [
        str(path) for path in save_figure(fig_go, output_dir / "figS10A_go_mf_sankey", dpi=dpi)
    ]

    kegg_height = max(8.6, 0.32 * max(kegg_edges["drug_label"].nunique(), len(kegg_summary)))
    fig_kegg, ax_kegg = plt.subplots(figsize=(14.0, kegg_height))
    fig_kegg.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.045)
    draw_sankey_panel(
        ax_kegg,
        kegg_edges,
        kegg_summary,
        panel_label="B",
        title="KEGG pathways",
        subtitle=sankey_subtitle,
    )
    outputs["supplementary_kegg_sankey"] = [
        str(path) for path in save_figure(fig_kegg, output_dir / "figS10B_kegg_sankey", dpi=dpi)
    ]
    return outputs


def write_readme(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    prediction_path: Path,
    bundle_path: Path,
    outputs: dict[str, list[str]],
    coverage: pd.DataFrame,
    go_summary: pd.DataFrame,
    kegg_summary: pd.DataFrame,
) -> Path:
    output_lines = [f"- `{path}`" for paths in outputs.values() for path in paths]
    readme = f"""# MCLRP-MFMR prediction-associated functional enrichment

## What this figure uses

- Prediction matrix: `{prediction_path}`
- Expression/response bundle: `{bundle_path}`
- Dataset: `{args.dataset}`
- Repository model variant: `{args.variant}`
- Display label: `{args.model_label}`
- Eligible drugs: {int(coverage['eligible'].sum())} / {len(coverage)}

The repository consistently names the new model **MCLRP-MFMR**. The user-facing name
"MCLRP-MRMF" is treated as referring to the same model; the manifest retains the exact
repository variant and prediction file so the data source is unambiguous.

## Analysis protocol

1. For each drug, use out-of-fold model predictions at response entries with available
   evaluation coverage (`M != 0`). Observed response magnitudes are **not** used in the
   gene score; they are used only to identify prediction coverage.
2. Compute Pearson correlation between each gene's expression and the predicted response.
3. Rank by absolute prediction correlation and retain the top {args.top_genes} unique gene
   symbols per drug, matching the paper's top-1000 design.
4. Query Enrichr `{args.go_library}` and `{args.kegg_library}`. Exact results are cached by
   database, drug, top-N, and SHA-256 hash of the submitted gene list.
5. Keep drug-term associations with adjusted p-value (FDR) < {args.fdr}; retain terms present
   in at least {args.min_drugs} drugs. The panels show up to {args.max_go_terms} GO-MF terms
   and {args.max_kegg_terms} KEGG terms.
6. Map each enriched term's overlap genes back to their drug-specific Pearson correlations.
   The mean overlap-gene correlation is a **post-enrichment directional annotation**; the
   Enrichr over-representation test itself remains unsigned.
7. The main bubble matrices encode within-query `-log10(FDR)` by area and the mean overlap-gene
   correlation by a blue-to-orange diverging scale. The adjacent bars show recurrence counts.
   For print readability, the main view shows the {args.main_go_terms} most recurrent GO-MF and
   {args.main_kegg_terms} most recurrent KEGG terms. Full selected tables and the original Sankey
   form are retained as supplementary outputs.

The analysis is association-based and supports biological interpretation; it does not
establish a causal mechanism. For AUC/IC50 response scales, a negative expression-prediction
correlation is aligned with lower predicted response, whereas a positive correlation is aligned
with higher predicted response. These labels do not convert enrichment into causal attribution.

## References

- MCLRP article and reference enrichment figure: https://pmc.ncbi.nlm.nih.gov/articles/PMC12781783/
- Enrichr enrichment service: https://maayanlab.cloud/Enrichr/

## Reproduce

```powershell
python plotting/scripts/plot_figure8_mfmr_enrichment.py --dataset {args.dataset} --variant {args.variant}
```

After the first successful run, add `--offline` to reproduce from exact-hash caches without
network access. Use `--refresh-cache` only when a fresh Enrichr retrieval is intentionally
required.

## Selected terms

- GO-MF: {len(go_summary)} terms; drug counts {go_summary['drug_count'].astype(int).tolist() if not go_summary.empty else []}
- KEGG: {len(kegg_summary)} terms; drug counts {kegg_summary['drug_count'].astype(int).tolist() if not kegg_summary.empty else []}

## Figure files

{chr(10).join(output_lines)}
"""
    path = output_dir / "README.md"
    path.write_text(readme, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    expression, observed, prediction, gene_symbols, drug_labels, prediction_path, bundle_path = load_model_inputs(
        args.dataset,
        args.variant,
        args.prediction_file,
        args.bundle_file,
    )
    ranking, coverage = rank_prediction_associated_genes(
        expression,
        observed,
        prediction,
        gene_symbols,
        drug_labels,
        top_n=args.top_genes,
        min_samples=args.min_samples,
    )
    ranking_path = table_dir / "fig8_prediction_top_genes_per_drug.csv"
    coverage_path = table_dir / "fig8_prediction_coverage.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    libraries = {"go_mf": args.go_library, "kegg": args.kegg_library}
    enrichment, cache_audit = run_enrichment(
        ranking,
        libraries=libraries,
        cache_dir=cache_dir,
        workers=args.workers,
        timeout=args.timeout,
        refresh_cache=args.refresh_cache,
        offline=args.offline,
    )
    enrichment_path = table_dir / "fig8_enrichment_all.csv"
    enrichment.to_csv(enrichment_path, index=False, encoding="utf-8-sig")

    go_edges, go_summary = select_shared_terms(
        enrichment.loc[enrichment["database_key"].eq("go_mf")].copy(),
        fdr_threshold=args.fdr,
        min_drugs=args.min_drugs,
        max_terms=args.max_go_terms,
    )
    kegg_edges, kegg_summary = select_shared_terms(
        enrichment.loc[enrichment["database_key"].eq("kegg")].copy(),
        fdr_threshold=args.fdr,
        min_drugs=args.min_drugs,
        max_terms=args.max_kegg_terms,
    )
    if go_edges.empty:
        raise RuntimeError("No GO-MF term passed the requested FDR and cross-drug criteria")
    if kegg_edges.empty:
        raise RuntimeError("No KEGG term passed the requested FDR and cross-drug criteria")

    go_edges = annotate_term_direction(go_edges, ranking)
    kegg_edges = annotate_term_direction(kegg_edges, ranking)

    go_edges_path = table_dir / "fig8_go_mf_shared_edges.csv"
    go_summary_path = table_dir / "fig8_go_mf_shared_terms.csv"
    kegg_edges_path = table_dir / "fig8_kegg_shared_edges.csv"
    kegg_summary_path = table_dir / "fig8_kegg_shared_terms.csv"
    go_edges.to_csv(go_edges_path, index=False, encoding="utf-8-sig")
    go_summary.to_csv(go_summary_path, index=False, encoding="utf-8-sig")
    kegg_edges.to_csv(kegg_edges_path, index=False, encoding="utf-8-sig")
    kegg_summary.to_csv(kegg_summary_path, index=False, encoding="utf-8-sig")

    outputs = render_outputs(
        go_edges,
        go_summary,
        kegg_edges,
        kegg_summary,
        drug_order=(
            coverage.loc[coverage["eligible"]]
            .sort_values("drug_idx")["drug_label"]
            .astype(str)
            .tolist()
        ),
        model_label=args.model_label,
        dataset=args.dataset,
        top_genes=args.top_genes,
        fdr=args.fdr,
        min_drugs=args.min_drugs,
        main_go_terms=args.main_go_terms,
        main_kegg_terms=args.main_kegg_terms,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    readme_path = write_readme(
        output_dir,
        args=args,
        prediction_path=prediction_path,
        bundle_path=bundle_path,
        outputs=outputs,
        coverage=coverage,
        go_summary=go_summary,
        kegg_summary=kegg_summary,
    )

    manifest = {
        "figure": "fig8_mfmr_biological_enrichment",
        "created_at": now_iso(),
        "dataset": args.dataset,
        "model_label": args.model_label,
        "repository_variant": args.variant,
        "prediction_file": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "bundle_file": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "prediction_shape": list(prediction.shape),
        "gene_count": int(expression.shape[1]),
        "parameters": {
            "ranking": "absolute Pearson correlation(expression, prediction); observed matrix used as coverage mask only",
            "direction_annotation": "mean Pearson correlation among each enriched term's overlap genes; enrichment remains unsigned",
            "top_genes_per_drug": int(args.top_genes),
            "minimum_prediction_pairs": int(args.min_samples),
            "fdr_threshold_strict_less_than": float(args.fdr),
            "minimum_drugs_per_term": int(args.min_drugs),
            "maximum_go_terms": int(args.max_go_terms),
            "maximum_kegg_terms": int(args.max_kegg_terms),
            "main_display_go_terms": int(args.main_go_terms),
            "main_display_kegg_terms": int(args.main_kegg_terms),
            "libraries": libraries,
        },
        "counts": {
            "eligible_drugs": int(coverage["eligible"].sum()),
            "top_gene_rows": int(len(ranking)),
            "enrichment_rows": int(len(enrichment)),
            "go_shared_terms": int(len(go_summary)),
            "go_edges": int(len(go_edges)),
            "kegg_shared_terms": int(len(kegg_summary)),
            "kegg_edges": int(len(kegg_edges)),
        },
        "tables": {
            "prediction_top_genes": str(ranking_path),
            "prediction_coverage": str(coverage_path),
            "enrichment_all": str(enrichment_path),
            "go_shared_edges": str(go_edges_path),
            "go_shared_terms": str(go_summary_path),
            "kegg_shared_edges": str(kegg_edges_path),
            "kegg_shared_terms": str(kegg_summary_path),
        },
        "cache_audit": cache_audit,
        "figures": outputs,
        "readme": str(readme_path),
    }
    manifest_path = output_dir / "fig8_mfmr_enrichment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "composite": outputs["composite"],
                "go_terms": len(go_summary),
                "kegg_terms": len(kegg_summary),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
