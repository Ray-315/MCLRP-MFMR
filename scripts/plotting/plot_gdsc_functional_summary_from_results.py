from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


CURRENT_FILE = Path(__file__).resolve()
ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Cannot locate project root")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.standardized_dataset_loaders import load_gdsc_standardized_bundle
from project_paths import GDSC_STANDARDIZED_DIR, LATEST_RESULTS_DIR


DEFAULT_VARIANTS = {
    "ERK_AUC": "A5_MFMR_Full",
    "ERK_IC50": "A5_MFMR_Full",
    "PI3K_AUC": "A5_MFMR_Full",
    "PI3K_IC50": "A5_MFMR_Full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GDSC figure-7-style summary inputs from MCLRP-MFMR results."
    )
    parser.add_argument("--dataset", choices=sorted(DEFAULT_VARIANTS), default="PI3K_AUC")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=1000, help="Top genes per drug exported for enrichment.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to plotting/outputs/requested_summary_figures/gdsc_functional_summary.",
    )
    return parser.parse_args()


def load_prediction_matrix(dataset: str, variant: str) -> np.ndarray:
    path = (
        LATEST_RESULTS_DIR
        / "ablation_mclrp_mfmr"
        / "shared_main"
        / dataset
        / "predictions"
        / f"{variant}_mean_prediction.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)["prediction"].astype(np.float32)


def vectorized_corr_p(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 8:
        nan_vec = np.full(X.shape[1], np.nan, dtype=np.float64)
        return nan_vec, nan_vec
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    x_norm = np.sqrt(np.sum(Xc * Xc, axis=0))
    y_norm = float(np.sqrt(np.sum(yc * yc)))
    denom = x_norm * y_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Xc.T @ yc) / denom
    r = np.clip(r, -1.0, 1.0)
    r[~np.isfinite(r)] = np.nan
    df = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = r * np.sqrt(df / np.maximum(1e-12, 1.0 - r * r))
    p = 2.0 * student_t.sf(np.abs(t_stat), df)
    p[~np.isfinite(p)] = np.nan
    return r.astype(np.float64), p.astype(np.float64)


def build_gene_ranking(
    X: np.ndarray,
    M_obs: np.ndarray,
    M_pred: np.ndarray,
    gene_symbols: np.ndarray,
    drug_labels: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_idx, drug_label in enumerate(drug_labels):
        valid = M_obs[:, drug_idx] != 0
        if valid.sum() < 30:
            continue
        y_obs = M_obs[valid, drug_idx].astype(float)
        y_pred = M_pred[valid, drug_idx].astype(float)
        Xv = X[valid].astype(float)
        r_obs_vec, p_obs_vec = vectorized_corr_p(Xv, y_obs)
        r_pred_vec, p_pred_vec = vectorized_corr_p(Xv, y_pred)
        for gene_idx, gene in enumerate(gene_symbols):
            r_obs = float(r_obs_vec[gene_idx])
            r_pred = float(r_pred_vec[gene_idx])
            if np.isnan(r_obs) or np.isnan(r_pred):
                continue
            p_obs = float(p_obs_vec[gene_idx])
            p_pred = float(p_pred_vec[gene_idx])
            score = abs(r_obs) + abs(r_pred)
            consistency = 1.0 - abs(r_obs - r_pred)
            rows.append(
                {
                    "drug_idx": int(drug_idx),
                    "drug_label": str(drug_label),
                    "gene_symbol": str(gene),
                    "r_observed": r_obs,
                    "p_observed": p_obs,
                    "r_predicted": r_pred,
                    "p_predicted": p_pred,
                    "abs_sum_score": score,
                    "consistency_score": consistency,
                    "signed_mean_r": (r_obs + r_pred) / 2.0,
                }
            )
    out = pd.DataFrame(rows)
    out.sort_values(
        ["drug_label", "abs_sum_score", "consistency_score", "p_predicted", "p_observed"],
        ascending=[True, False, False, True, True],
        inplace=True,
    )
    out["rank_within_drug"] = out.groupby("drug_label").cumcount() + 1
    return out


def export_top_gene_sets(ranking: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top = ranking.groupby("drug_label", group_keys=False).head(top_n).copy()
    top["direction"] = np.where(top["signed_mean_r"] >= 0, "positive", "negative")
    return top


def build_tissue_heatmap_table(
    observed: np.ndarray,
    predicted: np.ndarray,
    drug_labels: np.ndarray,
    cell_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_manifest = pd.read_csv(GDSC_STANDARDIZED_DIR / "model_manifest.csv")
    tissue_map = model_manifest.set_index("model_id")["Tissue"]
    tissues = pd.Index([tissue_map.get(str(cell_id), "Unknown") for cell_id in cell_ids], dtype=object)
    obs_df = pd.DataFrame(observed, columns=drug_labels)
    pred_df = pd.DataFrame(predicted, columns=drug_labels)
    obs_df["Tissue"] = tissues
    pred_df["Tissue"] = tissues
    obs_long = obs_df.melt(id_vars="Tissue", var_name="drug_label", value_name="observed")
    pred_long = pred_df.melt(id_vars="Tissue", var_name="drug_label", value_name="predicted")
    merged = obs_long.merge(pred_long, on=["Tissue", "drug_label"], how="inner")
    merged = merged[merged["observed"] != 0].copy()
    summary = (
        merged.groupby(["Tissue", "drug_label"], as_index=False)
        .agg(
            observed_mean=("observed", "mean"),
            predicted_mean=("predicted", "mean"),
            n=("observed", "size"),
        )
        .sort_values(["Tissue", "drug_label"])
    )
    heat = summary.pivot(index="Tissue", columns="drug_label", values="predicted_mean")
    return summary, heat


def draw_heatmap(heat: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 0.52 * heat.shape[1]), max(8, 0.36 * heat.shape[0])))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(np.arange(heat.shape[1]), labels=heat.columns)
    ax.set_yticks(np.arange(heat.shape[0]), labels=heat.index)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Drug")
    ax.set_ylabel("Tissue")
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean predicted response")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_gene_heatmap(top_gene_sets: pd.DataFrame, out_path: Path) -> None:
    panel = (
        top_gene_sets.groupby("drug_label", group_keys=False)
        .head(20)
        .pivot(index="gene_symbol", columns="drug_label", values="signed_mean_r")
    )
    if panel.empty:
        return
    ordered_genes = panel.abs().max(axis=1).sort_values(ascending=False).head(60).index
    panel = panel.loc[ordered_genes]
    fig, ax = plt.subplots(figsize=(max(10, 0.52 * panel.shape[1]), max(12, 0.24 * panel.shape[0])))
    vmax = float(np.nanmax(np.abs(panel.to_numpy())))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    im = ax.imshow(panel.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(panel.shape[1]), labels=panel.columns)
    ax.set_yticks(np.arange(panel.shape[0]), labels=panel.index)
    ax.set_title("Top Drug-Gene Correlations", fontsize=13, weight="bold")
    ax.set_xlabel("Drug")
    ax.set_ylabel("Gene")
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean gene-response correlation")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    variant = args.variant or DEFAULT_VARIANTS[args.dataset]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "plotting" / "outputs" / "requested_summary_figures" / "gdsc_functional_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_gdsc_standardized_bundle(args.dataset)
    pred = load_prediction_matrix(args.dataset, variant)
    if pred.shape != bundle.M.shape:
        raise ValueError(f"Prediction shape {pred.shape} != observed shape {bundle.M.shape}")

    ranking = build_gene_ranking(bundle.X, bundle.M, pred, bundle.gene_symbols, bundle.drug_labels)
    ranking_path = output_dir / f"{args.dataset}_{variant}_gene_ranking_long.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    top_gene_sets = export_top_gene_sets(ranking, args.top_n)
    top_path = output_dir / f"{args.dataset}_{variant}_top{args.top_n}_genes_per_drug.csv"
    top_gene_sets.to_csv(top_path, index=False, encoding="utf-8-sig")

    summary, tissue_heat = build_tissue_heatmap_table(bundle.M, pred, bundle.drug_labels, bundle.cell_ids)
    summary_path = output_dir / f"{args.dataset}_{variant}_tissue_mean_response.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    draw_heatmap(
        tissue_heat,
        title=f"{args.dataset} | {variant} | Tissue Mean Predicted Response",
        out_path=output_dir / f"{args.dataset}_{variant}_tissue_mean_heatmap.png",
    )
    draw_gene_heatmap(
        top_gene_sets,
        out_path=output_dir / f"{args.dataset}_{variant}_drug_gene_heatmap.png",
    )

    print(f"saved ranking: {ranking_path}")
    print(f"saved top genes: {top_path}")
    print(f"saved tissue summary: {summary_path}")


if __name__ == "__main__":
    main()
