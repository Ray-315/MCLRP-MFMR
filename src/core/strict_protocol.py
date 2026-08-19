from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge

from .progress_monitor import ThrottledProgressLogger


def top_var_indices(X: np.ndarray, topg: int) -> np.ndarray:
    topg_eff = int(max(1, min(int(topg), X.shape[1])))
    var = np.var(X, axis=0)
    if topg_eff >= X.shape[1]:
        return np.arange(X.shape[1], dtype=np.int64)
    idx = np.argpartition(var, -topg_eff)[-topg_eff:]
    return idx[np.argsort(var[idx])[::-1]].astype(np.int64)


def eff_n_components(comp: int, n_samples: int, n_features: int) -> int:
    upper = max(1, min(n_samples - 1 if n_samples > 1 else 1, n_features - 1 if n_features > 1 else 1))
    return int(max(1, min(int(comp), upper)))


def get_valid_train_rows(train_mask: np.ndarray) -> np.ndarray:
    rows = np.where(np.sum(train_mask, axis=1) > 0)[0]
    if len(rows) == 0:
        rows = np.arange(train_mask.shape[0], dtype=np.int64)
    return rows.astype(np.int64)


def make_train_stats(M: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainM = np.where(train_mask, M, np.nan)
    global_mean = np.nanmean(trainM)
    row_mean = np.nanmean(trainM, axis=1)
    col_mean = np.nanmean(trainM, axis=0)
    row_mean = np.where(np.isnan(row_mean), global_mean, row_mean).astype(np.float32)
    col_mean = np.where(np.isnan(col_mean), global_mean, col_mean).astype(np.float32)
    row_cnt = np.sum(train_mask, axis=1).astype(np.float32)
    filled = np.where(np.isnan(trainM), (row_mean[:, None] + col_mean[None, :]) / 2.0, trainM).astype(np.float32)
    return row_mean, col_mean, row_cnt, filled


def build_expr_pcs_pair(
    X_train: np.ndarray,
    X_test: np.ndarray,
    topg: int,
    comp: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    idx = top_var_indices(X_train, topg)
    Xtr = X_train[:, idx].astype(np.float32)
    Xte = X_test[:, idx].astype(np.float32)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    n_comp = eff_n_components(comp, Xtr.shape[0], Xtr.shape[1])
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Xp_tr = svd.fit_transform(Xtr).astype(np.float32)
    Xp_te = svd.transform(Xte).astype(np.float32)
    return Xp_tr, Xp_te


def select_high_var_genes_fit_transform(X_train: np.ndarray, X_all: np.ndarray, topg: int) -> np.ndarray:
    idx = top_var_indices(X_train, topg)
    Xtr = X_train[:, idx].astype(np.float32)
    Xall = X_all[:, idx].astype(np.float32)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-6
    return ((Xall - mean) / std).astype(np.float32)


def build_mutation_latent_pair(
    binary_train: np.ndarray,
    binary_test: np.ndarray,
    latent_dim: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n_comp = eff_n_components(latent_dim, binary_train.shape[0], binary_train.shape[1])
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    lat_tr = svd.fit_transform(binary_train.astype(np.float32)).astype(np.float32)
    lat_te = svd.transform(binary_test.astype(np.float32)).astype(np.float32)
    return lat_tr, lat_te


def fit_iterative_imputer_pair(
    M_masked: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    Xp_tr: np.ndarray,
    Xp_te: np.ndarray,
    max_iter: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    # The response matrix uses zero as the missing-value sentinel. IterativeImputer
    # only imputes NaNs, so sanitize the response block here to avoid treating
    # original missing entries as observed zeros.
    masked_tr = M_masked[tr].astype(np.float32, copy=True)
    masked_te = M_masked[te].astype(np.float32, copy=True)
    masked_tr[masked_tr == 0] = np.nan
    masked_te[masked_te == 0] = np.nan
    imp_input_tr = np.concatenate([masked_tr, Xp_tr], axis=1)
    imp_input_te = np.concatenate([masked_te, Xp_te], axis=1)
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=int(max_iter),
        random_state=int(random_state),
        initial_strategy="mean",
        skip_complete=True,
    )
    imp_tr = imputer.fit_transform(imp_input_tr).astype(np.float32)
    imp_te = imputer.transform(imp_input_te).astype(np.float32)
    return imp_tr, imp_te


def predict_shared_mfmr_seed_strict(
    X: np.ndarray,
    M: np.ndarray,
    folds: Sequence[np.ndarray],
    *,
    topg_imp: int,
    comp_imp: int,
    topg_ridge: int,
    comp_ridge: int,
    ridge_alpha: float,
    weight_imp: float,
    weight_ridge: float,
    imputer_max_iter: int = 3,
    min_train: int = 5,
    progress_label: str | None = None,
    progress_interval_sec: float = 30.0,
) -> np.ndarray:
    pred = np.zeros_like(M, dtype=np.float32)
    other_cols = [[k for k in range(M.shape[1]) if k != j] for j in range(M.shape[1])]
    for fold_idx, examdata in enumerate(folds):
        drug_logger = None
        if progress_label:
            drug_logger = ThrottledProgressLogger(
                f"{progress_label} fold={fold_idx + 1}/{len(folds)}",
                M.shape[1],
                unit="drugs",
                min_interval_sec=progress_interval_sec,
            )
            drug_logger.update(0, detail="started", force=True)
        test_mask = examdata != 0
        train_mask = (M != 0) & (~test_mask)
        M_masked = np.where(test_mask, np.nan, M).astype(np.float32)
        row_mean, _, row_cnt, filled = make_train_stats(M, train_mask)
        for j in range(M.shape[1]):
            tr = np.where(train_mask[:, j])[0]
            te = np.where(test_mask[:, j])[0]
            if len(te) == 0 or len(tr) < int(min_train):
                continue
            Xp_imp_tr, Xp_imp_te = build_expr_pcs_pair(X[tr], X[te], topg=topg_imp, comp=comp_imp, seed=fold_idx)
            Xp_ridge_tr, Xp_ridge_te = build_expr_pcs_pair(
                X[tr],
                X[te],
                topg=topg_ridge,
                comp=comp_ridge,
                seed=fold_idx + 17,
            )
            imp_tr, imp_te = fit_iterative_imputer_pair(
                M_masked,
                tr,
                te,
                Xp_imp_tr,
                Xp_imp_te,
                max_iter=imputer_max_iter,
                random_state=42 + fold_idx,
            )
            Xtr = np.concatenate(
                [filled[tr][:, other_cols[j]], Xp_ridge_tr, row_mean[tr, None], row_cnt[tr, None]],
                axis=1,
            )
            Xte = np.concatenate(
                [filled[te][:, other_cols[j]], Xp_ridge_te, row_mean[te, None], row_cnt[te, None]],
                axis=1,
            )
            model = Ridge(alpha=float(ridge_alpha))
            model.fit(Xtr, M[tr, j])
            ridge_test = model.predict(Xte).astype(np.float32)
            pred[te, j] = (float(weight_imp) * imp_te[:, j] + float(weight_ridge) * ridge_test).astype(np.float32)
            if drug_logger is not None:
                drug_logger.update(j + 1, detail=f"last_drug={j + 1}")
        if drug_logger is not None:
            drug_logger.update(M.shape[1], detail="fold complete", force=True)
    return pred.astype(np.float32)


def predict_cgp_mfmr_seed_strict(
    dataset_name: str,
    X: np.ndarray,
    M: np.ndarray,
    folds: Sequence[np.ndarray],
    *,
    topg_imp: int,
    comp_imp: int,
    topg_ridge: int,
    comp_ridge: int,
    ridge_alpha: float,
    final_alpha: float,
    weight_imp: float,
    weight_ridge: float,
    tissue: np.ndarray,
    pathway: np.ndarray,
    overall_mut: np.ndarray,
    binary_mut: np.ndarray,
    mutation_latent_dim: int,
    imputer_max_iter: int,
    min_train: int = 10,
    progress_label: str | None = None,
    progress_interval_sec: float = 30.0,
) -> np.ndarray:
    del dataset_name
    pred = np.zeros_like(M, dtype=np.float32)
    other_cols = [[k for k in range(M.shape[1]) if k != j] for j in range(M.shape[1])]
    for fold_idx, examdata in enumerate(folds):
        drug_logger = None
        if progress_label:
            drug_logger = ThrottledProgressLogger(
                f"{progress_label} fold={fold_idx + 1}/{len(folds)}",
                M.shape[1],
                unit="drugs",
                min_interval_sec=progress_interval_sec,
            )
            drug_logger.update(0, detail="started", force=True)
        test_mask = examdata != 0
        train_mask = (M != 0) & (~test_mask)
        M_masked = np.where(test_mask, np.nan, M).astype(np.float32)
        row_mean, col_mean, row_cnt, filled = make_train_stats(M, train_mask)
        for j in range(M.shape[1]):
            tr = np.where(train_mask[:, j])[0]
            te = np.where(test_mask[:, j])[0]
            if len(te) == 0 or len(tr) < int(min_train):
                continue
            Xp_imp_tr, Xp_imp_te = build_expr_pcs_pair(X[tr], X[te], topg=topg_imp, comp=comp_imp, seed=fold_idx)
            Xp_ridge_tr, Xp_ridge_te = build_expr_pcs_pair(
                X[tr],
                X[te],
                topg=topg_ridge,
                comp=comp_ridge,
                seed=fold_idx + 17,
            )
            latent_mut_tr, latent_mut_te = build_mutation_latent_pair(
                binary_mut[tr],
                binary_mut[te],
                latent_dim=mutation_latent_dim,
                seed=fold_idx + 31,
            )
            imp_tr, imp_te = fit_iterative_imputer_pair(
                M_masked,
                tr,
                te,
                Xp_imp_tr,
                Xp_imp_te,
                max_iter=imputer_max_iter,
                random_state=42 + fold_idx,
            )
            y = M[tr, j]
            Xtr = np.concatenate(
                [filled[tr][:, other_cols[j]], Xp_ridge_tr, row_mean[tr, None], row_cnt[tr, None]],
                axis=1,
            )
            Xte = np.concatenate(
                [filled[te][:, other_cols[j]], Xp_ridge_te, row_mean[te, None], row_cnt[te, None]],
                axis=1,
            )
            ridge_model = Ridge(alpha=float(ridge_alpha))
            ridge_model.fit(Xtr, y)
            ridge_train = ridge_model.predict(Xtr).astype(np.float32)
            ridge_test = ridge_model.predict(Xte).astype(np.float32)
            imp_train = imp_tr[:, j]
            imp_test = imp_te[:, j]
            base_train = (float(weight_imp) * imp_train + float(weight_ridge) * ridge_train).astype(np.float32)
            base_test = (float(weight_imp) * imp_test + float(weight_ridge) * ridge_test).astype(np.float32)
            meta_train = np.concatenate(
                [
                    imp_train[:, None],
                    ridge_train[:, None],
                    (imp_train * ridge_train)[:, None],
                    row_mean[tr, None],
                    np.full((len(tr), 1), col_mean[j], dtype=np.float32),
                    Xp_ridge_tr,
                ],
                axis=1,
            )
            meta_test = np.concatenate(
                [
                    imp_test[:, None],
                    ridge_test[:, None],
                    (imp_test * ridge_test)[:, None],
                    row_mean[te, None],
                    np.full((len(te), 1), col_mean[j], dtype=np.float32),
                    Xp_ridge_te,
                ],
                axis=1,
            )
            pathway_score_train = pathway[tr].mean(axis=1, keepdims=True).astype(np.float32)
            pathway_score_test = pathway[te].mean(axis=1, keepdims=True).astype(np.float32)
            base_terms_train = np.concatenate(
                [
                    base_train[:, None],
                    imp_train[:, None],
                    ridge_train[:, None],
                ],
                axis=1,
            )
            base_terms_test = np.concatenate(
                [
                    base_test[:, None],
                    imp_test[:, None],
                    ridge_test[:, None],
                ],
                axis=1,
            )
            mut_train = np.concatenate(
                [
                    tissue[tr],
                    pathway[tr],
                    overall_mut[tr],
                    latent_mut_tr,
                    pathway_score_train,
                    pathway_score_train * base_terms_train,
                ],
                axis=1,
            )
            mut_test = np.concatenate(
                [
                    tissue[te],
                    pathway[te],
                    overall_mut[te],
                    latent_mut_te,
                    pathway_score_test,
                    pathway_score_test * base_terms_test,
                ],
                axis=1,
            )
            final_model = Ridge(alpha=float(final_alpha))
            delta_train = (y - base_train).astype(np.float32)
            final_model.fit(np.concatenate([meta_train, mut_train], axis=1), delta_train)
            delta_test = final_model.predict(np.concatenate([meta_test, mut_test], axis=1)).astype(np.float32)
            pred[te, j] = (base_test + delta_test).astype(np.float32)
            if drug_logger is not None:
                drug_logger.update(j + 1, detail=f"last_drug={j + 1}")
        if drug_logger is not None:
            drug_logger.update(M.shape[1], detail="fold complete", force=True)
    return pred.astype(np.float32)
