from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.exceptions import ConvergenceWarning

from project_paths import SCRATCH_RESULTS_DIR

from .calc_pcc import calc_pcc
from .getcrossMatrixs import getcrossMatrixs
from .standardized_dataset_loaders import load_ccle_standardized_bundle
from .strict_protocol import build_expr_pcs_pair
from .strict_protocol import fit_iterative_imputer_pair

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

BASE_DIR = SCRATCH_RESULTS_DIR / 'strict_random_cv'
REFERENCE_FIXED_MAIN_PCC = 0.6963444191455191
DEFAULT_SEEDS = tuple(range(10))


@dataclass(frozen=True)
class RandomCVConfig:
    num_folds: int = 10
    seeds: Sequence[int] = DEFAULT_SEEDS
    topg_imp: int = 2000
    comp_imp: int = 12
    topg_ridge: int = 2500
    comp_ridge: int = 16
    ridge_alpha: float = 10.0
    weight_imp: float = 0.5
    weight_ridge: float = 0.5


def pct_gain(new: float, old: float) -> float:
    if old == 0:
        return float('nan')
    return (new - old) / abs(old) * 100.0


def build_expr_pcs(X: np.ndarray, topg: int, comp: int, seed: int = 0) -> np.ndarray:
    var = np.var(X, axis=0)
    idx = np.argsort(var)[::-1][:topg]
    Xsel = X[:, idx].astype(np.float32)
    Xsel = (Xsel - Xsel.mean(axis=0)) / (Xsel.std(axis=0) + 1e-6)
    return TruncatedSVD(n_components=comp, random_state=seed).fit_transform(Xsel).astype(np.float32)


def make_train_stats(M: np.ndarray, train_mask: np.ndarray):
    trainM = np.where(train_mask, M, np.nan)
    global_mean = np.nanmean(trainM)
    row_mean = np.nanmean(trainM, axis=1)
    col_mean = np.nanmean(trainM, axis=0)
    row_mean = np.where(np.isnan(row_mean), global_mean, row_mean).astype(np.float32)
    col_mean = np.where(np.isnan(col_mean), global_mean, col_mean).astype(np.float32)
    row_cnt = np.sum(train_mask, axis=1).astype(np.float32)
    filled = np.where(np.isnan(trainM), (row_mean[:, None] + col_mean[None, :]) / 2.0, trainM).astype(np.float32)
    return row_mean, col_mean, row_cnt, filled


def candidate_predictions(
    M: np.ndarray,
    X: np.ndarray,
    folds: Sequence[np.ndarray],
    config: RandomCVConfig,
) -> tuple[np.ndarray, np.ndarray]:
    d = M.shape[1]
    other_cols = [[k for k in range(d) if k != j] for j in range(d)]
    imp = np.zeros_like(M, dtype=np.float32)
    ridge_pred = np.zeros_like(M, dtype=np.float32)

    for fold_idx, examdata in enumerate(folds):
        test_mask = examdata != 0
        train_mask = (M != 0) & (~test_mask)
        M_masked = np.where(test_mask, np.nan, M).astype(np.float32)
        row_mean, col_mean, row_cnt, filled = make_train_stats(M, train_mask)
        for j in range(d):
            tr = np.where(train_mask[:, j])[0]
            te = np.where(test_mask[:, j])[0]
            if len(te) == 0 or len(tr) < 5:
                continue
            Xp_imp_tr, Xp_imp_te = build_expr_pcs_pair(
                X[tr],
                X[te],
                topg=config.topg_imp,
                comp=config.comp_imp,
                seed=fold_idx,
            )
            imp_tr, imp_te = fit_iterative_imputer_pair(
                M_masked,
                tr,
                te,
                Xp_imp_tr,
                Xp_imp_te,
                max_iter=3,
                random_state=42 + fold_idx,
            )
            imp[te, j] = imp_te[:, j]
            Xp_ridge_tr, Xp_ridge_te = build_expr_pcs_pair(
                X[tr],
                X[te],
                topg=config.topg_ridge,
                comp=config.comp_ridge,
                seed=fold_idx + 17,
            )
            Xtr = np.concatenate([
                filled[tr][:, other_cols[j]],
                Xp_ridge_tr,
                row_mean[tr, None],
                row_cnt[tr, None],
            ], axis=1)
            Xte = np.concatenate([
                filled[te][:, other_cols[j]],
                Xp_ridge_te,
                row_mean[te, None],
                row_cnt[te, None],
            ], axis=1)
            y = M[tr, j]
            model = Ridge(alpha=config.ridge_alpha)
            model.fit(Xtr, y)
            ridge_pred[te, j] = model.predict(Xte).astype(np.float32)
    return imp, ridge_pred


def evaluate_seed(
    M: np.ndarray,
    X: np.ndarray,
    seed: int,
    config: RandomCVConfig,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, Sequence[np.ndarray]]:
    folds = getcrossMatrixs(M, num_folds=config.num_folds, rng=np.random.default_rng(seed))
    imp, ridge_pred = candidate_predictions(M, X, folds, config)
    pred = config.weight_imp * imp + config.weight_ridge * ridge_pred
    overall_pcc, fold_pcc = calc_pcc(pred, M, folds)
    row = {
        'seed': int(seed),
        'overall_pcc': float(overall_pcc),
        'mean_fold_pcc': float(np.mean(fold_pcc)),
        'std_fold_pcc': float(np.std(fold_pcc, ddof=0)),
        'min_fold_pcc': float(np.min(fold_pcc)),
        'max_fold_pcc': float(np.max(fold_pcc)),
        'weight_imp': float(config.weight_imp),
        'weight_ridge': float(config.weight_ridge),
        'ridge_alpha': float(config.ridge_alpha),
    }
    return row, np.asarray(fold_pcc, dtype=np.float32), pred, imp, ridge_pred, folds


def run_main_random_cv(config: RandomCVConfig = RandomCVConfig()) -> dict:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_ccle_standardized_bundle()
    X = bundle.X.astype(np.float32)
    M = bundle.M.astype(np.float32)

    seed_rows = []
    fold_rows = []
    representative = None

    for i, seed in enumerate(config.seeds):
        row, fold_pcc, pred, imp, ridge_pred, folds = evaluate_seed(M, X, int(seed), config)
        seed_rows.append(row)
        for fold_idx, value in enumerate(fold_pcc, start=1):
            fold_rows.append({'seed': int(seed), 'fold': fold_idx, 'pcc': float(value)})
        if i == 0:
            representative = {
                'seed': int(seed),
                'prediction': pred,
                'imputer': imp,
                'ridge': ridge_pred,
                'folds': np.array(folds, dtype=np.float32),
            }

    df_seed = pd.DataFrame(seed_rows)
    df_fold = pd.DataFrame(fold_rows)
    mean_pcc = float(df_seed['overall_pcc'].mean())
    std_pcc = float(df_seed['overall_pcc'].std(ddof=0))
    min_pcc = float(df_seed['overall_pcc'].min())
    max_pcc = float(df_seed['overall_pcc'].max())
    all_above_ref = bool((df_seed['overall_pcc'] >= REFERENCE_FIXED_MAIN_PCC).all())

    summary = {
        'reference_fixed_main_pcc': float(REFERENCE_FIXED_MAIN_PCC),
        'random_cv_mean_pcc': mean_pcc,
        'random_cv_std_pcc': std_pcc,
        'random_cv_min_pcc': min_pcc,
        'random_cv_max_pcc': max_pcc,
        'random_cv_num_seeds': int(len(config.seeds)),
        'margin_vs_reference_mean': float(mean_pcc - REFERENCE_FIXED_MAIN_PCC),
        'pct_gain_vs_reference_mean': float(pct_gain(mean_pcc, REFERENCE_FIXED_MAIN_PCC)),
        'all_seeds_above_reference': all_above_ref,
        'config': {
            'num_folds': int(config.num_folds),
            'seeds': [int(s) for s in config.seeds],
            'topg_imp': int(config.topg_imp),
            'comp_imp': int(config.comp_imp),
            'topg_ridge': int(config.topg_ridge),
            'comp_ridge': int(config.comp_ridge),
            'ridge_alpha': float(config.ridge_alpha),
            'weight_imp': float(config.weight_imp),
            'weight_ridge': float(config.weight_ridge),
        },
    }

    df_seed.to_csv(BASE_DIR / 'strict_random_seed_results.csv', index=False, encoding='utf-8-sig')
    df_fold.to_csv(BASE_DIR / 'strict_random_fold_results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([summary]).to_csv(BASE_DIR / 'strict_random_summary.csv', index=False, encoding='utf-8-sig')
    (BASE_DIR / 'strict_random_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    if representative is not None:
        np.savez(
            BASE_DIR / f"prediction_output_random_seed{representative['seed']}.npz",
            prediction=representative['prediction'],
            imputer=representative['imputer'],
            ridge=representative['ridge'],
            folds=representative['folds'],
        )

    result = {
        'summary': summary,
        'per_seed': seed_rows,
    }
    return result


def main() -> None:
    result = run_main_random_cv()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
