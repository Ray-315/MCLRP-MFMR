from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.exceptions import ConvergenceWarning

from project_paths import CGP_RAW_DATA_DIR, SCRATCH_RESULTS_DIR

from .calc_pcc import calc_pcc
from .getcrossMatrixs import getcrossMatrixs
from .strict_protocol import build_expr_pcs_pair
from .strict_protocol import build_mutation_latent_pair
from .strict_protocol import fit_iterative_imputer_pair
from .utils_loader import load_npz_var

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

BASE_DIR = SCRATCH_RESULTS_DIR / 'strict_random_mutation_cv'
DEFAULT_SEEDS = tuple(range(10))
DATASETS = ('ERKAUC30', 'ERKIC50', 'PI3KAUC', 'PI3KIC50')
MAPK_GENES = (
    'ABL2', 'EGFR', 'FGFR3', 'JAK2', 'ALK', 'BRAF', 'EGFR.1', 'ERBB2',
    'FGFR2', 'FGFR3.1', 'FLT3', 'HRAS', 'KDR', 'KIT', 'KRAS', 'MAP2K4',
    'MET', 'NF1', 'NF2', 'NRAS', 'PDGFRA'
)
PI3K_GENES = (
    'AKT2', 'EGFR', 'ERBB2', 'HRAS', 'KRAS', 'NRAS', 'PIK3CA', 'PIK3R1',
    'PTEN', 'STK11', 'TSC1', 'CCND1', 'CCND2', 'CCND3', 'CDK4', 'CDK6',
    'CDKN2A', 'CDKN2C'
)
HOTSPOT_PATTERNS = {
    'BRAF': r'V600',
    'KRAS': r'G12|G13|Q61|A146',
    'NRAS': r'G12|G13|Q61',
    'PIK3CA': r'H1047|E545|E542|Q546|P539',
    'PTEN': r'\*|fs|\?',
}

@dataclass(frozen=True)
class MutationConfig:
    num_folds: int = 10
    base_weight_imp: float = 0.5
    base_weight_ridge: float = 0.5
    expr_topg_imp: int = 1500
    expr_comp_imp: int = 10
    expr_topg_ridge: int = 2000
    expr_comp_ridge: int = 12
    ridge_alpha: float = 10.0
    final_alpha: float = 20.0
    mutation_latent_dim: int = 24
    imputer_max_iter: int = 2


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


def _parse_gene_state(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = series.astype(str)
    mut = s.str.split('::').str[0].fillna('na')
    cn = s.str.split('::').str[-1].fillna('nci')
    mutated = ((~mut.str.startswith('wt')) & (~mut.str.startswith('na'))).astype(np.float32).to_numpy()
    amp = cn.str.contains('>=8').astype(np.float32).to_numpy()
    loss = cn.eq('0').astype(np.float32).to_numpy()
    return mutated, amp, loss, mut.to_numpy()


def _pathway_summary(gene_set: Sequence[str], df: pd.DataFrame) -> np.ndarray:
    muts, amps, losses, hots = [], [], [], []
    for g in gene_set:
        if g not in df.columns:
            continue
        m, a, l, tok = _parse_gene_state(df[g])
        muts.append(m)
        amps.append(a)
        losses.append(l)
        if g in HOTSPOT_PATTERNS:
            h = pd.Series(tok).str.contains(HOTSPOT_PATTERNS[g], regex=True, na=False).astype(np.float32).to_numpy()
            hots.append(h)

    def _stack_mean(lst: List[np.ndarray]) -> np.ndarray:
        if not lst:
            return np.zeros(len(df), dtype=np.float32)
        return np.mean(np.vstack(lst), axis=0).astype(np.float32)

    return np.vstack([
        _stack_mean(muts),
        _stack_mean(amps),
        _stack_mean(losses),
        _stack_mean(hots),
    ]).T.astype(np.float32)


def build_mutation_features(latent_dim: int = 24) -> Dict[str, np.ndarray]:
    s1 = pd.read_excel(CGP_RAW_DATA_DIR / 'Mutation.xlsx', sheet_name='Sheet1')
    s2 = pd.read_excel(CGP_RAW_DATA_DIR / 'Mutation.xlsx', sheet_name='Sheet2')
    tissue = pd.get_dummies(s1[['Cancer Type', 'Tissue']].astype(str), dummy_na=False).astype(np.float32).to_numpy()
    gene_cols = [c for c in s1.columns if c not in ['Cell Line', 'Cosmic_ID', 'Cancer Type', 'Tissue']]
    binary_feats: List[np.ndarray] = []
    for df in (s1, s2):
        for gene in gene_cols:
            if gene not in df.columns:
                continue
            mut, amp, loss, tok = _parse_gene_state(df[gene])
            binary_feats.extend([mut, amp, loss])
            if gene in HOTSPOT_PATTERNS:
                hotspot = pd.Series(tok).str.contains(HOTSPOT_PATTERNS[gene], regex=True, na=False).astype(np.float32).to_numpy()
                binary_feats.append(hotspot)
    binary = np.vstack(binary_feats).T.astype(np.float32)
    mapk = np.concatenate([_pathway_summary(MAPK_GENES, s1), _pathway_summary(MAPK_GENES, s2)], axis=1).astype(np.float32)
    pi3k = np.concatenate([_pathway_summary(PI3K_GENES, s1), _pathway_summary(PI3K_GENES, s2)], axis=1).astype(np.float32)
    overall = np.vstack([
        binary.mean(axis=1),
        binary[:, 1::4].mean(axis=1),
        binary[:, 2::4].mean(axis=1),
    ]).T.astype(np.float32)
    return {'tissue': tissue, 'mapk': mapk, 'pi3k': pi3k, 'overall': overall, 'binary': binary}


def make_train_stats(M: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainM = np.where(train_mask, M, np.nan)
    global_mean = np.nanmean(trainM)
    row_mean = np.nanmean(trainM, axis=1)
    col_mean = np.nanmean(trainM, axis=0)
    row_mean = np.where(np.isnan(row_mean), global_mean, row_mean).astype(np.float32)
    col_mean = np.where(np.isnan(col_mean), global_mean, col_mean).astype(np.float32)
    row_cnt = np.sum(train_mask, axis=1).astype(np.float32)
    filled = np.where(np.isnan(trainM), (row_mean[:, None] + col_mean[None, :]) / 2.0, trainM).astype(np.float32)
    return row_mean, col_mean, row_cnt, filled


def _dataset_pathway_features(dataset_name: str, mutation_feats: Dict[str, np.ndarray]) -> np.ndarray:
    return mutation_feats['mapk'] if dataset_name.startswith('ERK') else mutation_feats['pi3k']


def evaluate_dataset_seed(dataset_name: str, seed: int, config: MutationConfig, X: np.ndarray, mutation_feats: Dict[str, np.ndarray]) -> Dict[str, object]:
    M = load_npz_var(CGP_RAW_DATA_DIR / f'{dataset_name}.npz', dataset_name).astype(np.float32)
    folds = getcrossMatrixs(M, num_folds=config.num_folds, rng=np.random.default_rng(seed))
    D = M.shape[1]
    other_cols = [[k for k in range(D) if k != j] for j in range(D)]
    tissue = mutation_feats['tissue']
    pathway = _dataset_pathway_features(dataset_name, mutation_feats)
    overall_mut = mutation_feats['overall']
    binary_mut = mutation_feats['binary']
    base_pred = np.zeros_like(M, dtype=np.float32)
    final_pred = np.zeros_like(M, dtype=np.float32)
    for fold_idx, examdata in enumerate(folds):
        test_mask = examdata != 0
        train_mask = (M != 0) & (~test_mask)
        M_masked = np.where(test_mask, np.nan, M).astype(np.float32)
        row_mean, col_mean, row_cnt, filled = make_train_stats(M, train_mask)
        for j in range(D):
            tr = np.where(train_mask[:, j])[0]
            te = np.where(test_mask[:, j])[0]
            if len(te) == 0 or len(tr) < 10:
                continue
            Xp_imp_tr, Xp_imp_te = build_expr_pcs_pair(
                X[tr],
                X[te],
                topg=config.expr_topg_imp,
                comp=config.expr_comp_imp,
                seed=fold_idx,
            )
            imp_tr, imp_te = fit_iterative_imputer_pair(
                M_masked,
                tr,
                te,
                Xp_imp_tr,
                Xp_imp_te,
                max_iter=config.imputer_max_iter,
                random_state=42 + fold_idx,
            )
            Xp_ridge_tr, Xp_ridge_te = build_expr_pcs_pair(
                X[tr],
                X[te],
                topg=config.expr_topg_ridge,
                comp=config.expr_comp_ridge,
                seed=fold_idx + 17,
            )
            latent_mut_tr, latent_mut_te = build_mutation_latent_pair(
                binary_mut[tr],
                binary_mut[te],
                latent_dim=config.mutation_latent_dim,
                seed=fold_idx + 31,
            )
            Xtr = np.concatenate([filled[tr][:, other_cols[j]], Xp_ridge_tr, row_mean[tr, None], row_cnt[tr, None]], axis=1)
            Xte = np.concatenate([filled[te][:, other_cols[j]], Xp_ridge_te, row_mean[te, None], row_cnt[te, None]], axis=1)
            y = M[tr, j]
            ridge_model = Ridge(alpha=config.ridge_alpha)
            ridge_model.fit(Xtr, y)
            ridge_train = ridge_model.predict(Xtr).astype(np.float32)
            ridge_test = ridge_model.predict(Xte).astype(np.float32)
            imp_train = imp_tr[:, j]
            imp_test = imp_te[:, j]
            base_train = (config.base_weight_imp * imp_train + config.base_weight_ridge * ridge_train).astype(np.float32)
            base_test = (config.base_weight_imp * imp_test + config.base_weight_ridge * ridge_test).astype(np.float32)
            base_pred[te, j] = base_test.astype(np.float32)
            meta_train = np.concatenate([imp_train[:, None], ridge_train[:, None], (imp_train * ridge_train)[:, None], row_mean[tr, None], np.full((len(tr), 1), col_mean[j], dtype=np.float32), Xp_ridge_tr], axis=1)
            meta_test = np.concatenate([imp_test[:, None], ridge_test[:, None], (imp_test * ridge_test)[:, None], row_mean[te, None], np.full((len(te), 1), col_mean[j], dtype=np.float32), Xp_ridge_te], axis=1)
            pathway_score_train = pathway[tr].mean(axis=1, keepdims=True).astype(np.float32)
            pathway_score_test = pathway[te].mean(axis=1, keepdims=True).astype(np.float32)
            base_terms_train = np.concatenate([base_train[:, None], imp_train[:, None], ridge_train[:, None]], axis=1)
            base_terms_test = np.concatenate([base_test[:, None], imp_test[:, None], ridge_test[:, None]], axis=1)
            mut_train = np.concatenate([tissue[tr], pathway[tr], overall_mut[tr], latent_mut_tr, pathway_score_train, pathway_score_train * base_terms_train], axis=1)
            mut_test = np.concatenate([tissue[te], pathway[te], overall_mut[te], latent_mut_te, pathway_score_test, pathway_score_test * base_terms_test], axis=1)
            final_model = Ridge(alpha=config.final_alpha)
            delta_train = (y - base_train).astype(np.float32)
            final_model.fit(np.concatenate([meta_train, mut_train], axis=1), delta_train)
            delta_test = final_model.predict(np.concatenate([meta_test, mut_test], axis=1)).astype(np.float32)
            final_pred[te, j] = (base_test + delta_test).astype(np.float32)
    overall_base, fold_base = calc_pcc(base_pred, M, folds)
    overall_final, fold_final = calc_pcc(final_pred, M, folds)
    return {'dataset': dataset_name, 'seed': int(seed), 'base_pcc': float(overall_base), 'final_pcc': float(overall_final), 'abs_gain_final_minus_base': float(overall_final - overall_base), 'pct_gain_final_minus_base': float(pct_gain(overall_final, overall_base)), 'fold_base': np.asarray(fold_base, dtype=np.float32), 'fold_final': np.asarray(fold_final, dtype=np.float32), 'base_pred': base_pred, 'final_pred': final_pred, 'folds': np.asarray(folds, dtype=np.float32)}


def run_one_seed(seed: int, config: MutationConfig = MutationConfig(), save_predictions: bool = False) -> Dict[str, object]:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    X = load_npz_var(CGP_RAW_DATA_DIR / 'CGP_X.npz', 'X').astype(np.float32)
    mutation_feats = build_mutation_features(config.mutation_latent_dim)
    seed_rows, fold_rows = [], []
    for dataset in DATASETS:
        result = evaluate_dataset_seed(dataset, seed, config, X, mutation_feats)
        seed_rows.append({'seed': int(seed), 'dataset': dataset, 'base_pcc': result['base_pcc'], 'final_pcc': result['final_pcc'], 'abs_gain_final_minus_base': result['abs_gain_final_minus_base'], 'pct_gain_final_minus_base': result['pct_gain_final_minus_base']})
        for fold_idx, (b, f) in enumerate(zip(result['fold_base'], result['fold_final']), start=1):
            fold_rows.append({'seed': int(seed), 'dataset': dataset, 'fold': fold_idx, 'base_fold_pcc': float(b), 'final_fold_pcc': float(f), 'fold_gain': float(f - b)})
        if save_predictions and seed == 0:
            np.savez(BASE_DIR / f'{dataset}_random_mutation_seed{seed}.npz', base=result['base_pred'], final=result['final_pred'], folds=result['folds'])
    pd.DataFrame(seed_rows).to_csv(BASE_DIR / f'mutation_seed_{seed}_results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(fold_rows).to_csv(BASE_DIR / f'mutation_seed_{seed}_fold_results.csv', index=False, encoding='utf-8-sig')
    return {'seed': int(seed), 'datasets': seed_rows, 'config': asdict(config)}


def aggregate_results(config: MutationConfig = MutationConfig()) -> Dict[str, object]:
    seed_frames, fold_frames = [], []
    for seed in DEFAULT_SEEDS:
        sf = BASE_DIR / f'mutation_seed_{seed}_results.csv'
        ff = BASE_DIR / f'mutation_seed_{seed}_fold_results.csv'
        if sf.exists():
            seed_frames.append(pd.read_csv(sf))
        if ff.exists():
            fold_frames.append(pd.read_csv(ff))
    if not seed_frames or not fold_frames:
        raise FileNotFoundError('No per-seed mutation results found.')
    seed_df = pd.concat(seed_frames, ignore_index=True)
    fold_df = pd.concat(fold_frames, ignore_index=True)
    summary_rows = []
    for dataset in DATASETS:
        sdf = seed_df[seed_df['dataset'] == dataset].copy()
        fdf = fold_df[fold_df['dataset'] == dataset].copy()
        seed_t = ttest_rel(sdf['final_pcc'], sdf['base_pcc'], alternative='greater') if len(sdf) >= 2 else None
        fold_t = ttest_rel(fdf['final_fold_pcc'], fdf['base_fold_pcc'], alternative='greater') if len(fdf) >= 2 else None
        try:
            fold_w = wilcoxon(fdf['final_fold_pcc'], fdf['base_fold_pcc'], alternative='greater', zero_method='wilcox') if len(fdf) >= 2 else None
        except Exception:
            fold_w = None
        summary_rows.append({'dataset': dataset, 'num_seeds': int(len(sdf)), 'num_folds_total': int(len(fdf)), 'base_pcc_mean': float(sdf['base_pcc'].mean()), 'base_pcc_std': float(sdf['base_pcc'].std(ddof=0)), 'final_pcc_mean': float(sdf['final_pcc'].mean()), 'final_pcc_std': float(sdf['final_pcc'].std(ddof=0)), 'abs_gain_final_minus_base_mean': float((sdf['final_pcc'] - sdf['base_pcc']).mean()), 'pct_gain_final_minus_base_mean': float(pct_gain(sdf['final_pcc'].mean(), sdf['base_pcc'].mean())), 'min_seed_gain': float((sdf['final_pcc'] - sdf['base_pcc']).min()), 'max_seed_gain': float((sdf['final_pcc'] - sdf['base_pcc']).max()), 'mean_fold_gain': float((fdf['final_fold_pcc'] - fdf['base_fold_pcc']).mean()), 'seed_level_t_pvalue': None if seed_t is None or np.isnan(seed_t.pvalue) else float(seed_t.pvalue), 'fold_level_t_pvalue': None if fold_t is None or np.isnan(fold_t.pvalue) else float(fold_t.pvalue), 'fold_level_wilcoxon_pvalue': None if fold_w is None or np.isnan(fold_w.pvalue) else float(fold_w.pvalue), 'all_seed_gains_positive': bool(((sdf['final_pcc'] - sdf['base_pcc']) > 0).all()), 'all_seed_mean_fold_gains_positive': bool((fdf.groupby('seed')['fold_gain'].mean() > 0).all())})
    summary_df = pd.DataFrame(summary_rows)
    seed_df.to_csv(BASE_DIR / 'mutation_seed_results.csv', index=False, encoding='utf-8-sig')
    fold_df.to_csv(BASE_DIR / 'mutation_fold_results.csv', index=False, encoding='utf-8-sig')
    summary_df.to_csv(BASE_DIR / 'mutation_pcc_changes.csv', index=False, encoding='utf-8-sig')
    payload = {'config': asdict(config), 'summary': summary_rows}
    (BASE_DIR / 'mutation_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Strict repeated random 10x10 CV mutation benchmark')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    if args.seed is None and not args.aggregate:
        parser.error('Specify --seed or --aggregate')
    if args.seed is not None:
        print(json.dumps(run_one_seed(args.seed, save_predictions=args.save_predictions), indent=2, ensure_ascii=False))
    if args.aggregate:
        print(json.dumps(aggregate_results(), indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
