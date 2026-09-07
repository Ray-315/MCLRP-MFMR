"""Generate supplementary tables directly from frozen experiment CSV files."""
from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Generate Supplementary Tables S13-S18 and branch figure panels from CSV.')
parser.add_argument('--results-dir', type=Path, default=ROOT/'results/additional_experiments')
parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/additional_supplementary')
args = parser.parse_args()
RESULTS = args.results_dir
DEST = args.output_dir
DEST.mkdir(parents=True, exist_ok=True)
NAMES = {'CCLE': 'CCLE', 'ERKAUC30': 'CGP ERK-AUC', 'ERKIC50': 'CGP ERK-IC50', 'PI3KAUC': 'CGP PI3K-AUC', 'PI3KIC50': 'CGP PI3K-IC50', 'PRISM19Q4_independent': 'PRISM 19Q4'}
METHODS = {'imputer_only': 'Imputer', 'ridge_only': 'Ridge', 'fixed_fusion': 'Fixed fusion', 'adaptive_fusion': 'Adaptive fusion', 'mfmr_base': 'MFMR', 'original_mclrp': 'MCLRP'}

def pm(values):
    values = np.asarray(values, dtype=float)
    return rf'${values.mean():.4f}\pm{values.std(ddof=1):.4f}$'

def table(number, caption, headers, rows, note):
    spec = 'l' + 'r' * (len(headers) - 1)
    head = ' & '.join(headers) + r' \\'
    text = [r'\begingroup\footnotesize', r'\setlength{\tabcolsep}{3pt}', rf'\begin{{longtable}}{{{spec}}}',
            rf'\caption{{{caption}}}\label{{stab:additional-{number}}}\\', r'\toprule', head, r'\midrule\endfirsthead',
            rf'\multicolumn{{{len(headers)}}}{{l}}{{Table S{number} continued}}\\', r'\toprule', head, r'\midrule\endhead',
            r'\bottomrule\endfoot']
    text += [' & '.join(map(str, row)) + r' \\' for row in rows]
    text += [r'\end{longtable}', note, r'\par\endgroup', '']
    (DEST / f'MCLRP-MFMR-Supp-TableS{number}.tex').write_text('\n'.join(text), encoding='utf-8')

adaptive = pd.read_csv(RESULTS / 'adaptive_fusion/raw_seed_metrics.csv')
rows = []
for dataset in NAMES:
    for method in ['imputer_only', 'ridge_only', 'fixed_fusion', 'adaptive_fusion']:
        frame = adaptive[(adaptive.dataset == dataset) & (adaptive.method == method)]
        assert len(frame) == 10
        rows.append([NAMES[dataset], METHODS[method]] + [pm(frame[k]) for k in ['pcc', 'rmse', 'mae']])
table(13, 'Branch and fusion performance under nested evaluation.', ['Dataset', 'Method', 'PCC', 'RMSE', 'MAE'], rows,
      r'Values are mean $\pm$ sample SD across ten seeds. Primary tasks use complete outer OOF predictions; PRISM uses the same locked test in every seed.')

alphas = pd.read_csv(RESULTS / 'adaptive_fusion/alpha_summary.csv').set_index('dataset')
rows = []
for dataset in NAMES:
    pair = adaptive[adaptive.dataset == dataset].pivot(index='seed', columns='method', values='pcc')
    delta = pair.adaptive_fusion - pair.fixed_fusion
    half = t.ppf(.975, 9) * delta.std(ddof=1) / np.sqrt(10)
    a = alphas.loc[dataset]
    rows.append([NAMES[dataset], rf'${a["mean"]:.3f}\pm{a["std"]:.3f}$', int(a['count']), pm(delta), rf'$[{delta.mean()-half:.4f},\ {delta.mean()+half:.4f}]$'])
table(14, 'Selected imputer weights and paired adaptive-fusion changes.', ['Dataset', r'Imputer weight $a$', 'Folds', r'$\Delta$PCC', r'95\% interval'], rows,
      r'$\Delta$PCC is adaptive minus fixed fusion. Weight SD describes outer-fold selections (100 per primary task; 10 for PRISM); PCC SD and paired Student-$t$ intervals use ten seed-level differences. PRISM intervals describe algorithmic variation conditional on its locked test.')

mask = pd.read_csv(RESULTS / 'mask_sensitivity/raw_seed_metrics.csv')
rows = []
for dataset in list(NAMES)[:5]:
    for ratio in [.1, .15, .2, .25, .3]:
        for method in ['original_mclrp', 'mfmr_base']:
            frame = mask[(mask.dataset == dataset) & np.isclose(mask.mask_ratio, ratio) & (mask.method == method)]
            assert len(frame) == 10
            rows.append([NAMES[dataset], f'{ratio*100:.0f}\\%', METHODS[method]] + [pm(frame[k]) for k in ['pcc', 'rmse', 'mae']])
table(15, 'Performance across fractions of masked observed responses.', ['Dataset', 'Masked', 'Method', 'PCC', 'RMSE', 'MAE'], rows,
      r'Mean $\pm$ sample SD over ten seeds. Each ratio uses a constrained holdout, shared by both methods. Ratios refer to originally observed entries, excluding pre-existing missing values.')

slopes = pd.read_csv(RESULTS / 'mask_sensitivity/degradation_slopes.csv')
rows = []
for dataset in list(NAMES)[:5]:
    for method in ['original_mclrp', 'mfmr_base']:
        frame = mask[(mask.dataset == dataset) & (mask.method == method)].pivot(index='seed', columns='mask_ratio', values='pcc')
        row = slopes[(slopes.dataset == dataset) & (slopes.method == method)].iloc[0]
        rows.append([NAMES[dataset], METHODS[method], pm(frame[.3]-frame[.1])] + [f'{row[k]*.1:.4f}' for k in ['pcc_slope', 'rmse_slope', 'mae_slope']])
table(16, r'Changes from 10\% to 30\% masking and fitted degradation slopes.', ['Dataset', 'Method', r'$\Delta$PCC', 'PCC slope', 'RMSE slope', 'MAE slope'], rows,
      r'$\Delta$PCC is 30\% minus 10\% for each seed (mean $\pm$ sample SD). Slopes are fitted to the five mean metric values and expressed per 10-percentage-point increase in masking. Slopes are descriptive estimates.')

branch = pd.read_csv(RESULTS / 'branch_complementarity/raw_seed_metrics.csv')
rows = []
metrics = [('prediction_pearson', 'Prediction Pearson'), ('prediction_spearman', 'Prediction Spearman'), ('error_pearson', 'Residual Pearson'), ('error_spearman', 'Residual Spearman'), ('imputer_better_share', 'Imputer win fraction'), ('ridge_better_share', 'Ridge win fraction'), ('tie_share', 'Tie fraction'), ('fusion_delta_pcc_vs_imputer', 'PCC: fusion minus imputer'), ('fusion_delta_pcc_vs_ridge', 'PCC: fusion minus ridge'), ('fusion_rmse_gain_vs_imputer', 'RMSE: imputer minus fusion'), ('fusion_rmse_gain_vs_ridge', 'RMSE: ridge minus fusion'), ('fusion_mae_gain_vs_imputer', 'MAE: imputer minus fusion'), ('fusion_mae_gain_vs_ridge', 'MAE: ridge minus fusion')]
for dataset in NAMES:
    frame = branch[branch.dataset == dataset]
    for key, name in metrics:
        v = frame[key]
        h = t.ppf(.975,9) * v.std(ddof=1) / np.sqrt(10)
        rows.append([NAMES[dataset], name, pm(v), rf'$[{v.mean()-h:.4f},\ {v.mean()+h:.4f}]$'])
table(17, 'Held-out branch correlations, entry-wise wins, and fusion gains.', ['Dataset', 'Quantity', r'Mean $\pm$ SD', r'95\% interval'], rows,
      r'Sample SD and Student-$t$ intervals use ten seeds. Residuals are observed minus predicted response. Wins compare absolute errors, with tie tolerance $10^{-6}$. Unbounded $t$ intervals for rare fractions can extend beyond zero; raw proportions remain within $[0,1]$.')

quartiles = pd.read_csv(RESULTS / 'branch_complementarity/raw_quartile_metrics.csv')
rows = []
for dataset in NAMES:
    for q in range(1,5):
        for method in ['imputer_only', 'ridge_only', 'fixed_fusion']:
            frame = quartiles[(quartiles.dataset == dataset) & (quartiles.quartile == q) & (quartiles.method == method)]
            rows.append([NAMES[dataset], str(q), METHODS[method]] + [pm(frame[k]) for k in ['pcc','rmse','mae']])
table(18, 'Performance by held-out prediction-disagreement quartile.', ['Dataset', 'Quartile', 'Method', 'PCC', 'RMSE', 'MAE'], rows,
      r'Mean $\pm$ sample SD across ten seeds. Quartile 1 contains the smallest absolute imputer--ridge prediction differences and quartile 4 the largest. Stable sorting divides held-out entries into four approximately equal-sized groups within each dataset and seed.')

for folder, names in {
    'adaptive_fusion': ['selected_alpha_distribution', 'adaptive_minus_fixed_delta_pcc'],
    'mask_sensitivity': ['mask_sensitivity_pcc', 'mask_sensitivity_rmse', 'mask_sensitivity_mae'],
    'branch_complementarity': ['branch_correlations_and_shares', 'disagreement_quartile_performance'],
}.items():
    for name in names:
        source_panel = RESULTS / folder / (name + '.pdf')
        if source_panel.exists():
            shutil.copy2(source_panel, DEST / ('MCLRP-MFMR-' + name.replace('_','-') + '.pdf'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
colors = {'fixed_fusion':'#1f5b99','imputer_only':'#d07929','ridge_only':'#438c70'}
fig, axes = plt.subplots(2,3,figsize=(9,5.5),layout='constrained')
for axis,dataset in zip(axes.flat,NAMES):
    for method in ['fixed_fusion','imputer_only','ridge_only']:
        v = quartiles[(quartiles.dataset==dataset)&(quartiles.method==method)].groupby('quartile').pcc.agg(['mean','std'])
        axis.errorbar(v.index,v['mean'],yerr=v['std'],color=colors[method],marker='o',markersize=3,capsize=2,label=METHODS[method])
    axis.set(title=NAMES[dataset],xlabel='Disagreement quartile',ylabel='PCC',xticks=[1,2,3,4])
    axis.spines[['top','right']].set_visible(False)
axes[0,0].legend(fontsize=7,frameon=False)
fig.savefig(DEST/'MCLRP-MFMR-disagreement-quartile-performance.pdf',bbox_inches='tight')
plt.close(fig)

means=branch.groupby('dataset').mean(numeric_only=True).reindex(NAMES)
x=np.arange(6)
fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
axes[0].bar(x-.18,means.prediction_pearson,.36,label='Prediction',color='#1f5b99')
axes[0].bar(x+.18,means.error_pearson,.36,label='Residual',color='#d07929')
axes[0].set(ylabel='Pearson correlation',ylim=(0,1.16))
axes[0].legend(loc='upper center',ncol=2,frameon=False,fontsize=8)
bottom=np.zeros(6)
for key,label,color in [('imputer_better_share','Imputer','#d07929'),('ridge_better_share','Ridge','#438c70'),('tie_share','Tie','#b6b6b6')]:
    axes[1].bar(x,means[key],bottom=bottom,label=label,color=color,width=.65)
    bottom+=means[key].to_numpy()
axes[1].set(ylabel='Held-out absolute-error win fraction',ylim=(0,1.2))
axes[1].legend(loc='upper center',ncol=3,frameon=False,fontsize=8)
for axis in axes:
    axis.set_xticks(x,list(NAMES.values()),rotation=35,ha='right')
    axis.spines[['top','right']].set_visible(False)
fig.savefig(DEST/'MCLRP-MFMR-branch-correlations-and-shares.pdf',bbox_inches='tight')
plt.close(fig)
print('Generated Tables S13-S18 and publication figure panels from frozen outputs.')
