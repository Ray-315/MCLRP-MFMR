from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "original_mclrp": "Original MCLRP",
    "imputer_only": "Imputer only",
    "ridge_only": "Ridge only",
    "mfmr_base": "MFMR base",
    "mfmr_mutation": "MFMR mutation residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PRISM independent-test LaTeX tables.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.results_dir / "method_summary.csv")
    bootstrap = pd.read_csv(args.results_dir / "paired_drug_cluster_bootstrap.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in summary.itertuples(index=False):
        label = METHOD_LABELS.get(row.method, str(row.method).replace("_", " "))
        rows.append(
            f"{label} & {row.overall_pcc_mean:.4f} $\\pm$ {row.overall_pcc_std:.5f} "
            f"& {row.rmse_mean:.4f} $\\pm$ {row.rmse_std:.5f} "
            f"& {row.mae_mean:.4f} $\\pm$ {row.mae_std:.5f} \\\\"
        )
    table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Performance on the locked PRISM 19Q4 test set. Values are mean $\\pm$ SD across ten algorithm seeds evaluated on the same 2,265 test entries.}",
            "\\label{tab:prism-independent-performance}",
            "\\small",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Method & PCC & RMSE & MAE \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (args.output_dir / "MCLRP-MFMR-Supp-TableS11.tex").write_text(table, encoding="utf-8")

    delta_rows = []
    for row in bootstrap.itertuples(index=False):
        label = METHOD_LABELS.get(row.comparator, str(row.comparator).replace("_", " "))
        delta_rows.append(
            f"{label} & {row.delta_pcc:.4f} [{row.delta_pcc_ci_low:.4f}, {row.delta_pcc_ci_high:.4f}] "
            f"& {row.delta_rmse:.4f} [{row.delta_rmse_ci_low:.4f}, {row.delta_rmse_ci_high:.4f}] "
            f"& {row.delta_mae:.4f} [{row.delta_mae_ci_low:.4f}, {row.delta_mae_ci_high:.4f}] \\\\"
        )
    delta_table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Paired PRISM 19Q4 differences with drug-cluster bootstrap 95\\% intervals. PCC is MFMR base minus comparator; RMSE and MAE are comparator minus MFMR base, so positive values favour MFMR base.}",
            "\\label{tab:prism-independent-deltas}",
            "\\scriptsize",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Comparator & $\\Delta$PCC [95\\% CI] & $\\Delta$RMSE [95\\% CI] & $\\Delta$MAE [95\\% CI] \\\\",
            "\\midrule",
            *delta_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (args.output_dir / "MCLRP-MFMR-Supp-TableS12.tex").write_text(delta_table, encoding="utf-8")


if __name__ == "__main__":
    main()
