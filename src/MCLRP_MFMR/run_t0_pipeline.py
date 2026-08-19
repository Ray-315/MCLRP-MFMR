from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCLRP_MFMR.paths import RESULTS_DIR, data_inventory
from MCLRP_MFMR.t0_mfmr_protocol import BASELINE_METHODS, DATASET_CHOICES, SUPPORTED_METHODS
from MCLRP_MFMR.t0_mfmr_protocol import MFMRConfig, MutationHeadConfig, run_t0_benchmark
from MCLRP_MFMR.visualize_t0_results import visualize_t0_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict T0 MCLRP-MFMR training, evaluation, and visualization.")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=["CCLE"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, default=["global_mean", "row_col_mean", "mfmr_base"])
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--topg-imp", type=int, default=2000)
    parser.add_argument("--comp-imp", type=int, default=12)
    parser.add_argument("--topg-ridge", type=int, default=2500)
    parser.add_argument("--comp-ridge", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--weight-imp", type=float, default=0.5)
    parser.add_argument("--weight-ridge", type=float, default=0.5)
    parser.add_argument("--imputer-max-iter", type=int, default=3)
    parser.add_argument("--min-train-per-drug", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--final-alpha", type=float, default=20.0)
    parser.add_argument("--mutation-latent-dim", type=int, default=24)
    parser.add_argument("--residual-inner-cv", type=int, default=0)
    parser.add_argument("--max-drugs", type=int, default=None)
    parser.add_argument("--max-cell-lines", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "t0_mfmr")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--show-data-inventory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_data_inventory:
        print(json.dumps({"data_inventory": data_inventory()}, indent=2, ensure_ascii=False))
    benchmark_result = None
    if not args.visualize_only:
        seeds = tuple(int(s) for s in args.seeds)
        cfg = MFMRConfig(
            num_folds=int(args.num_folds),
            seeds=seeds,
            topg_imp=int(args.topg_imp),
            comp_imp=int(args.comp_imp),
            topg_ridge=int(args.topg_ridge),
            comp_ridge=int(args.comp_ridge),
            ridge_alpha=float(args.ridge_alpha),
            weight_imp=float(args.weight_imp),
            weight_ridge=float(args.weight_ridge),
            imputer_max_iter=int(args.imputer_max_iter),
            min_train_per_drug=int(args.min_train_per_drug),
            random_state=int(args.random_state),
        )
        mut_cfg = MutationHeadConfig(
            enabled=any(m in {"mfmr_mutation", "mutation_no_pathway", "mutation_no_latent"} for m in args.methods),
            final_alpha=float(args.final_alpha),
            mutation_latent_dim=int(args.mutation_latent_dim),
            residual_inner_cv=int(args.residual_inner_cv),
        )
        benchmark_result = run_t0_benchmark(
            datasets=tuple(args.datasets),
            methods=tuple(args.methods),
            seeds=seeds,
            config=cfg,
            mutation_config=mut_cfg,
            output_dir=args.output_dir,
            save_predictions=bool(args.save_predictions),
            max_cell_lines=args.max_cell_lines,
            max_drugs=args.max_drugs,
        )
    figure_result = visualize_t0_results(results_dir=args.output_dir)
    print(json.dumps({"benchmark": benchmark_result, "figures": figure_result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
