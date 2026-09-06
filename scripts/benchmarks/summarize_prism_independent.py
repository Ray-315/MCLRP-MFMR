from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MCLRP_MFMR.independent_t0 import paired_cluster_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the locked PRISM independent test.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference", default="mfmr_base")
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = np.load(args.bundle_dir / "bundle.npz", allow_pickle=True)
    split = np.load(args.results_dir / "locked_split.npz", allow_pickle=True)
    truth = bundle["M"].astype(np.float64)
    test_mask = split["test_mask"].astype(bool)

    paths = sorted((args.results_dir / "predictions").glob("*__seed_*.npz"))
    methods = sorted({path.name.split("__seed_", 1)[0] for path in paths})
    predictions: dict[str, np.ndarray] = {}
    for method in methods:
        method_paths = sorted((args.results_dir / "predictions").glob(f"{method}__seed_*.npz"))
        predictions[method] = np.mean(
            [np.load(path)["prediction"].astype(np.float64) for path in method_paths], axis=0
        )
    if args.reference not in predictions:
        raise ValueError(f"Missing reference predictions for {args.reference}")

    rows: list[dict[str, object]] = []
    for method in methods:
        if method == args.reference:
            continue
        result = paired_cluster_bootstrap(
            truth,
            predictions[args.reference],
            predictions[method],
            test_mask,
            iterations=args.iterations,
            seed=args.seed,
        )
        rows.append({"reference": args.reference, "comparator": method, **result})
    pd.DataFrame(rows).to_csv(args.results_dir / "paired_drug_cluster_bootstrap.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
