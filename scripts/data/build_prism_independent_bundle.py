from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MCLRP_MFMR.prism_independent import build_prism_independent_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the locked PRISM 19Q4 independent-test bundle.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-drugs", type=int, default=24)
    parser.add_argument("--metric", choices=("auc", "ic50"), default="auc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_prism_independent_bundle(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        num_drugs=args.num_drugs,
        metric=args.metric,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
