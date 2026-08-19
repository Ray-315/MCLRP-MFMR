from __future__ import annotations

import json
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1].parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCLRP_MFMR.t0_mfmr_protocol import run_leakage_safety_checks


def main() -> None:
    print(json.dumps(run_leakage_safety_checks(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
