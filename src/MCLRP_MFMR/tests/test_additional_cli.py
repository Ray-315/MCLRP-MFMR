from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('script', [
    'run_adaptive_fusion.py', 'run_mask_sensitivity.py',
    'analyze_branch_complementarity.py', 'aggregate_shards.py',
    'summarize_additional_experiments.py',
])
def test_cli_help_from_source_checkout(script):
    env = dict(os.environ, PYTHONPATH='')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/additional_experiments' / script), '--help'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
