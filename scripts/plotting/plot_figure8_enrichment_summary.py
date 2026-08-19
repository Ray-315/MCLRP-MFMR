from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root from plot_figure8_enrichment_summary")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import RESULTS_DIR
from plotting.scripts.plot_public_figure_utils import (
    default_explanation_root,
    default_plots_root,
    list_rendered_files,
    mirror_tree_with_prefix,
    run_python_script,
    write_publish_manifest,
)


DEFAULT_TASKS = ["ERK_AUC", "ERK_IC50", "PI3K_IC50", "PI3K_AUC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render renamed Figure 8 outputs from the enrichment summary pipeline.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(default_plots_root(8)))
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    explanation_root = default_explanation_root("fig7")

    build_script = PROJECT_ROOT / "scripts" / "main_figure" / "build_fig7_tables.py"
    polished_script = PROJECT_ROOT / "scripts" / "main_figure" / "plot_fig7_polished.py"
    task_variant_script = PROJECT_ROOT / "scripts" / "main_figure" / "render_fig7_task_variants.py"

    run_python_script(build_script, "--results-root", str(results_root), "--task", "PI3K_AUC")
    run_python_script(polished_script, "--results-root", str(results_root), "--stage", "all")
    run_python_script(task_variant_script, "--results-root", str(results_root), "--tasks", args.tasks)

    copied = mirror_tree_with_prefix(explanation_root, output_dir, old_prefix="fig7", new_prefix="fig8")
    manifest = write_publish_manifest(
        output_dir,
        {
            "figure": "fig8",
            "source": str(explanation_root),
            "tasks": [item.strip().upper() for item in args.tasks.split(",") if item.strip()],
            "copied_files": copied,
            "published_files": list_rendered_files(output_dir),
        },
        stem="fig8_publish",
    )
    print(json.dumps({"figure": "fig8", "output_dir": str(output_dir), "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
