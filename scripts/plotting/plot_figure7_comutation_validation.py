from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root from plot_figure7_comutation_validation")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import RESULTS_DIR
from plotting.scripts.plot_public_figure_utils import (
    clean_dir,
    default_explanation_root,
    default_plots_root,
    list_rendered_files,
    mirror_tree_with_prefix,
    run_python_script,
    write_publish_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render renamed Figure 7 outputs from the co-mutation validation pipeline.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(default_plots_root(7)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    explanation_root = default_explanation_root("fig6")
    clean_dir(explanation_root)

    build_script = PROJECT_ROOT / "scripts" / "main_figure" / "build_fig6_tables.py"
    base_panels_script = PROJECT_ROOT / "scripts" / "main_figure" / "plot_fig6_panels.py"
    polished_script = PROJECT_ROOT / "scripts" / "main_figure" / "plot_fig6_polished.py"
    extra_script = PROJECT_ROOT / "scripts" / "main_figure" / "plot_fig6_extra_panels.py"

    run_python_script(build_script, "--results-root", str(results_root))
    run_python_script(base_panels_script, "--results-root", str(results_root), "--stage", "all")
    run_python_script(polished_script, "--results-root", str(results_root), "--stage", "all")
    run_python_script(extra_script, "--results-root", str(results_root))

    copied = mirror_tree_with_prefix(explanation_root, output_dir, old_prefix="fig6", new_prefix="fig7")
    manifest = write_publish_manifest(
        output_dir,
        {
            "figure": "fig7",
            "source": str(explanation_root),
            "copied_files": copied,
            "published_files": list_rendered_files(output_dir),
        },
        stem="fig7_publish",
    )
    print(json.dumps({"figure": "fig7", "output_dir": str(output_dir), "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
