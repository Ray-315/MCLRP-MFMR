from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root from plot_figure6_mutation_network_validation")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import RESULTS_DIR
from plotting.scripts.plot_public_figure_utils import (
    clean_dir,
    default_explanation_root,
    default_plots_root,
    list_rendered_files,
    run_python_script,
    write_publish_manifest,
)


DEFAULT_TASKS = ["ERK_AUC", "ERK_IC50", "PI3K_AUC", "PI3K_IC50"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render renamed Figure 6 outputs from the mutation-network validation pipeline.")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(default_plots_root(6)))
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    return parser.parse_args()


def parse_tasks(raw: str) -> list[str]:
    tasks = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    return tasks or DEFAULT_TASKS.copy()


def renamed(name: str) -> str:
    return name.replace("fig5", "fig6")


def copy_if_exists(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def publish_task_group(explanation_root: Path, output_dir: Path, task_name: str, copied: list[str]) -> None:
    task_stem = task_name.lower()
    original_base = output_dir / "original" / task_stem
    grid_base = output_dir / "grid" / task_stem

    composite_dir = explanation_root / "composite"
    panels_dir = explanation_root / "panels"

    for ext in (".png", ".pdf", ".svg"):
        for stem in (
            f"fig5_{task_stem}_composite",
            f"fig5_{task_stem}_composite_polished",
            f"fig5_{task_stem}_composite_refined",
        ):
            copy_if_exists(composite_dir / f"{stem}{ext}", original_base / "composite" / f"{renamed(stem)}{ext}", copied)
        copy_if_exists(
            composite_dir / f"fig5_{task_stem}_equal_panel_grid{ext}",
            grid_base / "composite" / f"fig6_{task_stem}_equal_panel_grid{ext}",
            copied,
        )

        panel_prefixes = [
            f"fig5_{task_stem}_network",
            f"fig5_{task_stem}_network_polished",
            f"fig5_{task_stem}_two_panel_refined",
            f"fig5_network_two_panel_{task_stem}_polished",
            f"fig5_network_panelA_{task_stem}_mclrp_style_polished",
            f"fig5_network_panelB_{task_stem}_mclrp_style_polished",
        ]
        panel_prefixes.extend(f"fig5_{task_stem}_panel_{label}" for label in ["B", "C", "D", "E", "F", "G"])
        panel_prefixes.extend(f"fig5_{task_stem}_panel_{label}_polished" for label in ["B", "C", "D", "E", "F", "G"])
        for stem in panel_prefixes:
            copy_if_exists(panels_dir / f"{stem}{ext}", original_base / "panels" / f"{renamed(stem)}{ext}", copied)

    for suffix in (
        f"fig5_{task_stem}_equal_panel_grid_manifest.json",
        f"fig5_{task_stem}_composite_refined_manifest.json",
        f"fig5_{task_stem}_composite_polished_v2_final_manifest.json",
    ):
        copy_if_exists(explanation_root / suffix, original_base / renamed(suffix), copied)


def publish_common_tables(explanation_root: Path, output_dir: Path, copied: list[str]) -> None:
    tables_dir = explanation_root / "tables"
    target_dir = output_dir / "tables"
    for src in sorted(tables_dir.glob("*")):
        if src.is_file():
            copy_if_exists(src, target_dir / renamed(src.name), copied)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    explanation_root = default_explanation_root("fig5")
    clean_dir(explanation_root)

    build_script = PROJECT_ROOT / "scripts" / "main_figure" / "build_fig5_tables.py"
    rank_script = PROJECT_ROOT / "scripts" / "main_figure" / "rank_fig5_panels.py"
    base_panels_script = PROJECT_ROOT / "scripts" / "main_figure" / "plot_fig5_panels.py"
    polished_script = PROJECT_ROOT / "scripts" / "main_figure" / "plot_fig5_polished.py"
    sparse_network_script = PROJECT_ROOT / "scripts" / "main_figure" / "plot_fig5_network_mclrp_style.py"
    equal_grid_script = PROJECT_ROOT / "scripts" / "main_figure" / "compose_fig5_equal_panel_grid.py"

    run_python_script(build_script, "--results-root", str(results_root))
    run_python_script(rank_script, "--results-root", str(results_root))
    run_python_script(base_panels_script, "--results-root", str(results_root), "--stage", "all")
    run_python_script(polished_script, "--results-root", str(results_root), "--tasks", args.tasks)
    run_python_script(sparse_network_script, "--results-root", str(results_root), "--tasks", args.tasks)
    for task_name in parse_tasks(args.tasks):
        run_python_script(equal_grid_script, "--results-root", str(results_root), "--task", task_name, "--suffix", "equal_panel_grid")

    clean_dir(output_dir)
    copied: list[str] = []
    publish_common_tables(explanation_root, output_dir, copied)
    for task_name in parse_tasks(args.tasks):
        publish_task_group(explanation_root, output_dir, task_name, copied)
    manifest = write_publish_manifest(
        output_dir,
        {
            "figure": "fig6",
            "source": str(explanation_root),
            "tasks": parse_tasks(args.tasks),
            "copied_files": copied,
            "published_files": list_rendered_files(output_dir),
        },
        stem="fig6_publish",
    )
    print(json.dumps({"figure": "fig6", "output_dir": str(output_dir), "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
