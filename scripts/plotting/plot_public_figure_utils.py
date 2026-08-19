from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root from plot_public_figure_utils")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import EXPLANATION_RESULTS_DIR, PLOTS_RESULTS_DIR, RESULTS_DIR


def clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_python_script(script: Path, *args: str, cwd: Path | None = None) -> None:
    command = [sys.executable, str(script), *args]
    subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), check=True)


def rewrite_prefix(name: str, old_prefix: str, new_prefix: str) -> str:
    if name.startswith(old_prefix):
        return new_prefix + name[len(old_prefix) :]
    return name


def mirror_tree_with_prefix(source_root: Path, dest_root: Path, *, old_prefix: str, new_prefix: str) -> list[str]:
    clean_dir(dest_root)
    copied: list[str] = []
    for src in sorted(source_root.rglob("*")):
        rel = src.relative_to(source_root)
        if "cache" in rel.parts:
            continue
        rewritten_parts = [rewrite_prefix(part, old_prefix, new_prefix) for part in rel.parts]
        dst = dest_root.joinpath(*rewritten_parts)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def list_rendered_files(root: Path) -> list[str]:
    return [str(path) for path in sorted(root.rglob("*")) if path.is_file()]


def write_publish_manifest(dest_root: Path, payload: dict[str, object], *, stem: str) -> Path:
    manifest = dest_root / f"{stem}_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def default_results_root() -> Path:
    return RESULTS_DIR


def default_explanation_root(fig_name: str) -> Path:
    return EXPLANATION_RESULTS_DIR / fig_name


def default_plots_root(fig_number: int) -> Path:
    return PLOTS_RESULTS_DIR / f"fig{fig_number}"
