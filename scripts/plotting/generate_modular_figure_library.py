import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in CURRENT_FILE.parents if (parent / "project_paths.py").exists()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root from plotting script")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import RESULTS_DIR


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    spec.loader.exec_module(module)
    return module


BASE = load_module("plot_requested_benchmark_figures_modular_base", PROJECT_ROOT / "plotting" / "scripts" / "plot_requested_benchmark_figures.py")
AGG = load_module("aggregate_full_benchmark_suite_modular_base", PROJECT_ROOT / "scripts" / "benchmarks" / "full_suite" / "aggregate_full_benchmark_suite.py")


MODULE_LAYOUT = {
    "01_overview": "overview",
    "02_dataset_task": "dataset_task",
    "03_drug_level": "drug_level",
    "04_uplift": "uplift",
    "05_ablation": "ablation",
    "06_efficiency": "efficiency",
}

DATASET_PATH_ORDER = [
    ("CCLE", "CCLE"),
    ("GDSC", "ERK_AUC"),
    ("GDSC", "ERK_IC50"),
    ("GDSC", "PI3K_AUC"),
    ("GDSC", "PI3K_IC50"),
    ("CGP", "ERK_AUC"),
    ("CGP", "ERK_IC50"),
    ("CGP", "PI3K_AUC"),
    ("CGP", "PI3K_IC50"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a modular benchmark figure library with supplementary figures.")
    parser.add_argument("--input_dir", type=str, default=str(RESULTS_DIR), help="Root results directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "plotting" / "outputs" / "modular_figure_library"),
        help="Root output directory for modular figures.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing modular output directory before rendering.")
    return parser.parse_args()


def prepare_output_root(output_root: Path, overwrite: bool) -> dict[str, Path]:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    module_dirs = {}
    for folder_name, _ in MODULE_LAYOUT.items():
        path = output_root / folder_name
        path.mkdir(parents=True, exist_ok=True)
        module_dirs[folder_name] = path
    (module_dirs["03_drug_level"] / "current_task_panels").mkdir(parents=True, exist_ok=True)
    (module_dirs["04_uplift"] / "current_task_panels").mkdir(parents=True, exist_ok=True)
    return module_dirs


def discover_and_load(results_root: Path) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    source_paths = BASE.discover_sources(results_root)
    source_paths = {key: BASE.require_file(path) for key, path in source_paths.items()}

    overall_df = pd.read_csv(source_paths["overall"])
    task_df = pd.read_csv(source_paths["task_summary"])
    all_per_drug_df = BASE.prepare_all_per_drug_df(pd.read_csv(source_paths["all_per_drug"]))
    dataset_df = BASE.load_dataset_average_tables(source_paths)
    uplift_df = BASE.prepare_uplift_df(pd.read_csv(source_paths["uplift"]))
    uplift_summary_df = pd.read_csv(source_paths["uplift_summary"])
    ablation_summary_raw = pd.read_csv(source_paths["ablation_summary"])
    ablation_summary_df, ablation_aggregated_df = BASE.prepare_ablation_summary(ablation_summary_raw)
    ablation_seed_df = pd.read_csv(source_paths["ablation_seed"])
    ablation_fold_df = pd.read_csv(source_paths["ablation_fold"])
    return source_paths, {
        "overall_df": overall_df,
        "task_df": task_df,
        "all_per_drug_df": all_per_drug_df,
        "dataset_df": dataset_df,
        "uplift_df": uplift_df,
        "uplift_summary_df": uplift_summary_df,
        "ablation_summary_df": ablation_summary_df,
        "ablation_aggregated_df": ablation_aggregated_df,
        "ablation_seed_df": ablation_seed_df,
        "ablation_fold_df": ablation_fold_df,
    }


def load_runtime_table(benchmark_root: Path, task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, dataset in DATASET_PATH_ORDER:
        if group == "CCLE":
            metadata_path = benchmark_root / "CCLE" / "run_metadata.json"
            dataset_label = "CCLE"
            dataset_root = "CCLE"
        else:
            metadata_path = benchmark_root / group / dataset / "run_metadata.json"
            dataset_label = f"{group}-{dataset}"
            dataset_root = group
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        runtimes = metadata.get("model_runtimes_sec", {})
        for model, runtime_sec in runtimes.items():
            rows.append(
                {
                    "dataset_label": dataset_label,
                    "dataset_root": dataset_root,
                    "dataset": dataset,
                    "model": model,
                    "runtime_sec": float(runtime_sec),
                }
            )
    runtime_df = pd.DataFrame(rows)
    merge_cols = ["dataset_label", "model", "mean_pcc", "mean_scc", "num_best_pcc", "num_best_scc"]
    runtime_df = runtime_df.merge(task_df.loc[:, merge_cols], on=["dataset_label", "model"], how="left")
    return runtime_df


def copy_current_detail_panels(benchmark_root: Path, module_dirs: dict[str, Path]) -> list[Path]:
    copied_paths: list[Path] = []
    drug_root = module_dirs["03_drug_level"] / "current_task_panels"
    uplift_root = module_dirs["04_uplift"] / "current_task_panels"

    for group, dataset in DATASET_PATH_ORDER:
        if group == "CCLE":
            src_dir = benchmark_root / "CCLE"
            dst_dir = drug_root / "CCLE"
        else:
            src_dir = benchmark_root / group / dataset
            dst_dir = drug_root / group / dataset
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in src_dir.glob("*.png"):
            target = dst_dir / path.name
            shutil.copy2(path, target)
            copied_paths.append(target)

    uplift_root.mkdir(parents=True, exist_ok=True)
    for path in (benchmark_root / "UPLIFT_MFMR_vs_MCLRP").glob("*.png"):
        target = uplift_root / path.name
        shutil.copy2(path, target)
        copied_paths.append(target)
    return copied_paths


def collect_pngs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.png") if path.is_file())


def plot_best_pcc_stacked_barh(task_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = BASE.build_model_palette(model_order)
    pivot = (
        task_df.pivot(index="dataset_label", columns="model", values="num_best_pcc")
        .reindex(index=BASE.SUBTASK_LABEL_ORDER, columns=model_order)
        .fillna(0.0)
    )
    totals = pivot.sum(axis=1)
    y_pos = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    left = np.zeros(len(pivot), dtype=float)
    for model in model_order:
        values = pivot[model].to_numpy(dtype=float)
        bars = ax.barh(
            y_pos,
            values,
            left=left,
            color=palette[model],
            edgecolor="white",
            linewidth=0.8,
            height=0.72,
            label=BASE.MODEL_DISPLAY_NAMES.get(model, model),
        )
        for bar, value in zip(bars, values):
            if value >= 1.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(round(value))}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=BASE.DARK_GRAY,
                )
        left += values

    for ypos, total in zip(y_pos, totals.to_numpy(dtype=float)):
        ax.text(total + 0.35, ypos, f"n={int(round(total))}", va="center", ha="left", fontsize=8.5, color=BASE.DARK_GRAY)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([BASE.format_dataset_label_short(label) for label in pivot.index])
    ax.invert_yaxis()
    ax.set_xlabel("Number of drugs with best PCC")
    ax.set_ylabel("")
    ax.set_title("Best-PCC Drug Count Composition by Dataset")
    BASE.style_axis(ax, grid_axis="x")
    fig.legend(
        handles=[Patch(facecolor=palette[m], edgecolor="white", label=BASE.MODEL_DISPLAY_NAMES.get(m, m)) for m in model_order],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.20, right=0.96, top=0.90, bottom=0.16)
    return BASE.save_figure(fig, output_dir / "figure_best_pcc_stacked_barh")


def plot_dataset_rank_dotplot(task_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = BASE.build_model_palette(model_order)
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 13.8), sharex=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, dataset_label in zip(axes, BASE.SUBTASK_LABEL_ORDER):
        panel = task_df.loc[task_df["dataset_label"] == dataset_label, ["model", "mean_pcc"]].copy()
        panel["model"] = pd.Categorical(panel["model"], categories=model_order, ordered=True)
        panel = panel.sort_values("mean_pcc", ascending=True).reset_index(drop=True)
        y = np.arange(len(panel))

        ax.hlines(y, xmin=0.0, xmax=panel["mean_pcc"].to_numpy(dtype=float), color=BASE.LINE_BLUE, linewidth=1.5, zorder=1)
        ax.scatter(
            panel["mean_pcc"].to_numpy(dtype=float),
            y,
            s=88,
            color=[palette[model] for model in panel["model"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for xpos, ypos, value in zip(panel["mean_pcc"].to_numpy(dtype=float), y, panel["mean_pcc"].to_numpy(dtype=float)):
            ax.text(xpos + 0.012, ypos, f"{value:.3f}", va="center", ha="left", fontsize=7.6, color=BASE.DARK_GRAY)

        ax.set_yticks(y)
        ax.set_yticklabels([BASE.MODEL_DISPLAY_NAMES.get(model, model) for model in panel["model"]], fontsize=8)
        ax.set_xlim(0.0, max(0.85, float(task_df["mean_pcc"].max()) + 0.08))
        ax.set_xlabel("Mean PCC")
        ax.set_title(BASE.format_dataset_label_short(dataset_label))
        BASE.style_axis(ax, grid_axis="x")

    fig.suptitle("Model Ranking Dot Plot by Dataset", x=0.5, y=0.98, fontsize=BASE.TITLE_SIZE + 1)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06, wspace=0.30, hspace=0.36)
    return BASE.save_figure(fig, output_dir / "figure_dataset_rank_dotplot")


def plot_drug_delta_lollipop(uplift_df: pd.DataFrame, output_dir: Path) -> Path:
    root_palette = {"positive": BASE.DARK_TEAL, "negative": BASE.MID_ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(22.6, 8.8))
    axes = np.asarray(axes).reshape(-1)

    for ax, dataset_root in zip(axes, BASE.DATASET_ROOT_ORDER):
        panel = uplift_df.loc[uplift_df["dataset_root"] == dataset_root].copy()
        selected = pd.concat(
            [
                panel.nlargest(8, "delta_pcc"),
                panel.nsmallest(8, "delta_pcc"),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["display_drug"]).sort_values("delta_pcc", ascending=True)
        y = np.arange(len(selected))
        values = selected["delta_pcc"].to_numpy(dtype=float)
        colors = [root_palette["positive"] if value >= 0 else root_palette["negative"] for value in values]

        ax.hlines(y, xmin=0.0, xmax=values, color=BASE.LINE_BLUE, linewidth=2.0, zorder=1)
        ax.scatter(values, y, s=82, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        ax.axvline(0.0, color=BASE.DARK_GRAY, linewidth=0.9, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([BASE.truncate_label(text, max_len=32) for text in selected["display_drug"]], fontsize=7.8)
        ax.set_xlabel("Delta PCC")
        ax.set_title(dataset_root)
        BASE.style_axis(ax, grid_axis="x")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=root_palette["positive"], markerfacecolor=root_palette["positive"], markersize=8, label="Positive Delta PCC"),
        Line2D([0], [0], marker="o", linestyle="", color=root_palette["negative"], markerfacecolor=root_palette["negative"], markersize=8, label="Negative Delta PCC"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Top And Bottom Drug-Level Delta PCC by Dataset", x=0.5, y=0.98, fontsize=BASE.TITLE_SIZE + 1)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.12, wspace=0.35)
    return BASE.save_figure(fig, output_dir / "figure_drug_delta_pcc_lollipop")


def plot_baseline_vs_delta_bubble(uplift_df: pd.DataFrame, output_dir: Path) -> Path:
    root_palette = {"CCLE": BASE.DARK_BLUE, "GDSC": BASE.PRIMARY_BLUE, "CGP": BASE.MID_ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), sharey=True)

    for ax, dataset_root in zip(axes, BASE.DATASET_ROOT_ORDER):
        panel = uplift_df.loc[uplift_df["dataset_root"] == dataset_root].copy()
        n_obs = panel["n_obs"].to_numpy(dtype=float)
        size = 35.0 + 165.0 * (n_obs - n_obs.min()) / max(1.0, float(n_obs.max() - n_obs.min()))
        ax.scatter(
            panel["pcc_MCLRP"].to_numpy(dtype=float),
            panel["delta_pcc"].to_numpy(dtype=float),
            s=size,
            alpha=0.58,
            color=root_palette[dataset_root],
            edgecolor="white",
            linewidth=0.6,
        )
        rho = panel["pcc_MCLRP"].corr(panel["delta_pcc"], method="spearman")
        ax.axhline(0.0, color=BASE.DARK_GRAY, linewidth=0.9, linestyle="--")
        ax.set_title(dataset_root)
        ax.set_xlabel("Baseline MCLRP PCC")
        ax.set_ylabel("Delta PCC")
        ax.text(
            0.03,
            0.96,
            f"Spearman rho = {rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=BASE.DARK_GRAY,
        )
        BASE.style_axis(ax, grid_axis="both")

    fig.suptitle("Baseline Performance vs Improvement Bubble Plot", x=0.5, y=0.98, fontsize=BASE.TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.12)
    return BASE.save_figure(fig, output_dir / "figure_baseline_vs_delta_bubble")


def plot_delta_boxen(uplift_df: pd.DataFrame, output_dir: Path) -> Path:
    root_palette = {"CCLE": BASE.DARK_BLUE, "GDSC": BASE.PRIMARY_BLUE, "CGP": BASE.MID_ORANGE}
    work = uplift_df.copy()
    order = (
        work.groupby("dataset_label", as_index=False)["delta_pcc"]
        .mean()
        .sort_values("delta_pcc", ascending=False)["dataset_label"]
        .tolist()
    )
    palette = {label: root_palette[BASE.dataset_root_from_label(label)] for label in order}

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 5.8), sharey=False)
    for ax, metric_col, title in zip(axes, ["delta_pcc", "delta_scc"], ["Delta PCC Distribution", "Delta SCC Distribution"]):
        sns.boxenplot(
            data=work,
            x="dataset_label",
            y=metric_col,
            order=order,
            hue="dataset_label",
            dodge=False,
            palette=palette,
            linewidth=0.8,
            k_depth="trustworthy",
            ax=ax,
        )
        sns.stripplot(
            data=work,
            x="dataset_label",
            y=metric_col,
            order=order,
            color=BASE.DARK_GRAY,
            size=2.2,
            alpha=0.28,
            jitter=0.16,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.axhline(0.0, color=BASE.DARK_GRAY, linewidth=0.9, linestyle="--")
        ax.set_xticklabels([BASE.format_dataset_label_short(label) for label in order], rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel(metric_col.replace("delta_", "").upper())
        ax.set_title(title)
        BASE.style_axis(ax, grid_axis="y")

    fig.suptitle("Task-Wise Uplift Distributions", x=0.5, y=0.98, fontsize=BASE.TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.15)
    return BASE.save_figure(fig, output_dir / "figure_delta_boxen")


def plot_runtime_vs_pcc_bubble(runtime_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = BASE.build_model_palette(model_order)
    summary = (
        runtime_df.groupby("model", as_index=False)
        .agg(mean_runtime_sec=("runtime_sec", "mean"), mean_pcc=("mean_pcc", "mean"), total_best_pcc=("num_best_pcc", "sum"))
        .set_index("model")
        .reindex(model_order)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    sizes = 110.0 + 18.0 * summary["total_best_pcc"].to_numpy(dtype=float)
    ax.scatter(
        summary["mean_runtime_sec"].to_numpy(dtype=float),
        summary["mean_pcc"].to_numpy(dtype=float),
        s=sizes,
        color=[palette[model] for model in summary["model"]],
        edgecolor="white",
        linewidth=0.9,
        alpha=0.92,
    )
    for _, row in summary.iterrows():
        ax.text(
            float(row["mean_runtime_sec"]) * 1.06,
            float(row["mean_pcc"]) + 0.002,
            BASE.MODEL_DISPLAY_NAMES.get(row["model"], row["model"]),
            fontsize=8.5,
            color=BASE.DARK_GRAY,
            ha="left",
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per dataset (sec, log scale)")
    ax.set_ylabel("Mean PCC")
    ax.set_title("Efficiency Frontier: Runtime vs Mean PCC")
    BASE.style_axis(ax, grid_axis="both")
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.12, right=0.96)
    return BASE.save_figure(fig, output_dir / "figure_runtime_vs_pcc_bubble")


def plot_runtime_distribution_boxen(runtime_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> Path:
    palette = BASE.build_model_palette(model_order)
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    sns.boxenplot(
        data=runtime_df,
        x="model",
        y="runtime_sec",
        order=model_order,
        palette=[palette[model] for model in model_order],
        linewidth=0.8,
        k_depth="trustworthy",
        ax=ax,
    )
    sns.stripplot(
        data=runtime_df,
        x="model",
        y="runtime_sec",
        order=model_order,
        color=BASE.DARK_GRAY,
        size=2.4,
        alpha=0.35,
        jitter=0.14,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xticklabels([BASE.MODEL_DISPLAY_NAMES.get(model, model) for model in model_order], rotation=35, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Runtime per dataset (sec, log scale)")
    ax.set_title("Runtime Distribution Across Datasets")
    BASE.style_axis(ax, grid_axis="y")
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.10, right=0.97)
    return BASE.save_figure(fig, output_dir / "figure_runtime_boxen")


def plot_ablation_delta_dotplot(aggregated_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8), sharex=False)
    specs = [
        (axes[0], "shared_main", BASE.SHARED_VARIANT_ORDER, BASE.SHARED_VARIANT_LABELS, "Shared Backbone Delta vs Full"),
        (axes[1], "cgp_main", BASE.CGP_VARIANT_ORDER, BASE.CGP_VARIANT_LABELS, "Mutation Head Delta vs Full"),
    ]

    for ax, group_name, order, label_map, title in specs:
        palette = BASE.get_variant_palette(group_name)
        panel = aggregated_df.loc[aggregated_df["group"] == group_name].set_index("variant_id").reindex(order).reset_index()
        panel = panel.loc[panel["variant_id"] != order[-1]].copy()
        panel = panel.sort_values("delta_vs_full", ascending=True).reset_index(drop=True)
        y = np.arange(len(panel))
        values = panel["delta_vs_full"].to_numpy(dtype=float)
        ax.hlines(y, xmin=0.0, xmax=values, color=BASE.LINE_BLUE, linewidth=1.8)
        ax.scatter(values, y, s=88, color=[palette[variant] for variant in panel["variant_id"]], edgecolor="white", linewidth=0.8, zorder=3)
        ax.axvline(0.0, color=BASE.DARK_GRAY, linewidth=0.9, linestyle="--")
        for xpos, ypos in zip(values, y):
            offset = 0.004 if xpos >= 0 else -0.004
            ha = "left" if xpos >= 0 else "right"
            ax.text(xpos + offset, ypos, f"{xpos:+.3f}", va="center", ha=ha, fontsize=8, color=BASE.DARK_GRAY)
        ax.set_yticks(y)
        ax.set_yticklabels([label_map[variant] for variant in panel["variant_id"]])
        ax.set_xlabel("Delta PCC vs Full")
        ax.set_title(title)
        BASE.style_axis(ax, grid_axis="x")

    fig.suptitle("Ablation Delta Dot Plot", x=0.5, y=0.98, fontsize=BASE.TITLE_SIZE + 1)
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.20)
    return BASE.save_figure(fig, output_dir / "figure_ablation_delta_dotplot")


def write_module_index(output_root: Path, module_dirs: dict[str, Path]) -> None:
    lines = ["# Modular Figure Library", ""]
    for folder_name, module_name in MODULE_LAYOUT.items():
        lines.append(f"## {folder_name} {module_name}")
        for path in sorted(module_dirs[folder_name].rglob("*.png")):
            rel_path = path.relative_to(output_root)
            lines.append(f"- {rel_path.as_posix()}")
        lines.append("")
    (output_root / "module_index.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_root: Path, module_dirs: dict[str, Path], source_paths: dict[str, Path]) -> None:
    payload = {
        "modules": {
            folder_name: [str(path.relative_to(output_root)) for path in sorted(module_dirs[folder_name].rglob("*.png"))]
            for folder_name in module_dirs
        },
        "sources": {key: str(value) for key, value in source_paths.items()},
    }
    (output_root / "module_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    module_dirs = prepare_output_root(output_root, overwrite=args.overwrite)

    BASE.set_publication_style()
    source_paths, data = discover_and_load(results_root)
    benchmark_root = results_root / "latest" / "benchmark_10x10_mfmr_best_fullcv"
    model_order = BASE.build_model_order(data["overall_df"])
    runtime_df = load_runtime_table(benchmark_root, data["task_df"])

    BASE.plot_overall_model_comparison(data["overall_df"], model_order, module_dirs["01_overview"])
    BASE.plot_dataset_model_comparison(data["dataset_df"], model_order, module_dirs["01_overview"])
    BASE.plot_num_best(data["task_df"], model_order, module_dirs["01_overview"])
    BASE.plot_dataset_best_pcc_donuts(data["task_df"], model_order, module_dirs["01_overview"])
    plot_best_pcc_stacked_barh(data["task_df"], model_order, module_dirs["01_overview"])

    AGG.plot_all_drugs_mean(data["overall_df"], module_dirs["01_overview"])
    AGG.plot_dataset_heatmaps(data["task_df"], module_dirs["02_dataset_task"])
    AGG.plot_dataset_bargrids(data["task_df"], module_dirs["02_dataset_task"])
    BASE.plot_task_model_comparison(data["task_df"], model_order, module_dirs["02_dataset_task"])
    BASE.plot_mclrp_vs_mfmr_subtask_mean(data["task_df"], module_dirs["02_dataset_task"])
    plot_dataset_rank_dotplot(data["task_df"], model_order, module_dirs["02_dataset_task"])

    for dataset_root in BASE.DATASET_ROOT_ORDER:
        BASE.plot_per_drug_pairwise(data["uplift_df"], dataset_root, module_dirs["03_drug_level"])
        BASE.plot_per_drug_heatmap(data["all_per_drug_df"], dataset_root, model_order, module_dirs["03_drug_level"])
    plot_drug_delta_lollipop(data["uplift_df"], module_dirs["03_drug_level"])
    plot_baseline_vs_delta_bubble(data["uplift_df"], module_dirs["03_drug_level"])
    copy_current_detail_panels(benchmark_root, module_dirs)

    BASE.plot_top5_uplift(data["uplift_df"], module_dirs["04_uplift"])
    BASE.plot_delta_distribution(data["uplift_df"], data["uplift_summary_df"], module_dirs["04_uplift"])
    BASE.plot_nobs_vs_pcc(data["all_per_drug_df"], module_dirs["04_uplift"])
    plot_delta_boxen(data["uplift_df"], module_dirs["04_uplift"])

    BASE.plot_ablation_bar(data["ablation_aggregated_df"], module_dirs["05_ablation"])
    BASE.plot_ablation_heatmap(data["ablation_summary_df"], module_dirs["05_ablation"])
    BASE.plot_ablation_waterfall(data["ablation_aggregated_df"], module_dirs["05_ablation"])
    BASE.plot_ablation_seed_stability(data["ablation_seed_df"], module_dirs["05_ablation"])
    BASE.plot_ablation_boxplot(data["ablation_fold_df"], module_dirs["05_ablation"])
    plot_ablation_delta_dotplot(data["ablation_aggregated_df"], module_dirs["05_ablation"])

    plot_runtime_vs_pcc_bubble(runtime_df, model_order, module_dirs["06_efficiency"])
    plot_runtime_distribution_boxen(runtime_df, model_order, module_dirs["06_efficiency"])

    write_module_index(output_root, module_dirs)
    write_manifest(output_root, module_dirs, source_paths)

    payload = {folder_name: [str(path.relative_to(output_root)) for path in collect_pngs(module_dirs[folder_name])] for folder_name in module_dirs}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
