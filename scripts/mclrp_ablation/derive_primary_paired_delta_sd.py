"""Recover paired-difference SDs from archived paired t-test summaries.

The final-lock archive stores the paired mean difference, two-sided paired
t-test P value, and number of seed pairs. These quantities uniquely determine
the SD of the paired differences. Population SD is reported to match the
convention used by the manuscript's main performance table.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scipy.stats import t as student_t


DATASET_ORDER = ("CCLE", "ERKAUC30", "ERKIC50", "PI3KAUC", "PI3KIC50")
METRICS = ("overall_pcc", "rmse", "mae")


def paired_population_sd(mean_delta: float, p_value: float, n_pairs: int) -> float:
    if n_pairs <= 1:
        raise ValueError("At least two paired observations are required.")
    if not 0.0 < p_value < 1.0:
        raise ValueError(f"Invalid two-sided t-test P value: {p_value}")
    t_abs = float(student_t.isf(p_value / 2.0, df=n_pairs - 1))
    if not math.isfinite(t_abs) or t_abs <= 0.0:
        raise ValueError(f"Could not recover a finite t statistic from P={p_value}")
    sample_sd = abs(mean_delta) * math.sqrt(n_pairs) / t_abs
    return sample_sd * math.sqrt((n_pairs - 1) / n_pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistical-tests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.statistical_tests.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    indexed = {
        (row["dataset"], row["metric"]): row
        for row in rows
        if row["method"] == "original_mclrp"
        and row["reference_method"] == "mfmr_base"
        and row["metric"] in METRICS
    }

    output_rows: list[dict[str, str]] = []
    for dataset in DATASET_ORDER:
        record: dict[str, str] = {"dataset": dataset}
        for metric in METRICS:
            row = indexed.get((dataset, metric))
            if row is None:
                raise ValueError(f"Missing paired test for {dataset}/{metric}")
            n_pairs = int(row["n_pairs"])
            archived_mean = float(row["mean_delta"])
            p_value = float(row["paired_t_pvalue"])
            reported_mean = -archived_mean if metric == "overall_pcc" else archived_mean
            record[f"{metric}_mean"] = f"{reported_mean:.17g}"
            record[f"{metric}_sd_population"] = f"{paired_population_sd(archived_mean, p_value, n_pairs):.17g}"
            record[f"{metric}_paired_t_pvalue"] = f"{p_value:.17g}"
            record[f"{metric}_n_pairs"] = str(n_pairs)
        output_rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
