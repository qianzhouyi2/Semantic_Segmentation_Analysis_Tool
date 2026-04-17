from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

try:
    from scipy import stats as scipy_stats
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in import tests.
    scipy_stats = None

from src.reporting.exporter import write_csv, write_json, write_markdown


DEFAULT_METRICS = ("sample_miou", "pixel_accuracy", "sample_dice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two per-sample robustness CSV files with paired statistics.")
    parser.add_argument("--baseline", required=True, help="Baseline per_sample_metrics.csv")
    parser.add_argument("--candidate", required=True, help="Candidate per_sample_metrics.csv")
    parser.add_argument("--output-dir", default="results/reports/per_sample_robustness", help="Output directory.")
    parser.add_argument(
        "--metrics",
        default="sample_miou,pixel_accuracy,sample_dice",
        help="Comma or space separated metric columns to compare.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of paired bootstrap resamples used for the mean-delta confidence interval.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap resampling.")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Bootstrap confidence level for the paired mean delta interval.",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance below which a paired delta counts as tied.",
    )
    parser.add_argument(
        "--stat-tests",
        choices=("none", "sign", "wilcoxon", "both"),
        default="both",
        help="Optional paired significance tests to include in the summary.",
    )
    parser.add_argument(
        "--allow-partial-overlap",
        action="store_true",
        help="Allow the comparison to proceed on filename intersections when the two CSVs do not cover the same samples.",
    )
    return parser.parse_args()


def parse_metric_names(raw: str) -> list[str]:
    metrics = [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]
    if not metrics:
        raise ValueError("No metric columns were requested.")
    return metrics


def load_per_sample_csv(path: Path) -> tuple[dict[str, dict[str, float | int | str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise ValueError(f"Per-sample CSV is missing the filename column: {path}")
        rows: dict[str, dict[str, float | int | str]] = {}
        for row in reader:
            filename = str(row["filename"])
            if filename in rows:
                raise ValueError(f"Duplicate filename {filename!r} in {path}")
            parsed: dict[str, float | int | str] = {"filename": filename}
            for key, value in row.items():
                if key == "filename" or value is None or value == "":
                    continue
                if key in {"sample_index", "valid_class_count"}:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows[filename] = parsed
    return rows, reader.fieldnames or []


def bootstrap_mean_delta_ci(
    deltas: np.ndarray,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    if deltas.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, deltas.size, size=(bootstrap_samples, deltas.size))
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(bootstrap_means, alpha / 2.0))
    upper = float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0))
    return lower, upper


def exact_sign_test_pvalue(num_positive: int, num_negative: int) -> float | None:
    num_trials = num_positive + num_negative
    if num_trials == 0:
        return None
    observed = min(num_positive, num_negative)
    cumulative = sum(math.comb(num_trials, k) for k in range(observed + 1))
    pvalue = min(1.0, 2.0 * cumulative / float(2**num_trials))
    return float(pvalue)


def summarize_metric(
    *,
    metric_name: str,
    filenames: list[str],
    baseline_rows: dict[str, dict[str, float | int | str]],
    candidate_rows: dict[str, dict[str, float | int | str]],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    tie_tolerance: float,
    stat_tests: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline_values = np.asarray([float(baseline_rows[filename][metric_name]) for filename in filenames], dtype=np.float64)
    candidate_values = np.asarray([float(candidate_rows[filename][metric_name]) for filename in filenames], dtype=np.float64)
    deltas = candidate_values - baseline_values

    improved = deltas > tie_tolerance
    worsened = deltas < -tie_tolerance
    tied = ~(improved | worsened)

    bootstrap_lower, bootstrap_upper = bootstrap_mean_delta_ci(
        deltas,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )

    sign_test_pvalue = None
    if stat_tests in {"sign", "both"}:
        sign_test_pvalue = exact_sign_test_pvalue(int(improved.sum()), int(worsened.sum()))

    wilcoxon_pvalue = None
    wilcoxon_statistic = None
    if stat_tests in {"wilcoxon", "both"} and scipy_stats is not None and np.any(~tied):
        wilcoxon_result = scipy_stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", method="auto")
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)
        wilcoxon_statistic = float(wilcoxon_result.statistic)

    summary = {
        "metric": metric_name,
        "n_paired_samples": len(filenames),
        "baseline_mean": float(baseline_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "improved_count": int(improved.sum()),
        "worsened_count": int(worsened.sum()),
        "tied_count": int(tied.sum()),
        "improved_fraction": float(improved.mean()),
        "worsened_fraction": float(worsened.mean()),
        "tied_fraction": float(tied.mean()),
        "bootstrap_ci_lower": bootstrap_lower,
        "bootstrap_ci_upper": bootstrap_upper,
        "sign_test_pvalue": sign_test_pvalue,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
    }

    paired_rows = [
        {
            "filename": filename,
            "metric": metric_name,
            "baseline_value": float(baseline_rows[filename][metric_name]),
            "candidate_value": float(candidate_rows[filename][metric_name]),
            "delta": float(candidate_rows[filename][metric_name]) - float(baseline_rows[filename][metric_name]),
        }
        for filename in filenames
    ]
    return summary, paired_rows


def build_aligned_markdown_table(rows: list[list[str]], *, right_align: set[int]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(row))) + " |"

    def separator_cell(index: int) -> str:
        width = max(3, widths[index])
        if index in right_align:
            return "-" * (width - 1) + ":"
        return "-" * width

    separator = "| " + " | ".join(separator_cell(index) for index in range(len(widths))) + " |"
    return [format_row(rows[0]), separator, *(format_row(row) for row in rows[1:])]


def format_num(value: float | None, precision: int = 4) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    metric_names = parse_metric_names(args.metrics)

    baseline_rows, baseline_fieldnames = load_per_sample_csv(baseline_path)
    candidate_rows, candidate_fieldnames = load_per_sample_csv(candidate_path)

    baseline_filenames = set(baseline_rows.keys())
    candidate_filenames = set(candidate_rows.keys())
    common_filenames = sorted(baseline_filenames & candidate_filenames)
    baseline_only = sorted(baseline_filenames - candidate_filenames)
    candidate_only = sorted(candidate_filenames - baseline_filenames)

    if (baseline_only or candidate_only) and not args.allow_partial_overlap:
        raise ValueError(
            "Baseline and candidate per-sample CSVs do not contain the same filenames. "
            "Use --allow-partial-overlap to compare on the intersection only."
        )
    if not common_filenames:
        raise ValueError("No overlapping filenames were found between baseline and candidate per-sample CSVs.")

    available_metrics = set(baseline_fieldnames) & set(candidate_fieldnames)
    selected_metrics = [metric for metric in metric_names if metric in available_metrics]
    missing_metrics = [metric for metric in metric_names if metric not in available_metrics]
    if not selected_metrics:
        raise ValueError(
            "None of the requested metrics are present in both per-sample CSVs. "
            f"Requested={metric_names}, common={sorted(available_metrics)}"
        )

    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for metric_index, metric_name in enumerate(selected_metrics):
        summary_row, metric_paired_rows = summarize_metric(
            metric_name=metric_name,
            filenames=common_filenames,
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.seed + metric_index,
            tie_tolerance=args.tie_tolerance,
            stat_tests=args.stat_tests,
        )
        summary_rows.append(summary_row)
        paired_rows.extend(metric_paired_rows)

    output_dir = Path(args.output_dir)
    payload = {
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "n_baseline_samples": len(baseline_rows),
        "n_candidate_samples": len(candidate_rows),
        "n_paired_samples": len(common_filenames),
        "baseline_only_count": len(baseline_only),
        "candidate_only_count": len(candidate_only),
        "baseline_only_preview": baseline_only[:10],
        "candidate_only_preview": candidate_only[:10],
        "selected_metrics": selected_metrics,
        "missing_metrics": missing_metrics,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_level": args.confidence_level,
        "tie_tolerance": args.tie_tolerance,
        "stat_tests": args.stat_tests,
        "rows": summary_rows,
    }

    write_json(output_dir / "paired_robustness_stats.json", payload)
    write_csv(output_dir / "paired_robustness_stats.csv", summary_rows)
    write_csv(output_dir / "paired_sample_deltas.csv", paired_rows)

    table_rows = [[
        "metric",
        "baseline_mean",
        "candidate_mean",
        "mean_delta",
        "median_delta",
        "improved",
        "worsened",
        "tied",
        "bootstrap_ci",
        "sign_p",
        "wilcoxon_p",
    ]]
    for row in summary_rows:
        table_rows.append(
            [
                row["metric"],
                format_num(float(row["baseline_mean"])),
                format_num(float(row["candidate_mean"])),
                format_num(float(row["mean_delta"])),
                format_num(float(row["median_delta"])),
                str(row["improved_count"]),
                str(row["worsened_count"]),
                str(row["tied_count"]),
                f"[{format_num(row['bootstrap_ci_lower'])}, {format_num(row['bootstrap_ci_upper'])}]",
                format_num(row["sign_test_pvalue"]),
                format_num(row["wilcoxon_pvalue"]),
            ]
        )

    table_lines = build_aligned_markdown_table(
        table_rows,
        right_align={1, 2, 3, 4, 5, 6, 7, 9, 10},
    )
    write_markdown(
        output_dir / "paired_robustness_stats.md",
        "Paired Per-Sample Robustness Comparison",
        [
            f"- baseline: {baseline_path.resolve()}",
            f"- candidate: {candidate_path.resolve()}",
            f"- n_baseline_samples: {len(baseline_rows)}",
            f"- n_candidate_samples: {len(candidate_rows)}",
            f"- n_paired_samples: {len(common_filenames)}",
            f"- selected_metrics: {', '.join(selected_metrics)}",
            f"- missing_metrics: {', '.join(missing_metrics) if missing_metrics else '<none>'}",
            f"- stat_tests: {args.stat_tests}",
            "",
            *table_lines,
        ],
    )
    print(output_dir / "paired_robustness_stats.json")


if __name__ == "__main__":
    main()
