# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Benchmarking results evaluation and visualization module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MIN_METRICS_FOR_CORRELATION = 2


try:
    from .plots import plot_comparison, plt
except ImportError:
    try:
        import matplotlib.pyplot as plt

        from .plots import plot_comparison
    except ImportError:
        plot_comparison = None
        plt = None


def aggregate_results(
    benchmark_results: dict[str, dict],
) -> pd.DataFrame:
    """Aggregate results from multiple benchmarks into a single DataFrame.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict

    Returns:
        DataFrame with benchmarks as rows and metrics as columns

    """
    df = pd.DataFrame.from_dict(benchmark_results, orient="index")
    if "num_samples" in df.columns:
        df = df.drop(columns=["num_samples"])
    return df


def compare_benchmarks(
    benchmark_results: dict[str, dict],
    metric: str | None = None,
) -> pd.DataFrame:
    """Compare benchmarks across all metrics or a specific one.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        metric: Optional specific metric to compare

    Returns:
        DataFrame for comparison

    """
    df = aggregate_results(benchmark_results)
    if metric and metric in df.columns:
        return df[[metric]].sort_values(metric, ascending=False)
    return df


def summarize_benchmark(
    benchmark_results: dict[str, dict],
) -> pd.DataFrame:
    """Generate a summary table of all benchmarks with statistics.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict

    Returns:
        DataFrame with stats for each metric

    """
    df = aggregate_results(benchmark_results)

    numeric_df = df.select_dtypes(include=["number"])

    return pd.DataFrame(
        {
            "mean": numeric_df.mean(),
            "std": numeric_df.std(),
            "min": numeric_df.min(),
            "max": numeric_df.max(),
            "median": numeric_df.median(),
        },
    )


def save_results(
    benchmark_results: dict[str, dict],
    filepath: str,
    file_format: str = "csv",
) -> None:
    """Save benchmark results to file.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        filepath: Output file path
        file_format: File format ('csv', 'xlsx', 'json')

    """
    df = aggregate_results(benchmark_results)

    if file_format == "csv":
        df.to_csv(filepath)
    elif file_format == "xlsx":
        df.to_excel(filepath)
    elif file_format == "json":
        df.to_json(filepath, indent=2)
    else:
        msg = f"Unsupported format: {file_format}"
        raise ValueError(msg)


def load_results(filepath: str) -> dict[str, dict]:
    """Load benchmark results from file.

    Args:
        filepath: Path to saved results file

    Returns:
        Dict mapping benchmark name to its result dict

    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, index_col=0)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath, index_col=0)
    elif filepath.endswith(".json"):
        df = pd.read_json(filepath, orient="index")
    else:
        msg = f"Unsupported file format: {filepath}"
        raise ValueError(msg)

    return df.to_dict(orient="index")


def rank_benchmarks(
    benchmark_results: dict[str, dict],
    metric: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank benchmarks by a specific metric.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        metric: Metric to rank by
        ascending: Sort order (True = lowest first)

    Returns:
        DataFrame with rankings

    """
    df = compare_benchmarks(benchmark_results, metric=metric)
    return df.sort_values(metric, ascending=ascending)


def best_performance(
    benchmark_results: dict[str, dict],
    metric: str,
) -> tuple[str, float]:
    """Get the best performing benchmark for a metric.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        metric: Metric to evaluate

    Returns:
        Tuple of (best_benchmark_name, best_score)

    """
    df = compare_benchmarks(benchmark_results, metric=metric)
    max_idx = df[metric].idxmax()
    return max_idx, df.loc[max_idx, metric]


def worst_performance(
    benchmark_results: dict[str, dict],
    metric: str,
) -> tuple[str, float]:
    """Get the worst performing benchmark for a metric.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        metric: Metric to evaluate

    Returns:
        Tuple of (worst_benchmark_name, worst_score)

    """
    df = compare_benchmarks(benchmark_results, metric=metric)
    min_idx = df[metric].idxmin()
    return min_idx, df.loc[min_idx, metric]


def plot_correlation_matrix(
    benchmark_results: dict[str, dict],
    figsize: tuple[int, int] = (10, 8),
    title: str = "Metric Correlation Matrix",
) -> plt.Figure:
    """Plot correlation matrix between different metrics.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        figsize: Figure size as (width, height)
        title: Plot title

    Returns:
        matplotlib Figure object if available, otherwise None

    """
    if plt is None:
        msg = (
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
        raise ImportError(
            msg,
        )

    df = aggregate_results(benchmark_results)

    numeric_df = df.select_dtypes(include=["number"])

    if len(numeric_df.columns) < MIN_METRICS_FOR_CORRELATION:
        msg = "Need at least 2 numeric metrics for correlation"
        raise ValueError(msg)

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index)

    ax.figure.colorbar(im, ax=ax)

    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    ax.set_title(title)
    fig.tight_layout()

    return fig


def get_all_metrics(benchmark_results: dict[str, dict]) -> list[str]:
    """Get all unique metrics across all benchmarks.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict

    Returns:
        List of metric names

    """
    metrics = set()
    for results in benchmark_results.values():
        metrics.update(results.keys())
    return sorted(metrics)


def filter_metrics(
    benchmark_results: dict[str, dict],
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, dict]:
    """Filter benchmarks to keep only specific metrics.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        include_patterns: List of patterns to include (None = all)
        exclude_patterns: List of patterns to exclude (None = none)

    Returns:
        Filtered results dict

    """
    filtered = {}

    for name, results in benchmark_results.items():
        filtered[name] = {}
        for metric, value in results.items():
            if include_patterns is not None and not any(
                p in metric for p in include_patterns
            ):
                continue
            if exclude_patterns is not None and any(
                p in metric for p in exclude_patterns
            ):
                continue
            filtered[name][metric] = value

    return filtered


def evaluate_suite(
    suite_results: dict[str, dict],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate benchmark suite and optionally save results.

    Args:
        suite_results: Dict mapping benchmark name to its result dict
        output_dir: Optional directory to save plots and reports

    Returns:
        DataFrame with all aggregated results

    """
    df = aggregate_results(suite_results)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        numeric_metrics = [c for c in df.columns if df[c].dtype in ["float64", "int64"]]

        if len(numeric_metrics) > 1 and plt is not None:
            fig = plot_comparison(
                suite_results,
                metrics=numeric_metrics[:5],
                title="Benchmark Suite Comparison",
            )
            fig.savefig(Path(output_dir) / "benchmark_comparison.png", dpi=300)
            plt.close(fig)

        summary_df = summarize_benchmark(suite_results)
        summary_df.to_csv(Path(output_dir) / "summary_statistics.csv")

    return df
