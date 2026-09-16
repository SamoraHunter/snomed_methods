# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Plotting functions for benchmark results."""

from typing import List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def plot_comparison(
    benchmark_results: dict,
    metrics: List[str] | None = None,
    figsize: tuple[int, int] = (12, 6),
    title: str = "Benchmark Comparison",
) -> "plt.Figure":
    """Create a bar chart comparison of benchmarks.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        metrics: List of metric names to plot (default: all numeric)
        figsize: Figure size as (width, height)
        title: Plot title

    Returns:
        matplotlib Figure object if available, otherwise None
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import pandas as pd

    df = pd.DataFrame.from_dict(benchmark_results, orient="index")
    if "num_samples" in df.columns:
        df = df.drop(columns=["num_samples"])

    if metrics is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        metrics = [c for c in numeric_cols if "p_value" not in c]

    if not metrics:
        raise ValueError("No numeric metrics found to plot")

    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(df))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            offset = (i - len(metrics) / 2) * width + width / 2
            ax.bar(
                [pos + offset for pos in x],
                df[metric].values,
                width,
                label=metric,
            )

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return fig


def plot_metric_distribution(
    benchmark_results: dict,
    metric: str,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
) -> "plt.Figure":
    """Plot distribution of a specific metric across benchmarks.

    Args:
        benchmark_results: Dict mapping benchmark name to its result dict
        metric: Metric name to plot
        figsize: Figure size as (width, height)
        title: Plot title (default: f"{metric} Distribution")

    Returns:
        matplotlib Figure object if available, otherwise None
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import pandas as pd

    if title is None:
        title = f"{metric} Distribution"

    df = pd.DataFrame.from_dict(benchmark_results, orient="index")
    if "num_samples" in df.columns:
        df = df.drop(columns=["num_samples"])

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")

    fig, ax = plt.subplots(figsize=figsize)

    values = df[metric].dropna()

    bars = ax.bar(range(len(values)), values.values)

    ax.set_xlabel("Benchmark")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values.index, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    max_val = values.max()
    for bar, val in zip(bars, values.values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if max_val <= 1.0:
        ax.set_ylim(0, 1.1)

    fig.tight_layout()

    return fig
