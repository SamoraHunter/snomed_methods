# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""UMNSRS benchmarking module for semantic similarity and relatedness evaluation."""

from .dataset import (
    download_umnsrs,
    get_umnsrs_pairs,
    load_from_csv,
    save_umnsrs_to_csv,
)
from .evaluation import (
    average_precision,
    benchmark_results_to_dataframe,
    evaluate_model,
    f1_at_k,
    mean_absolute_error,
    mean_reciprocal_rank,
    mean_squared_error,
    pearson_correlation,
    precision_at_k,
    recall_at_k,
    root_mean_squared_error,
    spearman_correlation,
)

__all__ = [
    "average_precision",
    "benchmark_results_to_dataframe",
    "download_umnsrs",
    "evaluate_model",
    "f1_at_k",
    "get_umnsrs_pairs",
    "load_from_csv",
    "mean_absolute_error",
    "mean_reciprocal_rank",
    "mean_squared_error",
    "pearson_correlation",
    "precision_at_k",
    "recall_at_k",
    "root_mean_squared_error",
    "save_umnsrs_to_csv",
    "spearman_correlation",
]
