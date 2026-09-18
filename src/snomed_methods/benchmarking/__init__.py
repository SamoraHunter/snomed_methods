# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Benchmarking module for SNOMED methods evaluation."""

from .annotation import (
    evaluate_annotator,
    f1_at_k,
    generate_annotation_dataset,
    load_annotation_datasets,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from .hierarchy import (
    evaluate_hierarchy_expansion,
    generate_hierarchy_dataset,
    load_hierarchy_datasets,
)
from .mock_methods import (
    get_mock_methods,
    mock_annotator,
    mock_hierarchy_expansion,
    mock_mapper,
    mock_similarity,
    mock_term_lookup,
)
from .plots import plot_comparison, plot_metric_distribution
from .results import (
    aggregate_results,
    best_performance,
    compare_benchmarks,
    evaluate_suite,
    filter_metrics,
    get_all_metrics,
    load_results,
    plot_correlation_matrix,
    rank_benchmarks,
    save_results,
    summarize_benchmark,
    worst_performance,
)
from .term import (
    evaluate_term_lookup,
    generate_term_dataset,
    load_term_datasets,
)
from .umnsrs import (
    benchmark_results_to_dataframe,
    download_umnsrs,
    evaluate_model,
    get_umnsrs_pairs,
    mean_absolute_error,
    pearson_correlation,
    root_mean_squared_error,
    spearman_correlation,
)
from .vocabulary import (
    evaluate_mapper,
    generate_mapping_dataset,
    load_mapping_datasets,
)

try:
    from .embeddings import (
        accuracy_at_threshold,
        auc_pr,
        correlation_similarity,
        evaluate_embedding_similarity,
        generate_embedding_dataset,
        load_embedding_datasets,
        mean_squared_error_similarity,
        precision_recall_f1,
    )

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

__all__ = [
    "accuracy_at_threshold",
    "aggregate_results",
    "annotation",
    "auc_pr",
    "best_performance",
    "compare_benchmarks",
    "correlation_similarity",
    "download_umnsrs",
    "embeddings",
    "evaluate_annotator",
    "evaluate_embedding_similarity",
    "evaluate_hierarchy_expansion",
    "evaluate_mapper",
    "evaluate_model",
    "evaluate_suite",
    "evaluate_term_lookup",
    "f1_at_k",
    "filter_metrics",
    "generate_annotation_dataset",
    "generate_embedding_dataset",
    "generate_hierarchy_dataset",
    "generate_mapping_dataset",
    "generate_term_dataset",
    "get_all_metrics",
    "get_mock_methods",
    "get_umnsrs_pairs",
    "hierarchy",
    "load_annotation_datasets",
    "load_embedding_datasets",
    "load_hierarchy_datasets",
    "load_mapping_datasets",
    "load_results",
    "load_term_datasets",
    "mean_absolute_error",
    "mean_reciprocal_rank",
    "mean_squared_error_similarity",
    "mock_annotator",
    "mock_hierarchy_expansion",
    "mock_mapper",
    "mock_methods",
    "mock_similarity",
    "mock_term_lookup",
    "pearson_correlation",
    "plot_comparison",
    "plot_correlation_matrix",
    "plot_metric_distribution",
    "precision_at_k",
    "precision_recall_f1",
    "rank_benchmarks",
    "recall_at_k",
    "results",
    "root_mean_squared_error",
    "save_results",
    "spearman_correlation",
    "summarize_benchmark",
    "term",
    "umnsrs",
    "vocabulary",
    "worst_performance",
]

__version__ = "1.0.0"
