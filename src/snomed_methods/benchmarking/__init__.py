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
from .results import (
    aggregate_results,
    best_performance,
    compare_benchmarks,
    evaluate_suite,
    filter_metrics,
    get_all_metrics,
    load_results,
    plot_comparison,
    plot_correlation_matrix,
    plot_metric_distribution,
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
    average_precision,
    benchmark_results_to_dataframe,
    download_umnsrs,
    evaluate_model,
    get_umnsrs_pairs,
    mean_absolute_error,
    pearson_correlation,
    root_mean_squared_error,
    spearman_correlation,
)
from .umnsrs import (
    f1_at_k as umnsrs_f1_at_k,
)
from .umnsrs import (
    mean_reciprocal_rank as umnsrs_mean_reciprocal_rank,
)
from .umnsrs import (
    precision_at_k as umnsrs_precision_at_k,
)
from .umnsrs import (
    recall_at_k as umnsrs_recall_at_k,
)
from .vocabulary import (
    evaluate_mapper,
    generate_mapping_dataset,
    load_mapping_datasets,
)

__all__ = [
    "annotation",
    "hierarchy",
    "term",
    "vocabulary",
    "umnsrs",
    "generate_annotation_dataset",
    "load_annotation_datasets",
    "evaluate_annotator",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "mean_reciprocal_rank",
    "generate_hierarchy_dataset",
    "load_hierarchy_datasets",
    "evaluate_hierarchy_expansion",
    "generate_term_dataset",
    "load_term_datasets",
    "evaluate_term_lookup",
    "generate_mapping_dataset",
    "load_mapping_datasets",
    "evaluate_mapper",
    "download_umnsrs",
    "get_umnsrs_pairs",
    "evaluate_model",
    "spearman_correlation",
    "pearson_correlation",
    "mean_absolute_error",
    "root_mean_squared_error",
    "results",
    "evaluate_suite",
    "aggregate_results",
    "compare_benchmarks",
    "plot_comparison",
    "plot_metric_distribution",
    "summarize_benchmark",
    "save_results",
    "load_results",
    "rank_benchmarks",
    "best_performance",
    "worst_performance",
    "get_all_metrics",
    "filter_metrics",
    "plot_correlation_matrix",
]

__version__ = "1.0.0"
