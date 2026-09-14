"""Benchmarking utilities for embedding semantic similarity tasks."""

from .dataset import (
    generate_embedding_dataset,
    load_embedding_datasets,
)
from .evaluation import (
    accuracy_at_threshold,
    auc_pr,
    correlation_similarity,
    evaluate_embedding_similarity,
    mean_squared_error_similarity,
    precision_recall_f1,
)

__all__ = [
    "generate_embedding_dataset",
    "load_embedding_datasets",
    "evaluate_embedding_similarity",
    "accuracy_at_threshold",
    "precision_recall_f1",
    "auc_pr",
    "mean_squared_error_similarity",
    "correlation_similarity",
]
