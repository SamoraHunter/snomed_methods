# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
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
    "accuracy_at_threshold",
    "auc_pr",
    "correlation_similarity",
    "evaluate_embedding_similarity",
    "generate_embedding_dataset",
    "load_embedding_datasets",
    "mean_squared_error_similarity",
    "precision_recall_f1",
]
