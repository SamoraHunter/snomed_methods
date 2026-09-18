# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Benchmarking utilities for term lookup tasks."""

from .dataset import generate_term_dataset, load_term_datasets
from .evaluation import (
    average_precision,
    evaluate_term_lookup,
    exact_match_at_position,
    hit_rate,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "average_precision",
    "evaluate_term_lookup",
    "exact_match_at_position",
    "generate_term_dataset",
    "hit_rate",
    "load_term_datasets",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
