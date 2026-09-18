# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Evaluation metrics for vocabulary mapping benchmarking."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class MapperFunc(Protocol):
    """Type protocol for mapper functions."""

    def __call__(self, snomed_cui: str) -> list[str] | object: ...


def precision_at_k(
    predicted_codes: list[str],
    expected_codes: set[str],
    k: int = 10,
) -> float:
    """Compute Precision@K for vocabulary mapping.

    Args:
        predicted_codes: List of predicted codes in rank order
        expected_codes: Set of expected ground truth codes
        k: Number of top results to consider

    Returns:
        Precision at K (fraction of top-K that are relevant)

    """
    top_k = predicted_codes[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for code in top_k if code in expected_codes)
    return relevant_count / len(top_k)


def recall_at_k(
    predicted_codes: list[str],
    expected_codes: set[str],
    k: int = 10,
) -> float:
    """Compute Recall@K for vocabulary mapping.

    Args:
        predicted_codes: List of predicted codes in rank order
        expected_codes: Set of expected ground truth codes
        k: Number of top results to consider

    Returns:
        Recall at K (fraction of expected codes found in top-K)

    """
    top_k = set(predicted_codes[:k])
    relevant_found = len(top_k & expected_codes)
    return relevant_found / len(expected_codes) if expected_codes else 0.0


def coverage_rate(
    predicted_codes: list[str],
    expected_codes: set[str],
) -> float:
    """Compute Coverage Rate across all predictions.

    Args:
        predicted_codes: List of predicted codes
        expected_codes: Set of expected ground truth codes

    Returns:
        Fraction of expected codes covered by any prediction position

    """
    if not expected_codes:
        return 0.0
    pred_set = set(predicted_codes)
    relevant_found = len(pred_set & expected_codes)
    return relevant_found / len(expected_codes)


def exact_match_rate(
    predicted_codes: list[str],
    expected_codes: set[str],
) -> float:
    """Compute Exact Match Rate.

    Args:
        predicted_codes: List of predicted codes
        expected_codes: Set of expected ground truth codes

    Returns:
        1.0 if prediction set matches expected exactly, else 0.0

    """
    return 1.0 if set(predicted_codes) == expected_codes else 0.0


def f1_at_k(
    predicted_codes: list[str],
    expected_codes: set[str],
    k: int = 10,
) -> float:
    """Compute F1@K for vocabulary mapping.

    Args:
        predicted_codes: List of predicted codes in rank order
        expected_codes: Set of expected ground truth codes
        k: Number of top results to consider

    Returns:
        F1 score at K

    """
    p = precision_at_k(predicted_codes, expected_codes, k)
    r = recall_at_k(predicted_codes, expected_codes, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def mean_reciprocal_rank(
    predicted_codes: list[str],
    expected_codes: set[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        predicted_codes: List of predicted codes in rank order
        expected_codes: Set of expected ground truth codes

    Returns:
        Reciprocal rank of first relevant code found

    """
    for i, code in enumerate(predicted_codes):
        if code in expected_codes:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_mapper(
    mapper_func: MapperFunc,
    dataset: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate a vocabulary mapper on benchmark dataset.

    Args:
        mapper_func: Function that takes snomed_cui and returns list of mapped codes
        dataset: List of dicts with 'snomed_cui' and 'target_codes' keys
        k_values: List of K values for@K metrics (default: [1, 3, 5, 10])

    Returns:
        Dict with averaged metrics across all samples

    Example:
        >>> def my_mapper(snomed_cui):
        ...     return ["ICD_E11.9", "LOINC_4544-3"]
        >>> dataset = generate_mapping_dataset(10)
        >>> results = evaluate_mapper(my_mapper, dataset)

    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    all_f1s = {k: [] for k in k_values}
    coverage_scores = []
    reciprocal_ranks = []

    for sample in dataset:
        snomed_cui = sample["snomed_cui"]
        expected_codes = set(sample["target_codes"])

        prediction_result = mapper_func(snomed_cui)

        if isinstance(prediction_result, list):
            predicted_codes = prediction_result
        else:
            try:
                codes_list = getattr(prediction_result, "target_code", [])
                predicted_codes = list(codes_list)
            except (KeyError, TypeError, AttributeError):
                continue

        for k in k_values:
            p = precision_at_k(predicted_codes, expected_codes, k)
            r = recall_at_k(predicted_codes, expected_codes, k)
            f1 = f1_at_k(predicted_codes, expected_codes, k)

            all_precisions[k].append(p)
            all_recalls[k].append(r)
            all_f1s[k].append(f1)

        cov = coverage_rate(predicted_codes, expected_codes)
        rr = mean_reciprocal_rank(predicted_codes, expected_codes)

        coverage_scores.append(cov)
        reciprocal_ranks.append(rr)

    results = {}

    for k in k_values:
        if all_precisions[k]:
            results[f"precision@{k}"] = float(np.mean(all_precisions[k]))
            results[f"recall@{k}"] = float(np.mean(all_recalls[k]))
            results[f"f1@{k}"] = float(np.mean(all_f1s[k]))

    if coverage_scores:
        results["coverage_rate"] = float(np.mean(coverage_scores))

    if reciprocal_ranks:
        results["mrr"] = float(np.mean(reciprocal_ranks))

    results["num_samples"] = len(dataset)

    return results
