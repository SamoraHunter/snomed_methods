# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Evaluation metrics for term lookup benchmarking."""

from typing import Any, List, Tuple

import numpy as np


def recall_at_k(
    result_cuis: List[str],
    expected_cui: str,
    k: int = 10,
) -> float:
    """Compute Recall@K for term lookup.

    Args:
        result_cuis: List of CUIs from search results in rank order
        expected_cui: The expected ground truth CUI
        k: Number of top results to consider

    Returns:
        1.0 if expected CUI found in top-K, else 0.0
    """
    return 1.0 if expected_cui in result_cuis[:k] else 0.0


def precision_at_k(
    result_cuis: List[str],
    expected_cui: str,
    k: int = 10,
) -> float:
    """Compute Precision@K for term lookup.

    Args:
        result_cuis: List of CUIs from search results in rank order
        expected_cui: The expected ground truth CUI
        k: Number of top results to consider

    Returns:
        1.0 if expected CUI found at position <= k, else 0.0
    """
    return 1.0 if expected_cui in result_cuis[:k] else 0.0


def mean_reciprocal_rank(
    result_cuis: List[str],
    expected_cui: str,
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        result_cuis: List of CUIs from search results in rank order
        expected_cui: The expected ground truth CUI

    Returns:
        Reciprocal rank of expected CUI if found, else 0.0
    """
    for i, cui in enumerate(result_cuis):
        if cui == expected_cui:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    result_cuis: List[str],
    expected_cui: str,
) -> float:
    """Compute Average Precision for term lookup.

    Args:
        result_cuis: List of CUIs from search results in rank order
        expected_cui: The expected ground truth CUI

    Returns:
        Average precision: 1.0 if found at any position, 0.0 otherwise
    """
    for i, cui in enumerate(result_cuis):
        if cui == expected_cui:
            return 1.0 / (i + 1)
    return 0.0


def exact_match_at_position(
    result_cuis: List[str],
    expected_cui: str,
    position: int = 0,
) -> float:
    """Check if expected CUI is at specific position.

    Args:
        result_cuis: List of CUIs from search results
        expected_cui: The expected ground truth CUI
        position: Position to check (0-indexed)

    Returns:
        1.0 if expected CUI found at exact position, else 0.0
    """
    return float(len(result_cuis) > position and result_cuis[position] == expected_cui)


def hit_rate(
    result_cuis: List[str],
    expected_cui: str,
) -> float:
    """Compute Hit Rate (whether expected CUI appears anywhere).

    Args:
        result_cuis: List of CUIs from search results
        expected_cui: The expected ground truth CUI

    Returns:
        1.0 if expected CUI found at any position, else 0.0
    """
    return 1.0 if expected_cui in result_cuis else 0.0


def _evaluate_single_sample(
    lookup_func: Any,
    sample: dict,
) -> Tuple[dict, List[float], List[float]]:
    """Evaluate a single dataset sample and return results.

    Args:
        lookup_func: Function that takes term and returns list of (cui, term) tuples
        sample: Single dataset sample with 'term' and 'expected_cui'

    Returns:
        Tuple of (metrics_dict, reciprocal_ranks_list, exact_matches_list)
    """
    term = sample["term"]
    expected_cui = sample["expected_cui"]

    prediction_result = lookup_func(term)

    result_cuis = _extract_result_cuis(prediction_result)

    if not result_cuis:
        return {}, [], []

    all_recalls: dict = {}
    all_precisions: dict = {}

    for k in [1, 3, 5, 10]:
        r = recall_at_k(result_cuis, expected_cui, k)
        p = precision_at_k(result_cuis, expected_cui, k)
        all_recalls[k] = r
        all_precisions[k] = p

    rr = mean_reciprocal_rank(result_cuis, expected_cui)
    reciprocal_ranks = [rr]

    exact_matches: List[float] = []
    for pos in [0, 1, 2]:
        em = exact_match_at_position(result_cuis, expected_cui, pos)
        exact_matches.append(em)

    return (
        {
            "recalls": all_recalls,
            "precisions": all_precisions,
            "mrr": rr,
            "exact_matches": exact_matches,
            "result_cuis": result_cuis,
        },
        reciprocal_ranks,
        exact_matches,
    )


def evaluate_term_lookup(
    lookup_func: Any,
    dataset: List[dict],
    k_values: List[int] | None = None,
) -> dict:
    """Evaluate a term lookup method on benchmark dataset.

    Args:
        lookup_func: Function that takes term and returns list of (cui, term) tuples
        dataset: List of dicts with 'term' and 'expected_cui' keys
        k_values: List of K values for@K metrics (default: [1, 3, 5, 10])

    Returns:
        Dict with averaged metrics across all samples

    Example:
        >>> def my_lookup(term):
        ...     return [("123456789", "term")]
        >>> dataset = generate_term_dataset(10)
        >>> results = evaluate_term_lookup(my_lookup, dataset)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_recalls: dict = {k: [] for k in k_values}
    all_precisions: dict = {k: [] for k in k_values}

    reciprocal_ranks: List[float] = []
    exact_matches = {pos: [] for pos in [0, 1, 2]}

    for sample in dataset:
        metrics, rrs, ems = _evaluate_single_sample(lookup_func, sample)

        if not metrics:
            continue

        for k in k_values:
            all_recalls[k].append(metrics["recalls"].get(k, 0.0))
            all_precisions[k].append(metrics["precisions"].get(k, 0.0))

        reciprocal_ranks.extend(rrs)
        for pos, em in enumerate(ems):
            exact_matches[pos].append(em)

    results: dict = {}

    for k in k_values:
        if all_recalls[k]:
            results[f"recall@{k}"] = float(np.mean(all_recalls[k]))
            results[f"precision@{k}"] = float(np.mean(all_precisions[k]))
        else:
            results[f"recall@{k}"] = 0.0
            results[f"precision@{k}"] = 0.0

    if reciprocal_ranks:
        results["mrr"] = float(np.mean(reciprocal_ranks))
        results["hit_rate"] = float(
            sum(1 for rr in reciprocal_ranks if rr > 0) / len(reciprocal_ranks)
        )
    else:
        results["mrr"] = 0.0
        results["hit_rate"] = 0.0

    for pos, scores in exact_matches.items():
        results[f"exact_match@pos_{pos}"] = float(np.mean(scores))

    results["num_samples"] = len(dataset)

    return results


def _extract_result_cuis(prediction_result: Any) -> List[str]:
    """Extract result CUIs from various prediction formats."""
    if isinstance(prediction_result, list):
        if not prediction_result:
            return []
        first_item = prediction_result[0]
        if isinstance(first_item, tuple) and len(first_item) >= 2:
            return [str(c) for c, _ in prediction_result]
        try:
            return [str(c.get("concept_id", "")) for c in prediction_result]
        except Exception:
            return []
    try:
        concepts_list = getattr(prediction_result, "concepts", [])
        return [str(c) for c in concepts_list]
    except Exception:
        return []
