"""Evaluation metrics for hierarchy expansion benchmarking."""

from typing import List, Set

import numpy as np


def exact_match_rate(
    predicted_cuis: List[str],
    expected_cuis: Set[str],
) -> float:
    """Compute Exact Match Rate.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        expected_cuis: Set of expected ground truth CUIs

    Returns:
        1.0 if top-K prediction set matches expected exactly, else 0.0
    """
    return 1.0 if set(predicted_cuis) == expected_cuis else 0.0


def recall_at_k(
    predicted_cuis: List[str],
    expected_cuis: Set[str],
    k: int = 10,
) -> float:
    """Compute Recall@K for hierarchy expansion.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        expected_cuis: Set of expected ground truth CUIs
        k: Number of top results to consider

    Returns:
        Recall at K (fraction of expected items found in top-K)
    """
    top_k = set(predicted_cuis[:k])
    relevant_found = len(top_k & expected_cuis)
    return relevant_found / len(expected_cuis) if expected_cuis else 0.0


def precision_at_k(
    predicted_cuis: List[str],
    expected_cuis: Set[str],
    k: int = 10,
) -> float:
    """Compute Precision@K for hierarchy expansion.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        expected_cuis: Set of expected ground truth CUIs
        k: Number of top results to consider

    Returns:
        Precision at K (fraction of top-K that are relevant)
    """
    top_k = set(predicted_cuis[:k])
    if not top_k:
        return 0.0
    relevant_found = len(top_k & expected_cuis)
    return relevant_found / len(top_k)


def f1_at_k(
    predicted_cuis: List[str],
    expected_cuis: Set[str],
    k: int = 10,
) -> float:
    """Compute F1@K for hierarchy expansion.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        expected_cuis: Set of expected ground truth CUIs
        k: Number of top results to consider

    Returns:
        F1 score at K
    """
    p = precision_at_k(predicted_cuis, expected_cuis, k)
    r = recall_at_k(predicted_cuis, expected_cuis, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def jaccard_similarity(
    predicted_cuis: List[str],
    expected_cuis: Set[str],
    k: int = 10,
) -> float:
    """Compute Jaccard similarity between predicted and expected sets.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        expected_cuis: Set of expected ground truth CUIs
        k: Number of top results to consider

    Returns:
        Jaccard similarity score (intersection over union)
    """
    top_k = set(predicted_cuis[:k])
    intersection = len(top_k & expected_cuis)
    union = len(top_k | expected_cuis)
    return intersection / union if union else 0.0


def evaluate_hierarchy_expansion(
    expansion_func,
    dataset: List[dict],
    k_values: List[int] = None,
) -> dict:
    """Evaluate a hierarchy expansion method on benchmark dataset.

    Args:
        expansion_func: Function that takes seed_cui and returns list of related CUIs
        dataset: List of dicts with 'seed_cui' and 'expected_related' keys
        k_values: List of K values for@K metrics (default: [5, 10, 20])

    Returns:
        Dict with averaged metrics across all samples

    Example:
        >>> def my_expansion(seed_cui):
        ...     return ["C001", "C002", "C003"]
        >>> dataset = generate_hierarchy_dataset(10)
        >>> results = evaluate_hierarchy_expansion(my_expansion, dataset)
    """
    if k_values is None:
        k_values = [5, 10, 20]

    all_exact_matches = []
    all_recalls = {k: [] for k in k_values}
    all_precisions = {k: [] for k in k_values}
    all_f1s = {k: [] for k in k_values}
    all_jaccards = {k: [] for k in k_values}

    for sample in dataset:
        seed_cui = sample["seed_cui"]
        expected_related = set(sample["expected_related"])

        prediction_result = expansion_func(seed_cui)

        if isinstance(prediction_result, list):
            predicted_cuis = prediction_result
        else:
            try:
                predicted_cuis = list(prediction_result)
            except Exception:
                continue

        exact_match = 1.0 if set(predicted_cuis) == expected_related else 0.0
        all_exact_matches.append(exact_match)

        for k in k_values:
            r = recall_at_k(predicted_cuis, expected_related, k)
            p = precision_at_k(predicted_cuis, expected_related, k)
            f1 = f1_at_k(predicted_cuis, expected_related, k)
            j = jaccard_similarity(predicted_cuis, expected_related, k)

            all_recalls[k].append(r)
            all_precisions[k].append(p)
            all_f1s[k].append(f1)
            all_jaccards[k].append(j)

    results = {}

    for k in k_values:
        if all_recalls[k]:
            results[f"recall@{k}"] = float(np.mean(all_recalls[k]))
            results[f"precision@{k}"] = float(np.mean(all_precisions[k]))
            results[f"f1@{k}"] = float(np.mean(all_f1s[k]))
            results[f"jaccard@{k}"] = float(np.mean(all_jaccards[k]))

    if all_exact_matches:
        results["exact_match_rate"] = float(np.mean(all_exact_matches))

    results["num_samples"] = len(dataset)

    return results
