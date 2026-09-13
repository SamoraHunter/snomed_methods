"""Evaluation metrics and utilities for clinical concept annotation benchmarking."""

from typing import List, Set

import numpy as np


def precision_at_k(
    predicted_cuis: List[str],
    relevant_cuis: Set[str],
    k: int = 10,
) -> float:
    """Compute Precision@K for ranking tasks.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        relevant_cuis: Set of relevant/ground truth CUIs
        k: Number of top results to consider

    Returns:
        Precision at K (fraction of top-K results that are relevant)
    """
    top_k = predicted_cuis[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for cui in top_k if cui in relevant_cuis)
    return relevant_count / len(top_k)


def recall_at_k(
    predicted_cuis: List[str],
    relevant_cuis: Set[str],
    k: int = 10,
) -> float:
    """Compute Recall@K for ranking tasks.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        relevant_cuis: Set of relevant/ground truth CUIs
        k: Number of top results to consider

    Returns:
        Recall at K (fraction of relevant items found in top-K)
    """
    top_k = predicted_cuis[:k]
    if not relevant_cuis:
        return 0.0
    relevant_found = sum(1 for cui in top_k if cui in relevant_cuis)
    return relevant_found / len(relevant_cuis)


def f1_at_k(
    predicted_cuis: List[str],
    relevant_cuis: Set[str],
    k: int = 10,
) -> float:
    """Compute F1@K for ranking tasks.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        relevant_cuis: Set of relevant/ground truth CUIs
        k: Number of top results to consider

    Returns:
        F1 score at K
    """
    p = precision_at_k(predicted_cuis, relevant_cuis, k)
    r = recall_at_k(predicted_cuis, relevant_cuis, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def mean_reciprocal_rank(
    predicted_cuis: List[str],
    relevant_cuis: Set[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        relevant_cuis: Set of relevant/ground truth CUIs

    Returns:
        Reciprocal rank of first relevant item found (0 if none found)
    """
    for i, cui in enumerate(predicted_cuis):
        if cui in relevant_cuis:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    predicted_cuis: List[str],
    relevant_cuis: Set[str],
) -> float:
    """Compute Average Precision (AP) for ranking tasks.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        relevant_cuis: Set of relevant/ground truth CUIs

    Returns:
        Average precision across all relevant items
    """
    if not relevant_cuis:
        return 0.0

    ap_sum = 0.0
    relevant_count = 0

    for i, cui in enumerate(predicted_cuis):
        if cui in relevant_cuis:
            relevant_count += 1
            ap_sum += relevant_count / (i + 1)

    return ap_sum / len(relevant_cuis)


def exact_match_rate(
    predicted_cuis: List[str],
    relevant_cuis: Set[str],
) -> float:
    """Compute Exact Match Rate.

    Args:
        predicted_cuis: List of predicted CUIs in rank order
        relevant_cuis: Set of relevant/ground truth CUIs

    Returns:
        1.0 if first prediction matches any relevant, else 0.0
    """
    if not predicted_cuis or not relevant_cuis:
        return 0.0
    return 1.0 if predicted_cuis[0] in relevant_cuis else 0.0


def evaluate_annotator(
    annotator_func,
    dataset: List[dict],
    k_values: List[int] = None,
) -> dict:
    """Evaluate a concept annotator on annotation benchmark dataset.

    Args:
        annotator_func: Function that takes text and returns AnnotationResult
            or list of CUIs
        dataset: List of dicts with 'text' and 'gold_cuis' keys
        k_values: List of K values for@K metrics (default: [1, 3, 5, 10])

    Returns:
        Dict with averaged metrics across all samples

    Example:
        >>> def my_annotator(text):
        ...     return ["C001", "C002", "C003"]
        >>> dataset = generate_annotation_dataset(10)
        >>> results = evaluate_annotator(my_annotator, dataset)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_precisions: dict = {}
    all_recalls: dict = {}
    all_f1s: dict = {}

    reciprocal_ranks: List[float] = []
    exact_matches: List[float] = []

    for sample in dataset:
        text = sample["text"]
        gold_cuis = set(sample["gold_cuis"])

        predicted_cuis = _extract_predicted_cuis(annotator_func(text))

        if not predicted_cuis:
            continue

        for k in k_values:
            p = precision_at_k(predicted_cuis, gold_cuis, k)
            r = recall_at_k(predicted_cuis, gold_cuis, k)
            f1_val = f1_at_k(predicted_cuis, gold_cuis, k)

            if k not in all_precisions:
                all_precisions[k] = []
                all_recalls[k] = []
                all_f1s[k] = []

            all_precisions[k].append(p)
            all_recalls[k].append(r)
            all_f1s[k].append(f1_val)

        rr = mean_reciprocal_rank(predicted_cuis, gold_cuis)
        em = exact_match_rate(predicted_cuis, gold_cuis)

        reciprocal_ranks.append(rr)
        exact_matches.append(em)

    results: dict = {}

    for k in k_values:
        if all_precisions.get(k):
            results[f"precision@{k}"] = float(np.mean(all_precisions[k]))
            results[f"recall@{k}"] = float(np.mean(all_recalls[k]))
            results[f"f1@{k}"] = float(np.mean(all_f1s[k]))
        else:
            results[f"precision@{k}"] = 0.0
            results[f"recall@{k}"] = 0.0
            results[f"f1@{k}"] = 0.0

    results["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    results["exact_match_rate"] = (
        float(np.mean(exact_matches)) if exact_matches else 0.0
    )
    results["num_samples"] = len(dataset)

    return results


def _extract_predicted_cuis(prediction_result) -> List[str]:
    """Extract predicted CUIs from various prediction formats."""
    if hasattr(prediction_result, "top_concepts"):
        return [c.concept_id for c in prediction_result.top_concepts]
    if isinstance(prediction_result, list):
        return prediction_result
    try:
        concepts_dict = getattr(prediction_result, "concepts", [])
        return [c["concept_id"] for c in concepts_dict]
    except Exception:
        return []


def benchmark_results_to_dataframe(results: dict):
    """Convert benchmark results to pandas DataFrame.

    Args:
        results: Dict from evaluate_annotator

    Returns:
        pandas DataFrame with metrics as columns
    """
    try:
        import pandas as pd

        df = pd.DataFrame([results])
        return df.drop(columns=["num_samples"], errors="ignore")
    except ImportError:
        return None
