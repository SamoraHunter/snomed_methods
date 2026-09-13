"""Evaluation utilities for UMNSRS benchmarking."""

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def spearman_correlation(
    predictions: list[float],
    references: list[float],
) -> tuple[float, float]:
    """Compute Spearman rank correlation coefficient.

    Args:
        predictions: Model similarity/relatedness scores
        references: Human annotated ground truth scores

    Returns:
        Tuple of (correlation_coefficient, p_value)

    Raises:
        ValueError: If input lists have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have same length "
            f"(got {len(predictions)} vs {len(references)})"
        )

    corr, p_value = stats.spearmanr(predictions, references)

    return float(corr), float(p_value)


def pearson_correlation(
    predictions: list[float],
    references: list[float],
) -> tuple[float, float]:
    """Compute Pearson product-moment correlation coefficient.

    Args:
        predictions: Model similarity/relatedness scores
        references: Human annotated ground truth scores

    Returns:
        Tuple of (correlation_coefficient, p_value)

    Raises:
        ValueError: If input lists have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have same length "
            f"(got {len(predictions)} vs {len(references)})"
        )

    corr, p_value = stats.pearsonr(predictions, references)

    return float(corr), float(p_value)


def mean_absolute_error(
    predictions: list[float],
    references: list[float],
) -> float:
    """Compute Mean Absolute Error (MAE).

    Args:
        predictions: Model predicted scores
        references: Ground truth scores

    Returns:
        Mean absolute error
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have same length "
            f"(got {len(predictions)} vs {len(references)})"
        )

    return float(np.mean(np.abs(np.array(predictions) - np.array(references))))


def mean_squared_error(
    predictions: list[float],
    references: list[float],
) -> float:
    """Compute Mean Squared Error (MSE).

    Args:
        predictions: Model predicted scores
        references: Ground truth scores

    Returns:
        Mean squared error
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have same length "
            f"(got {len(predictions)} vs {len(references)})"
        )

    return float(np.mean((np.array(predictions) - np.array(references)) ** 2))


def root_mean_squared_error(
    predictions: list[float],
    references: list[float],
) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Args:
        predictions: Model predicted scores
        references: Ground truth scores

    Returns:
        Root mean squared error
    """
    return np.sqrt(mean_squared_error(predictions, references))


def evaluate_model(
    model_func: Callable[[str, str], float],
    dataset_pairs: list[dict],
    metric: str = "spearman",
) -> dict:
    """Evaluate a similarity/relatedness model on UMNSRS dataset.

    Args:
        model_func: Function that takes (text_1, text_2) and returns similarity score
        dataset_pairs: List of dicts with 'text_1', 'text_2', and 'label' keys
        metric: Evaluation metric(s). Options:
            - 'spearman': Spearman rank correlation
            - 'pearson': Pearson correlation
            - 'mae': Mean Absolute Error
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error

    Returns:
        Dict with metric names and their values

    Example:
        >>> def mock_model(text1, text2):
        ...     return 500.0
        >>> pairs = [{"text_1": "A", "text_2": "B", "label": 600.0}]
        >>> results = evaluate_model(mock_model, pairs)
    """
    if isinstance(metric, str):
        metric = [metric]

    valid_metrics = {
        "spearman": spearman_correlation,
        "pearson": pearson_correlation,
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
    }

    for m in metric:
        if m not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{m}'. Options: {list(valid_metrics.keys())}"
            )

    predictions = []
    references = []

    for pair in dataset_pairs:
        score = model_func(pair["text_1"], pair["text_2"])
        predictions.append(score)
        references.append(pair["label"])

    results = {}

    for metric_name in metric:
        metric_fn = valid_metrics[metric_name]

        if metric_name in ["spearman", "pearson"]:
            value, p_value = metric_fn(predictions, references)
            results[f"{metric_name}_correlation"] = value
            results[f"{metric_name}_p_value"] = p_value
        else:
            results[metric_name] = metric_fn(predictions, references)

    return results


def precision_at_k(
    predicted_cuis: list[str],
    relevant_cuis: set,
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
    relevant_count = sum(1 for cui in top_k if cui in relevant_cuis)
    return relevant_count / len(top_k) if top_k else 0.0


def recall_at_k(
    predicted_cuis: list[str],
    relevant_cuis: set,
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
    relevant_found = sum(1 for cui in top_k if cui in relevant_cuis)
    return relevant_found / len(relevant_cuis) if relevant_cuis else 0.0


def f1_at_k(
    predicted_cuis: list[str],
    relevant_cuis: set,
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
    predicted_cuis: list[str],
    relevant_cuis: set,
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
    predicted_cuis: list[str],
    relevant_cuis: set,
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


def benchmark_results_to_dataframe(
    benchmark_results: list[dict],
) -> "pd.DataFrame":
    """Convert benchmark results to pandas DataFrame.

    Args:
        benchmark_results: List of result dicts from evaluate_model

    Returns:
        pandas DataFrame with metrics as columns
    """
    import pandas as pd

    if not benchmark_results:
        return pd.DataFrame()

    return pd.DataFrame(benchmark_results)
