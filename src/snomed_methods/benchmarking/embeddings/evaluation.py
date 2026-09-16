"""Evaluation metrics and utilities for embedding semantic similarity benchmarking."""

from typing import List, Tuple

import numpy as np


def accuracy_at_threshold(
    similarity_scores: List[float],
    true_labels: List[bool],
    threshold: float = 0.5,
) -> float:
    """Compute classification accuracy at a given similarity threshold.

    Args:
        similarity_scores: List of predicted similarity scores (normalized 0-1)
        true_labels: List of ground truth binary labels
        threshold: Score threshold for classifying as similar

    Returns:
        Accuracy (fraction of correct classifications)
    """
    predictions = [score >= threshold for score in similarity_scores]
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    return correct / len(true_labels) if true_labels else 0.0


def precision_recall_f1(
    similarity_scores: List[float],
    true_labels: List[bool],
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 at given threshold.

    Args:
        similarity_scores: Predicted similarity scores
        true_labels: Ground truth binary labels
        threshold: Score threshold for positive classification

    Returns:
        Tuple of (precision, recall, f1)
    """
    predictions = [score >= threshold for score in similarity_scores]

    tp = sum(p and t for p, t in zip(predictions, true_labels))
    fp = sum(p and not t for p, t in zip(predictions, true_labels))
    fn = sum(not p and t for p, t in zip(predictions, true_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def auc_pr(
    similarity_scores: List[float],
    true_labels: List[bool],
) -> float:
    """Compute Area Under Precision-Recall Curve.

    Args:
        similarity_scores: Predicted similarity scores
        true_labels: Ground truth binary labels

    Returns:
        AUC-PR score (0-1, higher is better)
    """
    if not similarity_scores or not true_labels:
        return 0.0

    # Sort by scores descending
    sorted_pairs = sorted(
        zip(similarity_scores, true_labels), key=lambda x: x[0], reverse=True
    )

    precisions = []
    recalls = []

    tp = 0
    fp = 0
    total_positives = sum(true_labels)

    for _score, label in sorted_pairs:
        if label:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_positives if total_positives > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    # Compute AUC using trapezoidal rule
    if len(recalls) == 0:
        auc = 0.0
    elif len(recalls) == 1:
        # Single point: use rectangle rule (precision at that recall)
        auc = precisions[0] * recalls[0]
    else:
        auc = 0.0
        for i in range(1, len(recalls)):
            delta_recall = recalls[i] - recalls[i - 1]
            avg_precision = (precisions[i] + precisions[i - 1]) / 2
            auc += delta_recall * avg_precision

    return abs(auc)


def mean_squared_error_similarity(
    predicted_scores: List[float],
    reference_scores: List[float],
) -> float:
    """Compute MSE between predicted and reference similarity scores.

    Args:
        predicted_scores: Model-predicted similarity scores
        reference_scores: Ground truth similarity scores (e.g., UMNSRS scores)

    Returns:
        Mean squared error
    """
    if len(predicted_scores) != len(reference_scores):
        raise ValueError(
            f"Scores must have same length "
            f"(got {len(predicted_scores)} vs {len(reference_scores)})"
        )

    return float(
        np.mean((np.array(predicted_scores) - np.array(reference_scores)) ** 2)
    )


def correlation_similarity(
    predicted_scores: List[float],
    reference_scores: List[float],
    method: str = "spearman",
) -> Tuple[float, float]:
    """Compute correlation between predicted and reference similarity scores.

    Args:
        predicted_scores: Model-predicted similarity scores
        reference_scores: Ground truth similarity scores
        method: Correlation method ('spearman' or 'pearson')

    Returns:
        Tuple of (correlation_coefficient, p_value)
    """
    from scipy import stats

    if len(predicted_scores) != len(reference_scores):
        raise ValueError(
            f"Scores must have same length "
            f"(got {len(predicted_scores)} vs {len(reference_scores)})"
        )

    if method == "spearman":
        corr, p_value = stats.spearmanr(predicted_scores, reference_scores)
    elif method == "pearson":
        corr, p_value = stats.pearsonr(predicted_scores, reference_scores)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return float(corr), float(p_value)


def evaluate_embedding_similarity(
    embedder,
    dataset: List[dict],
    threshold: float = 0.5,
    use_umnsrs_scores: bool = True,
) -> dict:  # noqa: C901
    """Evaluate an embedding model on semantic similarity benchmark.

    Args:
        embedder: ClinicalConceptEmbedder instance
        dataset: List of dicts with 'concept_1', 'concept_2', and optionally
                 'umnsrs_score' or 'label' (UMNSRS format)
        threshold: Similarity threshold for classification
        use_umnsrs_scores: If True, use UMNSRS scores instead of binary labels

    Returns:
        Dict with evaluation metrics
    """
    try:
        import importlib.util

        if not importlib.util.find_spec("snomed_methods.llm_concept_embedder"):
            raise ImportError(
                "snomed_methods.llm_concept_embedder not available. "
                "Ensure the package is installed."
            )
    except ImportError:
        pass
    if not dataset:
        return {
            "num_samples": 0,
            "accuracy@threshold": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Compute similarity scores for all pairs
    similarity_scores = []
    true_labels = []
    reference_scores = []

    for sample in dataset:
        # Handle both formats: SNOMED CUI pairs or text pairs (UMNSRS)
        if "concept_1" in sample and "concept_2" in sample:
            concept_1 = str(sample["concept_1"])
            concept_2 = str(sample["concept_2"])
        elif "text_1" in sample and "text_2" in sample:
            concept_1 = str(sample["text_1"])
            concept_2 = str(sample["text_2"])
        else:
            # Skip samples without required fields
            continue

        try:
            # Get embeddings and compute cosine similarity
            emb1 = embedder.generate_embeddings([str(concept_1)], batch_size=1)[0]
            emb2 = embedder.generate_embeddings([str(concept_2)], batch_size=1)[0]

            # Compute cosine similarity
            score = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )
            # Normalize to [0, 1] for easier thresholding
            normalized_score = (score + 1) / 2

            similarity_scores.append(normalized_score)

            if use_umnsrs_scores:
                # Handle both "label" (UMNSRS) and "umnsrs_score" field names
                score_field = sample.get("label") or sample.get("umnsrs_score")
                if score_field is not None:
                    reference_scores.append(float(score_field) / 1000.0)
                    true_labels.append(float(score_field) >= 500)  # UMNSRS threshold
            else:
                true_labels.append(bool(sample.get("is_similar", False)))
                reference_scores.append(1.0 if sample.get("is_similar") else 0.0)
        except Exception:
            # Skip pairs that fail to embed
            continue

    if not similarity_scores:
        return {"num_samples": 0, "error": "No valid pairs could be evaluated"}

    # Compute metrics
    accuracy = accuracy_at_threshold(similarity_scores, true_labels, threshold)
    precision, recall, f1 = precision_recall_f1(
        similarity_scores, true_labels, threshold
    )

    if reference_scores:
        auc_pr_score = auc_pr(similarity_scores, true_labels)
        mse = mean_squared_error_similarity(similarity_scores, reference_scores)
        spearman_corr, spearman_p = correlation_similarity(
            similarity_scores, reference_scores, method="spearman"
        )
        pearson_corr, pearson_p = correlation_similarity(
            similarity_scores, reference_scores, method="pearson"
        )
    else:
        auc_pr_score = 0.0
        mse = 0.0
        spearman_corr = spearman_p = 0.0
        pearson_corr = pearson_p = 0.0

    return {
        "num_samples": len(similarity_scores),
        "accuracy@threshold": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_pr": auc_pr_score,
        "mse": mse,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,
    }
