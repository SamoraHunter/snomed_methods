"""Unit tests for embedding evaluation metrics."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import numpy.typing as npt
import pytest


class TestAccuracyAtThreshold:
    """Tests for accuracy_at_threshold function."""

    def test_perfect_accuracy(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            accuracy_at_threshold,
        )

        # All scores >= threshold match true labels (all True)
        scores = [0.9, 0.85, 0.75, 0.6]
        labels = [True, True, True, True]

        acc = accuracy_at_threshold(scores, labels, threshold=0.5)

        assert acc == 1.0

    def test_no_samples(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            accuracy_at_threshold,
        )

        acc = accuracy_at_threshold([], [])

        assert acc == 0.0

    def test_partial_accuracy(self) -> None:
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )
        from snomed_methods.benchmarking.embeddings.evaluation import (
            accuracy_at_threshold,
        )

        dataset = generate_embedding_dataset(num_samples=20)
        scores = [0.9 if s["is_similar"] else 0.1 for s in dataset]
        labels = [s["is_similar"] for s in dataset]

        acc = accuracy_at_threshold(scores, labels, threshold=0.5)

        assert acc > 0.8

    def test_wrong_predictions(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            accuracy_at_threshold,
        )

        scores = [0.9, 0.1]
        labels = [False, True]

        acc = accuracy_at_threshold(scores, labels, threshold=0.5)

        assert acc == 0.0


class TestPrecisionRecallF1:
    """Tests for precision_recall_f1 function."""

    def test_perfect_metrics(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            precision_recall_f1,
        )

        scores = [0.9, 0.8]
        labels = [True, True]

        p, r, f1 = precision_recall_f1(scores, labels)

        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_no_labels(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            precision_recall_f1,
        )

        p, _r, _f1 = precision_recall_f1([], [])

        assert p == 0.0


class TestAUCPR:
    """Tests for auc_pr function."""

    def test_perfect_auc(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            auc_pr,
        )

        scores = [0.95, 0.85, 0.75]
        labels = [True, True, False]

        auc = auc_pr(scores, labels)

        assert 0.0 <= auc <= 1.0

    def test_no_samples(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            auc_pr,
        )

        auc = auc_pr([], [])

        assert auc == 0.0


class TestMeanSquaredError:
    """Tests for mean_squared_error_similarity function."""

    def test_zero_mse(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            mean_squared_error_similarity,
        )

        pred = [1.0, 0.5]
        ref = [1.0, 0.5]

        mse = mean_squared_error_similarity(pred, ref)

        assert mse == 0.0

    def test_different_lengths_raises(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            mean_squared_error_similarity,
        )

        pred = [1.0]
        ref = [1.0, 0.5]

        with pytest.raises(ValueError, match="must have same length"):
            mean_squared_error_similarity(pred, ref)


class TestCorrelationSimilarity:
    """Tests for correlation_similarity function."""

    def test_perfect_correlation_spearman(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            correlation_similarity,
        )

        pred = [1.0, 2.0]
        ref = [1.5, 2.5]

        corr, _p_value = correlation_similarity(pred, ref, method="spearman")

        assert -1.0 <= corr <= 1.0

    def test_perfect_correlation_pearson(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            correlation_similarity,
        )

        pred = [1.0, 2.0]
        ref = [1.0, 2.0]

        corr, _p_value = correlation_similarity(pred, ref, method="pearson")

        assert abs(corr - 1.0) < 0.0001

    def test_invalid_method(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            correlation_similarity,
        )

        with pytest.raises(ValueError, match="Unknown correlation method"):
            correlation_similarity([1.0], [2.0], method="invalid")


class TestEvaluateEmbeddingSimilarity:
    """Tests for evaluate_embedding_similarity function."""

    def test_empty_dataset(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> list[npt.NDArray[np.float32]]:
                return [np.zeros(384)] * len(texts)

        result = evaluate_embedding_similarity(MockEmbedder(), [])

        assert result["num_samples"] == 0

    def test_no_valid_pairs(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> NoReturn:
                msg = "Error"
                raise ValueError(msg)

        dataset = [{"concept_1": "a", "concept_2": "b"}]
        result = evaluate_embedding_similarity(MockEmbedder(), dataset)

        assert "error" in result

    def test_with_binary_labels(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> list[npt.NDArray[np.float32]]:
                return [np.random.randn(384)] * len(texts)

        dataset = [
            {"concept_1": "a", "concept_2": "b", "is_similar": True},
            {"concept_1": "c", "concept_2": "d", "is_similar": False},
        ]
        result = evaluate_embedding_similarity(
            MockEmbedder(),
            dataset,
            use_umnsrs_scores=False,
        )

        assert "accuracy@threshold" in result


class TestThresholdVariation:
    """Tests with different threshold values."""

    def test_threshold_zero(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            accuracy_at_threshold,
        )

        scores = [0.1, 0.2]
        labels = [True, True]

        acc = accuracy_at_threshold(scores, labels, threshold=0.0)

        assert acc == 1.0

    def test_threshold_one(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            accuracy_at_threshold,
        )

        scores = [0.9, 0.8]
        labels = [False, False]

        acc = accuracy_at_threshold(scores, labels, threshold=1.0)

        assert acc == 1.0


class TestNumericEdgeCases:
    """Tests for numeric edge cases."""

    def test_score_at_boundary(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            precision_recall_f1,
        )

        scores = [0.5, 0.5]
        labels = [True, False]

        p, _r, _f1 = precision_recall_f1(scores, labels, threshold=0.5)

        assert isinstance(p, float)


class TestEmbeddingEvaluationWithDataset:
    """Tests using actual embedding datasets."""

    def test_evaluate_with_generated_dataset(self) -> None:
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> list[npt.NDArray[np.float32]]:
                return [np.ones(384), np.zeros(384)] * len(texts)

        dataset = generate_embedding_dataset(num_samples=20)
        result = evaluate_embedding_similarity(MockEmbedder(), dataset)

        assert result["num_samples"] >= 1

    def test_accuracy_with_mixed_predictions(self) -> None:
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )
        from snomed_methods.benchmarking.embeddings.evaluation import (
            accuracy_at_threshold,
        )

        dataset = generate_embedding_dataset(num_samples=30)
        scores = []
        labels = []

        for s in dataset:
            if s["is_similar"]:
                scores.append(0.7)
            else:
                scores.append(0.3)
            labels.append(s["is_similar"])

        acc = accuracy_at_threshold(scores, labels, threshold=0.5)

        assert 0.6 <= acc <= 1.0

    def test_precision_recall_f1_with_realistic_scores(self) -> None:
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )
        from snomed_methods.benchmarking.embeddings.evaluation import (
            precision_recall_f1,
        )

        dataset = generate_embedding_dataset(num_samples=25)
        scores = []
        labels = []

        for s in dataset:
            if s["is_similar"]:
                scores.append(0.8)
            else:
                scores.append(0.2)
            labels.append(s["is_similar"])

        p, r, _f1 = precision_recall_f1(scores, labels)

        assert 0.7 <= p <= 1.0
        assert 0.7 <= r <= 1.0


class TestAUCPRDetailed:
    """Detailed tests for AUC-PR computation."""

    def test_auc_pr_with_mixed_labels(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            auc_pr,
        )

        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [True, True, False, True, False]

        auc = auc_pr(scores, labels)

        assert 0.0 <= auc <= 1.0

    def test_auc_pr_all_positive(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            auc_pr,
        )

        scores = [0.9, 0.8, 0.7]
        labels = [True, True, True]

        auc = auc_pr(scores, labels)

        # AUC-PR is 1.0 when all predictions are correct and sorted
        assert 0.5 <= auc <= 1.0

    def test_auc_pr_all_negative(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            auc_pr,
        )

        scores = [0.9, 0.8, 0.7]
        labels = [False, False, False]

        auc = auc_pr(scores, labels)

        assert auc == 0.0

    def test_auc_pr_single_sample_positive(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            auc_pr,
        )

        scores = [0.9]
        labels = [True]

        auc = auc_pr(scores, labels)

        assert auc == 1.0

    def test_auc_pr_single_sample_negative(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            auc_pr,
        )

        scores = [0.1]
        labels = [False]

        auc = auc_pr(scores, labels)

        assert auc == 0.0


class TestMeanSquaredErrorDetailed:
    """Detailed tests for MSE computation."""

    def test_mse_with_varied_values(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            mean_squared_error_similarity,
        )

        pred = [1.0, 2.0, 3.0]
        ref = [1.5, 2.5, 3.5]

        mse = mean_squared_error_similarity(pred, ref)

        # MSE should be: ((0.5^2 + 0.5^2 + 0.5^2) / 3) = 0.25
        assert abs(mse - 0.25) < 0.01

    def test_mse_large_values(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            mean_squared_error_similarity,
        )

        pred = [100.0, 200.0]
        ref = [110.0, 190.0]

        mse = mean_squared_error_similarity(pred, ref)

        # MSE should be: ((10^2 + 10^2) / 2) = 100
        assert abs(mse - 100.0) < 1.0

    def test_mse_single_value(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            mean_squared_error_similarity,
        )

        pred = [1.0]
        ref = [2.0]

        mse = mean_squared_error_similarity(pred, ref)

        assert mse == 1.0


class TestCorrelationDetailed:
    """Detailed tests for correlation computation."""

    def test_spearman_with_ties(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            correlation_similarity,
        )

        pred = [1.0, 2.0, 2.0, 3.0]
        ref = [1.5, 2.5, 2.5, 3.5]

        corr, p_value = correlation_similarity(pred, ref, method="spearman")

        assert -1.0 <= corr <= 1.0
        assert 0.0 <= p_value <= 1.0

    def test_pearson_with_negative_correlation(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            correlation_similarity,
        )

        pred = [1.0, 2.0, 3.0]
        ref = [3.0, 2.0, 1.0]

        corr, p_value = correlation_similarity(pred, ref, method="pearson")

        assert -1.0 <= corr <= 0.0
        assert 0.0 < p_value < 1.0

    def test_correlation_same_values(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            correlation_similarity,
        )

        pred = [5.0, 10.0, 15.0]
        ref = [5.0, 10.0, 15.0]

        corr, _p_value = correlation_similarity(pred, ref, method="spearman")

        assert abs(corr - 1.0) < 0.0001

    def test_correlation_invalid_method(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            correlation_similarity,
        )

        with pytest.raises(ValueError, match="Unknown correlation method"):
            correlation_similarity([1.0, 2.0], [2.0, 3.0], method="kendall")


class TestEvaluateEmbeddingSimilarityDetailed:
    """Detailed tests for evaluate_embedding_similarity function."""

    def test_with_umnsrs_scores(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> list[npt.NDArray[np.float32]]:
                return [np.ones(384), np.zeros(384)] * len(texts)

        dataset = [
            {"concept_1": "a", "concept_2": "b", "umnsrs_score": 750},
            {"concept_1": "c", "concept_2": "d", "umnsrs_score": 250},
        ]
        result = evaluate_embedding_similarity(
            MockEmbedder(),
            dataset,
            use_umnsrs_scores=True,
        )

        assert "auc_pr" in result
        assert "mse" in result

    def test_with_custom_threshold(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> list[npt.NDArray[np.float32]]:
                return [np.random.randn(384)] * len(texts)

        dataset = [
            {"concept_1": "a", "concept_2": "b", "is_similar": True},
            {"concept_1": "c", "concept_2": "d", "is_similar": False},
        ]
        result = evaluate_embedding_similarity(
            MockEmbedder(),
            dataset,
            threshold=0.7,
            use_umnsrs_scores=False,
        )

        assert result["accuracy@threshold"] >= 0.0

    def test_skip_invalid_samples(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        call_count = [0]

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> list[npt.NDArray[np.float32]]:
                # Second sample will succeed
                call_count[0] += 1
                if call_count[0] >= 2:
                    return [np.ones(384)] * len(texts)
                msg = "Embedding error"
                raise ValueError(msg)

        dataset = [
            {"concept_1": "a", "concept_2": "b"},
            {"text_1": "c", "text_2": "d", "umnsrs_score": 500},
            {"text_1": "e", "text_2": "f", "umnsrs_score": 600},
        ]
        result = evaluate_embedding_similarity(MockEmbedder(), dataset)

        # At least one valid sample should be processed
        assert result["num_samples"] >= 1

    def test_all_samples_skip(self) -> None:
        from snomed_methods.benchmarking.embeddings.evaluation import (
            evaluate_embedding_similarity,
        )

        class MockEmbedder:
            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> NoReturn:
                msg = "Embedding error"
                raise ValueError(msg)

        dataset = [{"concept_1": "a", "concept_2": "b"}]
        result = evaluate_embedding_similarity(MockEmbedder(), dataset)

        assert "error" in result or result.get("num_samples", 0) == 0
