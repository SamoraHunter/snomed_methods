"""Unit tests for UMNSRS benchmarking utilities."""

import os

import pytest


class TestUMNSRSDataset:
    """Tests for UMNSRS dataset loading functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_dir(self, tmp_path) -> None:
        """Create a temporary directory for test data."""
        self.test_cache_dir = str(tmp_path / "umnsrs_test_data")
        os.makedirs(self.test_cache_dir, exist_ok=True)

    def test_download_invalid_subset(self) -> None:
        """Test that invalid subset raises ValueError."""
        from snomed_methods.benchmarking.umnsrs import download_umnsrs

        with pytest.raises(ValueError, match="Invalid subset"):
            download_umnsrs(subset="invalid_subsets")

    def test_get_pairs_structure(self) -> None:
        """Test get_umnsrs_pairs returns correct structure (with mock)."""
        # Test that the function works without actually downloading
        from snomed_methods.benchmarking.umnsrs import evaluate_model

        mock_pairs = [
            {"text_1": "A", "text_2": "B", "label": 600.0},
        ]

        def mock_model(_t1, _t2) -> float:
            return 500.0

        results = evaluate_model(mock_model, mock_pairs, metric="mae")
        assert "mae" in results

    def test_download_label_conversion(self) -> None:
        """Test that labels are converted to float by default."""
        from snomed_methods.benchmarking.umnsrs import download_umnsrs

        dataset = download_umnsrs(subset="relatedness", cache_dir=self.test_cache_dir)
        sample = dataset[0]
        assert isinstance(sample["label"], float)

    def test_download_label_conversion_disabled(self) -> None:
        """Test that labels can remain as strings."""
        from snomed_methods.benchmarking.umnsrs import download_umnsrs

        dataset = download_umnsrs(
            subset="relatedness",
            convert_labels_to_float=False,
            cache_dir=self.test_cache_dir,
        )
        sample = dataset[0]
        assert isinstance(sample["label"], str)


class TestBenchmarkEvaluation:
    """Tests for evaluation metrics."""

    def test_spearman_correlation(self) -> None:
        """Test Spearman correlation computation."""
        from snomed_methods.benchmarking.umnsrs import spearman_correlation

        predictions = [1.0, 2.0, 3.0]
        references = [1.1, 2.0, 2.9]

        corr, _p_value = spearman_correlation(predictions, references)

        assert isinstance(corr, float)
        assert -1 <= corr <= 1

    def test_mean_absolute_error(self) -> None:
        """Test MAE computation."""
        from snomed_methods.benchmarking.umnsrs import mean_absolute_error

        predictions = [10.0, 20.0]
        references = [12.0, 18.0]

        mae = mean_absolute_error(predictions, references)
        assert abs(mae - 2.0) < 0.01


class TestBenchmarkFunctions:
    """Tests for benchmark evaluation functions."""

    def test_evaluate_model(self) -> None:
        """Test evaluating with single metric."""
        from snomed_methods.benchmarking.umnsrs import evaluate_model

        def mock_model(_text1, _text2) -> float:
            return 500.0

        pairs = [
            {"text_1": "A", "text_2": "B", "label": 600.0},
        ]

        results = evaluate_model(mock_model, pairs, metric="spearman")
        assert "spearman_correlation" in results


class TestRankingMetrics:
    """Tests for ranking-based evaluation metrics (Recall@K, MRR, etc.)."""

    def test_precision_at_k_basic(self) -> None:
        """Test basic Precision@K calculation."""
        from snomed_methods.benchmarking.umnsrs import precision_at_k

        predicted = ["A", "B", "C", "D", "E"]
        relevant = {"A", "C", "F"}

        p5 = precision_at_k(predicted, relevant, k=5)
        assert abs(p5 - 0.4) < 0.01

        p3 = precision_at_k(predicted, relevant, k=3)
        assert abs(p3 - 0.6667) < 0.01

    def test_precision_at_k_empty(self) -> None:
        """Test Precision@K with empty lists."""
        from snomed_methods.benchmarking.umnsrs import precision_at_k

        p = precision_at_k([], set(), k=5)
        assert p == 0.0

    def test_recall_at_k_basic(self) -> None:
        """Test basic Recall@K calculation."""
        from snomed_methods.benchmarking.umnsrs import recall_at_k

        predicted = ["A", "B", "C", "D"]
        relevant = {"A", "C", "E", "F"}

        r4 = recall_at_k(predicted, relevant, k=4)
        assert abs(r4 - 0.5) < 0.01

    def test_f1_at_k_basic(self) -> None:
        """Test basic F1@K calculation."""
        from snomed_methods.benchmarking.umnsrs import f1_at_k

        predicted = ["A", "B", "C"]
        relevant = {"A", "D", "E"}

        f1 = f1_at_k(predicted, relevant, k=3)
        # Precision = 1/3, Recall = 1/3, F1 = 1/3
        assert abs(f1 - 0.3333) < 0.01

    def test_mean_reciprocal_rank_basic(self) -> None:
        """Test basic MRR calculation."""
        from snomed_methods.benchmarking.umnsrs import mean_reciprocal_rank

        predicted = ["A", "B", "C", "D"]
        relevant = {"C", "E"}

        rr = mean_reciprocal_rank(predicted, relevant)
        assert abs(rr - 1.0 / 3) < 0.01

    def test_mean_reciprocal_rank_not_found(self) -> None:
        """Test MRR when no relevant items found."""
        from snomed_methods.benchmarking.umnsrs import mean_reciprocal_rank

        predicted = ["A", "B"]
        relevant = {"C", "D"}

        rr = mean_reciprocal_rank(predicted, relevant)
        assert rr == 0.0

    def test_average_precision_basic(self) -> None:
        """Test basic Average Precision calculation."""
        from snomed_methods.benchmarking.umnsrs import average_precision

        predicted = ["A", "B", "C"]
        relevant = {"A", "C"}

        ap = average_precision(predicted, relevant)
        # At position 1: A is relevant -> prec=1/1, at position 3:
        # C is relevant -> prec=2/3, AP = (1/2) * (1 + 2/3) = 5/6 ≈ 0.8333
        assert abs(ap - 0.8333) < 0.01

    def test_average_precision_none_relevant(self) -> None:
        """Test Average Precision with no relevant items."""
        from snomed_methods.benchmarking.umnsrs import average_precision

        predicted = ["A", "B", "C"]
        relevant: set = set()

        ap = average_precision(predicted, relevant)
        assert ap == 0.0
