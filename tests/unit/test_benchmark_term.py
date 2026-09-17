"""Unit tests for term lookup benchmarking utilities."""


class TestTermDataset:
    """Tests for term dataset generation functionality."""

    def test_generate_term_dataset_structure(self):
        """Test that generated dataset has correct structure."""
        from snomed_methods.benchmarking.term import generate_term_dataset

        dataset = generate_term_dataset(num_samples=10)

        assert len(dataset) == 10
        sample = dataset[0]

        assert "term" in sample
        assert "expected_cui" in sample
        assert isinstance(sample["term"], str)
        assert isinstance(sample["expected_cui"], str)
        assert len(sample["term"]) > 0

    def test_generate_term_dataset_with_custom_terms(self):
        """Test custom medical terms."""
        from snomed_methods.benchmarking.term import generate_term_dataset

        custom_terms = [
            ("heart attack", "22298006"),
            ("stroke", "386157005"),
        ]

        dataset = generate_term_dataset(num_samples=5, medical_terms=custom_terms)

        for sample in dataset:
            assert sample["term"] in [t for t, _ in custom_terms]


class TestTermMetrics:
    """Tests for term lookup metrics."""

    def test_recall_at_k_found(self):
        """Test recall@K when CUI is found."""
        from snomed_methods.benchmarking.term import recall_at_k

        result_cuis = ["A", "B", "C", "D"]
        expected_cui = "B"

        r3 = recall_at_k(result_cuis, expected_cui, k=3)
        assert r3 == 1.0

    def test_recall_at_k_not_found(self):
        """Test recall@K when CUI is not found."""
        from snomed_methods.benchmarking.term import recall_at_k

        result_cuis = ["A", "B", "C"]
        expected_cui = "Z"

        r3 = recall_at_k(result_cuis, expected_cui, k=3)
        assert r3 == 0.0

    def test_mean_reciprocal_rank_found(self):
        """Test MRR when CUI is found."""
        from snomed_methods.benchmarking.term import mean_reciprocal_rank

        result_cuis = ["A", "B", "C"]
        expected_cui = "B"

        rr = mean_reciprocal_rank(result_cuis, expected_cui)
        assert abs(rr - 0.5) < 0.01

    def test_mean_reciprocal_rank_not_found(self):
        """Test MRR when CUI is not found."""
        from snomed_methods.benchmarking.term import mean_reciprocal_rank

        result_cuis = ["A", "B"]
        expected_cui = "Z"

        rr = mean_reciprocal_rank(result_cuis, expected_cui)
        assert rr == 0.0


class TestEvaluateTermLookup:
    """Tests for evaluate_term_lookup function."""

    def test_evaluate_with_tuple_prediction(self):
        """Test evaluation with (CUI, term) tuple prediction."""
        from snomed_methods.benchmarking.term import (
            evaluate_term_lookup,
        )

        def mock_lookup(term):
            return [("C001", "Matched Term"), ("C002", "Another Match")]

        dataset = [
            {"term": "test term", "expected_cui": "C001"},
        ]

        results = evaluate_term_lookup(mock_lookup, dataset)

        assert "recall@1" in results
        assert "num_samples" in results

    def test_evaluate_multiple_k_values(self):
        """Test evaluation with multiple K values."""
        from snomed_methods.benchmarking.term import (
            evaluate_term_lookup,
        )

        def mock_lookup(term):
            return [(f"C{i:03d}", f"Term {i}") for i in range(20)]

        dataset = [
            {"term": "test", "expected_cui": "C001"},
        ]

        results = evaluate_term_lookup(
            mock_lookup,
            dataset,
            k_values=[1, 3, 5],
        )

        assert "recall@1" in results
        assert "recall@3" in results
        assert "recall@5" in results


class TestTermIntegration:
    """Integration tests for term lookup benchmarking."""

    def test_full_pipeline(self):
        """Test complete evaluation pipeline."""
        from snomed_methods.benchmarking.term import (
            evaluate_term_lookup,
            generate_term_dataset,
        )

        def mock_lookup(term):
            return [(f"C{i:03d}", f"Matched_{i}") for i in range(15)]

        dataset = generate_term_dataset(num_samples=15)

        results = evaluate_term_lookup(mock_lookup, dataset)

        assert results["num_samples"] == 15
        assert "hit_rate" in results
        assert "mrr" in results
        assert "recall@1" in results

    def test_dataset_loading(self):
        """Test loading pre-generated datasets."""
        from snomed_methods.benchmarking.term import load_term_datasets

        datasets = load_term_datasets()

        assert "small" in datasets
        assert "medium" in datasets
        assert "large" in datasets

        for _name, data in datasets.items():
            assert len(data) > 0
            assert "term" in data[0]
            assert "expected_cui" in data[0]


class TestTermEdgeCases:
    """Tests for edge cases in term lookup evaluation."""

    def test_empty_prediction(self):
        """Test with empty prediction."""
        from snomed_methods.benchmarking.term import (
            evaluate_term_lookup,
        )

        def mock_lookup(term):
            return []

        dataset = [
            {"term": "test", "expected_cui": "C001"},
        ]

        results = evaluate_term_lookup(mock_lookup, dataset)

        assert "num_samples" in results

    def test_prediction_none(self):
        """Test with None-like prediction."""
        from snomed_methods.benchmarking.term import (
            evaluate_term_lookup,
        )

        def mock_lookup(term):
            return []

        dataset = [
            {"term": "test", "expected_cui": "C001"},
        ]

        results = evaluate_term_lookup(mock_lookup, dataset)

        assert results["num_samples"] == 1
