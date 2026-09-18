"""Unit tests for hierarchy expansion benchmarking utilities."""

from __future__ import annotations


class TestHierarchyDataset:
    """Tests for hierarchy dataset generation functionality."""

    def test_generate_hierarchy_dataset_structure(self) -> None:
        """Test that generated dataset has correct structure."""
        from snomed_methods.benchmarking.hierarchy import generate_hierarchy_dataset

        dataset = generate_hierarchy_dataset(num_samples=10)

        assert len(dataset) == 10
        sample = dataset[0]

        assert "seed_cui" in sample
        assert "expected_related" in sample
        assert isinstance(sample["seed_cui"], str)
        assert isinstance(sample["expected_related"], list)
        assert len(sample["expected_related"]) > 0

    def test_generate_hierarchy_dataset_with_custom_cuis(self) -> None:
        """Test custom seed CUIs."""
        from snomed_methods.benchmarking.hierarchy import generate_hierarchy_dataset

        custom_cuis = ["C100", "C200", "C300"]

        dataset = generate_hierarchy_dataset(num_samples=5, seed_cuis=custom_cuis)

        for sample in dataset:
            assert sample["seed_cui"] in custom_cuis


class TestHierarchyMetrics:
    """Tests for hierarchy expansion metrics."""

    def test_exact_match_rate(self) -> None:
        """Test exact match rate calculation."""
        from snomed_methods.benchmarking.hierarchy import exact_match_rate

        # Exact match
        predicted = ["A", "B", "C"]
        expected = {"A", "B", "C"}
        assert exact_match_rate(predicted, expected) == 1.0

    def test_exact_match_rate_partial(self) -> None:
        """Test exact match with partial overlap."""
        from snomed_methods.benchmarking.hierarchy import exact_match_rate

        predicted = ["A", "B", "D"]
        expected = {"A", "B", "C"}
        assert exact_match_rate(predicted, expected) == 0.0

    def test_recall_at_k(self) -> None:
        """Test recall@K calculation."""
        from snomed_methods.benchmarking.hierarchy import recall_at_k

        predicted = ["A", "B", "C", "D"]
        expected = {"A", "C", "E", "F"}

        r4 = recall_at_k(predicted, expected, k=4)
        assert abs(r4 - 0.5) < 0.01

    def test_precision_at_k(self) -> None:
        """Test precision@K calculation."""
        from snomed_methods.benchmarking.hierarchy import precision_at_k

        predicted = ["A", "B", "C"]
        expected = {"A", "D", "E"}

        p3 = precision_at_k(predicted, expected, k=3)
        assert abs(p3 - 0.3333) < 0.01

    def test_f1_at_k(self) -> None:
        """Test F1@K calculation."""
        from snomed_methods.benchmarking.hierarchy import f1_at_k

        predicted = ["A", "B"]
        expected = {"A", "C"}

        f1 = f1_at_k(predicted, expected, k=2)
        # Precision = 0.5, Recall = 0.5, F1 = 0.5
        assert abs(f1 - 0.5) < 0.01

    def test_jaccard_similarity(self) -> None:
        """Test Jaccard similarity calculation."""
        from snomed_methods.benchmarking.hierarchy import jaccard_similarity

        predicted = ["A", "B", "C"]
        expected = {"A", "C", "D"}

        # Intersection: A, C (size 2)
        # Union: A, B, C, D (size 4)
        # Jaccard = 2/4 = 0.5
        j = jaccard_similarity(predicted, expected, k=3)
        assert abs(j - 0.5) < 0.01


class TestEvaluateHierarchyExpansion:
    """Tests for evaluate_hierarchy_expansion function."""

    def test_evaluate_with_list_prediction(self) -> None:
        """Test evaluation with list-based prediction."""
        from snomed_methods.benchmarking.hierarchy import (
            evaluate_hierarchy_expansion,
        )

        def mock_expansion(_seed_cui) -> list[str]:
            return [f"C{_seed_cui[-3:]}01", f"C{_seed_cui[-3:]}02"]

        dataset = [
            {"seed_cui": "123456789", "expected_related": ["C78901"]},
        ]

        results = evaluate_hierarchy_expansion(mock_expansion, dataset)

        assert "exact_match_rate" in results
        assert "num_samples" in results

    def test_evaluate_multiple_k_values(self) -> None:
        """Test evaluation with multiple K values."""
        from snomed_methods.benchmarking.hierarchy import (
            evaluate_hierarchy_expansion,
        )

        def mock_expansion(_seed_cui) -> list[str]:
            return [f"C{i}" for i in range(30)]

        dataset = [
            {"seed_cui": "123", "expected_related": ["C1", "C2", "C3"]},
        ]

        results = evaluate_hierarchy_expansion(
            mock_expansion,
            dataset,
            k_values=[5, 10, 20],
        )

        assert "recall@5" in results
        assert "recall@10" in results
        assert "recall@20" in results


class TestHierarchyIntegration:
    """Integration tests for hierarchy benchmarking."""

    def test_full_pipeline(self) -> None:
        """Test complete evaluation pipeline."""
        from snomed_methods.benchmarking.hierarchy import (
            evaluate_hierarchy_expansion,
            generate_hierarchy_dataset,
        )

        def mock_expansion(_seed_cui) -> list[str]:
            return [f"C{_seed_cui[-3:]}_{i}" for i in range(20)]

        dataset = generate_hierarchy_dataset(num_samples=15)

        results = evaluate_hierarchy_expansion(mock_expansion, dataset)

        assert results["num_samples"] == 15
        assert "exact_match_rate" in results
        assert "recall@5" in results
        assert "precision@5" in results
        assert "f1@5" in results

    def test_dataset_loading(self) -> None:
        """Test loading pre-generated datasets."""
        from snomed_methods.benchmarking.hierarchy import load_hierarchy_datasets

        datasets = load_hierarchy_datasets()

        assert "small" in datasets
        assert "medium" in datasets
        assert "large" in datasets

        for data in datasets.values():
            assert len(data) > 0
            assert "seed_cui" in data[0]
            assert "expected_related" in data[0]


class TestHierarchyEdgeCases:
    """Tests for edge cases in hierarchy evaluation."""

    def test_empty_expected_set(self) -> None:
        """Test with empty expected set."""
        from snomed_methods.benchmarking.hierarchy import (
            evaluate_hierarchy_expansion,
        )

        def mock_expansion(_seed_cui) -> list[str]:
            return []

        dataset = [
            {"seed_cui": "123", "expected_related": []},
        ]

        results = evaluate_hierarchy_expansion(mock_expansion, dataset)

        # Should handle gracefully
        assert "num_samples" in results

    def test_empty_prediction(self) -> None:
        """Test with empty prediction."""
        from snomed_methods.benchmarking.hierarchy import (
            evaluate_hierarchy_expansion,
        )

        def mock_expansion(_seed_cui) -> list[str]:
            return []

        dataset = [
            {"seed_cui": "123", "expected_related": ["C001"]},
        ]

        results = evaluate_hierarchy_expansion(mock_expansion, dataset)

        assert "num_samples" in results
