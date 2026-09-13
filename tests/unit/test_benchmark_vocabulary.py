"""Unit tests for vocabulary mapping benchmarking utilities."""


class TestMappingDataset:
    """Tests for mapping dataset generation functionality."""

    def test_generate_mapping_dataset_structure(self):
        """Test that generated dataset has correct structure."""
        from snomed_methods.benchmarking.vocabulary import generate_mapping_dataset

        dataset = generate_mapping_dataset(num_samples=10)

        assert len(dataset) == 10
        sample = dataset[0]

        assert "snomed_cui" in sample
        assert "target_codes" in sample
        assert isinstance(sample["snomed_cui"], str)
        assert isinstance(sample["target_codes"], list)
        assert len(sample["target_codes"]) > 0

    def test_generate_mapping_dataset_num_samples(self):
        """Test number of samples generation."""
        from snomed_methods.benchmarking.vocabulary import generate_mapping_dataset

        dataset = generate_mapping_dataset(num_samples=25)

        assert len(dataset) == 25


class TestVocabularyMetrics:
    """Tests for vocabulary mapping metrics."""

    def test_precision_at_k(self):
        """Test precision@K calculation."""
        from snomed_methods.benchmarking.vocabulary import precision_at_k

        predicted = ["A", "B", "C"]
        expected = {"A", "D", "E"}

        p3 = precision_at_k(predicted, expected, k=3)
        assert abs(p3 - 0.3333) < 0.01

    def test_recall_at_k(self):
        """Test recall@K calculation."""
        from snomed_methods.benchmarking.vocabulary import recall_at_k

        predicted = ["A", "B", "C"]
        expected = {"A", "C", "E"}

        r3 = recall_at_k(predicted, expected, k=3)
        assert abs(r3 - 0.6667) < 0.01

    def test_coverage_rate(self):
        """Test coverage rate calculation."""
        from snomed_methods.benchmarking.vocabulary import coverage_rate

        predicted = ["A", "B", "C"]
        expected = {"A", "C", "E"}

        cov = coverage_rate(predicted, expected)
        assert abs(cov - 0.6667) < 0.01

    def test_exact_match_rate(self):
        """Test exact match rate calculation."""
        from snomed_methods.benchmarking.vocabulary import exact_match_rate

        predicted = ["A", "B"]
        expected = {"A", "B"}
        assert exact_match_rate(predicted, expected) == 1.0

    def test_mean_reciprocal_rank(self):
        """Test MRR calculation."""
        from snomed_methods.benchmarking.vocabulary import mean_reciprocal_rank

        predicted = ["A", "B", "C"]
        expected = {"B"}

        rr = mean_reciprocal_rank(predicted, expected)
        assert abs(rr - 0.5) < 0.01


class TestEvaluateMapper:
    """Tests for evaluate_mapper function."""

    def test_evaluate_with_list_prediction(self):
        """Test evaluation with list-based prediction."""
        from snomed_methods.benchmarking.vocabulary import (
            evaluate_mapper,
        )

        def mock_mapper(snomed_cui):
            return ["ICD_E11.9", "LOINC_4544-3"]

        dataset = [
            {"snomed_cui": "123", "target_codes": ["ICD_E11.9"]},
        ]

        results = evaluate_mapper(mock_mapper, dataset)

        assert "coverage_rate" in results
        assert "num_samples" in results

    def test_evaluate_multiple_k_values(self):
        """Test evaluation with multiple K values."""
        from snomed_methods.benchmarking.vocabulary import (
            evaluate_mapper,
        )

        def mock_mapper(snomed_cui):
            return [f"C{i}" for i in range(10)]

        dataset = [
            {"snomed_cui": "123", "target_codes": ["C1", "C2"]},
        ]

        results = evaluate_mapper(
            mock_mapper,
            dataset,
            k_values=[1, 3, 5],
        )

        assert "precision@1" in results
        assert "precision@3" in results
        assert "precision@5" in results


class TestVocabularyIntegration:
    """Integration tests for vocabulary mapping benchmarking."""

    def test_full_pipeline(self):
        """Test complete evaluation pipeline."""
        from snomed_methods.benchmarking.vocabulary import (
            evaluate_mapper,
            generate_mapping_dataset,
        )

        def mock_mapper(snomed_cui):
            return [f"ICD_{snomed_cui[:3]}", f"LOINC_{snomed_cui[-3:]}"]

        dataset = generate_mapping_dataset(num_samples=15)

        results = evaluate_mapper(mock_mapper, dataset)

        assert results["num_samples"] == 15
        assert "coverage_rate" in results
        assert "mrr" in results
        assert "precision@1" in results

    def test_dataset_loading(self):
        """Test loading pre-generated datasets."""
        from snomed_methods.benchmarking.vocabulary import load_mapping_datasets

        datasets = load_mapping_datasets()

        assert "small" in datasets
        assert "medium" in datasets
        assert "large" in datasets

        for name, data in datasets.items():
            assert len(data) > 0
            assert "snomed_cui" in data[0]
            assert "target_codes" in data[0]


class TestVocabularyEdgeCases:
    """Tests for edge cases in vocabulary evaluation."""

    def test_empty_prediction(self):
        """Test with empty prediction."""
        from snomed_methods.benchmarking.vocabulary import (
            evaluate_mapper,
        )

        def mock_mapper(snomed_cui):
            return []

        dataset = [
            {"snomed_cui": "123", "target_codes": ["C001"]},
        ]

        results = evaluate_mapper(mock_mapper, dataset)

        assert "num_samples" in results

    def test_empty_expected(self):
        """Test with empty expected set."""
        from snomed_methods.benchmarking.vocabulary import (
            evaluate_mapper,
        )

        def mock_mapper(snomed_cui):
            return ["C001", "C002"]

        dataset = [
            {"snomed_cui": "123", "target_codes": []},
        ]

        results = evaluate_mapper(mock_mapper, dataset)

        assert "num_samples" in results
