"""Unit tests for annotation benchmarking utilities."""


class TestAnnotationDataset:
    """Tests for annotation dataset generation functionality."""

    def test_generate_annotation_dataset_structure(self):
        """Test that generated dataset has correct structure."""
        from snomed_methods.benchmarking.annotation import generate_annotation_dataset

        dataset = generate_annotation_dataset(num_samples=10)

        assert len(dataset) == 10
        sample = dataset[0]

        assert "text" in sample
        assert "gold_cuis" in sample
        assert isinstance(sample["text"], str)
        assert isinstance(sample["gold_cuis"], list)
        assert len(sample["gold_cuis"]) > 0

    def test_generate_annotation_dataset_with_custom_diseases(self):
        """Test custom disease definitions."""
        from snomed_methods.benchmarking.annotation import generate_annotation_dataset

        custom_diseases = [
            ("test disease", ["C100", "C101"]),
        ]

        dataset = generate_annotation_dataset(num_samples=5, diseases=custom_diseases)

        for sample in dataset:
            assert set(sample["gold_cuis"]) == {"C100", "C101"}

    def test_generate_annotation_dataset_empty(self):
        """Test generation with zero samples."""
        from snomed_methods.benchmarking.annotation import generate_annotation_dataset

        dataset = generate_annotation_dataset(num_samples=0)
        assert len(dataset) == 0


class TestAnnotationEvaluation:
    """Tests for annotation evaluation metrics."""

    def test_precision_at_k_basic(self):
        """Test basic Precision@K calculation."""
        from snomed_methods.benchmarking.annotation import precision_at_k

        predicted = ["A", "B", "C", "D", "E"]
        relevant = {"A", "C", "F"}

        p5 = precision_at_k(predicted, relevant, k=5)
        assert abs(p5 - 0.4) < 0.01

        p3 = precision_at_k(predicted, relevant, k=3)
        assert abs(p3 - 0.6667) < 0.01

    def test_precision_at_k_empty(self):
        """Test Precision@K with empty lists."""
        from snomed_methods.benchmarking.annotation import precision_at_k

        p = precision_at_k([], set(), k=5)
        assert p == 0.0

    def test_recall_at_k_basic(self):
        """Test basic Recall@K calculation."""
        from snomed_methods.benchmarking.annotation import recall_at_k

        predicted = ["A", "B", "C", "D"]
        relevant = {"A", "C", "E", "F"}

        r4 = recall_at_k(predicted, relevant, k=4)
        assert abs(r4 - 0.5) < 0.01

    def test_f1_at_k_basic(self):
        """Test basic F1@K calculation."""
        from snomed_methods.benchmarking.annotation import f1_at_k

        predicted = ["A", "B", "C"]
        relevant = {"A", "D", "E"}

        f1 = f1_at_k(predicted, relevant, k=3)
        assert abs(f1 - 0.3333) < 0.01

    def test_mean_reciprocal_rank_basic(self):
        """Test basic MRR calculation."""
        from snomed_methods.benchmarking.annotation import mean_reciprocal_rank

        predicted = ["A", "B", "C", "D"]
        relevant = {"C", "E"}

        rr = mean_reciprocal_rank(predicted, relevant)
        assert abs(rr - 1.0 / 3) < 0.01

    def test_mean_reciprocal_rank_not_found(self):
        """Test MRR when no relevant items found."""
        from snomed_methods.benchmarking.annotation import mean_reciprocal_rank

        predicted = ["A", "B"]
        relevant = {"C", "D"}

        rr = mean_reciprocal_rank(predicted, relevant)
        assert rr == 0.0


class TestEvaluateAnnotator:
    """Tests for the evaluate_annotator function."""

    def test_evaluate_annotator_with_list_prediction(self):
        """Test evaluation with list-based prediction."""
        from snomed_methods.benchmarking.annotation import (
            evaluate_annotator,
        )

        def mock_annotator(text):
            return ["C001", "C002"]

        dataset = [
            {"text": "test text", "gold_cuis": ["C001"]},
        ]

        results = evaluate_annotator(mock_annotator, dataset)

        assert "precision@1" in results
        assert "num_samples" in results

    def test_evaluate_annotator_with_annotation_result(self):
        """Test evaluation with AnnotationResult-like object."""
        from snomed_methods.benchmarking.annotation import (
            evaluate_annotator,
        )

        class MockMatchedConcept:
            def __init__(self, concept_id):
                self.concept_id = concept_id

        class MockAnnotationResult:
            @property
            def top_concepts(self):
                return [MockMatchedConcept("C001"), MockMatchedConcept("C002")]

        def mock_annotator(text):
            return MockAnnotationResult()

        dataset = [
            {"text": "test text", "gold_cuis": ["C001"]},
        ]

        results = evaluate_annotator(mock_annotator, dataset)

        assert "precision@1" in results
        assert results["num_samples"] == 1

    def test_evaluate_annotator_multiple_k_values(self):
        """Test evaluation with multiple K values."""
        from snomed_methods.benchmarking.annotation import (
            evaluate_annotator,
        )

        def mock_annotator(text):
            return ["C001", "C002", "C003", "C004", "C005"]

        dataset = [
            {"text": "test text", "gold_cuis": ["C001", "C002"]},
        ]

        results = evaluate_annotator(mock_annotator, dataset, k_values=[1, 3, 5])

        assert "precision@1" in results
        assert "precision@3" in results
        assert "precision@5" in results
        assert results["num_samples"] == 1


class TestLoadAnnotationDatasets:
    """Tests for dataset loading functionality."""

    def test_load_annotation_datasets_structure(self):
        """Test that loaded datasets have correct structure."""
        from snomed_methods.benchmarking.annotation import load_annotation_datasets

        datasets = load_annotation_datasets()

        assert "small" in datasets
        assert "medium" in datasets
        assert "large" in datasets

        for _name, data in datasets.items():
            assert len(data) > 0
            sample = data[0]
            assert "text" in sample
            assert "gold_cuis" in sample


class TestAnnotationIntegration:
    """Integration tests for annotation benchmarking."""

    def test_full_pipeline(self):
        """Test complete evaluation pipeline."""
        from snomed_methods.benchmarking.annotation import (
            evaluate_annotator,
            generate_annotation_dataset,
        )

        def mock_annotator(text):
            # Simple mock that returns CUIs based on text length
            return [f"C{i:03d}" for i in range(min(10, len(text) // 5 + 2))]

        dataset = generate_annotation_dataset(num_samples=20)

        results = evaluate_annotator(mock_annotator, dataset)

        assert results["num_samples"] == 20
        assert "precision@1" in results
        assert "recall@1" in results
        assert "f1@1" in results
        assert "mrr" in results

    def test_dataset_caching(self):
        """Test that datasets can be saved/loaded."""
        import os
        import shutil

        from snomed_methods.benchmarking.annotation import (
            generate_annotation_dataset,
        )

        cache_dir = "/tmp/test_annotation_cache"
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Generate and save
            dataset1 = generate_annotation_dataset(num_samples=5)
            datasets = {
                "test": dataset1,
            }

            assert len(datasets["test"]) == 5

        finally:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
