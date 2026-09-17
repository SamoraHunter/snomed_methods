"""Unit tests for embedding dataset generation."""


class TestEmbeddingDatasetGeneration:
    """Tests for generate_embedding_dataset function."""

    def test_generate_with_default_parameters(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        dataset = generate_embedding_dataset()

        assert len(dataset) >= 50
        sample = dataset[0]
        assert "concept_1" in sample
        assert "concept_2" in sample
        assert "is_similar" in sample
        assert "similarity_label" in sample

    def test_generate_custom_num_samples(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        for num in [10, 50]:
            dataset = generate_embedding_dataset(num_samples=num)
            assert len(dataset) >= min(5, num)

    def test_generate_with_custom_snomed_concepts(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        custom_concepts = ["1001", "1002"]
        dataset = generate_embedding_dataset(
            num_samples=5, snomed_concepts=custom_concepts
        )

        assert len(dataset) == 5
        for sample in dataset:
            assert isinstance(sample["concept_1"], str)
            assert isinstance(sample["concept_2"], str)
            assert sample["concept_1"] != sample["concept_2"]

    def test_generate_handles_fewer_than_two_concepts(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        dataset = generate_embedding_dataset(num_samples=5, snomed_concepts=["1001"])

        assert len(dataset) == 5
        for sample in dataset:
            assert isinstance(sample["concept_1"], str)
            assert isinstance(sample["concept_2"], str)

    def test_similar_pairs_share_disease_group(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        dataset = generate_embedding_dataset(num_samples=10)

        diabetes_group = ["73211009", "237550006", "237600004"]
        hypertension_group = ["38341003", "59621000"]

        for sample in dataset:
            c1, c2 = sample["concept_1"], sample["concept_2"]

            if sample["is_similar"]:
                both_diabetes = c1 in diabetes_group and c2 in diabetes_group
                both_hypertension = (
                    c1 in hypertension_group and c2 in hypertension_group
                )
                assert both_diabetes or both_hypertension

    def test_dissimilar_pairs_different_groups(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        dataset = generate_embedding_dataset(num_samples=30)

        for sample in dataset:
            if not sample["is_similar"]:
                assert sample["similarity_label"] == "low"


class TestEmbeddingDatasetLabels:
    """Tests for similarity label assignment."""

    def test_high_similarity_label(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        dataset = generate_embedding_dataset(num_samples=10)

        for sample in dataset:
            if sample["is_similar"]:
                assert sample["similarity_label"] == "high"

    def test_low_similarity_label(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        dataset = generate_embedding_dataset(num_samples=10)

        for sample in dataset:
            if not sample["is_similar"]:
                assert sample["similarity_label"] == "low"


class TestLoadEmbeddingDatasets:
    """Tests for load_embedding_datasets function."""

    def test_load_returns_all_sizes(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            load_embedding_datasets,
        )

        datasets = load_embedding_datasets()

        assert "small" in datasets
        assert "medium" in datasets
        assert "large" in datasets

    def test_load_dataset_sizes(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            load_embedding_datasets,
        )

        datasets = load_embedding_datasets()

        assert len(datasets["small"]) >= 20
        assert len(datasets["medium"]) >= 50
        assert len(datasets["large"]) >= 50

    def test_load_with_custom_cache_dir(self, tmp_path):
        from snomed_methods.benchmarking.embeddings.dataset import (
            load_embedding_datasets,
        )

        datasets = load_embedding_datasets(cache_dir=str(tmp_path))

        assert len(datasets["small"]) > 0

    def test_generate_minimum_samples(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            generate_embedding_dataset,
        )

        dataset = generate_embedding_dataset(num_samples=2)

        assert len(dataset) >= 1
        sample = dataset[0]
        assert "concept_1" in sample
        assert "concept_2" in sample

    def test_load_datasets_all_have_valid_structure(self, tmp_path):
        from snomed_methods.benchmarking.embeddings.dataset import (
            load_embedding_datasets,
        )

        datasets = load_embedding_datasets(cache_dir=str(tmp_path))

        for _size_name, data in datasets.items():
            assert isinstance(data, list)
            if len(data) > 0:
                sample = data[0]
                assert "concept_1" in sample
                assert "concept_2" in sample
                assert "is_similar" in sample
                assert "similarity_label" in sample


class TestCreatePairsFromUMLSRS:
    """Tests for create_pairs_from_umnsrs function."""

    def test_basic_conversion(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            create_pairs_from_umnsrs,
        )

        umnsrs_pairs = [
            {"text_1": "diabetes", "text_2": "hypertension", "label": 750},
            {"text_1": "cancer", "text_2": "tumor", "label": 400},
        ]

        result = create_pairs_from_umnsrs(umnsrs_pairs)

        assert len(result) == 2
        assert result[0]["is_similar"] is True
        assert result[1]["is_similar"] is False

    def test_label_threshold(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            create_pairs_from_umnsrs,
        )

        umnsrs_pairs = [
            {"text_1": "a", "text_2": "b", "label": 500},
            {"text_1": "c", "text_2": "d", "label": 499},
            {"text_1": "e", "text_2": "f", "label": 501},
        ]

        result = create_pairs_from_umnsrs(umnsrs_pairs)

        assert result[0]["is_similar"] is True
        assert result[1]["is_similar"] is False
        assert result[2]["is_similar"] is True

    def test_default_label_value(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            create_pairs_from_umnsrs,
        )

        umnsrs_pairs = [
            {"text_1": "a", "text_2": "b"},
        ]

        result = create_pairs_from_umnsrs(umnsrs_pairs)

        assert len(result) == 1
        assert "umnsrs_score" in result[0]

    def test_preserves_original_fields(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            create_pairs_from_umnsrs,
        )

        umnsrs_pairs = [
            {"text_1": "test", "text_2": "concept", "label": 600},
        ]

        result = create_pairs_from_umnsrs(umnsrs_pairs)

        assert result[0]["concept_1"] == "test"
        assert result[0]["concept_2"] == "concept"

    def test_empty_input(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            create_pairs_from_umnsrs,
        )

        result = create_pairs_from_umnsrs([])

        assert len(result) == 0

    def test_label_value_at_threshold_boundary(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            create_pairs_from_umnsrs,
        )

        umnsrs_pairs = [
            {"text_1": "a", "text_2": "b", "label": 501},
            {"text_1": "c", "text_2": "d", "label": 499},
        ]

        result = create_pairs_from_umnsrs(umnsrs_pairs)

        assert result[0]["is_similar"] is True
        assert result[1]["is_similar"] is False

    def test_multiple_records_conversion(self):
        from snomed_methods.benchmarking.embeddings.dataset import (
            create_pairs_from_umnsrs,
        )

        umnsrs_pairs = [
            {"text_1": "diabetes", "text_2": "hypertension", "label": 750},
            {"text_1": "cancer", "text_2": "tumor", "label": 400},
            {"text_1": "fever", "text_2": "infection", "label": 650},
            {"text_1": "pain", "text_2": "symptom", "label": 550},
        ]

        result = create_pairs_from_umnsrs(umnsrs_pairs)

        assert len(result) == 4
        assert all("umnsrs_score" in r for r in result)
