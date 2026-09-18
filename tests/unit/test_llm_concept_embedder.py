#!/usr/bin/env python3
"""Unit tests for llm_concept_embedder module."""

from __future__ import annotations

import os
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pandas as pd


class TestClinicalConceptEmbedder(unittest.TestCase):
    """Test ClinicalConceptEmbedder class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        import snomed_methods.llm_concept_embedder as embedder_mod

        embedder_mod._SentenceTransformer = None

        self.test_concepts = pd.DataFrame(
            [
                {
                    "cui": "1001",
                    "preferred_name": "Meningioma",
                    "synonyms": "",
                    "type_id": "",
                },
                {
                    "cui": "2002",
                    "preferred_name": "Glioblastoma",
                    "synonyms": "GBM; Glioblastoma multiforme",
                    "type_id": "T-12",
                },
            ],
        )

    def tearDown(self) -> None:
        """Clean up after tests."""
        import snomed_methods.llm_concept_embedder as embedder_mod

        embedder_mod._SentenceTransformer = None

    def test_prepare_concept_text_basic(self) -> None:
        """Test basic concept text preparation."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()
            mock_model.encode = MagicMock(
                return_value=np.random.randn(2, 384).astype(np.float32),
            )
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
            )

            texts = embedder.prepare_concept_text(self.test_concepts)

            assert len(texts) == 2
            assert "Meningioma" in texts[0]
            assert "Concept ID: 1001" in texts[0]

    def test_prepare_concept_text_with_nan(self) -> None:
        """Test handling of NaN values."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
            )

            test_df = pd.DataFrame(
                [
                    {
                        "cui": "1001",
                        "preferred_name": np.nan,
                        "synonyms": None,
                        "type_id": "",
                    },
                ],
            )

            texts = embedder.prepare_concept_text(test_df)

            assert len(texts) == 1
            assert "Concept ID: 1001" in texts[0]

    def test_prepare_concept_text_missing_cui(self) -> None:
        """Test handling of missing cui column."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
            )

            test_df = pd.DataFrame(
                [
                    {
                        "concept_id": "1001",
                        "preferred_name": "Meningioma",
                    },
                ],
            )

            texts = embedder.prepare_concept_text(test_df)

            assert len(texts) == 1
            assert "Concept ID: 1001" in texts[0]

    def test_prepare_concept_text_semantic_tag(self) -> None:
        """Test semantic tag inclusion in formatted text."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
            )

            texts = embedder.prepare_concept_text(self.test_concepts)

            assert "Semantic Tag: T-12" in texts[1]

    def test_export_embeddings_pickle(self) -> None:
        """Test exporting embeddings to pickle."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
            )

            with TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test_embeddings.pkl")
                cui_to_embedding = {
                    "1001": np.random.randn(384).astype(np.float32),
                    "2002": np.random.randn(384).astype(np.float32),
                }

                embedder.export_embeddings(cui_to_embedding, output_path)

                assert os.path.exists(output_path)

    def test_export_embeddings_simple_filename(self) -> None:
        """Test export with simple filename (no directory path)."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
            )

            with TemporaryDirectory() as tmpdir:
                original_cwd = Path.cwd()
                try:
                    os.chdir(tmpdir)
                    output_path = "simple_test.pkl"
                    cui_to_embedding = {"1001": np.random.randn(384).astype(np.float32)}

                    embedder.export_embeddings(cui_to_embedding, output_path)

                    assert os.path.exists(output_path)
                finally:
                    os.chdir(original_cwd)

    def test_ollama_init_with_custom_url(self) -> None:
        """Test Ollama backend initialization with custom URL."""
        from unittest.mock import MagicMock, patch

        mock_ollama = MagicMock()
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="test-model",
                backend="ollama",
                ollama_base_url="http://custom-host:12345",
            )

            assert embedder.ollama_base_url == "http://custom-host:12345"
            assert embedder.backend == "ollama"


class TestConceptVectorSearch(unittest.TestCase):
    """Test ConceptVectorSearch class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_embeddings = {
            "C001": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "C002": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "C003": np.array([0.7, 0.8, 0.9], dtype=np.float32),
        }
        self.test_names = {
            "C001": "Meningioma",
            "C002": "Glioblastoma",
            "C003": "Astrocytoma",
        }

    def test_build_index(self) -> None:
        """Test building FAISS index."""
        import faiss

        mock_index = MagicMock()

        with patch("snomed_methods.llm_concept_embedder.faiss", spec=faiss):
            from snomed_methods.llm_concept_embedder import (
                ConceptVectorSearch,
            )
            from snomed_methods.llm_concept_embedder import (
                faiss as embedder_faiss,
            )

            embedder_faiss.IndexFlatIP.return_value = mock_index
            embedder_faiss.IndexHNSWFlat.return_value = mock_index

            search = ConceptVectorSearch(self.test_embeddings)
            search.build_index(index_type="FlatIP")

            assert search.index is not None
            assert len(search.cui_list) == 3

    def test_build_index_with_names(self) -> None:
        """Test building index with cui_to_name mapping."""
        with patch.dict("sys.modules", {"faiss": MagicMock()}):
            from snomed_methods.llm_concept_embedder import ConceptVectorSearch

            mock_faiss = sys.modules["faiss"]
            mock_index = MagicMock()
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.IndexHNSWFlat.return_value = mock_index

            search = ConceptVectorSearch(
                {"embeddings": self.test_embeddings, "names": self.test_names},
            )
            search.build_index()

            assert len(search.cui_list) == 3
            assert search.cui_to_name["C001"] == "Meningioma"

    def test_search_returns_results(self) -> None:
        """Test search returns valid results."""
        import os
        from unittest.mock import MagicMock, patch

        from snomed_methods.llm_concept_embedder import ConceptVectorSearch

        with TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "test.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"embeddings": self.test_embeddings}, f)

            mock_faiss = MagicMock()
            mock_index = MagicMock()
            mock_index.search.return_value = (
                np.array([[0.95, 0.85, 0.75]]),
                np.array([[0, 1, 2]]),
            )
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.IndexHNSWFlat.return_value = mock_index

            with patch.dict("sys.modules", {"faiss": mock_faiss}):
                search = ConceptVectorSearch(
                    pkl_path,
                    cui_to_name=self.test_names,
                )
                search.build_index(index_type="FlatIP")

                results = search.search(
                    query_embedding=np.random.randn(3).astype(np.float32),
                    top_k=3,
                )

                assert len(results) > 0
                for cui, name, score in results:
                    assert cui is not None
                    assert isinstance(name, str)
                    assert isinstance(score, float)

    def test_search_with_prebuilt_index(self) -> None:
        """Test query with pre-built index."""
        from unittest.mock import MagicMock, patch

        from snomed_methods.llm_concept_embedder import ConceptVectorSearch

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.95]]),
            np.array([[0]]),
        )
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            search = ConceptVectorSearch(
                {"embeddings": self.test_embeddings, "names": self.test_names},
            )
            search.build_index()

            query_vec = np.array([0.12, 0.22, 0.32], dtype=np.float32)

            results = search.search(query_embedding=query_vec, top_k=1)

            assert len(results) == 1
            assert results[0][0] == "C001"

    def test_load_embeddings_from_pkl_file(self) -> None:
        """Test loading embeddings from pickle file."""
        import os
        from unittest.mock import MagicMock, patch

        from snomed_methods.llm_concept_embedder import ConceptVectorSearch

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        with TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "test.pkl")
            data = {
                "embeddings": self.test_embeddings,
                "names": self.test_names,
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)

            with patch.dict("sys.modules", {"faiss": mock_faiss}):
                search = ConceptVectorSearch(pkl_path)
                search.build_index()

                assert len(search.cui_list) == 3
                assert search.cui_to_name["C001"] == "Meningioma"

    def test_embedder_reuse_for_query(self) -> None:
        """Test that embedder is reused for query embedding."""
        from snomed_methods.llm_concept_embedder import ConceptVectorSearch

        class MockEmbedder:
            def __init__(self) -> None:
                self.model_name_or_path = "mock-model"

            def generate_embeddings(
                self,
                texts: list[str],
                batch_size: int | None = None,
            ) -> npt.NDArray[np.float32]:
                return np.random.randn(len(texts), 384).astype(np.float32)

        search = ConceptVectorSearch(
            {"embeddings": self.test_embeddings},
            embedder=MockEmbedder(),
        )

        assert search.embedder is not None
        assert search.embedder.model_name_or_path == "mock-model"


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_hf_embedding_generation(self) -> None:
        """Test complete HF embedding pipeline."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()
            mock_model.encode = MagicMock(
                return_value=np.random.randn(2, 384).astype(np.float32),
            )
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            concepts = pd.DataFrame(
                [
                    {"cui": "1001", "preferred_name": "Meningioma"},
                    {"cui": "2002", "preferred_name": "Glioblastoma"},
                ],
            )

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
            )

            texts = embedder.prepare_concept_text(concepts)
            embeddings = embedder.generate_embeddings(texts, batch_size=2)

            assert embeddings.shape[0] == 2
            assert embeddings.shape[1] == 384

    def test_batch_size_default(self) -> None:
        """Test that generate_embeddings uses instance batch_size."""
        with patch(
            "src.snomed_methods.llm_concept_embedder.SentenceTransformer",
        ) as mock_st:
            mock_model = MagicMock()

            def encode_side_effect(
                texts: str | list[str],
                **_kwargs: object,
            ) -> npt.NDArray[np.float32]:
                num_texts = len(texts) if isinstance(texts, list) else 1
                return np.random.randn(num_texts, 384).astype(np.float32)

            mock_model.encode.side_effect = encode_side_effect
            mock_st.return_value = mock_model

            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            _ = pd.DataFrame([{"cui": "1", "preferred_name": "test"}] * 5)
            embedder = ClinicalConceptEmbedder(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                backend="hf",
                batch_size=3,
            )

            texts = ["test text"] * 5
            embeddings = embedder.generate_embeddings(texts)

            assert embeddings.shape[0] == 5


if __name__ == "__main__":
    unittest.main()
