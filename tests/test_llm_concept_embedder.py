#!/usr/bin/env python3
"""Unit tests for llm_concept_embedder module."""

import os
import pickle
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


class TestClinicalConceptEmbedder(unittest.TestCase):
    """Test ClinicalConceptEmbedder class."""

    def setUp(self):
        """Set up test fixtures."""
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
            ]
        )

    def test_prepare_concept_text_basic(self):
        """Test basic concept text preparation."""
        from llm_concept_embedder import ClinicalConceptEmbedder

        embedder = ClinicalConceptEmbedder(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            backend="hf",
        )

        texts = embedder.prepare_concept_text(self.test_concepts)

        self.assertEqual(len(texts), 2)
        self.assertIn("Meningioma", texts[0])
        self.assertIn("Concept ID: 1001", texts[0])

    def test_prepare_concept_text_with_nan(self):
        """Test handling of NaN values."""
        from llm_concept_embedder import ClinicalConceptEmbedder

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
                }
            ]
        )

        texts = embedder.prepare_concept_text(test_df)

        self.assertEqual(len(texts), 1)
        self.assertIn("Concept ID: 1001", texts[0])

    def test_prepare_concept_text_missing_cui(self):
        """Test handling of missing cui column."""
        from llm_concept_embedder import ClinicalConceptEmbedder

        embedder = ClinicalConceptEmbedder(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            backend="hf",
        )

        test_df = pd.DataFrame(
            [
                {
                    "concept_id": "1001",
                    "preferred_name": "Meningioma",
                }
            ]
        )

        texts = embedder.prepare_concept_text(test_df)

        self.assertEqual(len(texts), 1)
        self.assertIn("Concept ID: 1001", texts[0])

    def test_prepare_concept_text_semantic_tag(self):
        """Test semantic tag inclusion in formatted text."""
        from llm_concept_embedder import ClinicalConceptEmbedder

        embedder = ClinicalConceptEmbedder(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            backend="hf",
        )

        texts = embedder.prepare_concept_text(self.test_concepts)

        self.assertIn("Semantic Tag: T-12", texts[1])

    def test_export_embeddings_pickle(self):
        """Test exporting embeddings to pickle."""
        from llm_concept_embedder import ClinicalConceptEmbedder

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

            self.assertTrue(os.path.exists(output_path))

    def test_export_embeddings_simple_filename(self):
        """Test export with simple filename (no directory path)."""
        from llm_concept_embedder import ClinicalConceptEmbedder

        embedder = ClinicalConceptEmbedder(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            backend="hf",
        )

        with TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                output_path = "simple_test.pkl"
                cui_to_embedding = {"1001": np.random.randn(384).astype(np.float32)}

                embedder.export_embeddings(cui_to_embedding, output_path)

                self.assertTrue(os.path.exists(output_path))
            finally:
                os.chdir(original_cwd)

    def test_ollama_init_with_custom_url(self):
        """Test Ollama backend initialization with custom URL."""
        from unittest.mock import MagicMock, patch

        mock_ollama = MagicMock()
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from llm_concept_embedder import ClinicalConceptEmbedder

            embedder = ClinicalConceptEmbedder(
                model_name_or_path="test-model",
                backend="ollama",
                ollama_base_url="http://custom-host:12345",
            )

            self.assertEqual(embedder.ollama_base_url, "http://custom-host:12345")
            self.assertEqual(embedder.backend, "ollama")


class TestConceptVectorSearch(unittest.TestCase):
    """Test ConceptVectorSearch class."""

    def setUp(self):
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

    def test_build_index(self):
        """Test building FAISS index."""
        from unittest.mock import MagicMock, patch

        from llm_concept_embedder import ConceptVectorSearch

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            search = ConceptVectorSearch(self.test_embeddings)
            search.build_index(index_type="FlatIP")

            self.assertIsNotNone(search.index)
            self.assertEqual(len(search.cui_list), 3)

    def test_build_index_with_names(self):
        """Test building index with cui_to_name mapping."""
        from unittest.mock import MagicMock, patch

        from llm_concept_embedder import ConceptVectorSearch

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            search = ConceptVectorSearch(
                {"embeddings": self.test_embeddings, "names": self.test_names}
            )
            search.build_index()

            self.assertEqual(len(search.cui_list), 3)
            self.assertEqual(search.cui_to_name["C001"], "Meningioma")

    def test_search_returns_results(self):
        """Test search returns valid results."""
        import os
        from unittest.mock import MagicMock, patch

        from llm_concept_embedder import ConceptVectorSearch

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
                    query_embedding=np.random.randn(3).astype(np.float32), top_k=3
                )

                self.assertGreater(len(results), 0)
                for cui, name, score in results:
                    self.assertIsNotNone(cui)
                    self.assertIsInstance(name, str)
                    self.assertIsInstance(score, float)

    def test_search_with_prebuilt_index(self):
        """Test query with pre-built index."""
        from unittest.mock import MagicMock, patch

        from llm_concept_embedder import ConceptVectorSearch

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
                {"embeddings": self.test_embeddings, "names": self.test_names}
            )
            search.build_index()

            query_vec = np.array([0.12, 0.22, 0.32], dtype=np.float32)

            results = search.search(query_embedding=query_vec, top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], "C001")

    def test_load_embeddings_from_pkl_file(self):
        """Test loading embeddings from pickle file."""
        import os
        from unittest.mock import MagicMock, patch

        from llm_concept_embedder import ConceptVectorSearch

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

                self.assertEqual(len(search.cui_list), 3)
                self.assertEqual(search.cui_to_name["C001"], "Meningioma")

    def test_embedder_reuse_for_query(self):
        """Test that embedder is reused for query embedding."""
        from llm_concept_embedder import ConceptVectorSearch

        class MockEmbedder:
            def __init__(self):
                self.model_name_or_path = "mock-model"

            def generate_embeddings(self, texts, batch_size=None):
                return np.random.randn(len(texts), 384).astype(np.float32)

        search = ConceptVectorSearch(
            {"embeddings": self.test_embeddings},
            embedder=MockEmbedder(),
        )

        self.assertIsNotNone(search.embedder)
        self.assertEqual(search.embedder.model_name_or_path, "mock-model")


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_hf_embedding_generation(self):
        """Test complete HF embedding pipeline."""
        from llm_concept_embedder import ClinicalConceptEmbedder

        concepts = pd.DataFrame(
            [
                {"cui": "1001", "preferred_name": "Meningioma"},
                {"cui": "2002", "preferred_name": "Glioblastoma"},
            ]
        )

        embedder = ClinicalConceptEmbedder(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            backend="hf",
        )

        texts = embedder.prepare_concept_text(concepts)
        embeddings = embedder.generate_embeddings(texts, batch_size=2)

        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 384)

    def test_batch_size_default(self):
        """Test that generate_embeddings uses instance batch_size."""
        from llm_concept_embedder import ClinicalConceptEmbedder

        _ = pd.DataFrame([{"cui": "1", "preferred_name": "test"}] * 5)
        embedder = ClinicalConceptEmbedder(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            backend="hf",
            batch_size=3,
        )

        texts = ["test text"] * 5
        embeddings = embedder.generate_embeddings(texts)

        self.assertEqual(embeddings.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
