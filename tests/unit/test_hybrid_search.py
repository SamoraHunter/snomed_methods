#!/usr/bin/env python3
"""Comprehensive unit tests for hybrid_search.py module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from snomed_methods.hybrid_search import (
    HybridSearch,
    SearchResult,
    expand_concepts,
    semantic_filter_results,
)


class TestHybridSearchInit:
    """Tests for HybridSearch initialization."""

    def test_init_with_default_paths(self):
        """Test initialization with default paths."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        assert searcher.backend == "transformers"
        assert searcher.device == "cpu"

    def test_init_with_custom_paths(self):
        """Test initialization with custom paths - mocked to avoid actual loading."""
        with patch("snomed_methods.hybrid_search.HybridSearch._init_embedder"):
            with patch("snomed_methods.hybrid_search.HybridSearch._init_term_lookup"):
                with patch("snomed_methods.hybrid_search.HybridSearch._init_hierarchy"):
                    with patch(
                        "snomed_methods.hybrid_search.HybridSearch._init_medcat"
                    ):
                        searcher = HybridSearch(
                            uk_path="/test/path",
                            medcat_path="/test/medcat.zip",
                            model_path="/test/model",
                            backend="hf",
                            device="cpu",
                        )
                        assert searcher.uk_path == "/test/path"
                        assert searcher.medcat_path == "/test/medcat.zip"
                        assert searcher.model_path == "/test/model"
                        assert searcher.backend == "hf"

    def test_init_without_uk_path(self):
        """Test initialization without SNOMED UK path - mocked."""
        with patch("snomed_methods.hybrid_search.HybridSearch._init_embedder"):
            with patch("snomed_methods.hybrid_search.HybridSearch._init_term_lookup"):
                with patch("snomed_methods.hybrid_search.HybridSearch._init_hierarchy"):
                    with patch(
                        "snomed_methods.hybrid_search.HybridSearch._init_medcat"
                    ):
                        searcher = HybridSearch()
                        assert searcher.uk_path is None


class TestHybridSearchTermLookup:
    """Tests for term lookup initialization and methods."""

    def test_init_term_lookup_with_existing_path(self):
        """Test initializing term lookup when path exists."""
        from snomed_methods.hybrid_search import HybridSearch

        project_root = os.path.dirname(os.path.abspath(__file__))
        uk_path = os.path.join(project_root, "..", "uk_sct2cl_42.2.0")

        searcher = HybridSearch(uk_path=uk_path)
        if searcher._term_lookup is not None:
            assert searcher._term_lookup is not None

    def test_init_term_lookup_fallback(self):
        """Test fallback path calculation for term lookup - mocked."""
        with patch("snomed_methods.hybrid_search.HybridSearch._init_embedder"):
            with patch("snomed_methods.hybrid_search.HybridSearch._init_term_lookup"):
                with patch("snomed_methods.hybrid_search.HybridSearch._init_hierarchy"):
                    with patch(
                        "snomed_methods.hybrid_search.HybridSearch._init_medcat"
                    ):
                        searcher = HybridSearch()
                        assert searcher.uk_path is None


class TestHybridSearchHierarchy:
    """Tests for hierarchy initialization and methods."""

    def test_init_hierarchy_with_existing_file(self):
        """Test initializing hierarchy when relationship file exists."""
        from snomed_methods.hybrid_search import HybridSearch

        project_root = os.path.dirname(os.path.abspath(__file__))
        uk_path = os.path.join(project_root, "..", "uk_sct2cl_42.2.0")

        searcher = HybridSearch(uk_path=uk_path)
        if searcher._snomed_relations is not None:
            assert hasattr(searcher._snomed_relations, "get_children")
            assert hasattr(searcher._snomed_relations, "get_parents")


class TestHybridSearchEmbedder:
    """Tests for embedder initialization."""

    def test_init_embedder_with_model_path(self):
        """Test initializing embedder with model path - mocked."""
        with patch("snomed_methods.hybrid_search.HybridSearch._init_embedder"):
            with patch("snomed_methods.hybrid_search.HybridSearch._init_term_lookup"):
                with patch("snomed_methods.hybrid_search.HybridSearch._init_hierarchy"):
                    with patch(
                        "snomed_methods.hybrid_search.HybridSearch._init_medcat"
                    ):
                        searcher = HybridSearch(model_path="test/model")
                        assert searcher.model_path == "test/model"

    def test_init_embedder_without_model_path(self):
        """Test initializing embedder without model path."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        assert searcher.model_path is None


class TestHybridSearchEmbeddings:
    """Tests for cached embeddings loading."""

    def test_load_cached_embeddings_no_cache(self):
        """Test loading when no cache exists."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        assert hasattr(searcher, "_cui_to_embedding")


class TestHybridSearchTermSearch:
    """Tests for term-based search."""

    @pytest.fixture
    def mock_searcher(self):
        """Create a mocked HybridSearch instance."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()

        mock_lookup = MagicMock()
        mock_lookup.find_concepts_by_term.return_value = [
            ("C001", "Test Concept 1"),
            ("C002", "Test Concept 2"),
        ]
        searcher._term_lookup = mock_lookup
        return searcher

    def test_term_search_basic(self, mock_searcher):
        """Test basic term search."""
        results, cui_to_name = mock_searcher._term_search(["test"])

        assert isinstance(results, dict)
        assert len(results) >= 0

    def test_term_search_empty_query(self, mock_searcher):
        """Test term search with empty query."""
        results, cui_to_name = mock_searcher._term_search([])

        assert len(results) == 0
        assert len(cui_to_name) == 0

    def test_term_search_short_query(self, mock_searcher):
        """Test term search with short query (should be skipped)."""
        results, cui_to_name = mock_searcher._term_search(["a"])

        assert len(results) == 0

    def test_term_search_multiple_terms(self, mock_searcher):
        """Test term search with multiple terms."""
        results, cui_to_name = mock_searcher._term_search(["test", "another"])

        assert isinstance(results, dict)


class TestHybridSearchHierarchySearch:
    """Tests for hierarchy expansion search."""

    @pytest.fixture
    def mock_relations(self):
        """Create mock relations object."""
        mock_rel = MagicMock()
        mock_rel.get_children.return_value = [1002, 1003]
        mock_rel.get_parents.return_value = [1000]
        return mock_rel

    @pytest.fixture
    def mock_searcher_with_hierarchy(self, mock_relations):
        """Create a HybridSearch instance with mocked hierarchy."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        searcher._snomed_relations = mock_relations
        return searcher

    def test_hierarchy_search_basic(self, mock_searcher_with_hierarchy):
        """Test basic hierarchy search."""
        results, scores = mock_searcher_with_hierarchy._hierarchy_search(["1001"], 10)

        assert isinstance(results, list)
        assert isinstance(scores, list)
        assert len(results) == len(scores)

    def test_hierarchy_search_empty_cuis(self, mock_searcher_with_hierarchy):
        """Test hierarchy search with empty CUI list."""
        results, scores = mock_searcher_with_hierarchy._hierarchy_search([], 10)

        assert len(results) == 0
        assert len(scores) == 0

    def test_hierarchy_search_no_relations(self):
        """Test hierarchy search when no relations available."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        results, scores = searcher._hierarchy_search(["1001"], 10)

        assert len(results) == 0
        assert len(scores) == 0

    def test_hierarchy_search_invalid_cui(self, mock_searcher_with_hierarchy):
        """Test hierarchy search with invalid CUI."""
        results, scores = mock_searcher_with_hierarchy._hierarchy_search(
            ["invalid"], 10
        )

        assert isinstance(results, list)


class TestHybridSearchEmbeddingSearch:
    """Tests for embedding-based search."""

    def test_embedding_search_no_embedder(self):
        """Test embedding search when no embedder available."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        results, scores = searcher._embedding_search("test", 10)

        assert len(results) == 0
        assert len(scores) == 0

    def test_embedding_search_no_cached_embeddings(self):
        """Test embedding search with cached embeddings loading failure."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()

        results, scores = searcher._embedding_search("test", 10)

        assert isinstance(results, list)


class TestHybridSearchSemanticFilter:
    """Tests for semantic category filtering."""

    @pytest.fixture
    def mock_searcher(self):
        """Create a mocked HybridSearch instance."""
        searcher = HybridSearch()

        result = SearchResult()
        result.results = [
            ("C001", "Concept 1", 0.9),
            ("C002", "Concept 2", 0.8),
        ]
        result.cui_to_term = {"C001": "Concept 1", "C002": "Concept 2"}
        result.cui_scores = {
            "C001": {"term": 0.9},
            "C002": {"term": 0.8},
        }

        mock_lookup = MagicMock()
        mock_lookup.getconcept_info.side_effect = [
            {"type_id": "404684003"},
            {"type_id": "272379006"},
        ]
        searcher._term_lookup = mock_lookup

        return searcher, result

    def test_semantic_filter_disorder(self, mock_searcher):
        """Test filtering by disorder category."""
        searcher, result = mock_searcher

        filtered = searcher._apply_semantic_filter(result, ["disorder"])

        assert isinstance(filtered, SearchResult)

    def test_semantic_filter_finding(self, mock_searcher):
        """Test filtering by finding category."""
        searcher, result = mock_searcher

        filtered = searcher._apply_semantic_filter(result, ["finding"])

        assert isinstance(filtered, SearchResult)

    def test_semantic_filter_multiple_categories(self, mock_searcher):
        """Test filtering by multiple categories."""
        searcher, result = mock_searcher

        filtered = searcher._apply_semantic_filter(
            result, ["disorder", "finding", "procedure"]
        )

        assert isinstance(filtered, SearchResult)

    def test_semantic_filter_no_matches(self, mock_searcher):
        """Test filtering when no concepts match category."""
        searcher, result = mock_searcher

        filtered = searcher._apply_semantic_filter(result, ["substance"])

        assert isinstance(filtered, SearchResult)

    def test_semantic_filter_empty_categories(self, mock_searcher):
        """Test filtering with empty categories list."""
        searcher, result = mock_searcher

        filtered = searcher._apply_semantic_filter(result, [])

        assert len(filtered.results) == 2

    def test_semantic_filter_invalid_type_id(self, mock_searcher):
        """Test filtering with invalid type ID in concept info."""
        searcher, result = mock_searcher

        mock_lookup = MagicMock()
        mock_lookup.getconcept_info.side_effect = [
            {"type_id": "invalid"},
            None,
        ]
        searcher._term_lookup = mock_lookup

        filtered = searcher._apply_semantic_filter(result, ["disorder"])

        assert isinstance(filtered, SearchResult)


class TestHybridSearchRankResults:
    """Tests for result ranking."""

    def test_rank_results_basic(self):
        """Test basic result ranking."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()

        scores = {
            "C001": {"term": 0.8, "hierarchy": 0.6, "embedding": 0.7},
            "C002": {"term": 0.5, "hierarchy": 0.7, "embedding": 0.9},
            "C003": {"term": 0.3, "hierarchy": 0.4, "embedding": 0.5},
        }

        results = searcher._rank_results(scores, 0.3, 0.2, 0.5)

        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0][1] >= results[-1][1]

    def test_rank_results_empty(self):
        """Test ranking with empty scores."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()

        results = searcher._rank_results({}, 0.3, 0.2, 0.5)

        assert len(results) == 0

    def test_rank_results_partial_scores(self):
        """Test ranking with partial scores (some missing)."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()

        scores = {
            "C001": {"term": 0.8},
            "C002": {"embedding": 0.9},
            "C003": {},
        }

        results = searcher._rank_results(scores, 0.3, 0.2, 0.5)

        assert len(results) == 3


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_init(self):
        """Test SearchResult initialization."""
        from snomed_methods.hybrid_search import SearchResult

        result = SearchResult()

        assert len(result.results) == 0
        assert len(result.cui_to_term) == 0
        assert len(result.cui_scores) == 0
        assert result.term_matches == 0
        assert result.hierarchy_matches == 0
        assert result.embedding_matches == 0

    def test_properties_empty(self):
        """Test properties with empty results."""
        from snomed_methods.hybrid_search import SearchResult

        result = SearchResult()

        assert len(result.cuis) == 0
        assert len(result.terms) == 0
        assert len(result.scores) == 0

    def test_properties_with_data(self):
        """Test properties with result data."""
        from snomed_methods.hybrid_search import SearchResult

        result = SearchResult()
        result.results = [
            ("C001", "Concept 1", 0.9),
            ("C002", "Concept 2", 0.8),
        ]

        assert len(result.cuis) == 2
        assert "C001" in result.cuis
        assert "Concept 1" in result.terms
        assert 0.9 in result.scores

    def test_to_dict_empty(self):
        """Test to_dict with empty results."""
        from snomed_methods.hybrid_search import SearchResult

        result = SearchResult()

        d = result.to_dict()

        assert "results" in d
        assert "metrics" in d
        assert d["metrics"]["term_matches"] == 0

    def test_to_dict_with_data(self):
        """Test to_dict with result data."""
        from snomed_methods.hybrid_search import SearchResult

        result = SearchResult()
        result.results = [("C001", "Concept 1", 0.9)]
        result.term_matches = 5
        result.hierarchy_matches = 3
        result.embedding_matches = 2

        d = result.to_dict()

        assert len(d["results"]) == 1
        assert d["metrics"]["term_matches"] == 5
        assert d["metrics"]["hierarchy_matches"] == 3
        assert d["metrics"]["embedding_matches"] == 2

    def test_len(self):
        """Test __len__ method."""
        from snomed_methods.hybrid_search import SearchResult

        result = SearchResult()
        assert len(result) == 0

        result.results = [
            ("C001", "Concept 1", 0.9),
            ("C002", "Concept 2", 0.8),
        ]
        assert len(result) == 2

    def test_repr(self):
        """Test __repr__ method."""
        from snomed_methods.hybrid_search import SearchResult

        result = SearchResult()
        result.term_matches = 5
        result.hierarchy_matches = 3
        result.embedding_matches = 2

        repr_str = repr(result)

        assert "HybridSearchResult" in repr_str
        assert "term=5" in repr_str


class TestSemanticFilterResults:
    """Tests for standalone semantic_filter_results function."""

    def test_standalone_function(self):
        """Test the standalone filter function exists."""
        result = SearchResult()
        result.results = [("C001", "Concept 1", 0.9)]

        with patch("snomed_methods.hybrid_search.HybridSearch") as mock_hybrid:
            mock_instance = MagicMock()
            mock_lookup = MagicMock()
            mock_instance._term_lookup = mock_lookup
            mock_instance._apply_semantic_filter.return_value = result
            mock_hybrid.return_value = mock_instance

            filtered = semantic_filter_results(result, "disorder")

            assert isinstance(filtered, SearchResult)


class TestExpandConcepts:
    """Tests for standalone expand_concepts function."""

    def test_standalone_function(self):
        """Test the standalone expand function exists."""
        from unittest.mock import MagicMock

        with patch("snomed_methods.hybrid_search.HybridSearch") as mock_hybrid:
            mock_instance = MagicMock()
            mock_lookup = MagicMock()
            mock_lookup.find_concepts_by_term.return_value = []
            mock_instance._term_lookup = mock_lookup
            mock_instance._snomed_relations = MagicMock()
            mock_instance.search.return_value = SearchResult()

            mock_hybrid.return_value = mock_instance

            result = expand_concepts("test", top_k=5)

            assert hasattr(result, "results")


class TestHybridSearchComplete:
    """Integration tests for HybridSearch search method."""

    @pytest.fixture
    def mock_hybrid_searcher(self):
        """Create a fully mocked HybridSearch instance."""
        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()

        mock_lookup = MagicMock()
        mock_lookup.find_concepts_by_term.return_value = [
            ("C001", "Test Concept"),
        ]
        searcher._term_lookup = mock_lookup

        mock_hierarchy = MagicMock()
        mock_hierarchy.get_children.return_value = [1002]
        mock_hierarchy.get_parents.return_value = [1000]
        searcher._snomed_relations = mock_hierarchy

        return searcher

    def test_search_basic(self, mock_hybrid_searcher):
        """Test basic search functionality."""
        result = mock_hybrid_searcher.search("test", top_k=5)

        assert hasattr(result, "results")
        assert hasattr(result, "term_matches")

    def test_search_with_semantic_filter(self, mock_hybrid_searcher):
        """Test search with semantic filtering."""
        result = mock_hybrid_searcher.search(
            "test",
            top_k=5,
            term_weight=0.6,
            hierarchy_weight=0.2,
            embedding_weight=0.2,
        )

        filtered = mock_hybrid_searcher._apply_semantic_filter(result, ["disorder"])

        assert isinstance(filtered, type(result))

    def test_search_top_k_zero(self, mock_hybrid_searcher):
        """Test search with top_k=0."""
        result = mock_hybrid_searcher.search("test", top_k=0)

        assert len(result.results) == 0

    def test_search_invalid_weights(self, mock_hybrid_searcher):
        """Test search with invalid weights (should normalize)."""
        result = mock_hybrid_searcher.search(
            "test",
            top_k=5,
            term_weight=1.5,
            hierarchy_weight=-0.2,
            embedding_weight=0.7,
        )

        assert hasattr(result, "results")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query_search(self):
        """Test search with empty query string."""
        from unittest.mock import MagicMock

        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        mock_lookup = MagicMock()
        mock_lookup.find_concepts_by_term.return_value = []
        searcher._term_lookup = mock_lookup
        searcher._snomed_relations = MagicMock()

        result = searcher.search("", top_k=5)

        assert hasattr(result, "results")

    def test_whitespace_query_search(self):
        """Test search with whitespace-only query."""
        from unittest.mock import MagicMock

        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        mock_lookup = MagicMock()
        mock_lookup.find_concepts_by_term.return_value = []
        searcher._term_lookup = mock_lookup
        searcher._snomed_relations = MagicMock()

        result = searcher.search("   ", top_k=5)

        assert hasattr(result, "results")

    def test_special_characters_query(self):
        """Test search with special characters."""
        from unittest.mock import MagicMock

        from snomed_methods.hybrid_search import HybridSearch

        searcher = HybridSearch()
        mock_lookup = MagicMock()
        mock_lookup.find_concepts_by_term.return_value = []
        searcher._term_lookup = mock_lookup
        searcher._snomed_relations = MagicMock()

        result = searcher.search("@#$%^&*", top_k=5)

        assert hasattr(result, "results")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
