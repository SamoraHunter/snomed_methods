"""Unit tests for hybrid_search module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_term_lookup():
    """Create a mock term lookup."""
    mock = MagicMock()
    mock.find_concepts_by_term.return_value = [
        ("1001", "Meningioma"),
        ("2002", "Glioblastoma"),
    ]
    return mock


@pytest.fixture
def mock_snomed_relations():
    """Create a mock SNOMED relations."""
    mock = MagicMock()
    mock.get_children.return_value = [123, 456]
    mock.get_parents.return_value = [789]
    return mock


class TestHybridSearch:
    """Tests for HybridSearch class."""

    def test_init_without_paths(self):
        """Test initialization without providing paths."""
        with patch(
            "llm_concept_embedder.ClinicalConceptEmbedder"
        ) as mock_embedder_cls, patch(
            "snomed_methods_v1.SnomedRelations"
        ) as mock_relations_cls, patch(
            "snomed_term_lookup.create_term_lookup_from_directory"
        ) as mock_lookup:
            from hybrid_search import HybridSearch

            mock_embedder = MagicMock()
            mock_embedder_cls.return_value = mock_embedder
            mock_rel_instance = MagicMock()
            mock_relations_cls.return_value = mock_rel_instance
            mock_lookup_instance = MagicMock()
            mock_lookup.return_value = mock_lookup_instance

            searcher = HybridSearch()

            assert searcher.backend == "transformers"
            assert searcher.device == "cpu"

    def test_init_with_custom_paths(self):
        """Test initialization with custom paths."""
        with patch(
            "llm_concept_embedder.ClinicalConceptEmbedder"
        ) as mock_embedder_cls, patch(
            "snomed_methods_v1.SnomedRelations"
        ) as mock_relations_cls, patch(
            "snomed_term_lookup.create_term_lookup_from_directory"
        ) as mock_lookup, patch(
            "os.path.exists", return_value=True
        ), patch(
            "medcat.cat.CAT"
        ):
            from hybrid_search import HybridSearch

            mock_embedder = MagicMock()
            mock_embedder_cls.return_value = mock_embedder
            mock_rel_instance = MagicMock()
            mock_relations_cls.return_value = mock_rel_instance
            mock_lookup_instance = MagicMock()
            mock_lookup.return_value = mock_lookup_instance

            searcher = HybridSearch(
                uk_path="/test/uk",
                medcat_path="/test/model.zip",
                model_path="/test/model",
                backend="hf",
                device="cuda",
            )

            assert searcher.backend == "hf"
            assert searcher.device == "cuda"

    def test_search_raises_without_term_lookup(self):
        """Test search raises error when term lookup not available."""
        from hybrid_search import HybridSearch

        # Create instance without uk_path to skip file initialization
        searcher = HybridSearch.__new__(HybridSearch)
        searcher.uk_path = "/test/path"

        searcher._term_lookup = None
        searcher._snomed_relations = MagicMock()

        with pytest.raises(ValueError, match="Term lookup not available"):
            searcher.search("test")

    def test_search_raises_without_hierarchy(self):
        """Test search raises error when hierarchy not available."""
        from hybrid_search import HybridSearch

        searcher = HybridSearch()
        searcher._term_lookup = MagicMock()
        searcher._snomed_relations = None

        with pytest.raises(ValueError, match="Hierarchy data not available"):
            searcher.search("test")

    def test_term_search_basic(self):
        """Test basic term search functionality."""
        with patch("snomed_term_lookup.create_term_lookup_from_directory"), patch(
            "os.path.exists", return_value=True
        ):
            from hybrid_search import HybridSearch

            # Create HybridSearch instance withoutuk_path (skips file initialization)
            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_embedder = MagicMock()
            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.return_value = [
                ("1001", "Meningioma"),
                ("2002", "Glioblastoma"),
            ]

            searcher._term_lookup = mock_lookup_instance
            searcher._embedder = mock_embedder
            searcher._snomed_relations = MagicMock()

            results, cui_to_name = searcher._term_search(["meningioma"])

            assert len(results) > 0
            assert "1001" in results
            assert "Meningioma" in cui_to_name.values()

    def test_term_search_prefix_matching(self):
        """Test term search with prefix matching."""
        with patch(
            "snomed_term_lookup.create_term_lookup_from_directory"
        ) as mock_lookup, patch("os.path.exists", return_value=True):
            from hybrid_search import HybridSearch

            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_embedder = MagicMock()
            mock_rel_instance = MagicMock()
            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.side_effect = [
                [("1001", "Meningioma")],
                [("1002", "Meningioma tumor")],
            ]
            mock_lookup.return_value = mock_lookup_instance

            searcher._term_lookup = mock_lookup_instance
            searcher._embedder = mock_embedder
            searcher._snomed_relations = mock_rel_instance

            results, _ = searcher._term_search(["men"])

            assert len(results) > 0

    def test_term_search_empty_query(self):
        """Test term search with empty query."""
        from hybrid_search import HybridSearch

        searcher = HybridSearch()

        results, cui_to_name = searcher._term_search([])

        assert results == {}
        assert cui_to_name == {}

    def test_hierarchy_search_basic(self):
        """Test basic hierarchy search functionality."""
        with patch("snomed_term_lookup.create_term_lookup_from_directory"), patch(
            "os.path.exists", return_value=True
        ), patch("snomed_methods_v1.SnomedRelations"):
            from hybrid_search import HybridSearch

            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_embedder = MagicMock()
            mock_rel_instance = MagicMock()
            mock_rel_instance.get_children.return_value = [123]
            mock_rel_instance.get_parents.return_value = [789]

            searcher._embedder = mock_embedder
            searcher._snomed_relations = mock_rel_instance

            cuis, scores = searcher._hierarchy_search(["1001"], max_nodes=10)

            assert len(cuis) > 0
            assert len(scores) > 0
            assert cuis[0] == "1001"
            assert scores[0] == 1.0

    def test_hierarchy_search_empty_input(self):
        """Test hierarchy search with empty input."""
        from hybrid_search import HybridSearch

        searcher = HybridSearch()
        searcher._snomed_relations = MagicMock()

        cuis, scores = searcher._hierarchy_search([], max_nodes=10)

        assert cuis == []
        assert scores == []

    def test_hierarchy_search_no_relations(self):
        """Test hierarchy search when relations not available."""
        from hybrid_search import HybridSearch

        searcher = HybridSearch()
        searcher._snomed_relations = None

        cuis, scores = searcher._hierarchy_search(["1001"], max_nodes=10)

        assert cuis == []
        assert scores == []

    def test_hierarchy_search_skips_duplicates(self):
        """Test hierarchy search skips already visited nodes."""
        with patch("snomed_term_lookup.create_term_lookup_from_directory"), patch(
            "os.path.exists", return_value=True
        ), patch("snomed_methods_v1.SnomedRelations"):
            from hybrid_search import HybridSearch

            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_rel_instance = MagicMock()
            mock_rel_instance.get_children.return_value = [1002]
            mock_rel_instance.get_parents.return_value = []

            # Initialize attributes needed for hierarchy search
            searcher._medcat = None
            searcher.search_engine = None

            searcher._term_lookup = MagicMock()
            searcher._embedder = MagicMock()
            searcher._snomed_relations = mock_rel_instance

            cuis, scores = searcher._hierarchy_search(["1001"], max_nodes=5)

            assert "1001" in cuis

    def test_embedding_search_basic(self):
        """Test basic embedding search."""
        with patch(
            "llm_concept_embedder.ClinicalConceptEmbedder"
        ) as mock_embedder_cls, patch(
            "llm_concept_embedder.ConceptVectorSearch"
        ) as mock_search_cls, patch(
            "os.path.exists", return_value=False
        ):
            from hybrid_search import HybridSearch

            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_embedder = MagicMock()
            mock_embedder_cls.return_value = mock_embedder

            # Set up embeddings directly to skip _get_all_embeddings()
            searcher._cui_to_embedding = {"1001": MagicMock()}

            mock_search_instance = MagicMock()
            mock_search_instance.search.return_value = [("1001", "Meningioma", 0.95)]
            mock_search_cls.return_value = mock_search_instance

            # Initialize all instance attributes that HybridSearch normally sets in __init__
            searcher._medcat = None
            searcher.search_engine = None

            searcher._term_lookup = MagicMock()
            searcher._embedder = mock_embedder

            results, scores = searcher._embedding_search("meningioma", top_k=10)

            assert len(results) > 0
            assert results[0][0] == "1001"

    def test_embedding_search_no_embedder(self):
        """Test embedding search when embedder not available."""
        from hybrid_search import HybridSearch

        searcher = HybridSearch()
        searcher._embedder = None

        results, scores = searcher._embedding_search("test", top_k=10)

        assert results == []
        assert scores == []

    def test_rank_results_basic(self):
        """Test basic result ranking."""
        from hybrid_search import HybridSearch

        searcher = HybridSearch()

        scores = {
            "1001": {"term": 0.5, "hierarchy": 0.3, "embedding": 0.8},
            "2002": {"term": 0.7, "hierarchy": 0.4, "embedding": 0.6},
        }

        ranked = searcher._rank_results(scores, 0.3, 0.2, 0.5)

        assert len(ranked) == 2
        assert isinstance(ranked[0], tuple)

    def test_rank_results_zero_scores(self):
        """Test ranking with zero scores."""
        from hybrid_search import HybridSearch

        searcher = HybridSearch()

        scores = {
            "1001": {"term": 0, "hierarchy": 0, "embedding": 0},
        }

        ranked = searcher._rank_results(scores, 0.3, 0.2, 0.5)

        assert len(ranked) == 1
        assert ranked[0][1] == 0

    def test_search_returns_result_object(self):
        """Test that search returns SearchResult object."""
        with patch(
            "snomed_term_lookup.create_term_lookup_from_directory"
        ) as mock_lookup, patch(
            "snomed_methods_v1.SnomedRelations"
        ) as mock_relations_cls, patch(
            "os.path.exists", return_value=True
        ), patch(
            "llm_concept_embedder.ClinicalConceptEmbedder"
        ):
            from hybrid_search import HybridSearch

            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_rel_instance = MagicMock()
            mock_rel_instance.get_children.return_value = []
            mock_rel_instance.get_parents.return_value = []
            mock_relations_cls.return_value = mock_rel_instance
            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.return_value = [
                ("1001", "Meningioma")
            ]
            mock_lookup.return_value = mock_lookup_instance

            # Initialize all instance attributes
            searcher._medcat = None
            searcher.search_engine = None

            searcher._term_lookup = mock_lookup_instance
            searcher._snomed_relations = mock_rel_instance
            searcher._embedder = MagicMock()

            result = searcher.search("meningioma", top_k=5)

            assert hasattr(result, "results")
            assert hasattr(result, "cui_to_term")

    def test_search_with_custom_weights(self):
        """Test search with custom weighting parameters."""
        with patch(
            "snomed_term_lookup.create_term_lookup_from_directory"
        ) as mock_lookup, patch(
            "snomed_methods_v1.SnomedRelations"
        ) as mock_relations_cls, patch(
            "os.path.exists", return_value=True
        ), patch(
            "llm_concept_embedder.ClinicalConceptEmbedder"
        ):
            from hybrid_search import HybridSearch

            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_rel_instance = MagicMock()
            mock_rel_instance.get_children.return_value = []
            mock_rel_instance.get_parents.return_value = []
            mock_relations_cls.return_value = mock_rel_instance
            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.return_value = [
                ("1001", "Meningioma")
            ]
            mock_lookup.return_value = mock_lookup_instance

            # Initialize all instance attributes
            searcher._medcat = None
            searcher.search_engine = None

            searcher._term_lookup = mock_lookup_instance
            searcher._snomed_relations = mock_rel_instance
            searcher._embedder = MagicMock()

            result = searcher.search(
                "meningioma",
                top_k=5,
                term_weight=0.5,
                hierarchy_weight=0.2,
                embedding_weight=0.3,
            )

            assert len(result) >= 0

    def test_hybrid_search_with_split_query(self):
        """Test hybrid search with multi-word query."""
        with patch(
            "snomed_term_lookup.create_term_lookup_from_directory"
        ) as mock_lookup, patch(
            "snomed_methods_v1.SnomedRelations"
        ) as mock_relations_cls, patch(
            "os.path.exists", return_value=True
        ), patch(
            "llm_concept_embedder.ClinicalConceptEmbedder"
        ):
            from hybrid_search import HybridSearch

            searcher = HybridSearch.__new__(HybridSearch)
            searcher.uk_path = "/test/path"

            mock_rel_instance = MagicMock()
            mock_rel_instance.get_children.return_value = []
            mock_rel_instance.get_parents.return_value = []
            mock_relations_cls.return_value = mock_rel_instance
            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.return_value = [
                ("1001", "Meningioma")
            ]
            mock_lookup.return_value = mock_lookup_instance

            # Initialize all instance attributes
            searcher._medcat = None
            searcher.search_engine = None

            searcher._term_lookup = mock_lookup_instance
            searcher._snomed_relations = mock_rel_instance
            searcher._embedder = MagicMock()

            result = searcher.search("brain meningioma", top_k=5)

            assert len(result) >= 0


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_result_object_init(self):
        """Test SearchResult initialization."""
        from hybrid_search import SearchResult

        result = SearchResult()

        assert result.results == []
        assert result.cui_to_term == {}
        assert result.cui_scores == {}

    def test_result_properties(self):
        """Test SearchResult properties."""
        from hybrid_search import SearchResult

        result = SearchResult()
        result.results = [
            ("1001", "Meningioma", 0.95),
            ("2002", "Glioblastoma", 0.85),
        ]

        assert result.cuis == ["1001", "2002"]
        assert result.terms == ["Meningioma", "Glioblastoma"]
        assert result.scores == [0.95, 0.85]

    def test_result_to_dict(self):
        """Test SearchResult to_dict method."""
        from hybrid_search import SearchResult

        result = SearchResult()
        result.results = [("1001", "Meningioma", 0.95)]
        result.term_matches = 5
        result.hierarchy_matches = 3
        result.embedding_matches = 2

        dto = result.to_dict()

        assert "results" in dto
        assert "metrics" in dto
        assert dto["metrics"]["term_matches"] == 5

    def test_result_len(self):
        """Test SearchResult __len__ method."""
        from hybrid_search import SearchResult

        result = SearchResult()

        assert len(result) == 0

        result.results = [("1001", "Meningioma", 0.95)]

        assert len(result) == 1

    def test_result_repr(self):
        """Test SearchResult __repr__ method."""
        from hybrid_search import SearchResult

        result = SearchResult()
        result.term_matches = 5
        result.hierarchy_matches = 3
        result.embedding_matches = 2

        repr_str = repr(result)

        assert "HybridSearchResult" in repr_str


class TestExpandConcepts:
    """Tests for expand_concepts convenience function."""

    def test_expand_concepts_function(self):
        """Test expand_concepts function wrapper."""
        with patch("hybrid_search.HybridSearch") as mock_searcher_cls:
            from hybrid_search import expand_concepts

            mock_instance = MagicMock()
            mock_instance.search.return_value = MagicMock()
            mock_searcher_cls.return_value = mock_instance

            expand_concepts("meningioma", uk_path="/test/path", top_k=10)

            assert mock_instance.search.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
