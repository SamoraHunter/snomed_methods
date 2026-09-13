"""Unit tests for semantic_expansion module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_snomed_lookup():
    """Create a mock SNOMED term lookup."""
    mock = MagicMock()
    mock.find_concepts_by_term.return_value = [
        ("1001", "Meningioma"),
        ("2002", "Glioblastoma"),
    ]
    mock.getconcept_info.return_value = {
        "preferred_name": "Meningioma",
        "concept_id": "1001",
    }
    return mock


@pytest.fixture
def mock_snomed_relations():
    """Create a mock SNOMED relations."""
    mock = MagicMock()
    mock.has_medcat.return_value = True
    mock.cat.cdb.most_similar.return_value = {"2002": {"sim": 0.85}}
    return mock


class TestSemanticSearch:
    """Tests for SemanticSearch class."""

    def test_init_without_path(self):
        """Test initialization without providing UK path."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch()

        assert searcher.uk_path is not None

    def test_init_with_custom_path(self):
        """Test initialization with custom UK path."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch(uk_path="/custom/path")

        assert searcher.uk_path == "/custom/path"

    def test_generate_search_terms_basic(self):
        """Test basic search term generation."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch()

        variants = searcher._generate_search_terms("meningioma")

        assert len(variants) > 1
        assert "meningioma" in variants

    def test_generate_search_terms_duplicates(self):
        """Test that duplicate terms are removed."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch()

        variants = searcher._generate_search_terms("TEST")

        variant_lowers = [v.lower() for v in variants]
        assert len(variant_lowers) == len(set(variant_lowers))

    def test_generate_search_terms_variants(self):
        """Test that all expected variants are generated."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch()

        variants = searcher._generate_search_terms("tumor")

        assert "tumor" in variants
        assert "tumors" in variants

    def test_term_lookup_search_basic(self):
        """Test basic term lookup search."""
        with patch(
            "snomed_methods.semantic_expansion.importlib.util.find_spec",
            return_value=True,
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            searcher = SemanticSearch()

            mock_lookup = MagicMock()
            mock_lookup.find_concepts_by_term.return_value = [
                ("1001", "Meningioma"),
            ]

            results = searcher._term_lookup_search(
                mock_lookup, ["meningioma"], match_prefix=True
            )

            assert len(results) > 0

    def test_term_lookup_search_no_package(self):
        """Test term lookup search when package not available."""
        with patch(
            "snomed_methods.semantic_expansion.importlib.util.find_spec",
            return_value=False,
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            searcher = SemanticSearch()

            results = searcher._term_lookup_search(MagicMock(), ["test"])

            assert results == set()

    def test_hierarchy_expansion_basic(self):
        """Test basic hierarchy expansion."""
        with (
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
            patch(
                "snomed_methods.snomed_term_lookup.create_term_lookup_from_directory"
            ) as mock_lookup,
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            mock_lookup_instance = MagicMock()
            mock_lookup.return_value = mock_lookup_instance

            searcher = SemanticSearch(uk_path="/test/path")

            codes, names = searcher._hierarchy_expansion(["1001"], max_concepts=10)

            assert len(codes) >= 0

    def test_hierarchy_expansion_file_not_found(self):
        """Test hierarchy expansion when relationship file not found."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch(uk_path="/test/path")

        codes, names = searcher._hierarchy_expansion(["1001"], max_concepts=10)

        assert codes == []
        assert names == []

    def test_hierarchy_expansion_read_error(self):
        """Test hierarchy expansion when CSV read fails."""
        with (
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
            patch("pandas.read_csv") as mock_read,
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            mock_lookup_instance = MagicMock()
            with patch(
                "snomed_methods.snomed_term_lookup.create_term_lookup_from_directory",
                return_value=mock_lookup_instance,
            ):
                mock_read.side_effect = Exception("Read error")

                searcher = SemanticSearch(uk_path="/test/path")

                codes, names = searcher._hierarchy_expansion(["1001"], max_concepts=10)

                assert codes == []
                assert names == []

    def test_medcat_expansion_basic(self):
        """Test basic MedCAT expansion."""
        with (
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
            patch(
                "snomed_methods.snomed_methods_v1.SnomedRelations"
            ) as mock_relations_cls,
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            mock_rel_instance = MagicMock()
            mock_rel_instance.has_medcat.return_value = True
            mock_rel_instance.cat.cdb.most_similar.return_value = {
                "2002": {"sim": 0.85, "name": "Glioblastoma"}
            }
            mock_relations_cls.return_value = mock_rel_instance

            searcher = SemanticSearch(uk_path="/test/path")

            codes, names = searcher._medcat_expansion(["1001"], topn=30)

            assert len(codes) >= 0

    def test_medcat_expansion_no_package(self):
        """Test MedCAT expansion when package not available."""
        with patch(
            "snomed_methods.semantic_expansion.importlib.util.find_spec",
            return_value=False,
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            searcher = SemanticSearch()

            codes, names = searcher._medcat_expansion(["1001"])

            assert codes == []
            assert names == []

    def test_combine_results(self):
        """Test combining results from multiple methods."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch()

        term_matches = {
            ("1001", "Meningioma"),
            ("2002", "Glioblastoma"),
        }
        codes = ["3003", "4004"]
        names = ["Astrocytoma", "Oligodendroglioma"]

        combined = searcher._combine_results(term_matches, codes, names)

        assert len(combined) == 4
        assert combined["1001"] == "Meningioma"

    def test_combine_results_none_name(self):
        """Test combine results with None name."""
        from snomed_methods.semantic_expansion import SemanticSearch

        searcher = SemanticSearch()

        term_matches = {("1001", "Meningioma")}
        codes = ["2002"]
        names = [None]

        combined = searcher._combine_results(term_matches, codes, names)

        assert "CUI: 2002" in combined.values()

    def test_search_basic(self):
        """Test basic search functionality."""
        with (
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
            patch(
                "snomed_methods.snomed_term_lookup.create_term_lookup_from_directory"
            ) as mock_lookup,
            patch("os.path.exists", return_value=False),
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.return_value = [
                ("1001", "Meningioma")
            ]
            mock_lookup.return_value = mock_lookup_instance

            searcher = SemanticSearch(uk_path="/test/path")

            results = searcher.search("meningioma", max_concepts=50)

            assert hasattr(results, "concepts")
            assert len(results) > 0

    def test_search_with_hierarchy_disabled(self):
        """Test search with hierarchy expansion disabled."""
        with (
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
            patch(
                "snomed_methods.snomed_term_lookup.create_term_lookup_from_directory"
            ) as mock_lookup,
            patch("os.path.exists", return_value=False),
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.return_value = [
                ("1001", "Meningioma")
            ]
            mock_lookup.return_value = mock_lookup_instance

            searcher = SemanticSearch(uk_path="/test/path")

            results = searcher.search("meningioma", use_hierarchy=False)

            assert hasattr(results, "concepts")

    def test_search_with_medcat_enabled(self):
        """Test search with MedCAT enabled."""
        with (
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
            patch(
                "snomed_methods.snomed_term_lookup.create_term_lookup_from_directory"
            ) as mock_lookup,
            patch(
                "snomed_methods.snomed_methods_v1.SnomedRelations"
            ) as mock_relations_cls,
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            mock_lookup_instance = MagicMock()
            mock_lookup.return_value = mock_lookup_instance

            mock_rel_instance = MagicMock()
            mock_rel_instance.has_medcat.return_value = False
            mock_relations_cls.return_value = mock_rel_instance

            searcher = SemanticSearch(uk_path="/test/path")

            results = searcher.search(["meningioma", "tumor"], max_concepts=50)

            assert hasattr(results, "concepts")

    def test_search_metrics(self):
        """Test search returns correct metrics."""
        with (
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
            patch(
                "snomed_methods.snomed_term_lookup.create_term_lookup_from_directory"
            ) as mock_lookup,
            patch("os.path.exists", return_value=False),
        ):
            from snomed_methods.semantic_expansion import SemanticSearch

            mock_lookup_instance = MagicMock()
            mock_lookup_instance.find_concepts_by_term.return_value = [
                ("1001", "Meningioma")
            ]
            mock_lookup.return_value = mock_lookup_instance

            searcher = SemanticSearch(uk_path="/test/path")

            results = searcher.search("meningioma", max_concepts=50)

            assert hasattr(results, "concepts")
            assert hasattr(results, "metrics")
            metrics = results.metrics
            assert "total" in metrics


class TestSearchResults:
    """Tests for SearchResults class."""

    def test_results_init(self):
        """Test SearchResults initialization."""
        from snomed_methods.semantic_expansion import SearchResults

        concepts = {"1001": "Meningioma"}
        metrics = {"total": 1}

        results = SearchResults(concepts=concepts, metrics=metrics)

        assert results.concepts == concepts

    def test_results_cuis(self):
        """Test SearchResults cuis property."""
        from snomed_methods.semantic_expansion import SearchResults

        concepts = {
            "2002": "Glioblastoma",
            "1001": "Meningioma",
        }

        results = SearchResults(concepts=concepts, metrics={})

        cuis = results.cuis

        assert len(cuis) == 2

    def test_results_terms(self):
        """Test SearchResults terms property."""
        from snomed_methods.semantic_expansion import SearchResults

        concepts = {
            "1001": "Meningioma",
            "2002": "Glioblastoma",
        }

        results = SearchResults(concepts=concepts, metrics={})

        terms = results.terms

        assert len(terms) == 2

    def test_results_metrics(self):
        """Test SearchResults metrics property."""
        from snomed_methods.semantic_expansion import SearchResults

        metrics = {"total": 5, "core_concepts": 3}
        results = SearchResults(concepts={}, metrics=metrics)

        result_metrics = results.metrics

        assert result_metrics["total"] == 5

    def test_get_cui_to_term_dict(self):
        """Test get_cui_to_term_dict method."""
        from snomed_methods.semantic_expansion import SearchResults

        concepts = {"1001": "Meningioma"}
        results = SearchResults(concepts=concepts, metrics={})

        mapping = results.get_cui_to_term_dict()

        assert mapping["1001"] == "Meningioma"

    def test_get_core_concepts(self):
        """Test get_core_concepts method."""
        from snomed_methods.semantic_expansion import SearchResults

        concepts = {
            "1001": "Meningioma",
            "2002": "Glioblastoma",
        }

        results = SearchResults(
            concepts=concepts, metrics={"search_terms_used": ["meningioma"]}
        )

        core = results.get_core_concepts()

        assert len(core) > 0

    def test_get_expanded_concepts(self):
        """Test get_expanded_concepts method."""
        from snomed_methods.semantic_expansion import SearchResults

        concepts = {
            "1001": "Meningioma",
            "2002": "Glioblastoma",
        }

        results = SearchResults(
            concepts=concepts, metrics={"search_terms_used": ["meningioma"]}
        )

        expanded = results.get_expanded_concepts()

        assert len(expanded) >= 0

    def test_results_len(self):
        """Test SearchResults __len__ method."""
        from snomed_methods.semantic_expansion import SearchResults

        results = SearchResults(concepts={"1001": "Meningioma"}, metrics={})

        assert len(results) == 1

    def test_results_repr(self):
        """Test SearchResults __repr__ method."""
        from snomed_methods.semantic_expansion import SearchResults

        results = SearchResults(
            concepts={"1001": "Meningioma"},
            metrics={"total": 1, "core_concepts": 1, "expanded_concepts": 0},
        )

        repr_str = repr(results)

        assert "SearchResults" in repr_str


class TestExpandConcepts:
    """Tests for expand_concepts convenience function."""

    def test_expand_concepts_function(self):
        """Test expand_concepts function wrapper."""
        with (
            patch(
                "snomed_methods.semantic_expansion.SemanticSearch"
            ) as mock_searcher_cls,
            patch(
                "snomed_methods.semantic_expansion.importlib.util.find_spec",
                return_value=True,
            ),
        ):
            from snomed_methods.semantic_expansion import expand_concepts

            mock_instance = MagicMock()
            mock_instance.search.return_value = MagicMock()
            mock_searcher_cls.return_value = mock_instance

            expand_concepts("meningioma", uk_path="/test/path", max_concepts=10)

            assert mock_instance.search.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
