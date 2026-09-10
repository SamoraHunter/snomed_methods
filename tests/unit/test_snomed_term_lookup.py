"""
Unit tests for the SnomedTermLookup class.

This module provides comprehensive test coverage for SNOMED concept term lookup,
including:
- Exact and fuzzy matching strategies
- Case-insensitive and prefix matching
- Active/inactive concept filtering
- Special character handling in terms
- Batch search for multiple concepts at once

Tests verify correct behavior for SNOMED CT terminology operations including
text searching, concept information retrieval, and batch term lookup.
"""

import os
import tempfile

import pandas as pd
import pytest

from snomed_term_lookup import SnomedTermLookup, create_term_lookup_from_directory


@pytest.fixture
def sample_description_dataframe():
    """Create a sample description DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": ["100001", "100002", "100003", "100004", "100005"],
            "effectiveTime": ["20240101"] * 5,
            "active": [1, 1, 1, 0, 1],  # Last one is inactive
            "moduleId": ["999000011000000103"] * 5,
            "conceptId": [
                409681000000102,
                409681000000103,
                409681000000104,
                409681000000105,
                409681000000106,
            ],
            "languageCode": ["en"] * 5,
            "typeId": [
                900000000000003001,  # Preferred
                900000000000013009,  # Synonym
                900000000000013009,  # Synonym
                900000000000013009,  # Synonym
                900000000000013009,  # Synonym
            ],
            "term": [
                "[M]Meningioma",  # Preferred - active
                "Meningioma tumor",  # Synonym - active
                "Brain meningioma",  # Synonym - active
                "Old meningioma term",  # Synonym - inactive
                "Glioma brain",  # Different concept
            ],
            "caseSignificanceId": ["900000000000020000"] * 5,
        },
    )


@pytest.fixture
def temp_description_file(sample_description_dataframe):
    """Create a temporary description file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        desc_file = os.path.join(tmpdir, "sct2_Description_Test.txt")
        sample_description_dataframe.to_csv(desc_file, sep="\t", index=False)

        # Create minimal directory structure
        snapshot_dir = os.path.join(tmpdir, "Snapshot", "Terminology")
        os.makedirs(snapshot_dir, exist_ok=True)

        shutil = __import__("shutil")
        shutil.copy(
            desc_file,
            os.path.join(
                snapshot_dir,
                "sct2_Description_Test-en_GB1000000_20240101.txt",
            ),
        )

        yield tmpdir


class TestSnomedTermLookup:
    """
     Tests for the SnomedTermLookup class.

     The SnomedTermLookup class provides text search capabilities over SNOMED
    CT description tables.

     Behavior:
     - Loads SNOMED descriptions from tab-separated files
     - Supports case-insensitive and exact matching
     - Can filter by active status
     - Provides prefix matching for terms

     Parameters:
     - snomed_description_path: Path to SNOMED description file (optional)
     - active_only: Whether to include inactive concepts (default: True)

     This is the primary interface for searching SNOMED concepts by their
     descriptive terms, enabling text-based concept discovery.
    """

    def test_initialization_without_file(self):
        """
        Test that lookup can be initialized without a file path.

        Expected behavior: Instance created with df=None initially.

        SnomedTermLookup supports deferred loading - you can create an instance
        first and load data later, or instantiate directly with a file path.

        Input: SnomedTermLookup()
        State after:
        - self.df = None (no data loaded yet)
        - self.active_only = True (default setting)
        """
        lookup = SnomedTermLookup()
        assert lookup.df is None
        assert lookup.active_only is True

    def test_load_descriptions_invalid_file(self):
        """
        Test handling of invalid/non-existent file path.

        Expected behavior: Instance created with df=None (graceful failure).

        When the provided file path doesn't exist or can't be read,
        SnomedTermLookup should handle this gracefully without crashing,
        allowing callers to check if loading succeeded before proceeding.

        Input: snomed_description_path="/nonexistent/file.txt"
        State after:
        - self.df = None
        - No exception raised
        """

    def test_find_concepts_by_term_exact_match(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """
        Test exact term matching in SNOMED descriptions.

        Expected behavior: Returns concepts with terms matching exactly.

        Exact matching finds all concepts where the search term matches
        a term in the description table. The method should handle case-
        insensitivity via the ignore_case parameter.

        Input: term="Meningioma", ignore_case=True
        Return: List of (cui, term) tuples for matching concepts

        This is the most common search pattern where users know the
        preferred or accepted term name and want to find its concept ID.
        """
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term("Meningioma", ignore_case=True)

        assert len(results) > 0
        # Should find concepts with 'Meningioma' in their terms
        cui_list = [r[0] for r in results]
        assert any(
            str(cui) in ["409681000000102", "409681000000103"] for cui in cui_list
        )

    def test_find_concepts_by_term_case_sensitive(
        self, sample_description_dataframe, temp_description_file
    ):
        """
        Test case-sensitive and case-insensitive term matching.

        Expected behavior: Respects ignore_case parameter.

        When ignore_case=False (or default), search should be sensitive to
        uppercase/lowercase. When ignore_case=True, search should match
        regardless of case distinctions.

        Input:
        - Uppercase search term with ignore_case=False -> match expected
        - Lowercase search term with ignore_case=True -> match expected

        This versatility is important for SNOMED which has specific case
        conventions but users may search with varying capitalization.
        """
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        # Uppercase search should match
        results_upper = lookup.find_concepts_by_term("[M]Meningioma", ignore_case=False)
        assert len(results_upper) > 0

        # Lowercase search shouldn't match case-sensitive default
        results_lower = lookup.find_concepts_by_term("[m]meningioma", ignore_case=True)
        assert len(results_lower) > 0

    def test_find_concepts_by_term_match_prefix(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test prefix matching."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term("Meningioma", match_prefix=True)

        # Should find terms starting with 'Meningioma'
        for _cui, term in results:
            assert term.lower().startswith("meningioma")

    def test_find_concepts_by_term_top_n(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test limiting number of results."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term("Meningioma", top_n=2)

        assert len(results) <= 2

    def test_find_concepts_by_term_empty_dataframe(self):
        """Test finding concepts when DataFrame is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            desc_file = os.path.join(tmpdir, "empty.txt")
            pd.DataFrame().to_csv(desc_file, sep="\t", index=False)

            lookup = SnomedTermLookup(snomed_description_path=desc_file)
            results = lookup.find_concepts_by_term("test")

            assert results == []

    def test_find_concepts_by_term_none_df(self):
        """Test finding concepts when DataFrame is None."""
        lookup = SnomedTermLookup()
        results = lookup.find_concepts_by_term("test")

        assert results == []
        assert lookup.df is None

    def test_find_concepts_by_term_special_characters(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test matching terms with special characters."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term("[M]Meningioma", ignore_case=True)

        assert len(results) > 0

    def test_find_concepts_by_term_nan_terms(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test handling of NaN values in term column."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term("meningioma")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_find_concepts_by_term_empty_string_term(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test matching with empty string term."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term("", ignore_case=True)

        assert isinstance(results, list)

    def test_find_concepts_by_term_regex_special_chars(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test that special regex characters don't break matching."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term("[M]Men*", ignore_case=True)

        assert isinstance(results, list)

    def test_getconcept_info(self, sample_description_dataframe, temp_description_file):
        """Test getting concept information."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        # Test with valid CUI
        info = lookup.getconcept_info("409681000000102")

        assert info is not None
        assert info["concept_id"] == "409681000000102"
        assert info["preferred_name"] is not None

    def test_getconcept_info_nonexistent(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test getting information for non-existent CUI."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        info = lookup.getconcept_info("999999")

        assert info is None

    def test_getconcept_info_invalid_cui(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test handling of invalid CUI format."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        info = lookup.getconcept_info("invalid")

        assert info is None

    def test_getconcept_info_none_df(self):
        """Test getting concept info when DataFrame is None."""
        lookup = SnomedTermLookup()
        info = lookup.getconcept_info("123")

        assert info is None

    def test_getconcept_info_float_cui(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test handling of float CUI values."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        info = lookup.getconcept_info(409681000000102.0)

        assert info is not None
        assert "409681000000102" in info["concept_id"]

    def test_getconcept_info_string_cui(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test handling of string CUI values."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        info = lookup.getconcept_info("409681000000102")

        assert info is not None


class TestCreateTermLookupFromDirectory:
    """Tests for create_term_lookup_from_directory function."""

    def test_creates_lookup_from_directory(self, temp_description_file):
        """Test that lookup can be created from directory path."""
        lookup = create_term_lookup_from_directory(temp_description_file)

        assert lookup.df is not None
        assert len(lookup.df) > 0

    def test_raises_file_not_found_error_no_files(self):
        """Test that FileNotFoundError is raised when no description file found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                create_term_lookup_from_directory(tmpdir)

    def test_creates_lookup_with_active_false(
        self,
        temp_description_file,
    ):
        """Test creating lookup with active_only=False."""
        lookup = create_term_lookup_from_directory(
            temp_description_file, active_only=False
        )

        assert lookup.df is not None
        assert lookup.active_only is False

    def test_creates_lookup_with_custom_pattern(self, mocker):
        """Test that custom patterns can be used to find description files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            desc_dir = os.path.join(tmpdir, "Terminology")
            os.makedirs(desc_dir)

            desc_file = os.path.join(desc_dir, "sct2_Description_Custom.txt")
            pd.DataFrame({"id": ["1"], "conceptId": [123], "term": ["test"]}).to_csv(
                desc_file, sep="\t", index=False
            )

            mock_glob = mocker.patch("glob.glob")
            mock_glob.return_value = [desc_file]

            result = create_term_lookup_from_directory(tmpdir)

            assert isinstance(result, SnomedTermLookup)


class TestBatchSearch:
    """Tests for batch search functionality."""

    def test_find_concepts_batch(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test searching for multiple terms at once."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        terms = ["Meningioma", "Glioma"]
        results = lookup.find_concepts_batch(terms, ignore_case=True)

        assert len(results) > 0
        # Each result should be (term, cui, matched_term)
        for term, cui, matched in results:
            assert isinstance(term, str)
            assert isinstance(cui, str)
            assert isinstance(matched, str)

    def test_find_concepts_batch_empty_list(self):
        """Test batch search with empty list."""
        lookup = SnomedTermLookup()

        results = lookup.find_concepts_batch([])

        assert results == []

    def test_find_concepts_batch_with_prefix_match(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test batch search with prefix matching."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        terms = ["Meningioma"]
        results = lookup.find_concepts_batch(terms, match_prefix=True)

        assert len(results) >= 1

    @pytest.mark.skip(reason="requires rapidfuzz")
    def test_find_concepts_by_term_fuzzy_import_error(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test fuzzy matching when rapidfuzz is not available."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term_fuzzy("test")

        assert isinstance(results, list)

    @pytest.mark.skip(reason="requires rapidfuzz")
    def test_find_concepts_by_term_fuzzy_no_results(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test fuzzy matching with no matching terms."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term_fuzzy("nonexistentterm", min_score=80)

        assert isinstance(results, list)

    def test_find_concepts_by_term_fuzzy_empty_df(self):
        """Test fuzzy matching with empty DataFrame."""
        lookup = SnomedTermLookup()

        results = lookup.find_concepts_by_term_fuzzy("test")

        assert results == []

    @pytest.mark.skip(reason="requires rapidfuzz")
    def test_find_concepts_by_term_fuzzy_high_score(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test fuzzy matching with high minimum score."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term_fuzzy("Meningioma", min_score=95)

        assert isinstance(results, list)

    def test_find_concepts_by_term_case_insensitive(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test case-insensitive matching with different case queries."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results_upper = lookup.find_concepts_by_term("MENINGIOMA", ignore_case=True)
        results_lower = lookup.find_concepts_by_term("meningioma", ignore_case=True)

        assert len(results_upper) > 0
        assert len(results_lower) > 0

    def test_find_concepts_by_term_active_filtering(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test that active_only filtering works correctly."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup_active = SnomedTermLookup(
            snomed_description_path=desc_file, active_only=True
        )
        lookup_inactive = SnomedTermLookup(
            snomed_description_path=desc_file, active_only=False
        )

        assert len(lookup_active.df) < len(lookup_inactive.df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
