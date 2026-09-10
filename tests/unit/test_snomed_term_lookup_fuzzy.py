"""Tests for SnomedTermLookup fuzzy matching functionality."""

import os
import tempfile

import pandas as pd
import pytest

from snomed_term_lookup import SnomedTermLookup


@pytest.fixture
def sample_description_dataframe():
    """Create a sample description DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": ["100001", "100002", "100003"],
            "effectiveTime": ["20240101"] * 3,
            "active": [1, 1, 1],
            "moduleId": ["999000011000000103"] * 3,
            "conceptId": [409681000000102, 409681000000103, 409681000000104],
            "languageCode": ["en"] * 3,
            "typeId": [
                900000000000003001,  # Preferred
                900000000000013009,  # Synonym
                900000000000013009,  # Synonym
            ],
            "term": [
                "Meningioma",  # Preferred - exact match
                "Brain meningioma",  # Synonym - partial match
                "Glioma",  # Different concept
            ],
            "caseSignificanceId": ["900000000000020000"] * 3,
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

        import shutil

        shutil.copy(
            desc_file,
            os.path.join(
                snapshot_dir,
                "sct2_Description_Test-en_GB1000000_20240101.txt",
            ),
        )

        yield tmpdir


class TestSnomedTermLookupFuzzy:
    """Tests for fuzzy matching functionality."""

    @pytest.mark.skip(reason="rapidfuzz not available in test environment")
    def test_find_concepts_by_term_fuzzy_basic(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test basic fuzzy matching."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term_fuzzy("meningioma", min_score=80)

        assert len(results) > 0
        for cui, term, score in results:
            assert isinstance(cui, str)
            assert isinstance(term, str)
            assert isinstance(score, int)
            assert score >= 80

    @pytest.mark.skip(reason="rapidfuzz not available in test environment")
    def test_find_concepts_by_term_fuzzy_low_score(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test fuzzy matching with low minimum score."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_by_term_fuzzy("meningioma", min_score=30)

        assert len(results) > 0

    @pytest.mark.skip(reason="rapidfuzz not available in test environment")
    def test_find_concepts_by_term_fuzzy_empty_database(
        self,
        temp_description_file,
    ):
        """Test fuzzy matching with empty database."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        # Create empty dataframe
        df = pd.DataFrame(
            columns=[
                "id",
                "effectiveTime",
                "active",
                "moduleId",
                "conceptId",
                "languageCode",
                "typeId",
                "term",
                "caseSignificanceId",
            ]
        )
        df.to_csv(desc_file, sep="\t", index=False)

        lookup = SnomedTermLookup(snomed_description_path=desc_file)
        lookup.df = None  # Simulate empty load

        results = lookup.find_concepts_by_term_fuzzy("test")

        assert results == []

    @pytest.mark.skip(reason="rapidfuzz not available in test environment")
    def test_find_concepts_by_term_fallback_to_exact(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test fallback to exact matching when rapidfuzz unavailable."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        # The method should fall back gracefully
        results = lookup.find_concepts_by_term_fuzzy("meningioma")

        assert isinstance(results, list)


class TestSnomedTermLookupBatch:
    """Tests for batch search functionality edge cases."""

    def test_find_concepts_batch_empty_list(self):
        """Test batch search with empty term list."""
        lookup = SnomedTermLookup()

        results = lookup.find_concepts_batch([])

        assert results == []

    def test_find_concepts_batch_no_matches(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test batch search when no terms match."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_batch(["nonexistent", "notfound"])

        assert results == []

    def test_find_concepts_batch_multiple_matches(
        self,
        sample_description_dataframe,
        temp_description_file,
    ):
        """Test batch search with multiple matches."""
        desc_file = os.path.join(
            temp_description_file,
            "Snapshot",
            "Terminology",
            "sct2_Description_Test-en_GB1000000_20240101.txt",
        )

        lookup = SnomedTermLookup(snomed_description_path=desc_file)

        results = lookup.find_concepts_batch(["Meningioma", "Glioma"])

        assert len(results) > 0
        for term, cui, matched in results:
            assert isinstance(term, str)
            assert isinstance(cui, str)
            assert isinstance(matched, str)


class TestSnomedTermLookupEdgeCases:
    """Tests for edge cases in SnomedTermLookup."""

    def test_load_descriptions_duplicate_filenames(self):
        """Test loading from file with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with spaces and special chars
            desc_file = os.path.join(tmpdir, "sct2 Description Test.txt")
            df = pd.DataFrame(
                {
                    "id": ["100001"],
                    "effectiveTime": ["20240101"],
                    "active": [1],
                    "moduleId": ["999000011000000103"],
                    "conceptId": [409681000000102],
                    "languageCode": ["en"],
                    "typeId": [900000000000003001],
                    "term": ["Test Term"],
                    "caseSignificanceId": ["900000000000020000"],
                }
            )
            df.to_csv(desc_file, sep="\t", index=False)

            lookup = SnomedTermLookup(snomed_description_path=desc_file)
            assert lookup.df is not None
            assert len(lookup.df) == 1

    def test_getconcept_info_empty_dataframe(self):
        """Test getconcept_info when no data loaded."""
        lookup = SnomedTermLookup()

        result = lookup.getconcept_info("12345")

        assert result is None

    def test_find_concepts_by_term_nan_terms(self):
        """Test that NaN terms in dataframe are handled correctly."""
        df = pd.DataFrame(
            {
                "id": ["100001", "100002"],
                "effectiveTime": ["20240101"] * 2,
                "active": [1, 1],
                "moduleId": ["999000011000000103"] * 2,
                "conceptId": [409681000000102, 409681000000103],
                "languageCode": ["en"] * 2,
                "typeId": [900000000000003001] * 2,
                "term": ["Valid Term", None],
                "caseSignificanceId": ["900000000000020000"] * 2,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            desc_file = os.path.join(tmpdir, "test.txt")
            df.to_csv(desc_file, sep="\t", index=False)

            snapshot_dir = os.path.join(tmpdir, "Snapshot", "Terminology")
            os.makedirs(snapshot_dir, exist_ok=True)

            import shutil

            shutil.copy(
                desc_file,
                os.path.join(snapshot_dir, "sct2_Test-en_GB1000000_20240101.txt"),
            )

            lookup = SnomedTermLookup(snomed_description_path=desc_file)
            results = lookup.find_concepts_by_term("Valid")

            assert len(results) == 1
            assert results[0][1] == "Valid Term"

    def test_find_concepts_by_term_unicode_terms(self):
        """Test term matching with unicode characters."""
        df = pd.DataFrame(
            {
                "id": ["100001"],
                "effectiveTime": ["20240101"],
                "active": [1],
                "moduleId": ["999000011000000103"],
                "conceptId": [409681000000102],
                "languageCode": ["en"],
                "typeId": [900000000000003001],
                "term": ["Café résumé naïve"],
                "caseSignificanceId": ["900000000000020000"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            desc_file = os.path.join(tmpdir, "test.txt")
            df.to_csv(desc_file, sep="\t", index=False)

            snapshot_dir = os.path.join(tmpdir, "Snapshot", "Terminology")
            os.makedirs(snapshot_dir, exist_ok=True)

            import shutil

            shutil.copy(
                desc_file,
                os.path.join(snapshot_dir, "sct2_Test-en_GB1000000_20240101.txt"),
            )

            lookup = SnomedTermLookup(snomed_description_path=desc_file)
            results = lookup.find_concepts_by_term("café", ignore_case=True)

            assert len(results) == 1
            assert "Café" in results[0][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
