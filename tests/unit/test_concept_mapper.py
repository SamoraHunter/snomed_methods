"""Unit tests for ConceptMapper module.

Tests cover:
- ICD-10 mapping functionality
- LOINC mapping functionality
- RxNorm mapping functionality
- Bidirectional CUI mapping
- Multi-hop mappings
- Batch mapping operations
"""

import os
import tempfile

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def concept_mapper():
    """Create ConceptMapper instance once per test module and reuse it."""
    from src.snomed_methods.concept_mapper import ConceptMapper

    return ConceptMapper()


class TestConceptMapper:
    """Tests for the ConceptMapper class."""

    def test_initialization_without_paths(self):
        """Test that mapper can be initialized without file paths."""
        # Import directly to avoid heavy package-level imports
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        assert mapper.uk_path is None
        assert mapper.icd_map_file is None
        assert mapper.loinc_map_file is None

    def test_initialization_with_paths(self):
        """Test initialization with explicit file paths."""
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper(
            uk_path="/test/path",
            icd_map_file="/test/icd.map",
            loinc_map_file="/test/loinc.map",
        )
        assert mapper.uk_path == "/test/path"
        assert mapper.icd_map_file == "/test/icd.map"
        assert mapper.loinc_map_file == "/test/loinc.map"

    def test_load_icd_mapping_no_file(self, concept_mapper):
        """Test loading ICD mapping when no file is available."""
        result = concept_mapper.load_icd_mapping()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_load_loinc_mapping_no_file(self, concept_mapper):
        """Test loading LOINC mapping when no file is available."""
        result = concept_mapper.load_loinc_mapping()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_map_to_icd_no_mappings(self, concept_mapper):
        """Test mapping for a CUI with no ICD mappings."""
        result = concept_mapper.map_to_icd("999999")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_map_to_loinc_no_mappings(self, concept_mapper):
        """Test mapping for a CUI with no LOINC mappings."""
        result = concept_mapper.map_to_loinc("999999")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_map_to_rxnorm_no_medcat(self, concept_mapper):
        """Test RxNorm mapping without MedCAT model pack."""
        result = concept_mapper.map_to_rxnorm("409681000000102")
        assert isinstance(result, list)

    def test_map_bidirectional_snomed(self, concept_mapper):
        """Test bidirectional mapping from SNOMED."""
        result = concept_mapper.map_bidirectional("409681000000102", "SNOMED")
        assert isinstance(result, list)

    def test_map_bidirectional_icd10(self, concept_mapper):
        """Test bidirectional mapping from ICD-10."""
        result = concept_mapper.map_bidirectional("J44.1", "ICD10")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["target_vocab"] == "SNOMED (estimated)"

    def test_map_bidirectional_loinc(self, concept_mapper):
        """Test bidirectional mapping from LOINC."""
        result = concept_mapper.map_bidirectional("718-7", "LOINC")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["target_vocab"] == "SNOMED (estimated)"

    def test_multi_hop_mapping(self):
        """Test multi-hop mapping to multiple target vocabularies."""
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        result = mapper.multi_hop_mapping(
            "409681000000102", ["ICD-10", "LOINC"], max_hops=2
        )
        assert isinstance(result, dict)
        assert result["start_cui"] == "409681000000102"
        assert len(result["target_vocabs"]) == 2

    def test_get_all_mappings(self, concept_mapper):
        """Test getting all available mappings for a CUI."""
        result = concept_mapper.get_all_mappings("409681000000102")
        assert isinstance(result, dict)
        assert "snomed_cui" in result
        assert "icd_10" in result
        assert "loinc" in result
        assert "rxnorm" in result

    def test_export_mappings(self, concept_mapper):
        """Test exporting mappings to CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name
        try:
            count = concept_mapper.export_mappings(["409681000000102"], output_path)
            assert isinstance(count, int)
            if count > 0:
                assert os.path.exists(output_path)
                df = pd.read_csv(output_path)
                assert len(df) == count
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_map_concept_icd(self, concept_mapper):
        """Test map_concept function with ICD-10 target."""
        result = concept_mapper.map_to_icd("409681000000102")
        assert isinstance(result, list)

    def test_map_concept_loinc(self, concept_mapper):
        """Test map_concept function with LOINC target."""
        result = concept_mapper.map_to_loinc("409681000000102")
        assert isinstance(result, list)

    def test_map_concept_rxnorm(self, concept_mapper):
        """Test map_concept function with RxNorm target."""
        result = concept_mapper.map_to_rxnorm("409681000000102")
        assert isinstance(result, list)

    def test_batch_map_concepts(self, concept_mapper):
        """Test batch mapping of multiple concepts."""
        result = concept_mapper.batch_map_concepts(
            ["409681000000102", "154621002"], ["ICD-10", "LOINC"]
        )
        assert isinstance(result, dict)
        assert len(result) == 2
        for cui in result:
            assert isinstance(result[cui], dict)

    def test_batch_map_concepts_all_vocabs(self, concept_mapper):
        """Test batch mapping with all target vocabularies."""
        result = concept_mapper.batch_map_concepts(["409681000000102"])
        assert isinstance(result, dict)
        assert "409681000000102" in result


class TestConfidenceScoring:
    """Tests for confidence scoring functionality."""

    def test_confidence_calculation_basic(self, concept_mapper):
        """Test basic confidence score calculation."""
        row = pd.Series({"correlationId": "723899007"})
        confidence = concept_mapper._calculate_confidence(row)
        assert 0.9 <= confidence <= 1.0

    def test_confidence_calculation_missing_correlation(self, concept_mapper):
        """Test confidence score when correlation ID is missing."""
        row = pd.Series({})
        confidence = concept_mapper._calculate_confidence(row)
        assert 0.4 <= confidence <= 0.6


class TestICDDescriptions:
    """Tests for ICD-10 description lookup."""

    def test_icd_description_known_code(self, concept_mapper):
        """Test getting description for a known ICD code."""
        desc = concept_mapper._get_icd_description("C83.3")
        assert len(desc) > 0
        assert "mantle" in desc.lower() or "lymphoma" in desc.lower()

    def test_icd_description_unknown_code(self, concept_mapper):
        """Test getting description for an unknown ICD code."""
        desc = concept_mapper._get_icd_description("ZZZ99")
        assert len(desc) == 0


class TestLOINCDescriptions:
    """Tests for LOINC description lookup."""

    def test_loinc_description_known_code(self, concept_mapper):
        """Test getting description for a known LOINC code."""
        desc = concept_mapper._get_loinc_description("718-7")
        assert len(desc) > 0
        assert "hemoglobin" in desc.lower()

    def test_loinc_description_unknown_code(self, concept_mapper):
        """Test getting description for an unknown LOINC code."""
        desc = concept_mapper._get_loinc_description("99999-9")
        assert len(desc) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
