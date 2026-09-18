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

    def test_initialization_without_paths(self) -> None:
        """Test that mapper can be initialized without file paths."""
        # Import directly to avoid heavy package-level imports
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        assert mapper.uk_path is None
        assert mapper.icd_map_file is None
        assert mapper.loinc_map_file is None

    def test_initialization_with_paths(self) -> None:
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

    def test_load_icd_mapping_no_file(self, concept_mapper) -> None:
        """Test loading ICD mapping when no file is available."""
        result = concept_mapper.load_icd_mapping()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_load_loinc_mapping_no_file(self, concept_mapper) -> None:
        """Test loading LOINC mapping when no file is available."""
        result = concept_mapper.load_loinc_mapping()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_map_to_icd_no_mappings(self, concept_mapper) -> None:
        """Test mapping for a CUI with no ICD mappings."""
        result = concept_mapper.map_to_icd("999999")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_map_to_loinc_no_mappings(self, concept_mapper) -> None:
        """Test mapping for a CUI with no LOINC mappings."""
        result = concept_mapper.map_to_loinc("999999")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_map_to_rxnorm_no_medcat(self, concept_mapper) -> None:
        """Test RxNorm mapping without MedCAT model pack."""
        result = concept_mapper.map_to_rxnorm("409681000000102")
        assert isinstance(result, list)

    def test_map_bidirectional_snomed(self, concept_mapper) -> None:
        """Test bidirectional mapping from SNOMED."""
        result = concept_mapper.map_bidirectional("409681000000102", "SNOMED")
        assert isinstance(result, list)

    def test_map_bidirectional_icd10(self, concept_mapper) -> None:
        """Test bidirectional mapping from ICD-10."""
        result = concept_mapper.map_bidirectional("J44.1", "ICD10")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["target_vocab"] == "SNOMED (estimated)"

    def test_map_bidirectional_loinc(self, concept_mapper) -> None:
        """Test bidirectional mapping from LOINC."""
        result = concept_mapper.map_bidirectional("718-7", "LOINC")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["target_vocab"] == "SNOMED (estimated)"

    def test_multi_hop_mapping(self) -> None:
        """Test multi-hop mapping to multiple target vocabularies."""
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        result = mapper.multi_hop_mapping(
            "409681000000102",
            ["ICD-10", "LOINC"],
            max_hops=2,
        )
        assert isinstance(result, dict)
        assert result["start_cui"] == "409681000000102"
        assert len(result["target_vocabs"]) == 2

    def test_get_all_mappings(self, concept_mapper) -> None:
        """Test getting all available mappings for a CUI."""
        result = concept_mapper.get_all_mappings("409681000000102")
        assert isinstance(result, dict)
        assert "snomed_cui" in result
        assert "icd_10" in result
        assert "loinc" in result
        assert "rxnorm" in result

    def test_export_mappings(self, concept_mapper) -> None:
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

    def test_load_icd_mapping_with_full_columns(self, concept_mapper) -> None:
        """Test ICD mapping loads mapGroup, mapPriority, correlationId columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            tsv_path = f.name
        try:
            df_content = (
                "referencedComponentId\tmapTarget\tactive\tmapGroup\t"
                "mapPriority\tcorrelationId\n409681000000102\tDiabetes "
                "active\t1\t1\t1\t723899007"
            )
            with open(tsv_path, "w") as f:
                f.write(df_content)

            from src.snomed_methods.concept_mapper import ConceptMapper

            concept_mapper_with_file = ConceptMapper(icd_map_file=tsv_path)
            result = concept_mapper_with_file.load_icd_mapping()

            assert isinstance(result, dict)
            if "409681000000102" in result and len(result["409681000000102"]) > 0:
                mapping = result["409681000000102"][0]
                assert "map_group" in mapping
                assert "map_priority" in mapping
                assert "correlation_id" in mapping
        finally:
            if os.path.exists(tsv_path):
                os.unlink(tsv_path)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_map_concept_icd(self, concept_mapper) -> None:
        """Test map_concept function with ICD-10 target."""
        result = concept_mapper.map_to_icd("409681000000102")
        assert isinstance(result, list)

    def test_map_concept_loinc(self, concept_mapper) -> None:
        """Test map_concept function with LOINC target."""
        result = concept_mapper.map_to_loinc("409681000000102")
        assert isinstance(result, list)

    def test_map_concept_rxnorm(self, concept_mapper) -> None:
        """Test map_concept function with RxNorm target."""
        result = concept_mapper.map_to_rxnorm("409681000000102")
        assert isinstance(result, list)

    def test_batch_map_concepts(self, concept_mapper) -> None:
        """Test batch mapping of multiple concepts."""
        result = concept_mapper.batch_map_concepts(
            ["409681000000102", "154621002"],
            ["ICD-10", "LOINC"],
        )
        assert isinstance(result, dict)
        assert len(result) == 2
        for cui in result:
            assert isinstance(result[cui], dict)

    def test_batch_map_concepts_all_vocabs(self, concept_mapper) -> None:
        """Test batch mapping with all target vocabularies."""
        result = concept_mapper.batch_map_concepts(["409681000000102"])
        assert isinstance(result, dict)
        assert "409681000000102" in result


class TestConfidenceScoring:
    """Tests for confidence scoring functionality."""

    def test_confidence_calculation_basic(self, concept_mapper) -> None:
        """Test basic confidence score calculation."""
        row = pd.Series({"correlationId": "723899007"})
        confidence = concept_mapper._calculate_confidence(row)
        assert 0.9 <= confidence <= 1.0

    def test_confidence_calculation_missing_correlation(self, concept_mapper) -> None:
        """Test confidence score when correlation ID is missing."""
        row = pd.Series({})
        confidence = concept_mapper._calculate_confidence(row)
        assert 0.4 <= confidence <= 0.6

    def test_confidence_calculation_with_category_id(self, concept_mapper) -> None:
        """Test confidence score with mapCategoryId set."""
        row = pd.Series({"correlationId": "723895001", "mapCategoryId": "723886003"})
        confidence = concept_mapper._calculate_confidence(row)

        assert 0.4 <= confidence <= 1.0

    def test_confidence_calculation_high_correlation(self, concept_mapper) -> None:
        """Test confidence score with high correlation ID."""
        row = pd.Series({"correlationId": "723899007"})
        confidence = concept_mapper._calculate_confidence(row)

        assert confidence == 0.95

    def test_confidence_calculation_low_correlation(self, concept_mapper) -> None:
        """Test confidence score with low correlation ID."""
        row = pd.Series({"correlationId": "723896002"})
        confidence = concept_mapper._calculate_confidence(row)

        assert 0.6 <= confidence <= 0.7

    def test_confidence_calculation_category_id_only(self, concept_mapper) -> None:
        """Test confidence score with only mapCategoryId (no correlationId)."""
        row = pd.Series({"mapCategoryId": "723886003"})
        confidence = concept_mapper._calculate_confidence(row)

        assert 0.7 <= confidence <= 0.9


class TestConceptMapperEdgeCasesDetailed:
    """Additional edge case tests for ConceptMapper."""

    def test_batch_map_concepts_single_cui_with_vocabs(self, concept_mapper) -> None:
        result = concept_mapper.batch_map_concepts(["73211009"], ["ICD-10"])

        assert "73211009" in result
        assert "ICD-10" in result["73211009"]

    def test_multi_hop_mapping_with_rxnorm(self) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        result = mapper.multi_hop_mapping("409681000000102", ["RxNorm"], max_hops=2)

        assert "RxNorm" in result["target_vocabs"]


class TestICDDescriptions:
    """Tests for ICD-10 description lookup."""

    def test_icd_description_known_code(self, concept_mapper) -> None:
        """Test getting description for a known ICD code."""
        desc = concept_mapper._get_icd_description("C83.3")
        assert len(desc) > 0
        assert "mantle" in desc.lower() or "lymphoma" in desc.lower()

    def test_icd_description_unknown_code(self, concept_mapper) -> None:
        """Test getting description for an unknown ICD code."""
        desc = concept_mapper._get_icd_description("ZZZ99")
        assert len(desc) == 0


class TestLOINCDescriptions:
    """Tests for LOINC description lookup."""

    def test_loinc_description_known_code(self, concept_mapper) -> None:
        """Test getting description for a known LOINC code."""
        desc = concept_mapper._get_loinc_description("718-7")
        assert len(desc) > 0
        assert "hemoglobin" in desc.lower()

    def test_loinc_description_unknown_code(self, concept_mapper) -> None:
        """Test getting description for an unknown LOINC code."""
        desc = concept_mapper._get_loinc_description("99999-9")
        assert len(desc) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestConceptMapperEdgeCases:
    """Tests for ConceptMapper edge cases."""

    def test_multi_hop_mapping_empty_vocabs(self) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        result = mapper.multi_hop_mapping("409681000000102", [], max_hops=2)

        assert len(result["paths"]) == 0

    def test_export_mappings_empty_list(self, concept_mapper) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name
        try:
            count = concept_mapper.export_mappings([], output_path)
            assert count == 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_batch_map_concepts_empty_list(self, concept_mapper) -> None:
        result = concept_mapper.batch_map_concepts([])

        assert result == {}

    def test_map_bidirectional_unknown_vocab(self, concept_mapper) -> None:
        result = concept_mapper.map_bidirectional("409681000000102", "UNKNOWN")

        assert result == []


class TestUMLSCIMapperEdgeCases:
    """Tests for UMLSCIMapper edge cases."""

    def test_initialization_without_paths(self) -> None:
        from src.snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        assert mapper.uk_path is None
        assert mapper.international_path is None
        assert mapper.simple_map_file is None

    def test_is_umls_cui_valid(self) -> None:
        from src.snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()

        assert mapper._is_umls_cui("C0012345") is True
        assert mapper._is_umls_cui("c0012345") is False
        assert mapper._is_umls_cui("0012345") is False
        assert mapper._is_umls_cui("C001234") is False
        assert mapper._is_umls_cui("C00123456") is False

    def test_get_all_mappings_with_snomed(self) -> None:
        from src.snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.get_all_mappings("73211009")

        assert "snomed_cui" in result
        assert "umls_cuis" in result

    def test_get_all_mappings_with_umls(self) -> None:
        from src.snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.get_all_mappings("C0012345")

        assert "umls_cui" in result
        assert "snomed_concepts" in result

    def test_export_mappings_empty_list(self) -> None:
        import tempfile

        from src.snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name
        try:
            count = mapper.export_mappings([], output_path)
            assert count == 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestUMLSCIMapperConvenienceFunctions:
    """Tests for UMLSCIMapper convenience functions."""

    def test_map_to_umls_convenience(self) -> None:
        from src.snomed_methods.umlsci_mapper import map_to_umls

        result = map_to_umls("73211009")

        assert isinstance(result, list)

    def test_batch_map_to_umls(self) -> None:
        from src.snomed_methods.umlsci_mapper import batch_map_to_umls

        result = batch_map_to_umls(["73211009", "38341003"])

        assert isinstance(result, dict)
        for cui in result:
            assert isinstance(result[cui], list)


class TestBatchOperations:
    """Tests for batch operations."""

    def test_batch_map_concepts_with_empty_vocabs(self) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        result = mapper.batch_map_concepts(["409681000000102"], [])

        assert "409681000000102" in result
        assert result["409681000000102"] == {}

    def test_get_all_mappings_with_empty_cui(self) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        result = mapper.get_all_mappings("")

        assert "snomed_cui" in result


class TestConceptMapperPathsAndConfig:
    """Tests for path configuration and auto-detection."""

    def test_get_refset_name_with_icd_file(self, concept_mapper) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper(icd_map_file="/test/icd.map")
        name = mapper._get_refset_name()

        assert "icd.map" in name

    def test_get_refset_name_with_loinc_file(self, concept_mapper) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper(loinc_map_file="/test/loinc.map")
        name = mapper._get_refset_name()

        assert "loinc.map" in name

    def test_get_refset_name_default(self, concept_mapper) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        name = mapper._get_refset_name()

        assert "default SNOMED CT reference set" in name


class TestConceptMapperAutoDetectPaths:
    """Tests for _auto_detect_paths method."""

    def test_auto_detect_with_invalid_path(self, concept_mapper) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper(uk_path="/nonexistent/path")
        assert mapper.uk_path == "/nonexistent/path"

    def test_map_to_icd_after_load(self) -> None:
        from src.snomed_methods.concept_mapper import ConceptMapper

        mapper = ConceptMapper()
        result = mapper.map_to_icd("73211009")

        assert isinstance(result, list)

    def test_batch_map_concepts_single_cui_all_vocabularies(
        self,
        concept_mapper,
    ) -> None:
        result = concept_mapper.batch_map_concepts(["409681000000102"])

        assert "409681000000102" in result
