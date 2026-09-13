"""Unit tests for UMLSCIMapper module."""

import pytest


class TestUMLSCIMapper:
    """Tests for the UMLSCIMapper class."""

    def test_initialization_without_paths(self):
        """Test that mapper can be initialized without file paths."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        assert mapper.uk_path is None
        assert mapper.international_path is None
        assert mapper.simple_map_file is None

    def test_initialization_with_paths(self):
        """Test initialization with explicit file paths."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper(
            uk_path="/test/path",
            international_path="/test/intl",
            simple_map_file="/test/map.txt",
        )
        assert mapper.uk_path == "/test/path"
        assert mapper.international_path == "/test/intl"
        assert mapper.simple_map_file == "/test/map.txt"

    def test_load_snomed_to_umls_no_file(self):
        """Test loading when no mapping file is available."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.load_snomed_to_umls()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_map_to_umls_no_mappings(self):
        """Test mapping for a CUI with no mappings in file."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.map_to_umls("999999")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_map_from_umls_no_mappings(self):
        """Test reverse mapping for a CUI with no mappings."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.map_from_umls("C9999999")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_bidirectional_map_snomed(self):
        """Test bidirectional mapping from SNOMED."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.bidirectional_map("409681000000102", "SNOMED")
        assert isinstance(result, list)

    def test_bidirectional_map_umls(self):
        """Test bidirectional mapping from UMLS CUI."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.bidirectional_map("C0023957", "UMLS_CUI")
        assert isinstance(result, list)

    def test_get_all_mappings_snomed(self):
        """Get all mappings for SNOMED concept."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.get_all_mappings("409681000000102")
        assert isinstance(result, dict)
        assert "snomed_cui" in result
        assert "umls_cuis" in result

    def test_get_all_mappings_umls(self):
        """Get all mappings for UMLS CUI."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.get_all_mappings("C0023957")
        assert isinstance(result, dict)
        assert "umls_cui" in result
        assert "snomed_concepts" in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_map_to_umls(self):
        """Test map_to_umls function."""
        from snomed_methods.umlsci_mapper import map_to_umls

        result = map_to_umls("409681000000102")
        assert isinstance(result, list)

    def test_map_from_umls(self):
        """Test map_from_umls function."""
        from snomed_methods.umlsci_mapper import map_from_umls

        result = map_from_umls("C0023957")
        assert isinstance(result, list)

    def test_batch_map_to_umls(self):
        """Test batch mapping."""
        from snomed_methods.umlsci_mapper import batch_map_to_umls

        results = batch_map_to_umls(["409681000000102", "154621002"])
        assert isinstance(results, dict)
        assert len(results) == 2


class TestUmlsCuiDetection:
    """Tests for UMLS CUI format detection."""

    def test_valid_umls_cui(self):
        """Test valid UMLS CUI format."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        assert mapper._is_umls_cui("C0023957") is True
        assert mapper._is_umls_cui("C1234567") is True

    def test_invalid_umls_cui(self):
        """Test invalid UMLS CUI formats."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        assert mapper._is_umls_cui("C123456") is False  # Only 6 digits
        assert mapper._is_umls_cui("C12345678") is False  # 8 digits
        assert mapper._is_umls_cui("0023957") is False  # Missing C prefix
        assert mapper._is_umls_cui("I10") is False  # ICD code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
