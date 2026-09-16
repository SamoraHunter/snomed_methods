#!/usr/bin/env python3
"""Additional tests for UMLSCIMapper module."""

import pytest


class TestUMLSCIMapperAdditional:
    """Additional tests for UMLSCIMapper coverage."""

    def test_process_mapping_row_active_false(self):
        """Test processing inactive mapping row."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        mock_row = {
            "active": False,
            "referencedComponentId": "12345",
            "mapTarget": "C0001234",
        }
        result = mapper._process_mapping_row(mock_row, "/fake/file.map")
        assert result is None

    def test_process_mapping_row_invalid_umls(self):
        """Test processing with non-UMLS CUI target."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        mock_row = {
            "active": True,
            "referencedComponentId": "12345",
            "mapTarget": "NOTACUI",
        }
        result = mapper._process_mapping_row(mock_row, "/fake/file.map")
        assert result is None

    def test_process_mapping_row_valid(self):
        """Test processing valid mapping row."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        mock_row = {
            "active": True,
            "referencedComponentId": "12345",
            "mapTarget": "C0001234",
        }
        result = mapper._process_mapping_row(mock_row, "/fake/file.map")
        assert result is not None
        snomed_cui, mapping = result
        assert snomed_cui == "12345"
        assert mapping["umls_cui"] == "C0001234"

    def test_add_mapping_duplicate(self):
        """Test adding duplicate mapping."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        mapping = {"umls_cui": "C0001234", "confidence": 0.75, "source": "test.map"}
        result1 = mapper._add_mapping("12345", mapping)
        result2 = mapper._add_mapping("12345", mapping)
        assert result1 is not None
        assert result2 is None

    def test_is_umls_cui_valid(self):
        """Test valid UMLS CUI detection."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        assert mapper._is_umls_cui("C0001234") is True
        assert mapper._is_umls_cui("C9999999") is True

    def test_is_umls_cui_invalid(self):
        """Test invalid UMLS CUI detection."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        assert mapper._is_umls_cui("NOTACUI") is False
        assert mapper._is_umls_cui("C1234") is False
        assert mapper._is_umls_cui("") is False


class TestUMLSCIMapperBidirectional:
    """Tests for bidirectional mapping."""

    def test_bidirectional_map_empty_source(self):
        """Test bidirectional with empty source."""
        from snomed_methods.umlsci_mapper import UMLSCIMapper

        mapper = UMLSCIMapper()
        result = mapper.bidirectional_map("", "SNOMED")
        assert isinstance(result, list)
        assert len(result) == 0


class TestUMLSCIMapperConvenience:
    """Tests for convenience functions."""

    def test_batch_map_to_umls_empty(self):
        """Test batch mapping with empty list."""
        from snomed_methods.umlsci_mapper import batch_map_to_umls

        result = batch_map_to_umls([])
        assert isinstance(result, dict)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
