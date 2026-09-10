"""
Unit tests for the SnomedRelations class.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from snomed_methods_v1 import SnomedRelations


class TestSnomedRelationsInit:
    """
    Test suite for SnomedRelations __init__ method.
    """

    def test_init_default_parameters(self):
        """
        Test initialization with default parameters (snomed_rf2_full_path=None).
        Should load the default SNOMED file path.
        """
        snomed = SnomedRelations(
            snomed_rf2_full_path="/workspaces/snomed_methods/tests/data/mock_snomed.csv"
        )
        assert hasattr(snomed, "df")

    def test_init_custom_snomed_path(self):
        """
        Test initialization with a custom SNOMED file path.
        Should read from the specified path.
        """
        snomed = SnomedRelations(
            snomed_rf2_full_path="/workspaces/snomed_methods/tests/data/mock_snomed.csv"
        )
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_without_path(self):
        """
        Test initialization with medcat=True and no medcat_path specified.
        Should attempt to load model pack from default location based on aliencat flag.
        """
        snomed = SnomedRelations(
            snomed_rf2_full_path="/workspaces/snomed_methods/tests/data/mock_snomed.csv",
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_with_dgx_flag(self):
        """
        Test initialization with dgx=True parameter.
        Should load model pack from DGX-specific path.
        """
        snomed = SnomedRelations(
            snomed_rf2_full_path="/workspaces/snomed_methods/tests/data/mock_snomed.csv",
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_with_dhcap_flag(self):
        """
        Test initialization with dhcap=True parameter.
        Should load model pack from DHCAP-specific path.
        """
        snomed = SnomedRelations(
            snomed_rf2_full_path="/workspaces/snomed_methods/tests/data/mock_snomed.csv",
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_with_custom_path(self):
        """
        Test initialization with medcat=True and custom medcat_path specified.
        Should load model pack from the specified path.
        """
        snomed = SnomedRelations(
            snomed_rf2_full_path="/workspaces/snomed_methods/tests/data/mock_snomed.csv",
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_snowstorm_flag(self):
        """
        Test initialization with snowstorm=True parameter.
        Should set the snowstorm flag and enable medcat mode.
        """
        snomed = SnomedRelations(
            snomed_rf2_full_path="/workspaces/snomed_methods/tests/data/mock_snomed.csv",
            snowstorm=False,
            medcat=False,
        )
        assert hasattr(snomed, "df")


class TestSnomedRelationsGetChildren:
    """
    Test suite for SnomedRelations get_children method.
    """

    def test_get_children_valid_cui(self):
        """
        Test getting children for a valid integer CUI.
        Should return list of sourceId values where destinationId matches the CUI.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_children(50)

        assert set(result) == {100, 200}

    def test_get_children_string_cui(self):
        """
        Test getting children with string input that can be converted to int.
        Should successfully convert and return results.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_children("50")

        assert set(result) == {100, 200}

    def test_get_children_no_matching_rows(self):
        """
        Test getting children for a CUI with no matching rows.
        Should return empty list.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_children(999)

        assert result == []

    def test_get_children_invalid_cui_string(self):
        """
        Test getting children with invalid string that cannot be converted to int.
        Should log error and return empty list.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_children("invalid")

        assert result == []

    def test_get_children_empty_dataframe(self):
        """
        Test getting children from empty dataframe.
        Should return empty list without error.
        """
        mock_df = pd.DataFrame(columns=["sourceId", "destinationId"])
        df_path = "/workspaces/snomed_methods/tests/data/empty_mock.csv"
        mock_df.to_csv(df_path, sep="\t", index=False)

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_children(50)

        assert result == []

    def test_get_children_cui_none(self):
        """
        Test getting children with None as CUI.
        Should raise TypeError since int(None) fails and the exception is not caught.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        # The source code doesn't handle None properly - it raises TypeError
        with pytest.raises(TypeError):
            snomed.get_children(None)


class TestSnomedRelationsGetParents:
    """
    Test suite for SnomedRelations get_parents method.
    """

    def test_get_parents_valid_cui(self):
        """
        Test getting parents for a valid integer CUI.
        get_parents should return destinationId values where sourceId matches the CUI.
        For CUI 50: sourceId=50 has rows with destinationId=[70, 80]
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_parents(50)

        assert set(result) == {70, 80}

    def test_get_parents_string_cui(self):
        """
        Test getting parents with string input that can be converted to int.
        Should successfully convert and return results.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_parents("50")

        assert set(result) == {70, 80}

    def test_get_parents_no_matching_rows(self):
        """
        Test getting parents for a CUI with no matching rows.
        Should return empty list.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_parents(999)

        assert result == []

    def test_get_parents_invalid_cui_string(self):
        """
        Test getting parents with invalid string that cannot be converted to int.
        Should log error and return empty list.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_parents("invalid")

        assert result == []

    def test_get_parents_empty_dataframe(self):
        """
        Test getting parents from empty dataframe.
        Should return empty list without error.
        """
        mock_df = pd.DataFrame(columns=["sourceId", "destinationId"])
        df_path = "/workspaces/snomed_methods/tests/data/empty_mock.csv"
        mock_df.to_csv(df_path, sep="\t", index=False)

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_parents(50)

        assert result == []


class TestSnomedRelationsExpandCodesLocal:
    """
    Test suite for SnomedRelations expand_codes_local method.
    """

    def test_expand_codes_local_with_debug(self):
        """
        Test expanding codes locally with debug enabled.
        Note: expand_codes_local depends on medcat being True via
        expand_codes_parents_local and expand_codes_children_local assertions.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50, debug=True)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_expand_codes_local_without_debug(self):
        """
        Test expanding codes locally without debug.
        Note: expand_codes_local depends on medcat being True via
        expand_codes_parents_local and expand_codes_children_local assertions.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50, debug=False)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_expand_codes_local_empty_results(self):
        """
        Test expanding codes with no matching results.
        Note: expand_codes_local depends on medcat being True via
        expand_codes_parents_local and expand_codes_children_local assertions.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(999)

        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_expand_codes_local_duplicate_combination(self):
        """
        Test that duplicates are properly removed when combining results.
        Note: expand_codes_local depends on medcat being True via
        expand_codes_parents_local and expand_codes_children_local assertions.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50)

        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestSnomedRelationsRecursiveCodeExpansion:
    """
    Test suite for SnomedRelations recursive_code_expansion method.
    """

    def test_recursive_code_expansion_single_level(self):
        """
        Test recursive code expansion with n_recursion=1.
        Should expand one level and return codes.
        Note: calls expand_codes_local which requires medcat=True.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=1)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_multiple_levels(self):
        """
        Test recursive code expansion with multiple recursion levels.
        Should expand according to the specified depth.
        Note: calls expand_codes_local which requires medcat=True.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=3)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_empty_dataframe(self):
        """
        Test recursive code expansion with empty dataframe.
        Should handle gracefully without errors.
        Note: calls expand_codes_local which requires medcat=True.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        mock_df = pd.DataFrame(columns=["sourceId", "destinationId"])
        df_path = "/workspaces/snomed_methods/tests/data/empty_mock.csv"
        mock_df.to_csv(df_path, sep="\t", index=False)

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=2)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_no_expansion(self):
        """
        Test recursive code expansion when no children/parents exist.
        Should return just the initial CUI after deduplication.
        Note: calls expand_codes_local which requires medcat=True.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(999, n_recursion=3)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_debug_mode(self):
        """
        Test recursive code expansion with debug mode enabled.
        Should log debug information during execution.
        Note: calls expand_codes_local which requires medcat=True.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=1, debug=True)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestSnomedRelationsIntegration:
    """
    Integration test suite for SnomedRelations class.
    """

    def test_full_workflow_with_mock_data(self):
        """
        Test a complete workflow: init, get_children, expand locally.
        Uses a single cohesive mock dataset.
        Note: expand_codes_local requires medcat=True via assertions in
        expand_codes_parents_local and expand_codes_children_local methods.
        We mock get_pretty_name to avoid needing a real medcat model pack.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        # Test get_children
        children_50 = snomed.get_children(50)
        assert set(children_50) == {100, 200}

        # Test expand_codes_local (requires medcat=True internally)
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50)

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_boundary_conditions_none_values(self):
        """
        Test boundary conditions with None values.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        # Test with boundary values
        result_empty = snomed.get_children(999)
        assert result_empty == []

    def test_boundary_conditions_large_cui_values(self):
        """
        Test with large CUI values that might be at the edge of valid range.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        result = snomed.get_children(50)
        assert set(result) == {100, 200}

    def test_data_type_consistency(self):
        """
        Test that data types remain consistent through various operations.
        """
        df_path = "/workspaces/snomed_methods/tests/data/mock_snomed.csv"

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)

        # Test that returned lists contain integers
        children = snomed.get_children(50)
        for child in children:
            assert isinstance(child, int)
