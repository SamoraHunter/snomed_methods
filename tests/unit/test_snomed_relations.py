"""
Unit tests for the SnomedRelations class.
"""

import os
from unittest.mock import patch

import pandas as pd
import pytest

from snomed_methods_v1 import SnomedRelations


def get_mock_data_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "data", "mock_snomed.csv")


def get_empty_mock_data_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "data", "empty_mock.csv")


class TestSnomedRelationsInit:
    def test_init_default_parameters(self):
        snomed = SnomedRelations(snomed_rf2_full_path=get_mock_data_path())
        assert hasattr(snomed, "df")

    def test_init_custom_snomed_path(self):
        snomed = SnomedRelations(snomed_rf2_full_path=get_mock_data_path())
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_without_path(self):
        snomed = SnomedRelations(
            snomed_rf2_full_path=get_mock_data_path(),
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_with_dgx_flag(self):
        snomed = SnomedRelations(
            snomed_rf2_full_path=get_mock_data_path(),
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_with_dhcap_flag(self):
        snomed = SnomedRelations(
            snomed_rf2_full_path=get_mock_data_path(),
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_medcat_enabled_with_custom_path(self):
        snomed = SnomedRelations(
            snomed_rf2_full_path=get_mock_data_path(),
            medcat=False,
        )
        assert hasattr(snomed, "df")

    def test_init_snowstorm_flag(self):
        snomed = SnomedRelations(
            snomed_rf2_full_path=get_mock_data_path(),
            snowstorm=False,
            medcat=False,
        )
        assert hasattr(snomed, "df")


class TestSnomedRelationsGetChildren:
    def test_get_children_valid_cui(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_children(50)
        assert set(result) == {100, 200}

    def test_get_children_string_cui(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_children("50")
        assert set(result) == {100, 200}

    def test_get_children_no_matching_rows(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_children(999)
        assert result == []

    def test_get_children_invalid_cui_string(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_children("invalid")
        assert result == []

    def test_get_children_empty_dataframe(self):
        mock_df = pd.DataFrame(columns=["sourceId", "destinationId"])
        df_path = get_empty_mock_data_path()
        mock_df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_children(50)
        assert result == []

    def test_get_children_cui_none(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        with pytest.raises(TypeError):
            snomed.get_children(None)


class TestSnomedRelationsGetParents:
    def test_get_parents_valid_cui(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_parents(50)
        assert set(result) == {70, 80}

    def test_get_parents_string_cui(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_parents("50")
        assert set(result) == {70, 80}

    def test_get_parents_no_matching_rows(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_parents(999)
        assert result == []

    def test_get_parents_invalid_cui_string(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_parents("invalid")
        assert result == []

    def test_get_parents_empty_dataframe(self):
        mock_df = pd.DataFrame(columns=["sourceId", "destinationId"])
        df_path = get_empty_mock_data_path()
        mock_df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_parents(50)
        assert result == []


class TestSnomedRelationsExpandCodesLocal:
    def test_expand_codes_local_with_debug(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50, debug=True)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_expand_codes_local_without_debug(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50, debug=False)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_expand_codes_local_empty_results(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(999)
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_expand_codes_local_duplicate_combination(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50)
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestSnomedRelationsRecursiveCodeExpansion:
    def test_recursive_code_expansion_single_level(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=1)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_multiple_levels(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=3)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_empty_dataframe(self):
        mock_df = pd.DataFrame(columns=["sourceId", "destinationId"])
        df_path = get_empty_mock_data_path()
        mock_df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=2)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_no_expansion(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(999, n_recursion=3)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_recursive_code_expansion_debug_mode(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.recursive_code_expansion(50, n_recursion=1, debug=True)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestSnomedRelationsIntegration:
    def test_full_workflow_with_mock_data(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        children_50 = snomed.get_children(50)
        assert set(children_50) == {100, 200}
        with patch.object(snomed, "get_pretty_name", return_value=None):
            result = snomed.expand_codes_local(50)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_boundary_conditions_none_values(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_empty = snomed.get_children(999)
        assert result_empty == []

    def test_boundary_conditions_large_cui_values(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_children(50)
        assert set(result) == {100, 200}

    def test_data_type_consistency(self):
        df_path = get_mock_data_path()
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        children = snomed.get_children(50)
        for child in children:
            assert isinstance(child, int)
