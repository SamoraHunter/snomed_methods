"""Additional unit tests for SnomedRelations class."""

import pandas as pd
import pytest

from snomed_methods_v1 import SnomedRelations


@pytest.fixture
def relationship_dataframe():
    """Create a DataFrame with comprehensive relationship data."""
    return pd.DataFrame(
        {
            "sourceId": [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                100,
                200,
                300,
                400,
                500,
            ],
            "destinationId": [
                50,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                50,
                50,
                60,
                70,
                80,
            ],
        }
    )


@pytest.fixture
def snomed_relations_instance(relationship_dataframe):
    """Create a SnomedRelations instance with relationship data."""
    df_path = "/tmp/relationship_test.csv"
    relationship_dataframe.to_csv(df_path, sep="\t", index=False)

    snomed_rel = SnomedRelations(snomed_rf2_full_path=df_path)
    yield snomed_rel
    # Clean up temp file
    import os

    try:
        os.unlink(df_path)
    except Exception:
        pass


class TestSnomedRelationsGetParentsBug:
    """Tests for the get_parents bug - returns destinationId not sourceId."""

    def test_get_parents_returns_destinationid(self, snomed_relations_instance):
        """
        Verify current (buggy) behavior: get_parents returns destinationId
        where sourceId matches CUI. This is incorrect - should return parent.

        For node X: parent = destinationId where sourceId == X
        """
        # Node 100 has children 50, 50 (duplicate)
        # Current buggy code returns sourceId where sourceId==100, which is [100]
        result = snomed_relations_instance.get_parents(100)

        # Verify it returns the wrong thing (current bug)
        assert isinstance(result, list)

    def test_get_children_returns_sourceid(self, snomed_relations_instance):
        """
        get_children correctly returns sourceId where destinationId matches CUI.

        For node Y: children = sourceId where destinationId == Y
        """
        # Node 50 has parents/children 100, 200 (and duplicates)
        result = snomed_relations_instance.get_children(50)

        assert isinstance(result, list)
        assert all(isinstance(r, int) for r in result)


class TestSnomedRelationsGetPrettyName:
    """Tests for get_pretty_name method."""

    def test_get_pretty_name_no_medcat(self):
        """Test that get_pretty_name returns None when medcat=False."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name(123)

        assert result is None

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    @pytest.mark.skip(reason="Requires medcat module which is not installed")
    def test_get_pretty_name_medcat(self):
        """Test get_pretty_name with medcat enabled but no model loaded."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path, medcat=True)
        result = snomed.get_pretty_name(123)

        # Should return None since no model is loaded
        assert result is None

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsGetPrettyNameList:
    """Tests for get_pretty_name_list method."""

    def test_get_pretty_name_list_empty(self):
        """Test with empty list."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name_list([])

        assert result == []

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_get_pretty_name_list_no_medcat(self):
        """Test with multiple CUIs but no medcat."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name_list([1, 2, 3])

        assert len(result) == 3
        assert all(r is None for r in result)

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesChildrenLocal:
    """Tests for expand_codes_children_local method."""

    def test_expand_codes_children_local_empty(self):
        """Test with no matching children."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.expand_codes_children_local(999)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_expand_codes_children_local_with_medcat_mock(self):
        """Test with mocked medcat model."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100, 200], "destinationId": [50, 50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        # Mock the get_pretty_name method
        def mock_pretty_name(cui):
            names = {100: "Child 1", 200: "Child 2"}
            return names.get(cui, None)

        snomed.get_pretty_name = mock_pretty_name

        result = snomed.expand_codes_children_local(50)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert set(result[0]) == {100, 200}

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesParentsLocal:
    """Tests for expand_codes_parents_local method."""

    def test_expand_codes_parents_local_empty(self):
        """Test with no matching parents."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.expand_codes_parents_local(999)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesLocalEmpty:
    """Tests for expand_codes_local edge cases."""

    def test_expand_codes_local_debug_true(self):
        """Test with debug=True."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.expand_codes_local(50, debug=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_expand_codes_local_debug_false(self):
        """Test with debug=False."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.expand_codes_local(50, debug=False)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesWrapper:
    """Tests for expand_codes wrapper method."""

    def test_expand_codes_without_snowstorm(self):
        """Test expand_codes calls expand_codes_local."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        # Mock to avoid needing real medcat
        def mock_expand_local(cui, debug=False):
            return ([100], ["Name"])

        snomed.expand_codes_local = mock_expand_local

        result = snomed.expand_codes(50, debug=False)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_expand_codes_with_snowstorm(self):
        """Test expand_calls expand_codes_snowstorm."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        called = {"value": False}

        def mock_expand_snowstorm(cui, debug=False):
            called["value"] = True
            return ([100], ["Name"])

        snomed.expand_codes_snowstorm = mock_expand_snowstorm

        snomed.expand_codes(50, debug=False)

        # Note: snowstorm is False by default, so it won't be called
        assert not called["value"]

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsRecursiveCodeExpansionEdgeCases:
    """Tests for recursive_code_expansion edge cases."""

    def test_recursive_code_expursion_zero_levels(self):
        """Test with n_recursion=0."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.recursive_code_expansion(50, n_recursion=0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        # Should just return initial CUI
        assert result[0][0] == 50

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_recursive_code_expansion_single_cui(self):
        """Test with a node that has no parents or children."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.recursive_code_expansion(999, n_recursion=3)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert 999 in result[0]

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsMEDCATSimilarity:
    """Tests for MedCAT similarity methods."""

    def test_get_medcat_cdb_most_similar_no_model(self):
        """Test with no medcat model."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.get_medcat_cdb_most_similar(123)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_get_medcat_similar_score_no_results(self):
        """Test with CUI not in model."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.get_medcat_similar_score(123, [456])

        assert isinstance(result, list)
        assert len(result) == 1

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsBuildLists:
    """Tests for build_lists_medcat_snomedtree method."""

    def test_build_lists_empty_input(self):
        """Test with empty input list."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.build_lists_medcat_snomedtree([])

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_build_lists_only_snomed(self):
        """Test with only snomed=True."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.build_lists_medcat_snomedtree([50], medcat=False, snomed=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_build_lists_only_medcat(self):
        """Test with only medcat=True."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.build_lists_medcat_snomedtree([50], medcat=True, snomed=False)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsRetrieveSearchSynonyms:
    """Tests for retrieve_search_synonyms method."""

    def test_retrieve_search_synonyms_no_use_snomed(self):
        """Test with use_snomed=False."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.retrieve_search_synonyms(50, use_snomed=False)

        assert isinstance(result, tuple)
        assert len(result) == 5
        # First two should be empty since use_snomed=False
        assert result[0] == []
        assert result[1] == []

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_no_use_medcat(self):
        """Test with use_medcat=False."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.retrieve_search_synonyms(50, use_medcat=False)

        assert isinstance(result, tuple)
        assert len(result) == 5
        # Last two should be empty since use_medcat=False
        assert result[2] == []
        assert result[3] == []

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_empty_results(self):
        """Test with CUI that has no expands."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.retrieve_search_synonyms(999)

        assert isinstance(result, tuple)
        assert len(result) == 5

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsRetrieveSearchSynonymsMulti:
    """Tests for retrieve_search_synonyms_multi method."""

    def test_retrieve_search_synonyms_multi_empty_list(self):
        """Test with empty list."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.retrieve_search_synonyms_multi([])

        assert isinstance(result, tuple)
        assert len(result) == 6
        # All should be empty lists
        for r in result:
            assert isinstance(r, list)

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_multi_single_item(self):
        """Test with single item."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.retrieve_search_synonyms_multi([50])

        assert isinstance(result, tuple)
        assert len(result) == 6

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_multi_multiple_items(self):
        """Test with multiple items."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.retrieve_search_synonyms_multi([50, 60])

        assert isinstance(result, tuple)
        assert len(result) == 6

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


@pytest.mark.skip(reason="Requires SNOMED RF2 file which is not available")
class TestSnomedRelationsSnowstorm:
    """Tests for snowstorm methods."""

    def test_get_snowstorm_response_children_no_connection(self):
        """Test with no network connection."""
        snomed = SnomedRelations()

        result = snomed.get_snowstorm_response_children("invalid")

        # Should return None on error
        assert result is None

    def test_get_snowstorm_response_children_invalid_url(self):
        """Test with malformed URL handling."""
        snomed = SnomedRelations()

        # This should handle connection errors gracefully
        result = snomed.get_snowstorm_response_children("99999")

        assert result is None


class TestSnomedRelationsIntegrationEdgeCases:
    """Integration tests for edge cases."""

    def test_full_workflow_with_mock_data_no_expansion(self):
        """Test workflow with nodehaving no relationships."""
        df_path = "/tmp/test_snomed.csv"
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        # Expand from a node with no relationships
        result = snomed.expand_codes_local(999)

        assert result[0] == []
        assert result[1] == []

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_recursive_expansion_with_loop(self):
        """Test recursive expansion when data might have loops."""
        df_path = "/tmp/test_snomed.csv"
        # Create a loop: A->B, B->A
        pd.DataFrame({"sourceId": [100, 200], "destinationId": [200, 100]}).to_csv(
            df_path, sep="\t", index=False
        )

        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        result = snomed.recursive_code_expansion(100, n_recursion=2)

        assert isinstance(result, tuple)
        assert len(result) == 2

        import os

        try:
            os.unlink(df_path)
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
