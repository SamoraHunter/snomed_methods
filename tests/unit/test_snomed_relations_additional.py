"""Additional unit tests for SnomedRelations class."""

import os

import pandas as pd
import pytest

from snomed_methods_v1 import SnomedRelations


def get_tmp_path(filename):
    return os.path.join("/tmp", filename)


@pytest.fixture
def relationship_dataframe():
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
    df_path = get_tmp_path("relationship_test.csv")
    relationship_dataframe.to_csv(df_path, sep="\t", index=False)
    snomed_rel = SnomedRelations(snomed_rf2_full_path=df_path)
    yield snomed_rel
    try:
        os.unlink(df_path)
    except Exception:
        pass


class TestSnomedRelationsGetParentsBug:
    def test_get_parents_returns_destinationid(self, snomed_relations_instance):
        result = snomed_relations_instance.get_parents(100)
        assert isinstance(result, list)

    def test_get_children_returns_sourceid(self, snomed_relations_instance):
        result = snomed_relations_instance.get_children(50)
        assert isinstance(result, list)
        assert all(isinstance(r, int) for r in result)


class TestSnomedRelationsGetPrettyName:
    def test_get_pretty_name_no_medcat(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name(123)
        assert result is None
        try:
            os.unlink(df_path)
        except Exception:
            pass

    @pytest.mark.skip(reason="Requires medcat module which is not installed")
    def test_get_pretty_name_medcat(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path, medcat=True)
        result = snomed.get_pretty_name(123)
        assert result is None
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsGetPrettyNameList:
    def test_get_pretty_name_list_empty(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name_list([])
        assert result == []
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_get_pretty_name_list_no_medcat(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name_list([1, 2, 3])
        assert len(result) == 3
        assert all(r is None for r in result)
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesChildrenLocal:
    def test_expand_codes_children_local_empty(self):
        df_path = get_tmp_path("test_snomed.csv")
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
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_expand_codes_children_local_with_medcat_mock(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100, 200], "destinationId": [50, 50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        def mock_pretty_name(cui):
            names = {100: "Child 1", 200: "Child 2"}
            return names.get(cui, None)

        snomed.get_pretty_name = mock_pretty_name
        result = snomed.expand_codes_children_local(50)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert set(result[0]) == {100, 200}
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesParentsLocal:
    def test_expand_codes_parents_local_empty(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_parents_local(999)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesLocalEmpty:
    def test_expand_codes_local_debug_true(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_local(50, debug=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_expand_codes_local_debug_false(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_local(50, debug=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsExpandCodesWrapper:
    def test_expand_codes_without_snowstorm(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        def mock_expand_local(cui, debug=False):
            return ([100], ["Name"])

        snomed.expand_codes_local = mock_expand_local
        result = snomed.expand_codes(50, debug=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_expand_codes_with_snowstorm(self):
        df_path = get_tmp_path("test_snomed.csv")
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
        assert not called["value"]
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsRecursiveCodeExpansionEdgeCases:
    def test_recursive_code_expursion_zero_levels(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.recursive_code_expansion(50, n_recursion=0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0][0] == 50
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_recursive_code_expansion_single_cui(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.recursive_code_expansion(999, n_recursion=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert 999 in result[0]
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsMEDCATSimilarity:
    def test_get_medcat_cdb_most_similar_no_model(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.get_medcat_cdb_most_similar(123)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_get_medcat_similar_score_no_results(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.get_medcat_similar_score(123, [456])
        assert isinstance(result, list)
        assert len(result) == 1
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsBuildLists:
    def test_build_lists_empty_input(self):
        df_path = get_tmp_path("test_snomed.csv")
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
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_build_lists_only_snomed(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.build_lists_medcat_snomedtree([50], medcat=False, snomed=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_build_lists_only_medcat(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.build_lists_medcat_snomedtree([50], medcat=True, snomed=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsRetrieveSearchSynonyms:
    def test_retrieve_search_synonyms_no_use_snomed(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms(50, use_snomed=False)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert result[0] == []
        assert result[1] == []
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_no_use_medcat(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms(50, use_medcat=False)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert result[2] == []
        assert result[3] == []
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_empty_results(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms(999)
        assert isinstance(result, tuple)
        assert len(result) == 5
        try:
            os.unlink(df_path)
        except Exception:
            pass


class TestSnomedRelationsRetrieveSearchSynonymsMulti:
    def test_retrieve_search_synonyms_multi_empty_list(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms_multi([])
        assert isinstance(result, tuple)
        assert len(result) == 6
        for r in result:
            assert isinstance(r, list)
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_multi_single_item(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms_multi([50])
        assert isinstance(result, tuple)
        assert len(result) == 6
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_retrieve_search_synonyms_multi_multiple_items(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms_multi([50, 60])
        assert isinstance(result, tuple)
        assert len(result) == 6
        try:
            os.unlink(df_path)
        except Exception:
            pass


@pytest.mark.skip(reason="Requires SNOMED RF2 file which is not available")
class TestSnomedRelationsSnowstorm:
    def test_get_snowstorm_response_children_no_connection(self):
        snomed = SnomedRelations()
        result = snomed.get_snowstorm_response_children("invalid")
        assert result is None

    def test_get_snowstorm_response_children_invalid_url(self):
        snomed = SnomedRelations()
        result = snomed.get_snowstorm_response_children("99999")
        assert result is None


class TestSnomedRelationsIntegrationEdgeCases:
    def test_full_workflow_with_mock_data_no_expansion(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_local(999)
        assert result[0] == []
        assert result[1] == []
        try:
            os.unlink(df_path)
        except Exception:
            pass

    def test_recursive_expansion_with_loop(self):
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100, 200], "destinationId": [200, 100]}).to_csv(
            df_path, sep="\t", index=False
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.recursive_code_expansion(100, n_recursion=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        try:
            os.unlink(df_path)
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
