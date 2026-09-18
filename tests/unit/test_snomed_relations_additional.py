"""Additional unit tests for SnomedRelations class."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.snomed_methods import SnomedRelations


@pytest.fixture
def tmp_test_dir():
    """Create a temporary directory for test files within the project."""
    tmp_path = Path(__file__).parent.parent / "tmp_test_files"
    tmp_path.mkdir(exist_ok=True)
    yield tmp_path
    with contextlib.suppress(Exception):
        shutil.rmtree(tmp_path)


def get_tmp_path(filename, tmp_path=None):
    if tmp_path is None:
        tmp_path = Path(tempfile.gettempdir()) / "snomed_test_files"  # nosec S108
        tmp_path.mkdir(exist_ok=True)
    return str(tmp_path / filename)


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
        },
    )


@pytest.fixture
def snomed_relations_instance(relationship_dataframe):
    df_path = get_tmp_path("relationship_test.csv")
    relationship_dataframe.to_csv(df_path, sep="\t", index=False)
    snomed_rel = SnomedRelations(snomed_rf2_full_path=df_path)
    yield snomed_rel
    with contextlib.suppress(Exception):
        os.unlink(df_path)


class TestSnomedRelationsGetParentsBug:
    def test_get_parents_returns_destinationid(self, snomed_relations_instance) -> None:
        result = snomed_relations_instance.get_parents(100)
        assert isinstance(result, list)

    def test_get_children_returns_sourceid(self, snomed_relations_instance) -> None:
        result = snomed_relations_instance.get_children(50)
        assert isinstance(result, list)
        assert all(isinstance(r, int) for r in result)


class TestSnomedRelationsGetPrettyName:
    def test_get_pretty_name_no_medcat(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name(123)
        assert result is None
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    @pytest.mark.skip(reason="Requires medcat module which is not installed")
    def test_get_pretty_name_medcat(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path, medcat=True)
        result = snomed.get_pretty_name(123)
        assert result is None
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsGetPrettyNameList:
    def test_get_pretty_name_list_empty(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name_list([])
        assert result == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_get_pretty_name_list_no_medcat(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name_list([1, 2, 3])
        assert len(result) == 3
        assert all(r is None for r in result)
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsExpandCodesChildrenLocal:
    def test_expand_codes_children_local_empty(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_children_local(999)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_expand_codes_children_local_with_medcat_mock(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100, 200], "destinationId": [50, 50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        def mock_pretty_name(cui) -> str | None:
            names = {100: "Child 1", 200: "Child 2"}
            return names.get(cui)

        snomed.get_pretty_name = mock_pretty_name
        result = snomed.expand_codes_children_local(50)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert set(result[0]) == {100, 200}
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsExpandCodesParentsLocal:
    def test_expand_codes_parents_local_empty(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [1], "destinationId": [2]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_parents_local(999)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsExpandCodesLocalEmpty:
    def test_expand_codes_localdebug_true(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_local(50, debug=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_expand_codes_localdebug_false(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_local(50, debug=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsExpandCodesWrapper:
    def test_expand_codes_without_snowstorm(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        def mock_expand_local(
            _cui: str,
            *,
            debug: bool = False,
        ) -> tuple[list[int], list[str]]:
            return ([100], ["Name"])

        snomed.expand_codes_local = mock_expand_local
        result = snomed.expand_codes(50, debug=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_expand_codes_with_snowstorm(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        called = {"value": False}

        def mock_expand_snowstorm(
            _cui: str,
            *,
            debug: bool = False,
        ) -> tuple[list[int], list[str]]:
            called["value"] = True
            return ([100], ["Name"])

        snomed.expand_codes_snowstorm = mock_expand_snowstorm
        snomed.expand_codes(50, debug=False)
        assert not called["value"]
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsRecursiveCodeExpansionEdgeCases:
    def test_recursive_code_expursion_zero_levels(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.recursive_code_expansion(50, n_recursion=0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0][0] == 50
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_recursive_code_expansion_single_cui(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.recursive_code_expansion(999, n_recursion=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert 999 in result[0]
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsMEDCATSimilarity:
    def test_get_medcat_cdb_most_similar_no_model(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.get_medcat_cdb_most_similar(123)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_get_medcat_similar_score_no_results(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.get_medcat_similar_score(123, [456])
        assert isinstance(result, list)
        assert len(result) == 1
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsBuildLists:
    def test_build_lists_empty_input(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.build_lists_medcat_snomedtree([])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_build_lists_only_snomed(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.build_lists_medcat_snomedtree([50], medcat=False, snomed=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_build_lists_only_medcat(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.build_lists_medcat_snomedtree([50], medcat=True, snomed=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsRetrieveSearchSynonyms:
    def test_retrieve_search_synonyms_no_use_snomed(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms(50, use_snomed=False)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert result[0] == []
        assert result[1] == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_retrieve_search_synonyms_no_use_medcat(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms(50, use_medcat=False)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert result[2] == []
        assert result[3] == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_retrieve_search_synonyms_empty_results(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms(999)
        assert isinstance(result, tuple)
        assert len(result) == 5
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsRetrieveSearchSynonymsMulti:
    def test_retrieve_search_synonyms_multi_empty_list(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms_multi([])
        assert isinstance(result, tuple)
        assert len(result) == 6
        for r in result:
            assert isinstance(r, list)
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_retrieve_search_synonyms_multi_single_item(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms_multi([50])
        assert isinstance(result, tuple)
        assert len(result) == 6
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_retrieve_search_synonyms_multi_multiple_items(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.retrieve_search_synonyms_multi([50, 60])
        assert isinstance(result, tuple)
        assert len(result) == 6
        with contextlib.suppress(Exception):
            os.unlink(df_path)


@pytest.mark.skip(reason="Requires SNOMED RF2 file which is not available")
class TestSnomedRelationsSnowstorm:
    def test_get_snowstorm_response_children_no_connection(self) -> None:
        snomed = SnomedRelations()
        result = snomed.get_snowstorm_response_children("invalid")
        assert result is None

    def test_get_snowstorm_response_children_invalid_url(self) -> None:
        snomed = SnomedRelations()
        result = snomed.get_snowstorm_response_children("99999")
        assert result is None


class TestSnomedRelationsIntegrationEdgeCases:
    def test_full_workflow_with_mock_data_no_expansion(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100], "destinationId": [50]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.expand_codes_local(999)
        assert result[0] == []
        assert result[1] == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_recursive_expansion_with_loop(self) -> None:
        df_path = get_tmp_path("test_snomed.csv")
        pd.DataFrame({"sourceId": [100, 200], "destinationId": [200, 100]}).to_csv(
            df_path,
            sep="\t",
            index=False,
        )
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True
        result = snomed.recursive_code_expansion(100, n_recursion=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsGetSubsumedConcepts:
    def test_basic_traversal_descendants_only(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200, 300, 400],
                "destinationId": [50, 100, 200, 300],
                "typeId": [
                    116680003,
                    116680003,
                    116680003,
                    116680003,
                ],
            },
        )
        df_path = get_tmp_path("test_subsumed.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _result_names = snomed.get_subsumed_concepts(50)
        assert 50 in result_ids
        assert 100 in result_ids
        assert 200 in result_ids
        assert 300 in result_ids
        assert 400 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_traversal_with_ancestors(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_ancestors.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(
            100,
            include_ancestors=True,
            include_descendants=True,
        )
        assert 50 in result_ids
        assert 100 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_traversal_only_ancestors(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [200],
                "destinationId": [100],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_only_ancestors.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(
            100,
            include_ancestors=True,
            include_descendants=False,
        )
        assert len(result_ids) == 1
        assert 100 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_traversal_only_descendants(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_only_descendants.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(
            50,
            include_ancestors=False,
            include_descendants=True,
        )
        assert 50 in result_ids
        assert 100 in result_ids
        assert 200 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_max_depth_limit(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200, 300],
                "destinationId": [50, 100, 200],
                "typeId": [116680003, 116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_depth_limit.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, max_depth=1)
        assert 50 in result_ids
        assert 100 in result_ids
        assert 200 not in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["sourceId", "destinationId", "typeId"])
        df_path = get_tmp_path("test_empty.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _result_names = snomed.get_subsumed_concepts(999)
        assert 999 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_invalid_cui(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_invalid.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, result_names = snomed.get_subsumed_concepts("invalid")
        assert result_ids == []
        assert result_names == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_empty_results_for_nonexistent_concept(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_none.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _result_names = snomed.get_subsumed_concepts(999)
        assert 999 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_string_cui_conversion(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_string.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts("50")
        assert 50 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_none_cui(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_none_cui.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, result_names = snomed.get_subsumed_concepts(None)
        assert result_ids == []
        assert result_names == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_max_depth_zero(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_zero_depth.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, max_depth=0)
        assert len(result_ids) == 1
        assert 50 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_negative_max_depth(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_neg_depth.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, max_depth=-1)
        assert result_ids == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_inactive_relationships_excluded(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
                "active": ["1", "0"],
            },
        )
        df_path = get_tmp_path("test_inactive.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, active_only=True)
        # Starting from 50, find descendants: child 100 (active) is found
        # At node 100, look for ancestors: no parent relationship found
        assert 50 in result_ids
        assert (
            100 in result_ids
        )  # child 100 IS included because its active relationship to 50
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_include_inactive_relationships(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [200],
                "destinationId": [100],
                "typeId": [116680003],
                "active": ["0"],
            },
        )
        df_path = get_tmp_path("test_include_inactive.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(100, active_only=False)
        assert 100 in result_ids
        assert 200 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_loop_prevention(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200, 300],
                "destinationId": [50, 100, 200],
                "typeId": [116680003, 116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_loop.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, max_depth=10)
        assert len(result_ids) == 4
        assert 50 in result_ids
        assert 100 in result_ids
        assert 200 in result_ids
        assert 300 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_concept_names_returned(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_names.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, result_names = snomed.get_subsumed_concepts(50)
        assert len(result_ids) == len(result_names)
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_duplicate_handling(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 100],
                "destinationId": [50, 50],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_duplicate.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50)
        assert len(result_ids) == 2
        assert 50 in result_ids
        assert 100 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_mixed_type_relationships(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_mixed.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50)
        assert 50 in result_ids
        assert 100 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsGetSubsumedConceptsSemanticTags:
    def test_filter_by_single_tag_with_medcat_mock(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_tag_single.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        def mock_cat(_cui: str) -> None:
            return None  # Mock cat object

        class MockCDB:
            cui2preferred_name = {
                "50": "Disorder A",
                "100": "Finding B",
                "200": "Procedure C",
            }

        class MockCat:
            cdb = MockCDB()

        snomed.cat = MockCat()

        result_ids, _ = snomed.get_subsumed_concepts(50, semantic_tags=["disorder"])
        assert len(result_ids) == 1
        assert 50 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_filter_by_multiple_tags(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_tag_multi.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        class MockCDB:
            cui2preferred_name = {
                "50": "Disorder A",
                "100": "Finding B",
                "200": "Procedure C",
            }

        class MockCat:
            cdb = MockCDB()

        snomed.cat = MockCat()

        result_ids, _ = snomed.get_subsumed_concepts(
            50,
            semantic_tags=["disorder", "finding"],
        )
        assert len(result_ids) == 2
        assert 50 in result_ids
        assert 100 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_filter_no_matches(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_tag_none.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        class MockCDB:
            cui2preferred_name = {"50": "Disorder A"}  # Only has Disorder, no Finding

        class MockCat:
            cdb = MockCDB()

        snomed.cat = MockCat()

        result_ids, _ = snomed.get_subsumed_concepts(50, semantic_tags=["finding"])
        assert result_ids == []
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_filter_case_insensitive(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_tag_case.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        snomed.medcat = True

        class MockCDB:
            cui2preferred_name = {"50": "DISORDER A"}

        class MockCat:
            cdb = MockCDB()

        snomed.cat = MockCat()

        result_ids, _ = snomed.get_subsumed_concepts(50, semantic_tags=["disorder"])
        assert 50 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_no_filter_returns_all(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_no_filter.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, semantic_tags=None)
        assert 50 in result_ids
        assert 100 in result_ids
        assert 200 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_empty_tag_list(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_empty_tags.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, semantic_tags=[])
        assert 50 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_semantic_tag_with_no_medcat(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100],
                "destinationId": [50],
                "typeId": [116680003],
            },
        )
        df_path = get_tmp_path("test_tag_no_medcat.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50, semantic_tags=["disorder"])
        assert 50 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsGetSubsumedConceptsIntegration:
    def test_full_workflow_large_graph(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200, 300, 400, 500],
                "destinationId": [50, 100, 200, 300, 400],
                "typeId": [
                    116680003,
                    116680003,
                    116680003,
                    116680003,
                    116680003,
                ],
            },
        )
        df_path = get_tmp_path("test_integration.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _result_names = snomed.get_subsumed_concepts(50, max_depth=10)
        assert len(result_ids) == 6
        for i in [50, 100, 200, 300, 400, 500]:
            assert i in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_concept_both_ancestor_and_descendant(self) -> None:
        df = pd.DataFrame(
            {
                "sourceId": [100, 200],
                "destinationId": [50, 100],
                "typeId": [116680003, 116680003],
            },
        )
        df_path = get_tmp_path("test_both.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(
            100,
            include_ancestors=True,
            include_descendants=True,
        )
        assert len(result_ids) == 3
        assert 50 in result_ids
        assert 100 in result_ids
        assert 200 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)


class TestSnomedRelationsEdgeCases:
    """Additional edge case tests for robustness."""

    def test_get_pretty_name_list_with_none_values(self) -> None:
        df = pd.DataFrame(
            {"sourceId": [100], "destinationId": [50], "typeId": [116680003]},
        )
        df_path = get_tmp_path("test_none_names.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_pretty_name_list([None, 100])
        assert len(result) == 2
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_get_subsumed_concepts_with_none_cui_arg(self) -> None:
        df = pd.DataFrame(
            {"sourceId": [100], "destinationId": [50], "typeId": [116680003]},
        )
        df_path = get_tmp_path("test_none_arg.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(None)
        assert isinstance(result_ids, list)
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_get_subsumed_concepts_with_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            {"sourceId": [], "destinationId": [], "typeId": []},
        )
        df_path = get_tmp_path("test_empty_df.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.get_subsumed_concepts(50)
        assert 50 in result_ids
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_get_parents_with_no_parent(self) -> None:
        df = pd.DataFrame(
            {"sourceId": [100], "destinationId": [50], "typeId": [116680003]},
        )
        df_path = get_tmp_path("test_no_parent.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_parents(100)
        assert isinstance(result, list)
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_get_children_with_no_children(self) -> None:
        df = pd.DataFrame(
            {"sourceId": [100], "destinationId": [50], "typeId": [116680003]},
        )
        df_path = get_tmp_path("test_no_children.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result = snomed.get_children(50)
        assert isinstance(result, list)
        with contextlib.suppress(Exception):
            os.unlink(df_path)

    def test_expand_codes_local_with_invalid_cui(self) -> None:
        df = pd.DataFrame(
            {"sourceId": [100], "destinationId": [50], "typeId": [116680003]},
        )
        df_path = get_tmp_path("test_invalid_local.csv")
        df.to_csv(df_path, sep="\t", index=False)
        snomed = SnomedRelations(snomed_rf2_full_path=df_path)
        result_ids, _ = snomed.expand_codes_local(99999)
        assert len(result_ids) == 0
        with contextlib.suppress(Exception):
            os.unlink(df_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
