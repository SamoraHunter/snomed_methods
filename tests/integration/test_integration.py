import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.snomed_methods.snomed_methods_v1 import SnomedRelations


@pytest.fixture
def sample_relationships_df():
    """Create a temporary DataFrame with sample SNOMED CT relationships."""
    return pd.DataFrame(
        {
            "sourceId": [
                100001,
                100002,
                100003,
                100004,
                100005,
                100006,
                100007,
            ],
            "destinationId": [
                100000,
                100000,
                100001,
                100001,
                100002,
                100004,
                100006,
            ],
        }
    )


@pytest.fixture
def snomed_relations_instance(sample_relationships_df):
    """Create a SnomedRelations instance with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = os.path.join(tmpdir, "sample_relationships.tsv")
        sample_relationships_df.to_csv(temp_file, sep="\t", index=False)

        snomed_rel = SnomedRelations(snomed_rf2_full_path=temp_file)
        yield snomed_rel


class TestGetChildren:
    """Tests for the get_children method."""

    def test_get_children_returns_direct_children(self, snomed_relations_instance):
        """Test that get_children returns direct children of a parent node."""

        # Node 100000 has children 100001 and 100002
        children = snomed_relations_instance.get_children(100000)

        assert set(children) == {100001, 100002}

    def test_get_children_no_children(self, snomed_relations_instance):
        """Test that get_children returns empty list for nodes with no children."""
        # Node 100007 has no children (no row has destinationId == 100007)
        children = snomed_relations_instance.get_children(100007)

        assert children == []

    def test_get_children_invalid_cui(self, snomed_relations_instance):
        """Test that get_children handles invalid CUI gracefully."""
        children = snomed_relations_instance.get_children("invalid")

        assert children == []


class TestGetParents:
    """Tests for the get_parents method."""

    def test_get_parents_returns_direct_parents(self, snomed_relations_instance):
        """Test that get_parents returns direct parents of a child node."""
        # Node 100001 has parent 100000
        parents = snomed_relations_instance.get_parents(100001)

        assert parents == [100000]

    def test_get_parents_multiple_parents(self, snomed_relations_instance):
        """Test that get_parents returns all parents for a node."""
        # Node 100002 has parent 100000 (row: sourceId=100002, destinationId=100000)
        parents = snomed_relations_instance.get_parents(100002)

        assert parents == [100000]

    def test_get_parents_no_parents(self, snomed_relations_instance):
        """Test that get_parents returns empty list for nodes with no parents."""
        # Node 99999 has no parent (no row has sourceId == 99999)
        parents = snomed_relations_instance.get_parents(99999)

        assert parents == []

    def test_get_parents_invalid_cui(self, snomed_relations_instance):
        """Test that get_parents handles invalid CUI gracefully."""
        parents = snomed_relations_instance.get_parents("invalid")

        assert parents == []


class TestExpandCodesLocal:
    """Tests for the expand_codes_local method."""

    def test_expand_codes_local_returns_tuple(self, snomed_relations_instance):
        """Test that expand_codes_local returns a tuple of (codes, names)."""
        codes, names = snomed_relations_instance.expand_codes_local(100000)

        assert isinstance(codes, list)
        assert isinstance(names, list)

    def test_expand_codes_local_with_children(self, snomed_relations_instance):
        """Test expand_codes_local retrieves children."""
        # 100000 has children: 100001, 100002
        codes, names = snomed_relations_instance.expand_codes_local(100000)

        assert set(codes) == {100001, 100002}
        # Names will be empty since medcat is not enabled

    def test_expand_codes_local_with_parents(self, snomed_relations_instance):
        """Test expand_codes_local retrieves parents."""
        # 100001 has parent: 100000 and children: 100003, 100004
        codes, names = snomed_relations_instance.expand_codes_local(100001)

        assert set(codes) == {100000, 100003, 100004}
        assert len(names) == 0

    def test_expand_codes_local_no_results(self, snomed_relations_instance):
        """Test expand_codes_local with node having no parents or children."""
        codes, names = snomed_relations_instance.expand_codes_local(999999)

        assert codes == []
        assert names == []

    def test_expand_codes_local_removes_duplicates(self, snomed_relations_instance):
        """Test that duplicate codes are removed in expand_codes_local."""
        # 100003 has parent 100000
        # 100004 has parent 100003 and child 100005
        # Expanding from 100003 should not have duplicates
        codes, names = snomed_relations_instance.expand_codes_local(100003)

        assert len(codes) == len(set(codes))


class TestExpandCodesParentsLocal:
    """Tests for the expand_codes_parents_local method."""

    def test_expand_codes_parents_local_returns_tuple(self, snomed_relations_instance):
        """Test that expand_codes_parents_local returns a tuple of (codes, names)."""
        codes, names = snomed_relations_instance.expand_codes_parents_local(100001)

        assert isinstance(codes, list)
        assert isinstance(names, list)

    def test_expand_codes_parents_local_retrieves_parents(
        self, snomed_relations_instance
    ):
        """Test that expand_codes_parents_local retrieves parent codes."""
        codes, names = snomed_relations_instance.expand_codes_parents_local(100001)

        assert codes == [100000]
        assert len(names) == 0


class TestExpandCodesChildrenLocal:
    """Tests for the expand_codes_children_local method."""

    def test_expand_codes_children_local_returns_tuple(self, snomed_relations_instance):
        """Test that expand_codes_children_local returns a tuple of (codes, names)."""
        codes, names = snomed_relations_instance.expand_codes_children_local(100000)

        assert isinstance(codes, list)
        assert isinstance(names, list)

    def test_expand_codes_children_local_retrieves_children(
        self, snomed_relations_instance
    ):
        """Test that expand_codes_children_local retrieves child codes."""
        codes, names = snomed_relations_instance.expand_codes_children_local(100000)

        assert set(codes) == {100001, 100002}
        assert len(names) == 0


class TestIntegration:
    """Integration tests for the main workflow."""

    def test_full_workflow_with_sample_data(self, snomed_relations_instance):
        """Test complete integration flow with sample data."""
        # Define a root node
        root_cui = 100000

        # Expand codes locally from root
        all_codes, all_names = snomed_relations_instance.expand_codes_local(root_cui)

        # Verify we got children of root (100001 and 100002)
        assert 100001 in all_codes
        assert 100002 in all_codes

        # Verify no duplicates
        assert len(all_codes) == len(set(all_codes))

        # Test the expand_codes wrapper method
        all_codes_2, all_names_2 = snomed_relations_instance.expand_codes(root_cui)

        assert set(all_codes) == set(all_codes_2)

    def test_recursive_code_expansion_structure(self, snomed_relations_instance):
        """Test recursive code expansion returns expected structure."""
        root_cui = 100000

        # This will use expand_codes_local internally
        all_codes, all_names = snomed_relations_instance.recursive_code_expansion(
            root_cui, n_recursion=2
        )

        assert isinstance(all_codes, list)
        assert isinstance(all_names, list)

        # All codes should be integers
        for code in all_codes:
            assert isinstance(code, int)


@pytest.fixture
def sample_csv_file(sample_relationships_df):
    """Create a temporary CSV file with sample relationships."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / "sample_relationships.csv"
        # Save as TSV since the module expects tab-separated
        sample_relationships_df.to_csv(temp_file, sep="\t", index=False)
        yield str(temp_file)


class TestCSVLoading:
    """Tests for CSV file loading functionality."""

    def test_load_from_csv_path(self, sample_csv_file):
        """Test that SnomedRelations can load data from a CSV/TSV file."""
        snomed_rel = SnomedRelations(snomed_rf2_full_path=sample_csv_file)

        children = snomed_rel.get_children(100000)

        assert set(children) == {100001, 100002}

    def test_invalid_file_path(self):
        """Test handling of non-existent file path."""
        with pytest.raises(FileNotFoundError):
            SnomedRelations(snomed_rf2_full_path="/nonexistent/file.tsv")
