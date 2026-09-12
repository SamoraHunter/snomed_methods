#!/usr/bin/env python3
"""
Example usage of SnomedTermLookup for finding SNOMED CT codes by term.

This script demonstrates various ways to search for SNOMED concepts using
the SnomedTermLookup class. It includes examples for basic searching,
case-insensitive matching, fuzzy string matching, batch operations, and
integration with MedCAT.

Usage:
    python example_term_lookup.py

Set environment variables to customize behavior:
    SNOMED_DIR: Path to the SNOMED CT directory (default: UK clinical snapshot)
    MEDCAT_MODEL_PATH: Path to a medcat model pack for enhanced search

The script runs all examples sequentially and logs output at INFO level.
"""

import logging
import os

from snomed_term_lookup import create_term_lookup_from_directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def example_basic_search():
    """Demonstrate basic term search with prefix matching."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )
    sct2_dir = os.environ.get("SNOMED_DIR", default_dir)

    lookup = create_term_lookup_from_directory(sct2_dir)

    results = lookup.find_concepts_by_term("meningioma", match_prefix=True, top_n=10)

    for _cui, _term in results:
        pass


def example_case_insensitive():
    """Demonstrate case-insensitive search functionality."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )
    sct2_dir = os.environ.get("SNOMED_DIR", default_dir)

    lookup = create_term_lookup_from_directory(sct2_dir)

    lookup.find_concepts_by_term("Meningioma", ignore_case=True, top_n=5)


def example_fuzzy_search():
    """Demonstrate fuzzy string matching for typo-tolerant searches."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )
    sct2_dir = os.environ.get("SNOMED_DIR", default_dir)

    lookup = create_term_lookup_from_directory(sct2_dir)

    try:
        lookup.find_concepts_by_term_fuzzy("menignoma", min_score=70, top_n=5)
    except ImportError:
        pass


def example_batch_search():
    """Demonstrate batch searching for multiple terms at once."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )
    sct2_dir = os.environ.get("SNOMED_DIR", default_dir)

    lookup = create_term_lookup_from_directory(sct2_dir)

    terms_to_search = ["meningioma", "glioma", "tumor"]

    lookup.find_concepts_batch(terms_to_search, match_prefix=True)


def example_concept_info():
    """Demonstrate how to retrieve detailed information about a specific concept."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )
    sct2_dir = os.environ.get("SNOMED_DIR", default_dir)

    lookup = create_term_lookup_from_directory(sct2_dir)

    results = lookup.find_concepts_by_term("meningioma", top_n=1)

    if results:
        _cui = results[0][0]

        lookup.getconcept_info(_cui)


def example_with_medcat():
    """Demonstrate enhanced search capabilities using MedCAT integration."""
    try:
        from medcat.cat import CAT

        medcat_path = os.environ.get("MEDCAT_MODEL_PATH", None)

        if not medcat_path:
            return

        cat = CAT.load_model_pack(medcat_path)

        cui_list = list(cat.cdb.cui2preferred_name.keys())

        sample_cui = cui_list[0] if len(cui_list) > 0 else None

        if sample_cui:
            cat.cdb.cui2preferred_name.get(sample_cui)

    except ImportError:
        pass


def main():
    """Run all example functions sequentially."""
    example_basic_search()
    example_case_insensitive()
    example_fuzzy_search()
    example_batch_search()
    example_concept_info()
    example_with_medcat()


if __name__ == "__main__":
    main()
