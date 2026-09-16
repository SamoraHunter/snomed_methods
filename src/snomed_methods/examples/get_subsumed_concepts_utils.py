#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""
Utility functions for working with get_subsumed_concepts()

This module wraps the SNOMED CT complexity to provide simple, intuitive APIs.
"""

import os


def _get_snomed_path() -> str:
    """Get the default SNOMED relationship data path."""
    # Try multiple paths relative to different locations
    possible_paths = [
        os.path.join(
            "uk_sct2cl_42.2.0",
            "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            "Full",
            "Terminology",
            "sct2_Relationship_UKCLFull_GB1000000_20260603.txt",
        ),
    ]

    # Try from different working directories (relative to module location)
    for base_dir in [
        os.getcwd(),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
        ),
    ]:
        for rel_path in possible_paths:
            path = os.path.join(base_dir, rel_path)
            if os.path.exists(path):
                return path

    # Try environment variable
    env_path = os.environ.get("SNOMED_RF2_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    raise FileNotFoundError(
        "SNOMED RF2 file not found. Set SNOMED_RF2_PATH environment variable."
    )


def _get_lookup() -> object:
    """Get a SNOMED term lookup instance."""
    from snomed_term_lookup import create_term_lookup_from_directory

    # Get the UK Clinical data directory (one level up from relationships file)
    relationship_path = _get_snomed_path()
    uk_data_dir = os.path.dirname(os.path.dirname(relationship_path))

    return create_term_lookup_from_directory(uk_data_dir)


def _get_snomed_reader() -> object:
    """Get a SNOMED relations reader instance."""
    from snomed_methods_v1 import SnomedRelations

    return SnomedRelations(snomed_rf2_full_path=_get_snomed_path(), medcat=False)


def find_concept_cui(term: str, top_n: int = 5) -> tuple[str, str]:
    """
    Find a SNOMED concept CUI by term.

    Args:
        term: The search term (e.g., "meningioma")
        top_n: Number of results to return

    Returns:
        Tuple of (cui, preferred_name)

    Raises:
        ValueError: If no concepts found for the term
    """
    lookup = _get_lookup()
    matches = lookup.find_concepts_by_term(term, top_n=top_n)

    if not matches:
        raise ValueError(f"No concepts found for term: {term}")

    return str(matches[0][0]), matches[0][1]


def get_subsumed_concepts_for_terms(
    terms: list[str],
    max_depth: int = 5,
    include_ancestors: bool = False,
    include_descendants: bool = True,
) -> dict:
    """
    Find all concepts related to a list of terms via the SNOMED hierarchy.

    Args:
        terms: List of search terms (e.g., ["meningioma"])
        max_depth: Maximum traversal depth for each term
        include_ancestors: Whether to traverse upward (default False)
        include_descendants: Whether to traverse downward (default True)

    Returns:
        Dict with keys:
            - 'results': List of (term, cui, concept_ids) tuples
            - 'all_concepts': Set of all unique concept IDs found
            - 'config': The configuration used

    Example:
        >>> result = get_subsumed_concepts_for_terms(["meningioma"], max_depth=3)
    """
    snomed = _get_snomed_reader()

    results = []
    all_concept_ids = set()

    for term in terms:
        cui, preferred_name = find_concept_cui(term, top_n=1)

        concept_ids, _ = snomed.get_subsumed_concepts(
            cui,
            max_depth=max_depth,
            include_ancestors=include_ancestors,
            include_descendants=include_descendants,
        )

        results.append((term, cui, sorted(concept_ids)))
        all_concept_ids.update(concept_ids)

    return {
        "results": results,
        "all_concepts": sorted(all_concept_ids),
        "config": {
            "terms": terms,
            "max_depth": max_depth,
            "include_ancestors": include_ancestors,
            "include_descendants": include_descendants,
        },
    }


# Backward compatible wrapper
def get_subsumed_concepts_for_term(
    term: str,
    max_depth: int = 5,
    include_ancestors: bool = False,
    include_descendants: bool = True,
) -> tuple[list[int], list[str]]:
    """
    Find all concepts related to a single term via the SNOMED hierarchy.

    Args:
        term: Search term (e.g., "meningioma")
        max_depth: Maximum traversal depth (default 5)
        include_ancestors: Whether to traverse upward (default False)
        include_descendants: Whether to traverse downward (default True)

    Returns:
        Tuple of (concept_ids, concept_names) where concept_names are from MedCAT
    """
    snomed = _get_snomed_reader()

    cui, preferred_name = find_concept_cui(term, top_n=1)

    return snomed.get_subsumed_concepts(
        cui,
        max_depth=max_depth,
        include_ancestors=include_ancestors,
        include_descendants=include_descendants,
    )
