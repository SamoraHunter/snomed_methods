# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Mock/example implementations of benchmark methods for testing purposes.

This module provides simple placeholder functions that can be used to test
benchmark infrastructure without requiring real classifier/lookup implementations.
"""

from __future__ import annotations

import random

MAX_TARGET_CODES = 5
COMMON_DIGITS_THRESHOLD = 5
CHILDREN_THRESHOLD = 3
PARENTS_THRESHOLD = 2
MAX_APPROXIMATE_CUIS = 100


def mock_annotator(text: str) -> list[str]:
    """Mock annotator returning synthetic results based on text content.

    Args:
        text: Input clinical text to annotate

    Returns:
        List of predicted CUIs

    """
    text_lower = text.lower()

    cui_mapping = {
        "diabetes": ["C001", "C002", "C003"],
        "hypertension": ["C010", "C011"],
        "depression": ["C020", "C021", "C022"],
        "migraine": ["C030"],
        "pneumonia": ["C040", "C041"],
        "asthma": ["C050", "C051"],
    }

    predicted_cuis = []
    for keyword, cuis in cui_mapping.items():
        if keyword in text_lower:
            predicted_cuis.extend(cuis)

    predicted_cuis.extend(["C999", "C998", "C997"])

    return predicted_cuis[:10]


def mock_term_lookup(term: str) -> list[tuple[str, str]]:
    """Mock term lookup returning synthetic (CUI, matched_term) tuples.

    Args:
        term: Search term to look up

    Returns:
        List of (cui, matched_term) tuples

    """
    text_lower = term.lower()

    cui_mapping = {
        "diabetes": ["73211009", "237550006", "237600004"],
        "hypertension": ["38341003", "59621000"],
        "migraine": ["37796005"],
        "pneumonia": ["233604007"],
        "asthma": ["195967001"],
        "osteoporosis": ["230690007"],
        "arthritis": ["34825004"],
        "hyperlipidemia": ["414015000"],
        "depression": ["35489007", "228450005"],
        "chronic kidney disease": ["236422002"],
    }

    matched_cuis = []
    for keyword, cuis in cui_mapping.items():
        if keyword in text_lower:
            matched_cuis.extend(cuis)

    if not matched_cuis:
        seed_hash = hash(term)
        matched_cuis = [
            str(abs(seed_hash + i * 1000) % 1000000000).zfill(9) for i in range(15)
        ]

    return [(cui, f"Matched_{i}") for i, cui in enumerate(matched_cuis[:15])]


def simple_mapper(snomed_cui: str, mappings: dict[str, list[str]]) -> list[str]:
    """Mapper using direct lookup via concept mapping.

    Args:
        snomed_cui: SNOMED concept ID to map
        mappings: Dict of actual SNOMED to target code mappings

    Returns:
        List of mapped code strings (ICD and LOINC)

    """
    return mappings.get(snomed_cui, [])


def approximate_mapper(
    snomed_cui: str,
    mappings: dict[str, list[str]],
    all_cuis: list[str] | None = None,
) -> list[str]:
    """Mapper using approximate/string-based matching for SNOMED concepts.

    This method uses string similarity and partial matching to find mappings
    when exact lookup fails. It's designed to be DIFFERENT from ground truth
    extraction (which is direct lookup).

    Args:
        snomed_cui: SNOMED concept ID to map
        mappings: Dict of actual SNOMED to target code mappings
        all_cuis: Optional list of all available concept IDs

    Returns:
        List of mapped code strings

    """
    if snomed_cui in mappings:
        return mappings[snomed_cui]

    if not all_cuis:
        return []

    target_codes = []

    for cui in all_cuis[:MAX_APPROXIMATE_CUIS]:
        if len(target_codes) >= MAX_TARGET_CODES:
            break
        if cui == snomed_cui:
            continue

        common_digits = sum(1 for c1, c2 in zip(str(snomed_cui), str(cui)) if c1 == c2)
        prefix_match = str(snomed_cui).startswith(str(cui)[:3]) or str(cui).startswith(
            str(snomed_cui)[:3],
        )

        if common_digits >= COMMON_DIGITS_THRESHOLD or prefix_match:
            cui_mappings = mappings.get(cui, [])
            target_codes.extend(cui_mappings)

    return list(dict.fromkeys(target_codes))[:MAX_TARGET_CODES]


def random_mapper(
    snomed_cui: str,
    mappings: dict[str, list[str]],
    all_targets: list[str] | None = None,
) -> list[str]:
    """Mapper using random sampling of available target codes.

    This method randomly samples from all available mapping targets.
    It's a baseline that should perform poorly on benchmarks.

    Args:
        snomed_cui: SNOMED concept ID to map
        mappings: Dict of actual SNOMED to target code mappings
        all_targets: Optional list of all possible target codes

    Returns:
        List of randomly sampled mapped code strings

    """

    if all_targets is None:
        all_targets = []
        for targets in mappings.values():
            all_targets.extend(targets)
        all_targets = list(set(all_targets))

    if not all_targets:
        return []

    random.seed(hash(snomed_cui) % (2**32))
    num_return = min(5, len(all_targets))

    return random.sample(all_targets, num_return)


def mock_hierarchy_expansion(seed_cui: str) -> list[str]:
    """Mock hierarchy expansion returning synthetic related concepts.

    Args:
        seed_cui: Seed concept ID to expand from

    Returns:
        List of related concept IDs

    """
    # Use the SAME logic as generate_hierarchy_dataset for consistency
    seed_cuis = ["123456789", "987654321", "111222333", "444555666", "777888999"]

    children = []
    parents = []

    for idx, other in enumerate(seed_cuis):
        if seed_cui == other:
            continue
        diff = abs(hash(seed_cui + "!")) - abs(hash(other + "@"))
        if diff % 7 < CHILDREN_THRESHOLD:
            children.append(str(abs(hash(seed_cui + f"child{idx}")))[:9])
        elif diff % 5 < PARENTS_THRESHOLD:
            parents.append(str(abs(hash(seed_cui + f"parent{idx}")))[:9])

    expected_related = children[:5] + parents[:3]

    # Return expected related first, then additional random concepts for ranking
    all_concepts = expected_related + [
        str(abs(hash(seed_cui + f"extra{i}")))[:9] for i in range(15)
    ]

    return all_concepts[:20]


def mock_similarity(text1: str, text2: str) -> float:
    """Mock similarity function returning hash-based similarity score.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score (non-negative number)

    """
    combined = hash(text1 + text2)
    return float(abs(combined) % 1000)


def get_mock_methods(
    mappings: dict[str, list[str]] | None = None,
    all_cuis: list[str] | None = None,
) -> dict:
    """Get all mock methods as a dict for easy access.

    Args:
        mappings: Optional pre-loaded SNOMED to target code mappings
        all_cuis: Optional list of all available concept IDs

    Returns:
        Dict mapping method names to their functions

    """
    if mappings is None:
        mappings = {}

    return {
        "annotator": mock_annotator,
        "term_lookup": mock_term_lookup,
        "mapper_simple": lambda cui: simple_mapper(cui, mappings),
        "mapper_approximate": lambda cui: approximate_mapper(cui, mappings, all_cuis),
        "mapper_random": lambda cui: random_mapper(cui, mappings),
        "hierarchy_expansion": mock_hierarchy_expansion,
        "similarity": mock_similarity,
    }


def mock_mapper(snomed_cui: str) -> list[str]:
    """Mock mapper returning synthetic ICD-10 and LOINC codes.

    Args:
        snomed_cui: SNOMED concept ID to map

    Returns:
        List of mapped code strings (ICD and LOINC)

    """
    cui_to_codes = {
        "237550006": ["E11.9", "250.00", "LOINC_4544-3"],
        "237600004": ["E11.9", "250.01", "LOINC_17856-6"],
        "38341003": ["I10", "401.9", "LOINC_8867-4"],
        "59621000": ["I10", "401.1", "LOINC_8867-4"],
        "35489007": ["F33.9", "296.30", "LOINC_4544-3"],
        "228450005": ["F33.2", "296.33", "LOINC_24725-5"],
    }

    if snomed_cui in cui_to_codes:
        return cui_to_codes[snomed_cui]

    seed_hash = hash(snomed_cui)
    icd_codes = [
        f"ICD_E{abs(seed_hash + 1) % 99:02d}.{abs(seed_hash + 2) % 9}" for _ in range(3)
    ]
    loinc_codes = [f"LOINC_{718 + abs(seed_hash) % 100}" for _ in range(2)]

    return icd_codes + loinc_codes[:5]
