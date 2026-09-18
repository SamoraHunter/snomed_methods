# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Dataset generation and loading for embedding semantic similarity benchmarking."""

from __future__ import annotations

import pathlib
import random
from pathlib import Path

try:
    from datasets import DatasetDict
except ImportError:

    class DatasetDict:
        pass


MIN_GROUP_SIZE = 2
SIMILARITY_THRESHOLD = 500
SMALL_SIZE = 25
MEDIUM_SIZE = 100
LARGE_SIZE = 500

CACHE_DIR = Path(__file__).parent.parent / "cache"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embedding_datasets"


def generate_embedding_dataset(
    num_samples: int = 100,
    snomed_concepts: list[str] | None = None,
) -> list[dict]:
    """Generate synthetic embedding dataset for semantic similarity benchmarking.

    Creates concept pairs with known relationships (similar/dissimilar) to evaluate
    embedding quality. Similar concepts share semantic categories or hierarchy
    relationships.

    Args:
        num_samples: Number of concept pairs to generate
        snomed_concepts: Optional list of SNOMED CUIs to use

    Returns:
        List of dicts with keys: 'concept_1', 'concept_2', 'is_similar',
        'similarity_label'

    """
    if snomed_concepts is None:
        # Use real SNOMED CT concepts for benchmarking
        snomed_concepts = [
            "73211009",  # Diabetes mellitus
            "237550006",  # Type 2 diabetes
            "237600004",  # Impaired glucose tolerance
            "38341003",  # Hypertension
            "59621000",  # Diastolic hypertension
            "37796005",  # Migraine
            "233604007",  # Pneumonia
            "195967001",  # Asthma
            "230690007",  # Osteoporosis
            "34825004",  # Arthritis
        ]

    if len(snomed_concepts) < MIN_GROUP_SIZE:
        snomed_concepts = [str(100000 + i) for i in range(num_samples)]

    # Group related concepts by semantic category (disease groups)
    disease_groups: list[list[str]] = []

    # First, add any concepts that aren't in the default disease groups
    default_disease_concepts = {
        "73211009",
        "237550006",
        "237600004",
        "38341003",
        "59621000",
    }

    # Identify concepts not in default disease groups
    disease_groups.extend(
        [concept]
        for concept in snomed_concepts
        if concept not in default_disease_concepts
    )

    # Add default disease groups (for real SNOMED concepts)
    disease_groups.extend(
        [
            ["73211009", "237550006", "237600004"],  # Diabetes-related
            ["38341003", "59621000"],  # Hypertension-related
        ],
    )

    results = []
    group_pairs: list[tuple[str, str, bool]] = []

    # Generate similar pairs (same disease group)
    for group in disease_groups:
        if len(group) >= MIN_GROUP_SIZE:
            pairs = [
                (group[i], group[j], True)
                for i in range(len(group))
                for j in range(i + 1, min(len(group), i + 3))
            ]
            group_pairs.extend(pairs)

    # Generate dissimilar pairs (different disease groups)
    dissimilar_count = num_samples - len(group_pairs)
    if dissimilar_count > 0:
        all_concepts = [c for group in disease_groups for c in group]

        # If we can't generate enough unique dissimilar pairs, add synthetic concepts
        max_unique_pairs = len(all_concepts) * (len(all_concepts) - 1) // 2
        if max_unique_pairs < num_samples:
            additional_needed = num_samples - max_unique_pairs
            start_id = 100000 + len(snomed_concepts)
            synthetic_concepts = [str(start_id + i) for i in range(additional_needed)]
            all_concepts.extend(synthetic_concepts)

            # Add single-concept groups for synthetic concepts
            disease_groups.extend([syn_cui] for syn_cui in synthetic_concepts)

        attempted = 0
        seen_pairs: set[tuple[str, str]] = set()
        while (
            len(group_pairs) < num_samples
            and attempted < dissimilar_count * MIN_GROUP_SIZE
        ):
            c1, c2 = random.sample(all_concepts, 2)

            pair_key = tuple(sorted([c1, c2]))
            if pair_key in seen_pairs:
                attempted += 1
                continue

            c1_group = next((g for g in disease_groups if c1 in g), None)
            c2_group = next((g for g in disease_groups if c2 in g), None)

            is_same_group = (c1_group is not None) and (c1_group == c2_group)
            if is_same_group:
                attempted += 1
                continue

            seen_pairs.add(pair_key)
            group_pairs.append((c1, c2, False))
            attempted += 1

    # Create result samples
    for i in range(min(num_samples, len(group_pairs))):
        concept_1, concept_2, is_similar = group_pairs[i]

        similarity_label = "high" if is_similar else "low"

        results.append(
            {
                "concept_1": concept_1,
                "concept_2": concept_2,
                "is_similar": is_similar,
                "similarity_label": similarity_label,
            },
        )

    return results


def load_embedding_datasets(cache_dir: str | None = None) -> dict:
    """Load pre-generated embedding datasets or generate new ones.

    Args:
        cache_dir: Optional directory to cache generated data

    Returns:
        Dict with dataset names as keys and lists of samples

    """
    if cache_dir is None:
        cache_dir = EMBEDDING_CACHE_DIR

    pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)

    return {
        "small": generate_embedding_dataset(num_samples=SMALL_SIZE),
        "medium": generate_embedding_dataset(num_samples=MEDIUM_SIZE),
        "large": generate_embedding_dataset(num_samples=LARGE_SIZE),
    }


def create_pairs_from_umnsrs(umnsrs_pairs: list[dict]) -> list[dict]:
    """Convert UMNSRS pairs to embedding benchmark format.

    Args:
        umnsrs_pairs: List of UMNSRS pairs with 'text_1', 'text_2', 'label'

    Returns:
        List of dicts with concept pair info

    """
    results = []
    for pair in umnsrs_pairs:
        label = float(pair.get("label", SIMILARITY_THRESHOLD))
        is_similar = label >= SIMILARITY_THRESHOLD

        results.append(
            {
                "concept_1": pair.get("text_1", ""),
                "concept_2": pair.get("text_2", ""),
                "is_similar": is_similar,
                "similarity_label": "high" if is_similar else "low",
                "umnsrs_score": label,
            },
        )

    return results
