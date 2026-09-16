# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Dataset generation and loading for hierarchy expansion benchmarking."""

from typing import List, Optional

try:
    from datasets import DatasetDict
except ImportError:

    class DatasetDict:
        pass


def generate_hierarchy_dataset(
    num_samples: int = 100,
    seed_cuis: Optional[List[str]] = None,
) -> List[dict]:
    """Generate synthetic hierarchy expansion dataset for benchmarking.

    Each sample contains a seed CUI and expected related concepts based on
    simulated hierarchical relationships (parent-child, sibling relationships).

    Args:
        num_samples: Number of samples to generate
        seed_cuis: Optional list of seed CUIs to use

    Returns:
        List of dicts with keys: 'seed_cui', 'expected_related'
    """
    if seed_cuis is None:
        seed_cuis = [
            "123456789",
            "987654321",
            "111222333",
            "444555666",
            "777888999",
        ]

    if not seed_cuis:
        seed_cuis = [str(i) for i in range(20)]

    hierarchy_relations = {}

    for cui in seed_cuis:
        children = []
        parents = []

        for idx, other in enumerate(seed_cuis):
            if cui == other:
                continue
            diff = abs(hash(cui + "!")) - abs(hash(other + "@"))
            if diff % 7 < 3:
                children.append(str(abs(hash(cui + f"child{idx}")))[:9])
            elif diff % 5 < 2:
                parents.append(str(abs(hash(cui + f"parent{idx}")))[:9])

        hierarchy_relations[cui] = {
            "children": children,
            "parents": parents,
        }

    results = []

    for i in range(num_samples):
        seed_cui = seed_cuis[i % len(seed_cuis)]
        relations = hierarchy_relations[seed_cui]

        expected_related = relations["children"][:5] + relations["parents"][:3]

        if not expected_related:
            expected_related = [f"C{i}001", f"C{i}002"]

        results.append(
            {
                "seed_cui": seed_cui,
                "expected_related": expected_related,
                "num_expected": len(expected_related),
            }
        )

    return results


def load_hierarchy_datasets(cache_dir: Optional[str] = None) -> dict:
    """Load pre-generated hierarchy datasets or generate new ones.

    Args:
        cache_dir: Optional directory to cache generated data

    Returns:
        Dict with dataset names as keys and lists of samples
    """
    import os

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "cache", "hierarchy_datasets"
        )

    os.makedirs(cache_dir, exist_ok=True)

    return {
        "small": generate_hierarchy_dataset(num_samples=25),
        "medium": generate_hierarchy_dataset(num_samples=100),
        "large": generate_hierarchy_dataset(num_samples=500),
    }
