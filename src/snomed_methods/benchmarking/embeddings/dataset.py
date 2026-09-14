"""Dataset generation and loading for embedding semantic similarity benchmarking."""

import random
from typing import List, Optional

try:
    from datasets import DatasetDict
except ImportError:

    class DatasetDict:
        pass


def generate_embedding_dataset(  # noqa: C901
    num_samples: int = 100,
    snomed_concepts: Optional[List[str]] = None,
) -> List[dict]:
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

    if len(snomed_concepts) < 2:
        snomed_concepts = [str(100000 + i) for i in range(num_samples)]

    # Group related concepts by semantic category (disease groups)
    disease_groups = [
        ["73211009", "237550006", "237600004"],  # Diabetes-related
        ["38341003", "59621000"],  # Hypertension-related
        ["37796005"],  # Migraine (single concept)
        ["233604007"],  # Pneumonia (single concept)
        ["195967001"],  # Asthma (single concept)
    ]

    results = []
    group_pairs = []

    # Generate similar pairs (same disease group)
    for group in disease_groups:
        if len(group) >= 2:
            for i in range(len(group)):
                for j in range(i + 1, min(len(group), i + 3)):
                    group_pairs.append((group[i], group[j], True))

    # Generate dissimilar pairs (different disease groups)
    dissimilar_count = num_samples - len(group_pairs)
    if dissimilar_count > 0:
        all_concepts = [c for group in disease_groups for c in group]
        attempted = 0
        while len(group_pairs) < num_samples and attempted < dissimilar_count * 10:
            c1, c2 = random.sample(all_concepts, 2)
            c1_group = next((g for g in disease_groups if c1 in g), None)
            c2_group = next((g for g in disease_groups if c2 in g), None)
            if c1_group != c2_group and (c1, c2, False) not in group_pairs:
                group_pairs.append((c1, c2, False))
            attempted += 1

    # Create result samples
    for i in range(min(num_samples, len(group_pairs))):
        concept_1, concept_2, is_similar = group_pairs[i]

        if is_similar:
            similarity_label = "high"
        else:
            similarity_label = "low"

        results.append(
            {
                "concept_1": concept_1,
                "concept_2": concept_2,
                "is_similar": is_similar,
                "similarity_label": similarity_label,
            }
        )

    return results


def load_embedding_datasets(cache_dir: Optional[str] = None) -> dict:
    """Load pre-generated embedding datasets or generate new ones.

    Args:
        cache_dir: Optional directory to cache generated data

    Returns:
        Dict with dataset names as keys and lists of samples
    """
    import os

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "cache", "embedding_datasets"
        )

    os.makedirs(cache_dir, exist_ok=True)

    return {
        "small": generate_embedding_dataset(num_samples=25),
        "medium": generate_embedding_dataset(num_samples=100),
        "large": generate_embedding_dataset(num_samples=500),
    }


def create_pairs_from_umnsrs(umnsrs_pairs: List[dict]) -> List[dict]:
    """Convert UMNSRS pairs to embedding benchmark format.

    Args:
        umnsrs_pairs: List of UMNSRS pairs with 'text_1', 'text_2', 'label'

    Returns:
        List of dicts with concept pair info
    """
    results = []
    for pair in umnsrs_pairs:
        label = float(pair.get("label", 500))
        is_similar = label >= 500  # Threshold for similarity

        results.append(
            {
                "concept_1": pair.get("text_1", ""),
                "concept_2": pair.get("text_2", ""),
                "is_similar": is_similar,
                "similarity_label": "high" if is_similar else "low",
                "umnsrs_score": label,
            }
        )

    return results
