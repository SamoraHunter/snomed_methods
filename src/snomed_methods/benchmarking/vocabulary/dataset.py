"""Dataset generation and loading for vocabulary mapping benchmarking."""

from typing import List, Optional

try:
    from datasets import DatasetDict
except ImportError:

    class DatasetDict:
        pass


def generate_mapping_dataset(
    num_samples: int = 100,
    snomed_concepts: Optional[List[str]] = None,
) -> List[dict]:
    """Generate synthetic vocabulary mapping dataset for benchmarking.

    Each sample contains a SNOMED CUI and expected mapped codes from target
    vocabularies (ICD-10, LOINC, etc.).

    Args:
        num_samples: Number of samples to generate
        snomed_concepts: Optional list of SNOMED CUIs to use

    Returns:
        List of dicts with keys: 'snomed_cui', 'target_codes'
    """
    vocabulary_mappings = {
        "diabetes": [
            ("237550006", ["E11.9", "250.00", "LOINC_4544-3"]),
            ("237600004", ["E11.9", "250.01", "LOINC_17856-6"]),
        ],
        "hypertension": [
            ("38341003", ["I10", "401.9", "LOINC_8867-4"]),
            ("59621000", ["I10", "401.1", "LOINC_8867-4"]),
        ],
        "depression": [
            ("35489007", ["F33.9", "296.30", "LOINC_4544-3"]),
            ("228450005", ["F33.2", "296.33", "LOINC_24725-5"]),
        ],
    }

    snomed_concepts = []
    for _, pairs in vocabulary_mappings.items():
        for cui, _ in pairs:
            snomed_concepts.append(cui)
    snomed_concepts = list(set(snomed_concepts))

    if not snomed_concepts:
        snomed_concepts = [str(i) for i in range(num_samples)]

    results = []

    for i in range(num_samples):
        concept_tuple = None
        _name: str = ""
        for _name, pairs in vocabulary_mappings.items():
            if len(pairs) > 0:
                concept_tuple = pairs[i % len(pairs)]
                break

        if concept_tuple is None:
            snomed_cui = snomed_concepts[i % len(snomed_concepts)]
            target_codes = [f"ICD_{i}", f"LOINC_{i+10}"]
        else:
            snomed_cui, target_codes = concept_tuple

        results.append(
            {
                "snomed_cui": snomed_cui,
                "target_codes": target_codes.copy(),
                "vocab_type": _name if concept_tuple else "generic",
                "num_expected": len(target_codes),
            }
        )

    return results


def load_mapping_datasets(cache_dir: Optional[str] = None) -> dict:
    """Load pre-generated mapping datasets or generate new ones.

    Args:
        cache_dir: Optional directory to cache generated data

    Returns:
        Dict with dataset names as keys and lists of samples
    """
    import os

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "cache", "mapping_datasets"
        )

    os.makedirs(cache_dir, exist_ok=True)

    return {
        "small": generate_mapping_dataset(num_samples=25),
        "medium": generate_mapping_dataset(num_samples=100),
        "large": generate_mapping_dataset(num_samples=500),
    }
