"""Dataset generation and loading for term lookup benchmarking."""

from typing import List, Optional

try:
    from datasets import DatasetDict
except ImportError:

    class DatasetDict:
        pass


def generate_term_dataset(
    num_samples: int = 100,
    medical_terms: Optional[List[tuple]] = None,
) -> List[dict]:
    """Generate synthetic term lookup dataset for benchmarking.

    Each sample contains a clinical term and the expected SNOMED CUI that
    should be matched with high confidence.

    Args:
        num_samples: Number of samples to generate
        medical_terms: Optional list of (term, cui) tuples

    Returns:
        List of dicts with keys: 'term', 'expected_cui'
    """
    if medical_terms is None:
        medical_terms = [
            ("diabetes mellitus", "73211009"),
            ("hypertension", "38341003"),
            ("migraine", "37796005"),
            ("pneumonia", "233604007"),
            ("asthma", "195967001"),
            ("osteoporosis", "230690007"),
            ("arthritis", "34825004"),
            ("hyperlipidemia", "414015000"),
            ("depression", "35489007"),
            ("chronic kidney disease", "236422002"),
        ]

    if not medical_terms:
        medical_terms = [
            (f"medical condition {i}", str(100000 + i)) for i in range(num_samples)
        ]

    results = []

    for i in range(num_samples):
        term, cui = medical_terms[i % len(medical_terms)]

        results.append(
            {
                "term": term,
                "expected_cui": cui,
                "term_length": len(term),
            }
        )

    return results


def load_term_datasets(cache_dir: Optional[str] = None) -> dict:
    """Load pre-generated term datasets or generate new ones.

    Args:
        cache_dir: Optional directory to cache generated data

    Returns:
        Dict with dataset names as keys and lists of samples
    """
    import os

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "cache", "term_datasets"
        )

    os.makedirs(cache_dir, exist_ok=True)

    return {
        "small": generate_term_dataset(num_samples=25),
        "medium": generate_term_dataset(num_samples=100),
        "large": generate_term_dataset(num_samples=500),
    }
