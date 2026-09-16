# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Dataset generation and loading for clinical concept annotation benchmarking."""

import random
from typing import List, Optional

try:
    from datasets import DatasetDict
except ImportError:

    class DatasetDict:
        pass


def generate_annotation_dataset(
    num_samples: int = 100,
    cui_vocab: Optional[List[str]] = None,
    diseases: Optional[List[tuple]] = None,
) -> List[dict]:
    """Generate synthetic clinical note dataset for annotation benchmarking.

    Creates realistic clinical text and gold-standard CUI annotations based on
    disease-specific terminology patterns.

    Args:
        num_samples: Number of clinical notes to generate
        cui_vocab: Optional list of available CUIs to select from
        diseases: List of (disease_name, [related_cuis]) tuples for generation

    Returns:
        List of dicts with keys: 'text', 'gold_cuis'
    """
    if diseases is None:
        diseases = [
            ("diabetes mellitus", ["C001", "C002", "C003"]),
            ("hypertension", ["C010", "C011", "C012"]),
            ("depression", ["C020", "C021", "C022"]),
            ("migraine", ["C030", "C031"]),
            ("pneumonia", ["C040", "C041", "C042", "C043"]),
            ("asthma", ["C050", "C051", "C052"]),
            ("osteoporosis", ["C060", "C061"]),
            ("arthritis", ["C070", "C071", "C072"]),
            ("hyperlipidemia", ["C080", "C081"]),
            ("chronic kidney disease", ["C090", "C091"]),
        ]

    if cui_vocab is None:
        all_cuis = []
        for _, cuis in diseases:
            all_cuis.extend(cuis)
        cui_vocab = list(set(all_cuis))

    clinical_templates = [
        "Patient presents with {symptoms}. History of {condition}.",
        "The patient reports {symptoms}. Diagnosis: {condition}.",
        "{symptoms} observed. Possibly related to {condition}.",
        "Subject complaining of {symptoms}. Medical history includes {condition}.",
        "{symptoms} noted during examination. Patient has {condition}.",
    ]

    symptom_phrases = [
        "severe header",
        "persistent fever",
        "chest pain",
        "shortness of breath",
        "joint swelling",
        "fatigue",
        "nausea and vomiting",
        "abdominal discomfort",
        "dizziness",
        "muscle weakness",
        "rash",
        "cough",
    ]

    results = []

    for i in range(num_samples):
        disease_name, gold_cuis = diseases[i % len(diseases)]
        template = clinical_templates[i % len(clinical_templates)]

        symptoms = ", ".join(random.sample(symptom_phrases, k=2))
        text = template.format(symptoms=symptoms, condition=disease_name)

        results.append(
            {
                "text": text,
                "gold_cuis": gold_cuis.copy(),
                "disease_name": disease_name,
            }
        )

    return results


def load_annotation_datasets(cache_dir: Optional[str] = None) -> dict:
    """Load pre-generated annotation datasets or generate new ones.

    Args:
        cache_dir: Optional directory to cache generated data

    Returns:
        Dict with dataset names as keys and lists of samples
    """
    import os

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "cache", "annotation_datasets"
        )

    os.makedirs(cache_dir, exist_ok=True)

    return {
        "small": generate_annotation_dataset(num_samples=25),
        "medium": generate_annotation_dataset(num_samples=100),
        "large": generate_annotation_dataset(num_samples=500),
    }


def create_dataset_from_concepts(concept_list: List[tuple]) -> List[dict]:
    """Create annotation dataset from list of (text, [cuis]) tuples.

    Args:
        concept_list: List of (text, gold_cuis) tuples

    Returns:
        List of dicts with 'text' and 'gold_cuis' keys
    """
    return [{"text": text, "gold_cuis": cuis} for text, cuis in concept_list]
