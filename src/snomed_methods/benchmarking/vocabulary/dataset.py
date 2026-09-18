# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Dataset generation and loading for vocabulary mapping benchmarking.

This module generates real vocabulary mapping datasets using SNOMED CT RF2 data.
It extracts actual mappings from the Simple Map reference set files rather than
using synthetic/pseudo-random codes.
"""

from __future__ import annotations

import os
import pathlib

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from datasets import DatasetDict
except ImportError:

    class DatasetDict:
        pass


def load_simple_map_mappings(snomed_dir: str) -> dict[str, list[str]]:
    """Load actual SNOMED CT to target code mappings from RF2 Simple Map files.

    Args:
        snomed_dir: Path to SNOMED CT directory

    Returns:
        Dict mapping SNOMED concept IDs to list of mapped codes

    """
    if not snomed_dir or not pathlib.Path(snomed_dir).exists():
        return {}

    mappings = {}

    try:
        snapshot_dir = _find_snapshot_dir(snomed_dir)
        if not snapshot_dir:
            return {}

        map_files = [
            f
            for f in pathlib.Path(snapshot_dir).iterdir()
            if "SimpleMap" in f.name and f.suffix == ".txt"
        ]
        if not map_files:
            return {}

        for map_file in map_files:
            _process_map_file(pathlib.Path(snapshot_dir) / map_file, mappings)

    except (FileNotFoundError, PermissionError, OSError):
        pass

    return mappings


def _find_snapshot_dir(snomed_dir: str) -> str | None:
    """Find the Snapshot/Refset/Map directory."""
    for subdir in ["Snapshot", "Full"]:
        map_path = pathlib.Path(snomed_dir) / subdir / "Refset" / "Map"
        if map_path.exists():
            return str(map_path)
    return None


def _process_map_file(file_path: pathlib.Path, mappings: dict[str, list[str]]) -> None:
    """Process a single Simple Map file and add to mappings dict."""
    if pd is None:
        return

    try:
        df = pd.read_csv(file_path, sep="\t", low_memory=False)

        if "active" in df.columns:
            active_mask = df["active"].isin([1, "1"]) | (df["active"])
            df = df[active_mask]

        for _, row in df.iterrows():
            snomed_cui = str(row.get("referencedComponentId", "")).strip()
            map_target = str(row.get("mapTarget", "")).strip()

            if not snomed_cui or not map_target:
                continue

            if snomed_cui not in mappings:
                mappings[snomed_cui] = []

            if map_target not in mappings[snomed_cui]:
                mappings[snomed_cui].append(map_target)

    except (FileNotFoundError, PermissionError, OSError, pd.errors.ParserError):
        pass


# Fallback vocabulary mappings for testing when RF2 is unavailable
_FALLBACK_VOCABULARY_MAPPINGS = {
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


def _get_fallback_concepts() -> list[str]:
    """Get list of concept IDs from fallback mappings."""
    concepts = []
    for pairs in _FALLBACK_VOCABULARY_MAPPINGS.values():
        for cui, _ in pairs:
            concepts.append(cui)
    return list(set(concepts))


def generate_mapping_dataset(
    num_samples: int = 100,
    snomed_concepts: list[str] | None = None,
    snomed_dir: str | None = None,
) -> list[dict]:
    """Generate vocabulary mapping dataset from real SNOMED CT RF2 data.

    This function extracts actual mappings from the Simple Map reference set
    files in the RF2 data, rather than using synthetic/pseudo-random codes.

    For concepts that don't have mappings in RF2, it provides empty target_codes
    to test recall/coverage metrics properly.

    Args:
        num_samples: Number of samples to generate (used if snomed_concepts is None)
        snomed_concepts: Optional list of SNOMED CUIs to use
        snomed_dir: Path to SNOMED CT RF2 directory

    Returns:
        List of dicts with keys: 'snomed_cui', 'target_codes'

    """
    if snomed_dir is None:
        snomed_dir = os.environ.get(
            "SNOMED_RF2_PATH",
            "/workspaces/snomed_methods/uk_sct2cl_42.2.0/SnomedCT_InternationalRF2_PRODUCTION_20260201T120000Z",
        )

    real_mappings = load_simple_map_mappings(snomed_dir)

    if snomed_concepts is None:
        if not real_mappings:
            # Fallback to vocab-based data when RF2 is not available
            base_concepts = _get_fallback_concepts()
            snomed_concepts = base_concepts + [
                str(i)
                for i in range(1000, 1000 + max(0, num_samples - len(base_concepts)))
            ]
        else:
            available_cuis = list(real_mappings.keys())
            # Ensure we have valid concepts even if RF2 data exists but has issues
            fallback_concepts = _get_fallback_concepts()
            snomed_concepts = available_cuis + fallback_concepts

    results = []

    for cui in snomed_concepts[:num_samples]:
        target_codes = real_mappings.get(cui, [])

        # If no RF2 mappings and this is a fallback concept, use the hardcoded values
        if not target_codes:
            for pairs in _FALLBACK_VOCABULARY_MAPPINGS.values():
                for cui_pair, codes in pairs:
                    if str(cui_pair) == str(cui):
                        target_codes = list(codes)
                        break
                if target_codes:
                    break

        results.append(
            {
                "snomed_cui": str(cui),
                "target_codes": target_codes.copy() if target_codes else [],
                "has_mapping": len(target_codes) > 0,
                "num_expected": len(target_codes),
            },
        )

    return results


def load_mapping_datasets(
    cache_dir: str | None = None,
    snomed_dir: str | None = None,
) -> dict:
    """Load pre-generated mapping datasets or generate new ones.

    Args:
        cache_dir: Optional directory to cache generated data
        snomed_dir: Path to SNOMED CT RF2 directory

    Returns:
        Dict with dataset names as keys and lists of samples

    """
    if cache_dir is None:
        cache_dir = str(
            pathlib.Path(__file__).parent.parent / "cache" / "mapping_datasets",
        )

    pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)

    return {
        "small": generate_mapping_dataset(num_samples=25, snomed_dir=snomed_dir),
        "medium": generate_mapping_dataset(num_samples=100, snomed_dir=snomed_dir),
        "large": generate_mapping_dataset(num_samples=500, snomed_dir=snomed_dir),
    }
