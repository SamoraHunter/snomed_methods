#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""
UMLS CUI Mapper for SNOMED CT

Provides bidirectional mapping between SNOMED CT concepts and UMLS
Concept Unique Identifiers (CUIs) for interoperability with biomedical tools.
"""

import os
from typing import Dict, List, Optional


class UMLSCIMapper:
    """SNOMED CT to UMLS CUI mapper with bidirectional lookup."""

    def __init__(
        self,
        uk_path: Optional[str] = None,
        international_path: Optional[str] = None,
        simple_map_file: Optional[str] = None,
    ):
        """Initialize mapper with SNOMED data paths."""
        self.uk_path = uk_path
        self.international_path = international_path
        self.simple_map_file = simple_map_file

        self._snomed_to_umls = {}
        self._umls_to_snomed = {}

    def _auto_detect_paths(self) -> None:
        """Auto-detect RF2 mapping files."""
        import glob

        if not self.international_path:
            for pattern in ["uk_sct2cl_*/SnomedCT_InternationalRF2_*"]:
                found = glob.glob(pattern)
                if found:
                    self.international_path = os.path.dirname(found[0])
                    break

        if not self.simple_map_file and self.international_path:
            map_dir = os.path.join(self.international_path, "Snapshot", "Refset", "Map")
            if os.path.exists(map_dir):
                simple_maps = [f for f in os.listdir(map_dir) if "SimpleMap" in f]
                if simple_maps:
                    self.simple_map_file = os.path.join(
                        map_dir, sorted(simple_maps)[-1]
                    )

    def _get_mapping_files(self) -> List[str]:
        """Get list of mapping files to process."""
        files = []
        if self.simple_map_file and os.path.exists(self.simple_map_file):
            files.append(self.simple_map_file)
        elif not self.international_path:
            self._auto_detect_paths()
            if self.simple_map_file:
                files.append(self.simple_map_file)
        return files

    def _process_mapping_row(self, row: object, file_path: str) -> Optional[tuple]:
        """Process a single mapping row and return (snomed_cui, mapping) or None."""
        if not row.get("active"):
            return None

        snomed_cui = str(row.get("referencedComponentId", "")).strip()
        map_target = str(row.get("mapTarget", "")).strip()

        if not self._is_umls_cui(map_target):
            return None

        mapping = {
            "umls_cui": map_target,
            "confidence": 0.75,
            "source": os.path.basename(file_path),
        }

        return (snomed_cui, mapping)

    def _add_mapping(self, snomed_cui: str, mapping: dict) -> Optional[dict]:
        """Add a mapping if not already present. Returns updated mapping or None."""
        if snomed_cui not in self._snomed_to_umls:
            self._snomed_to_umls[snomed_cui] = []

        existing = {m["umls_cui"] for m in self._snomed_to_umls[snomed_cui]}
        if mapping["umls_cui"] not in existing:
            self._snomed_to_umls[snomed_cui].append(mapping)
            return mapping
        return None

    def load_snomed_to_umls(self) -> Dict[str, List[dict]]:
        """Load SNOMED CT to UMLS CUI mappings."""
        if self._snomed_to_umls:
            return self._snomed_to_umls

        files = self._get_mapping_files()

        try:
            import pandas as pd

            for file_path in files:
                df = pd.read_csv(file_path, sep="\t", low_memory=False)

                for _, row in df.iterrows():
                    result = self._process_mapping_row(row, file_path)
                    if result is None:
                        continue
                    snomed_cui, mapping = result
                    self._add_mapping(snomed_cui, mapping)

            return self._snomed_to_umls

        except Exception:
            return {}

    def _is_umls_cui(self, code: str) -> bool:
        """Check if code is UMLS CUI format (C + 7 digits)."""
        import re

        return bool(re.match(r"^C\d{7}$", str(code).strip()))

    def load_umls_to_snomed(self) -> Dict[str, List[dict]]:
        """Load reverse UMLS CUI to SNOMED CT mappings."""
        if self._umls_to_snomed:
            return self._umls_to_snomed

        self.load_snomed_to_umls()

        for snomed_cui, mappings in self._snomed_to_umls.items():
            for m in mappings:
                umls_cui = m["umls_cui"]
                if umls_cui not in self._umls_to_snomed:
                    self._umls_to_snomed[umls_cui] = []

                reverse_mapping = {
                    "snomed_cui": snomed_cui,
                    "confidence": m["confidence"],
                    "source": m["source"],
                }

                existing = {s["snomed_cui"] for s in self._umls_to_snomed[umls_cui]}
                if snomed_cui not in existing:
                    self._umls_to_snomed[umls_cui].append(reverse_mapping)

        return self._umls_to_snomed

    def map_to_umls(self, snomed_cui: str) -> List[dict]:
        """Map SNOMED CT concept to UMLS CUI(s)."""
        if not self._snomed_to_umls:
            self.load_snomed_to_umls()
        return self._snomed_to_umls.get(str(snomed_cui), [])

    def map_from_umls(self, umls_cui: str) -> List[dict]:
        """Map UMLS CUI to SNOMED CT concept(s)."""
        if not self._umls_to_snomed:
            self.load_umls_to_snomed()
        return self._umls_to_snomed.get(str(umls_cui), [])

    def bidirectional_map(self, concept_id: str, source_vocab: str) -> List[dict]:
        """Bidirectionally map between SNOMED and UMLS CUIs."""
        mappings = []
        concept_id_str = str(concept_id)

        if source_vocab.upper() == "SNOMED":
            for m in self.map_to_umls(concept_id_str):
                mappings.append(
                    {
                        "source_cui": concept_id_str,
                        "source_vocab": "SNOMED CT",
                        "target_cui": m["umls_cui"],
                        "target_vocab": "UMLS CUI",
                        "confidence": m["confidence"],
                        "source_file": m.get("source", ""),
                    }
                )

        elif source_vocab.upper() == "UMLS_CUI":
            for m in self.map_from_umls(concept_id_str):
                mappings.append(
                    {
                        "source_cui": concept_id_str,
                        "source_vocab": "UMLS CUI",
                        "target_cui": m["snomed_cui"],
                        "target_vocab": "SNOMED CT",
                        "confidence": m["confidence"],
                        "source_file": m.get("source", ""),
                    }
                )

        return mappings

    def get_all_mappings(self, concept_id: str) -> dict:
        """Get all available mappings for a concept."""
        if self._is_umls_cui(concept_id):
            return {
                "umls_cui": concept_id,
                "snomed_concepts": self.map_from_umls(concept_id),
            }
        return {
            "snomed_cui": str(concept_id),
            "umls_cuis": self.map_to_umls(concept_id),
        }

    def export_mappings(self, concepts: List[str], output_path: str) -> int:
        """Export SNOMED-to-UMLS mappings to CSV."""
        import pandas as pd

        records = []
        for concept in concepts:
            if self._is_umls_cui(concept):
                for m in self.map_from_umls(concept):
                    records.append(
                        {
                            "source_cui": concept,
                            "source_vocab": "UMLS CUI",
                            "target_cui": m["snomed_cui"],
                            "target_vocab": "SNOMED CT",
                            "confidence": m["confidence"],
                        }
                    )
            else:
                for m in self.map_to_umls(concept):
                    records.append(
                        {
                            "source_cui": concept,
                            "source_vocab": "SNOMED CT",
                            "target_cui": m["umls_cui"],
                            "target_vocab": "UMLS CUI",
                            "confidence": m["confidence"],
                        }
                    )

        if records:
            pd.DataFrame(records).to_csv(output_path, index=False)
            return len(records)

        return 0


def map_to_umls(snomed_cui: str) -> List[dict]:
    """Convenience function: SNOMED CT to UMLS CUI."""
    mapper = UMLSCIMapper()
    return mapper.map_to_umls(snomed_cui)


def map_from_umls(umls_cui: str) -> List[dict]:
    """Convenience function: UMLS CUI to SNOMED CT."""
    mapper = UMLSCIMapper()
    return mapper.map_from_umls(umls_cui)


def batch_map_to_umls(snomed_cuis: List[str]) -> Dict[str, List[dict]]:
    """Batch map SNOMED CT to UMLS CUIs."""
    mapper = UMLSCIMapper()
    results = {}
    for cui in snomed_cuis:
        results[str(cui)] = mapper.map_to_umls(cui)
    return results
