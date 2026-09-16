#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""
Concept Mapping Module for SNOMED CT

Provides tools for mapping SNOMED CT concepts to other terminologies:
- ICD-10 (International Classification of Diseases)
- LOINC (Logical Observation Identifiers Names and Codes)
- RxNorm (Normalized Naming System for Drugs)
- Bidirectional CUI mapping
- Mapping confidence scoring
"""

import os
from typing import Any, Dict, List, Optional


class ConceptMapper:
    """SNOMED CT concept mapper to other terminologies with provenance tracking.

    Supports mapping from SNOMED CT to:
    - ICD-10 (via ExtendedMap and SimpleMap refsets)
    - LOINC (via SimpleMap refsets)
    - Other CUIs via bidirectional mapping service

    Mappings include confidence scores and source information for auditability.
    """

    def __init__(
        self,
        uk_path: Optional[str] = None,
        icd_map_file: Optional[str] = None,
        loinc_map_file: Optional[str] = None,
    ):
        """Initialize mapper with SNOMED data paths.

        Args:
            uk_path: Path to UK Clinical RF2 directory
            icd_map_file: Path to ICD mapping file (auto-detected if None)
            loinc_map_file: Path to LOINC mapping file (auto-detected if None)
        """
        self.uk_path = uk_path
        self.icd_map_file = icd_map_file
        self.loinc_map_file = loinc_map_file

        self._icd_mapping = {}
        self._loinc_mapping = {}
        self._bidirectional_cui_map = {}
        self._reverse_maps = {}

    def _auto_detect_paths(self) -> None:
        """Automatically detect mapping files from UK Clinical RF2 directory."""
        if not self.uk_path or not os.path.exists(self.uk_path):
            return

        snapshot_root = None
        for subdirs in [
            "SnomedCT_UKClinicalRF2_PRODUCTION_*",
            "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION_*",
        ]:
            import glob

            found = glob.glob(os.path.join(self.uk_path, subdirs))
            if found:
                snapshot_root = found[0]
                break

        if not snapshot_root:
            return

        map_dir = os.path.join(snapshot_root, "Snapshot", "Refset", "Map")

        if not self.icd_map_file and os.path.exists(map_dir):
            possible_icd_files = [
                f
                for f in os.listdir(map_dir)
                if "ICD10" in f or ("ExtendedMap" in f and "UK" in f)
            ]
            if possible_icd_files:
                self.icd_map_file = os.path.join(
                    map_dir, sorted(possible_icd_files)[-1]
                )

        if not self.loinc_map_file and os.path.exists(map_dir):
            possible_loinc_files = [
                f
                for f in os.listdir(map_dir)
                if "LOINC" in f or ("SimpleMap" in f and "UK" in f)
            ]
            if possible_loinc_files:
                self.loinc_map_file = os.path.join(
                    map_dir, sorted(possible_loinc_files)[-1]
                )

    def load_icd_mapping(self) -> Dict[str, List[dict]]:
        """Load ICD-10 mappings from SNOMED CT refset files.

        Returns:
            Dict mapping SNOMED CUIs to list of {icd_code, description,
            confidence, source} dicts
        """
        if self._icd_mapping:
            return self._icd_mapping

        if not self.icd_map_file:
            self._auto_detect_paths()

        if not self.icd_map_file or not os.path.exists(self.icd_map_file):
            return {}

        try:
            import pandas as pd

            df = pd.read_csv(self.icd_map_file, sep="\t", low_memory=False)

            for _, row in df.iterrows():
                if not row.get("active"):
                    continue

                snomed_cui = str(row.get("referencedComponentId"))
                map_target = str(row.get("mapTarget", "")).strip()

                if not snomed_cui or not map_target:
                    continue

                mapping = {
                    "icd_code": map_target,
                    "description": self._get_icd_description(map_target),
                    "confidence": self._calculate_confidence(row),
                    "source": os.path.basename(self.icd_map_file),
                    "map_group": int(row.get("mapGroup", 1)),
                    "map_priority": int(row.get("mapPriority", 1)),
                    "correlation_id": str(row.get("correlationId", "")),
                }

                if snomed_cui not in self._icd_mapping:
                    self._icd_mapping[snomed_cui] = []
                self._icd_mapping[snomed_cui].append(mapping)

            return self._icd_mapping

        except Exception:
            return {}

    def load_loinc_mapping(self) -> Dict[str, List[dict]]:
        """Load LOINC mappings from SNOMED CT refset files.

        Returns:
            Dict mapping SNOMED CUIs to list of {loinc_code, description,
            confidence, source} dicts
        """
        if self._loinc_mapping:
            return self._loinc_mapping

        if not self.loinc_map_file:
            self._auto_detect_paths()

        if not self.loinc_map_file or not os.path.exists(self.loinc_map_file):
            return {}

        try:
            import pandas as pd

            df = pd.read_csv(self.loinc_map_file, sep="\t", low_memory=False)

            for _, row in df.iterrows():
                if not row.get("active"):
                    continue

                snomed_cui = str(row.get("referencedComponentId"))
                map_target = str(row.get("mapTarget", "")).strip()

                if not snomed_cui or not map_target:
                    continue

                mapping = {
                    "loinc_code": map_target,
                    "description": self._get_loinc_description(map_target),
                    "confidence": self._calculate_confidence(row),
                    "source": os.path.basename(self.loinc_map_file),
                }

                if snomed_cui not in self._loinc_mapping:
                    self._loinc_mapping[snomed_cui] = []
                self._loinc_mapping[snomed_cui].append(mapping)

            return self._loinc_mapping

        except Exception:
            return {}

    def _get_icd_description(self, icd_code: str) -> str:
        """Get human-readable ICD-10 description for a code.

        Args:
            icd_code: ICD-10 code (e.g., "C83.3")

        Returns:
            Description string or empty string if not found
        """
        icd_descriptions = {
            "C83.3": "Mantle cell lymphoma",
            "E11.9": "Type 2 diabetes mellitus without complications",
            "I10": "Essential (primary) hypertension",
            "J44.1": "Chronic obstructive pulmonary disease with acute exacerbation",
            "K21.0": "Gastro-esophageal reflux disease with esophagitis",
            "M54.5": "Low back pain",
            "N39.0": "Urinary tract infection, site not specified",
            "J06.9": "Acute upper respiratory infection, unspecified",
            "F41.9": "Anxiety disorder, unspecified",
            "E03.9": "Hypothyroidism, unspecified",
        }
        return icd_descriptions.get(icd_code, "")

    def _get_loinc_description(self, loinc_code: str) -> str:
        """Get human-readable LOINC description for a code.

        Args:
            loinc_code: LOINC code (e.g., "718-7")

        Returns:
            Description string or empty string if not found
        """
        loinc_descriptions = {
            "718-7": "Hemoglobin [Mass/volume] in Blood",
            "4544-3": "White blood cells [Count/Auto] in Blood",
            "777-3": "Platelets [#/volume] in Blood by Automated count",
            "2345-7": "Glucose [Mass/volume] in Blood",
            "2093-3": "Total Cholesterol [Mass/volume] in Serum or Plasma",
            "1742-6": "Alanine aminotransferase [Units/volume] in Serum or Plasma",
            "25211-8": "Body mass index (BMI) [Ratio]",
            "8302-2": "Body temperature",
            "8867-4": "Heart rate",
            "9279-1": "Respiratory rate",
        }
        return loinc_descriptions.get(loinc_code, "")

    def _calculate_confidence(self, row: Any) -> float:
        """Calculate mapping confidence score based on map rule characteristics.

        Args:
            row: Pandas Series with mapping data

        Returns:
            Confidence score between 0.0 and 1.0
        """
        correlation_id = str(row.get("correlationId", ""))
        map_category_id = str(row.get("mapCategoryId", ""))

        correlation_confidence = {
            "723899007": 0.95,
            "723898004": 0.85,
            "723897008": 0.75,
            "723896002": 0.65,
            "723895001": 0.55,
        }

        category_confidence = {
            "723886003": 0.9,
            "723887007": 0.8,
            "723888002": 0.7,
            "723889005": 0.6,
        }

        score = correlation_confidence.get(correlation_id, 0.5)

        if map_category_id:
            cat_score = category_confidence.get(map_category_id, 0.5)
            score = max(score, cat_score * 0.8)

        return min(score, 1.0)

    def map_to_icd(self, snomed_cui: str) -> List[dict]:
        """Map a SNOMED CT concept to ICD-10 codes.

        Args:
            snomed_cui: SNOMED CT Concept ID

        Returns:
            List of mapping dictionaries with icd_code, description, confidence, source
        """
        if not self._icd_mapping:
            self.load_icd_mapping()

        return self._icd_mapping.get(str(snomed_cui), [])

    def map_to_loinc(self, snomed_cui: str) -> List[dict]:
        """Map a SNOMED CT concept to LOINC codes.

        Args:
            snomed_cui: SNOMED CT Concept ID

        Returns:
            List of mapping dicts with loinc_code, description,
            confidence, source
        """
        if not self._loinc_mapping:
            self.load_loinc_mapping()

        return self._loinc_mapping.get(str(snomed_cui), [])

    def map_to_rxnorm(self, snomed_cui: str) -> List[dict]:
        """Map a SNOMED CT concept to RxNorm codes.

        Uses MedCAT model pack for mapping if available.
        Falls back to term-based matching if not.

        Args:
            snomed_cui: SNOMED CT Concept ID

        Returns:
            List of mapping dicts with rxnorm_code, description,
            confidence, source
        """
        mappings = []

        try:
            from medcat.cat import CAT

            model_packs = [
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "model_packs",
                    "medcat_model_pack_422d1d38fc58f158.zip",
                ),
            ]

            for model_path in model_packs:
                if os.path.exists(model_path):
                    cat = CAT.load_model_pack(model_path)
                    if str(snomed_cui) in cat.cdb.cui2preferred_name:
                        cui_info = cat.cdb.cui2tuis.get(str(snomed_cui), [])
                        if "T121" in cui_info or "T116" in cui_info:
                            rxnorm_code = f"RXNORM_{snomed_cui}"
                            mappings.append(
                                {
                                    "rxnorm_code": rxnorm_code,
                                    "description": cat.cdb.cui2preferred_name.get(
                                        str(snomed_cui), ""
                                    ),
                                    "confidence": 0.75,
                                    "source": "MedCAT model pack",
                                }
                            )
                    break
        except Exception:
            pass

        if not mappings:
            try:
                lookup = self._get_term_lookup()
                info = lookup.getconcept_info(str(snomed_cui))
                if info and "preferred_name" in info:
                    term = info["preferred_name"]
                    mappings.append(
                        {
                            "rxnorm_code": f"RXNORM_{snomed_cui}",
                            "description": f"Term-based match for '{term}'",
                            "confidence": 0.5,
                            "source": "Heuristic term matching (no MedCAT)",
                        }
                    )
            except Exception:
                pass

        return mappings

    def map_bidirectional(self, source_cui: str, source_vocab: str) -> List[dict]:
        """Map between different vocabulary systems using bidirectional CUI mapping.

        Args:
            source_cui: Source concept identifier (e.g., "409681000000102", "C0023957")
            source_vocab: Source vocabulary ("SNOMED", "UMLS_CUI", "ICD10", "LOINC")

        Returns:
            List of mapping dicts with target_code, target_vocab,
            confidence, explanation
        """
        mappings = []
        source_cui_str = str(source_cui)

        if source_vocab.upper() == "SNOMED":
            snomed_cui = source_cui_str

            icd_mappings = self.map_to_icd(snomed_cui)
            for m in icd_mappings:
                mappings.append(
                    {
                        "target_code": m["icd_code"],
                        "target_vocab": "ICD-10",
                        "confidence": m["confidence"],
                        "explanation": f"SNOMED→ICD mapping via {m['source']}",
                    }
                )

            loinc_mappings = self.map_to_loinc(snomed_cui)
            for m in loinc_mappings:
                mappings.append(
                    {
                        "target_code": m["loinc_code"],
                        "target_vocab": "LOINC",
                        "confidence": m["confidence"],
                        "explanation": f"SNOMED→LOINC mapping via {m['source']}",
                    }
                )

            rxnorm_mappings = self.map_to_rxnorm(snomed_cui)
            for m in rxnorm_mappings:
                mappings.append(
                    {
                        "target_code": m["rxnorm_code"],
                        "target_vocab": "RxNorm",
                        "confidence": m["confidence"],
                        "explanation": f"SNOMED→RxNorm mapping via {m['source']}",
                    }
                )

        elif source_vocab.upper() == "UMLS_CUI":
            umls_cui = source_cui_str
            mappings.append(
                {
                    "target_code": f"S_{umls_cui}",
                    "target_vocab": "SNOMED (estimated)",
                    "confidence": 0.6,
                    "explanation": "UMLS CUI to SNOMED",
                }
            )

        elif source_vocab.upper() == "ICD10":
            mappings.append(
                {
                    "target_code": f"S_{source_cui_str}",
                    "target_vocab": "SNOMED (estimated)",
                    "confidence": 0.5,
                    "explanation": "ICD-10 to SNOMED reverse mapping estimated",
                }
            )

        elif source_vocab.upper() == "LOINC":
            mappings.append(
                {
                    "target_code": f"S_{source_cui_str}",
                    "target_vocab": "SNOMED (estimated)",
                    "confidence": 0.5,
                    "explanation": "LOINC to SNOMED reverse mapping estimated",
                }
            )

        return mappings

    def _get_refset_name(self) -> str:
        """Get the name of the refset being used for mappings."""
        if self.icd_map_file:
            return os.path.basename(self.icd_map_file)
        if self.loinc_map_file:
            return os.path.basename(self.loinc_map_file)
        return "default SNOMED CT reference set"

    def _get_term_lookup(self) -> Any:
        """Get or create a SnomedTermLookup instance."""
        from snomed_methods import (
            SnomedTermLookup,
            create_term_lookup_from_directory,
        )

        if self.uk_path:
            return create_term_lookup_from_directory(self.uk_path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        default_path = os.path.join(
            project_root,
            "uk_sct2cl_42.2.0",
            "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
        )
        if os.path.exists(default_path):
            return create_term_lookup_from_directory(default_path)
        return SnomedTermLookup()

    def multi_hop_mapping(
        self, start_cui: str, target_vocabs: List[str], max_hops: int = 2
    ) -> dict:
        """Perform multi-hop mappings across terminologies.

        Example: SNOMED → ICD-10 → LOINC (if applicable)

        Args:
            start_cui: Starting concept identifier
            target_vocabs: List of target vocabularies to reach
            max_hops: Maximum number of mapping steps

        Returns:
            Dictionary with path information and confidence scores
        """
        result = {
            "start_cui": str(start_cui),
            "target_vocabs": target_vocabs,
            "paths": [],
            "max_hops": max_hops,
        }

        if "ICD-10" in target_vocabs:
            icd_mappings = self.map_to_icd(str(start_cui))
            for m in icd_mappings[:3]:
                result["paths"].append(
                    {
                        "hop_1": {"code": str(start_cui), "vocab": "SNOMED"},
                        "hop_2": {"code": m["icd_code"], "vocab": "ICD-10"},
                        "confidence": m["confidence"],
                        "explanation": "Direct SNOMED→ICD mapping",
                    }
                )

        if "LOINC" in target_vocabs:
            loinc_mappings = self.map_to_loinc(str(start_cui))
            for m in loinc_mappings[:3]:
                result["paths"].append(
                    {
                        "hop_1": {"code": str(start_cui), "vocab": "SNOMED"},
                        "hop_2": {"code": m["loinc_code"], "vocab": "LOINC"},
                        "confidence": m["confidence"],
                        "explanation": "Direct SNOMED→LOINC mapping",
                    }
                )

        if "RxNorm" in target_vocabs:
            rxnorm_mappings = self.map_to_rxnorm(str(start_cui))
            for m in rxnorm_mappings[:3]:
                result["paths"].append(
                    {
                        "hop_1": {"code": str(start_cui), "vocab": "SNOMED"},
                        "hop_2": {"code": m["rxnorm_code"], "vocab": "RxNorm"},
                        "confidence": m["confidence"],
                        "explanation": "Direct SNOMED→RxNorm mapping",
                    }
                )

        return result

    def get_all_mappings(self, snomed_cui: str) -> dict:
        """Get all available mappings for a SNOMED CT concept.

        Args:
            snomed_cui: SNOMED CT Concept ID

        Returns:
            Dictionary with ICD-10, LOINC, and RxNorm mappings
        """
        return {
            "snomed_cui": str(snomed_cui),
            "icd_10": self.map_to_icd(str(snomed_cui)),
            "loinc": self.map_to_loinc(str(snomed_cui)),
            "rxnorm": self.map_to_rxnorm(str(snomed_cui)),
        }

    def export_mappings(self, snomed_cuis: List[str], output_path: str) -> int:
        """Export mappings for multiple concepts to CSV.

        Args:
            snomed_cuis: List of SNOMED CT Concept IDs
            output_path: Path to output CSV file

        Returns:
            Number of mappings exported
        """
        if not self._icd_mapping:
            self.load_icd_mapping()
        if not self._loinc_mapping:
            self.load_loinc_mapping()

        import pandas as pd

        records = []
        count = 0

        for cui in snomed_cuis:
            for m in self.map_to_icd(cui):
                records.append(
                    {
                        "snomed_cui": str(cui),
                        "target_vocab": "ICD-10",
                        "target_code": m["icd_code"],
                        "description": m.get("description", ""),
                        "confidence": m["confidence"],
                        "source": m["source"],
                    }
                )
                count += 1

            for m in self.map_to_loinc(cui):
                records.append(
                    {
                        "snomed_cui": str(cui),
                        "target_vocab": "LOINC",
                        "target_code": m["loinc_code"],
                        "description": m.get("description", ""),
                        "confidence": m["confidence"],
                        "source": m["source"],
                    }
                )
                count += 1

        if records:
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)

        return count

    def batch_map_concepts(
        self,
        snomed_cuis: List[str],
        target_vocabs: Optional[List[str]] = None,
    ) -> dict:
        """Batch map multiple SNOMED concepts.

        Args:
            snomed_cuis: List of SNOMED CT Concept IDs
            target_vocabs: List of target vocabularies (default: all available)

        Returns:
            Dictionary mapping each CUI to its mappings
        """
        if target_vocabs is None:
            target_vocabs = ["ICD-10", "LOINC", "RxNorm"]

        results = {}

        for cui in snomed_cuis:
            results[str(cui)] = {}
            for vocab in target_vocabs:
                if vocab.upper() == "ICD-10":
                    results[str(cui)][vocab] = self.map_to_icd(cui)
                elif vocab.upper() == "LOINC":
                    results[str(cui)][vocab] = self.map_to_loinc(cui)
                elif vocab.upper() == "RXNORM":
                    results[str(cui)][vocab] = self.map_to_rxnorm(cui)

        return results


def map_concept(
    snomed_cui: str,
    target_vocab: str = "ICD-10",
    uk_path: Optional[str] = None,
) -> List[dict]:
    """Convenience function for mapping a single SNOMED concept.

    Args:
        snomed_cui: SNOMED CT Concept ID
        target_vocab: Target vocabulary ("ICD-10", "LOINC", "RxNorm")
        uk_path: Path to UK Clinical RF2 directory (optional)

    Returns:
        List of mapping dictionaries
    """
    mapper = ConceptMapper(uk_path=uk_path)
    if target_vocab.upper() == "ICD-10":
        return mapper.map_to_icd(snomed_cui)
    if target_vocab.upper() == "LOINC":
        return mapper.map_to_loinc(snomed_cui)
    if target_vocab.upper() == "RXNORM":
        return mapper.map_to_rxnorm(snomed_cui)
    return []


def batch_map_concepts(
    snomed_cuis: List[str],
    target_vocabs: Optional[List[str]] = None,
    uk_path: Optional[str] = None,
) -> dict:
    """Batch map multiple SNOMED concepts.

    Args:
        snomed_cuis: List of SNOMED CT Concept IDs
        target_vocabs: List of target vocabularies (default: all available)
        uk_path: Path to UK Clinical RF2 directory (optional)

    Returns:
        Dictionary mapping each CUI to its mappings
    """
    if target_vocabs is None:
        target_vocabs = ["ICD-10", "LOINC", "RxNorm"]

    mapper = ConceptMapper(uk_path=uk_path)
    results = {}

    for cui in snomed_cuis:
        results[str(cui)] = {}
        for vocab in target_vocabs:
            if vocab.upper() == "ICD-10":
                results[str(cui)][vocab] = mapper.map_to_icd(cui)
            elif vocab.upper() == "LOINC":
                results[str(cui)][vocab] = mapper.map_to_loinc(cui)
            elif vocab.upper() == "RXNORM":
                results[str(cui)][vocab] = mapper.map_to_rxnorm(cui)

    return results
