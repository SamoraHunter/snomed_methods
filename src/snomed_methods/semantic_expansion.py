#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""
Semantic Expansion Module for SNOMED CT.

This module provides generic functions to find all SNOMED CT concepts related
to any medical term(s) using multiple search strategies:

1. Term matching - exact and prefix matching on description terms
2. Hierarchy expansion - traversing parent-child relationships
3. Optional MedCAT semantic similarity (if model available)


Search Parameters:
    - max_concepts: Controls hierarchy expansion breadth (default=100)
    - top_n_per_term: Terms matched per search term (default=50)
    - use_hierarchy: Enable parent-child traversal (default=True)
    - use_medcat: Enable MedCAT similarity (default=False, requires model)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:

    from snomed_methods.snomed_term_lookup import SnomedTermLookup

SIMILARITY_THRESHOLD = 0.3


class SemanticSearch:
    """SNOMED CT semantic search with configurable expansion strategies."""

    def __init__(self, uk_path: str | None = None) -> None:
        """Initialize searcher with UK Clinical RF2 data path.

        Args:
            uk_path: Path to UK Clinical RF2 directory.
                Default: ../uk_sct2cl_42.2.0/
                SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z
                (relative to script)
        """
        if uk_path is None:
            script_dir = Path(__file__).resolve().parent
            project_root = (
                script_dir.parent if script_dir.name == "notebooks" else script_dir
            )
            uk_path = str(
                project_root
                / "uk_sct2cl_42.2.0"
                / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            )
        self.uk_path: str | None = uk_path

    def _generate_search_terms(self, term: str) -> list[str]:
        """Generate related search terms from input term.

        Creates variations including plurals and common prefixes to improve
        recall of related concepts.

        Args:
            term: Base medical term

        Returns:
            List of related search variants
        """
        variants = [
            f"{term}",
            f"{term.lower()}",
            f"{term}s",
            f"{term}ous",
            f"hyper{term}",
            f"hypo{term}",
            f"pseudo{term}",
            f"dys{term}",
        ]

        seen: set[str] = set()
        result: list[str] = []
        for v in variants:
            if v and v.lower() not in seen:
                seen.add(v.lower())
                result.append(v)

        return result

    def _term_lookup_search(
        self,
        lookup: SnomedTermLookup,
        terms: list[str],
        *,
        match_prefix: bool = True,
        top_n_per_term: int = 50,
    ) -> set[tuple[str, str]]:
        """Search concepts using term matching strategies.

        Performs both exact substring matching and prefix matching to find
        all descriptions containing the search terms.

        Args:
            lookup: SnomedTermLookup instance
            terms: List of terms to search for
            match_prefix: Also match terms starting with query
            top_n_per_term: Maximum results per term

        Returns:
            Set of (cui, term) tuples for matched concepts
        """
        import warnings

        if importlib.util.find_spec("snomed_term_lookup") is None:
            warnings.warn("snomed-term-lookup not available", stacklevel=2)
            return set()

        if not terms:
            return set()

        results: set[tuple[str, str]] = set()

        try:
            exact_matches: list[tuple[str, str]] = [
                match
                for search_term in terms
                for match in lookup.find_concepts_by_term(
                    search_term,
                    top_n=top_n_per_term,
                )
            ]
            results.update(exact_matches)
        except (KeyError, TypeError, ValueError):
            pass

        if match_prefix:
            try:
                prefix_matches: list[tuple[str, str]] = [
                    match
                    for search_term in terms
                    for match in lookup.find_concepts_by_term(
                        search_term,
                        match_prefix=True,
                        top_n=top_n_per_term // 2,
                    )
                ]
                results.update(prefix_matches)
            except (KeyError, TypeError, ValueError):
                pass

        return results

    def _load_relationship_file(self) -> pd.DataFrame | None:
        """Load and filter active relationships from the SNOMED relationship file.

        Returns:
            DataFrame with active relationships, or None if file not found.
        """
        rel_file = (
            Path(self.uk_path)
            / "Full"
            / "Terminology"
            / "sct2_Relationship_UKCLFull_GB1000000_20260603.txt"
        )

        if not rel_file.exists():
            return None

        try:
            rel_df = pd.read_csv(rel_file, sep="\t")
            return rel_df[rel_df["active"] == 1]
        except (OSError, ValueError):
            return None

    def _create_lookup_if_available(self) -> SnomedTermLookup | None:
        """Create SnomedTermLookup instance if available.

        Returns:
            SnomedTermLookup instance or None.
        """
        if importlib.util.find_spec("snomed_term_lookup") is not None:
            from snomed_methods.snomed_term_lookup import (
                create_term_lookup_from_directory,
            )

            return create_term_lookup_from_directory(self.uk_path)
        return None

    def _process_concept(
        self,
        current_cui: str,
        context: ProcessConceptContext,
    ) -> tuple[list[str], list[str], bool]:
        """Process a single concept and update context.

        Args:
            current_cui: Current concept ID to process
            context: ProcessConceptContext with all state

        Returns:
            Tuple of (updated_codes, updated_names, should_continue).
        """
        context.processed.add(current_cui)
        context.results_codes.append(current_cui)

        name = self._get_concept_name(current_cui, context.lookup)
        context.results_names.append(name or f"CUI: {current_cui}")

        if context.rel_active is None:
            return context.results_codes, context.results_names, True

        try:
            cui_int = int(current_cui)
        except ValueError:
            return context.results_codes, context.results_names, True

        parent_rows = context.rel_active[context.rel_active["sourceId"] == cui_int]
        for _, row in parent_rows.iterrows():
            p = str(row["destinationId"])
            if p not in context.processed and p not in context.queue:
                context.queue.append(p)

        child_rows = context.rel_active[context.rel_active["destinationId"] == cui_int]
        for _, row in child_rows.iterrows():
            c = str(row["sourceId"])
            if c not in context.processed and c not in context.queue:
                context.queue.append(c)

        return context.results_codes, context.results_names, True

    def _get_concept_name(
        self,
        current_cui: str,
        lookup: SnomedTermLookup | None,
    ) -> str | None:
        """Get the preferred name for a concept.

        Args:
            current_cui: Concept ID
            lookup: SnomedTermLookup instance or None

        Returns:
            Preferred name or None.
        """
        if not lookup:
            return None

        try:
            info = lookup.getconcept_info(current_cui)
            if info and "preferred_name" in info:
                return info["preferred_name"]
        except (KeyError, ValueError):
            pass

    def _hierarchy_expansion(
        self,
        cui_list: list[str],
        max_concepts: int = 100,
    ) -> tuple[list[str], list[str]]:
        """Expand concepts via SNOMED CT hierarchy traversal.

        Traverses parent-child relationships using iterative breadth-first
        search to find all connected concepts in the ontology tree.

        Args:
            cui_list: Starting concept IDs for expansion
            max_concepts: Maximum total concepts to retrieve

        Returns:
            Tuple of (codes, names) lists
        """
        rel_active = self._load_relationship_file()
        if rel_active is None:
            return [], []

        lookup = self._create_lookup_if_available()

        results_codes: list[str] = []
        results_names: list[str] = []
        processed: set[str] = set()
        queue: list[str] = list(cui_list[:max_concepts])

        while queue and len(processed) < max_concepts:
            current_cui = str(queue.pop(0))

            if current_cui in processed:
                continue

            context = ProcessConceptContext(
                lookup=lookup,
                rel_active=rel_active,
                results_codes=results_codes,
                results_names=results_names,
                processed=processed,
                queue=queue,
            )
            _, _, _ = self._process_concept(current_cui, context)

        return results_codes, results_names

    def _medcat_expansion(
        self,
        cui_list: list[str],
        context_type: str = "long",
        topn: int = 30,
    ) -> tuple[list[str], list[str]]:
        """Expand concepts using MedCAT semantic similarity.

        Uses a trained MedCAT model to find semantically similar concepts
        based on co-occurrence in text corpora.

        Args:
            cui_list: Starting concept IDs
            context_type: Context window for similarity calculation
            topn: Number of similar concepts per CUI

        Returns:
            Tuple of (codes, names) lists
        """
        if not importlib.util.find_spec("medcat"):
            return [], []

        try:
            from snomed_methods.snomed_methods_v1 import (
                SnomedRelations as _SnomedRelations,
            )
        except (ImportError, FileNotFoundError):
            return [], []

        snomed_path = (
            Path(self.uk_path)
            / "Full"
            / "Terminology"
            / "sct2_Relationship_UKCLFull_GB1000000_20260603.txt"
        )

        snomed = _SnomedRelations(medcat=True, snomed_rf2_full_path=str(snomed_path))

        if not snomed.has_medcat():
            return [], []

        all_codes: list[str] = []
        all_names: list[str] = []
        cdb = snomed.cat.cdb

        for cui in cui_list:
            try:
                sim_results = cdb.most_similar(
                    str(cui),
                    context_type=context_type,
                    topn=topn,
                )
            except (KeyError, TypeError, ValueError):
                continue

            for sim_cui, sim_data in sim_results.items():
                if sim_cui not in all_codes:
                    try:
                        sim_score = sim_data.get("sim", 0)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if sim_score > SIMILARITY_THRESHOLD:
                        try:
                            name = cdb.cui2preferred_name.get(
                                sim_cui,
                                f"CUI: {sim_cui}",
                            )
                        except (KeyError, TypeError, ValueError):
                            name = f"CUI: {sim_cui}"
                        all_codes.append(str(sim_cui))
                        all_names.append(name)

        return all_codes, all_names

    def _combine_results(
        self,
        term_matches: set[tuple[str, str]],
        codes: list[str],
        names: list[str],
    ) -> dict[str, str]:
        """Combine results from multiple search methods.

        Args:
            term_matches: Results from term lookup
            codes: Expanded concept codes
            names: Corresponding concept names

        Returns:
            Dictionary mapping CUI to preferred name
        """
        combined: dict[str, str] = {str(c): t for c, t in term_matches}

        for code, name in zip(codes, names):
            code_str = str(code)
            if code_str not in combined:
                if name and isinstance(name, str) and name != "None":
                    combined[code_str] = name
                else:
                    combined[code_str] = f"CUI: {code}"

        return combined

    def search(  # noqa: PLR0915
        self,
        term_or_terms: str | list[str],
        max_concepts: int = 100,
        *,
        top_n_per_term: int = 50,
        use_hierarchy: bool = True,
        use_medcat: bool = False,
    ) -> SearchResults:
        """Find all SNOMED CT concepts semantically related to the input.

        Args:
            term_or_terms: Single term string or list of terms
            max_concepts: Max concepts from hierarchy (default 100)
            top_n_per_term: Max results per search term (default 50)
            use_hierarchy: Enable hierarchy expansion (default True)
            use_medcat: Enable MedCAT semantic search (default False)

        Returns:
            SearchResults object with metrics, cuis, terms, and concepts
        """
        if isinstance(term_or_terms, str):
            search_terms = self._generate_search_terms(term_or_terms)
            base_term = term_or_terms.lower()
        else:
            search_terms = list(term_or_terms)
            base_term = "_".join(search_terms[:2])

        log_messages: list[str] = []
        log_messages.append(f"{'=' * 60}")
        log_messages.append("SEMANTIC EXPANSION SEARCH")
        log_messages.append(f"{'=' * 60}")
        log_messages.append(f"\nInput term(s): {search_terms}")
        log_messages.append("Configuration:")
        log_messages.append(f"  - max_concepts: {max_concepts}")
        log_messages.append(f"  - top_n_per_term: {top_n_per_term}")
        log_messages.append(f"  - use_hierarchy: {use_hierarchy}")
        log_messages.append(f"  - use_medcat: {use_medcat}")

        log_messages.append("\n[Step 1] Term Matching")
        log_messages.append(
            "        Searching for descriptions containing search terms...",
        )
        if importlib.util.find_spec("snomed_term_lookup") is None:
            msg = (
                "snomed-term-lookup package required. "
                "Install with: pip install snomed-term-lookup"
            )
            raise ImportError(
                msg,
            )

        from snomed_methods.snomed_term_lookup import (
            create_term_lookup_from_directory,
        )

        lookup = create_term_lookup_from_directory(self.uk_path)
        term_matches = self._term_lookup_search(
            lookup,
            search_terms,
            top_n_per_term=top_n_per_term,
        )
        log_messages.append(
            f"        Found {len(term_matches)} concepts via term matching",
        )

        cui_list = [str(c) for c, _ in term_matches]

        hierarchy_codes, hierarchy_names = [], []
        if use_hierarchy:
            log_messages.append("\n[Step 2] Hierarchy Expansion")
            log_messages.append("        Traversing parent-child relationships...")
            hierarchy_codes, hierarchy_names = self._hierarchy_expansion(
                cui_list,
                max_concepts=max_concepts,
            )
            log_messages.append(
                f"        Found {len(hierarchy_codes)} concepts via hierarchy",
            )

        medcat_codes, medcat_names = [], []
        if use_medcat:
            log_messages.append("\n[Step 3] MedCAT Semantic Expansion")
            log_messages.append("        Finding semantically similar concepts...")
            medcat_codes, medcat_names = self._medcat_expansion(
                cui_list,
                context_type="long",
                topn=30,
            )
            if not medcat_codes:
                log_messages.append("        MedCAT not available - skipping")
            else:
                log_messages.append(
                    f"        Found {len(medcat_codes)} concepts via MedCAT",
                )

        combined = self._combine_results(
            term_matches,
            hierarchy_codes + medcat_codes,
            hierarchy_names + medcat_names,
        )

        core_count = sum(1 for name in combined.values() if base_term in name.lower())
        expanded_count = len(combined) - core_count

        log_messages.append(f"\n{'=' * 60}")
        log_messages.append("SEARCH COMPLETE")
        log_messages.append(f"{'=' * 60}")
        log_messages.append(f"Total concepts: {len(combined)}")
        log_messages.append(f"  - Core (matches input): {core_count}")
        log_messages.append(f"  - Expanded (related): {expanded_count}")

        return SearchResults(
            concepts=combined,
            metrics={
                "total": len(combined),
                "core_concepts": core_count,
                "expanded_concepts": expanded_count,
                "search_terms_used": search_terms,
                "term_matching_results": len(term_matches),
                "hierarchy_expansion_results": len(hierarchy_codes),
                "medcat_results": len(medcat_codes),
            },
        )


class SearchResults:
    """Container for semantic search results with easy access to different formats."""

    def __init__(self, concepts: dict[str, str], metrics: dict[str, int]) -> None:
        self._concepts: dict[str, str] = concepts
        self._metrics: dict[str, int] = metrics

    @property
    def concepts(self) -> dict[str, str]:
        return dict(self._concepts)

    @property
    def cuis(self) -> list[str]:
        return sorted(self._concepts.keys())

    @property
    def terms(self) -> list[str]:
        return [self._concepts[cui] for cui in self.cuis]

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    def get_cui_to_term_dict(self) -> dict[str, str]:
        return self.concepts

    def get_core_concepts(
        self,
        base_term: str | None = None,
    ) -> list[tuple[str, str]]:
        if base_term is None and "search_terms_used" in self._metrics:
            base_term = self._metrics["search_terms_used"][0]

        if base_term:
            return [
                (cui, name)
                for cui, name in self._concepts.items()
                if base_term.lower() in name.lower()
            ]
        return list(self._concepts.items())

    def get_expanded_concepts(
        self,
        base_term: str | None = None,
    ) -> list[tuple[str, str]]:
        core = {c for c, _ in self.get_core_concepts(base_term)}
        return [(cui, name) for cui, name in self._concepts.items() if cui not in core]

    def __len__(self) -> int:
        return len(self._concepts)

    def __repr__(self) -> str:
        m = self._metrics
        return (
            f"SearchResults(total={m.get('total', 0)}, "
            f"core={m.get('core_concepts', 0)}, "
            f"expanded={m.get('expanded_concepts', 0)})"
        )


@dataclass
class ExpandConfig:
    """Configuration for concept expansion."""

    uk_path: str | None = None
    max_concepts: int = 100
    top_n_per_term: int = 50
    use_hierarchy: bool = True
    use_medcat: bool = False


@dataclass
class ProcessConceptContext:
    """Context for processing a concept during hierarchy expansion."""

    lookup: SnomedTermLookup | None
    rel_active: pd.DataFrame | None
    results_codes: list[str]
    results_names: list[str]
    processed: set[str]
    queue: list[str]


def expand_concepts(
    term_or_terms: str | list[str],
    uk_path: str | None = None,
    max_concepts: int = 100,
    top_n_per_term: int = 50,
    use_hierarchy: bool = True,
    use_medcat: bool = False,
) -> SearchResults:
    """Convenience function to perform semantic expansion.

    Wrapper around SemanticSearch class for simple usage.

    Args:
        term_or_terms: Single term or list of terms
        uk_path: Path to UK Clinical RF2 directory
        max_concepts: Maximum concepts from hierarchy (default 100)
        top_n_per_term: Max results per search term (default 50)
        use_hierarchy: Enable hierarchy expansion (default True)
        use_medcat: Enable MedCAT semantic search (default False)

    Returns:
        SearchResults object
    """
    config = ExpandConfig(
        uk_path=uk_path,
        max_concepts=max_concepts,
        top_n_per_term=top_n_per_term,
        use_hierarchy=use_hierarchy,
        use_medcat=use_medcat,
    )
    searcher = SemanticSearch(uk_path=config.uk_path)
    return searcher.search(
        term_or_terms,
        max_concepts=config.max_concepts,
        top_n_per_term=config.top_n_per_term,
        use_hierarchy=config.use_hierarchy,
        use_medcat=config.use_medcat,
    )


if __name__ == "__main__":
    config = ExpandConfig(max_concepts=50)
    results = expand_concepts("meningioma", config=config)
