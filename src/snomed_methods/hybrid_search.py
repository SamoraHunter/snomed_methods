#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""Hybrid Search Module for SNOMED CT.

Combines multiple search strategies with configurable re-ranking:
1. Term matching (BM25-style substring/fuzzy matching)
2. Hierarchy expansion (parent-child traversal)
3. Embedding similarity (semantic vector search)

Results are re-ranked using weighted combination of scores from each strategy.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np

try:
    from snomed_methods.llm_concept_embedder import (
        ClinicalConceptEmbedder,
        ConceptVectorSearch,
        load_concepts_from_medcat,
    )
except ImportError:
    ClinicalConceptEmbedder = None  # type: ignore[assignment]
    ConceptVectorSearch = None  # type: ignore[assignment]
    load_concepts_from_medcat = None  # type: ignore[assignment]

try:
    from snomed_methods.snomed_term_lookup import create_term_lookup_from_directory
except ImportError:
    create_term_lookup_from_directory = None  # type: ignore[assignment]

try:
    from snomed_methods.snomed_methods_v1 import SnomedRelations
except ImportError:
    SnomedRelations = None  # type: ignore[assignment]

try:
    from medcat.cat import CAT
except ImportError:
    CAT = None  # type: ignore[assignment]


class HybridSearch:
    """SNOMED CT hybrid search combining term matching, hierarchy expansion,
    and embedding similarity with configurable re-ranking.
    """

    MIN_TERM_LENGTH = 2

    @dataclass
    class SearchConfig:
        top_k: int = 20
        term_weight: float = 0.3
        hierarchy_weight: float = 0.2
        embedding_weight: float = 0.5
        max_hierarchy_nodes: int = 100
        semantic_filter: list[str] | None = None

    def __init__(
        self,
        uk_path: str | None = None,
        medcat_path: str | None = None,
        model_path: str | None = None,
        backend: str = "transformers",
        device: str = "cpu",
    ) -> None:
        """Initialize hybrid search with data paths.

        Args:
            uk_path: Path to UK Clinical RF2 directory
            medcat_path: Path to MedCAT model pack (for semantic similarity)
            model_path: Path to embedding model (e.g., SapBERT)
            backend: Embedding backend ("transformers", "hf", or "ollama")
            device: Device for embedding inference ("cpu" or "cuda")

        """
        self.uk_path = uk_path
        self.medcat_path = medcat_path
        self.model_path = model_path
        self.backend = backend
        self.device = device

        self._term_lookup = None
        self._snomed_relations = None
        self._embedder = None
        self.search_engine = None
        self._cui_to_embedding = None

        self._init_embedder()
        self._init_term_lookup()
        self._init_hierarchy()
        self._init_medcat()
        self._load_cached_embeddings_if_available()

    def _init_embedder(self) -> None:
        """Initialize embedding model and search engine."""
        if ClinicalConceptEmbedder is not None and self.model_path is not None:
            self._embedder = ClinicalConceptEmbedder(
                model_name_or_path=self.model_path,
                backend=self.backend,
                device=self.device,
            )

    def _init_term_lookup(self) -> None:
        """Initialize term lookup module."""
        if create_term_lookup_from_directory is not None:
            if self.uk_path is None:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                self.uk_path = str(
                    project_root
                    / "uk_sct2cl_42.2.0"
                    / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
                )

            if Path(self.uk_path).exists():
                self._term_lookup = create_term_lookup_from_directory(self.uk_path)

    def _init_hierarchy(self) -> None:
        """Initialize hierarchy expansion."""
        if SnomedRelations is not None:
            if self.uk_path is None:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                self.uk_path = str(
                    project_root
                    / "uk_sct2cl_42.2.0"
                    / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
                )

            rel_file = (
                Path(self.uk_path)
                / "Full"
                / "Terminology"
                / "sct2_Relationship_UKCLFull_GB1000000_20260603.txt"
            )

            if Path(rel_file).exists():
                self._snomed_relations = SnomedRelations(snomed_rf2_full_path=rel_file)

    def _init_medcat(self) -> None:
        """Initialize MedCAT model for semantic similarity."""
        self._medcat = None
        if CAT is not None:
            if self.medcat_path is not None and Path(self.medcat_path).exists():
                self._medcat = CAT.load_model_pack(self.medcat_path)
            else:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                default_medcat_path = str(
                    project_root
                    / "model_packs"
                    / "medcat_model_pack_422d1d38fc58f158.zip",
                )
                if Path(default_medcat_path).exists():
                    self._medcat = CAT.load_model_pack(default_medcat_path)

    def _load_cached_embeddings_if_available(self) -> None:
        """Load cached embeddings from disk to avoid regeneration."""
        if ConceptVectorSearch is None:
            return

        if self._cui_to_embedding is not None:
            return

        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        cached_embeddings_path = str(
            project_root / "tests" / "notebooks" / "outputs" / "concept_embeddings.pkl",
        )
        if Path(cached_embeddings_path).exists():
            with contextlib.suppress(Exception):
                self._cui_to_embedding = ConceptVectorSearch.load_embeddings_from_file(
                    cached_embeddings_path,
                )

    def search(  # noqa: C901
        self,
        query: str,
        *,
        config: SearchConfig | None = None,
    ) -> SearchResult:
        """Search for related SNOMED concepts using hybrid approach.

        Args:
            query: Input term(s) to search for
            config: Search configuration with weights and parameters

        Returns:
            SearchResult object with ranked concepts and scores

        """
        if self._term_lookup is None:
            msg = "Term lookup not available. UK Clinical RF2 path required."
            raise ValueError(
                msg,
            )

        if self._snomed_relations is None:
            msg = "Hierarchy data not available. Relationship file required."
            raise ValueError(
                msg,
            )

        results = SearchResult()

        if config is None:
            config = self.SearchConfig()

        query_lower = query.lower()
        query_terms = [query_lower.strip()]

        if " " in query_lower:
            parts = query_lower.split()
            query_terms.extend(parts)

        term_results, cui_to_term_name = self._term_search(query_terms)
        results.term_matches = len(term_results)
        results.cui_to_term.update(cui_to_term_name)

        hierarchy_cuis, hierarchy_scores = self._hierarchy_search(
            list(term_results.keys())[:10],
            config.max_hierarchy_nodes,
        )
        results.hierarchy_matches = len(hierarchy_cuis)
        for cui, score in zip(hierarchy_cuis, hierarchy_scores):
            if cui not in results.cui_scores:
                results.cui_scores[cui] = {}
            results.cui_scores[cui]["hierarchy"] = score

        embedding_results, embed_scores = self._embedding_search(
            query,
            config.top_k * 2,
        )
        results.embedding_matches = len(embedding_results)
        for i, (cui, name) in enumerate(embedding_results):
            if cui not in results.cui_to_term:
                results.cui_to_term[cui] = name
            if cui not in results.cui_scores:
                results.cui_scores[cui] = {}
            results.cui_scores[cui]["embedding"] = embed_scores[i]

        final_results = self._rank_results(
            results.cui_scores,
            config.term_weight,
            config.hierarchy_weight,
            config.embedding_weight,
        )

        for cui, combined_score in final_results[: config.top_k]:
            results.results.append(
                (cui, results.cui_to_term.get(cui, f"CUI: {cui}"), combined_score),
            )

        if config.semantic_filter:
            results = self._apply_semantic_filter(results, config.semantic_filter)

        return results

    def _term_search(
        self,
        query_terms: list[str],
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Perform term-based searching."""
        all_matches = {}
        cui_to_name = {}

        for q_term in query_terms:
            if len(q_term) < self.MIN_TERM_LENGTH:
                continue

            try:
                matches = self._term_lookup.find_concepts_by_term(
                    q_term,
                    ignore_case=True,
                    match_prefix=False,
                )

                for cui, term_name in matches[:50]:
                    score = 1.0 / (len(all_matches) + 1)
                    if cui not in all_matches or score > all_matches[cui]:
                        all_matches[cui] = score
                        cui_to_name[cui] = term_name

                prefix_matches = self._term_lookup.find_concepts_by_term(
                    q_term,
                    ignore_case=True,
                    match_prefix=True,
                )

                for cui, term_name in prefix_matches[:30]:
                    prefix_score = 1.5 / (len(all_matches) + 1)
                    if cui not in all_matches or prefix_score > all_matches[cui]:
                        all_matches[cui] = prefix_score
                        cui_to_name[cui] = term_name

            except (KeyError, TypeError, AttributeError):
                pass

        return all_matches, cui_to_name

    def _hierarchy_search(
        self,
        start_cuis: list[str],
        max_nodes: int,
    ) -> tuple[list[str], list[float]]:
        """Expand concepts via hierarchy traversal."""
        if not self._snomed_relations or not start_cuis:
            return [], []

        visited = set()
        queue = [(str(c), 0) for c in start_cuis if str(c) not in visited]
        results = []
        scores = []

        while queue and len(visited) < max_nodes:
            current, depth = queue.pop(0)

            if current in visited:
                continue
            visited.add(current)

            score = 1.0 / (depth + 1)
            results.append(current)
            scores.append(score)

            try:
                children = self._snomed_relations.get_children(int(current))
                parents = self._snomed_relations.get_parents(int(current))

                queue.extend(
                    (str(child), depth + 1)
                    for child in children[:5]
                    if str(child) not in visited
                )

                queue.extend(
                    (str(parent), depth + 1)
                    for parent in parents[:3]
                    if str(parent) not in visited
                )

            except (ValueError, TypeError):
                continue

        return results, scores

    def _embedding_search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[tuple[str, str]], list[float]]:
        """Search using embedding similarity."""
        if self._embedder is None:
            return [], []

        try:
            if (
                self.search_engine is None
                or not hasattr(self.search_engine, "index")
                or self.search_engine.index is None
            ):
                cui_to_embedding = self._get_all_embeddings()
                if len(cui_to_embedding) == 0:
                    return [], []

                names = {}
                for cui in cui_to_embedding:
                    if (
                        hasattr(self, "_medcat")
                        and self._medcat
                        and hasattr(self._medcat.cdb, "cui2preferred_name")
                    ):
                        name = self._medcat.cdb.cui2preferred_name.get(
                            cui,
                            f"CUI: {cui}",
                        )
                        names[cui] = name

                if ConceptVectorSearch is None:
                    return [], []

                self.search_engine = ConceptVectorSearch(
                    {"embeddings": cui_to_embedding, "names": names},
                    embedder=self._embedder,
                )
                self.search_engine.build_index(index_type="FlatIP")

            results = self.search_engine.search(query, top_k=top_k)

            if len(results) > 0 and isinstance(results[0], tuple):
                return [(str(c), n) for c, n, _ in results], [s for _, _, s in results]
        except (KeyError, TypeError, AttributeError):
            pass

        return [], []

    def _get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Load or generate embeddings for all concepts."""
        if ConceptVectorSearch is None:
            return {}

        if load_concepts_from_medcat is None:
            return {}

        if self._cui_to_embedding is not None:
            result = self._cui_to_embedding
        else:
            result = {}
            try:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                cached_embeddings_path = str(
                    project_root
                    / "tests"
                    / "notebooks"
                    / "outputs"
                    / "concept_embeddings.pkl",
                )
                if Path(cached_embeddings_path).exists():
                    result = ConceptVectorSearch.load_embeddings_from_file(
                        cached_embeddings_path,
                    )
                elif hasattr(self, "_medcat") and self._medcat:
                    concept_df = load_concepts_from_medcat(self._medcat)
                    texts = self._embedder.prepare_concept_text(concept_df)
                    embeddings = self._embedder.generate_embeddings(
                        texts,
                        batch_size=32,
                    )

                    for i, cui in enumerate(concept_df["cui"].tolist()):
                        result[cui] = embeddings[i]

            except (FileNotFoundError, PermissionError, OSError):
                pass

            self._cui_to_embedding = result

        return result

    SEMANTIC_CATEGORIES: ClassVar[dict[str, list[int]]] = {
        "disorder": [404684003, 272379006, 59881007, 441750002],
        "finding": [404684003, 272379006],
        "procedure": [71388002, 387713003, 363679005],
        "event": [410540007, 410541006, 272379006],
        "body structure": [272379006, 363679005, 410608005],
        "substance": [105590001, 763158003],
        "organism": [410607006, 370115009],
        "attribute": [246075003, 246076002],
        "linkage concept": [419890007, 326984008],
        "core concept": [410608005],
    }

    def _apply_semantic_filter(
        self,
        results: SearchResult,
        semantic_categories: list[str],
    ) -> SearchResult:
        """Filter search results by SNOMED semantic categories.

        Args:
            results: SearchResult object to filter
            semantic_categories: List of category names (e.g., 'disorder', 'finding')

        Returns:
            Filtered SearchResult object

        """
        allowed_type_ids = set()
        for category in semantic_categories:
            category_lower = category.lower().strip()
            if category_lower in self.SEMANTIC_CATEGORIES:
                allowed_type_ids.update(self.SEMANTIC_CATEGORIES[category_lower])

        if not allowed_type_ids:
            return results

        filtered_results = []
        cuis_to_keep = []

        for cui, term, score in results.results:
            info = self._term_lookup.getconcept_info(cui)
            if not info or "type_id" not in info:
                continue
            try:
                type_id = int(info["type_id"])
            except (ValueError, TypeError):
                continue
            if type_id in allowed_type_ids:
                filtered_results.append((cui, term, score))
                cuis_to_keep.append(cui)

        filtered_search_result = SearchResult()
        filtered_search_result.results = filtered_results
        for cui in cuis_to_keep:
            filtered_search_result.cui_to_term[cui] = results.cui_to_term.get(
                cui,
                f"CUI: {cui}",
            )
            if cui in results.cui_scores:
                filtered_search_result.cui_scores[cui] = results.cui_scores[cui]

        return filtered_search_result

    def _rank_results(
        self,
        scores: dict[str, dict],
        term_w: float,
        hier_w: float,
        embed_w: float,
    ) -> list[tuple[str, float]]:
        """Rank results using weighted combination of scores."""
        combined_scores = []

        for cui, score_dict in scores.items():
            term_score = score_dict.get("term", 0)
            hier_score = score_dict.get("hierarchy", 0)
            embed_score = score_dict.get("embedding", 0)

            norm_term = min(term_score / 1.0, 1.0) if term_score > 0 else 0
            norm_hier = min(hier_score / 1.0, 1.0) if hier_score > 0 else 0
            norm_embed = embed_score

            combined = term_w * norm_term + hier_w * norm_hier + embed_w * norm_embed

            combined_scores.append((cui, combined))

        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores


class SearchResult:
    """Container for hybrid search results."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, float]] = []
        self.cui_to_term: dict[str, str] = {}
        self.cui_scores: dict[str, dict] = {}
        self.term_matches: int = 0
        self.hierarchy_matches: int = 0
        self.embedding_matches: int = 0

    @property
    def cuis(self) -> list[str]:
        return [c for c, _, _ in self.results]

    @property
    def terms(self) -> list[str]:
        return [t for _, t, _ in self.results]

    @property
    def scores(self) -> list[float]:
        return [s for _, _, s in self.results]

    def to_dict(self) -> dict:
        return {
            "results": [
                {"cui": c, "term": t, "score": round(s, 4)} for c, t, s in self.results
            ],
            "metrics": {
                "term_matches": self.term_matches,
                "hierarchy_matches": self.hierarchy_matches,
                "embedding_matches": self.embedding_matches,
            },
        }

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        m = self.to_dict()["metrics"]
        return (
            f"HybridSearchResult(total={len(self)}, term={m['term_matches']}, "
            f"hier={m['hierarchy_matches']}, embed={m['embedding_matches']})"
        )


def semantic_filter_results(
    results: SearchResult,
    semantic_categories: str | list[str],
    uk_path: str | None = None,
) -> SearchResult:
    """Apply semantic category filtering to existing search results.

    Args:
        results: SearchResult from hybrid search
        semantic_categories: Category name(s) to filter by.
            Options: 'disorder', 'finding', 'procedure', 'event',
            'body structure', 'substance', 'organism'
        uk_path: Path to UK Clinical RF2 directory for term lookups

    Returns:
        Filtered SearchResult with only concepts from specified categories

    """
    if isinstance(semantic_categories, str):
        semantic_categories = [semantic_categories]

    searcher = HybridSearch(uk_path=uk_path)
    return searcher._apply_semantic_filter(results, semantic_categories)


def expand_concepts(
    term_or_terms: str | list[str],
    *,
    uk_path: str | None = None,
    medcat_path: str | None = None,
    model_path: str | None = None,
    config: HybridSearch.SearchConfig | None = None,
) -> SearchResult:
    """Convenience function for hybrid concept expansion.

    Args:
        term_or_terms: Input term(s)
        uk_path: Path to SNOMED UK Clinical RF2
        medcat_path: Path to MedCAT model pack
        model_path: Path to embedding model
        config: Search configuration with weights and parameters

    Returns:
        SearchResult object with ranked concepts

    """
    searcher = HybridSearch(
        uk_path=uk_path,
        medcat_path=medcat_path,
        model_path=model_path,
    )
    return searcher.search(term_or_terms, config=config)
