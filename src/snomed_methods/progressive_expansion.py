#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""
Progressive Concept Expansion Pipeline for SNOMED CT.

A multi-stage pipeline that iteratively enriches search results starting from
a user's initial term, returning a comprehensive list of related SNOMED concepts
with evidence tracking and confidence scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ExpansionConfig:
    """Configuration for progressive expansion pipeline."""

    uk_path: str | None = None
    medcat_path: str | None = None
    model_path: str | None = None
    backend: str = "transformers"
    stages: list[str] | None = None
    max_concepts: int = 100


class ExpansionResult:
    """Container for progressive expansion results with evidence tracking."""

    def __init__(
        self,
        query: str,
        stages_executed: list[str],
        scores: dict[str, float],
        sources: dict[str, list[str]],
        semantic_type: str | None = None,
    ) -> None:
        self.query = query
        self.stages_executed = stages_executed
        self.scores = scores
        self.sources = sources
        self.semantic_type = semantic_type

    @property
    def total_concepts(self) -> int:
        return len({c for cuis in self.sources.values() for c in cuis})

    @property
    def all_cuis(self) -> list[str]:
        all_cuis: set[str] = set()
        for cuis in self.sources.values():
            all_cuis.update(cuis)
        return sorted(all_cuis)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "stages_executed": self.stages_executed,
            "total_concepts": self.total_concepts,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "sources": self.sources,
            "semantic_type": self.semantic_type,
        }

    def __repr__(self) -> str:
        return (
            f"ExpansionResult(query='{self.query}', "
            f"concepts={self.total_concepts}, "
            f"stages={self.stages_executed})"
        )


class StageExecutor:
    """Base class for pipeline stages."""

    def execute(self, query: str, existing_cuis: set[str]) -> tuple[list[str], float]:
        raise NotImplementedError

    def get_weight(self) -> float:
        return 1.0


class TermMatchingStage(StageExecutor):
    """Stage 1: Direct fuzzy matching on SNOMED descriptions."""

    def __init__(
        self,
        uk_path: str | None = None,
        top_n: int = 50,
        *,
        match_prefix: bool = True,
    ) -> None:
        self.uk_path = uk_path
        self.top_n = top_n
        self.match_prefix = match_prefix
        self._lookup = None

    def _init_lookup(self) -> None:
        if self._lookup is not None:
            return
        try:
            from snomed_methods.snomed_term_lookup import (
                create_term_lookup_from_directory,
            )

            uk_path = self.uk_path or self._get_fallback_uk_path()
            uk_path = uk_path or self._get_fallback_uk_path()
            if Path(uk_path).exists():
                self._lookup = create_term_lookup_from_directory(uk_path)
        except ImportError:
            pass

    def _get_fallback_uk_path(self) -> str:
        return str(
            Path(__file__).resolve().parents[1]
            / "uk_sct2cl_42.2.0"
            / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
        )

    def execute(self, query: str, existing_cuis: set[str]) -> tuple[list[str], float]:
        self._init_lookup()
        if self._lookup is None:
            return [], 0.0
        new_cuis: list[str] = []
        confidence = 0.0

        # Try exact matches first
        try:
            exact_matches = self._lookup.find_concepts_by_term(query, ignore_case=True)
            for cui, _ in exact_matches[: self.top_n]:
                if cui not in existing_cuis and cui not in new_cuis:
                    new_cuis.append(cui)
                    confidence += 1.0 / (len(new_cuis) + 1)
        except (ValueError, TypeError, RuntimeError):
            pass

        # Also try prefix matches for better recall
        try:
            prefix_matches = self._lookup.find_concepts_by_term(
                query,
                ignore_case=True,
                match_prefix=True,
            )
            for cui, _ in prefix_matches[: int(self.top_n * 0.5)]:
                if cui not in existing_cuis and cui not in new_cuis:
                    new_cuis.append(cui)
                    confidence += 1.2 / (len(new_cuis) + 1)
        except (ValueError, TypeError, RuntimeError):
            pass

        return new_cuis, min(confidence, 1.0)

    def get_weight(self) -> float:
        return 0.5


class HierarchyExpansionStage(StageExecutor):
    """Stage 2: Parent-child traversal via RF2 relationships."""

    def __init__(
        self,
        uk_path: str | None = None,
        max_depth: int = 3,
        max_per_level: int = 10,
    ) -> None:
        self.uk_path = uk_path
        self.max_depth = max_depth
        self.max_per_level = max_per_level
        self._relations = None

    def _init_relations(self) -> None:
        if self._relations is not None:
            return
        try:
            from snomed_methods.snomed_methods_v1 import (
                SnomedRelations,
            )

            uk_path = self.uk_path or self._get_fallback_uk_path()
            rel_file = (
                Path(uk_path)
                / "Full"
                / "Terminology"
                / "sct2_Relationship_UKCLFull_GB1000000_20260603.txt"
            )
            if rel_file.exists():
                self._relations = SnomedRelations(snomed_rf2_full_path=str(rel_file))
        except ImportError:
            pass

    def _get_fallback_uk_path(self) -> str:
        return str(
            Path(__file__).resolve().parents[1]
            / "uk_sct2cl_42.2.0"
            / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
        )

    def execute(self, _query: str, existing_cuis: set[str]) -> tuple[list[str], float]:
        self._init_relations()
        if self._relations is None:
            return [], 0.0
        new_cuis: list[str] = []
        confidence = 0.0
        processed: set[str] = set(existing_cuis)
        queue: list[tuple[str, int]] = [(str(c), 0) for c in existing_cuis]
        while queue and len(processed) < self.max_per_level * (self.max_depth + 2):
            current, depth = queue.pop(0)
            if depth > self.max_depth:
                continue
            processed.add(current)
            if current not in existing_cuis:
                new_cuis.append(current)
                confidence += 1.0 / (depth + 1)
            try:
                cui_int = int(current)
                children = self._relations.get_children(cui_int)[: self.max_per_level]
                parents = self._relations.get_parents(cui_int)[: self.max_per_level]
                queue.extend(
                    (str(child), depth + 1)
                    for child in children
                    if str(child) not in processed
                )
                queue.extend(
                    (str(parent), depth + 1)
                    for parent in parents
                    if str(parent) not in processed
                )
            except (ValueError, TypeError):
                continue
        return new_cuis, min(confidence, 1.0)

    def get_weight(self) -> float:
        return 0.3


class EmbeddingSimilarityStage(StageExecutor):
    """Stage 3: Semantic search using LLM embeddings + FAISS."""

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "transformers",
        uk_path: str | None = None,
        top_k: int = 50,
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.uk_path = uk_path
        self.top_k = top_k
        self._embedder = None
        self._search_engine = None

    def _init_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from snomed_methods.llm_concept_embedder import (
                ClinicalConceptEmbedder,
            )

            model_path = self.model_path or self._get_default_model_path()
            if Path(model_path).exists():
                self._embedder = ClinicalConceptEmbedder(
                    model_name_or_path=(
                        str(model_path) if isinstance(model_path, Path) else model_path
                    ),
                    backend=self.backend,
                    device="cpu",
                )
        except ImportError:
            pass

    def _init_search_engine(self, seed_cuis: list[str]) -> None:
        if self._search_engine is not None:
            return
        try:
            from snomed_methods.llm_concept_embedder import (
                ConceptVectorSearch,
            )
            from snomed_methods.snomed_term_lookup import (
                create_term_lookup_from_directory,
            )

            if self._embedder is None:
                self._init_embedder()
            uk_path = self.uk_path or self._get_fallback_uk_path()
            lookup = create_term_lookup_from_directory(uk_path)
            cui_to_embedding: dict[str, np.ndarray] = {}
            names: dict[str, str] = {}
            for cui in seed_cuis:
                text = lookup.getconcept_info(cui)
                if text and "preferred_name" in text:
                    names[cui] = text["preferred_name"]
                    embedding = self._embedder.generate_embeddings(
                        [text["preferred_name"]],
                        batch_size=1,
                    )[0]
                    cui_to_embedding[cui] = embedding
            self._search_engine = ConceptVectorSearch(
                {"embeddings": cui_to_embedding, "names": names},
                embedder=self._embedder,
            )
            self._search_engine.build_index(index_type="FlatIP")
        except (ImportError, ValueError, TypeError, RuntimeError):
            pass

    def _get_default_model_path(self) -> str:
        return str(
            Path(__file__).resolve().parents[1]
            / "embedding_models"
            / "SapBERT-from-PubMedBERT-fulltext",
        )

    def _get_fallback_uk_path(self) -> str:
        return str(
            Path(__file__).resolve().parents[1]
            / "uk_sct2cl_42.2.0"
            / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
        )

    def execute(self, query: str, existing_cuis: set[str]) -> tuple[list[str], float]:
        if not existing_cuis:
            return [], 0.0
        self._init_embedder()
        if self._search_engine is None:
            self._init_search_engine(list(existing_cuis))
        if self._search_engine is None or self._embedder is None:
            return [], 0.0
        results: list[str] = []
        confidence = 0.0
        seen: set[str] = set()
        try:
            search_results = self._search_engine.search(query, top_k=self.top_k * 2)
            for cui, _name, score in search_results[: self.top_k]:
                if str(cui) not in existing_cuis and str(cui) not in seen:
                    results.append(str(cui))
                    confidence += float(score)
                    seen.add(str(cui))
        except (ValueError, TypeError, RuntimeError):
            pass
        return results, min(confidence, 1.0)

    def get_weight(self) -> float:
        return 0.3


class MedCatExpansionStage(StageExecutor):
    """Stage 4: Co-occurrence based expansion using MedCAT."""

    def __init__(
        self,
        medcat_path: str | None = None,
        top_n: int = 50,
        min_sim: float = 0.1,
    ) -> None:
        self.medcat_path = medcat_path
        self.top_n = top_n
        self.min_sim = min_sim
        self._medcat = None

    def _init_medcat(self) -> None:
        if self._medcat is not None:
            return
        try:
            from medcat.cat import CAT

            medcat_path = self.medcat_path or self._get_default_medcat_path()
            if Path(medcat_path).exists():
                self._medcat = CAT.load_model_pack(medcat_path)
        except ImportError:
            pass

    def _get_default_medcat_path(self) -> str:
        return str(
            Path(__file__).resolve().parents[1]
            / "model_packs"
            / "medcat_model_pack_422d1d38fc58f158.zip",
        )

    def execute(self, _query: str, existing_cuis: set[str]) -> tuple[list[str], float]:
        self._init_medcat()
        if self._medcat is None or not hasattr(self._medcat, "cdb"):
            return [], 0.0
        new_cuis: list[str] = []
        confidence = 0.0
        seen: set[str] = set()
        cdb = self._medcat.cdb
        for cui in list(existing_cuis)[:10]:
            sim_results = cdb.most_similar(
                cui,
                context_type="long",
                topn=self.top_n,
            )
            for sim_cui, sim_data in sim_results.items():
                if (
                    sim_cui not in existing_cuis
                    and sim_cui not in seen
                    and str(sim_cui) not in new_cuis
                ):
                    sim_score = sim_data.get("sim", 0)
                    if sim_score >= self.min_sim:
                        new_cuis.append(str(sim_cui))
                        confidence += sim_score
                        seen.add(sim_cui)
        return new_cuis, min(confidence, 1.0)

    def get_weight(self) -> float:
        return 0.2


class LLMExpansionStage(StageExecutor):
    """Stage 5: LLM-powered novel concept generation."""

    def __init__(
        self,
        backend: str = "ollama",
        model_name: str = "qwen2.5-coder",
        max_new_terms: int = 10,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.max_new_terms = max_new_terms

    def execute(
        self,
        query: str,
        _existing_cuis: set[str],
    ) -> tuple[list[str], float]:
        confidence = 0.5
        new_terms: list[str] = []
        prompt = (
            f'Given the medical term "{query}", '
            f"suggest {self.max_new_terms} related clinical terms or concepts. "
            "Format as comma-separated list only."
        )
        if self.backend == "ollama":
            new_terms = self._generate_ollama(prompt)
        else:
            new_terms = self._generate_hf(prompt)
        return new_terms[: self.max_new_terms], confidence

    def _generate_ollama(self, prompt: str) -> list[str]:
        try:
            import ollama

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical terminology assistant. "
                            "Respond with only comma-separated terms."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response["message"]["content"].strip()
            return [t.strip() for t in content.split(",") if t.strip()]
        except (ValueError, TypeError, RuntimeError):
            return []

    def _generate_hf(self, prompt: str) -> list[str]:
        try:
            from transformers import pipeline

            generator = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=128,
                temperature=0.3,
            )
            result = generator(prompt)[0]
            text = result["generated_text"].strip()
            if ":" in text:
                text = text.split(":")[-1]
            return [t.strip() for t in text.split(",") if t.strip()]
        except (ValueError, TypeError, RuntimeError):
            return []

    def get_weight(self) -> float:
        return 0.1


class ProgressiveExpansionPipeline:
    """
    Multi-stage pipeline for progressive concept expansion.
    """

    def __init__(
        self,
        uk_path: str | None = None,
        medcat_path: str | None = None,
        model_path: str | None = None,
        backend: str = "transformers",
    ) -> None:
        self.uk_path = uk_path
        self.medcat_path = medcat_path
        self.model_path = model_path
        self.backend = backend
        self.stages = {
            "term": TermMatchingStage(uk_path=uk_path, top_n=100),
            "hierarchy": HierarchyExpansionStage(
                uk_path=uk_path,
                max_depth=3,
                max_per_level=20,
            ),
            "embedding": EmbeddingSimilarityStage(
                uk_path=uk_path,
                model_path=model_path,
                backend=backend,
                top_k=50,
            ),
            "medcat": MedCatExpansionStage(
                medcat_path=medcat_path,
                top_n=50,
                min_sim=0.1,
            ),
            "llm": LLMExpansionStage(backend=backend, model_name="qwen2.5-coder"),
        }

    def expand(
        self,
        query: str,
        stages: list[str] | None = None,
        max_concepts: int = 100,
        stage_weights: dict[str, float] | None = None,
    ) -> ExpandedExpansionResult:
        if stages is None:
            stages = list(self.stages.keys())
        executed_stages: list[str] = []
        all_cuis: set[str] = set()
        stage_sources: dict[str, list[str]] = {}
        stage_scores: dict[str, float] = {}

        for stage_name in stages:
            if stage_name not in self.stages:
                continue
            stage = self.stages[stage_name]
            new_cuis, confidence = stage.execute(query, all_cuis)
            if new_cuis:
                executed_stages.append(stage_name)
                stage_sources[stage_name] = new_cuis
                stage_scores[stage_name] = confidence
                all_cuis.update(new_cuis)
            if len(all_cuis) >= max_concepts:
                break

        result = ExpandedExpansionResult(
            query=query,
            stages_executed=executed_stages,
            scores=stage_scores,
            sources=stage_sources,
        )

        if stage_weights is not None:
            result.set_stage_weights(stage_weights)

        return result


class ExpandedExpansionResult:
    """Enhanced result container with CUI and preferred name."""

    def __init__(
        self,
        query: str,
        stages_executed: list[str],
        scores: dict[str, float],
        sources: dict[str, list[str]],
    ) -> None:
        self.query = query
        self.stages_executed = stages_executed
        self.scores = scores
        self.sources = sources
        self._cui_to_name: dict[str, str] = {}
        self._stage_weights: dict[str, float] = {
            "term": 0.5,
            "hierarchy": 0.3,
            "embedding": 0.3,
            "medcat": 0.2,
            "llm": 0.1,
        }

    def set_stage_weights(self, weights: dict[str, float]) -> None:
        self._stage_weights.update(weights)

    @property
    def total_concepts(self) -> int:
        return len({c for cuis in self.sources.values() for c in cuis})

    @property
    def all_cuis(self) -> list[str]:
        all_cuis: set[str] = set()
        for cuis in self.sources.values():
            all_cuis.update(cuis)
        return sorted(all_cuis)

    def get_concepts_with_names(
        self,
        uk_path: str | None = None,
    ) -> list[tuple[str, str]]:
        """Get concepts with preferred names."""
        if not self.all_cuis:
            return []

        try:
            from snomed_methods.snomed_term_lookup import (
                create_term_lookup_from_directory,
            )

            uk_path = uk_path or self._get_fallback_uk_path()
            lookup = create_term_lookup_from_directory(uk_path)

            results: list[tuple[str, str]] = []
            for cui in sorted(self.all_cuis):
                info = lookup.getconcept_info(cui)
                if info and "preferred_name" in info:
                    preferred_name = info["preferred_name"]
                else:
                    preferred_name = f"CUI: {cui}"
                results.append((cui, preferred_name))

        except (ImportError, ValueError, TypeError, RuntimeError):
            return [(c, f"CUI: {c}") for c in self.all_cuis]

        return results

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        concepts = self.get_concepts_with_names()
        return {
            "query": self.query,
            "stages_executed": self.stages_executed,
            "total_concepts": len(concepts),
            "concepts": [{"cui": c, "name": n} for c, n in concepts[:10]],
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "sources": self.sources,
        }

    def to_dataframe(self) -> object | None:
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd

            concepts = self.get_concepts_with_names()
            rows = []
            for cui, name in concepts:
                row = {
                    "cui": cui,
                    "preferred_name": name,
                    "stages": ";".join(
                        s for s, cuis in self.sources.items() if cui in cuis
                    ),
                }
                for stage in self.stages_executed:
                    if cui in self.sources.get(stage, []):
                        row[f"score_{stage}"] = self.scores.get(stage, 0)
                rows.append(row)

            return pd.DataFrame(rows)
        except ImportError:
            return None

    @staticmethod
    def _get_fallback_uk_path() -> str:
        return str(
            Path(__file__).resolve().parents[1]
            / "uk_sct2cl_42.2.0"
            / "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
        )

    def __repr__(self) -> str:
        return (
            f"ExpandedExpansionResult(query='{self.query}', "
            f"concepts={self.total_concepts}, "
            f"stages={self.stages_executed})"
        )


def expand_progressive(
    query: str,
    config: ExpansionConfig | None = None,
) -> ExpandedExpansionResult:
    """Convenience function for progressive concept expansion."""
    if config is None:
        config = ExpansionConfig()
    pipeline = ProgressiveExpansionPipeline(
        uk_path=config.uk_path,
        medcat_path=config.medcat_path,
        model_path=config.model_path,
        backend=config.backend,
    )
    return pipeline.expand(
        query,
        stages=config.stages,
        max_concepts=config.max_concepts,
    )
