#!/usr/bin/env python3
"""
Progressive Concept Expansion Pipeline for SNOMED CT.

A multi-stage pipeline that iteratively enriches search results starting from
a user's initial term, returning a comprehensive list of related SNOMED concepts
with evidence tracking and confidence scoring.
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class ExpansionResult:
    """Container for progressive expansion results with evidence tracking."""

    def __init__(
        self,
        query: str,
        stages_executed: List[str],
        scores: Dict[str, float],
        sources: Dict[str, List[str]],
        semantic_type: Optional[str] = None,
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
    def all_cuis(self) -> List[str]:
        all_cuis: Set[str] = set()
        for cuis in self.sources.values():
            all_cuis.update(cuis)
        return sorted(all_cuis)

    def to_dict(self) -> Dict:
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

    def execute(self, query: str, existing_cuis: Set[str]) -> Tuple[List[str], float]:
        raise NotImplementedError

    def get_weight(self) -> float:
        return 1.0


class TermMatchingStage(StageExecutor):
    """Stage 1: Direct fuzzy matching on SNOMED descriptions."""

    def __init__(
        self, uk_path: Optional[str] = None, top_n: int = 50, match_prefix: bool = True
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
            if os.path.exists(uk_path):
                self._lookup = create_term_lookup_from_directory(uk_path)
        except ImportError:
            pass

    def _get_fallback_uk_path(self) -> str:
        # Try absolute path first, then relative to script directory
        uk_paths = [
            "/workspaces/snomed_methods/uk_sct2cl_42.2.0/SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "uk_sct2cl_42.2.0",
                "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            ),
        ]
        for path in uk_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"SNOMED UK Clinical RF2 not found: {uk_paths}")

    def execute(self, query: str, existing_cuis: Set[str]) -> Tuple[List[str], float]:
        self._init_lookup()
        if self._lookup is None:
            return [], 0.0
        new_cuis: List[str] = []
        confidence = 0.0

        # Try exact matches first
        try:
            exact_matches = self._lookup.find_concepts_by_term(query, ignore_case=True)
            for cui, _ in exact_matches[: self.top_n]:
                if cui not in existing_cuis and cui not in new_cuis:
                    new_cuis.append(cui)
                    confidence += 1.0 / (len(new_cuis) + 1)
        except Exception:
            pass

        # Also try prefix matches for better recall
        try:
            prefix_matches = self._lookup.find_concepts_by_term(
                query, ignore_case=True, match_prefix=True
            )
            for cui, _ in prefix_matches[: int(self.top_n * 0.5)]:
                if cui not in existing_cuis and cui not in new_cuis:
                    new_cuis.append(cui)
                    confidence += 1.2 / (len(new_cuis) + 1)
        except Exception:
            pass

        return new_cuis, min(confidence, 1.0)

    def get_weight(self) -> float:
        return 0.5


class HierarchyExpansionStage(StageExecutor):
    """Stage 2: Parent-child traversal via RF2 relationships."""

    def __init__(
        self,
        uk_path: Optional[str] = None,
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
            from snomed_methods.snomed_methods_v1 import SnomedRelations

            rel_file = os.path.join(
                self.uk_path or self._get_fallback_uk_path(),
                "Full",
                "Terminology",
                "sct2_Relationship_UKCLFull_GB1000000_20260603.txt",
            )
            if os.path.exists(rel_file):
                self._relations = SnomedRelations(snomed_rf2_full_path=rel_file)
        except ImportError:
            pass

    def _get_fallback_uk_path(self) -> str:
        # Try absolute path first, then relative to script directory
        uk_paths = [
            "/workspaces/snomed_methods/uk_sct2cl_42.2.0/SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "uk_sct2cl_42.2.0",
                "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            ),
        ]
        for path in uk_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"SNOMED UK Clinical RF2 not found: {uk_paths}")

    def execute(self, query: str, existing_cuis: Set[str]) -> Tuple[List[str], float]:
        self._init_relations()
        if self._relations is None:
            return [], 0.0
        new_cuis: List[str] = []
        confidence = 0.0
        processed: Set[str] = set(existing_cuis)
        queue: List[Tuple[str, int]] = [(str(c), 0) for c in existing_cuis]
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
                for child in children:
                    if str(child) not in processed:
                        queue.append((str(child), depth + 1))
                for parent in parents:
                    if str(parent) not in processed:
                        queue.append((str(parent), depth + 1))
            except (ValueError, TypeError):
                continue
        return new_cuis, min(confidence, 1.0)

    def get_weight(self) -> float:
        return 0.3


class EmbeddingSimilarityStage(StageExecutor):
    """Stage 3: Semantic search using LLM embeddings + FAISS."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "transformers",
        uk_path: Optional[str] = None,
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
            from snomed_methods.llm_concept_embedder import ClinicalConceptEmbedder

            model_path = self.model_path or self._get_default_model_path()
            if os.path.exists(model_path):
                self._embedder = ClinicalConceptEmbedder(
                    model_name_or_path=model_path,
                    backend=self.backend,
                    device="cpu",
                )
        except ImportError:
            pass

    def _init_search_engine(self, seed_cuis: List[str]) -> None:
        if self._search_engine is not None:
            return
        try:
            from snomed_methods.llm_concept_embedder import ConceptVectorSearch
            from snomed_methods.snomed_term_lookup import (
                create_term_lookup_from_directory,
            )

            if self._embedder is None:
                self._init_embedder()
            uk_path = self.uk_path or self._get_fallback_uk_path()
            lookup = create_term_lookup_from_directory(uk_path)
            cui_to_embedding: Dict[str, np.ndarray] = {}
            names: Dict[str, str] = {}
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
        except Exception:
            pass

    def _get_default_model_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(
            project_root, "embedding_models", "SapBERT-from-PubMedBERT-fulltext"
        )

    def _get_fallback_uk_path(self) -> str:
        # Try absolute path first, then relative to script directory
        uk_paths = [
            "/workspaces/snomed_methods/uk_sct2cl_42.2.0/SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "uk_sct2cl_42.2.0",
                "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
            ),
        ]
        for path in uk_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"SNOMED UK Clinical RF2 not found: {uk_paths}")

    def execute(self, query: str, existing_cuis: Set[str]) -> Tuple[List[str], float]:
        if not existing_cuis:
            return [], 0.0
        self._init_embedder()
        if self._search_engine is None:
            self._init_search_engine(list(existing_cuis))
        if self._search_engine is None or self._embedder is None:
            return [], 0.0
        results: List[str] = []
        confidence = 0.0
        seen: Set[str] = set()
        try:
            search_results = self._search_engine.search(query, top_k=self.top_k * 2)
            for cui, _name, score in search_results[: self.top_k]:
                if str(cui) not in existing_cuis and str(cui) not in seen:
                    results.append(str(cui))
                    confidence += float(score)
                    seen.add(str(cui))
        except Exception:
            pass
        return results, min(confidence, 1.0)

    def get_weight(self) -> float:
        return 0.3


class MedCatExpansionStage(StageExecutor):
    """Stage 4: Co-occurrence based expansion using MedCAT."""

    def __init__(
        self, medcat_path: Optional[str] = None, top_n: int = 50, min_sim: float = 0.1
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
            if os.path.exists(medcat_path):
                self._medcat = CAT.load_model_pack(medcat_path)
        except ImportError:
            pass

    def _get_default_medcat_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        return os.path.join(
            project_root, "model_packs", "medcat_model_pack_422d1d38fc58f158.zip"
        )

    def execute(self, query: str, existing_cuis: Set[str]) -> Tuple[List[str], float]:
        self._init_medcat()
        if self._medcat is None or not hasattr(self._medcat, "cdb"):
            return [], 0.0
        new_cuis: List[str] = []
        confidence = 0.0
        seen: Set[str] = set()
        cdb = self._medcat.cdb
        for cui in list(existing_cuis)[:10]:
            try:
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
            except Exception:
                continue
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

    def execute(self, query: str, existing_cuis: Set[str]) -> Tuple[List[str], float]:
        confidence = 0.5
        new_terms: List[str] = []
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

    def _generate_ollama(self, prompt: str) -> List[str]:
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
        except Exception:
            return []

    def _generate_hf(self, prompt: str) -> List[str]:
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
        except Exception:
            return []

    def get_weight(self) -> float:
        return 0.1


class ProgressiveExpansionPipeline:
    """
    Multi-stage pipeline for progressive concept expansion.
    """

    def __init__(
        self,
        uk_path: Optional[str] = None,
        medcat_path: Optional[str] = None,
        model_path: Optional[str] = None,
        backend: str = "transformers",
    ) -> None:
        self.uk_path = uk_path
        self.medcat_path = medcat_path
        self.model_path = model_path
        self.backend = backend
        self.stages = {
            "term": TermMatchingStage(uk_path=uk_path, top_n=100),
            "hierarchy": HierarchyExpansionStage(
                uk_path=uk_path, max_depth=3, max_per_level=20
            ),
            "embedding": EmbeddingSimilarityStage(
                uk_path=uk_path,
                model_path=model_path,
                backend=backend,
                top_k=50,
            ),
            "medcat": MedCatExpansionStage(
                medcat_path=medcat_path, top_n=50, min_sim=0.1
            ),
            "llm": LLMExpansionStage(backend=backend, model_name="qwen2.5-coder"),
        }

    def expand(
        self,
        query: str,
        stages: Optional[List[str]] = None,
        max_concepts: int = 100,
        stage_weights: Optional[Dict[str, float]] = None,
    ) -> "ExpandedExpansionResult":
        if stages is None:
            stages = list(self.stages.keys())
        executed_stages: List[str] = []
        all_cuis: Set[str] = set()
        stage_sources: Dict[str, List[str]] = {}
        stage_scores: Dict[str, float] = {}

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
        stages_executed: List[str],
        scores: Dict[str, float],
        sources: Dict[str, List[str]],
    ) -> None:
        self.query = query
        self.stages_executed = stages_executed
        self.scores = scores
        self.sources = sources
        self._cui_to_name: Dict[str, str] = {}
        self._stage_weights: Dict[str, float] = {
            "term": 0.5,
            "hierarchy": 0.3,
            "embedding": 0.3,
            "medcat": 0.2,
            "llm": 0.1,
        }

    def set_stage_weights(self, weights: Dict[str, float]) -> None:
        self._stage_weights.update(weights)

    @property
    def total_concepts(self) -> int:
        return len({c for cuis in self.sources.values() for c in cuis})

    @property
    def all_cuis(self) -> List[str]:
        all_cuis: Set[str] = set()
        for cuis in self.sources.values():
            all_cuis.update(cuis)
        return sorted(all_cuis)

    def get_concepts_with_names(
        self,
        uk_path: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Get concepts with preferred names."""
        if not self.all_cuis:
            return []

        try:
            from snomed_methods.snomed_term_lookup import (
                create_term_lookup_from_directory,
            )

            uk_path = uk_path or self._get_fallback_uk_path()
            lookup = create_term_lookup_from_directory(uk_path)

            results: List[Tuple[str, str]] = []
            for cui in sorted(self.all_cuis):
                info = lookup.getconcept_info(cui)
                if info and "preferred_name" in info:
                    preferred_name = info["preferred_name"]
                else:
                    preferred_name = f"CUI: {cui}"
                results.append((cui, preferred_name))

            return results
        except Exception:
            return [(c, f"CUI: {c}") for c in self.all_cuis]

    def to_dict(self) -> Dict:
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

    def to_dataframe(self) -> Any:
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        uk_path = os.path.join(
            project_root,
            "uk_sct2cl_42.2.0",
            "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
        )
        if not os.path.exists(uk_path):
            uk_path = (
                "/workspaces/snomed_methods/uk_sct2cl_42.2.0/"
                "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z"
            )
        return uk_path

    def __repr__(self) -> str:
        return (
            f"ExpandedExpansionResult(query='{self.query}', "
            f"concepts={self.total_concepts}, "
            f"stages={self.stages_executed})"
        )


def expand_progressive(
    query: str,
    uk_path: Optional[str] = None,
    medcat_path: Optional[str] = None,
    model_path: Optional[str] = None,
    backend: str = "transformers",
    stages: Optional[List[str]] = None,
    max_concepts: int = 100,
) -> ExpandedExpansionResult:
    """Convenience function for progressive concept expansion."""
    pipeline = ProgressiveExpansionPipeline(
        uk_path=uk_path,
        medcat_path=medcat_path,
        model_path=model_path,
        backend=backend,
    )
    return pipeline.expand(query, stages=stages, max_concepts=max_concepts)
