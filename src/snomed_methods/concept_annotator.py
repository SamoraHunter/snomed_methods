#!/usr/bin/env python3
"""
Clinical Concept Annotator Module

Maps free-text clinical notes to SNOMED CT concepts with evidence tracking.

Enables EHR integration by converting patient narratives, clinical notes,
and other free-text inputs into structured SNOMED CT concepts.
"""

from typing import Dict, List, Optional, Tuple


class AnnotationResult:
    """Container for annotation results with evidence tracking."""

    def __init__(self):
        self.concept_matches: List[MatchedConcept] = []
        self.evidence_trail: Dict[str, any] = {}
        self.text_terms: List[Tuple[str, float]] = []

    @property
    def concepts(self) -> List["MatchedConcept"]:
        return sorted(
            self.concept_matches,
            key=lambda c: c.total_score,
            reverse=True,
        )

    @property
    def top_concepts(self) -> List["MatchedConcept"]:
        """Return top 10 matched concepts."""
        return self.concepts[:10]

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "text_terms": [{"term": t, "score": s} for t, s in self.text_terms],
            "concepts": [
                c.to_dict()
                for c in sorted(
                    self.concept_matches,
                    key=lambda x: x.total_score,
                    reverse=True,
                )
            ],
            "evidence_trail": self.evidence_trail,
        }

    def __len__(self) -> int:
        return len(self.concept_matches)

    def __repr__(self) -> str:
        return (
            f"AnnotationResult(concepts={len(self)}, " f"terms={len(self.text_terms)})"
        )


class MatchedConcept:
    """Represents a matched SNOMED concept with evidence."""

    def __init__(
        self,
        concept_id: str,
        concept_name: str,
    ):
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.term_scores: Dict[str, float] = {}
        self.hierarchy_score: float = 0.0
        self.embedding_score: float = 0.0
        self.total_score: float = 0.0

    @property
    def best_term(self) -> Optional[Tuple[str, float]]:
        """Return the term with highest match score."""
        if not self.term_scores:
            return None
        best_term = max(self.term_scores.items(), key=lambda x: x[1])
        return (best_term[0], best_term[1])

    def add_term_score(self, term: str, score: float) -> None:
        """Add a term matching score."""
        if term not in self.term_scores:
            self.term_scores[term] = 0.0
        self.term_scores[term] = max(self.term_scores[term], score)

    def compute_total_score(
        self,
        term_weight: float = 0.6,
        hierarchy_weight: float = 0.2,
        embedding_weight: float = 0.2,
    ) -> float:
        """Compute weighted total score."""
        norm_term = min(max(self.term_scores.values()) if self.term_scores else 0, 1.0)
        norm_hier = min(self.hierarchy_score, 1.0)
        norm_embed = min(self.embedding_score, 1.0)

        self.total_score = (
            term_weight * norm_term
            + hierarchy_weight * norm_hier
            + embedding_weight * norm_embed
        )
        return self.total_score

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        evidence = {
            "term_scores": self.term_scores,
            "hierarchy_contribution": round(self.hierarchy_score, 4),
            "embedding_contribution": round(self.embedding_score, 4),
        }
        if hasattr(self, "_best_term") and self._best_term:
            evidence["best_match"] = (
                f"'{self._best_term[0]}' ({self._best_term[1]:.2f})"
            )

        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "total_score": round(self.total_score, 4),
            "evidence": evidence,
        }


class ClinicalConceptAnnotator:
    """Maps free-text clinical notes to SNOMED CT concepts."""

    def __init__(
        self,
        uk_path: Optional[str] = None,
        model_path: Optional[str] = None,
        backend: str = "transformers",
        device: str = "cpu",
    ):
        """Initialize annotator with data paths.

        Args:
            uk_path: Path to UK Clinical RF2 directory
            model_path: Path to embedding model (e.g., SapBERT)
            backend: Embedding backend ("transformers", "hf", or "ollama")
            device: Device for embeddings ("cpu" or "cuda")
        """
        self.uk_path = uk_path
        self.model_path = model_path
        self.backend = backend
        self.device = device

        self._hybrid_search = None
        self._term_lookup = None
        self._embedder = None

    def _get_hybrid_search(self):
        """Get or create HybridSearch instance."""
        if self._hybrid_search is None:
            try:
                from snomed_methods import HybridSearch

                self._hybrid_search = HybridSearch(
                    uk_path=self.uk_path,
                    model_path=self.model_path,
                    backend=self.backend,
                    device=self.device,
                )
            except ImportError:
                pass
        return self._hybrid_search

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess clinical text and extract key terms."""
        import re

        if not text or not isinstance(text, str):
            return []

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Extract clinical terms (words ≥ 3 chars)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        if len(words) <= 5:
            return words

        # For longer texts, extract n-grams and key terms
        ngrams = []
        for i in range(min(len(words) - 1, 0), min(len(words), 3)):
            if i + 2 <= len(words):
                ngrams.append(" ".join(words[i : i + 2]))

        # Add unigrams that are clinical-ish (no numbers/symbols)
        keywords = [w for w in words if len(w) >= 4]

        # Combine and deduplicate
        return list(dict.fromkeys(ngrams + keywords))[:10]

    def _score_concepts_from_terms(
        self, concepts: Dict[str, str], text_terms: List[str]
    ) -> Dict[str, MatchedConcept]:
        """Score SNOMED concepts based on term matching."""
        scored = {}

        for concept_id, concept_name in concepts.items():
            concept = MatchedConcept(concept_id, concept_name)

            # Score based on name/synonym matches
            all_names = [concept_name.lower()]
            if hasattr(self, "_term_lookup") and self._term_lookup:
                info = self._term_lookup.getconcept_info(concept_id)
                if info and "synonyms" in info:
                    all_names.extend([s.lower() for s in info["synonyms"]])

            for term in text_terms:
                for name in all_names:
                    if term in name or name.startswith(term):
                        score = 0.8 + (len(term) / len(name) * 0.2)
                        concept.add_term_score(term, score)
                        break

            scored[concept_id] = concept

        return scored

    def annotate(
        self,
        text: str,
        top_k: int = 10,
        max_concepts: int = 50,
    ) -> AnnotationResult:
        """Annotate clinical text with SNOMED concepts.

        Args:
            text: Clinical text to annotate
            top_k: Number of results to return
            max_concepts: Maximum concepts to consider

        Returns:
            AnnotationResult with matched concepts and evidence
        """
        result = AnnotationResult()

        if not text or not isinstance(text, str):
            return result

        # Preprocess text into clinical terms
        text_terms = self._preprocess_text(text)
        result.text_terms = [(t, 1.0) for t in text_terms[:5]]

        if not text_terms:
            return result

        # Use hybrid search to find related concepts
        _hybrid_search = None
        hybrid_search_available = None
        try:
            from snomed_methods import HybridSearch

            hybrid_search_available = HybridSearch
        except ImportError:
            pass

        if hybrid_search_available is None and self._hybrid_search is None:
            return result

        hybrid = self._get_hybrid_search()
        if hybrid is None:
            return result

        # Perform search with text as query
        try:
            search_result = hybrid.search(
                " ".join(text_terms),
                top_k=max_concepts,
                term_weight=0.6,
                hierarchy_weight=0.2,
                embedding_weight=0.2,
            )
        except Exception:
            return result

        # Score concepts based on term matches
        scored_concepts = self._score_concepts_from_terms(
            {c: n for c, n, _ in search_result.results[:max_concepts]},
            text_terms,
        )

        # Compute scores and build results
        for _concept_id, concept in scored_concepts.items():
            concept.compute_total_score()
            result.concept_matches.append(concept)

        result.evidence_trail = {
            "input_text": text,
            "extracted_terms": text_terms[:5],
            "total_search_results": len(search_result.results) if search_result else 0,
        }

        # Sort by score and take top_k
        result.concept_matches.sort(key=lambda c: c.total_score, reverse=True)
        result.concept_matches = result.concept_matches[:top_k]

        return result

    def batch_annotate(
        self,
        texts: List[str],
        top_k: int = 10,
        max_concepts: int = 50,
    ) -> Dict[str, AnnotationResult]:
        """Annotate multiple texts.

        Args:
            texts: List of clinical texts to annotate
            top_k: Number of results per text
            max_concepts: Maximum concepts per search

        Returns:
            Dictionary mapping input index or preview to result
        """
        results = {}

        for i, text in enumerate(texts):
            key = f"text_{i}" if len(str(i)) < 10 else text[:20] + "..."
            results[key] = self.annotate(text, top_k=top_k, max_concepts=max_concepts)

        return results


def annotate_text(
    text: str,
    uk_path: Optional[str] = None,
    model_path: Optional[str] = None,
    backend: str = "transformers",
    device: str = "cpu",
    top_k: int = 10,
) -> AnnotationResult:
    """Convenience function to annotate clinical text.

    Args:
        text: Clinical text to annotate
        uk_path: Path to UK Clinical RF2 directory
        model_path: Path to embedding model
        backend: Embedding backend
        device: Device for embeddings
        top_k: Number of results

    Returns:
        AnnotationResult with matched concepts
    """
    annotator = ClinicalConceptAnnotator(
        uk_path=uk_path,
        model_path=model_path,
        backend=backend,
        device=device,
    )
    return annotator.annotate(text, top_k=top_k)


def batch_annotate_texts(
    texts: List[str],
    uk_path: Optional[str] = None,
    model_path: Optional[str] = None,
    backend: str = "transformers",
    device: str = "cpu",
    top_k: int = 10,
) -> Dict[str, AnnotationResult]:
    """Annotate multiple clinical texts.

    Args:
        texts: List of clinical texts
        uk_path: Path to UK Clinical RF2 directory
        model_path: Path to embedding model
        backend: Embedding backend
        device: Device for embeddings
        top_k: Number of results per text

    Returns:
        Dictionary mapping indexes to AnnotationResults
    """
    annotator = ClinicalConceptAnnotator(
        uk_path=uk_path,
        model_path=model_path,
        backend=backend,
        device=device,
    )
    return annotator.batch_annotate(texts, top_k=top_k)
