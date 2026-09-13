"""Unit tests for ClinicalConceptAnnotator module."""

import pytest


class TestMatchedConcept:
    """Tests for MatchedConcept class."""

    def test_initialization(self):
        """Test concept initialization."""
        from snomed_methods.concept_annotator import MatchedConcept

        concept = MatchedConcept("409681000000102", "Meningioma")
        assert concept.concept_id == "409681000000102"
        assert concept.concept_name == "Meningioma"

    def test_add_term_score(self):
        """Test adding term scores."""
        from snomed_methods.concept_annotator import MatchedConcept

        concept = MatchedConcept("409681000000102", "Meningioma")
        concept.add_term_score("meningioma", 0.8)
        concept.add_term_score("brain tumor", 0.5)

        assert len(concept.term_scores) == 2
        assert concept.term_scores["meningioma"] == 0.8

    def test_add_term_score_overwrites_lower(self):
        """Test term score overwrites if higher."""
        from snomed_methods.concept_annotator import MatchedConcept

        concept = MatchedConcept("409681000000102", "Meningioma")
        concept.add_term_score("meningioma", 0.5)
        concept.add_term_score("meningioma", 0.9)

        assert concept.term_scores["meningioma"] == 0.9

    def test_best_term(self):
        """Test best term property."""
        from snomed_methods.concept_annotator import MatchedConcept

        concept = MatchedConcept("409681000000102", "Meningioma")
        assert concept.best_term is None

        concept.add_term_score("meningioma", 0.9)
        concept.add_term_score("brain tumor", 0.5)

        best = concept.best_term
        assert best[0] == "meningioma"
        assert best[1] == 0.9

    def test_compute_total_score(self):
        """Test total score computation."""
        from snomed_methods.concept_annotator import MatchedConcept

        concept = MatchedConcept("409681000000102", "Meningioma")
        concept.add_term_score("meningioma", 0.8)
        concept.hierarchy_score = 0.5
        concept.embedding_score = 0.6

        score = concept.compute_total_score(
            term_weight=0.6, hierarchy_weight=0.2, embedding_weight=0.2
        )

        assert score > 0


class TestAnnotationResult:
    """Tests for AnnotationResult class."""

    def test_initialization(self):
        """Test result initialization."""
        from snomed_methods.concept_annotator import AnnotationResult

        result = AnnotationResult()
        assert len(result) == 0
        assert result.text_terms == []
        assert result.evidence_trail == {}

    def test_add_concepts(self):
        """Test adding concepts to result."""
        from snomed_methods.concept_annotator import (
            AnnotationResult,
            MatchedConcept,
        )

        result = AnnotationResult()
        concept1 = MatchedConcept("409681000000102", "Meningioma")
        concept1.total_score = 0.9

        concept2 = MatchedConcept("154621002", "Bronchitis")
        concept2.total_score = 0.7

        result.concept_matches.extend([concept1, concept2])

        assert len(result) == 2
        top = result.top_concepts
        assert top[0].concept_id == "409681000000102"
        assert top[1].concept_id == "154621002"

    def test_to_dict(self):
        """Test result to dictionary conversion."""
        from snomed_methods.concept_annotator import AnnotationResult

        result = AnnotationResult()
        result.text_terms = [("chest pain", 1.0)]
        result.evidence_trail = {"test": True}

        d = result.to_dict()

        assert "text_terms" in d
        assert "concepts" in d
        assert "evidence_trail" in d


class TestClinicalConceptAnnotator:
    """Tests for ClinicalConceptAnnotator class."""

    def test_initialization(self):
        """Test annotator initialization."""
        from snomed_methods.concept_annotator import (
            ClinicalConceptAnnotator,
        )

        annotator = ClinicalConceptAnnotator()
        assert annotator.uk_path is None
        assert annotator.model_path is None

    def test_preprocess_text_short(self):
        """Test preprocessing short text."""
        from snomed_methods.concept_annotator import (
            ClinicalConceptAnnotator,
        )

        annotator = ClinicalConceptAnnotator()
        terms = annotator._preprocess_text("chest pain")

        assert len(terms) <= 5

    def test_preprocess_text_long(self):
        """Test preprocessing longer text."""
        from snomed_methods.concept_annotator import (
            ClinicalConceptAnnotator,
        )

        annotator = ClinicalConceptAnnotator()
        text = "patient presents with chest pain in left arm and shortness of breath"
        terms = annotator._preprocess_text(text)

        assert len(terms) > 0
        assert any("chest" in t or "pain" in t for t in terms)

    def test_annotate_empty_text(self):
        """Test annotate with empty text."""
        from snomed_methods.concept_annotator import (
            ClinicalConceptAnnotator,
        )

        annotator = ClinicalConceptAnnotator()
        result = annotator.annotate("")

        assert len(result) == 0
        assert result.text_terms == []

    def test_annotate_none_text(self):
        """Test annotate with None text."""
        from snomed_methods.concept_annotator import (
            ClinicalConceptAnnotator,
        )

        annotator = ClinicalConceptAnnotator()
        result = annotator.annotate(None)

        assert len(result) == 0

    def test_batch_annotate(self):
        """Test batch annotation."""
        from snomed_methods.concept_annotator import (
            ClinicalConceptAnnotator,
        )

        annotator = ClinicalConceptAnnotator()
        results = annotator.batch_annotate(["chest pain", "headache"])

        assert len(results) == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_annotate_text(self):
        """Test annotate_text convenience function."""
        from snomed_methods.concept_annotator import AnnotationResult, annotate_text

        result = annotate_text("patient has fever")
        assert isinstance(result, AnnotationResult)

    def test_batch_annotate_texts(self):
        """Test batch_annotate_texts convenience function."""
        from snomed_methods.concept_annotator import batch_annotate_texts

        result = batch_annotate_texts(["chest pain", "headache"])
        assert isinstance(result, dict)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
