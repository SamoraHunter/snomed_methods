# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
from snomed_methods.concept_annotator import (
    AnnotationResult,
    ClinicalConceptAnnotator,
    MatchedConcept,
    annotate_text,
    batch_annotate_texts,
)
from snomed_methods.concept_mapper import (
    ConceptMapper,
    batch_map_concepts,
    map_concept,
)
from snomed_methods.config import SnomedConfig, get_config
from snomed_methods.hybrid_search import (
    HybridSearch,
    SearchResult,
    semantic_filter_results,
)
from snomed_methods.llm_concept_embedder import (
    ClinicalConceptEmbedder,
    ConceptVectorSearch,
    load_concepts_from_medcat,
)
from snomed_methods.progressive_expansion import (
    ExpansionResult,
    ProgressiveExpansionPipeline,
    expand_progressive,
)
from snomed_methods.semantic_expansion import (
    SearchResults,
    SemanticSearch,
    expand_concepts,
)
from snomed_methods.snomed_methods_v1 import SnomedRelations
from snomed_methods.snomed_term_lookup import (
    SnomedTermLookup,
    create_term_lookup_from_directory,
)


def _get_umls_mapper() -> dict:
    """Lazy import for UMLSCIMapper to avoid torch dependency."""
    try:
        from snomed_methods.umlsci_mapper import (
            UMLSCIMapper,
            batch_map_to_umls,
            map_from_umls,
            map_to_umls,
        )

    except ImportError:
        return {}
    return {
        "UMLSCIMapper": UMLSCIMapper,
        "batch_map_to_umls": batch_map_to_umls,
        "map_from_umls": map_from_umls,
        "map_to_umls": map_to_umls,
    }


__all__ = [
    "AnnotationResult",
    "ClinicalConceptAnnotator",
    "ClinicalConceptEmbedder",
    "ConceptMapper",
    "ConceptVectorSearch",
    "ExpansionResult",
    "HybridSearch",
    "MatchedConcept",
    "ProgressiveExpansionPipeline",
    "SearchResult",
    "SearchResults",
    "SemanticSearch",
    "SnomedConfig",
    "SnomedRelations",
    "SnomedTermLookup",
    "annotate_text",
    "batch_annotate_texts",
    "batch_map_concepts",
    "create_term_lookup_from_directory",
    "expand_concepts",
    "expand_progressive",
    "get_config",
    "load_concepts_from_medcat",
    "map_concept",
]

# Add UMLSCIMapper components if available
_umls_components = _get_umls_mapper()
__all__ += list(_umls_components.keys())

globals().update(_umls_components)
