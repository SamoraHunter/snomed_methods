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


def _get_umls_mapper():
    """Lazy import for UMLSCIMapper to avoid torch dependency."""
    try:
        from snomed_methods.umlsci_mapper import (
            UMLSCIMapper,
            batch_map_to_umls,
            map_from_umls,
            map_to_umls,
        )

        return {
            "UMLSCIMapper": UMLSCIMapper,
            "batch_map_to_umls": batch_map_to_umls,
            "map_from_umls": map_from_umls,
            "map_to_umls": map_to_umls,
        }
    except ImportError:
        return {}


__all__ = [
    "SnomedRelations",
    "SnomedTermLookup",
    "create_term_lookup_from_directory",
    "HybridSearch",
    "SearchResult",
    "ClinicalConceptEmbedder",
    "ConceptVectorSearch",
    "load_concepts_from_medcat",
    "ExpansionResult",
    "ProgressiveExpansionPipeline",
    "expand_progressive",
    "SemanticSearch",
    "SearchResults",
    "expand_concepts",
    "AnnotationResult",
    "MatchedConcept",
    "ClinicalConceptAnnotator",
    "annotate_text",
    "batch_annotate_texts",
    "ConceptMapper",
    "map_concept",
    "batch_map_concepts",
]

# Add UMLSCIMapper components if available
_umls_components = _get_umls_mapper()
__all__.extend(_umls_components.keys())

globals().update(_umls_components)
