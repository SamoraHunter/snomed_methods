# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] - 2026-09-12

### Added
- Initial release of SNOMED Methods library
- `SnomedRelations` class for SNOMED CT concept hierarchy traversal
- `get_children()` and `get_parents()` methods for relationship navigation
- `expand_codes_local()` and `expand_codes_snowstorm()` for code expansion
- Recursive code expansion with configurable depth via `recursive_code_expansion()`
- MedCAT integration for concept similarity searches
- Snowstorm API support for remote SNOMED CT access
- `SnomedTermLookup` class for term-based concept searching
- Exact and fuzzy string matching via `find_concepts_by_term()` and `find_concepts_by_term_fuzzy()`
- Batch term search functionality via `find_concepts_batch()`
- Concept info retrieval via `getconcept_info()`
- `HybridSearch` module combining term matching, hierarchy expansion, and embeddings
- `SemanticSearch` class for multi-strategy concept expansion
- LLM-based clinical concept embedding via `ClinicalConceptEmbedder`
- Multiple backends: Hugging Face Transformers, SentenceTransformers, Ollama
- FAISS-based vector similarity search via `ConceptVectorSearch`
- `load_concepts_from_medcat()` and `load_concepts_from_cdb()` helper functions

### Features
- Navigate parent-child relationships in SNOMED CT hierarchy
- Expand concept codes recursively through hierarchy traversal
- Term matching with case-insensitive substring, prefix, and fuzzy matching
- Hybrid search combining multiple strategies with configurable re-ranking
- Local Hugging Face Transformers models (e.g., SapBERT) support
- Ollama API compatibility for remote inference
- FAISS index building for efficient similarity search
- Batch processing with checkpoint support

### Fixes
- Various bug fixes and stability improvements from initial development

[Unreleased]: https://github.com/SNOMED-Methods/snomed-methods/compare/v0.1.0...HEAD
[v0.1.0]: https://github.com/SNOMED-Methods/snomed-methods/tree/v0.1.0
