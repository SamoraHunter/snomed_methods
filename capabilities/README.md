# Machine-Readable Capabilities Index

This directory contains JSON files that describe all available capabilities in `snomed_methods` in a machine-readable format.

## Available Capability Files

| File | Category | Description |
|------|----------|-------------|
| `term_lookup.json` | Lookup | Term-based search of SNOMED CT descriptions with exact and fuzzy matching |
| `hierarchy_expansion.json` | Expansion | Traverse parent-child relationships in the SNOMED CT ontology |
| `semantic_search.json` | Semantic | Comprehensive concept discovery using combined term matching, hierarchy expansion, and optional MedCAT similarity |
| `embedding_generation.json` | Embeddings | Generate vector embeddings for clinical concepts using LLMs (HuggingFace, SentenceTransformers, Ollama) |
| `vector_search.json` | Search | FAISS-based similarity search for clinical concept embeddings |
| `mapping.json` | Mapping | Map SNOMED CT concepts to ICD-10, LOINC, RxNorm, and UMLS CUIs |

## File Format

Each JSON file contains:

- **name**: Human-readable capability name
- **version**: Capability version (typically "1.0")
- **category**: Functional category
- **description**: Detailed description of the capability
- **classes**: List of class definitions with methods, parameters, and returns
- **functions**: List of standalone functions (if any)
- **example_usage**: Code snippets demonstrating typical usage

## Usage

These files can be consumed by:
- Documentation generators
- API explorers
- IDE integrations
- Knowledge graph construction
- Capability-based system analysis

Example Python parser:

```python
import json
from pathlib import Path

def load_capabilities(cap_dir="capabilities"):
    """Load all capability definitions."""
    capabilities = {}

    for cap_file in Path(cap_dir).glob("*.json"):
        with open(cap_file) as f:
            data = json.load(f)
            capabilities[data["name"]] = data

    return capabilities

# Load and explore capabilities
caps = load_capabilities()
for name, cap in caps.items():
    print(f"{name}: {cap['description']}")
```

## Capability Categories

### 1. Term Lookup (`term_lookup.json`)
- Search by exact substring matching
- Fuzzy string matching (requires `rapidfuzz`)
- Batch term processing
- Case-sensitive/insensitive options

### 2. Hierarchy Expansion (`hierarchy_expansion.json`)
- Local expansion using SNOMED RF2 relationship data
- Snowstorm API integration for remote access
- Multi-level recursive expansion
- Parent and child traversal separately or combined

### 3. Semantic Search (`semantic_search.json`)
- Combined term matching + hierarchy expansion
- Optional MedCAT semantic similarity
- Configurable search depth and breadth
- Results with metrics including core vs expanded counts

### 4. Embedding Generation (`embedding_generation.json`)
- Multiple backends: HuggingFace, Transformers, Ollama
- Batch processing with checkpoint support
- Export to multiple formats (Pickle, NPZ, CSV, Parquet)
- Load concepts from MedCAT or SNOMED directly

### 5. Vector Search (`vector_search.json`)
- FAISS integration for fast similarity search
- Exact (FlatIP) and approximate (HNSW) index types
- Query by text (auto-embedding) or pre-computed vector
- Configurable result count and similarity threshold

### 6. Mapping (`mapping.json`)
- SNOMED CT to ICD-10 mapping via ExtendedMap and SimpleMap refsets
- SNOMED CT to LOINC mapping via SimpleMap refsets
- SNOMED CT to RxNorm (via UMLS CUIs)
- Bidirectional UMLS CUI mapping with confidence scores

## Recent Additions

### September 2026 Updates
- Added `mapping.json` capability definition for SNOMED CT ↔ ICD-10/LOINC/RxNorm/UMLS mappings

### July 2026 Updates
- Added `semantic_filter_results` function to `hybrid_search` module for post-search filtering by semantic categories
- Enhanced `HybridSearch.search()` with `semantic_filter` parameter for filtering results by SNOMED CT semantic types (disorder, finding, procedure, event, body structure, substance, organism)

## Version History

- **v1.0** (Current): Initial release covering all main modules
