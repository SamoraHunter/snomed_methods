# SNOMED Methods

A Python library for working with SNOMED CT (Systematized Nomenclature of Medicine - Clinical Terms).

## Documentation Index

- [Quick Start](#quick-start) - Installation and basic usage
- [Features](#features) - Library capabilities
- [Concept Embeddings](#concept-embeddings) - LLM-based semantic embeddings
- [Repository Structure](#repository-structure) - File organization
- [Development](#development) - Testing and code quality

## Quick Start

### Installation

```bash
pip install -e .
```

For development with testing and formatting tools:

```bash
pip install -e ".[dev]"
```

### Basic Usage

#### Navigate Concept Hierarchy

The `SnomedRelations` class provides methods for exploring SNOMED CT concept relationships.

```python
from snomed_methods_v1 import SnomedRelations

# Initialize with relationship data file
relations = SnomedRelations(
    snomed_rf2_full_path='/path/to/sct2_StatedRelationship_Full_INT_*.txt'
)

# Get children of a concept
children = relations.get_children(154621002)  # Meningioma

# Get parents of a concept
parents = relations.get_parents(154621002)

# Expand codes recursively
codes, names = relations.expand_codes(
    ['399187006'],
    mode='local',
    n_recursion=5
)
```

## Features

- **Concept Relationship Traversal**: Navigate parent-child relationships in SNOMED CT hierarchy
- **Code Expansion**: Automatically expand concept codes through recursive hierarchy traversal
- **MedCAT Integration**: Combine MedCAT entity recognition with SNOMED CT terminology
- **Snowstorm API Support**: Access remote Snowstorm terminology server API
- **Concept Embeddings**: LLM-based semantic embeddings for biomedical concepts using:
  - Local Hugging Face Transformers models (e.g., SapBERT, BioBERT)
  - SentenceTransformer models
  - Ollama API compatibility

**Note**: For detailed information on the Concept Embeddings module including usage examples and backend configuration, see the [Concept Embeddings](#concept-embeddings) section below.

## Concept Embeddings

The `llm_concept_embedder` module provides LLM-based semantic embeddings for biomedical concepts.

### Features

- **Local Hugging Face Transformers Models**: Load models like SapBERT directly from local storage
- **Ollama Backend**: Use remote Ollama instances for inference
- **FAISS Vector Search**: Efficient similarity search with FAISS index building
- **Batch Processing**: Generate embeddings in configurable batch sizes with checkpoint support

### Supported Backends

| Backend | Description | Example |
|---------|-------------|---------|
| `transformers` | Direct Hugging Face Transformers models (e.g., SapBERT, BioBERT) | ClinicalConceptEmbedder(model_path="path/to/model", backend="transformers") |
| `hf` | SentenceTransformer models | ClinicalConceptEmbedder(model_name="all-MiniLM-L6-v2", backend="hf") |
| `ollama` | Ollama API server | ClinicalConceptEmbedder(model_name="qwen2.5-coder", backend="ollama", ollama_base_url="http://localhost:11434") |

### Example Usage

```python
from llm_concept_embedder import (
    ClinicalConceptEmbedder,
    load_concepts_from_medcat,
    ConceptVectorSearch
)

# Initialize embedder with local SapBERT model
embedder = ClinicalConceptEmbedder(
    model_name_or_path="/workspaces/snomed_methods/embedding_models/SapBERT-from-PubMedBERT-fulltext",
    backend="transformers",
    device="cpu"
)

# Load concepts from MedCAT CDB
cat = CAT.load_model_pack("model_pack.zip")
concept_df = load_concepts_from_medcat(cat)

# Generate embeddings
embeddings = embedder.generate_embeddings(
    concept_texts=embedder.prepare_concept_text(concept_df),
    batch_size=32
)

# Create embedding dictionary and search engine
cui_to_embedding = {}
for i, row in enumerate(concept_df.itertuples()):
    cui_to_embedding[row.cui] = embeddings[i]

search_engine = ConceptVectorSearch(
    {'embeddings': cui_to_embedding, 'names': cat.cdb.cui2preferred_name},
    embedder=embedder
)
search_engine.build_index(index_type="FlatIP")

# Query for similar concepts
results = search_engine.search("meningioma", top_k=10)
```

### Model Configuration

To use the local SapBERT model included in this repository:

```python
embedder = ClinicalConceptEmbedder(
    model_name_or_path="/workspaces/snomed_methods/embedding_models/SapBERT-from-PubMedBERT-fulltext",
    backend="transformers",
    device="cpu"
)
```


## Requirements

- Python 3.8+
- pandas
- numpy

## Development

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/unit/test_snomed_term_lookup.py

# Run with coverage
pytest tests/ --cov=src --cov-report=xml
```

### Code Quality

```bash
# Format with Black
black .

# Lint with Ruff
ruff check .

# Type checking with mypy
mypy .
```

## License
