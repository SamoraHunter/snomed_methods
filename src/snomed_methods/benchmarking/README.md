# SNOMED Methods Benchmarking System

A comprehensive benchmarking framework for evaluating different SNOMED CT concept methods including annotation, hierarchy expansion, vocabulary mapping, and term lookup.

## Architecture

The benchmarking system is organized into four specialized modules:

### 1. Annotation (`annotation/`)
Evaluates clinical concept annotators that map free-text clinical notes to SNOMED CT concepts.

- **Metrics**: Precision@K, Recall@K, F1@K, MRR
- **Data Format**: `{text: str, gold_cuis: [str]}`
- **Key Functions**:
  - `generate_annotation_dataset()`: Create synthetic annotation datasets
  - `load_annotation_datasets()`: Load pre-generated datasets
  - `evaluate_annotator()`: Evaluate annotation methods

### 2. Hierarchy Expansion (`hierarchy/`)
Evaluates hierarchy traversal methods that expand SNOMED concepts via parent-child relationships.

- **Metrics**: Exact Match Rate, Recall@K, Precision@K, F1@K, Jaccard Similarity
- **Data Format**: `{seed_cui: str, expected_related: [str]}`
- **Key Functions**:
  - `generate_hierarchy_dataset()`: Create synthetic expansion datasets
  - `load_hierarchy_datasets()`: Load pre-generated datasets
  - `evaluate_hierarchy_expansion()`: Evaluate hierarchy methods

### 3. Vocabulary Mapping (`vocabulary/`)
Evaluates mapping of SNOMED CUIs to other terminologies (ICD-10, LOINC, etc.).

- **Metrics**: Precision@K, Recall@K, Coverage Rate, MRR
- **Data Format**: `{snomed_cui: str, target_codes: [str]}`
- **Key Functions**:
  - `generate_mapping_dataset()`: Create synthetic mapping datasets
  - `load_mapping_datasets()`: Load pre-generated datasets
  - `evaluate_mapper()`: Evaluate vocabulary mappers

### 4. Term Lookup (`term/`)
Evaluates term search methods that match clinical terms to SNOMED concepts.

- **Metrics**: Recall@K, Precision@K, MRR, Hit Rate
- **Data Format**: `{term: str, expected_cui: str}`
- **Key Functions**:
  - `generate_term_dataset()`: Create synthetic term datasets
  - `load_term_datasets()`: Load pre-generated datasets
  - `evaluate_term_lookup()`: Evaluate term lookup methods

## Installation

The benchmarking system is included with the snomed_methods package. Ensure you have:

```bash
pip install "snomed-methods[benchmark]"
```

Or alternatively, install directly from source.

## Usage Examples

### Benchmarking a Concept Annotator

```python
from snomed_methods.benchmarking.annotation import (
    evaluate_annotator,
    generate_annotation_dataset,
)

# Generate synthetic dataset
dataset = generate_annotation_dataset(num_samples=100)

# Define your annotator function
def my_annotator(text: str):
    # Your annotation logic here
    return ["C001", "C002", "C003"]

# Evaluate the annotator
results = evaluate_annotator(
    annotator_func=my_annotator,
    dataset=dataset,
    k_values=[1, 3, 5, 10],
)

print(f"Precision@5: {results['precision@5']:.4f}")
print(f"Recall@5: {results['recall@5']:.4f}")
print(f"F1@5: {results['f1@5']:.4f}")
print(f"MRR: {results['mrr']:.4f}")
```

### Benchmarking Hierarchy Expansion

```python
from snomed_methods.benchmarking.hierarchy import (
    evaluate_hierarchy_expansion,
    generate_hierarchy_dataset,
)

# Generate dataset
dataset = generate_hierarchy_dataset(num_samples=50)

# Define your expansion function
def my_expansion(seed_cui: str):
    # Your hierarchy expansion logic here
    return ["C100", "C200", "C300"]

# Evaluate
results = evaluate_hierarchy_expansion(
    expansion_func=my_expansion,
    dataset=dataset,
    k_values=[5, 10, 20],
)

print(f"Recall@10: {results['recall@10']:.4f}")
print(f"Precision@10: {results['precision@10']:.4f}")
print(f"F1@10: {results['f1@10']:.4f}")
```

### Benchmarking Vocabulary Mapper

```python
from snomed_methods.benchmarking.vocabulary import (
    evaluate_mapper,
    generate_mapping_dataset,
)

# Generate dataset
dataset = generate_mapping_dataset(num_samples=50)

# Define your mapper function
def my_mapper(snomed_cui: str):
    # Your vocabulary mapping logic here
    return ["ICD_E11.9", "LOINC_4544-3"]

# Evaluate
results = evaluate_mapper(
    mapper_func=my_mapper,
    dataset=dataset,
    k_values=[1, 3, 5],
)

print(f"Coverage Rate: {results['coverage_rate']:.4f}")
print(f"MRR: {results['mrr']:.4f}")
```

### Benchmarking Term Lookup

```python
from snomed_methods.benchmarking.term import (
    evaluate_term_lookup,
    generate_term_dataset,
)

# Generate dataset
dataset = generate_term_dataset(num_samples=50)

# Define your lookup function
def my_lookup(term: str):
    # Your term lookup logic here
    return [("123456789", "Matched Term")]

# Evaluate
results = evaluate_term_lookup(
    lookup_func=my_lookup,
    dataset=dataset,
    k_values=[1, 3, 5],
)

print(f"Hit Rate: {results['hit_rate']:.4f}")
print(f"MRR: {results['mrr']:.4f}")
```

### Loading Pre-generated Datasets

```python
from snomed_methods.benchmarking.annotation import load_annotation_datasets
from snomed_methods.benchmarking.hierarchy import load_hierarchy_datasets
from snomed_methods.benchmarking.vocabulary import load_mapping_datasets
from snomed_methods.benchmarking.term import load_term_datasets

# Load all datasets
annotation_data = load_annotation_datasets()
hierarchy_data = load_hierarchy_datasets()
vocabulary_data = load_mapping_datasets()
term_data = load_term_datasets()

print(f"Annotation: {len(annotation_data['medium'])} samples")
print(f"Hierarchy: {len(hierarchy_data['medium'])} samples")
```

## Metrics Reference

### Annotation Metrics
| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K predictions that are relevant |
| Recall@K | Fraction of relevant items found in top-K |
| F1@K | Harmonic mean of precision and recall at K |
| MRR | Mean Reciprocal Rank of first relevant item |

### Hierarchy Metrics
| Metric | Description |
|--------|-------------|
| Exact Match Rate | Whether predicted set matches expected exactly |
| Recall@K | Fraction of expected items in top-K |
| Precision@K | Fraction of top-K that are relevant |
| F1@K | Harmonic mean at K |
| Jaccard Similarity | Set overlap ratio |

### Vocabulary Metrics
| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K mappings that are correct |
| Recall@K | Fraction of expected mappings in top-K |
| Coverage Rate | Fraction of all expected mappings found |
| MRR | Mean Reciprocal Rank of first correct mapping |

### Term Metrics
| Metric | Description |
|--------|-------------|
| Recall@K | Whether expected CUI in top-K (binary) |
| Precision@K | Similar to recall for term lookup |
| MRR | Mean Reciprocal Rank of expected CUI |
| Hit Rate | Fraction of queries where CUI appears |

## Contributing

To add a new benchmarking module:

1. Create a subdirectory in `src/snomed_methods/benchmarking/`
2. Implement `dataset.py` with generation functions
3. Implement `evaluation.py` with metric calculations and evaluation function
4. Add exports to the module's `__init__.py`
5. Register imports in the parent `__init__.py`

## Testing

Run all tests:

```bash
pytest tests/unit/test_benchmark_*.py -v
```

Run specific test file:

```bash
pytest tests/unit/test_benchmark_annotation.py -v
```

## API Reference

See docstrings in individual modules for detailed function signatures.

### Main Module Exports

- `snomed_methods.benchmarking.annotation`
- `snomed_methods.benchmarking.hierarchy`
- `snomed_methods.benchmarking.vocabulary`
- `snomed_methods.benchmarking.term`
- `snomed_methods.benchmarking.umnsrs` (existing semantic similarity module)
