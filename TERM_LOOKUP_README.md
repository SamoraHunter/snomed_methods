# SNOMED CT Term Lookup

A module for finding SNOMED CT codes by term matching.

## Overview

This module provides tools to search for SNOMED CT concepts using term-based searches. It supports:
- Exact term matching
- Case-insensitive searching
- Prefix matching (find terms starting with a string)
- Batch searching multiple terms
- Fuzzy string matching (for typos and variants)
- Concept information retrieval

## Installation

```bash
pip install pandas rapidfuzz
```

For fuzzy matching (optional):
```bash
pip install rapidfuzz
```

## Quick Start

### Basic Usage

```python
from snomed_term_lookup import create_term_lookup_from_directory

# Create lookup from SNOMED directory
lookup = create_term_lookup_from_directory(
    '/path/to/snomed/RF2directory'
)

# Search for terms starting with 'meningioma'
results = lookup.find_concepts_by_term('meningioma', match_prefix=True, top_n=5)

for cui, term in results:
    print(f"{cui}: {term}")
```

### Using the Python Module

```python
import os
from snomed_term_lookup import create_term_lookup_from_directory

# Set SNOMED directory path (or use env variable)
os.environ['SNOMED_DIR'] = '/path/to/snomed/directory'

lookup = create_term_lookup_from_directory(os.environ['SNOMED_DIR'])

# Find concepts
results = lookup.find_concepts_by_term(
    'meningioma',
    ignore_case=True,
    match_prefix=False,
    top_n=10
)

print(f"Found {len(results)} concepts:")
for cui, term in results:
    print(f"  CUI: {cui}")
    print(f"  Term: {term}")
```

### Advanced Features

#### Batch Search (Multiple Terms)

```python
terms = ['meningioma', 'glioma', 'tumor']
results = lookup.find_concepts_batch(terms, ignore_case=True)

for term, cui, matched in results:
    print(f"{term} -> {cui}: {matched}")
```

#### Fuzzy Search (Typo Tolerance)

```python
# Find concepts even with typos
try:
    results = lookup.find_concepts_by_term_fuzzy(
        'menignoma',  # Typo: missing 'o'
        min_score=70,  # Minimum similarity score (0-100)
        top_n=5
    )

    for cui, term, score in results:
        print(f"Score: {score}, CUI: {cui}, Term: {term}")
except ImportError:
    print("Install rapidfuzz for fuzzy matching: pip install rapidfuzz")
```

#### Get Concept Information

```python
# Get detailed information about a concept
info = lookup.getconcept_info('409681000000102')

print(f"Concept ID: {info['concept_id']}")
print(f"Preferred Name: {info['preferred_name']}")
print(f"All Names: {info['all_names']}")
```

## API Reference

### SnomedTermLookup Class

#### `__init__(snomed_description_path, active_only=True)`

Initialize the lookup with a SNOMED description file.

**Parameters:**
- `snomed_description_path`: Path to `sct2_Description_*.txt` file
- `active_only`: If True, only include active concepts (default True)

#### `find_concepts_by_term(term, ignore_case=True, match_prefix=False, top_n=None)`

Find concepts by matching a term.

**Parameters:**
- `term`: The term to search for
- `ignore_case`: Case-insensitive matching (default True)
- `match_prefix`: Match terms starting with query (default False)
- `top_n`: Maximum number of results (default None, return all)

**Returns:** List of tuples `(cui, term)` for each match

#### `find_concepts_by_term_fuzzy(term, min_score=50, top_n=None)`

Find concepts using fuzzy string matching.

**Parameters:**
- `term`: The term to search for
- `min_score`: Minimum similarity score (0-100, default 50)
- `top_n`: Maximum number of results

**Returns:** List of tuples `(cui, term, score)`

#### `find_concepts_batch(terms, ignore_case=True, match_prefix=False)`

Search for multiple terms at once.

**Parameters:**
- `terms`: List of terms to search
- `ignore_case`: Case-insensitive matching
- `match_prefix`: Match prefix

**Returns:** List of tuples `(term, cui, matched_term)`

#### `getconcept_info(cui)`

Get detailed information about a concept.

**Parameters:**
- `cui`: The concept ID

**Returns:** Dictionary with concept details, or None if not found

### Helper Functions

#### `create_term_lookup_from_directory(sct2_dir, active_only=True)`

Create a SnomedTermLookup from a SNOMED directory. Automatically finds the description file.

**Parameters:**
- `sct2_dir`: Path to SNOMED directory (e.g.,包含SNOMED CT发布文件夹的目录)
- `active_only`: Only include active concepts

**Returns:** SnomedTermLookup instance ready for searching

## Requirements

- Python 3.8+
- pandas
- rapidfuzz (optional, for fuzzy matching)

## Examples

See `examples/example_term_lookup.py` for complete usage examples.

Run examples:

```bash
# Set SNOMED directory path
export SNOMED_DIR=/path/to/snomed/directory

# Run the example script
python examples/example_term_lookup.py
```

## Files

- `snomed_term_lookup.py`: Main module
- `tests/unit/test_snomed_term_lookup.py`: Unit tests
- `examples/example_term_lookup.py`: Usage examples

## Integration with MedCAT

You can combine this with MedCAT for enhanced matching:

```python
from snomed_term_lookup import SnomedTermLookup
from medcat.cat import CAT

# Load both models
lookup = create_term_lookup_from_directory('/path/to/snomed')
cat = CAT.load_model_pack('/path/to/medcat/model.zip')

# Use MedCAT to find entities in text, then look up their SNOMED CT codes
doc = cat("Patient has meningioma")
for entity in doc.entities:
    cui = entity['cui']

    # Look up the SNOMED CT term for this CUI
    info = lookup.getconcept_info(cui)
    if info:
        print(f"{entity['text']} -> {info['preferred_name']}")
```

## Notes

- The description file should be in standard SNOMED RF2 format
- Use `/Snapshot/Terminology/sct2_Description_*.txt` from your SNOMED release
- The `active_only` parameter filters by the 'active' column (default True)
- For fuzzy matching, install `rapidfuzz`: `pip install rapidfuzz`
