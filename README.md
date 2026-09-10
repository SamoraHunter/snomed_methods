# SNOMED Methods

A Python library for working with SNOMED CT (Systematized Nomenclature of Medicine - Clinical Terms).

## Documentation Index

- [Quick Start](#quick-start) - Installation and basic usage
- [Features](#features) - Library capabilities
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

## Repository Structure

```
snomed_methods/
├── snomed_methods_v1.py          # Core SNOMED operations class
├── pyproject.toml                # Package configuration and dependencies
├── setup/                        # Installation scripts
│   ├── install.py                # Python-based installer
│   └── install.sh                # Bash installer (Linux/Mac)
└── tests/                        # Unit and integration tests
    ├── unit/
    │   ├── test_snomed_relations.py
    │   ├── test_snomed_term_lookup.py
    │   └── test_setup_install.py
    └── integration/
        └── test_integration.py
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
```

### Code Quality

```bash
# Format with Black
black .

# Lint with Ruff
ruff check .
```

## License

This project is provided as-is for educational and development purposes.
