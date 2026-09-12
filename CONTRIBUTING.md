# Contributing to SNOMED Methods

Thank you for your interest in contributing to SNOMED Methods! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Testing Workflow](#testing-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Git Workflow](#git-workflow)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- git

### Clone the Repository

```bash
git clone https://github.com/SNOMED-Methods/snomed-methods.git
cd snomed-methods
```

### Create Virtual Environment

We recommend using a virtual environment to isolate dependencies:

```bash
python -m venv snomed_methods_env
source snomed_methods_env/bin/activate  # On Windows: snomed_methods_env\Scripts\activate
```

### Install Dependencies

For development with all tools:

```bash
pip install -e ".[dev,docs,medcat]"
```

This installs:
- Core dependencies (numpy, pandas, requests, tqdm)
- Development tools (pytest, black, ruff, mypy)
- Documentation tools (sphinx, furo)
- MedCAT support (if needed)

### Verify Installation

```bash
python -c "import snomed_methods_v1; print('Import successful')"
```

## Testing Workflow

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Module

```bash
pytest tests/unit/test_snomed_term_lookup.py
pytest tests/integration/test_integration.py
```

### Run with Verbose Output

```bash
pytest -v
pytest -vvs  # Very verbose with stdout
```

### Run Tests Matching a Pattern

```bash
pytest -k "term_lookup"
pytest -k "hierarchy or embedding"
```

## Code Quality Standards

### Formatting (Black)

All code must be formatted with Black:

```bash
black .
```

Check formatting without changes:

```bash
black --check .
```

### Linting (Ruff)

Linting is required before commit:

```bash
ruff check .
```

Auto-fix linting issues:

```bash
ruff check . --fix
```

### Type Checking (mypy)

Type hints are required for new code:

```bash
mypy .
```

### Pre-commit Hooks

Install pre-commit to automatically run checks:

```bash
pre-commit install
```

This runs on every commit:
- ruff check
- black formatting check
- mypy type checking

## Git Workflow

### Branch Naming

Use descriptive branch names following this convention:

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/description` | `feature/hybrid-search-enhancement` |
| Fix | `fix/description` | `fix/term-lookup-bug` |
| Docs | `docs/description` | `docs/update-readme` |
| Refactor | `refactor/description` | `refactor/cleanup-embedder` |

### Commit Messages

Use the following format for commit messages:

```
type(scope): description

body (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```
feat: Add HybridSearch module combining term matching and embeddings
fix: Resolve NoneType error in SnomedTermLookup initialization
docs: Update API documentation for SemanticSearch class
refactor: Simplify embedding generation pipeline
test: Add unit tests for concept expansion functionality
```

### Pull Request Process

1. **Create a branch** from `main` with your feature or fix

2. **Make your changes** following the code quality standards above

3. **Run checks locally**:
   ```bash
   ruff check . --fix
   black .
   pytest tests/
   mypy .
   ```

4. **Commit your changes** with clear, descriptive messages

5. **Push to your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

7. **PR Requirements**:
   - All tests must pass (CI checks)
   - Code must be formatted with Black
   - Linting must pass with no errors
   - Type hints required for new functions
   - Documentation updated if applicable
   - Changelog entry added for user-facing changes

### Code Review Guidelines

- Be respectful and constructive in feedback
- Focus on code quality, not personal preferences
- Suggest improvements with clear explanations
- Acknowledge good contributions
- Merge only when all CI checks pass

## Adding New Features

1. Create an issue describing the feature first (optional but recommended)
2. Fork the repository
3. Create a feature branch
4. Implement the feature following project conventions
5. Add tests for new functionality
6. Update documentation if needed
7. Submit a pull request

## Questions?

Open an issue with the `question` label or contact maintainers.
