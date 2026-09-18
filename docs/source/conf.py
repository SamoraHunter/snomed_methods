import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

project = "snomed-methods"
copyright_year = "2026"
author = "SNOMED Methods Contributors"

release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]

suppress_warnings = [
    "autodoc.imported_member",
]
