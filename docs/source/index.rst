snomed-methods documentation
============================

SNOMED CT Clinical Terminology Methods for Healthcare NLP

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   module_architecture
   workflows
   capabilities_index
   benchmarking

Features
--------

- **Term Lookup**: Fast substring and fuzzy matching on SNOMED CT descriptions
- **Hierarchy Expansion**: Traverse parent-child relationships locally or via Snowstorm API
- **Semantic Search**: Combine term matching with hierarchy expansion for comprehensive concept discovery
- **LLM Embeddings**: Generate embeddings using HuggingFace, Transformers, or local Ollama
- **Vector Search**: FAISS-based similarity search for finding semantically related concepts

Quick Start
-----------

.. code-block:: python

    from snomed_term_lookup import create_term_lookup_from_directory

    # Term lookup
    lookup = create_term_lookup_from_directory("/path/to/snomed")
    results = lookup.find_concepts_by_term("meningioma")

    from semantic_expansion import SemanticSearch

    # Combined search
    searcher = SemanticSearch()
    search_results = searcher.search("meningioma", max_concepts=100)

    from llm_concept_embedder import ClinicalConceptEmbedder, ConceptVectorSearch

    # Embeddings and search
    embedder = ClinicalConceptEmbedder("all-MiniLM-L6-v2", backend="hf")
    search_engine = ConceptVectorSearch("./embeddings.pkl")
    search_engine.build_index()
    results = search_engine.search("brain tumor", top_k=10)

Installation
------------

.. code-block:: bash

    # Using the installer script
    python setup/install.py [venv_name] [mode]

    # Or with pip
    pip install -e .

API Reference
-------------

.. toctree::
   :maxdepth: 4

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
