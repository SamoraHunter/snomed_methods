.. _module-architecture:

Module Architecture
===================

This document provides an overview of the snomed_methods module structure, including classes,
key methods, and their interactions.

Core Modules Overview
---------------------

The project consists of four main modules that work together to provide SNOMED CT concept
lookup, expansion, and semantic search capabilities.

.. code-block:: text

    Input Layer ──► Core Modules ──► Workflow Layer ──► Output Layer
         │                    │                   │                │
         ▼                    ▼                   ▼                ▼
    RF2 Files         SnomedRelations     Concept Expansion   CUI Lists
    Descriptions      TermLookup          Search Strategies     Names
    MedCAT Model      Embedder            Hierarchy Traversal   Similarity
    LLMs              SemanticSearch

Module Structure Diagram
------------------------

.. figure:: _static/module_architecture.md

   High-level module architecture showing data flow from input sources through
   core modules to workflow layer and final output.

SnomedRelations Module
^^^^^^^^^^^^^^^^^^^^^^

The ``SnomedRelations`` class provides access to SNOMED CT relationship data (stated
relationships) and implements hierarchy traversal operations.

.. autoclass:: snomed_methods_v1.SnomedRelations
   :members:
   :special-members: __init__

Key Methods:

- ``get_children(cui)`` - Retrieve child concepts for a given concept ID
- ``get_parents(cui)`` - Retrieve parent concepts for a given concept ID
- ``expand_codes(cui, use_snowstorm=False)`` - Expand concept hierarchy in both directions
- ``recursive_code_expansion(cui, n_recursion=3)`` - Multi-level hierarchy traversal
- ``retrieve_search_synonyms(cui)`` - Combine SNOMED and MedCAT synonyms

SnomedTermLookup Module
^^^^^^^^^^^^^^^^^^^^^^^

The ``SnomedTermLookup`` class provides term-based searching of SNOMED CT descriptions.

.. autoclass:: snomed_term_lookup.SnomedTermLookup
   :members:
   :special-members: __init__

Key Methods:

- ``find_concepts_by_term(term, ignore_case=True)`` - Substring matching on description terms
- ``find_concepts_by_term_fuzzy(term, min_score=50)`` - Fuzzy string matching
- ``find_concepts_batch(terms)`` - Batch term search for multiple terms
- ``getconcept_info(cui)`` - Get detailed information about a concept

ClinicalConceptEmbedder Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ClinicalConceptEmbedder`` class generates high-dimensional vector embeddings for
clinical concepts using various LLM backends.

.. autoclass:: llm_concept_embedder.ClinicalConceptEmbedder
   :members:
   :special-members: __init__

Key Methods:

- ``prepare_concept_text(concept_df)`` - Format concept data into text prompts
- ``generate_embeddings(concept_texts)`` - Generate embeddings using configured backend
- ``export_embeddings(embeddings_dict, output_path)`` - Save embeddings to disk

Supported Backends:

- **HF** (Hugging Face SentenceTransformer): Fast inference on CPU/GPU
- **Transformers**: Direct access to HuggingFace Transformers models
- **Ollama**: Local LLM inference via Ollama API

ConceptVectorSearch Class
'''''''''''''''''''''''''

.. autoclass:: llm_concept_embedder.ConceptVectorSearch
   :members:
   :special-members: __init__

Key Methods:

- ``build_index(index_type="FlatIP")`` - Build FAISS vector index for fast similarity search
- ``search(query_text=None, query_embedding=None, top_k=20)`` - Search for similar concepts

SemanticExpansion Module
^^^^^^^^^^^^^^^^^^^^^^^^

The ``SemanticSearch`` class provides unified semantic expansion using multiple strategies.

.. autoclass:: semantic_expansion.SemanticSearch
   :members:
   :special-members: __init__

Key Methods:

- ``search(term_or_terms, max_concepts=100)`` - Perform comprehensive semantic search
- ``expand_concepts(term, max_concepts=100)`` - Convenience function wrapper

Expand Strategies:
""""""""""""""""""

1. **Term Matching** (always enabled)
   - Exact substring matching on SNOMED description terms
   - Prefixed term matching for recall enhancement

2. **Hierarchy Expansion** (enabled by default)
   - Parent-child traversal via relationship data
   - Breadth-first search with configurable depth

3. **MedCAT Semantic Similarity** (optional)
   - Co-occurrence based similarity from MedCAT model
   - Threshold-based filtering of similar concepts

SearchResults Class
'''''''''''''''''''

.. autoclass:: semantic_expansion.SearchResults
   :members:

Capability Index
----------------

See :doc:`capabilities_index` for a machine-readable index of all capabilities.

Configuration
-------------

Environment Variables:
^^^^^^^^^^^^^^^^^^^^^^

- ``SNOMED_RF2_PATH`` - Path to SNOMED relationship data file
- ``SNOMED_DIR`` or ``SNOMED_DIR_PATH`` - Directory containing SNOMED RF2 files
- ``MEDCAT_MODEL_PATH`` - Default MedCAT model pack path
- ``MEDCAT_ALIENCAT_PATH``, ``MEDCAT_DGH_PATH``, etc. - Mode-specific paths

Default Paths:
"""""""""""""

.. code-block:: python

    # SNOMED default (can be overridden by env var)
    "/data/snomed/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/Full/Terminology/sct2_StatedRelationship_Full_INT_20231101.txt"

    # MedCAT default (can be overridden by mode-specific env vars)
    "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip"

Installation
------------

Run the installer script:

.. code-block:: bash

    python setup/install.py [venv_name] [mode]

Modes:
- **production** (default): Install minimal dependencies
- **dev**: Install with development extras including MedCAT

Usage Patterns
--------------

See :doc:`workflows` for detailed usage examples of each workflow.
