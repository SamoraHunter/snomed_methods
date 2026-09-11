.. _capabilities-index:

Capabilities Index
==================

This document provides a machine-readable index of all capabilities in snomed_methods.

Capability Categories
---------------------

1. **Term Lookup** - Search SNOMED CT descriptions by term
2. **Hierarchy Expansion** - Traverse parent-child relationships
3. **Semantic Similarity** - Find conceptually related concepts
4. **LLM Embeddings** - Generate vector embeddings using LLMs
5. **Vector Search** - Perform fast similarity search via FAISS

Machine-Readable Capability Index (JSON)
----------------------------------------

.. literalinclude:: ../../capabilities/term_lookup.json
   :language: json

.. literalinclude:: ../../capabilities/hierarchy_expansion.json
   :language: json

.. literalinclude:: ../../capabilities/semantic_search.json
   :language: json

.. literalinclude:: ../../capabilities/embedding_generation.json
   :language: json

.. literalinclude:: ../../capabilities/vector_search.json
   :language: json

API Reference Mapping
---------------------

+------------------------------+----------------------------------+-----------+
| Capability                   | API Class/Function               | Version   |
+==============================+==================================+===========+
| Term Lookup (Exact)          | ``SnomedTermLookup.find_concepts_by_term()``    | v1.0      |
+------------------------------+----------------------------------+-----------+
| Term Lookup (Fuzzy)          | ``SnomedTermLookup.find_concepts_by_term_fuzzy()`` | v1.0  |
+------------------------------+----------------------------------+-----------+
| Batch Term Search            | ``SnomedTermLookup.find_concepts_batch()``      | v1.0      |
+------------------------------+----------------------------------+-----------+
| Concept Info retrieval       | ``SnomedTermLookup.getconcept_info()``          | v1.0      |
+------------------------------+----------------------------------+-----------+
| Get Children                 | ``SnomedRelations.get_children()``              | v1.0      |
+------------------------------+----------------------------------+-----------+
| Get Parents                  | ``SnomedRelations.get_parents()``               | v1.0      |
+------------------------------+----------------------------------+-----------+
| Expand Codes (Local)         | ``SnomedRelations.expand_codes_local()``        | v1.0      |
+------------------------------+----------------------------------+-----------+
| Expand Codes (Snowstorm API)| ``SnomedRelations.expand_codes_snowstorm()``    | v1.0      |
+------------------------------+----------------------------------+-----------+
| Recursive Code Expansion     | ``SnomedRelations.recursive_code_expansion()``  | v1.0      |
+----------------------------+----------------------------------+-----------+
| Retrieve Search Synonyms     | ``SnomedRelations.retrieve_search_synonyms()``  | v1.0      |
+------------------------------+----------------------------------+-----------+
| Multi-term Synonym Retrieval | ``SnomedRelations.retrieve_search_synonyms_multi()`` | v1.0  |
+------------------------------+----------------------------------+-----------+
| Get MedCAT Similar Concepts  | ``SnomedRelations.get_medcat_cdb_most_similar()`` | v1.0  |
+------------------------------+----------------------------------+-----------+
| Semantic Tag Filtering       | ``SnomedRelations.get_subsumed_concepts()``     | v1.0      |
+------------------------------+----------------------------------+-----------+
| Prepare Concept Text         | ``ClinicalConceptEmbedder.prepare_concept_text()`` | v1.0  |
+------------------------------+----------------------------------+-----------+
| Generate Embeddings (HF)     | ``ClinicalConceptEmbedder.generate_embeddings()"`` | v1.0  |
+------------------------------+----------------------------------+-----------+
| Generate Embeddings (Ollama) | ``ClinicalConceptEmbedder._init_ollama_client()`` | v1.0  |
+------------------------------+----------------------------------+-----------+
| Export Embeddings            | ``ClinicalConceptEmbedder.export_embeddings()`` | v1.0      |
+------------------------------+----------------------------------+-----------+
| Build Vector Index           | ``ConceptVectorSearch.build_index()``           | v1.0      |
+------------------------------+----------------------------------+-----------+
| Search by Query Text         | ``ConceptVectorSearch.search(query_text=...)``  | v1.0      |
+------------------------------+----------------------------------+-----------+
| Semantic Search              | ``SemanticSearch.search()``                     | v1.0      |
+------------------------------+----------------------------------+-----------+

Configuration Capabilities
--------------------------

SNOMED RF2 Configuration:
"""""""""""""""""""""""""

- ``snomed_rf2_full_path`` - Custom relationship file path
- Environment variable: ``SNOMED_RF2_PATH``
- Default: SNOMED CT International Edition RF2

MedCAT Model Paths:
"""""""""""""""""""

- Default path via environment: ``MEDCAT_MODEL_PATH``
- Mode-specific paths: ``MEDCAT_ALIENCAT_PATH``, ``MEDCAT_DGH_PATH``, etc.

 Snowstorm API Configuration:
""""""""""""""""""""""""""""

- Endpoint: snowstorm.ihtsdotools.org
- Endpoints used: /children, /ancestors
- Configurable via user agent headers

LLM/Embedder Backend Options:
"""""""""""""""""""""""""""""

+----------+-----------------------------------+----------------------------+
| Backend  | Framework                         | Use Case                   |
+==========+===================================+============================+
| hf       | SentenceTransformer               | Fast CPU/GPU inference     |
+----------+-----------------------------------+----------------------------+
| transformers | HuggingFace Transformers        | Custom model architectures |
+----------+-----------------------------------+----------------------------+
| ollama   | Ollama API                        | Local LLM serving          |
+----------+-----------------------------------+----------------------------+

Vector Index Options:
"""""""""""""""""""""

- ``FlatIP`` - Exact search (INNER PRODUCT)
- ``HNSW`` - Approximate search for scalability

Export Formats
--------------

Embeddings:
- Pickle (.pkl) - Python native format
- NPZ (.npz) - NumPy compressed archive

Concept Dataframe:
- CSV (.csv) - Human-readable tabular data
- Parquet (.parquet) - Efficient columnar storage

Search Results:
- Dictionary (CUI -> Name mapping)
- List of tuples ((cui, name, score))
- Pandas DataFrame integrationReady

Error Handling Capabilities
---------------------------

1. **Missing Files**: Returns ``None`` or empty lists with warnings
2. **Invalid CUIs**: Graceful handling via try-except, returns empty results
3. **MedCAT Unavailable**: Falls back to SNOMED-only operations
4. **Embedding Failures**: Checkpoint support for resuming long jobs
5. **API Failures**: Snowstorm failures handled with None return

Environment Variable Reference
------------------------------

+------------------------------+----------------------------------+-------------+
| Variable                     | Description                      | Default     |
+==============================+==================================+=============+
| ``SNOMED_RF2_PATH``          | Path to SNOMED relationship file | See code    |
+------------------------------+----------------------------------+-------------+
| ``SNOMED_DIR``               | Directory containing SNOMED RF2  | ../uk_sct2cl|
+------------------------------+----------------------------------+-------------+
| ``MEDCAT_MODEL_PATH``        | MedCAT model pack path           |medcat_default|
+------------------------------+----------------------------------+-------------+
| ``MEDCAT_ALIENCAT_PATH``     | Aliencat model path              | Same as default |
+------------------------------+----------------------------------+-------------+
| ``MEDCAT_DGH_PATH``          | DGH model path                   | See code    |
+------------------------------+----------------------------------+-------------+
| ``OLLAMA_HOST``              | Ollama server URL                | http://localhost:11434|
+------------------------------+----------------------------------+-------------+

Compatibility Matrix
--------------------

+---------------------------+------------------+--------------------+--------+
| Feature                   | Python 3.8+      | Linux/Mac/Windows  | CUDA   |
+===========================+==================+====================+========+
| Term Lookup               | ✓                | ✓                  | N/A    |
+---------------------------+------------------+--------------------+--------+
| Hierarchy Expansion       | ✓                | ✓                  | N/A    |
+---------------------------+------------------+--------------------+--------+
| MedCAT Integration        | ✓                | ✓                  | ✓      |
+------------------- -------+------------------+--------------------+--------+
| SentenceTransformer (HF)  | ✓                | ✓                  | ✓      |
+---------------------------+------------------+--------------------+--------+
| Ollama Backend            | ✓                | ✓                  | N/A    |
+---------------------------+------------------+--------------------+--------+
| FAISS Index               | ✓                | ✓                  | ✓      |
+---------------------------+------------------+--------------------+--------+

Dependencies
------------

Core (Required):
- pandas >=1.5
- numpy >=1.24
- requests >=2.28
- tqdm >=4.65

Optional - MedCAT:
- medcat >=2.0

Optional - Embeddings:
- sentence-transformers >=2.2
- torch >=2.0 (for transformers backend)
- ollama >=0.1 (for Ollama backend)

Optional - Vector Search:
- faiss-cpu or faiss-gpu

Development Dependencies:
- rapidfuzz (for fuzzy matching)
- pytest, black, ruff (for development)

Version History
---------------

v1.0 (Current):
- Initial release
- Full term lookup functionality
- Hierarchy expansion with local and Snowstorm modes
- MedCAT integration for semantic similarity
- LLM embedding generation with HF/Ollama/Transformers backends
- FAISS-based vector similarity search
