.. _workflows:

Key Workflows and Usage Patterns
================================

This document describes the main workflows in snomed_methods with usage examples.

Basic Term Lookup Workflow
--------------------------

The basic term lookup workflow searches SNOMED CT descriptions for matching terms
and returns concept IDs (CUIs) with their preferred names.

.. figure:: _static/workflow_basic_lookup.md

   Sequence diagram showing basic term lookup flow.

Usage Example:

.. code-block:: python

    from snomed_term_lookup import create_term_lookup_from_directory

    # Initialize with SNOMED data directory
    lookup = create_term_lookup_from_directory("/path/to/snomed")

    # Basic substring search (case-insensitive by default)
    results = lookup.find_concepts_by_term("meningioma")
    print(f"Found {len(results)} concepts")

    for cui, term in results[:5]:
        print(f"{cui}: {term}")

    # Prefix matching
    prefix_results = lookup.find_concepts_by_term(
        "脑膜瘤",  # Chinese for meningioma
        match_prefix=True,
        top_n=10
    )

    # Fuzzy matching (requires rapidfuzz)
    fuzzy_results = lookup.find_concepts_by_term_fuzzy("meningoma", min_score=80)

    # Batch search
    terms = ["brain tumor", "liver cancer", "diabetes"]
    batch_results = lookup.find_concepts_batch(terms)

Concept Hierarchy Expansion Workflow
------------------------------------

The hierarchy expansion workflow traverses parent-child relationships in the SNOMED CT
ontology to retrieve related concepts.

.. figure:: _static/workflow_hierarchy_expansion.md

   Sequence diagram showing hierarchy expansion flow.

Usage Example:

.. code-block:: python

    from snomed_methods_v1 import SnomedRelations

    # Initialize with relationship data
    relations = SnomedRelations(
        medcat=False,  # Disable MedCAT for pure SNOMED operations
        snomed_rf2_full_path="/path/to/relationships.txt"
    )

    # Get direct children and parents
    child_codes, child_names = relations.expand_codes_children_local(363346000)  # Meningioma
    parent_codes, parent_names = relations.expand_codes_parents_local(363346000)

    print(f"Children: {len(child_codes)}, Parents: {len(parent_codes)}")

    # Full expansion (both directions)
    all_codes, all_names = relations.expand_codes(
        363346000,
        use_snowstorm=False  # Local processing
    )

    # Multi-level recursive expansion
    expanded_codes, expanded_names = relations.recursive_code_expansion(
        363346000,
        n_recursion=5  # Expand 5 levels deep
    )

    # Using Snowstorm API (remote)
    snowstorm_codes, snowstorm_names = relations.expand_codes(
        363346000,
        use_snowstorm=True
    )

    # With MedCAT for preferred names
    with_medcat = SnomedRelations(medcat=True)
    codes_with_names = with_medcat.recursive_code_expansion(363346000)

Combined Semantic Expansion Workflow
------------------------------------

The semantic expansion workflow combines multiple search strategies to find all concepts
semantically related to a given_term or list of terms.

.. figure:: _static/workflow_semantic_expansion.md

   Sequence diagram showing combined semantic expansion flow.

Usage Example:

.. code-block:: python

    from semantic_expansion import SemanticSearch

    # Initialize with SNOMED data path
    searcher = SemanticSearch(
        uk_path="/path/to/uk_sct2cl_42.2.0"
    )

    # Basic search with default settings
    results = searcher.search("meningioma")
    print(results.metrics)
    print(f"Total concepts: {len(results.concepts)}")

    # Custom configuration
    custom_results = searcher.search(
        term_or_terms=["brain tumor", "neoplasm"],
        max_concepts=500,  # Expand up to 500 concepts via hierarchy
        top_n_per_term=100,  # Get up to 100 matches per search term
        use_hierarchy=True,
        use_medcat=False  # Disable MedCAT for faster execution
    )

    # Access results in different formats
    print("Concepts (dict):", len(results.concepts))
    print("CUI list:", len(results.cuis))
    print("Term list:", len(results.terms))

    # Get core vs expanded concepts
    core = results.get_core_concepts("meningioma")
    expanded = results.get_expanded_concepts("meningioma")

    print(f"Core concepts (matching query): {len(core)}")
    print(f"Expanded concepts: {len(expanded)}")

Usage with MedCAT for Semantic Similarity:

.. code-block:: python

    from semantic_expansion import expand_concepts

    # Convenience function usage
    results = expand_concepts(
        term_or_terms="meningioma",
        uk_path="/path/to/snomed",
        max_concepts=200,
        top_n_per_term=50,
        use_hierarchy=True,
        use_medcat=True  # Enable MedCAT similarity search
    )

LLM Embedding & Similarity Search Workflow
------------------------------------------

This workflow generates high-dimensional embeddings for clinical concepts using various
LLM backends and performs efficient similarity search via FAISS.

.. figure:: _static/workflow_llm_embedding.md

   Sequence diagram showing LLM embedding and search flow.

Usage Example - Generating Embeddings:

.. code-block:: python

    from llm_concept_embedder import ClinicalConceptEmbedder

    # Initialize embedder with HuggingFace model
    embedder = ClinicalConceptEmbedder(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        backend="hf",  # Hugging Face SentenceTransformer
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Or use Ollama (local LLM)
    embedder = ClinicalConceptEmbedder(
        model_name_or_path="all-minilm",
        backend="ollama",
        ollama_base_url="http://localhost:11434"
    )

    # Prepare concept data
    from snomed_term_lookup import create_term_lookup_from_directory

    lookup = create_term_lookup_from_directory("/path/to/snomed")
    matches = lookup.find_concepts_by_term("meningioma", top_n=100)

    concept_df = pd.DataFrame([
        {"cui": cui, "preferred_name": term}
        for cui, term in matches
    ])

    # Format text prompts and generate embeddings
    text_prompts = embedder.prepare_concept_text(concept_df)
    embeddings = embedder.generate_embeddings(
        text_prompts,
        batch_size=64,
        checkpoint_path="./embeddings_checkpoint.pkl"
    )

    # Export embeddings
    cui_to_embedding = {
        cui: embeddings[i]
        for i, cui in enumerate(concept_df["cui"])
    }

    embedder.export_embeddings(
        cui_to_embedding,
        output_path="./outputs/concept_embeddings.pkl"
    )

Usage Example - Similarity Search:

.. code-block:: python

    from llm_concept_embedder import ConceptVectorSearch

    # Load previously saved embeddings
    search_engine = ConceptVectorSearch(
        embeddings_dict_or_path="./outputs/concept_embeddings.pkl",
        cui_to_name={
            cui: term
            for cui, term in matches
        },
        embedder=embedder  # Reuse embedder for query encoding
    )

    # Build FAISS index
    search_engine.build_index(index_type="FlatIP")  # or "HNSW" for faster approx search

    # Search for similar concepts
    results = search_engine.search(
        query_text="brain tumor",
        top_k=20
    )

    print("Top similar concepts:")
    for cui, name, score in results:
        print(f"{cui}: {name} (score: {score:.4f})")

    # Search with pre-computed embedding
    import numpy as np

    query_embedding = embedder.generate_embeddings(["brain tumor"])[0]
    results = search_engine.search(
        query_embedding=query_embedding,
        top_k=10
    )

Combined LLM + Hierarchy Workflow
---------------------------------

You can combine the embedding workflow with hierarchy expansion for enhanced concept
discovery.

.. code-block:: python

    from semantic_expansion import SemanticSearch
    from llm_concept_embedder import ClinicalConceptEmbedder, ConceptVectorSearch

    # Step 1: Get expanded concepts via hierarchy
    searcher = SemanticSearch(uk_path="/path/to/snomed")
    hierarchy_results = searcher.search(
        "meningioma",
        max_concepts=200,
        use_medcat=False
    )

    # Step 2: Generate embeddings for expanded concepts
    embedder = ClinicalConceptEmbedder("all-MiniLM-L6-v2", backend="hf")

    concept_df = pd.DataFrame([
        {"cui": cui, "preferred_name": name}
        for cui, name in hierarchy_results.concepts.items()
    ])

    text_prompts = embedder.prepare_concept_text(concept_df)
    embeddings = embedder.generate_embeddings(text_prompts)

    cui_to_embedding = {
        cui: embeddings[i]
        for i, cui in enumerate(concept_df["cui"])
    }

    # Step 3: Build vector search index
    search_engine = ConceptVectorSearch(
        cui_to_embedding,
        cui_to_name=hierarchy_results.concepts,
        embedder=embedder
    )
    search_engine.build_index()

    # Step 4: Query with natural language
    query_results = search_engine.search("abnormal brain growth", top_k=10)

    print("Concepts similar to 'abnormal brain growth':")
    for cui, name, score in query_results:
        print(f"{cui}: {name} (score: {score:.4f})")

Workflow Selection Guide
------------------------

+----------------------------+------------------------------+--------------------------+
| Use Case                   | Recommended Workflow         | Key Classes              |
+============================+==============================+==========================+
| Find exact concept matches | Basic Term Lookup            | SnomedTermLookup         |
+----------------------------+------------------------------+--------------------------+
| Traverse ontology tree     | Hierarchy Expansion          | SnomedRelations          |
+----------------------------+------------------------------+--------------------------+
| Comprehensive concept      | Combined Semantic Expansion  | SemanticSearch           |
| discovery                  |                              |                          |
+----------------------------+------------------------------+--------------------------+
| Embed concepts for ML      | LLM Embedding                | ClinicalConceptEmbedder  |
+----------------------------+------------------------------+--------------------------+
| Find semantically similar  | Vector Similarity Search     | ConceptVectorSearch      |
| concepts                   |                              |                          |
+----------------------------+------------------------------+--------------------------+

Performance Tips
----------------

1. **Use local processing** for simple hierarchy expansions (faster than API calls)
2. **Enable MedCAT only when needed** - it provides richer semantic similarity but is slower
3. **Batch embeddings generation** using appropriate batch_size for your hardware
4. **Use HNSW index type** for faster approximate search with large concept sets
5. **Enable checkpoints** during embedding generation for long-running jobs

Error Handling
--------------

Most methods include try-except blocks and will return empty lists or None on errors.
Always check results before processing:

.. code-block:: python

    results = lookup.find_concepts_by_term("nonexistent_term")
    if not results:
        print("No concepts found")

    relations = SnomedRelations(medcat=True)
    if not relations.has_medcat():
        print("MedCAT not available, using SNOMED only")
