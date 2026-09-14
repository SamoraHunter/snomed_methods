Benchmarking System
===================

The SNOMED Methods library includes a comprehensive benchmarking framework for evaluating different methods against standardized validation datasets. This system uses synthetic and real Hugging Face datasets to validate method performance across multiple clinical NLP tasks.

Overview
--------

The benchmarking suite evaluates methods across four core SNOMED CT tasks:

1. **Clinical Concept Annotation** - Mapping free-text clinical notes to SNOMED CT concepts
2. **Hierarchy Expansion** - Expanding SNOMED concepts through parent-child relationships
3. **Vocabulary Mapping** - Converting SNOMED CUIs to other terminologies (ICD-10, LOINC)
4. **Term Lookup** - Matching clinical terms to SNOMED concepts

Architecture
------------

The benchmarking system follows a modular design::

    src/snomed_methods/benchmarking/
    ├── annotation/           # Concept annotation evaluation
    ├── hierarchy/            # Hierarchy expansion evaluation
    ├── term/                 # Term lookup evaluation
    ├── vocabulary/           # Vocabulary mapping evaluation
    ├── umnsrs/               # Semantic similarity (UMNSRS from HF)
    ├── plots/                # Visualization utilities
    └── results.py            # Result aggregation and comparison

Dataset Generation
------------------

Annotation Dataset
^^^^^^^^^^^^^^^^^^

Generates synthetic clinical notes with gold-standard CUI annotations:

.. code-block:: python

    from snomed_methods.benchmarking.annotation import generate_annotation_dataset

    dataset = generate_annotation_dataset(num_samples=100)
    # Returns: List[{'text': str, 'gold_cuis': [str], 'disease_name': str}]

Format: Clinical notes with disease-specific terminology patterns and corresponding SNOMED concept IDs.

Hierarchy Dataset
^^^^^^^^^^^^^^^^^

Generates hierarchy expansion samples:

.. code-block:: python

    from snomed_methods.benchmarking.hierarchy import generate_hierarchy_dataset

    dataset = generate_hierarchy_dataset(num_samples=100)
    # Returns: List[{'seed_cui': str, 'expected_related': [str], 'num_expected': int}]

Format: Seed concepts with expected parent-child relationships.

Term Dataset
^^^^^^^^^^^^

Generates term-to-concept mapping samples:

.. code-block:: python

    from snomed_methods.benchmarking.term import generate_term_dataset

    dataset = generate_term_dataset(num_samples=100)
    # Returns: List[{'term': str, 'expected_cui': str, 'term_length': int}]

Format: Clinical terms with their expected SNOMED CUIs.

Vocabulary Mapping Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Generates SNOMED-to-external-vocabulary mappings:

.. code-block:: python

    from snomed_methods.benchmarking.vocabulary import generate_mapping_dataset

    dataset = generate_mapping_dataset(num_samples=100)
    # Returns: List[{'snomed_cui': str, 'target_codes': [str], 'vocab_type': str}]

Format: SNOMED concepts with expected mapped codes from target vocabularies.

UMNSRS Dataset (⭐)
^^^^^^^^^^^^^^^^^^

Loads the **UMNSRS dataset from Hugging Face Hub** - a real semantic similarity/relatedness benchmark:

.. code-block:: python

    from snomed_methods.benchmarking.umnsrs import download_umnsrs, get_umnsrs_pairs

    # Download from HF
    dataset = download_umnsrs("relatedness", split="train")
    pairs = get_umnsrs_pairs("similarity")

    # Returns: List[{'text_1': str, 'text_2': str, 'label': float}]

Format: Concept pairs with human-annotated similarity/relatedness scores (0-1000 scale).

Evaluation Metrics
------------------

Annotation Metrics
^^^^^^^^^^^^^^^^^^

+--------------------+---------------------------------------------------+
| Metric             | Description                                       |
+====================+===================================================+
| Precision@K        | Fraction of top-K predicted concepts that are     |
|                    | relevant                                          |
+--------------------+---------------------------------------------------+
| Recall@K           | Fraction of relevant concepts found in top-K      |
|                    | predictions                                       |
+--------------------+---------------------------------------------------+
| F1@K               | Harmonic mean of Precision and Recall at K        |
+--------------------+---------------------------------------------------+
| MRR                | Mean Reciprocal Rank - average rank of first      |
|                    | relevant concept                                  |
+--------------------+---------------------------------------------------+

Hierarchy Metrics
^^^^^^^^^^^^^^^^^

+---------------------+-------------------------------------------------------+
| Metric              | Description                                           |
+=====================+=======================================================+
| Exact Match Rate    | Whether predicted set matches expected related        |
|                     | concepts exactly                                      |
+---------------------+-------------------------------------------------------+
| Recall@K            | Fraction of expected hierarchy members in top-K       |
+---------------------+-------------------------------------------------------+
| Precision@K         | Fraction of top-K that are expected hierarchy         |
|                     | members                                               |
+---------------------+-------------------------------------------------------+
| F1@K                | Harmonic mean at K                                    |
+---------------------+-------------------------------------------------------+
| Jaccard Similarity  | Set overlap ratio between predicted and expected      |
+---------------------+-------------------------------------------------------+

Vocabulary Metrics
^^^^^^^^^^^^^^^^^^

+------------------+------------------------------------------------------+
| Metric           | Description                                          |
+==================+======================================================+
| Precision@K      | Fraction of top-K mapped codes that are correct      |
+------------------+------------------------------------------------------+
| Recall@K         | Fraction of expected mappings found in top-K         |
+------------------+------------------------------------------------------+
| Coverage Rate    | Fraction of all expected mappings successfully found |
+------------------+------------------------------------------------------+
| MRR              | Mean Reciprocal Rank of first correct mapping        |
+------------------+------------------------------------------------------+

Term Metrics
^^^^^^^^^^^^

+--------------+---------------------------------------------------------+
| Metric       | Description                                             |
+==============+=========================================================+
| Recall@K     | Binary: whether expected CUI appears in top-K           |
+--------------+---------------------------------------------------------+
| Precision@K  | Similar to recall for term lookup precision             |
+--------------+---------------------------------------------------------+
| MRR          | Mean Reciprocal Rank of expected CUI position           |
+--------------+---------------------------------------------------------+
| Hit Rate     | Fraction of queries where concept appears anywhere      |
+--------------+---------------------------------------------------------+

Semantic Similarity Metrics (UMNSRS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------+-----------------------------------------------+
| Metric               | Description                                   |
+======================+===============================================+
| Spearman Correlation | Rank correlation with human judgments         |
+----------------------+-----------------------------------------------+
| Pearson Correlation  | Linear correlation with human scores          |
+----------------------+-----------------------------------------------+
| MAE                  | Mean Absolute Error vs ground truth           |
+----------------------+-----------------------------------------------+
| RMSE                 | Root Mean Squared Error                       |
+----------------------+-----------------------------------------------+

Usage Examples
--------------

Evaluating a Concept Annotator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from snomed_methods.benchmarking.annotation import (
        evaluate_annotator,
        generate_annotation_dataset,
    )

    # Generate validation dataset (synthetic or from HF)
    dataset = generate_annotation_dataset(num_samples=100)

    # Define your method
    def my_annotator(text: str):
        # Your annotation implementation
        result = annotator.annotate(text, top_k=10)
        return [c.concept_id for c in result.top_concepts]

    # Evaluate
    results = evaluate_annotator(
        annotator_func=my_annotator,
        dataset=dataset,
        k_values=[1, 3, 5, 10],
    )

    print(f"Precision@5: {results['precision@5']:.4f}")
    print(f"Recall@5: {results['recall@5']:.4f}")
    print(f"MRR: {results['mrr']:.4f}")

Evaluating Vocabulary Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from snomed_methods.benchmarking.vocabulary import (
        evaluate_mapper,
        generate_mapping_dataset,
    )

    dataset = generate_mapping_dataset(num_samples=50)

    def my_mapper(snomed_cui: str):
        # Your mapping implementation
        mappings = mapper.map_to_icd10(snomed_cui)
        return [m.code for m in mappings]

    results = evaluate_mapper(
        mapper_func=my_mapper,
        dataset=dataset,
        k_values=[1, 3, 5],
    )

    print(f"Coverage Rate: {results['coverage_rate']:.4f}")

Using Real HF UMNSRS Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from snomed_methods.benchmarking.umnsrs import (
        download_umnsrs,
        evaluate_model,
    )

    # Load real human-annotated benchmark from Hugging Face
    dataset = download_umnsrs("relatedness")  # 588 concept pairs

    def my_similarity_model(text1: str, text2: str) -> float:
        # Your semantic similarity implementation
        return embedder.compute_similarity(text1, text2)

    results = evaluate_model(
        model_func=my_similarity_model,
        dataset_pairs=dataset,
        metric=["spearman", "pearson"],
    )

    print(f"Spearman Correlation: {results['spearman_correlation']:.4f}")

Evaluating Hierarchy Expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from snomed_methods.benchmarking.hierarchy import (
        evaluate_hierarchy_expansion,
        generate_hierarchy_dataset,
    )

    dataset = generate_hierarchy_dataset(num_samples=100)

    def my_expansion(seed_cui: str):
        rf2_path = "/path/to/snomed_rf2"
        relations = SnomedRelations(snomed_rf2_full_path=rf2_path)

        children = relations.get_children(seed_cui)[:10]
        parents = relations.get_parents(seed_cui)[:5]

        return children + parents

    results = evaluate_hierarchy_expansion(
        expansion_func=my_expansion,
        dataset=dataset,
        k_values=[5, 10, 20],
    )

    print(f"Recall@10: {results['recall@10']:.4f}")

Result Management
-----------------

Saving and Loading Results
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from snomed_methods.benchmarking import save_results, load_results

    # Save results to file
    save_results(results_dict, "benchmark_results.json", file_format="json")
    save_results(results_dict, "benchmark_results.csv", file_format="csv")

    # Load previously saved results
    loaded = load_results("benchmark_results.json")

Aggregating Multiple Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from snomed_methods.benchmarking import (
        aggregate_results,
        compare_benchmarks,
        summarize_benchmark,
    )

    # Compare multiple method versions
    all_results = {
        "method_v1": results_v1,
        "method_v2": results_v2,
        "method_v3": results_v3,
    }

    # Aggregate into DataFrame
    df = aggregate_results(all_results)

    # Compare specific metric
    best_method = compare_benchmarks(
        all_results,
        metric="precision@5"
    ).idxmax()

    # Summary statistics
    summary = summarize_benchmark(all_results)

Plotting Results
^^^^^^^^^^^^^^^^

.. code-block:: python

    from snomed_methods.benchmarking import (
        plot_comparison,
        plot_metric_distribution,
    )

    # Bar chart comparison across methods
    fig = plot_comparison(
        all_results,
        metrics=["precision@5", "recall@5", "f1@5"],
        title="Method Comparison",
    )
    fig.savefig("comparison.png")

    # Distribution of a single metric
    fig = plot_metric_distribution(
        all_results,
        metric="mrr",
        title="MRR Distribution",
    )
    fig.savefig("mrr_distribution.png")

Workflows
---------

Benchmark Suite Runner (benchmark_run_all.ipynb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``notebooks/benchmarks/benchmark_run_all.ipynb`` notebook provides a unified interface for running all benchmarks with real SNOMED methods.

Configuration needed:
- Set UK SNOMED RF2 data paths
- Configure annotator, term lookup, mapper, and hierarchy expansion implementations

Results Evaluation (benchmark_evaluate_results.ipynb)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``notebooks/benchmarks/benchmark_evaluate_results.ipynb`` notebook demonstrates:
- Loading benchmark results
- Generating summary statistics
- Creating comparison plots
- Exporting to CSV/JSON formats

Why Use Hugging Face Datasets?
-------------------------------

The UMNSRS dataset from Hugging Face Hub provides:

1. **Standardized Validation**: 588 concept pairs with human-annotated semantic relatedness scores
2. **Community Benchmark**: Widely used in biomedical NLP research for fair comparison
3. **Two Variants**: Relatedness (588 pairs) and Similarity (566 pairs) subsets
4. **Modified Versions**: Excludes control pairs for cleaner evaluation

 Contributing New Benchmarks
---------------------------

To add a new benchmark module:

1. Create subdirectory: ``src/snomed_methods/benchmarking/newtask/``
2. Implement ``dataset.py`` with generation functions
3. Implement ``evaluation.py`` with metrics and evaluation function
4. Add exports to ``__init__.py``
5. Register imports in parent ``benchmarking/__init__.py``

See existing modules for reference implementations.
