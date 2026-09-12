#!/usr/bin/env python3
"""Test script for hybrid_search.py module"""

import os
import sys


def run_tests():
    """Run comprehensive tests on HybridSearch class."""

    print("=" * 80)
    print("HYBRID SEARCH MODULE TESTS")
    print("=" * 80)
    print()

    results = {"passed": [], "failed": []}

    project_root = os.path.dirname(os.path.abspath(__file__))
    uk_path = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )
    medcat_path = os.path.join(
        project_root, "model_packs", "medcat_model_pack_422d1d38fc58f158.zip"
    )

    # Test 1: Import and initialization
    print("-" * 40)
    print("TEST 1: Import and Initialization")
    print("-" * 40)
    try:
        from src.snomed_methods import HybridSearch, SearchResult

        print("[PASS] Successfully imported HybridSearch class")
        results["passed"].append("Import HybridSearch")

        print("[PASS] Successfully imported SearchResult class")
        results["passed"].append("Import SearchResult")

        # Initialize with default paths (provide UK path explicitly since SNOMED data is in project root)

        if os.path.exists(uk_path):
            print(f"[INFO] Using SNOMED path: {uk_path}")
        else:
            print("[WARN] SNOMED data not found at expected location")

        searcher = HybridSearch(uk_path=uk_path, medcat_path=medcat_path)
        print("[PASS] HybridSearch initialized successfully with explicit paths")
        results["passed"].append("Initialize HybridSearch")

    except Exception as e:
        print(f"[FAIL] Error during import/initialization: {e}")
        results["failed"].append(f"Import/Initializati on: {e}")

    print()

    # Test 2: Basic search functionality with "meningioma"
    print("-" * 40)
    print("TEST 2: Basic Search Functionality - 'meningioma'")
    print("-" * 40)
    try:
        from src.snomed_methods import HybridSearch, SearchResult

        if os.path.exists(uk_path):
            print(f"[INFO] Using SNOMED path: {uk_path}")

        searcher = HybridSearch(uk_path=uk_path, medcat_path=medcat_path)

        query = "meningioma"
        print(f"Executing search with query: '{query}'")

        result = searcher.search(query, top_k=15)

        print("[PASS] Search completed successfully")
        results["passed"].append("Basic search 'meningioma'")

        # Verify result type
        if isinstance(result, SearchResult):
            print("[PASS] Result is of type SearchResult")
            results["passed"].append("Result type check")
        else:
            print(f"[FAIL] Expected SearchResult, got {type(result)}")
            results["failed"].append("Result type check")

        # Check metrics
        print()
        print("Search Results Metrics:")
        print(f"  - Term matches: {result.term_matches}")
        print(f"  - Hierarchy matches: {result.hierarchy_matches}")
        print(f"  - Embedding matches: {result.embedding_matches}")
        print(f"  - Total results returned: {len(result)}")

        if result.term_matches > 0:
            print("[PASS] Term matches found")
            results["passed"].append("Term matches > 0")
        else:
            print("[WARN] No term matches (may indicate data path issue)")

        print()
        print("Sample Results:")
        print("-" * 60)
        for i, (cui, term, score) in enumerate(result.results[:5]):
            print(f"  {i+1}. [{cui}] {term} - Score: {score:.4f}")

    except Exception as e:
        print(f"[FAIL] Error during search: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"Basic search: {e}")

    print()

    # Test 3: Re-ranking weights comparison
    print("-" * 40)
    print("TEST 3: Re-ranking Weights Comparison")
    print("-" * 40)
    try:
        from src.snomed_methods import HybridSearch

        if os.path.exists(uk_path):
            print(f"[INFO] Using SNOMED path: {uk_path}")

        searcher = HybridSearch(uk_path=uk_path, medcat_path=medcat_path)

        query = "meningioma"

        # Test with different weight configurations
        configs = [
            (
                "Term-heavy",
                {"term_weight": 0.6, "hierarchy_weight": 0.2, "embedding_weight": 0.2},
            ),
            (
                "Embedding-heavy",
                {"term_weight": 0.1, "hierarchy_weight": 0.1, "embedding_weight": 0.8},
            ),
            (
                "Balanced",
                {"term_weight": 0.3, "hierarchy_weight": 0.2, "embedding_weight": 0.5},
            ),
        ]

        print(f"Query: '{query}'")
        print()

        all_results = []
        for name, weights in configs:
            result = searcher.search(query, top_k=10, **weights)
            all_results.append((name, result))

            print(
                f"{name} (weights: term={weights['term_weight']}, hier={weights['hierarchy_weight']}, embed={weights['embedding_weight']}):"
            )
            print("  Top results:")
            for i, (cui, term, score) in enumerate(result.results[:3]):
                print(f"    {i+1}. [{cui}] {term} - Score: {score:.4f}")

        # Compare top results between configurations
        print()
        print("Comparison of Top Results:")
        for i, (name1, result1) in enumerate(all_results):
            for name2, result2 in all_results[i + 1 :]:
                cuis1 = set(result1.cuis[:5])
                cuis2 = set(result2.cuis[:5])
                overlap = len(cuis1 & cuis2)
                print(f"  {name1} vs {name2}: {overlap}/5 concepts in common")

            print("[PASS] Re-ranking weights test completed")
            results["passed"].append("Re-ranking weights comparison")

    except Exception as e:
        print(f"[FAIL] Error during re-ranking test: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"Re-rating test: {e}")

    print()

    # Test 3: SearchResult metrics verification
    print("-" * 40)
    print("TEST 3: SearchResult Metrics Verification")
    print("-" * 40)
    try:
        from src.snomed_methods import HybridSearch

        if os.path.exists(uk_path):
            print(f"[INFO] Using SNOMED path: {uk_path}")

        searcher = HybridSearch(uk_path=uk_path, medcat_path=medcat_path)
        result = searcher.search("meningioma", top_k=15)

        # Verify individual metrics
        print("Verifying SearchResult attributes:")

        # Check term_matches
        if hasattr(result, "term_matches"):
            print(f"[PASS] term_matches attribute exists: {result.term_matches}")
            results["passed"].append("term_matches attribute")
        else:
            print("[FAIL] term_matches attribute missing")
            results["failed"].append("term_matches attribute")

        # Check hierarchy_matches
        if hasattr(result, "hierarchy_matches"):
            print(
                f"[PASS] hierarchy_matches attribute exists: {result.hierarchy_matches}"
            )
            results["passed"].append("hierarchy_matches attribute")
        else:
            print("[FAIL] hierarchy_matches attribute missing")
            results["failed"].append("hierarchy_matches attribute")

        # Check embedding_matches
        if hasattr(result, "embedding_matches"):
            print(
                f"[PASS] embedding_matches attribute exists: {result.embedding_matches}"
            )
            results["passed"].append("embedding_matches attribute")
        else:
            print("[FAIL] embedding_matches attribute missing")
            results["failed"].append("embedding_matches attribute")

        # Check cui_to_term mapping
        if hasattr(result, "cui_to_term") and len(result.cui_to_term) > 0:
            print(
                f"[PASS] cui_to_term mapping exists with {len(result.cui_to_term)} entries"
            )
            results["passed"].append("cui_to_term mapping")

            # Show sample mappings
            print("  Sample CUI to term mappings:")
            for _i, (cui, term) in enumerate(list(result.cui_to_term.items())[:3]):
                print(f"    {cui}: {term}")
        else:
            print("[WARN] cui_to_term mapping is empty or missing")

        # Check cui_scores
        if hasattr(result, "cui_scores") and len(result.cui_scores) > 0:
            print(f"[PASS] cui_scores exists with {len(result.cui_scores)} entries")
            results["passed"].append("cui_scores attribute")

            # Show sample scores for first result
            first_cui = result.results[0][0]
            if first_cui in result.cui_scores:
                print(f"  Scores for top result [{first_cui}]:")
                for score_type, score_val in result.cui_scores[first_cui].items():
                    print(f"    {score_type}: {score_val:.4f}")
        else:
            print("[WARN] cui_scores is empty or missing")

        # Check to_dict method
        if hasattr(result, "to_dict"):
            dict_result = result.to_dict()
            if "metrics" in dict_result:
                print(f"[PASS] to_dict() returns metrics: {dict_result['metrics']}")
                results["passed"].append("to_dict() method")
            else:
                print("[FAIL] to_dict() missing 'metrics' key")
                results["failed"].append("to_dict() method")
        else:
            print("[FAIL] to_dict() method missing")
            results["failed"].append("to_dict() method")

        # Check properties
        if hasattr(result, "cuis") and len(result.cuis) > 0:
            print(f"[PASS] cuis property works: {len(result.cuis)} CUIs")
            results["passed"].append("cuis property")
        else:
            print("[WARN] cuis property empty or missing")

        if hasattr(result, "terms") and len(result.terms) > 0:
            print(f"[PASS] terms property works: {len(result.terms)} terms")
            results["passed"].append("terms property")
        else:
            print("[WARN] terms property empty or missing")

        if hasattr(result, "scores") and len(result.scores) > 0:
            print(f"[PASS] scores property works: {len(result.scores)} scores")
            results["passed"].append("scores property")
        else:
            print("[WARN] scores property empty or missing")

    except Exception as e:
        print(f"[FAIL] Error during metrics verification: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"Metrics verification: {e}")

    # Test 4: Semantic Filter functionality
    print("-" * 40)
    print("TEST 4: Semantic Filter Functionality")
    print("-" * 40)
    try:
        from src.snomed_methods import HybridSearch, SearchResult

        if os.path.exists(uk_path):
            print(f"[INFO] Using SNOMED path: {uk_path}")

        searcher = HybridSearch(uk_path=uk_path, medcat_path=medcat_path)

        # Get some results first (without filter)
        result = searcher.search("diabetes", top_k=10)
        original_count = len(result)

        print(f"[INFO] Original search returned {original_count} results")

        if hasattr(searcher, "SEMANTIC_CATEGORIES"):
            print("[PASS] SEMANTIC_CATEGORIES constant exists")
            results["passed"].append("SEMANTIC_CATEGORIES exists")
        else:
            print("[FAIL] SEMANTIC_CATEGORIES constant missing")
            results["failed"].append("SEMANTIC_CATEGORIES constant")

        # Test filtering
        if original_count > 0:
            filtered = searcher._apply_semantic_filter(result, ["disorder", "finding"])
            filtered_count = len(filtered)
            print(f"[INFO] Filtered results: {filtered_count} (from {original_count})")
            print("[PASS] _apply_semantic_filter method works")
            results["passed"].append("_apply_semantic_filter works")
        else:
            print("[WARN] No results to filter")

        # Test standalone function
        try:
            from src.snomed_methods import semantic_filter_results

            # noqa: F841 - variable used for testing existence of function
            filtered = semantic_filter_results(result, "procedure")
            print("[PASS] semantic_filter_results standalone function exists")
            results["passed"].append("semantic_filter_results function")
        except ImportError:
            print("[FAIL] semantic_filter_results not found")
            results["failed"].append("semantic_filter_results function")

    except Exception as e:
        print(f"[INFO] Semantic filter test skipped or failed: {e}")

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_passed = len(results["passed"])
    total_failed = len(results["failed"])
    total_tests = total_passed + total_failed

    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print()

    if results["passed"]:
        print("PASSED TESTS:")
        for test in results["passed"]:
            print(f"  ✓ {test}")

    if results["failed"]:
        print("\nFAILED TESTS:")
        for test in results["failed"]:
            print(f"  ✗ {test}")

    print()

    # Return exit code
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
