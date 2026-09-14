#!/usr/bin/env python3
"""Test script for rag.py module"""

import os
import sys


def run_tests():
    """Run comprehensive tests on RAG classes."""

    print("=" * 80)
    print("RAG MODULE TESTS")
    print("=" * 80)
    print()

    results = {"passed": [], "failed": []}

    project_root = os.path.dirname(os.path.abspath(__file__))
    uk_path = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )

    # Test 1: Import RAG module
    print("-" * 40)
    print("TEST 1: Import RAG Module")
    print("-" * 40)

    try:
        from snomed_methods.rag import RAGChat, RAGExplanations, RAGRetriever

        print("[PASS] Successfully imported RAG classes")
        results["passed"].append("Import RAG classes")

    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        results["failed"].append(f"Import: {e}")

    # Test 2: Mock embedder for testing
    print()
    print("-" * 40)
    print("TEST 2: Create Mock Embedder")
    print("-" * 40)

    try:
        import numpy as np

        class MockEmbedder:
            def generate_embeddings(self, texts, batch_size=None):
                return [np.random.randn(384) for _ in texts]

        mock_embedder = MockEmbedder()
        print("[PASS] Created mock embedder")
        results["passed"].append("Mock embedder")

    except Exception as e:
        print(f"[FAIL] Mock embedder error: {e}")
        results["failed"].append(f"Mock embedder: {e}")

    # Test 3: RAGRetriever with mocked data
    print()
    print("-" * 40)
    print("TEST 3: RAGRetriever Initialization")
    print("-" * 40)

    try:
        import numpy as np

        cui_to_embedding = {
            "C0023956": np.random.randn(384),
            "C0013421": np.random.randn(384),
            "C0011861": np.random.randn(384),
        }
        cui_to_name = {
            "C0023956": "Meningioma",
            "C0013421": "Diabetes Mellitus",
            "C0011861": "Hypertension",
        }

        retriever = RAGRetriever(
            embedder=mock_embedder,
            cui_to_embedding=cui_to_embedding,
            cui_to_name=cui_to_name,
        )
        print("[PASS] RAGRetriever initialized successfully")
        results["passed"].append("RAGRetriever init")

    except Exception as e:
        print(f"[FAIL] RAGRetriever initialization error: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"RAGRetriever init: {e}")

    # Test 4: Retrieval functionality
    print()
    print("-" * 40)
    print("TEST 4: Retrieval Functionality")
    print("-" * 40)

    try:
        results_list = retriever.retrieve("brain tumor", top_k=5)

        print(f"Retrieved {len(results_list)} concepts:")
        for cui, name, score in results_list[:3]:
            print(f"  - [{score:.3f}] {name} ({cui})")

        if len(results_list) > 0:
            print("[PASS] Retrieval returned results")
            results["passed"].append("Retrieval works")
        else:
            print("[FAIL] No retrieval results")
            results["failed"].append("Retrieval empty")

    except Exception as e:
        print(f"[FAIL] Retrieval error: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"Retrieval: {e}")

    # Test 5: RAGExplanations initialization
    print()
    print("-" * 40)
    print("TEST 5: RAGExplanations Initialization")
    print("-" * 40)

    try:
        explanations = RAGExplanations(embedder=mock_embedder, backend="ollama")
        print("[PASS] RAGExplanations initialized (backend: ollama)")
        results["passed"].append("RAGExplanations init")

    except Exception as e:
        print(f"[FAIL] RAGExplanations initialization error: {e}")
        results["failed"].append(f"RAGExplanations init: {e}")

    # Test 6: Batch explanations generation (mock)
    print()
    print("-" * 40)
    print("TEST 6: Batch Explanations Generation")
    print("-" * 40)

    try:
        retrieved = [
            ("C0023956", "Meningioma", 0.85),
            ("C0013421", "Diabetes Mellitus", 0.62),
        ]

        print("[INFO] Skipping actual LLM call (would require Ollama)")

        # Check method existence
        if hasattr(explanations, "generate_batch_explanations"):
            print("[PASS] generate_batch_explanations method exists")
            results["passed"].append("generate_batch_explanations method")

    except Exception as e:
        print(f"[FAIL] Explanations test error: {e}")
        results["failed"].append(f"Explanations: {e}")

    # Test 7: RAGChat initialization
    print()
    print("-" * 40)
    print("TEST 7: RAGChat Initialization")
    print("-" * 40)

    try:
        rag_chat = RAGChat(
            retriever=retriever,
            explanations=explanations if "explanations" in dir() else None,
            max_context_turns=5,
        )
        print("[PASS] RAGChat initialized")
        results["passed"].append("RAGChat init")

        # Test ask method
        response = rag_chat.ask(query="lung cancer", top_k=3)
        if "results" in response and len(response["results"]) > 0:
            print(f"[PASS] Chat ask() returned {len(response['results'])} results")
            results["passed"].append("Chat ask()")
        else:
            print("[FAIL] Chat ask() failed")
            results["failed"].append("Chat ask()")

    except Exception as e:
        print(f"[FAIL] RAGChat error: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"RAGChat: {e}")

    # Test 8: Follow-up query
    print()
    print("-" * 40)
    print("TEST 8: Follow-Up Query")
    print("-" * 40)

    try:
        feedback_response = rag_chat.follow_up(
            feedback="but only malignant",
            refine_with_previous=True,
        )
        if "results" in feedback_response:
            print(
                f"[PASS] Follow-up returned {len(feedback_response['results'])} results"
            )
            results["passed"].append("Chat follow-up()")
        else:
            print("[FAIL] Follow-up failed")
            results["failed"].append("Chat follow-up()")

    except Exception as e:
        print(f"[FAIL] Follow-up error: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"Follow-up: {e}")

    # Test 9: Reset conversation
    print()
    print("-" * 40)
    print("TEST 9: Reset Conversation")
    print("-" * 40)

    try:
        rag_chat.reset()
        if len(rag_chat.conversation_history) == 0:
            print("[PASS] Conversation history reset")
            results["passed"].append("Chat reset()")
        else:
            print(
                f"[FAIL] History not empty: {len(rag_chat.conversation_history)} turns"
            )
            results["failed"].append("Chat reset()")

    except Exception as e:
        print(f"[FAIL] Reset error: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"Reset: {e}")

    # Test 10: Save/Load index
    print()
    print("-" * 40)
    print("TEST 10: Save and Load Index")
    print("-" * 40)

    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rag_test.pkl")

            retriever.save_index(index_path)
            print("[PASS] RAG index saved")

            new_retriever = RAGRetriever(
                embedder=mock_embedder,
                index_path=index_path,
            )
            print("[PASS] RAG index loaded")
            results["passed"].append("Save/Load index")

    except Exception as e:
        print(f"[FAIL] Save/load error: {e}")
        import traceback

        traceback.print_exc()
        results["failed"].append(f"Save/Load index: {e}")

    # Summary
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

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
