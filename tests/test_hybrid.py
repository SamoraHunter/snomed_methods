#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, ".")

try:
    from hybrid_search import HybridSearch

    print("IMPORT_SUCCESS")

    project_root = os.path.dirname(os.path.abspath(__file__))
    uk_path = os.path.join(
        project_root,
        "uk_sct2cl_42.2.0",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z",
    )
    medcat_path = os.path.join(
        project_root, "model_packs", "medcat_model_pack_422d1d38fc58f158.zip"
    )
    model_path = os.path.join(
        project_root, "embedding_models", "SapBERT-from-PubMedBERT-fulltext"
    )

    searcher = HybridSearch(
        uk_path=uk_path, medcat_path=medcat_path, model_path=model_path
    )
    print("INIT_SUCCESS")

    result = searcher.search("diabetes", top_k=5)
    print(f"SEARCH_SUCCESS: {result}")
    print(f"Results: {len(result)} concepts found")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
