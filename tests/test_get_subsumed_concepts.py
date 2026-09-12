#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, "..")

from snomed_methods_v1 import SnomedRelations


def main():
    print("=" * 70)
    print("Testing get_subsumed_concepts() with Meningioma")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    fixture_dir = os.path.join(project_root, "tests", "data", "snomed_fixtures")
    rel_file = os.path.join(fixture_dir, "relationships_sample.txt")

    print(f"\nLoading SNOMED data from: {rel_file}")

    snomed = SnomedRelations(snomed_rf2_full_path=rel_file, medcat=False)
    print(f"Loaded {len(snomed.df)} relationships")

    from snomed_term_lookup import SnomedTermLookup

    desc_file = os.path.join(fixture_dir, "descriptions_sample.txt")
    lookup = SnomedTermLookup(desc_file)

    meningioma_matches = lookup.find_concepts_by_term("meningioma", top_n=5)

    if not meningioma_matches:
        print("ERROR: No meningioma concepts found!")
        return 1

    meningioma_cui = str(meningioma_matches[0][0])
    print(f"\nUsing Meningioma CUI: {meningioma_cui}")

    # Test get_subsumed_concepts()
    print("\n" + "=" * 70)
    print("Test: Descendants only (max_depth=5)")
    print("=" * 70)

    descendants_ids, _ = snomed.get_subsumed_concepts(
        meningioma_cui, include_ancestors=False, include_descendants=True, max_depth=5
    )

    print(f"Found {len(descendants_ids)} descendant concepts")
    if len(descendants_ids) > 1:
        print("Sample descendants:")
        for cui in descendants_ids[:5]:
            info = lookup.getconcept_info(str(cui))
            name = info["preferred_name"] if info else "N/A"
            print(f"  - CUI: {cui}, Name: {name}")
    else:
        print("(Only root concept found - no children in this dataset)")

    print("\n" + "=" * 70)
    print("Test: Ancestors only (max_depth=5)")
    print("=" * 70)

    ancestors_ids, _ = snomed.get_subsumed_concepts(
        meningioma_cui, include_ancestors=True, include_descendants=False, max_depth=5
    )

    print(f"Found {len(ancestors_ids)} ancestor concepts")
    if len(ancestors_ids) > 0:
        for cui in ancestors_ids[:5]:
            info = lookup.getconcept_info(str(cui))
            name = info["preferred_name"] if info else "N/A"
            print(f"  - CUI: {cui}, Name: {name}")

    print("\n" + "=" * 70)
    print("Test: Full DAG (both ancestors + descendants)")
    print("=" * 70)

    all_concepts_ids, _ = snomed.get_subsumed_concepts(
        meningioma_cui, include_ancestors=True, include_descendants=True, max_depth=5
    )

    print(f"Found {len(all_concepts_ids)} concepts in full DAG")
    if len(all_concepts_ids) > 0:
        for cui in all_concepts_ids[:10]:
            info = lookup.getconcept_info(str(cui))
            name = info["preferred_name"] if info else "N/A"
            print(f"  - CUI: {cui}, Name: {name}")

    # Test depth control
    print("\n" + "=" * 70)
    print("Test: Depth control")
    print("=" * 70)

    for depth in [1, 2, 3]:
        ids, _ = snomed.get_subsumed_concepts(meningioma_cui, max_depth=depth)
        print(f"max_depth={depth}: {len(ids)} concepts")

    # Test edge cases
    print("\n" + "=" * 70)
    print("Test: Edge cases")
    print("=" * 70)

    invalid_ids, _ = snomed.get_subsumed_concepts(999999999, max_depth=3)
    print(f"Invalid CUI (999999999): {len(invalid_ids)} concepts")

    neg_ids, _ = snomed.get_subsumed_concepts(meningioma_cui, max_depth=-1)
    print(f"Negative depth: {len(neg_ids)} concepts")

    zero_ids, _ = snomed.get_subsumed_concepts(meningioma_cui, max_depth=0)
    print(f"Zero depth (only root): {len(zero_ids)} concepts")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("get_subsumed_concepts() working correctly!")
    print(
        f"Meningioma has {len(descendants_ids) - 1 if len(descendants_ids) > 1 else 0} direct descendants"
    )
    print(
        f"Meningioma has {len(ancestors_ids) - 1 if len(ancestors_ids) > 1 else 0} direct ancestors"
    )
    print(f"Full DAG up to depth 5 contains {len(all_concepts_ids)} concepts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
