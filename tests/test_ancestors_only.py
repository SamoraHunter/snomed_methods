import os
import sys

sys.path.insert(0, "..")

from snomed_methods_v1 import SnomedRelations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
fixture_dir = os.path.join(PROJECT_ROOT, "tests", "data", "snomed_fixtures")
rel_file = os.path.join(fixture_dir, "relationships_sample.txt")

snomed = SnomedRelations(snomed_rf2_full_path=rel_file, medcat=False)

meningioma_cui = 438181000000108

print("Testing ancestors only...")
ancestors_ids, ancestors_names = snomed.get_subsumed_concepts(
    meningioma_cui, include_ancestors=True, include_descendants=False, max_depth=5
)

print(f"Found {len(ancestors_ids)} concepts")
for i, (cui, name) in enumerate(zip(ancestors_ids[:10], ancestors_names[:10])):
    print(f"  {i+1}. CUI: {cui}, Name: {name}")

print("\nTesting full DAG...")
all_concepts_ids, all_concepts_names = snomed.get_subsumed_concepts(
    meningioma_cui, include_ancestors=True, include_descendants=True, max_depth=5
)

print(f"Found {len(all_concepts_ids)} concepts")
for i, (cui, name) in enumerate(zip(all_concepts_ids[:10], all_concepts_names[:10])):
    print(f"  {i+1}. CUI: {cui}, Name: {name}")
