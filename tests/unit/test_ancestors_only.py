import os
import sys

sys.path.insert(0, "..")

from src.snomed_methods import SnomedRelations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
fixture_dir = os.path.join(PROJECT_ROOT, "data", "snomed_fixtures")
rel_file = os.path.join(fixture_dir, "relationships_sample.txt")

snomed = SnomedRelations(snomed_rf2_full_path=rel_file, medcat=False)

meningioma_cui = 438181000000108

ancestors_ids, ancestors_names = snomed.get_subsumed_concepts(
    meningioma_cui, include_ancestors=True, include_descendants=False, max_depth=5
)
for _ in zip(ancestors_ids[:10], ancestors_names[:10]):
    pass

all_concepts_ids, all_concepts_names = snomed.get_subsumed_concepts(
    meningioma_cui, include_ancestors=True, include_descendants=True, max_depth=5
)
for _ in zip(all_concepts_ids[:10], all_concepts_names[:10]):
    pass
