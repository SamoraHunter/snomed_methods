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

active_df = snomed._get_active_df()

type_id = 116680003

children_df = active_df[
    (active_df["destinationId"] == meningioma_cui) & (active_df["typeId"] == type_id)
]
for _ in children_df.head().iterrows():
    pass

parents_df = active_df[
    (active_df["sourceId"] == meningioma_cui) & (active_df["typeId"] == type_id)
]
for _ in parents_df.head().iterrows():
    pass

descendants_ids, descendants_names = snomed.get_subsumed_concepts(
    meningioma_cui,
    include_ancestors=False,
    include_descendants=True,
    max_depth=5,
)
for _ in enumerate(descendants_ids[:10]):
    pass
