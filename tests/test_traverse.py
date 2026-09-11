import sys

sys.path.insert(0, "..")

from snomed_methods_v1 import SnomedRelations

rel_file = "/workspaces/snomed_methods/uk_sct2cl_42.2.0/SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z/Full/Terminology/sct2_Relationship_UKCLFull_GB1000000_20260603.txt"

snomed = SnomedRelations(snomed_rf2_full_path=rel_file, medcat=False)

meningioma_cui = 438181000000108

# Test active df filtering
active_df = snomed._get_active_df()
print(f"Active relationships: {len(active_df)}")

type_id = 116680003

# Check children (destinationId == meningioma, sourceId are children)
children_df = active_df[
    (active_df["destinationId"] == meningioma_cui) & (active_df["typeId"] == type_id)
]
print(f"\nChildren of Meningioma (is_a): {len(children_df)}")
for _, row in children_df.head().iterrows():
    print(f"  child sourceId: {row['sourceId']}")

# Check parents (sourceId == meningioma, destinationId are parents)
parents_df = active_df[
    (active_df["sourceId"] == meningioma_cui) & (active_df["typeId"] == type_id)
]
print(f"\nParents of Meningioma (is_a): {len(parents_df)}")
for _, row in parents_df.head().iterrows():
    print(f"  parent destinationId: {row['destinationId']}")

# Now test the actual traversal
print("\n" + "=" * 70)
print("Testing get_subsumed_concepts()...")
print("=" * 70)

descendants_ids, descendants_names = snomed.get_subsumed_concepts(
    meningioma_cui, include_ancestors=False, include_descendants=True, max_depth=5
)

print(f"\nFound {len(descendants_ids)} concepts")
for i, (cui, name) in enumerate(zip(descendants_ids[:10], descendants_names[:10])):
    print(f"  {i+1}. CUI: {cui}, Name: {name}")
