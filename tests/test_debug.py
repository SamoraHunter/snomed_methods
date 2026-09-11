import sys

sys.path.insert(0, "..")
import pandas as pd

from snomed_methods_v1 import SnomedRelations

df = pd.DataFrame(
    {
        "sourceId": [100, 200],
        "destinationId": [50, 100],
        "typeId": [116680003, 116680003],
        "active": ["1", "0"],
    }
)

df_path = "/tmp/test_inactive.csv"
df.to_csv(df_path, sep="\t", index=False)
snomed = SnomedRelations(snomed_rf2_full_path=df_path)

print("Full df:")
print(df)
print()

print("Active only df:")
active_df = snomed._get_active_df()
print(active_df)
print()

# Check relationships
print("Children of 50 (where destinationId=50):")
children = active_df[active_df["destinationId"] == 50]
print(children)

print("\nParents of 100 (where sourceId=100):")
parents = active_df[active_df["sourceId"] == 100]
print(parents)

result_ids, _ = snomed.get_subsumed_concepts(50, active_only=True)
print(f"\nResult: {result_ids}")
