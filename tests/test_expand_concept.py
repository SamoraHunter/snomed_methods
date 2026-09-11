import sys

sys.path.insert(0, "..")

from snomed_term_lookup import create_term_lookup_from_directory

lookup = create_term_lookup_from_directory(
    "/workspaces/snomed_methods/uk_sct2cl_42.2.0/SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z"
)

concepts = [438181000000108, 127579001, 409681000000102]

for cui in concepts:
    info = lookup.getconcept_info(str(cui))
    print(f"CUI: {cui}")
    if info:
        print(f"  Name: {info.get('preferred_name', 'N/A')}")
        print(f"  Type: {info.get('type_id', 'N/A')}")
    else:
        print("  Not found")
