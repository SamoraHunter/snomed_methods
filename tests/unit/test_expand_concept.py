import os
import sys

sys.path.insert(0, "..")

from src.snomed_methods import SnomedTermLookup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
fixture_dir = os.path.join(PROJECT_ROOT, "tests", "data", "snomed_fixtures")
desc_file = os.path.join(fixture_dir, "descriptions_sample.txt")

lookup = SnomedTermLookup(desc_file)

concepts = [438181000000108, 127579001, 409681000000102]

for cui in concepts:
    info = lookup.getconcept_info(str(cui))
    print(f"CUI: {cui}")
    if info:
        print(f"  Name: {info.get('preferred_name', 'N/A')}")
        print(f"  Type: {info.get('type_id', 'N/A')}")
    else:
        print("  Not found")
