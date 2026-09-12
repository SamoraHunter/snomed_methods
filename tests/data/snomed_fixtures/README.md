# SNOMED CT UK Clinical RF2 Sample Fixtures

This directory contains a small sample of data from the UK Clinical Edition SNOMED RF2 release, extracted for use in unit tests.

## What's Included

### `relationships_sample.txt`
- A subset of **sct2_Relationship_UKCLFull_GB1000000_20260603.txt**
- Contains **50 sample relationships** from the UK Clinical Edition
- Format: SNOMED RF2 tab-separated file with header row
- All active relationships (active=1) from lines 95617-108145 of the original file

### `descriptions_sample.txt`
- A subset of **sct2_Description_UKCLFull-en_GB1000000_20260603.txt**
- Contains **51 sample descriptions** from the UK Clinical Edition
- Format: SNOMED RF2 tab-separated file with header row
- All active descriptions (active=1) from lines 95617-108145 of the original file

## Sample Concepts

The fixtures include data for concepts commonly used in tests:

| Concept ID | Term | Description Type |
|------------|------|------------------|
| 438181000000108 | [M]Meningioma NOS (morphologic abnormality) | SNOMED CT concept (morphologic abnormality) |
| 245931000000107 | Neoplasm of brain (disorder) | SNOMED CT concept (disorder) |

**Note:** The parent concept `127579001` referenced in the meningioma relationship does not exist in the UK Clinical Edition and is excluded from the fixture.

## Why Sample Data?

Full SNOMED RF2 files are large (typically 100MB+):
- sct2_Relationship: ~60MB
- sct2_Description: ~42MB

Sample fixtures enable:
- **Faster test execution** - No need to load multi-megabyte files for each test run
- **No external dependencies** - Tests work without downloading full RF2 releases
- **Consistent test data** - Version-controlled, reproducible test scenarios
- **Isolated testing environment** - Independent from production data

## Source Information

**Release:** SNOMED CT UK Clinical Edition v42.2.0
**Production Date:** 2026-06-03T12:00:00Z
**Version File:** `uk_sct2cl_42.2.0/SnomedCT_UKClinicalRF2_PRODUCTION_20260603T000001Z`

## Usage

Tests can reference these files directly:

```python
import os

fixture_dir = os.path.join("tests", "data", "snomed_fixtures")
rel_file = os.path.join(fixture_dir, "relationships_sample.txt")
desc_file = os.path.join(fixture_dir, "descriptions_sample.txt")

# For SnomedRelations:
from snomed_methods_v1 import SnomedRelations
snomed = SnomedRelations(snomed_rf2_full_path=rel_file, medcat=False)

# For SnomedTermLookup:
from snomed_term_lookup import SnomedTermLookup
lookup = SnomedTermLookup(desc_file)
```

## Line Counts

- **relationships_sample.txt**: 51 lines (1 header + 50 data rows)
- **descriptions_sample.txt**: 52 lines (1 header + 51 data rows)
