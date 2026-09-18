#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, "..")

from src.snomed_methods import SnomedRelations


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    fixture_dir = os.path.join(project_root, "data", "snomed_fixtures")
    rel_file = os.path.join(fixture_dir, "relationships_sample.txt")

    snomed = SnomedRelations(snomed_rf2_full_path=rel_file, medcat=False)

    from src.snomed_methods import SnomedTermLookup

    desc_file = os.path.join(fixture_dir, "descriptions_sample.txt")
    lookup = SnomedTermLookup(desc_file)

    meningioma_matches = lookup.find_concepts_by_term("meningioma", top_n=5)

    if not meningioma_matches:
        return 1

    meningioma_cui = str(meningioma_matches[0][0])

    descendants_ids, _ = snomed.get_subsumed_concepts(
        meningioma_cui,
        include_ancestors=False,
        include_descendants=True,
        max_depth=5,
    )
    if len(descendants_ids) > 1:
        for _cui in descendants_ids[:5]:
            lookup.getconcept_info(str(_cui))
    else:
        pass

    ancestors_ids, _ = snomed.get_subsumed_concepts(
        meningioma_cui,
        include_ancestors=True,
        include_descendants=False,
        max_depth=5,
    )
    if len(ancestors_ids) > 0:
        for _cui in ancestors_ids[:5]:
            lookup.getconcept_info(str(_cui))

    all_concepts_ids, _ = snomed.get_subsumed_concepts(
        meningioma_cui,
        include_ancestors=True,
        include_descendants=True,
        max_depth=5,
    )
    if len(all_concepts_ids) > 0:
        for _cui in all_concepts_ids[:10]:
            lookup.getconcept_info(str(_cui))

    for depth in [1, 2, 3]:
        _ids, _ = snomed.get_subsumed_concepts(meningioma_cui, max_depth=depth)

    _invalid_ids, _ = snomed.get_subsumed_concepts(999999999, max_depth=3)
    _neg_ids, _ = snomed.get_subsumed_concepts(meningioma_cui, max_depth=-1)
    _zero_ids, _ = snomed.get_subsumed_concepts(meningioma_cui, max_depth=0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
