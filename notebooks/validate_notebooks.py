#!/usr/bin/env python3
"""Validate notebooks without leaving execution output artifacts."""

import os
import sys
from pathlib import Path

import nbformat


def validate_notebook(notebook_path: str) -> tuple[bool, list[str]]:
    notebook = Path(notebook_path)
    if not notebook.exists():
        return False, [f"Notebook not found: {notebook_path}"]

    errors = []

    try:
        with open(notebook, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        return False, [f"Failed to read notebook: {e}"]

    hardcoded_patterns = [
        "/workspaces/snomed_methods",
        "C:\\\\",
        "/home/user/",
    ]

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            source = cell.source
            for j, line in enumerate(source.split("\n")):
                for pattern in hardcoded_patterns:
                    if pattern in line and not line.strip().startswith("#"):
                        errors.append(
                            f"Cell {i+1}, Line {j+1}: Potential hardcoded path found"
                        )

    project_root_patterns = [
        "project_root = os.path.abspath",
        'current_dir.endswith("notebooks")',
    ]

    has_dynamic_path = False
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source.lower()
            for pattern in project_root_patterns:
                if pattern in source:
                    has_dynamic_path = True
                    break

    if not has_dynamic_path and any(c.cell_type == "code" for c in nb.cells):
        errors.append("Notebook may lack dynamic PROJECT_ROOT/CURRENT_DIR setup")

    return len(errors) == 0, errors


def main():
    project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    notebooks_dir = project_root / "notebooks"

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))

    print(f"Validating {len(notebooks)} notebooks...\n")

    results: dict[str, tuple[str, list]] = {}
    for nb in notebooks:
        if any(x in nb.name for x in ["_executed", "nbconvert", "_tested"]):
            continue

        print(f"Validating: {nb.name}")
        success, errors = validate_notebook(str(nb))

        if success:
            print("  PASSED")
            results[nb.name] = ("passed", [])
        else:
            print(f"  FAILED ({len(errors)} issues)")
            for err in errors[:5]:
                print(f"     - {err}")
            if len(errors) > 5:
                print(f"     ... and {len(errors)-5} more")
            results[nb.name] = ("failed", errors)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r[0] == "passed")
    failed = sum(1 for r in results.values() if r[0] == "failed")

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed notebooks:")
        for name, (status, _) in results.items():
            if status == "failed":
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
