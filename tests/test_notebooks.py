#!/usr/bin/env python3
"""Test notebooks without leaving execution output artifacts.

This script executes each notebook and verifies it runs successfully.
It uses temp directory for each execution to avoid polluting the source dir.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path


def test_notebook(notebook_path: str) -> tuple[bool, list[str]]:
    """
    Test a notebook by executing it in a temp directory.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Tuple of (success: bool, errors: list of error messages)
    """
    notebook = Path(notebook_path)
    if not notebook.exists():
        return False, [f"Notebook not found: {notebook_path}"]

    # Create temp directory for execution
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_nb = Path(tmpdir) / notebook.name
        shutil.copy2(notebook, tmp_nb)

        python_bin = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "snomed_methods_env",
            "bin",
            "python",
        )
        cmd = (
            f'"{python_bin}" -m nbconvert '
            f"--to notebook --execute {tmp_nb} "
            f"--output {tmp_nb.stem}_tested.ipynb"
        )

        result = os.system(cmd)

        if result != 0:
            return False, ["Notebook execution failed"]

        return True, []


def main():
    """Test all notebooks in the notebooks directory."""
    project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    notebooks_dir = project_root / "notebooks"

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))

    print(f"Testing {len(notebooks)} notebooks...\n")

    results = {}
    for nb in notebooks:
        if "_executed" in nb.name or "nbconvert" in nb.name or "_tested" in nb.name:
            continue  # Skip auto-generated execution files

        print(f"Testing: {nb.name}")
        success, errors = test_notebook(str(nb))

        if success:
            print("  PASSED")
            results[nb.name] = ("passed", [])
        else:
            print("  FAILED")
            for err in errors:
                print(f"     Error: {err[:200]}...")
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
