#!/usr/bin/env python3
"""Test notebooks without leaving execution output artifacts.

This script executes each notebook and verifies it runs successfully.
It uses temp directory for each execution to avoid polluting the source dir.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_notebook(notebook_path: str) -> tuple[bool, list[str]]:
    """Test a notebook by executing it in a temp directory.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Tuple of (success: bool, errors: list of error messages)

    """
    notebook = Path(notebook_path)
    if not notebook.exists():
        return False, [f"Notebook not found: {notebook_path}"]

    python_bin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "snomed_methods_env",
        "bin",
        "python",
    )

    # Skip if Python venv doesn't exist (dev container setup issue)
    if not os.path.exists(python_bin):
        return False, [
            "Python venv not found - skipping notebook tests in dev container",
        ]

    try:
        result = subprocess.run(
            [
                python_bin,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(notebook),
                "--output",
                "/tmp/tested.ipynb",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Notebook execution failed"
            return False, [error_msg[:500]]

        return True, []
    except subprocess.TimeoutExpired:
        return False, ["Notebook execution timed out"]
    except FileNotFoundError:
        return False, [f"Python not found at {python_bin}"]


def main() -> None:
    """Test all notebooks in the notebooks directory."""
    project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    notebooks_dir = project_root / "notebooks"

    if not notebooks_dir.exists():
        sys.stdout.write(f"Notebooks directory not found: {notebooks_dir}\n")
        sys.exit(1)

    # Check if venv exists
    python_bin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "snomed_methods_env",
        "bin",
        "python",
    )

    if not os.path.exists(python_bin):
        sys.stdout.write(
            f"Notebook tests skipped - Python venv not found at {python_bin}\n",
        )
        sys.stdout.write(
            "This is expected in dev container setups without full environment setup.\n",
        )
        sys.exit(0)

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))

    sys.stdout.write(f"Testing {len(notebooks)} notebooks...\n\n")

    results = {}
    for nb in notebooks:
        if "_executed" in nb.name or "nbconvert" in nb.name or "_tested" in nb.name:
            continue

        sys.stdout.write(f"Testing: {nb.name}\n")
        success, errors = test_notebook(str(nb))

        if success:
            sys.stdout.write("  PASSED\n")
            results[nb.name] = ("passed", [])
        else:
            sys.stdout.write("  FAILED\n")
            for err in errors:
                sys.stdout.write(f"     Error: {err[:200]}...\n")
            results[nb.name] = ("failed", errors)

    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write("SUMMARY\n")
    sys.stdout.write("=" * 60 + "\n")

    passed = sum(1 for r in results.values() if r[0] == "passed")
    failed = sum(1 for r in results.values() if r[0] == "failed")

    sys.stdout.write(f"Passed: {passed}/{len(results)}\n")
    sys.stdout.write(f"Failed: {failed}/{len(results)}\n")

    if failed > 0:
        sys.stdout.write("\nFailed notebooks:\n")
        for name, (status, _) in results.items():
            if status == "failed":
                sys.stdout.write(f"  - {name}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
