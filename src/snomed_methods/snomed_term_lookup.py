#!/usr/bin/env python3
# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
"""SNOMED CT Term Lookup Module.

This module provides functionality for searching SNOMED CT concepts by term.
"""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from rapidfuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None


class SnomedTermLookup:
    """Lookup SNOMED CT concepts by term."""

    def __init__(
        self,
        snomed_description_path: str | None = None,
        *,
        active_only: bool = True,
    ) -> None:
        """Initialize with a SNOMED description file.

        Args:
            snomed_description_path: Path to sct2_Description_*.txt file
            active_only: If True, only include active concepts (default True)

        """
        self.snomed_description_path = snomed_description_path
        self.active_only = active_only
        self._load_data()

    def _load_data(self) -> None:
        """Load SNOMED description data from file."""
        if pd is None:
            msg = "pandas is required but not installed"
            raise ImportError(msg)

        if self.snomed_description_path is None:
            self.df = None
            self.cui_column = None
            self.concept_ids = []
            return

        try:
            df = pd.read_csv(self.snomed_description_path, sep="\t", low_memory=False)
        except pd.errors.EmptyDataError:
            # Empty file - treat as no data
            self.df = None
            self.cui_column = None
            self.concept_ids = []
            return

        if self.active_only and "active" in df.columns:
            # Handle both string and integer active column values
            df = df[
                (df["active"] == "1")
                | (df["active"] == 1)
                | (df["active"].astype(str) == "1")
            ]

        self.df = df

        if len(df.columns) == 0:
            self.cui_column = None
            self.concept_ids = []
            return

        # Handle different column naming in SNOMED RF2 format (case-insensitive)
        col_lower = {col.lower(): col for col in df.columns}

        if "conceptid" in col_lower:
            self.cui_column = col_lower["conceptid"]
        elif "sourceidentifier" in col_lower:
            self.cui_column = col_lower["sourceidentifier"]
        else:
            # Use the first column as concept ID
            self.cui_column = df.columns[0]

        self.concept_ids = df[self.cui_column].unique().tolist()

    def find_concepts_by_term(
        self,
        term: str,
        *,
        ignore_case: bool = True,
        match_prefix: bool = False,
        top_n: int | None = None,
    ) -> list[tuple[str, str]]:
        """Find concepts by matching a term.

        Args:
            term: The term to search for
            ignore_case: Case-insensitive matching (default True)
            match_prefix: Match terms starting with query (default False)
             top_n: Maximum number of results (default None, return all)

        Returns:
            List of tuples (cui, term) for each match

        """
        if self.df is None or len(self.df) == 0:
            return []

        search_df = self.df.copy()

        # Drop rows where term is NaN to avoid mask errors with non-boolean arrays
        search_df = search_df.dropna(subset=["term"])

        if len(search_df) == 0:
            return []

        if ignore_case:
            search_str_lower = term.lower()
            term_col_lower = search_df["term"].str.lower()

            if match_prefix:
                mask = term_col_lower.str.startswith(search_str_lower)
            else:
                # Use regex=False for literal string matching
                mask = term_col_lower.str.contains(search_str_lower, regex=False)
        # Case-sensitive matching - use original strings directly
        elif match_prefix:
            mask = search_df["term"].str.startswith(term)
        else:
            # For case-sensitive substring match without regex,
            # check each row explicitly to ensure true literal behavior
            target = term
            mask = []
            for val in search_df["term"]:
                if val is None or (isinstance(val, float) and str(val) == "nan"):
                    mask.append(False)
                else:
                    mask.append(target in str(val))

        results = search_df[mask]

        if top_n is not None:
            results = results.head(top_n)

        # Check if we have any results
        if len(results) == 0:
            return []

        return list(
            zip(
                results[self.cui_column].astype(str).tolist(),
                results["term"].tolist(),
            ),
        )

    def find_concepts_by_term_fuzzy(
        self,
        term: str,
        min_score: int = 50,
        top_n: int | None = None,
    ) -> list[tuple[str, str, int]]:
        """Find concepts using fuzzy string matching.

        Args:
            term: The term to search for
            min_score: Minimum similarity score (0-100, default 50)
            top_n: Maximum number of results

        Returns:
            List of tuples (_cui, _term, _score).
            Returns empty list if rapidfuzz not installed.

        """
        if fuzz is None or process is None:
            return []

        if self.df is None or len(self.df) == 0:
            return []

        df_with_terms = self.df.dropna(subset=["term"])
        if len(df_with_terms) == 0:
            return []

        terms = df_with_terms["term"].tolist()
        scores = process.extract(term, terms, scorer=fuzz.ratio)

        # Map index to concept ID using the column from df
        # process.extract returns (term_string, score, index)
        concept_values = df_with_terms[self.cui_column].tolist()
        results = [
            (concept_values[index], term_val, score)
            for term_val, score, index in scores
            if score >= min_score
        ]

        if top_n is not None:
            results = results[:top_n]

        return results

    def find_concepts_batch(
        self,
        terms: list[str],
        *,
        ignore_case: bool = True,
        match_prefix: bool = False,
    ) -> list[tuple[str, str, str]]:
        """Search for multiple terms at once.

        Args:
            terms: List of terms to search
            ignore_case: Case-insensitive matching
            match_prefix: Match prefix

        Returns:
            List of tuples (term, cui, matched_term)

        """
        results = []
        for term in terms:
            matches = self.find_concepts_by_term(
                term,
                ignore_case=ignore_case,
                match_prefix=match_prefix,
            )
            if matches:
                cui, matched_term = matches[0]
                results.append((term, cui, matched_term))
        return results

    def getconcept_info(self, cui: str) -> dict | None:
        """Get detailed information about a concept.

        Args:
            cui: The concept ID

        Returns:
            Dictionary with concept details, or None if not found

        """
        if self.df is None:
            return None

        try:
            row = self.df[self.df[self.cui_column] == int(cui)]
        except (ValueError, TypeError):
            return None

        if len(row) == 0:
            return None

        result = {
            "concept_id": str(row[self.cui_column].iloc[0]),
        }

        # Add available columns with snake_case keys
        column_mapping = {
            "term": "preferred_name",
            "typeId": "type_id",
            "preferredName": "preferred_name",
            "synonym": "synonym",
        }
        for df_col, result_key in column_mapping.items():
            if df_col in row.columns:
                result[result_key] = row[df_col].iloc[0]

        return result


def create_term_lookup_from_directory(
    sct2_dir: str,
    *,
    active_only: bool = True,
) -> SnomedTermLookup:
    """Create a SnomedTermLookup from a SNOMED directory.

    Automatically finds the description file.

    Args:
        sct2_dir: Path to SNOMED directory
        active_only: Only include active concepts

    Returns:
        SnomedTermLookup instance ready for searching

    Raises:
        FileNotFoundError: If description file not found

    """
    Path(sct2_dir) / "**" / "sct2_Description_*.txt"
    files = list(Path(sct2_dir).rglob("sct2_Description_*.txt"))

    if len(files) == 0:
        msg = (
            f"Description file not found in {sct2_dir}. "
            f"Expected file matching sct2_Description_*.txt"
        )
        raise FileNotFoundError(
            msg,
        )

    return SnomedTermLookup(files[0], active_only=active_only)


def main() -> int:
    """Run demo and return result count."""
    default_dir = Path(__file__).resolve().parents[1] / "uk_sct2cl_42.2.0"
    snomed_dir = os.environ.get("SNOMED_DIR", default_dir)

    if not pathlib.Path(snomed_dir).exists():
        sys.exit(1)

    demo_term_lookup = create_term_lookup_from_directory(snomed_dir)
    results = demo_term_lookup.find_concepts_by_term("meningioma", match_prefix=True)
    return len(results)


if __name__ == "__main__":
    count = main()
    sys.exit(0)
