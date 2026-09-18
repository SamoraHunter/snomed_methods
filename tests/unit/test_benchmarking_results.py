#!/usr/bin/env python3
"""Unit tests for benchmarking results evaluation and visualization module."""

import os
import tempfile

import pandas as pd
import pytest


@pytest.fixture
def sample_benchmark_results():
    """Create sample benchmark results dictionary."""
    return {
        "method_a": {
            "precision": 0.85,
            "recall": 0.92,
            "f1_score": 0.88,
            "accuracy": 0.91,
        },
        "method_b": {
            "precision": 0.78,
            "recall": 0.85,
            "f1_score": 0.81,
            "accuracy": 0.84,
        },
        "method_c": {
            "precision": 0.92,
            "recall": 0.79,
            "f1_score": 0.85,
            "accuracy": 0.88,
        },
    }


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_basic(self, sample_benchmark_results) -> None:
        """Test basic aggregation of benchmark results."""
        from src.snomed_methods.benchmarking.results import aggregate_results

        df = aggregate_results(sample_benchmark_results)

        assert len(df) == 3
        assert "method_a" in df.index
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert df.loc["method_a", "precision"] == 0.85

    def test_aggregate_removes_num_samples(self, sample_benchmark_results) -> None:
        """Test that num_samples column is removed if present."""
        from src.snomed_methods.benchmarking.results import aggregate_results

        results_with_samples = {
            "method_a": {"precision": 0.85, "num_samples": 100},
            "method_b": {"precision": 0.78, "num_samples": 200},
        }
        df = aggregate_results(results_with_samples)

        assert "num_samples" not in df.columns

    def test_aggregate_empty_input(self) -> None:
        """Test aggregation with empty input."""
        from src.snomed_methods.benchmarking.results import aggregate_results

        df = aggregate_results({})

        assert len(df) == 0

    def test_aggregate_single_method(self, sample_benchmark_results) -> None:
        """Test aggregation with single method."""
        from src.snomed_methods.benchmarking.results import aggregate_results

        single = {"method_a": sample_benchmark_results["method_a"]}
        df = aggregate_results(single)

        assert len(df) == 1
        assert "method_a" in df.index

    def test_aggregate_numeric_only(self, sample_benchmark_results) -> None:
        """Test that only numeric columns are included."""
        from src.snomed_methods.benchmarking.results import aggregate_results

        results_with_string = {
            "method_a": {
                "precision": 0.85,
                "description": "test method",
            },
        }
        df = aggregate_results(results_with_string)

        assert len(df) == 1
        assert "precision" in df.columns


class TestCompareBenchmarks:
    """Tests for compare_benchmarks function."""

    def test_compare_all_metrics(self, sample_benchmark_results) -> None:
        """Test comparing benchmarks across all metrics."""
        from src.snomed_methods.benchmarking.results import compare_benchmarks

        df = compare_benchmarks(sample_benchmark_results)

        assert len(df) == 3
        assert "precision" in df.columns
        assert "recall" in df.columns

    def test_compare_specific_metric(self, sample_benchmark_results) -> None:
        """Test comparing benchmarks for a specific metric."""
        from src.snomed_methods.benchmarking.results import compare_benchmarks

        df = compare_benchmarks(sample_benchmark_results, metric="precision")

        assert len(df) == 3
        assert list(df.columns) == ["precision"]

    def test_compare_metric_not_found(self, sample_benchmark_results) -> None:
        """Test comparing with non-existent metric returns all."""
        from src.snomed_methods.benchmarking.results import compare_benchmarks

        df = compare_benchmarks(sample_benchmark_results, metric="nonexistent")

        assert "precision" in df.columns
        assert "recall" in df.columns


class TestSummarizeBenchmark:
    """Tests for summarize_benchmark function."""

    def test_summarize_basic(self, sample_benchmark_results) -> None:
        """Test basic summary generation."""
        from src.snomed_methods.benchmarking.results import summarize_benchmark

        df = summarize_benchmark(sample_benchmark_results)

        assert len(df) == 4
        assert "precision" in df.index or "precision" in df.columns
        assert "mean" in df.columns
        assert isinstance(df.loc["precision", "mean"], float)
        assert isinstance(df.loc["recall", "std"], float)

    def test_summarize_single_value(self) -> None:
        """Test summary with single value per metric."""
        from src.snomed_methods.benchmarking.results import summarize_benchmark

        results = {"method_a": {"precision": 0.85}}
        df = summarize_benchmark(results)

        assert len(df) > 0
        for col in df.columns:
            assert isinstance(df.loc["precision", col], (int, float))

    def test_summarize_with_string_columns(self) -> None:
        """Test summary excludes non-numeric columns."""
        from src.snomed_methods.benchmarking.results import summarize_benchmark

        results = {
            "method_a": {
                "precision": 0.85,
                "name": "test",
            },
        }
        df = summarize_benchmark(results)

        assert len(df) > 0
        assert "precision" in df.index


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_csv(self, sample_benchmark_results) -> None:
        """Test saving results as CSV."""
        from src.snomed_methods.benchmarking.results import save_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            save_results(sample_benchmark_results, output_path, file_format="csv")

            df = pd.read_csv(output_path)
            assert len(df) == 3
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_json(self, sample_benchmark_results) -> None:
        """Test saving results as JSON."""
        from src.snomed_methods.benchmarking.results import save_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_results(sample_benchmark_results, output_path, file_format="json")

            import json

            with open(output_path) as f:
                data = json.load(f)
            assert len(data) == 4
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_invalid_format(self, sample_benchmark_results) -> None:
        """Test saving with invalid format raises error."""
        from src.snomed_methods.benchmarking.results import save_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                save_results(sample_benchmark_results, output_path, file_format="txt")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestLoadResults:
    """Tests for load_results function."""

    def test_load_csv(self, sample_benchmark_results) -> None:
        """Test loading results from CSV."""
        from src.snomed_methods.benchmarking.results import load_results, save_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            save_results(sample_benchmark_results, output_path, file_format="csv")
            loaded = load_results(output_path)

            assert len(loaded) == 3
            assert "method_a" in loaded
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_load_json(self, sample_benchmark_results) -> None:
        """Test loading results from JSON."""
        from src.snomed_methods.benchmarking.results import load_results, save_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_results(sample_benchmark_results, output_path, file_format="json")
            loaded = load_results(output_path)

            assert len(loaded) == 4
            assert "precision" in loaded or "method_a" in loaded
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_load_invalid_format(self) -> None:
        """Test loading with invalid format raises error."""
        from src.snomed_methods.benchmarking.results import load_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_results(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestRankBenchmarks:
    """Tests for rank_benchmarks function."""

    def test_rank_descending(self, sample_benchmark_results) -> None:
        """Test ranking in descending order."""
        from src.snomed_methods.benchmarking.results import rank_benchmarks

        df = rank_benchmarks(
            sample_benchmark_results,
            metric="precision",
            ascending=False,
        )

        values = df["precision"].tolist()
        assert values == sorted(values, reverse=True)

    def test_rank_ascending(self, sample_benchmark_results) -> None:
        """Test ranking in ascending order."""
        from src.snomed_methods.benchmarking.results import rank_benchmarks

        df = rank_benchmarks(
            sample_benchmark_results,
            metric="precision",
            ascending=True,
        )

        values = df["precision"].tolist()
        assert values == sorted(values)

    def test_rank_by_recall(self, sample_benchmark_results) -> None:
        """Test ranking by recall."""
        from src.snomed_methods.benchmarking.results import rank_benchmarks

        df = rank_benchmarks(sample_benchmark_results, metric="recall")

        method_a_idx = df.index.get_loc("method_a")
        assert df.iloc[method_a_idx]["recall"] == 0.92


class TestBestPerformance:
    """Tests for best_performance function."""

    def test_best_precision(self, sample_benchmark_results) -> None:
        """Test finding best by precision."""
        from src.snomed_methods.benchmarking.results import best_performance

        name, score = best_performance(sample_benchmark_results, metric="precision")

        assert name == "method_c"
        assert score == 0.92

    def test_best_recall(self, sample_benchmark_results) -> None:
        """Test finding best by recall."""
        from src.snomed_methods.benchmarking.results import best_performance

        name, score = best_performance(sample_benchmark_results, metric="recall")

        assert name == "method_a"
        assert score == 0.92


class TestWorstPerformance:
    """Tests for worst_performance function."""

    def test_worst_precision(self, sample_benchmark_results) -> None:
        """Test finding worst by precision."""
        from src.snomed_methods.benchmarking.results import worst_performance

        name, score = worst_performance(sample_benchmark_results, metric="precision")

        assert name == "method_b"
        assert score == 0.78

    def test_worst_recall(self, sample_benchmark_results) -> None:
        """Test finding worst by recall."""
        from src.snomed_methods.benchmarking.results import worst_performance

        name, score = worst_performance(sample_benchmark_results, metric="recall")

        assert name == "method_c"
        assert score == 0.79


class TestGetAllMetrics:
    """Tests for get_all_metrics function."""

    def test_get_all_metrics_basic(self, sample_benchmark_results) -> None:
        """Test getting all unique metrics."""
        from src.snomed_methods.benchmarking.results import get_all_metrics

        metrics = get_all_metrics(sample_benchmark_results)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "accuracy" in metrics

    def test_get_all_metrics_empty(self) -> None:
        """Test getting metrics from empty results."""
        from src.snomed_methods.benchmarking.results import get_all_metrics

        metrics = get_all_metrics({})

        assert len(metrics) == 0

    def test_get_all_metrics_sorted(self, sample_benchmark_results) -> None:
        """Test that returned metrics are sorted."""
        from src.snomed_methods.benchmarking.results import get_all_metrics

        metrics = get_all_metrics(sample_benchmark_results)

        assert metrics == sorted(metrics)


class TestFilterMetrics:
    """Tests for filter_metrics function."""

    def test_filter_include(self, sample_benchmark_results) -> None:
        """Test filtering with include patterns."""
        from src.snomed_methods.benchmarking.results import filter_metrics

        filtered = filter_metrics(sample_benchmark_results, include_patterns=["prec"])

        assert "precision" in filtered["method_a"]
        assert "recall" not in filtered["method_a"]

    def test_filter_exclude(self, sample_benchmark_results) -> None:
        """Test filtering with exclude patterns."""
        from src.snomed_methods.benchmarking.results import filter_metrics

        filtered = filter_metrics(sample_benchmark_results, exclude_patterns=["cal"])

        assert "precision" in filtered["method_a"]
        assert "recall" not in filtered["method_a"]

    def test_filter_both_include_exclude(self, sample_benchmark_results) -> None:
        """Test filtering with both include and exclude patterns."""
        from src.snomed_methods.benchmarking.results import filter_metrics

        filtered = filter_metrics(
            sample_benchmark_results,
            include_patterns=["f1", "acc"],
            exclude_patterns=["score"],
        )

        assert "f1_score" not in filtered["method_a"]
        assert "accuracy" in filtered["method_a"]

    def test_filter_no_patterns(self, sample_benchmark_results) -> None:
        """Test filtering with no patterns returns all."""
        from src.snomed_methods.benchmarking.results import filter_metrics

        filtered = filter_metrics(sample_benchmark_results)

        assert filtered == sample_benchmark_results


class TestEvaluateSuite:
    """Tests for evaluate_suite function."""

    def test_evaluate_basic(self, sample_benchmark_results) -> None:
        """Test basic suite evaluation."""
        from src.snomed_methods.benchmarking.results import evaluate_suite

        df = evaluate_suite(sample_benchmark_results)

        assert len(df) == 3
        assert "method_a" in df.index

    def test_evaluate_with_output_dir(self, sample_benchmark_results) -> None:
        """Test suite evaluation with output directory."""
        from src.snomed_methods.benchmarking.results import evaluate_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            df = evaluate_suite(sample_benchmark_results, output_dir=tmpdir)

            assert len(df) == 3
            assert "method_a" in df.index

    def test_evaluate_empty(self) -> None:
        """Test suite evaluation with empty results."""
        from src.snomed_methods.benchmarking.results import evaluate_suite

        df = evaluate_suite({})

        assert len(df) == 0


class TestBenchmarkingResultsEdgeCases:
    """Tests for edge cases in benchmarking results."""

    def test_aggregate_single_numeric_value(self) -> None:
        """Test aggregation with single numeric value."""
        from src.snomed_methods.benchmarking.results import aggregate_results

        results = {"method_a": {"precision": 0.85}}
        df = aggregate_results(results)

        assert len(df) == 1
        assert df.loc["method_a", "precision"] == 0.85

    def test_compare_metric_partial_match(self, sample_benchmark_results) -> None:
        """Test comparing with partial metric name."""
        from src.snomed_methods.benchmarking.results import compare_benchmarks

        df = compare_benchmarks(sample_benchmark_results, metric="prec")

        assert len(df) == 3
        assert "precision" in df.columns

    def test_rank_with_ties(self) -> None:
        """Test ranking with tied values."""
        from src.snomed_methods.benchmarking.results import rank_benchmarks

        results = {
            "method_a": {"score": 0.85},
            "method_b": {"score": 0.85},
            "method_c": {"score": 0.75},
        }
        df = rank_benchmarks(results, metric="score", ascending=False)

        assert len(df) == 3

    def test_filter_empty_results(self) -> None:
        """Test filtering empty results."""
        from src.snomed_methods.benchmarking.results import filter_metrics

        filtered = filter_metrics({}, include_patterns=["test"])

        assert filtered == {}

    def test_summary_with_all_same_values(self) -> None:
        """Test summary when all values are the same."""
        from src.snomed_methods.benchmarking.results import summarize_benchmark

        results = {
            "method_a": {"precision": 0.85},
            "method_b": {"precision": 0.85},
            "method_c": {"precision": 0.85},
        }
        df = summarize_benchmark(results)

        assert df.loc["precision", "std"] == 0
        assert df.loc["precision", "min"] == 0.85
        assert df.loc["precision", "max"] == 0.85

    def test_evaluate_suite_creates_plot(self, sample_benchmark_results) -> None:
        """Test that evaluate_suite creates plot when output_dir is given."""
        from pathlib import Path

        from src.snomed_methods.benchmarking.results import evaluate_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluate_suite(sample_benchmark_results, output_dir=tmpdir)

            plot_path = Path(tmpdir) / "benchmark_comparison.png"
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
