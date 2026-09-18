#!/usr/bin/env python3
"""Tests for ProgressiveExpansionPipeline."""

import pytest


def test_pipeline_import() -> None:
    from snomed_methods import ProgressiveExpansionPipeline, expand_progressive

    assert ProgressiveExpansionPipeline is not None
    assert expand_progressive is not None


def test_stage_executors_exist() -> None:
    from snomed_methods.progressive_expansion import (
        StageExecutor,
        TermMatchingStage,
    )

    assert issubclass(StageExecutor, object)
    assert issubclass(TermMatchingStage, StageExecutor)


def test_expansion_result() -> None:
    from snomed_methods.progressive_expansion import ExpansionResult

    result = ExpansionResult(
        query="test",
        stages_executed=["term"],
        scores={"term": 0.8},
        sources={"term": ["C1", "C2"]},
    )
    assert result.query == "test"
    assert result.total_concepts == 2


def test_stage_weights() -> None:
    from snomed_methods.progressive_expansion import (
        HierarchyExpansionStage,
        TermMatchingStage,
    )

    assert TermMatchingStage().get_weight() == 0.5
    assert HierarchyExpansionStage().get_weight() == 0.3


def test_pipeline_with_minimal_data() -> None:
    from snomed_methods.progressive_expansion import (
        ProgressiveExpansionPipeline,
    )

    try:
        p = ProgressiveExpansionPipeline(backend="transformers")
        result = p.expand("test", stages=["term"], max_concepts=5)
        assert hasattr(result, "total_concepts")
    except ImportError:
        pytest.skip("Required packages not available")
    except FileNotFoundError:
        pytest.skip("SNOMED UK Clinical RF2 data not available in test environment")
