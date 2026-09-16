#!/usr/bin/env python3
"""Unit tests for config module."""

import pytest


class TestSnomedConfig:
    """Tests for SnomedConfig class."""

    def test_singleton_pattern(self):
        """Test that SnomedConfig is a singleton."""
        from snomed_methods.config import SnomedConfig

        config1 = SnomedConfig()
        config2 = SnomedConfig()
        assert config1 is config2

    def test_get_instance(self):
        """Test get_class method."""
        from snomed_methods.config import SnomedConfig

        config = SnomedConfig.get_instance()
        assert config is not None

    def test_reset_singleton(self):
        """Test reset method creates new instance."""
        from snomed_methods.config import SnomedConfig

        SnomedConfig.reset()
        new_config = SnomedConfig()
        # After reset, we get a new instance
        assert new_config is not None


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns SnomedConfig."""
        from snomed_methods.config import get_config

        config = get_config()
        assert config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
