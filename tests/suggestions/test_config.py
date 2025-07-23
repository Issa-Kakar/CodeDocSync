"""
Comprehensive tests for suggestion configuration system.

Tests configuration classes, validation, loading/saving, and merging logic.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pytest
import yaml

from codedocsync.suggestions.config import (
    ConfigManager,
    RankingConfig,
    SuggestionConfig,
    config_manager,
    get_comprehensive_config,
    get_development_config,
    get_documentation_config,
    get_minimal_config,
)


class TestSuggestionConfig:
    """Test SuggestionConfig class."""

    def test_default_config_creation(self) -> None:
        """Test creating config with default values."""
        config = SuggestionConfig()

        assert config.default_style == "google"
        assert config.preserve_descriptions is True
        assert config.include_types is True
        assert config.max_line_length == 88
        assert config.confidence_threshold == 0.7
        assert config.max_suggestions_per_issue == 3

    def test_custom_config_creation(self) -> None:
        """Test creating config with custom values."""
        config = SuggestionConfig(
            default_style="numpy",
            max_line_length=100,
            confidence_threshold=0.8,
            include_examples=True,
        )

        assert config.default_style == "numpy"
        assert config.max_line_length == 100
        assert config.confidence_threshold == 0.8
        assert config.include_examples is True

    def test_config_validation_invalid_style(self) -> None:
        """Test validation fails for invalid default style."""
        with pytest.raises(ValueError, match="default_style must be one of"):
            SuggestionConfig(default_style="invalid_style")

    def test_config_validation_invalid_confidence(self) -> None:
        """Test validation fails for invalid confidence threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            SuggestionConfig(confidence_threshold=1.5)

        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            SuggestionConfig(confidence_threshold=-0.1)

    def test_config_validation_invalid_line_length(self) -> None:
        """Test validation fails for too small line length."""
        with pytest.raises(ValueError, match="max_line_length too small"):
            SuggestionConfig(max_line_length=30)

    def test_config_validation_invalid_indent_size(self) -> None:
        """Test validation fails for invalid indent size."""
        with pytest.raises(ValueError, match="indent_size must be between"):
            SuggestionConfig(indent_size=0)

        with pytest.raises(ValueError, match="indent_size must be between"):
            SuggestionConfig(indent_size=10)

    def test_config_validation_invalid_max_suggestions(self) -> None:
        """Test validation fails for invalid max suggestions."""
        with pytest.raises(
            ValueError, match="max_suggestions_per_issue must be positive"
        ):
            SuggestionConfig(max_suggestions_per_issue=0)

    def test_config_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config_dict = {
            "default_style": "sphinx",
            "max_line_length": 120,
            "confidence_threshold": 0.9,
            "include_examples": True,
        }

        config = SuggestionConfig.from_dict(config_dict)

        assert config.default_style == "sphinx"
        assert config.max_line_length == 120
        assert config.confidence_threshold == 0.9
        assert config.include_examples is True

    def test_config_from_dict_ignores_unknown_keys(self) -> None:
        """Test from_dict ignores unknown keys."""
        config_dict = {
            "default_style": "numpy",
            "unknown_key": "unknown_value",
            "another_unknown": 123,
        }

        config = SuggestionConfig.from_dict(config_dict)
        assert config.default_style == "numpy"
        # Should not have unknown attributes
        assert not hasattr(config, "unknown_key")

    def test_config_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = SuggestionConfig(
            default_style="rest",
            max_line_length=90,
            include_examples=True,
        )

        config_dict = config.to_dict()

        assert config_dict["default_style"] == "rest"
        assert config_dict["max_line_length"] == 90
        assert config_dict["include_examples"] is True
        # Should include all fields
        assert "confidence_threshold" in config_dict
        assert "preserve_descriptions" in config_dict

    def test_config_from_yaml_file(self) -> None:
        """Test loading config from YAML file."""
        yaml_content = """
        suggestions:
          default_style: numpy
          max_line_length: 100
          confidence_threshold: 0.8
          include_examples: true
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            config = SuggestionConfig.from_yaml_file(temp_path)

            assert config.default_style == "numpy"
            assert config.max_line_length == 100
            assert config.confidence_threshold == 0.8
            assert config.include_examples is True
        finally:
            temp_path.unlink()

    def test_config_from_yaml_file_missing_file(self) -> None:
        """Test loading config from missing file returns default."""
        non_existent_path = Path("/non/existent/file.yml")
        config = SuggestionConfig.from_yaml_file(non_existent_path)

        # Should return default config
        assert config.default_style == "google"
        assert config.max_line_length == 88

    def test_config_from_yaml_file_invalid_yaml(self) -> None:
        """Test loading config from invalid YAML raises error."""
        invalid_yaml = "invalid: yaml: content: {"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(invalid_yaml)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                SuggestionConfig.from_yaml_file(temp_path)
        finally:
            temp_path.unlink()

    def test_config_save_to_yaml(self) -> None:
        """Test saving config to YAML file."""
        config = SuggestionConfig(
            default_style="sphinx",
            max_line_length=120,
            include_examples=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_config.yml"

            config.save_to_yaml(temp_path)

            assert temp_path.exists()

            # Load and verify
            with open(temp_path) as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["suggestions"]["default_style"] == "sphinx"
            assert saved_data["suggestions"]["max_line_length"] == 120
            assert saved_data["suggestions"]["include_examples"] is True

    def test_config_get_style_config(self) -> None:
        """Test getting style-specific configuration."""
        config = SuggestionConfig()

        # Test Google style
        google_config = config.get_style_config("google")
        assert google_config["section_marker"] == ":"
        assert "{name}" in google_config["parameter_format"]
        assert "{type}" in google_config["parameter_format"]

        # Test NumPy style
        numpy_config = config.get_style_config("numpy")
        assert "\n" in numpy_config["section_marker"]
        assert "-" in numpy_config["section_marker"]

        # Test Sphinx style
        sphinx_config = config.get_style_config("sphinx")
        assert ":param" in sphinx_config["parameter_format"]
        assert ":type" in sphinx_config["parameter_format"]

        # Test unknown style defaults to Google
        unknown_config = config.get_style_config("unknown")
        assert unknown_config["section_marker"] == ":"

    def test_config_is_feature_enabled(self) -> None:
        """Test feature flag checking."""
        config = SuggestionConfig(
            include_types=True,
            include_examples=False,
            validate_syntax=True,
        )

        assert config.is_feature_enabled("types") is True
        assert config.is_feature_enabled("examples") is False
        assert config.is_feature_enabled("validation") is True
        assert config.is_feature_enabled("unknown_feature") is False

    def test_config_get_quality_thresholds(self) -> None:
        """Test getting quality control thresholds."""
        config = SuggestionConfig(
            confidence_threshold=0.8,
            require_actionable=True,
            validate_syntax=True,
        )

        thresholds = config.get_quality_thresholds()

        assert thresholds["confidence"] == 0.8
        assert thresholds["actionable"] == 1.0
        assert thresholds["syntax_valid"] == 1.0

    def test_config_abbreviation_map_defaults(self) -> None:
        """Test default abbreviation map."""
        config = SuggestionConfig()

        assert "param" in config.abbreviation_map
        assert config.abbreviation_map["param"] == "parameter"
        assert config.abbreviation_map["str"] == "string"
        assert config.abbreviation_map["int"] == "integer"

    def test_config_section_order_defaults(self) -> None:
        """Test default section ordering."""
        config = SuggestionConfig()

        google_order = config.section_order["google"]
        assert "summary" in google_order
        assert "args" in google_order
        assert "returns" in google_order

        numpy_order = config.section_order["numpy"]
        assert "parameters" in numpy_order
        assert "returns" in numpy_order


class TestRankingConfig:
    """Test RankingConfig class."""

    def test_default_ranking_config(self) -> None:
        """Test creating ranking config with defaults."""
        config = RankingConfig()

        assert config.severity_weight == 0.4
        assert config.confidence_weight == 0.3
        assert config.actionability_weight == 0.2
        assert config.impact_weight == 0.1
        assert config.min_confidence == 0.5
        assert config.max_suggestions == 10

    def test_ranking_config_validation_invalid_weights(self) -> None:
        """Test validation fails for invalid weights."""
        with pytest.raises(ValueError, match="All weights must be between 0.0 and 1.0"):
            RankingConfig(severity_weight=1.5)

        with pytest.raises(ValueError, match="All weights must be between 0.0 and 1.0"):
            RankingConfig(confidence_weight=-0.1)

    def test_ranking_config_validation_weight_sum(self) -> None:
        """Test validation of weight sum."""
        with pytest.raises(ValueError, match="Weights should sum to approximately 1.0"):
            RankingConfig(
                severity_weight=0.1,
                confidence_weight=0.1,
                actionability_weight=0.1,
                impact_weight=0.1,  # Sum = 0.4, too low
            )

    def test_ranking_config_validation_invalid_confidence(self) -> None:
        """Test validation fails for invalid confidence threshold."""
        with pytest.raises(ValueError, match="min_confidence must be between"):
            RankingConfig(min_confidence=1.5)

    def test_ranking_config_validation_invalid_max_suggestions(self) -> None:
        """Test validation fails for invalid max suggestions."""
        with pytest.raises(ValueError, match="max_suggestions must be positive"):
            RankingConfig(max_suggestions=0)

    def test_ranking_config_calculate_score(self) -> None:
        """Test score calculation for suggestions."""
        from codedocsync.suggestions.models import (
            Suggestion,
            SuggestionDiff,
            SuggestionMetadata,
            SuggestionType,
        )

        config = RankingConfig()

        # Create a mock suggestion
        diff = SuggestionDiff(
            original_lines=["test"],
            suggested_lines=["test"],
            start_line=1,
            end_line=1,
        )

        metadata = SuggestionMetadata(generator_type="test")

        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test",
            confidence=0.8,
            diff=diff,
            style="google",
            metadata=metadata,
            is_actionable=True,
        )

        # Mock severity attribute for testing
        # Note: Suggestion doesn't have severity, it's on the issue
        # but the config might expect it for ranking purposes

        score = config.calculate_score(suggestion)

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_ranking_config_get_severity_score(self) -> None:
        """Test severity score calculation."""
        config = RankingConfig()

        # Test known severities
        assert config._get_severity_score("low") == 0.0
        assert config._get_severity_score("medium") == 1 / 3
        assert config._get_severity_score("high") == 2 / 3
        assert config._get_severity_score("critical") == 1.0

        # Test unknown severity
        assert config._get_severity_score("unknown") == 0.5


class TestPredefinedConfigs:
    """Test predefined configuration functions."""

    def test_minimal_config(self) -> None:
        """Test minimal configuration."""
        config = get_minimal_config()

        assert config.confidence_threshold == 0.9
        assert config.prefer_minimal_changes is True
        assert config.include_examples is False
        assert config.max_suggestions_per_issue == 1
        assert config.require_actionable is True

    def test_comprehensive_config(self) -> None:
        """Test comprehensive configuration."""
        config = get_comprehensive_config()

        assert config.confidence_threshold == 0.5
        assert config.include_examples is True
        assert config.include_types is True
        assert config.include_default_values is True
        assert config.expand_abbreviations is True
        assert config.max_suggestions_per_issue == 3

    def test_development_config(self) -> None:
        """Test development configuration."""
        config = get_development_config()

        assert config.confidence_threshold == 0.6
        assert config.cache_suggestions is True
        assert config.merge_related_issues is True
        assert config.prefer_minimal_changes is True
        assert config.validate_syntax is True

    def test_documentation_config(self) -> None:
        """Test documentation configuration."""
        config = get_documentation_config()

        assert config.default_style == "google"
        assert config.preserve_descriptions is True
        assert config.preserve_examples is True
        assert config.include_types is True
        assert config.include_examples is True
        assert config.expand_abbreviations is True


class TestConfigManager:
    """Test ConfigManager class."""

    def test_config_manager_creation(self) -> None:
        """Test creating config manager."""
        manager = ConfigManager()
        assert manager._config_cache == {}

    def test_config_manager_load_config_default(self) -> None:
        """Test loading default config when no files exist."""
        manager = ConfigManager()
        config = manager.load_config()

        # Should return default config
        assert config.default_style == "google"
        assert config.max_line_length == 88

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_config_manager_load_config_with_user_config(
        self, mock_file: Any, mock_exists: Any
    ) -> None:
        """Test loading config with user configuration override."""
        manager = ConfigManager()

        # Mock file existence
        mock_exists.return_value = False

        user_config = {
            "default_style": "numpy",
            "max_line_length": 100,
        }

        config = manager.load_config(user_config=user_config)

        assert config.default_style == "numpy"
        assert config.max_line_length == 100

    def test_config_manager_merge_configs(self) -> None:
        """Test configuration merging."""
        manager = ConfigManager()

        base_config = SuggestionConfig(
            default_style="google",
            max_line_length=88,
            confidence_threshold=0.7,
        )

        override_config = SuggestionConfig(
            default_style="numpy",  # Should override
            confidence_threshold=0.9,  # Should override
            # max_line_length not specified, should keep base value
        )

        merged = manager._merge_configs(base_config, override_config)

        assert merged.default_style == "numpy"  # Overridden
        assert merged.confidence_threshold == 0.9  # Overridden
        assert merged.max_line_length == 88  # From base

    def test_config_manager_merge_configs_dictionaries(self) -> None:
        """Test merging configurations with dictionary fields."""
        manager = ConfigManager()

        base_config = SuggestionConfig()
        base_config.custom_templates = {"template1": "value1"}
        base_config.abbreviation_map = {"old": "old_value", "shared": "base_value"}

        override_config = SuggestionConfig()
        override_config.custom_templates = {"template2": "value2"}
        override_config.abbreviation_map = {
            "new": "new_value",
            "shared": "override_value",
        }

        merged = manager._merge_configs(base_config, override_config)

        # Custom templates should be merged
        assert "template1" in merged.custom_templates
        assert "template2" in merged.custom_templates

        # Abbreviation map should be merged with override taking precedence
        assert merged.abbreviation_map["old"] == "old_value"
        assert merged.abbreviation_map["new"] == "new_value"
        assert merged.abbreviation_map["shared"] == "override_value"

    def test_config_manager_caching(self) -> None:
        """Test configuration caching."""
        manager = ConfigManager()

        # First call
        config1 = manager.load_config()

        # Second call with same parameters should return cached result
        config2 = manager.load_config()

        assert config1 is config2  # Should be the same object

    def test_config_manager_clear_cache(self) -> None:
        """Test clearing configuration cache."""
        manager = ConfigManager()

        # Load config to populate cache
        manager.load_config()
        assert len(manager._config_cache) > 0

        # Clear cache
        manager.clear_cache()
        assert len(manager._config_cache) == 0


class TestGlobalConfigManager:
    """Test global config manager instance."""

    def test_global_config_manager_exists(self) -> None:
        """Test that global config manager exists."""
        assert config_manager is not None
        assert isinstance(config_manager, ConfigManager)

    def test_global_config_manager_usage(self) -> None:
        """Test using global config manager."""
        config = config_manager.load_config()
        assert isinstance(config, SuggestionConfig)


class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_config_with_yaml_file_and_overrides(self) -> None:
        """Test loading config from YAML with user overrides."""
        yaml_content = """
        suggestions:
          default_style: numpy
          max_line_length: 100
          confidence_threshold: 0.8
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            manager = ConfigManager()

            user_overrides = {
                "confidence_threshold": 0.9,  # Override YAML value
                "include_examples": True,  # New value
            }

            config = manager.load_config(
                config_path=temp_path,
                user_config=user_overrides,
            )

            # Should have YAML values
            assert config.default_style == "numpy"
            assert config.max_line_length == 100

            # Should have user override values
            assert config.confidence_threshold == 0.9
            assert config.include_examples is True

        finally:
            temp_path.unlink()

    def test_config_error_handling(self) -> None:
        """Test configuration error handling."""
        # Test invalid config values still work with fallbacks
        manager = ConfigManager()

        invalid_user_config = {
            "default_style": "invalid_style",  # Invalid
            "max_line_length": 100,  # Valid
        }

        # Should not raise exception, should use valid parts
        config = manager.load_config(user_config=invalid_user_config)

        # Should have default style (invalid was ignored)
        assert config.default_style == "google"
