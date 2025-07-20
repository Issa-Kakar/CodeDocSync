"""
Configuration system for suggestion generation.

This module provides comprehensive configuration options for controlling
how suggestions are generated, formatted, and validated.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SuggestionConfig:
    """Configuration for suggestion generation."""

    # Style preferences
    default_style: str = "google"  # Default if auto-detection fails
    preserve_descriptions: bool = True  # Keep existing descriptions
    preserve_examples: bool = True  # Keep existing examples
    style_precedence: list[str] = field(
        default_factory=lambda: ["google", "numpy", "sphinx", "rest"]
    )

    # Content generation
    include_types: bool = True  # Include type annotations in suggestions
    include_examples: bool = False  # Auto-generate examples
    include_default_values: bool = True  # Show parameter defaults
    expand_abbreviations: bool = True  # Expand common abbreviations

    # Formatting preferences
    max_line_length: int = 88  # For line wrapping
    indent_size: int = 4  # Spaces for indentation
    prefer_multiline: bool = True  # Use multiline format when beneficial
    sort_parameters: bool = False  # Sort parameters alphabetically

    # Quality control
    confidence_threshold: float = 0.7  # Minimum confidence to show
    require_actionable: bool = True  # Only show actionable suggestions
    validate_syntax: bool = True  # Validate generated Python syntax
    check_grammar: bool = False  # Basic grammar checking (future)

    # Generation behavior
    max_suggestions_per_issue: int = 3  # Limit suggestions per issue
    prefer_minimal_changes: bool = True  # Minimize diff size
    merge_related_issues: bool = True  # Combine related suggestions
    cache_suggestions: bool = True  # Cache generated suggestions

    # Advanced options
    custom_templates: dict[str, str] = field(default_factory=dict)
    abbreviation_map: dict[str, str] = field(
        default_factory=lambda: {
            "param": "parameter",
            "arg": "argument",
            "ret": "return",
            "val": "value",
            "obj": "object",
            "str": "string",
            "int": "integer",
            "bool": "boolean",
            "dict": "dictionary",
            "list": "list",
            "func": "function",
        }
    )

    # Section ordering for different styles
    section_order: dict[str, list[str]] = field(
        default_factory=lambda: {
            "google": [
                "summary",
                "args",
                "returns",
                "yields",
                "raises",
                "note",
                "example",
            ],
            "numpy": [
                "summary",
                "parameters",
                "returns",
                "yields",
                "raises",
                "see_also",
                "notes",
                "examples",
            ],
            "sphinx": ["summary", "param", "type", "returns", "rtype", "raises"],
            "rest": ["summary", "arguments", "returns", "raises", "examples"],
        }
    )

    def __post_init__(self):
        """Validate configuration values."""
        # Validate style
        valid_styles = ["google", "numpy", "sphinx", "rest"]
        if self.default_style not in valid_styles:
            raise ValueError(
                f"default_style must be one of {valid_styles}, got '{self.default_style}'"
            )

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # Validate line length
        if self.max_line_length < 40:
            raise ValueError(
                f"max_line_length too small: {self.max_line_length} (min 40)"
            )

        # Validate indent size
        if self.indent_size < 1 or self.indent_size > 8:
            raise ValueError(
                f"indent_size must be between 1 and 8, got {self.indent_size}"
            )

        # Validate max suggestions
        if self.max_suggestions_per_issue < 1:
            raise ValueError(
                f"max_suggestions_per_issue must be positive, "
                f"got {self.max_suggestions_per_issue}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SuggestionConfig":
        """Create configuration from dictionary."""
        # Filter out unknown keys to avoid TypeError
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "SuggestionConfig":
        """Load configuration from YAML file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

            # Extract suggestions section if it exists
            suggestions_config = config_data.get("suggestions", config_data)
            return cls.from_dict(suggestions_config)

        except FileNotFoundError:
            # Return default config if file doesn't exist
            return cls()
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {file_path}: {e}") from e

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def save_to_yaml(self, file_path: Path) -> None:
        """Save configuration to YAML file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"suggestions": self.to_dict()},
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
            )

    def get_style_config(self, style: str) -> dict[str, Any]:
        """Get style-specific configuration."""
        if style == "google":
            return {
                "section_marker": ":",
                "parameter_format": "{name} ({type}): {description}",
                "return_format": "{type}: {description}",
                "raises_format": "{exception}: {description}",
                "indent_continuation": 4,
            }
        elif style == "numpy":
            return {
                "section_marker": "\n" + "-" * 10,
                "parameter_format": "{name} : {type}\n    {description}",
                "return_format": "{type}\n    {description}",
                "raises_format": "{exception}\n    {description}",
                "indent_continuation": 4,
            }
        elif style == "sphinx":
            return {
                "section_marker": "",
                "parameter_format": ":param {name}: {description}\n:type {name}: {type}",
                "return_format": ":returns: {description}\n:rtype: {type}",
                "raises_format": ":raises {exception}: {description}",
                "indent_continuation": 0,
            }
        elif style == "rest":
            return {
                "section_marker": "",
                "parameter_format": ":param {name}: {description}",
                "return_format": ":return: {description}",
                "raises_format": ":raises {exception}: {description}",
                "indent_continuation": 0,
            }
        else:
            # Default to Google style
            return self.get_style_config("google")

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        feature_flags = {
            "types": self.include_types,
            "examples": self.include_examples,
            "defaults": self.include_default_values,
            "abbreviations": self.expand_abbreviations,
            "multiline": self.prefer_multiline,
            "sorting": self.sort_parameters,
            "grammar": self.check_grammar,
            "caching": self.cache_suggestions,
            "merging": self.merge_related_issues,
            "validation": self.validate_syntax,
        }

        return feature_flags.get(feature, False)

    def get_quality_thresholds(self) -> dict[str, float]:
        """Get quality control thresholds."""
        return {
            "confidence": self.confidence_threshold,
            "actionable": 1.0 if self.require_actionable else 0.0,
            "syntax_valid": 1.0 if self.validate_syntax else 0.0,
        }


@dataclass
class RankingConfig:
    """Configuration for suggestion ranking and filtering."""

    # Ranking weights (0.0 to 1.0)
    severity_weight: float = 0.4  # Issue severity importance
    confidence_weight: float = 0.3  # Suggestion confidence importance
    actionability_weight: float = 0.2  # How actionable the suggestion is
    impact_weight: float = 0.1  # Estimated impact/value

    # Filtering options
    min_confidence: float = 0.5  # Minimum confidence to include
    max_suggestions: int = 10  # Maximum suggestions to return
    group_by_severity: bool = True  # Group suggestions by severity
    prefer_ready_to_apply: bool = True  # Prioritize ready-to-apply suggestions

    # Severity priority order (higher index = higher priority)
    severity_order: list[str] = field(
        default_factory=lambda: ["low", "medium", "high", "critical"]
    )

    def __post_init__(self):
        """Validate ranking configuration."""
        weights = [
            self.severity_weight,
            self.confidence_weight,
            self.actionability_weight,
            self.impact_weight,
        ]

        # Check all weights are valid
        for weight in weights:
            if not 0.0 <= weight <= 1.0:
                raise ValueError("All weights must be between 0.0 and 1.0")

        # Check weights sum to approximately 1.0
        total_weight = sum(weights)
        if not 0.8 <= total_weight <= 1.2:  # Allow some tolerance
            raise ValueError(
                f"Weights should sum to approximately 1.0, got {total_weight:.2f}"
            )

        # Validate confidence threshold
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be between 0.0 and 1.0, got {self.min_confidence}"
            )

        # Validate max suggestions
        if self.max_suggestions < 1:
            raise ValueError(
                f"max_suggestions must be positive, got {self.max_suggestions}"
            )

    def calculate_score(self, suggestion: Any) -> float:
        """Calculate ranking score for a suggestion."""
        # Import here to avoid circular dependency
        from .models import Suggestion

        if not isinstance(suggestion, Suggestion):
            return 0.0

        # Get severity score (normalized to 0-1)
        severity_score = self._get_severity_score(
            getattr(suggestion, "severity", "medium")
        )

        # Get confidence score
        confidence_score = suggestion.confidence

        # Get actionability score
        actionability_score = 1.0 if suggestion.is_actionable else 0.5

        # Get impact score (based on quality and readiness)
        impact_score = suggestion.get_quality_score()

        # Calculate weighted score
        final_score = (
            self.severity_weight * severity_score
            + self.confidence_weight * confidence_score
            + self.actionability_weight * actionability_score
            + self.impact_weight * impact_score
        )

        return min(1.0, max(0.0, final_score))

    def _get_severity_score(self, severity: str) -> float:
        """Get normalized severity score."""
        try:
            index = self.severity_order.index(severity.lower())
            return index / (len(self.severity_order) - 1)
        except (ValueError, ZeroDivisionError):
            return 0.5  # Default for unknown severity


# Predefined configurations for common use cases
def get_minimal_config() -> SuggestionConfig:
    """Get configuration for minimal, conservative suggestions."""
    return SuggestionConfig(
        confidence_threshold=0.9,
        prefer_minimal_changes=True,
        include_examples=False,
        max_suggestions_per_issue=1,
        require_actionable=True,
    )


def get_comprehensive_config() -> SuggestionConfig:
    """Get configuration for comprehensive, detailed suggestions."""
    return SuggestionConfig(
        confidence_threshold=0.5,
        include_examples=True,
        include_types=True,
        include_default_values=True,
        expand_abbreviations=True,
        max_suggestions_per_issue=3,
        prefer_multiline=True,
    )


def get_development_config() -> SuggestionConfig:
    """Get configuration optimized for development workflow."""
    return SuggestionConfig(
        confidence_threshold=0.6,
        cache_suggestions=True,
        merge_related_issues=True,
        prefer_minimal_changes=True,
        validate_syntax=True,
        max_suggestions_per_issue=2,
    )


def get_documentation_config() -> SuggestionConfig:
    """Get configuration optimized for documentation quality."""
    return SuggestionConfig(
        default_style="google",
        preserve_descriptions=True,
        preserve_examples=True,
        include_types=True,
        include_examples=True,
        expand_abbreviations=True,
        prefer_multiline=True,
        confidence_threshold=0.7,
    )


class ConfigManager:
    """Manages configuration loading and precedence."""

    def __init__(self):
        self._config_cache: dict[str, SuggestionConfig] = {}

    def load_config(
        self,
        config_path: Path | None = None,
        project_path: Path | None = None,
        user_config: dict[str, Any] | None = None,
    ) -> SuggestionConfig:
        """
        Load configuration with proper precedence.

        Priority order:
        1. Explicit user_config parameter
        2. Explicit config_path file
        3. Project .codedocsync.yml
        4. User home ~/.codedocsync/config.yml
        5. Default configuration
        """
        cache_key = f"{config_path}:{project_path}:{hash(str(user_config))}"

        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Start with default config
        config = SuggestionConfig()

        # 4. Try user home config
        user_home = Path.home() / ".codedocsync" / "config.yml"
        if user_home.exists():
            try:
                config = SuggestionConfig.from_yaml_file(user_home)
            except Exception:
                pass  # Ignore errors, use default

        # 3. Try project config
        if project_path:
            project_config = project_path / ".codedocsync.yml"
            if project_config.exists():
                try:
                    project_settings = SuggestionConfig.from_yaml_file(project_config)
                    config = self._merge_configs(config, project_settings)
                except Exception:
                    pass  # Ignore errors, use previous config

        # 2. Try explicit config file
        if config_path and config_path.exists():
            try:
                file_config = SuggestionConfig.from_yaml_file(config_path)
                config = self._merge_configs(config, file_config)
            except Exception:
                pass  # Ignore errors, use previous config

        # 1. Apply user config (highest priority)
        if user_config:
            try:
                user_settings = SuggestionConfig.from_dict(user_config)
                config = self._merge_configs(config, user_settings)
            except Exception:
                pass  # Ignore errors, use previous config

        self._config_cache[cache_key] = config
        return config

    def _merge_configs(
        self, base: SuggestionConfig, override: SuggestionConfig
    ) -> SuggestionConfig:
        """Merge two configurations, with override taking precedence."""
        # Convert to dicts
        base_dict = base.to_dict()
        override_dict = override.to_dict()

        # Merge with override taking precedence
        merged = {**base_dict, **override_dict}

        # Handle special merging for dictionaries
        if "custom_templates" in base_dict and "custom_templates" in override_dict:
            merged["custom_templates"] = {
                **base_dict["custom_templates"],
                **override_dict["custom_templates"],
            }

        if "abbreviation_map" in base_dict and "abbreviation_map" in override_dict:
            merged["abbreviation_map"] = {
                **base_dict["abbreviation_map"],
                **override_dict["abbreviation_map"],
            }

        return SuggestionConfig.from_dict(merged)

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()


# Global config manager instance
config_manager = ConfigManager()
