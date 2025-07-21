"""
Configuration management for suggestion system.

Handles loading, merging, and managing configuration from multiple sources
with proper precedence and validation.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .config import SuggestionConfig
from .formatters.terminal_formatter import OutputStyle, TerminalFormatterConfig
from .ranking import RankingConfig, RankingStrategy

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSuggestionConfig:
    """Complete configuration for suggestion system."""

    # Core suggestion configuration
    suggestion: SuggestionConfig = field(default_factory=SuggestionConfig)

    # Ranking and filtering configuration
    ranking: RankingConfig = field(default_factory=RankingConfig)

    # Terminal formatter configuration
    terminal_formatter: TerminalFormatterConfig = field(
        default_factory=TerminalFormatterConfig
    )

    # Output configuration
    output_style: OutputStyle = OutputStyle.RICH
    json_indent: int = 2
    include_metadata: bool = True
    include_timestamps: bool = True

    # Performance configuration
    max_concurrent_generations: int = 10
    cache_suggestions: bool = True
    cache_ttl_hours: int = 24

    # Integration configuration
    auto_enhance_analysis: bool = True
    fallback_on_error: bool = True
    log_performance_metrics: bool = True


class SuggestionConfigManager:
    """Manage suggestion configuration with proper precedence."""

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._config_cache: IntegratedSuggestionConfig | None = None
        self._config_sources: list[str] = []

    def load_config(
        self,
        config_path: str | Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> IntegratedSuggestionConfig:
        """
        Load configuration with proper precedence.

        Priority order:
        1. CLI arguments/overrides
        2. Explicit config file path
        3. Project .codedocsync.yml
        4. User home ~/.codedocsync/config.yml
        5. Default configuration
        """
        # Start with default configuration
        config = IntegratedSuggestionConfig()
        config_sources = ["defaults"]

        # Load from user home directory
        user_config_path = self._get_user_config_path()
        if user_config_path.exists():
            try:
                user_config = self._load_config_file(user_config_path)
                config = self._merge_configs(config, user_config)
                config_sources.append(str(user_config_path))
                logger.debug(f"Loaded user config from {user_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")

        # Load from project directory
        project_config_path = self._find_project_config()
        if project_config_path and project_config_path.exists():
            try:
                project_config = self._load_config_file(project_config_path)
                config = self._merge_configs(config, project_config)
                config_sources.append(str(project_config_path))
                logger.debug(f"Loaded project config from {project_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load project config: {e}")

        # Load from explicit path
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    explicit_config = self._load_config_file(config_path)
                    config = self._merge_configs(config, explicit_config)
                    config_sources.append(str(config_path))
                    logger.debug(f"Loaded explicit config from {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load explicit config: {e}")
            else:
                logger.warning(f"Explicit config path does not exist: {config_path}")

        # Apply CLI overrides
        if cli_overrides:
            config = self._apply_cli_overrides(config, cli_overrides)
            config_sources.append("CLI overrides")

        # Validate final configuration
        self._validate_config(config)

        # Cache configuration and sources
        self._config_cache = config
        self._config_sources = config_sources

        logger.info(f"Loaded configuration from: {', '.join(config_sources)}")
        return config

    def get_cached_config(self) -> IntegratedSuggestionConfig | None:
        """Get cached configuration if available."""
        return self._config_cache

    def get_config_sources(self) -> list[str]:
        """Get list of configuration sources."""
        return self._config_sources.copy()

    def save_config(
        self,
        config: IntegratedSuggestionConfig,
        path: str | Path | None = None,
    ) -> Path:
        """Save configuration to file."""
        if path is None:
            path = self._get_user_config_path()
        else:
            path = Path(path)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary and save
        config_dict = self._config_to_dict(config)

        if path.suffix.lower() in [".yml", ".yaml"]:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif path.suffix.lower() == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)
        else:
            # Default to YAML
            path = path.with_suffix(".yml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {path}")
        return path

    def create_profile_config(self, profile: str) -> IntegratedSuggestionConfig:
        """Create configuration for a specific profile."""
        if profile == "strict":
            return self._create_strict_profile()
        elif profile == "permissive":
            return self._create_permissive_profile()
        elif profile == "development":
            return self._create_development_profile()
        elif profile == "production":
            return self._create_production_profile()
        elif profile == "fast":
            return self._create_fast_profile()
        else:
            logger.warning(f"Unknown profile: {profile}, using default")
            return IntegratedSuggestionConfig()

    def _load_config_file(self, path: Path) -> dict[str, Any]:
        """Load configuration from file."""
        with open(path, encoding="utf-8") as f:
            if path.suffix.lower() in [".yml", ".yaml"]:
                return yaml.safe_load(f) or {}
            elif path.suffix.lower() == ".json":
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

    def _merge_configs(
        self, base: IntegratedSuggestionConfig, override: dict[str, Any]
    ) -> IntegratedSuggestionConfig:
        """Merge configuration dictionaries."""
        # Convert base to dict for easier merging
        base_dict = self._config_to_dict(base)

        # Deep merge dictionaries
        merged_dict = self._deep_merge(base_dict, override)

        # Convert back to config object
        return self._dict_to_config(merged_dict)

    def _deep_merge(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_cli_overrides(
        self, config: IntegratedSuggestionConfig, overrides: dict[str, Any]
    ) -> IntegratedSuggestionConfig:
        """Apply CLI overrides to configuration."""
        # Convert to dict and apply overrides
        config_dict = self._config_to_dict(config)
        config_dict = self._deep_merge(config_dict, overrides)
        return self._dict_to_config(config_dict)

    def _config_to_dict(self, config: IntegratedSuggestionConfig) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result: dict[str, Any] = {}

        # Handle dataclass fields
        result["suggestion"] = asdict(config.suggestion)
        result["ranking"] = asdict(config.ranking)
        result["terminal_formatter"] = asdict(config.terminal_formatter)

        # Handle enum fields
        result["output_style"] = config.output_style.value

        # Handle other fields
        result["json_indent"] = config.json_indent
        result["include_metadata"] = config.include_metadata
        result["include_timestamps"] = config.include_timestamps
        result["max_concurrent_generations"] = config.max_concurrent_generations
        result["cache_suggestions"] = config.cache_suggestions
        result["cache_ttl_hours"] = config.cache_ttl_hours
        result["auto_enhance_analysis"] = config.auto_enhance_analysis
        result["fallback_on_error"] = config.fallback_on_error
        result["log_performance_metrics"] = config.log_performance_metrics

        # Handle enum fields in nested configs
        if "strategy" in result["ranking"]:
            result["ranking"]["strategy"] = config.ranking.strategy.value

        if "output_style" in result["terminal_formatter"]:
            result["terminal_formatter"][
                "output_style"
            ] = config.terminal_formatter.use_unicode

        return result

    def _dict_to_config(
        self, config_dict: dict[str, Any]
    ) -> IntegratedSuggestionConfig:
        """Convert dictionary to configuration object."""
        # Create base config
        config = IntegratedSuggestionConfig()

        # Update suggestion config
        if "suggestion" in config_dict:
            suggestion_dict = config_dict["suggestion"]
            for key, value in suggestion_dict.items():
                if hasattr(config.suggestion, key):
                    setattr(config.suggestion, key, value)

        # Update ranking config
        if "ranking" in config_dict:
            ranking_dict = config_dict["ranking"]
            for key, value in ranking_dict.items():
                if key == "strategy" and isinstance(value, str):
                    config.ranking.strategy = RankingStrategy(value)
                elif hasattr(config.ranking, key):
                    setattr(config.ranking, key, value)

        # Update terminal formatter config
        if "terminal_formatter" in config_dict:
            formatter_dict = config_dict["terminal_formatter"]
            for key, value in formatter_dict.items():
                if hasattr(config.terminal_formatter, key):
                    setattr(config.terminal_formatter, key, value)

        # Update other fields
        for key in [
            "json_indent",
            "include_metadata",
            "include_timestamps",
            "max_concurrent_generations",
            "cache_suggestions",
            "cache_ttl_hours",
            "auto_enhance_analysis",
            "fallback_on_error",
            "log_performance_metrics",
        ]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        # Handle output_style enum
        if "output_style" in config_dict:
            config.output_style = OutputStyle(config_dict["output_style"])

        return config

    def _validate_config(self, config: IntegratedSuggestionConfig) -> None:
        """Validate configuration values."""
        # Validate suggestion config
        if (
            config.suggestion.confidence_threshold < 0
            or config.suggestion.confidence_threshold > 1
        ):
            raise ValueError("confidence_threshold must be between 0 and 1")

        # Validate ranking config
        if config.ranking.min_confidence < 0 or config.ranking.min_confidence > 1:
            raise ValueError("min_confidence must be between 0 and 1")

        # Validate performance config
        if config.max_concurrent_generations < 1:
            raise ValueError("max_concurrent_generations must be positive")

        if config.cache_ttl_hours < 0:
            raise ValueError("cache_ttl_hours must be non-negative")

    def _get_user_config_path(self) -> Path:
        """Get user configuration file path."""
        home = Path.home()
        config_dir = home / ".codedocsync"
        return config_dir / "config.yml"

    def _find_project_config(self) -> Path | None:
        """Find project configuration file."""
        # Look for config in current directory and parent directories
        current = Path.cwd()

        config_names = [".codedocsync.yml", ".codedocsync.yaml", "codedocsync.yml"]

        for parent in [current] + list(current.parents):
            for name in config_names:
                config_path = parent / name
                if config_path.exists():
                    return config_path

        return None

    # Profile creation methods
    def _create_strict_profile(self) -> IntegratedSuggestionConfig:
        """Create strict configuration profile."""
        config = IntegratedSuggestionConfig()

        # Strict suggestion settings
        config.suggestion.confidence_threshold = 0.8
        config.suggestion.preserve_descriptions = True
        config.suggestion.include_types = True

        # Strict ranking settings
        config.ranking.min_confidence = 0.8
        config.ranking.copy_paste_ready_only = True
        config.ranking.allowed_severities = ["critical", "high"]

        return config

    def _create_permissive_profile(self) -> IntegratedSuggestionConfig:
        """Create permissive configuration profile."""
        config = IntegratedSuggestionConfig()

        # Permissive suggestion settings
        config.suggestion.confidence_threshold = 0.3
        config.suggestion.include_examples = True

        # Permissive ranking settings
        config.ranking.min_confidence = 0.3
        config.ranking.copy_paste_ready_only = False

        return config

    def _create_development_profile(self) -> IntegratedSuggestionConfig:
        """Create development configuration profile."""
        config = IntegratedSuggestionConfig()

        # Development-friendly settings
        config.suggestion.confidence_threshold = 0.5
        config.ranking.min_confidence = 0.5
        config.log_performance_metrics = True
        config.cache_suggestions = True

        # Show more information in development
        config.terminal_formatter.show_confidence = True
        config.terminal_formatter.show_diff = True
        config.include_metadata = True

        return config

    def _create_production_profile(self) -> IntegratedSuggestionConfig:
        """Create production configuration profile."""
        config = IntegratedSuggestionConfig()

        # Production-optimized settings
        config.suggestion.confidence_threshold = 0.7
        config.ranking.min_confidence = 0.7
        config.ranking.copy_paste_ready_only = True

        # Optimize for performance
        config.max_concurrent_generations = 5
        config.cache_suggestions = True
        config.cache_ttl_hours = 48

        # Minimal output for production
        config.output_style = OutputStyle.MINIMAL
        config.include_timestamps = False

        return config

    def _create_fast_profile(self) -> IntegratedSuggestionConfig:
        """Create fast configuration profile."""
        config = IntegratedSuggestionConfig()

        # Fast generation settings
        config.suggestion.confidence_threshold = 0.6
        config.ranking.min_confidence = 0.6
        config.ranking.max_total_suggestions = 10

        # Performance optimizations
        config.max_concurrent_generations = 20
        config.cache_suggestions = True
        config.terminal_formatter.compact_mode = True

        return config


# Global configuration manager instance
_config_manager = SuggestionConfigManager()


def get_config_manager() -> SuggestionConfigManager:
    """Get global configuration manager instance."""
    return _config_manager


def load_suggestion_config(**kwargs: Any) -> IntegratedSuggestionConfig:
    """Load suggestion configuration with the global manager."""
    return _config_manager.load_config(**kwargs)
