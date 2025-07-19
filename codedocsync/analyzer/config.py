"""
Configuration classes for the analyzer module.

This module provides configuration options for customizing rule behavior,
severity mappings, and analysis parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RuleEngineConfig:
    """Configuration for the rule engine."""

    # Rule selection
    enabled_rules: Optional[List[str]] = None  # None = all rules
    disabled_rules: Optional[List[str]] = None  # Rules to explicitly disable

    # Performance settings
    performance_mode: bool = False  # Skip expensive rules
    max_execution_time_ms: float = 5.0  # Per-function timeout

    # Severity overrides
    severity_overrides: Dict[str, str] = field(default_factory=dict)

    # Confidence settings
    confidence_threshold: float = 0.9  # Skip LLM if rule confidence >= this
    high_confidence_threshold: float = 0.95  # Mark as critical if >= this
    low_confidence_threshold: float = 0.5  # Require LLM if < this

    # Suggestion settings
    include_code_examples: bool = True  # Include code examples in suggestions
    markdown_formatting: bool = True  # Format suggestions as markdown
    max_suggestion_length: int = 500  # Maximum suggestion length in chars

    def get_effective_rules(self, all_rules: List[str]) -> List[str]:
        """Get the effective list of rules to run."""
        if self.enabled_rules is not None:
            # Use explicitly enabled rules
            effective = list(self.enabled_rules)
        else:
            # Start with all rules
            effective = list(all_rules)

        # Remove disabled rules
        if self.disabled_rules:
            effective = [rule for rule in effective if rule not in self.disabled_rules]

        return effective

    def get_severity(self, issue_type: str, default_severity: str) -> str:
        """Get effective severity for an issue type."""
        return self.severity_overrides.get(issue_type, default_severity)

    def should_skip_llm(self, confidence: float) -> bool:
        """Check if LLM analysis should be skipped based on confidence."""
        return confidence >= self.confidence_threshold

    def is_high_confidence(self, confidence: float) -> bool:
        """Check if this is a high-confidence result."""
        return confidence >= self.high_confidence_threshold

    def requires_llm(self, confidence: float) -> bool:
        """Check if LLM analysis is required."""
        return confidence < self.low_confidence_threshold


@dataclass
class AnalysisConfig:
    """Configuration for the complete analysis process."""

    # Rule engine configuration
    rule_engine: RuleEngineConfig = field(default_factory=RuleEngineConfig)

    # LLM configuration
    use_llm: bool = True  # Whether to use LLM analysis at all
    llm_provider: str = "openai"  # LLM provider (openai, anthropic, local)
    llm_model: str = "gpt-3.5-turbo"  # LLM model to use
    llm_temperature: float = 0.1  # Temperature for LLM calls
    llm_max_tokens: int = 1000  # Max tokens for LLM responses
    llm_timeout_seconds: float = 30.0  # Timeout for LLM calls

    # Caching configuration
    enable_cache: bool = True  # Whether to use caching
    cache_ttl_hours: int = 24  # Cache time-to-live
    max_cache_size_mb: int = 100  # Maximum cache size

    # Performance configuration
    parallel_analysis: bool = True  # Analyze multiple functions in parallel
    max_parallel_workers: int = 4  # Maximum number of parallel workers
    batch_size: int = 10  # Functions per batch for parallel processing

    # Output configuration
    include_rule_results: bool = False  # Include individual rule results
    sort_by_severity: bool = True  # Sort issues by severity
    include_performance_stats: bool = False  # Include timing information

    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []

        # Validate LLM configuration
        if self.use_llm:
            if not self.llm_provider:
                errors.append("llm_provider cannot be empty when use_llm=True")
            if not self.llm_model:
                errors.append("llm_model cannot be empty when use_llm=True")
            if self.llm_temperature < 0 or self.llm_temperature > 2:
                errors.append("llm_temperature must be between 0 and 2")
            if self.llm_max_tokens < 1:
                errors.append("llm_max_tokens must be positive")

        # Validate performance configuration
        if self.max_parallel_workers < 1:
            errors.append("max_parallel_workers must be positive")
        if self.batch_size < 1:
            errors.append("batch_size must be positive")

        # Validate cache configuration
        if self.cache_ttl_hours < 0:
            errors.append("cache_ttl_hours must be non-negative")
        if self.max_cache_size_mb < 1:
            errors.append("max_cache_size_mb must be positive")

        return errors


# Predefined configurations for common use cases


def get_fast_config() -> AnalysisConfig:
    """Get configuration optimized for speed."""
    config = AnalysisConfig()
    config.rule_engine.performance_mode = True
    config.rule_engine.confidence_threshold = 0.8  # Lower threshold = more LLM usage
    config.use_llm = False  # Skip LLM entirely for speed
    config.parallel_analysis = True
    config.max_parallel_workers = 8
    return config


def get_thorough_config() -> AnalysisConfig:
    """Get configuration optimized for thoroughness."""
    config = AnalysisConfig()
    config.rule_engine.performance_mode = False
    config.rule_engine.confidence_threshold = 0.95  # Higher threshold = less LLM usage
    config.use_llm = True
    config.llm_model = "gpt-4"  # Use better model
    config.include_rule_results = True
    config.include_performance_stats = True
    return config


def get_development_config() -> AnalysisConfig:
    """Get configuration suitable for development/testing."""
    config = AnalysisConfig()
    config.rule_engine.performance_mode = False
    config.use_llm = True
    config.enable_cache = False  # Disable cache for testing
    config.parallel_analysis = False  # Easier debugging
    config.include_rule_results = True
    config.include_performance_stats = True
    return config
