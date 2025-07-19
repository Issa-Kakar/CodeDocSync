"""
Configuration management for LLM-powered analysis.

This module provides the LLMConfig dataclass that manages all LLM-specific
settings including provider selection, model parameters, rate limiting,
and context management.

Critical Requirements:
- Temperature MUST be 0 for consistent outputs
- Model should be gpt-4o-mini for best price/performance
- Timeout is critical - never wait more than 30s
- Use @dataclass for automatic validation
"""

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM analyzer."""

    # Model configuration
    provider: str = "openai"  # Only OpenAI for now
    model: str = "gpt-4o-mini"  # Fast, cheap, good for this use case
    temperature: float = 0.0  # Deterministic - MUST be 0
    max_tokens: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3
    cache_ttl_days: int = 7

    # Rate limiting configuration
    requests_per_second: float = 10.0
    burst_size: int = 20

    # Context limits to prevent token overflow
    max_context_tokens: int = 2000
    max_functions_in_context: int = 3

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_provider()
        self._validate_model()
        self._validate_numeric_parameters()
        self._validate_rate_limits()
        self._validate_context_limits()
        self._validate_api_key()

    def _validate_provider(self) -> None:
        """Validate the LLM provider."""
        valid_providers = ["openai"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"provider must be one of {valid_providers}, got '{self.provider}'"
            )

    def _validate_model(self) -> None:
        """Validate the model selection."""
        # For OpenAI, validate against known models
        if self.provider == "openai":
            valid_models = [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ]
            if self.model not in valid_models:
                raise ValueError(
                    f"For OpenAI, model must be one of {valid_models}, got '{self.model}'"
                )

    def _validate_numeric_parameters(self) -> None:
        """Validate numeric configuration parameters."""
        # Temperature must be exactly 0 for deterministic outputs
        if self.temperature != 0.0:
            raise ValueError(
                f"temperature must be 0.0 for deterministic outputs, got {self.temperature}"
            )

        # Max tokens must be reasonable
        if not 100 <= self.max_tokens <= 4000:
            raise ValueError(
                f"max_tokens must be between 100 and 4000, got {self.max_tokens}"
            )

        # Timeout must be reasonable
        if not 5 <= self.timeout_seconds <= 120:
            raise ValueError(
                f"timeout_seconds must be between 5 and 120, got {self.timeout_seconds}"
            )

        # Retries must be reasonable
        if not 0 <= self.max_retries <= 10:
            raise ValueError(
                f"max_retries must be between 0 and 10, got {self.max_retries}"
            )

        # Cache TTL must be positive
        if self.cache_ttl_days <= 0:
            raise ValueError(
                f"cache_ttl_days must be positive, got {self.cache_ttl_days}"
            )

    def _validate_rate_limits(self) -> None:
        """Validate rate limiting configuration."""
        if not 0.1 <= self.requests_per_second <= 100:
            raise ValueError(
                f"requests_per_second must be between 0.1 and 100, got {self.requests_per_second}"
            )

        if not 1 <= self.burst_size <= 100:
            raise ValueError(
                f"burst_size must be between 1 and 100, got {self.burst_size}"
            )

    def _validate_context_limits(self) -> None:
        """Validate context management configuration."""
        if not 500 <= self.max_context_tokens <= 8000:
            raise ValueError(
                f"max_context_tokens must be between 500 and 8000, got {self.max_context_tokens}"
            )

        if not 1 <= self.max_functions_in_context <= 10:
            raise ValueError(
                f"max_functions_in_context must be between 1 and 10, got {self.max_functions_in_context}"
            )

    def _validate_api_key(self) -> None:
        """Validate that required API key is available."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OpenAI provider"
                )
            if not api_key.startswith("sk-"):
                raise ValueError(
                    "OPENAI_API_KEY must start with 'sk-', invalid format detected"
                )

    @classmethod
    def create_fast_config(cls) -> "LLMConfig":
        """Create configuration optimized for speed."""
        return cls(
            model="gpt-4o-mini",
            max_tokens=500,
            timeout_seconds=15,
            max_retries=1,
            requests_per_second=15.0,
            max_context_tokens=1000,
            max_functions_in_context=1,
        )

    @classmethod
    def create_balanced_config(cls) -> "LLMConfig":
        """Create configuration balanced between speed and thoroughness."""
        return cls()  # Uses defaults which are already balanced

    @classmethod
    def create_thorough_config(cls) -> "LLMConfig":
        """Create configuration optimized for thoroughness."""
        return cls(
            model="gpt-4o",
            max_tokens=1500,
            timeout_seconds=45,
            max_retries=5,
            requests_per_second=5.0,
            max_context_tokens=3000,
            max_functions_in_context=5,
            cache_ttl_days=14,
        )

    def get_model_identifier(self) -> str:
        """Get the full model identifier for API calls."""
        return f"{self.provider}/{self.model}"

    def estimate_cost_per_request(self) -> float:
        """Estimate cost per request in USD (rough approximation)."""
        # Rough cost estimates based on average token usage
        cost_per_1k_tokens = {
            "gpt-4o-mini": 0.0015,  # $0.15 per 1M tokens
            "gpt-4o": 0.03,  # $30 per 1M tokens
            "gpt-4-turbo": 0.01,  # $10 per 1M tokens
            "gpt-4": 0.03,  # $30 per 1M tokens
            "gpt-3.5-turbo": 0.002,  # $2 per 1M tokens
        }

        base_cost = cost_per_1k_tokens.get(self.model, 0.001)
        # Estimate ~1.5k tokens per request (prompt + completion)
        return base_cost * 1.5

    def get_summary(self) -> dict:
        """Get a summary of the current configuration."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "cache_ttl_days": self.cache_ttl_days,
            "requests_per_second": self.requests_per_second,
            "estimated_cost_per_request": self.estimate_cost_per_request(),
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        }
