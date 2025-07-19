"""
Tests for LLM configuration management.

This test suite covers the LLMConfig dataclass validation, factory methods,
and configuration profiles as required by Chunk 1.

Test Categories:
- Configuration validation
- Invalid model names rejected
- Token estimation accuracy (via model dependencies)
- Factory method behavior
- API key validation
"""

import os
import pytest
from unittest.mock import patch

from codedocsync.analyzer.llm_config import LLMConfig


class TestLLMConfigValidation:
    """Test configuration validation logic."""

    def test_default_configuration_valid(self):
        """Test that default configuration is valid."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = LLMConfig()
            assert config.provider == "openai"
            assert config.model == "gpt-4o-mini"
            assert config.temperature == 0.0
            assert config.max_tokens == 1000
            assert config.timeout_seconds == 30
            assert config.max_retries == 3
            assert config.cache_ttl_days == 7
            assert config.requests_per_second == 10.0
            assert config.burst_size == 20
            assert config.max_context_tokens == 2000
            assert config.max_functions_in_context == 3

    def test_invalid_provider_rejected(self):
        """Test that invalid providers are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with pytest.raises(ValueError, match="provider must be one of"):
                LLMConfig(provider="invalid_provider")

    def test_invalid_openai_model_rejected(self):
        """Test that invalid OpenAI models are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with pytest.raises(ValueError, match="For OpenAI, model must be one of"):
                LLMConfig(model="invalid-model-name")

    def test_valid_openai_models_accepted(self):
        """Test that all valid OpenAI models are accepted."""
        valid_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            for model in valid_models:
                config = LLMConfig(model=model)
                assert config.model == model

    def test_non_zero_temperature_rejected(self):
        """Test that non-zero temperature is rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with pytest.raises(
                ValueError, match="temperature must be 0.0 for deterministic outputs"
            ):
                LLMConfig(temperature=0.5)

            with pytest.raises(
                ValueError, match="temperature must be 0.0 for deterministic outputs"
            ):
                LLMConfig(temperature=1.0)

    def test_invalid_max_tokens_rejected(self):
        """Test that invalid max_tokens values are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # Too small
            with pytest.raises(
                ValueError, match="max_tokens must be between 100 and 4000"
            ):
                LLMConfig(max_tokens=50)

            # Too large
            with pytest.raises(
                ValueError, match="max_tokens must be between 100 and 4000"
            ):
                LLMConfig(max_tokens=5000)

    def test_invalid_timeout_rejected(self):
        """Test that invalid timeout values are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # Too small
            with pytest.raises(
                ValueError, match="timeout_seconds must be between 5 and 120"
            ):
                LLMConfig(timeout_seconds=2)

            # Too large
            with pytest.raises(
                ValueError, match="timeout_seconds must be between 5 and 120"
            ):
                LLMConfig(timeout_seconds=150)

    def test_invalid_max_retries_rejected(self):
        """Test that invalid max_retries values are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # Negative
            with pytest.raises(
                ValueError, match="max_retries must be between 0 and 10"
            ):
                LLMConfig(max_retries=-1)

            # Too large
            with pytest.raises(
                ValueError, match="max_retries must be between 0 and 10"
            ):
                LLMConfig(max_retries=15)

    def test_invalid_cache_ttl_rejected(self):
        """Test that invalid cache TTL values are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with pytest.raises(ValueError, match="cache_ttl_days must be positive"):
                LLMConfig(cache_ttl_days=0)

            with pytest.raises(ValueError, match="cache_ttl_days must be positive"):
                LLMConfig(cache_ttl_days=-1)

    def test_invalid_rate_limits_rejected(self):
        """Test that invalid rate limiting values are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # requests_per_second too small
            with pytest.raises(
                ValueError, match="requests_per_second must be between 0.1 and 100"
            ):
                LLMConfig(requests_per_second=0.05)

            # requests_per_second too large
            with pytest.raises(
                ValueError, match="requests_per_second must be between 0.1 and 100"
            ):
                LLMConfig(requests_per_second=150)

            # burst_size too small
            with pytest.raises(
                ValueError, match="burst_size must be between 1 and 100"
            ):
                LLMConfig(burst_size=0)

            # burst_size too large
            with pytest.raises(
                ValueError, match="burst_size must be between 1 and 100"
            ):
                LLMConfig(burst_size=150)

    def test_invalid_context_limits_rejected(self):
        """Test that invalid context limits are rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # max_context_tokens too small
            with pytest.raises(
                ValueError, match="max_context_tokens must be between 500 and 8000"
            ):
                LLMConfig(max_context_tokens=400)

            # max_context_tokens too large
            with pytest.raises(
                ValueError, match="max_context_tokens must be between 500 and 8000"
            ):
                LLMConfig(max_context_tokens=10000)

            # max_functions_in_context too small
            with pytest.raises(
                ValueError, match="max_functions_in_context must be between 1 and 10"
            ):
                LLMConfig(max_functions_in_context=0)

            # max_functions_in_context too large
            with pytest.raises(
                ValueError, match="max_functions_in_context must be between 1 and 10"
            ):
                LLMConfig(max_functions_in_context=15)


class TestLLMConfigAPIKeyValidation:
    """Test API key validation logic."""

    def test_missing_api_key_rejected(self):
        """Test that missing API key is rejected."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="OPENAI_API_KEY environment variable is required"
            ):
                LLMConfig()

    def test_invalid_api_key_format_rejected(self):
        """Test that invalid API key format is rejected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid-key-format"}):
            with pytest.raises(
                ValueError, match="OPENAI_API_KEY must start with 'sk-'"
            ):
                LLMConfig()

    def test_valid_api_key_accepted(self):
        """Test that valid API key format is accepted."""
        valid_keys = ["sk-test123", "sk-1234567890abcdef", "sk-proj-1234567890abcdef"]

        for key in valid_keys:
            with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
                config = LLMConfig()
                assert config.provider == "openai"


class TestLLMConfigFactoryMethods:
    """Test factory methods for different configuration profiles."""

    def test_create_fast_config(self):
        """Test fast configuration factory method."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = LLMConfig.create_fast_config()

            # Fast config should optimize for speed
            assert config.model == "gpt-4o-mini"
            assert config.max_tokens == 500
            assert config.timeout_seconds == 15
            assert config.max_retries == 1
            assert config.requests_per_second == 15.0
            assert config.max_context_tokens == 1000
            assert config.max_functions_in_context == 1

    def test_create_balanced_config(self):
        """Test balanced configuration factory method."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = LLMConfig.create_balanced_config()

            # Balanced config should use defaults
            assert config.model == "gpt-4o-mini"
            assert config.max_tokens == 1000
            assert config.timeout_seconds == 30
            assert config.max_retries == 3
            assert config.requests_per_second == 10.0

    def test_create_thorough_config(self):
        """Test thorough configuration factory method."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = LLMConfig.create_thorough_config()

            # Thorough config should optimize for completeness
            assert config.model == "gpt-4o"
            assert config.max_tokens == 1500
            assert config.timeout_seconds == 45
            assert config.max_retries == 5
            assert config.requests_per_second == 5.0
            assert config.max_context_tokens == 3000
            assert config.max_functions_in_context == 5
            assert config.cache_ttl_days == 14


class TestLLMConfigUtilityMethods:
    """Test utility methods on LLMConfig."""

    def test_get_model_identifier(self):
        """Test model identifier generation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = LLMConfig(model="gpt-4o-mini")
            assert config.get_model_identifier() == "openai/gpt-4o-mini"

            config = LLMConfig(model="gpt-4o")
            assert config.get_model_identifier() == "openai/gpt-4o"

    def test_estimate_cost_per_request(self):
        """Test cost estimation functionality."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # Test different models have different costs
            config_mini = LLMConfig(model="gpt-4o-mini")
            config_4o = LLMConfig(model="gpt-4o")

            cost_mini = config_mini.estimate_cost_per_request()
            cost_4o = config_4o.estimate_cost_per_request()

            # gpt-4o should be more expensive than gpt-4o-mini
            assert cost_4o > cost_mini
            assert cost_mini > 0
            assert cost_4o > 0

    def test_get_summary(self):
        """Test configuration summary generation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = LLMConfig()
            summary = config.get_summary()

            # Check all expected fields are present
            expected_fields = [
                "provider",
                "model",
                "temperature",
                "max_tokens",
                "timeout_seconds",
                "max_retries",
                "cache_ttl_days",
                "requests_per_second",
                "estimated_cost_per_request",
                "api_key_configured",
            ]

            for field in expected_fields:
                assert field in summary

            assert summary["api_key_configured"] is True
            assert summary["provider"] == "openai"
            assert summary["model"] == "gpt-4o-mini"
            assert summary["temperature"] == 0.0


class TestLLMConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_boundary_values_accepted(self):
        """Test that boundary values are accepted."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # Test minimum valid values
            config = LLMConfig(
                max_tokens=100,
                timeout_seconds=5,
                max_retries=0,
                requests_per_second=0.1,
                burst_size=1,
                max_context_tokens=500,
                max_functions_in_context=1,
                cache_ttl_days=1,
            )
            assert config.max_tokens == 100

            # Test maximum valid values
            config = LLMConfig(
                max_tokens=4000,
                timeout_seconds=120,
                max_retries=10,
                requests_per_second=100,
                burst_size=100,
                max_context_tokens=8000,
                max_functions_in_context=10,
            )
            assert config.max_tokens == 4000

    def test_configuration_immutability(self):
        """Test that configuration validation happens at creation time."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = LLMConfig()

            # Modifying the object after creation should not re-validate
            # (this is expected behavior for dataclasses)
            config.temperature = 0.5  # This would be invalid if validated
            assert config.temperature == 0.5

            # But creating a new config with invalid temperature should fail
            with pytest.raises(ValueError):
                LLMConfig(temperature=0.5)

    def test_multiple_configurations_independent(self):
        """Test that multiple configuration instances are independent."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config1 = LLMConfig.create_fast_config()
            config2 = LLMConfig.create_thorough_config()

            assert config1.model != config2.model
            assert config1.timeout_seconds != config2.timeout_seconds
            assert config1.max_retries != config2.max_retries
