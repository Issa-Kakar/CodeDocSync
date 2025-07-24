"""Tests for embedding configuration management."""

import logging

from codedocsync.matcher.semantic_models import EmbeddingModel
from codedocsync.storage.embedding_config import EmbeddingConfigManager


class TestEmbeddingConfigManager:
    """Test cases for EmbeddingConfigManager."""

    def test_load_api_keys_with_openai_key(self, monkeypatch):
        """Test loading API keys when OPENAI_API_KEY is set."""
        test_key = "test-openai-key-123"
        monkeypatch.setenv("OPENAI_API_KEY", test_key)

        manager = EmbeddingConfigManager()
        assert manager.api_keys["openai"] == test_key

    def test_load_api_keys_without_keys(self, monkeypatch, caplog):
        """Test loading API keys when no keys are set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING):
            manager = EmbeddingConfigManager()

        assert "openai" not in manager.api_keys
        assert "No OPENAI_API_KEY found in environment" in caplog.text

    def test_load_api_keys_warning_logged(self, monkeypatch, caplog):
        """Test that warning is logged when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING):
            EmbeddingConfigManager()

        assert "No OPENAI_API_KEY found in environment" in caplog.text

    def test_load_config_defaults(self, monkeypatch):
        """Test loading default configuration."""
        # Clean environment
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)

        manager = EmbeddingConfigManager()
        assert manager.config.primary_model == EmbeddingModel.OPENAI_SMALL
        assert manager.config.batch_size == 100

    def test_load_config_from_env_valid_model(self, monkeypatch):
        """Test loading configuration with valid model from environment."""
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-ada-002")

        manager = EmbeddingConfigManager()
        assert manager.config.primary_model == EmbeddingModel.OPENAI_ADA

    def test_load_config_from_env_invalid_model(self, monkeypatch, caplog):
        """Test loading configuration with invalid model from environment."""
        monkeypatch.setenv("EMBEDDING_MODEL", "invalid-model-name")

        with caplog.at_level(logging.WARNING):
            manager = EmbeddingConfigManager()

        # Should keep default
        assert manager.config.primary_model == EmbeddingModel.OPENAI_SMALL
        assert "Invalid EMBEDDING_MODEL: invalid-model-name" in caplog.text

    def test_load_config_invalid_batch_size(self, monkeypatch, caplog):
        """Test loading configuration with invalid batch size."""
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "not-a-number")

        with caplog.at_level(logging.WARNING):
            manager = EmbeddingConfigManager()

        # Should keep default
        assert manager.config.batch_size == 100
        assert "Invalid EMBEDDING_BATCH_SIZE" in caplog.text

    def test_load_config_valid_batch_size(self, monkeypatch):
        """Test loading configuration with valid batch size."""
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "50")

        manager = EmbeddingConfigManager()
        assert manager.config.batch_size == 50

    def test_get_api_key_exists(self, monkeypatch):
        """Test getting an API key that exists."""
        test_key = "test-key-123"
        monkeypatch.setenv("OPENAI_API_KEY", test_key)

        manager = EmbeddingConfigManager()
        assert manager.get_api_key("openai") == test_key

    def test_get_api_key_not_exists(self, monkeypatch):
        """Test getting an API key that doesn't exist."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        manager = EmbeddingConfigManager()
        assert manager.get_api_key("openai") is None
        assert manager.get_api_key("anthropic") is None

    def test_validate_config_with_openai_key(self, monkeypatch):
        """Test config validation when OpenAI key is present."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

        manager = EmbeddingConfigManager()
        assert manager.validate_config() is True

    def test_validate_config_without_required_key(self, monkeypatch, caplog):
        """Test config validation when required API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")

        with caplog.at_level(logging.ERROR):
            manager = EmbeddingConfigManager()
            result = manager.validate_config()

        assert result is False
        assert "OpenAI model selected but no API key found" in caplog.text

    def test_get_available_models_with_openai(self, monkeypatch):
        """Test getting available models when OpenAI key is present."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        manager = EmbeddingConfigManager()
        models = manager.get_available_models()

        assert EmbeddingModel.OPENAI_SMALL in models
        assert EmbeddingModel.OPENAI_ADA in models
        assert EmbeddingModel.OPENAI_LARGE in models
        assert EmbeddingModel.LOCAL_MINILM in models
        assert len(models) == 4

    def test_get_available_models_without_keys(self, monkeypatch):
        """Test getting available models when no API keys are present."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        manager = EmbeddingConfigManager()
        models = manager.get_available_models()

        # Only local model should be available
        assert models == [EmbeddingModel.LOCAL_MINILM]

    def test_local_model_always_available(self, monkeypatch):
        """Test that local model is always available regardless of API keys."""
        # Test with key
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        manager = EmbeddingConfigManager()
        assert EmbeddingModel.LOCAL_MINILM in manager.get_available_models()

        # Test without key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        manager = EmbeddingConfigManager()
        assert EmbeddingModel.LOCAL_MINILM in manager.get_available_models()
