import os
from unittest.mock import patch

from codedocsync.matcher.semantic_models import EmbeddingModel
from codedocsync.storage.embedding_config import EmbeddingConfigManager


class TestEmbeddingConfigManager:
    """Test the EmbeddingConfigManager class."""

    def test_init_without_api_keys(self):
        """Test initialization without API keys."""
        with patch.dict(os.environ, {}, clear=True):
            manager = EmbeddingConfigManager()

            assert manager.api_keys == {}
            assert manager.config.primary_model == EmbeddingModel.OPENAI_SMALL

    def test_init_with_openai_key(self):
        """Test initialization with OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            manager = EmbeddingConfigManager()

            assert manager.api_keys["openai"] == "test-key"

    def test_get_api_key(self):
        """Test getting API keys."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            manager = EmbeddingConfigManager()

            assert manager.get_api_key("openai") == "test-key"
            assert manager.get_api_key("nonexistent") is None

    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "EMBEDDING_MODEL": "text-embedding-ada-002",
            "EMBEDDING_BATCH_SIZE": "50",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            manager = EmbeddingConfigManager()

            assert manager.config.primary_model == EmbeddingModel.OPENAI_ADA
            assert manager.config.batch_size == 50

    def test_invalid_environment_config(self):
        """Test handling invalid environment configuration."""
        env_vars = {
            "EMBEDDING_MODEL": "invalid-model",
            "EMBEDDING_BATCH_SIZE": "invalid-number",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Should not raise exception, just use defaults
            manager = EmbeddingConfigManager()

            # Should fall back to defaults
            assert manager.config.primary_model == EmbeddingModel.OPENAI_SMALL
            assert manager.config.batch_size == 100

    def test_validate_config_with_openai_key(self):
        """Test config validation with OpenAI key available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            manager = EmbeddingConfigManager()

            assert manager.validate_config() is True

    def test_validate_config_without_required_key(self):
        """Test config validation without required API key."""
        with patch.dict(os.environ, {}, clear=True):
            manager = EmbeddingConfigManager()

            # Should fail validation because OpenAI model selected but no key
            assert manager.validate_config() is False

    def test_validate_config_with_local_model(self):
        """Test config validation with local model."""
        env_vars = {"EMBEDDING_MODEL": "all-MiniLM-L6-v2"}

        with patch.dict(os.environ, env_vars, clear=True):
            manager = EmbeddingConfigManager()
            manager.config.primary_model = EmbeddingModel.LOCAL_MINILM

            # Should pass validation even without API keys
            assert manager.validate_config() is True

    def test_get_available_models_with_openai(self):
        """Test getting available models with OpenAI key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            manager = EmbeddingConfigManager()

            available = manager.get_available_models()

            assert EmbeddingModel.OPENAI_SMALL in available
            assert EmbeddingModel.OPENAI_ADA in available
            assert EmbeddingModel.OPENAI_LARGE in available
            assert EmbeddingModel.LOCAL_MINILM in available

    def test_get_available_models_without_openai(self):
        """Test getting available models without OpenAI key."""
        with patch.dict(os.environ, {}, clear=True):
            manager = EmbeddingConfigManager()

            available = manager.get_available_models()

            # Should only have local model
            assert EmbeddingModel.LOCAL_MINILM in available
            assert EmbeddingModel.OPENAI_SMALL not in available
            assert EmbeddingModel.OPENAI_ADA not in available
            assert EmbeddingModel.OPENAI_LARGE not in available
