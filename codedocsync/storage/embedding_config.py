import logging
import os

from ..matcher.semantic_models import EmbeddingConfig, EmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingConfigManager:
    """Manages embedding configuration and API keys."""

    def __init__(self) -> None:
        self.api_keys = self._load_api_keys()
        self.config = self._load_config()

    def _load_api_keys(self) -> dict[str, str]:
        """Load API keys from environment variables."""
        keys = {}

        # OpenAI key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            keys["openai"] = openai_key
        else:
            logger.warning("No OPENAI_API_KEY found in environment")

        # Add other providers as needed
        return keys

    def _load_config(self) -> EmbeddingConfig:
        """Load embedding configuration from environment or defaults."""
        config = EmbeddingConfig()

        # Override from environment if set
        if os.getenv("EMBEDDING_MODEL"):
            try:
                config.primary_model = EmbeddingModel(os.getenv("EMBEDDING_MODEL"))
            except ValueError:
                logger.warning(
                    f"Invalid EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}"
                )

        if os.getenv("EMBEDDING_BATCH_SIZE"):
            try:
                batch_size_str = os.getenv("EMBEDDING_BATCH_SIZE")
                if batch_size_str is not None:
                    config.batch_size = int(batch_size_str)
            except ValueError:
                logger.warning("Invalid EMBEDDING_BATCH_SIZE")

        return config

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for provider."""
        return self.api_keys.get(provider)

    def validate_config(self) -> bool:
        """Validate that configuration is usable."""
        # Check if we have keys for the primary model
        if self.config.primary_model in [
            EmbeddingModel.OPENAI_SMALL,
            EmbeddingModel.OPENAI_LARGE,
            EmbeddingModel.OPENAI_ADA,
        ]:
            if not self.get_api_key("openai"):
                logger.error("OpenAI model selected but no API key found")
                return False

        return True

    def get_available_models(self) -> list[EmbeddingModel]:
        """Get list of models that are actually available."""
        available = []

        # Check OpenAI models
        if self.get_api_key("openai"):
            available.extend(
                [
                    EmbeddingModel.OPENAI_SMALL,
                    EmbeddingModel.OPENAI_ADA,
                    EmbeddingModel.OPENAI_LARGE,
                ]
            )

        # Local models are always available
        available.append(EmbeddingModel.LOCAL_MINILM)

        return available
