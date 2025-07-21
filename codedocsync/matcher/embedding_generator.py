import hashlib
import logging
import time

from tenacity import retry, stop_after_attempt, wait_exponential

from ..parser import ParsedFunction
from .semantic_models import EmbeddingConfig, EmbeddingModel, FunctionEmbedding

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for functions with fallback support."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        from ..storage.embedding_config import EmbeddingConfigManager

        self.config = config or EmbeddingConfig()
        self.config_manager = EmbeddingConfigManager()

        # Validate configuration
        if not self.config_manager.validate_config():
            raise ValueError("Invalid embedding configuration")

        # Initialize embedding providers
        self._init_providers()

        # Performance tracking
        self.stats = {
            "embeddings_generated": 0,
            "fallbacks_used": 0,
            "total_generation_time": 0.0,
            "batch_count": 0,
        }

    def _init_providers(self) -> None:
        """Initialize embedding providers based on available models."""
        self.providers = {}

        # Initialize OpenAI if available
        if self.config_manager.get_api_key("openai"):
            try:
                import openai

                openai.api_key = self.config_manager.get_api_key("openai")
                self.providers["openai"] = openai
                logger.info("Initialized OpenAI embedding provider")
            except ImportError:
                logger.warning("OpenAI library not installed")

        # Initialize local model as fallback
        try:
            from sentence_transformers import (  # type: ignore[import-untyped]
                SentenceTransformer,
            )

            self.providers["local"] = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Initialized local embedding model")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, local fallback unavailable"
            )

    def prepare_function_text(self, function: ParsedFunction) -> str:
        """
        Prepare function text for embedding.

        Strategy:
        1. Function signature with types
        2. First line of docstring (if available)
        3. NO source code (security risk)
        """
        # Build signature string
        signature_text = function.signature.to_string()

        # Add docstring summary if available
        docstring_text = ""
        if function.docstring:
            if hasattr(function.docstring, "summary"):
                # ParsedDocstring
                docstring_text = function.docstring.summary
            elif hasattr(function.docstring, "raw_text"):
                # RawDocstring - take first line
                first_line = function.docstring.raw_text.split("\n")[0].strip()
                docstring_text = first_line

        # Combine with clear separation
        if docstring_text:
            combined_text = f"{signature_text} | {docstring_text}"
        else:
            combined_text = signature_text

        # Truncate if too long (most models have token limits)
        max_length = 512  # Conservative limit
        if len(combined_text) > max_length:
            combined_text = combined_text[: max_length - 3] + "..."

        return combined_text

    def generate_function_id(self, function: ParsedFunction) -> str:
        """Generate stable ID for function."""
        # Use module path + function name
        module_path = function.file_path.replace("/", ".").replace("\\", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        return f"{module_path}.{function.signature.name}"

    def generate_signature_hash(self, function: ParsedFunction) -> str:
        """Generate hash of function signature for change detection."""
        signature_str = function.signature.to_string()
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_embedding(
        self, text: str, model: EmbeddingModel
    ) -> list[float] | None:
        """Generate embedding with retry logic."""
        try:
            if model in [
                EmbeddingModel.OPENAI_SMALL,
                EmbeddingModel.OPENAI_LARGE,
                EmbeddingModel.OPENAI_ADA,
            ]:
                return await self._generate_openai_embedding(text, model.value)
            elif model == EmbeddingModel.LOCAL_MINILM:
                return self._generate_local_embedding(text)
            else:
                raise ValueError(f"Unsupported model: {model}")

        except Exception as e:
            logger.error(f"Failed to generate embedding with {model}: {e}")
            raise

    async def _generate_openai_embedding(self, text: str, model: str) -> list[float]:
        """Generate embedding using OpenAI."""
        if "openai" not in self.providers:
            raise ValueError("OpenAI provider not initialized")

        import openai

        try:
            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(input=text, model=model)
            return response.data[0].embedding

        except openai.RateLimitError:
            logger.warning("OpenAI rate limit hit")
            raise
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def _generate_local_embedding(self, text: str) -> list[float]:
        """Generate embedding using local model."""
        if "local" not in self.providers:
            raise ValueError("Local model not initialized")

        model = self.providers["local"]
        embedding = model.encode(text, convert_to_numpy=True)
        return list(embedding.tolist())

    async def generate_function_embeddings(
        self, functions: list[ParsedFunction], use_cache: bool = True
    ) -> list[FunctionEmbedding]:
        """
        Generate embeddings for multiple functions.

        Uses batching and fallback models as needed.
        """
        start_time = time.time()
        embeddings = []

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(functions), batch_size):
            batch = functions[i : i + batch_size]
            batch_embeddings = await self._process_batch(batch, use_cache)
            embeddings.extend(batch_embeddings)
            self.stats["batch_count"] += 1

        # Update stats
        self.stats["embeddings_generated"] += len(embeddings)
        self.stats["total_generation_time"] += time.time() - start_time

        return embeddings

    async def _process_batch(
        self, functions: list[ParsedFunction], use_cache: bool
    ) -> list[FunctionEmbedding]:
        """Process a batch of functions."""
        embeddings = []

        for function in functions:
            # Prepare text
            text = self.prepare_function_text(function)
            function_id = self.generate_function_id(function)
            signature_hash = self.generate_signature_hash(function)

            # Try to generate embedding with fallback chain
            embedding_vector = None
            model_used = None

            # Try primary model first
            try:
                embedding_vector = await self.generate_embedding(
                    text, self.config.primary_model
                )
                model_used = self.config.primary_model.value

            except Exception as e:
                logger.warning(f"Primary model failed: {e}, trying fallbacks")
                self.stats["fallbacks_used"] += 1

                # Try fallback models
                for fallback_model in self.config.fallback_models:
                    try:
                        embedding_vector = await self.generate_embedding(
                            text, fallback_model
                        )
                        model_used = fallback_model.value
                        break
                    except Exception as e:
                        logger.warning(f"Fallback {fallback_model} failed: {e}")
                        continue

            # Create embedding object if successful
            if embedding_vector and model_used:
                embedding = FunctionEmbedding(
                    function_id=function_id,
                    embedding=embedding_vector,
                    model=model_used,
                    text_embedded=text,
                    timestamp=time.time(),
                    signature_hash=signature_hash,
                )
                embeddings.append(embedding)
            else:
                logger.error(f"Failed to generate embedding for {function_id}")

        return embeddings

    def get_stats(self) -> dict[str, float]:
        """Get generation statistics."""
        avg_time = (
            self.stats["total_generation_time"] / self.stats["embeddings_generated"]
            if self.stats["embeddings_generated"] > 0
            else 0
        )

        return {
            "embeddings_generated": self.stats["embeddings_generated"],
            "average_generation_time_ms": avg_time * 1000,
            "fallback_rate": (
                self.stats["fallbacks_used"] / self.stats["embeddings_generated"]
                if self.stats["embeddings_generated"] > 0
                else 0
            ),
            "batches_processed": self.stats["batch_count"],
        }
