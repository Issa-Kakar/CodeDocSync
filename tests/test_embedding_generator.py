import sys
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest

# Mock external dependencies at module level before other imports
mock_openai = MagicMock()
mock_openai.error = MagicMock()
mock_openai.error.RateLimitError = Exception
mock_openai.Embedding = MagicMock()
sys.modules["openai"] = mock_openai

mock_sentence_transformers = MagicMock()
sys.modules["sentence_transformers"] = mock_sentence_transformers

# These imports must come after the mocking  # noqa: E402
from codedocsync.matcher.embedding_generator import EmbeddingGenerator  # noqa: E402
from codedocsync.matcher.semantic_models import (  # noqa: E402
    EmbeddingConfig,
    EmbeddingModel,
    FunctionEmbedding,
)
from codedocsync.parser import (  # noqa: E402
    ParsedFunction,
    FunctionSignature,
    FunctionParameter,
    RawDocstring,
)


class TestEmbeddingGenerator:
    """Test embedding generation with various scenarios."""

    @pytest.fixture
    def mock_generator(self):
        """Create properly mocked embedding generator."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock valid configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = None
            mock_config_manager.return_value = mock_manager

            # Also mock the provider initialization
            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_st.return_value = Mock()
                yield EmbeddingGenerator()

    @pytest.fixture
    def sample_function(self):
        """Create a sample function for testing."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="test_function",
                parameters=[
                    FunctionParameter(
                        name="param1", type_annotation="str", default_value=None
                    ),
                    FunctionParameter(
                        name="param2", type_annotation="int", default_value="42"
                    ),
                ],
                return_type="bool",
            ),
            docstring=RawDocstring(
                raw_text="Test function that does something useful.\n\nDetailed description here.",
                line_number=2,
            ),
            file_path="test_module.py",
            line_number=1,
            end_line_number=10,
            source_code="def test_function(param1: str, param2: int = 42) -> bool:\n    return True",
        )

    @pytest.fixture
    def simple_function(self):
        """Create a simple function without docstring."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="simple_func", parameters=[], return_type="None"
            ),
            docstring=None,
            file_path="simple.py",
            line_number=1,
            end_line_number=2,
            source_code="def simple_func(): pass",
        )

    def test_prepare_function_text_with_docstring(
        self, mock_generator, sample_function
    ):
        """Test text preparation with docstring."""
        text = mock_generator.prepare_function_text(sample_function)

        expected = "def test_function(param1: str, param2: int = 42) -> bool | Test function that does something useful."
        assert text == expected

    def test_prepare_function_text_without_docstring(
        self, mock_generator, simple_function
    ):
        """Test text preparation without docstring."""
        text = mock_generator.prepare_function_text(simple_function)

        expected = "def simple_func() -> None"
        assert text == expected

    def test_prepare_function_text_truncation(self, mock_generator):
        """Test text truncation for very long text."""
        # Create function with very long signature
        long_params = [
            FunctionParameter(
                name=f"very_long_parameter_name_{i}", type_annotation="str"
            )
            for i in range(50)
        ]

        long_function = ParsedFunction(
            signature=FunctionSignature(
                name="function_with_very_long_signature_and_many_parameters",
                parameters=long_params,
                return_type="Dict[str, List[Optional[Union[str, int, float, bool]]]]",
            ),
            docstring=RawDocstring(
                raw_text="This is a very long docstring that describes a function in great detail with lots of information.",
                line_number=2,
            ),
            file_path="long_module.py",
            line_number=1,
            end_line_number=50,
            source_code="# very long function code",
        )

        text = mock_generator.prepare_function_text(long_function)

        # Should be truncated to 512 characters or less
        assert len(text) <= 512
        assert text.endswith("...")

    def test_generate_function_id(self, mock_generator, sample_function):
        """Test function ID generation."""
        function_id = mock_generator.generate_function_id(sample_function)

        assert function_id == "test_module.test_function"

    def test_generate_function_id_with_path(self, mock_generator):
        """Test function ID generation with complex path."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="my_func", parameters=[], return_type="None"
            ),
            docstring=None,
            file_path="src/package/module.py",
            line_number=1,
            end_line_number=2,
            source_code="def my_func(): pass",
        )

        function_id = mock_generator.generate_function_id(function)

        assert function_id == "src.package.module.my_func"

    def test_generate_signature_hash(self, mock_generator, sample_function):
        """Test signature hash generation."""
        hash1 = mock_generator.generate_signature_hash(sample_function)
        hash2 = mock_generator.generate_signature_hash(sample_function)

        # Should be consistent
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars
        assert isinstance(hash1, str)

    def test_signature_hash_changes_with_signature(
        self, mock_generator, sample_function
    ):
        """Test that signature hash changes when signature changes."""
        original_hash = mock_generator.generate_signature_hash(sample_function)

        # Modify the function signature
        sample_function.signature.name = "modified_function"
        modified_hash = mock_generator.generate_signature_hash(sample_function)

        assert original_hash != modified_hash

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock valid configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = "test-key"
            mock_config_manager.return_value = mock_manager

            config = EmbeddingConfig(primary_model=EmbeddingModel.OPENAI_SMALL)

            # No need to mock imports - already mocked at module level
            generator = EmbeddingGenerator(config)

            assert generator.config == config
            assert mock_manager.validate_config.called
            assert "openai" in generator.providers

    def test_init_with_invalid_config(self):
        """Test initialization with invalid configuration."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock invalid configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = False
            mock_config_manager.return_value = mock_manager

            with pytest.raises(ValueError, match="Invalid embedding configuration"):
                EmbeddingGenerator()

    def test_init_local_fallback(self):
        """Test initialization with local model fallback."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock configuration without OpenAI
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = None
            mock_config_manager.return_value = mock_manager

            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_model = Mock()
                mock_st.return_value = mock_model

                generator = EmbeddingGenerator()

                assert "local" in generator.providers
                assert generator.providers["local"] == mock_model

    async def test_generate_openai_embedding(self):
        """Test OpenAI embedding generation."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = "test-key"
            mock_config_manager.return_value = mock_manager

            # Use the module-level mock
            mock_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            mock_openai.Embedding.acreate = AsyncMock(return_value=mock_response)

            generator = EmbeddingGenerator()
            embedding = await generator._generate_openai_embedding(
                "test text", "text-embedding-3-small"
            )

            assert embedding == [0.1, 0.2, 0.3]
            mock_openai.Embedding.acreate.assert_called_once_with(
                input="test text", model="text-embedding-3-small"
            )

    async def test_generate_openai_embedding_rate_limit(self):
        """Test OpenAI embedding with rate limit error."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = "test-key"
            mock_config_manager.return_value = mock_manager

            # Use the module-level mock
            mock_openai.Embedding.acreate = AsyncMock(
                side_effect=mock_openai.error.RateLimitError("Rate limit")
            )

            generator = EmbeddingGenerator()

            with pytest.raises(Exception):
                await generator._generate_openai_embedding(
                    "test text", "text-embedding-3-small"
                )

    def test_generate_local_embedding(self):
        """Test local embedding generation."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = None
            mock_config_manager.return_value = mock_manager

            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                # Mock local model
                mock_model = Mock()
                mock_embedding = Mock()
                mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
                mock_model.encode.return_value = mock_embedding
                mock_st.return_value = mock_model

                generator = EmbeddingGenerator()
                embedding = generator._generate_local_embedding("test text")

                assert embedding == [0.1, 0.2, 0.3]
                mock_model.encode.assert_called_once_with(
                    "test text", convert_to_numpy=True
                )

    async def test_generate_function_embeddings_batch(self, sample_function):
        """Test batch processing of function embeddings."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = None
            mock_config_manager.return_value = mock_manager

            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                # Mock local model
                mock_model = Mock()
                mock_embedding = Mock()
                mock_embedding.tolist.return_value = [0.1] * 384
                mock_model.encode.return_value = mock_embedding
                mock_st.return_value = mock_model

                config = EmbeddingConfig(
                    batch_size=2, primary_model=EmbeddingModel.LOCAL_MINILM
                )
                generator = EmbeddingGenerator(config)

                # Test with 3 functions (should create 2 batches)
                functions = [sample_function] * 3
                embeddings = await generator.generate_function_embeddings(functions)

                assert len(embeddings) == 3
                assert generator.stats["batch_count"] == 2
                assert generator.stats["embeddings_generated"] == 3

                # Check embedding structure
                for embedding in embeddings:
                    assert isinstance(embedding, FunctionEmbedding)
                    assert embedding.function_id == "test_module.test_function"
                    assert embedding.model == "all-MiniLM-L6-v2"
                    assert len(embedding.embedding) == 384

    async def test_fallback_model_usage(self, sample_function):
        """Test fallback to secondary model when primary fails."""
        with patch(
            "codedocsync.matcher.embedding_generator.EmbeddingConfigManager"
        ) as mock_config_manager:
            # Mock configuration
            mock_manager = Mock()
            mock_manager.validate_config.return_value = True
            mock_manager.get_api_key.return_value = "test-key"
            mock_config_manager.return_value = mock_manager

            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                # Mock OpenAI failure
                mock_openai.Embedding.acreate = AsyncMock(
                    side_effect=Exception("OpenAI failed")
                )

                # Mock local model success
                mock_model = Mock()
                mock_embedding = Mock()
                mock_embedding.tolist.return_value = [0.1] * 384
                mock_model.encode.return_value = mock_embedding
                mock_st.return_value = mock_model

                generator = EmbeddingGenerator()
                embeddings = await generator.generate_function_embeddings(
                    [sample_function]
                )

                assert len(embeddings) == 1
                assert embeddings[0].model == "all-MiniLM-L6-v2"  # Local model
                assert generator.stats["fallbacks_used"] == 1

    def test_get_stats(self, mock_generator, sample_function):
        """Test statistics collection."""
        # Manually set some stats
        mock_generator.stats["embeddings_generated"] = 10
        mock_generator.stats["total_generation_time"] = 5.0
        mock_generator.stats["fallbacks_used"] = 2
        mock_generator.stats["batch_count"] = 3

        stats = mock_generator.get_stats()

        assert stats["embeddings_generated"] == 10
        assert stats["average_generation_time_ms"] == 500.0  # 5s / 10 * 1000
        assert stats["fallback_rate"] == 0.2  # 2/10
        assert stats["batches_processed"] == 3

    def test_get_stats_empty(self, mock_generator):
        """Test statistics with no generations."""
        stats = mock_generator.get_stats()

        assert stats["embeddings_generated"] == 0
        assert stats["average_generation_time_ms"] == 0
        assert stats["fallback_rate"] == 0
        assert stats["batches_processed"] == 0
