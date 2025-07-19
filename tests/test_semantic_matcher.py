import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from codedocsync.matcher.semantic_matcher import SemanticMatcher
from codedocsync.matcher.semantic_models import EmbeddingConfig, FunctionEmbedding
from codedocsync.parser import (
    ParsedFunction,
    FunctionSignature,
    FunctionParameter,
    RawDocstring,
)


class TestSemanticMatcher:
    """Test suite for SemanticMatcher - Part 1 implementation."""

    @pytest.fixture
    def sample_functions(self):
        """Create sample functions for testing."""
        functions = []
        for i in range(5):
            func = ParsedFunction(
                signature=FunctionSignature(
                    name=f"test_function_{i}",
                    parameters=[
                        FunctionParameter(
                            name="param1",
                            type_annotation="str",
                            default_value=None,
                            is_required=True,
                        )
                    ],
                    return_type="None",
                ),
                docstring=RawDocstring(
                    raw_text=f"Test function {i} documentation", line_number=i * 10 + 2
                ),
                file_path=f"test_module_{i % 2}.py",
                line_number=i * 10,
                end_line_number=i * 10 + 5,
                source_code=f"def test_function_{i}(param1: str): pass",
            )
            functions.append(func)
        return functions

    @pytest.fixture
    def mock_config(self):
        """Create mock embedding configuration."""
        return EmbeddingConfig(batch_size=10, timeout_seconds=30, cache_embeddings=True)

    def test_semantic_matcher_initialization(self, mock_config):
        """Test SemanticMatcher initialization."""
        with (
            patch(
                "codedocsync.matcher.semantic_matcher.VectorStore"
            ) as mock_vector_store,
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            matcher = SemanticMatcher("/test/project", config=mock_config)

            assert matcher.project_root == "/test/project"
            assert matcher.config == mock_config
            assert "functions_processed" in matcher.stats
            assert "embeddings_generated" in matcher.stats
            assert "index_preparation_time" in matcher.stats

    @pytest.mark.asyncio
    async def test_prepare_semantic_index_with_cache_hits(
        self, sample_functions, mock_config
    ):
        """Test semantic index preparation with cached embeddings."""
        with (
            patch(
                "codedocsync.matcher.semantic_matcher.VectorStore"
            ) as mock_vector_store,
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            # Setup mocks
            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance

            mock_generator_instance = Mock()
            mock_generator.return_value = mock_generator_instance
            mock_generator_instance.prepare_function_text.return_value = (
                "def test_function(): pass"
            )
            mock_generator_instance.generate_signature_hash.return_value = "test_hash"
            mock_generator_instance.generate_function_embeddings = AsyncMock(
                return_value=[]
            )

            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance

            # Mock cache hits for all functions
            cached_embedding = FunctionEmbedding(
                function_id="test.function",
                embedding=[0.1] * 1536,
                model="text-embedding-3-small",
                text_embedded="def test_function(): pass",
                timestamp=time.time(),
                signature_hash="test_hash",
            )
            mock_cache_instance.get.return_value = cached_embedding

            matcher = SemanticMatcher("/test/project", config=mock_config)

            # Test index preparation
            await matcher.prepare_semantic_index(sample_functions)

            # Verify cache was checked for each function
            assert mock_cache_instance.get.call_count == len(sample_functions)

            # Verify no new embeddings were generated (all cache hits)
            mock_generator_instance.generate_function_embeddings.assert_called_once_with(
                [], use_cache=True
            )

            # Verify vector store was populated
            mock_vector_store_instance.add_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_semantic_index_with_cache_misses(
        self, sample_functions, mock_config
    ):
        """Test semantic index preparation with cache misses requiring new embeddings."""
        with (
            patch(
                "codedocsync.matcher.semantic_matcher.VectorStore"
            ) as mock_vector_store,
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            # Setup mocks
            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get.return_value = None  # All cache misses

            mock_generator_instance = Mock()
            mock_generator.return_value = mock_generator_instance
            mock_generator_instance.prepare_function_text.return_value = (
                "def test_function(): pass"
            )
            mock_generator_instance.generate_signature_hash.return_value = "test_hash"

            # Mock new embedding generation
            new_embeddings = [
                FunctionEmbedding(
                    function_id=f"test.function_{i}",
                    embedding=[0.1] * 1536,
                    model="text-embedding-3-small",
                    text_embedded="def test_function(): pass",
                    timestamp=time.time(),
                    signature_hash="test_hash",
                )
                for i in range(len(sample_functions))
            ]
            mock_generator_instance.generate_function_embeddings = AsyncMock(
                return_value=new_embeddings
            )

            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance

            matcher = SemanticMatcher("/test/project", config=mock_config)

            # Test index preparation
            await matcher.prepare_semantic_index(sample_functions)

            # Verify new embeddings were generated
            mock_generator_instance.generate_function_embeddings.assert_called_once_with(
                sample_functions, use_cache=True
            )

            # Verify embeddings were cached
            assert mock_cache_instance.set.call_count == len(new_embeddings)

            # Verify vector store was populated with new embeddings
            mock_vector_store_instance.add_embeddings.assert_called_once()
            args = mock_vector_store_instance.add_embeddings.call_args[0]
            assert len(args[0]) == len(new_embeddings)  # vectors
            assert len(args[1]) == len(new_embeddings)  # metadata
            assert len(args[2]) == len(new_embeddings)  # ids

    @pytest.mark.asyncio
    async def test_prepare_semantic_index_force_reindex(
        self, sample_functions, mock_config
    ):
        """Test force reindexing ignores cache."""
        with (
            patch(
                "codedocsync.matcher.semantic_matcher.VectorStore"
            ) as mock_vector_store,
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            # Setup mocks
            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance

            mock_generator_instance = Mock()
            mock_generator.return_value = mock_generator_instance
            mock_generator_instance.prepare_function_text.return_value = (
                "def test_function(): pass"
            )
            mock_generator_instance.generate_signature_hash.return_value = "test_hash"
            mock_generator_instance.generate_function_embeddings = AsyncMock(
                return_value=[]
            )

            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance

            matcher = SemanticMatcher("/test/project", config=mock_config)

            # Test force reindex
            await matcher.prepare_semantic_index(sample_functions, force_reindex=True)

            # Verify cache was not checked when force_reindex=True
            mock_cache_instance.get.assert_not_called()

            # Verify all functions were sent for embedding generation
            mock_generator_instance.generate_function_embeddings.assert_called_once_with(
                sample_functions, use_cache=True
            )

    def test_create_placeholder_match_result(self, sample_functions):
        """Test placeholder match result creation."""
        with (
            patch("codedocsync.matcher.semantic_matcher.VectorStore"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingGenerator"),
        ):
            matcher = SemanticMatcher("/test/project")

            # Test with no previous results
            result = matcher.create_placeholder_match_result(sample_functions)

            assert result.total_functions == len(sample_functions)
            assert result.total_docs == len(sample_functions)
            assert len(result.matched_pairs) == 0
            assert len(result.unmatched_functions) == len(sample_functions)

    def test_get_stats(self):
        """Test statistics retrieval."""
        with (
            patch(
                "codedocsync.matcher.semantic_matcher.VectorStore"
            ) as mock_vector_store,
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            # Setup mock return values for stats
            mock_generator_instance = Mock()
            mock_generator.return_value = mock_generator_instance
            mock_generator_instance.get_stats.return_value = {"embeddings_generated": 5}

            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_stats.return_value = {"cache_hits": 10}

            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.get_stats.return_value = {
                "searches_performed": 3
            }

            matcher = SemanticMatcher("/test/project")

            # Update some stats
            matcher.stats["functions_processed"] = 10
            matcher.stats["semantic_matches_found"] = 2
            matcher.stats["total_time"] = 1.5
            matcher.stats["index_preparation_time"] = 0.8

            stats = matcher.get_stats()

            assert stats["functions_processed"] == 10
            assert stats["semantic_matches_found"] == 2
            assert stats["match_rate"] == 0.2
            assert stats["average_time_per_function_ms"] == 150.0
            assert stats["index_preparation_time_s"] == 0.8
            assert "embedding_stats" in stats
            assert "cache_stats" in stats
            assert "vector_store_stats" in stats

    def test_is_ready_for_matching(self):
        """Test readiness check for matching operations."""
        with (
            patch(
                "codedocsync.matcher.semantic_matcher.VectorStore"
            ) as mock_vector_store,
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingGenerator"),
        ):
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance

            matcher = SemanticMatcher("/test/project")

            # Test when vector store has embeddings
            mock_vector_store_instance.get_stats.return_value = {
                "collection_count": 100
            }
            assert matcher.is_ready_for_matching() == True

            # Test when vector store is empty
            mock_vector_store_instance.get_stats.return_value = {"collection_count": 0}
            assert matcher.is_ready_for_matching() == False

            # Test when vector store throws exception
            mock_vector_store_instance.get_stats.side_effect = Exception(
                "Connection failed"
            )
            assert matcher.is_ready_for_matching() == False

    def test_clear_index(self):
        """Test index clearing functionality."""
        with (
            patch("codedocsync.matcher.semantic_matcher.VectorStore"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingGenerator"),
        ):
            matcher = SemanticMatcher("/test/project")

            # Set some stats
            matcher.stats["embeddings_generated"] = 10
            matcher.stats["index_preparation_time"] = 2.5

            # Clear index
            matcher.clear_index()

            # Verify stats were reset
            assert matcher.stats["embeddings_generated"] == 0
            assert matcher.stats["index_preparation_time"] == 0.0

    def test_get_embedding_for_function(self, sample_functions):
        """Test getting cached embedding for specific function."""
        with (
            patch("codedocsync.matcher.semantic_matcher.VectorStore"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            # Setup mocks
            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance

            mock_generator_instance = Mock()
            mock_generator.return_value = mock_generator_instance
            mock_generator_instance.prepare_function_text.return_value = (
                "def test_function(): pass"
            )
            mock_generator_instance.generate_signature_hash.return_value = "test_hash"

            cached_embedding = FunctionEmbedding(
                function_id="test.function",
                embedding=[0.1] * 1536,
                model="text-embedding-3-small",
                text_embedded="def test_function(): pass",
                timestamp=time.time(),
                signature_hash="test_hash",
            )
            mock_cache_instance.get.return_value = cached_embedding

            matcher = SemanticMatcher("/test/project")

            # Test getting embedding
            result = matcher.get_embedding_for_function(sample_functions[0])

            assert result == cached_embedding
            mock_cache_instance.get.assert_called_once()

    def test_get_embedding_for_function_not_found(self, sample_functions):
        """Test getting embedding when not cached."""
        with (
            patch("codedocsync.matcher.semantic_matcher.VectorStore"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            # Setup mocks
            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get.return_value = None

            mock_generator_instance = Mock()
            mock_generator.return_value = mock_generator_instance
            mock_generator_instance.prepare_function_text.return_value = (
                "def test_function(): pass"
            )
            mock_generator_instance.generate_signature_hash.return_value = "test_hash"

            matcher = SemanticMatcher("/test/project")

            # Test getting embedding
            result = matcher.get_embedding_for_function(sample_functions[0])

            assert result is None
            mock_cache_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_preparation_performance(self, sample_functions):
        """Test that index preparation completes within reasonable time."""
        with (
            patch("codedocsync.matcher.semantic_matcher.VectorStore"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache") as mock_cache,
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            # Setup fast mocks
            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get.return_value = None

            mock_generator_instance = Mock()
            mock_generator.return_value = mock_generator_instance
            mock_generator_instance.prepare_function_text.return_value = (
                "def test(): pass"
            )
            mock_generator_instance.generate_signature_hash.return_value = "hash"
            mock_generator_instance.generate_function_embeddings = AsyncMock(
                return_value=[]
            )

            matcher = SemanticMatcher("/test/project")

            start_time = time.time()
            await matcher.prepare_semantic_index(sample_functions)
            duration = time.time() - start_time

            # Should complete quickly with mocks
            assert duration < 1.0  # Very generous for mocked operations
            assert matcher.stats["index_preparation_time"] > 0

    def test_semantic_matcher_with_custom_config(self):
        """Test SemanticMatcher with custom configuration."""
        custom_config = EmbeddingConfig(
            batch_size=50, timeout_seconds=60, cache_embeddings=False
        )

        with (
            patch("codedocsync.matcher.semantic_matcher.VectorStore"),
            patch("codedocsync.matcher.semantic_matcher.EmbeddingCache"),
            patch(
                "codedocsync.matcher.semantic_matcher.EmbeddingGenerator"
            ) as mock_generator,
        ):
            matcher = SemanticMatcher("/test/project", config=custom_config)

            assert matcher.config.batch_size == 50
            assert matcher.config.timeout_seconds == 60
            assert matcher.config.cache_embeddings == False

            # Verify config was passed to generator
            mock_generator.assert_called_once_with(custom_config)
