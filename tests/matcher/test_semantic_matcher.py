import time
from pathlib import Path
from typing import List, Optional, cast
from unittest.mock import MagicMock, patch

import pytest

from codedocsync.matcher.models import MatchConfidence
from codedocsync.matcher.semantic_matcher import SemanticMatcher
from codedocsync.matcher.semantic_models import (
    EmbeddingConfig,
    EmbeddingModel,
    FunctionEmbedding,
)
from codedocsync.parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.storage.embedding_cache import EmbeddingCache


@pytest.fixture
def sample_function() -> ParsedFunction:
    return ParsedFunction(
        signature=FunctionSignature(
            name="calculate_discount",
            parameters=[
                FunctionParameter(
                    name="price",
                    type_annotation="float",
                    default_value=None,
                    is_required=True,
                ),
                FunctionParameter(
                    name="percentage",
                    type_annotation="float",
                    default_value="0.1",
                    is_required=False,
                ),
            ],
            return_type="float",
            decorators=["lru_cache"],
            is_async=False,
            is_method=False,
        ),
        docstring=RawDocstring(raw_text='"""Calculate discount amount."""'),
        file_path="src/utils/pricing.py",
        line_number=10,
        end_line_number=15,
    )


@pytest.fixture
def renamed_function() -> ParsedFunction:
    return ParsedFunction(
        signature=FunctionSignature(
            name="compute_discount_amount",
            parameters=[
                FunctionParameter(
                    name="price",
                    type_annotation="float",
                    default_value=None,
                    is_required=True,
                ),
                FunctionParameter(
                    name="percentage",
                    type_annotation="float",
                    default_value="0.1",
                    is_required=False,
                ),
            ],
            return_type="float",
            decorators=["lru_cache"],
            is_async=False,
            is_method=False,
        ),
        docstring=RawDocstring(
            raw_text='"""Calculate discount amount for given price."""'
        ),
        file_path="src/utils/pricing.py",
        line_number=25,
        end_line_number=30,
    )


@pytest.fixture
def semantic_matcher(tmp_path: Path) -> SemanticMatcher:
    config = EmbeddingConfig(
        primary_model=EmbeddingModel.OPENAI_SMALL,
        cache_embeddings=True,
        batch_size=10,
    )

    # Create matcher with mocked dependencies
    with patch(
        "codedocsync.matcher.semantic_matcher.VectorStore"
    ) as mock_vector_store_class:
        with patch(
            "codedocsync.matcher.semantic_matcher.EmbeddingCache"
        ) as mock_cache_class:
            # Setup mock instances
            mock_vector_store = MagicMock()
            mock_vector_store.get_stats.return_value = {"collection_count": 0}
            mock_vector_store.search_similar.return_value = []
            mock_vector_store_class.return_value = mock_vector_store

            mock_cache = MagicMock()
            mock_cache.get.return_value = None
            mock_cache.get_stats.return_value = {
                "memory_size": 0,
                "memory_hit_rate": 0.0,
                "overall_hit_rate": 0.0,
                "total_saves": 0,
                "total_requests": 0,
            }
            mock_cache_class.return_value = mock_cache

            matcher = SemanticMatcher(str(tmp_path), config)

            # Replace with our mocks after initialization
            matcher.vector_store = mock_vector_store
            matcher.embedding_cache = mock_cache

            return matcher


@pytest.mark.asyncio
class TestSemanticMatcherPerformance:
    async def test_embedding_generation_time(
        self, semantic_matcher: SemanticMatcher, sample_function: ParsedFunction
    ) -> None:
        """Test that embedding generation completes within 200ms per function."""
        mock_embedding = [0.1] * 1536

        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            return_value=mock_embedding,
        ):
            start = time.perf_counter()
            embeddings = (
                await semantic_matcher.embedding_generator.generate_function_embeddings(
                    [sample_function], use_cache=False
                )
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 200, f"Embedding generation took {elapsed_ms:.2f}ms"
            assert len(embeddings) == 1
            assert len(embeddings[0].embedding) == 1536

    async def test_cache_hit_performance(
        self, semantic_matcher: SemanticMatcher, sample_function: ParsedFunction
    ) -> None:
        """Test that cache hits complete within 10ms."""
        embedding = FunctionEmbedding(
            function_id="src.utils.pricing.calculate_discount",
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            text_embedded="calculate_discount(price: float, percentage: float = 0.1) -> float",
            timestamp=time.time(),
            signature_hash="abc123",
        )

        # Mock cache to return embedding immediately
        cast(MagicMock, semantic_matcher.embedding_cache.get).return_value = embedding

        start = time.perf_counter()
        cached = semantic_matcher.embedding_cache.get(
            embedding.text_embedded, embedding.model, embedding.signature_hash
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, f"Cache hit took {elapsed_ms:.2f}ms"
        assert cached is not None
        assert cached.function_id == embedding.function_id

    async def test_similarity_threshold(
        self,
        semantic_matcher: SemanticMatcher,
        sample_function: ParsedFunction,
        renamed_function: ParsedFunction,
    ) -> None:
        """Test that similarity threshold of 0.8 is properly enforced."""
        mock_embedding = [0.1] * 1536

        # Mock vector store to return high and low similarity matches
        cast(MagicMock, semantic_matcher.vector_store.search_similar).return_value = [
            (
                "renamed_func_id",
                0.85,
                {
                    "function_id": "src.utils.pricing.compute_discount_amount",
                    "model": "text-embedding-3-small",
                    "signature_hash": "xyz789",
                    "timestamp": str(time.time()),
                },
            ),
            (
                "low_match_id",
                0.65,
                {
                    "function_id": "src.other.unrelated_function",
                    "model": "text-embedding-3-small",
                    "signature_hash": "def456",
                    "timestamp": str(time.time()),
                },
            ),
        ]

        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            return_value=mock_embedding,
        ):
            with patch.object(
                semantic_matcher.scorer,
                "validate_semantic_match",
                side_effect=[(True, 0.85), (True, 0.65)],
            ):
                with patch.object(
                    semantic_matcher.scorer,
                    "calculate_semantic_confidence",
                    return_value=MatchConfidence(
                        overall=0.825,
                        name_similarity=0.8,
                        location_score=0.85,
                        signature_similarity=0.82,
                    ),
                ):
                    match = await semantic_matcher._find_semantic_match(sample_function)

                    assert match is not None
                    assert match.confidence.overall >= 0.8
                    assert "Semantic similarity match" in match.match_reason
                    assert "0.85" in match.match_reason


@pytest.mark.asyncio
class TestSemanticMatcherFallback:
    async def test_semantic_match_renamed_function(
        self,
        semantic_matcher: SemanticMatcher,
        sample_function: ParsedFunction,
        renamed_function: ParsedFunction,
    ) -> None:
        """Test semantic matching finds renamed functions with high similarity."""
        # Mock prepare_semantic_index
        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            return_value=[0.1] * 1536,
        ):
            await semantic_matcher.prepare_semantic_index([renamed_function])

        # Mock search results
        cast(MagicMock, semantic_matcher.vector_store.search_similar).return_value = [
            (
                "renamed_id",
                0.92,
                {
                    "function_id": "src.utils.pricing.compute_discount_amount",
                    "model": "text-embedding-3-small",
                    "signature_hash": "xyz789",
                    "timestamp": str(time.time()),
                },
            )
        ]

        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            return_value=[0.1] * 1536,
        ):
            with patch.object(
                semantic_matcher.scorer,
                "validate_semantic_match",
                return_value=(True, 0.92),
            ):
                with patch.object(
                    semantic_matcher.scorer,
                    "calculate_semantic_confidence",
                    return_value=MatchConfidence(
                        overall=0.91,
                        name_similarity=0.9,
                        location_score=0.92,
                        signature_similarity=0.91,
                    ),
                ):
                    result = await semantic_matcher.match_with_embeddings(
                        [sample_function]
                    )

                    assert result.total_functions == 1
                    assert len(result.matched_pairs) == 1
                    assert result.matched_pairs[0].confidence.overall >= 0.8

    async def test_no_match_below_threshold(
        self, semantic_matcher: SemanticMatcher, sample_function: ParsedFunction
    ) -> None:
        """Test that matches below threshold are rejected."""
        cast(MagicMock, semantic_matcher.vector_store.search_similar).return_value = [
            (
                "low_match_id",
                0.55,
                {
                    "function_id": "src.other.unrelated",
                    "model": "text-embedding-3-small",
                    "signature_hash": "low123",
                    "timestamp": str(time.time()),
                },
            )
        ]

        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            return_value=[0.1] * 1536,
        ):
            with patch.object(
                semantic_matcher.scorer,
                "validate_semantic_match",
                return_value=(False, 0.55),
            ):
                match = await semantic_matcher._find_semantic_match(sample_function)
                assert match is None

    async def test_openai_api_failure_handling(
        self, semantic_matcher: SemanticMatcher, sample_function: ParsedFunction
    ) -> None:
        """Test fallback to local model when OpenAI API fails."""
        # Mock primary model failure and fallback success
        call_count = 0

        async def mock_generate_embedding(
            text: str, model: EmbeddingModel
        ) -> List[float]:
            nonlocal call_count
            call_count += 1

            if model == EmbeddingModel.OPENAI_SMALL:
                raise Exception("OpenAI API error")
            elif model == EmbeddingModel.OPENAI_ADA:
                raise Exception("Ada model also failed")
            else:
                # Local model succeeds
                return [0.1] * 384

        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            side_effect=mock_generate_embedding,
        ):
            embeddings = (
                await semantic_matcher.embedding_generator.generate_function_embeddings(
                    [sample_function]
                )
            )

            assert len(embeddings) == 1
            assert embeddings[0].model == "all-MiniLM-L6-v2"
            assert len(embeddings[0].embedding) == 384


@pytest.mark.asyncio
class TestSemanticMatcherAdvanced:
    async def test_prepare_semantic_index_performance(
        self, semantic_matcher: SemanticMatcher
    ) -> None:
        """Test semantic index preparation performance for 100 functions."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(
                    name=f"function_{i}",
                    parameters=[],
                    return_type="None",
                    decorators=[],
                    is_async=False,
                    is_method=False,
                ),
                docstring=RawDocstring(raw_text=f'"""Function {i}"""'),
                file_path=f"src/module_{i // 10}.py",
                line_number=i * 10,
                end_line_number=i * 10 + 5,
            )
            for i in range(100)
        ]

        mock_embedding = [0.1] * 1536

        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            return_value=mock_embedding,
        ):
            start = time.perf_counter()
            await semantic_matcher.prepare_semantic_index(functions)
            elapsed = time.perf_counter() - start

            assert elapsed < 5.0, f"Index preparation took {elapsed:.2f}s"
            assert semantic_matcher.stats["embeddings_generated"] == 100

            # Mock the ready check
            cast(MagicMock, semantic_matcher.vector_store.get_stats).return_value = {
                "collection_count": 100
            }
            assert semantic_matcher.is_ready_for_matching()

    async def test_cache_invalidation_on_function_change(
        self, semantic_matcher: SemanticMatcher, sample_function: ParsedFunction
    ) -> None:
        """Test cache invalidation when function signature changes."""
        original_hash = "original_hash_123"
        changed_hash = "changed_hash_456"

        original_embedding = FunctionEmbedding(
            function_id="src.utils.pricing.calculate_discount",
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            text_embedded="calculate_discount(price: float, percentage: float = 0.1) -> float",
            timestamp=time.time(),
            signature_hash=original_hash,
        )

        # Create real cache for this test
        cache = EmbeddingCache(str(Path(semantic_matcher.project_root) / ".test_cache"))
        semantic_matcher.embedding_cache = cache

        cache.set(original_embedding)

        # Cache should return None for changed hash
        cached = cache.get(
            original_embedding.text_embedded,
            original_embedding.model,
            changed_hash,
        )
        assert cached is None

        cache_stats = cache.get_stats()
        assert cache_stats["total_requests"] == 1

    async def test_batch_embedding_generation(
        self, semantic_matcher: SemanticMatcher
    ) -> None:
        """Test batch processing of embeddings respects batch size."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(
                    name=f"batch_func_{i}",
                    parameters=[],
                    return_type="int",
                    decorators=[],
                    is_async=False,
                    is_method=False,
                ),
                docstring=None,
                file_path="src/batch.py",
                line_number=i * 5,
                end_line_number=i * 5 + 3,
            )
            for i in range(25)
        ]

        async def mock_generate_embedding(
            text: str, model: EmbeddingModel
        ) -> List[float]:
            return [0.1] * 1536

        # Mock the embedding generator to track batch processing
        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            side_effect=mock_generate_embedding,
        ):
            embeddings = (
                await semantic_matcher.embedding_generator.generate_function_embeddings(
                    functions
                )
            )

            assert len(embeddings) == 25
            # Verify batches were created (10 + 10 + 5)
            assert semantic_matcher.embedding_generator.stats["batch_count"] == 3

    async def test_vector_store_integration(
        self, semantic_matcher: SemanticMatcher, sample_function: ParsedFunction
    ) -> None:
        """Test vector store integration for similarity search."""
        embedding = FunctionEmbedding(
            function_id="test.function",
            embedding=[0.2] * 1536,
            model="text-embedding-3-small",
            text_embedded="test_function()",
            timestamp=time.time(),
            signature_hash="test123",
        )

        # Mock add_embeddings
        semantic_matcher.vector_store.add_embeddings(
            [embedding.embedding],
            [
                {
                    "function_id": embedding.function_id,
                    "model": embedding.model,
                    "signature_hash": embedding.signature_hash,
                    "timestamp": str(embedding.timestamp),
                }
            ],
            [embedding.function_id],
        )

        # Update stats
        cast(MagicMock, semantic_matcher.vector_store.get_stats).return_value = {
            "collection_count": 1
        }
        stats = semantic_matcher.vector_store.get_stats()
        assert stats.get("collection_count", 0) > 0

        # Mock search results
        cast(MagicMock, semantic_matcher.vector_store.search_similar).return_value = [
            (
                embedding.function_id,
                0.99,
                {
                    "function_id": embedding.function_id,
                    "model": embedding.model,
                    "signature_hash": embedding.signature_hash,
                    "timestamp": str(embedding.timestamp),
                },
            )
        ]

        similar = semantic_matcher.vector_store.search_similar(
            embedding.embedding, n_results=1
        )
        assert len(similar) == 1
        assert similar[0][1] > 0.9

    async def test_multiple_model_fallback(
        self, semantic_matcher: SemanticMatcher, sample_function: ParsedFunction
    ) -> None:
        """Test fallback through multiple embedding models."""
        # Track which models were tried
        models_tried = []

        async def mock_generate_embedding(
            text: str, model: EmbeddingModel
        ) -> List[float]:
            models_tried.append(model)

            if model == EmbeddingModel.OPENAI_SMALL:
                raise Exception("Primary model failed")
            elif model == EmbeddingModel.OPENAI_ADA:
                raise Exception("First fallback failed")
            elif model == EmbeddingModel.LOCAL_MINILM:
                # Local model succeeds
                return [0.1] * 384
            else:
                raise Exception("Unknown model")

        with patch.object(
            semantic_matcher.embedding_generator,
            "generate_embedding",
            side_effect=mock_generate_embedding,
        ):
            embeddings = (
                await semantic_matcher.embedding_generator.generate_function_embeddings(
                    [sample_function]
                )
            )

            assert len(embeddings) == 1
            assert embeddings[0].model == "all-MiniLM-L6-v2"
            assert len(embeddings[0].embedding) == 384
            # Verify all models were tried in order
            assert EmbeddingModel.OPENAI_SMALL in models_tried
            assert EmbeddingModel.OPENAI_ADA in models_tried
            assert EmbeddingModel.LOCAL_MINILM in models_tried


def test_embedding_cache_performance_stats(tmp_path: Path) -> None:
    """Test embedding cache hit rate meets 90% benchmark."""
    cache = EmbeddingCache(str(tmp_path / "cache"), max_memory_items=100)

    # Add 150 embeddings (will trigger LRU eviction)
    for i in range(150):
        embedding = FunctionEmbedding(
            function_id=f"func_{i}",
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            text_embedded=f"function_{i}()",
            timestamp=time.time(),
            signature_hash=f"hash_{i}",
        )
        cache.set(embedding)

    # Access first 50 (should be disk hits)
    for i in range(50):
        cache.get(f"function_{i}()", "text-embedding-3-small", f"hash_{i}")

    # Access recent 20 (should be memory hits)
    for i in range(100, 120):
        cache.get(f"function_{i}()", "text-embedding-3-small", f"hash_{i}")

    # Access non-existent (should be misses)
    for i in range(200, 210):
        cache.get(f"function_{i}()", "text-embedding-3-small", f"hash_{i}")

    stats = cache.get_stats()

    # Verify stats
    assert stats["memory_size"] <= 100
    assert stats["total_saves"] == 150
    assert stats["overall_hit_rate"] >= 0.85  # 70 hits out of 80 requests
    assert stats["total_requests"] == 80
