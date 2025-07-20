import time

import pytest

from codedocsync.matcher.semantic_models import (
    EmbeddingConfig,
    EmbeddingModel,
    FunctionEmbedding,
    SemanticMatch,
    SemanticSearchResult,
)


class TestEmbeddingModel:
    """Test the EmbeddingModel enum."""

    def test_embedding_model_values(self):
        """Test that embedding models have correct string values."""
        assert EmbeddingModel.OPENAI_SMALL.value == "text-embedding-3-small"
        assert EmbeddingModel.OPENAI_LARGE.value == "text-embedding-3-large"
        assert EmbeddingModel.OPENAI_ADA.value == "text-embedding-ada-002"
        assert EmbeddingModel.LOCAL_MINILM.value == "all-MiniLM-L6-v2"


class TestEmbeddingConfig:
    """Test the EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.primary_model == EmbeddingModel.OPENAI_SMALL
        assert EmbeddingModel.OPENAI_ADA in config.fallback_models
        assert EmbeddingModel.LOCAL_MINILM in config.fallback_models
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
        assert config.cache_embeddings is True

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch sizes
        config = EmbeddingConfig(batch_size=1)
        assert config.batch_size == 1

        config = EmbeddingConfig(batch_size=2048)
        assert config.batch_size == 2048

        # Invalid batch sizes
        with pytest.raises(ValueError, match="Batch size must be between 1 and 2048"):
            EmbeddingConfig(batch_size=0)

        with pytest.raises(ValueError, match="Batch size must be between 1 and 2048"):
            EmbeddingConfig(batch_size=2049)

    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeout
        config = EmbeddingConfig(timeout_seconds=10)
        assert config.timeout_seconds == 10

        # Invalid timeout
        with pytest.raises(ValueError, match="Timeout must be at least 10 seconds"):
            EmbeddingConfig(timeout_seconds=9)


class TestFunctionEmbedding:
    """Test the FunctionEmbedding dataclass."""

    def test_valid_embedding_creation(self):
        """Test creating valid function embeddings."""
        embedding = FunctionEmbedding(
            function_id="module.function_name",
            embedding=[0.1] * 1536,  # OpenAI small model size
            model="text-embedding-3-small",
            text_embedded="def function_name(): pass",
            timestamp=time.time(),
            signature_hash="abc123",
        )

        assert embedding.function_id == "module.function_name"
        assert len(embedding.embedding) == 1536
        assert embedding.model == "text-embedding-3-small"

    def test_embedding_dimension_validation(self):
        """Test embedding dimension validation for known models."""
        # Valid dimensions for OpenAI small
        embedding = FunctionEmbedding(
            function_id="test",
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            text_embedded="test",
            timestamp=time.time(),
            signature_hash="hash",
        )
        assert len(embedding.embedding) == 1536

        # Invalid dimensions for OpenAI small
        with pytest.raises(ValueError, match="Invalid embedding dimension"):
            FunctionEmbedding(
                function_id="test",
                embedding=[0.1] * 1000,  # Wrong size
                model="text-embedding-3-small",
                text_embedded="test",
                timestamp=time.time(),
                signature_hash="hash",
            )

    def test_different_model_dimensions(self):
        """Test dimension validation for different models."""
        # OpenAI large model
        embedding = FunctionEmbedding(
            function_id="test",
            embedding=[0.1] * 3072,
            model="text-embedding-3-large",
            text_embedded="test",
            timestamp=time.time(),
            signature_hash="hash",
        )
        assert len(embedding.embedding) == 3072

        # Local MiniLM model
        embedding = FunctionEmbedding(
            function_id="test",
            embedding=[0.1] * 384,
            model="all-MiniLM-L6-v2",
            text_embedded="test",
            timestamp=time.time(),
            signature_hash="hash",
        )
        assert len(embedding.embedding) == 384

    def test_unknown_model_no_validation(self):
        """Test that unknown models don't trigger validation."""
        # Unknown model should not validate dimensions
        embedding = FunctionEmbedding(
            function_id="test",
            embedding=[0.1] * 999,  # Any size
            model="unknown-model",
            text_embedded="test",
            timestamp=time.time(),
            signature_hash="hash",
        )
        assert len(embedding.embedding) == 999


class TestSemanticMatch:
    """Test the SemanticMatch dataclass."""

    def test_valid_semantic_match(self):
        """Test creating valid semantic matches."""
        match = SemanticMatch(
            source_function="module.func1",
            matched_function="module.func2",
            similarity_score=0.85,
            embedding_model="text-embedding-3-small",
        )

        assert match.source_function == "module.func1"
        assert match.matched_function == "module.func2"
        assert match.similarity_score == 0.85
        assert match.embedding_model == "text-embedding-3-small"
        assert match.match_metadata == {}

    def test_similarity_score_validation(self):
        """Test similarity score validation."""
        # Valid scores
        match = SemanticMatch(
            source_function="func1",
            matched_function="func2",
            similarity_score=0.0,
            embedding_model="test",
        )
        assert match.similarity_score == 0.0

        match = SemanticMatch(
            source_function="func1",
            matched_function="func2",
            similarity_score=1.0,
            embedding_model="test",
        )
        assert match.similarity_score == 1.0

        # Invalid scores
        with pytest.raises(
            ValueError, match="Similarity score must be between 0 and 1"
        ):
            SemanticMatch(
                source_function="func1",
                matched_function="func2",
                similarity_score=-0.1,
                embedding_model="test",
            )

        with pytest.raises(
            ValueError, match="Similarity score must be between 0 and 1"
        ):
            SemanticMatch(
                source_function="func1",
                matched_function="func2",
                similarity_score=1.1,
                embedding_model="test",
            )

    def test_match_metadata(self):
        """Test match metadata handling."""
        metadata = {"confidence": 0.9, "method": "cosine"}
        match = SemanticMatch(
            source_function="func1",
            matched_function="func2",
            similarity_score=0.85,
            embedding_model="test",
            match_metadata=metadata,
        )

        assert match.match_metadata == metadata
        assert match.match_metadata["confidence"] == 0.9


class TestSemanticSearchResult:
    """Test the SemanticSearchResult dataclass."""

    def test_empty_search_result(self):
        """Test search result with no matches."""
        result = SemanticSearchResult(
            query_function="test_func",
            matches=[],
            search_time_ms=25.0,
            total_candidates=100,
        )

        assert result.query_function == "test_func"
        assert result.matches == []
        assert result.get_best_match() is None
        assert result.filter_by_threshold(0.5) == []

    def test_search_result_with_matches(self):
        """Test search result with multiple matches."""
        matches = [
            SemanticMatch("query", "match1", 0.7, "model"),
            SemanticMatch("query", "match2", 0.9, "model"),
            SemanticMatch("query", "match3", 0.6, "model"),
        ]

        result = SemanticSearchResult(
            query_function="query",
            matches=matches,
            search_time_ms=50.0,
            total_candidates=1000,
        )

        assert len(result.matches) == 3
        assert result.total_candidates == 1000
        assert result.search_time_ms == 50.0

    def test_get_best_match(self):
        """Test getting the best match."""
        matches = [
            SemanticMatch("query", "match1", 0.7, "model"),
            SemanticMatch("query", "match2", 0.9, "model"),  # Best
            SemanticMatch("query", "match3", 0.6, "model"),
        ]

        result = SemanticSearchResult(
            query_function="query",
            matches=matches,
            search_time_ms=50.0,
            total_candidates=100,
        )

        best = result.get_best_match()
        assert best is not None
        assert best.similarity_score == 0.9
        assert best.matched_function == "match2"

    def test_filter_by_threshold(self):
        """Test filtering matches by threshold."""
        matches = [
            SemanticMatch("query", "match1", 0.7, "model"),  # Above 0.65
            SemanticMatch("query", "match2", 0.9, "model"),  # Above 0.65
            SemanticMatch("query", "match3", 0.6, "model"),  # Below 0.65
        ]

        result = SemanticSearchResult(
            query_function="query",
            matches=matches,
            search_time_ms=50.0,
            total_candidates=100,
        )

        # Filter with threshold 0.65
        filtered = result.filter_by_threshold(0.65)
        assert len(filtered) == 2
        assert all(m.similarity_score >= 0.65 for m in filtered)

        # Filter with threshold 0.8
        filtered = result.filter_by_threshold(0.8)
        assert len(filtered) == 1
        assert filtered[0].similarity_score == 0.9
