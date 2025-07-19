import pytest
import tempfile
import time
from pathlib import Path

from codedocsync.storage.embedding_cache import EmbeddingCache
from codedocsync.matcher.semantic_models import FunctionEmbedding


class TestEmbeddingCache:
    """Test two-tier embedding caching functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding for testing."""
        return FunctionEmbedding(
            function_id="module.test_function",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            model="test-model",
            text_embedded="def test_function(): pass",
            timestamp=time.time(),
            signature_hash="abc123",
        )

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create cache instance with temporary directory."""
        return EmbeddingCache(cache_dir=temp_cache_dir, max_memory_items=3)

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization and database setup."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Check directory creation
        assert Path(temp_cache_dir).exists()

        # Check database file creation
        db_path = Path(temp_cache_dir) / "embeddings.db"
        assert db_path.exists()

        # Check initial metrics
        stats = cache.get_stats()
        assert stats["memory_size"] == 0
        assert stats["total_requests"] == 0

    def test_generate_cache_key(self, cache):
        """Test cache key generation."""
        key1 = cache._generate_cache_key("test text", "model1")
        key2 = cache._generate_cache_key("test text", "model1")
        key3 = cache._generate_cache_key("test text", "model2")
        key4 = cache._generate_cache_key("different text", "model1")

        # Same input should produce same key
        assert key1 == key2

        # Different inputs should produce different keys
        assert key1 != key3
        assert key1 != key4

        # Keys should be hex strings
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA256 hex

    def test_memory_cache_basic_operations(self, cache, sample_embedding):
        """Test basic memory cache operations."""
        # Initially empty
        result = cache.get("test text", "test-model")
        assert result is None
        assert cache.metrics["misses"] == 1

        # Add to cache
        cache.set(sample_embedding)
        assert cache.metrics["saves"] == 1

        # Retrieve from memory cache
        result = cache.get("def test_function(): pass", "test-model")
        assert result is not None
        assert result.function_id == "module.test_function"
        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert cache.metrics["memory_hits"] == 1

    def test_memory_cache_lru_eviction(self, temp_cache_dir, sample_embedding):
        """Test LRU eviction in memory cache."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir, max_memory_items=2)

        # Add 3 embeddings (should evict first one)
        embedding1 = FunctionEmbedding(
            function_id="module.func1",
            embedding=[0.1],
            model="test-model",
            text_embedded="func1",
            timestamp=time.time(),
            signature_hash="hash1",
        )
        embedding2 = FunctionEmbedding(
            function_id="module.func2",
            embedding=[0.2],
            model="test-model",
            text_embedded="func2",
            timestamp=time.time(),
            signature_hash="hash2",
        )
        embedding3 = FunctionEmbedding(
            function_id="module.func3",
            embedding=[0.3],
            model="test-model",
            text_embedded="func3",
            timestamp=time.time(),
            signature_hash="hash3",
        )

        cache.set(embedding1)
        cache.set(embedding2)
        cache.set(embedding3)

        # Memory cache should only have 2 items
        assert len(cache.memory_cache) == 2

        # First embedding should be evicted from memory (but still on disk)
        result1 = cache.get("func1", "test-model")
        assert result1 is not None  # Found on disk
        assert cache.metrics["disk_hits"] == 1

        # Second and third should be in memory
        result2 = cache.get("func2", "test-model")
        result3 = cache.get("func3", "test-model")
        assert result2 is not None
        assert result3 is not None

    def test_signature_hash_validation(self, cache, sample_embedding):
        """Test signature hash validation for cache invalidation."""
        # Add to cache
        cache.set(sample_embedding)

        # Retrieve with correct signature hash
        result = cache.get("def test_function(): pass", "test-model", "abc123")
        assert result is not None
        assert cache.metrics["memory_hits"] == 1

        # Retrieve with incorrect signature hash (function changed)
        result = cache.get("def test_function(): pass", "test-model", "different_hash")
        assert result is None
        assert cache.metrics["misses"] == 1

        # Original embedding should be removed from cache
        result = cache.get("def test_function(): pass", "test-model", "abc123")
        assert result is None  # Should be invalidated

    def test_disk_persistence(self, temp_cache_dir, sample_embedding):
        """Test disk persistence across cache instances."""
        # Create cache and add embedding
        cache1 = EmbeddingCache(cache_dir=temp_cache_dir)
        cache1.set(sample_embedding)

        # Create new cache instance (simulating restart)
        cache2 = EmbeddingCache(cache_dir=temp_cache_dir)

        # Should find embedding on disk
        result = cache2.get("def test_function(): pass", "test-model")
        assert result is not None
        assert result.function_id == "module.test_function"
        assert cache2.metrics["disk_hits"] == 1

    def test_disk_cache_hit_count_tracking(self, cache, sample_embedding):
        """Test hit count tracking in disk cache."""
        # Add to cache
        cache.set(sample_embedding)

        # Clear memory cache to force disk access
        cache.memory_cache.clear()

        # Access multiple times
        for _ in range(3):
            result = cache.get("def test_function(): pass", "test-model")
            assert result is not None

        assert cache.metrics["disk_hits"] == 3

    def test_clear_old_entries(self, cache):
        """Test clearing old cache entries."""
        # Create old embedding
        old_embedding = FunctionEmbedding(
            function_id="module.old_func",
            embedding=[0.1],
            model="test-model",
            text_embedded="old func",
            timestamp=time.time() - (40 * 24 * 60 * 60),  # 40 days ago
            signature_hash="old_hash",
        )

        # Create recent embedding
        recent_embedding = FunctionEmbedding(
            function_id="module.recent_func",
            embedding=[0.2],
            model="test-model",
            text_embedded="recent func",
            timestamp=time.time(),
            signature_hash="recent_hash",
        )

        cache.set(old_embedding)
        cache.set(recent_embedding)

        # Clear entries older than 30 days
        deleted_count = cache.clear_old_entries(max_age_days=30)
        assert deleted_count == 1

        # Old embedding should be gone
        result = cache.get("old func", "test-model")
        assert result is None

        # Recent embedding should remain
        result = cache.get("recent func", "test-model")
        assert result is not None

    def test_cache_statistics(self, cache, sample_embedding):
        """Test comprehensive cache statistics."""
        # Initial stats
        stats = cache.get_stats()
        assert stats["memory_size"] == 0
        assert stats["memory_hit_rate"] == 0
        assert stats["overall_hit_rate"] == 0
        assert stats["total_saves"] == 0
        assert stats["total_requests"] == 0

        # Add embedding
        cache.set(sample_embedding)

        # Memory hit
        cache.get("def test_function(): pass", "test-model")

        # Memory miss
        cache.get("nonexistent", "test-model")

        # Clear memory and force disk hit
        cache.memory_cache.clear()
        cache.get("def test_function(): pass", "test-model")

        stats = cache.get_stats()
        assert stats["memory_size"] == 1  # Reloaded to memory
        assert stats["memory_hit_rate"] == 0.5  # 1 hit / 2 requests
        assert stats["overall_hit_rate"] == 2 / 3  # 2 hits / 3 requests
        assert stats["total_saves"] == 1
        assert stats["total_requests"] == 3

    def test_concurrent_access_simulation(self, cache):
        """Test cache behavior under simulated concurrent access."""
        embeddings = []
        for i in range(10):
            embedding = FunctionEmbedding(
                function_id=f"module.func_{i}",
                embedding=[float(i)],
                model="test-model",
                text_embedded=f"func {i}",
                timestamp=time.time(),
                signature_hash=f"hash_{i}",
            )
            embeddings.append(embedding)
            cache.set(embedding)

        # Access all embeddings multiple times
        for _ in range(3):
            for i in range(10):
                result = cache.get(f"func {i}", "test-model")
                assert result is not None
                assert result.function_id == f"module.func_{i}"

        stats = cache.get_stats()
        assert stats["total_requests"] >= 30

    def test_cache_with_different_models(self, cache):
        """Test caching with different embedding models."""
        # Same text, different models
        text = "def test_function(): pass"

        embedding1 = FunctionEmbedding(
            function_id="module.test_function",
            embedding=[0.1, 0.2],
            model="model1",
            text_embedded=text,
            timestamp=time.time(),
            signature_hash="hash1",
        )

        embedding2 = FunctionEmbedding(
            function_id="module.test_function",
            embedding=[0.3, 0.4],
            model="model2",
            text_embedded=text,
            timestamp=time.time(),
            signature_hash="hash1",
        )

        cache.set(embedding1)
        cache.set(embedding2)

        # Should retrieve different embeddings for different models
        result1 = cache.get(text, "model1")
        result2 = cache.get(text, "model2")

        assert result1 is not None
        assert result2 is not None
        assert result1.embedding == [0.1, 0.2]
        assert result2.embedding == [0.3, 0.4]

    def test_error_handling_corrupted_disk_data(self, cache, temp_cache_dir):
        """Test error handling when disk data is corrupted."""
        # Manually insert corrupted data into database
        import sqlite3

        db_path = Path(temp_cache_dir) / "embeddings.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Insert invalid JSON
        cursor.execute(
            """
            INSERT INTO embeddings
            (cache_key, function_id, embedding_json, model, text_embedded,
             signature_hash, timestamp, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "corrupted_key",
                "module.test",
                "invalid json data",
                "test-model",
                "test text",
                "hash",
                time.time(),
                time.time(),
            ),
        )
        conn.commit()
        conn.close()

        # Should handle corrupted data gracefully
        result = cache._get_from_disk("corrupted_key")
        assert result is None

    def test_large_embedding_handling(self, cache):
        """Test handling of large embeddings."""
        # Create large embedding (typical for large models)
        large_embedding = FunctionEmbedding(
            function_id="module.large_func",
            embedding=[0.1] * 3072,  # OpenAI large model size
            model="text-embedding-3-large",
            text_embedded="def large_function(): pass",
            timestamp=time.time(),
            signature_hash="large_hash",
        )

        cache.set(large_embedding)

        # Should handle large embedding correctly
        result = cache.get("def large_function(): pass", "text-embedding-3-large")
        assert result is not None
        assert len(result.embedding) == 3072
        assert result.model == "text-embedding-3-large"
