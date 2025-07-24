"""Tests for embedding cache functionality."""

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

from codedocsync.matcher.semantic_models import FunctionEmbedding
from codedocsync.storage.embedding_cache import EmbeddingCache


class TestEmbeddingCache:
    """Test cases for EmbeddingCache."""

    def test_init_creates_cache_dir(self, temp_cache_dir):
        """Test that initialization creates cache directory."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        assert Path(temp_cache_dir).exists()
        cache.close()

    def test_init_creates_database(self, temp_cache_dir):
        """Test that initialization creates SQLite database."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        db_path = Path(temp_cache_dir) / "embeddings.db"
        assert db_path.exists()

        # Verify table structure
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()

        assert ("embeddings",) in tables
        cache.close()

    def test_get_memory_hit(self, temp_cache_dir, mock_embedding):
        """Test getting embedding from memory cache."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Manually add to memory cache
        cache_key = cache._generate_cache_key(
            mock_embedding.text_embedded, mock_embedding.model
        )
        cache.memory_cache[cache_key] = mock_embedding

        # Get should hit memory cache
        result = cache.get(mock_embedding.text_embedded, mock_embedding.model)

        assert result == mock_embedding
        assert cache.metrics["memory_hits"] == 1
        assert cache.metrics["disk_hits"] == 0
        assert cache.metrics["misses"] == 0
        cache.close()

    def test_get_disk_hit(self, temp_cache_dir, mock_embedding):
        """Test getting embedding from disk cache."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Save to disk only
        cache_key = cache._generate_cache_key(
            mock_embedding.text_embedded, mock_embedding.model
        )
        cache._save_to_disk(cache_key, mock_embedding)

        # Get should hit disk cache
        result = cache.get(mock_embedding.text_embedded, mock_embedding.model)

        assert result.function_id == mock_embedding.function_id
        assert result.embedding == mock_embedding.embedding
        assert cache.metrics["memory_hits"] == 0
        assert cache.metrics["disk_hits"] == 1
        assert cache.metrics["misses"] == 0
        cache.close()

    def test_get_miss(self, temp_cache_dir):
        """Test cache miss."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        result = cache.get("nonexistent text", "test-model")

        assert result is None
        assert cache.metrics["memory_hits"] == 0
        assert cache.metrics["disk_hits"] == 0
        assert cache.metrics["misses"] == 1
        cache.close()

    def test_set_adds_to_both_caches(self, temp_cache_dir, mock_embedding):
        """Test that set adds to both memory and disk cache."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        cache.set(mock_embedding)

        # Check memory cache
        cache_key = cache._generate_cache_key(
            mock_embedding.text_embedded, mock_embedding.model
        )
        assert cache_key in cache.memory_cache

        # Check disk cache
        disk_result = cache._get_from_disk(cache_key)
        assert disk_result is not None
        assert disk_result.function_id == mock_embedding.function_id

        assert cache.metrics["saves"] == 1
        cache.close()

    def test_lru_eviction(self, temp_cache_dir):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir, max_memory_items=3)

        # Add 4 embeddings
        embeddings = []
        for i in range(4):
            embedding = FunctionEmbedding(
                function_id=f"func_{i}",
                embedding=[float(i), float(i + 1)],
                model="test-model",
                text_embedded=f"def func_{i}(): pass",
                signature_hash=f"hash_{i}",
                timestamp=time.time(),
            )
            embeddings.append(embedding)
            cache.set(embedding)
            time.sleep(0.01)  # Ensure different timestamps

        # First embedding should be evicted from memory
        cache_key_0 = cache._generate_cache_key(
            embeddings[0].text_embedded, embeddings[0].model
        )
        assert cache_key_0 not in cache.memory_cache
        assert len(cache.memory_cache) == 3

        # But should still be on disk
        disk_result = cache._get_from_disk(cache_key_0)
        assert disk_result is not None
        cache.close()

    def test_lru_move_to_end(self, temp_cache_dir):
        """Test that accessing an item moves it to end of LRU."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir, max_memory_items=3)

        # Add 3 embeddings
        embeddings = []
        for i in range(3):
            embedding = FunctionEmbedding(
                function_id=f"func_{i}",
                embedding=[float(i)],
                model="test-model",
                text_embedded=f"def func_{i}(): pass",
                signature_hash=f"hash_{i}",
                timestamp=time.time(),
            )
            embeddings.append(embedding)
            cache.set(embedding)

        # Access first embedding
        cache.get(embeddings[0].text_embedded, embeddings[0].model)

        # Add fourth embedding - second should be evicted, not first
        new_embedding = FunctionEmbedding(
            function_id="func_new",
            embedding=[99.0],
            model="test-model",
            text_embedded="def func_new(): pass",
            signature_hash="hash_new",
            timestamp=time.time(),
        )
        cache.set(new_embedding)

        # First should still be in memory (was accessed)
        cache_key_0 = cache._generate_cache_key(
            embeddings[0].text_embedded, embeddings[0].model
        )
        assert cache_key_0 in cache.memory_cache

        # Second should be evicted
        cache_key_1 = cache._generate_cache_key(
            embeddings[1].text_embedded, embeddings[1].model
        )
        assert cache_key_1 not in cache.memory_cache
        cache.close()

    def test_signature_mismatch_invalidates_cache(self, temp_cache_dir, mock_embedding):
        """Test that signature mismatch invalidates cache entry."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Set embedding
        cache.set(mock_embedding)

        # Try to get with different signature
        result = cache.get(
            mock_embedding.text_embedded,
            mock_embedding.model,
            signature_hash="different_hash",
        )

        assert result is None
        assert cache.metrics["misses"] == 1

        # Original should be deleted
        cache_key = cache._generate_cache_key(
            mock_embedding.text_embedded, mock_embedding.model
        )
        assert cache_key not in cache.memory_cache
        cache.close()

    def test_signature_match_returns_cached(self, temp_cache_dir, mock_embedding):
        """Test that matching signature returns cached value."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Set embedding
        cache.set(mock_embedding)

        # Get with matching signature
        result = cache.get(
            mock_embedding.text_embedded,
            mock_embedding.model,
            signature_hash=mock_embedding.signature_hash,
        )

        assert result == mock_embedding
        assert cache.metrics["memory_hits"] == 1
        cache.close()

    def test_save_to_disk(self, temp_cache_dir, mock_embedding):
        """Test saving embedding to disk."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        cache_key = cache._generate_cache_key(
            mock_embedding.text_embedded, mock_embedding.model
        )

        cache._save_to_disk(cache_key, mock_embedding)

        # Verify in database
        conn = sqlite3.connect(str(cache.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM embeddings WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[1] == mock_embedding.function_id  # function_id
        assert json.loads(row[2]) == mock_embedding.embedding  # embedding_json
        cache.close()

    def test_load_from_disk(self, temp_cache_dir, mock_embedding):
        """Test loading embedding from disk."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        cache_key = cache._generate_cache_key(
            mock_embedding.text_embedded, mock_embedding.model
        )

        # Save first
        cache._save_to_disk(cache_key, mock_embedding)

        # Load back
        result = cache._get_from_disk(cache_key)

        assert result is not None
        assert result.function_id == mock_embedding.function_id
        assert result.embedding == mock_embedding.embedding
        assert result.model == mock_embedding.model
        cache.close()

    def test_delete_from_disk(self, temp_cache_dir, mock_embedding):
        """Test deleting embedding from disk."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        cache_key = cache._generate_cache_key(
            mock_embedding.text_embedded, mock_embedding.model
        )

        # Save and verify it exists
        cache._save_to_disk(cache_key, mock_embedding)
        assert cache._get_from_disk(cache_key) is not None

        # Delete
        cache._delete_from_disk(cache_key)

        # Verify deleted
        assert cache._get_from_disk(cache_key) is None
        cache.close()

    def test_clear_old_entries(self, temp_cache_dir):
        """Test clearing old cache entries."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Add old and new embeddings
        old_time = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
        new_time = time.time()

        # Manually insert old entry
        conn = sqlite3.connect(str(cache.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO embeddings
            (cache_key, function_id, embedding_json, model, text_embedded,
             signature_hash, timestamp, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old_key",
                "old_func",
                "[1,2]",
                "model",
                "text",
                "hash",
                old_time,
                old_time,
            ),
        )
        conn.commit()
        conn.close()

        # Add new entry
        new_embedding = FunctionEmbedding(
            function_id="new_func",
            embedding=[3, 4],
            model="model",
            text_embedded="new text",
            signature_hash="new_hash",
            timestamp=new_time,
        )
        cache.set(new_embedding)

        # Clear old entries
        deleted = cache.clear_old_entries(max_age_days=30)

        assert deleted == 1

        # Verify old is gone, new remains
        assert cache._get_from_disk("old_key") is None
        new_key = cache._generate_cache_key(
            new_embedding.text_embedded, new_embedding.model
        )
        assert cache._get_from_disk(new_key) is not None
        cache.close()

    def test_get_stats(self, temp_cache_dir, mock_embedding):
        """Test getting cache statistics."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Generate some activity
        cache.set(mock_embedding)
        cache.get(mock_embedding.text_embedded, mock_embedding.model)  # Memory hit
        cache.get("nonexistent", "model")  # Miss

        stats = cache.get_stats()

        assert stats["memory_size"] == 1
        assert stats["memory_hit_rate"] == 0.5  # 1 hit out of 2 requests
        assert stats["overall_hit_rate"] == 0.5
        assert stats["total_saves"] == 1
        assert stats["total_requests"] == 2
        cache.close()

    def test_close(self, temp_cache_dir, mock_embedding):
        """Test closing cache clears memory."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Add to cache
        cache.set(mock_embedding)
        assert len(cache.memory_cache) == 1

        # Close
        cache.close()

        # Memory should be cleared
        assert len(cache.memory_cache) == 0

    def test_error_handling_disk_operations(
        self, temp_cache_dir, mock_embedding, caplog
    ):
        """Test error handling in disk operations."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Mock database error
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            # Should handle error gracefully
            result = cache._get_from_disk("any_key")
            assert result is None
            assert "Failed to load from disk cache" in caplog.text

            # Save should also handle error
            cache._save_to_disk("any_key", mock_embedding)
            assert "Failed to save to disk cache" in caplog.text

        cache.close()
