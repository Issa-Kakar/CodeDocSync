import json
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ..matcher.semantic_models import FunctionEmbedding

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Two-tier caching for embeddings: in-memory LRU + disk persistence.
    """

    def __init__(
        self, cache_dir: str = ".codedocsync_cache", max_memory_items: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory LRU cache
        self.memory_cache: OrderedDict[str, FunctionEmbedding] = OrderedDict()
        self.max_memory_items = max_memory_items

        # SQLite for disk persistence
        self.db_path = self.cache_dir / "embeddings.db"
        self._init_db()

        # Performance metrics
        self.metrics = {"memory_hits": 0, "disk_hits": 0, "misses": 0, "saves": 0}

    def _init_db(self) -> None:
        """Initialize SQLite database for embeddings."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                function_id TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                model TEXT NOT NULL,
                text_embedded TEXT NOT NULL,
                signature_hash TEXT NOT NULL,
                timestamp REAL NOT NULL,
                hit_count INTEGER DEFAULT 0,
                last_accessed REAL NOT NULL
            )
        """
        )

        # Create indices for performance
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_function_id ON embeddings(function_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signature_hash ON embeddings(signature_hash)
        """
        )

        conn.commit()
        conn.close()

    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        import hashlib

        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(
        self, text: str, model: str, signature_hash: str | None = None
    ) -> FunctionEmbedding | None:
        """
        Get embedding from cache.

        Args:
            text: Text that was embedded
            model: Model used for embedding
            signature_hash: Optional hash to verify function hasn't changed

        Returns:
            FunctionEmbedding if found and valid, None otherwise
        """
        cache_key = self._generate_cache_key(text, model)

        # Check memory cache first
        if cache_key in self.memory_cache:
            embedding = self.memory_cache[cache_key]
            # Move to end (LRU)
            self.memory_cache.move_to_end(cache_key)

            # Verify signature if provided
            if signature_hash and embedding.signature_hash != signature_hash:
                # Function has changed, invalidate cache
                del self.memory_cache[cache_key]
                self._delete_from_disk(cache_key)
                self.metrics["misses"] += 1
                return None

            self.metrics["memory_hits"] += 1
            return embedding

        # Check disk cache
        embedding = self._get_from_disk(cache_key)
        if embedding:
            # Verify signature if provided
            if signature_hash and embedding.signature_hash != signature_hash:
                # Function has changed, invalidate cache
                self._delete_from_disk(cache_key)
                self.metrics["misses"] += 1
                return None

            # Add to memory cache
            self._add_to_memory_cache(cache_key, embedding)
            self.metrics["disk_hits"] += 1
            return embedding

        self.metrics["misses"] += 1
        return None

    def set(self, embedding: FunctionEmbedding) -> None:
        """Save embedding to cache."""
        cache_key = self._generate_cache_key(embedding.text_embedded, embedding.model)

        # Add to memory cache
        self._add_to_memory_cache(cache_key, embedding)

        # Save to disk
        self._save_to_disk(cache_key, embedding)

        self.metrics["saves"] += 1

    def _add_to_memory_cache(self, key: str, embedding: FunctionEmbedding) -> None:
        """Add to memory cache with LRU eviction."""
        # Remove oldest if at capacity
        if len(self.memory_cache) >= self.max_memory_items:
            self.memory_cache.popitem(last=False)

        self.memory_cache[key] = embedding

    def _get_from_disk(self, cache_key: str) -> FunctionEmbedding | None:
        """Load embedding from disk."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT function_id, embedding_json, model, text_embedded,
                       signature_hash, timestamp
                FROM embeddings
                WHERE cache_key = ?
            """,
                (cache_key,),
            )

            row = cursor.fetchone()
            if row:
                # Update access stats
                cursor.execute(
                    """
                    UPDATE embeddings
                    SET hit_count = hit_count + 1,
                        last_accessed = ?
                    WHERE cache_key = ?
                """,
                    (time.time(), cache_key),
                )
                conn.commit()

                # Reconstruct embedding
                embedding = FunctionEmbedding(
                    function_id=row[0],
                    embedding=json.loads(row[1]),
                    model=row[2],
                    text_embedded=row[3],
                    signature_hash=row[4],
                    timestamp=row[5],
                )
                return embedding

        except Exception as e:
            logger.error(f"Failed to load from disk cache: {e}")
        finally:
            conn.close()

        return None

    def _save_to_disk(self, cache_key: str, embedding: FunctionEmbedding) -> None:
        """Save embedding to disk."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (cache_key, function_id, embedding_json, model, text_embedded,
                 signature_hash, timestamp, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    cache_key,
                    embedding.function_id,
                    json.dumps(embedding.embedding),
                    embedding.model,
                    embedding.text_embedded,
                    embedding.signature_hash,
                    embedding.timestamp,
                    time.time(),
                ),
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Failed to save to disk cache: {e}")
        finally:
            conn.close()

    def _delete_from_disk(self, cache_key: str) -> None:
        """Delete embedding from disk."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete from disk: {e}")
        finally:
            conn.close()

    def clear_old_entries(self, max_age_days: int = 30) -> int:
        """Remove old cache entries."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                DELETE FROM embeddings
                WHERE last_accessed < ?
            """,
                (cutoff_time,),
            )

            deleted = cursor.rowcount
            conn.commit()

            logger.info(f"Cleared {deleted} old cache entries")
            return deleted

        except Exception as e:
            logger.error(f"Failed to clear old entries: {e}")
            return 0
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = (
            self.metrics["memory_hits"]
            + self.metrics["disk_hits"]
            + self.metrics["misses"]
        )

        memory_hit_rate = (
            self.metrics["memory_hits"] / total_requests if total_requests > 0 else 0
        )

        overall_hit_rate = (
            (self.metrics["memory_hits"] + self.metrics["disk_hits"]) / total_requests
            if total_requests > 0
            else 0
        )

        return {
            "memory_size": len(self.memory_cache),
            "memory_hit_rate": memory_hit_rate,
            "overall_hit_rate": overall_hit_rate,
            "total_saves": self.metrics["saves"],
            "total_requests": total_requests,
        }
