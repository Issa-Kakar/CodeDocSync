"""
Advanced Cache Management for LLM Analyzer - Chunk 4 Implementation

This module provides high-performance caching for LLM responses with:
- Connection pooling for concurrent access
- WAL mode for better read performance
- Automatic cleanup and size management
- Performance optimization strategies
- Cache warming for high-value functions

Performance Requirements:
- Cache operations should be <10ms
- Support concurrent access from multiple threads
- Automatic cleanup when cache exceeds 1GB
- >90% cache hit rate for frequently accessed functions
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
import zlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any

from ..parser.ast_parser import ParsedFunction
from .llm_models import LLMAnalysisResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    total_entries: int
    valid_entries: int
    expired_entries: int
    cache_size_mb: float
    hit_rate: float
    average_response_time_ms: float
    most_accessed_keys: list[tuple[str, int]]
    cache_efficiency_score: float


class ConnectionPool:
    """
    SQLite connection pool for concurrent access.

    Manages a pool of database connections to avoid blocking
    and improve performance for concurrent operations.
    """

    def __init__(self, db_path: str, max_connections: int = 5) -> None:
        """Initialize connection pool."""
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._initialized = False

    def _init_connection(self) -> sqlite3.Connection:
        """Create and configure a new database connection."""
        conn = sqlite3.Connection(
            self.db_path,
            timeout=30.0,
            isolation_level=None,  # Autocommit mode
            check_same_thread=False,
        )

        # Configure for performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")

        return conn

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[sqlite3.Connection, None]:
        """Get a connection from the pool."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    # Initialize pool with connections
                    for _ in range(self.max_connections):
                        self._pool.put(self._init_connection())
                    self._initialized = True

        # Get connection from pool
        conn = await asyncio.get_event_loop().run_in_executor(None, self._pool.get)

        try:
            yield conn
        finally:
            # Return connection to pool
            await asyncio.get_event_loop().run_in_executor(None, self._pool.put, conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()


class LLMCache:
    """
    High-performance cache for LLM responses.

    Features:
    - Connection pooling for concurrent access
    - Automatic compression for large responses
    - Intelligent expiration based on file modification times
    - Cache warming for high-value functions
    - Performance monitoring and optimization
    """

    def __init__(self, db_path: str = ".codedocsync_cache/llm.db") -> None:
        """
        Initialize with SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection pool (max 5 connections)
        self.connection_pool = ConnectionPool(str(self.db_path), max_connections=5)

        # Performance tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_response_time": 0.0,
            "last_cleanup": time.time(),
        }

        # Compression threshold (1KB)
        self.compression_threshold = 1024

        # Initialize database schema
        asyncio.create_task(self._init_database())

    async def _init_database(self) -> None:
        """Initialize database schema with optimizations."""
        async with self.connection_pool.get_connection() as conn:
            # Create main cache table with additional columns
            await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    response_size INTEGER DEFAULT 0,
                    is_compressed INTEGER DEFAULT 0,
                    function_complexity INTEGER DEFAULT 0,
                    file_path TEXT,
                    analysis_types TEXT
                )
                """,
            )

            # Create performance indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_created_at ON llm_cache(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_accessed_at ON llm_cache(accessed_at)",
                "CREATE INDEX IF NOT EXISTS idx_model ON llm_cache(model)",
                "CREATE INDEX IF NOT EXISTS idx_request_hash ON llm_cache(request_hash)",
                "CREATE INDEX IF NOT EXISTS idx_access_count ON llm_cache(access_count)",
                "CREATE INDEX IF NOT EXISTS idx_file_path ON llm_cache(file_path)",
                "CREATE INDEX IF NOT EXISTS idx_complexity ON llm_cache(function_complexity)",
            ]

            for index_sql in indexes:
                await asyncio.get_event_loop().run_in_executor(
                    None, conn.execute, index_sql
                )

    async def get(self, cache_key: str) -> LLMAnalysisResponse | None:
        """
        Get cached response with freshness check.

        Args:
            cache_key: Unique cache key for the request

        Returns:
            Cached response if found and valid, None otherwise
        """
        start_time = time.time()

        try:
            async with self.connection_pool.get_connection() as conn:
                # Check if exists and not expired
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    SELECT response_json, is_compressed, created_at
                    FROM llm_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )

                row = await asyncio.get_event_loop().run_in_executor(
                    None, cursor.fetchone
                )

                if not row:
                    self.stats["misses"] += 1
                    return None

                response_json, is_compressed, created_at = row

                # Check if expired (7 days default)
                if time.time() - created_at > (7 * 24 * 3600):
                    self.stats["misses"] += 1
                    return None

                # Update accessed_at timestamp and increment access_count
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    UPDATE llm_cache
                    SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )

                # Decompress if needed
                if is_compressed:
                    # response_json is stored as base64-encoded compressed data
                    import base64

                    compressed_data = base64.b64decode(response_json)
                    response_json = zlib.decompress(compressed_data).decode("utf-8")

                # Deserialize JSON to response object
                response_data = json.loads(response_json)
                response = LLMAnalysisResponse(**response_data)
                response.cache_hit = True

                self.stats["hits"] += 1
                response_time = (time.time() - start_time) * 1000
                self.stats["total_response_time"] += response_time

                return response

        except Exception as e:
            logger.warning(f"Cache get error for key {cache_key}: {e}")
            self.stats["misses"] += 1
            return None

    async def set(
        self,
        cache_key: str,
        response: LLMAnalysisResponse,
        ttl_days: int = 7,
        function_complexity: int = 0,
        file_path: str = "",
        analysis_types: list[str] | None = None,
    ) -> None:
        """
        Cache response with TTL and metadata.

        Args:
            cache_key: Unique cache key
            response: LLM analysis response to cache
            ttl_days: Time to live in days
            function_complexity: Complexity score for prioritization
            file_path: Source file path for invalidation
            analysis_types: Types of analysis performed
        """
        try:
            # Serialize to JSON
            response_dict = asdict(response)
            response_dict["cache_hit"] = False  # Reset cache_hit flag
            response_json = json.dumps(response_dict)

            # Compress if large (>1KB)
            is_compressed = False
            if len(response_json) > self.compression_threshold:
                # Compress and base64 encode for safe storage as TEXT
                import base64

                compressed_data = zlib.compress(response_json.encode("utf-8"))
                response_json = base64.b64encode(compressed_data).decode("ascii")
                is_compressed = True

            analysis_types_str = ",".join(analysis_types) if analysis_types else ""

            async with self.connection_pool.get_connection() as conn:
                # Store with metadata
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    INSERT OR REPLACE INTO llm_cache
                    (cache_key, request_hash, response_json, model, response_size,
                     is_compressed, function_complexity, file_path, analysis_types)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        cache_key[:16],  # Use first 16 chars as request hash
                        response_json,
                        response.model_used,
                        len(response_json),
                        1 if is_compressed else 0,
                        function_complexity,
                        file_path,
                        analysis_types_str,
                    ),
                )

            # Trigger cleanup if cache getting large
            await self._maybe_cleanup()

        except Exception as e:
            logger.warning(f"Cache set error for key {cache_key}: {e}")

    async def _maybe_cleanup(self) -> None:
        """Trigger cleanup if cache size exceeds limits."""
        # Only check every 5 minutes
        if time.time() - self.stats["last_cleanup"] < 300:
            return

        try:
            cache_size = self.db_path.stat().st_size / (1024 * 1024)  # MB

            if cache_size > 1024:  # 1GB limit
                await self._cleanup_cache()
                self.stats["last_cleanup"] = time.time()

        except Exception as e:
            logger.warning(f"Cache cleanup check failed: {e}")

    async def _cleanup_cache(self) -> None:
        """Clean up old and unused cache entries."""
        async with self.connection_pool.get_connection() as conn:
            # Delete expired entries first
            await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                """
                DELETE FROM llm_cache
                WHERE created_at < datetime('now', '-7 days')
                """,
            )

            # If still too large, delete least recently used entries
            cursor = await asyncio.get_event_loop().run_in_executor(
                None, conn.execute, "SELECT COUNT(*) FROM llm_cache"
            )
            count = (
                await asyncio.get_event_loop().run_in_executor(None, cursor.fetchone)
            )[0]

            if count > 10000:  # Limit to 10k entries
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    DELETE FROM llm_cache
                    WHERE cache_key IN (
                        SELECT cache_key FROM llm_cache
                        ORDER BY access_count ASC, accessed_at ASC
                        LIMIT ?
                    )
                    """,
                    (count - 8000,),  # Keep 8k entries
                )

            # Vacuum to reclaim space
            await asyncio.get_event_loop().run_in_executor(None, conn.execute, "VACUUM")

    async def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        try:
            async with self.connection_pool.get_connection() as conn:
                # Basic counts
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None, conn.execute, "SELECT COUNT(*) FROM llm_cache"
                )
                total_entries = (
                    await asyncio.get_event_loop().run_in_executor(
                        None, cursor.fetchone
                    )
                )[0]

                # Valid entries (not expired)
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    SELECT COUNT(*) FROM llm_cache
                    WHERE created_at > datetime('now', '-7 days')
                    """,
                )
                valid_entries = (
                    await asyncio.get_event_loop().run_in_executor(
                        None, cursor.fetchone
                    )
                )[0]

                # Most accessed keys
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    SELECT cache_key, access_count FROM llm_cache
                    ORDER BY access_count DESC LIMIT 10
                    """,
                )
                most_accessed = await asyncio.get_event_loop().run_in_executor(
                    None, cursor.fetchall
                )

                # Cache size
                cache_size_mb = self.db_path.stat().st_size / (1024 * 1024)

                # Calculate hit rate
                total_requests = self.stats["hits"] + self.stats["misses"]
                hit_rate = (
                    self.stats["hits"] / total_requests if total_requests > 0 else 0.0
                )

                # Average response time
                avg_response_time = (
                    self.stats["total_response_time"] / self.stats["hits"]
                    if self.stats["hits"] > 0
                    else 0.0
                )

                # Cache efficiency score (combination of hit rate and response time)
                efficiency_score = (
                    hit_rate * 0.7 + (1.0 / (1.0 + avg_response_time / 100)) * 0.3
                )

                return CacheStats(
                    total_entries=total_entries,
                    valid_entries=valid_entries,
                    expired_entries=total_entries - valid_entries,
                    cache_size_mb=cache_size_mb,
                    hit_rate=hit_rate,
                    average_response_time_ms=avg_response_time,
                    most_accessed_keys=most_accessed,
                    cache_efficiency_score=efficiency_score,
                )

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return CacheStats(0, 0, 0, 0.0, 0.0, 0.0, [], 0.0)

    async def invalidate_by_file(self, file_path: str) -> int:
        """
        Invalidate cache entries for a specific file.

        Args:
            file_path: Path to the file whose cache entries should be invalidated

        Returns:
            Number of entries invalidated
        """
        try:
            async with self.connection_pool.get_connection() as conn:
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    "DELETE FROM llm_cache WHERE file_path = ?",
                    (file_path,),
                )
                return cursor.rowcount

        except Exception as e:
            logger.warning(f"Error invalidating cache for file {file_path}: {e}")
            return 0

    async def warm_cache(
        self, functions: list[ParsedFunction], max_concurrent: int = 5
    ) -> dict[str, Any]:
        """
        Pre-populate cache for high-value functions.

        Identifies functions that would benefit from caching:
        - Public API functions (no leading underscore)
        - Frequently changed functions
        - Complex functions (>50 lines)

        Args:
            functions: List of functions to potentially cache
            max_concurrent: Maximum concurrent warming operations

        Returns:
            Statistics about the warming operation
        """
        # This is a placeholder - actual implementation would require
        # integration with the LLM analyzer to generate responses
        logger.info(f"Cache warming requested for {len(functions)} functions")

        # Identify high-value targets
        high_value_functions = []
        for func in functions:
            if self._is_high_value_function(func):
                high_value_functions.append(func)

        return {
            "total_functions": len(functions),
            "high_value_functions": len(high_value_functions),
            "warming_completed": False,  # Would be True after actual warming
            "cache_entries_created": 0,
            "warming_time_ms": 0.0,
        }

    def _is_high_value_function(self, func: ParsedFunction) -> bool:
        """Determine if a function is high-value for caching."""
        # Public API functions (no leading underscore)
        if not func.signature.name.startswith("_"):
            return True

        # Complex functions based on line count estimation
        if hasattr(func, "line_count") and func.line_count > 50:
            return True

        # Functions with complex parameter lists
        if len(func.signature.parameters) > 5:
            return True

        return False

    async def close(self) -> None:
        """Close all database connections."""
        self.connection_pool.close_all()
