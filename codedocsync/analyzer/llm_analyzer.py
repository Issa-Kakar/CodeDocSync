"""
LLM Analyzer Foundation - Chunk 1 Implementation

This module provides the foundation for LLM-powered semantic analysis.
Includes class structure, initialization, rate limiting, and cache schema.

Key Requirements (Chunk 1):
- Use openai library directly (not litellm since we're OpenAI-only)
- Implement token bucket rate limiter in __init__
- Cache must use SQLite (create table if not exists)
- Must validate API key exists in environment
- Performance target: Foundation setup in <100ms
"""

import os
import sqlite3
import time
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .llm_config import LLMConfig


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Ensures we don't exceed API rate limits while allowing bursts.
    """

    def __init__(self, rate: float, burst_size: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            burst_size: Maximum burst capacity
        """
        self.rate = rate
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update

            # Add tokens based on time passed
            self.tokens = min(self.burst_size, self.tokens + time_passed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
        """
        while not await self.acquire(tokens):
            # Calculate wait time needed
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(min(wait_time, 1.0))  # Max 1 second waits


class LLMAnalyzer:
    """
    Analyzes code-doc consistency using LLM for semantic understanding.

    This is the foundation implementation (Chunk 1) focusing on:
    - Proper initialization and configuration
    - Rate limiting using token bucket algorithm
    - SQLite cache setup with proper schema
    - OpenAI client initialization with retry wrapper
    - Performance monitoring setup
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize with configuration and OpenAI client.

        Args:
            config: LLM configuration (creates default if None)

        Raises:
            ImportError: If openai package is not available
            ValueError: If configuration is invalid or API key missing
        """
        # Validate OpenAI availability
        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for LLM analysis. "
                "Install with: pip install openai"
            )

        # Set up configuration (this validates API key and other settings)
        self.config = config or LLMConfig()

        # Initialize OpenAI client with retry wrapper
        self._init_openai_client()

        # Set up rate limiter (token bucket algorithm)
        self.rate_limiter = TokenBucket(
            rate=self.config.requests_per_second, burst_size=self.config.burst_size
        )

        # Initialize cache connection
        self.cache_db_path = self._init_cache_database()

        # Set up performance monitoring
        self.performance_stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_used": 0,
            "total_response_time_ms": 0.0,
            "errors_encountered": 0,
        }

        # Track initialization success
        self._initialized_at = time.time()

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client with proper configuration."""
        # Get API key from environment (already validated in config)
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )

        # Store model identifier for API calls
        self.model_id = self.config.model

    def _init_cache_database(self) -> Path:
        """
        Initialize SQLite cache database with proper schema.

        Returns:
            Path to the cache database file
        """
        # Create cache directory
        cache_dir = Path.home() / ".cache" / "codedocsync"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_db_path = cache_dir / "llm_cache.db"

        # Create database schema if it doesn't exist
        with sqlite3.connect(cache_db_path) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")

            # Create main cache table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON llm_cache(created_at)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model ON llm_cache(model)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_hash ON llm_cache(request_hash)
            """
            )

            conn.commit()

        return cache_db_path

    def _generate_cache_key(
        self, function_signature: str, docstring: str, analysis_types: list, model: str
    ) -> str:
        """
        Generate deterministic cache key.

        Cache key generation is critical for cache effectiveness.
        Must be deterministic and collision-resistant.

        Args:
            function_signature: String representation of function signature
            docstring: Raw docstring text
            analysis_types: List of analysis types being performed
            model: Model identifier

        Returns:
            MD5 hash as cache key
        """
        # Create content string with all relevant parameters
        content_parts = [
            function_signature,
            docstring,
            "|".join(sorted(analysis_types)),  # Sort for deterministic ordering
            model,
        ]
        content = "|".join(content_parts)

        # Generate MD5 hash (sufficient for cache keys)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics and performance metrics
        """
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Get basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM llm_cache")
                total_entries = cursor.fetchone()[0]

                # Get valid entries (not expired)
                ttl_seconds = self.config.cache_ttl_days * 24 * 3600
                cutoff_time = time.time() - ttl_seconds
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM llm_cache WHERE created_at > ?",
                    (cutoff_time,),
                )
                valid_entries = cursor.fetchone()[0]

                # Get access statistics
                cursor = conn.execute("SELECT SUM(access_count) FROM llm_cache")
                total_accesses = cursor.fetchone()[0] or 0

                # Get database file size
                db_size_mb = self.cache_db_path.stat().st_size / (1024 * 1024)

                # Calculate hit rate
                total_requests = (
                    self.performance_stats["cache_hits"]
                    + self.performance_stats["cache_misses"]
                )
                hit_rate = (
                    self.performance_stats["cache_hits"] / total_requests
                    if total_requests > 0
                    else 0.0
                )

                return {
                    "total_entries": total_entries,
                    "valid_entries": valid_entries,
                    "expired_entries": total_entries - valid_entries,
                    "total_accesses": total_accesses,
                    "database_size_mb": round(db_size_mb, 2),
                    "cache_hit_rate": round(hit_rate, 3),
                    "cache_file_path": str(self.cache_db_path),
                    **self.performance_stats,
                }

        except Exception as e:
            return {
                "error": f"Failed to get cache stats: {e}",
                "cache_enabled": False,
            }

    def clear_expired_cache_entries(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        ttl_seconds = self.config.cache_ttl_days * 24 * 3600
        cutoff_time = time.time() - ttl_seconds

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM llm_cache WHERE created_at < ?", (cutoff_time,)
                )
                return cursor.rowcount

        except Exception:
            return 0

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration and system state.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "config_valid": True,
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "cache_accessible": False,
            "openai_client_initialized": hasattr(self, "openai_client"),
            "rate_limiter_configured": hasattr(self, "rate_limiter"),
            "errors": [],
        }

        # Test cache accessibility
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("SELECT 1")
            validation_results["cache_accessible"] = True
        except Exception as e:
            validation_results["errors"].append(f"Cache not accessible: {e}")

        # Validate configuration
        try:
            # This will raise if configuration is invalid
            _ = self.config.get_summary()
        except Exception as e:
            validation_results["config_valid"] = False
            validation_results["errors"].append(f"Configuration invalid: {e}")

        return validation_results

    def get_initialization_summary(self) -> Dict[str, Any]:
        """
        Get summary of initialization status and configuration.

        Returns:
            Dictionary with initialization details
        """
        return {
            "initialized_at": self._initialized_at,
            "initialization_time_ms": (time.time() - self._initialized_at) * 1000,
            "config_summary": self.config.get_summary(),
            "cache_db_path": str(self.cache_db_path),
            "model_id": self.model_id,
            "rate_limit_config": {
                "requests_per_second": self.config.requests_per_second,
                "burst_size": self.config.burst_size,
            },
            "validation_results": self.validate_configuration(),
        }


# Factory functions for different use cases
def create_fast_analyzer() -> LLMAnalyzer:
    """Create analyzer optimized for speed."""
    return LLMAnalyzer(LLMConfig.create_fast_config())


def create_balanced_analyzer() -> LLMAnalyzer:
    """Create analyzer with balanced speed/thoroughness."""
    return LLMAnalyzer(LLMConfig.create_balanced_config())


def create_thorough_analyzer() -> LLMAnalyzer:
    """Create analyzer optimized for thoroughness."""
    return LLMAnalyzer(LLMConfig.create_thorough_config())
