"""
Tests for LLMAnalyzer initialization and foundation components.

This test suite covers the LLMAnalyzer class initialization, cache table creation,
rate limiter setup, and configuration validation as required by Chunk 1.

Test Categories:
- Initialization with/without API key
- Cache table creation and schema validation
- Rate limiter initialization and functionality
- OpenAI client setup
- Performance monitoring setup
- Configuration validation
"""

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from codedocsync.analyzer.llm_analyzer import LLMAnalyzer, TokenBucket
from codedocsync.analyzer.llm_config import LLMConfig


class TestLLMAnalyzerInitialization:
    """Test basic LLMAnalyzer initialization."""

    def test_initialization_with_api_key(self):
        """Test successful initialization with valid API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch(
                "codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"
            ) as mock_openai:
                analyzer = LLMAnalyzer()

                # Check all components were initialized
                assert analyzer.config is not None
                assert analyzer.rate_limiter is not None
                assert analyzer.cache_db_path is not None
                assert analyzer.performance_stats is not None
                assert hasattr(analyzer, "_initialized_at")

                # Check OpenAI client was created
                mock_openai.assert_called_once()

    def test_initialization_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="OPENAI_API_KEY environment variable is required"
            ):
                LLMAnalyzer()

    def test_initialization_without_openai_package(self):
        """Test initialization fails without openai package."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.HAS_OPENAI", False):
                with pytest.raises(ImportError, match="openai package is required"):
                    LLMAnalyzer()

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                custom_config = LLMConfig.create_fast_config()
                analyzer = LLMAnalyzer(config=custom_config)

                assert analyzer.config is custom_config
                assert analyzer.model_id == custom_config.model

    def test_initialization_performance_target(self):
        """Test that initialization completes within performance target (<100ms)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                import time

                start_time = time.time()
                analyzer = LLMAnalyzer()
                end_time = time.time()

                initialization_time_ms = (end_time - start_time) * 1000

                # Should complete within 100ms (being generous with test timing)
                assert (
                    initialization_time_ms < 200
                )  # 200ms to account for test overhead

                # Also check the analyzer's own timing
                summary = analyzer.get_initialization_summary()
                assert summary["initialization_time_ms"] < 200


class TestCacheTableCreation:
    """Test cache database initialization and schema."""

    def test_cache_table_creation(self):
        """Test that cache table is created with correct schema."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Mock the cache directory to use temp directory
                    with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                        analyzer = LLMAnalyzer()

                        # Check that database file was created
                        assert analyzer.cache_db_path.exists()

                        # Check table schema
                        with sqlite3.connect(analyzer.cache_db_path) as conn:
                            cursor = conn.execute(
                                """
                                SELECT name FROM sqlite_master
                                WHERE type='table' AND name='llm_cache'
                            """
                            )
                            assert cursor.fetchone() is not None

                            # Check table structure
                            cursor = conn.execute("PRAGMA table_info(llm_cache)")
                            columns = {row[1]: row[2] for row in cursor.fetchall()}

                            expected_columns = {
                                "cache_key": "TEXT",
                                "request_hash": "TEXT",
                                "response_json": "TEXT",
                                "model": "TEXT",
                                "created_at": "TIMESTAMP",
                                "accessed_at": "TIMESTAMP",
                                "access_count": "INTEGER",
                            }

                            for col_name, col_type in expected_columns.items():
                                assert col_name in columns
                                assert col_type in columns[col_name]

    def test_cache_indexes_creation(self):
        """Test that cache indexes are created."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                        analyzer = LLMAnalyzer()

                        # Check indexes were created
                        with sqlite3.connect(analyzer.cache_db_path) as conn:
                            cursor = conn.execute(
                                """
                                SELECT name FROM sqlite_master
                                WHERE type='index' AND name LIKE 'idx_%'
                            """
                            )
                            indexes = [row[0] for row in cursor.fetchall()]

                            expected_indexes = [
                                "idx_created_at",
                                "idx_model",
                                "idx_request_hash",
                            ]

                            for index_name in expected_indexes:
                                assert index_name in indexes

    def test_cache_wal_mode_enabled(self):
        """Test that WAL mode is enabled for better concurrency."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                        analyzer = LLMAnalyzer()

                        # Check WAL mode is enabled
                        with sqlite3.connect(analyzer.cache_db_path) as conn:
                            cursor = conn.execute("PRAGMA journal_mode")
                            journal_mode = cursor.fetchone()[0]
                            assert journal_mode.lower() == "wal"

    def test_cache_stats_empty_database(self):
        """Test cache stats with empty database."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                        analyzer = LLMAnalyzer()

                        stats = analyzer.get_cache_stats()

                        assert stats["total_entries"] == 0
                        assert stats["valid_entries"] == 0
                        assert stats["total_accesses"] == 0
                        assert stats["database_size_mb"] >= 0
                        assert stats["cache_hit_rate"] == 0.0


class TestRateLimiterInitialization:
    """Test rate limiter setup and functionality."""

    def test_rate_limiter_initialization(self):
        """Test that rate limiter is properly initialized."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                config = LLMConfig(requests_per_second=5.0, burst_size=10)
                analyzer = LLMAnalyzer(config=config)

                assert isinstance(analyzer.rate_limiter, TokenBucket)
                assert analyzer.rate_limiter.rate == 5.0
                assert analyzer.rate_limiter.burst_size == 10
                assert analyzer.rate_limiter.tokens == 10  # Should start full

    @pytest.mark.asyncio
    async def test_rate_limiter_token_acquisition(self):
        """Test rate limiter token acquisition."""
        rate_limiter = TokenBucket(rate=2.0, burst_size=5)

        # Should be able to acquire tokens initially
        assert await rate_limiter.acquire(1) is True
        assert await rate_limiter.acquire(2) is True
        assert await rate_limiter.acquire(2) is True

        # Should fail when bucket is empty
        assert await rate_limiter.acquire(1) is False

    @pytest.mark.asyncio
    async def test_rate_limiter_token_replenishment(self):
        """Test that tokens are replenished over time."""
        rate_limiter = TokenBucket(rate=10.0, burst_size=5)  # 10 tokens per second

        # Drain the bucket
        await rate_limiter.acquire(5)
        assert await rate_limiter.acquire(1) is False

        # Wait a bit and check if tokens are replenished
        await asyncio.sleep(0.2)  # 0.2 seconds should add ~2 tokens
        assert await rate_limiter.acquire(1) is True
        assert await rate_limiter.acquire(1) is True

    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_tokens(self):
        """Test waiting for tokens functionality."""
        rate_limiter = TokenBucket(rate=5.0, burst_size=2)

        # Drain the bucket
        await rate_limiter.acquire(2)

        # Should wait and then succeed
        start_time = asyncio.get_event_loop().time()
        await rate_limiter.wait_for_tokens(1)
        end_time = asyncio.get_event_loop().time()

        # Should have waited at least some time (but not too long for tests)
        wait_time = end_time - start_time
        assert 0.1 < wait_time < 1.0  # Should wait between 0.1 and 1 second


class TestOpenAIClientSetup:
    """Test OpenAI client initialization."""

    def test_openai_client_configuration(self):
        """Test that OpenAI client is configured correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch(
                "codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"
            ) as mock_openai:
                config = LLMConfig(timeout_seconds=45, max_retries=5)
                analyzer = LLMAnalyzer(config=config)

                # Check that OpenAI client was created with correct parameters
                mock_openai.assert_called_once_with(
                    api_key="sk-test123", timeout=45, max_retries=5
                )

                assert analyzer.model_id == config.model

    def test_model_identifier_storage(self):
        """Test that model identifier is stored correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                config = LLMConfig(model="gpt-4o")
                analyzer = LLMAnalyzer(config=config)

                assert analyzer.model_id == "gpt-4o"


class TestPerformanceMonitoringSetup:
    """Test performance monitoring initialization."""

    def test_performance_stats_initialization(self):
        """Test that performance stats are initialized correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                analyzer = LLMAnalyzer()

                expected_stats = {
                    "requests_made": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "total_tokens_used": 0,
                    "total_response_time_ms": 0.0,
                    "errors_encountered": 0,
                }

                for key, expected_value in expected_stats.items():
                    assert key in analyzer.performance_stats
                    assert analyzer.performance_stats[key] == expected_value


class TestConfigurationValidation:
    """Test configuration validation methods."""

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                        analyzer = LLMAnalyzer()

                        validation = analyzer.validate_configuration()

                        assert validation["config_valid"] is True
                        assert validation["api_key_configured"] is True
                        assert validation["cache_accessible"] is True
                        assert validation["openai_client_initialized"] is True
                        assert validation["rate_limiter_configured"] is True
                        assert len(validation["errors"]) == 0

    def test_validate_configuration_missing_api_key(self):
        """Test configuration validation with missing API key."""
        # This test needs to be careful since __init__ will fail without API key
        # We'll test the validation logic after successful initialization
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                analyzer = LLMAnalyzer()

                # Now patch the environment to simulate missing key
                with patch.dict(os.environ, {}, clear=True):
                    validation = analyzer.validate_configuration()
                    assert validation["api_key_configured"] is False

    def test_get_initialization_summary(self):
        """Test initialization summary generation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                config = LLMConfig(requests_per_second=15.0, burst_size=30)
                analyzer = LLMAnalyzer(config=config)

                summary = analyzer.get_initialization_summary()

                expected_fields = [
                    "initialized_at",
                    "initialization_time_ms",
                    "config_summary",
                    "cache_db_path",
                    "model_id",
                    "rate_limit_config",
                    "validation_results",
                ]

                for field in expected_fields:
                    assert field in summary

                assert summary["rate_limit_config"]["requests_per_second"] == 15.0
                assert summary["rate_limit_config"]["burst_size"] == 30
                assert summary["model_id"] == config.model


class TestCacheKeyGeneration:
    """Test cache key generation functionality."""

    def test_cache_key_deterministic(self):
        """Test that cache key generation is deterministic."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                analyzer = LLMAnalyzer()

                # Same inputs should produce same key
                key1 = analyzer._generate_cache_key(
                    "def test(a: int) -> str:",
                    "Test function",
                    ["behavior", "examples"],
                    "gpt-4o-mini",
                )

                key2 = analyzer._generate_cache_key(
                    "def test(a: int) -> str:",
                    "Test function",
                    ["behavior", "examples"],
                    "gpt-4o-mini",
                )

                assert key1 == key2

    def test_cache_key_different_for_different_inputs(self):
        """Test that different inputs produce different cache keys."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                analyzer = LLMAnalyzer()

                base_args = [
                    "def test(a: int) -> str:",
                    "Test function",
                    ["behavior"],
                    "gpt-4o-mini",
                ]

                base_key = analyzer._generate_cache_key(*base_args)

                # Different function signature
                key1 = analyzer._generate_cache_key(
                    "def test(a: str) -> str:", base_args[1], base_args[2], base_args[3]
                )
                assert key1 != base_key

                # Different docstring
                key2 = analyzer._generate_cache_key(
                    base_args[0], "Different docstring", base_args[2], base_args[3]
                )
                assert key2 != base_key

                # Different analysis types
                key3 = analyzer._generate_cache_key(
                    base_args[0], base_args[1], ["examples"], base_args[3]
                )
                assert key3 != base_key

                # Different model
                key4 = analyzer._generate_cache_key(
                    base_args[0], base_args[1], base_args[2], "gpt-4o"
                )
                assert key4 != base_key

    def test_cache_key_order_independent_for_analysis_types(self):
        """Test that analysis types order doesn't affect cache key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            with patch("codedocsync.analyzer.llm_analyzer.openai.AsyncOpenAI"):
                analyzer = LLMAnalyzer()

                key1 = analyzer._generate_cache_key(
                    "def test() -> None:",
                    "Test function",
                    ["behavior", "examples", "edge_cases"],
                    "gpt-4o-mini",
                )

                key2 = analyzer._generate_cache_key(
                    "def test() -> None:",
                    "Test function",
                    ["edge_cases", "behavior", "examples"],
                    "gpt-4o-mini",
                )

                assert key1 == key2
