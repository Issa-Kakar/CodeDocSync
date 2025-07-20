import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from codedocsync.matcher.semantic_models import FunctionEmbedding
from codedocsync.matcher.semantic_optimizer import SemanticOptimizer
from codedocsync.parser import FunctionSignature, ParsedFunction


class TestSemanticOptimizer:
    """Comprehensive tests for SemanticOptimizer performance optimization functionality."""

    @pytest.fixture
    def sample_functions(self):
        """Generate sample functions for testing."""
        functions = []
        for i in range(100):
            func = ParsedFunction(
                signature=FunctionSignature(
                    name=f"function_{i}", parameters=[], return_type="None"
                ),
                docstring=None,
                file_path=f"module_{i % 10}.py",
                line_number=i * 10,
                end_line_number=i * 10 + 5,
                source_code=f"def function_{i}(): pass",
            )
            functions.append(func)
        return functions

    @pytest.fixture
    def mock_optimizer(self):
        """Create optimizer with mocked psutil for consistent testing."""
        with patch("psutil.Process") as mock_process:
            # Mock memory info to return predictable values
            mock_memory = Mock()
            mock_memory.rss = 1024 * 1024 * 1024  # 1GB in bytes
            mock_process.return_value.memory_info.return_value = mock_memory
            mock_process.return_value.cpu_percent.return_value = 25.0

            optimizer = SemanticOptimizer(max_memory_mb=500)
            yield optimizer
            optimizer.cleanup()

    def test_optimizer_initialization(self, mock_optimizer):
        """Test SemanticOptimizer initialization."""
        assert mock_optimizer.max_memory_mb == 500
        assert isinstance(mock_optimizer.executor, ThreadPoolExecutor)
        assert mock_optimizer.initial_memory == 1024  # 1GB / 1024 / 1024

        # Check initial stats
        stats = mock_optimizer.get_stats()
        assert stats["gc_triggered_count"] == 0
        assert stats["batches_optimized"] == 0
        assert stats["parallel_operations"] == 0

    def test_batch_size_optimization_high_memory(self, mock_optimizer):
        """Test batch size optimization with high memory availability."""
        # Mock high memory available (current = initial, so 400MB available)
        with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
            mock_memory.return_value.rss = (
                mock_optimizer.initial_memory * 1024 * 1024
            )  # Same as initial

            batch_size = mock_optimizer.optimize_batch_size(1000)
            assert batch_size == 100  # Should use base batch size
            assert mock_optimizer.stats["batches_optimized"] == 1

    def test_batch_size_optimization_medium_memory(self, mock_optimizer):
        """Test batch size optimization with medium memory availability."""
        # Mock medium memory (350MB used = 150MB available)
        with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
            mock_memory.return_value.rss = (
                (mock_optimizer.initial_memory + 350) * 1024 * 1024
            )

            batch_size = mock_optimizer.optimize_batch_size(1000)
            assert batch_size == 50  # Should use reduced batch size
            assert mock_optimizer.stats["batches_optimized"] == 1

    def test_batch_size_optimization_low_memory(self, mock_optimizer):
        """Test batch size optimization with low memory availability."""
        # Mock low memory (450MB used = 50MB available)
        with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
            mock_memory.return_value.rss = (
                (mock_optimizer.initial_memory + 450) * 1024 * 1024
            )

            batch_size = mock_optimizer.optimize_batch_size(1000)
            assert batch_size == 25  # Should use minimum batch size
            assert mock_optimizer.stats["batches_optimized"] == 1

    def test_batch_size_respects_function_count(self, mock_optimizer):
        """Test that batch size never exceeds total function count."""
        batch_size = mock_optimizer.optimize_batch_size(10)
        assert batch_size <= 10

    @pytest.mark.asyncio
    async def test_parallel_embedding_generation_success(
        self, mock_optimizer, sample_functions
    ):
        """Test successful parallel embedding generation."""
        # Create mock embedding results
        mock_embeddings = [
            FunctionEmbedding(
                function_id=f"test.function_{i}",
                embedding=[0.1] * 384,
                model="test-model",
                text_embedded=f"def function_{i}(): pass",
                timestamp=time.time(),
                signature_hash=f"hash_{i}",
            )
            for i in range(5)
        ]

        # Mock generator function
        async def mock_generator(batch):
            await asyncio.sleep(0.01)  # Simulate API call
            return mock_embeddings[: len(batch)]

        # Create batches
        batches = [sample_functions[i : i + 5] for i in range(0, 15, 5)]

        # Test parallel generation
        results = await mock_optimizer.parallel_embedding_generation(
            batches, mock_generator
        )

        assert len(results) == 15  # 3 batches * 5 embeddings each
        assert mock_optimizer.stats["parallel_operations"] == 1
        assert mock_optimizer.stats["total_optimization_time"] > 0

    @pytest.mark.asyncio
    async def test_parallel_embedding_generation_with_errors(
        self, mock_optimizer, sample_functions
    ):
        """Test parallel embedding generation with some batch failures."""
        mock_embeddings = [
            FunctionEmbedding(
                function_id=f"test.function_{i}",
                embedding=[0.1] * 384,
                model="test-model",
                text_embedded=f"def function_{i}(): pass",
                timestamp=time.time(),
                signature_hash=f"hash_{i}",
            )
            for i in range(5)
        ]

        call_count = 0

        async def mock_generator_with_errors(batch):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail the second batch
                raise Exception("API error")
            return mock_embeddings[: len(batch)]

        batches = [sample_functions[i : i + 5] for i in range(0, 15, 5)]

        results = await mock_optimizer.parallel_embedding_generation(
            batches, mock_generator_with_errors
        )

        # Should get results from 2 successful batches (10 embeddings)
        assert len(results) == 10
        assert mock_optimizer.stats["parallel_operations"] == 1

    def test_should_trigger_gc_below_threshold(self, mock_optimizer):
        """Test GC trigger when memory usage is below threshold."""
        # Mock memory usage at 70% (350MB used)
        with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
            mock_memory.return_value.rss = (
                (mock_optimizer.initial_memory + 350) * 1024 * 1024
            )

            should_trigger = mock_optimizer.should_trigger_gc()
            assert not should_trigger
            assert mock_optimizer.stats["gc_triggered"] == 0

    def test_should_trigger_gc_above_threshold(self, mock_optimizer):
        """Test GC trigger when memory usage exceeds threshold."""
        # Mock memory usage at 90% (450MB used)
        with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
            mock_memory.return_value.rss = (
                (mock_optimizer.initial_memory + 450) * 1024 * 1024
            )

            with patch("gc.collect") as mock_gc:
                should_trigger = mock_optimizer.should_trigger_gc()
                assert should_trigger
                assert mock_optimizer.stats["gc_triggered"] == 1
                mock_gc.assert_called_once()

    def test_optimize_vector_store_queries_with_batch_support(self, mock_optimizer):
        """Test vector store query optimization with batch support."""
        # Mock vector store with batch support
        mock_vector_store = Mock()
        mock_vector_store.search_similar_batch.return_value = [
            [("id1", 0.9, {"test": "meta1"})],
            [("id2", 0.8, {"test": "meta2"})],
        ]

        queries = [[0.1] * 384, [0.2] * 384]

        results = mock_optimizer.optimize_vector_store_queries(
            queries, mock_vector_store
        )

        assert len(results) == 2
        mock_vector_store.search_similar_batch.assert_called_once()

    def test_optimize_vector_store_queries_without_batch_support(self, mock_optimizer):
        """Test vector store query optimization without batch support."""
        # Mock vector store without batch support
        mock_vector_store = Mock()
        del mock_vector_store.search_similar_batch  # Remove batch method
        mock_vector_store.search_similar.return_value = [("id1", 0.9, {"test": "meta"})]

        queries = [[0.1] * 384, [0.2] * 384]

        results = mock_optimizer.optimize_vector_store_queries(
            queries, mock_vector_store
        )

        assert len(results) == 2
        assert mock_vector_store.search_similar.call_count == 2

    def test_optimize_vector_store_queries_with_batch_errors(self, mock_optimizer):
        """Test vector store query optimization with batch errors falling back to individual queries."""
        # Mock vector store that fails batch but succeeds individual
        mock_vector_store = Mock()
        mock_vector_store.search_similar_batch.side_effect = Exception("Batch failed")
        mock_vector_store.search_similar.return_value = [("id1", 0.9, {"test": "meta"})]

        queries = [[0.1] * 384, [0.2] * 384]

        results = mock_optimizer.optimize_vector_store_queries(
            queries, mock_vector_store
        )

        assert len(results) == 2
        mock_vector_store.search_similar_batch.assert_called_once()
        assert mock_vector_store.search_similar.call_count == 2

    def test_estimate_processing_time_default_cache_rate(self, mock_optimizer):
        """Test processing time estimation with default cache hit rate."""
        estimates = mock_optimizer.estimate_processing_time(100)

        assert "embedding_generation" in estimates
        assert "vector_indexing" in estimates
        assert "similarity_search" in estimates
        assert "memory_overhead" in estimates
        assert "total_estimated" in estimates

        # With 90% cache hit rate, should only generate 10 embeddings
        assert estimates["embedding_generation"] == 10 * 0.1  # 10 functions * 100ms
        assert estimates["vector_indexing"] == 100 * 0.01  # 100 functions * 10ms
        assert estimates["similarity_search"] == 100 * 0.05  # 100 functions * 50ms
        assert estimates["total_estimated"] > 0

    def test_estimate_processing_time_custom_cache_rate(self, mock_optimizer):
        """Test processing time estimation with custom cache hit rate."""
        estimates = mock_optimizer.estimate_processing_time(100, cache_hit_rate=0.5)

        # With 50% cache hit rate, should generate 50 embeddings
        assert estimates["embedding_generation"] == 50 * 0.1  # 50 functions * 100ms

    def test_monitor_system_resources(self, mock_optimizer):
        """Test system resource monitoring."""
        with patch.object(mock_optimizer.process, "cpu_percent", return_value=45.0):
            with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
                mock_memory.return_value.rss = (
                    (mock_optimizer.initial_memory + 200) * 1024 * 1024
                )

                resources = mock_optimizer.monitor_system_resources()

                assert "cpu_percent" in resources
                assert "memory_mb" in resources
                assert "memory_percent" in resources
                assert "memory_available_mb" in resources
                assert "warnings" in resources

                assert resources["cpu_percent"] == 45.0
                assert resources["memory_mb"] == mock_optimizer.initial_memory + 200
                assert (
                    resources["memory_percent"]
                    == ((mock_optimizer.initial_memory + 200) / 500) * 100
                )

    def test_monitor_system_resources_high_usage_warnings(self, mock_optimizer):
        """Test system resource monitoring with high usage warnings."""
        with patch.object(mock_optimizer.process, "cpu_percent", return_value=85.0):
            with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
                # Set memory to 95% usage (475MB)
                mock_memory.return_value.rss = 475 * 1024 * 1024

                resources = mock_optimizer.monitor_system_resources()

                assert len(resources["warnings"]) == 2
                assert "High memory usage (>90%)" in resources["warnings"]
                assert "High CPU usage (>80%)" in resources["warnings"]

    def test_get_optimization_recommendations_high_memory(self, mock_optimizer):
        """Test optimization recommendations for high memory usage."""
        with patch.object(mock_optimizer, "monitor_system_resources") as mock_monitor:
            mock_monitor.return_value = {"memory_percent": 85.0}

            recommendations = mock_optimizer.get_optimization_recommendations(
                100, {"avg_time_per_function": 0.1, "cache_hit_rate": 0.9}
            )

            assert any("reducing batch size" in rec for rec in recommendations)

    def test_get_optimization_recommendations_slow_performance(self, mock_optimizer):
        """Test optimization recommendations for slow performance."""
        with patch.object(mock_optimizer, "monitor_system_resources") as mock_monitor:
            mock_monitor.return_value = {"memory_percent": 50.0}

            recommendations = mock_optimizer.get_optimization_recommendations(
                100,
                {
                    "avg_time_per_function": 0.3,
                    "cache_hit_rate": 0.9,
                },  # >200ms per function
            )

            assert any("parallel processing" in rec for rec in recommendations)

    def test_get_optimization_recommendations_large_project(self, mock_optimizer):
        """Test optimization recommendations for large projects."""
        with patch.object(mock_optimizer, "monitor_system_resources") as mock_monitor:
            mock_monitor.return_value = {"memory_percent": 50.0}

            recommendations = mock_optimizer.get_optimization_recommendations(
                1500,  # Large project
                {"avg_time_per_function": 0.1, "cache_hit_rate": 0.9},
            )

            assert any("local embedding models" in rec for rec in recommendations)

    def test_get_optimization_recommendations_frequent_gc(self, mock_optimizer):
        """Test optimization recommendations for frequent garbage collection."""
        # Trigger GC multiple times
        mock_optimizer.stats["gc_triggered"] = 8

        with patch.object(mock_optimizer, "monitor_system_resources") as mock_monitor:
            mock_monitor.return_value = {"memory_percent": 50.0}

            recommendations = mock_optimizer.get_optimization_recommendations(
                100, {"avg_time_per_function": 0.1, "cache_hit_rate": 0.9}
            )

            assert any("increasing memory limit" in rec for rec in recommendations)

    def test_get_optimization_recommendations_low_cache_hit_rate(self, mock_optimizer):
        """Test optimization recommendations for low cache hit rate."""
        with patch.object(mock_optimizer, "monitor_system_resources") as mock_monitor:
            mock_monitor.return_value = {"memory_percent": 50.0}

            recommendations = mock_optimizer.get_optimization_recommendations(
                100,
                {
                    "avg_time_per_function": 0.1,
                    "cache_hit_rate": 0.6,
                },  # Low cache hit rate
            )

            assert any("cache hit rate" in rec for rec in recommendations)

    def test_get_stats_comprehensive(self, mock_optimizer):
        """Test comprehensive statistics retrieval."""
        # Simulate some operations
        mock_optimizer.stats["gc_triggered"] = 2
        mock_optimizer.stats["batches_optimized"] = 5
        mock_optimizer.stats["parallel_operations"] = 3
        mock_optimizer.stats["total_optimization_time"] = 1.5

        stats = mock_optimizer.get_stats()

        expected_keys = [
            "memory_initial_mb",
            "memory_current_mb",
            "memory_peak_mb",
            "memory_efficiency",
            "gc_triggered_count",
            "batches_optimized",
            "parallel_operations",
            "total_optimization_time",
            "avg_optimization_time",
        ]

        for key in expected_keys:
            assert key in stats

        assert stats["gc_triggered_count"] == 2
        assert stats["batches_optimized"] == 5
        assert stats["parallel_operations"] == 3
        assert stats["avg_optimization_time"] == 0.5  # 1.5 / 3

    def test_cleanup(self, mock_optimizer):
        """Test resource cleanup."""
        with patch("gc.collect") as mock_gc:
            with patch.object(mock_optimizer.executor, "shutdown") as mock_shutdown:
                mock_optimizer.cleanup()

                mock_shutdown.assert_called_once_with(wait=True)
                mock_gc.assert_called_once()

    def test_peak_memory_tracking(self, mock_optimizer):
        """Test that peak memory is tracked correctly."""
        initial_peak = mock_optimizer.stats["memory_peak_mb"]

        # Simulate memory increase during batch optimization
        with patch.object(mock_optimizer.process, "memory_info") as mock_memory:
            mock_memory.return_value.rss = (
                (mock_optimizer.initial_memory + 300) * 1024 * 1024
            )

            mock_optimizer.optimize_batch_size(100)

            assert mock_optimizer.stats["memory_peak_mb"] > initial_peak

    @pytest.mark.asyncio
    async def test_concurrent_semaphore_limiting(
        self, mock_optimizer, sample_functions
    ):
        """Test that semaphore properly limits concurrent operations."""
        call_times = []

        async def mock_generator(batch):
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate longer API call
            return []

        # Create 5 batches (should be limited to 3 concurrent)
        batches = [sample_functions[i : i + 2] for i in range(0, 10, 2)]

        start_time = time.time()
        await mock_optimizer.parallel_embedding_generation(batches, mock_generator)
        total_time = time.time() - start_time

        # With 5 batches and max 3 concurrent, should take at least 2 rounds
        # (3 concurrent + 2 more) = at least 0.2 seconds total
        assert total_time >= 0.15  # Allow some tolerance for timing
        assert len(call_times) == 5
