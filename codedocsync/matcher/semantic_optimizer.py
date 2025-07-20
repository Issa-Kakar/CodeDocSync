import asyncio
import gc
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil

from ..parser import ParsedFunction
from .semantic_models import FunctionEmbedding

logger = logging.getLogger(__name__)


class SemanticOptimizer:
    """
    Performance optimizations for semantic matching.

    Handles:
    - Batch processing optimization
    - Memory management
    - Concurrent operations
    - Resource monitoring
    """

    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Monitor resource usage
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Performance tracking
        self.stats = {
            "gc_triggered": 0,
            "batches_optimized": 0,
            "parallel_operations": 0,
            "memory_peak_mb": self.initial_memory,
            "total_optimization_time": 0.0,
        }

    def optimize_batch_size(self, total_functions: int) -> int:
        """
        Dynamically determine optimal batch size.

        Considers:
        - Available memory
        - Number of functions
        - API rate limits
        """
        # Base batch size
        base_batch = 100

        # Adjust based on available memory
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.initial_memory
        memory_available = self.max_memory_mb - memory_used

        # Update peak memory tracking
        self.stats["memory_peak_mb"] = max(self.stats["memory_peak_mb"], current_memory)

        if memory_available < 100:
            # Low memory, use smaller batches
            optimized_batch = min(25, total_functions)
        elif memory_available < 200:
            optimized_batch = min(50, total_functions)
        else:
            # Plenty of memory
            optimized_batch = min(base_batch, total_functions)

        self.stats["batches_optimized"] += 1
        logger.debug(
            f"Optimized batch size: {optimized_batch} (memory available: {memory_available:.1f}MB)"
        )

        return optimized_batch

    async def parallel_embedding_generation(
        self, function_batches: list[list[ParsedFunction]], generator_func: Callable
    ) -> list[FunctionEmbedding]:
        """
        Generate embeddings in parallel with controlled concurrency.
        """
        start_time = time.time()

        # Limit concurrent API calls to avoid rate limits
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent batches

        async def process_batch_with_limit(batch):
            async with semaphore:
                return await generator_func(batch)

        # Process all batches
        tasks = [process_batch_with_limit(batch) for batch in function_batches]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle errors
        all_embeddings = []
        error_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
                error_count += 1
            else:
                all_embeddings.extend(result)

        # Update stats
        self.stats["parallel_operations"] += 1
        self.stats["total_optimization_time"] += time.time() - start_time

        if error_count > 0:
            logger.warning(
                f"Parallel processing completed with {error_count} batch errors"
            )

        return all_embeddings

    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.initial_memory

        # Trigger GC if using >80% of allowed memory
        should_trigger = memory_used > (self.max_memory_mb * 0.8)

        if should_trigger:
            logger.info(
                f"Memory usage {memory_used:.1f}MB exceeds 80% threshold, triggering GC"
            )
            gc.collect()
            self.stats["gc_triggered"] += 1

            # Update memory tracking after GC
            new_memory = self.process.memory_info().rss / 1024 / 1024
            logger.info(
                f"Memory after GC: {new_memory:.1f}MB (freed {current_memory - new_memory:.1f}MB)"
            )

        return should_trigger

    def optimize_vector_store_queries(
        self, queries: list[list[float]], vector_store: Any
    ) -> list[list[Any]]:
        """
        Optimize vector store queries with batching.

        Some vector stores support batch queries for better performance.
        """
        start_time = time.time()

        # Check if vector store supports batch queries
        if hasattr(vector_store, "search_similar_batch"):
            # Use batch search
            batch_size = 10
            results = []

            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size]
                try:
                    batch_results = vector_store.search_similar_batch(batch)
                    results.extend(batch_results)
                except Exception as e:
                    logger.warning(
                        f"Batch query failed, falling back to individual queries: {e}"
                    )
                    # Fallback to individual queries for this batch
                    for query in batch:
                        try:
                            result = vector_store.search_similar(query)
                            results.append(result)
                        except Exception as query_error:
                            logger.error(f"Individual query failed: {query_error}")
                            results.append([])

            logger.debug(
                f"Batch query optimization completed in {time.time() - start_time:.3f}s"
            )
            return results
        else:
            # Fall back to individual queries
            logger.debug(
                "Vector store doesn't support batch queries, using individual queries"
            )
            results = []
            for query in queries:
                try:
                    result = vector_store.search_similar(query)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Individual query failed: {e}")
                    results.append([])

            return results

    def estimate_processing_time(
        self, num_functions: int, cache_hit_rate: float = 0.9
    ) -> dict[str, float]:
        """Estimate processing time for semantic matching."""
        # Based on performance benchmarks
        embeddings_to_generate = int(num_functions * (1 - cache_hit_rate))

        estimates = {
            "embedding_generation": embeddings_to_generate * 0.1,  # 100ms per function
            "vector_indexing": num_functions * 0.01,  # 10ms per function
            "similarity_search": num_functions * 0.05,  # 50ms per function
            "memory_overhead": max(
                0.5, num_functions * 0.001
            ),  # Base overhead + scaling
            "total_estimated": 0.0,
        }

        estimates["total_estimated"] = sum(estimates.values())

        logger.debug(
            f"Estimated processing time for {num_functions} functions: {estimates['total_estimated']:.1f}s"
        )
        return estimates

    def monitor_system_resources(self) -> dict[str, Any]:
        """Monitor current system resource usage."""
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        current_memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = (current_memory_mb / self.max_memory_mb) * 100

        system_stats = {
            "cpu_percent": cpu_percent,
            "memory_mb": current_memory_mb,
            "memory_percent": memory_percent,
            "memory_available_mb": self.max_memory_mb - current_memory_mb,
            "peak_memory_mb": self.stats["memory_peak_mb"],
            "threads_active": (
                self.executor._threads if hasattr(self.executor, "_threads") else 0
            ),
        }

        # Check for resource warnings
        warnings = []
        if memory_percent > 90:
            warnings.append("High memory usage (>90%)")
        if cpu_percent > 80:
            warnings.append("High CPU usage (>80%)")

        system_stats["warnings"] = warnings
        return system_stats

    def get_optimization_recommendations(
        self, num_functions: int, current_performance: dict[str, float]
    ) -> list[str]:
        """Generate optimization recommendations based on current performance."""
        recommendations = []

        # Memory recommendations
        memory_usage = self.monitor_system_resources()["memory_percent"]
        if memory_usage > 80:
            recommendations.append(
                "Consider reducing batch size or enabling more aggressive caching"
            )

        # Performance recommendations
        if (
            current_performance.get("avg_time_per_function", 0) > 0.2
        ):  # >200ms per function
            recommendations.append("Enable parallel processing for better throughput")

        # Scale recommendations
        if num_functions > 1000:
            recommendations.append(
                "For large projects, consider using local embedding models to reduce API costs"
            )

        # GC recommendations
        if self.stats["gc_triggered"] > 5:
            recommendations.append(
                "Frequent garbage collection detected - consider increasing memory limit"
            )

        # Cache recommendations
        cache_stats = current_performance.get("cache_hit_rate", 0)
        if cache_stats < 0.8:
            recommendations.append(
                "Low cache hit rate - ensure functions haven't changed significantly"
            )

        return recommendations

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        current_memory = self.process.memory_info().rss / 1024 / 1024

        return {
            "memory_initial_mb": self.initial_memory,
            "memory_current_mb": current_memory,
            "memory_peak_mb": self.stats["memory_peak_mb"],
            "memory_efficiency": (current_memory - self.initial_memory)
            / max(1, self.stats.get("functions_processed", 1)),
            "gc_triggered_count": self.stats["gc_triggered"],
            "batches_optimized": self.stats["batches_optimized"],
            "parallel_operations": self.stats["parallel_operations"],
            "total_optimization_time": self.stats["total_optimization_time"],
            "avg_optimization_time": (
                self.stats["total_optimization_time"]
                / max(1, self.stats["parallel_operations"])
                if self.stats["parallel_operations"] > 0
                else 0
            ),
        }

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up SemanticOptimizer resources")

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        # Force garbage collection
        gc.collect()

        # Log final stats
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_freed = self.stats["memory_peak_mb"] - final_memory
        logger.info(f"Cleanup complete - freed {memory_freed:.1f}MB memory")
