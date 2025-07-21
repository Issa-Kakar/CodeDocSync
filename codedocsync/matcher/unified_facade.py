import gc
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import psutil  # type: ignore[import-untyped]

from ..parser import IntegratedParser
from ..utils.config import CodeDocSyncConfig
from .contextual_facade import ContextualMatchingFacade
from .facade import MatchingFacade
from .models import MatchResult
from .semantic_matcher import SemanticMatcher

logger = logging.getLogger(__name__)


class MemoryUsageStats(TypedDict):
    initial_mb: float
    peak_mb: float
    final_mb: float


class ErrorStats(TypedDict):
    parsing_errors: int
    matching_errors: int
    total_errors: int


class CacheStats(TypedDict):
    hits: int
    misses: int
    hit_rate: float


class ThroughputStats(TypedDict):
    functions_per_second: float
    files_per_second: float


class MatchesByType(TypedDict):
    direct: int
    contextual: int
    semantic: int


class StatsDict(TypedDict):
    total_time: float
    parsing_time: float
    direct_matching_time: float
    contextual_matching_time: float
    semantic_matching_time: float
    semantic_indexing_time: float
    files_processed: int
    functions_processed: int
    matches_by_type: MatchesByType
    memory_usage: MemoryUsageStats
    errors: ErrorStats
    cache_stats: CacheStats
    throughput: ThroughputStats


class EnhancedMatchResult(MatchResult):
    """MatchResult with additional metadata."""

    def __init__(self, base_result: MatchResult):
        super().__init__(
            matched_pairs=base_result.matched_pairs,
            unmatched_functions=base_result.unmatched_functions,
            total_functions=base_result.total_functions,
            match_duration_ms=base_result.match_duration_ms,
        )
        self.metadata: dict[str, Any] = {}


class UnifiedMatchingFacade:
    """
    Unified interface for all three matching strategies.

    Implements the complete matching pipeline:
    1. Direct matching (90% of cases)
    2. Contextual matching (8% of cases)
    3. Semantic matching (2% of cases)

    Enhanced with advanced performance monitoring, memory management,
    and production-ready optimization features.
    """

    def __init__(self, config: CodeDocSyncConfig | None = None):
        self.config = config or CodeDocSyncConfig()

        # Enhanced statistics tracking
        self.stats: StatsDict = {
            "total_time": 0.0,
            "parsing_time": 0.0,
            "direct_matching_time": 0.0,
            "contextual_matching_time": 0.0,
            "semantic_matching_time": 0.0,
            "semantic_indexing_time": 0.0,
            "files_processed": 0,
            "functions_processed": 0,
            "matches_by_type": {"direct": 0, "contextual": 0, "semantic": 0},
            # New performance metrics
            "memory_usage": {"initial_mb": 0.0, "peak_mb": 0.0, "final_mb": 0.0},
            "errors": {"parsing_errors": 0, "matching_errors": 0, "total_errors": 0},
            "cache_stats": {"hits": 0, "misses": 0, "hit_rate": 0.0},
            "throughput": {"functions_per_second": 0.0, "files_per_second": 0.0},
        }

        # Initialize process monitoring
        self.process = psutil.Process()
        self.stats["memory_usage"]["initial_mb"] = (
            self.process.memory_info().rss / 1024 / 1024
        )

        # Progress tracking
        self.progress_callback: Callable[[str, int, int], None] | None = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """Set a callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, phase: str, current: int, total: int) -> None:
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(phase, current, total)

    def _monitor_memory(self) -> float:
        """Monitor current memory usage and update peak."""
        current_mb: float = self.process.memory_info().rss / 1024 / 1024
        if current_mb > self.stats["memory_usage"]["peak_mb"]:
            self.stats["memory_usage"]["peak_mb"] = current_mb
        return current_mb

    def _should_trigger_gc(self, memory_threshold_mb: float = 500.0) -> bool:
        """Check if garbage collection should be triggered."""
        current_mb = self._monitor_memory()
        return current_mb > memory_threshold_mb

    async def match_project(
        self,
        project_path: str,
        use_cache: bool = True,
        enable_semantic: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> EnhancedMatchResult:
        """
        Perform complete matching on a project with advanced monitoring.

        Args:
            project_path: Root directory of the project
            use_cache: Whether to use cached parsing/embeddings
            enable_semantic: Whether to use semantic matching
            progress_callback: Optional callback for progress updates

        Returns:
            Unified MatchResult with all matches and comprehensive metadata
        """
        start_time = time.time()
        project_path_obj = Path(project_path).resolve()

        if progress_callback:
            self.set_progress_callback(progress_callback)

        logger.info(
            f"Starting enhanced unified matching for project: {project_path_obj}"
        )

        try:
            # Phase 1: Enhanced file discovery and parsing
            logger.info("Phase 1: Enhanced parsing with monitoring...")
            parse_start = time.time()

            all_functions = []
            python_files = self._discover_python_files(project_path_obj)

            if not python_files:
                logger.warning(f"No Python files found in {project_path_obj}")
                return self._create_empty_result()

            parser = IntegratedParser()
            parsing_errors = 0

            for i, file_path in enumerate(python_files):
                self._update_progress("Parsing", i + 1, len(python_files))

                try:
                    functions = parser.parse_file(str(file_path))
                    all_functions.extend(functions)
                    self.stats["files_processed"] += 1

                    # Trigger GC periodically for large projects
                    if i > 0 and i % 100 == 0 and self._should_trigger_gc():
                        gc.collect()
                        logger.debug(
                            f"GC triggered at file {i}, memory: {self._monitor_memory():.1f}MB"
                        )

                except Exception as e:
                    parsing_errors += 1
                    logger.error(f"Failed to parse {file_path}: {e}")

            self.stats["parsing_time"] = time.time() - parse_start
            self.stats["functions_processed"] = len(all_functions)
            self.stats["errors"]["parsing_errors"] = parsing_errors

            logger.info(
                f"Parsed {len(all_functions)} functions from {len(python_files)} files "
                f"({parsing_errors} errors)"
            )

            # Phase 2: Direct matching with error handling
            logger.info("Phase 2: Enhanced direct matching...")
            direct_start = time.time()

            try:
                direct_facade = MatchingFacade(config=self.config)
                direct_result = direct_facade.match_project(str(project_path_obj))

                # Enhanced direct match analysis
                direct_matches = [
                    m
                    for m in direct_result.matched_pairs
                    if m.confidence.overall >= 0.8
                ]
                self.stats["matches_by_type"]["direct"] = len(direct_matches)

            except Exception as e:
                logger.error(f"Direct matching failed: {e}")
                self.stats["errors"]["matching_errors"] += 1
                direct_result = self._create_empty_result()
                direct_matches = []

            self.stats["direct_matching_time"] = time.time() - direct_start

            # Phase 3: Enhanced contextual matching
            logger.info("Phase 3: Enhanced contextual matching...")
            context_match_start = time.time()

            try:
                contextual_facade = ContextualMatchingFacade()
                contextual_result = contextual_facade.match_project(
                    str(project_path_obj)
                )

                # Enhanced contextual match analysis
                prev_matched = {m.function.signature.name for m in direct_matches}
                contextual_matches = [
                    m
                    for m in contextual_result.matched_pairs
                    if m.function.signature.name not in prev_matched
                    and m.confidence.overall >= 0.7
                ]
                self.stats["matches_by_type"]["contextual"] = len(contextual_matches)

            except Exception as e:
                logger.error(f"Contextual matching failed: {e}")
                self.stats["errors"]["matching_errors"] += 1
                contextual_result = direct_result  # Fallback to direct result
                contextual_matches = []

            self.stats["contextual_matching_time"] = time.time() - context_match_start
            enhanced_result = EnhancedMatchResult(contextual_result)

            # Phase 4: Enhanced semantic matching with monitoring
            if enable_semantic and getattr(
                self.config.matcher, "enable_semantic", True
            ):
                logger.info("Phase 4: Enhanced semantic matching with monitoring...")

                try:
                    semantic_matcher = SemanticMatcher(str(project_path_obj))

                    # Build semantic index with progress tracking
                    index_start = time.time()
                    await semantic_matcher.prepare_semantic_index(
                        all_functions, force_reindex=False
                    )
                    self.stats["semantic_indexing_time"] = time.time() - index_start

                    # Monitor memory before intensive semantic operations
                    pre_semantic_memory = self._monitor_memory()
                    logger.debug(f"Pre-semantic memory: {pre_semantic_memory:.1f}MB")

                    # Perform semantic matching
                    semantic_start = time.time()
                    semantic_result = await semantic_matcher.match_with_embeddings(
                        all_functions, [direct_result, contextual_result]
                    )
                    self.stats["semantic_matching_time"] = time.time() - semantic_start

                    # Enhanced semantic match analysis
                    all_prev_matched = {
                        m.function.signature.name
                        for m in contextual_result.matched_pairs
                        if m.confidence.overall >= 0.7
                    }
                    semantic_matches = [
                        m
                        for m in semantic_result.matched_pairs
                        if m.function.signature.name not in all_prev_matched
                    ]
                    self.stats["matches_by_type"]["semantic"] = len(semantic_matches)

                    enhanced_result = EnhancedMatchResult(semantic_result)

                    # Add semantic stats to result
                    enhanced_result.metadata["semantic_stats"] = (
                        semantic_matcher.get_stats()
                    )

                    # Log semantic performance
                    post_semantic_memory = self._monitor_memory()
                    logger.info(
                        f"Semantic matching: {len(semantic_matches)} matches, "
                        f"memory delta: {post_semantic_memory - pre_semantic_memory:.1f}MB"
                    )

                except Exception as e:
                    logger.error(f"Semantic matching failed: {e}")
                    self.stats["errors"]["matching_errors"] += 1
                    # Continue with contextual result as fallback

            # Finalize statistics
            self._finalize_stats(start_time, len(all_functions), len(python_files))

            # Add comprehensive metadata
            self._add_comprehensive_metadata(enhanced_result)

            logger.info(
                f"Enhanced unified matching completed in {self.stats['total_time']:.2f}s"
            )
            return enhanced_result

        except Exception as e:
            logger.error(f"Critical error in unified matching: {e}")
            self.stats["errors"]["total_errors"] += 1
            raise

    def match_file(self, file_path: str | Path) -> MatchResult:
        """
        Match functions in a single file using the unified pipeline.

        Args:
            file_path: Path to Python file

        Returns:
            MatchResult with matched pairs
        """
        # Use the basic MatchingFacade for single file matching
        from .facade import MatchingFacade

        # Create a basic facade with same config
        basic_facade = MatchingFacade(self.config)

        # Use its match_file method
        return basic_facade.match_file(file_path)

    def _create_empty_result(self) -> EnhancedMatchResult:
        """Create an empty match result for error cases."""
        base_result = MatchResult(
            total_functions=0, matched_pairs=[], unmatched_functions=[]
        )
        return EnhancedMatchResult(base_result)

    def _finalize_stats(
        self, start_time: float, total_functions: int, total_files: int
    ) -> None:
        """Finalize performance statistics."""
        self.stats["total_time"] = time.time() - start_time
        self.stats["memory_usage"]["final_mb"] = self._monitor_memory()
        self.stats["errors"]["total_errors"] = (
            self.stats["errors"]["parsing_errors"]
            + self.stats["errors"]["matching_errors"]
        )

        # Calculate throughput
        if self.stats["total_time"] > 0:
            self.stats["throughput"]["functions_per_second"] = (
                total_functions / self.stats["total_time"]
            )
            self.stats["throughput"]["files_per_second"] = (
                total_files / self.stats["total_time"]
            )

    def _add_comprehensive_metadata(self, result: EnhancedMatchResult) -> None:
        """Add comprehensive metadata to result."""
        result.metadata.update(
            {
                "unified_stats": self.get_comprehensive_stats(),
                "system_info": {
                    "cpu_count": os.cpu_count(),
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "platform": os.name,
                },
                "performance_profile": self._generate_performance_profile(),
            }
        )

    def _generate_performance_profile(self) -> dict[str, str]:
        """Generate performance profile for optimization insights."""
        total_time = self.stats["total_time"]

        if total_time == 0:
            return {"status": "no_data"}

        profile = {}

        # Phase time analysis
        parsing_pct = (self.stats["parsing_time"] / total_time) * 100
        direct_pct = (self.stats["direct_matching_time"] / total_time) * 100
        contextual_pct = (self.stats["contextual_matching_time"] / total_time) * 100
        semantic_pct = (self.stats["semantic_matching_time"] / total_time) * 100

        if parsing_pct > 50:
            profile["bottleneck"] = "parsing"
            profile["recommendation"] = "Consider using cache or parallel parsing"
        elif semantic_pct > 40:
            profile["bottleneck"] = "semantic_matching"
            profile["recommendation"] = "Consider reducing semantic batch size"
        elif contextual_pct > 30:
            profile["bottleneck"] = "contextual_matching"
            profile["recommendation"] = "Consider optimizing import resolution"
        elif direct_pct > 20:
            profile["bottleneck"] = "direct_matching"
            profile["recommendation"] = (
                "Consider optimizing signature similarity calculation"
            )
        else:
            profile["bottleneck"] = "none"
            profile["recommendation"] = "Performance is well balanced"

        # Memory analysis
        memory_growth = (
            self.stats["memory_usage"]["peak_mb"]
            - self.stats["memory_usage"]["initial_mb"]
        )
        if memory_growth > 200:
            profile["memory_concern"] = "high_growth"
        elif memory_growth > 100:
            profile["memory_concern"] = "moderate_growth"
        else:
            profile["memory_concern"] = "acceptable"

        return profile

    def _discover_python_files(self, project_path: Path) -> list[Path]:
        """Discover Python files with exclusions."""
        exclusions = {".git", "__pycache__", "venv", "env", ".venv", "build", "dist"}

        python_files = []
        for path in project_path.rglob("*.py"):
            # Check if any parent directory is in exclusions
            if any(part in exclusions for part in path.parts):
                continue
            python_files.append(path)

        return python_files

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics (backward compatibility)."""
        return self.get_comprehensive_stats()

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get enhanced comprehensive statistics with performance insights."""
        matches_by_type = self.stats["matches_by_type"]
        total_matches = (
            matches_by_type["direct"]
            + matches_by_type["contextual"]
            + matches_by_type["semantic"]
        )

        return {
            "total_time_seconds": self.stats["total_time"],
            "phase_times": {
                "parsing": self.stats["parsing_time"],
                "direct_matching": self.stats["direct_matching_time"],
                "contextual_matching": self.stats["contextual_matching_time"],
                "semantic_indexing": self.stats["semantic_indexing_time"],
                "semantic_matching": self.stats["semantic_matching_time"],
            },
            "files_processed": self.stats["files_processed"],
            "functions_processed": self.stats["functions_processed"],
            "total_matches": total_matches,
            "matches_by_type": self.stats["matches_by_type"],
            "match_distribution": {
                "direct": (
                    f"{self.stats['matches_by_type']['direct'] / total_matches * 100:.1f}%"
                    if total_matches > 0
                    else "0%"
                ),
                "contextual": (
                    f"{self.stats['matches_by_type']['contextual'] / total_matches * 100:.1f}%"
                    if total_matches > 0
                    else "0%"
                ),
                "semantic": (
                    f"{self.stats['matches_by_type']['semantic'] / total_matches * 100:.1f}%"
                    if total_matches > 0
                    else "0%"
                ),
            },
            # Enhanced metrics
            "memory_usage": self.stats["memory_usage"],
            "error_summary": self.stats["errors"],
            "throughput": self.stats["throughput"],
            "efficiency_metrics": self._calculate_efficiency_metrics(),
        }

    def _calculate_efficiency_metrics(self) -> dict[str, Any]:
        """Calculate efficiency metrics for performance analysis."""
        total_time = self.stats["total_time"]
        total_functions = self.stats["functions_processed"]
        matches_by_type = self.stats["matches_by_type"]

        if total_time == 0 or total_functions == 0:
            return {"status": "no_data"}

        return {
            "functions_per_mb": total_functions
            / max(self.stats["memory_usage"]["peak_mb"], 1),
            "matches_per_second": (
                matches_by_type["direct"]
                + matches_by_type["contextual"]
                + matches_by_type["semantic"]
            )
            / total_time,
            "error_rate": self.stats["errors"]["total_errors"]
            / max(total_functions, 1),
            "memory_efficiency": total_functions
            / max(
                self.stats["memory_usage"]["peak_mb"]
                - self.stats["memory_usage"]["initial_mb"],
                1,
            ),
        }

    def print_summary(self) -> None:
        """Print enhanced comprehensive matching summary."""
        stats = self.get_comprehensive_stats()

        print("\n=== Enhanced Unified Matching Summary ===")
        print(f"Total time: {stats['total_time_seconds']:.2f}s")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Functions processed: {stats['functions_processed']}")
        print(f"Total matches: {stats['total_matches']}")

        print("\n--- Performance Metrics ---")
        if "throughput" in stats:
            print(
                f"Functions/second: {stats['throughput']['functions_per_second']:.1f}"
            )
            print(f"Files/second: {stats['throughput']['files_per_second']:.1f}")

        print("\n--- Memory Usage ---")
        if "memory_usage" in stats:
            print(f"Initial: {stats['memory_usage']['initial_mb']:.1f}MB")
            print(f"Peak: {stats['memory_usage']['peak_mb']:.1f}MB")
            print(f"Final: {stats['memory_usage']['final_mb']:.1f}MB")
            print(
                f"Growth: {stats['memory_usage']['peak_mb'] - stats['memory_usage']['initial_mb']:.1f}MB"
            )

        print("\n--- Time Breakdown ---")
        for phase, time_val in stats["phase_times"].items():
            if time_val > 0:
                print(
                    f"{phase}: {time_val:.2f}s ({time_val / stats['total_time_seconds'] * 100:.1f}%)"
                )

        print("\n--- Match Distribution ---")
        for match_type, count in stats["matches_by_type"].items():
            if count > 0:
                print(
                    f"{match_type}: {count} matches ({stats['match_distribution'][match_type]})"
                )

        print("\n--- Error Summary ---")
        if "error_summary" in stats:
            errors = stats["error_summary"]
            print(f"Parsing errors: {errors['parsing_errors']}")
            print(f"Matching errors: {errors['matching_errors']}")
            print(f"Total errors: {errors['total_errors']}")

        print("\n--- Efficiency Metrics ---")
        if (
            "efficiency_metrics" in stats
            and stats["efficiency_metrics"].get("status") != "no_data"
        ):
            eff = stats["efficiency_metrics"]
            print(f"Functions per MB: {eff['functions_per_mb']:.1f}")
            print(f"Matches per second: {eff['matches_per_second']:.1f}")
            print(f"Error rate: {eff['error_rate']:.3f}")

    def get_performance_recommendations(self) -> list[str]:
        """Get performance recommendations based on current metrics."""
        recommendations = []
        stats = self.get_comprehensive_stats()

        # Parsing performance
        if (
            stats.get("phase_times", {}).get("parsing", 0)
            > stats["total_time_seconds"] * 0.5
        ):
            recommendations.append(
                "Consider enabling parsing cache to improve performance"
            )

        # Memory usage
        memory_growth = stats.get("memory_usage", {}).get("peak_mb", 0) - stats.get(
            "memory_usage", {}
        ).get("initial_mb", 0)
        if memory_growth > 500:
            recommendations.append(
                "High memory usage detected - consider processing smaller batches"
            )

        # Error rates
        error_rate = stats.get("efficiency_metrics", {}).get("error_rate", 0)
        if error_rate > 0.1:
            recommendations.append(
                "High error rate - check file permissions and syntax"
            )

        # Semantic matching performance
        semantic_time = stats.get("phase_times", {}).get("semantic_matching", 0)
        if semantic_time > 60:  # More than 1 minute
            recommendations.append(
                "Semantic matching is slow - consider reducing batch size or disabling for large projects"
            )

        if not recommendations:
            recommendations.append("Performance is good - no specific recommendations")

        return recommendations

    async def cleanup(self) -> None:
        """Clean up resources and perform final garbage collection."""
        gc.collect()
        logger.info("Unified matching facade cleanup completed")
