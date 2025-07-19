import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .facade import MatchingFacade
from .contextual_facade import ContextualMatchingFacade
from .semantic_matcher import SemanticMatcher
from .models import MatchResult
from ..parser import IntegratedParser
from ..utils.config import CodeDocSyncConfig

logger = logging.getLogger(__name__)


class UnifiedMatchingFacade:
    """
    Unified interface for all three matching strategies.

    Implements the complete matching pipeline:
    1. Direct matching (90% of cases)
    2. Contextual matching (8% of cases)
    3. Semantic matching (2% of cases)

    This is the first half implementation focusing on core integration.
    """

    def __init__(self, config: Optional[CodeDocSyncConfig] = None):
        self.config = config or CodeDocSyncConfig()
        self.stats = {
            "total_time": 0.0,
            "parsing_time": 0.0,
            "direct_matching_time": 0.0,
            "contextual_matching_time": 0.0,
            "semantic_matching_time": 0.0,
            "semantic_indexing_time": 0.0,
            "files_processed": 0,
            "matches_by_type": {"direct": 0, "contextual": 0, "semantic": 0},
        }

    async def match_project(
        self, project_path: str, use_cache: bool = True, enable_semantic: bool = True
    ) -> MatchResult:
        """
        Perform complete matching on a project.

        Args:
            project_path: Root directory of the project
            use_cache: Whether to use cached parsing/embeddings
            enable_semantic: Whether to use semantic matching

        Returns:
            Unified MatchResult with all matches
        """
        start_time = time.time()
        project_path = Path(project_path).resolve()

        logger.info(f"Starting unified matching for project: {project_path}")

        # Phase 1: Parse all Python files
        logger.info("Phase 1: Parsing Python files...")
        parse_start = time.time()

        all_functions = []
        python_files = self._discover_python_files(project_path)

        parser = IntegratedParser(cache_enabled=use_cache)
        for file_path in python_files:
            try:
                functions = parser.parse_file(str(file_path))
                all_functions.extend(functions)
                self.stats["files_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        self.stats["parsing_time"] = time.time() - parse_start
        logger.info(
            f"Parsed {len(all_functions)} functions from {len(python_files)} files"
        )

        # Phase 2: Direct matching
        logger.info("Phase 2: Direct matching...")
        direct_start = time.time()

        direct_facade = MatchingFacade(config=self.config)
        direct_result = direct_facade.match_project(str(project_path))

        self.stats["direct_matching_time"] = time.time() - direct_start

        # Count direct matches
        direct_matches = [
            m for m in direct_result.matched_pairs if m.confidence.overall >= 0.8
        ]
        self.stats["matches_by_type"]["direct"] = len(direct_matches)

        # Phase 3: Contextual matching
        logger.info("Phase 3: Contextual matching for remaining functions...")
        context_match_start = time.time()

        contextual_facade = ContextualMatchingFacade()
        contextual_result = contextual_facade.match_project(str(project_path))

        self.stats["contextual_matching_time"] = time.time() - context_match_start

        # Count new contextual matches (excluding those already well-matched by direct)
        prev_matched = {m.function.signature.name for m in direct_matches}
        contextual_matches = [
            m
            for m in contextual_result.matched_pairs
            if m.function.signature.name not in prev_matched
            and m.confidence.overall >= 0.7
        ]
        self.stats["matches_by_type"]["contextual"] = len(contextual_matches)

        # Start with contextual result as our final result
        final_result = contextual_result

        # Phase 4: Semantic matching (if enabled)
        if enable_semantic and getattr(self.config.matching, "enable_semantic", True):
            logger.info("Phase 4: Semantic matching for remaining functions...")

            # Initialize semantic matcher
            semantic_matcher = SemanticMatcher(str(project_path))

            # Build semantic index
            index_start = time.time()
            await semantic_matcher.prepare_semantic_index(
                all_functions, force_reindex=False
            )
            self.stats["semantic_indexing_time"] = time.time() - index_start

            # Perform semantic matching
            semantic_start = time.time()
            semantic_result = await semantic_matcher.match_with_embeddings(
                all_functions, [direct_result, contextual_result]
            )
            self.stats["semantic_matching_time"] = time.time() - semantic_start

            # Count new semantic matches
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

            final_result = semantic_result

            # Add semantic stats to result
            if not hasattr(final_result, "metadata"):
                final_result.metadata = {}
            final_result.metadata["semantic_stats"] = semantic_matcher.get_stats()

        # Calculate total time
        self.stats["total_time"] = time.time() - start_time

        # Add unified stats to result
        if not hasattr(final_result, "metadata"):
            final_result.metadata = {}
        final_result.metadata["unified_stats"] = self.get_stats()

        logger.info(f"Unified matching completed in {self.stats['total_time']:.2f}s")
        return final_result

    def _discover_python_files(self, project_path: Path) -> List[Path]:
        """Discover Python files with exclusions."""
        exclusions = {".git", "__pycache__", "venv", "env", ".venv", "build", "dist"}

        python_files = []
        for path in project_path.rglob("*.py"):
            # Check if any parent directory is in exclusions
            if any(part in exclusions for part in path.parts):
                continue
            python_files.append(path)

        return python_files

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_matches = sum(self.stats["matches_by_type"].values())

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
        }

    def print_summary(self) -> None:
        """Print comprehensive matching summary."""
        stats = self.get_stats()

        print("\n=== Unified Matching Summary ===")
        print(f"Total time: {stats['total_time_seconds']:.2f}s")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Total matches: {stats['total_matches']}")

        print("\n--- Time Breakdown ---")
        for phase, time_val in stats["phase_times"].items():
            if time_val > 0:
                print(
                    f"{phase}: {time_val:.2f}s ({time_val/stats['total_time_seconds']*100:.1f}%)"
                )

        print("\n--- Match Distribution ---")
        for match_type, count in stats["matches_by_type"].items():
            if count > 0:
                print(
                    f"{match_type}: {count} matches ({stats['match_distribution'][match_type]})"
                )
