"""
High-level interface for contextual matching.

Combines direct and contextual matching for best results.
"""

import fnmatch
import logging
import time
from pathlib import Path

from ..parser import IntegratedParser
from ..utils.config import CodeDocSyncConfig
from .contextual_matcher import ContextualMatcher
from .direct_matcher import DirectMatcher
from .models import MatchResult

logger = logging.getLogger(__name__)


class ContextualMatchingFacade:
    """
    High-level interface for contextual matching.

    Combines direct and contextual matching for best results.
    """

    def __init__(self, config: CodeDocSyncConfig | None = None):
        self.config = config or CodeDocSyncConfig()
        self.stats = {
            "total_time": 0.0,
            "parsing_time": 0.0,
            "direct_matching_time": 0.0,
            "contextual_matching_time": 0.0,
            "files_processed": 0,
        }

    def match_project(self, project_path: str, use_cache: bool = True) -> MatchResult:
        """
        Perform complete matching on a project.

        Args:
            project_path: Root directory of the project
            use_cache: Whether to use cached parsing results

        Returns:
            Combined MatchResult with all matches
        """
        start_time = time.time()
        project_path = Path(project_path).resolve()

        # Initialize components
        parser = IntegratedParser()
        direct_matcher = DirectMatcher()
        contextual_matcher = ContextualMatcher(str(project_path))

        # Phase 1: Build project context
        logger.info("Building project context...")
        context_start = time.time()
        contextual_matcher.analyze_project()
        self.stats["contextual_matching_time"] += time.time() - context_start

        # Phase 2: Parse and match all files
        logger.info("Parsing and matching functions...")
        all_functions = []
        python_files = self._discover_python_files(project_path)

        parse_start = time.time()
        for file_path in python_files:
            try:
                functions = parser.parse_file(str(file_path))
                all_functions.extend(functions)
                self.stats["files_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        self.stats["parsing_time"] = time.time() - parse_start

        # Phase 3: Direct matching
        direct_start = time.time()
        direct_result = direct_matcher.match_functions(all_functions)
        self.stats["direct_matching_time"] = time.time() - direct_start

        # Phase 4: Contextual matching for low-confidence matches
        context_match_start = time.time()
        final_result = contextual_matcher.match_with_context(
            all_functions, direct_result
        )
        self.stats["contextual_matching_time"] += time.time() - context_match_start

        # Add statistics to result
        self.stats["total_time"] = time.time() - start_time
        final_result.metadata = {
            "performance": self.stats,
            "matcher_stats": {
                "direct": direct_matcher.get_stats(),
                "contextual": contextual_matcher.stats,
            },
        }

        return final_result

    def match_file(
        self, file_path: str, project_path: str | None = None
    ) -> MatchResult:
        """
        Match a single file with optional project context.

        Args:
            file_path: Python file to analyze
            project_path: Optional project root for context

        Returns:
            MatchResult for the file
        """
        file_path = Path(file_path).resolve()

        # If no project path, use file's parent directory
        if not project_path:
            project_path = file_path.parent

        # For single file, we still build context but only for its dependencies
        parser = IntegratedParser()
        contextual_matcher = ContextualMatcher(str(project_path))

        # Parse the file
        functions = parser.parse_file(str(file_path))

        # Build minimal context (just this file and its imports)
        contextual_matcher._analyze_file(str(file_path))

        # Try direct matching first
        direct_matcher = DirectMatcher()
        direct_result = direct_matcher.match_functions(functions)

        # Enhance with contextual matching
        return contextual_matcher.match_with_context(functions, direct_result)

    def _discover_python_files(self, project_path: Path) -> list[Path]:
        """Discover Python files respecting exclusions."""
        # Default exclusions if not specified in config
        exclusions = {
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            ".venv/*",
            "venv/*",
            "env/*",
            "build/*",
            "dist/*",
            ".tox/*",
            "node_modules/*",
        }

        # Add config exclusions if available
        if hasattr(self.config, "ignore") and hasattr(self.config.ignore, "paths"):
            exclusions.update(self.config.ignore.paths)

        python_files = []
        for path in project_path.rglob("*.py"):
            # Check exclusions
            if any(self._matches_pattern(str(path), pattern) for pattern in exclusions):
                continue

            python_files.append(path)

        return python_files

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches exclusion pattern."""
        return fnmatch.fnmatch(path, pattern)

    def print_summary(self) -> None:
        """Print performance summary."""
        print("\n=== Performance Summary ===")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Parsing: {self.stats['parsing_time']:.2f}s")
        print(f"Direct matching: {self.stats['direct_matching_time']:.2f}s")
        print(f"Contextual matching: {self.stats['contextual_matching_time']:.2f}s")
