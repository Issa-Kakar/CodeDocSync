"""High-level facade for matching operations."""

import logging
from pathlib import Path
from typing import Optional, Union
from codedocsync.parser import IntegratedParser
from codedocsync.utils.config import CodeDocSyncConfig
from .direct_matcher import DirectMatcher
from .models import MatchResult

logger = logging.getLogger(__name__)


class MatchingFacade:
    """High-level interface for matching operations."""

    def __init__(self, config: Optional[CodeDocSyncConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or CodeDocSyncConfig()

        # Initialize matchers based on config
        self.direct_matcher = DirectMatcher()

    def match_file(self, file_path: Union[str, Path]) -> MatchResult:
        """
        Parse and match functions in a single file.

        Args:
            file_path: Path to Python file

        Returns:
            MatchResult with all matches

        Example:
            >>> facade = MatchingFacade()
            >>> result = facade.match_file("mymodule.py")
            >>> print(f"Matched {result.match_rate:.1%} of functions")
        """
        parser = IntegratedParser()
        functions = parser.parse_file(str(file_path))

        if not functions:
            logger.warning(f"No functions found in {file_path}")
            return MatchResult(total_functions=0)

        return self.direct_matcher.match_functions(functions)

    def match_project(self, project_path: Union[str, Path]) -> MatchResult:
        """
        Parse and match all Python files in a project.

        Args:
            project_path: Root directory of project

        Returns:
            Combined MatchResult for entire project
        """
        project_path = Path(project_path)

        # Find all Python files
        python_files = list(project_path.rglob("*.py"))

        # Filter out common excluded directories
        excluded_dirs = {".venv", "venv", "__pycache__", ".git", "build", "dist"}
        python_files = [
            f
            for f in python_files
            if not any(excluded in f.parts for excluded in excluded_dirs)
        ]

        logger.info(f"Found {len(python_files)} Python files in {project_path}")

        # Parse all files
        all_functions = []
        parser = IntegratedParser()

        for file_path in python_files:
            try:
                functions = parser.parse_file(str(file_path))
                all_functions.extend(functions)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        logger.info(f"Parsed {len(all_functions)} functions total")

        # Match all functions
        return self.direct_matcher.match_functions(all_functions)
