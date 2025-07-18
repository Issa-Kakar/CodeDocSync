import logging
from pathlib import Path
from typing import List, Optional

from .models import MatchResult, MatchedPair, MatchType, MatchConfidence
from .contextual_models import ContextualMatcherState, ImportStatement, FunctionLocation
from .module_resolver import ModuleResolver
from .function_registry import FunctionRegistry
from .import_parser import ImportParser
from ..parser import ParsedFunction, IntegratedParser

logger = logging.getLogger(__name__)


class ContextualMatcher:
    """
    Matches functions to documentation using module context and imports.

    Handles:
    - Functions imported from other modules
    - Functions that moved between files
    - Cross-file documentation references
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.state = ContextualMatcherState()
        self.module_resolver = ModuleResolver(project_root)
        self.function_registry = FunctionRegistry()
        self.import_parser = ImportParser()
        self.integrated_parser = IntegratedParser()

        # Performance metrics
        self.stats = {
            "files_analyzed": 0,
            "imports_resolved": 0,
            "cross_file_matches": 0,
            "moved_functions": 0,
        }

    def analyze_project(
        self, python_files: Optional[List[str]] = None
    ) -> ContextualMatcherState:
        """
        Analyze all Python files to build project context.

        Args:
            python_files: Optional list of files, otherwise discover all

        Returns:
            Populated ContextualMatcherState
        """
        if python_files is None:
            python_files = self._discover_python_files()

        # Phase 1: Parse all modules and build registry
        logger.info(f"Analyzing {len(python_files)} Python files")

        for file_path in python_files:
            self._analyze_file(file_path)

        # Phase 2: Resolve all import chains
        self._resolve_all_imports()

        logger.info(
            f"Analysis complete: {self.stats['files_analyzed']} files, "
            f"{len(self.function_registry.functions)} functions"
        )

        return self.state

    def match_with_context(
        self,
        functions: List[ParsedFunction],
        direct_match_result: Optional[MatchResult] = None,
    ) -> MatchResult:
        """
        Perform contextual matching on functions.

        Args:
            functions: Functions to match (with their docstrings)
            direct_match_result: Previous direct matching results to enhance

        Returns:
            MatchResult with contextual matches
        """
        matches = []
        unmatched_functions = []
        unmatched_docs = []

        # Start with direct match results if provided
        if direct_match_result:
            matches.extend(direct_match_result.matches)
            # Track which functions need contextual matching
            matched_function_ids = {
                m.function.signature.name
                for m in direct_match_result.matches
                if m.confidence.overall >= 0.8
            }
        else:
            matched_function_ids = set()

        # Process each function
        for func in functions:
            if func.signature.name in matched_function_ids:
                continue  # Already well-matched

            # Try contextual matching
            contextual_match = self._find_contextual_match(func)

            if contextual_match:
                matches.append(contextual_match)
                self.stats["cross_file_matches"] += 1
            else:
                unmatched_functions.append(func)

        # Build result
        return MatchResult(
            total_functions=len(functions),
            total_docs=len(functions),  # Assuming integrated parsing
            matches=matches,
            unmatched_functions=unmatched_functions,
            unmatched_docs=unmatched_docs,
        )

    def _analyze_file(self, file_path: str) -> None:
        """Analyze a single file and update state."""
        try:
            # Get module path
            module_path = self.module_resolver.resolve_module_path(file_path)
            if not module_path:
                logger.warning(f"Could not resolve module path for {file_path}")
                return

            # Build module info
            module_info = self.import_parser.build_module_info(file_path, module_path)

            # Parse functions
            functions = self.integrated_parser.parse_file(file_path)

            # Register functions
            for func in functions:
                self.function_registry.register_function(func, module_info)

            # Store module info
            self.state.add_module(module_info)
            self.stats["files_analyzed"] += 1

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")

    def _find_contextual_match(self, function: ParsedFunction) -> Optional[MatchedPair]:
        """
        Find a contextual match for a function.

        Tries in order:
        1. Check if function was imported from elsewhere
        2. Check if function moved to different file
        3. Check for documentation in different location
        """
        # Strategy 1: Imported function
        match = self._match_imported_function(function)
        if match:
            return match

        # Strategy 2: Moved function
        match = self._match_moved_function(function)
        if match:
            return match

        # Strategy 3: Cross-file documentation
        match = self._match_cross_file_docs(function)
        if match:
            return match

        return None

    def _match_imported_function(
        self, function: ParsedFunction
    ) -> Optional[MatchedPair]:
        """Match a function that might be imported from another module."""
        # Get the module containing this function
        module_path = self.module_resolver.resolve_module_path(function.file_path)
        if not module_path:
            return None

        module_info = self.state.module_tree.get(module_path)
        if not module_info:
            return None

        # Check imports for this function name
        for import_stmt in module_info.imports:
            if function.signature.name in import_stmt.imported_names:
                # Function is imported, find source
                source_module = self.module_resolver.resolve_import(
                    import_stmt, module_path
                )
                if source_module:
                    # Look up the original function
                    locations = self.function_registry.find_function(
                        function.signature.name, source_module
                    )
                    if locations:
                        # Found original function
                        # Note: In a full implementation, we'd load and match
                        # the original function's documentation here
                        return self._create_import_match(
                            function, locations[0], import_stmt
                        )

        return None

    def _match_moved_function(self, function: ParsedFunction) -> Optional[MatchedPair]:
        """Match a function that moved to a different file."""
        # Look for functions with same name in other modules
        module_path = self.module_resolver.resolve_module_path(function.file_path)

        moved_location = self.function_registry.find_moved_function(
            function, module_path or ""
        )

        if moved_location:
            # Compare signatures to confirm it's the same function
            confidence = self._calculate_signature_similarity(function, moved_location)

            if confidence > 0.8:
                self.stats["moved_functions"] += 1
                return self._create_moved_match(function, moved_location)

        return None

    def _match_cross_file_docs(self, function: ParsedFunction) -> Optional[MatchedPair]:
        """Find documentation in a different file."""
        # For now, this is a placeholder - will be implemented in Chunk 4
        # This would search for documentation in:
        # - Module-level docstrings
        # - Package __init__.py files
        # - Related documentation files
        return None

    def _calculate_signature_similarity(
        self, function: ParsedFunction, location: FunctionLocation
    ) -> float:
        """Calculate similarity between function signatures."""
        # In full implementation, load the other function and compare
        # For now, return a placeholder
        # This would compare parameters, return types, decorators
        return 0.85  # Placeholder

    def _create_import_match(
        self,
        function: ParsedFunction,
        original_location: FunctionLocation,
        import_stmt: ImportStatement,
    ) -> MatchedPair:
        """Create a match for an imported function."""
        return MatchedPair(
            function=function,
            docstring=function.docstring,  # Will be enhanced in full impl
            match_type=MatchType.CONTEXTUAL,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=1.0,  # Same name
                location_score=0.8,  # Different file but imported
                signature_similarity=0.9,
            ),
            match_reason=f"Imported from {original_location.canonical_module}",
        )

    def _create_moved_match(
        self, function: ParsedFunction, new_location: FunctionLocation
    ) -> MatchedPair:
        """Create a match for a moved function."""
        return MatchedPair(
            function=function,
            docstring=function.docstring,
            match_type=MatchType.CONTEXTUAL,
            confidence=MatchConfidence(
                overall=0.85,
                name_similarity=1.0,
                location_score=0.7,  # Different location
                signature_similarity=0.85,
            ),
            match_reason=f"Function moved from {new_location.canonical_module}",
        )

    def _discover_python_files(self) -> List[str]:
        """Discover all Python files in the project."""
        python_files = []

        for path in self.project_root.rglob("*.py"):
            # Skip common exclusions
            if any(part.startswith(".") for part in path.parts):
                continue
            if any(part in {"__pycache__", "venv", "env"} for part in path.parts):
                continue

            python_files.append(str(path))

        return python_files

    def _resolve_all_imports(self) -> None:
        """Resolve all import chains in the project."""
        # This is where we'd build the complete import graph
        # For now, track basic stats
        for module_info in self.state.module_tree.values():
            self.stats["imports_resolved"] += len(module_info.imports)
