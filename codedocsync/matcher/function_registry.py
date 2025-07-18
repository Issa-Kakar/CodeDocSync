from typing import Dict, List, Optional, Set
import logging

from .contextual_models import FunctionLocation, ModuleInfo
from ..parser import ParsedFunction

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """Global registry of all functions in the project."""

    def __init__(self):
        self.functions: Dict[str, FunctionLocation] = {}
        # Secondary indices for efficient lookup
        self.by_module: Dict[str, Set[str]] = {}
        self.by_name: Dict[str, Set[str]] = {}

    def register_function(
        self, function: ParsedFunction, module_info: ModuleInfo
    ) -> str:
        """
        Register a function and return its canonical name.

        Returns:
            Canonical function name (module.path.function_name)
        """
        canonical_name = f"{module_info.module_path}.{function.signature.name}"

        # Check for duplicates
        if canonical_name in self.functions:
            logger.warning(f"Duplicate function registration: {canonical_name}")

        # Create function location
        location = FunctionLocation(
            canonical_module=module_info.module_path,
            function_name=function.signature.name,
            line_number=function.line_number,
            is_exported=function.signature.name in module_info.exports
            or not function.signature.name.startswith("_"),
        )

        # Register in main index
        self.functions[canonical_name] = location

        # Update secondary indices
        if module_info.module_path not in self.by_module:
            self.by_module[module_info.module_path] = set()
        self.by_module[module_info.module_path].add(canonical_name)

        if function.signature.name not in self.by_name:
            self.by_name[function.signature.name] = set()
        self.by_name[function.signature.name].add(canonical_name)

        return canonical_name

    def find_function(
        self, name: str, hint_module: Optional[str] = None
    ) -> List[FunctionLocation]:
        """
        Find functions by name, optionally with module hint.

        Args:
            name: Function name to search for
            hint_module: Optional module to prioritize

        Returns:
            List of matching functions, sorted by relevance
        """
        # Exact canonical name match
        if "." in name and name in self.functions:
            return [self.functions[name]]

        # Search by function name
        matches = []
        canonical_names = self.by_name.get(name, set())

        for canonical_name in canonical_names:
            location = self.functions[canonical_name]
            matches.append(location)

        # Sort by relevance if hint provided
        if hint_module and matches:

            def relevance_score(loc: FunctionLocation) -> float:
                if loc.canonical_module == hint_module:
                    return 0.0  # Exact match
                elif hint_module.startswith(loc.canonical_module + "."):
                    return 0.5  # Parent module
                elif loc.canonical_module.startswith(hint_module + "."):
                    return 0.5  # Child module
                else:
                    return 1.0  # Different module tree

            matches.sort(key=relevance_score)

        return matches

    def find_moved_function(
        self, old_function: ParsedFunction, old_module: str
    ) -> Optional[FunctionLocation]:
        """
        Find a function that might have moved to a different module.

        Uses signature similarity to identify likely moves.
        """
        name = old_function.signature.name
        candidates = self.by_name.get(name, set())

        if not candidates:
            return None

        # Filter out the original location
        candidates = {c for c in candidates if not c.startswith(old_module + ".")}

        if len(candidates) == 1:
            # Only one candidate, likely moved here
            canonical_name = next(iter(candidates))
            return self.functions[canonical_name]

        # Multiple candidates - need signature comparison
        # This will be implemented in the matcher
        return None

    def get_module_functions(self, module_path: str) -> List[FunctionLocation]:
        """Get all functions in a module."""
        canonical_names = self.by_module.get(module_path, set())
        return [self.functions[name] for name in canonical_names]
