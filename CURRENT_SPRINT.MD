## CURRENT_SPRINT.MD - Contextual Matcher Implementation

```markdown
# Current Sprint: Contextual Matcher Implementation

## Sprint Overview
The contextual matcher is the second layer of our three-layer matching system, handling ~8% of cases where direct name matching fails. It uses file paths, module structure, and import statements to match functions that have been relocated or are referenced across files. This is our most complex component yet, requiring careful state management and cross-file analysis.

## Critical Context
- The direct matcher only handles same-file, same-name matching
- Contextual matcher must handle: relocated functions, cross-file references, import chains
- Performance is critical: we're now analyzing multiple files simultaneously
- Must integrate seamlessly with existing direct matcher results

## Implementation Chunks

---

## Chunk 1: Foundation and Data Models (Estimated: 3-4 hours)

### Objectives
- Create the contextual matcher data models
- Build the import parsing infrastructure
- Set up the module resolution system

### Required Files to Create
1. `matcher/contextual_models.py` - Core data models for contextual matching
2. `matcher/import_parser.py` - AST-based import extraction
3. `tests/test_contextual_models.py` - Model validation tests
4. `tests/test_import_parser.py` - Import parsing tests

### Detailed Implementation

#### 1. Create `matcher/contextual_models.py`
```python
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import re

class ImportType(Enum):
    """Types of Python import statements."""
    STANDARD = "standard"          # import module
    FROM = "from"                  # from module import name
    RELATIVE = "relative"          # from . import module
    WILDCARD = "wildcard"          # from module import *

@dataclass
class ImportStatement:
    """Represents a parsed import statement."""
    import_type: ImportType
    module_path: str               # 'os.path' or '.utils'
    imported_names: List[str]      # ['join', 'exists'] or ['*']
    aliases: Dict[str, str]        # {'DataFrame': 'pd.DataFrame'}
    line_number: int
    level: int = 0                 # Relative import level (number of dots)

    def __post_init__(self):
        """Validate import statement."""
        if self.import_type == ImportType.WILDCARD and len(self.imported_names) != 1:
            raise ValueError("Wildcard imports must have exactly one '*' entry")
        if self.level > 0 and self.import_type != ImportType.RELATIVE:
            raise ValueError("Only relative imports can have level > 0")

@dataclass
class ModuleInfo:
    """Information about a Python module."""
    module_path: str               # 'mypackage.submodule'
    file_path: str                 # '/path/to/mypackage/submodule.py'
    imports: List[ImportStatement] = field(default_factory=list)
    exports: Set[str] = field(default_factory=set)  # __all__ or public names
    functions: Dict[str, 'FunctionLocation'] = field(default_factory=dict)
    is_package: bool = False       # True if __init__.py

    def get_canonical_name(self, function_name: str) -> str:
        """Get fully qualified function name."""
        return f"{self.module_path}.{function_name}"

@dataclass
class FunctionLocation:
    """Tracks where a function is defined and how it's accessed."""
    canonical_module: str          # Original definition module
    function_name: str
    line_number: int
    import_paths: Set[str] = field(default_factory=set)  # All ways to import
    is_exported: bool = True       # False if name starts with _

@dataclass
class CrossFileMatch:
    """A match between a function and documentation in different files."""
    function: ParsedFunction
    documentation: ParsedDocstring
    match_reason: str              # "imported_function", "moved_function", etc
    import_chain: List[str]        # Steps to resolve the import
    confidence: float

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

@dataclass
class ContextualMatcherState:
    """Global state for contextual matching across files."""
    module_tree: Dict[str, ModuleInfo] = field(default_factory=dict)
    import_graph: Dict[str, Set[str]] = field(default_factory=dict)
    function_registry: Dict[str, FunctionLocation] = field(default_factory=dict)

    def add_module(self, module_info: ModuleInfo) -> None:
        """Add a module to the state."""
        self.module_tree[module_info.module_path] = module_info
        # Update import graph
        for imp in module_info.imports:
            if module_info.module_path not in self.import_graph:
                self.import_graph[module_info.module_path] = set()
            self.import_graph[module_info.module_path].add(imp.module_path)
2. Create matcher/import_parser.py
pythonimport ast
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import logging

from .contextual_models import (
    ImportStatement, ImportType, ModuleInfo, FunctionLocation
)

logger = logging.getLogger(__name__)

class ImportParser:
    """Parses Python imports and module structure."""

    def parse_imports(self, file_path: str) -> Tuple[List[ImportStatement], Set[str]]:
        """
        Parse all imports and exports from a Python file.

        Returns:
            Tuple of (imports, exports)
            - imports: List of ImportStatement objects
            - exports: Set of exported names (from __all__ or public names)
        """
        imports = []
        exports = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(self._parse_import(node))
                elif isinstance(node, ast.ImportFrom):
                    imports.extend(self._parse_import_from(node))

            # Extract exports (__all__)
            exports = self._extract_exports(tree)

            # If no explicit exports, find all public names
            if not exports:
                exports = self._extract_public_names(tree)

            return imports, exports

        except Exception as e:
            logger.error(f"Failed to parse imports from {file_path}: {e}")
            return [], set()

    def _parse_import(self, node: ast.Import) -> List[ImportStatement]:
        """Parse 'import' statement."""
        imports = []
        for alias in node.names:
            imports.append(ImportStatement(
                import_type=ImportType.STANDARD,
                module_path=alias.name,
                imported_names=[],
                aliases={alias.asname: alias.name} if alias.asname else {},
                line_number=node.lineno
            ))
        return imports

    def _parse_import_from(self, node: ast.ImportFrom) -> List[ImportStatement]:
        """Parse 'from ... import ...' statement."""
        if node.module is None:
            # Relative import like 'from . import something'
            module_path = '.' * node.level
            import_type = ImportType.RELATIVE
        else:
            module_path = node.module
            import_type = ImportType.RELATIVE if node.level > 0 else ImportType.FROM

        # Handle wildcards
        imported_names = []
        aliases = {}

        for alias in node.names:
            if alias.name == '*':
                import_type = ImportType.WILDCARD
                imported_names = ['*']
                break
            else:
                imported_names.append(alias.name)
                if alias.asname:
                    aliases[alias.asname] = alias.name

        return [ImportStatement(
            import_type=import_type,
            module_path=module_path,
            imported_names=imported_names,
            aliases=aliases,
            line_number=node.lineno,
            level=node.level
        )]

    def _extract_exports(self, tree: ast.AST) -> Set[str]:
        """Extract __all__ exports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        # Extract list of strings
                        if isinstance(node.value, ast.List):
                            return {
                                elt.value for elt in node.value.elts
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                            }
        return set()

    def _extract_public_names(self, tree: ast.AST) -> Set[str]:
        """Extract all public function and class names."""
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):
                    names.add(node.name)
        return names

    def build_module_info(self, file_path: str, module_path: str) -> ModuleInfo:
        """Build complete module information."""
        imports, exports = self.parse_imports(file_path)

        # Determine if it's a package
        path = Path(file_path)
        is_package = path.name == '__init__.py'

        return ModuleInfo(
            module_path=module_path,
            file_path=file_path,
            imports=imports,
            exports=exports,
            is_package=is_package
        )
Testing Requirements for Chunk 1
Create comprehensive tests that validate:

Import parsing for all Python import types
Export detection from all and public names
Module info building with proper error handling
Data model validation and constraints

Critical Implementation Notes

Import parser MUST handle syntax errors gracefully
Always validate that module paths are valid Python identifiers
Track line numbers for all imports for debugging
Handle encoding issues (try UTF-8, fallback to latin-1)

Success Criteria

All import types are correctly parsed
Module structure is accurately represented
Tests cover edge cases (empty files, syntax errors, circular imports)
Performance: <10ms per file for import parsing


Chunk 2: Module Resolution and Function Registry (Estimated: 4-5 hours)
Objectives

Build the module resolution system
Create the global function registry
Implement import chain resolution

Required Files to Create

matcher/module_resolver.py - Module and import resolution logic
matcher/function_registry.py - Global function tracking
tests/test_module_resolver.py - Resolution testing
tests/test_function_registry.py - Registry testing

Detailed Implementation
1. Create matcher/module_resolver.py
pythonimport os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

from .contextual_models import ModuleInfo, ImportStatement, ImportType
from .import_parser import ImportParser

logger = logging.getLogger(__name__)

class ModuleResolver:
    """Resolves module paths and import chains."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.import_parser = ImportParser()
        self.module_cache: Dict[str, ModuleInfo] = {}
        self._python_paths = self._calculate_python_paths()

    def _calculate_python_paths(self) -> List[Path]:
        """Calculate Python module search paths."""
        paths = [self.project_root]

        # Add parent directories that contain __init__.py
        current = self.project_root
        while current.parent != current:
            if (current.parent / '__init__.py').exists():
                paths.append(current.parent)
                current = current.parent
            else:
                break

        return paths

    def resolve_module_path(self, file_path: str) -> Optional[str]:
        """
        Convert file path to module path.

        Example:
            /project/src/utils/helpers.py -> src.utils.helpers
            /project/src/utils/__init__.py -> src.utils
        """
        file_path = Path(file_path).resolve()

        # Find the base path this file is relative to
        for base_path in self._python_paths:
            try:
                relative_path = file_path.relative_to(base_path)

                # Convert to module path
                parts = list(relative_path.parts)

                # Handle __init__.py
                if parts[-1] == '__init__.py':
                    parts = parts[:-1]
                elif parts[-1].endswith('.py'):
                    parts[-1] = parts[-1][:-3]
                else:
                    continue  # Not a Python file

                module_path = '.'.join(parts)
                return module_path

            except ValueError:
                # file_path is not relative to this base_path
                continue

        return None

    def resolve_import(
        self,
        import_stmt: ImportStatement,
        current_module: str
    ) -> Optional[str]:
        """
        Resolve an import statement to absolute module path.

        Args:
            import_stmt: The import statement to resolve
            current_module: The module containing the import

        Returns:
            Absolute module path or None if can't resolve
        """
        if import_stmt.import_type == ImportType.RELATIVE:
            # Handle relative imports
            if import_stmt.level == 0:
                return import_stmt.module_path

            # Go up 'level' packages from current module
            parts = current_module.split('.')
            if import_stmt.level > len(parts):
                logger.warning(
                    f"Relative import goes beyond top-level package: "
                    f"{import_stmt} in {current_module}"
                )
                return None

            # Go up the required levels
            base_parts = parts[:-import_stmt.level]

            if import_stmt.module_path and import_stmt.module_path != '.':
                # from ..package import something
                base_parts.append(import_stmt.module_path.lstrip('.'))

            return '.'.join(base_parts) if base_parts else None

        else:
            # Absolute import
            return import_stmt.module_path

    def build_import_chain(
        self,
        target_function: str,
        from_module: str,
        to_module: str
    ) -> Optional[List[str]]:
        """
        Build the import chain to access a function from another module.

        Returns:
            List of import steps, or None if not accessible
        """
        if from_module == to_module:
            return [target_function]  # Same module, direct access

        # Check direct imports
        from_info = self.module_cache.get(from_module)
        if not from_info:
            return None

        for import_stmt in from_info.imports:
            resolved = self.resolve_import(import_stmt, from_module)
            if resolved == to_module:
                # Direct import found
                if import_stmt.import_type == ImportType.WILDCARD:
                    return [target_function]
                elif target_function in import_stmt.imported_names:
                    # Check for alias
                    for alias, real_name in import_stmt.aliases.items():
                        if real_name == target_function:
                            return [alias]
                    return [target_function]
                elif not import_stmt.imported_names:
                    # import module
                    alias = import_stmt.aliases.get(to_module, to_module)
                    return [f"{alias}.{target_function}"]

        # TODO: Check indirect imports (through parent packages)
        return None

    def find_module_file(self, module_path: str) -> Optional[str]:
        """Find the file path for a module."""
        # Convert module path to possible file paths
        path_parts = module_path.split('.')

        for base_path in self._python_paths:
            # Try as regular module
            file_path = base_path / Path(*path_parts).with_suffix('.py')
            if file_path.exists():
                return str(file_path)

            # Try as package
            package_path = base_path / Path(*path_parts) / '__init__.py'
            if package_path.exists():
                return str(package_path)

        return None
2. Create matcher/function_registry.py
pythonfrom typing import Dict, List, Optional, Set
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
        self,
        function: ParsedFunction,
        module_info: ModuleInfo
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
            is_exported=function.signature.name in module_info.exports or
                       not function.signature.name.startswith('_')
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
        self,
        name: str,
        hint_module: Optional[str] = None
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
        if '.' in name and name in self.functions:
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
                elif hint_module.startswith(loc.canonical_module + '.'):
                    return 0.5  # Parent module
                elif loc.canonical_module.startswith(hint_module + '.'):
                    return 0.5  # Child module
                else:
                    return 1.0  # Different module tree

            matches.sort(key=relevance_score)

        return matches

    def find_moved_function(
        self,
        old_function: ParsedFunction,
        old_module: str
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
        candidates = {
            c for c in candidates
            if not c.startswith(old_module + '.')
        }

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
Testing Requirements for Chunk 2
Create tests that validate:

Module path resolution for various file structures
Import resolution including relative imports
Function registration and lookup
Import chain building

Critical Implementation Notes

Module resolver MUST handle missing init.py gracefully
Function registry must track all functions, even private ones
Import resolution must handle circular imports without infinite loops
Always preserve the canonical location of functions

Success Criteria

Module paths correctly resolved for all project structures
Import chains accurately built
Function registry efficiently searches by name and module
Performance: <20ms for import resolution per function


Chunk 3: Contextual Matcher Core Implementation (Estimated: 5-6 hours)
Objectives

Implement the main ContextualMatcher class
Build matching logic for relocated and imported functions
Integrate with direct matcher results

Required Files to Create

matcher/contextual_matcher.py - Main contextual matching logic
tests/test_contextual_matcher.py - Core functionality tests
Update matcher/__init__.py to export new components

Detailed Implementation
1. Create matcher/contextual_matcher.py
pythonimport logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import MatchResult, MatchedPair, MatchType, MatchConfidence
from .contextual_models import (
    ContextualMatcherState, CrossFileMatch, ModuleInfo, ImportStatement
)
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
            "moved_functions": 0
        }

    def analyze_project(
        self,
        python_files: Optional[List[str]] = None
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
        direct_match_result: Optional[MatchResult] = None
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
                m.function.signature.name for m in direct_match_result.matches
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
            unmatched_docs=unmatched_docs
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
            module_info = self.import_parser.build_module_info(
                file_path, module_path
            )

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

    def _find_contextual_match(
        self,
        function: ParsedFunction
    ) -> Optional[MatchedPair]:
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
        self,
        function: ParsedFunction
    ) -> Optional[MatchedPair]:
        """Match a function that might be imported from another module."""
        # Get the module containing this function
        module_path = self.module_resolver.resolve_module_path(
            function.file_path
        )
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

    def _match_moved_function(
        self,
        function: ParsedFunction
    ) -> Optional[MatchedPair]:
        """Match a function that moved to a different file."""
        # Look for functions with same name in other modules
        module_path = self.module_resolver.resolve_module_path(
            function.file_path
        )

        moved_location = self.function_registry.find_moved_function(
            function, module_path or ""
        )

        if moved_location:
            # Compare signatures to confirm it's the same function
            confidence = self._calculate_signature_similarity(
                function, moved_location
            )

            if confidence > 0.8:
                self.stats["moved_functions"] += 1
                return self._create_moved_match(function, moved_location)

        return None

    def _calculate_signature_similarity(
        self,
        function: ParsedFunction,
        location: 'FunctionLocation'
    ) -> float:
        """Calculate similarity between function signatures."""
        # In full implementation, load the other function and compare
        # For now, return a placeholder
        # This would compare parameters, return types, decorators
        return 0.85  # Placeholder

    def _create_import_match(
        self,
        function: ParsedFunction,
        original_location: 'FunctionLocation',
        import_stmt: ImportStatement
    ) -> MatchedPair:
        """Create a match for an imported function."""
        return MatchedPair(
            function=function,
            docstring=function.docstring,  # Will be enhanced in full impl
            match_type=MatchType.CONTEXTUAL,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=1.0,  # Same name
                location_score=0.8,   # Different file but imported
                signature_similarity=0.9
            ),
            match_reason=f"Imported from {original_location.canonical_module}"
        )

    def _create_moved_match(
        self,
        function: ParsedFunction,
        new_location: 'FunctionLocation'
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
                signature_similarity=0.85
            ),
            match_reason=f"Function moved from {new_location.canonical_module}"
        )

    def _discover_python_files(self) -> List[str]:
        """Discover all Python files in the project."""
        python_files = []

        for path in self.project_root.rglob("*.py"):
            # Skip common exclusions
            if any(part.startswith('.') for part in path.parts):
                continue
            if any(part in {'__pycache__', 'venv', 'env'} for part in path.parts):
                continue

            python_files.append(str(path))

        return python_files

    def _resolve_all_imports(self) -> None:
        """Resolve all import chains in the project."""
        # This is where we'd build the complete import graph
        # For now, track basic stats
        for module_info in self.state.module_tree.values():
            self.stats["imports_resolved"] += len(module_info.imports)
Testing Requirements for Chunk 3
Create tests that cover:

Basic contextual matching scenarios
Import resolution and matching
Function relocation detection
Integration with direct matcher results

Critical Implementation Notes

MUST integrate smoothly with direct matcher results
Never downgrade high-confidence direct matches
Track performance metrics for analysis
Handle missing imports gracefully

Success Criteria

Correctly identifies imported functions
Detects relocated functions with high accuracy
Integrates with direct matching results
Performance: <100ms for 100-file project analysis


Chunk 4: Cross-File Documentation Matching (Estimated: 4-5 hours)
Objectives

Implement cross-file documentation detection
Handle module and class-level docstrings
Support documentation in init.py files

Required Files to Create/Update

Update matcher/contextual_matcher.py with cross-file logic
Create matcher/doc_location_finder.py - Find docs in various locations
Create tests/test_cross_file_matching.py

Detailed Implementation
1. Create matcher/doc_location_finder.py
pythonimport ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..parser import ParsedDocstring, DocstringParser

logger = logging.getLogger(__name__)

class DocLocationFinder:
    """Finds documentation in non-standard locations."""

    def __init__(self):
        self.docstring_parser = DocstringParser()
        self._cache: Dict[str, List[ParsedDocstring]] = {}

    def find_module_docs(self, module_path: str) -> Dict[str, ParsedDocstring]:
        """
        Find all docstrings in a module that might document functions.

        Returns:
            Dict mapping potential function names to their docs
        """
        if module_path in self._cache:
            return self._cache[module_path]

        docs = {}

        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=module_path)

            # Check module docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                # Parse for function documentation sections
                func_docs = self._extract_function_docs_from_module(module_doc)
                docs.update(func_docs)

            # Check class docstrings for method documentation
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        method_docs = self._extract_method_docs_from_class(
                            class_doc, node.name
                        )
                        docs.update(method_docs)

            self._cache[module_path] = docs
            return docs

        except Exception as e:
            logger.error(f"Failed to find docs in {module_path}: {e}")
            return {}

    def find_package_docs(self, package_path: str) -> Dict[str, ParsedDocstring]:
        """
        Find documentation in package __init__.py files.

        Some projects document all module functions in __init__.py
        """
        init_file = Path(package_path) / '__init__.py'
        if not init_file.exists():
            return {}

        return self.find_module_docs(str(init_file))

    def _extract_function_docs_from_module(
        self,
        module_docstring: str
    ) -> Dict[str, ParsedDocstring]:
        """
        Extract function documentation from module docstring.

        Looks for sections like:
        Functions:
        ----------
        function_name: Description
            Full documentation here
        """
        docs = {}

        # Common patterns for function documentation in module docstrings
        patterns = [
            r"Functions:\s*\n[-=]+\s*\n(.*?)(?=\n\n|\Z)",
            r"Available Functions:\s*\n(.*?)(?=\n\n|\Z)",
            r"Public API:\s*\n(.*?)(?=\n\n|\Z)"
        ]

        # This is simplified - in reality we'd have more sophisticated parsing
        # For now, we'll rely on the docstring parser to handle structured content

        return docs

    def _extract_method_docs_from_class(
        self,
        class_docstring: str,
        class_name: str
    ) -> Dict[str, ParsedDocstring]:
        """
        Extract method documentation from class docstring.

        Some styles document all methods in the class docstring.
        """
        docs = {}

        # Look for method documentation sections
        # This would need more sophisticated parsing in practice

        return docs

    def find_related_docs(
        self,
        function_name: str,
        module_path: str,
        search_radius: int = 2
    ) -> List[Tuple[str, ParsedDocstring]]:
        """
        Search for documentation in related files.

        Args:
            function_name: Name of the function
            module_path: Current module path
            search_radius: How many directories up to search

        Returns:
            List of (file_path, docstring) tuples
        """
        related_docs = []

        # Search in parent directories up to search_radius
        current_path = Path(module_path).parent

        for _ in range(search_radius):
            # Check __init__.py in current directory
            init_file = current_path / '__init__.py'
            if init_file.exists():
                docs = self.find_package_docs(str(current_path))
                if function_name in docs:
                    related_docs.append((str(init_file), docs[function_name]))

            # Check README or docs files
            for doc_file in ['README.md', 'DOCS.md', f'{function_name}.md']:
                doc_path = current_path / doc_file
                if doc_path.exists():
                    # Parse markdown for function documentation
                    # This would need a markdown parser in practice
                    pass

            # Move up one directory
            if current_path.parent == current_path:
                break
            current_path = current_path.parent

        return related_docs
2. Update matcher/contextual_matcher.py - Add cross-file methods
Add these methods to the ContextualMatcher class:
pythondef _match_cross_file_docs(
    self,
    function: ParsedFunction
) -> Optional[MatchedPair]:
    """Find documentation in a different file."""

    # Skip if function already has good documentation
    if function.docstring and hasattr(function.docstring, 'summary'):
        if len(function.docstring.summary) > 20:  # Has substantial docs
            return None

    # Initialize doc finder
    if not hasattr(self, 'doc_finder'):
        self.doc_finder = DocLocationFinder()

    # Search strategies:

    # 1. Check module-level documentation
    module_docs = self.doc_finder.find_module_docs(function.file_path)
    if function.signature.name in module_docs:
        return self._create_cross_file_match(
            function,
            module_docs[function.signature.name],
            function.file_path,
            "module docstring"
        )

    # 2. Check parent package documentation
    module_path = self.module_resolver.resolve_module_path(function.file_path)
    if module_path and '.' in module_path:
        parent_package = '.'.join(module_path.split('.')[:-1])
        parent_file = self.module_resolver.find_module_file(parent_package)

        if parent_file:
            package_docs = self.doc_finder.find_package_docs(
                str(Path(parent_file).parent)
            )
            if function.signature.name in package_docs:
                return self._create_cross_file_match(
                    function,
                    package_docs[function.signature.name],
                    parent_file,
                    "package __init__.py"
                )

    # 3. Check related documentation files
    related_docs = self.doc_finder.find_related_docs(
        function.signature.name,
        function.file_path
    )

    if related_docs:
        # Use the first matching documentation
        doc_file, docstring = related_docs[0]
        return self._create_cross_file_match(
            function,
            docstring,
            doc_file,
            "related documentation file"
        )

    return None

def _create_cross_file_match(
    self,
    function: ParsedFunction,
    docstring: ParsedDocstring,
    doc_location: str,
    location_type: str
) -> MatchedPair:
    """Create a match for cross-file documentation."""

    # Calculate confidence based on documentation quality
    doc_quality = self._assess_doc_quality(docstring, function)

    return MatchedPair(
        function=function,
        docstring=docstring,
        match_type=MatchType.CONTEXTUAL,
        confidence=MatchConfidence(
            overall=0.7 * doc_quality,  # Lower base confidence for cross-file
            name_similarity=1.0,  # Exact name match required
            location_score=0.5,   # Different file
            signature_similarity=doc_quality  # Based on parameter matching
        ),
        match_reason=f"Documentation found in {location_type} at {doc_location}"
    )

def _assess_doc_quality(
    self,
    docstring: ParsedDocstring,
    function: ParsedFunction
) -> float:
    """Assess how well documentation matches the function."""
    score = 1.0

    # Check parameter coverage
    if hasattr(docstring, 'parameters'):
        doc_params = {p.name for p in docstring.parameters}
        func_params = {p.name for p in function.signature.parameters}

        # Penalize missing or extra parameters
        missing = func_params - doc_params
        extra = doc_params - func_params

        score -= 0.1 * len(missing)
        score -= 0.05 * len(extra)

    # Check if return type is documented
    if function.signature.return_type and not docstring.returns:
        score -= 0.1

    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))
Testing Requirements for Chunk 4
Create comprehensive tests for:

Module-level documentation extraction
Package init.py documentation
Cross-file matching scenarios
Documentation quality assessment

Critical Implementation Notes

Cache parsed documentation to avoid repeated parsing
Handle various documentation styles in module docstrings
Don't match if function already has good documentation
Lower confidence for cross-file matches

Success Criteria

Finds documentation in module and class docstrings
Correctly extracts from init.py files
Quality assessment accurately scores matches
Performance: <50ms per file for doc extraction


Chunk 5: Integration and Performance Optimization (Estimated: 3-4 hours)
Objectives

Integrate contextual matcher with the full system
Optimize performance for large codebases
Add comprehensive error handling and logging
Create the facade for easy usage

Required Files to Create/Update

Create matcher/contextual_facade.py - High-level interface
Update matcher/__init__.py - Export all components
Create tests/test_contextual_integration.py
Update main CLI to support contextual matching

Detailed Implementation
1. Create matcher/contextual_facade.py
pythonimport logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import MatchResult
from .contextual_matcher import ContextualMatcher
from .direct_matcher import DirectMatcher
from ..parser import IntegratedParser, ParsedFunction
from ..utils.config import CodeDocSyncConfig

logger = logging.getLogger(__name__)

class ContextualMatchingFacade:
    """
    High-level interface for contextual matching.

    Combines direct and contextual matching for best results.
    """

    def __init__(self, config: Optional[CodeDocSyncConfig] = None):
        self.config = config or CodeDocSyncConfig()
        self.stats = {
            "total_time": 0.0,
            "parsing_time": 0.0,
            "direct_matching_time": 0.0,
            "contextual_matching_time": 0.0,
            "files_processed": 0
        }

    def match_project(
        self,
        project_path: str,
        use_cache: bool = True
    ) -> MatchResult:
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
        parser = IntegratedParser(cache_enabled=use_cache)
        direct_matcher = DirectMatcher(
            fuzzy_threshold=self.config.matching.fuzzy_threshold
        )
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
            all_functions,
            direct_result
        )
        self.stats["contextual_matching_time"] += time.time() - context_match_start

        # Add statistics to result
        self.stats["total_time"] = time.time() - start_time
        final_result.metadata = {
            "performance": self.stats,
            "matcher_stats": {
                "direct": direct_matcher.get_stats(),
                "contextual": contextual_matcher.stats
            }
        }

        return final_result

    def match_file(
        self,
        file_path: str,
        project_path: Optional[str] = None
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

    def _discover_python_files(self, project_path: Path) -> List[Path]:
        """Discover Python files respecting exclusions."""
        exclusions = set(self.config.ignore.paths)

        python_files = []
        for path in project_path.rglob("*.py"):
            # Check exclusions
            if any(
                self._matches_pattern(str(path), pattern)
                for pattern in exclusions
            ):
                continue

            python_files.append(path)

        return python_files

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches exclusion pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

    def print_summary(self) -> None:
        """Print performance summary."""
        print("\n=== Performance Summary ===")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Parsing: {self.stats['parsing_time']:.2f}s")
        print(f"Direct matching: {self.stats['direct_matching_time']:.2f}s")
        print(f"Contextual matching: {self.stats['contextual_matching_time']:.2f}s")
2. Update CLI for contextual matching
Add to cli/main.py:
python@app.command()
def match_contextual(
    path: Path = typer.Argument(..., help="Project directory to analyze"),
    output_format: str = typer.Option("terminal", "--format", "-f"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    show_stats: bool = typer.Option(False, "--stats", help="Show performance statistics")
):
    """
    Perform contextual matching on a project.

    Uses both direct and contextual matching for best results.
    """
    # Load configuration
    if config:
        config_obj = CodeDocSyncConfig.from_yaml(str(config))
    else:
        config_obj = CodeDocSyncConfig()

    # Create facade and run matching
    facade = ContextualMatchingFacade(config_obj)

    with Progress() as progress:
        task = progress.add_task("Analyzing project...", total=None)

        result = facade.match_project(str(path))

        progress.update(task, completed=True)

    # Format output
    if output_format == "json":
        output = _format_json_result(result)
    else:
        output = _format_terminal_result(result)

    # Save or print
    if output_file:
        output_file.write_text(output)
        console.print(f"✅ Results saved to {output_file}")
    else:
        console.print(output)

    # Show statistics if requested
    if show_stats:
        facade.print_summary()
Testing Requirements for Chunk 5
Create integration tests that verify:

Full project analysis workflow
Performance meets requirements
Configuration is properly applied
Results combine direct and contextual matches correctly

Critical Implementation Notes

Cache project context between runs when possible
Use parallel processing for parsing multiple files
Ensure memory usage stays reasonable for large projects
Provide clear progress feedback for long operations

Success Criteria

Complete project matching in <30s for 1000 files
Memory usage <500MB for large projects
Seamless integration of direct and contextual results
Clear performance metrics and debugging info


Testing Strategy
Each chunk must have comprehensive tests BEFORE moving to the next chunk:

Unit Tests: Test each component in isolation
Integration Tests: Test component interactions
Performance Tests: Verify performance requirements
Edge Case Tests: Handle errors, empty files, circular imports

Common Pitfalls to Avoid

Import Resolution: Don't assume imports are always valid
Circular Dependencies: Detect and break cycles
Memory Leaks: Clear caches periodically
Performance: Use lazy loading and caching aggressively
Error Recovery: Never let one file failure stop analysis

Success Metrics
The contextual matcher is complete when:

It correctly identifies 95%+ of relocated functions
It resolves import chains accurately
Performance meets all targets (<100ms for 100 files)
It integrates seamlessly with direct matcher
All tests pass with >90% coverage
