from dataclasses import dataclass, field
from typing import List, Dict, Set, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..parser import ParsedFunction, ParsedDocstring


class ImportType(Enum):
    """Types of Python import statements."""

    STANDARD = "standard"  # import module
    FROM = "from"  # from module import name
    RELATIVE = "relative"  # from . import module
    WILDCARD = "wildcard"  # from module import *


@dataclass
class ImportStatement:
    """Represents a parsed import statement."""

    import_type: ImportType
    module_path: str  # 'os.path' or '.utils'
    imported_names: List[str]  # ['join', 'exists'] or ['*']
    aliases: Dict[str, str]  # {'DataFrame': 'pd.DataFrame'}
    line_number: int
    level: int = 0  # Relative import level (number of dots)

    def __post_init__(self):
        """Validate import statement."""
        if self.import_type == ImportType.WILDCARD and len(self.imported_names) != 1:
            raise ValueError("Wildcard imports must have exactly one '*' entry")
        if self.level > 0 and self.import_type != ImportType.RELATIVE:
            raise ValueError("Only relative imports can have level > 0")


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    module_path: str  # 'mypackage.submodule'
    file_path: str  # '/path/to/mypackage/submodule.py'
    imports: List[ImportStatement] = field(default_factory=list)
    exports: Set[str] = field(default_factory=set)  # __all__ or public names
    functions: Dict[str, "FunctionLocation"] = field(default_factory=dict)
    is_package: bool = False  # True if __init__.py

    def get_canonical_name(self, function_name: str) -> str:
        """Get fully qualified function name."""
        return f"{self.module_path}.{function_name}"


@dataclass
class FunctionLocation:
    """Tracks where a function is defined and how it's accessed."""

    canonical_module: str  # Original definition module
    function_name: str
    line_number: int
    import_paths: Set[str] = field(default_factory=set)  # All ways to import
    is_exported: bool = True  # False if name starts with _


@dataclass
class CrossFileMatch:
    """A match between a function and documentation in different files."""

    function: "ParsedFunction"
    documentation: "ParsedDocstring"
    match_reason: str  # "imported_function", "moved_function", etc
    import_chain: List[str]  # Steps to resolve the import
    confidence: float

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )


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
