Chunk 1: Data Models and Project Setup
I'm building CodeDocSync, a tool that detects inconsistencies between Python code and documentation. I'm on Days 3-4 of the implementation, focusing on the AST Parser.

Please implement the core data models for the parser module according to this specification:

1. Create the file structure:
   - codedocsync/parser/__init__.py
   - codedocsync/parser/ast_parser.py
   - codedocsync/utils/errors.py

2. In codedocsync/parser/ast_parser.py, implement these dataclasses with validation:

```python
@dataclass
class FunctionParameter:
    """Single function parameter with validation."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True

    # Add validation in __post_init__ for parameter names and type annotations
    # Parameter names must match Python identifier rules
    # Type annotations should validate against common patterns (List[], Dict[], Optional[], Union[])

@dataclass
class FunctionSignature:
    """Complete function signature with validation."""
    name: str
    parameters: List[FunctionParameter] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)

    # Add to_string() method that converts to readable signature string

@dataclass
class ParsedFunction:
    """Represents a parsed function with all metadata."""
    signature: FunctionSignature
    docstring: Optional[str] = None  # Raw docstring for now
    file_path: str
    line_number: int
    end_line_number: int
    source_code: str

In codedocsync/utils/errors.py, create custom exceptions:

pythonclass ParsingError(Exception):
    """Base exception for parsing errors."""
    def __init__(self, message: str, recovery_hint: Optional[str] = None):
        self.message = message
        self.recovery_hint = recovery_hint
        super().__init__(self.message)
Include proper imports, type hints, and follow the validation patterns shown in the technical spec.

## **Chunk 2: Core AST Parsing Logic**
Now implement the core AST parsing functionality in codedocsync/parser/ast_parser.py:

Add the main parsing function:

pythondef parse_python_file(file_path: str) -> List[ParsedFunction]:
    """Parse Python file with comprehensive error handling.

    Performance targets:
    - Small file (<100 lines): <10ms
    - Medium file (100-1000 lines): <50ms
    - Large file (>1000 lines): <200ms
    """

Implement the helper function to extract function details:

pythondef _extract_function(
    node: ast.FunctionDef,
    file_path: str,
    source_content: str
) -> ParsedFunction:
    """Extract function details with error recovery.

    Handles:
    - Functions without docstrings
    - Complex decorators
    - Nested functions
    - Lambda expressions in defaults
    """

Add a helper to extract function signatures:

pythondef _extract_signature(node: ast.FunctionDef) -> FunctionSignature:
    """Extract signature information from AST node."""
The implementation should:

Use Python's built-in ast module
Extract all function metadata (name, parameters, return types, decorators)
Handle both regular and async functions
Extract docstrings using ast.get_docstring()
Track line numbers for error reporting
Include the function's source code

Follow the error handling patterns from the technical spec, including try-except blocks for file operations and AST parsing.

## **Chunk 3: Error Handling and Edge Cases**
Enhance the AST parser with comprehensive error handling and edge case support:

Update parse_python_file to handle:

Syntax errors with partial parsing (parse up to the error line)
File not found with helpful recovery hints
Permission errors
Encoding issues (default to UTF-8)
Empty files
Files with only imports/comments


Enhance _extract_function to handle:

Functions with no parameters
Functions with *args and **kwargs
Complex type annotations (Union, Optional, List, Dict)
Default values that are function calls or complex expressions
Nested functions (include parent context)
Lambda functions
Property decorators (@property, @setter)
Class methods and static methods


Add these utility functions:

pythondef _get_decorator_names(decorators: List[ast.expr]) -> List[str]:
    """Extract decorator names handling both simple and complex decorators."""

def _extract_parameter(arg: ast.arg, defaults: List, index: int) -> FunctionParameter:
    """Extract parameter information with proper default handling."""

def _get_annotation_string(annotation: ast.expr) -> Optional[str]:
    """Convert AST annotation to string representation."""

Add logging throughout:

Use Python's logging module
Log warnings for syntax errors and partial parses
Log info for successful parses with timing
Log errors with full context



Include performance timing using time.time() and log parse duration for benchmarking.

## **Chunk 4: Comprehensive Test Suite**
Create a comprehensive test suite in codedocsync/tests/test_ast_parser.py:

Test basic parsing functionality:

pythondef test_parse_simple_function():
    """Test parsing a basic function."""

def test_parse_function_with_parameters():
    """Test function with various parameter types."""

def test_parse_async_function():
    """Test async function parsing."""

def test_parse_decorated_function():
    """Test functions with decorators."""

Test edge cases:

pythondef test_parse_empty_file():
    """Empty file should return empty list."""

def test_parse_syntax_error():
    """Syntax errors should attempt partial parsing."""

def test_parse_unicode_content():
    """Test files with unicode in docstrings and comments."""

def test_parse_complex_annotations():
    """Test complex type annotations (Union, Optional, etc)."""

Test error handling:

pythondef test_file_not_found():
    """Should raise ParsingError with recovery hint."""

def test_permission_denied():
    """Should handle permission errors gracefully."""

def test_malformed_decorators():
    """Should handle invalid decorator syntax."""

Test performance:

pythondef test_parsing_performance_small_file():
    """Small files should parse in <10ms."""

def test_parsing_performance_large_file():
    """Large files should parse in <200ms."""

Add fixtures for test data:

python@pytest.fixture
def sample_python_file(tmp_path):
    """Create a temporary Python file for testing."""

@pytest.fixture
def complex_function_file(tmp_path):
    """Create a file with complex function examples."""
Use pytest features like parametrize for testing multiple scenarios and tmp_path for temporary files.

## **Chunk 5: Performance Optimization and Integration**
Finalize the AST parser with performance optimizations and module integration:

Add performance monitoring to ast_parser.py:

pythonimport time
from functools import lru_cache

# Add caching for frequently parsed files
@lru_cache(maxsize=100)
def _get_cached_ast(file_content_hash: str) -> ast.AST:
    """Cache parsed AST trees for repeated analysis."""

Optimize the parsing for large files:

pythondef parse_python_file_lazy(file_path: str) -> Generator[ParsedFunction, None, None]:
    """Lazy parsing using generators for memory efficiency."""

Create the module's public API in codedocsync/parser/init.py:

pythonfrom .ast_parser import parse_python_file, ParsedFunction, FunctionSignature, FunctionParameter
from ..utils.errors import ParsingError

__all__ = ['parse_python_file', 'ParsedFunction', 'FunctionSignature', 'FunctionParameter', 'ParsingError']

Add a CLI command for testing the parser in codedocsync/cli/main.py:

python@app.command()
def parse(
    file: Path = typer.Argument(..., help="Python file to parse"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """Parse a Python file and display extracted functions."""
    try:
        functions = parse_python_file(str(file))
        if json_output:
            # Output as JSON
        else:
            # Pretty print with Rich
    except ParsingError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        if e.recovery_hint:
            console.print(f"[yellow]Hint:[/yellow] {e.recovery_hint}")
        raise typer.Exit(1)

Update the architecture files:


Add entry to CHANGELOG.md for the AST parser implementation
Ensure ARCHITECTURE.md reflects the parser component accurately

Make sure all imports are correct and the module is properly integrated with the project structure.

## **Additional Context for the Agent**

When implementing, remember:
- The AST parser is performance-critical (it's the foundation for all analysis)
- We're using Python's built-in `ast` module as the primary parser (2.4x faster than alternatives)
- Error recovery is important - partial parsing is better than failing completely
- The parser should handle real-world Python code with all its complexity
- This is part of a larger system, so the interfaces must be clean and well-documented

Each chunk builds on the previous one, so implement them in order. The technical spec provides many implementation details and examples that should be followed closely.
