# CodeDocSync Architecture

**Version: 0.1.1**

## Project Overview

CodeDocSync is a CLI tool that automatically detects inconsistencies between Python code and its documentation. It uses AST parsing, rule-based analysis, and minimal semantic search to identify when functions have changed but their docstrings haven't, preventing documentation drift that causes developer confusion and AI tool inefficiency.

## High-Level Architecture
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   CLI       │────▶│   Parser     │────▶│  Matcher    │
│  (Typer)    │     │  (AST/DS)    │     │  (3-Layer)  │
└─────────────┘     └──────────────┘     └─────────────┘
│                     │
▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Storage    │◀────│   Analyzer   │◀────│   Cache     │
│ (ChromaDB)  │     │  (LLM/Rules) │     │  (3-Layer)  │
└─────────────┘     └──────────────┘     └─────────────┘
│
▼
┌──────────────┐
│   Reports    │
│(Term/JSON/HTML)
└──────────────┘

## Implementation Invariants (NEVER VIOLATE)

1. **Data Flow**: AST Parser → Docstring Parser → Matcher (never reverse)
2. **Model Ownership**:
   - parser/ owns all parsing models
   - analyzer/ owns all analysis models
   - Never create models outside their modules
3. **Function Naming**: parse_* for parsing, analyze_* for analysis, match_* for matching
4. **Error Handling**: Every public function must handle None inputs gracefully
5. **Testing**: Each chunk must have tests BEFORE moving to next chunk

## Component Specifications

### Parser Module (`parser/`)

**Purpose**: Extract code structure and documentation from Python files with high accuracy.

**Subcomponents**:
- `ast_parser.py`: Parse Python files into structured function data (~450 lines)
- `docstring_parser.py`: Parse docstrings in multiple formats (~380 lines)
- `code_analyzer.py`: Extract behavioral patterns and exceptions (~290 lines)

**Key Data Models**:
```python
@dataclass
class FunctionParameter:
    name: str                      # Valid Python identifier
    type_annotation: Optional[str] # String representation of type
    default_value: Optional[str]   # String representation of default
    is_required: bool             # True if no default value
    kind: ParameterKind           # POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, etc.

@dataclass
class ParsedDocstring:
    format: str                    # 'google', 'numpy', 'sphinx', 'rest'
    summary: str                   # First line(s) of docstring
    description: Optional[str]     # Extended description
    parameters: List[DocParam]     # Parsed parameter documentation
    returns: Optional[DocReturn]   # Return value documentation
    raises: List[DocRaises]        # Documented exceptions
    examples: List[str]           # Code examples if present
    raw_text: str                 # Original docstring text
Python Parameter Rules (Critical for correctness):

Parameter order in signatures MUST be: positional-only, positional-or-keyword, *args, keyword-only, **kwargs
args.defaults contains defaults for ALL positional parameters (both positional-only AND regular)
Keyword-only parameters have their own kw_defaults list with None for required params
The * separator is syntax, not a parameter name

Error Handling Requirements:

Syntax errors: Parse up to error line, log warning, continue
Encoding issues: Try UTF-8 first, fallback to latin-1
Missing docstrings: Return None, not empty string
Malformed docstrings: Best-effort parsing with logged warnings

Cache Strategy
Three-Layer Caching (prevents memory leaks):

File Hash Cache: MD5(content) → parsed AST (disk-based)
Function Cache: function_signature → parsed docstring (LRU in-memory, max 1000)
Embedding Cache: text+model → vector (disk-based with memory LRU layer)

Cache Keys: Use content hash, NOT file path + content (prevents duplication)
Docstring Format Detection
Auto-detection Priority:

Check for explicit format markers (e.g., :param: for Sphinx)
Analyze structure patterns (e.g., Args: section for Google)
Use project-wide default from config
Fallback to 'google' style

Format Indicators:

Google: Sections like Args:, Returns:, Raises:
NumPy: Sections with underlines like Parameters\n----------
Sphinx: reST directives like :param name:, :returns:
REST: Similar to Sphinx but with different conventions

Key Architectural Decisions
ADR-001: Python 3.10+ Requirement
Context: Need modern features for type hints and performance
Decision: Require Python 3.10+
Reason: Match statements, better typing, ast.unparse() for defaults
ADR-002: AST Parser Choice
Context: Need fast, reliable Python code parsing
Decision: Primary: built-in ast, Secondary: tree-sitter for watch mode
Reason: AST module 2.4x faster for batch processing; tree-sitter adds incremental parsing
ADR-003: Minimal RAG with ChromaDB
Context: Need semantic fallback for renamed functions (2% of cases)
Decision: ChromaDB for embedding-based matching
Reason: Zero-config setup, handles 1M embeddings, built for AI applications
ADR-004: Docstring Parser Library
Context: Need robust multi-format docstring parsing
Decision: Use docstring_parser library with custom enhancements
Reason: Actively maintained, supports all major formats, extensible
ADR-005: Parameter Validation Strategy
Context: Prevent bugs from invalid parameter names or types
Decision: Validate at data model creation with regex patterns
Reason: Fail fast with clear errors rather than propagate invalid data
Integration Points

External APIs: OpenAI/Anthropic via litellm for semantic analysis
Libraries: docstring_parser>=0.16 for parsing, ast for code analysis
File System: Watch mode monitoring, Git integration for incremental analysis
CI/CD: GitHub Actions, GitLab CI, pre-commit hooks

Code Quality Standards
Type Safety: All functions handling AST nodes must use proper Union types and None checks. Example:
pythondef process_node(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> ParsedFunction:
    # Handle both sync and async functions
Validation First: Validate data at boundaries. Use Pydantic models or manual validation in __post_init__.
Error Context: All errors must include recovery hints and context:
pythonraise ParsingError(
    f"Failed to parse {file_path}",
    recovery_hint="Check file encoding or syntax",
    line_number=error_line
)
Performance Targets:

Small file (<100 lines): <10ms
Medium file (100-1000 lines): <50ms
Large file (>1000 lines): <200ms
Docstring parsing: <5ms per docstring

AI Development Guidelines

No Component Invention: Only implement components listed in this document
Test-First for Edge Cases: Write tests before implementing complex parsing logic
Explicit Over Implicit: When in doubt, be explicit about behavior
Incremental Progress: Each chunk should be testable independently

## Critical Interfaces

### ParsedFunction (from AST parser)
```python
@dataclass
class ParsedFunction:
    signature: FunctionSignature
    docstring: Optional[Union[RawDocstring, ParsedDocstring]]  # <-- Critical!
    file_path: str
    line_number: int

## Data Handling Guidelines (CRITICAL)

### Union Type Handling
When working with Union types, ALWAYS check the actual type before accessing attributes:

```python
# WRONG - This caused bugs!
if func.docstring:
    text = str(func.docstring)  # NO! Returns object representation

# CORRECT
if func.docstring:
    if isinstance(func.docstring, RawDocstring):
        text = func.docstring.raw_text
    elif isinstance(func.docstring, ParsedDocstring):
        text = func.docstring.raw_text
Serialization Rules

Never use str() on custom objects for serialization
Always access the specific attribute containing the data
For JSON serialization, create explicit serialization methods

Example:
pythondef _serialize_docstring(docstring: Optional[Union[RawDocstring, ParsedDocstring]]) -> Optional[Dict[str, Any]]:
    if docstring is None:
        return None

    if isinstance(docstring, RawDocstring):
        return {
            "type": "raw",
            "text": docstring.raw_text,  # NOT str(docstring)!
            "line_number": docstring.line_number
        }
    elif isinstance(docstring, ParsedDocstring):
        return {
            "type": "parsed",
            "format": docstring.format,
            "summary": docstring.summary,
            "raw_text": docstring.raw_text
        }
Component Pipeline & Data Flow
Complete Pipeline
1. AST Parser extracts:
   - FunctionSignature (parameters, types, decorators)
   - RawDocstring (raw text + basic metadata)

2. Docstring Parser converts:
   - RawDocstring → ParsedDocstring (structured data)

3. Matcher finds relationships:
   - Direct: Same name, same file
   - Contextual: Import/module aware
   - Semantic: Embedding similarity

4. Analyzer checks consistency:
   - Rule Engine: Fast structural checks
   - LLM: Semantic understanding
Matcher Component Contract
The matcher MUST:

Accept List[ParsedFunction] as input
Return List[MatchedPair] where each pair links function to its documentation
Handle cases where documentation exists in different locations (class docstrings, module docstrings)
Provide confidence scores for each match

Performance Contracts
Component performance requirements:

Matcher (Direct): <1ms per function
Matcher (Fuzzy): <5ms per function
Matcher (Semantic): <50ms per function (includes embedding time)
Memory: <100MB for 10,000 functions

Common Pitfalls & Prevention
AST Line Number Calculation
WRONG: Assuming docstring is always at node.lineno + 1
RIGHT: Use ast.get_docstring() which handles multi-line definitions correctly
Type Checking Before Access
ALWAYS check instance type before accessing Union type attributes:
python# Every function handling Union types MUST have explicit type checks
if isinstance(obj, SpecificType):
    # Now safe to access SpecificType attributes
Caching Keys
WRONG: Using mutable objects or file paths as cache keys
RIGHT: Use content hash (MD5/SHA) for cache keys to prevent cache poisoning



## Existing Interfaces (Must Honor)

### From parser/ast_parser.py
```python
# These exist and cannot be changed:
def parse_python_file(file_path: str) -> List[ParsedFunction]
def parse_python_file_lazy(file_path: str) -> Generator[ParsedFunction, None, None]

@dataclass
class ParsedFunction:
    signature: FunctionSignature
    docstring: Optional[RawDocstring]  # Will accept ParsedDocstring after integration
    file_path: str
    line_number: int
    end_line_number: int
    source_code: str


    Integration Contract
When docstring parser is complete, modify ParsedFunction.docstring to accept:
Union[RawDocstring, ParsedDocstring]
Module Public APIs (What's Exposed)
parser/init.py exports:

parse_python_file, parse_python_file_lazy
ParsedFunction, FunctionSignature, FunctionParameter
ParsingError (base exception)
