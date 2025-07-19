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

## Contextual Matcher Architecture

### Import Resolution System
The contextual matcher MUST track and resolve imports to understand cross-file relationships:

```python
# Import tracking requirements:
1. Standard imports: import module, from module import name
2. Relative imports: from . import module, from ..package import name
3. Aliased imports: import module as alias, from module import name as alias
4. Wildcard imports: from module import * (track with warning)
Module Resolution Rules
python# Resolution priority order:
1. Explicit imports in the same file
2. Module-level __all__ exports
3. Package __init__.py exports
4. Parent package imports (for relative imports)
5. Standard library modules (ignore for matching)
Function Relocation Detection
Functions may move between files during refactoring. Detection strategy:
python# Relocation indicators (in priority order):
1. Same function name + similar signature (>80% parameter match)
2. Import statement changes pointing to new location
3. Git history showing file moves (optional enhancement)
4. Deprecation decorators with forwarding imports
Cross-File Reference Tracking
python@dataclass
class CrossFileReference:
    source_file: str          # File containing the reference
    source_line: int          # Line number of reference
    target_function: str      # Referenced function name
    import_chain: List[str]   # Import path to reach function
    confidence: float         # Confidence in reference resolution
Performance Constraints for Contextual Matching

Import parsing: <10ms per file
Reference resolution: <20ms per function
Module tree building: <100ms for 100 files
Memory: <10MB per 1000 imports

State Management for Multi-File Analysis
The contextual matcher MUST maintain state across files:
pythonclass ContextualMatcherState:
    # Module-level state
    module_tree: Dict[str, ModuleInfo]      # Full module hierarchy
    import_graph: Dict[str, Set[str]]       # Import dependencies
    function_index: Dict[str, FunctionInfo]  # Global function registry

    # File-level caches
    import_cache: Dict[str, List[Import]]   # Parsed imports per file
    export_cache: Dict[str, Set[str]]       # Exported names per module
Error Recovery for Import Resolution
python# Import resolution failures MUST be handled gracefully:
1. Circular imports: Track visited modules, break cycles
2. Missing modules: Log warning, continue with partial info
3. Dynamic imports: Mark as unresolved, use heuristics
4. Import errors: Fallback to string matching
Critical Invariants for Contextual Matching

Never modify AST: Contextual matcher is read-only
Preserve function identity: A function has ONE canonical location
Import precedence: Local imports override global imports
Module boundaries: Respect all and private conventions
Performance first: Skip deep analysis if confidence is already high

Integration Points with Direct Matcher
python# Contextual matcher enhances direct matcher results:
1. If direct match confidence < 0.8, try contextual
2. Validate direct matches against import structure
3. Upgrade match confidence based on import evidence
4. Never downgrade a high-confidence direct match

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

## Semantic Matching Architecture

### Embedding Strategy
The semantic matcher uses embeddings as a last-resort fallback (~2% of cases) when direct and contextual matching fail. Design principles:

```python
# Embedding content strategy (what gets embedded):
1. Primary embedding: Function signature + first line of docstring
   - Provides semantic understanding of function purpose
   - Example: "def authenticate_user(username: str, password: str) -> User: Authenticate a user with credentials"

2. Extended embedding (for complex cases): Signature + full docstring summary
   - Used when primary embedding yields low confidence matches
   - Includes parameter descriptions for better semantic understanding

3. Source code embeddings: NEVER embed function source code
   - Security risk (may contain secrets)
   - Too noisy for similarity matching
   - Violates separation of concerns
Vector Store Contract
ChromaDB integration requirements:
python# Collection management rules:
1. One collection per project (named by project hash)
2. Metadata stored with each embedding:
   - function_name: str
   - module_path: str
   - signature_hash: str (for change detection)
   - timestamp: float
   - embedding_model: str

# Persistence strategy:
- Local disk persistence in .codedocsync_cache/chroma/
- Collections expire after 30 days of no access
- Maximum 10GB total storage before oldest collections are pruned
Semantic Matching Thresholds
python# Confidence scoring for semantic matches:
SEMANTIC_SIMILARITY_THRESHOLDS = {
    "high_confidence": 0.85,      # Very likely the same function
    "medium_confidence": 0.75,     # Possibly the same function
    "low_confidence": 0.65,        # Needs manual verification
    "no_match": 0.65              # Below this, don't even suggest
}

# Match validation rules:
1. Semantic matches MUST have signature similarity > 0.5
2. Parameter count difference must be <= 3
3. Never match if module paths are >2 levels apart
4. Downgrade confidence by 0.1 for each naming convention difference
Performance Requirements for Semantic Operations
python# Semantic matcher performance targets:
- Embedding generation: <100ms per function (batched)
- Similarity search: <50ms per query
- Full semantic matching: <200ms per function
- Memory usage: <500MB for 10k embeddings
- Cache hit rate: >90% for unchanged functions
Embedding Cache Architecture
python@dataclass
class EmbeddingCacheEntry:
    text_hash: str           # SHA256 of input text
    embedding: List[float]   # The embedding vector
    model: str              # Model used to generate
    timestamp: float        # For TTL management
    hit_count: int = 0      # For LRU eviction

# Cache hierarchy:
1. In-memory LRU cache: 1000 most recent embeddings
2. Disk cache: SQLite for persistence
3. ChromaDB: Long-term storage with search capability
Integration with Matching Pipeline
The semantic matcher integrates as the final fallback:
python# Matching pipeline order (STRICT):
1. DirectMatcher.match_functions() -> MatchResult
   - If confidence >= 0.8, STOP (use direct match)

2. ContextualMatcher.match_with_context(functions, direct_result) -> MatchResult
   - If confidence >= 0.7, STOP (use contextual match)

3. SemanticMatcher.match_with_embeddings(functions, [direct_result, contextual_result]) -> MatchResult
   - Only called for functions with no good match (confidence < 0.7)
   - Never downgrades existing matches

# Critical invariant:
# Each matcher MUST preserve high-confidence matches from previous matchers
Semantic Matcher State Management
python@dataclass
class SemanticMatcherState:
    vector_store: ChromaDB.Collection
    embedding_cache: EmbeddingCache
    model_config: EmbeddingModelConfig
    performance_stats: Dict[str, float]

    # State constraints:
    # - Vector store must be initialized before first use
    # - Cache must be warmed from disk on startup
    # - Stats must be persisted every 100 operations
Error Recovery for External Services
python# Embedding service failures:
1. Primary: OpenAI text-embedding-3-small
2. Fallback 1: OpenAI text-embedding-ada-002
3. Fallback 2: Local sentence-transformers model
4. Final fallback: Skip semantic matching entirely

# Vector store failures:
1. Connection timeout: Retry with exponential backoff (max 3 attempts)
2. Storage full: Prune oldest 20% of embeddings
3. Corruption: Rebuild collection from embedding cache
4. Complete failure: Fall back to in-memory only mode
