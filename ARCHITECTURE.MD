# CodeDocSync Architecture

**Version: 0.2.0**

## Project Overview

CodeDocSync is a CLI tool that automatically detects inconsistencies between Python code and its documentation. It uses AST parsing, rule-based analysis, and semantic search to identify when functions have changed but their docstrings haven't, preventing documentation drift that causes developer confusion and AI tool inefficiency.

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

1. **Data Flow**: AST Parser → Docstring Parser → Matcher → Analyzer (never reverse)
2. **Model Ownership**: Each module owns its models - never create models outside their modules
3. **Function Naming**: parse_* for parsing, analyze_* for analysis, match_* for matching
4. **Error Handling**: Every public function must handle None inputs gracefully
5. **Testing**: Each component must have tests BEFORE moving to next component

## Component Specifications

### Parser Module (`parser/`)

**Purpose**: Extract code structure and documentation from Python files with high accuracy.

**Core Data Models**:
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
    parameters: List[DocParam]     # Parsed parameter documentation
    returns: Optional[DocReturn]   # Return value documentation
    raises: List[DocRaises]        # Documented exceptions
    raw_text: str                 # Original docstring text
Python Parameter Rules:

Parameter order: positional-only, positional-or-keyword, *args, keyword-only, **kwargs
args.defaults contains defaults for ALL positional parameters
The * separator is syntax, not a parameter name

Error Handling Requirements:

Syntax errors: Parse up to error line, log warning, continue
Encoding issues: Try UTF-8 first, fallback to latin-1
Missing docstrings: Return None, not empty string

Matcher Module (matcher/)
Three-Layer Matching Strategy:

DirectMatcher (90% of cases): Same file, same/similar names, confidence ≥ 0.7
ContextualMatcher (8% of cases): Import-aware, cross-file, moved functions
SemanticMatcher (2% of cases): Embedding-based fallback for renamed functions

Performance Targets:

Direct: <1ms per function
Contextual: <20ms per function
Semantic: <200ms per function (includes embedding)

Analyzer Module (analyzer/)
Two-Phase Analysis:

RuleEngine: Fast structural checks (parameter names, types, counts)
LLMAnalyzer: Semantic understanding for behavioral changes

Storage Module (storage/)
Three-Layer Caching:

File Hash Cache: MD5(content) → parsed AST (disk-based)
Function Cache: function_signature → parsed docstring (LRU in-memory, max 1000)
Embedding Cache: text+model → vector (SQLite + memory LRU)

Key Architectural Decisions
ADR-001: Python 3.10+ Requirement

Decision: Require Python 3.10+
Reason: Match statements, better typing, ast.unparse()

ADR-002: AST Parser Choice

Decision: Built-in ast module
Reason: 2.4x faster than alternatives for batch processing

ADR-003: Minimal RAG with ChromaDB

Decision: ChromaDB for embedding storage
Reason: Zero-config, handles 1M embeddings, built for AI

ADR-004: Docstring Parser Library

Decision: docstring_parser>=0.16
Reason: Actively maintained, supports all formats

Critical Data Flows
Parsing Pipeline
python# 1. AST Parser extracts structure
ParsedFunction(
    signature=FunctionSignature(...),
    docstring=RawDocstring(raw_text, line_number),
    ...
)

# 2. Docstring Parser converts to structured format
ParsedFunction(
    signature=FunctionSignature(...),
    docstring=ParsedDocstring(format, parameters, returns, ...),
    ...
)
Matching Pipeline
python# Functions flow through matchers in order
DirectMatcher.match() → confidence >= 0.7 → STOP
     ↓ (if < 0.7)
ContextualMatcher.match() → confidence >= 0.7 → STOP
     ↓ (if < 0.7)
SemanticMatcher.match() → final fallback
Data Type Guidelines
Union Type Handling
python# ALWAYS check type before access
if isinstance(func.docstring, RawDocstring):
    text = func.docstring.raw_text  # NOT str(docstring)!
elif isinstance(func.docstring, ParsedDocstring):
    text = func.docstring.raw_text
Critical Interfaces
python@dataclass
class ParsedFunction:
    signature: FunctionSignature
    docstring: Optional[Union[RawDocstring, ParsedDocstring]]
    file_path: str
    line_number: int

@dataclass
class MatchedPair:
    function: ParsedFunction
    documentation: Optional[ParsedDocstring]
    confidence: MatchConfidence
    match_type: MatchType
    match_reason: str
    docstring: Optional[Union[RawDocstring, ParsedDocstring]] = None
Integration Requirements
External Dependencies

LLM: OpenAI/Anthropic via litellm
Embeddings: OpenAI
Vector Store: ChromaDB (persistent, project-specific collections)
CLI: Typer + Rich for terminal UI

Performance Contracts

Small file (<100 lines): <10ms parsing
Medium project (100 files): <30s full analysis
Large project (1000 files): <5 minutes full analysis
Memory usage: <500MB for 10k functions

Common Pitfalls

Never use str() on custom objects - Always access specific attributes
Check instance type before Union access - Prevents AttributeError
Use content hash for cache keys - Not file paths (prevents cache poisoning)
Respect confidence thresholds - Never downgrade high-confidence matches

Module Public APIs
parser/init.py

parse_python_file(), parse_python_file_lazy()
ParsedFunction, FunctionSignature, FunctionParameter
DocstringParser, ParsedDocstring

matcher/init.py

DirectMatcher, ContextualMatcher, SemanticMatcher
UnifiedMatchingFacade
MatchResult, MatchedPair, MatchConfidence

analyzer/init.py (upcoming)

RuleEngine, LLMAnalyzer
AnalysisResult, InconsistencyIssue

### Analyzer Module Data Models

```python
@dataclass
class InconsistencyIssue:
    """Single documentation inconsistency."""
    issue_type: str  # See ISSUE_TYPES constant
    severity: str    # 'critical', 'high', 'medium', 'low'
    description: str # Human-readable description
    suggestion: str  # Actionable fix suggestion
    line_number: int # Line where issue occurs
    confidence: float = 1.0  # 0.0-1.0, determines if LLM needed
    details: Dict[str, Any] = field(default_factory=dict)  # Additional context

@dataclass
class AnalysisResult:
    """Complete analysis result for a matched pair."""
    matched_pair: MatchedPair
    issues: List[InconsistencyIssue]
    used_llm: bool
    analysis_time_ms: float
    cache_hit: bool = False

# Rule categories that MUST be implemented
RULE_CATEGORIES = {
    "structural": ["parameter_names", "parameter_types", "parameter_count", "return_type"],
    "completeness": ["missing_params", "missing_returns", "missing_raises", "undocumented_kwargs"],
    "consistency": ["type_mismatches", "default_mismatches", "parameter_order"],
}

# Issue type constants
ISSUE_TYPES = {
    "parameter_name_mismatch": "critical",
    "parameter_missing": "critical",
    "parameter_type_mismatch": "high",
    "return_type_mismatch": "high",
    "missing_raises": "medium",
    "parameter_order_different": "medium",
    "description_outdated": "medium",
    "example_invalid": "low",
}
Add Integration Contract:
python# Analyzer module public API
from codedocsync.analyzer import analyze_matched_pair, RuleEngine, LLMAnalyzer

async def analyze_matched_pair(
    pair: MatchedPair,
    config: AnalysisConfig,
    cache: Optional[AnalysisCache] = None
) -> AnalysisResult:
    """Main entry point for analysis."""
    pass
Add Caching Strategy:
markdown### Analysis Caching Strategy

1. **Rule Results Cache**: In-memory LRU (no persistence needed)
2. **LLM Results Cache**: SQLite with function signature hash as key
3. **Cache Key Generation**: MD5(function_signature + docstring + model)
4. **Cache Invalidation**: On function or docstring change
