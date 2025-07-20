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
6. **Code Quality**: All code must pass pre-commit hooks before committing (see Code Quality section)

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
Reason: Match statements, better typing, ast.unparse(), parenthesized context managers

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

### LLM Analyzer Specifications

**Purpose**: Provide semantic analysis for complex inconsistencies that rule-based checks cannot detect.

**Core Responsibilities**:
1. Behavioral consistency checking (does the code do what the docs say?)
2. Example validation (are code examples in docstrings valid?)
3. Edge case documentation (are important edge cases documented?)
4. Version/deprecation info validation
5. Complex type consistency beyond structural checks

**LLM Integration Architecture**:
```python
@dataclass
class LLMAnalysisRequest:
    function: ParsedFunction
    docstring: ParsedDocstring
    analysis_type: str  # 'behavior', 'examples', 'edge_cases', etc.
    context: Dict[str, Any]  # Related functions, imports, etc.

@dataclass
class LLMAnalysisResponse:
    issues: List[InconsistencyIssue]
    raw_response: str
    model_used: str
    tokens_used: int
    response_time_ms: float
Prompt Engineering Principles:

Structured Output: Always request JSON with specific schema
Context Limiting: Include only relevant code context (max 2000 tokens)
Temperature 0: Deterministic outputs for consistency
Confidence Scoring: LLM must provide confidence for each issue
Example-Driven: Include examples of correct analysis in prompts

LLM Performance Requirements:

Response time: <2s per function (p95)
Cache hit rate: >80% after warm-up
Token usage: <1000 per analysis (optimize prompts)
Fallback time: <100ms to switch to rule-only analysis

Caching Strategy for LLM:
Cache Key: MD5(function_signature + docstring + analysis_type + model)
Cache Storage: SQLite with JSON responses
Cache TTL: 7 days (configurable)
Cache Size Limit: 1GB (rotate oldest entries)
Integration Flow
Rule Engine Analysis (confidence > 0.9) → Skip LLM → Return results
                    ↓ (confidence ≤ 0.9)
        Check LLM Cache → Hit → Return cached + rule results
                    ↓ Miss
        Call LLM with context → Merge with rule results → Cache → Return
                    ↓ (timeout/error)
        Return rule-only results with degraded confidence
Critical LLM Invariants

Never Block on LLM: Always have rule-based fallback
Cache Everything: LLM calls are expensive, cache aggressively
Validate Output: LLM output must match expected schema or reject
Context Window: Never exceed 4000 tokens total (prompt + completion)
Rate Limiting: Implement token bucket with 10 req/s limit

## Code Quality and Pre-commit Hooks

This project uses strict code quality enforcement through pre-commit hooks that run automatically before each commit. **All code must pass these checks before being committed.**

### Pre-commit Hook Configuration

**IMPORTANT**: All tool versions are synchronized between `pyproject.toml` and `.pre-commit-config.yaml` to ensure consistent code quality enforcement.

The project uses the following pre-commit hooks (defined in `.pre-commit-config.yaml`):

1. **Standard Pre-commit Hooks** (v4.6.0):
   - `check-yaml`: Validates YAML file syntax
   - `end-of-file-fixer`: Ensures files end with a newline
   - `trailing-whitespace`: Removes trailing whitespace

2. **Ruff** (v0.12.3):
   - `ruff --fix`: Fast Python linter with auto-fix capabilities
   - `ruff-format`: Fast Python code formatter (alternative to Black)

3. **Black** (v25.1.0):
   - `black`: Python code formatter for consistent style

4. **Mypy** (v1.17.0):
   - `mypy`: Static type checker for Python code

### Development Workflow

**CRITICAL**: When writing code, follow this workflow to avoid commit failures:

1. **Write code following these style guidelines**:
   - Use double quotes for strings (Ruff/Black default)
   - Keep line length ≤ 88 characters (Black default)
   - Use trailing commas in multi-line structures
   - Follow PEP 8 naming conventions
   - End files with a single newline
   - Remove trailing whitespace

2. **Before committing, run pre-commit manually** (optional but recommended):
   ```bash
   pre-commit run --all-files
   ```

3. **Commit normally**:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

4. **If pre-commit hooks fail**:
   - Review the changes made by the hooks
   - Add the modified files: `git add -A`
   - Commit again: `git commit -m "Your commit message"`

### Common Pre-commit Issues and Solutions

**Issue**: Ruff/Black formatting conflicts
- **Solution**: Let both tools run, they are configured to work together

**Issue**: Files missing final newline
- **Solution**: Automatic fix by `end-of-file-fixer`, just re-commit

**Issue**: Trailing whitespace
- **Solution**: Automatic fix by `trailing-whitespace`, just re-commit

**Issue**: YAML syntax errors
- **Solution**: Fix YAML syntax manually, then re-commit

### Bypassing Pre-commit (Emergency Only)

**ONLY use in emergencies**:
```bash
git commit --no-verify -m "Emergency commit message"
```

### Code Style Guidelines for Consistency

To minimize pre-commit hook failures, follow these guidelines when writing code:

**String Formatting**:
```python
# Good
description = "This is a string"
f"Function {name} has {count} parameters"

# Avoid
description = 'This is a string'  # Single quotes
```

**Line Length and Formatting**:
```python
# Good
def long_function_name(
    parameter_one: str,
    parameter_two: int,
    parameter_three: bool = True,
) -> Dict[str, Any]:
    pass

# Good
result = some_function(
    arg1="value1",
    arg2="value2",
    arg3="value3",
)
```

**Import Organization**:
```python
# Standard library
import json
import re
from typing import List, Dict, Any

# Third party
import pytest
from rich.console import Console

# Local imports
from .models import InconsistencyIssue
from .prompt_templates import format_prompt
```

**Docstring Style**:
```python
def function_name(param: str) -> bool:
    """
    Brief description of function.

    Args:
        param: Description of parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When validation fails
    """
    pass
```

### Editor Configuration

**Recommended VS Code settings** (`.vscode/settings.json`):
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true
}
```

### Suggestion Generator Specifications

**Purpose**: Generate actionable, copy-paste ready fix suggestions for detected inconsistencies.

**Core Responsibilities**:
1. Transform issues into properly formatted docstrings
2. Support multiple docstring styles (Google, NumPy, Sphinx, REST)
3. Provide confidence scores for suggestions
4. Generate minimal diffs for easy review
5. Format suggestions for different output contexts (terminal, JSON, HTML)

**Data Models**:
```python
@dataclass
class Suggestion:
    """A single fix suggestion with formatting."""
    original_text: str          # Current docstring/code
    suggested_text: str         # Fixed version
    diff_preview: str          # Minimal diff view
    confidence: float          # 0.0-1.0 confidence
    style: str                 # Docstring style used
    copy_paste_ready: bool     # If suggestion can be directly applied

@dataclass
class SuggestionContext:
    """Context needed to generate suggestions."""
    issue: InconsistencyIssue
    function: ParsedFunction
    docstring: Optional[ParsedDocstring]
    project_style: str  # From config
    surrounding_code: Optional[str]  # For context-aware suggestions
Integration Pattern:
InconsistencyIssue → SuggestionGenerator → Formatted Suggestion
                           ↓
                    Style Formatter
                           ↓
                    Output Formatter
Critical Invariants:

Suggestions must preserve all valid existing documentation
Generated docstrings must be syntactically valid Python
Style must be consistent with project configuration
Suggestions must be actionable (no vague instructions)
Terminal output must be ANSI-safe, JSON must escape properly
