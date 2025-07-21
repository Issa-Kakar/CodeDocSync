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

### CLI Module (`cli/`)

**Purpose**: Provide a clean, maintainable command-line interface organized into logical submodules.

**Structure** (Refactored from single 2191-line main.py):
- `cli/main.py`: App initialization and command registration (~100 lines)
- `cli/analyze.py`: `analyze` and `analyze_function` commands for documentation analysis (~425 lines)
- `cli/match.py`: `match`, `match_contextual`, `match_unified` commands for function matching (~538 lines)
- `cli/parse.py`: `parse` command for AST parsing demonstration (~130 lines)
- `cli/suggest.py`: `suggest` and `suggest_interactive` commands for fix suggestions (~386 lines)
- `cli/cache.py`: `clear_cache` command for cache management (~106 lines)
- `cli/placeholders.py`: `watch` and `check` placeholder commands for future features (~25 lines)
- `cli/formatting.py`: Shared formatting and display utilities (~300 lines)

**Key Design Decisions**:
- Modular structure for maintainability
- Shared formatting utilities to avoid duplication
- Clear separation of concerns by command type
- Backward compatibility maintained with wrapper in original main.py

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

## Poetry Command Execution Guide

### Issue: Poetry commands fail in certain shell environments

**Problem**: When using Poetry in certain environments (particularly Windows Git Bash and some CI/CD environments), the `poetry run` command may fail with various shell-specific errors. This is a known issue with Poetry's shell detection and command execution.

### Universal Solution: Direct Virtual Environment Execution

Instead of using `poetry run`, execute commands directly using the virtual environment's Python interpreter. This method bypasses shell-specific bugs and works universally across all environments.

#### Step-by-Step Guide:

1. **Find Your Virtual Environment Path** (run once per project):
   ```bash
   poetry env info --path
   ```
   This will output a path like:
   - Windows: `C:\Users\username\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12`
   - macOS/Linux: `/Users/username/Library/Caches/pypoetry/virtualenvs/codedocsync-5yfwj9Sn-py3.12`

2. **Construct the Python Executable Path**:
   - Windows: `<path>\Scripts\python.exe`
   - macOS/Linux: `<path>/bin/python`

3. **Execute Commands Using Direct Path**:

   **Instead of:**
   ```bash
   poetry run codedocsync analyze .
   poetry run mypy .
   poetry run pytest
   ```

   **Use:**
   ```bash
   # Windows example
   "C:\Users\username\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe" -m codedocsync analyze .
   "C:\Users\username\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe" -m mypy .
   "C:\Users\username\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe" -m pytest
   ```

#### Setting Up Aliases (Recommended)

For convenience, create shell aliases:

**Windows (Git Bash):**
```bash
# Add to ~/.bashrc
alias pyexec='"C:\Users\username\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe"'
```

**macOS/Linux:**
```bash
# Add to ~/.bashrc or ~/.zshrc
alias pyexec='/Users/username/Library/Caches/pypoetry/virtualenvs/codedocsync-5yfwj9Sn-py3.12/bin/python'
```

Then use:
```bash
pyexec -m codedocsync analyze .
pyexec -m mypy .
pyexec -m pytest
```

#### CI/CD Integration

For CI/CD pipelines, store the virtual environment path in an environment variable:

```yaml
# GitHub Actions example
- name: Get Poetry venv path
  run: echo "VENV_PATH=$(poetry env info --path)" >> $GITHUB_ENV

- name: Run tests
  run: ${{ env.VENV_PATH }}/bin/python -m pytest
```

#### Why This Works

- **No Shell Interpretation**: Commands go directly to Python, avoiding shell-specific parsing issues
- **Universal Compatibility**: Works on all platforms and shell environments
- **Reliable**: Eliminates Poetry's shell detection bugs
- **Performance**: Slightly faster as it skips Poetry's wrapper overhead

#### When to Use This Method

- Always when experiencing `poetry run` failures
- In CI/CD pipelines for reliability
- When working in non-standard shell environments
- For scripts that need to call Poetry commands

### Note on Python Version

This project requires Python 3.10+ due to:
- Modern type annotations (`X | Y` union syntax)
- Parenthesized context managers
- Match statements
- Enhanced error messages

Ensure your Poetry environment uses Python 3.10 or higher by checking:
```bash
"<venv_path>\Scripts\python.exe" --version  # Windows
<venv_path>/bin/python --version             # macOS/Linux
```

## Systematic Code Quality Fix Plan

### Overview

This section documents the systematic approach for fixing linting (ruff) and type checking (mypy) issues in the CodeDocSync codebase. The plan includes automated tracking, prioritized fixing strategies, and helper scripts to manage the process efficiently.

### Current Status (as of 2025-01-21)

- **Initial State**:
  - Ruff: 104 errors (10 auto-fixed, leaving 94)
  - Mypy: 556 errors
- **After Black formatting**:
  - Ruff: 86 errors
- **Main Error Categories**:
  - 29 B904: raise-without-from errors
  - 18 UP038: isinstance with tuple instead of union
  - 17 B007: unused loop control variables
  - 8 B017: assert blind exception
  - Various other minor issues

### Fix Infrastructure

Two helper scripts have been created to manage the fixing process:

#### 1. fix_tracker.py

A tracking script that monitors progress and helps identify which files to fix next:

```python
# Usage examples:
python fix_tracker.py summary              # Show overall progress
python fix_tracker.py mypy                 # Show mypy strategy
python fix_tracker.py file <path>          # Show errors in specific file
python fix_tracker.py mark-fixed <path>    # Mark file as fixed
```

**Features**:
- Groups errors by file and sorts by error count (fix easier files first)
- Groups errors by type to identify patterns
- Tracks which files have been fixed
- Persists progress in `fix_progress.json`
- Provides strategic categorization for mypy fixes

#### 2. fix_project.py

The main execution script for systematic fixes:

```python
# Usage examples:
python fix_project.py --phase status       # Show current status
python fix_project.py --phase all          # Run all auto-fixes
python fix_project.py --phase examples     # Show fix examples
python fix_project.py --phase examples --error-code B904  # Specific fix example
```

**Features**:
- Runs ruff and black with proper error handling
- Shows detailed status with error breakdowns
- Provides fix examples for common error patterns
- Uses direct Python path to avoid Poetry issues on Windows

### Systematic Fix Strategy

#### Phase 1: Ruff Issues (Estimated 4-6 hours)

**Priority Order**:
1. **B904 (raise-without-from)**: Add `from e` to exception chains
2. **UP038 (isinstance)**: Change `isinstance(x, (A, B))` to `isinstance(x, A | B)`
3. **B007 (unused-loop-control-variable)**: Rename to `_` or use the variable
4. **F841 (unused-variable)**: Remove or use the variable
5. **Other minor issues**: Handle case by case

**Fix Patterns**:

```python
# B904: raise-without-from
# Before:
except SomeError as e:
    raise NewError("message")

# After:
except SomeError as e:
    raise NewError("message") from e

# UP038: isinstance union
# Before:
isinstance(x, (str, int))

# After:
isinstance(x, str | int)

# B007: unused loop variable
# Before:
for item in items:
    print("processing")

# After:
for _ in items:
    print("processing")
```

#### Phase 2: Mypy Issues (Estimated 8-10 hours)

**Strategic Categories** (fix in this order):

1. **Critical Path Files** (Priority 1):
   - `codedocsync/__init__.py`
   - `codedocsync/cli/*.py`
   - `codedocsync/analyzer/__init__.py`
   - `codedocsync/suggestions/__init__.py`

2. **Core Functionality** (Priority 2):
   - `codedocsync/analyzer/*.py`
   - `codedocsync/matcher/*.py`
   - `codedocsync/parser/*.py`

3. **Generators and Templates** (Priority 3):
   - `codedocsync/suggestions/generators/*.py`
   - `codedocsync/suggestions/templates/*.py`

4. **Utilities and Tests** (Priority 4):
   - Helper modules
   - Test files (can often be ignored)

**Common Mypy Fixes**:

```python
# Missing return type for __post_init__
# Before:
def __post_init__(self):
    # code

# After:
def __post_init__(self) -> None:
    # code

# Attribute access errors
# Before:
if self.docstring.type_annotation:  # Error: no type_annotation

# After:
if hasattr(self.docstring, 'type_str') and self.docstring.type_str:
    # or fix the actual attribute name

# Union type handling
# Before:
def process(doc: Union[RawDocstring, ParsedDocstring]):
    return doc.summary  # Error: RawDocstring has no summary

# After:
def process(doc: Union[RawDocstring, ParsedDocstring]):
    if isinstance(doc, ParsedDocstring):
        return doc.summary
    else:
        # Handle RawDocstring case
        return None
```

### Execution Workflow

1. **Initial Setup**:
   ```bash
   # Run initial status check
   python fix_project.py --phase status

   # Create initial tracking data
   python fix_tracker.py summary
   ```

2. **Fix Ruff Issues**:
   ```bash
   # For each error type, check examples
   python fix_project.py --phase examples --error-code B904

   # Fix files with fewest errors first
   python fix_tracker.py summary  # See "Next Files to Fix"

   # After fixing a file
   python fix_tracker.py mark-fixed <file_path> ruff
   ```

3. **Fix Mypy Issues**:
   ```bash
   # Get strategic overview
   python fix_tracker.py mypy

   # Fix by category (critical path first)
   # Check specific file errors
   python fix_tracker.py file codedocsync/__init__.py

   # After fixing
   python fix_tracker.py mark-fixed <file_path> mypy
   ```

4. **Validate Progress**:
   ```bash
   # Run full check periodically
   python fix_project.py --phase all

   # Verify with pre-commit
   pre-commit run --all-files
   ```

### Best Practices

1. **Fix One Error Type at a Time**: Focus on one ruff error code across all files before moving to the next
2. **Test After Each Major Change**: Run relevant tests to ensure fixes don't break functionality
3. **Commit Frequently**: Make small, focused commits for each type of fix
4. **Document Non-Obvious Fixes**: Add comments when the fix might not be immediately clear
5. **Skip Complex Test Files**: For mypy, consider adding test files to ignore list if fixes are too complex

### Progress Tracking

The `fix_progress.json` file maintains the current state:

```json
{
  "phase": "ruff",
  "ruff_fixed_files": ["file1.py", "file2.py"],
  "mypy_fixed_files": ["file3.py"],
  "timestamp": "2025-01-21T10:30:00",
  "statistics": {
    "initial_ruff_errors": 104,
    "current_ruff_errors": 86,
    "initial_mypy_errors": 556,
    "current_mypy_errors": 556
  }
}
```

This systematic approach ensures consistent progress while maintaining code quality and functionality throughout the fixing process.
