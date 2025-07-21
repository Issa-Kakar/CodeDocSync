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
6. **Code Quality**: All code must pass pre-commit hooks before committing

## Component Specifications

### CLI Module (`cli/`)
**Structure** (Refactored from single 2191-line main.py):
- `cli/main.py`: App initialization and command registration
- `cli/analyze.py`: Analysis commands
- `cli/match.py`: Matching commands
- `cli/parse.py`: Parsing demonstration
- `cli/suggest.py`: Fix suggestions
- `cli/cache.py`: Cache management
- `cli/formatting.py`: Shared formatting utilities

### Parser Module (`parser/`)
**Core Data Models**:
```python
@dataclass
class ParsedFunction:
    signature: FunctionSignature
    docstring: Optional[Union[RawDocstring, ParsedDocstring]]
    file_path: str
    line_number: int

@dataclass
class FunctionParameter:
    name: str
    type_annotation: Optional[str]
    default_value: Optional[str]
    is_required: bool
    kind: ParameterKind
```

Key Rules:
- Parameter order: positional-only, positional-or-keyword, *args, keyword-only, **kwargs
- Syntax errors: Parse up to error line, log warning, continue
- Missing docstrings: Return None, not empty string

### Matcher Module (`matcher/`)
Three-Layer Strategy:
- DirectMatcher (90%): Same file, confidence ≥ 0.7
- ContextualMatcher (8%): Cross-file, import-aware
- SemanticMatcher (2%): Embedding-based fallback

Performance Targets:
- Direct: <1ms per function
- Contextual: <20ms per function
- Semantic: <200ms per function

### Analyzer Module (`analyzer/`)
Two-Phase Analysis:
- RuleEngine: Fast structural checks (<5ms)
- LLMAnalyzer: Semantic understanding (<2s)

Core Models:
```python
@dataclass
class InconsistencyIssue:
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    suggestion: str
    line_number: int
    confidence: float = 1.0
```

### Suggestion Generator (`suggestions/`)
Key Components:

- Template system for multiple docstring styles
- Issue-specific generators (parameters, returns, exceptions, behavior, examples)
- Smart merging for partial updates
- Multi-format output (terminal, JSON)

### Storage Module (`storage/`)
Three-Layer Caching:
- File Hash Cache: MD5(content) → parsed AST
- Function Cache: LRU in-memory (max 1000)
- Embedding Cache: SQLite + memory LRU

## Critical Interfaces
```python
@dataclass
class MatchedPair:
    function: ParsedFunction
    documentation: Optional[ParsedDocstring]
    confidence: MatchConfidence
    match_type: MatchType
    match_reason: str
    docstring: Optional[Union[RawDocstring, ParsedDocstring]] = None
```

## Poetry Command Execution Guide
Issue: Poetry commands fail in Windows Git Bash and some CI/CD environments
Solution: Use direct virtual environment Python interpreter path.

Find Virtual Environment Path:
```bash
poetry env info --path
```

Execute Commands:
```bash
# Windows
"C:\path\to\venv\Scripts\python.exe" -m codedocsync analyze .

# macOS/Linux
/path/to/venv/bin/python -m codedocsync analyze .

```

Setup Alias (recommended):
```bash
# Add to ~/.bashrc or equivalent
alias pyexec='"C:\path\to\venv\Scripts\python.exe"'
```

**Important**: This project requires Python 3.10+ for modern syntax features. The local development environment needs Python 3.12.11 selected as the interpreter for Poetry to work correctly.
Pre-commit Hooks Configuration
The project enforces code quality through pre-commit hooks:

Standard hooks: YAML validation, EOF fixes, whitespace cleanup
Ruff (v0.12.3): Fast Python linter with auto-fix
Black (v25.1.0): Code formatter (88 char line length)
Mypy (v1.17.0): Static type checker

Workflow:
```bash
# Run manually before commit
pre-commit run --all-files
```

## Known Issues and Workarounds
### Poetry Command Execution
- **Issue**: Poetry commands fail in Windows Git Bash
- **Workaround**: Use direct virtual environment Python path (see ARCHITECTURE.MD)
- **Note**: Local development requires Python 3.12.1 interpreter selection

### Type Annotations
- **Status**: 0 mypy errors (fully resolved)
- **Key Pattern**: FunctionParameter uses `type_annotation` not `type_str`

```bash
# Emergency bypass (use sparingly)
git commit --no-verify -m "Emergency commit"
```

## Key Architectural Decisions
ADR-001: Python 3.10+ requirement for modern features
ADR-002: Built-in ast module (2.4x faster than alternatives)
ADR-003: ChromaDB for embeddings (zero-config, handles 1M embeddings)
ADR-004: docstring_parser>=0.16 (actively maintained, all formats)
## Performance Contracts
- Small file (<100 lines): <10ms parsing
- Medium project (100 files): <30s full analysis
- Large project (1000 files): <5 minutes full analysis
- Memory usage: <500MB for 10k functions

## Common Pitfalls
- Never use str() on custom objects - access specific attributes
- Check instance type before Union access
- Use content hash for cache keys, not file paths
- Respect confidence thresholds in matching
