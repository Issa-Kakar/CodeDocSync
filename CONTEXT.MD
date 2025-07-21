# CodeDocSync Development Context

**Version: 0.2.0** | **Status: Week 4 Ready (Performance & Optimization)**

## What We're Building

CodeDocSync is a CLI tool that automatically detects inconsistencies between Python code and its documentation. It uses AST parsing, rule-based analysis, and semantic search to identify when functions have changed but their docstrings haven't.

## Core Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   CLI       │────▶│   Parser     │────▶│  Matcher    │
│  (Typer)    │     │  (AST/DS)    │     │  (3-Layer)  │
└─────────────┘     └──────────────┘     └─────────────┘
                                             │
                                             ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Storage    │◀────│   Analyzer   │◀────│ Suggestions │
│ (ChromaDB)  │     │  (LLM/Rules) │     │ (Templates) │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Data Flow (NEVER VIOLATE)
AST Parser → Docstring Parser → Matcher → Analyzer → Suggestions

## Implementation Invariants

1. **Model Ownership**: Each module owns its models - never create models outside their modules
2. **Function Naming**: parse_* for parsing, analyze_* for analysis, match_* for matching
3. **Error Handling**: Every public function must handle None inputs gracefully
4. **Performance Targets**:
   - AST parsing: <50ms for medium files
   - Rule analysis: <5ms per function
   - LLM analysis: <2s per function
   - Cache operations: <10ms

## Critical Data Models

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
    type_annotation: Optional[str]  # NOT type_str!
    default_value: Optional[str]
    is_required: bool
    kind: ParameterKind

@dataclass
class MatchedPair:
    function: ParsedFunction
    documentation: Optional[ParsedDocstring]
    confidence: MatchConfidence  # dataclass, not enum
    match_type: MatchType        # dataclass, not enum
    match_reason: str
```

## Technology Stack

- **Python 3.10+**: Required for modern syntax
- **AST**: Built-in `ast` module (2.4x faster)
- **Docstrings**: docstring_parser v0.16+
- **LLM**: litellm (OpenAI/Anthropic abstraction)
- **Embeddings**: ChromaDB + OpenAI text-embedding-3-small
- **CLI**: Typer + Rich

## Module Structure

```
codedocsync/
├── cli/          # Modular commands (analyze, match, parse, suggest)
├── parser/       # AST + docstring parsing
├── matcher/      # 3-layer matching (direct, contextual, semantic)
├── analyzer/     # Rule engine + LLM analysis
├── suggestions/  # Fix generation with templates
└── storage/      # ChromaDB + caching layers
```

## Current Implementation State

### Completed (Weeks 1-3) ✅
- Parser: AST + multi-format docstring parsing
- Matcher: 3-layer strategy with OpenAI embeddings
- Analyzer: 12 rules + 6 LLM analysis types
- Suggestions: All docstring styles + smart merging
- Storage: 3-tier caching + ChromaDB
- Code Quality: 0 mypy/ruff errors

### Next Phase (Week 4)
1. Implement caching optimizations
2. Add parallel processing for large projects
3. Implement incremental analysis
4. Performance benchmarking suite

## Critical Implementation Notes

### Attribute Names (ALWAYS VERIFY)
- `type_annotation` not `type_str` for FunctionParameter
- `issues` not `enhanced_issues` for EnhancedAnalysisResult
- `rich_suggestion` not `suggestion_object` for EnhancedIssue

### Import Paths
- `codedocsync.parser.ast_parser` not `codedocsync.parser.models`
- `DocstringReturns` (plural) not `DocstringReturn`

### Error Handling Strategy
- LLM failures → fall back to rules
- Parsing errors → continue with partial results
- Network issues → use cached data

## Environment Setup

### Poetry on Windows Git Bash
```bash
# Find venv path
poetry env info --path

# Use direct path
"C:\path\to\venv\Scripts\python.exe" -m codedocsync analyze .
```

### Pre-commit Hooks
- Ruff: v0.12.3 (linting with auto-fix)
- Black: v25.1.0 (88 char lines)
- Mypy: v1.17.0 (static typing)

### Local Development
Requires Python 3.12.1 selected as interpreter for Poetry.

## Key Commands

```bash
# Development
poetry install
pre-commit run --all-files

# Testing
poetry run pytest tests/ -xvs

# Usage
poetry run codedocsync analyze .
poetry run codedocsync suggest . --output suggestions.json
```
