# CodeDocSync - Claude Context Guide

## ⚠️ CRITICAL: Python Environment Setup

**System has dual Python installations - MUST use venv (Python 3.13.5) not system Python (3.9.12)**

```bash
# Activate venv EVERY session:
source .venv/Scripts/activate  # Git Bash
.\.venv\Scripts\activate      # PowerShell

# Verify: python --version  # Must show 3.13.5

# Git Bash pipe workaround:
python -m mypy codedocsync > mypy_output.txt 2>&1  # Don't use pipes
```

**Common Issues**: Wrong Python = syntax errors on `list[str]`, `x | None`. Always activate venv!

## MCP Integration (Enhanced Capabilities)

This project supports MCP servers for enhanced Claude Desktop capabilities:

1. **Filesystem Server**: Direct file access to project without explicit paths
2. **Memory Server**: Persistent knowledge across sessions (claude-memory.json)

**Benefits**:
- Natural file navigation and editing
- Remembers project patterns, decisions, and known issues
- No need to re-explain context between sessions

**Setup**: Requires Node.js and Claude Desktop with MCP support. Configuration goes in `%APPDATA%\Claude\claude_desktop_config.json`.

## Project Overview

**CodeDocSync** is an intelligent CLI tool that detects inconsistencies between Python code and its documentation using:
- AST parsing for code analysis
- Multiple docstring format support (Google, NumPy, Sphinx)
- LLM-powered semantic analysis
- Three-tier matching system (Direct, Contextual, Semantic)

**Current Status**: Core implementation complete, Week 4 optimizations in progress
**MyPy Compliance**: ✅ 0 errors
**Details**: See IMPLEMENTATION_STATE.MD

## Project Structure

```
codedocsync/
├── analyzer/       # LLM and rule-based analysis
├── cli/           # Command-line interface (Typer + Rich)
├── matcher/       # Code-to-doc matching algorithms
├── parser/        # AST and docstring parsing
├── storage/       # Vector store and embeddings (ChromaDB)
├── suggestions/   # Docstring generation and formatting
└── utils/         # Shared utilities and configuration

scripts/            # Development and testing utilities
├── memory_safe_test_runner.py     # Prevents test crashes
├── find_problematic_test.py       # Identifies problematic tests
├── safely_remove_suggestion_tests.py  # Clean test removal
└── identify_failing_tests.py      # Parse test results
```

## Critical Data Models

### Parser Module (`parser/`)
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

### Key Dependencies

- **Core**: typer, rich, pydantic, python-dotenv
- **Parsing**: docstring_parser (v0.16+), ast
- **Matching**: rapidfuzz, chromadb, sentence-transformers
- **LLM**: openai, tenacity
- **Dev**: pytest, black, mypy, ruff, pre-commit

### Type Hints (MANDATORY Python 3.10+)

✅ Use: `list[str]`, `dict[str, int]`, `str | None`, `int | float`
❌ NOT: `List[str]`, `Dict[str, int]`, `Optional[str]`, `Union[int, float]`

### Critical Code Patterns

```python
# isinstance with unions: Use tuple not union type
isinstance(x, (int, float))  # ✅
isinstance(x, int | float)   # ❌

# Optional AST types need typing import
from typing import Optional
Optional[ast.AST]  # ✅
```

## Testing and Quality Checks

```bash
# ALWAYS activate venv first!
source .venv/Scripts/activate

# Run tests (use memory-safe runner to prevent crashes)
python scripts/memory_safe_test_runner.py

# Run specific test patterns safely
python scripts/memory_safe_test_runner.py --pattern "test_name"

# Type checking
python -m mypy codedocsync

# For module-specific type checking (useful in Git Bash):
python -m mypy codedocsync/cli
python -m mypy codedocsync/analyzer
python -m mypy codedocsync/parser

# Code formatting
python -m black codedocsync

# Linting
python -m ruff check codedocsync

# Run the main CLI
python -m codedocsync analyze .
```

## 🚨 Memory Safety Guidelines (CRITICAL)

**System crashes from problematic tests - follow these rules:**

1. **NEVER use**: `sys.getsizeof(gc.get_objects())` - crashes system by measuring ALL Python objects
2. **Avoid module-level test code** - causes Windows crashes
3. **Use memory-safe runner**: `python scripts/memory_safe_test_runner.py`
4. **For memory profiling**: Use `tracemalloc`, not gc.get_objects()
5. **Test crashes during collection** = module-level code problem

## Current Focus

**RAG Sprint Continuation (16-20 hours)** - Addressing critical gaps in RAG implementation:
- Persistence layer for accepted suggestions
- Generator coverage (4 of 5 generators need RAG enhancement)
- Curated examples expansion (5 → 75+)
- Improvement measurement system
See RAG_SPRINT_CONTINUATION.MD for detailed plan.

## Architecture Highlights

### Three-Tier Matching System
1. **Direct Matcher** (90% of cases): Exact/fuzzy name matching
2. **Contextual Matcher** (8% of cases): Module/import aware
3. **Semantic Matcher** (2% of cases): Embedding-based similarity
   - Uses ChromaDB for vector storage
   - Requires OpenAI API key for embeddings
   - Use `--no-semantic` flag to explicitly disable

### Analysis Pipeline
1. Parse Python files with AST
2. Extract docstrings with format detection
3. Match code to documentation
4. Apply rule-based checks
5. Use LLM for semantic validation
6. Generate fix suggestions

### RAG-Enhanced Suggestions (⚠️ PARTIAL - Core infrastructure only)
- **Bootstrap corpus**: 143 examples from CodeDocSync + 5 curated (need 75+)
- **Self-improvement**: ❌ No persistence - accepted suggestions lost on exit
- **Retrieval**: Basic string similarity (not semantic)
- **Performance**: ✅ <100ms retrieval, <50MB memory
- **Generator coverage**: Only ParameterGenerator enhanced (1 of 5)
- **Effectiveness**: Limited impact on suggestion quality
- **Storage**: Bootstrap in `data/bootstrap_corpus.json`, no persistence for accepted

#### RAG Commands:
- `python -m codedocsync analyze . --no-rag` - Disable RAG enhancement
- `python -m codedocsync accept-suggestion <file> <function> <issue_type>` - Mark suggestion as accepted
- `python -m codedocsync rag-stats` - View corpus statistics and performance metrics

## Configuration Files

- `.codedocsync.yml`: Project-specific analysis configuration
- `.env`: API keys and environment settings
- `pyproject.toml`: Project dependencies and tool configs

## Quick Command Reference

```bash
# Activate environment (REQUIRED)
source .venv/Scripts/activate

# Analyze current directory
python -m codedocsync analyze .

# Check specific file
python -m codedocsync analyze path/to/file.py

# CI/CD mode
python -m codedocsync check . --ci --fail-on-critical

# Run with all matching strategies (default)
python -m codedocsync analyze .

# Run without semantic matching (optional)
python -m codedocsync analyze . --no-semantic

# Run without RAG enhancement (optional)
python -m codedocsync analyze . --no-rag

# Accept a suggestion (⚠️ WARNING: Not persisted across sessions)
python -m codedocsync accept-suggestion file.py function_name issue_type

# View RAG corpus statistics
python -m codedocsync rag-stats

# Get help
python -m codedocsync --help
```

### Quality Tools
- **Formatting**: black (88 char lines)
- **Linting**: ruff with auto-fix
- **Type checking**: mypy (see pyproject.toml)

## Critical Reminders

1. **ALWAYS activate venv** - wrong Python = broken code
2. **NEVER use old typing imports** - Python 3.10+ only
3. **NEVER use `sys.getsizeof(gc.get_objects())`** - memory bomb
4. **Use isinstance(x, (int, float))** not isinstance(x, int | float)
5. **Follow established patterns** - read existing code first

## Key Files for Context

- `IMPLEMENTATION_STATE.MD`: Detailed progress tracking and component status
- `TEST_FIXES_LOG.md`: Test infrastructure history and lessons learned
- `pyproject.toml`: Project dependencies and tool configurations
- `.codedocsync.yml`: Project-specific analysis configuration

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
