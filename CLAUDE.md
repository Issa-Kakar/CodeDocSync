# CodeDocSync - Claude Context Guide

## CRITICAL: Python Environment Setup

**MANDATORY BEFORE ANY CODE EXECUTION:**

The system has multiple Python installations:
- **System Default**: Anaconda Python 3.9.12 (INCOMPATIBLE)
- **Project Python**: Python 3.13.5 in .venv (REQUIRED)

### Environment Activation (REQUIRED EVERY SESSION)

```bash
# Git Bash (preferred)
source .venv/Scripts/activate

# Windows PowerShell
.\.venv\Scripts\activate

# VERIFY activation worked:
python --version  # MUST show Python 3.13.5
which python     # MUST show .venv/Scripts/python
```

### Alternative: Direct Venv Python Usage

```bash
# If activation fails, use direct paths:
.venv/Scripts/python.exe <command>
.venv/Scripts/python.exe -m pip install <package>
.venv/Scripts/python.exe -m pytest
```

### Common Environment Issues

1. **Python 3.9 SyntaxError**: You're using system Python - activate venv!
2. **"invalid syntax" on `list[str] | None`**: Wrong Python version
3. **Poetry issues on Windows**: Use `python -m poetry` instead of `poetry`
4. **New terminal loses venv**: Must activate in EVERY new session

### Git Bash Command Workarounds

Due to Git Bash limitations on Windows, use file redirection instead of pipes:

```bash
# ❌ WRONG (Unix pipes don't work)
python -m mypy codedocsync | head -50

# ✅ CORRECT (Use file redirection)
python -m mypy codedocsync > mypy_output.txt 2>&1
cat mypy_output.txt  # or open in editor

# ✅ For specific modules (faster)
python -m mypy codedocsync/cli > cli_mypy.txt 2>&1
python -m mypy codedocsync/analyzer > analyzer_mypy.txt 2>&1
```

**Note**: The ".venv/Scripts/activate: line 40: uname: command not found" warning is harmless and can be ignored.

## Project Overview

**CodeDocSync** is an intelligent CLI tool that detects inconsistencies between Python code and its documentation using:
- AST parsing for code analysis
- Multiple docstring format support (Google, NumPy, Sphinx)
- LLM-powered semantic analysis
- Three-tier matching system (Direct, Contextual, Semantic)

**Current Stage**: Week 4 of 6-week sprint
**Core Components**: ✅ All working (Parser, Matcher, Storage, CLI, Analyzer)
**Test Suite**: 95.9% passing (446/465 tests)
**MyPy Compliance**: ✅ 0 errors

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

### Dependencies

- **Core**: typer, rich, pydantic, python-dotenv
- **Parsing**: docstring_parser (v0.16+), built-in ast, astor
- **Matching**: rapidfuzz, chromadb (optional), sentence-transformers
- **LLM**: openai, tenacity
- **Dev**: pytest, black, mypy, ruff, pre-commit

**Note**: ChromaDB is optional but currently has a runtime incompatibility with NumPy 2.0+. The project uses NumPy 2.3.1, which causes ChromaDB imports to fail. Until ChromaDB is updated or imports are made lazy, semantic matching (2% of cases) will be unavailable. The tool gracefully degrades to Direct/Contextual matching (98% of functionality).

### Type Hints Convention

**ALWAYS use modern Python 3.10+ syntax:**
- ✅ `list[str]` not ❌ `List[str]`
- ✅ `dict[str, int]` not ❌ `Dict[str, int]`
- ✅ `str | None` not ❌ `Optional[str]`
- ✅ `int | float` not ❌ `Union[int, float]`

### Common Code Patterns to Follow

1. **isinstance() with Union types**:
   ```python
   # ✅ CORRECT
   isinstance(x, (int, float))

   # ❌ WRONG
   isinstance(x, int | float)
   ```

2. **Regex flags**:
   ```python
   # ✅ CORRECT
   re.MULTILINE | re.IGNORECASE

   # ❌ WRONG
   Union[re.MULTILINE, re.IGNORECASE]
   ```

3. **Optional AST types**:
   ```python
   # ✅ CORRECT
   from typing import Optional
   Optional[ast.AST]

   # ❌ WRONG
   ast.Optional
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

### MyPy Configuration

The project uses relaxed mypy settings appropriate for a portfolio project in active development (see pyproject.toml). This focuses on functionality and performance over type perfection, appropriate for a 6-week sprint portfolio project.

## Memory Safety Guidelines

**CRITICAL: Test Development Best Practices**

After experiencing system crashes from problematic tests, follow these rules:

1. **Avoid Module-Level Code in Tests**
   - Don't create large data structures at import time
   - Keep test data creation inside test methods or fixtures

2. **NEVER Use Dangerous Functions**
   - ❌ **NEVER**: `sys.getsizeof(gc.get_objects())` - This attempts to measure ALL Python objects in memory (gigabytes) and will crash your system
   - ✅ **Use**: `tracemalloc` for proper memory profiling

3. **Be Careful with Test Fixtures**
   - Avoid creating massive objects in fixtures
   - Use lazy fixture evaluation when possible
   - Clean up resources in fixture teardown

4. **Watch for Memory-Intensive Operations**
   - Avoid unbounded loops or recursion
   - Don't create extremely long strings (e.g., "x" * 10000000)
   - Be careful with cartesian products or combinatorial explosions

5. **Use the Memory-Safe Test Runner**
   ```bash
   # Always use this instead of pytest directly
   python scripts/memory_safe_test_runner.py

   # Set memory limits for safety
   python scripts/memory_safe_test_runner.py --memory-limit 1024

   # Clean up test artifacts
   python scripts/memory_safe_test_runner.py --clean-only
   ```

6. **Test Collection Issues**
   - If tests crash during collection (before running), the issue is in imports
   - Check for circular imports between test modules
   - Look for module-level code that executes on import

## Current Implementation Focus

### Week 4: Performance & Optimization (Current Phase)
- **Priority**: Reimplement suggestion tests with memory safety before proceeding
- See `IMPLEMENTATION_STATE.MD` for detailed progress and next steps

### Known Issues to Address
1. **ChromaDB NumPy 2.0+ Compatibility Issue (Critical)**
   - ChromaDB is incompatible with NumPy 2.0+ (uses deprecated `np.float_`)
   - Causes runtime error when importing ChromaDB with NumPy 2.3.1
   - **Current Impact**: Module-level import causes immediate failure
   - **Workaround**: Make ChromaDB import truly lazy (only import when semantic matching requested)
   - **Docker Strategy**: Future Docker containers could provide isolated environments with compatible versions

2. ChromaDB installation may fail on Windows (requires C++ compiler)
   - Solution: Use `--no-semantic` flag or let the tool gracefully degrade
   - This only affects 2% of matching cases (semantic matching)

## Architecture Highlights

### Three-Tier Matching System
1. **Direct Matcher** (90% of cases): Exact/fuzzy name matching
2. **Contextual Matcher** (8% of cases): Module/import aware
3. **Semantic Matcher** (2% of cases): Embedding-based similarity
   - Requires ChromaDB (optional dependency)
   - Gracefully disabled if ChromaDB unavailable
   - Use `--no-semantic` flag to explicitly disable

### Analysis Pipeline
1. Parse Python files with AST
2. Extract docstrings with format detection
3. Match code to documentation
4. Apply rule-based checks
5. Use LLM for semantic validation
6. Generate fix suggestions

### Performance Strategy
- File-level hash caching
- AST serialization cache
- Embedding persistence
- Multiprocessing for CPU-bound tasks
- Async I/O for LLM calls

## Development Workflow

1. **Always check Python version first**
2. **Read existing code before modifying** - follow established patterns
3. **Use modern type hints** - no legacy typing
4. **Test incrementally** - run mypy and ruff after changes
5. **Follow project structure** - keep components separated

## Configuration Files

- `.codedocsync.yml`: Project-specific analysis configuration
- `.env`: API keys and environment settings
- `pyproject.toml`: Project dependencies and tool configs

### Error Handling Strategy
- LLM failures → fall back to rules
- Parsing errors → continue with partial results
- Network issues → use cached data

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

# Run without semantic matching (if ChromaDB unavailable)
python -m codedocsync match-unified . --no-semantic

# Get help
python -m codedocsync --help
```

### Pre-commit Hooks
- Ruff: v0.12.3 (linting with auto-fix)
- Black: v25.1.0 (88 char lines)
- Mypy: v1.17.0 (static typing with relaxed settings for portfolio project)

## Performance Contracts
- Small file (<100 lines): <10ms parsing
- Medium project (100 files): <30s full analysis
- Large project (1000 files): <5 minutes full analysis
- Memory usage: <500MB for 10k functions

## Important Reminders

1. **NEVER commit without activating venv** - wrong Python = broken code
2. **NEVER use old typing imports** - project requires Python 3.10+
3. **ALWAYS check imports exist** - don't assume availability
4. **ALWAYS handle Windows paths** - use forward slashes or raw strings
5. **NEVER mix Union in isinstance()** - use tuples instead
6. **NEVER use `sys.getsizeof(gc.get_objects())` in tests** - it's a memory bomb
7. **AVOID module-level test code** - causes Windows crashes

## Key Files for Context

- `IMPLEMENTATION_STATE.MD`: Detailed progress tracking and component status
- `TEST_FIXES_LOG.md`: Test infrastructure history and lessons learned
- `pyproject.toml`: Project dependencies and tool configurations
- `.codedocsync.yml`: Project-specific analysis configuration
