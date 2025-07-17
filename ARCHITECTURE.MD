# CodeDocSync Architecture

**Version: 0.1.0**

## Project Overview

CodeDocSync is a CLI tool that automatically detects inconsistencies between Python code and its documentation. It uses AST parsing, rule-based analysis, and minimal semantic search to identify when functions have changed but their docstrings haven't, preventing documentation drift that causes developer confusion and AI tool inefficiency.

## High-Level Architecture

┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   CLI       │────▶│   Parser     │────▶│  Matcher    │
│  (Typer)    │     │  (AST/TS)    │     │  (3-Layer)  │
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

## Component List

| Component | Purpose | Key Interfaces |
|-----------|---------|----------------|
| **cli/** | Command-line interface with analyze/watch/check modes | `main.py`: Entry point, `formatters.py`: Output formatting |
| **parser/** | Extract code structure and docstrings from Python files | `ast_parser.py`: AST traversal, `docstring_parser.py`: Parse Google/NumPy/Sphinx styles |
| **matcher/** | Match code to documentation using 3-layer approach | `direct_matcher.py`: 90% name-based, `semantic_matcher.py`: 2% embedding fallback |
| **analyzer/** | Detect inconsistencies using rules and LLM | `llm_analyzer.py`: Semantic checking, `rule_engine.py`: Fast pre-checks |
| **storage/** | Manage embeddings and configuration | `vector_store.py`: ChromaDB abstraction, `embedding_cache.py`: LRU cache |
| **integrations/** | CI/CD and editor integration | `github_actions.py`: Pipeline templates, `git_hooks.py`: Pre-commit |
| **utils/** | Performance and parallelization utilities | `parallel.py`: Multiprocessing, `cache.py`: File hash caching |

## Key Architectural Decisions

### ADR-001: Python 3.10+ Requirement
**Context:** Need modern features for type hints and performance
**Decision:** Require Python 3.10+
**Alternatives:** Support 3.8+ (rejected - limits features)
**Reason:** Match statements, better typing, performance improvements worth the restriction

### ADR-002: AST Parser Choice
**Context:** Need fast, reliable Python code parsing
**Decision:** Primary: built-in `ast`, Secondary: tree-sitter for watch mode
**Alternatives:** Only tree-sitter (rejected - overhead for simple parsing)
**Reason:** AST module 2.4x faster for batch processing; tree-sitter adds incremental parsing

### ADR-003: Minimal RAG with ChromaDB
**Context:** Need semantic fallback for renamed functions
**Decision:** ChromaDB for 2% of matches requiring embeddings
**Alternatives:** No semantic matching (rejected - misses refactorings)
**Reason:** Zero-config setup, handles 1M embeddings, built for AI applications

### ADR-004: LLM Provider Abstraction
**Context:** Support multiple LLM providers for flexibility
**Decision:** Use litellm for unified interface
**Alternatives:** Direct OpenAI only (rejected - vendor lock-in)
**Reason:** Automatic fallbacks, cost tracking, provider switching without code changes

## Integration Points

- **External APIs:** OpenAI/Anthropic via litellm for semantic analysis
- **Databases:** ChromaDB/hnswlib for vector similarity search
- **File System:** Watch mode monitoring, Git integration for incremental analysis
- **CI/CD:** GitHub Actions, GitLab CI, pre-commit hooks
- **Configuration:** YAML config files, environment variables for API keys

## Code Quality and Formatting

**Ruff Integration**: This project uses Ruff for code linting and formatting. All code must be Ruff-compliant before commits. Pre-commit hooks enforce this standard. Use `ruff format` and `ruff check --fix` to ensure compliance.

**Static Type Checking**: MyPy is used for static type analysis. All code must pass `mypy codedocsync/` without errors. Key type safety patterns:
- Use `Union[ast.FunctionDef, ast.AsyncFunctionDef]` for functions handling both sync and async functions
- Handle `Optional[ast.expr]` types properly with null checks before passing to functions expecting non-optional types
- Use `# type: ignore` comments sparingly and only for third-party imports without type stubs

**Pre-commit Quality Gates**: Before any commit, the following commands must pass:
- `ruff check . --fix` - Fix all linting issues
- `ruff format .` - Format all code consistently
- `mypy codedocsync/` - Pass all type checks
- `python -m pytest tests/` - All tests must pass

**IDE Integration**: Install the Ruff extension in your IDE for real-time linting feedback. Configure your IDE to run these checks on save to catch issues early.

## AI Guardrails

Use ARCHITECTURE.md as canonical context for all AI-assisted development. Do not invent components or modify the 3-layer matching approach (direct 90%, contextual 8%, semantic 2%) without updating this document.
