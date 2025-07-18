# Changelog

All notable changes to CodeDocSync will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with Poetry setup
- ARCHITECTURE.md documenting system design and components
- CHANGELOG.md for tracking project evolution
- Core module structure: cli/, parser/, matcher/, analyzer/, storage/, integrations/, utils/
- AST Parser implementation with comprehensive data models:
  - `FunctionParameter` dataclass with validation for parameter names and type annotations
  - `FunctionSignature` dataclass with `to_string()` method for readable output
  - `ParsedFunction` dataclass representing complete function metadata
  - `parse_python_file()` function with error handling and performance targets
- Custom exception hierarchy in `utils/errors.py`:
  - `ParsingError` base class with recovery hints
  - `ValidationError` for data validation issues
  - `FileAccessError` for file system errors
  - `SyntaxParsingError` for syntax parsing errors
- Parser module public API in `parser/__init__.py`
- Core AST parsing functionality with comprehensive error handling:
  - `parse_python_file()` function with performance targets (<10ms for small files, <50ms for medium files, <200ms for large files)
  - `_extract_function()` helper for extracting function details from AST nodes with error recovery
  - `_extract_signature()` helper for extracting complete function signatures including parameters, return types, and decorators
  - Support for both regular and async functions with proper metadata extraction
  - Robust error handling for file access, syntax errors, and encoding issues
  - Helper functions for decorator extraction (`_get_decorator_names()`) and annotation parsing (`_get_annotation_string()`)
  - Default value extraction with fallback for complex expressions
- Enhanced error handling and edge case support:
  - Syntax error recovery with partial parsing up to error line
  - Alternative encoding support (fallback to latin-1 from utf-8)
  - Empty file and imports-only file detection
  - Permission error handling with recovery hints
  - Comprehensive logging throughout parsing process with timing benchmarks
  - Enhanced function parameter extraction supporting *args, **kwargs, positional-only, and keyword-only parameters
  - Complex type annotation handling (Union, Optional, List, Dict, custom generics)
  - Default value extraction for function calls, lambda expressions, and complex expressions
  - Property decorator support (@property, @setter, @getter, @deleter)
  - Class method and static method detection
  - Nested function context preservation
  - Lambda function handling in default values
  - Fallback mechanisms for older Python versions without ast.unparse()
  - Enhanced method detection considering decorators and parameter patterns
- Comprehensive test suite for AST parser (Chunk 4):
  - 25 test cases covering basic parsing, edge cases, error handling, and performance
  - Test fixtures for various Python syntax patterns and edge cases
  - Validation tests for data models (FunctionParameter, FunctionSignature, ParsedFunction)
  - Performance tests ensuring <50ms for small files, <200ms for large files
  - Error handling tests for file access, syntax errors, and encoding issues
  - Support for complex parameter types (*args, **kwargs, keyword-only, positional-only)
  - Unicode content testing and malformed syntax recovery
  - Comprehensive decorator testing including complex decorator patterns
  - Cross-platform compatibility (Windows permission handling)
- Enhanced parameter validation to support special syntax markers
- Ruff compliance and formatting integration with pre-commit hooks
- Updated ARCHITECTURE.MD with Ruff integration guidelines
- Performance optimization and integration features (Chunk 5):
  - Added LRU caching for AST parsing with `@lru_cache(maxsize=100)` decorator
  - Implemented `parse_python_file_lazy()` generator function for memory-efficient parsing of large files
  - Enhanced file content hashing with MD5 for cache key generation
  - Updated parser module public API to expose lazy parsing functionality
  - Added comprehensive CLI `parse` command with Rich table formatting and JSON output options
  - Integrated performance monitoring and benchmarking throughout parser lifecycle
  - Added function count tracking in lazy parsing mode
- Docstring Parser foundation (Chunk 1):
  - Created `docstring_models.py` with comprehensive data models for parsed docstrings
  - Added `DocstringFormat` enum supporting Google, NumPy, Sphinx, REST, and Unknown formats
  - Implemented `DocstringParameter` dataclass with validation for parameter names including *args/**kwargs support
  - Added `DocstringReturns` and `DocstringRaises` dataclasses for return and exception documentation
  - Created `ParsedDocstring` main dataclass with metadata including validity flags and parse errors
  - Built base `DocstringParser` class structure with format mapping to third-party library
  - Updated parser module `__init__.py` to export all new docstring models and parser class
- Docstring Parser format detection and parsing logic (Chunk 2):
  - Implemented comprehensive `detect_format()` method with hierarchical detection heuristics
  - Added format-specific pattern matching for Google, NumPy, Sphinx, and REST docstring styles
  - Built complete `parse()` method with automatic format detection and error handling
  - Created `_convert_to_parsed_docstring()` method to bridge third-party parser to internal models
  - Implemented robust `_extract_examples()` method for extracting code examples across formats
  - Added fallback parsing with `_create_error_docstring()` for graceful handling of malformed docstrings
  - Built `_extract_parameters_fallback()` method with regex patterns for each docstring format
  - Enhanced error recovery with best-effort parameter extraction when primary parsing fails
  - Integrated with `docstring_parser` library for reliable multi-format docstring parsing
- Docstring Parser integration and AST compatibility (Chunk 3 & 4):
  - Created `RawDocstring` dataclass for structured storage of raw docstring text with metadata
  - Updated `ParsedFunction.docstring` field to accept `Union[RawDocstring, ParsedDocstring]` for seamless integration
  - Built complete `IntegratedParser` class combining AST and docstring parsing capabilities
  - Implemented caching mechanism in `IntegratedParser` for improved parsing performance
  - Added lazy parsing support with `parse_file_lazy()` method for memory-efficient large file processing
  - Enhanced parser module exports to include all new models and parsers
  - Added `docstring_parser>=0.16` dependency for reliable multi-format docstring parsing
  - Fixed docstring style mapping to use correct enum values (NUMPYDOC vs NUMPY, EPYDOC for Sphinx)
- Comprehensive Testing Suite (Chunk 5):
  - Created `tests/test_docstring_parser.py` with 17 comprehensive test cases covering all docstring parsing functionality
  - Added format detection tests for Google, NumPy, Sphinx, and REST docstring styles
  - Implemented parsing tests for each supported format with parameter, return, and exception extraction
  - Added malformed docstring handling tests with graceful error recovery
  - Created edge case tests for empty docstrings, Unicode content, and complex type annotations
  - Built parameter validation tests for *args/**kwargs and special parameter naming patterns
  - Added error recovery tests for completely invalid docstrings and empty sections
  - Implemented fallback parameter extraction tests for when primary parsing fails
  - Created `tests/test_integrated_parser.py` with 14 integration test cases
  - Added file parsing tests with temporary files and various content scenarios
  - Implemented cache functionality tests with stats and clearing mechanisms
  - Added lazy parsing tests for memory-efficient large file processing
  - Created performance tests for large files (50+ functions) with timing validation
  - Added Unicode content handling tests for international character support
  - Implemented error handling tests for malformed files, empty files, and non-existent files
  - Added Windows-compatible file handling with proper resource cleanup
  - All tests pass on Windows platform with comprehensive error handling
- Docstring Parser performance optimization and final integration (Chunk 6):
  - Enhanced `DocstringParser` class with LRU caching for format detection using `@functools.lru_cache(maxsize=1000)`
  - Implemented intelligent parse result caching with MD5-based cache keys and FIFO eviction strategy
  - Added configurable cache size (default 500 entries) to prevent memory bloat
  - Created efficient `parse_batch()` method using ThreadPoolExecutor with 4 workers for parallel processing
  - Built cache management methods: `get_cache_stats()` for monitoring and `clear_cache()` for memory cleanup
  - Separated caching logic with `_parse_uncached()` method for clean architecture
  - Added comprehensive cache statistics including hit ratios and memory usage tracking
  - Optimized batch processing for large-scale docstring parsing with thread-safe operations
  - Enhanced `IntegratedParser` compatibility with all new caching features
  - Performance targets: <5ms per docstring parse with cache hits, <50ms for batch processing

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Fixed incorrect default value assignment for positional-only and regular arguments in AST parser - the logic now correctly handles `args.defaults` as containing defaults for trailing positional parameters (both positional-only and regular combined)
- Fixed incorrect keyword-only parameter name formatting in `_extract_signature()` - keyword-only parameters now have correct names without `*` prefix (the `*` is a syntax marker, not part of the parameter name)

### Security
- Nothing yet

## [0.1.0] - 2025-01-08

### Added
- Project inception with comprehensive blueprint
- Two-document system (ARCHITECTURE.md + CHANGELOG.md) for AI-resistant documentation
- 6-week implementation roadmap with daily milestones
- Research-driven technical decisions based on AST parsing performance studies
