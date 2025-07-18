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
- Direct Matcher foundation (Chunk 1):
  - Created `matcher/models.py` with comprehensive data models for matching results
  - Added `MatchType` enum supporting EXACT, FUZZY, CONTEXTUAL, SEMANTIC, and NO_MATCH match types
  - Implemented `MatchConfidence` dataclass with validation for confidence scores including name similarity, location score, and signature similarity
  - Added `MatchedPair` dataclass linking functions to their documentation with confidence scoring and match reasoning
  - Created `MatchResult` dataclass with summary statistics including match rate calculation and match type counting
  - Built `MatchingError` exception class with recovery hints for robust error handling
  - Updated matcher module `__init__.py` to export all new models and matcher classes
  - Added comprehensive test suite in `tests/test_matcher_models.py` with validation and summary testing
- Direct Matcher exact matching implementation (Chunk 2):
  - Implemented complete `DirectMatcher` class with `match_functions()` method for same-file function-documentation matching
  - Added exact matching logic with `_try_exact_match()` method using high confidence threshold (≥0.95)
  - Built comprehensive confidence calculation system with `_calculate_exact_match_confidence()` method
  - Implemented signature similarity calculation using Jaccard similarity for parameter matching
  - Added efficient file grouping with `_group_by_file()` method for processing multiple files
  - Integrated statistics tracking with match type counting and performance monitoring
  - Created helper methods for stats management: `_reset_stats()` and `get_stats()`
  - Added robust error handling and logging throughout the matching process
  - Built comprehensive test suite in `tests/test_direct_matcher_exact.py` with 10 test cases covering exact matching functionality, edge cases, and performance validation
  - Performance targets met: <1ms per function for exact matching operations
  - Full integration with existing parser output (RawDocstring and ParsedDocstring support)
- Direct Matcher fuzzy matching implementation (Chunk 3):
  - Enhanced `DirectMatcher` class with fuzzy name matching capabilities using `rapidfuzz` library
  - Added configurable fuzzy threshold parameter (default 0.85) for similarity matching
  - Implemented pattern-based transformations for common naming conventions (snake_case ↔ camelCase, abbreviations)
  - Built comprehensive fuzzy matching logic with `_try_fuzzy_match()` method supporting:
    - Pattern-based transformations (get_user → getUser, calc_total → calculate_total)
    - String similarity matching using rapidfuzz.fuzz.ratio with configurable threshold
    - Cross-function similarity validation with parameter count and line distance checks
  - Added fuzzy match validation with `_validate_fuzzy_match()` method preventing false positives:
    - Parameter count similarity checking (max difference of 2 parameters)
    - Line distance validation (within 100 lines for same context)
    - Return type compatibility checking when available
  - Implemented detailed confidence scoring for fuzzy matches with weighted calculation:
    - Name similarity (50% weight) using fuzzy string matching
    - Location score (20% weight) based on line distance
    - Signature similarity (30% weight) using Jaccard coefficient
  - Built signature similarity calculation between different functions with `_calculate_signature_similarity_between()`
  - Added `rapidfuzz>=3.13.0` dependency for high-performance fuzzy string matching
  - Enhanced statistics tracking to include fuzzy match counts and performance metrics
  - Created comprehensive test suite in `tests/test_direct_matcher_fuzzy.py` with 9 test cases covering:
    - Snake_case to camelCase pattern matching
    - Fuzzy threshold enforcement and validation
    - Cross-function match prevention for different signatures
    - Line distance and parameter count validation
    - Confidence scoring accuracy and edge cases
  - Performance targets met: <5ms per function for fuzzy matching operations (actual: ~0.01ms)
  - All existing exact matching tests continue to pass with no regression
- Direct Matcher integration and configuration (Chunk 4):
  - Created `MatcherConfig` class in `utils/config.py` with comprehensive validation
  - Added configurable fuzzy matching threshold, line distance limits, and custom pattern support
  - Built `MatchingFacade` class in `matcher/facade.py` for high-level matching operations
  - Implemented project-wide matching with automatic Python file discovery and exclusion filtering
  - Added CLI `match` command in `main.py` with Rich table output and JSON export options
  - Enhanced configuration system with YAML file loading and validation
  - Created comprehensive integration tests in `tests/test_matcher_integration.py` covering:
    - End-to-end file and project matching workflows
    - Custom pattern configuration and application
    - Empty file handling and excluded directory filtering
    - Configuration loading from YAML files
  - Built performance test suite in `tests/test_matcher_performance.py` validating:
    - Exact matching performance (<1ms per function requirement met)
    - Fuzzy matching performance (<5ms per function requirement met)
    - Memory usage optimization for large-scale processing (10k+ functions)
    - Multi-file matching efficiency with proper file grouping
  - All performance targets met with comprehensive validation and edge case handling
- DirectMatcher core implementation refinement and testing (Chunk 5):
  - Fixed fundamental design flaw where DirectMatcher was matching functions to other functions instead of functions to their own docstrings
  - Corrected architecture to match functions with their own documentation (function → own docstring) rather than cross-function matching
  - Removed inappropriate fuzzy matching between different functions (this will be implemented in ContextualMatcher later)
  - Simplified DirectMatcher to focus on exact matching of functions to their own docstrings
  - Updated MatchingFacade to work with simplified DirectMatcher interface
  - Fixed pydantic v2 compatibility issues in config system (updated @validator to @field_validator)
  - Corrected integration tests to match actual expected behavior of direct matching
  - Fixed performance tests with proper RawDocstring constructor calls
  - All integration tests now pass with corrected expectations
  - All performance tests pass with <1ms per function requirement met
  - DirectMatcher now correctly implements the architecture: each function is matched to its own docstring documentation

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Fixed incorrect default value assignment for positional-only and regular arguments in AST parser - the logic now correctly handles `args.defaults` as containing defaults for trailing positional parameters (both positional-only and regular combined)
- Fixed incorrect keyword-only parameter name formatting in `_extract_signature()` - keyword-only parameters now have correct names without `*` prefix (the `*` is a syntax marker, not part of the parameter name)
- Fixed RawDocstring serialization bugs in CLI parse command - replaced `str(docstring)` with `docstring.raw_text` in both JSON serialization and pretty-print output to prevent object representation display instead of actual docstring content

**Note**: Always use `.raw_text` attribute when displaying RawDocstring objects, never `str()` which returns object representation.

### Security
- Nothing yet

## [0.1.0] - 2025-01-08

### Added
- Project inception with comprehensive blueprint
- Two-document system (ARCHITECTURE.md + CHANGELOG.md) for AI-resistant documentation
- 6-week implementation roadmap with daily milestones
- Research-driven technical decisions based on AST parsing performance studies
