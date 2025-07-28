# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-01-28

### Fixed

#### Critical Bug Fixes

- **Bug 1 - Async/Await Runtime Error**: Investigated UnifiedMatchingFacade async method calls. Found that the implementation correctly uses `await` when calling async methods from SemanticMatcher (`prepare_semantic_index` and `match_with_embeddings`). The `match_project` method is properly declared as async and all CLI commands correctly use `asyncio.run()` to execute it. No fix was required as the code was already correct.

- **Bug 2 - Nested Function Parsing**: Investigated the parser extracting nested functions. The current behavior of using `ast.walk()` to extract all functions (including nested ones and methods inside classes) is intentional and expected by the test suite. The issue mentioned in the bug report about "losing context" may refer to not tracking the nesting hierarchy, but changing this behavior would break existing functionality. No fix was applied as the current implementation matches the expected behavior defined by the comprehensive test suite.
