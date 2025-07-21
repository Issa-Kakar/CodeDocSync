# Comprehensive List of Files Needing Fixes

## Ruff Issues (9 errors, 8 fixable)
- 5 x W293 (blank-line-with-whitespace)
- 1 x I001 (unsorted-imports)
- 1 x UP038 (non-pep604-isinstance)
- 1 x UP045 (non-pep604-annotation-optional)
- 1 x W292 (missing-newline-at-end-of-file)

## Black Formatting Issues (2 files)
1. test_mypy_fixes.py
2. codedocsync/suggestions/type_formatter.py

## Mypy Issues (Focus on suggestions module - highest priority)
### Main problem files in suggestions module:
1. codedocsync/suggestions/style_detector.py - Type annotation issues with dict
2. codedocsync/suggestions/errors.py - Missing type annotations, incorrect Suggestion calls
3. codedocsync/suggestions/converter.py - Missing return annotations, attribute errors
4. codedocsync/suggestions/generators/return_generator.py - Missing type annotations
5. codedocsync/suggestions/generators/raises_generator.py - Missing annotations
6. codedocsync/suggestions/generators/edge_case_handlers.py - Missing annotations
7. codedocsync/suggestions/performance.py - Missing return type annotations
8. codedocsync/suggestions/base.py - Missing type annotations
9. codedocsync/suggestions/type_formatter.py - Unreachable code

### Analyzer module (remaining issues):
1. codedocsync/analyzer/prompt_templates.py - Unreachable code

## Fix Strategy
1. First fix all ruff issues (can be done automatically)
2. Then fix black formatting issues
3. Apply automated mypy fixes to suggestions module
4. Handle remaining manual mypy fixes
