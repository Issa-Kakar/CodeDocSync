CURRENT_SPRINT.MD: Rule Engine Implementation
This sprint implements the Analyzer module's Rule Engine component, providing fast, deterministic validation of function-documentation consistency before any LLM analysis. The implementation will be split into 5 chunks for systematic development.
Sprint Overview
Goal: Implement a high-performance rule engine that catches 80% of documentation issues without LLM calls, achieving <5ms per function analysis.
Key Principles:

Rules run BEFORE LLM analysis - they are the first line of defense
High confidence rule detections (>0.9) skip LLM analysis entirely
All rules must be deterministic and fast
Rules produce structured InconsistencyIssue objects with actionable suggestions
The engine must integrate seamlessly with the existing MatchedPair output from matchers


CHUNK 1: Data Models and Core Infrastructure (codedocsync/analyzer/init.py and models.py)
Context for AI Agent
You are implementing the foundational data models and infrastructure for the Analyzer module. This module receives MatchedPair objects from the matcher module and produces AnalysisResult objects containing detected inconsistencies.
Critical Implementation Requirements

Create analyzer/init.py that properly exports all public APIs
Create analyzer/models.py with complete data models
Ensure compatibility with existing MatchedPair from matcher module
Include comprehensive validation in data models

Detailed Implementation Guide
Step 1: Create analyzer/models.py
python"""
Data models for the analyzer module.

CRITICAL: These models define the contract between the analyzer and other modules.
All fields must be validated and documented.
"""
Define these dataclasses with full validation:

InconsistencyIssue: Core model for a single detected issue

issue_type: Must be from predefined ISSUE_TYPES
severity: Must be one of ['critical', 'high', 'medium', 'low']
description: Clear, specific description of what's wrong
suggestion: Actionable fix (can include code snippets)
line_number: Where the issue occurs
confidence: Float 0.0-1.0, used for LLM routing decisions
details: Dict for additional context (e.g., {"expected": "user_id", "actual": "userId"})


RuleCheckResult: Result from a single rule check

rule_name: String identifier for the rule
passed: Boolean
confidence: Float indicating certainty
issues: List[InconsistencyIssue]
execution_time_ms: Performance tracking


AnalysisResult: Complete analysis output

matched_pair: The input MatchedPair
issues: All detected issues
used_llm: Whether LLM analysis was performed
analysis_time_ms: Total time
rule_results: Optional list of individual rule results
cache_hit: Whether results came from cache



Step 2: Define Constants
Create comprehensive constants for:

ISSUE_TYPES: Mapping of issue type to default severity
SEVERITY_WEIGHTS: Numeric values for sorting (critical=4, high=3, etc.)
CONFIDENCE_THRESHOLDS: When to skip LLM (e.g., HIGH_CONFIDENCE = 0.9)

Step 3: Create analyzer/init.py
Export the public API cleanly:

All models from models.py
Future exports for RuleEngine, LLMAnalyzer (use TYPE_CHECKING imports)
Main analyze_matched_pair function signature

Validation Examples
The InconsistencyIssue should validate:
python# Valid
issue = InconsistencyIssue(
    issue_type="parameter_name_mismatch",
    severity="critical",
    description="Parameter 'user_id' in function does not match 'userId' in docstring",
    suggestion="Change docstring parameter to 'user_id' to match function signature",
    line_number=45,
    confidence=0.95,
    details={"expected": "user_id", "actual": "userId", "position": 0}
)

# Should raise ValueError
issue = InconsistencyIssue(
    issue_type="invalid_type",  # Not in ISSUE_TYPES
    severity="very_high",  # Invalid severity
    ...
)
Integration Points
Remember these critical integration requirements:

Import MatchedPair from codedocsync.matcher
Import ParsedFunction, ParsedDocstring from codedocsync.parser
Use Union types correctly for docstring field (can be RawDocstring or ParsedDocstring)
Maintain compatibility with the confidence scoring from matchers


CHUNK 2: Rule Engine Core Implementation (codedocsync/analyzer/rule_engine.py)
Context for AI Agent
You are implementing the RuleEngine class that performs fast, deterministic checks on function-documentation pairs. This is the heart of the analysis system and must be extremely performant (<5ms per function).
Critical Implementation Requirements

Implement all rule categories: structural, completeness, consistency
Each rule must return confidence scores for routing decisions
Performance is critical - avoid regex where possible, use early returns
Generate specific, actionable suggestions for each issue

Detailed Implementation Guide
Step 1: Create RuleEngine Class Structure
python"""
Rule-based analysis engine for fast documentation validation.

Performance target: <5ms per function
Confidence threshold: Issues with confidence > 0.9 skip LLM analysis
"""
Design the class with:

Configuration options (which rules to run)
Performance tracking via decorators
Clean separation between rule categories

Step 2: Implement Structural Rules
These rules check the structure matches between function and documentation:

parameter_names: Check exact name matches

Compare function parameter names with docstring parameter names
Handle *args and **kwargs specially
Ignore 'self' for methods, 'cls' for classmethods
Confidence: 1.0 for exact matches, 0.0 for mismatches


parameter_types: Validate type annotations match documented types

Parse type strings carefully (handle Optional, List, etc.)
Map common variations (int vs integer, str vs string)
Confidence: 0.9 for exact match, 0.7 for equivalent types


parameter_count: Ensure all parameters are documented

Count must match (excluding self/cls)
Check for extra parameters in docs
Confidence: 1.0 (this is deterministic)


return_type: Validate return type consistency

Handle None vs no return
Parse complex return types
Check yields for generators



Step 3: Implement Completeness Rules
These ensure documentation is complete:

missing_params: Find undocumented parameters

List all function params not in docstring
Generate template for missing params
Skip 'self', 'cls', but flag *args/**kwargs


missing_returns: Check return documentation

Detect functions with return statements but no Returns section
Handle generator functions (yields)
Skip if function only returns None


missing_raises: Find undocumented exceptions

AST parse function body for raise statements
Compare with documented exceptions
Only flag direct raises, not re-raises



Step 4: Implement Consistency Rules
These check for inconsistencies in documented values:

default_value_mismatch: Verify default values match

Compare function defaults with documented defaults
Handle None, strings, numbers, booleans
Parse simple expressions


parameter_order: Check if parameter order matches

Order should be consistent (excluding keyword-only)
This is "medium" severity as it's not breaking



Step 5: Implement Check Method
The main entry point should:

Accept a MatchedPair
Run all enabled rules
Aggregate results
Return RuleCheckResult with timing

Rule Implementation Example
Here's the pattern each rule should follow:
pythondef _check_parameter_names(self, pair: MatchedPair) -> RuleCheckResult:
    """Check parameter name consistency."""
    start_time = time.time()
    issues = []

    # Extract parameters from function and docstring
    func_params = self._get_function_params(pair.function)
    doc_params = self._get_doc_params(pair.documentation)

    # Compare and generate issues
    # ... implementation ...

    return RuleCheckResult(
        rule_name="parameter_names",
        passed=len(issues) == 0,
        confidence=0.95 if issues else 1.0,
        issues=issues,
        execution_time_ms=(time.time() - start_time) * 1000
    )
Performance Optimizations

Use set operations for name comparisons
Cache parsed type strings
Early return on first critical issue (if configured)
Avoid AST parsing in rules (should be done in parser)


CHUNK 3: Rule Engine Advanced Features and Integration (codedocsync/analyzer/rule_engine_utils.py and completion of rule_engine.py)
Context for AI Agent
You are completing the rule engine implementation by adding utility functions, suggestion generation, and integration with the broader system. This chunk focuses on making the rules actionable and useful.
Critical Implementation Requirements

Create intelligent suggestion generators for each issue type
Implement utility functions for type parsing and comparison
Add configuration support for customizing rule behavior
Integrate with the analyzer module's public API

Detailed Implementation Guide
Step 1: Create rule_engine_utils.py
This file contains shared utilities for rule implementation:

Type parsing utilities:

normalize_type_string(): Convert variations to canonical form
compare_types(): Check if two type strings are equivalent
extract_base_type(): Get base type from Optional, List, etc.


Parameter extraction helpers:

extract_function_params(): Get params from FunctionSignature
extract_doc_params(): Get params from ParsedDocstring
should_ignore_param(): Check if param should be skipped (self, cls)


Suggestion generators:

generate_parameter_suggestion(): Create fix for parameter issues
generate_docstring_template(): Create template for missing docs
format_code_suggestion(): Format suggestions as markdown code blocks



Step 2: Complete Advanced Rules in rule_engine.py
Add these sophisticated rules:

**Check for undocumented *args and kwargs:

These often lack documentation
Suggest generic templates if missing


Validate Optional parameters:

Parameters with defaults should be Optional in type hints
Check consistency between signature and docs


Check class method decorators:

@staticmethod shouldn't have 'self'
@classmethod should have 'cls'



Step 3: Implement Suggestion Generation
Each issue needs an actionable suggestion. Examples:
For parameter_name_mismatch:
Suggestion: "Update the docstring parameter name from 'userId' to 'user_id':\n\n
Args:
    user_id: The user's identifier  # Changed from 'userId'
    action: The action to perform
"
For missing_params:
Suggestion: "Add the following parameters to your docstring:\n\n
Args:
    ...existing params...
    cache_timeout: [TODO: Add description]
    retry_count: [TODO: Add description]
"
Step 4: Add Configuration Support
Allow users to customize rule behavior:

Rule enablement: Which rules to run
Severity overrides: Change default severities
Confidence thresholds: When to skip LLM
Performance mode: Skip expensive rules

Step 5: Create Integration Function
Add to rule_engine.py:
pythondef check_matched_pair(
    self,
    pair: MatchedPair,
    enabled_rules: Optional[List[str]] = None,
    confidence_threshold: float = 0.9
) -> List[InconsistencyIssue]:
    """Main entry point for rule checking."""
This should:

Run only enabled rules (or all if None)
Aggregate all issues
Sort by severity
Return formatted issues

Utility Function Examples
Type comparison should handle common variations:
python# These should be considered equivalent:
"str" == "string"
"int" == "integer"
"List[str]" == "list[str]"
"Optional[int]" == "int | None"
"Dict[str, Any]" == "dict"
Error Handling
Remember to handle:

Malformed docstrings gracefully
Missing type annotations
Complex default values that can't be parsed
Recursive or complex types


CHUNK 4: LLM Analyzer and Prompt Templates (codedocsync/analyzer/llm_analyzer.py and prompt_templates.py)
Context for AI Agent
You are implementing the LLM-powered analysis component that handles complex semantic inconsistencies the rule engine can't catch. This includes behavioral mismatches, outdated descriptions, and nuanced issues.
Critical Implementation Requirements

LLM calls only happen when rule confidence is low (<0.9)
Implement retry logic and fallbacks for API failures
Use structured prompts for consistent output
Cache LLM results aggressively
Performance target: <500ms per function (cached)

Detailed Implementation Guide
Step 1: Create prompt_templates.py
Design modular, reusable prompts:

Base analysis prompt template:

Include function signature, docstring, and source code
Specify exact JSON output format
Provide examples of good analysis


Specialized prompts for:

Behavioral consistency checking
Example code validation
Edge case detection
Description accuracy


Prompt engineering best practices:

Use clear role definition
Provide specific examples
Request structured JSON output
Include confidence scores



Example structure:
pythonBEHAVIOR_ANALYSIS_PROMPT = """
You are analyzing Python function documentation for consistency.

Function signature:
{signature}

Current documentation:
{docstring}

Function implementation:
{source_code}

Check if the documentation accurately describes what the function does.
Look for:
1. Outdated behavior descriptions
2. Missing edge cases
3. Incorrect examples
4. Side effects not documented

Return JSON:
{
  "issues": [
    {
      "type": "behavior_mismatch",
      "description": "specific issue",
      "suggestion": "how to fix",
      "confidence": 0.0-1.0
    }
  ]
}
"""
Step 2: Create llm_analyzer.py
Implement the LLMAnalyzer class:

Initialize with configuration:

LLM provider (OpenAI, Anthropic, etc.)
Model selection
Temperature settings
Retry configuration


Implement analyze_consistency method:

Accept MatchedPair and rule results
Determine which LLM checks to run
Format prompts with actual data
Parse and validate LLM responses


Add specialized analysis methods:

analyze_behavior(): Check if description matches implementation
analyze_examples(): Validate code examples work
analyze_edge_cases(): Find undocumented edge cases
analyze_version_info(): Check if version/deprecation info is current


Implement caching layer:

Generate cache key from function + docstring + model
Store results in SQLite
Set TTL for cache entries
Handle cache invalidation


Add retry logic:

Exponential backoff for rate limits
Fallback to simpler models
Graceful degradation on failures



Step 3: Implement Response Parsing
Create robust parsing for LLM outputs:

JSON extraction: Handle markdown code blocks
Validation: Ensure required fields exist
Normalization: Convert to InconsistencyIssue objects
Confidence adjustment: LLM issues typically have lower confidence

Step 4: Integration with Rule Engine
The LLM analyzer should:

Receive rule engine results
Skip analysis if high-confidence rules found critical issues
Focus on areas rules couldn't analyze
Merge results appropriately

Error Handling Strategy
pythonasync def analyze_with_retry(self, pair: MatchedPair, max_retries: 3):
    """Analyze with comprehensive error handling."""
    errors = []

    for attempt in range(max_retries):
        try:
            return await self._analyze_internal(pair)
        except RateLimitError as e:
            wait_time = 2 ** attempt * 1.0
            await asyncio.sleep(wait_time)
            errors.append(e)
        except APIError as e:
            # Try fallback model
            if self.fallback_model:
                self.current_model = self.fallback_model
            errors.append(e)

    # Return degraded results
    return self._create_fallback_result(pair, errors)
Caching Implementation
The cache should:

Use SQLite for persistence
Include model version in cache key
Implement LRU eviction in memory
Support manual invalidation


CHUNK 5: Integration, Testing, and CLI Commands (codedocsync/analyzer/integration.py, tests, and CLI updates)
Context for AI Agent
You are completing the analyzer module by implementing the main integration point, comprehensive tests, and CLI commands. This final chunk brings everything together into a working system.
Critical Implementation Requirements

Create the main analyze_matched_pair function that orchestrates everything
Implement comprehensive test coverage for all components
Add new CLI commands for analysis
Ensure performance targets are met
Handle all edge cases gracefully

Detailed Implementation Guide
Step 1: Create analyzer/integration.py
Implement the main orchestration function:
pythonasync def analyze_matched_pair(
    pair: MatchedPair,
    config: Optional[AnalysisConfig] = None,
    cache: Optional[AnalysisCache] = None,
    rule_engine: Optional[RuleEngine] = None,
    llm_analyzer: Optional[LLMAnalyzer] = None
) -> AnalysisResult:
    """
    Main entry point for analyzing a matched function-documentation pair.

    Flow:
    1. Run rule engine checks (fast path)
    2. Determine if LLM analysis needed based on confidence
    3. Run LLM analysis if required
    4. Merge and sort results
    5. Generate final suggestions
    """
Implementation details:

Initialize components if not provided
Check cache first
Run rules, measure performance
Decide on LLM based on rule confidence
Merge results maintaining severity order
Cache final results

Step 2: Create Comprehensive Tests
Create tests/analyzer/ directory with:

test_models.py: Validate all data models

Test validation logic
Test edge cases
Test serialization


test_rule_engine.py: Test each rule

Parameter name matching (exact, case differences)
Type parsing and comparison
Missing parameter detection
Performance benchmarks


test_llm_analyzer.py: Test LLM integration

Mock LLM responses
Test retry logic
Test caching
Test fallback behavior


test_integration.py: End-to-end tests

Complete analysis flow
Performance requirements
Error handling
Cache behavior



Step 3: Update CLI Commands
Add to cli/main.py:

analyze command (update existing):
python@app.command()
def analyze(
    path: Path,
    --rules-only: bool = False,  # Skip LLM analysis
    --confidence-threshold: float = 0.9,
    --cache-dir: Optional[Path] = None,
    --parallel: bool = True
):

analyze-function command (new):
python@app.command()
def analyze_function(
    file: Path,
    function_name: str,
    --verbose: bool = False
):
"""Analyze a specific function in detail."""

clear-cache command (new):
python@app.command()
def clear_cache(
    --llm-only: bool = False,
    --older-than-days: int = 7
):
"""Clear analysis cache."""


Step 4: Performance Optimization
Ensure targets are met:

Parallel analysis:

Use asyncio for LLM calls
Process multiple files concurrently
Batch cache lookups


Memory optimization:

Stream large files
Limit concurrent analyses
Clear caches periodically


Progress tracking:

Show progress bar during analysis
Display current file/function
Show cache hit rate



Step 5: Integration Testing
Create integration tests that verify:

Full pipeline flow:
python# Parse file -> Match functions -> Analyze -> Report
result = await full_analyze_file("test_file.py")
assert len(result.issues) > 0
assert result.performance.total_ms < 1000

Configuration handling:

Custom severity mappings
Disabled rules
LLM provider switching


Error recovery:

Malformed files
API failures
Missing dependencies



Final Integration Checklist
Ensure these work together:

✓ Parser output feeds into Matcher
✓ Matcher output feeds into Analyzer
✓ Rule engine runs first, fast
✓ LLM only runs when needed
✓ Results are cached appropriately
✓ CLI provides useful commands
✓ Performance targets are met
✓ All errors handled gracefully

Example Test Case
python@pytest.mark.asyncio
async def test_complete_analysis_flow():
    """Test the complete analysis pipeline."""
    # Setup
    test_file = create_test_file("""
    def process_user(user_id: int, action: str = 'view'):
        '''Process user action.

        Args:
            userId: User ID  # Wrong name!
            action: Action to perform
        '''
        return {"user": user_id, "action": action}
    """)

    # Parse
    functions = parse_python_file(test_file)
    assert len(functions) == 1

    # Match (direct matcher should handle this)
    pairs = match_functions(functions)
    assert len(pairs) == 1

    # Analyze
    result = await analyze_matched_pair(pairs[0])

    # Verify
    assert len(result.issues) >= 1
    assert any(i.issue_type == "parameter_name_mismatch" for i in result.issues)
    assert result.analysis_time_ms < 100  # Should be fast (rules only)
    assert not result.used_llm  # High confidence rule issue
