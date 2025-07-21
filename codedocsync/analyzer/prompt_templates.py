"""
Prompt templates for LLM-powered documentation analysis.

This module provides modular, reusable prompts for different types of semantic
analysis that the rule engine cannot catch. All prompts follow best practices:
- Clear role definition
- Specific examples
- Structured JSON output
- Confidence scores
"""

from typing import Any

# Base template for all analysis prompts
BASE_ANALYSIS_PROMPT = """You are an expert Python documentation analyzer. Your job is to identify inconsistencies between function implementations and their documentation that automated rules cannot catch.

You will analyze a function's signature, implementation, and current documentation to find semantic issues like:
- Outdated behavior descriptions
- Missing edge cases
- Incorrect examples
- Undocumented side effects
- Behavioral mismatches

CRITICAL: Return ONLY valid JSON in the exact format specified. No markdown, no code blocks, no extra text."""


# Behavioral consistency analysis
BEHAVIOR_ANALYSIS_PROMPT = (
    BASE_ANALYSIS_PROMPT
    + """

FUNCTION SIGNATURE:
{signature}

CURRENT DOCUMENTATION:
{docstring}

FUNCTION IMPLEMENTATION:
{source_code}

CONTEXT:
Rule-based analysis found: {rule_issues}

TASK: Check if the documentation accurately describes what the function actually does.

Look specifically for:
1. Behavior described in docstring vs actual implementation
2. Edge cases handled in code but not documented
3. Error conditions that should be documented
4. Side effects (file I/O, network calls, state changes) not mentioned
5. Return value descriptions that don't match implementation

Return JSON in this EXACT format:
{{
  "issues": [
    {{
      "type": "behavior_mismatch",
      "description": "Specific description of the inconsistency",
      "suggestion": "Actionable fix for the documentation",
      "confidence": 0.85,
      "line_number": 10,
      "details": {{"expected": "described behavior", "actual": "implemented behavior"}}
    }}
  ],
  "analysis_notes": "Brief explanation of what was checked",
  "confidence": 0.90
}}

EXAMPLE RESPONSE:
{{
  "issues": [
    {{
      "type": "behavior_mismatch",
      "description": "Function handles None input gracefully but docstring doesn't mention this behavior",
      "suggestion": "Add note about None handling: 'If input is None, returns empty list.'",
      "confidence": 0.90,
      "line_number": 5,
      "details": {{"missing_behavior": "None handling"}}
    }}
  ],
  "analysis_notes": "Checked parameter handling, return value consistency, and error conditions",
  "confidence": 0.88
}}"""
)


# Example code validation
EXAMPLE_VALIDATION_PROMPT = (
    BASE_ANALYSIS_PROMPT
    + """

FUNCTION SIGNATURE:
{signature}

FUNCTION IMPLEMENTATION:
{source_code}

DOCUMENTATION WITH EXAMPLES:
{docstring}

TASK: Validate that any code examples in the documentation actually work with the current implementation.

Check for:
1. Syntax errors in examples
2. Examples that would raise exceptions
3. Examples that don't match current parameter names/types
4. Examples showing outdated usage patterns
5. Missing imports or context needed for examples

Return JSON in this EXACT format:
{{
  "issues": [
    {{
      "type": "example_invalid",
      "description": "Specific problem with the example",
      "suggestion": "Corrected example or explanation",
      "confidence": 0.95,
      "line_number": 15,
      "details": {{"example_line": "problematic code", "error": "specific error"}}
    }}
  ],
  "analysis_notes": "Summary of example validation",
  "confidence": 0.92
}}"""
)


# Edge case detection
EDGE_CASE_ANALYSIS_PROMPT = (
    BASE_ANALYSIS_PROMPT
    + """

FUNCTION SIGNATURE:
{signature}

FUNCTION IMPLEMENTATION:
{source_code}

CURRENT DOCUMENTATION:
{docstring}

TASK: Identify edge cases handled in the implementation but not documented.

Look for:
1. Input validation (empty strings, None, negative numbers)
2. Boundary conditions (empty lists, single items, very large inputs)
3. Type coercion or conversion
4. Fallback behaviors
5. Resource constraints (memory, file limits)

Return JSON in this EXACT format:
{{
  "issues": [
    {{
      "type": "missing_edge_case",
      "description": "Edge case handled in code but not documented",
      "suggestion": "Documentation addition for the edge case",
      "confidence": 0.80,
      "line_number": 8,
      "details": {{"condition": "specific condition", "behavior": "what happens"}}
    }}
  ],
  "analysis_notes": "Edge cases checked",
  "confidence": 0.85
}}"""
)


# Version and deprecation analysis
VERSION_ANALYSIS_PROMPT = (
    BASE_ANALYSIS_PROMPT
    + """

FUNCTION SIGNATURE:
{signature}

FUNCTION IMPLEMENTATION:
{source_code}

CURRENT DOCUMENTATION:
{docstring}

TASK: Check if version information, deprecation warnings, or "since" notes in documentation are current and accurate.

Look for:
1. Outdated version numbers
2. Deprecated parameters still documented as current
3. New features not marked with version info
4. Incorrect deprecation warnings
5. Missing migration guidance

Return JSON in this EXACT format:
{{
  "issues": [
    {{
      "type": "version_info_outdated",
      "description": "Specific version/deprecation issue",
      "suggestion": "Updated version information",
      "confidence": 0.75,
      "line_number": 12,
      "details": {{"current_version": "mentioned", "actual_status": "current state"}}
    }}
  ],
  "analysis_notes": "Version info checked",
  "confidence": 0.80
}}"""
)


# Type annotation consistency
TYPE_CONSISTENCY_PROMPT = (
    BASE_ANALYSIS_PROMPT
    + """

FUNCTION SIGNATURE:
{signature}

FUNCTION IMPLEMENTATION:
{source_code}

CURRENT DOCUMENTATION:
{docstring}

TASK: Check for subtle type inconsistencies that rule-based analysis might miss.

Look for:
1. Documentation mentions types not reflected in type hints
2. Generic types that could be more specific
3. Union types where documentation only mentions one type
4. Return types that vary based on input but aren't documented
5. Duck typing assumptions not explained

Return JSON in this EXACT format:
{{
  "issues": [
    {{
      "type": "type_documentation_mismatch",
      "description": "Subtle type inconsistency",
      "suggestion": "Clarification for type handling",
      "confidence": 0.70,
      "line_number": 6,
      "details": {{"documented_type": "what docs say", "actual_behavior": "what code does"}}
    }}
  ],
  "analysis_notes": "Type consistency checked",
  "confidence": 0.75
}}"""
)


# Performance and complexity analysis
PERFORMANCE_ANALYSIS_PROMPT = (
    BASE_ANALYSIS_PROMPT
    + """

FUNCTION SIGNATURE:
{signature}

FUNCTION IMPLEMENTATION:
{source_code}

CURRENT DOCUMENTATION:
{docstring}

TASK: Check if performance characteristics mentioned in documentation match the implementation.

Look for:
1. Time complexity claims (O(n), O(log n)) vs actual algorithm
2. Memory usage warnings that don't match implementation
3. Performance tips that are outdated
4. Scalability notes that need updating
5. Caching or optimization mentions

Return JSON in this EXACT format:
{{
  "issues": [
    {{
      "type": "performance_mismatch",
      "description": "Performance documentation inconsistency",
      "suggestion": "Updated performance information",
      "confidence": 0.65,
      "line_number": 20,
      "details": {{"claimed_complexity": "documented", "actual_complexity": "implemented"}}
    }}
  ],
  "analysis_notes": "Performance characteristics checked",
  "confidence": 0.70
}}"""
)


# Prompt selection mapping
PROMPT_TEMPLATES = {
    "behavior_analysis": BEHAVIOR_ANALYSIS_PROMPT,
    "example_validation": EXAMPLE_VALIDATION_PROMPT,
    "edge_case_analysis": EDGE_CASE_ANALYSIS_PROMPT,
    "version_analysis": VERSION_ANALYSIS_PROMPT,
    "type_consistency": TYPE_CONSISTENCY_PROMPT,
    "performance_analysis": PERFORMANCE_ANALYSIS_PROMPT,
}


def get_prompt_template(analysis_type: str) -> str:
    """
    Get a prompt template for a specific analysis type.

    Args:
        analysis_type: Type of analysis ('behavior_analysis', 'example_validation', etc.)

    Returns:
        str: The prompt template

    Raises:
        ValueError: If analysis_type is not supported
    """
    if analysis_type not in PROMPT_TEMPLATES:
        available = list(PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown analysis type '{analysis_type}'. Available: {available}"
        )

    return PROMPT_TEMPLATES[analysis_type]


def format_prompt(
    analysis_type: str,
    signature: str,
    source_code: str,
    docstring: str,
    rule_issues: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Format a prompt template with actual function data.

    Args:
        analysis_type: Type of analysis to perform
        signature: Function signature string
        source_code: Function implementation source code
        docstring: Current documentation
        rule_issues: Optional summary of rule-based issues found
        **kwargs: Additional formatting variables

    Returns:
        str: Formatted prompt ready for LLM

    Raises:
        ValueError: If analysis_type is not supported
        KeyError: If required formatting variables are missing
    """
    template = get_prompt_template(analysis_type)

    # Base formatting variables
    format_vars = {
        "signature": signature,
        "source_code": source_code,
        "docstring": docstring,
    }

    # Add rule_issues if provided
    if rule_issues is not None:
        format_vars["rule_issues"] = rule_issues

    # Add any additional variables
    format_vars.update(kwargs)

    try:
        return template.format(**format_vars)
    except KeyError as e:
        raise KeyError(f"Missing required formatting variable: {e}") from e


def get_available_analysis_types() -> list[str]:
    """Get list of available analysis types."""
    return list(PROMPT_TEMPLATES.keys())


def validate_llm_response(response_data: dict[str, Any]) -> bool:
    """
    Validate that an LLM response has the expected structure.

    Args:
        response_data: Parsed JSON response from LLM

    Returns:
        bool: True if response has valid structure
    """
    if not isinstance(response_data, dict):
        return False

    # Must have 'issues' field that's a list
    if "issues" not in response_data or not isinstance(response_data["issues"], list):
        return False

    # Must have 'confidence' field that's a number
    if "confidence" not in response_data or not isinstance(
        response_data["confidence"], int | float
    ):
        return False

    # Check each issue has required fields
    for issue in response_data["issues"]:
        if not isinstance(issue, dict):
            return False

        required_fields = [
            "type",
            "description",
            "suggestion",
            "confidence",
            "line_number",
        ]
        for field in required_fields:
            if field not in issue:
                return False

        # Validate confidence is a number between 0 and 1
        if (
            not isinstance(issue["confidence"], int | float)
            or not 0 <= issue["confidence"] <= 1
        ):
            return False

    return True


# Issue type mapping from LLM responses to our constants
LLM_ISSUE_TYPE_MAPPING = {
    "behavior_mismatch": "description_outdated",
    "example_invalid": "example_invalid",
    "missing_edge_case": "description_outdated",
    "version_info_outdated": "description_outdated",
    "type_documentation_mismatch": "parameter_type_mismatch",
    "performance_mismatch": "description_outdated",
}


def map_llm_issue_type(llm_type: str) -> str:
    """
    Map LLM-specific issue types to our standard issue types.

    Args:
        llm_type: Issue type from LLM response

    Returns:
        str: Mapped issue type from our ISSUE_TYPES constants
    """
    return LLM_ISSUE_TYPE_MAPPING.get(llm_type, "description_outdated")
