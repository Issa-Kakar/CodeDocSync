"""
LLM output parsing and validation utilities.

This module handles parsing LLM responses into structured InconsistencyIssue objects
with comprehensive validation and error recovery. Supports both successful parsing
and graceful degradation for malformed responses.

Critical Requirements:
- Each issue must have valid issue_type from ISSUE_TYPES
- Confidence must be 0.0-1.0
- Line numbers must be positive integers
- Suggestions must be actionable (not vague)
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .models import ISSUE_TYPES, InconsistencyIssue
from .prompt_templates import map_llm_issue_type

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""

    issues: list[InconsistencyIssue]
    success: bool
    error_message: str | None = None
    raw_response: str | None = None
    confidence: float = 0.0
    analysis_notes: str | None = None


class LLMOutputParser:
    """Parse and validate LLM responses."""

    def __init__(self, strict_validation: bool = True) -> None:
        """
        Initialize parser.

        Args:
            strict_validation: If True, reject responses with any validation errors.
                              If False, attempt to recover what we can.
        """
        self.strict_validation = strict_validation

    def parse_analysis_response(
        self, raw_response: str, expected_format: str = "json"
    ) -> ParseResult:
        """
        Parse LLM response into structured issues.

        Args:
            raw_response: Raw text response from LLM
            expected_format: Expected format ('json' is only supported format)

        Returns:
            ParseResult with parsed issues or error information
        """
        if expected_format != "json":
            return ParseResult(
                issues=[],
                success=False,
                error_message=f"Unsupported format: {expected_format}",
                raw_response=raw_response,
            )

        return self._parse_json_response(raw_response)

    def _parse_json_response(self, raw_response: str) -> ParseResult:
        """Parse JSON response from LLM."""
        try:
            # Try to parse as JSON
            response_data = self._extract_and_parse_json(raw_response)

            # Validate structure
            validation_result = self._validate_response_structure(response_data)
            if not validation_result[0]:
                if self.strict_validation:
                    return ParseResult(
                        issues=[],
                        success=False,
                        error_message=f"Invalid response structure: {validation_result[1]}",
                        raw_response=raw_response,
                    )
                else:
                    # Try to extract what we can
                    return self._recover_from_malformed_response(
                        raw_response, response_data
                    )

            # Parse issues
            issues = self._parse_issues_from_response(response_data)

            # Extract metadata
            confidence = response_data.get("confidence", 0.0)
            analysis_notes = response_data.get("analysis_notes", "")

            return ParseResult(
                issues=issues,
                success=True,
                confidence=confidence,
                analysis_notes=analysis_notes,
                raw_response=raw_response,
            )

        except json.JSONDecodeError as e:
            return self._handle_json_error(raw_response, str(e))
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return ParseResult(
                issues=[],
                success=False,
                error_message=f"Unexpected parsing error: {e}",
                raw_response=raw_response,
            )

    def _extract_and_parse_json(self, raw_response: str) -> dict[str, Any]:
        """Extract JSON from response that might have extra text."""
        # First try direct parsing
        try:
            return json.loads(raw_response.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text using a more robust approach
        # Look for JSON that starts with { and contains "issues"
        start_idx = raw_response.find("{")
        while start_idx != -1:
            # Try to find a valid JSON object starting from this position
            brace_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False

            for i in range(start_idx, len(raw_response)):
                char = raw_response[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

            if end_idx > start_idx:
                potential_json = raw_response[start_idx:end_idx]
                if '"issues"' in potential_json:
                    try:
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        pass

            # Look for next {
            start_idx = raw_response.find("{", start_idx + 1)

        # If all else fails, try to parse the whole thing
        return json.loads(raw_response)

    def _validate_response_structure(
        self, response_data: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Validate that response has expected structure.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(response_data, dict):
            return False, "Response must be a JSON object"

        # Must have 'issues' field that's a list
        if "issues" not in response_data:
            return False, "Response must have 'issues' field"

        if not isinstance(response_data["issues"], list):
            return False, "'issues' field must be a list"

        # Must have 'confidence' field that's a number
        if "confidence" not in response_data:
            return False, "Response must have 'confidence' field"

        if not isinstance(response_data["confidence"], int | float):
            return False, "'confidence' field must be a number"

        # Validate confidence range
        if not 0 <= response_data["confidence"] <= 1:
            return (
                False,
                f"'confidence' must be between 0 and 1, got {response_data['confidence']}",
            )

        return True, ""

    def _parse_issues_from_response(
        self, response_data: dict[str, Any]
    ) -> list[InconsistencyIssue]:
        """Parse individual issues from validated response data."""
        issues = []

        for i, issue_data in enumerate(response_data["issues"]):
            try:
                issue = self._parse_single_issue(issue_data)
                if issue:
                    issues.append(issue)
            except Exception as e:
                logger.warning(f"Failed to parse issue {i}: {e}")
                if self.strict_validation:
                    raise
                # In non-strict mode, skip problematic issues
                continue

        return issues

    def _parse_single_issue(
        self, issue_data: dict[str, Any]
    ) -> InconsistencyIssue | None:
        """Parse a single issue from LLM response data."""
        if not isinstance(issue_data, dict):
            raise ValueError("Issue must be a JSON object")

        # Required fields
        required_fields = [
            "type",
            "description",
            "suggestion",
            "confidence",
            "line_number",
        ]
        for field in required_fields:
            if field not in issue_data:
                raise ValueError(f"Issue missing required field: {field}")

        # Extract and validate fields
        llm_issue_type = issue_data["type"]
        description = issue_data["description"]
        suggestion = issue_data["suggestion"]
        confidence = issue_data["confidence"]
        line_number = issue_data["line_number"]
        details = issue_data.get("details", {})

        # Validate types
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be non-empty string")

        if not isinstance(suggestion, str) or not suggestion.strip():
            raise ValueError("suggestion must be non-empty string")

        if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
            raise ValueError(
                f"confidence must be number between 0 and 1, got {confidence}"
            )

        if not isinstance(line_number, int) or line_number < 1:
            raise ValueError(f"line_number must be positive integer, got {line_number}")

        # Map LLM issue type to our standard types
        issue_type = map_llm_issue_type(llm_issue_type)

        # Determine severity from our issue type mapping
        severity = ISSUE_TYPES.get(issue_type, "medium")

        # Create InconsistencyIssue
        return InconsistencyIssue(
            issue_type=issue_type,
            severity=severity,
            description=description.strip(),
            suggestion=suggestion.strip(),
            line_number=line_number,
            confidence=float(confidence),
            details=details,
        )

    def _handle_json_error(self, raw_response: str, error_msg: str) -> ParseResult:
        """Handle JSON parsing errors with recovery attempts."""
        if self.strict_validation:
            return ParseResult(
                issues=[],
                success=False,
                error_message=f"JSON parsing failed: {error_msg}",
                raw_response=raw_response,
            )

        # Try to recover something useful from the response
        return self._recover_from_malformed_response(raw_response, None)

    def _recover_from_malformed_response(
        self, raw_response: str, partial_data: dict[str, Any] | None = None
    ) -> ParseResult:
        """Attempt to recover useful information from malformed response."""
        issues = []

        # Try to extract any issues using regex
        recovered_issues = self._extract_issues_with_regex(raw_response)
        issues.extend(recovered_issues)

        # If we have partial JSON data, try to use it
        if partial_data and isinstance(partial_data, dict):
            if "issues" in partial_data and isinstance(partial_data["issues"], list):
                for issue_data in partial_data["issues"]:
                    try:
                        issue = self._parse_single_issue(issue_data)
                        if issue:
                            issues.append(issue)
                    except Exception:
                        continue

        return ParseResult(
            issues=issues,
            success=len(issues) > 0,
            error_message=(
                "Recovered from malformed response"
                if issues
                else "Could not recover any issues"
            ),
            raw_response=raw_response,
            confidence=0.3,  # Low confidence for recovered data
        )

    def _extract_issues_with_regex(self, text: str) -> list[InconsistencyIssue]:
        """Extract issues using regex patterns as fallback."""
        issues = []

        # Pattern to find issue-like content
        patterns = [
            r"(?:issue|problem|error):\s*(.+?)(?:suggestion|fix):\s*(.+?)(?:\n|$)",
            r'(?:description):\s*"([^"]+)".*?(?:suggestion):\s*"([^"]+)"',
            r"- (.+?)\s*\(line\s*(\d+)\)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    if len(match.groups()) >= 2:
                        description = match.group(1).strip()
                        suggestion = (
                            match.group(2).strip()
                            if len(match.groups()) > 1
                            else "Review manually"
                        )
                        line_number = 1  # Default line number

                        # Try to extract line number if present
                        line_match = re.search(
                            r"line\s*(\d+)", description + suggestion
                        )
                        if line_match:
                            line_number = int(line_match.group(1))

                        if description and suggestion:
                            issues.append(
                                InconsistencyIssue(
                                    issue_type="description_outdated",
                                    severity="medium",
                                    description=description[
                                        :200
                                    ],  # Truncate if too long
                                    suggestion=suggestion[:200],
                                    line_number=line_number,
                                    confidence=0.3,  # Low confidence for regex-extracted issues
                                    details={"extracted_by": "regex_fallback"},
                                )
                            )
                except Exception:
                    continue

        return issues

    def validate_issue_actionability(self, issue: InconsistencyIssue) -> bool:
        """
        Check if an issue suggestion is actionable (not vague).

        Args:
            issue: Issue to validate

        Returns:
            True if suggestion is actionable, False if too vague
        """
        suggestion = issue.suggestion.lower().strip()

        # Vague phrases that indicate non-actionable suggestions
        vague_phrases = [
            "fix this",
            "review this",
            "check this",
            "update this",
            "improve this",
            "clarify this",
            "consider this",
            "think about",
            "look at",
            "make sure",
            "be careful",
            "pay attention",
        ]

        # Check if suggestion is too short or contains only vague phrases
        if len(suggestion) < 10:
            return False

        for phrase in vague_phrases:
            if phrase in suggestion and len(suggestion) < 50:
                return False

        # Check if suggestion contains specific actions
        actionable_indicators = [
            "add",
            "remove",
            "change",
            "replace",
            "specify",
            "include",
            "document",
            "mention",
            "note that",
            "explain",
            "describe",
            "list",
            "provide example",
            "use",
            "implement",
        ]

        for indicator in actionable_indicators:
            if indicator in suggestion:
                return True

        return False

    def filter_actionable_issues(
        self, issues: list[InconsistencyIssue]
    ) -> list[InconsistencyIssue]:
        """Filter issues to only include those with actionable suggestions."""
        actionable = []
        for issue in issues:
            if self.validate_issue_actionability(issue):
                actionable.append(issue)
            else:
                logger.debug(
                    f"Filtered out non-actionable issue: {issue.description[:50]}..."
                )

        return actionable

    def get_parsing_statistics(self, results: list[ParseResult]) -> dict[str, Any]:
        """Get statistics about parsing success rates."""
        if not results:
            return {
                "total_responses": 0,
                "successful_parses": 0,
                "success_rate": 0.0,
                "average_issues_per_response": 0.0,
                "common_errors": [],
            }

        successful = [r for r in results if r.success]
        total_issues = sum(len(r.issues) for r in results)

        # Count error types
        error_counts: dict[str, int] = {}
        for result in results:
            if not result.success and result.error_message:
                error_type = result.error_message.split(":")[0]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {
            "total_responses": len(results),
            "successful_parses": len(successful),
            "success_rate": len(successful) / len(results),
            "average_issues_per_response": total_issues / len(results),
            "common_errors": sorted(
                error_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


# Convenience functions for common parsing scenarios
def parse_llm_response(raw_response: str, strict: bool = True) -> ParseResult:
    """
    Parse a single LLM response.

    Args:
        raw_response: Raw text response from LLM
        strict: Whether to use strict validation

    Returns:
        ParseResult with parsed issues
    """
    parser = LLMOutputParser(strict_validation=strict)
    return parser.parse_analysis_response(raw_response)


def parse_and_filter_response(
    raw_response: str, filter_actionable: bool = True
) -> list[InconsistencyIssue]:
    """
    Parse response and return filtered list of actionable issues.

    Args:
        raw_response: Raw text response from LLM
        filter_actionable: Whether to filter out non-actionable suggestions

    Returns:
        List of validated InconsistencyIssue objects
    """
    parser = LLMOutputParser(strict_validation=False)
    result = parser.parse_analysis_response(raw_response)

    if not result.success:
        logger.warning(f"Failed to parse LLM response: {result.error_message}")
        return []

    issues = result.issues

    if filter_actionable:
        issues = parser.filter_actionable_issues(issues)

    return issues


def validate_response_format(response_data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate LLM response format without full parsing.

    Args:
        response_data: Parsed JSON response

    Returns:
        Tuple of (is_valid, error_message)
    """
    parser = LLMOutputParser()
    return parser._validate_response_structure(response_data)
