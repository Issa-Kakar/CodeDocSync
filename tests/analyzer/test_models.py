"""
Test suite for analyzer data models.

Tests validation logic, edge cases, and serialization for all analyzer models.
"""

import pytest
from codedocsync.analyzer.models import (
    InconsistencyIssue,
    RuleCheckResult,
    AnalysisResult,
    ISSUE_TYPES,
    SEVERITY_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
)
from codedocsync.matcher import MatchedPair, MatchConfidence, MatchType
from codedocsync.parser import ParsedFunction, FunctionSignature


class TestInconsistencyIssue:
    """Test InconsistencyIssue model validation and functionality."""

    def test_valid_issue_creation(self):
        """Test creating a valid InconsistencyIssue."""
        issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Parameter 'user_id' in function does not match 'userId' in docstring",
            suggestion="Change docstring parameter to 'user_id' to match function signature",
            line_number=45,
            confidence=0.95,
            details={"expected": "user_id", "actual": "userId", "position": 0},
        )

        assert issue.issue_type == "parameter_name_mismatch"
        assert issue.severity == "critical"
        assert issue.confidence == 0.95
        assert issue.line_number == 45
        assert issue.details["expected"] == "user_id"

    def test_invalid_issue_type(self):
        """Test validation of invalid issue type."""
        with pytest.raises(ValueError, match="issue_type must be one of"):
            InconsistencyIssue(
                issue_type="invalid_type",
                severity="critical",
                description="Test description",
                suggestion="Test suggestion",
                line_number=1,
            )

    def test_invalid_severity(self):
        """Test validation of invalid severity."""
        with pytest.raises(ValueError, match="severity must be one of"):
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="very_high",
                description="Test description",
                suggestion="Test suggestion",
                line_number=1,
            )

    def test_invalid_confidence(self):
        """Test validation of confidence range."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Test description",
                suggestion="Test suggestion",
                line_number=1,
                confidence=1.5,
            )

    def test_invalid_line_number(self):
        """Test validation of line number."""
        with pytest.raises(ValueError, match="line_number must be positive"):
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Test description",
                suggestion="Test suggestion",
                line_number=0,
            )

    def test_empty_description(self):
        """Test validation of empty description."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="   ",
                suggestion="Test suggestion",
                line_number=1,
            )

    def test_empty_suggestion(self):
        """Test validation of empty suggestion."""
        with pytest.raises(ValueError, match="suggestion cannot be empty"):
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Test description",
                suggestion="   ",
                line_number=1,
            )

    def test_severity_weight_property(self):
        """Test severity_weight property."""
        issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Test description",
            suggestion="Test suggestion",
            line_number=1,
        )
        assert issue.severity_weight == SEVERITY_WEIGHTS["critical"]

    def test_is_high_confidence(self):
        """Test is_high_confidence method."""
        high_confidence_issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Test description",
            suggestion="Test suggestion",
            line_number=1,
            confidence=0.95,
        )
        assert high_confidence_issue.is_high_confidence()

        low_confidence_issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Test description",
            suggestion="Test suggestion",
            line_number=1,
            confidence=0.5,
        )
        assert not low_confidence_issue.is_high_confidence()


class TestRuleCheckResult:
    """Test RuleCheckResult model validation and functionality."""

    def test_valid_rule_result_creation(self):
        """Test creating a valid RuleCheckResult."""
        result = RuleCheckResult(
            rule_name="parameter_names",
            passed=False,
            confidence=0.95,
            issues=[
                InconsistencyIssue(
                    issue_type="parameter_name_mismatch",
                    severity="critical",
                    description="Test description",
                    suggestion="Test suggestion",
                    line_number=1,
                )
            ],
            execution_time_ms=2.5,
        )

        assert result.rule_name == "parameter_names"
        assert not result.passed
        assert result.confidence == 0.95
        assert len(result.issues) == 1
        assert result.execution_time_ms == 2.5

    def test_invalid_confidence(self):
        """Test validation of confidence range."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            RuleCheckResult(rule_name="test_rule", passed=True, confidence=2.0)

    def test_negative_execution_time(self):
        """Test validation of execution time."""
        with pytest.raises(ValueError, match="execution_time_ms must be non-negative"):
            RuleCheckResult(
                rule_name="test_rule",
                passed=True,
                confidence=1.0,
                execution_time_ms=-1.0,
            )

    def test_empty_rule_name(self):
        """Test validation of rule name."""
        with pytest.raises(ValueError, match="rule_name cannot be empty"):
            RuleCheckResult(rule_name="   ", passed=True, confidence=1.0)

    def test_passed_with_issues_validation(self):
        """Test logical validation: passed=True with issues should fail."""
        issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Test description",
            suggestion="Test suggestion",
            line_number=1,
        )

        with pytest.raises(ValueError, match="passed but has .* issues"):
            RuleCheckResult(
                rule_name="test_rule", passed=True, confidence=1.0, issues=[issue]
            )


class TestAnalysisResult:
    """Test AnalysisResult model validation and functionality."""

    def create_mock_matched_pair(self):
        """Create a mock matched pair for testing."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_function",
                parameters=[],
                return_annotation=None,
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/file.py",
            line_number=10,
        )

        return MatchedPair(
            function=function,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

    def test_valid_analysis_result_creation(self):
        """Test creating a valid AnalysisResult."""
        pair = self.create_mock_matched_pair()

        issues = [
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Test description",
                suggestion="Test suggestion",
                line_number=1,
            )
        ]

        result = AnalysisResult(
            matched_pair=pair,
            issues=issues,
            used_llm=True,
            analysis_time_ms=100.5,
            cache_hit=False,
        )

        assert result.matched_pair == pair
        assert len(result.issues) == 1
        assert result.used_llm
        assert result.analysis_time_ms == 100.5
        assert not result.cache_hit

    def test_negative_analysis_time(self):
        """Test validation of analysis time."""
        pair = self.create_mock_matched_pair()

        with pytest.raises(ValueError, match="analysis_time_ms must be non-negative"):
            AnalysisResult(matched_pair=pair, analysis_time_ms=-1.0)

    def test_critical_issues_property(self):
        """Test critical_issues property."""
        pair = self.create_mock_matched_pair()

        issues = [
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Critical issue",
                suggestion="Test suggestion",
                line_number=1,
            ),
            InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="High issue",
                suggestion="Test suggestion",
                line_number=2,
            ),
        ]

        result = AnalysisResult(matched_pair=pair, issues=issues)
        critical_issues = result.critical_issues

        assert len(critical_issues) == 1
        assert critical_issues[0].severity == "critical"

    def test_has_critical_issues_property(self):
        """Test has_critical_issues property."""
        pair = self.create_mock_matched_pair()

        # With critical issues
        critical_issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Critical issue",
            suggestion="Test suggestion",
            line_number=1,
        )
        result_with_critical = AnalysisResult(
            matched_pair=pair, issues=[critical_issue]
        )
        assert result_with_critical.has_critical_issues

        # Without critical issues
        high_issue = InconsistencyIssue(
            issue_type="missing_returns",
            severity="high",
            description="High issue",
            suggestion="Test suggestion",
            line_number=1,
        )
        result_without_critical = AnalysisResult(matched_pair=pair, issues=[high_issue])
        assert not result_without_critical.has_critical_issues

    def test_get_issues_by_severity(self):
        """Test get_issues_by_severity method."""
        pair = self.create_mock_matched_pair()

        issues = [
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Critical issue",
                suggestion="Test suggestion",
                line_number=1,
            ),
            InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="High issue",
                suggestion="Test suggestion",
                line_number=2,
            ),
            InconsistencyIssue(
                issue_type="parameter_order_different",
                severity="medium",
                description="Medium issue",
                suggestion="Test suggestion",
                line_number=3,
            ),
        ]

        result = AnalysisResult(matched_pair=pair, issues=issues)
        by_severity = result.get_issues_by_severity()

        assert len(by_severity["critical"]) == 1
        assert len(by_severity["high"]) == 1
        assert len(by_severity["medium"]) == 1
        assert len(by_severity["low"]) == 0

    def test_get_sorted_issues(self):
        """Test get_sorted_issues method."""
        pair = self.create_mock_matched_pair()

        issues = [
            InconsistencyIssue(
                issue_type="example_invalid",
                severity="low",
                description="Low issue",
                suggestion="Test suggestion",
                line_number=1,
            ),
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Critical issue",
                suggestion="Test suggestion",
                line_number=2,
            ),
            InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="High issue",
                suggestion="Test suggestion",
                line_number=3,
            ),
        ]

        result = AnalysisResult(matched_pair=pair, issues=issues)
        sorted_issues = result.get_sorted_issues()

        # Should be sorted by severity weight (critical, high, low)
        assert sorted_issues[0].severity == "critical"
        assert sorted_issues[1].severity == "high"
        assert sorted_issues[2].severity == "low"

    def test_get_summary(self):
        """Test get_summary method."""
        pair = self.create_mock_matched_pair()

        issues = [
            InconsistencyIssue(
                issue_type="parameter_name_mismatch",
                severity="critical",
                description="Critical issue",
                suggestion="Test suggestion",
                line_number=1,
            ),
            InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="High issue",
                suggestion="Test suggestion",
                line_number=2,
            ),
        ]

        result = AnalysisResult(
            matched_pair=pair, issues=issues, used_llm=True, analysis_time_ms=150.0
        )

        summary = result.get_summary()

        assert summary["total_issues"] == 2
        assert summary["critical"] == 1
        assert summary["high"] == 1
        assert summary["medium"] == 0
        assert summary["low"] == 0
        assert summary["used_llm"] is True
        assert summary["analysis_time_ms"] == 150.0
        assert summary["function_name"] == "test_function"
        assert summary["file_path"] == "/test/file.py"


class TestConstants:
    """Test module constants."""

    def test_issue_types_completeness(self):
        """Test that all required issue types are defined."""
        required_types = [
            "parameter_name_mismatch",
            "parameter_missing",
            "parameter_type_mismatch",
            "return_type_mismatch",
            "missing_raises",
            "parameter_order_different",
            "description_outdated",
            "example_invalid",
            "missing_params",
            "missing_returns",
            "undocumented_kwargs",
            "type_mismatches",
            "default_mismatches",
            "parameter_count_mismatch",
        ]

        for issue_type in required_types:
            assert issue_type in ISSUE_TYPES

    def test_severity_weights(self):
        """Test severity weights are properly ordered."""
        assert SEVERITY_WEIGHTS["critical"] > SEVERITY_WEIGHTS["high"]
        assert SEVERITY_WEIGHTS["high"] > SEVERITY_WEIGHTS["medium"]
        assert SEVERITY_WEIGHTS["medium"] > SEVERITY_WEIGHTS["low"]

    def test_confidence_thresholds(self):
        """Test confidence thresholds are properly ordered."""
        assert (
            CONFIDENCE_THRESHOLDS["HIGH_CONFIDENCE"]
            > CONFIDENCE_THRESHOLDS["MEDIUM_CONFIDENCE"]
        )
        assert (
            CONFIDENCE_THRESHOLDS["MEDIUM_CONFIDENCE"]
            > CONFIDENCE_THRESHOLDS["LOW_CONFIDENCE"]
        )
