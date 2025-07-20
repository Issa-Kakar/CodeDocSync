"""
Test suite for rule engine implementation.

Tests each rule implementation, performance requirements, and edge cases.
"""

import time

from codedocsync.analyzer.rule_engine import RuleEngine
from codedocsync.matcher import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser import (
    DocstringParameter,
    DocstringReturns,
    FunctionParameter,
    FunctionSignature,
    ParsedDocstring,
    ParsedFunction,
    RawDocstring,
)


class TestRuleEngine:
    """Test RuleEngine class functionality."""

    def test_rule_engine_initialization(self):
        """Test RuleEngine initialization with different configurations."""
        # Default initialization
        engine = RuleEngine()
        assert engine.enabled_rules is None  # All rules enabled
        assert not engine.performance_mode
        assert engine.confidence_threshold == 0.9

        # Custom initialization
        engine = RuleEngine(
            enabled_rules=["parameter_names", "parameter_types"],
            performance_mode=True,
            confidence_threshold=0.8,
        )
        assert engine.enabled_rules == ["parameter_names", "parameter_types"]
        assert engine.performance_mode
        assert engine.confidence_threshold == 0.8

    def create_mock_function(
        self,
        name: str = "test_function",
        parameters=None,
        return_annotation: str = None,
        file_path: str = "/test/file.py",
        line_number: int = 10,
    ):
        """Create a mock ParsedFunction for testing."""
        if parameters is None:
            parameters = []

        return ParsedFunction(
            signature=FunctionSignature(
                name=name,
                parameters=parameters,
                return_annotation=return_annotation,
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path=file_path,
            line_number=line_number,
        )

    def create_mock_docstring(self, parameters=None, returns=None, format="google"):
        """Create a mock ParsedDocstring for testing."""
        if parameters is None:
            parameters = []

        return ParsedDocstring(
            format=format,
            summary="Test function summary",
            parameters=parameters,
            returns=returns,
            raises=[],
            raw_text="Test docstring",
        )

    def create_matched_pair(self, function, docstring):
        """Create a MatchedPair for testing."""
        return MatchedPair(
            function=function,
            documentation=docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )


class TestParameterNameRule:
    """Test parameter name matching rule."""

    def test_parameter_names_match(self):
        """Test when parameter names match exactly."""
        engine = RuleEngine()

        # Create function with parameters
        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="user_id",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                ),
                FunctionParameter(
                    name="action",
                    type_annotation="str",
                    default_value="'view'",
                    is_required=False,
                ),
            ]
        )

        # Create docstring with matching parameters
        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="user_id", description="User ID", type="int"),
                DocstringParameter(
                    name="action", description="Action to perform", type="str"
                ),
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should have no parameter name issues
        name_issues = [i for i in issues if i.issue_type == "parameter_name_mismatch"]
        assert len(name_issues) == 0

    def test_parameter_names_mismatch(self):
        """Test when parameter names don't match."""
        engine = RuleEngine()

        # Function with user_id parameter
        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="user_id",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                )
            ]
        )

        # Docstring with userId parameter (camelCase)
        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="userId", description="User ID", type="int")
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should detect parameter name mismatch
        name_issues = [i for i in issues if i.issue_type == "parameter_name_mismatch"]
        assert len(name_issues) > 0
        assert "user_id" in name_issues[0].description
        assert "userId" in name_issues[0].description

    def test_missing_parameters_in_docstring(self):
        """Test when function has parameters not documented."""
        engine = RuleEngine()

        # Function with two parameters
        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="user_id",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                ),
                FunctionParameter(
                    name="action",
                    type_annotation="str",
                    default_value="'view'",
                    is_required=False,
                ),
            ]
        )

        # Docstring with only one parameter
        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="user_id", description="User ID", type="int")
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should detect missing parameter
        missing_issues = [i for i in issues if i.issue_type == "missing_params"]
        assert len(missing_issues) > 0
        assert "action" in missing_issues[0].description

    def test_self_parameter_ignored(self):
        """Test that 'self' parameter is ignored in methods."""
        engine = RuleEngine()

        # Function with self parameter (method)
        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="self",
                    type_annotation=None,
                    default_value=None,
                    is_required=True,
                ),
                FunctionParameter(
                    name="user_id",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                ),
            ]
        )

        # Docstring without self parameter (correct)
        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="user_id", description="User ID", type="int")
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should not complain about missing 'self' parameter
        missing_issues = [i for i in issues if "self" in i.description.lower()]
        assert len(missing_issues) == 0


class TestTypeMatchingRule:
    """Test parameter type matching rule."""

    def test_exact_type_match(self):
        """Test when types match exactly."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="count",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                )
            ]
        )

        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="count", description="Count value", type="int")
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should have no type mismatch issues
        type_issues = [i for i in issues if i.issue_type == "parameter_type_mismatch"]
        assert len(type_issues) == 0

    def test_equivalent_type_match(self):
        """Test when types are equivalent but different format."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="name",
                    type_annotation="str",
                    default_value=None,
                    is_required=True,
                )
            ]
        )

        # Docstring uses "string" instead of "str"
        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="name", description="Name value", type="string")
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should not flag equivalent types as mismatches
        type_issues = [i for i in issues if i.issue_type == "parameter_type_mismatch"]
        assert len(type_issues) == 0

    def test_type_mismatch(self):
        """Test when types clearly don't match."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="count",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                )
            ]
        )

        # Docstring has wrong type
        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="count", description="Count value", type="str")
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should detect type mismatch
        type_issues = [i for i in issues if i.issue_type == "parameter_type_mismatch"]
        assert len(type_issues) > 0
        assert "int" in type_issues[0].description
        assert "str" in type_issues[0].description


class TestReturnTypeRule:
    """Test return type validation rule."""

    def test_return_type_match(self):
        """Test when return types match."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function(return_annotation="str")

        docstring = TestRuleEngine().create_mock_docstring(
            returns=DocstringReturns(description="Returns a string", type="str")
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should have no return type issues
        return_issues = [i for i in issues if i.issue_type == "return_type_mismatch"]
        assert len(return_issues) == 0

    def test_missing_return_documentation(self):
        """Test when function returns something but no return docs."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function(return_annotation="str")

        # Docstring without return documentation
        docstring = TestRuleEngine().create_mock_docstring(returns=None)

        pair = TestRuleEngine().create_matched_pair(function, docstring)
        issues = engine.check_matched_pair(pair)

        # Should detect missing return documentation
        return_issues = [i for i in issues if i.issue_type == "missing_returns"]
        assert len(return_issues) > 0


class TestPerformanceRequirements:
    """Test that performance requirements are met."""

    def test_rule_engine_performance(self):
        """Test that rule engine meets <5ms per function target."""
        engine = RuleEngine()

        # Create a typical function with several parameters
        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="user_id",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                ),
                FunctionParameter(
                    name="action",
                    type_annotation="str",
                    default_value="'view'",
                    is_required=False,
                ),
                FunctionParameter(
                    name="timeout",
                    type_annotation="float",
                    default_value="30.0",
                    is_required=False,
                ),
            ],
            return_annotation="dict",
        )

        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(name="user_id", description="User ID", type="int"),
                DocstringParameter(name="action", description="Action", type="str"),
                DocstringParameter(name="timeout", description="Timeout", type="float"),
            ],
            returns=DocstringReturns(description="Result dict", type="dict"),
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)

        # Time the rule checking
        start_time = time.time()
        engine.check_matched_pair(pair)
        execution_time_ms = (time.time() - start_time) * 1000

        # Should complete in under 5ms
        assert (
            execution_time_ms < 5.0
        ), f"Rule engine took {execution_time_ms}ms, should be <5ms"

    def test_performance_mode_faster(self):
        """Test that performance mode is faster than normal mode."""
        normal_engine = RuleEngine(performance_mode=False)
        performance_engine = RuleEngine(performance_mode=True)

        # Create a complex function for testing
        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name=f"param_{i}",
                    type_annotation="str",
                    default_value=None,
                    is_required=True,
                )
                for i in range(10)
            ]
        )

        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(
                    name=f"param_{i}", description=f"Parameter {i}", type="str"
                )
                for i in range(10)
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)

        # Time normal mode
        start_time = time.time()
        normal_engine.check_matched_pair(pair)
        normal_time = time.time() - start_time

        # Time performance mode
        start_time = time.time()
        performance_engine.check_matched_pair(pair)
        performance_time = time.time() - start_time

        # Performance mode should be faster (or at least not significantly slower)
        assert (
            performance_time <= normal_time * 1.5
        ), "Performance mode should be faster"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_docstring(self):
        """Test handling of functions with no docstring."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function()
        # Pair with no documentation
        pair = MatchedPair(
            function=function,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

        # Should handle gracefully without crashing
        issues = engine.check_matched_pair(pair)
        assert isinstance(issues, list)

    def test_raw_docstring_handling(self):
        """Test handling of raw (unparsed) docstring."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function()
        function.docstring = RawDocstring(
            raw_text="This is a raw docstring without structured parsing",
            line_number=11,
        )

        pair = MatchedPair(
            function=function,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

        # Should handle gracefully
        issues = engine.check_matched_pair(pair)
        assert isinstance(issues, list)

    def test_malformed_type_annotations(self):
        """Test handling of malformed or complex type annotations."""
        engine = RuleEngine()

        function = TestRuleEngine().create_mock_function(
            parameters=[
                FunctionParameter(
                    name="complex_param",
                    type_annotation="Dict[str, Optional[List[Union[int, str]]]]",
                    default_value=None,
                    is_required=True,
                )
            ]
        )

        docstring = TestRuleEngine().create_mock_docstring(
            parameters=[
                DocstringParameter(
                    name="complex_param", description="Complex parameter", type="dict"
                )
            ]
        )

        pair = TestRuleEngine().create_matched_pair(function, docstring)

        # Should handle complex types gracefully
        issues = engine.check_matched_pair(pair)
        assert isinstance(issues, list)
