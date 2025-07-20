"""
Tests for the Edge Case Handlers.

Tests cover special Python constructs like properties, class methods, magic methods,
async functions, and other edge cases requiring specialized documentation.
"""

import pytest
from unittest.mock import Mock

from codedocsync.suggestions.generators.edge_case_handlers import (
    EdgeCaseSuggestionGenerator,
    SpecialConstructAnalyzer,
    PropertyMethodHandler,
    ClassMethodHandler,
)
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionContext,
    SuggestionType,
)
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.analyzer.models import InconsistencyIssue


class TestSpecialConstructAnalyzer:
    """Test the special construct analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create special construct analyzer."""
        return SpecialConstructAnalyzer()

    def test_analyze_property_getter(self, analyzer):
        """Test analyzing property getter."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["property"]

        constructs = analyzer.analyze_function(function)

        assert len(constructs) >= 1
        property_construct = next(
            c for c in constructs if c.construct_type == "property_getter"
        )
        assert property_construct.requires_special_handling is True
        assert property_construct.confidence == 1.0

    def test_analyze_property_setter(self, analyzer):
        """Test analyzing property setter."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["name.setter"]

        constructs = analyzer.analyze_function(function)

        setter_constructs = [
            c for c in constructs if c.construct_type == "property_setter"
        ]
        assert len(setter_constructs) >= 1

    def test_analyze_property_deleter(self, analyzer):
        """Test analyzing property deleter."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["name.deleter"]

        constructs = analyzer.analyze_function(function)

        deleter_constructs = [
            c for c in constructs if c.construct_type == "property_deleter"
        ]
        assert len(deleter_constructs) >= 1

    def test_analyze_classmethod(self, analyzer):
        """Test analyzing classmethod."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["classmethod"]

        constructs = analyzer.analyze_function(function)

        classmethod_constructs = [
            c for c in constructs if c.construct_type == "classmethod"
        ]
        assert len(classmethod_constructs) >= 1
        assert classmethod_constructs[0].confidence == 1.0

    def test_analyze_staticmethod(self, analyzer):
        """Test analyzing staticmethod."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["staticmethod"]

        constructs = analyzer.analyze_function(function)

        staticmethod_constructs = [
            c for c in constructs if c.construct_type == "staticmethod"
        ]
        assert len(staticmethod_constructs) >= 1

    def test_analyze_overload(self, analyzer):
        """Test analyzing overloaded function."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["overload"]

        constructs = analyzer.analyze_function(function)

        overload_constructs = [c for c in constructs if c.construct_type == "overload"]
        assert len(overload_constructs) >= 1

    def test_analyze_async_function(self, analyzer):
        """Test analyzing async function."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = []
        function.signature.is_async = True

        constructs = analyzer.analyze_function(function)

        async_constructs = [
            c for c in constructs if c.construct_type == "async_function"
        ]
        assert len(async_constructs) >= 1

    def test_analyze_generator_function(self, analyzer):
        """Test analyzing generator function."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = []
        function.source_code = """
def number_generator():
    for i in range(10):
        yield i
"""

        constructs = analyzer.analyze_function(function)

        generator_constructs = [
            c for c in constructs if c.construct_type == "generator"
        ]
        assert len(generator_constructs) >= 1
        assert generator_constructs[0].confidence >= 0.8

    def test_analyze_magic_method(self, analyzer):
        """Test analyzing magic methods."""
        magic_methods = ["__init__", "__str__", "__len__", "__eq__"]

        for method_name in magic_methods:
            function = Mock()
            function.signature = Mock()
            function.signature.decorators = []
            function.signature.name = method_name

            constructs = analyzer.analyze_function(function)

            magic_constructs = [
                c for c in constructs if c.construct_type == "magic_method"
            ]
            assert len(magic_constructs) >= 1

    def test_analyze_context_manager(self, analyzer):
        """Test analyzing context manager methods."""
        context_methods = ["__enter__", "__exit__", "__aenter__", "__aexit__"]

        for method_name in context_methods:
            function = Mock()
            function.signature = Mock()
            function.signature.decorators = []
            function.signature.name = method_name

            constructs = analyzer.analyze_function(function)

            context_constructs = [
                c for c in constructs if c.construct_type == "context_manager"
            ]
            assert len(context_constructs) >= 1

    def test_is_generator_function_syntax_error(self, analyzer):
        """Test generator detection with syntax error."""
        result = analyzer._is_generator_function("def incomplete(")
        assert result is False

    def test_no_special_constructs(self, analyzer):
        """Test analyzing regular function with no special constructs."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = []
        function.signature.name = "regular_function"
        function.signature.is_async = False
        function.source_code = """
def regular_function():
    return "hello"
"""

        constructs = analyzer.analyze_function(function)

        # Should not find any special constructs
        assert len(constructs) == 0


class TestPropertyMethodHandler:
    """Test the property method handler."""

    @pytest.fixture
    def handler(self):
        """Create property method handler."""
        return PropertyMethodHandler()

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "user_name"
        function.signature.return_annotation = "str"
        function.line_number = 10
        return function

    @pytest.fixture
    def mock_docstring(self):
        """Create mock docstring."""
        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Get user name"
        docstring.raw_text = '"""Get user name."""'
        return docstring

    @pytest.fixture
    def mock_context(self, mock_function, mock_docstring):
        """Create mock context."""
        issue = InconsistencyIssue(
            issue_type="property_documentation",
            severity="medium",
            description="Property needs proper documentation",
            suggestion="",
            line_number=10,
        )

        return SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=issue
        )

    def test_handle_property_getter(self, handler, mock_context):
        """Test handling property getter."""
        suggestion = handler.handle_property_getter(mock_context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.FULL_DOCSTRING
        assert suggestion.confidence >= 0.9

        # Should have return documentation but no parameters
        suggested_text = suggestion.suggested_text
        assert "Returns:" in suggested_text or "returns" in suggested_text.lower()
        # Should not have Args/Parameters section for getter
        assert "Args:" not in suggested_text and "Parameters:" not in suggested_text

    def test_handle_property_setter(self, handler, mock_context):
        """Test handling property setter."""
        # Add a parameter for the setter
        param = Mock()
        param.name = "value"
        param.type_annotation = "str"
        param.is_required = True
        mock_context.function.signature.parameters = [param]

        suggestion = handler.handle_property_setter(mock_context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.FULL_DOCSTRING
        assert suggestion.confidence >= 0.9

        # Should have parameter documentation but no return
        suggested_text = suggestion.suggested_text
        assert "value" in suggested_text
        assert "Args:" in suggested_text or "Parameters:" in suggested_text

    def test_handle_property_deleter(self, handler, mock_context):
        """Test handling property deleter."""
        suggestion = handler.handle_property_deleter(mock_context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.FULL_DOCSTRING
        assert suggestion.confidence >= 0.9

        # Should mention deletion
        suggested_text = suggestion.suggested_text
        assert "delete" in suggested_text.lower()

    def test_infer_property_type(self, handler, mock_function):
        """Test property type inference."""
        # Test with return annotation
        mock_function.signature.return_annotation = "int"
        result = handler._infer_property_type(mock_function)
        assert result == "int"

        # Test with name-based inference
        mock_function.signature.return_annotation = None
        mock_function.signature.name = "item_count"
        result = handler._infer_property_type(mock_function)
        assert result == "int"

        # Test with boolean inference
        mock_function.signature.name = "is_valid"
        result = handler._infer_property_type(mock_function)
        assert result == "bool"

    def test_generate_property_description(self, handler, mock_function):
        """Test property description generation."""
        mock_function.signature.name = "user_name"
        result = handler._generate_property_description(mock_function)
        assert "user name" in result.lower()

        # Test with get_ prefix
        mock_function.signature.name = "get_user_data"
        result = handler._generate_property_description(mock_function)
        assert "user data" in result.lower()

    def test_generate_setter_description(self, handler, mock_function):
        """Test setter description generation."""
        mock_function.signature.name = "user_name"
        result = handler._generate_setter_description(mock_function)
        assert "set" in result.lower()
        assert "user name" in result.lower()


class TestClassMethodHandler:
    """Test the class method handler."""

    @pytest.fixture
    def handler(self):
        """Create class method handler."""
        return ClassMethodHandler()

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "create_instance"
        function.line_number = 5

        # Mock parameters including cls
        cls_param = Mock()
        cls_param.name = "cls"
        cls_param.type_annotation = None
        cls_param.is_required = True

        value_param = Mock()
        value_param.name = "value"
        value_param.type_annotation = "str"
        value_param.is_required = True

        function.signature.parameters = [cls_param, value_param]
        return function

    @pytest.fixture
    def mock_docstring(self):
        """Create mock docstring."""
        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Create instance"
        docstring.raw_text = '"""Create instance."""'
        return docstring

    @pytest.fixture
    def mock_context(self, mock_function, mock_docstring):
        """Create mock context."""
        issue = InconsistencyIssue(
            issue_type="classmethod_documentation",
            severity="medium",
            description="Classmethod needs proper documentation",
            suggestion="",
            line_number=5,
        )

        return SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=issue
        )

    def test_handle_classmethod(self, handler, mock_context):
        """Test handling classmethod."""
        suggestion = handler.handle_classmethod(mock_context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert suggestion.confidence >= 0.9

        # Should exclude 'cls' parameter from documentation
        suggested_text = suggestion.suggested_text
        assert "value" in suggested_text
        # Should not document cls parameter
        lines = suggested_text.split("\n")
        cls_mentioned = any("cls" in line and "Args:" not in line for line in lines)
        assert not cls_mentioned

    def test_handle_staticmethod(self, handler, mock_context):
        """Test handling staticmethod."""
        # Remove cls parameter for staticmethod
        value_param = mock_context.function.signature.parameters[1]
        mock_context.function.signature.parameters = [value_param]

        suggestion = handler.handle_staticmethod(mock_context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert suggestion.confidence >= 0.9

        # Should document all parameters for staticmethod
        suggested_text = suggestion.suggested_text
        assert "value" in suggested_text


class TestEdgeCaseSuggestionGenerator:
    """Test the main edge case suggestion generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(default_style="google", max_line_length=88,)

    @pytest.fixture
    def generator(self, config):
        """Create edge case suggestion generator."""
        return EdgeCaseSuggestionGenerator(config)

    @pytest.fixture
    def mock_docstring(self):
        """Create mock docstring."""
        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Test function"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Test function."""'
        return docstring

    @pytest.fixture
    def mock_issue(self):
        """Create mock issue."""
        return InconsistencyIssue(
            issue_type="edge_case_documentation",
            severity="medium",
            description="Edge case needs documentation",
            suggestion="",
            line_number=10,
        )

    def test_property_getter_delegation(self, generator, mock_docstring, mock_issue):
        """Test delegation to property getter handler."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["property"]
        function.signature.name = "name"
        function.signature.return_annotation = "str"
        function.line_number = 10

        context = SuggestionContext(
            function=function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator.generate(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence >= 0.9

    def test_classmethod_delegation(self, generator, mock_docstring, mock_issue):
        """Test delegation to classmethod handler."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["classmethod"]
        function.signature.name = "create"
        function.signature.parameters = []
        function.line_number = 10

        context = SuggestionContext(
            function=function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator.generate(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence >= 0.9

    def test_async_function_handling(self, generator, mock_docstring, mock_issue):
        """Test async function handling."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = []
        function.signature.is_async = True
        function.signature.name = "fetch_data"
        function.line_number = 10

        context = SuggestionContext(
            function=function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._handle_async_function(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE
        assert "async" in suggestion.suggested_text.lower()
        assert "await" in suggestion.suggested_text.lower()

    def test_generator_function_handling(self, generator, mock_docstring, mock_issue):
        """Test generator function handling."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = []
        function.signature.name = "number_generator"
        function.line_number = 10
        function.source_code = """
def number_generator():
    for i in range(10):
        yield i
"""

        context = SuggestionContext(
            function=function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._handle_generator_function(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RETURN_UPDATE
        assert "Generator" in suggestion.suggested_text

    def test_magic_method_handling(self, generator, mock_docstring, mock_issue):
        """Test magic method handling."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = []
        function.signature.name = "__init__"
        function.line_number = 10

        context = SuggestionContext(
            function=function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._handle_magic_method(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE
        assert "Initialize" in suggestion.suggested_text

    def test_overloaded_function_handling(self, generator, mock_docstring, mock_issue):
        """Test overloaded function handling."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["overload"]
        function.signature.name = "process"
        function.line_number = 10

        context = SuggestionContext(
            function=function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._handle_overloaded_function(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE
        assert "overload" in suggestion.suggested_text.lower()

    def test_magic_method_descriptions(self, generator):
        """Test magic method description generation."""
        magic_methods = {
            "__init__": "Initialize",
            "__str__": "string representation",
            "__len__": "length",
            "__eq__": "equality",
            "__call__": "callable",
        }

        for method_name, expected_desc in magic_methods.items():
            function = Mock()
            function.signature = Mock()
            function.signature.name = method_name

            docstring = Mock()
            docstring.format = "google"
            docstring.raw_text = '"""Magic method."""'

            issue = InconsistencyIssue(
                issue_type="magic_method_documentation",
                severity="medium",
                description="Magic method needs documentation",
                suggestion="",
                line_number=1,
            )

            context = SuggestionContext(
                function=function, docstring=docstring, issue=issue
            )

            suggestion = generator._handle_magic_method(context)

            assert expected_desc.lower() in suggestion.suggested_text.lower()

    def test_no_special_constructs_fallback(
        self, generator, mock_docstring, mock_issue
    ):
        """Test fallback when no special constructs are found."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = []
        function.signature.name = "regular_function"
        function.signature.is_async = False
        function.line_number = 10
        function.source_code = """
def regular_function():
    return "hello"
"""

        context = SuggestionContext(
            function=function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator.generate(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence == 0.1  # Generic fallback


class TestEdgeCaseIntegration:
    """Integration tests for edge case handlers."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(default_style="google")

    @pytest.fixture
    def generator(self, config):
        """Create edge case suggestion generator."""
        return EdgeCaseSuggestionGenerator(config)

    def test_complete_workflow_property_getter(self, generator):
        """Test complete workflow for property getter."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["property"]
        function.signature.name = "full_name"
        function.signature.return_annotation = "str"
        function.line_number = 8

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Get full name"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Get full name."""'

        issue = InconsistencyIssue(
            issue_type="property_documentation",
            severity="medium",
            description="Property getter needs proper documentation",
            suggestion="Format as property getter",
            line_number=8,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify property-specific formatting
        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.FULL_DOCSTRING
        assert suggestion.confidence >= 0.9

        suggested_text = suggestion.suggested_text

        # Should have return documentation
        assert "Returns:" in suggested_text or "returns" in suggested_text.lower()
        assert "str" in suggested_text

        # Should NOT have parameter documentation for getter
        assert "Args:" not in suggested_text
        assert "Parameters:" not in suggested_text

    def test_complete_workflow_classmethod_creation(self, generator):
        """Test complete workflow for classmethod creation."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["classmethod"]
        function.signature.name = "from_dict"
        function.line_number = 15

        # Parameters including cls
        cls_param = Mock()
        cls_param.name = "cls"
        cls_param.type_annotation = None
        cls_param.is_required = True

        data_param = Mock()
        data_param.name = "data"
        data_param.type_annotation = "Dict[str, Any]"
        data_param.is_required = True

        function.signature.parameters = [cls_param, data_param]

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Create instance from dictionary"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Create instance from dictionary."""'

        issue = InconsistencyIssue(
            issue_type="classmethod_documentation",
            severity="medium",
            description="Classmethod parameters need documentation",
            suggestion="Document parameters excluding cls",
            line_number=15,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify classmethod-specific handling
        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence >= 0.9

        suggested_text = suggestion.suggested_text

        # Should document the data parameter
        assert "data" in suggested_text
        assert "Dict[str, Any]" in suggested_text or "dict" in suggested_text.lower()

        # Should NOT document cls parameter
        # Check that cls is not mentioned in parameter documentation
        lines = suggested_text.split("\n")
        parameter_section_started = False
        cls_documented = False

        for line in lines:
            if "Args:" in line or "Parameters:" in line:
                parameter_section_started = True
            elif (
                parameter_section_started and line.strip() and not line.startswith(" ")
            ):
                # End of parameter section
                break
            elif parameter_section_started and "cls:" in line:
                cls_documented = True

        assert not cls_documented

    def test_multiple_special_constructs(self, generator):
        """Test handling function with multiple special constructs."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["property"]  # Property has precedence
        function.signature.name = "__len__"  # Also a magic method
        function.signature.return_annotation = "int"
        function.line_number = 3

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Get length"
        docstring.raw_text = '"""Get length."""'

        issue = InconsistencyIssue(
            issue_type="special_construct_documentation",
            severity="medium",
            description="Special construct needs documentation",
            suggestion="",
            line_number=3,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Should handle as property (first construct detected)
        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.FULL_DOCSTRING
