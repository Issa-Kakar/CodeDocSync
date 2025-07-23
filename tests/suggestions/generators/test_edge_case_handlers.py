"""
Tests for the Edge Case Handlers.

Tests cover special Python constructs like properties, class methods, magic methods,
async functions, and other edge cases requiring specialized documentation.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.generators.edge_case_handlers import (
    ClassMethodHandler,
    EdgeCaseSuggestionGenerator,
    PropertyMethodHandler,
    SpecialConstructAnalyzer,
)
from codedocsync.suggestions.models import (
    SuggestionContext,
    SuggestionType,
)


class TestSpecialConstructAnalyzer:
    """Test the special construct analyzer."""

    @pytest.fixture
    def analyzer(self) -> Any:
        """Create special construct analyzer."""
        return SpecialConstructAnalyzer()

    def test_analyze_property_getter(self, analyzer: Any) -> None:
        """Test analyzing property getter."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.decorators = ["property"]

        constructs = analyzer.analyze_function(function)

        assert len(constructs) >= 1
        property_construct = next(
            c for c in constructs if c.construct_type == "property_getter"
        )
        assert property_construct.requires_special_handling is True

    def test_analyze_classmethod(self, analyzer: Any) -> None:
        """Test analyzing class method."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.decorators = ["classmethod"]

        constructs = analyzer.analyze_function(function)

        assert len(constructs) >= 1
        classmethod_construct = next(
            c for c in constructs if c.construct_type == "classmethod"
        )
        assert classmethod_construct.requires_special_handling is True

    def test_analyze_magic_method(self, analyzer: Any) -> None:
        """Test analyzing magic method."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "__init__"
        function.signature.decorators = []

        constructs = analyzer.analyze_function(function)

        assert len(constructs) >= 1
        magic_construct = next(
            c for c in constructs if c.construct_type == "magic_method"
        )
        assert magic_construct.requires_special_handling is True

    def test_analyze_async_function(self, analyzer: Any) -> None:
        """Test analyzing async function."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.is_async = True
        function.signature.decorators = []

        constructs = analyzer.analyze_function(function)

        assert len(constructs) >= 1
        async_construct = next(
            c for c in constructs if c.construct_type == "async_function"
        )
        assert async_construct.requires_special_handling is True


class TestPropertyMethodHandler:
    """Test the property method handler."""

    @pytest.fixture
    def handler(self) -> Any:
        """Create property method handler."""
        return PropertyMethodHandler()

    @pytest.fixture
    def property_context(self) -> Any:
        """Create a context for property methods."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "username"
        function.signature.decorators = ["property"]
        function.signature.return_annotation = "str"
        function.signature.parameters = []

        issue = InconsistencyIssue(
            issue_type="missing_docstring",
            severity="high",
            description="Property getter has no docstring",
            suggestion="Add a docstring",
            line_number=10,
        )

        return SuggestionContext(
            issue=issue,
            function=function,
            project_style="google",
        )

    def test_generate_property_getter_docstring(
        self, handler: Any, property_context: Any
    ) -> None:
        """Test generating docstring for property getter."""
        suggestion = handler.generate_suggestion(property_context)

        assert suggestion is not None
        assert suggestion.suggestion_type == SuggestionType.FULL_DOCSTRING
        assert "Get the username" in suggestion.suggested_text
        assert "Returns:" in suggestion.suggested_text
        assert "str:" in suggestion.suggested_text

    def test_property_setter_detection(self, handler: Any) -> None:
        """Test detecting property setter."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "username"
        function.signature.decorators = ["username.setter"]

        assert handler._is_property_setter(function) is True

    def test_property_deleter_detection(self, handler: Any) -> None:
        """Test detecting property deleter."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "username"
        function.signature.decorators = ["username.deleter"]

        assert handler._is_property_deleter(function) is True


class TestClassMethodHandler:
    """Test the class method handler."""

    @pytest.fixture
    def handler(self) -> Any:
        """Create class method handler."""
        return ClassMethodHandler()

    @pytest.fixture
    def classmethod_context(self) -> Any:
        """Create a context for class methods."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "from_config"
        function.signature.decorators = ["classmethod"]
        function.signature.return_annotation = "MyClass"

        # Mock parameters
        cls_param: Mock = Mock()
        cls_param.name = "cls"
        cls_param.type_annotation = None

        config_param: Mock = Mock()
        config_param.name = "config"
        config_param.type_annotation = "dict"

        function.signature.parameters = [cls_param, config_param]

        issue = InconsistencyIssue(
            issue_type="missing_docstring",
            severity="high",
            description="Class method has no docstring",
            suggestion="Add a docstring",
            line_number=20,
        )

        return SuggestionContext(
            issue=issue,
            function=function,
            project_style="google",
        )

    def test_generate_classmethod_docstring(
        self, handler: Any, classmethod_context: Any
    ) -> None:
        """Test generating docstring for class method."""
        suggestion = handler.generate_suggestion(classmethod_context)

        assert suggestion is not None
        assert suggestion.suggestion_type == SuggestionType.FULL_DOCSTRING
        assert "Create instance from configuration" in suggestion.suggested_text
        assert "Args:" in suggestion.suggested_text
        assert "config (dict):" in suggestion.suggested_text
        assert "Returns:" in suggestion.suggested_text
        assert "MyClass:" in suggestion.suggested_text

    def test_staticmethod_handling(self, handler: Any) -> None:
        """Test that static methods are also handled."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.decorators = ["staticmethod"]

        constructs = handler._analyze_constructs(function)
        assert "staticmethod" in constructs


class TestEdgeCaseSuggestionGenerator:
    """Test the main edge case suggestion generator."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create edge case suggestion generator."""
        config = SuggestionConfig()
        return EdgeCaseSuggestionGenerator(config)

    @pytest.fixture
    def mock_handlers(self) -> dict[str, Any]:
        """Create mock handlers."""
        return {
            "property": Mock(),
            "classmethod": Mock(),
        }

    def test_generate_suggestion_delegates_to_handlers(
        self, generator: Any, property_context: Any, mock_handlers: dict[str, Any]
    ) -> None:
        """Test that generator delegates to appropriate handlers."""
        # Mock the handlers
        generator._handlers = mock_handlers
        mock_handlers["property"].can_handle.return_value = True
        mock_handlers["classmethod"].can_handle.return_value = False

        mock_suggestion: Mock = Mock()
        mock_handlers["property"].generate_suggestion.return_value = mock_suggestion

        result = generator.generate_suggestion(property_context)

        assert result == mock_suggestion
        mock_handlers["property"].generate_suggestion.assert_called_once()
        mock_handlers["classmethod"].generate_suggestion.assert_not_called()

    def test_generate_suggestion_no_handler(
        self, generator: Any, property_context: Any
    ) -> None:
        """Test when no handler can handle the context."""
        # Make all handlers unable to handle
        for handler in generator._handlers.values():
            handler.can_handle = Mock(return_value=False)

        result = generator.generate_suggestion(property_context)

        assert result is None

    def test_async_function_suggestion(self, generator: Any) -> None:
        """Test generating suggestion for async function."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "fetch_data"
        function.signature.is_async = True
        function.signature.return_annotation = "dict"
        function.signature.parameters = []

        issue = InconsistencyIssue(
            issue_type="missing_docstring",
            severity="high",
            description="Async function has no docstring",
            suggestion="Add a docstring",
            line_number=30,
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            project_style="google",
        )

        suggestion = generator.generate_suggestion(context)

        assert suggestion is not None
        assert (
            "async" in suggestion.suggested_text.lower()
            or "asynchronous" in suggestion.suggested_text.lower()
        )

    def test_magic_method_suggestion(self, generator: Any) -> None:
        """Test generating suggestion for magic method."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "__str__"
        function.signature.return_annotation = "str"
        function.signature.parameters = []

        issue = InconsistencyIssue(
            issue_type="missing_docstring",
            severity="medium",
            description="Magic method has no docstring",
            suggestion="Add a docstring",
            line_number=40,
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            project_style="google",
        )

        suggestion = generator.generate_suggestion(context)

        assert suggestion is not None
        assert "string representation" in suggestion.suggested_text.lower()
