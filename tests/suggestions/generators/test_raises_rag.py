"""Test RAG enhancement for RaisesGenerator."""

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.suggestions.generators.raises_generator import (
    RaisesSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext


class TestRaisesGeneratorRAG:
    """Test RAG enhancement functionality in RaisesGenerator."""

    @pytest.fixture
    def mock_function(self):
        """Create a mock function for testing."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="validate_input",
                parameters=[
                    FunctionParameter(
                        name="data",
                        type_annotation="dict[str, Any]",
                        default_value=None,
                        is_required=True,
                    ),
                ],
                return_type="bool",
            ),
            docstring=RawDocstring(raw_text='"""Validate input."""', line_number=11),
            file_path="test.py",
            line_number=10,
            end_line_number=20,  # Required field
        )
        # Add detected exceptions
        function.detected_exceptions = ["ValueError", "TypeError"]
        # Add source code so exception analysis can work
        function.source_code = '''def validate_input(data: dict[str, Any]) -> bool:
    """Validate input."""
    if not data:
        raise ValueError("Data cannot be empty")
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")
    return True'''
        return function

    @pytest.fixture
    def rag_examples(self):
        """Create RAG examples with exception documentation."""
        return [
            {
                "signature": "def validate_data(input_data: dict) -> bool:",
                "docstring": '''"""Validate input data format.

                Args:
                    input_data: Dictionary to validate.

                Returns:
                    True if valid, False otherwise.

                Raises:
                    ValueError: If data format is invalid or missing required fields.
                    TypeError: If input_data is not a dictionary.
                """''',
                "similarity": 0.85,
            },
            {
                "signature": "def check_values(values: dict) -> None:",
                "docstring": '''"""Check values in dictionary.

                Raises:
                    ValueError: If any value is empty or None.
                    KeyError: If required keys are missing.
                """''',
                "similarity": 0.75,
            },
        ]

    def test_rag_enhancement_used(self, mock_function, rag_examples):
        """Test that RAG enhancement is used when examples are available."""
        generator = RaisesSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Missing exception documentation",
                suggestion="Document the exceptions",
                line_number=10,
            ),
            function=mock_function,
            docstring=mock_function.docstring,  # Use docstring from function
            related_functions=rag_examples,
        )

        suggestion = generator.generate(context)

        # Verify RAG was used
        assert suggestion.metadata.used_rag_examples is True

        # Verify exceptions are documented
        assert "ValueError" in suggestion.suggested_text
        assert "TypeError" in suggestion.suggested_text

    def test_semantic_exception_matching(self, mock_function, rag_examples):
        """Test semantic matching between exception types."""
        generator = RaisesSuggestionGenerator()

        # Test the semantic matching method directly
        assert generator._exception_types_match("ValueError", "ValidationError") is True
        assert generator._exception_types_match("KeyError", "LookupError") is True
        assert (
            generator._exception_types_match("IOError", "FileError") is True
        )  # Both have "io" and "file"
        assert generator._exception_types_match("ValueError", "TypeError") is False

    def test_exception_description_synthesis(self, mock_function, rag_examples):
        """Test that exception descriptions are synthesized from patterns."""
        generator = RaisesSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Missing exception documentation",
                suggestion="Document the exceptions",
                line_number=10,
            ),
            function=mock_function,
            docstring=mock_function.docstring,  # Use docstring from function
            related_functions=rag_examples,
        )

        suggestion = generator.generate(context)

        # Should learn from examples
        assert any(
            phrase in suggestion.suggested_text.lower()
            for phrase in ["invalid", "format", "not a dictionary", "empty"]
        )

    def test_rag_without_similar_exceptions(self, mock_function):
        """Test RAG behavior when examples have different exceptions."""
        generator = RaisesSuggestionGenerator()

        # Examples with different exceptions
        different_examples = [
            {
                "signature": "def network_call() -> str:",
                "docstring": '''"""Make network call.

                Raises:
                    ConnectionError: If network is unavailable.
                    TimeoutError: If request times out.
                """''',
                "similarity": 0.6,
            }
        ]

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Missing exception documentation",
                suggestion="Document the exceptions",
                line_number=10,
            ),
            function=mock_function,
            docstring=mock_function.docstring,  # Use docstring from function
            related_functions=different_examples,
        )

        suggestion = generator.generate(context)

        # Should still generate ValueError and TypeError
        assert "ValueError" in suggestion.suggested_text
        assert "TypeError" in suggestion.suggested_text

        # Should not include unrelated exceptions
        assert "ConnectionError" not in suggestion.suggested_text
        assert "TimeoutError" not in suggestion.suggested_text

    def test_fallback_without_rag(self, mock_function):
        """Test that generator works without RAG examples."""
        generator = RaisesSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Missing exception documentation",
                suggestion="Document the exceptions",
                line_number=10,
            ),
            function=mock_function,
            docstring=mock_function.docstring,  # Use docstring from function
            related_functions=[],  # No RAG examples
        )

        suggestion = generator.generate(context)

        # Should not use RAG
        assert suggestion.metadata.used_rag_examples is False

        # Should still document exceptions
        assert "ValueError" in suggestion.suggested_text
        assert "TypeError" in suggestion.suggested_text

    def test_edge_case_descriptions(self, mock_function, rag_examples):
        """Test that edge case patterns are properly adapted."""
        generator = RaisesSuggestionGenerator()

        # Add edge case example
        edge_examples = rag_examples + [
            {
                "signature": "def process_empty(items: list) -> None:",
                "docstring": '''"""Process items.

                Raises:
                    ValueError: If items list is empty or contains None values.
                """''',
                "similarity": 0.8,
            }
        ]

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Missing exception documentation",
                suggestion="Document the exceptions",
                line_number=10,
            ),
            function=mock_function,
            docstring=mock_function.docstring,  # Use docstring from function
            related_functions=edge_examples,
        )

        suggestion = generator.generate(context)

        # Should adapt edge case language
        assert any(
            word in suggestion.suggested_text.lower()
            for word in ["empty", "none", "missing"]
        )

    def test_grammar_fixes(self, mock_function, rag_examples):
        """Test that exception descriptions have proper grammar."""
        generator = RaisesSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Missing exception documentation",
                suggestion="Document the exceptions",
                line_number=10,
            ),
            function=mock_function,
            docstring=mock_function.docstring,  # Use docstring from function
            related_functions=rag_examples,
        )

        suggestion = generator.generate(context)

        # Check each raises line
        lines = suggestion.suggested_text.split("\n")

        # Parse the raises section properly handling multi-line descriptions
        i = 0
        while i < len(lines):
            line = lines[i]
            if ":" in line and any(exc in line for exc in ["ValueError", "TypeError"]):
                # Found a raises line, collect the full description
                desc_part = line.split(":", 1)[1].strip()

                # Check if description continues on next lines (indented)
                j = i + 1
                while j < len(lines) and lines[j].startswith("        "):
                    desc_part += " " + lines[j].strip()
                    j += 1

                # Now check the complete description
                assert desc_part.endswith(
                    "."
                ), f"Description doesn't end with period: '{desc_part}'"
                # Should be capitalized
                assert desc_part[
                    0
                ].isupper(), f"Description not capitalized: '{desc_part}'"
                # No double spaces
                assert "  " not in desc_part, f"Double spaces in: '{desc_part}'"

                i = j  # Skip to next unprocessed line
            else:
                i += 1
