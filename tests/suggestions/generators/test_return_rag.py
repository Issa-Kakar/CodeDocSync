"""Tests for RAG enhancement in ReturnGenerator."""

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
)
from codedocsync.parser.docstring_models import DocstringFormat, ParsedDocstring
from codedocsync.suggestions.generators.return_generator import (
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext


class TestReturnGeneratorRAG:
    """Test RAG enhancement in ReturnGenerator."""

    @pytest.fixture
    def mock_function_with_complex_return(self):
        """Create a function with complex return type."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="process_data",
                parameters=[
                    FunctionParameter(
                        name="data",
                        type_annotation="list[dict[str, Any]]",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_type="dict[str, list[float]]",
            ),
            docstring=ParsedDocstring(
                format=DocstringFormat.GOOGLE,
                summary="Process data and return statistics",
                raw_text='"""Process data and return statistics"""',
            ),
            file_path="test.py",
            line_number=20,
            end_line_number=25,
        )

    @pytest.fixture
    def rag_return_examples(self):
        """Create RAG examples with return documentation."""
        return [
            {
                "signature": "def analyze_metrics(metrics: list[dict]) -> dict[str, list[float]]:",
                "docstring": '''"""Analyze metrics and compute statistics.

                Args:
                    metrics: List of metric dictionaries

                Returns:
                    A dictionary mapping metric names to their computed values.
                    Each key is a metric name (str) and each value is a list of
                    statistical measures: [mean, median, std_dev, min, max].

                    Example return value:
                    {
                        'cpu_usage': [45.2, 44.0, 12.3, 10.0, 89.0],
                        'memory': [1024.5, 1000.0, 256.7, 512.0, 2048.0]
                    }
                """''',
                "similarity": 0.8,
                "source_file": "analytics/metrics.py",
            },
            {
                "signature": "def aggregate_results(results: list) -> dict[str, Any]:",
                "docstring": '''"""Aggregate analysis results.

                Returns:
                    Dictionary containing aggregated statistics with the following structure:
                    - 'summary': Overall summary statistics
                    - 'by_category': Results grouped by category
                    - 'outliers': List of outlier values detected
                """''',
                "similarity": 0.6,
                "source_file": "analytics/aggregator.py",
            },
        ]

    def test_return_generator_rag_enhancement(
        self, mock_function_with_complex_return, rag_return_examples
    ):
        """Test ReturnGenerator uses RAG for complex return types."""
        generator = ReturnSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="Return value not documented",
                suggestion="Add return documentation",
                line_number=20,
            ),
            function=mock_function_with_complex_return,
            docstring=mock_function_with_complex_return.docstring,
            related_functions=rag_return_examples,
        )

        suggestion = generator.generate(context)

        # Verify RAG was used
        assert suggestion.metadata.used_rag_examples is True

        # Should describe the dictionary structure
        assert (
            "dictionary" in suggestion.suggested_text.lower()
            or "dict" in suggestion.suggested_text.lower()
        )

        # Should explain the nested structure
        assert "str" in suggestion.suggested_text
        assert "list" in suggestion.suggested_text

    def test_return_generator_semantic_adaptation(
        self, mock_function_with_complex_return
    ):
        """Test ReturnGenerator adapts descriptions semantically."""
        generator = ReturnSuggestionGenerator()

        # Example with different context but similar structure
        semantic_examples = [
            {
                "signature": "def get_user_scores(user_id: str) -> dict[str, list[int]]:",
                "docstring": '''"""Get all scores for a user.

                Returns:
                    A mapping of test names to score lists. Each test name maps to
                    a chronologically ordered list of scores achieved by the user.
                    Scores range from 0 to 100.
                """''',
                "similarity": 0.7,
                "source_file": "scoring.py",
            }
        ]

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_returns",
                severity="medium",
                description="Return description is vague",
                suggestion="Clarify return value structure",
                line_number=20,
            ),
            function=mock_function_with_complex_return,
            docstring=mock_function_with_complex_return.docstring,
            related_functions=semantic_examples,
        )

        suggestion = generator.generate(context)

        # Should adapt the pattern but not copy content
        assert "test names" not in suggestion.suggested_text
        assert "user" not in suggestion.suggested_text.lower()

        # Should maintain structural description pattern
        assert (
            "mapping" in suggestion.suggested_text.lower()
            or "dictionary" in suggestion.suggested_text.lower()
        )

    def test_return_generator_optional_return_handling(self):
        """Test ReturnGenerator handles Optional return types with RAG."""
        generator = ReturnSuggestionGenerator()

        # Function that might return None
        optional_function = ParsedFunction(
            signature=FunctionSignature(
                name="find_item",
                parameters=[
                    FunctionParameter(
                        name="item_id",
                        type_annotation="str",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_type="Item | None",
            ),
            docstring=ParsedDocstring(
                format=DocstringFormat.GOOGLE,
                summary="Find an item by ID",
                raw_text='"""Find an item by ID"""',
            ),
            file_path="test.py",
            line_number=30,
            end_line_number=35,
        )

        # RAG examples showing None handling
        none_examples = [
            {
                "signature": "def search_database(query: str) -> Record | None:",
                "docstring": '''"""Search for a record in the database.

                Returns:
                    The matching Record object if found, or None if no record
                    matches the search query. None is also returned if the
                    database connection fails.
                """''',
                "similarity": 0.75,
                "source_file": "db/search.py",
            }
        ]

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_returns",
                severity="medium",
                description="Optional return not explained",
                suggestion="Clarify when None is returned",
                line_number=30,
            ),
            function=optional_function,
            docstring=optional_function.docstring,
            related_functions=none_examples,
        )

        suggestion = generator.generate(context)

        # Should explain None case
        assert "None" in suggestion.suggested_text
        assert suggestion.metadata.used_rag_examples is True

    def test_return_generator_no_rag_simple_type(self):
        """Test ReturnGenerator doesn't use RAG for simple types."""
        generator = ReturnSuggestionGenerator()

        # Simple function returning int
        simple_function = ParsedFunction(
            signature=FunctionSignature(
                name="count_items", parameters=[], return_type="int"
            ),
            docstring=ParsedDocstring(
                format=DocstringFormat.GOOGLE,
                summary="Count items",
                raw_text='"""Count items"""',
            ),
            file_path="test.py",
            line_number=40,
            end_line_number=42,
        )

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="No return documentation",
                suggestion="Add return documentation",
                line_number=40,
            ),
            function=simple_function,
            docstring=simple_function.docstring,
            related_functions=[],  # Empty RAG examples
        )

        suggestion = generator.generate(context)

        # Should not use RAG for simple int return
        assert suggestion.metadata.used_rag_examples is False

        # Should still generate appropriate description
        assert (
            "int" in suggestion.suggested_text or "number" in suggestion.suggested_text
        )
