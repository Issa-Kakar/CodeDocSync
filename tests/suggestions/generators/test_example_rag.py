"""Tests for RAG enhancement in ExampleGenerator."""

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
)
from codedocsync.parser.docstring_models import DocstringFormat, ParsedDocstring
from codedocsync.suggestions.generators.example_generator import (
    ExampleSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext


class TestExampleGeneratorRAG:
    """Test RAG enhancement in ExampleGenerator."""

    @pytest.fixture
    def mock_function(self):
        """Create a mock function for testing."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="calculate_discount",
                parameters=[
                    FunctionParameter(
                        name="price",
                        type_annotation="float",
                        default_value=None,
                        is_required=True,
                    ),
                    FunctionParameter(
                        name="discount_rate",
                        type_annotation="float",
                        default_value="0.1",
                        is_required=False,
                    ),
                ],
                return_type="float",
            ),
            docstring=None,
            file_path="test.py",
            line_number=10,
            end_line_number=15,
        )

    @pytest.fixture
    def rag_examples(self):
        """Create RAG examples with proper similarity scores."""
        return [
            {
                "signature": "def apply_discount(amount: float, percentage: float = 0.2) -> float:",
                "docstring": """Apply a percentage discount to an amount.

                Args:
                    amount: The original amount
                    percentage: The discount percentage (default: 0.2)

                Returns:
                    The discounted amount

                Examples:
                    >>> apply_discount(100.0)
                    80.0
                    >>> apply_discount(100.0, 0.5)
                    50.0
                    >>> apply_discount(0, 0.5)
                    0.0
                """,
                "similarity": 0.85,
                "source_file": "utils/pricing.py",
            },
            {
                "signature": "def calculate_price(base: float, tax_rate: float = 0.08) -> float:",
                "docstring": """Calculate final price with tax.

                Args:
                    base: Base price
                    tax_rate: Tax rate to apply

                Returns:
                    Final price including tax

                Example:
                    >>> calculate_price(100)
                    108.0
                """,
                "similarity": 0.45,
                "source_file": "utils/pricing.py",
            },
        ]

    def test_example_generator_rag_enhancement(self, mock_function, rag_examples):
        """Test ExampleGenerator uses RAG examples."""
        generator = ExampleSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_examples",
                severity="medium",
                description="Function lacks usage examples",
                suggestion="Add usage examples to demonstrate function behavior",
                line_number=10,
            ),
            function=mock_function,
            docstring=None,
            related_functions=rag_examples,
        )

        suggestion = generator.generate(context)

        # Verify RAG was used
        assert suggestion.metadata.used_rag_examples is True

        # Verify example adaptation
        assert "calculate_discount" in suggestion.suggested_text
        assert ">>>" in suggestion.suggested_text

        # Should have adapted the examples from related functions
        assert (
            "100" in suggestion.suggested_text or "100.0" in suggestion.suggested_text
        )

    def test_example_generator_no_rag_fallback(self, mock_function):
        """Test ExampleGenerator falls back when no RAG examples available."""
        generator = ExampleSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_examples",
                severity="medium",
                description="Function lacks usage examples",
                suggestion="Add usage examples",
                line_number=10,
            ),
            function=mock_function,
            docstring=None,
            related_functions=None,  # No RAG examples
        )

        suggestion = generator.generate(context)

        # Verify RAG was not used
        assert suggestion.metadata.used_rag_examples is False

        # Should still generate a suggestion
        assert suggestion.suggested_text
        assert "calculate_discount" in suggestion.suggested_text

    def test_example_generator_low_similarity_filtering(self, mock_function):
        """Test ExampleGenerator filters out low similarity examples."""
        generator = ExampleSuggestionGenerator()

        # Create examples with low similarity
        low_similarity_examples = [
            {
                "signature": "def unrelated_function(data: dict) -> str:",
                "docstring": """Process data.

                Example:
                    >>> unrelated_function({'key': 'value'})
                    'processed'
                """,
                "similarity": 0.2,  # Below threshold
                "source_file": "other.py",
            }
        ]

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_examples",
                severity="medium",
                description="Function lacks examples",
                suggestion="Add examples",
                line_number=10,
            ),
            function=mock_function,
            docstring=None,
            related_functions=low_similarity_examples,
        )

        suggestion = generator.generate(context)

        # Should not use low similarity examples
        assert suggestion.metadata.used_rag_examples is False
        assert "unrelated_function" not in suggestion.suggested_text

    def test_example_generator_complex_adaptation(self, mock_function):
        """Test ExampleGenerator handles complex example adaptation."""
        generator = ExampleSuggestionGenerator()

        # Complex RAG example with setup code
        complex_examples = [
            {
                "signature": "def compute_total(items: list[dict], tax: float = 0.08) -> float:",
                "docstring": """Compute total with tax.

                Examples:
                    Basic usage:
                    >>> items = [{'price': 10}, {'price': 20}]
                    >>> compute_total(items)
                    32.4

                    Custom tax rate:
                    >>> items = [{'price': 100}]
                    >>> compute_total(items, tax=0.1)
                    110.0

                    Empty list:
                    >>> compute_total([])
                    0.0
                """,
                "similarity": 0.7,
                "source_file": "calc.py",
            }
        ]

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="example_incomplete",
                severity="low",
                description="Examples are incomplete",
                suggestion="Add more comprehensive examples",
                line_number=10,
            ),
            function=mock_function,
            docstring=ParsedDocstring(
                format=DocstringFormat.GOOGLE,
                summary="Calculate discount",
                raw_text='"""Calculate discount"""',
            ),
            related_functions=complex_examples,
        )

        suggestion = generator.generate(context)

        # Should use RAG examples
        assert suggestion.metadata.used_rag_examples is True

        # Should adapt function name
        assert "calculate_discount" in suggestion.suggested_text

        # Should include multiple examples
        assert suggestion.suggested_text.count(">>>") >= 2
