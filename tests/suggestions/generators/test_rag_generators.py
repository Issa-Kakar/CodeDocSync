"""Comprehensive tests for all RAG-enhanced generators."""

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
)
from codedocsync.parser.docstring_models import DocstringFormat, ParsedDocstring
from codedocsync.suggestions.generators import (
    BehaviorSuggestionGenerator,
    ExampleSuggestionGenerator,
    ParameterSuggestionGenerator,
    RaisesSuggestionGenerator,
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext


class TestAllGeneratorsRAG:
    """Test RAG enhancement across all generators."""

    @pytest.fixture
    def comprehensive_function(self):
        """Create a function that needs all types of documentation."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="process_transaction",
                parameters=[
                    FunctionParameter(
                        name="transaction_data",
                        type_annotation="dict[str, Any]",
                        default_value=None,
                        is_required=True,
                    ),
                    FunctionParameter(
                        name="validate",
                        type_annotation="bool",
                        default_value="True",
                        is_required=False,
                    ),
                    FunctionParameter(
                        name="timeout",
                        type_annotation="float | None",
                        default_value="None",
                        is_required=False,
                    ),
                ],
                return_type="TransactionResult",
            ),
            docstring=ParsedDocstring(
                format=DocstringFormat.GOOGLE,
                summary="Process a transaction",
                raw_text='"""Process a transaction"""',
            ),
            file_path="transaction.py",
            line_number=50,
            end_line_number=70,
            source_code='''def process_transaction(transaction_data, validate=True, timeout=None):
    """Process a transaction"""
    if not transaction_data:
        raise ValueError("Transaction data is required")

    if timeout and timeout < 0:
        raise ValueError("Timeout must be positive")

    # Process transaction logic here
    return TransactionResult(success=True)
''',
        )

    @pytest.fixture
    def comprehensive_rag_examples(self):
        """Create comprehensive RAG examples covering all aspects."""
        return [
            {
                "signature": "def execute_payment(payment_info: dict, verify: bool = True) -> PaymentResult:",
                "docstring": '''"""Execute a payment transaction.

                Args:
                    payment_info: Dictionary containing payment details including:
                        - amount (float): Payment amount
                        - currency (str): Three-letter currency code
                        - recipient (str): Recipient identifier
                    verify: Whether to verify the payment before execution (default: True)

                Returns:
                    PaymentResult object containing:
                        - success (bool): Whether payment was successful
                        - transaction_id (str): Unique transaction identifier
                        - timestamp (datetime): When the transaction was processed

                Raises:
                    PaymentError: If payment validation fails
                    NetworkError: If connection to payment processor fails
                    TimeoutError: If payment processing exceeds timeout

                Examples:
                    >>> payment = {'amount': 100.0, 'currency': 'USD', 'recipient': 'user123'}
                    >>> result = execute_payment(payment)
                    >>> print(result.success)
                    True

                    >>> # Skip verification for faster processing
                    >>> result = execute_payment(payment, verify=False)

                Note:
                    All payments are logged for audit purposes. Large payments
                    (over $10,000) require additional verification steps.
                """''',
                "similarity": 0.85,
                "source_file": "payments/processor.py",
            },
            {
                "signature": "def validate_transaction(data: dict, strict: bool = False) -> bool:",
                "docstring": '''"""Validate transaction data.

                Validates transaction data against business rules. In strict mode,
                all optional fields must be present and valid.

                Args:
                    data: Transaction data to validate
                    strict: Enable strict validation mode

                Returns:
                    True if validation passes, False otherwise

                Raises:
                    ValueError: If required fields are missing
                    TypeError: If field types are incorrect
                """''',
                "similarity": 0.7,
                "source_file": "validation/rules.py",
            },
        ]

    def test_parameter_generator_rag_comprehensive(
        self, comprehensive_function, comprehensive_rag_examples
    ):
        """Test ParameterGenerator with comprehensive RAG examples."""
        generator = ParameterSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_params",
                severity="high",
                description="Parameter 'transaction_data' not documented",
                suggestion="Add parameter documentation",
                line_number=50,
                details={"parameter_name": "transaction_data"},
            ),
            function=comprehensive_function,
            docstring=comprehensive_function.docstring,
            related_functions=comprehensive_rag_examples,
        )

        suggestion = generator.generate(context)

        # Should use RAG
        assert suggestion.metadata.used_rag_examples is True

        # Should describe dictionary structure
        assert (
            "dict" in suggestion.suggested_text
            or "Dictionary" in suggestion.suggested_text
        )

        # Should have parameter documentation
        assert "transaction_data" in suggestion.suggested_text

    def test_return_generator_rag_comprehensive(
        self, comprehensive_function, comprehensive_rag_examples
    ):
        """Test ReturnGenerator with comprehensive RAG examples."""
        generator = ReturnSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="Return value not documented",
                suggestion="Add return documentation",
                line_number=50,
            ),
            function=comprehensive_function,
            docstring=comprehensive_function.docstring,
            related_functions=comprehensive_rag_examples,
        )

        suggestion = generator.generate(context)

        # Should adapt PaymentResult pattern to TransactionResult
        assert "TransactionResult" in suggestion.suggested_text

        # Should describe object structure
        assert (
            "object" in suggestion.suggested_text
            or "containing" in suggestion.suggested_text
        )

    def test_raises_generator_rag_comprehensive(
        self, comprehensive_function, comprehensive_rag_examples
    ):
        """Test RaisesGenerator with comprehensive RAG examples."""
        generator = RaisesSuggestionGenerator()

        # Add raises to the docstring
        comprehensive_function.docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Process a transaction",
            raw_text='''"""Process a transaction

            Raises:
                ValueError: When input is invalid
            """''',
        )

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Raises section incomplete",
                suggestion="Add more exception documentation",
                line_number=50,
            ),
            function=comprehensive_function,
            docstring=comprehensive_function.docstring,
            related_functions=comprehensive_rag_examples,
        )

        suggestion = generator.generate(context)

        # Should suggest additional exceptions based on patterns
        assert suggestion.metadata.used_rag_examples is True

        # Could suggest TimeoutError based on timeout parameter
        assert "Error" in suggestion.suggested_text

    def test_behavior_generator_rag_comprehensive(
        self, comprehensive_function, comprehensive_rag_examples
    ):
        """Test BehaviorGenerator with comprehensive RAG examples."""
        generator = BehaviorSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="description_outdated",
                severity="medium",
                description="Function behavior not clear",
                suggestion="Clarify function behavior",
                line_number=50,
            ),
            function=comprehensive_function,
            docstring=comprehensive_function.docstring,
            related_functions=comprehensive_rag_examples,
        )

        suggestion = generator.generate(context)

        # Should learn vocabulary from examples
        assert suggestion.metadata.used_rag_examples is True

        # Should adapt validation/verification concepts
        docstring_lower = suggestion.suggested_text.lower()
        assert "validat" in docstring_lower or "verif" in docstring_lower

    def test_example_generator_rag_comprehensive(
        self, comprehensive_function, comprehensive_rag_examples
    ):
        """Test ExampleGenerator with comprehensive RAG examples."""
        generator = ExampleSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="missing_examples",
                severity="medium",
                description="No usage examples",
                suggestion="Add usage examples",
                line_number=50,
            ),
            function=comprehensive_function,
            docstring=comprehensive_function.docstring,
            related_functions=comprehensive_rag_examples,
        )

        suggestion = generator.generate(context)

        # Should use RAG examples
        assert suggestion.metadata.used_rag_examples is True

        # Should have Python prompt
        assert ">>>" in suggestion.suggested_text

        # Should adapt function name
        assert "process_transaction" in suggestion.suggested_text

    def test_all_generators_metadata_consistency(
        self, comprehensive_function, comprehensive_rag_examples
    ):
        """Test all generators produce consistent metadata."""
        generators = [
            ParameterSuggestionGenerator(),
            ReturnSuggestionGenerator(),
            RaisesSuggestionGenerator(),
            BehaviorSuggestionGenerator(),
            ExampleSuggestionGenerator(),
        ]

        issue_types = [
            "missing_params",
            "missing_returns",
            "missing_raises",
            "description_outdated",
            "missing_examples",
        ]

        for generator, issue_type in zip(generators, issue_types, strict=False):
            context = SuggestionContext(
                issue=InconsistencyIssue(
                    issue_type=issue_type,
                    severity="medium",
                    description=f"Test {issue_type}",
                    suggestion="Fix it",
                    line_number=50,
                    details=(
                        {"parameter_name": "validate"}
                        if "parameter" in issue_type
                        else {}
                    ),
                ),
                function=comprehensive_function,
                docstring=comprehensive_function.docstring,
                related_functions=comprehensive_rag_examples,
            )

            suggestion = generator.generate(context)

            # All should have proper metadata
            assert suggestion.metadata.generator_type == generator.__class__.__name__
            assert suggestion.metadata.generator_version == "1.0.0"
            assert isinstance(suggestion.metadata.used_rag_examples, bool)

    def test_generators_handle_empty_rag_gracefully(self, comprehensive_function):
        """Test all generators handle empty RAG corpus gracefully."""
        generators = [
            ParameterSuggestionGenerator(),
            ReturnSuggestionGenerator(),
            RaisesSuggestionGenerator(),
            BehaviorSuggestionGenerator(),
            ExampleSuggestionGenerator(),
        ]

        for generator in generators:
            context = SuggestionContext(
                issue=InconsistencyIssue(
                    issue_type="description_outdated",
                    severity="low",
                    description="Test issue",
                    suggestion="Fix it",
                    line_number=50,
                ),
                function=comprehensive_function,
                docstring=comprehensive_function.docstring,
                related_functions=[],  # Empty RAG
            )

            # Should not crash
            suggestion = generator.generate(context)
            assert suggestion is not None
            assert suggestion.metadata.used_rag_examples is False

    def test_generators_filter_low_similarity(self, comprehensive_function):
        """Test all generators filter out low similarity examples."""
        # Create low similarity examples
        low_similarity_examples = [
            {
                "signature": "def unrelated_function() -> None:",
                "docstring": "Does something completely different",
                "similarity": 0.1,  # Very low
                "source_file": "unrelated.py",
            }
        ]

        generators = [
            ParameterSuggestionGenerator(),
            ReturnSuggestionGenerator(),
            RaisesSuggestionGenerator(),
            BehaviorSuggestionGenerator(),
            ExampleSuggestionGenerator(),
        ]

        for generator in generators:
            context = SuggestionContext(
                issue=InconsistencyIssue(
                    issue_type="description_outdated",
                    severity="low",
                    description="Test",
                    suggestion="Test",
                    line_number=50,
                ),
                function=comprehensive_function,
                docstring=comprehensive_function.docstring,
                related_functions=low_similarity_examples,
            )

            suggestion = generator.generate(context)

            # Should not use low similarity examples
            assert suggestion.metadata.used_rag_examples is False
