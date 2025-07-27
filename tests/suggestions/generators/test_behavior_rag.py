"""Tests for BehaviorGenerator RAG enhancement functionality."""

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
)
from codedocsync.parser.docstring_models import DocstringFormat, ParsedDocstring
from codedocsync.suggestions.generators.behavior_generator import (
    BehaviorSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext


class TestBehaviorGeneratorRAG:
    """Test RAG enhancement for BehaviorGenerator."""

    @pytest.fixture
    def mock_function(self):
        """Create a mock function for testing."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="process_user_data",
                parameters=[
                    FunctionParameter(
                        name="user_data",
                        type_annotation="dict[str, Any]",
                        default_value=None,
                        is_required=True,
                    ),
                    FunctionParameter(
                        name="config",
                        type_annotation="dict[str, str]",
                        default_value=None,
                        is_required=False,
                    ),
                ],
                return_type="dict[str, Any]",
            ),
            docstring=None,
            file_path="test.py",
            line_number=10,
            end_line_number=20,
        )

    @pytest.fixture
    def rag_examples(self):
        """Create RAG examples for testing."""
        return [
            {
                "signature": "def transform_user_records(records: dict[str, Any], settings: dict) -> dict:",
                "docstring": '''"""Transform user records based on configuration settings.

                Processes each user record according to the transformation rules
                defined in the settings. Validates data integrity and handles
                missing fields gracefully.

                Args:
                    records: Dictionary of user records to transform.
                    settings: Configuration settings for transformation.

                Returns:
                    Dictionary containing transformed user data with
                    validation results and processing metadata.
                """''',
                "similarity": 0.85,
            },
            {
                "signature": "def validate_user_input(input_data: dict) -> dict:",
                "docstring": '''"""Validates user input data against schema.

                Works with JSON schema validation to ensure data integrity.
                """''',
                "similarity": 0.70,
            },
        ]

    def test_vocabulary_extraction(self, rag_examples):
        """Test vocabulary extraction from RAG examples."""
        generator = BehaviorSuggestionGenerator()

        vocabulary = generator._extract_behavior_vocabulary(rag_examples)

        # Verify vocabulary categories
        assert "verbs" in vocabulary
        assert "patterns" in vocabulary
        assert "technical" in vocabulary

        # Check extracted verbs - look for transforms or processes
        # The docstrings use "Transform" and "Processes" which should be lowercased
        verbs = vocabulary.get("verbs", [])
        assert "transforms" in verbs or "transform" in verbs or "processes" in verbs

        # Check patterns
        assert any("based on" in pattern for pattern in vocabulary["patterns"])

        # Check technical terms
        assert "JSON" in vocabulary["technical"]

    def test_behavior_generation_with_rag(self, mock_function, rag_examples):
        """Test behavior description generation using RAG."""
        generator = BehaviorSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="description_outdated",
                severity="medium",
                description="Description needs update",
                suggestion="Update the description",
                line_number=10,
            ),
            function=mock_function,
            docstring=ParsedDocstring(
                format=DocstringFormat.GOOGLE,
                summary="Old description",
                description=None,
                raw_text='"""Old description"""',
            ),
            related_functions=rag_examples,
        )

        suggestion = generator.generate(context)

        # Verify RAG was used
        assert suggestion.metadata.used_rag_examples is True

        # Verify description quality
        assert (
            "process" in suggestion.suggested_text.lower()
            or "transform" in suggestion.suggested_text.lower()
        )
        assert "user" in suggestion.suggested_text.lower()
        assert "data" in suggestion.suggested_text.lower()

    def test_tokenize_function_name(self):
        """Test function name tokenization."""
        generator = BehaviorSuggestionGenerator()

        # Test snake_case
        tokens = generator._tokenize_function_name("process_user_data")
        assert tokens == ["process", "user", "data"]

        # Test camelCase
        tokens = generator._tokenize_function_name("processUserData")
        assert tokens == ["process", "user", "data"]

        # Test mixed
        tokens = generator._tokenize_function_name("process_userData")
        assert tokens == ["process", "user", "data"]

    def test_parameter_description(self):
        """Test parameter description generation."""
        generator = BehaviorSuggestionGenerator()

        # Test with type annotation
        param = FunctionParameter(
            name="user_list",
            type_annotation="list[User]",
            default_value=None,
            is_required=True,
        )
        desc = generator._describe_parameter_for_behavior(param)
        assert desc == "the user list list"

        # Test without type
        param = FunctionParameter(
            name="config",
            type_annotation=None,
            default_value=None,
            is_required=True,
        )
        desc = generator._describe_parameter_for_behavior(param)
        assert desc == "the given config"

    def test_return_phrase_generation(self):
        """Test return phrase generation."""
        generator = BehaviorSuggestionGenerator()

        # Test bool return
        phrase = generator._generate_return_phrase_for_behavior("bool", [])
        assert phrase == "returns True if successful"

        # Test dict return
        phrase = generator._generate_return_phrase_for_behavior("dict[str, Any]", [])
        assert phrase == "returns the results as a dictionary"

        # Test None return
        phrase = generator._generate_return_phrase_for_behavior("None", [])
        assert phrase is None

        # Test with existing return pattern
        phrase = generator._generate_return_phrase_for_behavior(
            "bool", ["returns the result"]
        )
        assert phrase is None  # Should not duplicate

    def test_technical_terms_relevance(self, mock_function):
        """Test finding relevant technical terms."""
        generator = BehaviorSuggestionGenerator()

        tech_terms = ["JSON", "API", "database", "user", "config"]

        relevant = generator._find_relevant_technical_terms(mock_function, tech_terms)

        # Should find terms that appear in function context
        assert "user" in relevant
        assert "config" in relevant
        assert "database" not in relevant  # Not in function context

    def test_grammar_correction(self):
        """Test grammar correction in descriptions."""
        generator = BehaviorSuggestionGenerator()

        # Test capitalization
        result = generator._ensure_behavior_grammar("processes data")
        assert result == "Processes data."

        # Test double spaces
        result = generator._ensure_behavior_grammar("Processes  the  data")
        assert result == "Processes the data."

        # Test duplicate "and"
        result = generator._ensure_behavior_grammar("Validates and and returns data")
        assert result == "Validates and returns data."

    def test_full_vocabulary_based_generation(self, mock_function):
        """Test complete behavior generation from vocabulary."""
        generator = BehaviorSuggestionGenerator()

        vocabulary = {
            "verbs": ["processes", "transforms", "validates"],
            "patterns": ["based on configuration", "using settings"],
            "technical": ["JSON", "API", "user", "config"],
        }

        description = generator._generate_behavior_from_vocabulary(
            mock_function, vocabulary, []
        )

        # Verify description components
        assert description.startswith("Processes")  # Capitalized verb
        assert "user data" in description  # Parameter description
        assert (
            "based on configuration" in description or "using settings" in description
        )  # Pattern
        assert "returns" in description  # Return phrase
        assert description.endswith(".")  # Proper ending

    def test_behavior_without_rag_fallback(self, mock_function):
        """Test behavior generation without RAG examples."""
        generator = BehaviorSuggestionGenerator()

        context = SuggestionContext(
            issue=InconsistencyIssue(
                issue_type="description_outdated",
                severity="medium",
                description="Description needs update",
                suggestion="Update the description",
                line_number=10,
            ),
            function=mock_function,
            docstring=None,
            related_functions=[],  # No RAG examples
        )

        suggestion = generator.generate(context)

        # Should not use RAG
        assert suggestion.metadata.used_rag_examples is False

        # Should still generate something
        assert suggestion.suggested_text
        assert len(suggestion.suggested_text) > 10
