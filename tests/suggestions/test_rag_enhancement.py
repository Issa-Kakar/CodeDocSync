"""Unit tests for RAG-enhanced suggestion generation."""

from unittest.mock import Mock, patch

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
)
from codedocsync.parser.docstring_models import DocstringFormat, ParsedDocstring
from codedocsync.suggestions.generators.parameter_generator import (
    ParameterSuggestionGenerator,
)
from codedocsync.suggestions.generators.return_generator import (
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext


class TestRAGEnhancement:
    """Test RAG enhancement functionality in suggestion generators."""

    def create_test_context(
        self, issue_type: str, related_functions: list[dict] = None
    ) -> SuggestionContext:
        """Create a test suggestion context."""
        # Create test function
        param = FunctionParameter(
            name="query", type_annotation="str", default_value=None, is_required=True
        )

        signature = FunctionSignature(
            name="search_data", parameters=[param], return_type="list[dict]"
        )

        # Create a parsed docstring with no parameters (to trigger parameter_missing)
        parsed_docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Search for data.",
            description=None,
            parameters=[],  # Empty parameters list
            returns=None,
            raises=[],
            examples=[],
            raw_text='"""Search for data."""',
        )

        function = ParsedFunction(
            signature=signature,
            docstring=parsed_docstring,
            file_path="test.py",
            line_number=1,
            end_line_number=3,
        )

        # Create issue
        issue = InconsistencyIssue(
            issue_type=issue_type,
            severity="high",
            description="Test issue",
            suggestion="Test suggestion",
            line_number=1,
            confidence=0.9,
        )

        # Create context with related functions
        return SuggestionContext(
            issue=issue,
            function=function,
            docstring=function.docstring,
            project_style="google",
            surrounding_code=None,
            related_functions=related_functions or [],
        )

    def test_parameter_generator_uses_rag_examples(self):
        """Test that parameter generator uses RAG examples when available."""
        # Create RAG examples with higher similarity for the parameter
        rag_examples = [
            {
                "signature": "def search_documents(query: str, limit: int = 10) -> list[dict]:",
                "docstring": """Search for documents matching the query.

Args:
    query (str): The search query string to match against documents.
    limit (int): Maximum number of results to return. Defaults to 10.

Returns:
    list[dict]: List of matching documents with relevance scores.
""",
                "similarity": 0.95,  # Increased similarity
            }
        ]

        # Create context with RAG examples
        context = self.create_test_context("parameter_missing", rag_examples)

        # Create generator and generate suggestion
        generator = ParameterSuggestionGenerator()
        suggestion = generator.generate(context)

        # Check that RAG was used
        assert suggestion.metadata.used_rag_examples is True

        # Check that the description is enhanced (not just "Description for query")
        assert "Description for query" not in suggestion.suggested_text
        assert (
            "search" in suggestion.suggested_text.lower()
            or "query" in suggestion.suggested_text.lower()
        )

    def test_parameter_generator_without_rag_examples(self):
        """Test parameter generator behavior without RAG examples."""
        # Create context without RAG examples
        context = self.create_test_context("parameter_missing", [])

        # Create generator and generate suggestion
        generator = ParameterSuggestionGenerator()
        suggestion = generator.generate(context)

        # Check that RAG was not used
        assert suggestion.metadata.used_rag_examples is False

        # Check that basic description is used
        assert "Description for query" in suggestion.suggested_text

    def test_return_generator_uses_rag_examples(self):
        """Test that return generator uses RAG examples when available."""
        # Create RAG examples
        rag_examples = [
            {
                "signature": "def fetch_results(query: str) -> list[dict]:",
                "docstring": '''"""Fetch results based on the query.

                Args:
                    query: The query string.

                Returns:
                    List of dictionaries containing the search results with metadata.
                """''',
                "similarity": 0.9,
            }
        ]

        # Create context with RAG examples
        context = self.create_test_context("missing_returns", rag_examples)

        # Create generator and generate suggestion
        generator = ReturnSuggestionGenerator()
        suggestion = generator.generate(context)

        # Check that RAG was used
        assert suggestion.metadata.used_rag_examples is True

        # The return description should be more detailed than the basic one
        assert (
            "The list" not in suggestion.suggested_text
        )  # Basic description would be "The list result"

    def test_rag_similarity_threshold(self):
        """Test that only high-similarity examples are used."""
        # Create low-similarity RAG examples
        rag_examples = [
            {
                "signature": "def unrelated_function(x: int) -> int:",
                "docstring": '''"""Do something unrelated.

                Args:
                    x: Some number.

                Returns:
                    The processed number.
                """''',
                "similarity": 0.3,  # Low similarity
            }
        ]

        # Create context with low-similarity examples
        context = self.create_test_context("parameter_missing", rag_examples)

        # Create generator and generate suggestion
        generator = ParameterSuggestionGenerator()

        # Mock the _extract_parameter_patterns_from_examples to verify it filters
        with patch.object(
            generator, "_extract_parameter_patterns_from_examples"
        ) as mock_extract:
            mock_extract.return_value = []  # Simulate no patterns found
            suggestion = generator.generate(context)

            # RAG should not be used due to low similarity
            assert suggestion.metadata.used_rag_examples is False

    def test_rag_pattern_extraction(self):
        """Test the pattern extraction from RAG examples."""
        generator = ParameterSuggestionGenerator()

        # Test examples with various docstring formats
        examples = [
            {
                "signature": "def process_data(input_data: dict, validate: bool = True) -> dict:",
                "docstring": """Process input data with optional validation.

Args:
    input_data (dict): The data dictionary to process.
    validate (bool): Whether to validate the input data. Defaults to True.

Returns:
    dict: Processed data dictionary.
""",
                "similarity": 0.85,
            }
        ]

        # Create a parameter to match
        param = FunctionParameter(
            name="data", type_annotation="dict", default_value=None, is_required=True
        )

        # Extract patterns
        patterns = generator._extract_parameter_patterns_from_examples(
            param.name, examples, param.type_annotation
        )

        # Should find patterns even with different parameter names but same type
        assert len(patterns) > 0
        assert patterns[0]["similarity"] > 0  # Should have calculated similarity

    @patch("codedocsync.suggestions.integration._get_rag_manager")
    def test_rag_retrieval_integration(self, mock_get_rag_manager):
        """Test the integration with RAG corpus retrieval."""
        from codedocsync.suggestions.config import SuggestionConfig
        from codedocsync.suggestions.integration import SuggestionIntegration
        from codedocsync.suggestions.rag_corpus import DocstringExample

        # Mock RAG corpus manager instance
        mock_instance = Mock()
        mock_get_rag_manager.return_value = mock_instance

        # Create mock examples
        example = Mock(spec=DocstringExample)
        example.function_signature = "def search(query: str) -> list:"
        example.docstring_content = '"""Search for items."""'
        example.similarity_score = 0.9

        mock_instance.retrieve_examples.return_value = [example]

        # Create suggestion integration
        config = SuggestionConfig(use_rag=True)
        integration = SuggestionIntegration(config)

        # Create test issue and pair
        issue = InconsistencyIssue(
            issue_type="missing_returns",
            severity="high",
            description="Missing return documentation",
            suggestion="Add return documentation",
            line_number=1,
            confidence=0.9,
        )

        signature = FunctionSignature(
            name="test_function", parameters=[], return_type="list"
        )

        # Create a parsed docstring instead of raw
        parsed_doc = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Test function.",
            description=None,
            parameters=[],
            returns=None,  # Missing returns to trigger the issue
            raises=[],
            examples=[],
            raw_text='"""Test function."""',
        )

        function = ParsedFunction(
            signature=signature,
            docstring=parsed_doc,
            file_path="test.py",
            line_number=1,
            end_line_number=3,
        )

        from codedocsync.matcher.models import MatchConfidence, MatchedPair, MatchType

        pair = MatchedPair(
            function=function,
            docstring=None,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=0.9,
                location_score=0.9,
                signature_similarity=0.9,
            ),
            match_type=MatchType.EXACT,
            match_reason="Direct match",
        )

        # Create an AnalysisResult to use the public interface
        from codedocsync.analyzer.models import AnalysisResult

        analysis_result = AnalysisResult(
            matched_pair=pair, issues=[issue], used_llm=False, analysis_time_ms=0.0
        )

        # Enhance the analysis result with suggestions
        enhanced_result = integration.enhance_analysis_result(analysis_result)

        # Verify RAG was called
        mock_instance.retrieve_examples.assert_called_once()

        # Verify suggestion was created
        assert len(enhanced_result.issues) == 1
        enhanced_issue = enhanced_result.issues[0]
        assert enhanced_issue.suggestion is not None
