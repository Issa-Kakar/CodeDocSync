"""Tests for RAG-enhanced suggestion integration."""

from unittest.mock import Mock, patch

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.matcher.models import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser.ast_parser import (
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.integration import SuggestionIntegration
from codedocsync.suggestions.rag_corpus import DocstringExample


class TestRAGIntegration:
    """Test RAG integration in suggestion generation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a test function
        self.test_function = ParsedFunction(
            signature=FunctionSignature(
                name="process_data", parameters=[], return_type="list[dict]"
            ),
            docstring=RawDocstring(
                raw_text="Process data and return results.",
                line_number=5,
            ),
            file_path="test.py",
            line_number=1,
            end_line_number=10,
        )

        # Create a test issue
        self.test_issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="medium",
            description="Missing parameter documentation",
            suggestion="Add parameter documentation",
            line_number=1,
        )

        # Create a matched pair
        self.test_pair = MatchedPair(
            function=self.test_function,
            docstring=None,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=0.9,
                location_score=0.9,
                signature_similarity=0.9,
            ),
            match_type=MatchType.EXACT,
            match_reason="Direct name match",
        )

    def test_create_context_with_rag_enabled(self):
        """Test that RAG examples are retrieved when enabled."""
        # Create config with RAG enabled
        config = SuggestionConfig(use_rag=True)
        integration = SuggestionIntegration(config)

        # Mock RAG corpus manager
        mock_rag_manager = Mock()
        mock_examples = [
            DocstringExample(
                function_name="analyze_data",
                module_path="example.py",
                function_signature="def analyze_data(data: list[dict]) -> dict:",
                docstring_format="google",
                docstring_content='"""Analyze data and return statistics."""',
                has_params=True,
                has_returns=True,
                has_examples=False,
                complexity_score=3,
                quality_score=4,
            )
        ]
        mock_rag_manager.retrieve_examples.return_value = mock_examples

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=mock_rag_manager,
        ):
            # Create context
            context = integration._create_context(self.test_issue, self.test_pair)

            # Verify RAG retrieval was called
            mock_rag_manager.retrieve_examples.assert_called_once()
            call_args = mock_rag_manager.retrieve_examples.call_args
            assert call_args[1]["function"] == self.test_function
            assert call_args[1]["n_results"] == 3
            assert "min_similarity" in call_args[1]

            # Verify related_functions populated
            assert len(context.related_functions) == 1
            assert (
                context.related_functions[0]["signature"]
                == mock_examples[0].function_signature
            )
            assert (
                context.related_functions[0]["docstring"]
                == mock_examples[0].docstring_content
            )
            # Note: similarity_score is not part of DocstringExample dataclass
            # The integration uses getattr with a default of 0.8

    def test_create_context_with_rag_disabled(self):
        """Test that RAG is not used when disabled."""
        # Create config with RAG disabled
        config = SuggestionConfig(use_rag=False)
        integration = SuggestionIntegration(config)

        # Mock RAG corpus manager (should not be called)
        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager"
        ) as mock_rag_class:
            # Create context
            context = integration._create_context(self.test_issue, self.test_pair)

            # Verify RAG was not instantiated
            mock_rag_class.assert_not_called()

            # Verify related_functions is empty
            assert context.related_functions == []

    def test_create_context_rag_failure_graceful(self):
        """Test that RAG failures are handled gracefully."""
        # Create config with RAG enabled
        config = SuggestionConfig(use_rag=True)
        integration = SuggestionIntegration(config)

        # Mock RAG corpus manager initialization to raise exception
        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            side_effect=Exception("RAG initialization error"),
        ):
            # Create context - should not raise
            context = integration._create_context(self.test_issue, self.test_pair)

            # Verify related_functions is empty (fallback)
            assert context.related_functions == []

    def test_create_context_with_multiple_rag_examples(self):
        """Test handling multiple RAG examples."""
        # Create config with RAG enabled
        config = SuggestionConfig(use_rag=True)
        integration = SuggestionIntegration(config)

        # Mock RAG corpus manager with multiple examples
        mock_rag_manager = Mock()
        mock_examples = [
            DocstringExample(
                function_name="process_items",
                module_path="example1.py",
                function_signature="def process_items(items: list) -> dict:",
                docstring_format="google",
                docstring_content='"""Process items and return summary."""',
                has_params=True,
                has_returns=True,
                has_examples=False,
                complexity_score=2,
                quality_score=4,
            ),
            DocstringExample(
                function_name="handle_data",
                module_path="example2.py",
                function_signature="def handle_data(data: dict) -> list:",
                docstring_format="google",
                docstring_content='"""Handle data and return results."""',
                has_params=True,
                has_returns=True,
                has_examples=False,
                complexity_score=2,
                quality_score=4,
            ),
            DocstringExample(
                function_name="transform_records",
                module_path="example3.py",
                function_signature="def transform_records(records: list) -> list:",
                docstring_format="google",
                docstring_content='"""Transform records to new format."""',
                has_params=True,
                has_returns=True,
                has_examples=False,
                complexity_score=3,
                quality_score=4,
            ),
        ]
        mock_rag_manager.retrieve_examples.return_value = mock_examples

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=mock_rag_manager,
        ):
            # Create context
            context = integration._create_context(self.test_issue, self.test_pair)

            # Verify all examples are included
            assert len(context.related_functions) == 3

            # Verify they're sorted by similarity (highest first)
            # Since retrieve_examples returns tuples of (example, score)
            # we don't have direct access to similarity scores in context
