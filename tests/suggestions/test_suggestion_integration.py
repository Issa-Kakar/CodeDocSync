"""Tests for RAG-enhanced suggestion integration."""

from unittest.mock import Mock, patch

from codedocsync.analyzer.models import (
    InconsistencyIssue,
    IssueSeverity,
    MatchConfidence,
    MatchedPair,
    MatchType,
)
from codedocsync.parser.models import FunctionSignature, ParsedFunction, RawDocstring
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.integration import SuggestionIntegration
from codedocsync.suggestions.rag_corpus import RAGExample


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
                content="Process data and return results.",
                format_type="google",
                line_number=5,
            ),
            file_path="test.py",
            line_number=1,
        )

        # Create a test issue
        self.test_issue = InconsistencyIssue(
            issue_type="missing_parameters",
            severity=IssueSeverity.MEDIUM,
            message="Missing parameter documentation",
            file_path="test.py",
            line_number=1,
            affected_element="process_data",
        )

        # Create a matched pair
        self.test_pair = MatchedPair(
            function=self.test_function,
            documentation=None,
            confidence=MatchConfidence(value=0.9),
            match_type=MatchType(value="direct"),
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
            RAGExample(
                function_signature="def analyze_data(data: list[dict]) -> dict:",
                docstring_text='"""Analyze data and return statistics."""',
                issue_type="missing_parameters",
                similarity_score=0.85,
                source_file="example.py",
            )
        ]
        mock_rag_manager.retrieve_similar_examples.return_value = mock_examples

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=mock_rag_manager,
        ):
            # Create context
            context = integration._create_context(self.test_issue, self.test_pair)

            # Verify RAG retrieval was called
            mock_rag_manager.retrieve_similar_examples.assert_called_once()
            call_args = mock_rag_manager.retrieve_similar_examples.call_args
            assert "missing_parameters" in call_args[1]["issue_type"]
            assert call_args[1]["n_results"] == 3

            # Verify related_functions populated
            assert len(context.related_functions) == 1
            assert (
                context.related_functions[0]["signature"]
                == mock_examples[0].function_signature
            )
            assert (
                context.related_functions[0]["docstring"]
                == mock_examples[0].docstring_text
            )
            assert (
                context.related_functions[0]["similarity"]
                == mock_examples[0].similarity_score
            )

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

        # Mock RAG corpus manager to raise exception
        mock_rag_manager = Mock()
        mock_rag_manager.retrieve_similar_examples.side_effect = Exception("RAG error")

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=mock_rag_manager,
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
            RAGExample(
                function_signature="def process_items(items: list) -> dict:",
                docstring_text='"""Process items and return summary."""',
                issue_type="missing_parameters",
                similarity_score=0.90,
                source_file="example1.py",
            ),
            RAGExample(
                function_signature="def handle_data(data: dict) -> list:",
                docstring_text='"""Handle data and return results."""',
                issue_type="missing_parameters",
                similarity_score=0.85,
                source_file="example2.py",
            ),
            RAGExample(
                function_signature="def transform_records(records: list) -> list:",
                docstring_text='"""Transform records to new format."""',
                issue_type="missing_parameters",
                similarity_score=0.80,
                source_file="example3.py",
            ),
        ]
        mock_rag_manager.retrieve_similar_examples.return_value = mock_examples

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=mock_rag_manager,
        ):
            # Create context
            context = integration._create_context(self.test_issue, self.test_pair)

            # Verify all examples are included
            assert len(context.related_functions) == 3

            # Verify they're sorted by similarity (highest first)
            assert context.related_functions[0]["similarity"] == 0.90
            assert context.related_functions[1]["similarity"] == 0.85
            assert context.related_functions[2]["similarity"] == 0.80
