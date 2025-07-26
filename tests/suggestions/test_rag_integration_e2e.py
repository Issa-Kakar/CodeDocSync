"""End-to-end integration tests for RAG-enhanced suggestions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.matcher.models import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.integration import SuggestionIntegration
from codedocsync.suggestions.rag_corpus import RAGCorpusManager


class TestRAGEnhancedSuggestions:
    """Test full pipeline of RAG-enhanced suggestion generation."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test corpus
        self.temp_dir = tempfile.mkdtemp()
        self.corpus_path = Path(self.temp_dir) / "test_corpus.json"

        # Create test corpus with relevant examples
        test_corpus = {
            "examples": [
                {
                    "function_signature": "def calculate_metrics(data: list[float], window_size: int = 10) -> dict[str, float]:",
                    "docstring_text": '''"""Calculate statistical metrics for the given data.

                    Args:
                        data: List of numerical values to analyze.
                        window_size: Size of the rolling window for moving averages. Defaults to 10.

                    Returns:
                        Dictionary containing calculated metrics including mean, std, and moving average.
                    """''',
                    "issue_type": "missing_parameters",
                    "source_file": "metrics.py",
                },
                {
                    "function_signature": "def process_batch(items: list[dict], config: dict | None = None) -> tuple[list[dict], dict]:",
                    "docstring_text": '''"""Process a batch of items according to configuration.

                    Args:
                        items: List of items to process, each item should have 'id' and 'data' keys.
                        config: Optional configuration dictionary. If None, uses default settings.

                    Returns:
                        Tuple containing:
                            - Processed items list
                            - Processing statistics dictionary
                    """''',
                    "issue_type": "missing_parameters",
                    "source_file": "batch_processor.py",
                },
            ],
            "metadata": {
                "version": "1.0",
                "created_at": "2025-01-25",
                "total_examples": 2,
            },
        }

        with open(self.corpus_path, "w") as f:
            json.dump(test_corpus, f)

    @pytest.mark.xfail(
        reason="RAGCorpusManager.load_corpus and SuggestionIntegration.generate_suggestion not yet implemented"
    )
    def test_parameter_suggestion_with_rag(self):
        """Test that parameter suggestions are enhanced by RAG examples."""
        # Create a function missing parameter documentation
        test_function = ParsedFunction(
            signature=FunctionSignature(
                name="analyze_dataset",
                parameters=[
                    FunctionParameter(
                        name="dataset",
                        type_annotation="list[dict]",
                        default_value=None,
                        is_required=True,
                    ),
                    FunctionParameter(
                        name="options",
                        type_annotation="dict | None",
                        default_value="None",
                        is_required=False,
                    ),
                ],
                return_type="dict",
            ),
            docstring=RawDocstring(
                raw_text='"""Analyze the dataset."""',
                line_number=2,
            ),
            file_path="analyzer.py",
            line_number=1,
            end_line_number=3,
        )

        # Create issue and pair
        issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="medium",
            description="Parameters not documented",
            suggestion="Add Args section with parameter documentation",
            line_number=1,
        )

        pair = MatchedPair(
            function=test_function,
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

        # Initialize RAG corpus manager with test corpus
        rag_manager = RAGCorpusManager()
        rag_manager.load_corpus(str(self.corpus_path))

        # Test with RAG enabled
        config_with_rag = SuggestionConfig(use_rag=True)
        integration_with_rag = SuggestionIntegration(config_with_rag)

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=rag_manager,
        ):
            suggestion_with_rag = integration_with_rag.generate_suggestion(issue, pair)

        # Test with RAG disabled
        config_no_rag = SuggestionConfig(use_rag=False)
        integration_no_rag = SuggestionIntegration(config_no_rag)

        suggestion_no_rag = integration_no_rag.generate_suggestion(issue, pair)

        # Verify RAG-enhanced suggestion has better quality
        assert suggestion_with_rag is not None
        assert suggestion_no_rag is not None

        # RAG-enhanced should mention specific details like those in examples
        # (e.g., mentioning what keys dict should have, default behavior, etc.)
        assert "Args:" in suggestion_with_rag.suggested_fix
        assert "dataset" in suggestion_with_rag.suggested_fix
        assert "options" in suggestion_with_rag.suggested_fix

    @pytest.mark.xfail(
        reason="RAGCorpusManager.load_corpus and SuggestionIntegration.generate_suggestion not yet implemented"
    )
    def test_return_type_suggestion_with_rag(self):
        """Test that return type suggestions are enhanced by RAG examples."""
        # Create a function missing return documentation
        test_function = ParsedFunction(
            signature=FunctionSignature(
                name="compute_statistics",
                parameters=[
                    FunctionParameter(
                        name="values",
                        type_annotation="list[float]",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_type="tuple[dict[str, float], list[float]]",
            ),
            docstring=RawDocstring(
                raw_text='''"""Compute statistics for values.

                Args:
                    values: List of numerical values.
                """''',
                line_number=2,
            ),
            file_path="stats.py",
            line_number=1,
            end_line_number=5,
        )

        # Create issue for missing return documentation
        issue = InconsistencyIssue(
            issue_type="missing_returns",
            severity="medium",
            description="Return value not documented",
            suggestion="Add Returns section with return type documentation",
            line_number=1,
        )

        pair = MatchedPair(
            function=test_function,
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

        # Initialize RAG corpus manager
        rag_manager = RAGCorpusManager()
        rag_manager.load_corpus(str(self.corpus_path))

        # Generate suggestion with RAG
        config = SuggestionConfig(use_rag=True)
        integration = SuggestionIntegration(config)

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=rag_manager,
        ):
            suggestion = integration.generate_suggestion(issue, pair)

        # Verify suggestion quality
        assert suggestion is not None
        assert "Returns:" in suggestion.suggested_fix
        assert (
            "Tuple containing:" in suggestion.suggested_fix
            or "tuple" in suggestion.suggested_fix.lower()
        )

    @pytest.mark.xfail(
        reason="RAGCorpusManager.load_corpus and SuggestionIntegration.generate_suggestion not yet implemented"
    )
    def test_rag_performance_impact(self):
        """Test that RAG enhancement has acceptable performance impact."""
        import time

        # Create test data
        test_function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type="None"
            ),
            docstring=None,
            file_path="test.py",
            line_number=1,
            end_line_number=3,
        )

        issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="high",
            description="No docstring",
            suggestion="Add docstring to document the function",
            line_number=1,
        )

        pair = MatchedPair(
            function=test_function,
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

        # Initialize managers
        rag_manager = RAGCorpusManager()
        rag_manager.load_corpus(str(self.corpus_path))

        config = SuggestionConfig(use_rag=True)
        integration = SuggestionIntegration(config)

        # Measure time with RAG
        start_time = time.time()
        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=rag_manager,
        ):
            for _ in range(5):  # Run multiple times
                integration.generate_suggestion(issue, pair)
        rag_time = time.time() - start_time

        # Measure time without RAG
        config_no_rag = SuggestionConfig(use_rag=False)
        integration_no_rag = SuggestionIntegration(config_no_rag)

        start_time = time.time()
        for _ in range(5):
            integration_no_rag.generate_suggestion(issue, pair)
        no_rag_time = time.time() - start_time

        # RAG should add less than 100ms per suggestion (requirement from task)
        time_difference_per_suggestion = (rag_time - no_rag_time) / 5
        assert time_difference_per_suggestion < 0.1  # 100ms

    @pytest.mark.xfail(
        reason="SuggestionIntegration.generate_suggestion not yet implemented"
    )
    def test_rag_graceful_degradation(self):
        """Test that system works when RAG corpus is unavailable."""
        # Create test data
        test_function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type="None"
            ),
            docstring=None,
            file_path="test.py",
            line_number=1,
            end_line_number=3,
        )

        issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="high",
            description="No docstring",
            suggestion="Add docstring to document the function",
            line_number=1,
        )

        pair = MatchedPair(
            function=test_function,
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

        config = SuggestionConfig(use_rag=True)
        integration = SuggestionIntegration(config)

        # Mock RAG manager to simulate corpus load failure
        mock_rag_manager = Mock()
        mock_rag_manager.retrieve_examples.side_effect = Exception(
            "Corpus not available"
        )

        with patch(
            "codedocsync.suggestions.integration.RAGCorpusManager",
            return_value=mock_rag_manager,
        ):
            # Should not raise exception
            suggestion = integration.generate_suggestion(issue, pair)

            # Should still generate a suggestion
            assert suggestion is not None
            assert suggestion.suggested_fix is not None
