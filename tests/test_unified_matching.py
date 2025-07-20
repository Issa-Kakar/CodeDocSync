"""Tests for unified matching facade (First Half Implementation)."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile

from codedocsync.matcher.unified_facade import UnifiedMatchingFacade
from codedocsync.matcher.models import (
    MatchResult,
    MatchedPair,
    MatchType,
    MatchConfidence,
)
from codedocsync.parser import ParsedFunction, FunctionSignature, RawDocstring
from codedocsync.utils.config import CodeDocSyncConfig


class TestUnifiedMatchingFacade:
    """Test the UnifiedMatchingFacade class basic functionality."""

    def test_initialization(self):
        """Test facade initializes correctly."""
        facade = UnifiedMatchingFacade()

        assert facade.config is not None
        assert facade.stats["total_time"] == 0.0
        assert facade.stats["matches_by_type"]["direct"] == 0
        assert facade.stats["matches_by_type"]["contextual"] == 0
        assert facade.stats["matches_by_type"]["semantic"] == 0

    def test_initialization_with_config(self):
        """Test facade initialization with custom config."""
        config = CodeDocSyncConfig()
        facade = UnifiedMatchingFacade(config)

        assert facade.config is config

    def test_discover_python_files(self):
        """Test Python file discovery with exclusions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "main.py").touch()
            (temp_path / "utils.py").touch()
            (temp_path / ".git").mkdir()
            (temp_path / ".git" / "config").touch()
            (temp_path / "__pycache__").mkdir()
            (temp_path / "__pycache__" / "main.pyc").touch()
            (temp_path / "venv").mkdir()
            (temp_path / "venv" / "lib.py").touch()

            facade = UnifiedMatchingFacade()
            python_files = facade._discover_python_files(temp_path)

            # Should find only the actual Python files
            file_names = [f.name for f in python_files]
            assert "main.py" in file_names
            assert "utils.py" in file_names
            assert len(python_files) == 2

    def test_get_stats_empty(self):
        """Test stats retrieval with no matches."""
        facade = UnifiedMatchingFacade()
        stats = facade.get_stats()

        assert stats["total_time_seconds"] == 0.0
        assert stats["total_matches"] == 0
        assert stats["files_processed"] == 0
        assert stats["match_distribution"]["direct"] == "0%"

    def test_get_stats_with_matches(self):
        """Test stats calculation with some matches."""
        facade = UnifiedMatchingFacade()
        facade.stats["matches_by_type"]["direct"] = 5
        facade.stats["matches_by_type"]["contextual"] = 3
        facade.stats["matches_by_type"]["semantic"] = 2
        facade.stats["total_time"] = 10.5

        stats = facade.get_stats()

        assert stats["total_time_seconds"] == 10.5
        assert stats["total_matches"] == 10
        assert stats["match_distribution"]["direct"] == "50.0%"
        assert stats["match_distribution"]["contextual"] == "30.0%"
        assert stats["match_distribution"]["semantic"] == "20.0%"

    @patch("codedocsync.matcher.unified_facade.MatchingFacade")
    @patch("codedocsync.matcher.unified_facade.ContextualMatchingFacade")
    @patch("codedocsync.matcher.unified_facade.SemanticMatcher")
    @patch("codedocsync.matcher.unified_facade.IntegratedParser")
    @pytest.mark.asyncio
    async def test_match_project_basic(
        self, mock_parser, mock_semantic, mock_contextual, mock_direct
    ):
        """Test basic project matching workflow."""
        # Setup mocks
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance

        # Create sample function
        sample_function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type="None"
            ),
            docstring=RawDocstring(raw_text="Test function", line_number=2),
            file_path="test.py",
            line_number=1,
            end_line_number=3,
            source_code="def test_func(): pass",
        )
        mock_parser_instance.parse_file.return_value = [sample_function]

        # Mock direct matching facade
        mock_direct_facade = Mock()
        mock_direct.return_value = mock_direct_facade
        direct_result = MatchResult(
            total_functions=1,
            matched_pairs=[
                MatchedPair(
                    function=sample_function,
                    docstring=sample_function.docstring,
                    match_type=MatchType.EXACT,
                    confidence=MatchConfidence(
                        overall=0.9,
                        name_similarity=1.0,
                        location_score=1.0,
                        signature_similarity=1.0,
                    ),
                    match_reason="Exact match",
                )
            ],
            unmatched_functions=[],
        )
        mock_direct_facade.match_project.return_value = direct_result

        # Mock contextual matching facade
        mock_contextual_facade = Mock()
        mock_contextual.return_value = mock_contextual_facade
        contextual_result = MatchResult(
            total_functions=1,
            matched_pairs=[
                MatchedPair(
                    function=sample_function,
                    docstring=sample_function.docstring,
                    match_type=MatchType.EXACT,
                    confidence=MatchConfidence(
                        overall=0.9,
                        name_similarity=1.0,
                        location_score=1.0,
                        signature_similarity=1.0,
                    ),
                    match_reason="Exact match",
                )
            ],
            unmatched_functions=[],
        )
        mock_contextual_facade.match_project.return_value = contextual_result

        # Mock semantic matcher
        mock_semantic_instance = Mock()
        mock_semantic.return_value = mock_semantic_instance
        mock_semantic_instance.prepare_semantic_index = AsyncMock()
        mock_semantic_instance.match_with_embeddings = AsyncMock(
            return_value=contextual_result
        )
        mock_semantic_instance.get_stats.return_value = {"test": "stats"}

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").touch()

            facade = UnifiedMatchingFacade()

            result = await facade.match_project(str(temp_path), enable_semantic=True)

            # Verify the result
            assert result is not None
            assert len(result.matched_pairs) == 1
            assert result.matched_pairs[0].function.signature.name == "test_func"

            # Verify stats were updated
            assert facade.stats["files_processed"] > 0
            assert facade.stats["total_time"] > 0

    @patch("codedocsync.matcher.unified_facade.MatchingFacade")
    @patch("codedocsync.matcher.unified_facade.ContextualMatchingFacade")
    @patch("codedocsync.matcher.unified_facade.IntegratedParser")
    @pytest.mark.asyncio
    async def test_match_project_without_semantic(
        self, mock_parser, mock_contextual, mock_direct
    ):
        """Test project matching without semantic matching."""
        # Setup mocks (similar to above but without semantic)
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_file.return_value = []

        mock_direct_facade = Mock()
        mock_direct.return_value = mock_direct_facade
        empty_result = MatchResult(
            total_functions=0, matched_pairs=[], unmatched_functions=[],
        )
        mock_direct_facade.match_project.return_value = empty_result

        mock_contextual_facade = Mock()
        mock_contextual.return_value = mock_contextual_facade
        mock_contextual_facade.match_project.return_value = empty_result

        with tempfile.TemporaryDirectory() as temp_dir:
            facade = UnifiedMatchingFacade()

            result = await facade.match_project(str(temp_dir), enable_semantic=False)

            assert result is not None
            assert len(result.matched_pairs) == 0
            # Semantic matching time should be 0
            assert facade.stats["semantic_matching_time"] == 0.0

    def test_print_summary(self, capsys):
        """Test summary printing functionality."""
        facade = UnifiedMatchingFacade()
        facade.stats["total_time"] = 5.0
        facade.stats["files_processed"] = 10
        facade.stats["matches_by_type"]["direct"] = 8
        facade.stats["matches_by_type"]["contextual"] = 2
        facade.stats["parsing_time"] = 1.0
        facade.stats["direct_matching_time"] = 2.0
        facade.stats["contextual_matching_time"] = 2.0

        facade.print_summary()

        captured = capsys.readouterr()
        assert "Unified Matching Summary" in captured.out
        assert "Total time: 5.00s" in captured.out
        assert "Files processed: 10" in captured.out
        assert "Total matches: 10" in captured.out


class TestUnifiedMatchingIntegration:
    """Test unified matching integration scenarios."""

    @patch("codedocsync.matcher.unified_facade.Path.rglob")
    def test_discover_files_empty_directory(self, mock_rglob):
        """Test file discovery in empty directory."""
        mock_rglob.return_value = []

        facade = UnifiedMatchingFacade()
        result = facade._discover_python_files(Path("/fake/path"))

        assert result == []

    def test_stats_calculation_edge_cases(self):
        """Test statistics calculation edge cases."""
        facade = UnifiedMatchingFacade()

        # Test with zero matches
        stats = facade.get_stats()
        assert stats["match_distribution"]["direct"] == "0.0%"

        # Test with only one match type
        facade.stats["matches_by_type"]["direct"] = 1
        stats = facade.get_stats()
        assert stats["match_distribution"]["direct"] == "100.0%"
        assert stats["match_distribution"]["contextual"] == "0.0%"

    @pytest.mark.asyncio
    async def test_error_handling_during_parsing(self):
        """Test error handling when file parsing fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "invalid.py").write_text("invalid python syntax {{{")

            facade = UnifiedMatchingFacade()

            # This should not crash, but handle the error gracefully
            with patch(
                "codedocsync.matcher.unified_facade.IntegratedParser"
            ) as mock_parser:
                mock_parser_instance = Mock()
                mock_parser.return_value = mock_parser_instance
                mock_parser_instance.parse_file.side_effect = Exception("Parse error")

                with patch(
                    "codedocsync.matcher.unified_facade.MatchingFacade"
                ) as mock_direct:
                    mock_direct_facade = Mock()
                    mock_direct.return_value = mock_direct_facade
                    empty_result = MatchResult(
                        total_functions=0, matched_pairs=[], unmatched_functions=[],
                    )
                    mock_direct_facade.match_project.return_value = empty_result

                    with patch(
                        "codedocsync.matcher.unified_facade.ContextualMatchingFacade"
                    ) as mock_contextual:
                        mock_contextual_facade = Mock()
                        mock_contextual.return_value = mock_contextual_facade
                        mock_contextual_facade.match_project.return_value = empty_result

                        # Should complete without crashing
                        result = await facade.match_project(
                            str(temp_path), enable_semantic=False
                        )
                        assert result is not None
