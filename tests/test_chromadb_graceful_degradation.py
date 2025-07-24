"""
Test ChromaDB graceful degradation.

This test verifies that the system works properly when ChromaDB is not available.
"""

import sys
from unittest.mock import patch

import pytest

from codedocsync.matcher import UnifiedMatchingFacade


def test_unified_facade_without_chromadb() -> None:
    """Test that UnifiedMatchingFacade works without ChromaDB."""
    # Mock ChromaDB to be unavailable
    with patch.dict(sys.modules, {"chromadb": None}):
        # This should not raise an error - facade should work without ChromaDB
        facade = UnifiedMatchingFacade()

        # The facade should still be functional even without semantic matching
        # It will use direct and contextual matching only
        assert facade is not None


def test_analyze_command_with_no_semantic_flag() -> None:
    """Test that analyze command properly handles --no-semantic flag."""
    import typer
    from typer.testing import CliRunner

    from codedocsync.cli.analyze import analyze

    app = typer.Typer()
    app.command()(analyze)

    runner = CliRunner()

    # Test help text includes --no-semantic
    result = runner.invoke(app, ["--help"])
    assert "--no-semantic" in result.output
    assert "Disable semantic matching" in result.output


def test_semantic_matcher_import_error() -> None:
    """Test that SemanticMatcher raises proper error when ChromaDB unavailable."""
    with patch("codedocsync.matcher.semantic_matcher.VECTOR_STORE_AVAILABLE", False):
        from codedocsync.matcher.semantic_matcher import SemanticMatcher

        with pytest.raises(ImportError) as exc_info:
            SemanticMatcher(project_root=".")

        assert "ChromaDB is not installed" in str(exc_info.value)
        assert "pip install chromadb" in str(exc_info.value)


def test_vector_store_import_error() -> None:
    """Test that VectorStore raises proper error when ChromaDB unavailable."""
    with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", False):
        from codedocsync.storage.vector_store import VectorStore

        with pytest.raises(ImportError) as exc_info:
            VectorStore()

        assert "ChromaDB is not installed" in str(exc_info.value)
