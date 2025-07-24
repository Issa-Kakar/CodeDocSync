"""Common fixtures for storage module tests."""

import time
from unittest.mock import MagicMock, patch

import pytest

from codedocsync.matcher.semantic_models import FunctionEmbedding


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory."""
    return str(tmp_path / "test_cache")


@pytest.fixture
def mock_embedding():
    """Minimal FunctionEmbedding for tests."""
    return FunctionEmbedding(
        function_id="test_func_123",
        embedding=[0.1, 0.2, 0.3],  # Small embedding
        model="test-model",
        text_embedded="def test(): pass",
        signature_hash="abc123",
        timestamp=time.time(),
    )


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB module."""
    with patch("codedocsync.storage.vector_store.chromadb") as mock:
        # Set up basic mock behavior
        mock_client = MagicMock()
        mock_collection = MagicMock()

        mock.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_or_create_collection.return_value = mock_collection

        # Make chromadb available as an attribute for import checks
        mock.__version__ = "0.4.0"

        yield mock


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clean environment for each test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)


@pytest.fixture
def mock_psutil():
    """Mock psutil for memory monitoring tests."""
    with patch("codedocsync.storage.performance_monitor.psutil") as mock:
        # Set up basic mock behavior
        mock_process = MagicMock()
        mock.Process.return_value = mock_process

        # Mock memory info
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.memory_info.return_value = mock_memory_info

        yield mock
