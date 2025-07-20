import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codedocsync.storage.vector_store import VectorStore


class TestVectorStore:
    """Test the VectorStore class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB to avoid actual database operations in tests."""
        with patch("codedocsync.storage.vector_store.chromadb") as mock_chromadb:
            # Mock client
            mock_client = Mock()
            mock_chromadb.PersistentClient.return_value = mock_client

            # Mock collection
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = mock_collection

            yield mock_chromadb, mock_client, mock_collection

    def test_vector_store_initialization(self, temp_cache_dir, mock_chromadb):
        """Test VectorStore initialization."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        store = VectorStore(cache_dir=temp_cache_dir, project_id="test_project")

        # Check that cache directory was created
        assert Path(temp_cache_dir).exists()

        # Check project ID
        assert store.project_id == "test_project"
        assert store.collection_name == "functions_test_project"

        # Check that ChromaDB client was initialized
        mock_chromadb_module.PersistentClient.assert_called_once()

        # Check that collection was created (since get_collection should fail)
        mock_client.create_collection.assert_called_once()

    def test_project_id_generation(self, temp_cache_dir, mock_chromadb):
        """Test automatic project ID generation."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        store = VectorStore(cache_dir=temp_cache_dir)

        # Should generate a project ID based on current directory
        assert store.project_id is not None
        assert len(store.project_id) == 32  # MD5 hash length
        assert store.collection_name.startswith("functions_")

    def test_add_embeddings(self, temp_cache_dir, mock_chromadb):
        """Test adding embeddings to the vector store."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        store = VectorStore(cache_dir=temp_cache_dir)

        # Test data
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [
            {"function_id": "func1", "model": "test"},
            {"function_id": "func2", "model": "test"},
        ]
        ids = ["id1", "id2"]

        # Add embeddings
        store.add_embeddings(embeddings, metadatas, ids)

        # Check that collection.add was called
        mock_collection.add.assert_called_once()

        # Check that metadata was updated with timestamps
        call_args = mock_collection.add.call_args
        assert "embeddings" in call_args.kwargs
        assert "metadatas" in call_args.kwargs
        assert "ids" in call_args.kwargs

        # Check timestamp was added
        added_metadata = call_args.kwargs["metadatas"]
        for metadata in added_metadata:
            assert "indexed_at" in metadata

        # Check metrics updated
        assert store.metrics["embeddings_stored"] == 2

    def test_add_embeddings_error_handling(self, temp_cache_dir, mock_chromadb):
        """Test error handling in add_embeddings."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb
        mock_collection.add.side_effect = Exception("Database error")

        store = VectorStore(cache_dir=temp_cache_dir)

        embeddings = [[0.1, 0.2, 0.3]]
        metadatas = [{"function_id": "func1"}]
        ids = ["id1"]

        with pytest.raises(ValueError, match="Failed to store embeddings"):
            store.add_embeddings(embeddings, metadatas, ids)

    def test_search_similar(self, temp_cache_dir, mock_chromadb):
        """Test similarity search."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.3]],  # Distances (will be converted to similarity)
            "metadatas": [
                [
                    {"function_id": "func1", "model": "test"},
                    {"function_id": "func2", "model": "test"},
                ]
            ],
        }

        store = VectorStore(cache_dir=temp_cache_dir)

        query_embedding = [0.1, 0.2, 0.3]
        results = store.search_similar(query_embedding, n_results=5, min_similarity=0.6)

        # Check that query was called
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding], n_results=5
        )

        # Check results format
        assert len(results) == 2  # Both results above min_similarity (0.9, 0.7)

        # Check first result
        id_, similarity, metadata = results[0]
        assert id_ == "id1"
        assert similarity == 0.9  # 1 - 0.1
        assert metadata["function_id"] == "func1"

        # Check metrics updated
        assert store.metrics["searches_performed"] == 1
        assert store.metrics["total_search_time"] >= 0

    def test_search_similar_with_filtering(self, temp_cache_dir, mock_chromadb):
        """Test similarity search with min_similarity filtering."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        # Mock search results with varying similarities
        mock_collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "distances": [[0.1, 0.3, 0.5]],  # Similarities: 0.9, 0.7, 0.5
            "metadatas": [
                [
                    {"function_id": "func1"},
                    {"function_id": "func2"},
                    {"function_id": "func3"},
                ]
            ],
        }

        store = VectorStore(cache_dir=temp_cache_dir)

        # Search with min_similarity = 0.65
        results = store.search_similar([0.1, 0.2], min_similarity=0.65)

        # Should only return first two results (0.9 and 0.7 >= 0.65)
        assert len(results) == 2
        assert results[0][1] == 0.9  # First similarity
        assert results[1][1] == 0.7  # Second similarity

    def test_search_similar_error_handling(self, temp_cache_dir, mock_chromadb):
        """Test error handling in search_similar."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb
        mock_collection.query.side_effect = Exception("Search error")

        store = VectorStore(cache_dir=temp_cache_dir)

        # Should return empty list on error
        results = store.search_similar([0.1, 0.2])
        assert results == []

        # Metrics should still be updated
        assert store.metrics["searches_performed"] == 1

    def test_get_by_id(self, temp_cache_dir, mock_chromadb):
        """Test getting embedding by ID."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        # Mock get result
        mock_collection.get.return_value = {
            "ids": ["test_id"],
            "embeddings": [[0.1, 0.2, 0.3]],
            "metadatas": [{"function_id": "test_func"}],
        }

        store = VectorStore(cache_dir=temp_cache_dir)

        result = store.get_by_id("test_id")

        # Check that get was called
        mock_collection.get.assert_called_once_with(ids=["test_id"])

        # Check result
        assert result is not None
        embedding, metadata = result
        assert embedding == [0.1, 0.2, 0.3]
        assert metadata["function_id"] == "test_func"

        # Check metrics
        assert store.metrics["cache_hits"] == 1

    def test_get_by_id_not_found(self, temp_cache_dir, mock_chromadb):
        """Test getting embedding by ID when not found."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        # Mock empty result
        mock_collection.get.return_value = {"ids": []}

        store = VectorStore(cache_dir=temp_cache_dir)

        result = store.get_by_id("nonexistent_id")
        assert result is None
        assert store.metrics["cache_hits"] == 0

    def test_clear_old_embeddings(self, temp_cache_dir, mock_chromadb):
        """Test clearing old embeddings."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb

        # Mock old and new embeddings
        current_time = time.time()
        old_time = current_time - (40 * 24 * 60 * 60)  # 40 days ago

        mock_collection.get.return_value = {
            "ids": ["old_id", "new_id"],
            "metadatas": [
                {"indexed_at": str(old_time)},  # Old
                {"indexed_at": str(current_time)},  # New
            ],
        }

        store = VectorStore(cache_dir=temp_cache_dir)

        # Clear embeddings older than 30 days
        deleted_count = store.clear_old_embeddings(max_age_days=30)

        # Should delete one old embedding
        assert deleted_count == 1
        mock_collection.delete.assert_called_once_with(ids=["old_id"])

    def test_get_stats(self, temp_cache_dir, mock_chromadb):
        """Test getting performance statistics."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb
        mock_collection.count.return_value = 100

        store = VectorStore(cache_dir=temp_cache_dir)

        # Simulate some activity
        store.metrics["embeddings_stored"] = 50
        store.metrics["searches_performed"] = 10
        store.metrics["total_search_time"] = 0.5  # 500ms total
        store.metrics["cache_hits"] = 5

        stats = store.get_stats()

        assert stats["embeddings_stored"] == 50
        assert stats["searches_performed"] == 10
        assert stats["average_search_time_ms"] == 50.0  # 500ms / 10 searches
        assert stats["cache_hit_rate"] == 0.5  # 5 hits / 10 searches
        assert stats["collection_count"] == 100

    def test_get_stats_no_activity(self, temp_cache_dir, mock_chromadb):
        """Test getting stats with no activity."""
        mock_chromadb_module, mock_client, mock_collection = mock_chromadb
        mock_collection.count.return_value = 0

        store = VectorStore(cache_dir=temp_cache_dir)

        stats = store.get_stats()

        # Should not divide by zero
        assert stats["average_search_time_ms"] == 0
        assert stats["cache_hit_rate"] == 0
        assert stats["collection_count"] == 0
