"""Tests for vector store functionality."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codedocsync.storage.vector_store import VectorStore


class TestVectorStore:
    """Test cases for VectorStore."""

    def test_init_without_chromadb_raises(self):
        """Test that initialization raises ImportError when ChromaDB is not available."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", False):
            with pytest.raises(ImportError, match="chromadb is required"):
                VectorStore()

    def test_init_creates_collection(self, mock_chromadb, temp_cache_dir):
        """Test that initialization creates a new collection."""
        # Mock collection doesn't exist
        mock_chromadb.PersistentClient.return_value.get_collection.side_effect = (
            Exception("Not found")
        )

        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Should create cache directory
        assert Path(temp_cache_dir).exists()

        # Should create collection
        mock_chromadb.PersistentClient.return_value.create_collection.assert_called_once()
        assert store.collection_name.startswith("functions_")

        store.close()

    def test_init_loads_existing_collection(self, mock_chromadb, temp_cache_dir):
        """Test that initialization loads existing collection."""
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value.get_collection.return_value = (
            mock_collection
        )

        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Should load existing collection
        mock_chromadb.PersistentClient.return_value.get_collection.assert_called_once()
        mock_chromadb.PersistentClient.return_value.create_collection.assert_not_called()
        assert store.collection == mock_collection

        store.close()

    def test_add_embeddings_success(self, mock_chromadb, temp_cache_dir):
        """Test successfully adding embeddings."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [{"function": "func1"}, {"function": "func2"}]
        ids = ["id1", "id2"]

        store.add_embeddings(embeddings, metadatas, ids)

        # Should call collection.add
        store.collection.add.assert_called_once()
        call_args = store.collection.add.call_args[1]
        assert call_args["embeddings"] == embeddings
        assert call_args["ids"] == ids

        # Should add timestamp to metadata
        for metadata in call_args["metadatas"]:
            assert "indexed_at" in metadata

        # Should update metrics
        assert store.metrics["embeddings_stored"] == 2

        store.close()

    def test_add_embeddings_failure(self, mock_chromadb, temp_cache_dir):
        """Test handling failure when adding embeddings."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Mock failure
        store.collection.add.side_effect = Exception("Storage error")

        with pytest.raises(ValueError, match="Failed to store embeddings"):
            store.add_embeddings([[0.1, 0.2]], [{"func": "test"}], ["id1"])

        store.close()

    def test_search_similar_with_results(self, mock_chromadb, temp_cache_dir):
        """Test searching for similar embeddings with results."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Mock search results
        store.collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "distances": [[0.2, 0.5, 0.8]],  # ChromaDB distances
            "metadatas": [[{"func": "f1"}, {"func": "f2"}, {"func": "f3"}]],
        }

        results = store.search_similar([0.1, 0.2, 0.3], n_results=5, min_similarity=0.7)

        # Should return filtered results (distances 0.2 and 0.5 -> similarities 0.9 and 0.75)
        assert len(results) == 2
        assert results[0][0] == "id1"
        assert abs(results[0][1] - 0.9) < 0.01  # similarity ~0.9
        assert results[1][0] == "id2"
        assert abs(results[1][1] - 0.75) < 0.01  # similarity ~0.75

        # Should update metrics
        assert store.metrics["searches_performed"] == 1
        assert store.metrics["total_search_time"] > 0

        store.close()

    def test_search_similar_no_results(self, mock_chromadb, temp_cache_dir):
        """Test searching when no results match criteria."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Mock no results
        store.collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        results = store.search_similar([0.1, 0.2, 0.3])

        assert len(results) == 0
        assert store.metrics["searches_performed"] == 1

        store.close()

    def test_search_similar_filters_by_similarity(self, mock_chromadb, temp_cache_dir):
        """Test that search filters results by minimum similarity."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Mock results with varying distances
        store.collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "distances": [[0.1, 0.7, 1.5]],  # -> similarities ~0.95, ~0.65, ~0.25
            "metadatas": [[{"f": "1"}, {"f": "2"}, {"f": "3"}]],
        }

        # High threshold
        results = store.search_similar([0.1, 0.2], min_similarity=0.8)
        assert len(results) == 1
        assert results[0][0] == "id1"

        # Medium threshold
        results = store.search_similar([0.1, 0.2], min_similarity=0.6)
        assert len(results) == 2

        # Low threshold
        results = store.search_similar([0.1, 0.2], min_similarity=0.2)
        assert len(results) == 3

        store.close()

    def test_get_by_id_exists(self, mock_chromadb, temp_cache_dir):
        """Test getting embedding by ID when it exists."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Mock get result
        store.collection.get.return_value = {
            "ids": ["test_id"],
            "embeddings": [[0.1, 0.2, 0.3]],
            "metadatas": [{"function": "test_func"}],
        }

        result = store.get_by_id("test_id")

        assert result is not None
        embedding, metadata = result
        assert embedding == [0.1, 0.2, 0.3]
        assert metadata["function"] == "test_func"
        assert store.metrics["cache_hits"] == 1

        store.close()

    def test_get_by_id_not_exists(self, mock_chromadb, temp_cache_dir):
        """Test getting embedding by ID when it doesn't exist."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Mock empty result
        store.collection.get.return_value = {
            "ids": [],
            "embeddings": [],
            "metadatas": [],
        }

        result = store.get_by_id("nonexistent_id")

        assert result is None
        assert store.metrics["cache_hits"] == 0

        store.close()

    def test_clear_old_embeddings(self, mock_chromadb, temp_cache_dir):
        """Test clearing old embeddings."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        old_time = str(time.time() - 40 * 24 * 60 * 60)  # 40 days ago
        new_time = str(time.time() - 10 * 24 * 60 * 60)  # 10 days ago

        # Mock get all items
        store.collection.get.return_value = {
            "ids": ["old1", "old2", "new1"],
            "embeddings": [[0.1], [0.2], [0.3]],
            "metadatas": [
                {"indexed_at": old_time},
                {"indexed_at": old_time},
                {"indexed_at": new_time},
            ],
        }

        deleted = store.clear_old_embeddings(max_age_days=30)

        assert deleted == 2
        store.collection.delete.assert_called_once_with(ids=["old1", "old2"])

        store.close()

    def test_get_stats(self, mock_chromadb, temp_cache_dir):
        """Test getting statistics."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Set up metrics
        store.metrics = {
            "embeddings_stored": 100,
            "searches_performed": 50,
            "total_search_time": 5.0,
            "cache_hits": 10,
        }
        store.collection.count.return_value = 100

        stats = store.get_stats()

        assert stats["embeddings_stored"] == 100
        assert stats["searches_performed"] == 50
        assert stats["average_search_time_ms"] == 100.0  # 5.0 / 50 * 1000
        assert stats["cache_hit_rate"] == 0.2  # 10 / 50
        assert stats["collection_count"] == 100

        store.close()

    def test_close_deletes_collection(self, mock_chromadb, temp_cache_dir):
        """Test that close deletes the collection."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)
            collection_name = store.collection_name

        store.close()

        # Should delete collection and reset client
        mock_chromadb.PersistentClient.return_value.delete_collection.assert_called_once_with(
            collection_name
        )
        mock_chromadb.PersistentClient.return_value.reset.assert_called_once()

    def test_context_manager(self, mock_chromadb, temp_cache_dir):
        """Test using VectorStore as context manager."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            with VectorStore(cache_dir=temp_cache_dir) as store:
                # Should be able to use store
                assert store.collection is not None
                collection_name = store.collection_name

        # Should call close on exit
        mock_chromadb.PersistentClient.return_value.delete_collection.assert_called_once_with(
            collection_name
        )

    def test_generate_project_id_deterministic(self, mock_chromadb, temp_cache_dir):
        """Test that project ID generation is deterministic."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store1 = VectorStore(cache_dir=temp_cache_dir)
            id1 = store1.project_id
            store1.close()

            store2 = VectorStore(cache_dir=temp_cache_dir)
            id2 = store2.project_id
            store2.close()

        # Same directory should produce same ID
        assert id1 == id2

        # ID should be a valid hex string (MD5)
        assert len(id1) == 32
        assert all(c in "0123456789abcdef" for c in id1)

    def test_error_handling_in_operations(self, mock_chromadb, temp_cache_dir, caplog):
        """Test error handling in various operations."""
        with patch("codedocsync.storage.vector_store.CHROMADB_AVAILABLE", True):
            store = VectorStore(cache_dir=temp_cache_dir)

        # Test search failure handling
        store.collection.query.side_effect = Exception("Query error")
        results = store.search_similar([0.1, 0.2])
        assert results == []
        assert "Search failed" in caplog.text

        # Test get_by_id failure handling
        store.collection.get.side_effect = Exception("Get error")
        result = store.get_by_id("test_id")
        assert result is None
        assert "Failed to get embedding" in caplog.text

        store.close()
