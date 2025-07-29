import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment,misc]
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB collections for semantic search."""

    def __init__(
        self, cache_dir: str = ".codedocsync_cache", project_id: str | None = None
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required for VectorStore functionality")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.chroma_dir = self.cache_dir / "chroma"
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Generate project-specific collection name
        self.project_id = project_id or self._generate_project_id()
        self.collection_name = f"functions_{self.project_id[:16]}"

        # Initialize or get collection
        self.collection = self._init_collection()

        # Track performance metrics
        self.metrics = {
            "embeddings_stored": 0,
            "searches_performed": 0,
            "total_search_time": 0.0,
            "cache_hits": 0,
        }

    def _generate_project_id(self) -> str:
        """Generate unique project ID from current directory."""
        project_path = Path.cwd().absolute()
        return hashlib.md5(str(project_path).encode()).hexdigest()

    def _init_collection(self) -> Any:
        """Initialize or get existing collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"project_id": self.project_id, "created_at": time.time()},
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection

    def add_embeddings(
        self,
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        ids: list[str],
    ) -> None:
        """Add embeddings to the collection with metadata."""
        try:
            # Add timestamp to metadata
            for metadata in metadatas:
                metadata["indexed_at"] = str(time.time())

            self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

            self.metrics["embeddings_stored"] += len(embeddings)
            logger.debug(f"Added {len(embeddings)} embeddings to collection")

        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise ValueError(f"Failed to store embeddings: {e}") from e

    def search_similar(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        min_similarity: float = 0.65,
    ) -> list[tuple[str, float, dict[str, str]]]:
        """
        Search for similar embeddings.

        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        start_time = time.time()
        similar_items = []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            # Extract and filter results
            if results["ids"] and results["ids"][0]:
                for i, id_ in enumerate(results["ids"][0]):
                    # ChromaDB returns cosine distances in [0,2] range, convert to similarity [0,1]
                    distance = results["distances"][0][i]
                    similarity = 1 - (
                        distance / 2
                    )  # Proper cosine distance to similarity conversion

                    if similarity >= min_similarity:
                        metadata = results["metadatas"][0][i]
                        similar_items.append((id_, similarity, metadata))

        except Exception as e:
            logger.error(f"Search failed: {e}")

        # Update metrics regardless of success/failure
        self.metrics["searches_performed"] += 1
        self.metrics["total_search_time"] += time.time() - start_time

        return similar_items

    def get_by_id(self, function_id: str) -> tuple[list[float], dict[str, str]] | None:
        """Get specific embedding by ID."""
        try:
            result = self.collection.get(ids=[function_id])
            if result["ids"]:
                embedding = result["embeddings"][0]
                metadata = result["metadatas"][0]
                self.metrics["cache_hits"] += 1
                return embedding, metadata
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding {function_id}: {e}")
            return None

    def clear_old_embeddings(self, max_age_days: int = 30) -> int:
        """Remove embeddings older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        # Get all items
        all_items = self.collection.get()

        # Find old items
        old_ids = []
        for i, metadata in enumerate(all_items["metadatas"]):
            if float(metadata.get("indexed_at", 0)) < cutoff_time:
                old_ids.append(all_items["ids"][i])

        # Delete old items
        if old_ids:
            self.collection.delete(ids=old_ids)
            logger.info(f"Deleted {len(old_ids)} old embeddings")

        return len(old_ids)

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        avg_search_time = (
            self.metrics["total_search_time"] / self.metrics["searches_performed"]
            if self.metrics["searches_performed"] > 0
            else 0
        )

        return {
            "embeddings_stored": self.metrics["embeddings_stored"],
            "searches_performed": self.metrics["searches_performed"],
            "average_search_time_ms": avg_search_time * 1000,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / self.metrics["searches_performed"]
                if self.metrics["searches_performed"] > 0
                else 0
            ),
            "collection_count": self.collection.count(),
        }

    def close(self) -> None:
        """Close ChromaDB client and release resources."""
        try:
            # Reset the client to force cleanup (without deleting data)
            if hasattr(self, "client"):
                try:
                    # Force reset to clean up resources
                    self.client.reset()
                    logger.info("ChromaDB client reset successfully")
                except Exception as e:
                    logger.debug(f"Client reset failed: {e}")

            # Clear local references
            if hasattr(self, "collection"):
                self.collection = None

            # Note: The collection data is preserved in ChromaDB storage

            logger.info("VectorStore closed and resources released")
        except Exception as e:
            logger.error(f"Error closing VectorStore: {e}")

    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, "client") and self.client is not None:
                self.close()
        except Exception:
            # Suppress errors during garbage collection
            pass

    def __enter__(self) -> "VectorStore":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup resources."""
        self.close()
