"""Storage components for CodeDocSync."""

from .embedding_config import EmbeddingConfigManager

# ChromaDB is optional for testing
try:
    from .vector_store import VectorStore

    __all__ = [
        "VectorStore",
        "EmbeddingConfigManager",
    ]
except ImportError:
    __all__ = [
        "EmbeddingConfigManager",
    ]
