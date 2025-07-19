from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class EmbeddingModel(Enum):
    """Supported embedding models."""

    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    OPENAI_ADA = "text-embedding-ada-002"
    LOCAL_MINILM = "all-MiniLM-L6-v2"  # For local fallback


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    primary_model: EmbeddingModel = EmbeddingModel.OPENAI_SMALL
    fallback_models: List[EmbeddingModel] = field(
        default_factory=lambda: [EmbeddingModel.OPENAI_ADA, EmbeddingModel.LOCAL_MINILM]
    )
    batch_size: int = 100  # For batch processing
    max_retries: int = 3
    timeout_seconds: int = 30
    cache_embeddings: bool = True

    def __post_init__(self):
        if self.batch_size < 1 or self.batch_size > 2048:
            raise ValueError("Batch size must be between 1 and 2048")
        if self.timeout_seconds < 10:
            raise ValueError("Timeout must be at least 10 seconds")


@dataclass
class FunctionEmbedding:
    """Embedding for a function with metadata."""

    function_id: str  # Canonical function path
    embedding: List[float]
    model: str
    text_embedded: str  # What was actually embedded
    timestamp: float
    signature_hash: str  # For change detection

    def __post_init__(self):
        # Validate embedding dimensions based on model
        expected_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
        }

        if self.model in expected_dims:
            if len(self.embedding) != expected_dims[self.model]:
                raise ValueError(
                    f"Invalid embedding dimension for {self.model}: "
                    f"expected {expected_dims[self.model]}, got {len(self.embedding)}"
                )


@dataclass
class SemanticMatch:
    """A semantic similarity match between functions."""

    source_function: str  # Function looking for match
    matched_function: str  # Potentially matching function
    similarity_score: float  # 0-1 similarity
    embedding_model: str
    match_metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.similarity_score <= 1:
            raise ValueError("Similarity score must be between 0 and 1")


@dataclass
class SemanticSearchResult:
    """Results from semantic search operation."""

    query_function: str
    matches: List[SemanticMatch]
    search_time_ms: float
    total_candidates: int  # How many were searched

    def get_best_match(self) -> Optional[SemanticMatch]:
        """Get the highest scoring match."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.similarity_score)

    def filter_by_threshold(self, threshold: float) -> List[SemanticMatch]:
        """Get matches above similarity threshold."""
        return [m for m in self.matches if m.similarity_score >= threshold]
