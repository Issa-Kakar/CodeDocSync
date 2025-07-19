import time
import logging
from typing import List, Optional, Dict, Any

from ..parser import ParsedFunction
from ..storage.vector_store import VectorStore
from ..storage.embedding_cache import EmbeddingCache
from .semantic_models import EmbeddingConfig, FunctionEmbedding
from .embedding_generator import EmbeddingGenerator
from .models import MatchResult, MatchedPair, MatchType, MatchConfidence

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """
    Semantic similarity matching using embeddings.

    This is the final fallback when direct and contextual matching fail.
    Only handles ~2% of cases but critical for major refactorings.

    Note: This is Part 1 implementation focusing on index preparation.
    The complete similarity search will be implemented in Part 2.
    """

    def __init__(self, project_root: str, config: Optional[EmbeddingConfig] = None):
        self.project_root = project_root
        self.config = config or EmbeddingConfig()

        # Initialize components
        self.vector_store = VectorStore(project_id=None)  # Auto-generate from path
        self.embedding_cache = EmbeddingCache()
        self.embedding_generator = EmbeddingGenerator(self.config)

        # Performance tracking
        self.stats = {
            "functions_processed": 0,
            "embeddings_generated": 0,
            "searches_performed": 0,
            "semantic_matches_found": 0,
            "total_time": 0.0,
            "index_preparation_time": 0.0,
        }

    async def prepare_semantic_index(
        self, all_functions: List[ParsedFunction], force_reindex: bool = False
    ) -> None:
        """
        Prepare the semantic search index with all functions.

        This should be called once before matching to build the index.
        This method processes all functions and creates embeddings for semantic search.

        Args:
            all_functions: List of functions to index
            force_reindex: If True, regenerate all embeddings even if cached
        """
        start_time = time.time()
        logger.info(f"Preparing semantic index for {len(all_functions)} functions")

        # Generate embeddings for all functions
        embeddings_needed = []
        embeddings_cached = []

        for function in all_functions:
            # Check cache first if not forcing reindex
            text = self.embedding_generator.prepare_function_text(function)
            signature_hash = self.embedding_generator.generate_signature_hash(function)

            if not force_reindex:
                cached = self.embedding_cache.get(
                    text, self.config.primary_model.value, signature_hash
                )
                if cached:
                    embeddings_cached.append(cached)
                    continue

            embeddings_needed.append(function)

        logger.info(
            f"Found {len(embeddings_cached)} cached embeddings, "
            f"need to generate {len(embeddings_needed)}"
        )

        # Generate missing embeddings
        new_embeddings = []
        if embeddings_needed:
            new_embeddings = (
                await self.embedding_generator.generate_function_embeddings(
                    embeddings_needed, use_cache=True
                )
            )

            # Cache the new embeddings
            for embedding in new_embeddings:
                self.embedding_cache.set(embedding)

        # Combine all embeddings
        all_embeddings = embeddings_cached + new_embeddings

        # Store in vector database
        if all_embeddings:
            # Prepare for batch insert
            ids = [e.function_id for e in all_embeddings]
            vectors = [e.embedding for e in all_embeddings]
            metadatas = [
                {
                    "function_id": e.function_id,
                    "model": e.model,
                    "signature_hash": e.signature_hash,
                    "timestamp": str(e.timestamp),
                }
                for e in all_embeddings
            ]

            self.vector_store.add_embeddings(vectors, metadatas, ids)
            logger.info(f"Added {len(all_embeddings)} embeddings to vector store")

        # Update stats
        self.stats["embeddings_generated"] += len(new_embeddings)
        index_time = time.time() - start_time
        self.stats["index_preparation_time"] = index_time
        self.stats["total_time"] += index_time

        logger.info(f"Semantic index prepared in {index_time:.2f}s")

    def create_placeholder_match_result(
        self,
        functions: List[ParsedFunction],
        previous_results: Optional[List[MatchResult]] = None,
    ) -> MatchResult:
        """
        Create a placeholder match result for Part 1 implementation.

        This method provides the interface that will be completed in Part 2.
        For now, it returns the previous results or an empty result.

        Args:
            functions: Functions to find matches for
            previous_results: Results from direct/contextual matching

        Returns:
            MatchResult with existing matches preserved
        """
        logger.info("Creating placeholder match result (Part 1 implementation)")

        # Start with previous results if provided
        all_matches = []
        if previous_results:
            for result in previous_results:
                all_matches.extend(result.matched_pairs)

        # Create result preserving existing matches
        return MatchResult(
            total_functions=len(functions),
            total_docs=len(functions),  # Assuming integrated parsing
            matched_pairs=all_matches,
            unmatched_functions=[
                f
                for f in functions
                if not any(
                    m.function.signature.name == f.signature.name for m in all_matches
                )
            ],
            unmatched_docs=[],
        )

    def _create_semantic_match_placeholder(
        self,
        source_function: ParsedFunction,
        matched_function_id: str,
        similarity_score: float,
        metadata: Dict[str, str],
    ) -> MatchedPair:
        """
        Create a placeholder MatchedPair for semantic match.

        This will be fully implemented in Part 2 with actual matching logic.
        """
        # Create basic confidence (will be enhanced in Part 2)
        confidence = MatchConfidence(
            overall=similarity_score,
            name_similarity=similarity_score,
            location_score=0.5,  # Unknown location relationship
            signature_similarity=0.7,  # Assumed similar based on embedding
        )

        return MatchedPair(
            function=source_function,
            docstring=source_function.docstring,
            match_type=MatchType.SEMANTIC,
            confidence=confidence,
            match_reason=f"Semantic similarity match with {matched_function_id} (score: {similarity_score:.2f})",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get matcher statistics."""
        avg_time = self.stats["total_time"] / max(self.stats["functions_processed"], 1)

        match_rate = self.stats["semantic_matches_found"] / max(
            self.stats["functions_processed"], 1
        )

        return {
            "functions_processed": self.stats["functions_processed"],
            "semantic_matches_found": self.stats["semantic_matches_found"],
            "match_rate": match_rate,
            "average_time_per_function_ms": avg_time * 1000,
            "index_preparation_time_s": self.stats["index_preparation_time"],
            "embedding_stats": self.embedding_generator.get_stats(),
            "cache_stats": self.embedding_cache.get_stats(),
            "vector_store_stats": self.vector_store.get_stats(),
        }

    def is_ready_for_matching(self) -> bool:
        """
        Check if the semantic matcher is ready for matching operations.

        Returns:
            True if index is prepared and components are initialized
        """
        try:
            # Check if vector store has embeddings
            stats = self.vector_store.get_stats()
            return stats.get("collection_count", 0) > 0
        except Exception as e:
            logger.error(f"Failed to check readiness: {e}")
            return False

    def clear_index(self) -> None:
        """Clear the semantic index for re-indexing."""
        try:
            # Clear vector store (this would need implementation in VectorStore)
            # For now, log the intention
            logger.info("Clearing semantic index (placeholder implementation)")

            # Reset stats
            self.stats["embeddings_generated"] = 0
            self.stats["index_preparation_time"] = 0.0

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")

    def get_embedding_for_function(
        self, function: ParsedFunction
    ) -> Optional[FunctionEmbedding]:
        """
        Get cached embedding for a specific function.

        Args:
            function: Function to get embedding for

        Returns:
            FunctionEmbedding if cached, None otherwise
        """
        try:
            text = self.embedding_generator.prepare_function_text(function)
            signature_hash = self.embedding_generator.generate_signature_hash(function)

            return self.embedding_cache.get(
                text, self.config.primary_model.value, signature_hash
            )
        except Exception as e:
            logger.error(f"Failed to get embedding for {function.signature.name}: {e}")
            return None
