import logging
import time
from typing import Any

from ..parser import ParsedFunction
from ..storage.embedding_cache import EmbeddingCache
from ..storage.vector_store import VectorStore
from .embedding_generator import EmbeddingGenerator
from .models import MatchedPair, MatchResult, MatchType
from .semantic_models import EmbeddingConfig, FunctionEmbedding
from .semantic_scorer import SemanticScorer

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """
    Semantic similarity matching using embeddings.

    This is the final fallback when direct and contextual matching fail.
    Only handles ~2% of cases but critical for major refactorings.
    """

    def __init__(
        self, project_root: str, config: EmbeddingConfig | None = None
    ) -> None:
        self.project_root = project_root
        self.config = config or EmbeddingConfig()

        # Initialize components
        self.vector_store = VectorStore(project_id=None)  # Auto-generate from path
        self.embedding_cache = EmbeddingCache()
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.scorer = SemanticScorer()

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
        self, all_functions: list[ParsedFunction], force_reindex: bool = False
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

    async def match_with_embeddings(
        self,
        functions: list[ParsedFunction],
        previous_results: list[MatchResult] | None = None,
    ) -> MatchResult:
        """
        Perform semantic matching on functions.

        Args:
            functions: Functions to find matches for
            previous_results: Results from direct/contextual matching

        Returns:
            MatchResult with semantic matches added
        """
        start_time = time.time()
        matches = []
        unmatched_functions = []

        # Start with previous results if provided
        high_confidence_matches = set()
        if previous_results:
            for result in previous_results:
                for match in result.matched_pairs:
                    # Keep high confidence matches
                    if match.confidence.overall >= 0.7:
                        matches.append(match)
                        high_confidence_matches.add(match.function.signature.name)

        # Process only functions without good matches
        functions_to_process = [
            f for f in functions if f.signature.name not in high_confidence_matches
        ]

        logger.info(
            f"Semantic matching for {len(functions_to_process)} functions "
            f"(skipping {len(high_confidence_matches)} with good matches)"
        )

        # Perform semantic search for each function
        for function in functions_to_process:
            semantic_match = await self._find_semantic_match(function)

            if semantic_match:
                matches.append(semantic_match)
                self.stats["semantic_matches_found"] += 1
            else:
                unmatched_functions.append(function)

        # Update stats
        self.stats["functions_processed"] += len(functions_to_process)
        self.stats["total_time"] += time.time() - start_time

        # Build result
        return MatchResult(
            total_functions=len(functions),
            matched_pairs=matches,
            unmatched_functions=unmatched_functions,
        )

    async def _find_semantic_match(
        self, function: ParsedFunction
    ) -> MatchedPair | None:
        """Find semantic match for a single function."""
        try:
            # Generate embedding for query function
            text = self.embedding_generator.prepare_function_text(function)

            # Try cache first
            signature_hash = self.embedding_generator.generate_signature_hash(function)
            cached_embedding = self.embedding_cache.get(
                text, self.config.primary_model.value, signature_hash
            )

            if cached_embedding:
                query_embedding = cached_embedding.embedding
            else:
                # Generate new embedding
                embeddings = (
                    await self.embedding_generator.generate_function_embeddings(
                        [function], use_cache=True
                    )
                )
                if not embeddings:
                    logger.warning(
                        f"Failed to generate embedding for {function.signature.name}"
                    )
                    return None

                query_embedding = embeddings[0].embedding
                # Cache it
                self.embedding_cache.set(embeddings[0])

            # Search for similar functions
            similar_items = self.vector_store.search_similar(
                query_embedding,
                n_results=10,
                min_similarity=0.65,  # Get top 10 candidates
            )
            self.stats["searches_performed"] += 1

            if not similar_items:
                return None

            # Score and validate matches
            best_match = None
            best_score = 0.0

            for item_id, similarity, metadata in similar_items:
                # Skip self-matches
                if metadata[
                    "function_id"
                ] == self.embedding_generator.generate_function_id(function):
                    continue

                # Validate match with additional checks
                is_valid, adjusted_score = self.scorer.validate_semantic_match(
                    function, metadata["function_id"], similarity
                )

                if is_valid and adjusted_score > best_score:
                    best_score = adjusted_score
                    best_match = (item_id, adjusted_score, metadata)

            # Create match if found
            if best_match and best_score >= 0.65:
                return self._create_semantic_match(
                    function,
                    best_match[0],  # matched function ID
                    best_match[1],  # score
                    best_match[2],  # metadata
                )

            return None

        except Exception as e:
            logger.error(f"Semantic matching failed for {function.signature.name}: {e}")
            return None

    def _create_semantic_match(
        self,
        source_function: ParsedFunction,
        matched_function_id: str,
        similarity_score: float,
        metadata: dict[str, str],
    ) -> MatchedPair:
        """Create a MatchedPair for semantic match."""
        # In a real implementation, we'd load the matched function
        # For now, we'll create a placeholder match

        confidence = self.scorer.calculate_semantic_confidence(
            similarity_score, source_function
        )

        return MatchedPair(
            function=source_function,
            docstring=source_function.docstring,  # Will be updated when we load actual match
            match_type=MatchType.SEMANTIC,
            confidence=confidence,
            match_reason=f"Semantic similarity match with {matched_function_id} (score: {similarity_score:.2f})",
        )

    def get_stats(self) -> dict[str, Any]:
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
            return bool(stats.get("collection_count", 0) > 0)
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
    ) -> FunctionEmbedding | None:
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
