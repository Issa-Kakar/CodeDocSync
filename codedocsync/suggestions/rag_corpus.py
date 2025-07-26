"""
RAG Corpus Manager for self-improving documentation suggestions.

This module implements a Retrieval-Augmented Generation (RAG) system that learns
from high-quality docstring examples to provide better suggestions.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..parser import ParsedDocstring, ParsedFunction
from ..storage.embedding_cache import EmbeddingCache
from ..storage.vector_store import VectorStore


class StorageError(Exception):
    """Error related to storage operations."""

    pass


logger = logging.getLogger(__name__)


@dataclass
class DocstringExample:
    """Represents a high-quality docstring example for the corpus."""

    function_name: str
    module_path: str
    function_signature: str
    docstring_format: str
    docstring_content: str
    has_params: bool
    has_returns: bool
    has_examples: bool
    complexity_score: int  # 1-5, based on function complexity
    quality_score: int  # 1-5, based on docstring completeness
    embedding_id: str | None = None
    source: str = "bootstrap"  # bootstrap, accepted, or good_example
    timestamp: float = 0.0
    category: str | None = None  # Optional category field for compatibility
    similarity_score: float | None = (
        None  # Optional similarity score for retrieval results
    )


class RAGCorpusManager:
    """Manages the RAG corpus for enhanced documentation suggestions."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_cache: EmbeddingCache | None = None,
        corpus_dir: str = "data",
        collection_name: str = "rag_corpus",
        enable_embeddings: bool = False,  # Disabled by default for graceful degradation
    ):
        """
        Initialize the RAG corpus manager.

        Args:
            vector_store: Vector store for similarity search. Creates new if None.
            embedding_cache: Cache for embeddings. Creates new if None.
            corpus_dir: Directory containing corpus JSON files.
            collection_name: Name for the ChromaDB collection.
        """
        self.corpus_dir = Path(corpus_dir)
        self.collection_name = collection_name
        self.enable_embeddings = enable_embeddings

        # Initialize vector store with custom collection only if embeddings enabled
        self.vector_store: VectorStore | None = None
        self.embedding_cache: EmbeddingCache | None = None

        if enable_embeddings:
            if vector_store:
                self.vector_store = vector_store
            else:
                try:
                    self.vector_store = VectorStore(
                        cache_dir=".codedocsync_cache", project_id=collection_name
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize vector store: {e}")
                    self.enable_embeddings = False

        # Initialize embedding cache only if embeddings enabled
        if enable_embeddings and self.vector_store:
            if embedding_cache:
                self.embedding_cache = embedding_cache
            else:
                try:
                    self.embedding_cache = EmbeddingCache()
                except Exception as e:
                    logger.warning(f"Failed to initialize embedding cache: {e}")
                    self.enable_embeddings = False

        # Store examples in memory as fallback
        self.memory_corpus: list[DocstringExample] = []

        # Track metrics
        self._metrics = {
            "examples_loaded": 0,
            "examples_added": 0,
            "retrievals_performed": 0,
            "searches_performed": 0,
            "examples_retrieved": 0,
            "total_retrieval_time": 0.0,
            "avg_retrieval_time_ms": 0.0,
        }

        # Load bootstrap corpus on initialization
        self._load_bootstrap_corpus()

        # Load persisted metrics
        self._load_persisted_metrics()

    def _load_bootstrap_corpus(self) -> None:
        """Load the bootstrap corpus from JSON files."""
        bootstrap_file = self.corpus_dir / "bootstrap_corpus.json"
        curated_file = self.corpus_dir / "curated_examples.json"

        loaded_count = 0

        # Load bootstrap corpus
        if bootstrap_file.exists():
            try:
                with open(bootstrap_file, encoding="utf-8") as f:
                    data = json.load(f)
                    examples = data.get("examples", [])

                for ex_data in examples:
                    example = DocstringExample(**ex_data)
                    if not example.timestamp:
                        example.timestamp = time.time()

                    # Store example
                    try:
                        self._store_example(example)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to store bootstrap example: {e}")

                logger.info(f"Loaded {loaded_count} bootstrap examples")

            except Exception as e:
                logger.error(f"Failed to load bootstrap corpus: {e}")

        # Load curated examples
        if curated_file.exists():
            try:
                with open(curated_file, encoding="utf-8") as f:
                    data = json.load(f)
                    examples = data.get("examples", [])

                curated_count = 0
                for ex_data in examples:
                    example = DocstringExample(**ex_data)
                    example.source = "curated"
                    if not example.timestamp:
                        example.timestamp = time.time()

                    try:
                        self._store_example(example)
                        curated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to store curated example: {e}")

                logger.info(f"Loaded {curated_count} curated examples")

            except Exception as e:
                logger.warning(f"No curated examples loaded: {e}")

        self._metrics["examples_loaded"] = loaded_count

    def _store_example(self, example: DocstringExample) -> None:
        """Store an example in the corpus."""
        # Always store in memory corpus
        self.memory_corpus.append(example)

        # Try to store in vector store if embeddings are enabled
        if self.enable_embeddings and self.vector_store and self.embedding_cache:
            try:
                # Create text representation for embedding
                # text_repr = self._create_text_representation(example)

                # Generate embedding - need to handle the actual method available
                # For now, skip embedding generation as it requires async/specific API
                logger.debug(
                    f"Embeddings disabled for {example.function_name} - using memory corpus only"
                )

                # Create unique ID
                example.embedding_id = (
                    f"{example.source}_{example.function_name}_{int(example.timestamp)}"
                )

            except Exception as e:
                logger.debug(
                    f"Failed to generate embedding for {example.function_name}: {e}"
                )
                # Continue with memory-only storage

    def _create_text_representation(self, example: DocstringExample) -> str:
        """Create text representation of example for embedding."""
        parts = [
            f"Function: {example.function_name}",
            f"Signature: {example.function_signature}",
            f"Format: {example.docstring_format}",
            f"Docstring: {example.docstring_content}",
        ]
        return "\n".join(parts)

    def add_good_example(
        self,
        function: ParsedFunction,
        docstring: ParsedDocstring,
        quality_score: int = 4,
    ) -> None:
        """
        Add a good example from analysis to the corpus.

        Args:
            function: The parsed function with good documentation.
            docstring: The parsed docstring.
            quality_score: Quality score for the example (1-5).
        """
        try:
            # Create example
            example = DocstringExample(
                function_name=function.signature.name,
                module_path=function.file_path,
                function_signature=str(function.signature),
                docstring_format=(
                    docstring.format.value
                    if hasattr(docstring.format, "value")
                    else str(docstring.format)
                ),
                docstring_content=(
                    function.docstring.raw_text if function.docstring else ""
                ),
                has_params=bool(docstring.parameters),
                has_returns=bool(docstring.returns),
                has_examples=bool(
                    hasattr(docstring, "examples") and docstring.examples
                ),
                complexity_score=self._calculate_complexity(function),
                quality_score=quality_score,
                source="good_example",
                timestamp=time.time(),
            )

            # Store the example
            self._store_example(example)
            self._metrics["examples_added"] += 1

            logger.debug(f"Added good example: {example.function_name}")

        except Exception as e:
            logger.error(f"Failed to add good example: {e}")

    def add_accepted_suggestion(
        self,
        function: ParsedFunction,
        suggested_docstring: str,
        docstring_format: str,
        issue_type: str,
    ) -> None:
        """
        Add an accepted suggestion to the corpus for learning.

        Args:
            function: The function that was documented.
            suggested_docstring: The accepted docstring content.
            docstring_format: The docstring format used.
            issue_type: The type of issue that was fixed.
        """
        try:
            # Parse the suggested docstring to extract features
            from ..parser.docstring_parser import DocstringParser

            parser = DocstringParser()
            parsed = parser.parse(suggested_docstring)

            if not parsed:
                logger.warning("Failed to parse accepted suggestion")
                return

            # Create example
            example = DocstringExample(
                function_name=function.signature.name,
                module_path=function.file_path,
                function_signature=str(function.signature),
                docstring_format=docstring_format,
                docstring_content=suggested_docstring,
                has_params=bool(parsed.parameters),
                has_returns=bool(parsed.returns),
                has_examples=bool(hasattr(parsed, "examples") and parsed.examples),
                complexity_score=self._calculate_complexity(function),
                quality_score=4,  # Accepted suggestions get quality 4
                source="accepted",
                timestamp=time.time(),
            )

            # Store the example
            self._store_example(example)
            self._metrics["examples_added"] += 1

            logger.info(f"Added accepted suggestion for {example.function_name}")

        except Exception as e:
            logger.error(f"Failed to add accepted suggestion: {e}")

    def retrieve_examples(
        self, function: ParsedFunction, n_results: int = 3, min_similarity: float = 0.7
    ) -> list[DocstringExample]:
        """
        Retrieve similar examples for a function.

        Args:
            function: The function to find examples for.
            n_results: Number of examples to retrieve.
            min_similarity: Minimum similarity score.

        Returns:
            List of similar DocstringExample objects.
        """
        start_time = time.time()

        try:
            logger.debug(
                f"Retrieving examples for function '{function.signature.name}' "
                f"(n_results={n_results}, min_similarity={min_similarity})"
            )

            # If embeddings are enabled and available, use vector search
            if self.enable_embeddings and self.vector_store and self.embedding_cache:
                # Create query text
                # query_text = f"Function: {function.signature.name}\nSignature: {str(function.signature)}"

                # For now, fall back to memory search as embedding generation requires specific API
                logger.debug("Using memory-based search for examples")

            # Use memory-based search (simple heuristics)
            scored_examples = []
            total_corpus_size = len(self.memory_corpus)
            logger.debug(
                f"Searching through {total_corpus_size} examples in memory corpus"
            )

            for example in self.memory_corpus:
                score = self._calculate_similarity_score(function, example)
                if score >= min_similarity:
                    scored_examples.append((example, score))

            logger.info(
                f"RAG: Searched {total_corpus_size} examples, found {len(scored_examples)} with similarity >= {min_similarity}"
            )

            # Sort by score and take top n
            scored_examples.sort(key=lambda x: x[1], reverse=True)

            # Return examples with scores attached
            results = []
            for example, score in scored_examples[:n_results]:
                # Create a copy with similarity score included
                example_with_score = DocstringExample(
                    function_name=example.function_name,
                    module_path=example.module_path,
                    function_signature=example.function_signature,
                    docstring_format=example.docstring_format,
                    docstring_content=example.docstring_content,
                    has_params=example.has_params,
                    has_returns=example.has_returns,
                    has_examples=example.has_examples,
                    complexity_score=example.complexity_score,
                    quality_score=example.quality_score,
                    embedding_id=example.embedding_id,
                    source=example.source,
                    timestamp=example.timestamp,
                    similarity_score=score,  # Include similarity score in the dataclass
                )
                results.append(example_with_score)

            # Update metrics
            retrieval_time = time.time() - start_time
            self._metrics["retrievals_performed"] += 1
            self._metrics["searches_performed"] += 1
            self._metrics["total_retrieval_time"] += retrieval_time
            if results:
                self._metrics["examples_retrieved"] = self._metrics.get(
                    "examples_retrieved", 0
                ) + len(results)

            # Persist metrics for cross-process visibility
            self._persist_metrics()

            if results:
                top_scores = [getattr(r, "similarity_score", 0) for r in results[:3]]
                logger.debug(
                    f"Retrieved {len(results)} examples in {retrieval_time * 1000:.1f}ms. "
                    f"Top scores: {', '.join(f'{s:.2f}' for s in top_scores)}"
                )
            else:
                logger.info(
                    f"RAG: No examples found with similarity >= {min_similarity} in {retrieval_time * 1000:.1f}ms"
                )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve examples: {e}")
            return []

    def _calculate_similarity_score(
        self, function: ParsedFunction, example: DocstringExample
    ) -> float:
        """Calculate similarity between a function and an example using heuristics."""
        score = 0.0

        # Name similarity (20% weight - reduced from 40%)
        name_similarity = self._string_similarity(
            function.signature.name, example.function_name
        )
        score += name_similarity * 0.2

        # Parameter count similarity (30% weight - increased from 20%)
        func_param_count = len(function.signature.parameters)
        # Parse parameter count from signature string
        example_param_count = self._extract_parameter_count(example.function_signature)
        param_diff = abs(func_param_count - example_param_count)
        param_score = 1.0 / (1.0 + param_diff * 0.5)
        score += param_score * 0.3

        # Return type presence (30% weight - increased from 20%)
        has_return = bool(function.signature.return_type)
        if has_return == example.has_returns:
            score += 0.3

        # Complexity similarity (20% weight - same)
        func_complexity = self._calculate_complexity(function)
        complexity_diff = abs(func_complexity - example.complexity_score)
        complexity_score = 1.0 / (1.0 + complexity_diff * 0.3)
        score += complexity_score * 0.2

        return score

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity based on common substrings."""
        s1 = s1.lower()
        s2 = s2.lower()

        if s1 == s2:
            return 1.0

        # Check for common prefixes/suffixes
        common_prefix_len = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                common_prefix_len += 1
            else:
                break

        # Normalize by average length
        avg_len = (len(s1) + len(s2)) / 2
        return common_prefix_len / avg_len if avg_len > 0 else 0.0

    def _reconstruct_example(
        self, id_: str, metadata: dict[str, str]
    ) -> DocstringExample | None:
        """Reconstruct a DocstringExample from stored metadata."""
        try:
            # Need to load the full docstring content from somewhere
            # For now, we'll create a partial example from metadata
            example = DocstringExample(
                function_name=metadata.get("function_name", ""),
                module_path=metadata.get("module_path", ""),
                function_signature="",  # Would need to store this
                docstring_format=metadata.get("docstring_format", "google"),
                docstring_content="",  # Would need to store this
                has_params=metadata.get("has_params", "False") == "True",
                has_returns=metadata.get("has_returns", "False") == "True",
                has_examples=metadata.get("has_examples", "False") == "True",
                complexity_score=int(metadata.get("complexity_score", "1")),
                quality_score=int(metadata.get("quality_score", "1")),
                embedding_id=id_,
                source=metadata.get("source", "unknown"),
                timestamp=float(metadata.get("timestamp", "0")),
            )
            return example

        except Exception as e:
            logger.error(f"Failed to reconstruct example: {e}")
            return None

    def _calculate_complexity(self, function: ParsedFunction) -> int:
        """Calculate complexity score for a function."""
        score = 1

        # Add points for parameters
        param_count = len(function.signature.parameters)
        if param_count >= 3:
            score += 1
        if param_count >= 5:
            score += 1

        # Add points for type annotations
        if all(p.type_annotation for p in function.signature.parameters):
            score += 1

        # Add point for return type
        if function.signature.return_type:
            score += 1

        return min(score, 5)

    def _extract_parameter_count(self, signature: str) -> int:
        """Extract parameter count from a function signature string."""
        # Handle edge cases
        if "(" not in signature or ")" not in signature:
            return 0

        # Extract the parameter section
        try:
            start = signature.index("(") + 1
            # Find the matching closing parenthesis, accounting for nested parentheses
            paren_depth = 1
            end = start
            while end < len(signature) and paren_depth > 0:
                if signature[end] == "(":
                    paren_depth += 1
                elif signature[end] == ")":
                    paren_depth -= 1
                end += 1

            params_str = signature[start : end - 1].strip()

            # Empty parameter list
            if not params_str:
                return 0

            # Count parameters by splitting on commas at the top level
            # (not inside brackets or parentheses)
            param_count = 1  # At least one parameter if not empty
            bracket_depth = 0
            paren_depth = 0

            for char in params_str:
                if char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                elif char == "," and bracket_depth == 0 and paren_depth == 0:
                    param_count += 1

            return param_count

        except (ValueError, IndexError):
            # If parsing fails, make a simple estimate
            return signature.count(",") + (
                1 if "(" in signature and ")" in signature else 0
            )

    def get_stats(self) -> dict[str, Any]:
        """Get corpus statistics."""
        avg_retrieval_time = (
            self._metrics["total_retrieval_time"]
            / self._metrics["retrievals_performed"]
            if self._metrics["retrievals_performed"] > 0
            else 0
        )

        # Get corpus size
        corpus_size = len(self.memory_corpus)

        stats: dict[str, Any] = {
            "corpus_size": corpus_size,
            "examples_loaded": self._metrics["examples_loaded"],
            "examples_added": self._metrics["examples_added"],
            "retrievals_performed": self._metrics["retrievals_performed"],
            "searches_performed": self._metrics.get("searches_performed", 0),
            "examples_retrieved": self._metrics.get("examples_retrieved", 0),
            "average_retrieval_time_ms": avg_retrieval_time * 1000,
            "embeddings_enabled": self.enable_embeddings,
        }

        # Add vector store stats if available
        vector_store_stats: dict[str, Any]
        if self.vector_store:
            try:
                vector_store_stats = self.vector_store.get_stats()
            except Exception:
                vector_store_stats = {"status": "unavailable"}
        else:
            vector_store_stats = {"status": "disabled"}

        stats["vector_store_stats"] = vector_store_stats

        return stats

    def _persist_metrics(self) -> None:
        """Save metrics to file for cross-process visibility."""
        metrics_file = self.corpus_dir / "rag_metrics.json"
        try:
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({**self._metrics, "last_updated": time.time()}, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to persist metrics: {e}")

    def _load_persisted_metrics(self) -> None:
        """Load metrics from file if available."""
        metrics_file = self.corpus_dir / "rag_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    saved_metrics = json.load(f)
                    # Only use if recent (within 24 hours)
                    if time.time() - saved_metrics.get("last_updated", 0) < 86400:
                        self._metrics.update(
                            {
                                k: v
                                for k, v in saved_metrics.items()
                                if k != "last_updated"
                            }
                        )
                        logger.debug("Loaded persisted metrics from file")
            except Exception as e:
                logger.debug(f"Failed to load persisted metrics: {e}")
                # Ignore errors, use default metrics
