"""
RAG Corpus Manager for self-improving documentation suggestions.

This module implements a Retrieval-Augmented Generation (RAG) system that learns
from high-quality docstring examples to provide better suggestions.
"""

import json
import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from ..parser import FunctionSignature, ParsedDocstring, ParsedFunction
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
    category: str | None = None  # Category for organization (async_patterns, etc.)
    similarity_score: float = 0.0  # Added to track similarity during retrieval
    issue_types: list[str] | None = None  # Track which issues were fixed
    original_issue: str | None = None  # The specific issue this suggestion addressed
    improvement_score: float | None = None  # Calculated improvement metric


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

        # Load persisted metrics first to get historical data
        self._load_persisted_metrics()

        # Load bootstrap corpus on initialization
        self._load_bootstrap_corpus()

        # Load accepted suggestions from disk
        self.load_accepted_suggestions()

        # Update examples_loaded count after loading all examples
        self._metrics["examples_loaded"] = len(self.memory_corpus)

        # Persist the updated metrics
        self._persist_metrics()

    def _load_bootstrap_corpus(self) -> None:
        """Load the bootstrap corpus from JSON files."""
        bootstrap_file = self.corpus_dir / "bootstrap_corpus.json"
        curated_file = self.corpus_dir / "curated_examples.json"

        loaded_count = 0
        curated_count = 0

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

        # Count is now updated in __init__ after all examples are loaded

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

            # Calculate improvement score based on acceptance
            improvement_score = 1.0  # Base score for being accepted

            # Bonus for fixing critical issues
            if issue_type in [
                "missing_docstring",
                "parameter_mismatch",
                "return_mismatch",
            ]:
                improvement_score += 0.2

            # Create example with improvement tracking
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
                issue_types=[issue_type],  # Track the issue type
                original_issue=issue_type,  # Store the specific issue
                improvement_score=improvement_score,  # Track improvement
            )

            # Store the example
            self._store_example(example)
            self._metrics["examples_added"] += 1

            logger.info(f"Added accepted suggestion for {example.function_name}")

            # Persist to disk
            self.save_accepted_suggestions()

        except Exception as e:
            logger.error(f"Failed to add accepted suggestion: {e}")

    def save_accepted_suggestions(self) -> None:
        """Persist accepted suggestions to disk with versioning."""
        accepted_path = self.corpus_dir / "accepted_suggestions.json"

        # Create backup if file exists
        if accepted_path.exists():
            backup_path = (
                self.corpus_dir / f"accepted_suggestions.{int(time.time())}.backup"
            )
            shutil.copy2(accepted_path, backup_path)
            self._cleanup_old_backups()

        # Filter accepted suggestions from memory corpus
        accepted_examples = [ex for ex in self.memory_corpus if ex.source == "accepted"]

        # Prepare data structure
        data = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "total_accepted": len(accepted_examples),
            "examples": [self._serialize_example(ex) for ex in accepted_examples],
        }

        # Atomic write
        temp_path = accepted_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(accepted_path)
            logger.info(f"Saved {len(accepted_examples)} accepted suggestions to disk")
        except Exception as e:
            logger.error(f"Failed to save accepted suggestions: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _serialize_example(self, example: DocstringExample) -> dict[str, Any]:
        """Serialize a DocstringExample to a dictionary."""
        # Use asdict but exclude None values to save space
        data = asdict(example)
        return {k: v for k, v in data.items() if v is not None}

    def _cleanup_old_backups(self) -> None:
        """Keep only the most recent 5 backup files."""
        backup_files = sorted(
            self.corpus_dir.glob("accepted_suggestions.*.backup"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep only the 5 most recent backups
        for backup_file in backup_files[5:]:
            try:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to remove backup {backup_file}: {e}")

    def load_accepted_suggestions(self) -> None:
        """Load accepted suggestions from disk with version compatibility."""
        accepted_path = self.corpus_dir / "accepted_suggestions.json"

        if not accepted_path.exists():
            logger.debug("No accepted suggestions file found")
            return

        try:
            with open(accepted_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check version compatibility
            version = data.get("version", "0.0.0")
            if not self._is_compatible_version(version):
                logger.warning(f"Incompatible accepted suggestions version: {version}")
                return

            # Load examples
            loaded_count = 0
            for ex_data in data.get("examples", []):
                try:
                    # Ensure compatibility with new fields
                    if "issue_types" not in ex_data and "original_issue" in ex_data:
                        ex_data["issue_types"] = [ex_data["original_issue"]]

                    example = DocstringExample(**ex_data)
                    example.source = "accepted"  # Ensure source is correct

                    # Store example
                    self._store_example(example)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load accepted example: {e}")

            logger.info(f"Loaded {loaded_count} accepted suggestions from disk")

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted accepted suggestions file: {e}")
            # Try to restore from backup
            self._restore_from_backup()
        except Exception as e:
            logger.error(f"Failed to load accepted suggestions: {e}")

    def _is_compatible_version(self, version: str) -> bool:
        """Check if the file version is compatible."""
        # For now, we support version 1.x.x
        try:
            major_version = int(version.split(".")[0])
            return major_version == 1
        except (ValueError, IndexError):
            return False

    def _restore_from_backup(self) -> None:
        """Attempt to restore from the most recent backup."""
        backup_files = sorted(
            self.corpus_dir.glob("accepted_suggestions.*.backup"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not backup_files:
            logger.warning("No backup files available for restoration")
            return

        for backup_file in backup_files:
            try:
                # Copy backup to main file
                accepted_path = self.corpus_dir / "accepted_suggestions.json"
                shutil.copy2(backup_file, accepted_path)
                logger.info(f"Restored from backup: {backup_file}")

                # Try loading again
                self.load_accepted_suggestions()
                return
            except Exception as e:
                logger.warning(f"Failed to restore from backup {backup_file}: {e}")

        logger.error("Failed to restore from any backup")

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
        self, query_func: ParsedFunction, example: DocstringExample
    ) -> float:
        """Calculate similarity using advanced heuristics."""

        # Initialize weights
        weights = {
            "name_similarity": 0.25,
            "signature_similarity": 0.20,
            "type_compatibility": 0.15,
            "complexity_match": 0.10,
            "module_context": 0.10,
            "recency_boost": 0.10,
            "quality_weight": 0.10,
        }

        scores = {}

        # 1. Enhanced name similarity with semantic understanding
        scores["name_similarity"] = self._calculate_semantic_name_similarity(
            query_func.signature.name, example.function_name
        )

        # 2. Signature similarity (parameter names and order)
        scores["signature_similarity"] = self._calculate_signature_similarity(
            query_func.signature, example.function_signature
        )

        # 3. Type compatibility scoring
        scores["type_compatibility"] = self._calculate_type_compatibility(
            query_func, example
        )

        # 4. Complexity matching (prefer similar complexity)
        complexity_diff = abs(
            self._calculate_complexity(query_func) - example.complexity_score
        )
        scores["complexity_match"] = 1.0 / (1.0 + complexity_diff * 0.5)

        # 5. Module context bonus
        if self._same_module_context(query_func, example):
            scores["module_context"] = 1.0
        elif self._related_module(query_func, example):
            scores["module_context"] = 0.5
        else:
            scores["module_context"] = 0.0

        # 6. Recency boost for accepted suggestions
        if example.source == "accepted":
            days_old = (time.time() - example.timestamp) / 86400
            scores["recency_boost"] = 1.0 / (1.0 + days_old * 0.1)  # Decay over time
        else:
            scores["recency_boost"] = 0.0

        # 7. Quality weighting
        scores["quality_weight"] = example.quality_score / 5.0

        # Calculate weighted sum
        total_score = sum(scores[key] * weights[key] for key in weights)

        # Apply boosts
        if example.source == "accepted":
            total_score *= 1.2  # 20% boost for accepted suggestions

        return min(total_score, 1.0)  # Cap at 1.0

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

    def _calculate_semantic_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate semantic similarity between function names."""
        if name1 == name2:
            return 1.0

        # Tokenize both names
        tokens1 = self._tokenize_name(name1)
        tokens2 = self._tokenize_name(name2)

        # Check for verb synonyms
        verb_synonyms = {
            "get": {"fetch", "retrieve", "obtain", "find", "load"},
            "set": {"update", "modify", "change", "assign"},
            "create": {"make", "build", "generate", "construct", "init"},
            "delete": {"remove", "drop", "destroy", "clear"},
            "check": {"verify", "validate", "test", "ensure"},
            "calculate": {"compute", "determine", "evaluate"},
            "parse": {"process", "analyze", "interpret"},
            "save": {"store", "persist", "write"},
            "load": {"read", "fetch", "get", "retrieve"},
        }

        # Check if first tokens are verb synonyms
        if tokens1 and tokens2:
            verb1 = tokens1[0]
            verb2 = tokens2[0]

            for base_verb, synonyms in verb_synonyms.items():
                if (
                    (verb1 == base_verb and verb2 in synonyms)
                    or (verb2 == base_verb and verb1 in synonyms)
                    or (verb1 in synonyms and verb2 in synonyms)
                ):
                    # High similarity for verb synonyms
                    remaining_similarity = self._calculate_token_similarity(
                        tokens1[1:], tokens2[1:]
                    )
                    return 0.8 + 0.2 * remaining_similarity

        # Fall back to fuzzy matching
        return fuzz.token_set_ratio(name1, name2) / 100.0

    def _tokenize_name(self, name: str) -> list[str]:
        """Tokenize function/variable name into words."""
        # Handle snake_case
        tokens = name.split("_")

        # Handle camelCase
        result = []
        for token in tokens:
            # Split on capital letters
            subtokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", token)
            if subtokens:
                result.extend(subtokens)
            else:
                result.append(token)

        return [t.lower() for t in result if t]

    def _calculate_token_similarity(
        self, tokens1: list[str], tokens2: list[str]
    ) -> float:
        """Calculate similarity between two token lists."""
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        # Calculate Jaccard similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _calculate_signature_similarity(
        self, sig1: FunctionSignature, sig2_str: str
    ) -> float:
        """Calculate similarity between function signatures."""
        # Parse the signature string
        sig2_info = self._parse_signature_string(sig2_str)
        if not sig2_info:
            return 0.0

        score = 0.0

        # Parameter count similarity (40%)
        param_diff = abs(len(sig1.parameters) - len(sig2_info.get("parameters", [])))
        score += 0.4 * (1.0 / (1.0 + param_diff * 0.3))

        # Parameter name overlap (30%)
        if sig1.parameters and sig2_info.get("parameters"):
            sig1_names = {p.name for p in sig1.parameters}
            sig2_names = {p["name"] for p in sig2_info["parameters"]}
            overlap = len(sig1_names & sig2_names)
            max_params = max(len(sig1_names), len(sig2_names))
            score += 0.3 * (overlap / max_params if max_params > 0 else 0)

        # Parameter order bonus (30%)
        if sig1.parameters and sig2_info.get("parameters"):
            order_matches = sum(
                1
                for i, p1 in enumerate(sig1.parameters)
                if i < len(sig2_info["parameters"])
                and p1.name == sig2_info["parameters"][i]["name"]
            )
            score += 0.3 * (order_matches / len(sig1.parameters))

        return score

    def _parse_signature_string(self, signature: str) -> dict[str, Any]:
        """Parse a function signature string into components."""
        try:
            # Extract function name and parameters
            match = re.match(
                r"(?:async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+))?", signature
            )
            if not match:
                return {}

            func_name = match.group(1)
            params_str = match.group(2)
            return_type = match.group(3)

            # Parse parameters
            parameters = []
            if params_str.strip():
                # Split by comma, handling nested brackets
                param_parts = self._split_parameters(params_str)
                for param in param_parts:
                    param_info = self._parse_parameter(param.strip())
                    if param_info:
                        parameters.append(param_info)

            return {
                "name": func_name,
                "parameters": parameters,
                "return_type": return_type.strip() if return_type else None,
            }
        except Exception:
            return {}

    def _split_parameters(self, params_str: str) -> list[str]:
        """Split parameter string handling nested brackets."""
        params = []
        current: list[str] = []
        depth = 0

        for char in params_str:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                params.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        if current:
            params.append("".join(current).strip())

        return params

    def _parse_parameter(self, param_str: str) -> dict[str, str] | None:
        """Parse a single parameter string."""
        # Handle various formats: name, name: type, name: type = default
        if ":" in param_str:
            name_part, rest = param_str.split(":", 1)
            name = name_part.strip()
            if "=" in rest:
                type_part, default = rest.split("=", 1)
                return {
                    "name": name,
                    "type": type_part.strip(),
                    "default": default.strip(),
                }
            else:
                return {"name": name, "type": rest.strip()}
        elif "=" in param_str:
            name, default = param_str.split("=", 1)
            return {"name": name.strip(), "default": default.strip()}
        else:
            return {"name": param_str.strip()} if param_str.strip() else None

    def _calculate_type_compatibility(
        self, query_func: ParsedFunction, example: DocstringExample
    ) -> float:
        """Calculate type compatibility between query function and example."""
        sig2_info = self._parse_signature_string(example.function_signature)
        if not sig2_info:
            return 0.0

        score = 0.0

        # Return type compatibility (50%)
        if query_func.signature.return_type:
            example_return = sig2_info.get("return_type")
            if example_return:
                if self._types_compatible(
                    query_func.signature.return_type, example_return
                ):
                    score += 0.5
                elif self._types_related(
                    query_func.signature.return_type, example_return
                ):
                    score += 0.25

        # Parameter type compatibility (50%)
        if query_func.signature.parameters and sig2_info.get("parameters"):
            param_scores = []

            for p1 in query_func.signature.parameters:
                if not p1.type_annotation:
                    continue

                # Find matching parameter by name
                for p2 in sig2_info["parameters"]:
                    if p1.name == p2["name"] and p2.get("type"):
                        if self._types_compatible(p1.type_annotation, p2["type"]):
                            param_scores.append(1.0)
                        elif self._types_related(p1.type_annotation, p2["type"]):
                            param_scores.append(0.5)
                        else:
                            param_scores.append(0.0)
                        break

            if param_scores:
                score += 0.5 * (sum(param_scores) / len(param_scores))

        return score

    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible."""
        type1 = self._normalize_type(type1)
        type2 = self._normalize_type(type2)

        if type1 == type2:
            return True

        # Common type equivalences
        equivalences = [
            {"str", "string"},
            {"int", "integer"},
            {"bool", "boolean"},
            {"dict", "dictionary", "mapping"},
            {"list", "array", "sequence"},
            {"float", "number", "double"},
            {"none", "null", "nonetype"},
        ]

        for equiv_set in equivalences:
            if type1 in equiv_set and type2 in equiv_set:
                return True

        # Handle Optional types
        if "| none" in type1 and "| none" in type2:
            base1 = type1.replace("| none", "").strip()
            base2 = type2.replace("| none", "").strip()
            return self._types_compatible(base1, base2)

        return False

    def _normalize_type(self, type_str: str) -> str:
        """Normalize a type string for comparison."""
        if not type_str:
            return ""

        # Remove whitespace and convert to lowercase
        normalized = type_str.strip().lower()

        # Remove quotes
        normalized = normalized.replace('"', "").replace("'", "")

        # Handle Optional notation
        if normalized.startswith("optional[") and normalized.endswith("]"):
            normalized = normalized[9:-1] + " | none"

        return normalized

    def _types_related(self, type1: str, type2: str) -> bool:
        """Check if two types are related but not identical."""
        type1 = self._normalize_type(type1)
        type2 = self._normalize_type(type2)

        # Check for subtype relationships
        relationships = [
            ("int", "float"),  # int is subtype of float
            ("list", "sequence"),
            ("dict", "mapping"),
            ("str", "any"),
            ("int", "any"),
            ("float", "any"),
        ]

        for t1, t2 in relationships:
            if (type1 == t1 and type2 == t2) or (type1 == t2 and type2 == t1):
                return True

        # Check for generic relationships
        base1 = type1.split("[")[0] if "[" in type1 else type1
        base2 = type2.split("[")[0] if "[" in type2 else type2

        return base1 == base2 and type1 != type2

    def _same_module_context(
        self, func: ParsedFunction, example: DocstringExample
    ) -> bool:
        """Check if function and example are from the same module."""
        func_module = (
            func.file_path.replace(".py", "").replace("/", ".").replace("\\", ".")
        )
        example_module = example.module_path

        # Normalize paths
        func_parts = func_module.split(".")
        example_parts = example_module.split(".") if example_module else []

        # Check for exact match or parent/child relationship
        if not func_parts or not example_parts:
            return False

        return (
            func_parts[: len(example_parts)] == example_parts
            or example_parts[: len(func_parts)] == func_parts
        )

    def _related_module(self, func: ParsedFunction, example: DocstringExample) -> bool:
        """Check if modules are related (e.g., same package)."""
        func_module = (
            func.file_path.replace(".py", "").replace("/", ".").replace("\\", ".")
        )
        example_module = example.module_path or ""

        # Check if they share a common package
        func_parts = func_module.split(".")
        example_parts = example_module.split(".")

        if len(func_parts) < 2 or len(example_parts) < 2:
            return False

        # Same top-level package
        return func_parts[0] == example_parts[0]

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
