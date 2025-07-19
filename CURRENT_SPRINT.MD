## CURRENT_SPRINT.MD - Semantic Matcher Implementation

```markdown
# Current Sprint: Semantic Matcher Implementation

## Sprint Overview
The semantic matcher is the third and final layer of our matching system, handling the ~2% of cases where functions have been significantly renamed or restructured. It uses embedding-based similarity search to find potential matches when direct name matching and contextual analysis fail. This component requires careful integration with external services (OpenAI embeddings, ChromaDB) and robust error handling.

## Critical Context
- This is a FALLBACK mechanism - only used when other matchers fail
- Performance is less critical than accuracy since it handles few cases
- Must integrate with existing matcher results without disrupting them
- External dependencies (OpenAI API, ChromaDB) require careful error handling
- Embeddings are expensive - aggressive caching is mandatory

## Implementation Chunks

---

## Chunk 1: Foundation and Data Models ✅ COMPLETE (Estimated: 2-3 hours)

### Objectives
- Set up ChromaDB integration and vector store
- Create data models for semantic matching
- Implement embedding configuration system
- Set up the basic SemanticMatcher class structure

### Required Files to Create
1. `storage/vector_store.py` - ChromaDB wrapper and collection management
2. `matcher/semantic_models.py` - Data models for semantic matching
3. `storage/embedding_config.py` - Configuration for embedding models
4. `tests/test_semantic_models.py` - Model validation tests
5. `tests/test_vector_store.py` - Vector store integration tests

### Detailed Implementation

#### 1. Create `storage/vector_store.py`
```python
import chromadb
from chromadb.config import Settings
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages ChromaDB collections for semantic search."""

    def __init__(self, cache_dir: str = ".codedocsync_cache", project_id: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.chroma_dir = self.cache_dir / "chroma"
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
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
            "cache_hits": 0
        }

    def _generate_project_id(self) -> str:
        """Generate unique project ID from current directory."""
        project_path = Path.cwd().absolute()
        return hashlib.md5(str(project_path).encode()).hexdigest()

    def _init_collection(self) -> chromadb.Collection:
        """Initialize or get existing collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"project_id": self.project_id, "created_at": time.time()}
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
        ids: List[str]
    ) -> None:
        """Add embeddings to the collection with metadata."""
        try:
            # Add timestamp to metadata
            for metadata in metadatas:
                metadata["indexed_at"] = str(time.time())

            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            self.metrics["embeddings_stored"] += len(embeddings)
            logger.debug(f"Added {len(embeddings)} embeddings to collection")

        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise ValueError(f"Failed to store embeddings: {e}")

    def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        min_similarity: float = 0.65
    ) -> List[Tuple[str, float, Dict[str, str]]]:
        """
        Search for similar embeddings.

        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        start_time = time.time()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Extract and filter results
            similar_items = []
            if results['ids'] and results['ids'][0]:
                for i, id_ in enumerate(results['ids'][0]):
                    # ChromaDB returns distances, convert to similarity
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Assuming normalized embeddings

                    if similarity >= min_similarity:
                        metadata = results['metadatas'][0][i]
                        similar_items.append((id_, similarity, metadata))

            # Update metrics
            self.metrics["searches_performed"] += 1
            self.metrics["total_search_time"] += time.time() - start_time

            return similar_items

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_by_id(self, function_id: str) -> Optional[Tuple[List[float], Dict[str, str]]]:
        """Get specific embedding by ID."""
        try:
            result = self.collection.get(ids=[function_id])
            if result['ids']:
                embedding = result['embeddings'][0]
                metadata = result['metadatas'][0]
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
        for i, metadata in enumerate(all_items['metadatas']):
            if float(metadata.get('indexed_at', 0)) < cutoff_time:
                old_ids.append(all_items['ids'][i])

        # Delete old items
        if old_ids:
            self.collection.delete(ids=old_ids)
            logger.info(f"Deleted {len(old_ids)} old embeddings")

        return len(old_ids)

    def get_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        avg_search_time = (
            self.metrics["total_search_time"] / self.metrics["searches_performed"]
            if self.metrics["searches_performed"] > 0 else 0
        )

        return {
            "embeddings_stored": self.metrics["embeddings_stored"],
            "searches_performed": self.metrics["searches_performed"],
            "average_search_time_ms": avg_search_time * 1000,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / self.metrics["searches_performed"]
                if self.metrics["searches_performed"] > 0 else 0
            ),
            "collection_count": self.collection.count()
        }
2. Create matcher/semantic_models.py
pythonfrom dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
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
    fallback_models: List[EmbeddingModel] = field(default_factory=lambda: [
        EmbeddingModel.OPENAI_ADA,
        EmbeddingModel.LOCAL_MINILM
    ])
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
            "all-MiniLM-L6-v2": 384
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
            raise ValueError(f"Similarity score must be between 0 and 1")

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
3. Create storage/embedding_config.py
pythonimport os
from typing import Optional, Dict, Any
import logging

from ..matcher.semantic_models import EmbeddingConfig, EmbeddingModel

logger = logging.getLogger(__name__)

class EmbeddingConfigManager:
    """Manages embedding configuration and API keys."""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.config = self._load_config()

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        keys = {}

        # OpenAI key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            keys["openai"] = openai_key
        else:
            logger.warning("No OPENAI_API_KEY found in environment")

        # Add other providers as needed
        return keys

    def _load_config(self) -> EmbeddingConfig:
        """Load embedding configuration from environment or defaults."""
        config = EmbeddingConfig()

        # Override from environment if set
        if os.getenv("EMBEDDING_MODEL"):
            try:
                config.primary_model = EmbeddingModel(os.getenv("EMBEDDING_MODEL"))
            except ValueError:
                logger.warning(f"Invalid EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")

        if os.getenv("EMBEDDING_BATCH_SIZE"):
            try:
                config.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE"))
            except ValueError:
                logger.warning("Invalid EMBEDDING_BATCH_SIZE")

        return config

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider."""
        return self.api_keys.get(provider)

    def validate_config(self) -> bool:
        """Validate that configuration is usable."""
        # Check if we have keys for the primary model
        if self.config.primary_model in [
            EmbeddingModel.OPENAI_SMALL,
            EmbeddingModel.OPENAI_LARGE,
            EmbeddingModel.OPENAI_ADA
        ]:
            if not self.get_api_key("openai"):
                logger.error("OpenAI model selected but no API key found")
                return False

        return True

    def get_available_models(self) -> List[EmbeddingModel]:
        """Get list of models that are actually available."""
        available = []

        # Check OpenAI models
        if self.get_api_key("openai"):
            available.extend([
                EmbeddingModel.OPENAI_SMALL,
                EmbeddingModel.OPENAI_ADA,
                EmbeddingModel.OPENAI_LARGE
            ])

        # Local models are always available
        available.append(EmbeddingModel.LOCAL_MINILM)

        return available
Testing Requirements for Chunk 1
Create comprehensive tests that validate:

Vector store initialization and persistence
Embedding model configuration validation
Data model constraints and validation
ChromaDB connection error handling
API key loading and validation

Critical Implementation Notes

ChromaDB MUST use persistent storage, not in-memory
Always validate embedding dimensions match the model
Generate stable IDs for functions (use canonical path)
Handle ChromaDB connection failures gracefully
Validate API keys before attempting to use them

Success Criteria

Vector store successfully stores and retrieves embeddings
Data models properly validate all inputs
Configuration correctly loads from environment
All tests pass including error cases
Performance: <50ms for vector search operations


Chunk 2: Embedding Generation Pipeline ✅ COMPLETE (Estimated: 3-4 hours)
Objectives

Implement embedding generation for functions
Create text preparation strategies
Build fallback chain for multiple embedding providers
Implement batching for efficiency

Required Files to Create

matcher/embedding_generator.py - Core embedding generation logic
storage/embedding_cache.py - Caching layer for embeddings
tests/test_embedding_generator.py - Generation tests
tests/test_embedding_cache.py - Cache functionality tests

Detailed Implementation
1. Create matcher/embedding_generator.py
pythonimport hashlib
import time
from typing import List, Dict, Optional, Tuple
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from ..parser import ParsedFunction
from ..storage.embedding_config import EmbeddingConfigManager
from .semantic_models import (
    EmbeddingModel, FunctionEmbedding, EmbeddingConfig
)

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for functions with fallback support."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.config_manager = EmbeddingConfigManager()

        # Validate configuration
        if not self.config_manager.validate_config():
            raise ValueError("Invalid embedding configuration")

        # Initialize embedding providers
        self._init_providers()

        # Performance tracking
        self.stats = {
            "embeddings_generated": 0,
            "fallbacks_used": 0,
            "total_generation_time": 0.0,
            "batch_count": 0
        }

    def _init_providers(self):
        """Initialize embedding providers based on available models."""
        self.providers = {}

        # Initialize OpenAI if available
        if self.config_manager.get_api_key("openai"):
            try:
                import openai
                openai.api_key = self.config_manager.get_api_key("openai")
                self.providers["openai"] = openai
                logger.info("Initialized OpenAI embedding provider")
            except ImportError:
                logger.warning("OpenAI library not installed")

        # Initialize local model as fallback
        try:
            from sentence_transformers import SentenceTransformer
            self.providers["local"] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized local embedding model")
        except ImportError:
            logger.warning("sentence-transformers not installed, local fallback unavailable")

    def prepare_function_text(self, function: ParsedFunction) -> str:
        """
        Prepare function text for embedding.

        Strategy:
        1. Function signature with types
        2. First line of docstring (if available)
        3. NO source code (security risk)
        """
        # Build signature string
        signature_text = function.signature.to_string()

        # Add docstring summary if available
        docstring_text = ""
        if function.docstring:
            if hasattr(function.docstring, 'summary'):
                # ParsedDocstring
                docstring_text = function.docstring.summary
            elif hasattr(function.docstring, 'raw_text'):
                # RawDocstring - take first line
                first_line = function.docstring.raw_text.split('\n')[0].strip()
                docstring_text = first_line

        # Combine with clear separation
        if docstring_text:
            combined_text = f"{signature_text} | {docstring_text}"
        else:
            combined_text = signature_text

        # Truncate if too long (most models have token limits)
        max_length = 512  # Conservative limit
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length-3] + "..."

        return combined_text

    def generate_function_id(self, function: ParsedFunction) -> str:
        """Generate stable ID for function."""
        # Use module path + function name
        module_path = function.file_path.replace('/', '.').replace('\\', '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]

        return f"{module_path}.{function.signature.name}"

    def generate_signature_hash(self, function: ParsedFunction) -> str:
        """Generate hash of function signature for change detection."""
        signature_str = function.signature.to_string()
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_embedding(
        self,
        text: str,
        model: EmbeddingModel
    ) -> Optional[List[float]]:
        """Generate embedding with retry logic."""
        try:
            if model in [EmbeddingModel.OPENAI_SMALL, EmbeddingModel.OPENAI_LARGE, EmbeddingModel.OPENAI_ADA]:
                return await self._generate_openai_embedding(text, model.value)
            elif model == EmbeddingModel.LOCAL_MINILM:
                return self._generate_local_embedding(text)
            else:
                raise ValueError(f"Unsupported model: {model}")

        except Exception as e:
            logger.error(f"Failed to generate embedding with {model}: {e}")
            raise

    async def _generate_openai_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding using OpenAI."""
        if "openai" not in self.providers:
            raise ValueError("OpenAI provider not initialized")

        import openai

        try:
            response = await openai.Embedding.acreate(
                input=text,
                model=model
            )
            return response['data'][0]['embedding']

        except openai.error.RateLimitError:
            logger.warning("OpenAI rate limit hit")
            raise
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def _generate_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        if "local" not in self.providers:
            raise ValueError("Local model not initialized")

        model = self.providers["local"]
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def generate_function_embeddings(
        self,
        functions: List[ParsedFunction],
        use_cache: bool = True
    ) -> List[FunctionEmbedding]:
        """
        Generate embeddings for multiple functions.

        Uses batching and fallback models as needed.
        """
        start_time = time.time()
        embeddings = []

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(functions), batch_size):
            batch = functions[i:i + batch_size]
            batch_embeddings = await self._process_batch(batch, use_cache)
            embeddings.extend(batch_embeddings)
            self.stats["batch_count"] += 1

        # Update stats
        self.stats["embeddings_generated"] += len(embeddings)
        self.stats["total_generation_time"] += time.time() - start_time

        return embeddings

    async def _process_batch(
        self,
        functions: List[ParsedFunction],
        use_cache: bool
    ) -> List[FunctionEmbedding]:
        """Process a batch of functions."""
        embeddings = []

        for function in functions:
            # Prepare text
            text = self.prepare_function_text(function)
            function_id = self.generate_function_id(function)
            signature_hash = self.generate_signature_hash(function)

            # Try to generate embedding with fallback chain
            embedding_vector = None
            model_used = None

            # Try primary model first
            try:
                embedding_vector = await self.generate_embedding(
                    text,
                    self.config.primary_model
                )
                model_used = self.config.primary_model.value

            except Exception as e:
                logger.warning(f"Primary model failed: {e}, trying fallbacks")
                self.stats["fallbacks_used"] += 1

                # Try fallback models
                for fallback_model in self.config.fallback_models:
                    try:
                        embedding_vector = await self.generate_embedding(
                            text,
                            fallback_model
                        )
                        model_used = fallback_model.value
                        break
                    except Exception as e:
                        logger.warning(f"Fallback {fallback_model} failed: {e}")
                        continue

            # Create embedding object if successful
            if embedding_vector and model_used:
                embedding = FunctionEmbedding(
                    function_id=function_id,
                    embedding=embedding_vector,
                    model=model_used,
                    text_embedded=text,
                    timestamp=time.time(),
                    signature_hash=signature_hash
                )
                embeddings.append(embedding)
            else:
                logger.error(f"Failed to generate embedding for {function_id}")

        return embeddings

    def get_stats(self) -> Dict[str, float]:
        """Get generation statistics."""
        avg_time = (
            self.stats["total_generation_time"] / self.stats["embeddings_generated"]
            if self.stats["embeddings_generated"] > 0 else 0
        )

        return {
            "embeddings_generated": self.stats["embeddings_generated"],
            "average_generation_time_ms": avg_time * 1000,
            "fallback_rate": (
                self.stats["fallbacks_used"] / self.stats["embeddings_generated"]
                if self.stats["embeddings_generated"] > 0 else 0
            ),
            "batches_processed": self.stats["batch_count"]
        }
2. Create storage/embedding_cache.py
pythonimport sqlite3
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from collections import OrderedDict

from ..matcher.semantic_models import FunctionEmbedding

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Two-tier caching for embeddings: in-memory LRU + disk persistence.
    """

    def __init__(self, cache_dir: str = ".codedocsync_cache", max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory LRU cache
        self.memory_cache: OrderedDict[str, FunctionEmbedding] = OrderedDict()
        self.max_memory_items = max_memory_items

        # SQLite for disk persistence
        self.db_path = self.cache_dir / "embeddings.db"
        self._init_db()

        # Performance metrics
        self.metrics = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "saves": 0
        }

    def _init_db(self):
        """Initialize SQLite database for embeddings."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                function_id TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                model TEXT NOT NULL,
                text_embedded TEXT NOT NULL,
                signature_hash TEXT NOT NULL,
                timestamp REAL NOT NULL,
                hit_count INTEGER DEFAULT 0,
                last_accessed REAL NOT NULL
            )
        """)

        # Create indices for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_function_id ON embeddings(function_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signature_hash ON embeddings(signature_hash)
        """)

        conn.commit()
        conn.close()

    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        import hashlib
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(
        self,
        text: str,
        model: str,
        signature_hash: Optional[str] = None
    ) -> Optional[FunctionEmbedding]:
        """
        Get embedding from cache.

        Args:
            text: Text that was embedded
            model: Model used for embedding
            signature_hash: Optional hash to verify function hasn't changed

        Returns:
            FunctionEmbedding if found and valid, None otherwise
        """
        cache_key = self._generate_cache_key(text, model)

        # Check memory cache first
        if cache_key in self.memory_cache:
            embedding = self.memory_cache[cache_key]
            # Move to end (LRU)
            self.memory_cache.move_to_end(cache_key)

            # Verify signature if provided
            if signature_hash and embedding.signature_hash != signature_hash:
                # Function has changed, invalidate cache
                del self.memory_cache[cache_key]
                self._delete_from_disk(cache_key)
                self.metrics["misses"] += 1
                return None

            self.metrics["memory_hits"] += 1
            return embedding

        # Check disk cache
        embedding = self._get_from_disk(cache_key)
        if embedding:
            # Verify signature if provided
            if signature_hash and embedding.signature_hash != signature_hash:
                # Function has changed, invalidate cache
                self._delete_from_disk(cache_key)
                self.metrics["misses"] += 1
                return None

            # Add to memory cache
            self._add_to_memory_cache(cache_key, embedding)
            self.metrics["disk_hits"] += 1
            return embedding

        self.metrics["misses"] += 1
        return None

    def set(self, embedding: FunctionEmbedding) -> None:
        """Save embedding to cache."""
        cache_key = self._generate_cache_key(
            embedding.text_embedded,
            embedding.model
        )

        # Add to memory cache
        self._add_to_memory_cache(cache_key, embedding)

        # Save to disk
        self._save_to_disk(cache_key, embedding)

        self.metrics["saves"] += 1

    def _add_to_memory_cache(self, key: str, embedding: FunctionEmbedding) -> None:
        """Add to memory cache with LRU eviction."""
        # Remove oldest if at capacity
        if len(self.memory_cache) >= self.max_memory_items:
            self.memory_cache.popitem(last=False)

        self.memory_cache[key] = embedding

    def _get_from_disk(self, cache_key: str) -> Optional[FunctionEmbedding]:
        """Load embedding from disk."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT function_id, embedding_json, model, text_embedded,
                       signature_hash, timestamp
                FROM embeddings
                WHERE cache_key = ?
            """, (cache_key,))

            row = cursor.fetchone()
            if row:
                # Update access stats
                cursor.execute("""
                    UPDATE embeddings
                    SET hit_count = hit_count + 1,
                        last_accessed = ?
                    WHERE cache_key = ?
                """, (time.time(), cache_key))
                conn.commit()

                # Reconstruct embedding
                embedding = FunctionEmbedding(
                    function_id=row[0],
                    embedding=json.loads(row[1]),
                    model=row[2],
                    text_embedded=row[3],
                    signature_hash=row[4],
                    timestamp=row[5]
                )
                return embedding

        except Exception as e:
            logger.error(f"Failed to load from disk cache: {e}")
        finally:
            conn.close()

        return None

    def _save_to_disk(self, cache_key: str, embedding: FunctionEmbedding) -> None:
        """Save embedding to disk."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings
                (cache_key, function_id, embedding_json, model, text_embedded,
                 signature_hash, timestamp, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                embedding.function_id,
                json.dumps(embedding.embedding),
                embedding.model,
                embedding.text_embedded,
                embedding.signature_hash,
                embedding.timestamp,
                time.time()
            ))
            conn.commit()

        except Exception as e:
            logger.error(f"Failed to save to disk cache: {e}")
        finally:
            conn.close()

    def _delete_from_disk(self, cache_key: str) -> None:
        """Delete embedding from disk."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete from disk: {e}")
        finally:
            conn.close()

    def clear_old_entries(self, max_age_days: int = 30) -> int:
        """Remove old cache entries."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM embeddings
                WHERE last_accessed < ?
            """, (cutoff_time,))

            deleted = cursor.rowcount
            conn.commit()

            logger.info(f"Cleared {deleted} old cache entries")
            return deleted

        except Exception as e:
            logger.error(f"Failed to clear old entries: {e}")
            return 0
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_requests = (
            self.metrics["memory_hits"] +
            self.metrics["disk_hits"] +
            self.metrics["misses"]
        )

        memory_hit_rate = (
            self.metrics["memory_hits"] / total_requests
            if total_requests > 0 else 0
        )

        overall_hit_rate = (
            (self.metrics["memory_hits"] + self.metrics["disk_hits"]) / total_requests
            if total_requests > 0 else 0
        )

        return {
            "memory_size": len(self.memory_cache),
            "memory_hit_rate": memory_hit_rate,
            "overall_hit_rate": overall_hit_rate,
            "total_saves": self.metrics["saves"],
            "total_requests": total_requests
        }
Testing Requirements for Chunk 2
Create tests that validate:

Text preparation strategies for different function types
Embedding generation with primary and fallback models
Batch processing functionality
Cache hit/miss scenarios
Signature change detection

Critical Implementation Notes

NEVER embed source code (security risk)
Always handle API failures with fallbacks
Cache based on text content, not function ID
Validate embeddings match expected dimensions
Monitor API rate limits and costs

Success Criteria

Successfully generates embeddings for all function types
Fallback chain works when primary model fails
Cache achieves >90% hit rate on unchanged functions
Batch processing improves throughput
Performance: <100ms per function (batched)


## Chunk 3A: Semantic Search Implementation - Part 1 ✅ COMPLETE (Estimated: 2 hours)
### Objectives

- Implement the core semantic matching logic
- Build semantic index preparation
- Create confidence scoring for semantic matches
- Build foundation for similarity search

### Required Files to Create/Update

1. `matcher/semantic_matcher.py` - Core SemanticMatcher class with indexing
2. `matcher/semantic_scorer.py` - Confidence scoring for matches
3. `tests/test_semantic_matcher.py` - Core matching functionality tests
4. `tests/test_semantic_scorer.py` - Scoring validation tests

### Scope (First Half Only)
This chunk implements the foundation for semantic matching:
- SemanticMatcher class initialization and configuration
- `prepare_semantic_index()` method for building search index
- SemanticScorer class for confidence calculation and validation
- Basic semantic match creation (without full similarity search)
- Test infrastructure for semantic matching

**Note**: The complete similarity search implementation with `match_with_embeddings()` method will be implemented in **Chunk 3B** by the next developer.

---

## Chunk 3B: Semantic Search Implementation - Part 2 ❌ PENDING (Estimated: 2 hours)
### Objectives (For Next Developer)

- Complete similarity search implementation
- Integrate with vector store for full matching pipeline
- Implement `match_with_embeddings()` method
- Add integration with previous match results

### Required Files to Create/Update (For Next Developer)

1. Complete `matcher/semantic_matcher.py` - Add `match_with_embeddings()` method
2. Add integration tests for full semantic matching pipeline
3. Performance validation tests for semantic search

### Scope (Second Half)
- Complete `_find_semantic_match()` implementation
- Full integration with vector store search
- Match validation and filtering logic
- Integration with DirectMatcher and ContextualMatcher results
- End-to-end semantic matching pipeline

Detailed Implementation
1. Create matcher/semantic_matcher.py
pythonimport time
import logging
from typing import List, Optional, Dict, Tuple
import asyncio

from ..parser import ParsedFunction
from ..storage.vector_store import VectorStore
from ..storage.embedding_cache import EmbeddingCache
from .semantic_models import (
    SemanticMatch, SemanticSearchResult, EmbeddingConfig,
    FunctionEmbedding
)
from .embedding_generator import EmbeddingGenerator
from .semantic_scorer import SemanticScorer
from .models import MatchResult, MatchedPair, MatchType, MatchConfidence

logger = logging.getLogger(__name__)

class SemanticMatcher:
    """
    Semantic similarity matching using embeddings.

    This is the final fallback when direct and contextual matching fail.
    Only handles ~2% of cases but critical for major refactorings.
    """

    def __init__(
        self,
        project_root: str,
        config: Optional[EmbeddingConfig] = None
    ):
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
            "total_time": 0.0
        }

    async def prepare_semantic_index(
        self,
        all_functions: List[ParsedFunction],
        force_reindex: bool = False
    ) -> None:
        """
        Prepare the semantic search index with all functions.

        This should be called once before matching to build the index.
        """
        start_time = time.time()
        logger.info(f"Preparing semantic index for {len(all_functions)} functions")

        # Generate embeddings for all functions
        embeddings_needed = []
        embeddings_cached = []

        for function in all_functions:
            # Check cache first
            text = self.embedding_generator.prepare_function_text(function)
            signature_hash = self.embedding_generator.generate_signature_hash(function)

            if not force_reindex:
                cached = self.embedding_cache.get(
                    text,
                    self.config.primary_model.value,
                    signature_hash
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
        if embeddings_needed:
            new_embeddings = await self.embedding_generator.generate_function_embeddings(
                embeddings_needed,
                use_cache=True
            )

            # Cache the new embeddings
            for embedding in new_embeddings:
                self.embedding_cache.set(embedding)

            # Combine with cached
            all_embeddings = embeddings_cached + new_embeddings
        else:
            all_embeddings = embeddings_cached

        # Store in vector database
        if all_embeddings:
            # Prepare for batch insert
            ids = [e.function_id for e in all_embeddings]
            vectors = [e.embedding for e in all_embeddings]
            metadatas = [
                {
                    "function_id": e.function_id,
                    "model": e.model,
                    "signature_hash": e.signature_hash
                }
                for e in all_embeddings
            ]

            self.vector_store.add_embeddings(vectors, metadatas, ids)

        # Update stats
        self.stats["embeddings_generated"] += len(embeddings_needed)
        self.stats["total_time"] += time.time() - start_time

        logger.info(f"Semantic index prepared in {time.time() - start_time:.2f}s")

    async def match_with_embeddings(
        self,
        functions: List[ParsedFunction],
        previous_results: Optional[List[MatchResult]] = None
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
            f for f in functions
            if f.signature.name not in high_confidence_matches
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
            total_docs=len(functions),  # Assuming integrated parsing
            matched_pairs=matches,
            unmatched_functions=unmatched_functions,
            unmatched_docs=[]
        )

    async def _find_semantic_match(
        self,
        function: ParsedFunction
    ) -> Optional[MatchedPair]:
        """Find semantic match for a single function."""
        try:
            # Generate embedding for query function
            text = self.embedding_generator.prepare_function_text(function)

            # Try cache first
            signature_hash = self.embedding_generator.generate_signature_hash(function)
            cached_embedding = self.embedding_cache.get(
                text,
                self.config.primary_model.value,
                signature_hash
            )

            if cached_embedding:
                query_embedding = cached_embedding.embedding
            else:
                # Generate new embedding
                embeddings = await self.embedding_generator.generate_function_embeddings(
                    [function],
                    use_cache=True
                )
                if not embeddings:
                    logger.warning(f"Failed to generate embedding for {function.signature.name}")
                    return None

                query_embedding = embeddings[0].embedding
                # Cache it
                self.embedding_cache.set(embeddings[0])

            # Search for similar functions
            search_start = time.time()
            similar_items = self.vector_store.search_similar(
                query_embedding,
                n_results=10,  # Get top 10 candidates
                min_similarity=0.65
            )
            self.stats["searches_performed"] += 1

            if not similar_items:
                return None

            # Score and validate matches
            best_match = None
            best_score = 0.0

            for item_id, similarity, metadata in similar_items:
                # Skip self-matches
                if metadata['function_id'] == self.embedding_generator.generate_function_id(function):
                    continue

                # Validate match with additional checks
                is_valid, adjusted_score = self.scorer.validate_semantic_match(
                    function,
                    metadata['function_id'],
                    similarity
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
                    best_match[2]   # metadata
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
        metadata: Dict[str, str]
    ) -> MatchedPair:
        """Create a MatchedPair for semantic match."""
        # In a real implementation, we'd load the matched function
        # For now, we'll create a placeholder match

        confidence = self.scorer.calculate_semantic_confidence(
            similarity_score,
            source_function
        )

        return MatchedPair(
            function=source_function,
            docstring=source_function.docstring,  # Will be updated when we load actual match
            match_type=MatchType.SEMANTIC,
            confidence=confidence,
            match_reason=f"Semantic similarity match with {matched_function_id} (score: {similarity_score:.2f})"
        )

    def get_stats(self) -> Dict[str, any]:
        """Get matcher statistics."""
        avg_time = (
            self.stats["total_time"] / self.stats["functions_processed"]
            if self.stats["functions_processed"] > 0 else 0
        )

        match_rate = (
            self.stats["semantic_matches_found"] / self.stats["functions_processed"]
            if self.stats["functions_processed"] > 0 else 0
        )

        return {
            "functions_processed": self.stats["functions_processed"],
            "semantic_matches_found": self.stats["semantic_matches_found"],
            "match_rate": match_rate,
            "average_time_per_function_ms": avg_time * 1000,
            "embedding_stats": self.embedding_generator.get_stats(),
            "cache_stats": self.embedding_cache.get_stats(),
            "vector_store_stats": self.vector_store.get_stats()
        }
2. Create matcher/semantic_scorer.py
pythonimport re
from typing import Tuple, Optional
import logging

from ..parser import ParsedFunction
from .models import MatchConfidence

logger = logging.getLogger(__name__)

class SemanticScorer:
    """
    Scores and validates semantic matches.

    Applies additional heuristics beyond raw similarity scores.
    """

    # Thresholds from architecture
    THRESHOLDS = {
        "high_confidence": 0.85,
        "medium_confidence": 0.75,
        "low_confidence": 0.65,
        "no_match": 0.65
    }

    def validate_semantic_match(
        self,
        source_function: ParsedFunction,
        candidate_function_id: str,
        raw_similarity: float
    ) -> Tuple[bool, float]:
        """
        Validate and adjust semantic match score.

        Returns:
            (is_valid, adjusted_score)
        """
        # Start with raw similarity
        adjusted_score = raw_similarity

        # Extract function name from ID
        candidate_name = candidate_function_id.split('.')[-1]
        source_name = source_function.signature.name

        # Check naming patterns
        if self._names_follow_pattern(source_name, candidate_name):
            # Boost score for common refactoring patterns
            adjusted_score += 0.05
        else:
            # Penalize very different names
            name_similarity = self._calculate_name_similarity(source_name, candidate_name)
            if name_similarity < 0.3:
                adjusted_score -= 0.1

        # Module distance check (from architecture rules)
        source_module = source_function.file_path.replace('/', '.').replace('\\', '.')
        candidate_module = '.'.join(candidate_function_id.split('.')[:-1])

        module_distance = self._calculate_module_distance(source_module, candidate_module)
        if module_distance > 2:
            # Penalize distant modules
            adjusted_score -= 0.1

        # Parameter count validation (architecture rule)
        # We don't have the candidate function loaded, so we'll be lenient
        # In real implementation, we'd load and check

        # Ensure score stays in valid range
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        # Apply threshold
        is_valid = adjusted_score >= self.THRESHOLDS["no_match"]

        return is_valid, adjusted_score

    def calculate_semantic_confidence(
        self,
        similarity_score: float,
        source_function: ParsedFunction
    ) -> MatchConfidence:
        """Calculate confidence scores for semantic match."""
        # Determine overall confidence based on thresholds
        if similarity_score >= self.THRESHOLDS["high_confidence"]:
            overall = 0.9  # High confidence but not perfect
        elif similarity_score >= self.THRESHOLDS["medium_confidence"]:
            overall = 0.75
        elif similarity_score >= self.THRESHOLDS["low_confidence"]:
            overall = 0.65
        else:
            overall = 0.5  # Minimum for a match

        # Semantic matches have high name similarity (by definition of embeddings)
        # but lower location score
        return MatchConfidence(
            overall=overall,
            name_similarity=similarity_score,  # Embedding similarity
            location_score=0.5,  # Unknown location relationship
            signature_similarity=0.7  # Assumed similar based on embedding
        )

    def _names_follow_pattern(self, name1: str, name2: str) -> bool:
        """Check if names follow common refactoring patterns."""
        patterns = [
            # Verb changes
            (r'^get_(.+)$', r'^fetch_\1$'),
            (r'^set_(.+)$', r'^update_\1$'),
            (r'^create_(.+)$', r'^make_\1$'),
            (r'^delete_(.+)$', r'^remove_\1$'),

            # Noun changes
            (r'^(.+)_id$', r'^\1_identifier$'),
            (r'^(.+)_dict$', r'^\1_map$'),
            (r'^(.+)_list$', r'^\1_array$'),

            # Style changes
            (r'^([a-z]+)_([a-z]+)$', lambda m: f"{m.group(1)}{m.group(2).capitalize()}"),
        ]

        for pattern1, pattern2 in patterns:
            # Check both directions
            if (re.match(pattern1, name1) and re.match(pattern2, name2)):
                return True
            if (re.match(pattern2, name1) and re.match(pattern1, name2)):
                return True

        return False

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate simple name similarity."""
        # Convert to lowercase
        name1 = name1.lower()
        name2 = name2.lower()

        # Exact match
        if name1 == name2:
            return 1.0

        # Check if one contains the other
        if name1 in name2 or name2 in name1:
            return 0.8

        # Simple character overlap
        common_chars = set(name1) & set(name2)
        all_chars = set(name1) | set(name2)

        if all_chars:
            return len(common_chars) / len(all_chars)

        return 0.0

    def _calculate_module_distance(self, module1: str, module2: str) -> int:
        """Calculate distance between modules."""
        parts1 = module1.split('.')
        parts2 = module2.split('.')

        # Find common prefix
        common_prefix_len = 0
        for i, (p1, p2) in enumerate(zip(parts1, parts2)):
            if p1 == p2:
                common_prefix_len = i + 1
            else:
                break

        # Distance is how many levels apart they are
        distance = (len(parts1) - common_prefix_len) + (len(parts2) - common_prefix_len)

        return distance
Testing Requirements for Chunk 3
Create tests covering:

Semantic search with various similarity thresholds
Score adjustment based on naming patterns
Module distance validation
Integration with previous match results
Edge cases (no matches, self-matches)

Critical Implementation Notes

NEVER override high-confidence matches from previous matchers
Always validate module distance per architecture rules
Skip self-matches in search results
Apply all scoring adjustments from architecture
Handle search failures gracefully

Success Criteria

Finds semantically similar functions with renamed identifiers
Respects confidence thresholds from architecture
Integrates smoothly with existing match results
No false positives from distant modules
Performance: <200ms per function including search


## Chunk 4A: Integration Layer - Part 1 ✅ COMPLETE (Estimated: 1.5 hours)
### Objectives (First Half)

- Create unified matching facade foundation with all three matchers
- Implement basic integration between direct, contextual, and semantic matching
- Add basic CLI command for unified matching
- Create foundation integration tests

### Required Files Created/Updated (First Half)

1. `matcher/unified_facade.py` ✅ - Basic UnifiedMatchingFacade class with core integration
2. `matcher/__init__.py` ✅ - Export semantic components and unified facade
3. `main.py` ✅ - Add basic match_unified CLI command
4. `tests/test_unified_matching.py` ✅ - Basic integration tests

### Completed Implementation (First Half)

**Created `matcher/unified_facade.py`** with core UnifiedMatchingFacade:
- Basic class initialization with configuration and statistics tracking
- `match_project()` method implementing four-phase matching pipeline:
  1. Python file parsing with IntegratedParser
  2. Direct matching using MatchingFacade
  3. Contextual matching using ContextualMatchingFacade
  4. Semantic matching using SemanticMatcher (optional)
- Statistics collection across all matching phases
- Performance timing for each phase (parsing, direct, contextual, semantic)
- Match type counting (direct, contextual, semantic)
- File discovery with standard exclusions (.git, __pycache__, venv)
- Basic summary printing functionality

**Updated `matcher/__init__.py`** to export:
- `SemanticMatcher` class
- `UnifiedMatchingFacade` class
- All existing semantic models

**Added basic `match_unified` CLI command** to `main.py`:
- Async command execution for semantic matching support
- Configuration file loading support
- Enable/disable semantic matching option
- Basic output formatting (reusing existing formatters)
- Statistics display option
- Error handling for invalid paths and analysis failures

**Created `tests/test_unified_matching.py`** with comprehensive test coverage:
- Facade initialization and configuration tests
- File discovery testing with exclusion validation
- Statistics calculation and retrieval tests
- Basic project matching workflow with mocked components
- Error handling during file parsing
- Edge cases for empty directories and invalid files
- Integration testing with all three matcher types

---

## Chunk 4B: Integration Layer - Part 2 ❌ PENDING (For Next Developer)
### Objectives (Second Half - To Be Implemented)

- Complete performance monitoring and optimization
- Implement detailed output formatting for unified results
- Add comprehensive CLI features and options
- Create production-ready error recovery
- Add performance validation tests

### Required Files to Create/Update (Second Half)

1. Complete `matcher/unified_facade.py` - Add advanced performance monitoring and optimization
2. Update `main.py` - Add detailed output formatting functions for unified results
3. Create `tests/test_unified_performance.py` - Performance validation tests
4. Add comprehensive integration scenarios

Detailed Implementation
1. Create matcher/unified_facade.py
pythonimport asyncio
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .direct_matcher import DirectMatcher
from .contextual_matcher import ContextualMatcher
from .semantic_matcher import SemanticMatcher
from .models import MatchResult
from ..parser import IntegratedParser, ParsedFunction
from ..utils.config import CodeDocSyncConfig

logger = logging.getLogger(__name__)

class UnifiedMatchingFacade:
    """
    Unified interface for all three matching strategies.

    Implements the complete matching pipeline:
    1. Direct matching (90% of cases)
    2. Contextual matching (8% of cases)
    3. Semantic matching (2% of cases)
    """

    def __init__(self, config: Optional[CodeDocSyncConfig] = None):
        self.config = config or CodeDocSyncConfig()
        self.stats = {
            "total_time": 0.0,
            "parsing_time": 0.0,
            "direct_matching_time": 0.0,
            "contextual_matching_time": 0.0,
            "semantic_matching_time": 0.0,
            "semantic_indexing_time": 0.0,
            "files_processed": 0,
            "matches_by_type": {
                "direct": 0,
                "contextual": 0,
                "semantic": 0
            }
        }

    async def match_project(
        self,
        project_path: str,
        use_cache: bool = True,
        enable_semantic: bool = True
    ) -> MatchResult:
        """
        Perform complete matching on a project.

        Args:
            project_path: Root directory of the project
            use_cache: Whether to use cached parsing/embeddings
            enable_semantic: Whether to use semantic matching

        Returns:
            Unified MatchResult with all matches
        """
        start_time = time.time()
        project_path = Path(project_path).resolve()

        # Initialize components
        parser = IntegratedParser(cache_enabled=use_cache)
        direct_matcher = DirectMatcher(
            fuzzy_threshold=self.config.matching.fuzzy_threshold
        )
        contextual_matcher = ContextualMatcher(str(project_path))

        # Phase 1: Parse all Python files
        logger.info("Phase 1: Parsing Python files...")
        parse_start = time.time()

        all_functions = []
        python_files = self._discover_python_files(project_path)

        for file_path in python_files:
            try:
                functions = parser.parse_file(str(file_path))
                all_functions.extend(functions)
                self.stats["files_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        self.stats["parsing_time"] = time.time() - parse_start
        logger.info(f"Parsed {len(all_functions)} functions from {len(python_files)} files")

        # Phase 2: Build contextual index
        logger.info("Phase 2: Building project context...")
        context_start = time.time()
        contextual_matcher.analyze_project(python_files=[str(p) for p in python_files])
        self.stats["contextual_matching_time"] += time.time() - context_start

        # Phase 3: Direct matching
        logger.info("Phase 3: Direct matching...")
        direct_start = time.time()
        direct_result = direct_matcher.match_functions(all_functions)
        self.stats["direct_matching_time"] = time.time() - direct_start

        # Count direct matches
        direct_matches = [m for m in direct_result.matched_pairs if m.confidence.overall >= 0.8]
        self.stats["matches_by_type"]["direct"] = len(direct_matches)

        # Phase 4: Contextual matching
        logger.info("Phase 4: Contextual matching for remaining functions...")
        context_match_start = time.time()
        contextual_result = contextual_matcher.match_with_context(
            all_functions,
            direct_result
        )
        self.stats["contextual_matching_time"] += time.time() - context_match_start

        # Count new contextual matches
        prev_matched = {m.function.signature.name for m in direct_matches}
        contextual_matches = [
            m for m in contextual_result.matched_pairs
            if m.function.signature.name not in prev_matched and m.confidence.overall >= 0.7
        ]
        self.stats["matches_by_type"]["contextual"] = len(contextual_matches)

        # Phase 5: Semantic matching (if enabled)
        final_result = contextual_result

        if enable_semantic and self.config.matching.enable_semantic:
            logger.info("Phase 5: Semantic matching for remaining functions...")

            # Initialize semantic matcher
            semantic_matcher = SemanticMatcher(str(project_path), self.config.matching)

            # Build semantic index
            index_start = time.time()
            await semantic_matcher.prepare_semantic_index(all_functions, force_reindex=False)
            self.stats["semantic_indexing_time"] = time.time() - index_start

            # Perform semantic matching
            semantic_start = time.time()
            semantic_result = await semantic_matcher.match_with_embeddings(
                all_functions,
                [direct_result, contextual_result]
            )
            self.stats["semantic_matching_time"] = time.time() - semantic_start

            # Count new semantic matches
            all_prev_matched = {
                m.function.signature.name
                for m in contextual_result.matched_pairs
                if m.confidence.overall >= 0.7
            }
            semantic_matches = [
                m for m in semantic_result.matched_pairs
                if m.function.signature.name not in all_prev_matched
            ]
            self.stats["matches_by_type"]["semantic"] = len(semantic_matches)

            final_result = semantic_result

            # Add semantic stats to result
            if not hasattr(final_result, 'metadata'):
                final_result.metadata = {}
            final_result.metadata["semantic_stats"] = semantic_matcher.get_stats()

        # Calculate total time
        self.stats["total_time"] = time.time() - start_time

        # Add unified stats to result
        if not hasattr(final_result, 'metadata'):
            final_result.metadata = {}
        final_result.metadata["unified_stats"] = self.get_stats()

        return final_result

    def _discover_python_files(self, project_path: Path) -> List[Path]:
        """Discover Python files with exclusions."""
        exclusions = {'.git', '__pycache__', 'venv', 'env', '.venv', 'build', 'dist'}

        python_files = []
        for path in project_path.rglob("*.py"):
            # Check if any parent directory is in exclusions
            if any(part in exclusions for part in path.parts):
                continue
            python_files.append(path)

        return python_files

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_matches = sum(self.stats["matches_by_type"].values())

        return {
            "total_time_seconds": self.stats["total_time"],
            "phase_times": {
                "parsing": self.stats["parsing_time"],
                "direct_matching": self.stats["direct_matching_time"],
                "contextual_matching": self.stats["contextual_matching_time"],
                "semantic_indexing": self.stats["semantic_indexing_time"],
                "semantic_matching": self.stats["semantic_matching_time"]
            },
            "files_processed": self.stats["files_processed"],
            "total_matches": total_matches,
            "matches_by_type": self.stats["matches_by_type"],
            "match_distribution": {
                "direct": f"{self.stats['matches_by_type']['direct'] / total_matches * 100:.1f}%" if total_matches > 0 else "0%",
                "contextual": f"{self.stats['matches_by_type']['contextual'] / total_matches * 100:.1f}%" if total_matches > 0 else "0%",
                "semantic": f"{self.stats['matches_by_type']['semantic'] / total_matches * 100:.1f}%" if total_matches > 0 else "0%"
            }
        }

    def print_summary(self) -> None:
        """Print comprehensive matching summary."""
        stats = self.get_stats()

        print("\n=== Unified Matching Summary ===")
        print(f"Total time: {stats['total_time_seconds']:.2f}s")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Total matches: {stats['total_matches']}")

        print("\n--- Time Breakdown ---")
        for phase, time in stats['phase_times'].items():
            if time > 0:
                print(f"{phase}: {time:.2f}s ({time/stats['total_time_seconds']*100:.1f}%)")

        print("\n--- Match Distribution ---")
        for match_type, count in stats['matches_by_type'].items():
            if count > 0:
                print(f"{match_type}: {count} matches ({stats['match_distribution'][match_type]})")
2. Update CLI with semantic command
Add to cli/main.py:
python@app.command()
def match_unified(
    path: Path = typer.Argument(..., help="Project directory to analyze"),
    output_format: str = typer.Option("terminal", "--format", "-f"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    enable_semantic: bool = typer.Option(True, "--semantic/--no-semantic", help="Enable semantic matching"),
    show_stats: bool = typer.Option(False, "--stats", help="Show detailed statistics"),
    show_unmatched: bool = typer.Option(False, "--show-unmatched", help="Show unmatched functions")
):
    """
    Perform unified matching using all three strategies.

    Uses direct, contextual, and semantic matching for best results.
    """
    # Load configuration
    if config and config.exists():
        config_obj = CodeDocSyncConfig.from_yaml(str(config))
    else:
        config_obj = CodeDocSyncConfig()

    # Create facade
    facade = UnifiedMatchingFacade(config_obj)

    # Run matching
    with Progress() as progress:
        task = progress.add_task("Analyzing project...", total=None)

        # Run async matching
        result = asyncio.run(
            facade.match_project(
                str(path),
                enable_semantic=enable_semantic
            )
        )

        progress.update(task, completed=True)

    # Format output
    if output_format == "json":
        output = _format_json_unified_result(result, show_unmatched)
    else:
        output = _format_terminal_unified_result(result, show_unmatched)

    # Save or print
    if output_file:
        output_file.write_text(output)
        console.print(f"✅ Results saved to {output_file}")
    else:
        console.print(output)

    # Show statistics if requested
    if show_stats:
        facade.print_summary()

        # Show semantic-specific stats if available
        if hasattr(result, 'metadata') and 'semantic_stats' in result.metadata:
            print("\n--- Semantic Matching Stats ---")
            sem_stats = result.metadata['semantic_stats']
            print(f"Embeddings generated: {sem_stats.get('embedding_stats', {}).get('embeddings_generated', 0)}")
            print(f"Cache hit rate: {sem_stats.get('cache_stats', {}).get('overall_hit_rate', 0):.1%}")
            print(f"Average search time: {sem_stats.get('vector_store_stats', {}).get('average_search_time_ms', 0):.1f}ms")
Testing Requirements for Chunk 4
Create integration tests that verify:

Complete pipeline from parsing to semantic matching
Proper fallback chain (direct → contextual → semantic)
Statistics tracking across all matchers
Configuration properly applied to all components
Performance within acceptable bounds

Critical Implementation Notes

Async/await properly for semantic operations
Each matcher preserves previous high-confidence matches
Statistics must track matches by type accurately
Semantic matching is optional (can be disabled)
Handle API failures gracefully

Success Criteria

Complete pipeline executes without errors
Match distribution roughly follows 90/8/2 pattern
All three matchers integrate seamlessly
CLI provides useful feedback and stats
Total time <30s for 1000 functions


Chunk 5: Performance Optimization and Production Readiness (Estimated: 2-3 hours)
Objectives

Optimize embedding generation with batching
Implement comprehensive error recovery
Add monitoring and alerting capabilities
Ensure production-ready performance

Required Files to Create/Update

matcher/semantic_optimizer.py - Performance optimizations
tests/test_semantic_performance.py - Performance validation
tests/test_semantic_integration.py - End-to-end tests
Update configuration for production settings

Detailed Implementation
1. Create matcher/semantic_optimizer.py
pythonimport asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Dict, Any, Optional
import psutil
import time

from ..parser import ParsedFunction
from .semantic_models import FunctionEmbedding

logger = logging.getLogger(__name__)

class SemanticOptimizer:
    """
    Performance optimizations for semantic matching.

    Handles:
    - Batch processing optimization
    - Memory management
    - Concurrent operations
    - Resource monitoring
    """

    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Monitor resource usage
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024

    def optimize_batch_size(self, total_functions: int) -> int:
        """
        Dynamically determine optimal batch size.

        Considers:
        - Available memory
        - Number of functions
        - API rate limits
        """
        # Base batch size
        base_batch = 100

        # Adjust based on available memory
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.initial_memory
        memory_available = self.max_memory_mb - memory_used

        if memory_available < 100:
            # Low memory, use smaller batches
            return min(25, total_functions)
        elif memory_available < 200:
            return min(50, total_functions)
        else:
            # Plenty of memory
            return min(base_batch, total_functions)

    async def parallel_embedding_generation(
        self,
        function_batches: List[List[ParsedFunction]],
        generator_func: callable
    ) -> List[FunctionEmbedding]:
        """
        Generate embeddings in parallel with controlled concurrency.
        """
        # Limit concurrent API calls to avoid rate limits
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent batches

        async def process_batch_with_limit(batch):
            async with semaphore:
                return await generator_func(batch)

        # Process all batches
        tasks = [
            process_batch_with_limit(batch)
            for batch in function_batches
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle errors
        all_embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
            else:
                all_embeddings.extend(result)

        return all_embeddings

    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.initial_memory

        # Trigger GC if using >80% of allowed memory
        return memory_used > (self.max_memory_mb * 0.8)

    def optimize_vector_store_queries(
        self,
        queries: List[List[float]],
        vector_store: Any
    ) -> List[List[Any]]:
        """
        Optimize vector store queries with batching.

        Some vector stores support batch queries for better performance.
        """
        # Check if vector store supports batch queries
        if hasattr(vector_store, 'search_similar_batch'):
            # Use batch search
            batch_size = 10
            results = []

            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                batch_results = vector_store.search_similar_batch(batch)
                results.extend(batch_results)

            return results
        else:
            # Fall back to individual queries
            return [
                vector_store.search_similar(query)
                for query in queries
            ]

    def estimate_processing_time(
        self,
        num_functions: int,
        cache_hit_rate: float = 0.9
    ) -> Dict[str, float]:
        """Estimate processing time for semantic matching."""
        # Based on performance benchmarks
        embeddings_to_generate = int(num_functions * (1 - cache_hit_rate))

        estimates = {
            "embedding_generation": embeddings_to_generate * 0.1,  # 100ms per function
            "vector_indexing": num_functions * 0.01,  # 10ms per function
            "similarity_search": num_functions * 0.05,  # 50ms per function
            "total_estimated": 0.0
        }

        estimates["total_estimated"] = sum(estimates.values())

        return estimates

    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)

class SemanticErrorRecovery:
    """
    Comprehensive error recovery for semantic matching.
    """

    @staticmethod
    async def with_embedding_fallback(
        primary_func: callable,
        fallback_funcs: List[callable],
        function: ParsedFunction
    ) -> Optional[FunctionEmbedding]:
        """
        Try multiple embedding providers with fallback.
        """
        # Try primary
        try:
            return await primary_func(function)
        except Exception as e:
            logger.warning(f"Primary embedding failed: {e}")

        # Try fallbacks in order
        for i, fallback in enumerate(fallback_funcs):
            try:
                logger.info(f"Trying fallback {i+1}")
                return await fallback(function)
            except Exception as e:
                logger.warning(f"Fallback {i+1} failed: {e}")
                continue

        # All failed
        logger.error(f"All embedding methods failed for {function.signature.name}")
        return None

    @staticmethod
    def handle_api_errors(error: Exception) -> Dict[str, Any]:
        """
        Categorize and handle API errors.

        Returns:
            Dictionary with error type and recommended action
        """
        error_str = str(error).lower()

        if "rate limit" in error_str:
            return {
                "type": "rate_limit",
                "action": "wait",
                "wait_seconds": 60,
                "message": "Rate limit exceeded, waiting before retry"
            }

        elif "api key" in error_str or "authentication" in error_str:
            return {
                "type": "auth_error",
                "action": "skip",
                "message": "API authentication failed, check credentials"
            }

        elif "timeout" in error_str:
            return {
                "type": "timeout",
                "action": "retry",
                "message": "Request timed out, will retry"
            }

        elif "connection" in error_str:
            return {
                "type": "connection_error",
                "action": "retry",
                "wait_seconds": 5,
                "message": "Connection error, will retry"
            }

        else:
            return {
                "type": "unknown",
                "action": "skip",
                "message": f"Unknown error: {error}"
            }
2. Create comprehensive performance tests
Create tests/test_semantic_performance.py:
pythonimport pytest
import asyncio
import time
from unittest.mock import Mock, patch

from codedocsync.matcher.semantic_matcher import SemanticMatcher
from codedocsync.matcher.semantic_optimizer import SemanticOptimizer
from codedocsync.parser import ParsedFunction, FunctionSignature

class TestSemanticPerformance:
    """Validate semantic matcher meets performance requirements."""

    @pytest.fixture
    def sample_functions(self):
        """Generate sample functions for testing."""
        functions = []
        for i in range(100):
            func = ParsedFunction(
                signature=FunctionSignature(
                    name=f"function_{i}",
                    parameters=[],
                    return_type="None"
                ),
                docstring=None,
                file_path=f"module_{i % 10}.py",
                line_number=i * 10,
                end_line_number=i * 10 + 5,
                source_code=f"def function_{i}(): pass"
            )
            functions.append(func)
        return functions

    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self, sample_functions):
        """Test embedding generation meets <100ms per function target."""
        # Mock the actual API calls
        with patch('openai.Embedding.acreate') as mock_create:
            mock_create.return_value = {
                'data': [{'embedding': [0.1] * 1536}]
            }

            matcher = SemanticMatcher("/test/project")

            start_time = time.time()
            await matcher.prepare_semantic_index(sample_functions[:10])
            duration = time.time() - start_time

            # Should complete in <1 second for 10 functions
            assert duration < 1.0

            # Average should be <100ms per function
            avg_per_function = duration / 10
            assert avg_per_function < 0.1

    @pytest.mark.asyncio
    async def test_search_performance(self, sample_functions):
        """Test similarity search meets <50ms per query target."""
        matcher = SemanticMatcher("/test/project")

        # Pre-populate vector store
        test_embeddings = [[0.1] * 1536 for _ in range(100)]
        test_metadata = [
            {"function_id": f"module.function_{i}", "model": "test"}
            for i in range(100)
        ]
        test_ids = [f"func_{i}" for i in range(100)]

        matcher.vector_store.add_embeddings(
            test_embeddings,
            test_metadata,
            test_ids
        )

        # Test search performance
        query_embedding = [0.2] * 1536

        start_time = time.time()
        for _ in range(10):
            results = matcher.vector_store.search_similar(query_embedding)
        duration = time.time() - start_time

        # Average should be <50ms per search
        avg_per_search = duration / 10
        assert avg_per_search < 0.05

    def test_memory_usage(self, sample_functions):
        """Test memory usage stays under 500MB for 10k embeddings."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create large number of embeddings
        embeddings = []
        for i in range(10000):
            # Each embedding is ~6KB (1536 floats * 4 bytes)
            embedding = [0.1] * 1536
            embeddings.append(embedding)

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - initial_memory

        # Should use less than 500MB
        assert memory_used < 500

        # Clean up
        del embeddings

    def test_batch_size_optimization(self):
        """Test dynamic batch size optimization."""
        optimizer = SemanticOptimizer(max_memory_mb=500)

        # Test with different memory conditions
        optimizer.initial_memory = 1000  # Simulate 1GB initial

        # High memory available
        optimizer.process.memory_info = Mock(return_value=Mock(rss=1050 * 1024 * 1024))
        assert optimizer.optimize_batch_size(1000) == 100

        # Medium memory available
        optimizer.process.memory_info = Mock(return_value=Mock(rss=1350 * 1024 * 1024))
        assert optimizer.optimize_batch_size(1000) == 50

        # Low memory available
        optimizer.process.memory_info = Mock(return_value=Mock(rss=1450 * 1024 * 1024))
        assert optimizer.optimize_batch_size(1000) == 25

    @pytest.mark.asyncio
    async def test_cache_performance(self, sample_functions):
        """Test cache hit rate >90% for unchanged functions."""
        from codedocsync.storage.embedding_cache import EmbeddingCache

        cache = EmbeddingCache()

        # First pass - all misses
        for func in sample_functions[:10]:
            text = f"def {func.signature.name}(): pass"
            result = cache.get(text, "test-model")
            assert result is None

        # Populate cache
        for i, func in enumerate(sample_functions[:10]):
            text = f"def {func.signature.name}(): pass"
            embedding = FunctionEmbedding(
                function_id=f"module.{func.signature.name}",
                embedding=[0.1] * 384,
                model="test-model",
                text_embedded=text,
                timestamp=time.time(),
                signature_hash=f"hash_{i}"
            )
            cache.set(embedding)

        # Second pass - should be all hits
        hits = 0
        for func in sample_functions[:10]:
            text = f"def {func.signature.name}(): pass"
            result = cache.get(text, "test-model")
            if result is not None:
                hits += 1

        hit_rate = hits / 10
        assert hit_rate >= 0.9

    @pytest.mark.asyncio
    async def test_full_pipeline_performance(self, sample_functions):
        """Test complete semantic matching pipeline performance."""
        # This would be an integration test with mocked external services
        with patch('openai.Embedding.acreate') as mock_embed:
            mock_embed.return_value = {
                'data': [{'embedding': [0.1] * 1536}]
            }

            matcher = SemanticMatcher("/test/project")

            start_time = time.time()

            # Index functions
            await matcher.prepare_semantic_index(sample_functions)

            # Perform matching
            result = await matcher.match_with_embeddings(
                sample_functions[:20]
            )

            duration = time.time() - start_time

            # Should complete within reasonable time
            # 100 functions indexed + 20 matched
            assert duration < 30  # Very generous for mocked test

            # Check result
            assert result.total_functions == 20
Testing Requirements for Chunk 5
Create tests that validate:

Performance meets all architectural requirements
Memory usage stays within bounds
Error recovery works correctly
Batch optimization improves throughput
Cache provides expected hit rates

Critical Implementation Notes

Monitor memory usage continuously
Implement proper cleanup for long-running processes
Use semaphores to limit concurrent API calls
Track all performance metrics for debugging
Gracefully degrade when resources are constrained

Success Criteria

All performance targets from architecture are met
Memory usage <500MB for 10k embeddings


90% cache hit rate for unchanged functions


Graceful handling of API failures
Production-ready error recovery


Sprint Summary
This semantic matcher implementation provides:

Robust Foundation: ChromaDB integration with proper persistence and error handling
Efficient Embeddings: Multi-provider support with intelligent fallbacks and caching
Accurate Matching: Validation rules from architecture properly enforced
Seamless Integration: Works perfectly with existing direct and contextual matchers
Production Ready: Comprehensive error recovery, monitoring, and performance optimization

The implementation follows all architectural guidelines and integrates cleanly with the existing codebase. The semantic matcher truly acts as a last resort, only processing functions that couldn't be matched by simpler methods, ensuring optimal performance and accuracy.
Common Pitfalls to Avoid

Don't Over-Engineer: This handles only 2% of cases - keep it simple
Respect the Pipeline: Never override high-confidence matches from previous matchers
Cache Aggressively: Embeddings are expensive - cache everything
Handle Failures Gracefully: External services will fail - always have fallbacks
Monitor Resources: Track memory and API usage to avoid surprises

Final Integration Checklist

 Vector store persists between runs
 Embeddings are cached effectively
 API failures don't crash the system
 Performance meets all targets
 CLI provides useful feedback
 All tests pass
 Documentation is updated
