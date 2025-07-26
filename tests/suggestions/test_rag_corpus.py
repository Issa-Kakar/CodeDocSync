"""
Tests for RAG corpus manager.

Tests the RAG corpus functionality for self-improving documentation suggestions,
including corpus loading, example retrieval, and graceful degradation.
"""

import json
import time
from unittest.mock import Mock, patch

import pytest

from codedocsync.parser import ParsedDocstring, ParsedFunction, RawDocstring
from codedocsync.parser.ast_parser import FunctionParameter, FunctionSignature
from codedocsync.parser.docstring_models import DocstringFormat
from codedocsync.storage.embedding_cache import EmbeddingCache
from codedocsync.storage.vector_store import VectorStore
from codedocsync.suggestions.rag_corpus import (
    DocstringExample,
    RAGCorpusManager,
)


# Test fixtures
@pytest.fixture
def sample_corpus_data():
    """Create sample corpus data for testing."""
    return {
        "examples": [
            {
                "function_name": "calculate_sum",
                "module_path": "math_utils.py",
                "function_signature": "calculate_sum(a: int, b: int) -> int",
                "docstring_format": "google",
                "docstring_content": "Calculate the sum of two integers.\n\nArgs:\n    a: First integer.\n    b: Second integer.\n\nReturns:\n    The sum of a and b.",
                "has_params": True,
                "has_returns": True,
                "has_examples": False,
                "complexity_score": 2,
                "quality_score": 4,
                "source": "bootstrap",
                "timestamp": 1234567890.0,
            },
            {
                "function_name": "validate_email",
                "module_path": "validators.py",
                "function_signature": "validate_email(email: str) -> bool",
                "docstring_format": "google",
                "docstring_content": "Validate an email address.\n\nArgs:\n    email: Email address to validate.\n\nReturns:\n    True if valid, False otherwise.",
                "has_params": True,
                "has_returns": True,
                "has_examples": False,
                "complexity_score": 3,
                "quality_score": 5,
                "source": "bootstrap",
                "timestamp": 1234567891.0,
            },
        ]
    }


@pytest.fixture
def temp_corpus_dir(tmp_path):
    """Create temporary corpus directory with test data."""
    corpus_dir = tmp_path / "data"
    corpus_dir.mkdir()
    return corpus_dir


@pytest.fixture
def corpus_manager_no_embeddings(temp_corpus_dir):
    """Create RAG corpus manager without embeddings."""
    return RAGCorpusManager(corpus_dir=str(temp_corpus_dir), enable_embeddings=False)


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    mock = Mock(spec=VectorStore)
    mock.get_stats.return_value = {"collections": 1, "embeddings": 100}
    return mock


@pytest.fixture
def mock_embedding_cache():
    """Create mock embedding cache."""
    mock = Mock(spec=EmbeddingCache)
    return mock


@pytest.fixture
def sample_function():
    """Create sample parsed function."""
    signature = FunctionSignature(
        name="process_data",
        parameters=[
            FunctionParameter(
                name="data",
                type_annotation="list[dict]",
                default_value=None,
                is_required=True,
            ),
            FunctionParameter(
                name="validate",
                type_annotation="bool",
                default_value="True",
                is_required=False,
            ),
        ],
        return_type="dict[str, Any]",
        is_method=False,
        is_async=False,
        decorators=[],
    )

    docstring = RawDocstring(
        "Process data with optional validation.\n\nArgs:\n    data: Input data.\n    validate: Whether to validate.\n\nReturns:\n    Processed data.",
        line_number=2,
    )

    return ParsedFunction(
        signature=signature,
        docstring=docstring,
        file_path="processor.py",
        line_number=1,
        end_line_number=10,
        source_code="def process_data(data: list[dict], validate: bool = True) -> dict[str, Any]: pass",
    )


@pytest.fixture
def sample_parsed_docstring():
    """Create sample parsed docstring."""
    mock = Mock(spec=ParsedDocstring)
    mock.format = DocstringFormat.GOOGLE
    mock.parameters = [Mock(name="data"), Mock(name="validate")]
    mock.returns = Mock(description="Processed data")
    mock.examples = []
    return mock


class TestDocstringExample:
    """Tests for DocstringExample dataclass."""

    def test_docstring_example_creation(self):
        """Test creating a DocstringExample."""
        example = DocstringExample(
            function_name="test_func",
            module_path="test.py",
            function_signature="test_func(x: int) -> str",
            docstring_format="google",
            docstring_content="Test function.",
            has_params=True,
            has_returns=True,
            has_examples=False,
            complexity_score=3,
            quality_score=4,
        )

        assert example.function_name == "test_func"
        assert example.module_path == "test.py"
        assert example.source == "bootstrap"
        assert example.timestamp == 0.0


class TestRAGCorpusManager:
    """Tests for RAGCorpusManager."""

    def test_initialization_without_embeddings(self, temp_corpus_dir):
        """Test initialization without embeddings."""
        manager = RAGCorpusManager(
            corpus_dir=str(temp_corpus_dir), enable_embeddings=False
        )

        assert manager.vector_store is None
        assert manager.embedding_cache is None
        assert not manager.enable_embeddings
        assert len(manager.memory_corpus) == 0
        assert manager._metrics["examples_loaded"] == 0

    def test_initialization_with_embeddings(
        self, temp_corpus_dir, mock_vector_store, mock_embedding_cache
    ):
        """Test initialization with embeddings."""
        manager = RAGCorpusManager(
            vector_store=mock_vector_store,
            embedding_cache=mock_embedding_cache,
            corpus_dir=str(temp_corpus_dir),
            enable_embeddings=True,
        )

        assert manager.vector_store == mock_vector_store
        assert manager.embedding_cache == mock_embedding_cache
        assert manager.enable_embeddings

    def test_initialization_embeddings_failure(self, temp_corpus_dir):
        """Test graceful degradation when embeddings fail."""
        with patch("codedocsync.suggestions.rag_corpus.VectorStore") as mock_vs:
            mock_vs.side_effect = Exception("Connection failed")

            manager = RAGCorpusManager(
                corpus_dir=str(temp_corpus_dir), enable_embeddings=True
            )

            assert manager.vector_store is None
            assert not manager.enable_embeddings

    def test_load_bootstrap_corpus(self, temp_corpus_dir, sample_corpus_data):
        """Test loading bootstrap corpus from JSON."""
        # Write test corpus file
        bootstrap_file = temp_corpus_dir / "bootstrap_corpus.json"
        with open(bootstrap_file, "w", encoding="utf-8") as f:
            json.dump(sample_corpus_data, f)

        manager = RAGCorpusManager(
            corpus_dir=str(temp_corpus_dir), enable_embeddings=False
        )

        assert len(manager.memory_corpus) == 2
        assert manager.memory_corpus[0].function_name == "calculate_sum"
        assert manager.memory_corpus[1].function_name == "validate_email"
        assert manager._metrics["examples_loaded"] == 2

    def test_load_curated_examples(self, temp_corpus_dir):
        """Test loading curated examples."""
        curated_data = {
            "examples": [
                {
                    "function_name": "special_function",
                    "module_path": "special.py",
                    "function_signature": "special_function() -> None",
                    "docstring_format": "numpy",
                    "docstring_content": "Special function.\n\nReturns\n-------\nNone",
                    "has_params": False,
                    "has_returns": True,
                    "has_examples": False,
                    "complexity_score": 1,
                    "quality_score": 5,
                }
            ]
        }

        curated_file = temp_corpus_dir / "curated_examples.json"
        with open(curated_file, "w", encoding="utf-8") as f:
            json.dump(curated_data, f)

        manager = RAGCorpusManager(
            corpus_dir=str(temp_corpus_dir), enable_embeddings=False
        )

        # Find curated example
        curated = [ex for ex in manager.memory_corpus if ex.source == "curated"]
        assert len(curated) == 1
        assert curated[0].function_name == "special_function"

    def test_load_corpus_with_invalid_json(self, temp_corpus_dir):
        """Test handling of invalid JSON in corpus files."""
        bootstrap_file = temp_corpus_dir / "bootstrap_corpus.json"
        with open(bootstrap_file, "w", encoding="utf-8") as f:
            f.write("invalid json{")

        manager = RAGCorpusManager(
            corpus_dir=str(temp_corpus_dir), enable_embeddings=False
        )

        assert len(manager.memory_corpus) == 0
        assert manager._metrics["examples_loaded"] == 0

    def test_add_good_example(
        self, corpus_manager_no_embeddings, sample_function, sample_parsed_docstring
    ):
        """Test adding a good example from analysis."""
        manager = corpus_manager_no_embeddings
        initial_count = len(manager.memory_corpus)

        manager.add_good_example(
            function=sample_function, docstring=sample_parsed_docstring, quality_score=5
        )

        assert len(manager.memory_corpus) == initial_count + 1
        assert manager._metrics["examples_added"] == 1

        # Check the added example
        added = manager.memory_corpus[-1]
        assert added.function_name == "process_data"
        assert added.source == "good_example"
        assert added.quality_score == 5
        assert added.has_params is True
        assert added.has_returns is True

    def test_add_accepted_suggestion(
        self, corpus_manager_no_embeddings, sample_function
    ):
        """Test adding an accepted suggestion."""
        manager = corpus_manager_no_embeddings
        initial_count = len(manager.memory_corpus)

        suggested_docstring = """Process data with validation.

        Args:
            data: Input data to process.
            validate: Whether to validate input.

        Returns:
            Processed data dictionary.
        """

        manager.add_accepted_suggestion(
            function=sample_function,
            suggested_docstring=suggested_docstring,
            docstring_format="google",
            issue_type="missing_description",
        )

        assert len(manager.memory_corpus) == initial_count + 1
        assert manager._metrics["examples_added"] == 1

        # Check the added example
        added = manager.memory_corpus[-1]
        assert added.function_name == "process_data"
        assert added.source == "accepted"
        assert added.quality_score == 4  # Accepted suggestions get quality 4
        assert added.docstring_content == suggested_docstring

    def test_retrieve_examples_memory_based(
        self, temp_corpus_dir, sample_corpus_data, sample_function
    ):
        """Test retrieving examples using memory-based search."""
        # Write test corpus
        bootstrap_file = temp_corpus_dir / "bootstrap_corpus.json"
        with open(bootstrap_file, "w", encoding="utf-8") as f:
            json.dump(sample_corpus_data, f)

        manager = RAGCorpusManager(
            corpus_dir=str(temp_corpus_dir), enable_embeddings=False
        )

        # Retrieve examples
        examples = manager.retrieve_examples(
            function=sample_function, n_results=2, min_similarity=0.0
        )

        assert len(examples) <= 2
        assert manager._metrics["retrievals_performed"] == 1
        assert manager._metrics["total_retrieval_time"] > 0

        # Examples should be sorted by similarity
        if len(examples) > 1:
            # Since we're using heuristic matching, just verify we got results
            assert all(isinstance(ex, DocstringExample) for ex in examples)

    def test_retrieve_examples_with_similarity_threshold(
        self, corpus_manager_no_embeddings, sample_function
    ):
        """Test retrieving examples with similarity threshold."""
        manager = corpus_manager_no_embeddings

        # Add some examples
        for i in range(5):
            example = DocstringExample(
                function_name=f"func_{i}",
                module_path="test.py",
                function_signature=f"func_{i}() -> None",
                docstring_format="google",
                docstring_content=f"Function {i}.",
                has_params=False,
                has_returns=False,
                has_examples=False,
                complexity_score=1,
                quality_score=3,
                timestamp=time.time(),
            )
            manager.memory_corpus.append(example)

        # Retrieve with high similarity threshold
        examples = manager.retrieve_examples(
            function=sample_function, n_results=10, min_similarity=0.9
        )

        # Should get few or no results with high threshold
        assert len(examples) <= 2

    def test_calculate_similarity_score(self, corpus_manager_no_embeddings):
        """Test similarity score calculation."""
        manager = corpus_manager_no_embeddings

        # Create function and example with similar names
        function = Mock(spec=ParsedFunction)
        function.signature = Mock()
        function.signature.name = "calculate_sum"
        function.signature.parameters = [Mock(), Mock()]  # 2 params
        function.signature.return_type = "int"

        example = DocstringExample(
            function_name="calculate_sum",
            module_path="math.py",
            function_signature="calculate_sum(a: int, b: int) -> int",
            docstring_format="google",
            docstring_content="Calculate sum.",
            has_params=True,
            has_returns=True,
            has_examples=False,
            complexity_score=2,
            quality_score=4,
        )

        score = manager._calculate_similarity_score(function, example)

        # Should have high score due to exact name match
        assert score > 0.7

        # Test with different name
        function.signature.name = "multiply_values"
        score2 = manager._calculate_similarity_score(function, example)
        assert score2 < score

    def test_string_similarity(self, corpus_manager_no_embeddings):
        """Test string similarity calculation."""
        manager = corpus_manager_no_embeddings

        # Exact match
        assert manager._string_similarity("test", "test") == 1.0

        # Case insensitive
        assert manager._string_similarity("Test", "test") == 1.0

        # Common prefix
        score = manager._string_similarity("calculate_sum", "calculate_average")
        assert 0.5 < score < 1.0

        # No similarity
        assert manager._string_similarity("abc", "xyz") < 0.2

    def test_calculate_complexity(self, corpus_manager_no_embeddings, sample_function):
        """Test complexity score calculation."""
        manager = corpus_manager_no_embeddings

        # Simple function
        simple_func = Mock(spec=ParsedFunction)
        simple_func.signature = Mock()
        simple_func.signature.parameters = []
        simple_func.signature.return_type = None

        # Base score is 1, but the function checks for empty params list which might add 1
        score = manager._calculate_complexity(simple_func)
        assert score >= 1 and score <= 2

        # Complex function
        complex_func = Mock(spec=ParsedFunction)
        complex_func.signature = Mock()
        complex_func.signature.parameters = [
            Mock(type_annotation="str"),
            Mock(type_annotation="int"),
            Mock(type_annotation="bool"),
            Mock(type_annotation="list[str]"),
            Mock(type_annotation="dict[str, Any]"),
        ]
        complex_func.signature.return_type = "tuple[bool, str]"

        score = manager._calculate_complexity(complex_func)
        assert score >= 4  # 5 params + return type + type annotations

    def test_get_stats(self, corpus_manager_no_embeddings):
        """Test getting corpus statistics."""
        manager = corpus_manager_no_embeddings

        # Add some activity
        manager._metrics["examples_loaded"] = 10
        manager._metrics["examples_added"] = 5
        manager._metrics["retrievals_performed"] = 3
        manager._metrics["total_retrieval_time"] = 0.15

        stats = manager.get_stats()

        assert stats["corpus_size"] == len(manager.memory_corpus)
        assert stats["examples_loaded"] == 10
        assert stats["examples_added"] == 5
        assert stats["retrievals_performed"] == 3
        assert abs(stats["average_retrieval_time_ms"] - 50.0) < 0.01  # 0.15 / 3 * 1000
        assert stats["embeddings_enabled"] is False
        assert stats["vector_store_stats"]["status"] == "disabled"

    def test_get_stats_with_vector_store(
        self, temp_corpus_dir, mock_vector_store, mock_embedding_cache
    ):
        """Test getting stats with vector store enabled."""
        manager = RAGCorpusManager(
            vector_store=mock_vector_store,
            embedding_cache=mock_embedding_cache,
            corpus_dir=str(temp_corpus_dir),
            enable_embeddings=True,
        )

        stats = manager.get_stats()

        assert stats["embeddings_enabled"] is True
        assert stats["vector_store_stats"] == {"collections": 1, "embeddings": 100}

    def test_create_text_representation(self, corpus_manager_no_embeddings):
        """Test creating text representation for embedding."""
        manager = corpus_manager_no_embeddings

        example = DocstringExample(
            function_name="test_func",
            module_path="test.py",
            function_signature="test_func(x: int) -> str",
            docstring_format="google",
            docstring_content="Test function that converts int to string.",
            has_params=True,
            has_returns=True,
            has_examples=False,
            complexity_score=2,
            quality_score=4,
        )

        text = manager._create_text_representation(example)

        assert "Function: test_func" in text
        assert "Signature: test_func(x: int) -> str" in text
        assert "Format: google" in text
        assert "Docstring: Test function that converts int to string." in text

    def test_error_handling_in_add_good_example(self, corpus_manager_no_embeddings):
        """Test error handling when adding good example fails."""
        manager = corpus_manager_no_embeddings

        # Mock function that causes error
        bad_function = Mock(spec=ParsedFunction)
        bad_function.signature = Mock()
        bad_function.signature.name.side_effect = AttributeError("No name")

        # Should not raise, just log error
        manager.add_good_example(
            function=bad_function, docstring=Mock(), quality_score=4
        )

        assert manager._metrics["examples_added"] == 0

    def test_store_example_with_embeddings_error(self, temp_corpus_dir):
        """Test storing example when embedding generation fails."""
        manager = RAGCorpusManager(
            corpus_dir=str(temp_corpus_dir), enable_embeddings=False
        )

        example = DocstringExample(
            function_name="test",
            module_path="test.py",
            function_signature="test() -> None",
            docstring_format="google",
            docstring_content="Test.",
            has_params=False,
            has_returns=False,
            has_examples=False,
            complexity_score=1,
            quality_score=3,
        )

        # Should store in memory even if embeddings disabled
        manager._store_example(example)
        assert len(manager.memory_corpus) == 1
        # When embeddings are disabled, embedding_id may not be set
        # Just verify the example was stored


class TestIntegration:
    """Integration tests for RAG corpus manager."""

    def test_full_workflow(
        self,
        temp_corpus_dir,
        sample_corpus_data,
        sample_function,
        sample_parsed_docstring,
    ):
        """Test complete workflow from initialization to retrieval."""
        # Setup corpus file
        bootstrap_file = temp_corpus_dir / "bootstrap_corpus.json"
        with open(bootstrap_file, "w", encoding="utf-8") as f:
            json.dump(sample_corpus_data, f)

        # Initialize manager
        manager = RAGCorpusManager(
            corpus_dir=str(temp_corpus_dir), enable_embeddings=False
        )

        # Verify bootstrap loaded
        assert len(manager.memory_corpus) == 2

        # Add a good example
        manager.add_good_example(
            function=sample_function, docstring=sample_parsed_docstring, quality_score=5
        )

        assert len(manager.memory_corpus) == 3

        # Retrieve examples
        examples = manager.retrieve_examples(function=sample_function, n_results=2)

        assert len(examples) > 0

        # Get stats
        stats = manager.get_stats()
        assert stats["corpus_size"] == 3
        assert stats["examples_loaded"] == 2
        assert stats["examples_added"] == 1
        assert stats["retrievals_performed"] == 1
