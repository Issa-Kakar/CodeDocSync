"""Integration tests for RAG corpus with expanded curated examples."""

import time

import pytest

from codedocsync.suggestions.generators import (
    ExampleSuggestionGenerator,
    ParameterSuggestionGenerator,
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.rag_corpus import RAGCorpusManager


class TestRAGCorpusExpansionIntegration:
    """Test RAG corpus integration with expanded curated examples."""

    @pytest.fixture
    def rag_manager(self):
        """Create RAG corpus manager instance."""
        return RAGCorpusManager()

    def test_loading_expanded_corpus(self, rag_manager):
        """Test that expanded corpus loads successfully."""
        # Get corpus stats
        stats = rag_manager.get_corpus_stats()

        # Verify expanded corpus is loaded
        assert stats["total_examples"] >= 218  # 143 bootstrap + 75+ curated
        assert stats["curated_examples"] >= 75
        assert stats["bootstrap_examples"] == 143

        # Check collections are created
        assert "corpus_functions" in stats["collections"]
        assert "corpus_parameters" in stats["collections"]

    def test_similarity_scoring_with_new_examples(self, rag_manager):
        """Test that similarity scoring works with new examples."""
        # Test async pattern
        async_results = rag_manager.find_similar_functions(
            "fetch_data",
            "async def fetch_data(url: str, timeout: int = 30) -> dict[str, Any]:",
            limit=5,
        )

        assert len(async_results) > 0
        assert all(0 <= result["score"] <= 1.0 for result in async_results)

        # Test REST API pattern
        api_results = rag_manager.find_similar_functions(
            "update_user",
            "def update_user(user_id: str, data: dict) -> UserResponse:",
            limit=5,
        )

        assert len(api_results) > 0

        # Test data science pattern
        ml_results = rag_manager.find_similar_functions(
            "train_model",
            "def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf') -> Model:",
            limit=5,
        )

        assert len(ml_results) > 0

    def test_generator_pattern_extraction(self, rag_manager):
        """Test that generators can extract patterns from new examples."""
        # Create generators with RAG enhancement
        param_gen = ParameterSuggestionGenerator(rag_corpus=rag_manager)
        return_gen = ReturnSuggestionGenerator(rag_corpus=rag_manager)
        example_gen = ExampleSuggestionGenerator(rag_corpus=rag_manager)

        # Test async pattern extraction
        async_signature = "async def process_batch(items: list[str], max_concurrent: int = 10) -> list[Result]:"

        # Each generator should be able to find relevant patterns
        param_patterns = param_gen._extract_patterns_from_similar_functions(
            "process_batch", async_signature
        )
        assert len(param_patterns) > 0

        return_patterns = return_gen._extract_patterns_from_similar_functions(
            "process_batch", async_signature
        )
        assert len(return_patterns) > 0

        # Test REST API pattern extraction
        api_signature = "def create_resource(data: ResourceSchema, user: User = Depends(get_current_user)) -> ResourceResponse:"

        example_patterns = example_gen._extract_patterns_from_similar_functions(
            "create_resource", api_signature
        )
        assert len(example_patterns) > 0

    def test_retrieval_performance(self, rag_manager):
        """Test that retrieval performance remains under 100ms."""
        test_queries = [
            ("async_function", "async def async_function() -> None:"),
            ("api_endpoint", "def api_endpoint(request: Request) -> Response:"),
            ("train_model", "def train_model(X: np.ndarray, y: np.ndarray) -> Model:"),
            ("rate_limit", "def rate_limit(calls: int = 10) -> Callable:"),
        ]

        for func_name, signature in test_queries:
            start_time = time.time()
            results = rag_manager.find_similar_functions(func_name, signature, limit=5)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms

            assert (
                elapsed < 100
            ), f"Retrieval for '{func_name}' took {elapsed:.2f}ms (>100ms)"
            assert len(results) > 0, f"No results found for '{func_name}'"

    def test_memory_usage_reasonable(self, rag_manager):
        """Test that memory usage remains reasonable with expanded corpus."""
        import tracemalloc

        tracemalloc.start()

        # Perform multiple retrievals
        for i in range(100):
            rag_manager.find_similar_functions(
                f"test_func_{i}", f"def test_func_{i}(x: int) -> str:", limit=5
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 500MB for 100 queries)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 500, f"Peak memory usage too high: {peak_mb:.2f}MB"

    def test_category_filtering(self, rag_manager):
        """Test filtering by category if implemented."""
        # Check if we can access examples by category
        all_examples = []

        # This assumes the corpus has a way to access examples
        # If not implemented, this test can be skipped
        if hasattr(rag_manager, "_curated_examples"):
            all_examples = rag_manager._curated_examples
        elif (
            hasattr(rag_manager, "corpus_data") and "curated" in rag_manager.corpus_data
        ):
            all_examples = rag_manager.corpus_data["curated"]

        if all_examples:
            categories = {}
            for ex in all_examples:
                if "category" in ex:
                    cat = ex["category"]
                    categories[cat] = categories.get(cat, 0) + 1

            # Verify categories are present
            expected_categories = {
                "async_patterns",
                "rest_api_patterns",
                "data_science_patterns",
                "advanced_patterns",
            }

            for cat in expected_categories:
                assert cat in categories, f"Category '{cat}' not found in corpus"
                assert categories[cat] > 0, f"No examples for category '{cat}'"

    def test_quality_score_influence(self, rag_manager):
        """Test that quality scores influence retrieval results."""
        # High-quality examples (score=5) should generally rank higher
        results = rag_manager.find_similar_functions(
            "validate_input", "def validate_input(data: dict) -> bool:", limit=10
        )

        if results:
            # Extract quality scores from results
            quality_scores = []
            for result in results:
                if "metadata" in result and "quality_score" in result["metadata"]:
                    quality_scores.append(result["metadata"]["quality_score"])

            if quality_scores:
                # Average quality score of top results should be high
                avg_quality = sum(quality_scores) / len(quality_scores)
                assert (
                    avg_quality >= 3.5
                ), f"Average quality score too low: {avg_quality}"

    def test_diverse_pattern_coverage(self, rag_manager):
        """Test that diverse patterns are covered in retrieval results."""
        # Test various programming patterns
        patterns_to_test = [
            # Async patterns
            ("async_generator", "async def async_generator() -> AsyncIterator[str]:"),
            ("async_context", "async def async_context() -> AsyncContextManager:"),
            # REST patterns
            (
                "get_items",
                "def get_items(skip: int = 0, limit: int = 100) -> list[Item]:",
            ),
            ("delete_resource", "def delete_resource(resource_id: str) -> None:"),
            # Data science patterns
            (
                "preprocess_data",
                "def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:",
            ),
            (
                "evaluate_model",
                "def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:",
            ),
            # Advanced patterns
            ("cached_property", "def cached_property(func: Callable) -> property:"),
            ("retry_decorator", "def retry_decorator(retries: int = 3) -> Callable:"),
        ]

        pattern_coverage = {}

        for func_name, signature in patterns_to_test:
            results = rag_manager.find_similar_functions(func_name, signature, limit=3)
            pattern_coverage[func_name] = len(results) > 0

        # Should find relevant examples for most patterns
        covered = sum(pattern_coverage.values())
        total = len(patterns_to_test)
        coverage_rate = covered / total

        assert (
            coverage_rate >= 0.7
        ), f"Pattern coverage too low: {covered}/{total} ({coverage_rate:.1%})"
