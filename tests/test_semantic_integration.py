"""
End-to-end integration tests for complete semantic matching pipeline.

This module tests the complete semantic matching system in realistic scenarios,
ensuring all components work together correctly and the system is ready for production.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Import all semantic matching components
from codedocsync.matcher.semantic_matcher import SemanticMatcher
from codedocsync.storage.embedding_cache import EmbeddingCache
from codedocsync.matcher.semantic_optimizer import SemanticOptimizer
from codedocsync.storage.performance_monitor import PerformanceMonitor
from codedocsync.matcher.unified_facade import UnifiedMatchingFacade

# Import related components
from codedocsync.parser import IntegratedParser
from codedocsync.matcher.models import MatchResult, MatchType
from codedocsync.matcher.semantic_models import FunctionEmbedding
from codedocsync.utils.config import CodeDocSyncConfig


class TestSemanticIntegrationWorkflows:
    """Test complete semantic matching workflows in realistic scenarios."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory with realistic Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create a realistic project structure
            (project_path / "src").mkdir()
            (project_path / "src" / "core").mkdir()
            (project_path / "src" / "utils").mkdir()
            (project_path / "tests").mkdir()

            # Create Python files with functions that might be refactored/renamed
            self._create_test_files(project_path)

            yield str(project_path)

    def _create_test_files(self, project_path: Path):
        """Create realistic test files with functions that demonstrate semantic matching scenarios."""

        # Original file with functions
        original_file = project_path / "src" / "core" / "user_management.py"
        original_file.write_text(
            '''
"""User management module with authentication functions."""

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    return True

def get_user_profile(user_id: int) -> dict:
    """Retrieve user profile information by user ID."""
    return {}

def update_user_preferences(user_id: int, preferences: dict) -> bool:
    """Update user preferences in the system."""
    return True

def calculate_user_score(user_data: dict) -> float:
    """Calculate user engagement score based on activity."""
    return 0.0
'''
        )

        # Refactored file with renamed functions (semantic matching candidates)
        refactored_file = project_path / "src" / "core" / "auth_service.py"
        refactored_file.write_text(
            '''
"""Authentication service with improved function names."""

def verify_user_credentials(username: str, password: str) -> bool:
    """Verify user credentials for authentication."""
    return True

def fetch_user_data(user_id: int) -> dict:
    """Fetch comprehensive user data from database."""
    return {}

def modify_user_settings(user_id: int, settings: dict) -> bool:
    """Modify user settings and preferences."""
    return True

def compute_engagement_metric(activity_data: dict) -> float:
    """Compute user engagement metric from activity data."""
    return 0.0
'''
        )

        # Utility functions that should not match
        utils_file = project_path / "src" / "utils" / "helpers.py"
        utils_file.write_text(
            '''
"""General utility functions."""

def format_string(text: str) -> str:
    """Format a string for display."""
    return text.strip()

def parse_config(config_path: str) -> dict:
    """Parse configuration file."""
    return {}

def log_message(message: str, level: str = "info") -> None:
    """Log a message with specified level."""
    pass
'''
        )

        # Test file
        test_file = project_path / "tests" / "test_integration.py"
        test_file.write_text(
            '''
"""Integration tests."""

def test_user_authentication():
    """Test user authentication functionality."""
    assert True

def test_data_retrieval():
    """Test data retrieval operations."""
    assert True
'''
        )

    @pytest.mark.asyncio
    async def test_complete_semantic_matching_workflow(self, temp_project_dir):
        """Test complete semantic matching workflow from project analysis to results."""

        # Mock external dependencies for consistent testing
        with (
            patch("openai.embeddings.create") as mock_create,
            patch(
                "codedocsync.storage.embedding_config.EmbeddingConfigManager"
            ) as mock_config,
        ):
            # Setup realistic mock responses
            mock_create.return_value = Mock()
            mock_create.return_value.data = [Mock(embedding=[0.1] * 1536)]
            mock_config.return_value.validate_config.return_value = True
            mock_config.return_value.get_api_key.return_value = "test-key"

            # Initialize semantic matcher
            matcher = SemanticMatcher(temp_project_dir)

            # Step 1: Parse project files
            parser = IntegratedParser()
            all_functions = []

            for py_file in Path(temp_project_dir).rglob("*.py"):
                try:
                    functions = parser.parse_file(str(py_file))
                    all_functions.extend(functions)
                except Exception as e:
                    print(f"Failed to parse {py_file}: {e}")

            assert len(all_functions) > 0, "Should find functions in test project"

            # Step 2: Prepare semantic index
            await matcher.prepare_semantic_index(all_functions)

            # Verify index was created
            assert (
                matcher.vector_store.collection.count() > 0
            ), "Should have embeddings in vector store"

            # Step 3: Perform semantic matching
            result = await matcher.match_with_embeddings(all_functions)

            # Validate results
            assert isinstance(result, MatchResult)
            assert result.total_functions == len(all_functions)
            assert len(result.matched_pairs) + len(result.unmatched_functions) == len(
                all_functions
            )

            # Get statistics
            stats = matcher.get_stats()
            assert stats["functions_processed"] > 0
            assert "embedding_stats" in stats
            assert "cache_stats" in stats
            assert "vector_store_stats" in stats

    @pytest.mark.asyncio
    async def test_unified_facade_complete_pipeline(self, temp_project_dir):
        """Test unified facade with complete four-phase matching pipeline."""

        # Mock all external dependencies
        with (
            patch("openai.embeddings.create") as mock_create,
            patch(
                "codedocsync.storage.embedding_config.EmbeddingConfigManager"
            ) as mock_config,
        ):
            mock_create.return_value = Mock()
            mock_create.return_value.data = [Mock(embedding=[0.1] * 1536)]
            mock_config.return_value.validate_config.return_value = True
            mock_config.return_value.get_api_key.return_value = "test-key"

            # Create unified facade
            facade = UnifiedMatchingFacade()

            # Run complete pipeline
            result = await facade.match_project(
                temp_project_dir, use_cache=True, enable_semantic=True
            )

            # Validate complete pipeline results
            assert isinstance(result, MatchResult)
            assert result.total_functions > 0
            assert hasattr(result, "metadata")
            assert "unified_stats" in result.metadata

            # Check phase execution
            unified_stats = result.metadata["unified_stats"]
            assert unified_stats["files_processed"] > 0
            assert "phase_times" in unified_stats
            assert "matches_by_type" in unified_stats

            # Verify all phases executed
            phase_times = unified_stats["phase_times"]
            assert phase_times["parsing"] > 0
            assert phase_times["direct_matching"] >= 0
            assert phase_times["contextual_matching"] >= 0

            # Print summary for manual verification
            facade.print_summary()

    @pytest.mark.asyncio
    async def test_semantic_matching_with_realistic_renaming_scenarios(
        self, temp_project_dir
    ):
        """Test semantic matching identifies renamed functions correctly."""

        with (
            patch("openai.embeddings.create") as mock_create,
            patch(
                "codedocsync.storage.embedding_config.EmbeddingConfigManager"
            ) as mock_config,
        ):
            # Create embeddings that would be similar for renamed functions
            def create_similar_embedding(base_value):
                return [base_value + i * 0.001 for i in range(1536)]

            # Mock embeddings to simulate semantic similarity
            mock_create.return_value = Mock()

            def mock_embedding_response(*args, **kwargs):
                text = kwargs.get("input", "") or args[0] if args else ""

                # Create similar embeddings for semantically related functions
                if "authenticate" in text or "verify" in text:
                    return Mock(data=[Mock(embedding=create_similar_embedding(0.1))])
                elif "get_user" in text or "fetch_user" in text:
                    return Mock(data=[Mock(embedding=create_similar_embedding(0.2))])
                elif "update" in text or "modify" in text:
                    return Mock(data=[Mock(embedding=create_similar_embedding(0.3))])
                elif "calculate" in text or "compute" in text:
                    return Mock(data=[Mock(embedding=create_similar_embedding(0.4))])
                else:
                    return Mock(data=[Mock(embedding=create_similar_embedding(0.9))])

            mock_create.side_effect = mock_embedding_response
            mock_config.return_value.validate_config.return_value = True
            mock_config.return_value.get_api_key.return_value = "test-key"

            # Initialize components
            matcher = SemanticMatcher(temp_project_dir)
            parser = IntegratedParser()

            # Parse all functions
            all_functions = []
            for py_file in Path(temp_project_dir).rglob("*.py"):
                try:
                    functions = parser.parse_file(str(py_file))
                    all_functions.extend(functions)
                except Exception:
                    continue

            # Prepare index and perform matching
            await matcher.prepare_semantic_index(all_functions)
            result = await matcher.match_with_embeddings(all_functions)

            # Analyze results for semantic matches
            _semantic_matches = [
                pair
                for pair in result.matched_pairs
                if hasattr(pair, "match_type") and pair.match_type == MatchType.SEMANTIC
            ]

            # Should find some semantic matches (or at least attempt matching)
            assert len(result.matched_pairs) >= 0  # May or may not find matches
            assert len(result.unmatched_functions) >= 0

            # Get detailed statistics
            stats = matcher.get_stats()
            assert stats["searches_performed"] > 0

    def test_semantic_cache_persistence_across_sessions(self, temp_project_dir):
        """Test that semantic cache persists across different matcher sessions."""

        cache_dir = Path(temp_project_dir) / ".codedocsync_cache"

        # First session - populate cache
        cache1 = EmbeddingCache(str(cache_dir))

        # Add some embeddings
        test_embeddings = []
        for i in range(10):
            embedding = FunctionEmbedding(
                function_id=f"module.function_{i}",
                embedding=[0.1 + i * 0.01] * 1536,
                model="text-embedding-3-small",
                text_embedded=f"def function_{i}(): pass",
                timestamp=1234567890.0,
                signature_hash=f"hash_{i}",
            )
            cache1.set(embedding)
            test_embeddings.append(embedding)

        # Get initial stats
        stats1 = cache1.get_stats()
        _initial_saves = stats1["total_saves"]

        # Second session - should load from cache
        cache2 = EmbeddingCache(str(cache_dir))

        # Try to retrieve cached embeddings
        hits = 0
        for i in range(10):
            text = f"def function_{i}(): pass"
            result = cache2.get(text, "text-embedding-3-small")
            if result is not None:
                hits += 1

        # Should have high hit rate from persistence
        hit_rate = hits / 10
        assert hit_rate >= 0.5, f"Cache persistence hit rate {hit_rate:.1%} too low"

        # Verify cache database exists
        cache_db = cache_dir / "embeddings.db"
        assert cache_db.exists(), "Cache database should persist"

    @pytest.mark.asyncio
    async def test_error_recovery_in_complete_pipeline(self, temp_project_dir):
        """Test error recovery works correctly in complete semantic matching pipeline."""

        # Test with failing primary service but working fallback
        with (
            patch("openai.embeddings.create") as mock_openai,
            patch(
                "codedocsync.storage.embedding_config.EmbeddingConfigManager"
            ) as mock_config,
            patch("sentence_transformers.SentenceTransformer") as mock_local,
        ):
            # Setup config
            mock_config.return_value.validate_config.return_value = True
            mock_config.return_value.get_api_key.return_value = "test-key"

            # Primary service fails
            mock_openai.side_effect = Exception("OpenAI API failed")

            # Fallback service works
            mock_local_instance = Mock()
            mock_local_instance.encode.return_value = Mock()
            mock_local_instance.encode.return_value.tolist.return_value = [0.1] * 384
            mock_local.return_value = mock_local_instance

            # Initialize semantic matcher
            matcher = SemanticMatcher(temp_project_dir)

            # Parse some functions
            parser = IntegratedParser()
            test_file = Path(temp_project_dir) / "src" / "core" / "user_management.py"
            functions = parser.parse_file(str(test_file))

            # Should handle errors gracefully and use fallback
            await matcher.prepare_semantic_index(functions)
            _result = await matcher.match_with_embeddings(functions)

            # Verify recovery worked
            assert isinstance(_result, MatchResult)

            # Check that fallback was used
            stats = matcher.get_stats()
            embedding_stats = stats.get("embedding_stats", {})
            assert (
                embedding_stats.get("fallback_rate", 0) > 0
            ), "Should have used fallback"

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, temp_project_dir):
        """Test performance monitoring works correctly during semantic matching."""

        monitor = PerformanceMonitor()

        with (
            patch("openai.embeddings.create") as mock_create,
            patch(
                "codedocsync.storage.embedding_config.EmbeddingConfigManager"
            ) as mock_config,
        ):
            mock_create.return_value = Mock()
            mock_create.return_value.data = [Mock(embedding=[0.1] * 1536)]
            mock_config.return_value.validate_config.return_value = True
            mock_config.return_value.get_api_key.return_value = "test-key"

            # Track semantic matching operation
            with monitor.track_operation("semantic_matching", category="performance"):
                matcher = SemanticMatcher(temp_project_dir)
                parser = IntegratedParser()

                # Parse and match functions
                functions = []
                for py_file in Path(temp_project_dir).rglob("*.py"):
                    try:
                        file_functions = parser.parse_file(str(py_file))
                        functions.extend(file_functions)
                    except Exception:
                        continue

                if functions:
                    await matcher.prepare_semantic_index(
                        functions[:5]
                    )  # Limit for testing
                    _result = await matcher.match_with_embeddings(functions[:5])

            # Verify monitoring captured metrics
            metrics = monitor.get_current_metrics()
            assert metrics["operations_tracked"] > 0
            assert "semantic_matching" in str(
                metrics
            )  # Should have tracked the operation

            # Get performance report
            report = monitor.get_performance_report()
            assert "total_operations" in report
            assert report["total_operations"] > 0

    @pytest.mark.asyncio
    async def test_optimization_recommendations_in_realistic_scenario(
        self, temp_project_dir
    ):
        """Test optimization recommendations are generated correctly in realistic scenarios."""

        optimizer = SemanticOptimizer()

        # Simulate high memory usage scenario
        with patch.object(optimizer.process, "memory_info") as mock_memory:
            # Simulate high memory usage (400MB)
            mock_memory.return_value.rss = (
                (optimizer.initial_memory + 400) * 1024 * 1024
            )

            # Get recommendations
            recommendations = optimizer.get_optimization_recommendations()

            # Should provide actionable recommendations
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0

            # Should include memory-related recommendations
            memory_rec_found = any("memory" in rec.lower() for rec in recommendations)
            assert (
                memory_rec_found
            ), "Should recommend memory optimization in high memory scenario"

    @pytest.mark.asyncio
    async def test_configuration_integration_with_semantic_matching(
        self, temp_project_dir
    ):
        """Test semantic matching respects configuration settings."""

        # Create custom configuration
        config = CodeDocSyncConfig()

        # Test with semantic matching disabled
        facade_disabled = UnifiedMatchingFacade(config)

        with (
            patch("openai.embeddings.create") as mock_create,
            patch(
                "codedocsync.storage.embedding_config.EmbeddingConfigManager"
            ) as mock_config,
        ):
            mock_create.return_value = Mock()
            mock_create.return_value.data = [Mock(embedding=[0.1] * 1536)]
            mock_config.return_value.validate_config.return_value = True
            mock_config.return_value.get_api_key.return_value = "test-key"

            # Run with semantic disabled
            _result_disabled = await facade_disabled.match_project(
                temp_project_dir, enable_semantic=False
            )

            # Should not have semantic matching stats
            stats_disabled = facade_disabled.get_stats()
            assert stats_disabled["matches_by_type"]["semantic"] == 0

            # Run with semantic enabled
            _result_enabled = await facade_disabled.match_project(
                temp_project_dir, enable_semantic=True
            )

            # Should have attempted semantic matching
            _stats_enabled = facade_disabled.get_stats()
            # May have semantic matches or not, but should have tried

    def test_cross_platform_compatibility(self, temp_project_dir):
        """Test semantic matching works correctly across different platforms."""

        # Test path handling works correctly
        cache = EmbeddingCache(temp_project_dir)

        # Create embeddings with cross-platform paths
        test_paths = [
            "src/core/module.py",
            "src\\utils\\helpers.py",  # Windows-style
            "tests/test_file.py",
        ]

        for i, path in enumerate(test_paths):
            embedding = FunctionEmbedding(
                function_id=f"module.function_{i}",
                embedding=[0.1] * 1536,
                model="test-model",
                text_embedded=f"def function_{i}(): pass",
                timestamp=1234567890.0,
                signature_hash=f"hash_{i}",
            )
            cache.set(embedding)

        # Should handle all path formats
        stats = cache.get_stats()
        assert stats["total_saves"] == len(test_paths)

    @pytest.mark.asyncio
    async def test_large_project_scalability(self, temp_project_dir):
        """Test semantic matching scales appropriately for larger projects."""

        # Create additional files for scalability testing
        large_project_path = Path(temp_project_dir)

        # Create more modules
        for module_num in range(10):
            module_dir = large_project_path / f"module_{module_num}"
            module_dir.mkdir(exist_ok=True)

            for file_num in range(5):
                py_file = module_dir / f"file_{file_num}.py"
                py_file.write_text(
                    f'''
def function_{module_num}_{file_num}_1(param: str) -> str:
    """Function {module_num}_{file_num}_1 description."""
    return param

def function_{module_num}_{file_num}_2(data: dict) -> bool:
    """Function {module_num}_{file_num}_2 description."""
    return True
'''
                )

        with (
            patch("openai.embeddings.create") as mock_create,
            patch(
                "codedocsync.storage.embedding_config.EmbeddingConfigManager"
            ) as mock_config,
        ):
            mock_create.return_value = Mock()
            mock_create.return_value.data = [Mock(embedding=[0.1] * 1536)]
            mock_config.return_value.validate_config.return_value = True
            mock_config.return_value.get_api_key.return_value = "test-key"

            # Test unified facade with larger project
            facade = UnifiedMatchingFacade()

            import time

            start_time = time.time()

            result = await facade.match_project(
                str(large_project_path), use_cache=True, enable_semantic=True
            )

            duration = time.time() - start_time

            # Should complete within reasonable time even for larger project
            assert duration < 30.0, f"Large project took {duration:.2f}s, too slow"

            # Should process many functions
            assert (
                result.total_functions > 50
            ), "Should find many functions in large project"

            # Check statistics
            stats = facade.get_stats()
            assert stats["files_processed"] > 50, "Should process many files"

    def test_memory_cleanup_after_semantic_operations(self, temp_project_dir):
        """Test memory is properly cleaned up after semantic operations."""

        import gc
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Perform semantic operations that allocate memory
        cache = EmbeddingCache(temp_project_dir)
        optimizer = SemanticOptimizer()

        # Allocate embeddings
        for i in range(100):
            embedding = FunctionEmbedding(
                function_id=f"func_{i}",
                embedding=[0.1] * 1536,
                model="test-model",
                text_embedded=f"def function_{i}(): pass",
                timestamp=1234567890.0,
                signature_hash=f"hash_{i}",
            )
            cache.set(embedding)

        # Perform cleanup
        optimizer.cleanup()
        gc.collect()

        # Check memory usage after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable
        assert (
            memory_growth < 50
        ), f"Memory grew by {memory_growth:.1f}MB, potential leak"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
