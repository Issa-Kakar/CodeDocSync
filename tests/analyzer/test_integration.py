"""
Test suite for analyzer integration.

End-to-end tests for the complete analysis pipeline, performance requirements,
and error handling.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock

from codedocsync.analyzer import (
    analyze_matched_pair,
    analyze_multiple_pairs,
    AnalysisCache,
    AnalysisConfig,
    get_development_config,
)
from codedocsync.analyzer.models import AnalysisResult
from codedocsync.matcher import MatchedPair, MatchConfidence, MatchType
from codedocsync.parser import (
    ParsedFunction,
    FunctionSignature,
    FunctionParameter,
    ParsedDocstring,
    DocstringParameter,
    DocstringReturns,
)


class TestAnalysisCache:
    """Test AnalysisCache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with different sizes."""
        cache = AnalysisCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache.cache) == 0

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = AnalysisCache()
        config = get_development_config()

        # Create test data
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func",
                parameters=[],
                return_annotation=None,
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/file.py",
            line_number=10,
        )

        pair = MatchedPair(
            function=function,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

        key1 = cache._generate_key(pair, config)
        key2 = cache._generate_key(pair, config)

        # Same inputs should generate same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 0

    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache."""
        cache = AnalysisCache()
        config = get_development_config()

        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func",
                parameters=[],
                return_annotation=None,
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/file.py",
            line_number=10,
        )

        pair = MatchedPair(
            function=function,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

        result = AnalysisResult(matched_pair=pair)

        # Store in cache
        cache.put(pair, config, result)

        # Retrieve from cache
        cached_result = cache.get(pair, config)

        assert cached_result is not None
        assert cached_result.matched_pair == pair
        assert cached_result.cache_hit

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = AnalysisCache(max_size=2)
        config = get_development_config()

        # Create three different pairs
        pairs = []
        for i in range(3):
            function = ParsedFunction(
                signature=FunctionSignature(
                    name=f"test_func_{i}",
                    parameters=[],
                    return_annotation=None,
                    is_async=False,
                    decorators=[],
                ),
                docstring=None,
                file_path=f"/test/file_{i}.py",
                line_number=10,
            )

            pair = MatchedPair(
                function=function,
                documentation=None,
                confidence=MatchConfidence.HIGH,
                match_type=MatchType.DIRECT,
                match_reason="Test match",
            )
            pairs.append(pair)

        # Store first two (fills cache)
        for i in range(2):
            result = AnalysisResult(matched_pair=pairs[i])
            cache.put(pairs[i], config, result)

        assert len(cache.cache) == 2

        # Store third (should evict first)
        result = AnalysisResult(matched_pair=pairs[2])
        cache.put(pairs[2], config, result)

        assert len(cache.cache) == 2

        # First should be evicted, second and third should remain
        assert cache.get(pairs[0], config) is None
        assert cache.get(pairs[1], config) is not None
        assert cache.get(pairs[2], config) is not None


class TestAnalyzeMatchedPair:
    """Test the main analyze_matched_pair function."""

    def create_test_pair(self, with_issues=False):
        """Create a test matched pair."""
        # Create function with parameter issue if requested
        parameters = []
        doc_parameters = []

        if with_issues:
            # Function parameter: user_id
            parameters.append(
                FunctionParameter(
                    name="user_id",
                    type_annotation="int",
                    default_value=None,
                    is_required=True,
                )
            )
            # Docstring parameter: userId (mismatch)
            doc_parameters.append(
                DocstringParameter(name="userId", description="User ID", type="int")
            )

        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_function",
                parameters=parameters,
                return_annotation="str" if with_issues else None,
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/file.py",
            line_number=10,
        )

        docstring = ParsedDocstring(
            format="google",
            summary="Test function",
            parameters=doc_parameters,
            returns=(
                DocstringReturns(description="Returns string", type="str")
                if with_issues
                else None
            ),
            raises=[],
            raw_text="Test docstring",
        )

        return MatchedPair(
            function=function,
            documentation=docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

    @pytest.mark.asyncio
    async def test_analyze_matched_pair_basic(self):
        """Test basic analyze_matched_pair functionality."""
        pair = self.create_test_pair(with_issues=True)

        result = await analyze_matched_pair(pair)

        assert isinstance(result, AnalysisResult)
        assert result.matched_pair == pair
        assert isinstance(result.issues, list)
        assert result.analysis_time_ms > 0
        assert not result.cache_hit  # First analysis

    @pytest.mark.asyncio
    async def test_analyze_with_cache_hit(self):
        """Test analyze_matched_pair with cache hit."""
        pair = self.create_test_pair()
        cache = AnalysisCache()

        # First analysis (cache miss)
        result1 = await analyze_matched_pair(pair, cache=cache)
        assert not result1.cache_hit

        # Second analysis (cache hit)
        result2 = await analyze_matched_pair(pair, cache=cache)
        assert result2.cache_hit

    @pytest.mark.asyncio
    async def test_analyze_with_custom_config(self):
        """Test analyze_matched_pair with custom configuration."""
        pair = self.create_test_pair()

        config = AnalysisConfig(
            use_llm=False,  # Disable LLM for this test
            rule_engine=get_development_config().rule_engine,
        )

        result = await analyze_matched_pair(pair, config=config)

        assert not result.used_llm
        assert isinstance(result.issues, list)

    @pytest.mark.asyncio
    async def test_analyze_with_rule_engine_issues(self):
        """Test analyze_matched_pair with rule engine detecting issues."""
        pair = self.create_test_pair(with_issues=True)

        config = AnalysisConfig(
            use_llm=False,  # Only use rules to isolate behavior
            rule_engine=get_development_config().rule_engine,
        )

        result = await analyze_matched_pair(pair, config=config)

        # Should detect parameter name mismatch
        assert len(result.issues) > 0
        name_mismatches = [
            i for i in result.issues if i.issue_type == "parameter_name_mismatch"
        ]
        assert len(name_mismatches) > 0

    @pytest.mark.asyncio
    async def test_analyze_with_high_confidence_skip_llm(self):
        """Test that high-confidence rule issues skip LLM analysis."""
        pair = self.create_test_pair(with_issues=True)

        config = AnalysisConfig(
            use_llm=True, rule_engine=get_development_config().rule_engine
        )

        # Mock LLM analyzer to detect if it was called
        llm_analyzer = Mock()
        llm_analyzer.analyze_consistency = AsyncMock()

        result = await analyze_matched_pair(
            pair, config=config, llm_analyzer=llm_analyzer
        )

        # LLM should not be called for high-confidence rule issues
        assert not result.used_llm
        llm_analyzer.analyze_consistency.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_invalid_input(self):
        """Test analyze_matched_pair with invalid input."""
        # Test with None pair
        with pytest.raises(ValueError, match="matched_pair cannot be None"):
            await analyze_matched_pair(None)

        # Test with pair having None function
        invalid_pair = MatchedPair(
            function=None,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

        with pytest.raises(ValueError, match="matched_pair.function cannot be None"):
            await analyze_matched_pair(invalid_pair)

    @pytest.mark.asyncio
    async def test_analyze_with_component_failures(self):
        """Test analyze_matched_pair with component failures."""
        pair = self.create_test_pair()

        # Mock rule engine to fail
        rule_engine = Mock()
        rule_engine.check_matched_pair.side_effect = Exception("Rule engine error")

        # Should continue with empty rule results
        result = await analyze_matched_pair(pair, rule_engine=rule_engine)

        assert isinstance(result, AnalysisResult)
        assert isinstance(result.issues, list)


class TestAnalyzeMultiplePairs:
    """Test the analyze_multiple_pairs function."""

    def create_test_pairs(self, count=3):
        """Create multiple test pairs."""
        pairs = []
        for i in range(count):
            function = ParsedFunction(
                signature=FunctionSignature(
                    name=f"test_function_{i}",
                    parameters=[],
                    return_annotation=None,
                    is_async=False,
                    decorators=[],
                ),
                docstring=None,
                file_path=f"/test/file_{i}.py",
                line_number=10,
            )

            docstring = ParsedDocstring(
                format="google",
                summary=f"Test function {i}",
                parameters=[],
                returns=None,
                raises=[],
                raw_text=f"Test docstring {i}",
            )

            pair = MatchedPair(
                function=function,
                documentation=docstring,
                confidence=MatchConfidence.HIGH,
                match_type=MatchType.DIRECT,
                match_reason="Test match",
            )
            pairs.append(pair)

        return pairs

    @pytest.mark.asyncio
    async def test_analyze_multiple_pairs_sequential(self):
        """Test analyzing multiple pairs sequentially."""
        pairs = self.create_test_pairs(3)

        config = AnalysisConfig(
            parallel_analysis=False,  # Force sequential
            use_llm=False,
        )

        results = await analyze_multiple_pairs(pairs, config=config)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, AnalysisResult)
            assert result.matched_pair == pairs[i]

    @pytest.mark.asyncio
    async def test_analyze_multiple_pairs_parallel(self):
        """Test analyzing multiple pairs in parallel."""
        pairs = self.create_test_pairs(5)

        config = AnalysisConfig(
            parallel_analysis=True, max_parallel_workers=2, batch_size=2, use_llm=False
        )

        start_time = time.time()
        results = await analyze_multiple_pairs(pairs, config=config)
        execution_time = time.time() - start_time

        assert len(results) == 5
        for i, result in enumerate(results):
            assert isinstance(result, AnalysisResult)
            assert result.matched_pair == pairs[i]

        # Parallel should be reasonably fast
        assert execution_time < 2.0  # Should complete in under 2 seconds

    @pytest.mark.asyncio
    async def test_analyze_empty_pairs_list(self):
        """Test analyzing empty list of pairs."""
        results = await analyze_multiple_pairs([])
        assert results == []

    @pytest.mark.asyncio
    async def test_analyze_single_pair_via_multiple(self):
        """Test analyzing single pair via analyze_multiple_pairs."""
        pairs = self.create_test_pairs(1)

        config = AnalysisConfig(use_llm=False)

        results = await analyze_multiple_pairs(pairs, config=config)

        assert len(results) == 1
        assert isinstance(results[0], AnalysisResult)


class TestPerformanceRequirements:
    """Test that performance requirements are met."""

    @pytest.mark.asyncio
    async def test_rule_only_analysis_performance(self):
        """Test that rule-only analysis meets performance targets."""
        pair = TestAnalyzeMatchedPair().create_test_pair(with_issues=True)

        config = AnalysisConfig(
            use_llm=False,  # Rules only
            rule_engine=get_development_config().rule_engine,
        )

        start_time = time.time()
        result = await analyze_matched_pair(pair, config=config)
        execution_time_ms = (time.time() - start_time) * 1000

        # Should complete quickly (rule engine target is <5ms per function)
        assert execution_time_ms < 50  # Allow some overhead for integration
        assert result.analysis_time_ms > 0

    @pytest.mark.asyncio
    async def test_cached_analysis_performance(self):
        """Test that cached analysis is very fast."""
        pair = TestAnalyzeMatchedPair().create_test_pair()
        cache = AnalysisCache()

        config = AnalysisConfig(use_llm=False)

        # First analysis (populate cache)
        await analyze_matched_pair(pair, config=config, cache=cache)

        # Second analysis (from cache)
        start_time = time.time()
        result = await analyze_matched_pair(pair, config=config, cache=cache)
        execution_time_ms = (time.time() - start_time) * 1000

        assert result.cache_hit
        assert execution_time_ms < 10  # Cache access should be very fast

    @pytest.mark.asyncio
    async def test_multiple_pairs_performance(self):
        """Test performance of analyzing multiple pairs."""
        pairs = TestAnalyzeMultiplePairs().create_test_pairs(10)

        config = AnalysisConfig(
            parallel_analysis=True, max_parallel_workers=4, use_llm=False
        )

        start_time = time.time()
        results = await analyze_multiple_pairs(pairs, config=config)
        execution_time = time.time() - start_time

        assert len(results) == 10
        # Should complete reasonably quickly
        assert execution_time < 5.0  # 10 functions in under 5 seconds


class TestConfigurationHandling:
    """Test different configuration scenarios."""

    @pytest.mark.asyncio
    async def test_default_configuration(self):
        """Test analysis with default configuration."""
        pair = TestAnalyzeMatchedPair().create_test_pair()

        result = await analyze_matched_pair(pair)

        assert isinstance(result, AnalysisResult)
        # Default config should use development settings

    @pytest.mark.asyncio
    async def test_fast_configuration(self):
        """Test analysis with fast configuration."""
        pair = TestAnalyzeMatchedPair().create_test_pair()

        from codedocsync.analyzer.config import get_fast_config

        config = get_fast_config()

        result = await analyze_matched_pair(pair, config=config)

        assert isinstance(result, AnalysisResult)
        # Fast config should prefer speed over thoroughness

    @pytest.mark.asyncio
    async def test_thorough_configuration(self):
        """Test analysis with thorough configuration."""
        pair = TestAnalyzeMatchedPair().create_test_pair()

        from codedocsync.analyzer.config import get_thorough_config

        config = get_thorough_config()

        result = await analyze_matched_pair(pair, config=config)

        assert isinstance(result, AnalysisResult)
        # Thorough config should be more comprehensive


class TestErrorRecovery:
    """Test error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_rule_engine_error_recovery(self):
        """Test recovery from rule engine errors."""
        pair = TestAnalyzeMatchedPair().create_test_pair()

        # Mock rule engine to fail
        rule_engine = Mock()
        rule_engine.check_matched_pair.side_effect = Exception("Rule engine failure")

        # Should not crash, should continue with empty rule results
        result = await analyze_matched_pair(pair, rule_engine=rule_engine)

        assert isinstance(result, AnalysisResult)
        assert isinstance(result.issues, list)

    @pytest.mark.asyncio
    async def test_llm_analyzer_error_recovery(self):
        """Test recovery from LLM analyzer errors."""
        pair = TestAnalyzeMatchedPair().create_test_pair()

        config = AnalysisConfig(use_llm=True)

        # Mock LLM analyzer to fail
        llm_analyzer = Mock()
        llm_analyzer.analyze_consistency = AsyncMock(
            side_effect=Exception("LLM failure")
        )

        # Should not crash, should continue with rule results only
        result = await analyze_matched_pair(
            pair, config=config, llm_analyzer=llm_analyzer
        )

        assert isinstance(result, AnalysisResult)
        assert not result.used_llm
