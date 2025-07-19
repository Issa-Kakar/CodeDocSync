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
from codedocsync.analyzer.integration import (
    _should_use_llm,
    _determine_analysis_types,
    _create_llm_request,
    _merge_results,
)
from codedocsync.analyzer.models import (
    AnalysisResult,
    InconsistencyIssue,
    RuleCheckResult,
)
from codedocsync.analyzer.llm_models import LLMAnalysisRequest
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


class TestHelperFunctions:
    """Test the helper functions in integration module."""

    def create_test_data(self):
        """Create test data for helper function tests."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="calculate_total",
                parameters=[
                    FunctionParameter(
                        name="items",
                        type_annotation="List[int]",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_annotation="int",
                is_async=False,
                decorators=["@cached_property"],
            ),
            docstring=None,
            file_path="/test/calculations.py",
            line_number=10,
            body="if len(items) == 0:\n    return 0\nreturn sum(items)",
        )

        docstring = ParsedDocstring(
            format="google",
            summary="Calculate total of items",
            parameters=[
                DocstringParameter(
                    name="items", description="List of integers", type="List[int]"
                )
            ],
            returns=DocstringReturns(description="Total sum", type="int"),
            raises=[],
            raw_text="Calculate total of items.\n\nArgs:\n    items: List of integers\n\nReturns:\n    Total sum",
            examples=[">>> calculate_total([1, 2, 3])\n6"],
        )

        pair = MatchedPair(
            function=function,
            documentation=docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Direct match",
        )

        return function, docstring, pair

    def test_should_use_llm_low_confidence(self):
        """Test _should_use_llm with low confidence rules."""
        _, _, pair = self.create_test_data()
        config = get_development_config()

        # Create low confidence rule result
        rule_results = [
            RuleCheckResult(
                rule_name="parameter_type_check",
                passed=False,
                confidence=0.7,  # Low confidence
                issues=[],
                execution_time_ms=1.0,
            )
        ]

        assert _should_use_llm(rule_results, config, pair) is True

    def test_should_use_llm_high_confidence(self):
        """Test _should_use_llm with high confidence rules."""
        _, _, pair = self.create_test_data()
        config = get_development_config()

        # Create high confidence rule result
        rule_results = [
            RuleCheckResult(
                rule_name="parameter_name_check",
                passed=True,
                confidence=0.95,  # High confidence
                issues=[],
                execution_time_ms=1.0,
            )
        ]

        assert _should_use_llm(rule_results, config, pair) is False

    def test_should_use_llm_with_examples(self):
        """Test _should_use_llm when docstring has examples."""
        _, _, pair = self.create_test_data()
        config = get_development_config()

        # High confidence rules but docstring has examples
        rule_results = [
            RuleCheckResult(
                rule_name="all_checks",
                passed=True,
                confidence=0.95,
                issues=[],
                execution_time_ms=1.0,
            )
        ]

        # Should use LLM because docstring has examples
        assert _should_use_llm(rule_results, config, pair) is True

    def test_should_use_llm_with_decorators(self):
        """Test _should_use_llm with behavior-affecting decorators."""
        _, _, pair = self.create_test_data()
        config = get_development_config()

        rule_results = [
            RuleCheckResult(
                rule_name="all_checks",
                passed=True,
                confidence=0.95,
                issues=[],
                execution_time_ms=1.0,
            )
        ]

        # Should use LLM because function has @cached_property decorator
        assert _should_use_llm(rule_results, config, pair) is True

    def test_should_use_llm_disabled(self):
        """Test _should_use_llm when LLM is disabled in config."""
        _, _, pair = self.create_test_data()
        config = get_development_config()
        config.use_llm = False

        rule_results = [
            RuleCheckResult(
                rule_name="test",
                passed=False,
                confidence=0.5,  # Low confidence
                issues=[],
                execution_time_ms=1.0,
            )
        ]

        # Should not use LLM when disabled
        assert _should_use_llm(rule_results, config, pair) is False

    def test_determine_analysis_types_basic(self):
        """Test _determine_analysis_types for basic function."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="simple_func",
                parameters=[],
                return_annotation=None,
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/file.py",
            line_number=1,
        )

        pair = MatchedPair(
            function=function,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test",
        )

        types = _determine_analysis_types(pair, [])

        # Should default to behavior analysis
        assert types == ["behavior"]

    def test_determine_analysis_types_with_examples(self):
        """Test _determine_analysis_types when docstring has examples."""
        _, _, pair = self.create_test_data()

        types = _determine_analysis_types(pair, [])

        # Should include examples analysis
        assert "examples" in types

    def test_determine_analysis_types_with_conditionals(self):
        """Test _determine_analysis_types for functions with conditionals."""
        _, _, pair = self.create_test_data()

        types = _determine_analysis_types(pair, [])

        # Should include edge_cases analysis due to if statement in body
        assert "edge_cases" in types

    def test_determine_analysis_types_with_deprecated(self):
        """Test _determine_analysis_types for deprecated functions."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="old_func",
                parameters=[],
                return_annotation=None,
                is_async=False,
                decorators=["@deprecated"],
            ),
            docstring=None,
            file_path="/test/file.py",
            line_number=1,
        )

        pair = MatchedPair(
            function=function,
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test",
        )

        types = _determine_analysis_types(pair, [])

        # Should include version_info analysis
        assert "version_info" in types

    def test_create_llm_request(self):
        """Test _create_llm_request function."""
        _, _, pair = self.create_test_data()
        config = get_development_config()

        rule_results = [
            RuleCheckResult(
                rule_name="test_rule",
                passed=False,
                confidence=0.8,
                issues=[],
                execution_time_ms=1.0,
            )
        ]

        request = _create_llm_request(pair, rule_results, config)

        assert isinstance(request, LLMAnalysisRequest)
        assert request.function == pair.function
        assert request.docstring == pair.documentation
        assert len(request.analysis_types) > 0
        assert request.rule_results == rule_results
        assert isinstance(request.related_functions, list)

    def test_create_llm_request_with_config_override(self):
        """Test _create_llm_request with config analysis types override."""
        _, _, pair = self.create_test_data()
        config = get_development_config()
        config.llm_analysis_types = ["behavior", "performance"]

        rule_results = []

        request = _create_llm_request(pair, rule_results, config)

        # Should use config override
        assert request.analysis_types == ["behavior", "performance"]

    def test_merge_results_no_duplicates(self):
        """Test _merge_results with no duplicates."""
        rule_issues = [
            InconsistencyIssue(
                issue_type="parameter_missing",
                severity="critical",
                description="Parameter 'x' is missing",
                suggestion="Add parameter 'x'",
                line_number=10,
                confidence=0.95,
            )
        ]

        llm_issues = [
            InconsistencyIssue(
                issue_type="edge_case_missing",
                severity="medium",
                description="Edge case not documented",
                suggestion="Document empty list case",
                line_number=15,
                confidence=0.85,
            )
        ]

        merged = _merge_results(rule_issues, llm_issues)

        assert len(merged) == 2
        assert merged[0].severity == "critical"  # Higher severity first
        assert merged[1].severity == "medium"

    def test_merge_results_with_duplicates(self):
        """Test _merge_results with duplicate issues."""
        rule_issues = [
            InconsistencyIssue(
                issue_type="parameter_type_mismatch",
                severity="high",
                description="Type mismatch",
                suggestion="Fix type to int",
                line_number=10,
                confidence=0.85,
            )
        ]

        llm_issues = [
            InconsistencyIssue(
                issue_type="parameter_type_mismatch",
                severity="high",
                description="Type mismatch detected",
                suggestion="Change type annotation to int",
                line_number=10,
                confidence=0.9,
            )
        ]

        merged = _merge_results(rule_issues, llm_issues)

        # Should merge into one issue
        assert len(merged) == 1
        assert merged[0].confidence == 0.9  # Higher confidence wins
        assert "Alternatively:" in merged[0].suggestion  # Suggestions combined

    def test_merge_results_sorting(self):
        """Test _merge_results sorting by severity and line number."""
        issues = [
            InconsistencyIssue(
                issue_type="test1",
                severity="low",
                description="Low severity",
                suggestion="Fix",
                line_number=20,
                confidence=0.8,
            ),
            InconsistencyIssue(
                issue_type="test2",
                severity="critical",
                description="Critical issue",
                suggestion="Fix now",
                line_number=10,
                confidence=0.9,
            ),
            InconsistencyIssue(
                issue_type="test3",
                severity="critical",
                description="Another critical",
                suggestion="Fix now",
                line_number=5,
                confidence=0.9,
            ),
        ]

        merged = _merge_results(issues, [])

        # Should be sorted by severity (critical first) then line number
        assert merged[0].line_number == 5  # First critical (lower line)
        assert merged[1].line_number == 10  # Second critical
        assert merged[2].severity == "low"  # Low severity last


class TestIntegrationFlow:
    """Test the complete integration flow with all components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_llm(self):
        """Test complete pipeline with rule engine and LLM."""
        # Create function with issues that trigger LLM
        function = ParsedFunction(
            signature=FunctionSignature(
                name="complex_calculation",
                parameters=[
                    FunctionParameter(
                        name="data",
                        type_annotation="Dict[str, Any]",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_annotation="float",
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/calc.py",
            line_number=25,
            complexity=15,  # High complexity
        )

        docstring = ParsedDocstring(
            format="google",
            summary="Perform complex calculation",
            parameters=[
                DocstringParameter(
                    name="data",
                    description="Input data",
                    type="dict",  # Type mismatch
                )
            ],
            returns=DocstringReturns(
                description="Calculated result",
                type="float",
            ),
            raises=[],
            raw_text="Docstring text",
        )

        pair = MatchedPair(
            function=function,
            documentation=docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Direct match",
        )

        config = get_development_config()
        config.use_llm = True

        # Mock LLM analyzer
        mock_llm = Mock()
        mock_llm.analyze_function = AsyncMock(
            return_value=Mock(
                issues=[
                    InconsistencyIssue(
                        issue_type="behavior_inconsistency",
                        severity="medium",
                        description="Function behavior not fully documented",
                        suggestion="Add description of edge cases",
                        line_number=25,
                        confidence=0.85,
                    )
                ]
            )
        )

        result = await analyze_matched_pair(pair, config=config, llm_analyzer=mock_llm)

        assert result.used_llm is True
        assert len(result.issues) >= 2  # Rule issues + LLM issues

        # Check that issues are properly merged and sorted
        critical_issues = [i for i in result.issues if i.severity == "critical"]
        medium_issues = [i for i in result.issues if i.severity == "medium"]

        # Critical issues should come before medium issues
        if critical_issues and medium_issues:
            assert result.issues.index(critical_issues[0]) < result.issues.index(
                medium_issues[0]
            )

    @pytest.mark.asyncio
    async def test_pipeline_skips_llm_for_high_confidence(self):
        """Test that pipeline skips LLM for high-confidence critical issues."""
        # Create function with critical parameter mismatch
        function = ParsedFunction(
            signature=FunctionSignature(
                name="process_data",
                parameters=[
                    FunctionParameter(
                        name="user_id",
                        type_annotation="int",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_annotation=None,
                is_async=False,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/process.py",
            line_number=10,
        )

        docstring = ParsedDocstring(
            format="google",
            summary="Process user data",
            parameters=[
                DocstringParameter(
                    name="userId",  # Name mismatch - critical issue
                    description="User identifier",
                    type="int",
                )
            ],
            returns=None,
            raises=[],
            raw_text="Process user data",
        )

        pair = MatchedPair(
            function=function,
            documentation=docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Direct match",
        )

        config = get_development_config()
        config.use_llm = True

        # Mock LLM analyzer
        mock_llm = Mock()
        mock_llm.analyze_function = AsyncMock()

        result = await analyze_matched_pair(pair, config=config, llm_analyzer=mock_llm)

        # Should not use LLM due to high-confidence critical issue
        assert result.used_llm is False
        mock_llm.analyze_function.assert_not_called()

        # Should still have the critical issue
        assert len(result.issues) > 0
        assert any(i.severity == "critical" for i in result.issues)
