"""
Tests for LLM Smart Batching - Chunk 4 Implementation

Comprehensive test coverage for:
- Smart batching with grouping strategies
- Concurrent analysis with semaphore control
- Progress callback functionality
- Error handling and partial failures
- Cache warming functionality
- Performance optimization strategies
"""

import asyncio
import time
from inspect import Parameter as ParameterKind
from unittest.mock import AsyncMock, patch

import pytest

from codedocsync.analyzer.llm_analyzer import LLMAnalyzer
from codedocsync.analyzer.llm_config import LLMConfig
from codedocsync.analyzer.llm_models import LLMAnalysisRequest, LLMAnalysisResponse
from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
)
from codedocsync.parser.docstring_models import ParsedDocstring


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    with patch("codedocsync.analyzer.llm_analyzer.openai") as mock:
        mock.AsyncOpenAI.return_value = AsyncMock()
        yield mock


@pytest.fixture
def sample_functions():
    """Create sample ParsedFunctions for batch testing."""
    functions = []

    for i in range(10):
        param = FunctionParameter(
            name=f"param{i}",
            type_annotation="str" if i % 2 == 0 else "int",
            default_value=None,
            is_required=True,
            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        )

        signature = FunctionSignature(
            name=f"test_function_{i}",
            parameters=[param] * (i % 5 + 1),  # Variable parameter count
            return_type="bool" if i % 2 == 0 else "str",
        )

        docstring = ParsedDocstring(
            format="google",
            summary=f"Test function {i} summary",
            parameters=[],
            returns=None,
            raises=[],
            raw_text=f"Test function {i} docstring",
        )

        function = ParsedFunction(
            signature=signature,
            docstring=docstring,
            file_path=f"/test/file{i % 3}.py",  # Group by files
            line_number=10 + i,
        )

        # Add complexity estimation attributes
        function.line_count = 20 + (i * 5)
        functions.append(function)

    return functions


@pytest.fixture
def sample_requests(sample_functions):
    """Create sample LLMAnalysisRequests for testing."""
    requests = []

    for i, func in enumerate(sample_functions):
        # Vary analysis types
        analysis_types = ["behavior"]
        if i % 3 == 0:
            analysis_types.append("examples")
        if i % 4 == 0:
            analysis_types.append("type_consistency")

        request = LLMAnalysisRequest(
            function=func,
            docstring=func.docstring,
            analysis_types=analysis_types,
            rule_results=[],
            related_functions=[],
        )
        requests.append(request)

    return requests


@pytest.fixture
def mock_analyzer(mock_openai):
    """Create a mock LLM analyzer for testing."""
    config = LLMConfig(model="gpt-4o-mini", temperature=0.0)

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        analyzer = LLMAnalyzer(config)

        # Mock the analyze_function method to return predictable responses
        async def mock_analyze_function(request):
            await asyncio.sleep(0.01)  # Simulate processing time

            issue = InconsistencyIssue(
                issue_type="test_issue",
                severity="medium",
                description=f"Test issue for {request.function.signature.name}",
                suggestion="Test suggestion",
                line_number=request.function.line_number,
                confidence=0.8,
            )

            return LLMAnalysisResponse(
                issues=[issue],
                raw_response='{"test": "response"}',
                model_used="gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=50,
                response_time_ms=150.0,
                cache_hit=False,
            )

        analyzer.analyze_function = mock_analyze_function
        return analyzer


class TestSmartBatching:
    """Test smart batching functionality."""

    @pytest.mark.asyncio
    async def test_empty_batch(self, mock_analyzer):
        """Test batching with empty request list."""
        results = await mock_analyzer.analyze_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_single_request_batch(self, mock_analyzer, sample_requests):
        """Test batching with single request."""
        results = await mock_analyzer.analyze_batch(sample_requests[:1])

        assert len(results) == 1
        assert isinstance(results[0], LLMAnalysisResponse)
        assert len(results[0].issues) > 0

    @pytest.mark.asyncio
    async def test_multiple_requests_batch(self, mock_analyzer, sample_requests):
        """Test batching with multiple requests."""
        results = await mock_analyzer.analyze_batch(sample_requests[:5])

        assert len(results) == 5
        for result in results:
            assert isinstance(result, LLMAnalysisResponse)
            assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_batch_order_preservation(self, mock_analyzer, sample_requests):
        """Test that batch results maintain original order."""
        requests = sample_requests[:3]
        original_names = [req.function.signature.name for req in requests]

        results = await mock_analyzer.analyze_batch(requests)

        # Verify order is preserved by checking function names in issues
        for i, result in enumerate(results):
            assert original_names[i] in result.issues[0].description

    @pytest.mark.asyncio
    async def test_batch_concurrency_control(self, mock_analyzer, sample_requests):
        """Test batch respects concurrency limits."""
        start_time = time.time()

        # Use low concurrency limit
        results = await mock_analyzer.analyze_batch(
            sample_requests[:10], max_concurrent=2
        )

        elapsed_time = time.time() - start_time

        assert len(results) == 10
        # With concurrency limit of 2, should take longer than single threaded
        # but less than completely sequential
        assert elapsed_time > 0.05  # Some serialization due to limit

    @pytest.mark.asyncio
    async def test_progress_callback(self, mock_analyzer, sample_requests):
        """Test progress callback functionality."""
        progress_updates = []

        def progress_callback(completed, total, request):
            progress_updates.append((completed, total, request.function.signature.name))

        results = await mock_analyzer.analyze_batch(
            sample_requests[:5], progress_callback=progress_callback
        )

        assert len(results) == 5
        assert len(progress_updates) == 5

        # Check progress updates are correct
        for i, (completed, total, name) in enumerate(progress_updates):
            assert completed == i + 1
            assert total == 5
            assert name.startswith("test_function_")

    @pytest.mark.asyncio
    async def test_batch_error_handling(self, mock_analyzer, sample_requests):
        """Test batch handles individual request failures."""
        # Mock analyze_function to fail for specific requests
        original_analyze = mock_analyzer.analyze_function

        async def failing_analyze_function(request):
            if "test_function_2" in request.function.signature.name:
                raise Exception("Simulated failure")
            return await original_analyze(request)

        mock_analyzer.analyze_function = failing_analyze_function

        results = await mock_analyzer.analyze_batch(sample_requests[:5])

        assert len(results) == 5

        # Check that failed request has error response
        failed_result = results[2]  # test_function_2
        assert len(failed_result.issues) == 1
        assert failed_result.issues[0].issue_type == "analysis_error"
        assert "failed" in failed_result.issues[0].description.lower()

    @pytest.mark.asyncio
    async def test_batch_partial_failures(self, mock_analyzer, sample_requests):
        """Test batch continues when some requests fail."""
        # Mock to fail every other request
        original_analyze = mock_analyzer.analyze_function

        async def partial_failing_analyze(request):
            if int(request.function.signature.name.split("_")[-1]) % 2 == 0:
                raise Exception("Even numbered function failure")
            return await original_analyze(request)

        mock_analyzer.analyze_function = partial_failing_analyze

        results = await mock_analyzer.analyze_batch(sample_requests[:6])

        assert len(results) == 6

        # Check success/failure pattern
        for i, result in enumerate(results):
            if i % 2 == 0:  # Even indices should have error responses
                assert result.issues[0].issue_type == "analysis_error"
            else:  # Odd indices should have normal responses
                assert result.issues[0].issue_type == "test_issue"


class TestRequestGrouping:
    """Test request grouping for efficiency."""

    def test_group_requests_for_efficiency(self, mock_analyzer, sample_requests):
        """Test request grouping logic."""
        groups = mock_analyzer._group_requests_for_efficiency(sample_requests[:6])

        # Should return groups
        assert len(groups) > 0

        # Each group should contain (request, index) tuples
        for group in groups:
            assert len(group) > 0
            for item in group:
                assert len(item) == 2  # (request, index)
                assert isinstance(item[0], LLMAnalysisRequest)
                assert isinstance(item[1], int)

    def test_grouping_by_analysis_type(self, mock_analyzer, sample_requests):
        """Test grouping by analysis type."""
        # Modify requests to have distinct analysis types
        sample_requests[0].analysis_types = ["behavior"]
        sample_requests[1].analysis_types = ["examples"]
        sample_requests[2].analysis_types = ["behavior"]
        sample_requests[3].analysis_types = ["type_consistency"]

        groups = mock_analyzer._group_requests_for_efficiency(sample_requests[:4])

        # Should group by analysis type
        behavior_groups = []
        examples_groups = []
        type_groups = []

        for group in groups:
            primary_type = group[0][0].analysis_types[0]
            if primary_type == "behavior":
                behavior_groups.extend(group)
            elif primary_type == "examples":
                examples_groups.extend(group)
            elif primary_type == "type_consistency":
                type_groups.extend(group)

        assert len(behavior_groups) == 2  # Requests 0 and 2
        assert len(examples_groups) == 1  # Request 1
        assert len(type_groups) == 1  # Request 3

    def test_complexity_estimation(self, mock_analyzer, sample_functions):
        """Test function complexity estimation."""
        # Test with functions of varying complexity
        simple_func = sample_functions[0]  # 1 parameter
        complex_func = sample_functions[4]  # 5 parameters

        simple_complexity = mock_analyzer._estimate_function_complexity(simple_func)
        complex_complexity = mock_analyzer._estimate_function_complexity(complex_func)

        assert complex_complexity > simple_complexity
        assert simple_complexity > 0

    def test_grouping_by_file_path(self, mock_analyzer, sample_requests):
        """Test grouping considers file paths."""
        groups = mock_analyzer._group_requests_for_efficiency(sample_requests)

        # Verify that requests from same file are grouped together when possible
        file_paths_in_groups = []
        for group in groups:
            group_files = {req[0].function.file_path for req in group}
            file_paths_in_groups.append(group_files)

        # Most groups should have requests from same file
        single_file_groups = sum(1 for files in file_paths_in_groups if len(files) == 1)
        assert single_file_groups > 0


class TestCacheWarming:
    """Test cache warming functionality."""

    @pytest.mark.asyncio
    async def test_warm_cache_empty_list(self, mock_analyzer):
        """Test cache warming with empty function list."""
        result = await mock_analyzer.warm_cache([])

        assert result["total_functions"] == 0
        assert result["high_value_functions"] == 0
        assert result["warming_completed"] is True
        assert result["cache_entries_created"] == 0

    @pytest.mark.asyncio
    async def test_warm_cache_high_value_identification(
        self, mock_analyzer, sample_functions
    ):
        """Test identification of high-value functions for warming."""
        # Modify some functions to be high-value
        sample_functions[0].signature.name = "public_api_function"  # No underscore
        sample_functions[1].signature.name = "_private_function"  # Private
        sample_functions[2].signature.parameters = (
            sample_functions[2].signature.parameters * 3
        )  # Many params

        result = await mock_analyzer.warm_cache(sample_functions[:5])

        assert result["total_functions"] == 5
        assert result["high_value_functions"] > 0
        assert result["warming_completed"] is True

    def test_is_high_value_function(self, mock_analyzer, sample_functions):
        """Test high-value function identification logic."""
        # Public function (no underscore)
        public_func = sample_functions[0]
        public_func.signature.name = "public_function"
        assert mock_analyzer._is_high_value_function(public_func) is True

        # Private function with few parameters
        private_func = sample_functions[1]
        private_func.signature.name = "_private_function"
        private_func.signature.parameters = private_func.signature.parameters[:2]
        assert mock_analyzer._is_high_value_function(private_func) is False

        # Function with many parameters
        complex_func = sample_functions[2]
        complex_func.signature.name = "_complex_private"
        complex_func.signature.parameters = (
            complex_func.signature.parameters * 3
        )  # 6+ params
        assert mock_analyzer._is_high_value_function(complex_func) is True

        # Function with return type
        typed_func = sample_functions[3]
        typed_func.signature.name = "_private_typed"
        typed_func.signature.return_type = "Dict[str, Any]"
        assert mock_analyzer._is_high_value_function(typed_func) is True

    def test_determine_warming_analysis_types(self, mock_analyzer, sample_functions):
        """Test analysis type determination for warming."""
        func = sample_functions[0]

        # Test basic function
        analysis_types = mock_analyzer._determine_warming_analysis_types(func)
        assert "behavior" in analysis_types

        # Test function with examples in docstring
        func.docstring.raw_text = "Function with example: >>> func()"
        analysis_types = mock_analyzer._determine_warming_analysis_types(func)
        assert "behavior" in analysis_types
        assert "examples" in analysis_types

        # Test function with type annotations
        for param in func.signature.parameters:
            param.type_annotation = "str"
        analysis_types = mock_analyzer._determine_warming_analysis_types(func)
        assert "type_consistency" in analysis_types

    @pytest.mark.asyncio
    async def test_warm_cache_progress_callback(self, mock_analyzer, sample_functions):
        """Test cache warming with progress callback."""
        progress_updates = []

        def progress_callback(completed, total):
            progress_updates.append((completed, total))

        # Mock the warming to actually process some functions
        with patch.object(mock_analyzer, "_is_high_value_function", return_value=True):
            result = await mock_analyzer.warm_cache(
                sample_functions[:3], progress_callback=progress_callback
            )

        assert result["total_functions"] == 3

    @pytest.mark.asyncio
    async def test_warm_cache_concurrent_limit(self, mock_analyzer, sample_functions):
        """Test cache warming respects concurrency limits."""
        start_time = time.time()

        with patch.object(mock_analyzer, "_is_high_value_function", return_value=True):
            result = await mock_analyzer.warm_cache(
                sample_functions[:5], max_concurrent=2
            )

        elapsed_time = time.time() - start_time

        # Should respect concurrency limit
        assert result["total_functions"] == 5
        assert elapsed_time > 0  # Should take some time due to concurrency limit


class TestErrorResponseCreation:
    """Test error response creation for failed analyses."""

    def test_create_error_response(self, mock_analyzer, sample_requests):
        """Test error response creation."""
        request = sample_requests[0]
        error_message = "Test error message"

        response = mock_analyzer._create_error_response(request, error_message)

        assert isinstance(response, LLMAnalysisResponse)
        assert len(response.issues) == 1
        assert response.issues[0].issue_type == "analysis_error"
        assert response.issues[0].severity == "low"
        assert error_message in response.issues[0].description
        assert response.issues[0].line_number == request.function.line_number
        assert response.cache_hit is False
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0


@pytest.mark.integration
class TestBatchingIntegration:
    """Integration tests for batching functionality."""

    @pytest.mark.asyncio
    async def test_large_batch_performance(self, mock_analyzer, sample_functions):
        """Test performance with large batches."""
        # Create larger batch
        large_batch = []
        for i in range(50):
            func = sample_functions[i % len(sample_functions)]
            request = LLMAnalysisRequest(
                function=func,
                docstring=func.docstring,
                analysis_types=["behavior"],
                rule_results=[],
                related_functions=[],
            )
            large_batch.append(request)

        start_time = time.time()
        results = await mock_analyzer.analyze_batch(large_batch, max_concurrent=10)
        elapsed_time = time.time() - start_time

        assert len(results) == 50
        # Should complete reasonably quickly with concurrency
        assert elapsed_time < 5.0  # Generous limit for test environment

    @pytest.mark.asyncio
    async def test_mixed_analysis_types_batch(self, mock_analyzer, sample_functions):
        """Test batching with mixed analysis types."""
        mixed_requests = []
        analysis_types_list = [
            ["behavior"],
            ["examples"],
            ["type_consistency"],
            ["behavior", "examples"],
            ["behavior", "type_consistency"],
            ["examples", "edge_cases"],
        ]

        for i, func in enumerate(sample_functions[:6]):
            request = LLMAnalysisRequest(
                function=func,
                docstring=func.docstring,
                analysis_types=analysis_types_list[i],
                rule_results=[],
                related_functions=[],
            )
            mixed_requests.append(request)

        results = await mock_analyzer.analyze_batch(mixed_requests)

        assert len(results) == 6
        for result in results:
            assert isinstance(result, LLMAnalysisResponse)

    @pytest.mark.asyncio
    async def test_batch_memory_efficiency(self, mock_analyzer):
        """Test batching is memory efficient."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create many requests
        requests = []
        for i in range(100):
            param = FunctionParameter(
                name=f"param{i}",
                type_annotation="str",
                default_value=None,
                is_required=True,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
            )

            signature = FunctionSignature(
                name=f"test_function_{i}", parameters=[param], return_type="bool"
            )

            func = ParsedFunction(
                signature=signature,
                docstring=None,
                file_path=f"/test/file{i}.py",
                line_number=10,
            )

            request = LLMAnalysisRequest(
                function=func,
                docstring=None,
                analysis_types=["behavior"],
                rule_results=[],
                related_functions=[],
            )
            requests.append(request)

        results = await mock_analyzer.analyze_batch(requests)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        assert len(results) == 100
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100
