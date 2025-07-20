"""
Test suite for LLM analyzer core logic (Chunk 3).

Tests the core LLM analysis functionality including:
- analyze_function method
- OpenAI API calls with retry logic
- Smart prompt building
- Result merging with rule engine
- Caching and error handling
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from codedocsync.analyzer.llm_analyzer import LLMAnalyzer
from codedocsync.analyzer.llm_config import LLMConfig
from codedocsync.analyzer.llm_models import LLMAnalysisRequest, LLMAnalysisResponse
from codedocsync.analyzer.models import InconsistencyIssue, RuleCheckResult
from codedocsync.parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedDocstring,
    ParsedFunction,
)


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    with patch("codedocsync.analyzer.llm_analyzer.openai") as mock:
        mock.AsyncOpenAI.return_value = AsyncMock()
        yield mock


@pytest.fixture
def sample_function():
    """Create a sample ParsedFunction for testing."""
    signature = FunctionSignature(
        name="test_function",
        parameters=[
            FunctionParameter(
                name="param1",
                type_annotation="str",
                default_value=None,
                is_required=True,
            ),
            FunctionParameter(
                name="param2",
                type_annotation="int",
                default_value="10",
                is_required=False,
            ),
        ],
        return_type="bool",
    )

    return ParsedFunction(
        signature=signature, docstring=None, file_path="/test/file.py", line_number=10
    )


@pytest.fixture
def sample_docstring():
    """Create a sample ParsedDocstring for testing."""
    return ParsedDocstring(
        format="google",
        summary="Test function that does something",
        parameters=[],
        returns=None,
        raises=[],
        raw_text="Test function that does something\n\nArgs:\n    param1: A string parameter\n    param2: An integer parameter\n\nReturns:\n    True if successful",
    )


@pytest.fixture
def sample_request(sample_function, sample_docstring):
    """Create a sample LLMAnalysisRequest for testing."""
    return LLMAnalysisRequest(
        function=sample_function,
        docstring=sample_docstring,
        analysis_types=["behavior", "examples"],
        rule_results=[],
        related_functions=[],
    )


@pytest.fixture
def analyzer():
    """Create LLMAnalyzer instance for testing."""
    config = LLMConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        timeout_seconds=30,
        max_retries=3,
        cache_ttl_days=7,
        max_context_tokens=2000,
    )

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch("codedocsync.analyzer.llm_analyzer.openai"):
            return LLMAnalyzer(config)


class TestAnalyzeFunction:
    """Test the main analyze_function method."""

    @pytest.mark.asyncio
    async def test_successful_analysis(self, analyzer, sample_request, mock_openai):
        """Test successful LLM analysis."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "issues": [
                    {
                        "type": "behavior_mismatch",
                        "description": "Function behavior doesn't match docs",
                        "suggestion": "Update documentation to reflect actual behavior",
                        "confidence": 0.85,
                        "line_number": 1,
                        "details": {},
                    }
                ],
                "analysis_notes": "Checked behavior consistency",
                "confidence": 0.85,
            }
        )
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 75

        analyzer.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Execute analysis
        result = await analyzer.analyze_function(sample_request)

        # Verify result
        assert isinstance(result, LLMAnalysisResponse)
        assert len(result.issues) == 1
        assert (
            result.issues[0].issue_type == "description_outdated"
        )  # Mapped from behavior_mismatch
        assert result.model_used == "gpt-4o-mini"
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 75
        assert not result.cache_hit

    @pytest.mark.asyncio
    async def test_cache_hit(self, analyzer, sample_request):
        """Test cache hit scenario."""
        # Pre-populate cache
        cache_key = analyzer._generate_cache_key(
            function_signature=sample_request.get_function_signature_str(),
            docstring=sample_request.docstring.raw_text,
            analysis_types=sample_request.analysis_types,
            model=analyzer.config.model,
        )

        cached_response = LLMAnalysisResponse(
            issues=[],
            raw_response="cached response",
            model_used="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            response_time_ms=10.0,
            cache_hit=True,
        )

        await analyzer._store_cache(cache_key, cached_response)

        # Execute analysis
        result = await analyzer.analyze_function(sample_request)

        # Verify cache hit
        assert result.cache_hit
        assert result.raw_response == "cached response"
        assert analyzer.performance_stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_request_too_large(self, analyzer, sample_request):
        """Test handling of requests that exceed token limits."""
        # Mock estimate_tokens to return a large value
        with patch.object(sample_request, "estimate_tokens", return_value=5000):
            with pytest.raises(ValueError, match="Request too large"):
                await analyzer.analyze_function(sample_request)

    @pytest.mark.asyncio
    async def test_invalid_request(self, analyzer):
        """Test handling of invalid requests."""
        with pytest.raises(ValueError, match="LLMAnalysisRequest cannot be None"):
            await analyzer.analyze_function(None)


class TestCallOpenAI:
    """Test the _call_openai method."""

    @pytest.mark.asyncio
    async def test_successful_api_call(self, analyzer, mock_openai):
        """Test successful OpenAI API call."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        analyzer.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Execute API call
        response_text, token_usage = await analyzer._call_openai(
            user_prompt="Test prompt", system_prompt="Test system prompt"
        )

        # Verify result
        assert response_text == "Test response"
        assert token_usage["prompt_tokens"] == 100
        assert token_usage["completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, analyzer, mock_openai):
        """Test retry logic on rate limit error."""
        import openai

        # Mock rate limit error then success
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success after retry"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        analyzer.openai_client.chat.completions.create = AsyncMock(
            side_effect=[
                openai.RateLimitError("Rate limit exceeded", response=None, body=None),
                mock_response,
            ]
        )

        # Execute with retry
        start_time = time.time()
        response_text, token_usage = await analyzer._call_openai("Test prompt")
        elapsed = time.time() - start_time

        # Verify retry occurred (should take at least 1 second for backoff)
        assert elapsed >= 1.0
        assert response_text == "Success after retry"
        assert analyzer.openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, analyzer, mock_openai):
        """Test timeout handling."""
        # Mock timeout
        analyzer.openai_client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        # Should raise TimeoutError after retries
        with pytest.raises(asyncio.TimeoutError):
            await analyzer._call_openai("Test prompt")

        # Should have attempted retries
        assert (
            analyzer.openai_client.chat.completions.create.call_count
            == analyzer.config.max_retries + 1
        )


class TestBuildAnalysisPrompt:
    """Test the _build_analysis_prompt method."""

    def test_basic_prompt_building(self, analyzer, sample_function, sample_docstring):
        """Test basic prompt building functionality."""
        analysis_types = ["behavior"]
        context = {"rule_results": [], "related_functions": []}

        system_prompt, user_prompt = analyzer._build_analysis_prompt(
            sample_function, sample_docstring, analysis_types, context
        )

        # Verify system prompt
        assert "expert Python documentation analyzer" in system_prompt
        assert "valid JSON only" in system_prompt

        # Verify user prompt contains function details
        assert "test_function" in user_prompt
        assert "param1: str" in user_prompt
        assert "param2: int = 10" in user_prompt
        assert "Test function that does something" in user_prompt

    def test_multiple_analysis_types(self, analyzer, sample_function, sample_docstring):
        """Test prompt building with multiple analysis types."""
        analysis_types = ["behavior", "examples", "edge_cases"]
        context = {"rule_results": [], "related_functions": []}

        system_prompt, user_prompt = analyzer._build_analysis_prompt(
            sample_function, sample_docstring, analysis_types, context
        )

        # Should mention additional analysis types
        assert "ADDITIONAL ANALYSIS" in user_prompt
        assert "examples" in user_prompt
        assert "edge_cases" in user_prompt

    def test_rule_results_context(self, analyzer, sample_function, sample_docstring):
        """Test prompt building with rule results context."""
        # Create mock rule result
        rule_issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Parameter not documented",
            suggestion="Add parameter documentation",
            line_number=5,
        )

        rule_result = RuleCheckResult(
            rule_name="missing_params",
            passed=False,
            confidence=0.95,
            issues=[rule_issue],
            execution_time_ms=5.0,
        )

        context = {"rule_results": [rule_result], "related_functions": []}
        analysis_types = ["behavior"]

        system_prompt, user_prompt = analyzer._build_analysis_prompt(
            sample_function, sample_docstring, analysis_types, context
        )

        # Should include rule results in prompt
        assert "missing_params" in user_prompt
        assert "Parameter not documented" in user_prompt

    def test_fallback_prompt(self, analyzer):
        """Test fallback prompt when template formatting fails."""
        fallback = analyzer._build_fallback_prompt(
            signature="def test():",
            source_code="return True",
            docstring="Test function",
        )

        assert "def test():" in fallback
        assert "return True" in fallback
        assert "Test function" in fallback
        assert "JSON" in fallback


class TestMergeWithRuleResults:
    """Test the merge_with_rule_results method."""

    def test_high_confidence_rule_priority(self, analyzer):
        """Test that high-confidence rule results take priority."""
        # High-confidence rule issue
        rule_issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Rule found missing parameter",
            suggestion="Add parameter docs",
            line_number=1,
            confidence=0.95,
        )

        rule_result = RuleCheckResult(
            rule_name="missing_params",
            passed=False,
            confidence=0.95,
            issues=[rule_issue],
            execution_time_ms=5.0,
        )

        # LLM issue for same thing
        llm_issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="high",
            description="LLM found missing parameter",
            suggestion="Add parameter documentation",
            line_number=1,
            confidence=0.8,
        )

        merged = analyzer.merge_with_rule_results([llm_issue], [rule_result])

        # Should prefer rule result and combine suggestions
        assert len(merged) == 1
        assert "Rule found missing parameter" in merged[0].description
        assert "Add parameter docs" in merged[0].suggestion
        assert "Additionally" in merged[0].suggestion

    def test_llm_enhancement_of_low_confidence_rules(self, analyzer):
        """Test LLM enhancement of low-confidence rule results."""
        # Low-confidence rule issue
        rule_issue = InconsistencyIssue(
            issue_type="description_outdated",
            severity="medium",
            description="Possibly outdated description",
            suggestion="Review description",
            line_number=1,
            confidence=0.6,
        )

        rule_result = RuleCheckResult(
            rule_name="description_check",
            passed=False,
            confidence=0.6,
            issues=[rule_issue],
            execution_time_ms=5.0,
        )

        # LLM provides better analysis
        llm_issue = InconsistencyIssue(
            issue_type="description_outdated",
            severity="medium",
            description="Function behavior changed but docs not updated",
            suggestion="Update docstring to reflect new behavior",
            line_number=1,
            confidence=0.85,
        )

        merged = analyzer.merge_with_rule_results([llm_issue], [rule_result])

        # Should use LLM result when it provides better analysis
        assert len(merged) == 1
        assert "Function behavior changed" in merged[0].description

    def test_distinct_issues_preserved(self, analyzer):
        """Test that distinct issues are all preserved."""
        rule_issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Missing parameter docs",
            suggestion="Add parameter",
            line_number=1,
            confidence=0.95,
        )

        rule_result = RuleCheckResult(
            rule_name="missing_params",
            passed=False,
            confidence=0.95,
            issues=[rule_issue],
            execution_time_ms=5.0,
        )

        llm_issue = InconsistencyIssue(
            issue_type="description_outdated",
            severity="medium",
            description="Behavior description outdated",
            suggestion="Update behavior description",
            line_number=10,
            confidence=0.8,
        )

        merged = analyzer.merge_with_rule_results([llm_issue], [rule_result])

        # Should have both distinct issues
        assert len(merged) == 2
        issue_types = [issue.issue_type for issue in merged]
        assert "parameter_missing" in issue_types
        assert "description_outdated" in issue_types

    def test_sorting_by_severity_and_line(self, analyzer):
        """Test that merged results are sorted by severity and line number."""
        issues = [
            InconsistencyIssue(
                issue_type="description_outdated",
                severity="medium",
                description="Medium issue line 10",
                suggestion="Fix it",
                line_number=10,
                confidence=0.8,
            ),
            InconsistencyIssue(
                issue_type="parameter_missing",
                severity="critical",
                description="Critical issue line 5",
                suggestion="Fix it",
                line_number=5,
                confidence=0.9,
            ),
            InconsistencyIssue(
                issue_type="return_type_mismatch",
                severity="high",
                description="High issue line 1",
                suggestion="Fix it",
                line_number=1,
                confidence=0.85,
            ),
        ]

        merged = analyzer.merge_with_rule_results(issues, [])

        # Should be sorted by severity (critical > high > medium), then line number
        assert merged[0].severity == "critical"
        assert merged[1].severity == "high"
        assert merged[2].severity == "medium"


class TestCaching:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_store_and_retrieve(self, analyzer):
        """Test storing and retrieving from cache."""
        response = LLMAnalysisResponse(
            issues=[],
            raw_response="test response",
            model_used="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            response_time_ms=150.0,
            cache_hit=False,
        )

        cache_key = "test_cache_key"

        # Store in cache
        await analyzer._store_cache(cache_key, response)

        # Retrieve from cache
        cached = await analyzer._check_cache(cache_key)

        assert cached is not None
        assert cached.raw_response == "test response"
        assert cached.cache_hit

    @pytest.mark.asyncio
    async def test_cache_expiration(self, analyzer):
        """Test cache expiration logic."""
        # Set very short TTL for testing
        analyzer.config.cache_ttl_days = 0.000001  # ~0.1 seconds

        response = LLMAnalysisResponse(
            issues=[],
            raw_response="test response",
            model_used="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            response_time_ms=150.0,
            cache_hit=False,
        )

        cache_key = "test_cache_key"

        # Store in cache
        await analyzer._store_cache(cache_key, response)

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired
        cached = await analyzer._check_cache(cache_key)
        assert cached is None


class TestPerformanceMetrics:
    """Test performance tracking."""

    @pytest.mark.asyncio
    async def test_performance_stats_updated(
        self, analyzer, sample_request, mock_openai
    ):
        """Test that performance stats are properly updated."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"issues": [], "analysis_notes": "Test", "confidence": 0.8}
        )
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        analyzer.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        initial_stats = analyzer.performance_stats.copy()

        # Execute analysis
        await analyzer.analyze_function(sample_request)

        # Verify stats updated
        assert (
            analyzer.performance_stats["requests_made"]
            == initial_stats["requests_made"] + 1
        )
        assert (
            analyzer.performance_stats["total_tokens_used"]
            == initial_stats["total_tokens_used"] + 150
        )
        assert (
            analyzer.performance_stats["cache_misses"]
            == initial_stats["cache_misses"] + 1
        )

    @pytest.mark.asyncio
    async def test_error_stats_updated(self, analyzer, sample_request, mock_openai):
        """Test that error stats are properly updated."""
        # Mock API error
        import openai

        analyzer.openai_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError("Test error", response=None, body=None)
        )

        initial_errors = analyzer.performance_stats["errors_encountered"]

        # Should raise error and update stats
        with pytest.raises(openai.APIError):
            await analyzer.analyze_function(sample_request)

        assert analyzer.performance_stats["errors_encountered"] == initial_errors + 1
