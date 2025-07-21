"""
Test suite for LLM analyzer implementation.

Tests LLM integration, caching, retry logic, and fallback behavior.
"""

import asyncio
import json
from unittest.mock import Mock, patch

import pytest

from codedocsync.analyzer.llm_analyzer import LLMAnalyzer
from codedocsync.analyzer.llm_cache import LLMCache
from codedocsync.analyzer.llm_errors import LLMError as LLMAnalysisError
from codedocsync.analyzer.llm_models import LLMAnalysisResponse as LLMAnalysisResult
from codedocsync.matcher import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser.ast_parser import FunctionSignature, ParsedFunction
from codedocsync.parser.docstring_models import ParsedDocstring


class TestLLMCache:
    """Test LLM caching functionality."""

    def test_cache_initialization(self, tmp_path):
        """Test cache initialization."""
        cache = LLMCache(cache_dir=tmp_path, ttl_hours=12)

        assert cache.db_path == tmp_path / "llm_analysis_cache.db"
        assert cache.ttl_seconds == 12 * 3600
        assert cache.db_path.exists()

    def test_cache_key_generation(self, tmp_path):
        """Test cache key generation."""
        cache = LLMCache(cache_dir=tmp_path)

        # Create test data
        function_sig = "def test_func(param: str) -> int"
        docstring = "Test docstring"
        model = "gpt-3.5-turbo"

        key1 = cache._generate_cache_key(function_sig, docstring, model)
        key2 = cache._generate_cache_key(function_sig, docstring, model)
        key3 = cache._generate_cache_key(function_sig, "Different docstring", model)

        # Same inputs should generate same key
        assert key1 == key2
        # Different inputs should generate different keys
        assert key1 != key3

    def test_cache_put_and_get(self, tmp_path):
        """Test storing and retrieving from cache."""
        cache = LLMCache(cache_dir=tmp_path)

        # Test data
        function_sig = "def test_func(param: str) -> int"
        docstring = "Test docstring"
        model = "gpt-3.5-turbo"
        result = {"issues": [], "confidence": 0.9}

        # Store in cache
        cache.put(function_sig, docstring, model, result)

        # Retrieve from cache
        cached_result = cache.get(function_sig, docstring, model)

        assert cached_result == result

    def test_cache_expiration(self, tmp_path):
        """Test cache TTL expiration."""
        # Create cache with very short TTL
        cache = LLMCache(
            cache_dir=tmp_path, ttl_hours=0
        )  # 0 hours = immediate expiration

        function_sig = "def test_func(param: str) -> int"
        docstring = "Test docstring"
        model = "gpt-3.5-turbo"
        result = {"issues": [], "confidence": 0.9}

        # Store in cache
        cache.put(function_sig, docstring, model, result)

        # Should not retrieve expired entry
        cached_result = cache.get(function_sig, docstring, model)
        assert cached_result is None

    def test_cache_stats(self, tmp_path):
        """Test cache statistics."""
        cache = LLMCache(cache_dir=tmp_path)

        # Initially empty
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

        # Add entry and test hit
        function_sig = "def test_func(param: str) -> int"
        docstring = "Test docstring"
        model = "gpt-3.5-turbo"
        result = {"issues": [], "confidence": 0.9}

        cache.put(function_sig, docstring, model, result)
        cache.get(function_sig, docstring, model)  # Hit
        cache.get("different", "sig", "model")  # Miss

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    def test_cache_clear(self, tmp_path):
        """Test cache clearing functionality."""
        cache = LLMCache(cache_dir=tmp_path)

        # Add some entries
        for i in range(3):
            cache.put(f"func_{i}", f"docstring_{i}", "model", {"data": i})

        stats = cache.get_stats()
        assert stats["total_entries"] == 3

        # Clear cache
        cleared_count = cache.clear()
        assert cleared_count == 3

        stats = cache.get_stats()
        assert stats["total_entries"] == 0


class TestLLMAnalyzer:
    """Test LLMAnalyzer functionality."""

    def create_mock_matched_pair(self):
        """Create a mock matched pair for testing."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_function",
                parameters=[],
                return_annotation=None,
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
            parameters=[],
            returns=None,
            raises=[],
            raw_text="Test function docstring",
        )

        return MatchedPair(
            function=function,
            documentation=docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

    def test_llm_analyzer_initialization(self, tmp_path):
        """Test LLMAnalyzer initialization."""
        analyzer = LLMAnalyzer(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.1,
            cache_dir=tmp_path,
        )

        assert analyzer.provider == "openai"
        assert analyzer.model == "gpt-3.5-turbo"
        assert analyzer.temperature == 0.1
        assert analyzer.cache is not None

    def test_llm_analyzer_no_cache(self):
        """Test LLMAnalyzer without caching."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", enable_cache=False
        )

        assert analyzer.cache is None

    @pytest.mark.asyncio
    async def test_analyze_consistency_cached(self, tmp_path):
        """Test analyze_consistency with cache hit."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path
        )

        pair = self.create_mock_matched_pair()

        # Mock cache to return a result
        mock_result = {
            "issues": [
                {
                    "type": "description_outdated",
                    "description": "Function behavior has changed",
                    "suggestion": "Update the docstring",
                    "confidence": 0.8,
                }
            ],
            "analysis_notes": "Test analysis",
            "confidence": 0.8,
        }

        with patch.object(analyzer.cache, "get", return_value=mock_result):
            result = await analyzer.analyze_consistency(pair)

            assert result.cache_hit
            assert len(result.issues) == 1
            assert result.issues[0].issue_type == "description_outdated"
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_analyze_consistency_mock_llm(self, tmp_path):
        """Test analyze_consistency with mocked LLM call."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path
        )

        pair = self.create_mock_matched_pair()

        # Mock the LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "issues": [
                    {
                        "type": "behavior_mismatch",
                        "description": "Function behavior doesn't match description",
                        "suggestion": "Update docstring to reflect actual behavior",
                        "confidence": 0.85,
                    }
                ],
                "analysis_notes": "Detected behavioral inconsistency",
                "confidence": 0.85,
            }
        )

        # Mock the cache to return None (cache miss)
        with (
            patch.object(analyzer.cache, "get", return_value=None),
            patch.object(analyzer.cache, "put"),
            patch(
                "codedocsync.analyzer.llm_analyzer.acompletion",
                return_value=mock_response,
            ) as mock_completion,
        ):
            result = await analyzer.analyze_consistency(pair)

            assert not result.cache_hit
            assert len(result.issues) == 1
            assert result.issues[0].issue_type == "behavior_mismatch"
            assert result.confidence == 0.85
            assert "behavioral inconsistency" in result.analysis_notes

            # Verify LLM was called
            mock_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_with_retry_success(self, tmp_path):
        """Test retry logic with eventual success."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path, max_retries=3
        )

        pair = self.create_mock_matched_pair()

        # Mock first call to fail, second to succeed
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps(
            {"issues": [], "analysis_notes": "No issues found", "confidence": 0.9}
        )

        call_count = 0

        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API error")
            return mock_response

        with (
            patch.object(analyzer.cache, "get", return_value=None),
            patch.object(analyzer.cache, "put"),
            patch(
                "codedocsync.analyzer.llm_analyzer.acompletion",
                side_effect=mock_completion,
            ),
        ):
            result = await analyzer.analyze_consistency(pair)

            assert len(result.issues) == 0
            assert result.confidence == 0.9
            assert call_count == 2  # First failed, second succeeded

    @pytest.mark.asyncio
    async def test_analyze_with_retry_failure(self, tmp_path):
        """Test retry logic with ultimate failure."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path, max_retries=2
        )

        pair = self.create_mock_matched_pair()

        # Mock all calls to fail
        async def mock_completion(*args, **kwargs):
            raise Exception("Persistent API error")

        with (
            patch.object(analyzer.cache, "get", return_value=None),
            patch(
                "codedocsync.analyzer.llm_analyzer.acompletion",
                side_effect=mock_completion,
            ),
        ):
            with pytest.raises(LLMAnalysisError):
                await analyzer.analyze_consistency(pair)

    @pytest.mark.asyncio
    async def test_specialized_analysis_methods(self, tmp_path):
        """Test specialized analysis methods."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path
        )

        pair = self.create_mock_matched_pair()

        # Mock the main analyze_consistency method
        async def mock_analyze(pair, analysis_types=None):
            return LLMAnalysisResult(
                issues=[],
                analysis_notes=f"Analysis type: {analysis_types[0] if analysis_types else 'default'}",
                confidence=0.9,
                llm_model="gpt-3.5-turbo",
                execution_time_ms=100.0,
            )

        with patch.object(analyzer, "analyze_consistency", side_effect=mock_analyze):
            # Test behavior analysis
            result = await analyzer.analyze_behavior(pair)
            assert "behavior_analysis" in result.analysis_notes

            # Test example analysis
            result = await analyzer.analyze_examples(pair)
            assert "example_validation" in result.analysis_notes

            # Test edge case analysis
            result = await analyzer.analyze_edge_cases(pair)
            assert "edge_case_analysis" in result.analysis_notes

            # Test version analysis
            result = await analyzer.analyze_version_info(pair)
            assert "version_analysis" in result.analysis_notes

    def test_cache_stats_integration(self, tmp_path):
        """Test cache statistics integration."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path
        )

        stats = analyzer.get_cache_stats()
        assert "total_entries" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    def test_cache_clear_integration(self, tmp_path):
        """Test cache clearing integration."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path
        )

        # Should work without errors
        cleared_count = analyzer.clear_cache()
        assert isinstance(cleared_count, int)

    def test_no_cache_stats(self):
        """Test stats when cache is disabled."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", enable_cache=False
        )

        stats = analyzer.get_cache_stats()
        assert stats == {"cache_enabled": False}

    def test_no_cache_clear(self):
        """Test clear when cache is disabled."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", enable_cache=False
        )

        cleared_count = analyzer.clear_cache()
        assert cleared_count == 0


class TestLLMResponseParsing:
    """Test LLM response parsing and validation."""

    def test_valid_json_response(self):
        """Test parsing of valid JSON response."""
        from codedocsync.analyzer.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(provider="openai", model="gpt-3.5-turbo")

        # Valid JSON response
        response_text = json.dumps(
            {
                "issues": [
                    {
                        "type": "behavior_mismatch",
                        "description": "Function behavior changed",
                        "suggestion": "Update docstring",
                        "confidence": 0.8,
                    }
                ],
                "analysis_notes": "Found one issue",
                "confidence": 0.8,
            }
        )

        result = analyzer._parse_llm_response(response_text, "gpt-3.5-turbo", 100.0)

        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "behavior_mismatch"
        assert result.confidence == 0.8
        assert result.analysis_notes == "Found one issue"

    def test_invalid_json_response(self):
        """Test handling of invalid JSON response."""
        from codedocsync.analyzer.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(provider="openai", model="gpt-3.5-turbo")

        # Invalid JSON
        response_text = "This is not valid JSON"

        # Should raise an error or return a fallback result
        with pytest.raises(LLMAnalysisError):
            analyzer._parse_llm_response(response_text, "gpt-3.5-turbo", 100.0)

    def test_missing_fields_response(self):
        """Test handling of response with missing required fields."""
        from codedocsync.analyzer.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(provider="openai", model="gpt-3.5-turbo")

        # JSON missing required fields
        response_text = json.dumps(
            {
                "issues": [],
                # Missing analysis_notes and confidence
            }
        )

        # Should handle gracefully with defaults
        result = analyzer._parse_llm_response(response_text, "gpt-3.5-turbo", 100.0)

        assert len(result.issues) == 0
        assert result.analysis_notes == ""  # Should have default
        assert 0.0 <= result.confidence <= 1.0  # Should have valid default


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_network_timeout(self, tmp_path):
        """Test handling of network timeouts."""
        analyzer = LLMAnalyzer(
            provider="openai",
            model="gpt-3.5-turbo",
            cache_dir=tmp_path,
            timeout_seconds=1.0,
        )

        pair = TestLLMAnalyzer().create_mock_matched_pair()

        # Mock a timeout
        async def mock_completion(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return Mock()

        with (
            patch.object(analyzer.cache, "get", return_value=None),
            patch(
                "codedocsync.analyzer.llm_analyzer.acompletion",
                side_effect=mock_completion,
            ),
        ):
            with pytest.raises(LLMAnalysisError):
                await analyzer.analyze_consistency(pair)

    @pytest.mark.asyncio
    async def test_malformed_function_data(self, tmp_path):
        """Test handling of malformed function data."""
        analyzer = LLMAnalyzer(
            provider="openai", model="gpt-3.5-turbo", cache_dir=tmp_path
        )

        # Create malformed matched pair
        pair = MatchedPair(
            function=None,  # Invalid: None function
            documentation=None,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Test match",
        )

        # Should handle gracefully
        with pytest.raises((ValueError, LLMAnalysisError)):
            await analyzer.analyze_consistency(pair)
