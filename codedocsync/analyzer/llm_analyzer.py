"""
LLM Analyzer Foundation - Chunk 1 Implementation

This module provides the foundation for LLM-powered semantic analysis.
Includes class structure, initialization, rate limiting, and cache schema.

Key Requirements (Chunk 1):
- Use openai library directly (not litellm since we're OpenAI-only)
- Implement token bucket rate limiter in __init__
- Cache must use SQLite (create table if not exists)
- Must validate API key exists in environment
- Performance target: Foundation setup in <100ms
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .llm_config import LLMConfig
from .llm_errors import (
    CircuitBreaker,
    LLMAPIKeyError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMTimeoutError,
    RetryStrategy,
)
from .llm_models import LLMAnalysisRequest, LLMAnalysisResponse
from .llm_output_parser import parse_llm_response
from .models import AnalysisResult, InconsistencyIssue, RuleCheckResult
from .prompt_templates import format_prompt

# Import ParsedFunction for type hints
if TYPE_CHECKING:
    from ..parser.models import ParsedFunction

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Ensures we don't exceed API rate limits while allowing bursts.
    """

    def __init__(self, rate: float, burst_size: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            burst_size: Maximum burst capacity
        """
        self.rate = rate
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update

            # Add tokens based on time passed
            self.tokens = min(self.burst_size, self.tokens + time_passed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
        """
        while not await self.acquire(tokens):
            # Calculate wait time needed
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(min(wait_time, 1.0))  # Max 1 second waits


class LLMAnalyzer:
    """
    Analyzes code-doc consistency using LLM for semantic understanding.

    This is the foundation implementation (Chunk 1) focusing on:
    - Proper initialization and configuration
    - Rate limiting using token bucket algorithm
    - SQLite cache setup with proper schema
    - OpenAI client initialization with retry wrapper
    - Performance monitoring setup
    """

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize with configuration and OpenAI client.

        Args:
            config: LLM configuration (creates default if None)

        Raises:
            ImportError: If openai package is not available
            ValueError: If configuration is invalid or API key missing
        """
        # Validate OpenAI availability
        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for LLM analysis. "
                "Install with: pip install openai"
            )

        # Set up configuration (this validates API key and other settings)
        self.config = config or LLMConfig()

        # Initialize OpenAI client with retry wrapper
        self._init_openai_client()

        # Set up rate limiter (token bucket algorithm)
        self.rate_limiter = TokenBucket(
            rate=self.config.requests_per_second, burst_size=self.config.burst_size
        )

        # Initialize cache connection
        self.cache_db_path = self._init_cache_database()

        # Set up performance monitoring
        self.performance_stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_used": 0,
            "total_response_time_ms": 0.0,
            "errors_encountered": 0,
        }

        # Set up circuit breaker for error handling
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=60, expected_exception=LLMError
        )

        # Set up retry strategy
        self.retry_strategy = RetryStrategy(
            max_retries=self.config.max_retries,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Track initialization success
        self._initialized_at = time.time()

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client with proper configuration."""
        # Get API key from environment (already validated in config)
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )

        # Store model identifier for API calls
        self.model_id = self.config.model

    def _init_cache_database(self) -> Path:
        """
        Initialize SQLite cache database with proper schema.

        Returns:
            Path to the cache database file
        """
        # Create cache directory
        cache_dir = Path.home() / ".cache" / "codedocsync"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_db_path = cache_dir / "llm_cache.db"

        # Create database schema if it doesn't exist
        with sqlite3.connect(cache_db_path) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")

            # Create main cache table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON llm_cache(created_at)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model ON llm_cache(model)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_hash ON llm_cache(request_hash)
            """
            )

            conn.commit()

        return cache_db_path

    def _generate_cache_key(
        self, function_signature: str, docstring: str, analysis_types: list, model: str
    ) -> str:
        """
        Generate deterministic cache key.

        Cache key generation is critical for cache effectiveness.
        Must be deterministic and collision-resistant.

        Args:
            function_signature: String representation of function signature
            docstring: Raw docstring text
            analysis_types: List of analysis types being performed
            model: Model identifier

        Returns:
            MD5 hash as cache key
        """
        # Create content string with all relevant parameters
        content_parts = [
            function_signature,
            docstring,
            "|".join(sorted(analysis_types)),  # Sort for deterministic ordering
            model,
        ]
        content = "|".join(content_parts)

        # Generate MD5 hash (sufficient for cache keys)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics and performance metrics
        """
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Get basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM llm_cache")
                total_entries = cursor.fetchone()[0]

                # Get valid entries (not expired)
                ttl_seconds = self.config.cache_ttl_days * 24 * 3600
                cutoff_time = time.time() - ttl_seconds
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM llm_cache WHERE created_at > ?",
                    (cutoff_time,),
                )
                valid_entries = cursor.fetchone()[0]

                # Get access statistics
                cursor = conn.execute("SELECT SUM(access_count) FROM llm_cache")
                total_accesses = cursor.fetchone()[0] or 0

                # Get database file size
                db_size_mb = self.cache_db_path.stat().st_size / (1024 * 1024)

                # Calculate hit rate
                total_requests = (
                    self.performance_stats["cache_hits"]
                    + self.performance_stats["cache_misses"]
                )
                hit_rate = (
                    self.performance_stats["cache_hits"] / total_requests
                    if total_requests > 0
                    else 0.0
                )

                return {
                    "total_entries": total_entries,
                    "valid_entries": valid_entries,
                    "expired_entries": total_entries - valid_entries,
                    "total_accesses": total_accesses,
                    "database_size_mb": round(db_size_mb, 2),
                    "cache_hit_rate": round(hit_rate, 3),
                    "cache_file_path": str(self.cache_db_path),
                    **self.performance_stats,
                }

        except Exception as e:
            return {
                "error": f"Failed to get cache stats: {e}",
                "cache_enabled": False,
            }

    def clear_expired_cache_entries(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        ttl_seconds = self.config.cache_ttl_days * 24 * 3600
        cutoff_time = time.time() - ttl_seconds

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM llm_cache WHERE created_at < ?", (cutoff_time,)
                )
                return cursor.rowcount

        except Exception:
            return 0

    def validate_configuration(self) -> dict[str, Any]:
        """
        Validate the current configuration and system state.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "config_valid": True,
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "cache_accessible": False,
            "openai_client_initialized": hasattr(self, "openai_client"),
            "rate_limiter_configured": hasattr(self, "rate_limiter"),
            "errors": [],
        }

        # Test cache accessibility
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("SELECT 1")
            validation_results["cache_accessible"] = True
        except Exception as e:
            validation_results["errors"].append(f"Cache not accessible: {e}")

        # Validate configuration
        try:
            # This will raise if configuration is invalid
            _ = self.config.get_summary()
        except Exception as e:
            validation_results["config_valid"] = False
            validation_results["errors"].append(f"Configuration invalid: {e}")

        return validation_results

    def get_initialization_summary(self) -> dict[str, Any]:
        """
        Get summary of initialization status and configuration.

        Returns:
            Dictionary with initialization details
        """
        return {
            "initialized_at": self._initialized_at,
            "initialization_time_ms": (time.time() - self._initialized_at) * 1000,
            "config_summary": self.config.get_summary(),
            "cache_db_path": str(self.cache_db_path),
            "model_id": self.model_id,
            "rate_limit_config": {
                "requests_per_second": self.config.requests_per_second,
                "burst_size": self.config.burst_size,
            },
            "validation_results": self.validate_configuration(),
        }

    # ========== CHUNK 3: Core LLM Analysis Logic ==========

    async def analyze_function(
        self, request: LLMAnalysisRequest, cache: dict[str, Any] | None = None
    ) -> LLMAnalysisResponse:
        """
        Perform LLM analysis with caching and error handling.

        Args:
            request: LLM analysis request with function and docstring
            cache: Optional analysis cache (reserved for future use)

        Returns:
            LLMAnalysisResponse with structured results

        Raises:
            ValueError: If request is invalid
            openai.APIError: If API call fails after retries
            asyncio.TimeoutError: If request times out
        """
        start_time = time.time()

        # Validate request
        if not request:
            raise ValueError("LLMAnalysisRequest cannot be None")

        # Validate token estimate to prevent context overflow
        estimated_tokens = request.estimate_tokens()
        if estimated_tokens > self.config.max_context_tokens:
            raise ValueError(
                f"Request too large: {estimated_tokens} tokens exceeds "
                f"limit of {self.config.max_context_tokens}"
            )

        # Generate cache key
        cache_key = self._generate_cache_key(
            function_signature=request.get_function_signature_str(),
            docstring=request.docstring.raw_text,
            analysis_types=request.analysis_types,
            model=self.config.model,
        )

        # Check cache first
        cached_response = await self._check_cache(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for function {request.function.signature.name}")
            self.performance_stats["cache_hits"] += 1
            return cached_response

        self.performance_stats["cache_misses"] += 1

        # Check rate limits before making API call
        await self.rate_limiter.wait_for_tokens(1)

        try:
            # Build prompts for requested analysis types
            system_prompt, user_prompt = self._build_analysis_prompt(
                request.function,
                request.docstring,
                request.analysis_types,
                {
                    "rule_results": request.rule_results,
                    "related_functions": request.related_functions,
                },
            )

            # Call OpenAI API
            raw_response, token_usage = await self._call_openai(
                user_prompt=user_prompt, system_prompt=system_prompt
            )

            # Parse response
            parse_result = parse_llm_response(raw_response, strict=False)

            if not parse_result.success:
                logger.warning(
                    f"Failed to parse LLM response for {request.function.signature.name}: "
                    f"{parse_result.error_message}"
                )

            # Create response object
            response_time_ms = (time.time() - start_time) * 1000
            response = LLMAnalysisResponse(
                issues=parse_result.issues,
                raw_response=raw_response,
                model_used=self.config.model,
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
                response_time_ms=response_time_ms,
                cache_hit=False,
            )

            # Cache the response
            await self._store_cache(cache_key, response)

            # Update performance stats
            self.performance_stats["requests_made"] += 1
            self.performance_stats["total_tokens_used"] += response.total_tokens
            self.performance_stats["total_response_time_ms"] += response_time_ms

            logger.debug(
                f"LLM analysis completed for {request.function.signature.name}: "
                f"{len(response.issues)} issues found in {response_time_ms:.1f}ms"
            )

            return response

        except Exception as e:
            self.performance_stats["errors_encountered"] += 1
            logger.error(
                f"LLM analysis failed for {request.function.signature.name}: {e}"
            )
            raise

    async def _call_openai(
        self, user_prompt: str, system_prompt: str | None = None
    ) -> tuple[str, dict[str, int]]:
        """
        Make OpenAI API call with retry logic.

        Args:
            user_prompt: The main prompt for analysis
            system_prompt: Optional system prompt for role definition

        Returns:
            Tuple of (response_text, token_usage_dict)

        Raises:
            openai.APIError: If API call fails after retries
            asyncio.TimeoutError: If request times out
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        # Retry logic with exponential backoff
        max_retries = self.config.max_retries
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Apply timeout to the entire operation
                async with asyncio.timeout(self.config.timeout_seconds):
                    response = await self.openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                # Extract response text
                response_text = response.choices[0].message.content

                # Extract token usage
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }

                total_tokens = (
                    token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                )
                logger.debug(f"OpenAI API call successful: {total_tokens} tokens used")

                return response_text, token_usage

            except asyncio.TimeoutError:
                logger.warning(f"OpenAI API call timed out (attempt {attempt + 1})")
                if attempt == max_retries:
                    raise
                await asyncio.sleep(base_delay * (2**attempt))

            except openai.RateLimitError as e:
                logger.warning(f"Rate limit exceeded (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    raise

                # Extract retry-after from headers if available
                retry_after = getattr(e, "retry_after", None) or (
                    base_delay * (2**attempt)
                )
                await asyncio.sleep(min(retry_after, 60))  # Cap at 60 seconds

            except openai.APIError as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    raise
                await asyncio.sleep(base_delay * (2**attempt))

            except Exception as e:
                logger.error(f"Unexpected error in OpenAI API call: {e}")
                raise

    def _build_analysis_prompt(
        self, function, docstring, analysis_types: list[str], context: dict[str, Any]
    ) -> tuple[str, str]:
        """
        Build optimized prompt for analysis type.

        Args:
            function: ParsedFunction object
            docstring: ParsedDocstring object
            analysis_types: List of analysis types to perform
            context: Additional context (rule_results, related_functions, etc.)

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # For multiple analysis types, use behavior analysis as primary template
        # and mention other types in context
        primary_analysis = analysis_types[0] if analysis_types else "behavior"

        # Map analysis types to prompt template names
        analysis_mapping = {
            "behavior": "behavior_analysis",
            "examples": "example_validation",
            "edge_cases": "edge_case_analysis",
            "version_info": "version_analysis",
            "type_consistency": "type_consistency",
            "performance": "performance_analysis",
        }

        template_name = analysis_mapping.get(primary_analysis, "behavior_analysis")

        # Get function signature string
        signature = function.signature
        params = []
        for param in signature.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str += f": {param.type_annotation}"
            if param.default_value:
                param_str += f" = {param.default_value}"
            params.append(param_str)

        param_str = ", ".join(params)
        return_str = f" -> {signature.return_type}" if signature.return_type else ""
        signature_str = f"def {signature.name}({param_str}){return_str}:"

        # Get source code if available (truncate if too long)
        source_code = (
            getattr(function, "source_code", "") or "# Source code not available"
        )
        if len(source_code) > 2000:
            source_code = source_code[:2000] + "\n# ... (truncated)"

        # Build rule issues summary
        rule_results = context.get("rule_results", [])
        if rule_results:
            rule_issues = []
            for result in rule_results:
                if not result.passed:
                    for issue in result.issues:
                        rule_issues.append(f"- {result.rule_name}: {issue.description}")
            rule_issues_str = (
                "\n".join(rule_issues) if rule_issues else "No rule violations found"
            )
        else:
            rule_issues_str = "No rule analysis performed"

        # Build the prompt
        try:
            user_prompt = format_prompt(
                analysis_type=template_name,
                signature=signature_str,
                source_code=source_code,
                docstring=docstring.raw_text,
                rule_issues=rule_issues_str,
            )
        except Exception as e:
            logger.warning(
                f"Failed to format prompt with template {template_name}: {e}"
            )
            # Fallback to basic prompt
            user_prompt = self._build_fallback_prompt(
                signature_str, source_code, docstring.raw_text
            )

        # System prompt for role definition
        system_prompt = (
            "You are an expert Python documentation analyzer. "
            "Your job is to identify semantic inconsistencies between function "
            "implementations and their documentation that automated rules cannot catch. "
            "Always respond with valid JSON only, no additional text."
        )

        # Add analysis type context if multiple types requested
        if len(analysis_types) > 1:
            other_types = [t for t in analysis_types[1:]]
            user_prompt += f"\n\nADDITIONAL ANALYSIS: Also consider {', '.join(other_types)} if relevant."

        return system_prompt, user_prompt

    def _build_fallback_prompt(
        self, signature: str, source_code: str, docstring: str
    ) -> str:
        """Build a basic fallback prompt when template formatting fails."""
        return f"""
        Analyze this Python function for documentation inconsistencies:

        FUNCTION SIGNATURE:
        {signature}

        IMPLEMENTATION:
        {source_code}

        CURRENT DOCUMENTATION:
        {docstring}

        Find issues where the documentation doesn't match the implementation.
        Return JSON with this format:
        {{
            "issues": [
                {{
                    "type": "description_outdated",
                    "description": "Issue description",
                    "suggestion": "Specific fix suggestion",
                    "confidence": 0.8,
                    "line_number": 1,
                    "details": {{}}
                }}
            ],
            "analysis_notes": "Summary of analysis",
            "confidence": 0.8
        }}
        """

    def merge_with_rule_results(
        self, llm_issues: list[InconsistencyIssue], rule_results: list[RuleCheckResult]
    ) -> list[InconsistencyIssue]:
        """
        Merge LLM and rule engine results intelligently.

        Args:
            llm_issues: Issues found by LLM analysis
            rule_results: Results from rule engine

        Returns:
            Combined and deduplicated list of issues
        """
        merged_issues = []

        # First, add all high-confidence rule issues
        for rule_result in rule_results:
            if rule_result.passed:
                continue

            for issue in rule_result.issues:
                if issue.is_high_confidence():
                    merged_issues.append(issue)

        # Add LLM issues, avoiding duplicates
        for llm_issue in llm_issues:
            # Check if similar issue already exists from rules
            is_duplicate = False
            for existing_issue in merged_issues:
                if self._is_similar_issue(llm_issue, existing_issue):
                    # Combine suggestions if both have them
                    if (
                        llm_issue.suggestion
                        and existing_issue.suggestion
                        and llm_issue.suggestion != existing_issue.suggestion
                    ):
                        combined_suggestion = (
                            f"{existing_issue.suggestion} "
                            f"Additionally: {llm_issue.suggestion}"
                        )
                        existing_issue.suggestion = combined_suggestion
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged_issues.append(llm_issue)

        # Add low-confidence rule issues if no LLM equivalent found
        for rule_result in rule_results:
            if rule_result.passed:
                continue

            for issue in rule_result.issues:
                if not issue.is_high_confidence():
                    # Check if LLM provided better analysis for this issue
                    has_llm_equivalent = any(
                        self._is_similar_issue(issue, llm_issue)
                        for llm_issue in llm_issues
                    )

                    if not has_llm_equivalent:
                        merged_issues.append(issue)

        # Sort by severity and line number
        merged_issues.sort(key=lambda x: (-x.severity_weight, x.line_number))

        logger.debug(
            f"Merged {len(llm_issues)} LLM issues with "
            f"{sum(len(r.issues) for r in rule_results)} rule issues -> "
            f"{len(merged_issues)} total"
        )

        return merged_issues

    def _is_similar_issue(
        self, issue1: InconsistencyIssue, issue2: InconsistencyIssue
    ) -> bool:
        """
        Check if two issues are similar enough to be considered duplicates.

        Args:
            issue1: First issue to compare
            issue2: Second issue to compare

        Returns:
            True if issues are similar enough to merge
        """
        # Same issue type and close line numbers
        if (
            issue1.issue_type == issue2.issue_type
            and abs(issue1.line_number - issue2.line_number) <= 2
        ):
            return True

        # Similar descriptions (basic check)
        desc1_words = set(issue1.description.lower().split())
        desc2_words = set(issue2.description.lower().split())

        if len(desc1_words) > 0 and len(desc2_words) > 0:
            overlap = len(desc1_words & desc2_words)
            similarity = overlap / max(len(desc1_words), len(desc2_words))
            if similarity > 0.6:  # 60% word overlap
                return True

        return False

    async def _check_cache(self, cache_key: str) -> LLMAnalysisResponse | None:
        """Check if cached response exists and is not expired."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT response_json, model, created_at
                    FROM llm_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                response_json, model, created_at = row

                # Check if expired
                ttl_seconds = self.config.cache_ttl_days * 24 * 3600
                if time.time() - created_at > ttl_seconds:
                    # Entry expired, remove it
                    conn.execute(
                        "DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,)
                    )
                    return None

                # Update access statistics
                conn.execute(
                    """
                    UPDATE llm_cache
                    SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )

                # Deserialize response
                response_data = json.loads(response_json)
                response_data["cache_hit"] = True

                return LLMAnalysisResponse(**response_data)

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            return None

    async def _store_cache(self, cache_key: str, response: LLMAnalysisResponse) -> None:
        """Store response in cache."""
        try:
            # Convert response to JSON-serializable dict
            response_dict = {
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "suggestion": issue.suggestion,
                        "line_number": issue.line_number,
                        "confidence": issue.confidence,
                        "details": issue.details,
                    }
                    for issue in response.issues
                ],
                "raw_response": response.raw_response,
                "model_used": response.model_used,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "response_time_ms": response.response_time_ms,
                "cache_hit": False,
            }

            response_json = json.dumps(response_dict)
            request_hash = hashlib.md5(cache_key.encode()).hexdigest()

            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO llm_cache
                    (cache_key, request_hash, response_json, model, created_at, accessed_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (cache_key, request_hash, response_json, response.model_used),
                )

        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def analyze_batch(
        self,
        requests: list[LLMAnalysisRequest],
        max_concurrent: int = 10,
        progress_callback: callable | None = None,
    ) -> list[LLMAnalysisResponse]:
        """
        Analyze multiple functions efficiently with smart batching.

        Implements batching strategies:
        - Group similar functions together (better cache hits)
        - Prioritize by file modification time
        - Limit concurrent requests to avoid rate limits
        - Use asyncio.gather with return_exceptions=True

        Args:
            requests: List of LLM analysis requests
            max_concurrent: Maximum concurrent analysis operations
            progress_callback: Optional callback function for progress updates

        Returns:
            List of LLM analysis responses (same order as requests)
        """
        if not requests:
            return []

        logger.info(f"Starting batch analysis of {len(requests)} requests")

        # Group by analysis type for cache efficiency
        grouped_requests = self._group_requests_for_efficiency(requests)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Track progress
        completed = 0
        total = len(requests)
        results = [None] * total  # Maintain original order

        async def analyze_single_with_semaphore(
            request: LLMAnalysisRequest, index: int
        ):
            """Analyze single request with semaphore control."""
            async with semaphore:
                try:
                    result = await self.analyze_function(request)

                    # Update progress
                    nonlocal completed
                    completed += 1

                    if progress_callback:
                        await asyncio.get_event_loop().run_in_executor(
                            None, progress_callback, completed, total, request
                        )

                    return result, index

                except Exception as e:
                    logger.warning(f"Batch analysis failed for request {index}: {e}")
                    # Return a minimal error response
                    return self._create_error_response(request, str(e)), index

        # Execute all requests with concurrency control
        # Process in groups to optimize cache hits
        all_tasks = []
        for group in grouped_requests:
            group_tasks = [
                analyze_single_with_semaphore(req, idx) for req, idx in group
            ]
            all_tasks.extend(group_tasks)

        # Wait for all requests to complete, handling partial failures
        completed_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Process results and maintain original order
        for result in completed_results:
            if isinstance(result, Exception):
                logger.error(f"Batch analysis exception: {result}")
                continue

            response, original_index = result
            results[original_index] = response

        # Fill any None results with error responses
        for i, result in enumerate(results):
            if result is None:
                results[i] = self._create_error_response(
                    requests[i], "Failed to process request"
                )

        logger.info(f"Batch analysis completed: {len(results)} results")
        return results

    def _group_requests_for_efficiency(
        self, requests: list[LLMAnalysisRequest]
    ) -> list[list[tuple[LLMAnalysisRequest, int]]]:
        """
        Group requests for better cache efficiency and performance.

        Grouping strategies:
        1. Group by analysis type (similar prompts = better cache hits)
        2. Prioritize by function complexity (complex functions first)
        3. Group by file path (similar context)

        Args:
            requests: List of requests to group

        Returns:
            List of groups, each containing (request, original_index) tuples
        """
        # Create request-index pairs for tracking
        indexed_requests = [(req, i) for i, req in enumerate(requests)]

        # Group by analysis type first
        type_groups = {}
        for req, idx in indexed_requests:
            primary_type = req.analysis_types[0] if req.analysis_types else "behavior"
            if primary_type not in type_groups:
                type_groups[primary_type] = []
            type_groups[primary_type].append((req, idx))

        # Further group by file path within each type
        final_groups = []
        for analysis_type, type_group in type_groups.items():
            # Sort by function complexity (more complex first for better parallelization)
            type_group.sort(
                key=lambda x: self._estimate_function_complexity(x[0].function),
                reverse=True,
            )

            # Group by file path
            file_groups = {}
            for req, idx in type_group:
                file_path = req.function.file_path
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append((req, idx))

            # Add file groups to final groups
            final_groups.extend(file_groups.values())

        return final_groups

    def _estimate_function_complexity(self, function: "ParsedFunction") -> int:
        """
        Estimate function complexity for prioritization.

        Args:
            function: Function to analyze

        Returns:
            Complexity score (higher = more complex)
        """
        complexity = 0

        # Parameter count contributes to complexity
        complexity += len(function.signature.parameters) * 2

        # Type annotations suggest more complex functions
        for param in function.signature.parameters:
            if param.type_annotation:
                complexity += 1

        if function.signature.return_type:
            complexity += 1

        # Estimate based on line count if available
        if hasattr(function, "line_count"):
            complexity += function.line_count // 10
        elif hasattr(function, "source_code") and function.source_code:
            complexity += len(function.source_code.split("\n")) // 10

        return complexity

    def _create_error_response(
        self, request: LLMAnalysisRequest, error_message: str
    ) -> LLMAnalysisResponse:
        """
        Create an error response for failed analysis.

        Args:
            request: Original request that failed
            error_message: Error description

        Returns:
            Error response with minimal information
        """
        from .models import InconsistencyIssue

        # Create a generic error issue
        error_issue = InconsistencyIssue(
            issue_type="analysis_error",
            severity="low",
            description=f"Analysis failed: {error_message}",
            suggestion="Manual review recommended",
            line_number=request.function.line_number,
            confidence=0.0,
        )

        return LLMAnalysisResponse(
            issues=[error_issue],
            raw_response=f"Error: {error_message}",
            model_used=self.config.model,
            prompt_tokens=0,
            completion_tokens=0,
            response_time_ms=0.0,
            cache_hit=False,
        )

    async def warm_cache(
        self,
        functions: list["ParsedFunction"],
        max_concurrent: int = 5,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """
        Pre-populate cache for high-value functions.

        Identifies functions that would benefit from caching:
        - Public API functions (no leading underscore)
        - Frequently changed functions
        - Complex functions (>50 lines or >5 parameters)

        Args:
            functions: List of functions to potentially cache
            max_concurrent: Maximum concurrent warming operations
            progress_callback: Optional progress callback

        Returns:
            Statistics about the warming operation
        """
        start_time = time.time()

        # Identify high-value targets
        high_value_functions = []
        for func in functions:
            if self._is_high_value_function(func):
                high_value_functions.append(func)

        logger.info(
            f"Cache warming: {len(high_value_functions)}/{len(functions)} high-value functions identified"
        )

        if not high_value_functions:
            return {
                "total_functions": len(functions),
                "high_value_functions": 0,
                "warming_completed": True,
                "cache_entries_created": 0,
                "warming_time_ms": 0.0,
                "skipped_existing": 0,
            }

        # Create analysis requests for high-value functions
        requests = []
        skipped_existing = 0

        for func in high_value_functions:
            # Check if already cached
            if hasattr(func, "docstring") and func.docstring:
                # Create minimal request for cache key generation
                cache_key = self._generate_cache_key(
                    str(func.signature),
                    str(func.docstring),
                    ["behavior"],  # Default analysis type
                    self.config.model,
                )

                # Check if already cached
                cached_response = await self._get_cached_response(cache_key)
                if cached_response:
                    skipped_existing += 1
                    continue

            # Create analysis request
            from .llm_models import LLMAnalysisRequest

            request = LLMAnalysisRequest(
                function=func,
                docstring=func.docstring,
                analysis_types=self._determine_warming_analysis_types(func),
                rule_results=[],  # No rule results for warming
                related_functions=[],
            )
            requests.append(request)

        if not requests:
            return {
                "total_functions": len(functions),
                "high_value_functions": len(high_value_functions),
                "warming_completed": True,
                "cache_entries_created": 0,
                "warming_time_ms": (time.time() - start_time) * 1000,
                "skipped_existing": skipped_existing,
            }

        # Perform batch analysis with reduced concurrency
        warming_semaphore = asyncio.Semaphore(
            min(max_concurrent, 3)
        )  # Conservative limit

        successful_warming = 0

        async def warm_single_function(request):
            """Warm cache for a single function."""
            async with warming_semaphore:
                try:
                    await self.analyze_function(request)
                    return True
                except Exception as e:
                    logger.warning(
                        f"Cache warming failed for {request.function.signature.name}: {e}"
                    )
                    return False

        # Execute warming with progress tracking
        warming_tasks = [warm_single_function(req) for req in requests]

        if progress_callback:
            # Track progress
            completed = 0

            async def track_progress():
                nonlocal completed
                while completed < len(warming_tasks):
                    await asyncio.sleep(0.5)
                    current_completed = sum(1 for task in warming_tasks if task.done())
                    if current_completed > completed:
                        completed = current_completed
                        await asyncio.get_event_loop().run_in_executor(
                            None, progress_callback, completed, len(warming_tasks)
                        )

        results = await asyncio.gather(*warming_tasks, return_exceptions=True)
        successful_warming = sum(1 for result in results if result is True)

        warming_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Cache warming completed: {successful_warming}/{len(requests)} functions cached"
        )

        return {
            "total_functions": len(functions),
            "high_value_functions": len(high_value_functions),
            "warming_completed": True,
            "cache_entries_created": successful_warming,
            "warming_time_ms": warming_time_ms,
            "skipped_existing": skipped_existing,
        }

    def _is_high_value_function(self, func: "ParsedFunction") -> bool:
        """Determine if a function is high-value for caching."""
        # Public API functions (no leading underscore)
        if not func.signature.name.startswith("_"):
            return True

        # Complex functions based on parameter count
        if len(func.signature.parameters) > 5:
            return True

        # Functions with complex type annotations
        complex_types = 0
        for param in func.signature.parameters:
            if param.type_annotation and any(
                keyword in param.type_annotation.lower()
                for keyword in ["union", "optional", "dict", "list", "callable"]
            ):
                complex_types += 1

        if complex_types > 2:
            return True

        # Functions with return type annotations
        if func.signature.return_type:
            return True

        return False

    def _determine_warming_analysis_types(self, func: "ParsedFunction") -> list[str]:
        """Determine which analysis types to use for cache warming."""
        analysis_types = ["behavior"]  # Always include behavior analysis

        # Add example validation if docstring likely contains examples
        if hasattr(func, "docstring") and func.docstring:
            docstring_text = str(func.docstring)
            if any(
                keyword in docstring_text.lower()
                for keyword in ["example", ">>>", "test"]
            ):
                analysis_types.append("examples")

        # Add type consistency for complex typed functions
        if any(param.type_annotation for param in func.signature.parameters):
            analysis_types.append("type_consistency")

        return analysis_types

    # ========== CHUNK 5: Error Handling & Graceful Degradation ==========

    async def analyze_with_fallback(
        self, request: LLMAnalysisRequest
    ) -> AnalysisResult:
        """
        Analyze with multiple fallback strategies.

        Implements graceful degradation through multiple fallback strategies:
        1. Try full LLM analysis with circuit breaker protection
        2. Try simpler prompts with reduced complexity
        3. Try rule engine only analysis
        4. Return minimal analysis as last resort

        Args:
            request: LLM analysis request

        Returns:
            AnalysisResult with appropriate confidence adjustment
        """
        strategies = [
            # 1. Try full LLM analysis
            (self._analyze_with_llm, 1.0, "full_llm"),
            # 2. Try simpler prompts
            (self._analyze_with_simple_prompts, 0.8, "simple_llm"),
            # 3. Try rule engine only
            (self._analyze_with_rules_only, 0.6, "rules_only"),
            # 4. Return minimal analysis
            (self._minimal_analysis, 0.3, "minimal"),
        ]

        last_error = None

        for strategy_func, confidence_multiplier, strategy_name in strategies:
            try:
                logger.debug(
                    f"Attempting {strategy_name} strategy for {request.function.signature.name}"
                )

                result = await strategy_func(request)

                # Adjust confidence if using degraded strategy
                if confidence_multiplier < 1.0:
                    for issue in result.issues:
                        issue.confidence *= confidence_multiplier
                    result.degraded = True
                    result.degradation_reason = (
                        f"Used {strategy_name} fallback strategy"
                    )
                else:
                    result.degraded = False

                logger.info(f"Analysis successful using {strategy_name} strategy")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue

        # All strategies failed - return empty analysis
        logger.error(
            f"All fallback strategies failed for {request.function.signature.name}"
        )
        return self._empty_analysis(request, last_error)

    async def _analyze_with_llm(self, request: LLMAnalysisRequest) -> AnalysisResult:
        """
        Perform full LLM analysis with circuit breaker protection.

        Args:
            request: LLM analysis request

        Returns:
            AnalysisResult from LLM analysis

        Raises:
            LLMError: If circuit breaker is open or analysis fails
        """
        # Check circuit breaker state
        if not self.circuit_breaker.can_execute():
            raise LLMError(f"Circuit breaker is {self.circuit_breaker.state.value}")

        # Use circuit breaker to protect LLM call
        async def protected_llm_call():
            try:
                # Convert OpenAI errors to our error types
                response = await self.analyze_function(request)
                return self._llm_response_to_analysis_result(request, response)

            except openai.RateLimitError as e:
                raise LLMRateLimitError(str(e), getattr(e, "retry_after", None))
            except openai.APIError as e:
                if "api key" in str(e).lower():
                    raise LLMAPIKeyError(str(e))
                else:
                    raise LLMNetworkError(str(e))
            except asyncio.TimeoutError as e:
                raise LLMTimeoutError(str(e))
            except Exception as e:
                raise LLMError(f"Unexpected error: {e}")

        return await self.circuit_breaker.call(protected_llm_call)

    async def _analyze_with_simple_prompts(
        self, request: LLMAnalysisRequest
    ) -> AnalysisResult:
        """
        Perform LLM analysis with simplified prompts.

        Uses shorter, simpler prompts to reduce the chance of failure.

        Args:
            request: LLM analysis request

        Returns:
            AnalysisResult from simplified LLM analysis
        """
        # Create simplified request with only behavior analysis
        simplified_request = LLMAnalysisRequest(
            function=request.function,
            docstring=request.docstring,
            analysis_types=["behavior"],  # Only basic behavior analysis
            rule_results=[],  # Remove rule context to simplify
            related_functions=[],  # Remove related functions to reduce complexity
        )

        # Use shorter timeout and simpler configuration
        original_timeout = self.config.timeout_seconds
        original_max_tokens = self.config.max_tokens

        try:
            # Temporarily reduce complexity
            self.config.timeout_seconds = min(15, original_timeout)
            self.config.max_tokens = min(500, original_max_tokens)

            response = await self.analyze_function(simplified_request)
            return self._llm_response_to_analysis_result(simplified_request, response)

        finally:
            # Restore original configuration
            self.config.timeout_seconds = original_timeout
            self.config.max_tokens = original_max_tokens

    async def _analyze_with_rules_only(
        self, request: LLMAnalysisRequest
    ) -> AnalysisResult:
        """
        Perform rule-based analysis only.

        Falls back to rule engine when LLM analysis is not available.

        Args:
            request: LLM analysis request

        Returns:
            AnalysisResult from rule engine only
        """
        from .config import RuleEngineConfig
        from .rule_engine import RuleEngine

        # Create rule engine with default configuration
        rule_config = RuleEngineConfig()
        rule_engine = RuleEngine(rule_config)

        # Create matched pair for rule analysis
        from ..matcher.models import MatchConfidence, MatchedPair, MatchType

        matched_pair = MatchedPair(
            function=request.function,
            documentation=request.docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="Rule-only analysis",
            docstring=request.docstring,
        )

        # Run rule analysis
        rule_results = rule_engine.analyze(matched_pair)

        # Convert rule results to issues
        issues = []
        for result in rule_results:
            if not result.passed:
                issues.extend(result.issues)

        start_time = time.time()
        analysis_time = (time.time() - start_time) * 1000

        return AnalysisResult(
            matched_pair=matched_pair,
            issues=issues,
            used_llm=False,
            analysis_time_ms=analysis_time,
            cache_hit=False,
        )

    async def _minimal_analysis(self, request: LLMAnalysisRequest) -> AnalysisResult:
        """
        Return minimal analysis when all other strategies fail.

        Creates a basic analysis result with minimal information.

        Args:
            request: LLM analysis request

        Returns:
            Minimal AnalysisResult
        """
        from ..matcher.models import MatchConfidence, MatchedPair, MatchType

        # Create minimal matched pair
        matched_pair = MatchedPair(
            function=request.function,
            documentation=request.docstring,
            confidence=MatchConfidence.LOW,
            match_type=MatchType.SEMANTIC,
            match_reason="Minimal fallback analysis",
            docstring=request.docstring,
        )

        # Create basic analysis issue
        minimal_issue = InconsistencyIssue(
            issue_type="analysis_unavailable",
            severity="low",
            description="Full analysis unavailable, manual review recommended",
            suggestion="Review function documentation manually for consistency",
            line_number=request.function.line_number,
            confidence=0.1,
        )

        return AnalysisResult(
            matched_pair=matched_pair,
            issues=[minimal_issue],
            used_llm=False,
            analysis_time_ms=1.0,
            cache_hit=False,
        )

    def _empty_analysis(
        self, request: LLMAnalysisRequest, error: Exception = None
    ) -> AnalysisResult:
        """
        Create empty analysis result when all strategies fail.

        Args:
            request: Original analysis request
            error: Last error encountered

        Returns:
            Empty AnalysisResult with error information
        """
        from ..matcher.models import MatchConfidence, MatchedPair, MatchType

        # Create error matched pair
        matched_pair = MatchedPair(
            function=request.function,
            documentation=request.docstring,
            confidence=MatchConfidence.NONE,
            match_type=MatchType.SEMANTIC,
            match_reason="Analysis failed",
            docstring=request.docstring,
        )

        # Create error issue
        error_message = str(error) if error else "Analysis failed for unknown reason"
        error_issue = InconsistencyIssue(
            issue_type="analysis_error",
            severity="low",
            description=f"Analysis failed: {error_message}",
            suggestion="Manual review required - automated analysis unavailable",
            line_number=request.function.line_number,
            confidence=0.0,
            details={"error": error_message, "all_strategies_failed": True},
        )

        return AnalysisResult(
            matched_pair=matched_pair,
            issues=[error_issue],
            used_llm=False,
            analysis_time_ms=0.0,
            cache_hit=False,
        )

    def _llm_response_to_analysis_result(
        self, request: LLMAnalysisRequest, response: LLMAnalysisResponse
    ) -> AnalysisResult:
        """
        Convert LLMAnalysisResponse to AnalysisResult.

        Args:
            request: Original request
            response: LLM response

        Returns:
            AnalysisResult with LLM data
        """
        from ..matcher.models import MatchConfidence, MatchedPair, MatchType

        # Create matched pair
        matched_pair = MatchedPair(
            function=request.function,
            documentation=request.docstring,
            confidence=MatchConfidence.HIGH,
            match_type=MatchType.DIRECT,
            match_reason="LLM analysis",
            docstring=request.docstring,
        )

        return AnalysisResult(
            matched_pair=matched_pair,
            issues=response.issues,
            used_llm=True,
            analysis_time_ms=response.response_time_ms,
            cache_hit=response.cache_hit,
        )

    def get_circuit_breaker_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return self.circuit_breaker.get_stats()

    def get_retry_stats(self) -> dict[str, Any]:
        """Get retry strategy statistics."""
        return self.retry_strategy.get_retry_stats()

    def reset_error_handling(self) -> None:
        """Reset circuit breaker and retry statistics."""
        self.circuit_breaker.reset()
        self.retry_strategy.retry_history.clear()


# Factory functions for different use cases
def create_fast_analyzer() -> LLMAnalyzer:
    """Create analyzer optimized for speed."""
    return LLMAnalyzer(LLMConfig.create_fast_config())


def create_balanced_analyzer() -> LLMAnalyzer:
    """Create analyzer with balanced speed/thoroughness."""
    return LLMAnalyzer(LLMConfig.create_balanced_config())


def create_thorough_analyzer() -> LLMAnalyzer:
    """Create analyzer optimized for thoroughness."""
    return LLMAnalyzer(LLMConfig.create_thorough_config())
