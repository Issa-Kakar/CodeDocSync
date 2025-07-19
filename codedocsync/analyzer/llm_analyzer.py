"""
LLM-powered analysis component for semantic documentation consistency checking.

This module provides the LLMAnalyzer class that handles complex semantic
inconsistencies that the rule engine cannot catch. It includes retry logic,
fallbacks, caching, and specialized analysis methods.

Performance target: <500ms per function (cached)
"""

import asyncio
import json
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from litellm import acompletion

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

from .models import InconsistencyIssue, RuleCheckResult, ISSUE_TYPES
from .prompt_templates import (
    format_prompt,
    get_available_analysis_types,
    validate_llm_response,
    map_llm_issue_type,
)
from .config import AnalysisConfig
from codedocsync.matcher import MatchedPair
from codedocsync.parser import ParsedFunction, ParsedDocstring, RawDocstring


@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis."""

    issues: List[InconsistencyIssue]
    analysis_notes: str
    confidence: float
    llm_model: str
    execution_time_ms: float
    cache_hit: bool = False
    error: Optional[str] = None


class LLMAnalysisError(Exception):
    """Exception raised when LLM analysis fails."""

    pass


class LLMCache:
    """SQLite-based cache for LLM analysis results."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        """
        Initialize LLM cache.

        Args:
            cache_dir: Directory for cache database (uses temp if None)
            ttl_hours: Time-to-live for cache entries in hours
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "codedocsync"

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "llm_analysis_cache.db"
        self.ttl_seconds = ttl_hours * 3600

        self._init_database()

    def _init_database(self):
        """Initialize the cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON llm_cache(created_at)
            """
            )

    def _generate_cache_key(
        self, function_signature: str, docstring: str, analysis_type: str, model: str
    ) -> str:
        """Generate cache key for function + docstring + analysis type + model."""
        content = f"{function_signature}|{docstring}|{analysis_type}|{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(
        self, function_signature: str, docstring: str, analysis_type: str, model: str
    ) -> Optional[LLMAnalysisResult]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(
            function_signature, docstring, analysis_type, model
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT result_json, created_at FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            result_json, created_at = row

            # Check if expired
            if time.time() - created_at > self.ttl_seconds:
                # Remove expired entry
                conn.execute("DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,))
                return None

            # Update access count
            conn.execute(
                "UPDATE llm_cache SET access_count = access_count + 1 WHERE cache_key = ?",
                (cache_key,),
            )

            try:
                result_data = json.loads(result_json)

                # Convert back to LLMAnalysisResult
                issues = []
                for issue_data in result_data.get("issues", []):
                    issue = InconsistencyIssue(
                        issue_type=issue_data["issue_type"],
                        severity=issue_data["severity"],
                        description=issue_data["description"],
                        suggestion=issue_data["suggestion"],
                        line_number=issue_data["line_number"],
                        confidence=issue_data["confidence"],
                        details=issue_data.get("details", {}),
                    )
                    issues.append(issue)

                return LLMAnalysisResult(
                    issues=issues,
                    analysis_notes=result_data["analysis_notes"],
                    confidence=result_data["confidence"],
                    llm_model=result_data["llm_model"],
                    execution_time_ms=result_data["execution_time_ms"],
                    cache_hit=True,
                )

            except (json.JSONDecodeError, KeyError, ValueError):
                # Corrupted cache entry, remove it
                conn.execute("DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,))
                return None

    def set(
        self,
        function_signature: str,
        docstring: str,
        analysis_type: str,
        model: str,
        result: LLMAnalysisResult,
    ):
        """Cache analysis result."""
        cache_key = self._generate_cache_key(
            function_signature, docstring, analysis_type, model
        )

        # Convert result to JSON
        result_data = {
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
                for issue in result.issues
            ],
            "analysis_notes": result.analysis_notes,
            "confidence": result.confidence,
            "llm_model": result.llm_model,
            "execution_time_ms": result.execution_time_ms,
        }

        result_json = json.dumps(result_data)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, result_json, model, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, result_json, model, time.time()),
            )

    def clear_expired(self) -> int:
        """Remove expired cache entries. Returns number of entries removed."""
        cutoff_time = time.time() - self.ttl_seconds

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM llm_cache WHERE created_at < ?", (cutoff_time,)
            )
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(access_count) FROM llm_cache")
            row = cursor.fetchone()
            entry_count, total_accesses = row

            cursor = conn.execute(
                "SELECT COUNT(*) FROM llm_cache WHERE created_at > ?",
                (time.time() - self.ttl_seconds,),
            )
            valid_entries = cursor.fetchone()[0]

            return {
                "total_entries": entry_count or 0,
                "valid_entries": valid_entries or 0,
                "total_accesses": total_accesses or 0,
                "cache_file_size_mb": (
                    self.db_path.stat().st_size / (1024 * 1024)
                    if self.db_path.exists()
                    else 0
                ),
            }


class LLMAnalyzer:
    """LLM-powered analyzer for semantic documentation consistency."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize LLM analyzer.

        Args:
            config: Analysis configuration (creates default if None)

        Raises:
            ImportError: If litellm is not available
            ValueError: If configuration is invalid
        """
        if not HAS_LITELLM:
            raise ImportError(
                "litellm package is required for LLM analysis. "
                "Install with: pip install litellm"
            )

        self.config = config or AnalysisConfig()

        # Validate LLM configuration
        if self.config.use_llm and not self.config.llm_provider:
            raise ValueError("llm_provider must be specified when use_llm is True")

        # Initialize cache
        self.cache = (
            LLMCache(ttl_hours=self.config.cache_ttl_hours)
            if self.config.enable_cache
            else None
        )

        # Model fallback chain
        self.primary_model = f"{self.config.llm_provider}/{self.config.llm_model}"
        self.fallback_models = [
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku-20240307",
        ]
        if self.primary_model in self.fallback_models:
            self.fallback_models.remove(self.primary_model)

    async def analyze_consistency(
        self,
        pair: MatchedPair,
        rule_results: Optional[List[RuleCheckResult]] = None,
        analysis_types: Optional[List[str]] = None,
    ) -> LLMAnalysisResult:
        """
        Analyze function-documentation consistency using LLM.

        Args:
            pair: The matched function-documentation pair
            rule_results: Optional results from rule engine
            analysis_types: Types of analysis to perform (default: behavior_analysis)

        Returns:
            LLMAnalysisResult: Analysis results with detected issues

        Raises:
            LLMAnalysisError: If analysis fails unrecoverably
        """
        if not self.config.use_llm:
            return LLMAnalysisResult(
                issues=[],
                analysis_notes="LLM analysis disabled in configuration",
                confidence=1.0,
                llm_model="none",
                execution_time_ms=0.0,
            )

        start_time = time.time()

        # Default to behavior analysis if no types specified
        if analysis_types is None:
            analysis_types = ["behavior_analysis"]

        # Validate analysis types
        available_types = get_available_analysis_types()
        for analysis_type in analysis_types:
            if analysis_type not in available_types:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Extract function information
        function_signature = self._get_function_signature(pair.function)
        source_code = self._get_source_code(pair.function)
        docstring = self._get_docstring_text(pair)

        if not docstring:
            return LLMAnalysisResult(
                issues=[],
                analysis_notes="No docstring available for analysis",
                confidence=1.0,
                llm_model=self.primary_model,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Aggregate results from all analysis types
        all_issues = []
        analysis_notes = []
        min_confidence = 1.0
        model_used = self.primary_model

        for analysis_type in analysis_types:
            try:
                result = await self._analyze_single_type(
                    analysis_type=analysis_type,
                    function_signature=function_signature,
                    source_code=source_code,
                    docstring=docstring,
                    rule_results=rule_results,
                )

                all_issues.extend(result.issues)
                analysis_notes.append(f"{analysis_type}: {result.analysis_notes}")
                min_confidence = min(min_confidence, result.confidence)
                model_used = result.llm_model

            except Exception as e:
                analysis_notes.append(f"{analysis_type}: Failed - {str(e)}")
                min_confidence = 0.5  # Lower confidence if any analysis fails

        execution_time = (time.time() - start_time) * 1000

        return LLMAnalysisResult(
            issues=all_issues,
            analysis_notes="; ".join(analysis_notes),
            confidence=min_confidence,
            llm_model=model_used,
            execution_time_ms=execution_time,
        )

    async def _analyze_single_type(
        self,
        analysis_type: str,
        function_signature: str,
        source_code: str,
        docstring: str,
        rule_results: Optional[List[RuleCheckResult]] = None,
    ) -> LLMAnalysisResult:
        """Analyze a single analysis type."""
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(
                function_signature, docstring, analysis_type, self.primary_model
            )
            if cached_result:
                return cached_result

        # Prepare rule issues summary
        rule_issues_summary = "None"
        if rule_results:
            rule_issues = []
            for result in rule_results:
                if not result.passed:
                    rule_issues.append(
                        f"{result.rule_name}: {len(result.issues)} issues"
                    )
            if rule_issues:
                rule_issues_summary = "; ".join(rule_issues)

        # Format prompt
        prompt = format_prompt(
            analysis_type=analysis_type,
            signature=function_signature,
            source_code=source_code,
            docstring=docstring,
            rule_issues=rule_issues_summary,
        )

        # Try analysis with retry logic
        result = await self._analyze_with_retry(prompt, analysis_type)

        # Cache the result
        if self.cache and not result.error:
            self.cache.set(
                function_signature, docstring, analysis_type, result.llm_model, result
            )

        return result

    async def _analyze_with_retry(
        self, prompt: str, analysis_type: str, max_retries: int = 3
    ) -> LLMAnalysisResult:
        """Analyze with comprehensive error handling and retries."""
        models_to_try = [self.primary_model] + self.fallback_models
        errors = []

        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    start_time = time.time()

                    response = await acompletion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens,
                        timeout=self.config.llm_timeout_seconds,
                    )

                    execution_time = (time.time() - start_time) * 1000

                    # Parse response
                    content = response.choices[0].message.content.strip()
                    result = self._parse_llm_response(content, model, execution_time)

                    return result

                except Exception as e:
                    error_msg = f"Model {model}, attempt {attempt + 1}: {str(e)}"
                    errors.append(error_msg)

                    # Wait before retry (exponential backoff)
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 1.0
                        await asyncio.sleep(wait_time)

        # All attempts failed, return degraded result
        return LLMAnalysisResult(
            issues=[],
            analysis_notes=f"LLM analysis failed: {'; '.join(errors)}",
            confidence=0.0,
            llm_model="failed",
            execution_time_ms=0.0,
            error=f"All models failed: {'; '.join(errors)}",
        )

    def _parse_llm_response(
        self, content: str, model: str, execution_time_ms: float
    ) -> LLMAnalysisResult:
        """Parse LLM response into structured result."""
        try:
            # Try to extract JSON from response
            # Sometimes LLMs wrap JSON in code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            else:
                json_str = content

            response_data = json.loads(json_str)

            # Validate response structure
            if not validate_llm_response(response_data):
                raise ValueError("Response does not match expected structure")

            # Convert to our issue objects
            issues = []
            for issue_data in response_data.get("issues", []):
                # Map LLM issue type to our constants
                mapped_type = map_llm_issue_type(issue_data["type"])

                # Get default severity for the mapped type
                default_severity = ISSUE_TYPES.get(mapped_type, "medium")

                issue = InconsistencyIssue(
                    issue_type=mapped_type,
                    severity=default_severity,
                    description=issue_data["description"],
                    suggestion=issue_data["suggestion"],
                    line_number=issue_data.get("line_number", 1),
                    confidence=float(issue_data["confidence"]),
                    details=issue_data.get("details", {}),
                )
                issues.append(issue)

            return LLMAnalysisResult(
                issues=issues,
                analysis_notes=response_data.get(
                    "analysis_notes", "LLM analysis completed"
                ),
                confidence=float(response_data["confidence"]),
                llm_model=model,
                execution_time_ms=execution_time_ms,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Could not parse response, return error result
            return LLMAnalysisResult(
                issues=[],
                analysis_notes=f"Failed to parse LLM response: {str(e)}",
                confidence=0.0,
                llm_model=model,
                execution_time_ms=execution_time_ms,
                error=f"Parse error: {str(e)}",
            )

    def _get_function_signature(self, function: ParsedFunction) -> str:
        """Extract readable function signature."""
        sig = function.signature
        params = []

        for param in sig.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str += f": {param.type_annotation}"
            if param.default_value:
                param_str += f" = {param.default_value}"
            params.append(param_str)

        param_str = ", ".join(params)
        return_str = f" -> {sig.return_type}" if sig.return_type else ""

        return f"def {sig.name}({param_str}){return_str}:"

    def _get_source_code(self, function: ParsedFunction) -> str:
        """Extract function source code."""
        # For now, return a placeholder since we don't have source code extraction
        # In a full implementation, this would extract the actual function body
        return f"# Function implementation at {function.file_path}:{function.line_number}\n# Source code extraction not yet implemented"

    def _get_docstring_text(self, pair: MatchedPair) -> str:
        """Extract docstring text from matched pair."""
        # Check function's docstring first
        if pair.function.docstring:
            if isinstance(pair.function.docstring, RawDocstring):
                return pair.function.docstring.raw_text
            elif isinstance(pair.function.docstring, ParsedDocstring):
                return pair.function.docstring.raw_text

        # Check paired docstring
        if pair.docstring:
            if isinstance(pair.docstring, RawDocstring):
                return pair.docstring.raw_text
            elif isinstance(pair.docstring, ParsedDocstring):
                return pair.docstring.raw_text

        return ""

    async def analyze_behavior(self, pair: MatchedPair) -> LLMAnalysisResult:
        """Specialized method for behavior analysis."""
        return await self.analyze_consistency(
            pair, analysis_types=["behavior_analysis"]
        )

    async def analyze_examples(self, pair: MatchedPair) -> LLMAnalysisResult:
        """Specialized method for example validation."""
        return await self.analyze_consistency(
            pair, analysis_types=["example_validation"]
        )

    async def analyze_edge_cases(self, pair: MatchedPair) -> LLMAnalysisResult:
        """Specialized method for edge case detection."""
        return await self.analyze_consistency(
            pair, analysis_types=["edge_case_analysis"]
        )

    async def analyze_version_info(self, pair: MatchedPair) -> LLMAnalysisResult:
        """Specialized method for version/deprecation analysis."""
        return await self.analyze_consistency(pair, analysis_types=["version_analysis"])

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get LLM cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """Clear LLM cache. Returns number of entries removed."""
        if not self.cache:
            return 0

        if older_than_hours is None:
            # Clear all entries
            with sqlite3.connect(self.cache.db_path) as conn:
                cursor = conn.execute("DELETE FROM llm_cache")
                return cursor.rowcount
        else:
            # Clear entries older than specified hours
            cutoff_time = time.time() - (older_than_hours * 3600)
            with sqlite3.connect(self.cache.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM llm_cache WHERE created_at < ?", (cutoff_time,)
                )
                return cursor.rowcount
