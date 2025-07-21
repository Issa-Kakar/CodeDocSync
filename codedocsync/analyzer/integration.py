"""
Integration module for the analyzer - orchestrates rule engine and LLM analysis.

This module contains the main analyze_matched_pair function that brings together
all analyzer components into a cohesive analysis pipeline.
"""

import ast
import asyncio
import logging
import time
from typing import Any

from codedocsync.matcher import MatchedPair
from codedocsync.parser import ParsedDocstring, ParsedFunction

from .config import AnalysisConfig, get_development_config
from .llm_analyzer import LLMAnalyzer
from .llm_models import LLMAnalysisRequest
from .models import AnalysisResult, InconsistencyIssue, RuleCheckResult
from .rule_engine import RuleEngine

# Configure logger
logger = logging.getLogger(__name__)


class IntegrationMetrics:
    """Simple metrics collector for integration monitoring."""

    def __init__(self) -> None:
        self.total_analyses = 0
        self.cache_hits = 0
        self.llm_calls = 0
        self.rule_only_analyses = 0
        self.total_issues_found = 0
        self.total_time_ms = 0.0
        self.errors = 0

    def record_analysis(
        self, result: AnalysisResult, cache_hit: bool, error: bool = False
    ) -> None:
        """Record metrics from an analysis run."""
        self.total_analyses += 1

        if error:
            self.errors += 1
            return

        if cache_hit:
            self.cache_hits += 1

        if result.used_llm:
            self.llm_calls += 1
        else:
            self.rule_only_analyses += 1

        self.total_issues_found += len(result.issues)
        self.total_time_ms += result.analysis_time_ms

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "total_analyses": self.total_analyses,
            "cache_hit_rate": (
                self.cache_hits / self.total_analyses if self.total_analyses > 0 else 0
            ),
            "llm_usage_rate": (
                self.llm_calls / self.total_analyses if self.total_analyses > 0 else 0
            ),
            "avg_issues_per_analysis": (
                self.total_issues_found / self.total_analyses
                if self.total_analyses > 0
                else 0
            ),
            "avg_time_ms": (
                self.total_time_ms / self.total_analyses
                if self.total_analyses > 0
                else 0
            ),
            "error_rate": (
                self.errors / self.total_analyses if self.total_analyses > 0 else 0
            ),
        }

    def log_stats(self) -> None:
        """Log current statistics."""
        stats = self.get_stats()
        logger.info(f"Integration metrics: {stats}")


# Global metrics instance
metrics = IntegrationMetrics()


def _should_use_llm(
    rule_results: list[RuleCheckResult], config: AnalysisConfig, pair: MatchedPair
) -> bool:
    """
    Determine if LLM analysis is needed based on rule results and context.

    Use LLM if:
    1. Any rule has confidence < 0.9
    2. Config explicitly requests semantic analysis
    3. Function is complex (cyclomatic complexity > 10)
    4. Docstring has examples to validate
    5. Function has decorators affecting behavior
    """
    if not config.use_llm:
        return False

    # Check rule confidence
    low_confidence_rules = [r for r in rule_results if r.confidence < 0.9]
    if low_confidence_rules:
        logger.debug(
            f"Found {len(low_confidence_rules)} low confidence rules, using LLM"
        )
        return True

    # Check if semantic analysis explicitly requested
    if config.llm_analysis_types and "behavior" in config.llm_analysis_types:
        logger.debug("Semantic analysis explicitly requested in config")
        return True

    # Check function complexity
    function = pair.function
    if hasattr(function, "complexity") and function.complexity > 10:
        logger.debug(
            f"Function {function.signature.name} has high complexity ({function.complexity}), using LLM"
        )
        return True

    # Check if docstring has examples
    if pair.docstring and hasattr(pair.docstring, "examples"):
        if pair.docstring.examples:
            return True

    # Check for behavior-affecting decorators
    if hasattr(function.signature, "decorators"):
        behavior_decorators = {
            "classmethod",
            "staticmethod",
            "property",
            "cached_property",
        }
        for decorator in function.signature.decorators:
            if any(dec in str(decorator) for dec in behavior_decorators):
                logger.debug(f"Found behavior-affecting decorator: {decorator}")
                return True

    logger.debug("No conditions met for LLM usage")
    return False


def _determine_analysis_types(
    pair: MatchedPair, rule_results: list[RuleCheckResult]
) -> list[str]:
    """Determine which LLM analysis types are needed."""
    analysis_types = []
    logger.debug(f"Determining analysis types for {pair.function.signature.name}")

    # Always check behavior for complex functions
    if hasattr(pair.function, "complexity") and pair.function.complexity > 5:
        analysis_types.append("behavior")

    # Check examples if docstring contains them
    if pair.docstring and hasattr(pair.docstring, "examples"):
        if pair.docstring.examples:
            analysis_types.append("examples")

    # Check edge cases for functions with conditionals
    if hasattr(pair.function, "body") and pair.function.body:
        # Simple heuristic: check for if/try statements
        try:
            if any(
                isinstance(node, ast.If | ast.Try)
                for node in ast.walk(ast.parse(pair.function.body))
            ):
                analysis_types.append("edge_cases")
        except (SyntaxError, TypeError, ValueError):
            # If we can't parse the body, skip edge case analysis
            logger.debug(
                f"Could not parse function body for edge case analysis: {pair.function.signature.name}"
            )

    # Check version info if decorators present
    if hasattr(pair.function.signature, "decorators"):
        if any("deprecated" in str(d) for d in pair.function.signature.decorators):
            analysis_types.append("version_info")

    # Check type consistency for low confidence type rules
    type_rules = [
        r for r in rule_results if "type" in r.rule_name and r.confidence < 0.9
    ]
    if type_rules:
        analysis_types.append("type_consistency")

    # Default to behavior if no specific types identified
    if not analysis_types:
        analysis_types = ["behavior"]

    logger.debug(f"Selected analysis types: {analysis_types}")
    return analysis_types


def _find_related_functions(
    function: ParsedFunction, max_functions: int = 3
) -> list[ParsedFunction]:
    """Find related functions for context (simplified version)."""
    # In a real implementation, this would:
    # 1. Find functions in the same class
    # 2. Find functions with similar names
    # 3. Find functions that call or are called by this function
    # For now, return empty list
    return []


def _create_llm_request(
    pair: MatchedPair, rule_results: list[RuleCheckResult], config: AnalysisConfig
) -> LLMAnalysisRequest:
    """Create optimized LLM request from matched pair."""
    # Determine which analysis types needed
    analysis_types = _determine_analysis_types(pair, rule_results)

    # Override with config if specified
    if config.llm_analysis_types:
        analysis_types = config.llm_analysis_types

    # Gather context
    related_functions = _find_related_functions(pair.function)

    # Build request
    return LLMAnalysisRequest(
        function=pair.function,
        docstring=pair.docstring
        or ParsedDocstring(
            format="none",
            summary="",
            parameters=[],
            returns=None,
            raises=[],
            raw_text="",
        ),
        analysis_types=analysis_types,
        rule_results=rule_results,
        related_functions=related_functions[:3],  # Limit context
    )


def _merge_results(
    rule_issues: list[InconsistencyIssue], llm_issues: list[InconsistencyIssue]
) -> list[InconsistencyIssue]:
    """
    Merge rule engine and LLM results intelligently.

    Rules:
    1. De-duplicate by issue type + line number
    2. Prefer high-confidence results
    3. Combine suggestions when both exist
    4. Sort by severity, then line number
    """
    # Create a map for deduplication
    issue_map: dict[tuple, InconsistencyIssue] = {}

    # Process rule issues first (generally higher confidence)
    for issue in rule_issues:
        key = (issue.issue_type, issue.line_number)
        if key not in issue_map or issue.confidence > issue_map[key].confidence:
            issue_map[key] = issue

    # Process LLM issues
    for issue in llm_issues:
        key = (issue.issue_type, issue.line_number)
        if key in issue_map:
            # Combine suggestions if both exist
            existing = issue_map[key]
            if existing.confidence < 0.9 and issue.confidence >= 0.8:
                # LLM has good confidence, merge suggestions
                combined_suggestion = existing.suggestion
                if issue.suggestion not in combined_suggestion:
                    combined_suggestion += f"\n\nAlternatively: {issue.suggestion}"
                issue.suggestion = combined_suggestion
                issue_map[key] = issue
        else:
            # New issue from LLM
            issue_map[key] = issue

    # Convert back to list and sort
    merged_issues = list(issue_map.values())

    # Sort by severity weight (high to low), then line number
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    merged_issues.sort(
        key=lambda x: (-severity_order.get(x.severity, 0), x.line_number)
    )

    logger.debug(
        f"Merged {len(rule_issues)} rule issues and {len(llm_issues)} LLM issues into {len(merged_issues)} unique issues"
    )
    return merged_issues


class AnalysisCache:
    """Simple in-memory cache for analysis results."""

    def __init__(self, max_size: int = 1000) -> None:
        self.cache: dict = {}
        self.max_size = max_size

    def _generate_key(self, pair: MatchedPair, config: AnalysisConfig) -> str:
        """Generate cache key for a matched pair and config."""
        # Create a simple hash based on function signature and config
        func_sig = f"{pair.function.file_path}:{pair.function.line_number}:{pair.function.signature.name}"
        config_hash = hash(
            (
                tuple(config.rule_engine.enabled_rules or []),
                config.use_llm,
                config.llm_model,
                config.rule_engine.confidence_threshold,
            )
        )
        return f"{func_sig}:{config_hash}"

    def get(self, pair: MatchedPair, config: AnalysisConfig) -> AnalysisResult | None:
        """Get cached analysis result."""
        key = self._generate_key(pair, config)
        result = self.cache.get(key)
        if result:
            # Update cache hit flag
            result.cache_hit = True
        return result

    def put(
        self, pair: MatchedPair, config: AnalysisConfig, result: AnalysisResult
    ) -> None:
        """Store analysis result in cache."""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        key = self._generate_key(pair, config)
        result.cache_hit = False  # This is a fresh result
        self.cache[key] = result


async def analyze_matched_pair(
    pair: MatchedPair,
    config: AnalysisConfig | None = None,
    cache: AnalysisCache | None = None,
    rule_engine: RuleEngine | None = None,
    llm_analyzer: LLMAnalyzer | None = None,
) -> AnalysisResult:
    """
    Main entry point for analyzing a matched function-documentation pair.

    This function orchestrates the complete analysis process:
    1. Run rule engine checks (fast path)
    2. Determine if LLM analysis needed based on confidence
    3. Run LLM analysis if required
    4. Merge and sort results
    5. Generate final suggestions

    Args:
        pair: The matched function-documentation pair to analyze
        config: Optional analysis configuration (uses defaults if None)
        cache: Optional cache instance for performance
        rule_engine: Optional rule engine instance (creates default if None)
        llm_analyzer: Optional LLM analyzer instance (creates default if None)

    Returns:
        AnalysisResult: Complete analysis with all detected issues

    Raises:
        ValueError: If the matched pair is invalid
        AnalysisError: If analysis fails unrecoverably
    """
    start_time = time.time()

    logger.info(
        f"Starting analysis for {pair.function.signature.name if pair and pair.function else 'unknown function'}"
    )

    # Validate inputs
    if not pair:
        logger.error("matched_pair cannot be None")
        raise ValueError("matched_pair cannot be None")
    if not pair.function:
        logger.error("matched_pair.function cannot be None")
        raise ValueError("matched_pair.function cannot be None")

    # Use defaults if not provided
    if config is None:
        config = get_development_config()

    # Check cache first
    if cache and config.enable_cache:
        cached_result = cache.get(pair, config)
        if cached_result:
            logger.info(f"Cache hit for {pair.function.signature.name}")
            metrics.record_analysis(cached_result, cache_hit=True)
            return cached_result

    # Initialize components if not provided
    if rule_engine is None:
        rule_engine = RuleEngine(
            enabled_rules=config.rule_engine.enabled_rules,
            performance_mode=config.rule_engine.performance_mode,
            severity_overrides=config.rule_engine.severity_overrides,
            confidence_threshold=config.rule_engine.confidence_threshold,
        )

    if llm_analyzer is None and config.use_llm:
        # Create LLMConfig from AnalysisConfig
        from .llm_config import LLMConfig

        llm_config = LLMConfig(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout_seconds=int(config.llm_timeout_seconds),
        )
        llm_analyzer = LLMAnalyzer(llm_config)

    # Step 1: Run rule engine checks (fast path)
    rule_start_time = time.time()
    try:
        logger.debug("Running rule engine checks")
        rule_issues = rule_engine.check_matched_pair(
            pair, config.rule_engine.confidence_threshold
        )
        rule_time_ms = (time.time() - rule_start_time) * 1000
        logger.info(
            f"Rule engine completed in {rule_time_ms:.2f}ms, found {len(rule_issues)} issues"
        )
    except Exception as e:
        # If rule engine fails, continue with empty rule results
        rule_issues = []
        rule_time_ms = (time.time() - rule_start_time) * 1000
        logger.warning(f"Rule engine failed: {str(e)}, continuing with empty results")

    # Convert rule issues to RuleCheckResult for context
    rule_results = []
    if rule_issues:
        rule_results.append(
            RuleCheckResult(
                rule_name="rule_engine_combined",
                passed=len(rule_issues) == 0,
                confidence=(
                    min([issue.confidence for issue in rule_issues])
                    if rule_issues
                    else 1.0
                ),
                issues=rule_issues,
                execution_time_ms=rule_time_ms,
            )
        )

    # Step 2: Determine if LLM analysis needed
    needs_llm = _should_use_llm(rule_results, config, pair)
    used_llm = False
    llm_issues: list[InconsistencyIssue] = []

    # Step 3: Run LLM analysis if needed
    if needs_llm and llm_analyzer:
        try:
            logger.info(f"Running LLM analysis for {pair.function.signature.name}")
            llm_request = _create_llm_request(pair, rule_results, config)
            llm_result = await llm_analyzer.analyze_function(llm_request)
            llm_issues = llm_result.issues
            used_llm = True
            logger.info(
                f"LLM analysis completed, found {len(llm_issues)} additional issues"
            )
        except Exception as e:
            # If LLM fails, continue with just rule results
            logger.warning(
                f"LLM analysis failed: {str(e)}, continuing with rule results only"
            )
            pass

    # Step 4: Merge results intelligently
    final_issues = _merge_results(rule_issues, llm_issues)

    # Step 5: Create final result
    total_time_ms = (time.time() - start_time) * 1000

    logger.info(
        f"Analysis completed in {total_time_ms:.2f}ms: {len(final_issues)} total issues, LLM used: {used_llm}"
    )

    result = AnalysisResult(
        matched_pair=pair,
        issues=final_issues,
        used_llm=used_llm,
        analysis_time_ms=total_time_ms,
        rule_results=None,  # Include if requested in config
        cache_hit=False,
    )

    # Add rule results if requested
    if config.include_rule_results and rule_results:
        result.rule_results = rule_results

    # Step 6: Cache the result
    if cache and config.enable_cache:
        cache.put(pair, config, result)
        logger.debug(f"Cached analysis result for {pair.function.signature.name}")

    # Record metrics
    metrics.record_analysis(result, cache_hit=False)

    # Log stats periodically (every 100 analyses)
    if metrics.total_analyses % 100 == 0:
        metrics.log_stats()

    return result


async def analyze_multiple_pairs(
    pairs: list[MatchedPair],
    config: AnalysisConfig | None = None,
    cache: AnalysisCache | None = None,
    rule_engine: RuleEngine | None = None,
    llm_analyzer: LLMAnalyzer | None = None,
) -> list[AnalysisResult]:
    """
    Analyze multiple matched pairs with optional parallelization.

    Args:
        pairs: List of matched pairs to analyze
        config: Analysis configuration
        cache: Optional shared cache
        rule_engine: Shared rule engine instance
        llm_analyzer: Shared LLM analyzer instance

    Returns:
        List of analysis results in the same order as input pairs
    """
    if not pairs:
        return []

    # Use defaults if not provided
    if config is None:
        config = get_development_config()

    if config.parallel_analysis and len(pairs) > 1:
        # Parallel analysis with batching
        batch_size = config.batch_size
        max_workers = min(config.max_parallel_workers, len(pairs))

        # Use semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_workers)

        async def analyze_with_semaphore(pair: MatchedPair) -> AnalysisResult:
            async with semaphore:
                return await analyze_matched_pair(
                    pair, config, cache, rule_engine, llm_analyzer
                )

        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]

            batch_results = await asyncio.gather(
                *[analyze_with_semaphore(pair) for pair in batch]
            )
            results.extend(batch_results)

        return results
    else:
        # Sequential analysis
        results = []
        for pair in pairs:
            result = await analyze_matched_pair(
                pair, config, cache, rule_engine, llm_analyzer
            )
            results.append(result)
        return results


class AnalysisError(Exception):
    """Exception raised when analysis fails unrecoverably."""

    pass


def get_integration_metrics() -> dict[str, Any]:
    """Get current integration metrics for monitoring."""
    return metrics.get_stats()


def reset_integration_metrics() -> None:
    """Reset integration metrics (useful for testing)."""
    global metrics
    metrics = IntegrationMetrics()
