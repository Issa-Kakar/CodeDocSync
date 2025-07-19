"""
Integration module for the analyzer - orchestrates rule engine and LLM analysis.

This module contains the main analyze_matched_pair function that brings together
all analyzer components into a cohesive analysis pipeline.
"""

import asyncio
import time
from typing import Optional, List

from codedocsync.matcher import MatchedPair
from .models import AnalysisResult, InconsistencyIssue, RuleCheckResult
from .rule_engine import RuleEngine
from .llm_analyzer import LLMAnalyzer
from .config import AnalysisConfig, get_development_config


class AnalysisCache:
    """Simple in-memory cache for analysis results."""

    def __init__(self, max_size: int = 1000):
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

    def get(
        self, pair: MatchedPair, config: AnalysisConfig
    ) -> Optional[AnalysisResult]:
        """Get cached analysis result."""
        key = self._generate_key(pair, config)
        result = self.cache.get(key)
        if result:
            # Update cache hit flag
            result.cache_hit = True
        return result

    def put(self, pair: MatchedPair, config: AnalysisConfig, result: AnalysisResult):
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
    config: Optional[AnalysisConfig] = None,
    cache: Optional[AnalysisCache] = None,
    rule_engine: Optional[RuleEngine] = None,
    llm_analyzer: Optional[LLMAnalyzer] = None,
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

    # Validate inputs
    if not pair:
        raise ValueError("matched_pair cannot be None")
    if not pair.function:
        raise ValueError("matched_pair.function cannot be None")

    # Use defaults if not provided
    if config is None:
        config = get_development_config()

    # Check cache first
    if cache and config.enable_cache:
        cached_result = cache.get(pair, config)
        if cached_result:
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
        llm_analyzer = LLMAnalyzer(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            cache_dir=None,  # Use default cache directory
        )

    # Step 1: Run rule engine checks (fast path)
    rule_start_time = time.time()
    try:
        rule_issues = rule_engine.check_matched_pair(
            pair, config.rule_engine.confidence_threshold
        )
        rule_time_ms = (time.time() - rule_start_time) * 1000
    except Exception:
        # If rule engine fails, continue with empty rule results
        rule_issues = []
        rule_time_ms = (time.time() - rule_start_time) * 1000
        # In production, you might want to log this error

    # Step 2: Determine if LLM analysis needed
    used_llm = False
    llm_issues: List[InconsistencyIssue] = []

    if config.use_llm and llm_analyzer:
        # Check if we have high-confidence rule issues that skip LLM
        high_confidence_issues = [
            issue
            for issue in rule_issues
            if issue.confidence >= config.rule_engine.confidence_threshold
        ]

        # Only run LLM if we don't have high-confidence critical issues
        should_run_llm = not high_confidence_issues or not any(
            issue.severity == "critical" for issue in high_confidence_issues
        )

        if should_run_llm:
            # Step 3: Run LLM analysis
            try:
                # Create rule results for LLM context
                rule_results = (
                    [
                        RuleCheckResult(
                            rule_name="combined_rules",
                            passed=len(rule_issues) == 0,
                            confidence=(
                                min([issue.confidence for issue in rule_issues])
                                if rule_issues
                                else 1.0
                            ),
                            issues=rule_issues,
                            execution_time_ms=rule_time_ms,
                        )
                    ]
                    if rule_issues
                    else []
                )

                llm_result = await llm_analyzer.analyze_consistency(
                    pair, rule_results=rule_results
                )
                llm_issues = llm_result.issues
                used_llm = True
            except Exception:
                # If LLM fails, continue with just rule results
                pass
                # In production, you might want to log this error

    # Step 4: Merge and sort results
    all_issues = rule_issues + llm_issues

    # Remove duplicates (simple deduplication by description)
    seen_descriptions = set()
    unique_issues = []
    for issue in all_issues:
        if issue.description not in seen_descriptions:
            unique_issues.append(issue)
            seen_descriptions.add(issue.description)

    # Sort by severity if requested
    if config.sort_by_severity:
        unique_issues = sorted(
            unique_issues, key=lambda x: x.severity_weight, reverse=True
        )

    # Step 5: Create final result
    total_time_ms = (time.time() - start_time) * 1000

    result = AnalysisResult(
        matched_pair=pair,
        issues=unique_issues,
        used_llm=used_llm,
        analysis_time_ms=total_time_ms,
        rule_results=None,  # Include if requested in config
        cache_hit=False,
    )

    # Add rule results if requested
    if config.include_rule_results and rule_issues:
        result.rule_results = [
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
        ]

    # Step 6: Cache the result
    if cache and config.enable_cache:
        cache.put(pair, config, result)

    return result


async def analyze_multiple_pairs(
    pairs: List[MatchedPair],
    config: Optional[AnalysisConfig] = None,
    cache: Optional[AnalysisCache] = None,
    rule_engine: Optional[RuleEngine] = None,
    llm_analyzer: Optional[LLMAnalyzer] = None,
) -> List[AnalysisResult]:
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

        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]

            # Use semaphore to limit concurrent executions
            semaphore = asyncio.Semaphore(max_workers)

            async def analyze_with_semaphore(pair):
                async with semaphore:
                    return await analyze_matched_pair(
                        pair, config, cache, rule_engine, llm_analyzer
                    )

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
