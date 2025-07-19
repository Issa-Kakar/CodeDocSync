"""
Analyzer module for detecting inconsistencies between functions and documentation.

This module provides rule-based and LLM-powered analysis to identify when
function signatures don't match their documentation. It's designed to run fast
deterministic checks first, then fall back to LLM analysis only when needed.
"""

from typing import TYPE_CHECKING

# Import all models and constants
from .models import (
    InconsistencyIssue,
    RuleCheckResult,
    AnalysisResult,
    ISSUE_TYPES,
    SEVERITY_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
    RULE_CATEGORIES,
)

# Import implemented components
from .rule_engine import RuleEngine

# Import utility functions
from .rule_engine_utils import (
    normalize_type_string,
    compare_types,
    extract_base_type,
    generate_parameter_suggestion,
    generate_docstring_template,
    format_code_suggestion,
    get_parameter_statistics,
)

# Import configuration classes
from .config import (
    RuleEngineConfig,
    AnalysisConfig,
    get_fast_config,
    get_thorough_config,
    get_development_config,
)

# Import LLM components (now implemented)
from .llm_analyzer import LLMAnalyzer, LLMAnalysisResult, LLMCache, LLMAnalysisError
from .prompt_templates import (
    get_prompt_template,
    format_prompt,
    get_available_analysis_types,
    validate_llm_response,
)

# TYPE_CHECKING imports for future components
if TYPE_CHECKING:
    from .models import AnalysisCache

# Import matcher models we depend on
from codedocsync.matcher import MatchedPair


# Import main integration functions
from .integration import (
    analyze_matched_pair,
    analyze_multiple_pairs,
    AnalysisCache,
    AnalysisError,
)


__all__ = [
    # Data models
    "InconsistencyIssue",
    "RuleCheckResult",
    "AnalysisResult",
    "LLMAnalysisResult",
    # Constants
    "ISSUE_TYPES",
    "SEVERITY_WEIGHTS",
    "CONFIDENCE_THRESHOLDS",
    "RULE_CATEGORIES",
    # Components
    "RuleEngine",
    "LLMAnalyzer",
    "LLMCache",
    # Integration components
    "AnalysisCache",
    # Exceptions
    "LLMAnalysisError",
    "AnalysisError",
    # Utility functions
    "normalize_type_string",
    "compare_types",
    "extract_base_type",
    "generate_parameter_suggestion",
    "generate_docstring_template",
    "format_code_suggestion",
    "get_parameter_statistics",
    # LLM utilities
    "get_prompt_template",
    "format_prompt",
    "get_available_analysis_types",
    "validate_llm_response",
    # Configuration
    "RuleEngineConfig",
    "AnalysisConfig",
    "get_fast_config",
    "get_thorough_config",
    "get_development_config",
    # Main entry points
    "analyze_matched_pair",
    "analyze_multiple_pairs",
    # Re-exported dependency
    "MatchedPair",
]
