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

# Import LLM components (Chunk 1 foundation)
from .llm_config import LLMConfig
from .llm_models import LLMAnalysisRequest, LLMAnalysisResponse, VALID_ANALYSIS_TYPES
from .llm_analyzer import (
    LLMAnalyzer,
    TokenBucket,
    create_fast_analyzer,
    create_balanced_analyzer,
    create_thorough_analyzer,
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
    "LLMAnalysisRequest",
    "LLMAnalysisResponse",
    # Constants
    "ISSUE_TYPES",
    "SEVERITY_WEIGHTS",
    "CONFIDENCE_THRESHOLDS",
    "RULE_CATEGORIES",
    "VALID_ANALYSIS_TYPES",
    # Components
    "RuleEngine",
    "LLMAnalyzer",
    "TokenBucket",
    # Integration components (future chunks)
    "AnalysisCache",
    # Exceptions (future chunks)
    "AnalysisError",
    # Utility functions
    "normalize_type_string",
    "compare_types",
    "extract_base_type",
    "generate_parameter_suggestion",
    "generate_docstring_template",
    "format_code_suggestion",
    "get_parameter_statistics",
    # Configuration
    "RuleEngineConfig",
    "AnalysisConfig",
    "LLMConfig",
    "get_fast_config",
    "get_thorough_config",
    "get_development_config",
    # LLM factory functions
    "create_fast_analyzer",
    "create_balanced_analyzer",
    "create_thorough_analyzer",
    # Main entry points (future chunks)
    "analyze_matched_pair",
    "analyze_multiple_pairs",
    # Re-exported dependency
    "MatchedPair",
]
