"""
Analyzer module for detecting inconsistencies between functions and documentation.

This module provides rule-based and LLM-powered analysis to identify when
function signatures don't match their documentation. It's designed to run fast
deterministic checks first, then fall back to LLM analysis only when needed.
"""

from typing import TYPE_CHECKING

# Import configuration classes
from .config import (
    AnalysisConfig,
    RuleEngineConfig,
    get_development_config,
    get_fast_config,
    get_thorough_config,
)
from .llm_analyzer import (
    LLMAnalyzer,
    TokenBucket,
    create_balanced_analyzer,
    create_fast_analyzer,
    create_thorough_analyzer,
)

# Import LLM components (Chunk 1 foundation)
from .llm_config import LLMConfig
from .llm_models import VALID_ANALYSIS_TYPES, LLMAnalysisRequest, LLMAnalysisResponse

# Import all models and constants
from .models import (
    CONFIDENCE_THRESHOLDS,
    ISSUE_TYPES,
    RULE_CATEGORIES,
    SEVERITY_WEIGHTS,
    AnalysisResult,
    InconsistencyIssue,
    RuleCheckResult,
)

# Import implemented components
from .rule_engine import RuleEngine

# Import utility functions
from .rule_engine_utils import (
    compare_types,
    extract_base_type,
    format_code_suggestion,
    generate_docstring_template,
    generate_parameter_suggestion,
    get_parameter_statistics,
    normalize_type_string,
)

# TYPE_CHECKING imports for future components
if TYPE_CHECKING:
    pass

# Import matcher models we depend on
from codedocsync.matcher import MatchedPair

# Import main integration functions
from .integration import (
    AnalysisCache,
    AnalysisError,
    analyze_matched_pair,
    analyze_multiple_pairs,
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
