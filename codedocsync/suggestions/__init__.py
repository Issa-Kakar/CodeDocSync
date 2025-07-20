"""
Suggestion Generator Module for CodeDocSync.

This module provides comprehensive suggestion generation for fixing documentation
inconsistencies detected by the analyzer. It supports multiple docstring styles,
intelligent style detection, and high-quality suggestion formatting.

Core Components:
- Data models for representing suggestions and metadata
- Base generator interface for extensible suggestion types
- Style detection system for automatic docstring style recognition
- Configuration system for customizing suggestion behavior

Key Features:
- Multi-style support (Google, NumPy, Sphinx, reStructuredText)
- Automatic style detection from existing code
- High-confidence suggestion validation
- Actionable, copy-paste ready suggestions
- Comprehensive diff generation
- Quality scoring and ranking
"""

from typing import Any, Dict, Optional

# Base generator interface
from .base import (
    BaseSuggestionGenerator,
    with_suggestion_fallback,
)

# Configuration system
from .config import (
    ConfigManager,
    # Main configuration classes
    SuggestionConfig,
    # Global config manager
    config_manager,
    get_comprehensive_config,
    get_development_config,
    get_documentation_config,
    # Predefined configurations
    get_minimal_config,
)

# Configuration management (Chunk 5)
from .config_manager import (
    IntegratedSuggestionConfig,
    SuggestionConfigManager,
    get_config_manager,
    load_suggestion_config,
)

# Style conversion
from .converter import (
    DocstringStyleConverter,
    batch_convert_docstrings,
    convert_docstring,
)

# Output formatters (Chunk 5)
from .formatters import (
    JSONSuggestionFormatter,
    OutputStyle,
    TerminalSuggestionFormatter,
)

# Integration layer (Chunk 5)
from .integration import (
    EnhancedAnalysisResult,
    EnhancedIssue,
    SuggestionBatchProcessor,
    SuggestionIntegration,
    enhance_multiple_with_suggestions,
    enhance_with_suggestions,
)

# Core data models
from .models import (
    DocstringStyle,
    StyleDetectionError,
    # Main suggestion classes
    Suggestion,
    SuggestionBatch,
    SuggestionContext,
    SuggestionDiff,
    # Exceptions
    SuggestionError,
    SuggestionGenerationError,
    SuggestionMetadata,
    # Enums
    SuggestionType,
    SuggestionValidationError,
)

# Ranking and filtering (Chunk 5)
from .ranking import (
    PriorityBooster,
    RankingConfig,
    RankingMetrics,
    RankingStrategy,
    SuggestionFilter,
    SuggestionRanker,
    create_balanced_ranker,
    create_permissive_ranker,
    create_strict_ranker,
)

# Style detection system
from .style_detector import (
    DocstringStyleDetector,
    style_detector,  # Global detector instance
)

# Type formatting
from .type_formatter import (
    TypeAnnotationFormatter,
    extract_type_from_ast,
    format_type_for_style,
)

# Version information
__version__ = "1.0.0"

# Module metadata
__author__ = "CodeDocSync Team"
__description__ = "Intelligent suggestion generation for documentation inconsistencies"
__license__ = "MIT"

# Public API exports
__all__ = [
    # Core classes
    "Suggestion",
    "SuggestionBatch",
    "SuggestionContext",
    "SuggestionDiff",
    "SuggestionMetadata",
    # Enums
    "SuggestionType",
    "DocstringStyle",
    # Base generator
    "BaseSuggestionGenerator",
    "with_suggestion_fallback",
    # Configuration
    "SuggestionConfig",
    "ConfigManager",
    "config_manager",
    # Predefined configs
    "get_minimal_config",
    "get_comprehensive_config",
    "get_development_config",
    "get_documentation_config",
    # Style detection
    "DocstringStyleDetector",
    "style_detector",
    # Type formatting
    "TypeAnnotationFormatter",
    "format_type_for_style",
    "extract_type_from_ast",
    # Style conversion
    "DocstringStyleConverter",
    "convert_docstring",
    "batch_convert_docstrings",
    # Integration layer (Chunk 5)
    "SuggestionIntegration",
    "SuggestionBatchProcessor",
    "EnhancedIssue",
    "EnhancedAnalysisResult",
    "enhance_with_suggestions",
    "enhance_multiple_with_suggestions",
    # Output formatters (Chunk 5)
    "TerminalSuggestionFormatter",
    "JSONSuggestionFormatter",
    "OutputStyle",
    # Ranking and filtering (Chunk 5)
    "SuggestionRanker",
    "SuggestionFilter",
    "RankingStrategy",
    "RankingConfig",
    "RankingMetrics",
    "PriorityBooster",
    "create_strict_ranker",
    "create_permissive_ranker",
    "create_balanced_ranker",
    # Configuration management (Chunk 5)
    "SuggestionConfigManager",
    "IntegratedSuggestionConfig",
    "get_config_manager",
    "load_suggestion_config",
    # Exceptions
    "SuggestionError",
    "StyleDetectionError",
    "SuggestionGenerationError",
    "SuggestionValidationError",
    # Module metadata
    "__version__",
    "__author__",
    "__description__",
    "__license__",
]


def create_suggestion_context(
    issue, function, docstring=None, project_style="google", **kwargs
) -> SuggestionContext:
    """
    Factory function to create a SuggestionContext with sensible defaults.

    Args:
        issue: InconsistencyIssue from analyzer
        function: ParsedFunction from parser
        docstring: Optional ParsedDocstring
        project_style: Docstring style to use (default: "google")
        **kwargs: Additional context parameters

    Returns:
        Configured SuggestionContext instance
    """
    return SuggestionContext(
        issue=issue,
        function=function,
        docstring=docstring,
        project_style=project_style,
        **kwargs,
    )


def validate_suggestion_quality(suggestion: Suggestion) -> dict[str, Any]:
    """
    Validate and analyze the quality of a suggestion.

    Args:
        suggestion: Suggestion to validate

    Returns:
        Dictionary with quality metrics and validation results
    """
    from .base import BaseSuggestionGenerator

    # Create a temporary generator for validation
    generator = type(
        "TempGenerator",
        (BaseSuggestionGenerator,),
        {
            "generate": lambda self, ctx: suggestion,
            "__init__": lambda self: super().__init__(),
        },
    )()

    is_valid = generator.validate_suggestion(suggestion)

    return {
        "is_valid": is_valid,
        "quality_score": suggestion.get_quality_score(),
        "confidence": suggestion.confidence,
        "is_actionable": suggestion.is_actionable,
        "validation_passed": suggestion.validation_passed,
        "copy_paste_ready": suggestion.copy_paste_ready,
        "is_ready_to_apply": suggestion.is_ready_to_apply,
        "is_high_confidence": suggestion.is_high_confidence,
        "affected_sections": suggestion.affected_sections,
        "metadata": (
            suggestion.metadata.to_dict()
            if hasattr(suggestion.metadata, "to_dict")
            else vars(suggestion.metadata)
        ),
    }


def detect_project_style(
    project_path: str, config: SuggestionConfig | None = None
) -> str:
    """
    Detect the predominant docstring style in a project.

    Args:
        project_path: Path to the project directory
        config: Optional configuration for detection

    Returns:
        Detected style name ("google", "numpy", "sphinx", or "rest")
    """
    detector = DocstringStyleDetector(config)
    return detector.detect_from_project(project_path)


def create_diff_preview(
    original: str, suggested: str, filename: str = "docstring"
) -> str:
    """
    Create a unified diff preview between original and suggested text.

    Args:
        original: Original text
        suggested: Suggested text
        filename: Filename to use in diff headers

    Returns:
        Unified diff string
    """
    original_lines = original.splitlines(keepends=True)
    suggested_lines = suggested.splitlines(keepends=True)

    diff = SuggestionDiff(
        original_lines=original_lines,
        suggested_lines=suggested_lines,
        start_line=1,
        end_line=max(len(original_lines), len(suggested_lines)),
    )

    return diff.to_unified_diff(filename)


# Integration helpers for working with the analyzer module
def enhance_analysis_result_with_suggestions(
    analysis_result, config: SuggestionConfig | None = None
):
    """
    Enhance an AnalysisResult with suggestions.

    This function integrates with the analyzer module and adds rich suggestions
    to analysis results using the SuggestionIntegration system.

    Args:
        analysis_result: AnalysisResult from analyzer module
        config: Optional suggestion configuration

    Returns:
        EnhancedAnalysisResult with suggestions
    """
    # Use the integration layer to enhance the result
    return enhance_with_suggestions(analysis_result, config)


# Quality constants for suggestion evaluation
QUALITY_THRESHOLDS = {
    "HIGH_QUALITY": 0.8,  # High-quality suggestions ready for auto-apply
    "MEDIUM_QUALITY": 0.6,  # Good suggestions needing review
    "LOW_QUALITY": 0.4,  # Suggestions needing significant review
    "MIN_ACCEPTABLE": 0.3,  # Minimum threshold for showing suggestions
}

CONFIDENCE_LEVELS = {
    "VERY_HIGH": 0.9,  # Extremely confident in suggestion
    "HIGH": 0.8,  # High confidence
    "MEDIUM": 0.6,  # Medium confidence
    "LOW": 0.4,  # Low confidence, needs review
    "VERY_LOW": 0.2,  # Very low confidence, mostly informational
}

# Style priorities for auto-detection fallback
STYLE_PRIORITY_ORDER = ["google", "numpy", "sphinx", "rest"]

# Common suggestion templates (for use in future generators)
COMMON_TEMPLATES = {
    "parameter_missing": "Add missing parameter '{param_name}' to docstring",
    "parameter_type_mismatch": "Update parameter '{param_name}' type from '{old_type}' to '{new_type}'",
    "return_missing": "Add return documentation for {return_type}",
    "raises_missing": "Document {exception_type} exception that can be raised",
    "description_outdated": "Update function description to reflect current behavior",
}


def get_module_info() -> dict[str, Any]:
    """
    Get information about the suggestions module.

    Returns:
        Dictionary with module metadata and capabilities
    """
    return {
        "name": "codedocsync.suggestions",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "capabilities": {
            "supported_styles": ["google", "numpy", "sphinx", "rest"],
            "suggestion_types": [t.value for t in SuggestionType],
            "auto_style_detection": True,
            "quality_scoring": True,
            "diff_generation": True,
            "validation": True,
            "caching": True,
        },
        "chunk_status": {
            "chunk_1": "✅ Complete - Core Data Models and Base Infrastructure",
            "chunk_2": "✅ Complete - Template-Based Suggestion Engine",
            "chunk_3": "✅ Complete - Style-Specific Formatters",
            "chunk_4": "✅ Complete - Issue-Specific Strategies",
            "chunk_5": "✅ Complete - Integration Layer and Output Formatting",
            "chunk_6": "⏳ Pending - Testing Suite and Production Integration",
        },
        "integration_ready": True,  # Chunk 5 completed
    }


# Module initialization message
def _print_initialization_info():
    """Print module initialization information (for development)."""
    import os

    if os.getenv("CODEDOCSYNC_DEBUG"):
        print(f"CodeDocSync Suggestions Module {__version__} initialized")
        print("✅ Chunk 1: Core Data Models and Base Infrastructure - Complete")
        print("✅ Chunk 2: Template-Based Suggestion Engine - Complete")
        print("✅ Chunk 3: Style-Specific Formatters - Complete")
        print("✅ Chunk 4: Issue-Specific Strategies - Complete")
        print("✅ Chunk 5: Integration Layer and Output Formatting - Complete")
        print("🎉 Suggestion system ready for integration!")


# Initialize module (only in debug mode)
_print_initialization_info()
