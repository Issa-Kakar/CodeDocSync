"""
Test suite for the suggestions module.

This package contains comprehensive tests for all components of the
suggestion generation system implemented in Chunk 1.
"""

# Test module information
__version__ = "1.0.0"
__description__ = "Test suite for CodeDocSync suggestion generation system"

# Test coverage targets
COVERAGE_TARGETS = {
    "models": 95,  # Data models should have very high coverage
    "config": 90,  # Configuration system coverage
    "style_detector": 85,  # Style detection coverage
    "base": 80,  # Base generator coverage
    "overall": 85,  # Overall module coverage target
}

# Test categories
TEST_CATEGORIES = {
    "unit": ["test_models", "test_config", "test_style_detector", "test_base",],
    "integration": [
        # Integration tests will be added in future chunks
    ],
    "performance": [
        # Performance tests will be added in future chunks
    ],
}


def get_test_info():
    """Get information about the test suite."""
    return {
        "version": __version__,
        "description": __description__,
        "coverage_targets": COVERAGE_TARGETS,
        "test_categories": TEST_CATEGORIES,
        "chunk_status": "Chunk 1: Foundation tests complete",
    }
