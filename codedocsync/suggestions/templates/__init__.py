"""
Template system for docstring generation.

This module provides template-based docstring generation for different styles
including Google, NumPy, Sphinx, and reStructuredText formats.
"""

# Import DocstringStyle for convenience
from ..models import DocstringStyle

from .base import (
    DocstringTemplate,
    TemplateRegistry,
    template_registry,
    get_template,
    TemplateMerger,
)
from .google_template import GoogleStyleTemplate

# Register Google template
template_registry.register(DocstringStyle.GOOGLE, GoogleStyleTemplate)

__all__ = [
    "DocstringTemplate",
    "TemplateRegistry",
    "template_registry",
    "get_template",
    "TemplateMerger",
    "GoogleStyleTemplate",
    "DocstringStyle",
]
