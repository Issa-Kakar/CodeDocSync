"""
Template system for docstring generation.

This module provides template-based docstring generation for different styles
including Google, NumPy, Sphinx, and reStructuredText formats.
"""

# Import DocstringStyle for convenience
from ..models import DocstringStyle
from .base import (
    DocstringTemplate,
    TemplateMerger,
    TemplateRegistry,
    get_template,
    template_registry,
)
from .google_template import GoogleStyleTemplate
from .numpy_template import NumpyStyleTemplate
from .sphinx_template import SphinxStyleTemplate

# Register all templates
template_registry.register(DocstringStyle.GOOGLE, GoogleStyleTemplate)
template_registry.register(DocstringStyle.NUMPY, NumpyStyleTemplate)
template_registry.register(DocstringStyle.SPHINX, SphinxStyleTemplate)

__all__ = [
    "DocstringTemplate",
    "TemplateRegistry",
    "template_registry",
    "get_template",
    "TemplateMerger",
    "GoogleStyleTemplate",
    "NumpyStyleTemplate",
    "SphinxStyleTemplate",
    "DocstringStyle",
]
