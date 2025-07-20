"""
Docstring style converter for transforming between different formats.

This module provides functionality to convert parsed docstrings between
different styles (Google, NumPy, Sphinx, REST) while preserving all
information and applying style-specific formatting rules.
"""

from typing import Dict, List, Optional, Any
import logging

from .models import DocstringStyle
from .templates.base import get_template
from .type_formatter import TypeAnnotationFormatter
from ..parser.docstring_models import (
    ParsedDocstring,
    DocstringParameter,
    DocstringReturns,
    DocstringRaises,
)

logger = logging.getLogger(__name__)


class DocstringStyleConverter:
    """Convert between different docstring styles."""

    def __init__(self):
        """Initialize the style converter."""
        self._type_formatters: Dict[DocstringStyle, TypeAnnotationFormatter] = {}
        self._conversion_stats = {
            "conversions_performed": 0,
            "information_preserved": 0.0,
            "common_issues": [],
        }

    def convert(
        self,
        docstring: ParsedDocstring,
        from_style: DocstringStyle,
        to_style: DocstringStyle,
        preserve_formatting: bool = True,
        max_line_length: int = 88,
    ) -> str:
        """
        Convert parsed docstring to different style.

        Args:
            docstring: Parsed docstring to convert
            from_style: Original docstring style
            to_style: Target docstring style
            preserve_formatting: Whether to preserve custom formatting
            max_line_length: Maximum line length for the target style

        Returns:
            Converted docstring as a formatted string

        Raises:
            ValueError: If conversion between styles is not supported
        """
        if from_style == to_style:
            # No conversion needed, but reformat with template
            return self._reformat_same_style(docstring, to_style, max_line_length)

        logger.info(f"Converting docstring from {from_style.value} to {to_style.value}")

        try:
            # Get type formatter for target style
            type_formatter = self._get_type_formatter(to_style)

            # Convert type annotations to target style
            converted_params = self._convert_parameters(
                docstring.parameters, type_formatter
            )
            converted_returns = self._convert_returns(docstring.returns, type_formatter)
            converted_raises = self._convert_raises(docstring.raises, type_formatter)

            # Get template for target style
            template = get_template(to_style, max_line_length=max_line_length)

            # Generate converted docstring
            converted = template.render_complete_docstring(
                summary=docstring.summary or "",
                description=docstring.description,
                parameters=converted_params,
                returns=converted_returns,
                raises=converted_raises,
                examples=self._convert_examples(docstring, from_style, to_style),
            )

            self._conversion_stats["conversions_performed"] += 1
            logger.debug(f"Successfully converted docstring to {to_style.value}")

            return converted

        except Exception as e:
            logger.error(f"Failed to convert docstring: {e}")
            raise ValueError(
                f"Conversion from {from_style.value} to {to_style.value} failed: {e}"
            )

    def convert_batch(
        self,
        docstrings: List[ParsedDocstring],
        from_style: DocstringStyle,
        to_style: DocstringStyle,
        **kwargs,
    ) -> List[str]:
        """
        Convert multiple docstrings in batch.

        Args:
            docstrings: List of parsed docstrings to convert
            from_style: Original docstring style
            to_style: Target docstring style
            **kwargs: Additional arguments passed to convert()

        Returns:
            List of converted docstring strings
        """
        results = []
        errors = []

        for i, docstring in enumerate(docstrings):
            try:
                converted = self.convert(docstring, from_style, to_style, **kwargs)
                results.append(converted)
            except Exception as e:
                logger.warning(f"Failed to convert docstring {i}: {e}")
                errors.append((i, str(e)))
                results.append(None)  # Placeholder for failed conversion

        if errors:
            logger.warning(
                f"Failed to convert {len(errors)} out of {len(docstrings)} docstrings"
            )

        return results

    def estimate_conversion_quality(
        self,
        docstring: ParsedDocstring,
        from_style: DocstringStyle,
        to_style: DocstringStyle,
    ) -> Dict[str, Any]:
        """
        Estimate the quality of conversion between styles.

        Args:
            docstring: Docstring to analyze
            from_style: Source style
            to_style: Target style

        Returns:
            Dictionary with quality metrics and warnings
        """
        quality = {
            "information_loss_risk": "low",
            "formatting_changes": [],
            "unsupported_features": [],
            "confidence": 0.95,
            "warnings": [],
        }

        # Check for style-specific features that might not convert well
        if from_style == DocstringStyle.NUMPY and to_style == DocstringStyle.GOOGLE:
            if any("See Also" in str(section) for section in [docstring.description]):
                quality["unsupported_features"].append("See Also sections")
                quality["confidence"] -= 0.1

        if from_style == DocstringStyle.SPHINX and to_style in {
            DocstringStyle.GOOGLE,
            DocstringStyle.NUMPY,
        }:
            quality["formatting_changes"].append("Cross-references will be simplified")
            quality["confidence"] -= 0.05

        # Check for complex type annotations
        if docstring.parameters:
            complex_types = [
                p
                for p in docstring.parameters
                if p.type_annotation
                and any(
                    pattern in p.type_annotation
                    for pattern in ["Callable", "Protocol", "TypeVar"]
                )
            ]
            if complex_types:
                quality["warnings"].append(
                    f"Complex types found in {len(complex_types)} parameters"
                )
                quality["confidence"] -= 0.1

        return quality

    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get statistics about conversions performed."""
        return self._conversion_stats.copy()

    def _reformat_same_style(
        self, docstring: ParsedDocstring, style: DocstringStyle, max_line_length: int
    ) -> str:
        """Reformat docstring in the same style with consistent formatting."""
        template = get_template(style, max_line_length=max_line_length)

        return template.render_complete_docstring(
            summary=docstring.summary or "",
            description=docstring.description,
            parameters=docstring.parameters,
            returns=docstring.returns,
            raises=docstring.raises,
        )

    def _get_type_formatter(self, style: DocstringStyle) -> TypeAnnotationFormatter:
        """Get or create type formatter for style."""
        if style not in self._type_formatters:
            self._type_formatters[style] = TypeAnnotationFormatter(style)
        return self._type_formatters[style]

    def _convert_parameters(
        self,
        parameters: Optional[List[DocstringParameter]],
        type_formatter: TypeAnnotationFormatter,
    ) -> Optional[List[DocstringParameter]]:
        """Convert parameter type annotations to target style."""
        if not parameters:
            return None

        converted = []
        for param in parameters:
            converted_param = DocstringParameter(
                name=param.name,
                type_annotation=(
                    type_formatter.format_for_docstring(param.type_annotation)
                    if param.type_annotation
                    else None
                ),
                description=param.description,
                is_optional=param.is_optional,
                default_value=param.default_value,
            )
            converted.append(converted_param)

        return converted

    def _convert_returns(
        self,
        returns: Optional[DocstringReturns],
        type_formatter: TypeAnnotationFormatter,
    ) -> Optional[DocstringReturns]:
        """Convert return type annotation to target style."""
        if not returns:
            return None

        return DocstringReturns(
            type_annotation=(
                type_formatter.format_for_docstring(returns.type_annotation)
                if returns.type_annotation
                else None
            ),
            description=returns.description,
        )

    def _convert_raises(
        self,
        raises: Optional[List[DocstringRaises]],
        type_formatter: TypeAnnotationFormatter,
    ) -> Optional[List[DocstringRaises]]:
        """Convert exception information to target style."""
        if not raises:
            return None

        # Raises sections don't typically need type conversion,
        # but we ensure consistency
        converted = []
        for exception in raises:
            converted_exception = DocstringRaises(
                exception_type=exception.exception_type,
                description=exception.description,
            )
            converted.append(converted_exception)

        return converted

    def _convert_examples(
        self,
        docstring: ParsedDocstring,
        from_style: DocstringStyle,
        to_style: DocstringStyle,
    ) -> Optional[List[str]]:
        """Convert examples to target style format."""
        # Examples conversion is complex and style-dependent
        # For now, preserve examples as-is
        # TODO: Implement style-specific example formatting

        if hasattr(docstring, "examples") and docstring.examples:
            return docstring.examples

        return None

    def _handle_style_specific_sections(
        self,
        docstring: ParsedDocstring,
        from_style: DocstringStyle,
        to_style: DocstringStyle,
    ) -> Dict[str, List[str]]:
        """Handle conversion of style-specific sections."""
        sections = {}

        # NumPy-specific sections
        if from_style == DocstringStyle.NUMPY:
            # Handle "See Also", "Notes", etc.
            if hasattr(docstring, "see_also"):
                if to_style == DocstringStyle.SPHINX:
                    # Convert to Sphinx cross-references
                    sections["see_also"] = self._convert_see_also_to_sphinx(
                        docstring.see_also
                    )
                else:
                    # Convert to simple text
                    sections["see_also"] = [f"See also: {docstring.see_also}"]

        # Sphinx-specific sections
        if from_style == DocstringStyle.SPHINX:
            # Handle cross-references, directives
            if hasattr(docstring, "references"):
                sections["references"] = self._simplify_sphinx_references(
                    docstring.references
                )

        return sections

    def _convert_see_also_to_sphinx(self, see_also: str) -> List[str]:
        """Convert NumPy See Also section to Sphinx format."""
        # Simple conversion - in practice this would be more sophisticated
        return [f".. seealso:: {see_also}"]

    def _simplify_sphinx_references(self, references: str) -> List[str]:
        """Simplify Sphinx cross-references for other styles."""
        # Remove Sphinx-specific markup
        simplified = references.replace(":func:", "").replace(":class:", "")
        simplified = simplified.replace(":mod:", "").replace(":meth:", "")
        return [simplified]


class ConversionPreset:
    """Predefined conversion configurations."""

    @staticmethod
    def scientific_to_api() -> Dict[str, Any]:
        """Convert from scientific (NumPy) to API documentation (Google)."""
        return {
            "preserve_formatting": True,
            "max_line_length": 88,
            "simplify_types": True,
        }

    @staticmethod
    def api_to_sphinx() -> Dict[str, Any]:
        """Convert from API docs (Google) to Sphinx documentation."""
        return {
            "preserve_formatting": True,
            "max_line_length": 100,  # Sphinx often uses longer lines
            "enable_cross_references": True,
        }

    @staticmethod
    def legacy_cleanup() -> Dict[str, Any]:
        """Configuration for cleaning up legacy docstrings."""
        return {
            "preserve_formatting": False,  # Reformat everything
            "max_line_length": 88,
            "strict_validation": True,
        }


def convert_docstring(
    docstring: ParsedDocstring,
    from_style: DocstringStyle,
    to_style: DocstringStyle,
    **kwargs,
) -> str:
    """
    Convenience function to convert a single docstring.

    Args:
        docstring: Parsed docstring to convert
        from_style: Source style
        to_style: Target style
        **kwargs: Additional conversion options

    Returns:
        Converted docstring string
    """
    converter = DocstringStyleConverter()
    return converter.convert(docstring, from_style, to_style, **kwargs)


def batch_convert_docstrings(
    docstrings: List[ParsedDocstring],
    from_style: DocstringStyle,
    to_style: DocstringStyle,
    **kwargs,
) -> List[str]:
    """
    Convenience function to convert multiple docstrings.

    Args:
        docstrings: List of parsed docstrings
        from_style: Source style
        to_style: Target style
        **kwargs: Additional conversion options

    Returns:
        List of converted docstring strings
    """
    converter = DocstringStyleConverter()
    return converter.convert_batch(docstrings, from_style, to_style, **kwargs)
