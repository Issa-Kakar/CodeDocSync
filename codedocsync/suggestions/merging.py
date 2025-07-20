"""
Smart merging algorithm for partial docstring updates.

This module provides intelligent merging capabilities that preserve existing
docstring content while applying targeted fixes to specific sections.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .models import DocstringStyle
from ..parser.docstring_models import DocstringParameter


class SectionType(Enum):
    """Types of docstring sections."""

    SUMMARY = "summary"
    DESCRIPTION = "description"
    PARAMETERS = "parameters"
    RETURNS = "returns"
    RAISES = "raises"
    EXAMPLES = "examples"
    NOTES = "notes"
    SEE_ALSO = "see_also"
    CUSTOM = "custom"


@dataclass
class SectionBoundary:
    """Represents the boundaries of a docstring section."""

    section_type: SectionType
    start_line: int
    end_line: int
    header_line: Optional[int] = None
    content_lines: List[int] = None

    def __post_init__(self):
        """Initialize content lines if not provided."""
        if self.content_lines is None:
            self.content_lines = list(range(self.start_line, self.end_line + 1))


class DocstringMerger:
    """Merge new suggestions with existing docstrings."""

    def __init__(self, style: DocstringStyle = DocstringStyle.GOOGLE):
        """Initialize merger with docstring style."""
        self.style = style
        self._section_patterns = self._get_section_patterns()

    def merge_partial_update(
        self,
        existing_docstring: str,
        new_section: str,
        section_type: str,
        preserve_formatting: bool = True,
    ) -> str:
        """Replace just one section of docstring."""
        if not existing_docstring.strip():
            return new_section

        existing_lines = existing_docstring.split("\n")
        new_section_lines = new_section.split("\n")

        # Parse existing docstring structure
        boundaries = self._parse_section_boundaries(existing_lines)

        # Find target section
        target_section = (
            SectionType(section_type)
            if section_type in [e.value for e in SectionType]
            else SectionType.CUSTOM
        )

        # Perform merge
        if target_section in boundaries:
            # Replace existing section
            merged_lines = self._replace_section(
                existing_lines, boundaries[target_section], new_section_lines
            )
        else:
            # Insert new section
            merged_lines = self._insert_new_section(
                existing_lines, new_section_lines, target_section
            )

        return "\n".join(merged_lines)

    def merge_multiple_sections(
        self,
        existing_docstring: str,
        section_updates: Dict[str, str],
        preserve_descriptions: bool = True,
    ) -> str:
        """Merge multiple section updates into existing docstring."""
        current_docstring = existing_docstring

        # Apply updates in order of priority
        section_priority = [
            SectionType.SUMMARY,
            SectionType.DESCRIPTION,
            SectionType.PARAMETERS,
            SectionType.RETURNS,
            SectionType.RAISES,
            SectionType.EXAMPLES,
        ]

        for section_type in section_priority:
            section_key = section_type.value
            if section_key in section_updates:
                current_docstring = self.merge_partial_update(
                    current_docstring,
                    section_updates[section_key],
                    section_key,
                    preserve_formatting=preserve_descriptions,
                )

        # Handle any remaining custom sections
        for section_key, content in section_updates.items():
            if section_key not in [s.value for s in section_priority]:
                current_docstring = self.merge_partial_update(
                    current_docstring, content, section_key
                )

        return current_docstring

    def preserve_custom_content(
        self,
        original_docstring: str,
        updated_docstring: str,
        preserve_sections: Set[str] = None,
    ) -> str:
        """Preserve custom sections and content from original docstring."""
        if preserve_sections is None:
            preserve_sections = {"notes", "see_also", "todo", "warning", "deprecated"}

        original_lines = original_docstring.split("\n")
        updated_lines = updated_docstring.split("\n")

        # Parse both docstrings
        original_boundaries = self._parse_section_boundaries(original_lines)

        # Extract custom sections from original
        custom_sections = {}
        for section_type, boundary in original_boundaries.items():
            if (
                section_type == SectionType.CUSTOM
                or section_type.value in preserve_sections
            ):
                section_lines = original_lines[
                    boundary.start_line : boundary.end_line + 1
                ]
                custom_sections[section_type] = section_lines

        # Insert custom sections into updated docstring
        if custom_sections:
            result_lines = updated_lines[:]

            # Find insertion point (before closing quotes)
            insert_point = len(result_lines) - 1
            for i in range(len(result_lines) - 1, -1, -1):
                if result_lines[i].strip().endswith('"""'):
                    insert_point = i
                    break

            # Insert custom sections
            for section_type, section_lines in custom_sections.items():
                if result_lines[insert_point - 1].strip():
                    result_lines.insert(insert_point, "")
                    insert_point += 1

                for line in section_lines:
                    result_lines.insert(insert_point, line)
                    insert_point += 1

            return "\n".join(result_lines)

        return updated_docstring

    def smart_parameter_merge(
        self,
        original_params: List[DocstringParameter],
        new_params: List[DocstringParameter],
        preserve_descriptions: bool = True,
    ) -> List[DocstringParameter]:
        """Intelligently merge parameter lists."""
        if not preserve_descriptions:
            return new_params

        # Create mapping of original parameter descriptions
        original_map = {param.name: param for param in original_params}

        # Merge parameters
        merged_params = []
        for new_param in new_params:
            if new_param.name in original_map and preserve_descriptions:
                original_param = original_map[new_param.name]

                # Preserve description if new one is generic or empty
                description = new_param.description
                if (
                    not description
                    or self._is_generic_description(description)
                    and original_param.description
                ):
                    description = original_param.description

                # Create merged parameter
                merged_param = DocstringParameter(
                    name=new_param.name,
                    type_str=new_param.type_str,  # Use new type info
                    description=description,
                    is_optional=new_param.is_optional,
                    default_value=new_param.default_value,
                )
                merged_params.append(merged_param)
            else:
                merged_params.append(new_param)

        return merged_params

    def _parse_section_boundaries(
        self, lines: List[str]
    ) -> Dict[SectionType, SectionBoundary]:
        """Parse docstring lines to identify section boundaries."""
        boundaries = {}
        current_section = None
        section_start = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Skip docstring delimiters
            if line_stripped.startswith('"""') or line_stripped.endswith('"""'):
                continue

            # Check for section headers
            section_type = self._identify_section_type(line_stripped)

            if section_type:
                # Save previous section
                if current_section and section_start is not None:
                    boundaries[current_section] = SectionBoundary(
                        section_type=current_section,
                        start_line=section_start,
                        end_line=i - 1,
                    )

                # Start new section
                current_section = section_type
                section_start = i
            elif i == 0 and line_stripped and not section_type:
                # First line is likely summary
                current_section = SectionType.SUMMARY
                section_start = i

        # Save last section
        if current_section and section_start is not None:
            boundaries[current_section] = SectionBoundary(
                section_type=current_section,
                start_line=section_start,
                end_line=len(lines) - 1,
            )

        return boundaries

    def _identify_section_type(self, line: str) -> Optional[SectionType]:
        """Identify section type from line content."""
        line_lower = line.lower().strip()

        # Check against known patterns for current style
        for section_type, patterns in self._section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line_lower):
                    return section_type

        # Check for custom sections (lines ending with colon)
        if line.strip().endswith(":") and len(line.strip()) > 1:
            return SectionType.CUSTOM

        return None

    def _get_section_patterns(self) -> Dict[SectionType, List[str]]:
        """Get section patterns for current style."""
        if self.style == DocstringStyle.GOOGLE:
            return {
                SectionType.PARAMETERS: [
                    r"^args?:$",
                    r"^arguments?:$",
                    r"^parameters?:$",
                ],
                SectionType.RETURNS: [r"^returns?:$", r"^return:$"],
                SectionType.RAISES: [r"^raises?:$", r"^raise:$", r"^exceptions?:$"],
                SectionType.EXAMPLES: [r"^examples?:$", r"^example:$"],
                SectionType.NOTES: [r"^notes?:$", r"^note:$"],
                SectionType.SEE_ALSO: [r"^see also:$"],
            }
        elif self.style == DocstringStyle.NUMPY:
            return {
                SectionType.PARAMETERS: [r"^parameters$", r"^params$"],
                SectionType.RETURNS: [r"^returns$", r"^return$"],
                SectionType.RAISES: [r"^raises$", r"^exceptions$"],
                SectionType.EXAMPLES: [r"^examples$", r"^example$"],
                SectionType.NOTES: [r"^notes$", r"^note$"],
                SectionType.SEE_ALSO: [r"^see also$"],
            }
        elif self.style == DocstringStyle.SPHINX:
            return {
                SectionType.PARAMETERS: [r"^:param.*:$", r"^:parameters?:$"],
                SectionType.RETURNS: [r"^:returns?:$", r"^:return:$"],
                SectionType.RAISES: [r"^:raises?:$", r"^:raise:$"],
                SectionType.EXAMPLES: [r"^:examples?:$"],
            }
        else:
            # Default to Google style patterns
            return self._get_section_patterns()

    def _replace_section(
        self,
        existing_lines: List[str],
        boundary: SectionBoundary,
        new_section_lines: List[str],
    ) -> List[str]:
        """Replace existing section with new content."""
        result = existing_lines[: boundary.start_line]
        result.extend(new_section_lines)
        result.extend(existing_lines[boundary.end_line + 1 :])
        return result

    def _insert_new_section(
        self,
        existing_lines: List[str],
        new_section_lines: List[str],
        section_type: SectionType,
    ) -> List[str]:
        """Insert new section at appropriate location."""
        # Find insertion point based on section order
        insertion_order = [
            SectionType.SUMMARY,
            SectionType.DESCRIPTION,
            SectionType.PARAMETERS,
            SectionType.RETURNS,
            SectionType.RAISES,
            SectionType.EXAMPLES,
            SectionType.NOTES,
            SectionType.SEE_ALSO,
        ]

        boundaries = self._parse_section_boundaries(existing_lines)

        # Find best insertion point
        insert_point = len(existing_lines)

        # Find position in insertion order
        try:
            target_index = insertion_order.index(section_type)
        except ValueError:
            # Custom section, insert before closing
            target_index = len(insertion_order)

        # Find last section that comes before target
        best_end_line = -1
        for existing_section, boundary in boundaries.items():
            try:
                existing_index = insertion_order.index(existing_section)
                if existing_index < target_index and boundary.end_line > best_end_line:
                    best_end_line = boundary.end_line
            except ValueError:
                # Custom section
                continue

        if best_end_line >= 0:
            insert_point = best_end_line + 1
        else:
            # Insert after summary/description
            summary_boundary = boundaries.get(SectionType.SUMMARY)
            desc_boundary = boundaries.get(SectionType.DESCRIPTION)

            if desc_boundary:
                insert_point = desc_boundary.end_line + 1
            elif summary_boundary:
                insert_point = summary_boundary.end_line + 1
            else:
                # Find first non-empty line after opening quotes
                for i, line in enumerate(existing_lines):
                    if line.strip() and not line.strip().startswith('"""'):
                        insert_point = i + 1
                        break

        # Insert with proper spacing
        result = existing_lines[:insert_point]

        # Add blank line if needed
        if result and result[-1].strip():
            result.append("")

        result.extend(new_section_lines)
        result.extend(existing_lines[insert_point:])

        return result

    def _is_generic_description(self, description: str) -> bool:
        """Check if description appears to be generic/auto-generated."""
        generic_patterns = [
            r"^description for \w+",
            r"^parameter \w+",
            r"^the \w+ parameter",
            r"^\w+ parameter$",
            r"^todo:? ",
            r"^fixme:? ",
        ]

        desc_lower = description.lower().strip()

        for pattern in generic_patterns:
            if re.match(pattern, desc_lower):
                return True

        return len(desc_lower) < 10  # Very short descriptions are likely generic

    def extract_section_content(
        self, docstring: str, section_type: str
    ) -> Optional[str]:
        """Extract content of specific section from docstring."""
        lines = docstring.split("\n")
        boundaries = self._parse_section_boundaries(lines)

        target_section = (
            SectionType(section_type)
            if section_type in [e.value for e in SectionType]
            else SectionType.CUSTOM
        )

        if target_section in boundaries:
            boundary = boundaries[target_section]
            section_lines = lines[boundary.start_line : boundary.end_line + 1]
            return "\n".join(section_lines)

        return None

    def merge_with_confidence_weighting(
        self,
        original_content: str,
        new_content: str,
        confidence: float,
        threshold: float = 0.8,
    ) -> str:
        """Merge content based on confidence level."""
        if confidence >= threshold:
            return new_content
        elif confidence >= 0.5:
            # Partial merge - preserve descriptions but update structure
            return self.merge_partial_update(
                original_content,
                new_content,
                SectionType.PARAMETERS.value,
                preserve_formatting=True,
            )
        else:
            # Low confidence - minimal changes
            return original_content

    def validate_merge_result(self, merged_docstring: str) -> Tuple[bool, List[str]]:
        """Validate that merge result is syntactically correct."""
        errors = []

        # Check for balanced quotes
        quote_count = merged_docstring.count('"""')
        if quote_count % 2 != 0:
            errors.append("Unbalanced docstring quotes")

        # Check for empty sections
        lines = merged_docstring.split("\n")
        boundaries = self._parse_section_boundaries(lines)

        for section_type, boundary in boundaries.items():
            section_lines = lines[boundary.start_line : boundary.end_line + 1]
            content_lines = [
                line
                for line in section_lines
                if line.strip() and not line.strip().endswith(":")
            ]

            if not content_lines:
                errors.append(f"Empty {section_type.value} section")

        # Check for proper indentation
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('"""'):
                # Check if line is properly indented (at least 4 spaces for content)
                if not line.startswith("    ") and line.strip() != line:
                    errors.append(f"Improper indentation at line {i + 1}")

        return len(errors) == 0, errors
