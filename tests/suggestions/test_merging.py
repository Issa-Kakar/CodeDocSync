"""Tests for smart docstring merging functionality."""

import pytest

from codedocsync.parser.docstring_models import DocstringParameter
from codedocsync.suggestions.merging import (
    DocstringMerger,
    SectionBoundary,
    SectionType,
)
from codedocsync.suggestions.models import DocstringStyle


class TestDocstringMerger:
    """Test docstring merging functionality."""

    @pytest.fixture
    def merger(self):
        """Create DocstringMerger instance."""
        return DocstringMerger(DocstringStyle.GOOGLE)

    def test_merge_partial_update_parameters(self, merger):
        """Test merging parameter section update."""
        existing_docstring = '''"""
        Function summary.

        Args:
            old_param: Old parameter description

        Returns:
            Something useful
        """'''

        new_section = """Args:
    new_param: New parameter description
    another_param: Another parameter"""

        result = merger.merge_partial_update(
            existing_docstring, new_section, "parameters"
        )

        assert "new_param: New parameter description" in result
        assert "another_param: Another parameter" in result
        assert "old_param: Old parameter description" not in result
        assert "Returns:" in result  # Should preserve returns section
        assert "Something useful" in result

    def test_merge_partial_update_new_section(self, merger):
        """Test adding new section that doesn't exist."""
        existing_docstring = '''"""
        Function summary.

        Returns:
            Something useful
        """'''

        new_section = """Args:
    param1: First parameter
    param2: Second parameter"""

        result = merger.merge_partial_update(
            existing_docstring, new_section, "parameters"
        )

        assert "Args:" in result
        assert "param1: First parameter" in result
        assert "param2: Second parameter" in result
        assert "Returns:" in result
        assert "Something useful" in result

    def test_merge_multiple_sections(self, merger):
        """Test merging multiple section updates."""
        existing_docstring = '''"""
        Function summary.

        Args:
            old_param: Old description

        Returns:
            Old return description
        """'''

        section_updates = {
            "parameters": """Args:
    new_param: New parameter
    another_param: Another parameter""",
            "returns": """Returns:
    New return description""",
        }

        result = merger.merge_multiple_sections(existing_docstring, section_updates)

        assert "new_param: New parameter" in result
        assert "another_param: Another parameter" in result
        assert "New return description" in result
        assert "old_param: Old description" not in result
        assert "Old return description" not in result

    def test_preserve_custom_content(self, merger):
        """Test preserving custom sections during merge."""
        original_docstring = '''"""
        Function summary.

        Args:
            param: Parameter description

        Returns:
            Return description

        Note:
            This is a custom note that should be preserved.

        Warning:
            This is a warning that should be preserved.
        """'''

        updated_docstring = '''"""
        Updated function summary.

        Args:
            new_param: New parameter description

        Returns:
            Updated return description
        """'''

        result = merger.preserve_custom_content(original_docstring, updated_docstring)

        assert "Updated function summary" in result
        assert "new_param: New parameter description" in result
        assert "Updated return description" in result
        assert "This is a custom note that should be preserved" in result
        assert "This is a warning that should be preserved" in result

    def test_smart_parameter_merge_preserve_descriptions(self, merger):
        """Test intelligent parameter merging with description preservation."""
        original_params = [
            DocstringParameter(
                name="param1",
                type_str="str",
                description="Detailed description that should be preserved",
            ),
            DocstringParameter(
                name="param2",
                type_str="int",
                description="Another detailed description",
            ),
        ]

        new_params = [
            DocstringParameter(
                name="param1",
                type_str="Optional[str]",  # Type updated
                description="Generic description",  # Generic description
            ),
            DocstringParameter(
                name="param3",  # New parameter
                type_str="bool",
                description="New parameter description",
            ),
        ]

        merged = merger.smart_parameter_merge(
            original_params, new_params, preserve_descriptions=True
        )

        assert len(merged) == 2

        # First parameter should have new type but preserved description
        param1 = next(p for p in merged if p.name == "param1")
        assert param1.type_str == "Optional[str]"
        assert param1.description == "Detailed description that should be preserved"

        # New parameter should have its description
        param3 = next(p for p in merged if p.name == "param3")
        assert param3.description == "New parameter description"

    def test_smart_parameter_merge_no_preservation(self, merger):
        """Test parameter merging without description preservation."""
        original_params = [
            DocstringParameter(name="param1", description="Original description")
        ]

        new_params = [DocstringParameter(name="param1", description="New description")]

        merged = merger.smart_parameter_merge(
            original_params, new_params, preserve_descriptions=False
        )

        assert len(merged) == 1
        assert merged[0].description == "New description"

    def test_parse_section_boundaries_google_style(self, merger):
        """Test parsing section boundaries for Google style."""
        lines = [
            '"""Function summary.',
            "",
            "Detailed description.",
            "",
            "Args:",
            "    param1: First parameter",
            "    param2: Second parameter",
            "",
            "Returns:",
            "    Return value",
            "",
            "Raises:",
            "    ValueError: When something is wrong",
            "",
            "Examples:",
            "    >>> example()",
            "    True",
            '"""',
        ]

        boundaries = merger._parse_section_boundaries(lines)

        assert SectionType.SUMMARY in boundaries
        assert SectionType.PARAMETERS in boundaries
        assert SectionType.RETURNS in boundaries
        assert SectionType.RAISES in boundaries
        assert SectionType.EXAMPLES in boundaries

        # Check parameter section boundaries
        params_boundary = boundaries[SectionType.PARAMETERS]
        assert params_boundary.start_line == 4  # "Args:" line
        assert params_boundary.end_line == 6  # Last parameter line

    def test_identify_section_type_google(self, merger):
        """Test identifying section types for Google style."""
        test_cases = [
            ("Args:", SectionType.PARAMETERS),
            ("Arguments:", SectionType.PARAMETERS),
            ("Parameters:", SectionType.PARAMETERS),
            ("Returns:", SectionType.RETURNS),
            ("Return:", SectionType.RETURNS),
            ("Raises:", SectionType.RAISES),
            ("Exceptions:", SectionType.RAISES),
            ("Examples:", SectionType.EXAMPLES),
            ("Notes:", SectionType.NOTES),
            ("See Also:", SectionType.SEE_ALSO),
            ("Custom Section:", SectionType.CUSTOM),
            ("Regular line", None),
        ]

        for line, expected in test_cases:
            result = merger._identify_section_type(line)
            assert result == expected

    def test_extract_section_content(self, merger):
        """Test extracting specific section content."""
        docstring = '''"""
        Function summary.

        Args:
            param1: First parameter
            param2: Second parameter

        Returns:
            Return value
        """'''

        # Extract parameters section
        params_content = merger.extract_section_content(docstring, "parameters")

        assert "Args:" in params_content
        assert "param1: First parameter" in params_content
        assert "param2: Second parameter" in params_content
        assert "Returns:" not in params_content

    def test_merge_with_confidence_weighting(self, merger):
        """Test confidence-based merging."""
        original = "Original content"
        new_content = "New content"

        # High confidence - use new content
        result = merger.merge_with_confidence_weighting(original, new_content, 0.9)
        assert result == new_content

        # Medium confidence - partial merge
        result = merger.merge_with_confidence_weighting(original, new_content, 0.6)
        # Should attempt partial merge (implementation dependent)

        # Low confidence - keep original
        result = merger.merge_with_confidence_weighting(original, new_content, 0.3)
        assert result == original

    def test_validate_merge_result(self, merger):
        """Test validation of merge results."""
        # Valid docstring
        valid_docstring = '''"""
        Valid docstring.

        Args:
            param: Parameter description

        Returns:
            Return value
        """'''

        is_valid, errors = merger.validate_merge_result(valid_docstring)
        assert is_valid
        assert len(errors) == 0

        # Invalid docstring - unbalanced quotes
        invalid_docstring = '''"""
        Invalid docstring.

        Args:
            param: Parameter description
        """'''

        is_valid, errors = merger.validate_merge_result(invalid_docstring)
        assert not is_valid
        assert "Unbalanced docstring quotes" in errors[0]

    def test_merge_empty_existing_docstring(self, merger):
        """Test merging with empty existing docstring."""
        new_section = """Args:
    param: Parameter description"""

        result = merger.merge_partial_update("", new_section, "parameters")
        assert result == new_section

    def test_merge_preserve_spacing(self, merger):
        """Test that merging preserves proper spacing."""
        existing_docstring = '''"""
        Function summary.

        Args:
            old_param: Old parameter

        Returns:
            Something useful
        """'''

        new_section = """Args:
    new_param: New parameter"""

        result = merger.merge_partial_update(
            existing_docstring, new_section, "parameters"
        )

        # Should maintain blank lines between sections
        lines = result.split("\n")

        # Find Args and Returns sections
        args_line = None
        returns_line = None

        for i, line in enumerate(lines):
            if line.strip() == "Args:":
                args_line = i
            elif line.strip() == "Returns:":
                returns_line = i

        # Should have blank line between Args and Returns sections
        assert args_line is not None
        assert returns_line is not None
        assert returns_line > args_line + 1  # At least one line between


class TestSectionBoundary:
    """Test SectionBoundary functionality."""

    def test_section_boundary_creation(self):
        """Test creating section boundary."""
        boundary = SectionBoundary(
            section_type=SectionType.PARAMETERS, start_line=5, end_line=10
        )

        assert boundary.section_type == SectionType.PARAMETERS
        assert boundary.start_line == 5
        assert boundary.end_line == 10
        assert boundary.content_lines == [5, 6, 7, 8, 9, 10]

    def test_section_boundary_with_content_lines(self):
        """Test creating section boundary with explicit content lines."""
        boundary = SectionBoundary(
            section_type=SectionType.RETURNS,
            start_line=5,
            end_line=10,
            content_lines=[6, 7, 8],
        )

        assert boundary.content_lines == [6, 7, 8]


class TestMergerEdgeCases:
    """Test edge cases for merger."""

    @pytest.fixture
    def merger(self):
        """Create DocstringMerger instance."""
        return DocstringMerger(DocstringStyle.GOOGLE)

    def test_is_generic_description(self, merger):
        """Test detection of generic descriptions."""
        generic_descriptions = [
            "Description for param",
            "Parameter param",
            "The param parameter",
            "param parameter",
            "TODO: add description",
            "Short",  # Very short
        ]

        specific_descriptions = [
            "The username for authentication with the API server",
            "Configuration object containing timeout and retry settings",
            "List of validated data records from the input file",
        ]

        for desc in generic_descriptions:
            assert merger._is_generic_description(desc)

        for desc in specific_descriptions:
            assert not merger._is_generic_description(desc)

    def test_merge_with_malformed_docstring(self, merger):
        """Test merging with malformed docstring."""
        malformed_docstring = '''"""
        Malformed docstring
        Args
            missing_colon: This should still work
        """'''

        new_section = """Args:
    new_param: New parameter"""

        # Should handle gracefully
        result = merger.merge_partial_update(
            malformed_docstring, new_section, "parameters"
        )
        assert "new_param: New parameter" in result

    def test_merge_numpy_style_sections(self):
        """Test merging with NumPy style."""
        numpy_merger = DocstringMerger(DocstringStyle.NUMPY)

        existing_docstring = '''"""
        Function summary.

        Parameters
        ----------
        old_param : str
            Old parameter description

        Returns
        -------
        bool
            Success status
        """'''

        new_section = """Parameters
----------
new_param : int
    New parameter description"""

        result = numpy_merger.merge_partial_update(
            existing_docstring, new_section, "parameters"
        )

        assert "new_param : int" in result
        assert "New parameter description" in result
        assert "Returns" in result  # Should preserve returns section
