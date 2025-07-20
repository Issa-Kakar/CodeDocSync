"""Tests for cross-file documentation matching functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

from codedocsync.matcher.contextual_matcher import ContextualMatcher
from codedocsync.matcher.doc_location_finder import DocLocationFinder
from codedocsync.parser import (
    FunctionSignature,
    ParsedDocstring,
    ParsedFunction,
    RawDocstring,
)


class TestDocLocationFinder:
    """Test DocLocationFinder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.doc_finder = DocLocationFinder()

    def test_find_module_docs_with_function_documentation(self):
        """Test finding function documentation in module docstring."""
        # Create a temporary file with module-level function documentation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""
Module with function documentation.

This module contains several utility functions.

Functions:
----------
calculate_total: Calculate the total of a list of numbers.
    Args:
        numbers: List of numbers to sum.
    Returns:
        The sum of all numbers.

format_output: Format output string with given parameters.
    Args:
        template: Template string.
        **kwargs: Template parameters.
    Returns:
        Formatted string.
"""

def some_function():
    pass
'''
            )
            f.flush()

            try:
                docs = self.doc_finder.find_module_docs(f.name)

                # Should find function documentation
                assert "calculate_total" in docs
                assert "format_output" in docs

                # Check that documentation is properly parsed
                calc_doc = docs["calculate_total"]
                assert isinstance(calc_doc, ParsedDocstring)
                assert (
                    "numbers" in calc_doc.summary or "numbers" in calc_doc.description
                )

            finally:
                os.unlink(f.name)

    def test_find_module_docs_with_class_methods(self):
        """Test finding method documentation in class docstring."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""Module with class method documentation."""

class Calculator:
    """
    A calculator class.

    Methods:
    --------
    add: Add two numbers together.
        Args:
            a: First number.
            b: Second number.
        Returns:
            Sum of a and b.

    multiply: Multiply two numbers.
        Args:
            a: First number.
            b: Second number.
        Returns:
            Product of a and b.
    """

    def add(self, a, b):
        pass

    def multiply(self, a, b):
        pass
'''
            )
            f.flush()

            try:
                docs = self.doc_finder.find_module_docs(f.name)

                # Should find method documentation with class prefix
                assert "Calculator.add" in docs
                assert "Calculator.multiply" in docs

                # Check that documentation is properly parsed
                add_doc = docs["Calculator.add"]
                assert isinstance(add_doc, ParsedDocstring)

            finally:
                os.unlink(f.name)

    def test_find_module_docs_caching(self):
        """Test that module documentation is cached."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""Simple module."""

def test_func():
    pass
'''
            )
            f.flush()

            try:
                # First call should parse the file
                docs1 = self.doc_finder.find_module_docs(f.name)

                # Second call should return cached result
                docs2 = self.doc_finder.find_module_docs(f.name)

                assert docs1 is docs2  # Should be the same object

                # Check cache stats
                stats = self.doc_finder.get_cache_stats()
                assert stats["cached_files"] == 1

            finally:
                os.unlink(f.name)

    def test_find_module_docs_encoding_fallback(self):
        """Test encoding fallback for non-UTF-8 files."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            # Write content with Latin-1 encoding
            content = '''"""Module with special characters: café"""

def test_func():
    pass
'''.encode(
                "latin-1"
            )
            f.write(content)
            f.flush()

            try:
                docs = self.doc_finder.find_module_docs(f.name)

                # Should still work with encoding fallback
                assert isinstance(docs, dict)

            finally:
                os.unlink(f.name)

    def test_find_module_docs_syntax_error(self):
        """Test handling of syntax errors in modules."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""Module with syntax error."""

def test_func():
    invalid syntax here
'''
            )
            f.flush()

            try:
                docs = self.doc_finder.find_module_docs(f.name)

                # Should return empty dict for syntax errors
                assert docs == {}

            finally:
                os.unlink(f.name)

    def test_find_package_docs(self):
        """Test finding documentation in package __init__.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create package structure
            package_dir = Path(tmpdir) / "testpackage"
            package_dir.mkdir()

            init_file = package_dir / "__init__.py"
            init_file.write_text(
                '''"""
Package documentation.

Functions:
----------
package_func: A function documented in the package.
    Args:
        param: Function parameter.
    Returns:
        Function result.
"""

def package_func(param):
    pass
'''
            )

            docs = self.doc_finder.find_package_docs(str(package_dir))

            assert "package_func" in docs
            assert isinstance(docs["package_func"], ParsedDocstring)

    def test_find_package_docs_no_init(self):
        """Test finding package docs when __init__.py doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = self.doc_finder.find_package_docs(tmpdir)
            assert docs == {}

    def test_find_related_docs(self):
        """Test finding documentation in related files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            base_dir = Path(tmpdir)

            # Create parent __init__.py with documentation
            parent_init = base_dir / "__init__.py"
            parent_init.write_text(
                '''"""
Parent package documentation.

Functions:
----------
related_func: A function documented in parent package.
    Args:
        param: Function parameter.
    Returns:
        Function result.
"""
'''
            )

            # Create a module file
            module_file = base_dir / "module.py"
            module_file.write_text(
                """def related_func(param):
    pass
"""
            )

            related_docs = self.doc_finder.find_related_docs(
                "related_func", str(module_file)
            )

            assert len(related_docs) == 1
            doc_file, docstring = related_docs[0]
            assert str(parent_init) == doc_file
            assert isinstance(docstring, ParsedDocstring)

    def test_clear_cache(self):
        """Test clearing the documentation cache."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""Test module."""

def test_func():
    pass
'''
            )
            f.flush()

            try:
                # Add something to cache
                self.doc_finder.find_module_docs(f.name)

                # Verify cache has content
                stats = self.doc_finder.get_cache_stats()
                assert stats["cached_files"] > 0

                # Clear cache
                self.doc_finder.clear_cache()

                # Verify cache is empty
                stats = self.doc_finder.get_cache_stats()
                assert stats["cached_files"] == 0
                assert stats["total_docs"] == 0

            finally:
                os.unlink(f.name)

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""
Test module.

Functions:
----------
func1: First function.
    Returns:
        Result 1.

func2: Second function.
    Returns:
        Result 2.
"""

def func1():
    pass

def func2():
    pass
'''
            )
            f.flush()

            try:
                # Add to cache
                docs = self.doc_finder.find_module_docs(f.name)

                # Check stats
                stats = self.doc_finder.get_cache_stats()
                assert stats["cached_files"] == 1
                assert stats["total_docs"] == len(docs)

            finally:
                os.unlink(f.name)


class TestContextualMatcherCrossFile:
    """Test ContextualMatcher cross-file matching functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.project_root = tmpdir
            self.matcher = ContextualMatcher(tmpdir)

    def test_match_cross_file_docs_with_module_documentation(self):
        """Test matching function with module-level documentation."""
        # Create a function without docstring
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_function", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def test_function():\n    pass",
        )

        # Mock the doc finder to return documentation
        mock_doc_finder = Mock()
        mock_parsed_doc = ParsedDocstring(
            format="google",
            summary="Test function documentation",
            description="Detailed description of test function",
            parameters=[],
            returns=None,
            raises=[],
            examples=[],
            raw_text="Test function documentation",
        )
        mock_doc_finder.find_module_docs.return_value = {
            "test_function": mock_parsed_doc
        }

        # Mock the module resolver
        self.matcher.module_resolver = Mock()
        self.matcher.module_resolver.resolve_module_path.return_value = "test.module"

        # Set the mock doc finder
        self.matcher.doc_finder = mock_doc_finder

        # Test cross-file documentation matching
        result = self.matcher._match_cross_file_docs(function)

        assert result is not None
        assert result.function == function
        assert result.match_type.value == "contextual"
        assert "module docstring" in result.match_reason

    def test_match_cross_file_docs_with_package_documentation(self):
        """Test matching function with package-level documentation."""
        # Create a function without docstring
        function = ParsedFunction(
            signature=FunctionSignature(
                name="package_function", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/test/package/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def package_function():\n    pass",
        )

        # Mock the doc finder
        mock_doc_finder = Mock()
        mock_parsed_doc = ParsedDocstring(
            format="google",
            summary="Package function documentation",
            description="Function documented in package init",
            parameters=[],
            returns=None,
            raises=[],
            examples=[],
            raw_text="Package function documentation",
        )
        mock_doc_finder.find_module_docs.return_value = {}  # No module docs
        mock_doc_finder.find_package_docs.return_value = {
            "package_function": mock_parsed_doc
        }

        # Mock the module resolver
        self.matcher.module_resolver = Mock()
        self.matcher.module_resolver.resolve_module_path.return_value = (
            "test.package.module"
        )
        self.matcher.module_resolver.find_module_file.return_value = (
            "/test/package/__init__.py"
        )

        # Set the mock doc finder
        self.matcher.doc_finder = mock_doc_finder

        # Test cross-file documentation matching
        result = self.matcher._match_cross_file_docs(function)

        assert result is not None
        assert result.function == function
        assert "package __init__.py" in result.match_reason

    def test_match_cross_file_docs_with_related_documentation(self):
        """Test matching function with related documentation files."""
        # Create a function without docstring
        function = ParsedFunction(
            signature=FunctionSignature(
                name="related_function", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def related_function():\n    pass",
        )

        # Mock the doc finder
        mock_doc_finder = Mock()
        mock_parsed_doc = ParsedDocstring(
            format="google",
            summary="Related function documentation",
            description="Function documented in related file",
            parameters=[],
            returns=None,
            raises=[],
            examples=[],
            raw_text="Related function documentation",
        )
        mock_doc_finder.find_module_docs.return_value = {}
        mock_doc_finder.find_package_docs.return_value = {}
        mock_doc_finder.find_related_docs.return_value = [
            ("/test/docs.md", mock_parsed_doc)
        ]

        # Mock the module resolver
        self.matcher.module_resolver = Mock()
        self.matcher.module_resolver.resolve_module_path.return_value = "test.module"

        # Set the mock doc finder
        self.matcher.doc_finder = mock_doc_finder

        # Test cross-file documentation matching
        result = self.matcher._match_cross_file_docs(function)

        assert result is not None
        assert result.function == function
        assert result.docstring == mock_parsed_doc
        assert "related documentation file" in result.match_reason

    def test_match_cross_file_docs_skips_well_documented_functions(self):
        """Test that cross-file matching skips functions with good documentation."""
        # Create a function with good documentation
        function = ParsedFunction(
            signature=FunctionSignature(
                name="well_documented_function",
                parameters=[],
                return_type=None,
                decorators=[],
            ),
            docstring=ParsedDocstring(
                format="google",
                summary="This function is already well documented with a long summary",
                description="It has detailed description too",
                parameters=[],
                returns=None,
                raises=[],
                examples=[],
                raw_text="This function is already well documented",
            ),
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def well_documented_function():\n    pass",
        )

        # Test cross-file documentation matching
        result = self.matcher._match_cross_file_docs(function)

        # Should return None because function already has good documentation
        assert result is None

    def test_match_cross_file_docs_with_raw_docstring(self):
        """Test cross-file matching with function having raw docstring."""
        # Create a function with substantial raw docstring
        function = ParsedFunction(
            signature=FunctionSignature(
                name="raw_documented_function",
                parameters=[],
                return_type=None,
                decorators=[],
            ),
            docstring=RawDocstring(
                raw_text="This function has a substantial raw docstring already",
                line_number=11,
            ),
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def raw_documented_function():\n    pass",
        )

        # Test cross-file documentation matching
        result = self.matcher._match_cross_file_docs(function)

        # Should return None because function already has good documentation
        assert result is None

    def test_match_cross_file_docs_no_documentation_found(self):
        """Test cross-file matching when no documentation is found."""
        # Create a function without docstring
        function = ParsedFunction(
            signature=FunctionSignature(
                name="undocumented_function",
                parameters=[],
                return_type=None,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def undocumented_function():\n    pass",
        )

        # Mock the doc finder to return no documentation
        mock_doc_finder = Mock()
        mock_doc_finder.find_module_docs.return_value = {}
        mock_doc_finder.find_package_docs.return_value = {}
        mock_doc_finder.find_related_docs.return_value = []

        # Mock the module resolver
        self.matcher.module_resolver = Mock()
        self.matcher.module_resolver.resolve_module_path.return_value = "test.module"

        # Set the mock doc finder
        self.matcher.doc_finder = mock_doc_finder

        # Test cross-file documentation matching
        result = self.matcher._match_cross_file_docs(function)

        assert result is None

    def test_assess_doc_quality_perfect_match(self):
        """Test documentation quality assessment with perfect match."""
        # Create function with parameters and return type
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func",
                parameters=[Mock(name="param1"), Mock(name="param2")],
                return_type="str",
                decorators=[],
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def test_func(param1, param2) -> str:\n    pass",
        )

        # Create documentation with matching parameters and return
        docstring = ParsedDocstring(
            format="google",
            summary="Test function",
            description="Test function description",
            parameters=[Mock(name="param1"), Mock(name="param2")],
            returns=Mock(),
            raises=[],
            examples=[],
            raw_text="Test function",
        )

        quality = self.matcher._assess_doc_quality(docstring, function)

        # Should be perfect match
        assert quality == 1.0

    def test_assess_doc_quality_missing_parameters(self):
        """Test documentation quality assessment with missing parameters."""
        # Create function with parameters
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func",
                parameters=[
                    Mock(name="param1"),
                    Mock(name="param2"),
                    Mock(name="param3"),
                ],
                return_type=None,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def test_func(param1, param2, param3):\n    pass",
        )

        # Create documentation missing one parameter
        docstring = ParsedDocstring(
            format="google",
            summary="Test function",
            description="Test function description",
            parameters=[Mock(name="param1"), Mock(name="param2")],
            returns=None,
            raises=[],
            examples=[],
            raw_text="Test function",
        )

        quality = self.matcher._assess_doc_quality(docstring, function)

        # Should be penalized for missing parameter
        assert quality == 0.9  # 1.0 - 0.1 * 1 missing parameter

    def test_assess_doc_quality_extra_parameters(self):
        """Test documentation quality assessment with extra parameters."""
        # Create function with parameters
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func",
                parameters=[Mock(name="param1")],
                return_type=None,
                decorators=[],
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def test_func(param1):\n    pass",
        )

        # Create documentation with extra parameters
        docstring = ParsedDocstring(
            format="google",
            summary="Test function",
            description="Test function description",
            parameters=[Mock(name="param1"), Mock(name="param2"), Mock(name="param3")],
            returns=None,
            raises=[],
            examples=[],
            raw_text="Test function",
        )

        quality = self.matcher._assess_doc_quality(docstring, function)

        # Should be penalized for extra parameters
        assert quality == 0.9  # 1.0 - 0.05 * 2 extra parameters

    def test_assess_doc_quality_missing_return_documentation(self):
        """Test documentation quality assessment with missing return documentation."""
        # Create function with return type
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type="str", decorators=[]
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def test_func() -> str:\n    pass",
        )

        # Create documentation without return documentation
        docstring = ParsedDocstring(
            format="google",
            summary="Test function",
            description="Test function description",
            parameters=[],
            returns=None,
            raises=[],
            examples=[],
            raw_text="Test function",
        )

        quality = self.matcher._assess_doc_quality(docstring, function)

        # Should be penalized for missing return documentation
        assert quality == 0.9  # 1.0 - 0.1 for missing return

    def test_create_cross_file_match(self):
        """Test creating cross-file match with proper confidence calculation."""
        # Create function and documentation
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/test/module.py",
            line_number=10,
            end_line_number=12,
            source_code="def test_func():\n    pass",
        )

        docstring = ParsedDocstring(
            format="google",
            summary="Test function",
            description="Test function description",
            parameters=[],
            returns=None,
            raises=[],
            examples=[],
            raw_text="Test function",
        )

        # Create cross-file match
        match = self.matcher._create_cross_file_match(
            function, docstring, "/test/docs.py", "test location"
        )

        assert match.function == function
        assert match.match_type.value == "contextual"
        assert match.confidence.name_similarity == 1.0
        assert match.confidence.location_score == 0.5
        assert "test location" in match.match_reason
        assert "/test/docs.py" in match.match_reason
