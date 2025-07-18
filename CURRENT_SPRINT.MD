## CURRENT_SPRINT.MD

```markdown
# Current Sprint: Docstring Parser Implementation

## Quick Implementation Reference
- Data Models: `DocstringParameter`, `ParsedDocstring` in `docstring_models.py`
- Main Parser: `DocstringParser` class in `docstring_parser.py`
- Format Detection: `detect_format()` - returns `DocstringFormat` enum
- Integration Point: `IntegratedParser` combines AST + docstring parsing
- Key Validation: Parameter names must match `^[a-zA-Z_][a-zA-Z0-9_]*$` (except *args/**kwargs)

**Sprint Duration**: Days 5-7 (Week 1)
**Previous Sprint**: AST Parser Implementation ✅
**Sprint Goal**: Build a robust docstring parser that handles multiple formats with auto-detection

## Sprint Overview

We need to implement the docstring parsing component that:
1. Auto-detects docstring format (Google, NumPy, Sphinx, REST)
2. Parses docstrings into structured data with validation
3. Handles malformed docstrings gracefully
4. Integrates with the AST parser output
5. Provides a unified interface regardless of format

## Implementation Chunks

### Chunk 1: Data Models and Base Structure
**Estimated Time**: 45 minutes
**Files to Create/Modify**:
- `codedocsync/parser/docstring_models.py` (new)
- `codedocsync/parser/docstring_parser.py` (new)
- `codedocsync/parser/__init__.py` (update)

**Implementation**:
```python
# docstring_models.py - Create these exact models with validation

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import re

class DocstringFormat(Enum):
    """Supported docstring formats."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    REST = "rest"
    UNKNOWN = "unknown"

@dataclass
class DocstringParameter:
    """Single parameter documentation."""
    name: str
    type_str: Optional[str] = None
    description: str = ""
    is_optional: bool = False
    default_value: Optional[str] = None

    def __post_init__(self):
        # Validate parameter name
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.name):
            # Allow *args and **kwargs
            if not re.match(r'^(\*{1,2})?[a-zA-Z_][a-zA-Z0-9_]*$', self.name):
                raise ValueError(f"Invalid parameter name: {self.name}")

@dataclass
class DocstringReturns:
    """Return value documentation."""
    type_str: Optional[str] = None
    description: str = ""

@dataclass
class DocstringRaises:
    """Exception documentation."""
    exception_type: str
    description: str = ""

    def __post_init__(self):
        # Validate exception type is valid identifier
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$', self.exception_type):
            raise ValueError(f"Invalid exception type: {self.exception_type}")

@dataclass
class ParsedDocstring:
    """Complete parsed docstring with all components."""
    format: DocstringFormat
    summary: str
    description: Optional[str] = None
    parameters: List[DocstringParameter] = field(default_factory=list)
    returns: Optional[DocstringReturns] = None
    raises: List[DocstringRaises] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    raw_text: str = ""

    # Additional metadata
    is_valid: bool = True
    parse_errors: List[str] = field(default_factory=list)

    def get_parameter(self, name: str) -> Optional[DocstringParameter]:
        """Get parameter by name."""
        return next((p for p in self.parameters if p.name == name), None)

# docstring_parser.py - Base structure

import logging
from typing import Optional, Dict, Type
from docstring_parser import parse as third_party_parse
from docstring_parser.common import DocstringStyle

from .docstring_models import ParsedDocstring, DocstringFormat

logger = logging.getLogger(__name__)

class DocstringParser:
    """Main docstring parser with format auto-detection."""

    # Mapping of our formats to docstring_parser styles
    FORMAT_MAPPING = {
        DocstringFormat.GOOGLE: DocstringStyle.GOOGLE,
        DocstringFormat.NUMPY: DocstringStyle.NUMPY,
        DocstringFormat.SPHINX: DocstringStyle.SPHINX,
        DocstringFormat.REST: DocstringStyle.REST,
    }

    def parse(self, docstring: Optional[str]) -> Optional[ParsedDocstring]:
        """Parse docstring with auto-detection."""
        if not docstring:
            return None

        # Implementation in next chunk
        pass
Update parser/__init__.py:
python# Add new exports
from .docstring_models import (
    DocstringFormat, DocstringParameter, DocstringReturns,
    DocstringRaises, ParsedDocstring
)
from .docstring_parser import DocstringParser

__all__ = [
    # ... existing exports ...
    'DocstringFormat', 'DocstringParameter', 'DocstringReturns',
    'DocstringRaises', 'ParsedDocstring', 'DocstringParser'
]
Chunk 2: Format Detection Logic
Estimated Time: 60 minutes
Files to Modify:

codedocsync/parser/docstring_parser.py

Implementation Details:
python# Add these methods to DocstringParser class

def detect_format(self, docstring: str) -> DocstringFormat:
    """Auto-detect docstring format using heuristics.

    Detection priority:
    1. Explicit markers (most reliable)
    2. Section patterns
    3. Indentation patterns
    4. Default fallback
    """
    lines = docstring.strip().split('\n')
    if not lines:
        return DocstringFormat.UNKNOWN

    # Check for Sphinx/REST markers
    sphinx_markers = [':param', ':type', ':returns:', ':rtype:', ':raises:']
    if any(marker in docstring for marker in sphinx_markers):
        # Distinguish between Sphinx and REST based on additional patterns
        if ':Example:' in docstring or '.. code-block::' in docstring:
            return DocstringFormat.REST
        return DocstringFormat.SPHINX

    # Check for NumPy style sections with underlines
    numpy_sections = ['Parameters', 'Returns', 'Raises', 'Examples', 'Notes']
    for i, line in enumerate(lines[:-1]):
        if line.strip() in numpy_sections and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and all(c == '-' for c in next_line):
                return DocstringFormat.NUMPY

    # Check for Google style sections
    google_sections = ['Args:', 'Arguments:', 'Returns:', 'Return:', 'Raises:',
                      'Example:', 'Examples:', 'Note:', 'Notes:']
    for line in lines:
        if any(line.strip().startswith(section) for section in google_sections):
            return DocstringFormat.GOOGLE

    # Default to Google style as it's most common
    return DocstringFormat.GOOGLE

def parse(self, docstring: Optional[str]) -> Optional[ParsedDocstring]:
    """Parse docstring with auto-detection and error handling."""
    if not docstring:
        return None

    # Clean docstring
    docstring = docstring.strip()
    if not docstring:
        return None

    # Detect format
    detected_format = self.detect_format(docstring)

    try:
        # Use third-party parser
        if detected_format in self.FORMAT_MAPPING:
            style = self.FORMAT_MAPPING[detected_format]
            parsed = third_party_parse(docstring, style=style)
            return self._convert_to_parsed_docstring(parsed, detected_format, docstring)
        else:
            # Unknown format - best effort with Google style
            logger.warning(f"Unknown docstring format, using Google style as fallback")
            parsed = third_party_parse(docstring, style=DocstringStyle.GOOGLE)
            return self._convert_to_parsed_docstring(parsed, DocstringFormat.UNKNOWN, docstring)

    except Exception as e:
        logger.error(f"Failed to parse docstring: {e}")
        # Return partially parsed docstring with error info
        return self._create_error_docstring(docstring, detected_format, str(e))
Chunk 3: Conversion and Error Handling
Estimated Time: 60 minutes
Files to Modify:

codedocsync/parser/docstring_parser.py

Implementation:
python# Add these methods to DocstringParser class

def _convert_to_parsed_docstring(
    self,
    parsed: Any,  # docstring_parser.Docstring object
    format: DocstringFormat,
    raw_text: str
) -> ParsedDocstring:
    """Convert third-party parsed object to our model.

    Handle differences between formats:
    - Google/NumPy use 'Args' section
    - Sphinx/REST use :param: directives
    - Different type annotation styles
    """
    parameters = []
    for param in parsed.params:
        parameters.append(DocstringParameter(
            name=param.arg_name,
            type_str=param.type_name,
            description=param.description or "",
            is_optional=param.is_optional,
            default_value=param.default
        ))

    returns = None
    if parsed.returns:
        returns = DocstringReturns(
            type_str=parsed.returns.type_name,
            description=parsed.returns.description or ""
        )

    raises = []
    for exc in parsed.raises:
        raises.append(DocstringRaises(
            exception_type=exc.type_name,
            description=exc.description or ""
        ))

    # Extract examples based on format
    examples = self._extract_examples(parsed, format)

    return ParsedDocstring(
        format=format,
        summary=parsed.short_description or "",
        description=parsed.long_description,
        parameters=parameters,
        returns=returns,
        raises=raises,
        examples=examples,
        raw_text=raw_text,
        is_valid=True,
        parse_errors=[]
    )

def _extract_examples(self, parsed: Any, format: DocstringFormat) -> List[str]:
    """Extract code examples from docstring.

    Different formats store examples differently:
    - Google: In 'Example:' or 'Examples:' section
    - NumPy: In 'Examples' section with >>> prefixes
    - Sphinx: In .. code-block:: directives
    """
    examples = []

    # Check if parser extracted examples
    if hasattr(parsed, 'examples') and parsed.examples:
        for example in parsed.examples:
            if hasattr(example, 'snippet'):
                examples.append(example.snippet)
            else:
                examples.append(str(example))

    # For some formats, examples might be in meta
    if hasattr(parsed, 'meta') and parsed.meta:
        for meta in parsed.meta:
            if hasattr(meta, 'description') and meta.description:
                if any(marker in str(meta) for marker in ['Example', 'example', '>>>']):
                    examples.append(meta.description)

    return examples

def _create_error_docstring(
    self,
    raw_text: str,
    format: DocstringFormat,
    error: str
) -> ParsedDocstring:
    """Create ParsedDocstring for parsing failures.

    Best-effort extraction:
    1. Extract summary (first line)
    2. Try to extract obvious sections
    3. Mark as invalid with error details
    """
    lines = raw_text.strip().split('\n')
    summary = lines[0] if lines else ""

    # Best-effort parameter extraction
    parameters = self._extract_parameters_fallback(raw_text, format)

    return ParsedDocstring(
        format=format,
        summary=summary,
        description=None,
        parameters=parameters,
        returns=None,
        raises=[],
        examples=[],
        raw_text=raw_text,
        is_valid=False,
        parse_errors=[f"Parsing failed: {error}"]
    )

def _extract_parameters_fallback(
    self,
    raw_text: str,
    format: DocstringFormat
) -> List[DocstringParameter]:
    """Fallback parameter extraction for failed parsing.

    Use regex patterns specific to each format.
    """
    parameters = []

    if format == DocstringFormat.GOOGLE:
        # Look for Args: section
        import re
        args_match = re.search(r'Args?:\s*\n((?:\s+\w+.*\n?)*)', raw_text, re.MULTILINE)
        if args_match:
            args_text = args_match.group(1)
            # Simple pattern: name (type): description
            param_pattern = r'^\s+(\w+)\s*\(([^)]+)\)?\s*:\s*(.*)$'
            for match in re.finditer(param_pattern, args_text, re.MULTILINE):
                parameters.append(DocstringParameter(
                    name=match.group(1),
                    type_str=match.group(2) if match.group(2) else None,
                    description=match.group(3)
                ))

    elif format == DocstringFormat.SPHINX:
        # Look for :param name: patterns
        param_pattern = r':param\s+(\w+):\s*(.*)$'
        type_pattern = r':type\s+(\w+):\s*(.*)$'

        # First pass: collect parameters
        param_dict = {}
        for match in re.finditer(param_pattern, raw_text, re.MULTILINE):
            param_dict[match.group(1)] = {
                'description': match.group(2)
            }

        # Second pass: add types
        for match in re.finditer(type_pattern, raw_text, re.MULTILINE):
            name = match.group(1)
            if name in param_dict:
                param_dict[name]['type'] = match.group(2)

        # Convert to parameters
        for name, info in param_dict.items():
            parameters.append(DocstringParameter(
                name=name,
                type_str=info.get('type'),
                description=info.get('description', '')
            ))

    return parameters
Chunk 4: Integration with AST Parser
Estimated Time: 45 minutes
Files to Create/Modify:

codedocsync/parser/integrated_parser.py (new)
codedocsync/parser/__init__.py (update)

Implementation:
python# integrated_parser.py - Combines AST and docstring parsing

import logging
from typing import List, Optional
from pathlib import Path

from .ast_parser import parse_python_file, ParsedFunction
from .docstring_parser import DocstringParser
from .docstring_models import ParsedDocstring

logger = logging.getLogger(__name__)

class IntegratedParser:
    """Combines AST and docstring parsing for complete function analysis."""

    def __init__(self):
        self.docstring_parser = DocstringParser()
        self._cache = {}  # Simple cache for parsed docstrings

    def parse_file(self, file_path: str) -> List[ParsedFunction]:
        """Parse file and enrich with parsed docstrings.

        Steps:
        1. Parse AST to get functions
        2. Parse each function's docstring
        3. Attach parsed docstring to function
        4. Return enriched functions
        """
        # Get functions from AST parser
        functions = parse_python_file(file_path)

        # Enrich with parsed docstrings
        for func in functions:
            if func.docstring:
                # Check cache first
                cache_key = hash(func.docstring.raw_text)
                if cache_key in self._cache:
                    func.docstring = self._cache[cache_key]
                else:
                    # Parse docstring
                    parsed = self.docstring_parser.parse(func.docstring.raw_text)
                    if parsed:
                        func.docstring = parsed
                        self._cache[cache_key] = parsed
                    else:
                        # Keep raw docstring if parsing fails
                        logger.warning(
                            f"Failed to parse docstring for {func.signature.name} "
                            f"in {file_path}:{func.line_number}"
                        )

        return functions

    def parse_file_lazy(self, file_path: str):
        """Generator version for memory efficiency."""
        # Get lazy parser from AST module
        from .ast_parser import parse_python_file_lazy

        for func in parse_python_file_lazy(file_path):
            if func.docstring:
                parsed = self.docstring_parser.parse(func.docstring.raw_text)
                if parsed:
                    func.docstring = parsed
            yield func

# Update ParsedFunction to handle both raw and parsed docstrings
# In ast_parser.py, modify ParsedFunction dataclass:
# docstring: Optional[Union[RawDocstring, ParsedDocstring]] = None
Chunk 5: Comprehensive Testing
Estimated Time: 90 minutes
Files to Create:

tests/test_docstring_parser.py
tests/fixtures/docstring_examples.py

Test Implementation:
python# test_docstring_parser.py

import pytest
from codedocsync.parser.docstring_parser import DocstringParser
from codedocsync.parser.docstring_models import (
    DocstringFormat, DocstringParameter, DocstringReturns, DocstringRaises
)

class TestDocstringParser:
    """Comprehensive tests for docstring parsing."""

    @pytest.fixture
    def parser(self):
        return DocstringParser()

    def test_format_detection(self, parser):
        """Test format auto-detection."""
        test_cases = [
            # Google style
            ('''"""
            Summary here.

            Args:
                param1 (str): Description
                param2: Another param

            Returns:
                bool: Success flag
            """''', DocstringFormat.GOOGLE),

            # NumPy style
            ('''"""
            Summary here.

            Parameters
            ----------
            param1 : str
                Description

            Returns
            -------
            bool
                Success flag
            """''', DocstringFormat.NUMPY),

            # Sphinx style
            ('''"""
            Summary here.

            :param param1: Description
            :type param1: str
            :returns: Success flag
            :rtype: bool
            """''', DocstringFormat.SPHINX),
        ]

        for docstring, expected_format in test_cases:
            detected = parser.detect_format(docstring)
            assert detected == expected_format

    def test_google_style_parsing(self, parser):
        """Test Google-style docstring parsing."""
        docstring = '''"""
        Calculate the sum of two numbers.

        This function adds two numbers and returns the result.
        It supports both integers and floats.

        Args:
            a (int | float): First number
            b (int | float): Second number
            round_result (bool, optional): Whether to round. Defaults to False.

        Returns:
            float: The sum of a and b

        Raises:
            TypeError: If inputs are not numeric
            ValueError: If result would overflow

        Example:
            >>> add(2, 3)
            5
            >>> add(2.5, 3.7)
            6.2
        """'''

        parsed = parser.parse(docstring)

        assert parsed is not None
        assert parsed.format == DocstringFormat.GOOGLE
        assert parsed.summary == "Calculate the sum of two numbers."
        assert "supports both integers and floats" in parsed.description

        # Check parameters
        assert len(parsed.parameters) == 3
        param_a = parsed.get_parameter('a')
        assert param_a.type_str == "int | float"
        assert param_a.description == "First number"

        param_round = parsed.get_parameter('round_result')
        assert param_round.is_optional
        assert "Defaults to False" in param_round.description

        # Check returns
        assert parsed.returns.type_str == "float"
        assert parsed.returns.description == "The sum of a and b"

        # Check raises
        assert len(parsed.raises) == 2
        assert any(r.exception_type == "TypeError" for r in parsed.raises)

        # Check examples
        assert len(parsed.examples) >= 1
        assert "add(2, 3)" in str(parsed.examples)

    def test_malformed_docstring_handling(self, parser):
        """Test handling of malformed docstrings."""
        malformed = '''"""
        This docstring has issues

        Args:
            param1 (: Missing type
            param2 No colon here
            param3 (int):  # No description

        Returns
            Missing colon
        """'''

        parsed = parser.parse(malformed)

        assert parsed is not None
        assert not parsed.is_valid
        assert len(parsed.parse_errors) > 0
        assert parsed.summary == "This docstring has issues"
        # Should still extract what it can
        assert len(parsed.parameters) >= 1

    def test_edge_cases(self, parser):
        """Test edge cases and special scenarios."""
        test_cases = [
            # Empty docstring
            ("", None),

            # Only whitespace
            ("   \n  \n   ", None),

            # Single line
            ("Just a summary.", lambda p: p.summary == "Just a summary."),

            # Unicode content
            ("""Unicode test 中文 émojis 🎉""", lambda p: "🎉" in p.summary),

            # Very long summary
            ("x" * 500, lambda p: len(p.summary) == 500),
        ]

        for docstring, check in test_cases:
            parsed = parser.parse(docstring)
            if check is None:
                assert parsed is None
            else:
                assert parsed is not None
                assert check(parsed)

    def test_parameter_variations(self, parser):
        """Test various parameter documentation styles."""
        docstring = '''"""
        Test various parameter styles.

        Args:
            simple: No type annotation
            typed (str): With type
            *args: Variable positional
            **kwargs: Variable keyword
            complex (List[Dict[str, Any]]): Complex type
            optional (int, optional): Optional parameter
        """'''

        parsed = parser.parse(docstring)

        assert len(parsed.parameters) == 6

        # Check special parameters handled correctly
        args_param = parsed.get_parameter('*args')
        assert args_param is not None

        kwargs_param = parsed.get_parameter('**kwargs')
        assert kwargs_param is not None
Chunk 6: Performance Optimization and Final Integration
Estimated Time: 45 minutes
Files to Modify:

codedocsync/parser/docstring_parser.py
codedocsync/parser/__init__.py

Optimizations:
python# Add to docstring_parser.py

import functools
from concurrent.futures import ThreadPoolExecutor
import hashlib

class DocstringParser:
    """Enhanced with caching and performance optimizations."""

    def __init__(self, cache_size: int = 500):
        self._format_cache = {}  # Cache format detection
        self._parse_cache = {}   # Cache full parse results
        self._cache_size = cache_size

    @functools.lru_cache(maxsize=1000)
    def detect_format(self, docstring: str) -> DocstringFormat:
        """Cached format detection."""
        # Implementation from Chunk 2
        # ... existing code ...

    def parse(self, docstring: Optional[str]) -> Optional[ParsedDocstring]:
        """Parse with caching."""
        if not docstring:
            return None

        # Generate cache key
        cache_key = hashlib.md5(docstring.encode()).hexdigest()

        # Check cache
        if cache_key in self._parse_cache:
            return self._parse_cache[cache_key]

        # Parse (existing implementation)
        result = self._parse_uncached(docstring)

        # Cache result
        if len(self._parse_cache) >= self._cache_size:
            # Simple FIFO eviction
            oldest = next(iter(self._parse_cache))
            del self._parse_cache[oldest]

        self._parse_cache[cache_key] = result
        return result

    def parse_batch(self, docstrings: List[Optional[str]]) -> List[Optional[ParsedDocstring]]:
        """Parse multiple docstrings efficiently."""
        results = []

        # Use thread pool for CPU-bound parsing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.parse, ds) for ds in docstrings]
            for future in futures:
                results.append(future.result())

        return results

# Update __init__.py with final exports
__all__ = [
    # AST parser exports
    'parse_python_file', 'parse_python_file_lazy',
    'ParsedFunction', 'FunctionSignature', 'FunctionParameter',

    # Docstring parser exports
    'DocstringParser', 'IntegratedParser',
    'DocstringFormat', 'ParsedDocstring', 'DocstringParameter',
    'DocstringReturns', 'DocstringRaises',

    # Errors
    'ParsingError', 'ValidationError'
]
Sprint Completion Checklist

 All data models created with validation
 Format detection implemented and tested
 Conversion from third-party parser working
 Error handling and fallback parsing
 Integration with AST parser
 Comprehensive test suite (>90% coverage)
 Performance optimization (caching, batch processing)
 Updated CHANGELOG.md with all changes
 All code passes type checking and linting

Notes for AI Implementation

Import Order: Always import standard library first, then third-party, then local
Error Messages: Include file path and line numbers where applicable
Validation: Fail fast at model creation, not during processing
Testing: Each chunk should have corresponding tests
Documentation: Update module docstrings as you implement

Common Pitfalls to Avoid

Don't assume docstring format - always detect
Handle *args and **kwargs as valid parameter names
Preserve original text for error reporting
Don't fail completely on malformed sections
Cache keys should use content hash, not object identity
