"""
Test fixtures and utilities for suggestion module testing.

Provides reusable test data, mock objects, and utility functions
for comprehensive testing of the suggestion generation system.
"""

from inspect import Parameter as ParameterKind

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.parser.docstring_models import (
    DocstringRaises,
    DocstringReturns,
    ParsedDocstring,
)


def create_test_function(
    name: str = "test_func",
    params: list[str] | None = None,
    return_type: str | None = None,
    docstring: str | None = None,
    line_number: int = 10,
    file_path: str = "test.py",
) -> ParsedFunction:
    """Create test function with specified attributes."""
    if params is None:
        params = ["param1", "param2"]

    # Create function parameters
    parameters = []
    for i, param_name in enumerate(params):
        parameters.append(
            FunctionParameter(
                name=param_name,
                type_annotation="str" if i == 0 else "int",
                default_value=None,
                is_required=True,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
            )
        )

    # Create function signature
    signature = FunctionSignature(
        name=name,
        parameters=parameters,
        return_type=return_type,
        is_method=False,
        is_classmethod=False,
        is_staticmethod=False,
        is_async=False,
        is_generator=False,
        decorators=[],
    )

    # Create raw docstring if provided
    raw_docstring = RawDocstring(docstring, line_number + 1) if docstring else None

    return ParsedFunction(
        signature=signature,
        docstring=raw_docstring,
        file_path=file_path,
        line_number=line_number,
        source_code=f"def {name}({', '.join(params)}): pass",
    )


def create_parsed_docstring(
    summary: str = "Test function.",
    params: dict[str, str] | None = None,
    returns: str | None = None,
    raises: dict[str, str] | None = None,
    examples: list[str] | None = None,
    format_style: str = "google",
) -> ParsedDocstring:
    """Create a parsed docstring with specified components."""
    if params is None:
        params = {}

    # Create parameter documentation
    doc_params = []
    for name, desc in params.items():
        doc_params.append(
            DocstringParam(
                arg_name=name,
                description=desc,
                type_name="str",
                is_optional=False,
                default=None,
            )
        )

    # Create return documentation
    doc_returns = None
    if returns:
        doc_returns = DocstringReturns(
            description=returns,
            type_name="bool",
            is_generator=False,
            return_name=None,
        )

    # Create raises documentation
    doc_raises = []
    if raises:
        for exc_type, desc in raises.items():
            doc_raises.append(DocstringRaises(type_name=exc_type, description=desc))

    # Build raw text
    raw_text = summary
    if params:
        raw_text += "\n\nArgs:"
        for name, desc in params.items():
            raw_text += f"\n    {name}: {desc}"
    if returns:
        raw_text += f"\n\nReturns:\n    {returns}"
    if raises:
        raw_text += "\n\nRaises:"
        for exc_type, desc in raises.items():
            raw_text += f"\n    {exc_type}: {desc}"

    return ParsedDocstring(
        summary=summary,
        description="",
        args=doc_params,
        returns=doc_returns,
        yields=None,
        receives=None,
        raises=doc_raises,
        warns=[],
        see_also=[],
        notes=[],
        references=[],
        examples=examples or [],
        attributes=[],
        raw_text=raw_text,
    )


def create_test_issue(
    issue_type: str = "parameter_name_mismatch",
    severity: str = "critical",
    description: str | None = None,
    line_number: int = 10,
    confidence: float = 0.95,
) -> InconsistencyIssue:
    """Create test inconsistency issue."""
    if description is None:
        description = f"Test issue of type {issue_type}"

    return InconsistencyIssue(
        issue_type=issue_type,
        severity=severity,
        description=description,
        suggestion=f"Fix {issue_type}",
        line_number=line_number,
        confidence=confidence,
        details={},
    )


# Docstring examples for different styles
DOCSTRING_EXAMPLES = {
    "google_simple": '''"""
    Simple function description.

    Args:
        param1: First parameter
        param2: Second parameter

    Returns:
        Something useful
    """''',
    "google_complex": '''"""
    Complex function with many features.

    This function demonstrates multiple docstring sections
    and advanced formatting.

    Args:
        data (List[Dict[str, Any]]): Input data structure
            with nested types and long description that
            spans multiple lines.
        config (Optional[Config]): Configuration object.
            Defaults to None.
        *args: Variable positional arguments
        **kwargs: Variable keyword arguments

    Returns:
        Tuple[bool, str]: Success status and message

    Raises:
        ValueError: If data is invalid
        TypeError: If config is wrong type
        RuntimeError: On processing error

    Examples:
        Basic usage:

        >>> result = complex_func([{"a": 1}])
        >>> print(result)
        (True, "Success")

        With configuration:

        >>> config = Config(debug=True)
        >>> result = complex_func(data, config)
    """''',
    "numpy_simple": '''"""
    Simple function description.

    Parameters
    ----------
    param1 : str
        First parameter
    param2 : int
        Second parameter

    Returns
    -------
    bool
        Something useful
    """''',
    "numpy_complex": '''"""
    Complex function with many features.

    Long description here with multiple paragraphs
    explaining the function behavior.

    Parameters
    ----------
    data : np.ndarray
        Input data array with shape (n, m) where n is
        the number of samples and m is the number of
        features.
    axis : int, optional
        Axis along which to operate. Default is 0.
    method : {'mean', 'median', 'mode'}, optional
        Statistical method to use. Default is 'mean'.

    Returns
    -------
    result : np.ndarray
        Processed array with reduced dimensionality
    stats : dict
        Dictionary containing processing statistics

    Raises
    ------
    ValueError
        If axis is out of bounds
    TypeError
        If data is not array-like

    See Also
    --------
    related_func : Related functionality
    another_func : Another related function

    Notes
    -----
    This function uses optimized NumPy operations
    for performance. Memory usage is O(n).

    Examples
    --------
    >>> data = np.random.rand(100, 10)
    >>> result, stats = complex_func(data)
    >>> result.shape
    (100,)
    """''',
    "sphinx_simple": '''"""
    Simple function description.

    :param param1: First parameter
    :type param1: str
    :param param2: Second parameter
    :type param2: int
    :returns: Something useful
    :rtype: bool
    """''',
    "sphinx_complex": '''"""
    Complex function with many features.

    Detailed description with reStructuredText formatting
    and multiple paragraphs.

    :param data: Input data structure with complex types
    :type data: List[Dict[str, Any]]
    :param config: Configuration object, defaults to None
    :type config: Optional[Config]
    :param \\*args: Variable positional arguments
    :param \\**kwargs: Variable keyword arguments
    :returns: Tuple of success status and message
    :rtype: Tuple[bool, str]
    :raises ValueError: If data is invalid
    :raises TypeError: If config is wrong type
    :raises RuntimeError: On processing error

    .. seealso::
       :func:`related_func` -- Related functionality
       :class:`Config` -- Configuration class

    .. note::
       This function is thread-safe.

    .. code-block:: python

       result = complex_func([{"a": 1}], Config())
       print(result)  # (True, "Success")
    """''',
    "malformed": '''"""This is a malformed docstring
    with no proper sections Args: param1 badly formatted
    Returns something maybe
    """''',
}


# Test data for edge cases
EDGE_CASE_FUNCTIONS = {
    "property_getter": '''
@property
def name(self) -> str:
    """Get the name."""
    return self._name
''',
    "property_setter": '''
@name.setter
def name(self, value: str) -> None:
    """Set the name."""
    self._name = value
''',
    "classmethod": '''
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "MyClass":
    """Create instance from dictionary."""
    return cls(**data)
''',
    "staticmethod": '''
@staticmethod
def validate_data(data: Any) -> bool:
    """Validate input data."""
    return isinstance(data, dict)
''',
    "async_function": '''
async def fetch_data(url: str) -> Dict[str, Any]:
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
''',
    "generator": '''
def fibonacci(n: int) -> Generator[int, None, None]:
    """Generate Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
''',
    "magic_method": '''
def __init__(self, name: str, value: int) -> None:
    """Initialize the object."""
    self.name = name
    self.value = value
''',
}


# Performance test data
def generate_large_function(num_params: int = 100) -> ParsedFunction:
    """Generate function with many parameters for performance testing."""
    params = [f"param{i}" for i in range(num_params)]
    return create_test_function(
        name="large_function",
        params=params,
        docstring=f"Function with {num_params} parameters.\n\nArgs:\n"
        + "\n".join(f"    param{i}: Parameter {i}" for i in range(10))
        + "\n    ... and more",
    )


def generate_complex_docstring(num_sections: int = 10) -> str:
    """Generate complex docstring with many sections."""
    sections = ["Summary of the function."]

    if num_sections > 0:
        sections.append("\nArgs:")
        for i in range(min(5, num_sections)):
            sections.append(f"    param{i}: Description of parameter {i}")

    if num_sections > 1:
        sections.append("\nReturns:")
        sections.append("    Complex return value description")

    if num_sections > 2:
        sections.append("\nRaises:")
        for exc in ["ValueError", "TypeError", "RuntimeError"][: num_sections - 2]:
            sections.append(f"    {exc}: When something goes wrong")

    if num_sections > 5:
        sections.append("\nExamples:")
        sections.append("    >>> func()")
        sections.append("    42")

    return "\n".join(sections)
