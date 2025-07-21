"""Integration tests for docstring parser.

Tests cover format detection, extraction accuracy, error handling, and performance.
Benchmarks:
- Auto-detection accuracy: > 95%
- Parse time per docstring: < 5ms
"""

import time

from codedocsync.parser.docstring_models import DocstringFormat
from codedocsync.parser.docstring_parser import DocstringParser


class TestFormatDetection:
    """Test automatic docstring format detection accuracy."""

    def test_auto_detect_google_style(self) -> None:
        """Test detection of Google-style docstrings."""

        google_docstrings = [
            """Process data and return results.

            Args:
                data (str): The input data to process
                validate (bool): Whether to validate input

            Returns:
                dict: Processing results with keys 'status' and 'output'

            Raises:
                ValueError: If data is invalid
                RuntimeError: If processing fails
            """,
            """Simple function with minimal Google docstring.

            Args:
                x: First value
                y: Second value

            Returns:
                Sum of x and y
            """,
            """Function with examples section.

            Args:
                items (list): List of items to process

            Example:
                >>> process_items([1, 2, 3])
                [2, 4, 6]

            Note:
                This function doubles each item
            """,
        ]

        for docstring in google_docstrings:
            detected = DocstringParser.detect_format(docstring)
            assert (
                detected == DocstringFormat.GOOGLE
            ), f"Failed to detect Google format in: {docstring[:50]}..."

    def test_auto_detect_numpy_style(self) -> None:
        """Test detection of NumPy-style docstrings."""

        numpy_docstrings = [
            """Calculate the weighted average.

            Parameters
            ----------
            values : array_like
                Array of values to average
            weights : array_like, optional
                Weights for each value, defaults to equal weights

            Returns
            -------
            float
                The weighted average

            Raises
            ------
            ValueError
                If values and weights have different shapes
            """,
            """Compute matrix multiplication.

            Parameters
            ----------
            a : ndarray
                First matrix
            b : ndarray
                Second matrix

            Returns
            -------
            ndarray
                Product of a and b

            Examples
            --------
            >>> matmul([[1, 2]], [[3], [4]])
            array([[11]])
            """,
            """Simple NumPy style with notes.

            Parameters
            ----------
            x : int
                Input value

            Notes
            -----
            This is a demonstration function.
            """,
        ]

        for docstring in numpy_docstrings:
            detected = DocstringParser.detect_format(docstring)
            assert (
                detected == DocstringFormat.NUMPY
            ), f"Failed to detect NumPy format in: {docstring[:50]}..."

    def test_auto_detect_sphinx_style(self) -> None:
        """Test detection of Sphinx-style docstrings."""

        sphinx_docstrings = [
            """Execute database query.

            :param query: SQL query string
            :type query: str
            :param params: Query parameters
            :type params: dict
            :returns: Query results
            :rtype: list[dict]
            :raises DatabaseError: If query execution fails
            """,
            """Process configuration file.

            :param path: Path to config file
            :param validate: Whether to validate config
            :type validate: bool
            :returns: Parsed configuration
            :rtype: dict
            """,
            """Simple Sphinx docstring.

            :param name: User name
            :returns: Greeting message
            """,
        ]

        for docstring in sphinx_docstrings:
            detected = DocstringParser.detect_format(docstring)
            assert (
                detected == DocstringFormat.SPHINX
            ), f"Failed to detect Sphinx format in: {docstring[:50]}..."

    def test_auto_detect_rest_style(self) -> None:
        """Test detection of reStructuredText-style docstrings."""

        rest_docstrings = [
            """Parse REST API response.

            :param response: HTTP response object
            :type response: Response
            :returns: Parsed data
            :rtype: dict

            :Example:

            .. code-block:: python

                data = parse_response(resp)
                print(data['status'])
            """,
            """Generate documentation.

            :param source: Source directory
            :param output: Output directory
            :returns: Generated file paths
            :rtype: list[str]

            .. note::
               This uses Sphinx internally

            .. code-block:: python

               generate_docs('src/', 'docs/')
            """,
        ]

        for docstring in rest_docstrings:
            detected = DocstringParser.detect_format(docstring)
            assert (
                detected == DocstringFormat.REST
            ), f"Failed to detect REST format in: {docstring[:50]}..."


class TestExtractionAccuracy:
    """Test accurate extraction of docstring components."""

    def test_extract_all_param_info(self) -> None:
        """Test extraction of parameter name, type, and description."""
        parser = DocstringParser()

        google_doc = """Function with detailed parameter info.

        Args:
            user_id (int): Unique identifier for the user
            settings (dict[str, Any]): User configuration settings
            notify (bool, optional): Whether to send notifications. Defaults to True.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """

        result = parser.parse(google_doc)
        assert result is not None
        assert len(result.parameters) == 5

        # Check first parameter
        user_id_param = result.get_parameter("user_id")
        assert user_id_param is not None
        assert user_id_param.name == "user_id"
        assert user_id_param.type_str == "int"
        assert "Unique identifier" in user_id_param.description

        # Check complex type
        settings_param = result.get_parameter("settings")
        assert settings_param is not None
        assert settings_param.type_str == "dict[str, Any]"

        # Check optional parameter
        notify_param = result.get_parameter("notify")
        assert notify_param is not None
        assert notify_param.is_optional is True
        assert notify_param.default_value == "True"

        # Check *args and **kwargs
        args_param = result.get_parameter("args")
        assert args_param is not None
        kwargs_param = result.get_parameter("kwargs")
        assert kwargs_param is not None

    def test_extract_returns_section(self) -> None:
        """Test extraction of return value information."""
        parser = DocstringParser()

        docstrings_with_returns = [
            # Google style
            """Calculate result.

            Returns:
                tuple[int, str]: A tuple containing (status_code, message)
            """,
            # NumPy style
            """Process data.

            Returns
            -------
            pd.DataFrame
                Processed data with columns 'id', 'value', and 'timestamp'
            """,
            # Sphinx style
            """Get user info.

            :returns: User information dictionary
            :rtype: dict[str, Any]
            """,
        ]

        for docstring in docstrings_with_returns:
            result = parser.parse(docstring)
            assert result is not None
            assert result.returns is not None
            assert result.returns.type_str is not None or result.returns.description

    def test_extract_raises_section(self) -> None:
        """Test extraction of exception information."""
        parser = DocstringParser()

        google_doc = """Validate input data.

        Raises:
            ValueError: If input data is malformed or missing required fields
            TypeError: If input is not a dictionary
            ValidationError: If business rules validation fails
        """

        result = parser.parse(google_doc)
        assert result is not None
        assert len(result.raises) == 3

        # Check exception types
        exception_types = {exc.exception_type for exc in result.raises}
        assert exception_types == {"ValueError", "TypeError", "ValidationError"}

        # Check descriptions
        value_error = next(
            exc for exc in result.raises if exc.exception_type == "ValueError"
        )
        assert "malformed" in value_error.description

    def test_preserve_examples_section(self) -> None:
        """Test preservation of code examples."""
        parser = DocstringParser()

        numpy_doc = """Apply transformation to data.

        Parameters
        ----------
        data : array_like
            Input data
        transform : callable
            Transformation function

        Examples
        --------
        >>> data = [1, 2, 3, 4]
        >>> transform = lambda x: x ** 2
        >>> apply_transform(data, transform)
        [1, 4, 9, 16]

        >>> import numpy as np
        >>> data = np.array([[1, 2], [3, 4]])
        >>> apply_transform(data, np.sqrt)
        array([[1.0, 1.414], [1.732, 2.0]])
        """

        result = parser.parse(numpy_doc)
        assert result is not None
        assert len(result.examples) >= 1
        # Check that example code is preserved
        example_text = " ".join(result.examples)
        assert "lambda x: x ** 2" in example_text or "np.sqrt" in example_text

    def test_handle_malformed_docstrings(self) -> None:
        """Test graceful handling of malformed docstrings."""
        parser = DocstringParser()

        malformed_docs = [
            # Missing colons
            """Malformed Google style.

            Args
                x  First parameter
                y  Second parameter

            Returns
                Some value
            """,
            # Incomplete sections
            """Incomplete docstring.

            Parameters
            ----------
            """,
            # Mixed formats
            """Mixed format docstring.

            Args:
                x: First value

            :param y: Second value
            :returns: Sum of x and y
            """,
        ]

        for docstring in malformed_docs:
            result = parser.parse(docstring)
            assert result is not None  # Should not crash
            assert result.summary  # Should extract at least the summary

    def test_extract_complex_types(self) -> None:
        """Test extraction of complex type annotations."""
        parser = DocstringParser()

        complex_types_doc = """Function with complex types.

        Args:
            callback (Callable[[int, str], None]): Callback function
            data (Union[List[Dict[str, Any]], pd.DataFrame]): Input data
            config (Optional[ConfigType]): Configuration object
            items (Sequence[T]): Generic sequence of items
            mapping (Mapping[str, List[int]]): String to list mapping
        """

        result = parser.parse(complex_types_doc)
        assert result is not None
        assert len(result.parameters) == 5

        # Check complex callable type
        callback = result.get_parameter("callback")
        assert callback is not None
        assert callback.type_str is not None and "Callable" in callback.type_str

        # Check union type
        data = result.get_parameter("data")
        assert data is not None
        assert data.type_str is not None and "Union" in data.type_str

        # Check generic type
        items = result.get_parameter("items")
        assert items is not None
        assert items.type_str is not None and "Sequence" in items.type_str

    def test_optional_parameters_with_defaults(self) -> None:
        """Test handling of optional parameters with default values."""
        parser = DocstringParser()

        doc_with_defaults = """Configure system settings.

        Args:
            host (str): Server hostname
            port (int, optional): Server port. Defaults to 8080.
            timeout (float, optional): Connection timeout in seconds. Defaults to 30.0.
            debug (bool, optional): Enable debug mode. Defaults to False.
            retries (int, optional): Number of retry attempts. Defaults to 3.
            config (dict, optional): Additional configuration. Defaults to None.
        """

        result = parser.parse(doc_with_defaults)
        assert result is not None

        # Check required parameter
        host = result.get_parameter("host")
        assert host is not None
        assert not host.is_optional

        # Check optional parameters with defaults
        optional_params = {
            "port": "8080",
            "timeout": "30.0",
            "debug": "False",
            "retries": "3",
            "config": "None",
        }

        for param_name, expected_default in optional_params.items():
            param = result.get_parameter(param_name)
            assert param is not None
            assert param.is_optional
            assert param.default_value == expected_default


class TestMixedFormats:
    """Test handling of projects with multiple docstring formats."""

    def test_mixed_format_handling(self) -> None:
        """Test parser handles different formats in same session."""
        parser = DocstringParser()

        # Different formats that might appear in same project
        mixed_docs = [
            # Google style
            """Google style function.

            Args:
                x (int): Input value

            Returns:
                int: Processed value
            """,
            # NumPy style
            """NumPy style function.

            Parameters
            ----------
            x : int
                Input value

            Returns
            -------
            int
                Processed value
            """,
            # Sphinx style
            """Sphinx style function.

            :param x: Input value
            :type x: int
            :returns: Processed value
            :rtype: int
            """,
        ]

        results = []
        formats_detected = set()

        for docstring in mixed_docs:
            result = parser.parse(docstring)
            assert result is not None
            results.append(result)
            formats_detected.add(result.format)

            # Verify consistent extraction despite format differences
            assert len(result.parameters) == 1
            param = result.parameters[0]
            assert param.name == "x"
            assert param.type_str == "int"
            assert result.returns is not None

        # Ensure all formats were detected correctly
        assert len(formats_detected) == 3
        assert DocstringFormat.GOOGLE in formats_detected
        assert DocstringFormat.NUMPY in formats_detected
        assert DocstringFormat.SPHINX in formats_detected


class TestPerformance:
    """Test parser performance meets requirements."""

    def test_parse_time_performance(self) -> None:
        """Test single docstring parse time < 5ms."""
        parser = DocstringParser()

        # Typical docstring
        docstring = """Process user data and generate report.

        This function takes user data, validates it, processes it according to
        business rules, and generates a comprehensive report.

        Args:
            user_data (dict): User information including name, email, preferences
            options (ProcessOptions): Processing configuration
            validate (bool, optional): Whether to validate input. Defaults to True.
            format (str, optional): Output format. Defaults to "json".

        Returns:
            Report: Generated report object containing:
                - summary: Executive summary
                - details: Detailed analysis
                - metrics: Performance metrics

        Raises:
            ValueError: If user_data is invalid or missing required fields
            ProcessingError: If processing fails due to business rule violations
            FormatError: If requested format is not supported

        Example:
            >>> data = {"name": "John", "email": "john@example.com"}
            >>> report = process_user_data(data, options)
            >>> print(report.summary)
        """

        # Warm up cache
        parser.parse(docstring)

        # Measure parse time
        start_time = time.perf_counter()
        result = parser.parse(docstring)
        parse_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert result is not None
        assert parse_time < 5.0, f"Parse time {parse_time:.2f}ms exceeds 5ms limit"

    def test_batch_parsing_performance(self) -> None:
        """Test batch parsing performance with thread pool."""
        parser = DocstringParser()

        # Generate 100 varied docstrings
        docstrings: list[str | None] = []
        for i in range(100):
            doc = f"""Function {i} with parameters.

            Args:
                param1 (str): First parameter
                param2 (int): Second parameter
                param3 (bool, optional): Third parameter. Defaults to False.

            Returns:
                dict: Result containing status and data

            Raises:
                ValueError: If parameters are invalid
            """
            docstrings.append(doc)

        # Measure batch parsing time
        start_time = time.perf_counter()
        results = parser.parse_batch(docstrings)
        batch_time = time.perf_counter() - start_time

        assert len(results) == 100
        assert all(r is not None for r in results)

        # Average time per docstring should be much less than 5ms due to parallelism
        avg_time = (batch_time * 1000) / 100
        assert avg_time < 5.0, f"Average parse time {avg_time:.2f}ms exceeds 5ms limit"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_and_special_characters(self) -> None:
        """Test handling of unicode and special characters in docstrings."""
        parser = DocstringParser()

        unicode_doc = """Function with unicode documentation.

        Args:
            text (str): Text that may contain émojis 🚀, spëcial characters, and 中文
            currency (str): Currency symbol like €, £, ¥, or ₹
            equation (str): Mathematical symbols like ∑, ∏, ∫, or ∞

        Returns:
            str: Processed text with preserved special characters

        Example:
            >>> process("Hello 世界! 🌍")
            "Processed: Hello 世界! 🌍"
        """

        result = parser.parse(unicode_doc)
        assert result is not None
        assert len(result.parameters) == 3

        # Verify unicode is preserved
        assert "émojis" in result.parameters[0].description
        assert "🚀" in result.parameters[0].description
        assert "中文" in result.parameters[0].description

        # Check example preservation
        assert result.examples
        assert any("世界" in ex for ex in result.examples)
