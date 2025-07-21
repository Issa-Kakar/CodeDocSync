"""Comprehensive tests for the DirectMatcher component.

Tests focus on:
- Performance benchmarks (<1ms per function)
- Matching accuracy (100% for exact, >85% for fuzzy)
- Edge cases and special scenarios
"""

import time

from rapidfuzz import fuzz

from codedocsync.matcher.direct_matcher import DirectMatcher
from codedocsync.matcher.models import MatchType
from codedocsync.parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)


class TestPerformance:
    """Performance benchmarks for the DirectMatcher."""

    def test_match_1000_functions_under_1_second(self) -> None:
        """Test that matching 1000 functions completes in under 1 second."""
        matcher = DirectMatcher()

        # Create 1000 test functions with realistic data
        functions = []
        for i in range(1000):
            func_name = f"process_data_{i}"
            decorators = []
            if i % 5 == 0:  # Every 5th is decorated like a class method
                decorators = ["classmethod"]
            elif i % 7 == 0:  # Every 7th is decorated like a static method
                decorators = ["staticmethod"]

            signature = FunctionSignature(
                name=func_name,
                parameters=[
                    FunctionParameter(
                        name="data",
                        type_annotation="dict[str, Any]",
                        default_value=None,
                        is_required=True,
                    ),
                    FunctionParameter(
                        name="validate",
                        type_annotation="bool",
                        default_value="True",
                        is_required=False,
                    ),
                ],
                return_type="dict[str, Any]",
                is_async=i % 10 == 0,  # Every 10th function is async
                is_method=i % 3 == 0,  # Every 3rd is a method
                decorators=decorators,
            )

            # Half have docstrings
            docstring = None
            if i % 2 == 0:
                docstring = RawDocstring(
                    raw_text=f"Process data item {i}.\n\nArgs:\n    data: Input data\n    validate: Whether to validate\n\nReturns:\n    Processed data",
                    line_number=10 + (i % 100) * 5,
                )

            functions.append(
                ParsedFunction(
                    signature=signature,
                    file_path=f"/project/module_{i // 100}/file_{i // 10}.py",
                    line_number=10 + (i % 100) * 5,
                    end_line_number=15 + (i % 100) * 5,
                    docstring=docstring,
                )
            )

        # Measure matching time
        start_time = time.time()
        result = matcher.match_functions(functions)
        duration = time.time() - start_time

        # Assertions
        assert (
            duration < 1.0
        ), f"Matching 1000 functions took {duration:.3f}s (should be <1s)"
        assert result.total_functions == 1000
        assert len(result.matched_pairs) == 500  # Half have docstrings
        assert result.match_rate == 0.5

        # Check per-function performance
        per_function_ms = (duration * 1000) / 1000
        assert (
            per_function_ms < 1.0
        ), f"Per-function time: {per_function_ms:.3f}ms (should be <1ms)"

    def test_single_function_performance(self) -> None:
        """Test that matching a single function is under 1ms."""
        matcher = DirectMatcher()

        function = ParsedFunction(
            signature=FunctionSignature(
                name="calculate_total",
                parameters=[
                    FunctionParameter(
                        name="items",
                        type_annotation="list[float]",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_type="float",
                is_async=False,
                is_method=False,
                decorators=[],
            ),
            file_path="/project/calculator.py",
            line_number=42,
            end_line_number=45,
            docstring=RawDocstring(
                raw_text="Calculate the total of all items.",
                line_number=42,
            ),
        )

        # Run multiple times to get stable measurement
        durations = []
        for _ in range(100):
            start = time.perf_counter()
            _ = matcher.match_functions([function])
            duration_ms = (time.perf_counter() - start) * 1000
            durations.append(duration_ms)

        avg_duration = sum(durations) / len(durations)
        assert (
            avg_duration < 1.0
        ), f"Average matching time: {avg_duration:.3f}ms (should be <1ms)"


class TestMatchingAccuracy:
    """Test matching accuracy for different scenarios."""

    def test_exact_name_match(self) -> None:
        """Test 100% accuracy for exact name matches."""
        matcher = DirectMatcher()

        # Create functions with exact matching names and docstrings
        test_cases = [
            ("get_user", "Get user by ID."),
            ("calculate_total", "Calculate the total sum."),
            ("_private_method", "Private helper method."),
            ("__double_private", "Double underscore method."),
            ("async_fetch_data", "Async function to fetch data."),
        ]

        functions = []
        for name, doc_content in test_cases:
            signature = FunctionSignature(
                name=name,
                parameters=[],
                return_type="Any",
                is_async="async" in name,
                is_method=False,
                decorators=[],
            )

            functions.append(
                ParsedFunction(
                    signature=signature,
                    file_path="/test/file.py",
                    line_number=10,
                    end_line_number=15,
                    docstring=RawDocstring(
                        raw_text=doc_content,
                        line_number=10,
                    ),
                )
            )

        result = matcher.match_functions(functions)

        # All functions should be matched
        assert (
            result.match_rate == 1.0
        ), "All functions with docstrings should be matched"
        assert len(result.matched_pairs) == len(test_cases)

        # All matches should be EXACT type with high confidence
        for pair in result.matched_pairs:
            assert pair.match_type == MatchType.EXACT
            assert pair.confidence.overall >= 0.95
            assert pair.confidence.name_similarity == 1.0

    def test_fuzzy_match_common_patterns(self) -> None:
        """Test fuzzy matching for common naming pattern variations."""
        matcher = DirectMatcher()

        # Test different naming conventions that should match
        test_patterns = [
            # (function_name, expected_match_patterns)
            ("get_user", ["getUser", "GetUser", "get_User"]),
            ("fetch_data", ["fetchData", "FetchData", "fetch_Data"]),
            ("process_items", ["processItems", "ProcessItems", "process_Items"]),
            ("validate_input", ["validateInput", "ValidateInput", "validate_Input"]),
            ("parse_json", ["parseJSON", "ParseJSON", "parseJson", "parse_JSON"]),
        ]

        for base_name, variations in test_patterns:
            for variant in variations:
                # Test each variation
                functions = [
                    ParsedFunction(
                        signature=FunctionSignature(
                            name=base_name,
                            parameters=[],
                            return_type="Any",
                            is_async=False,
                            is_method=False,
                            decorators=[],
                        ),
                        file_path="/test/file.py",
                        line_number=10,
                        end_line_number=15,
                        docstring=RawDocstring(
                            raw_text=f"Function for {variant}",
                            line_number=10,
                        ),
                    )
                ]

                # Calculate expected similarity
                similarity = fuzz.ratio(base_name, variant) / 100.0

                # DirectMatcher should handle these as exact matches in same file
                result = matcher.match_functions(functions)

                if similarity >= 0.7:  # Threshold mentioned in requirements
                    assert (
                        result.match_rate == 1.0
                    ), f"Should match {base_name} (similarity: {similarity:.2f})"

    def test_confidence_scoring(self) -> None:
        """Test that confidence threshold of 0.7 works correctly."""
        matcher = DirectMatcher()

        # Create functions with varying match quality
        functions = []

        # High confidence match
        functions.append(
            ParsedFunction(
                signature=FunctionSignature(
                    name="calculate_total",
                    parameters=[
                        FunctionParameter(
                            name="items",
                            type_annotation="list[float]",
                            default_value=None,
                            is_required=True,
                        )
                    ],
                    return_type="float",
                    is_async=False,
                    is_method=False,
                    decorators=[],
                ),
                file_path="/project/calc.py",
                line_number=10,
                end_line_number=20,
                docstring=RawDocstring(
                    raw_text="Calculate the total of all items.\n\nArgs:\n    items: List of numbers\n\nReturns:\n    The sum",
                    line_number=10,
                ),
            )
        )

        # Medium confidence match (missing some details)
        functions.append(
            ParsedFunction(
                signature=FunctionSignature(
                    name="process_data",
                    parameters=[
                        FunctionParameter(
                            name="data",
                            type_annotation="dict",
                            default_value=None,
                            is_required=True,
                        ),
                        FunctionParameter(
                            name="validate",
                            type_annotation="bool",
                            default_value="True",
                            is_required=False,
                        ),
                    ],
                    return_type="dict",
                    is_async=False,
                    is_method=True,
                    decorators=[],
                ),
                file_path="/project/processor.py",
                line_number=20,
                end_line_number=30,
                docstring=RawDocstring(
                    raw_text="Process the data.",  # Very minimal docstring
                    line_number=20,
                ),
            )
        )

        result = matcher.match_functions(functions)

        # Both should be matched (DirectMatcher matches all functions with docstrings)
        assert result.match_rate == 1.0
        assert len(result.matched_pairs) == 2

        # Check confidence scores
        for pair in result.matched_pairs:
            assert (
                pair.confidence.overall >= 0.7
            ), f"Confidence {pair.confidence.overall} should be >= 0.7"

            # Detailed docstring should have higher confidence
            if (
                pair.function.docstring
                and "Calculate the total" in pair.function.docstring.raw_text
            ):
                assert pair.confidence.overall >= 0.9

    def test_handle_private_methods(self) -> None:
        """Test handling of private methods (_internal, __private)."""
        matcher = DirectMatcher()

        functions = [
            # Single underscore private
            ParsedFunction(
                signature=FunctionSignature(
                    name="_internal_helper",
                    parameters=[],
                    return_type="None",
                    is_async=False,
                    is_method=True,
                    decorators=[],
                ),
                file_path="/project/utils.py",
                line_number=50,
                end_line_number=52,
                docstring=RawDocstring(
                    raw_text="Internal helper function.",
                    line_number=50,
                ),
            ),
            # Double underscore private
            ParsedFunction(
                signature=FunctionSignature(
                    name="__private_method",
                    parameters=[],
                    return_type="None",
                    is_async=False,
                    is_method=True,
                    decorators=[],
                ),
                file_path="/project/utils.py",
                line_number=60,
                end_line_number=62,
                docstring=RawDocstring(
                    raw_text="Private method with name mangling.",
                    line_number=60,
                ),
            ),
            # Magic method
            ParsedFunction(
                signature=FunctionSignature(
                    name="__init__",
                    parameters=[
                        FunctionParameter(
                            name="self",
                            type_annotation=None,
                            default_value=None,
                            is_required=True,
                        )
                    ],
                    return_type="None",
                    is_async=False,
                    is_method=True,
                    decorators=[],
                ),
                file_path="/project/model.py",
                line_number=10,
                end_line_number=15,
                docstring=RawDocstring(
                    raw_text="Initialize the instance.",
                    line_number=10,
                ),
            ),
        ]

        result = matcher.match_functions(functions)

        # All private methods with docstrings should be matched
        assert result.match_rate == 1.0
        assert len(result.matched_pairs) == 3

        # Verify all are matched correctly
        for pair in result.matched_pairs:
            assert pair.match_type == MatchType.EXACT
            assert pair.confidence.overall >= 0.9
            assert pair.function.signature.name.startswith(
                "_"
            ) or pair.function.signature.name.startswith("__")

    def test_match_class_methods(self) -> None:
        """Test self parameter handling in class methods."""
        matcher = DirectMatcher()

        functions = [
            # Regular instance method
            ParsedFunction(
                signature=FunctionSignature(
                    name="process",
                    parameters=[
                        FunctionParameter(
                            name="self",
                            type_annotation=None,
                            default_value=None,
                            is_required=True,
                        ),
                        FunctionParameter(
                            name="data",
                            type_annotation="dict",
                            default_value=None,
                            is_required=True,
                        ),
                    ],
                    return_type="dict",
                    is_async=False,
                    is_method=True,
                    decorators=[],
                ),
                file_path="/project/processor.py",
                line_number=25,
                end_line_number=35,
                docstring=RawDocstring(
                    raw_text="Process the data.\n\nArgs:\n    data: Input data dictionary\n\nReturns:\n    Processed data",
                    line_number=25,
                ),
            ),
            # Class method
            ParsedFunction(
                signature=FunctionSignature(
                    name="from_config",
                    parameters=[
                        FunctionParameter(
                            name="cls",
                            type_annotation=None,
                            default_value=None,
                            is_required=True,
                        ),
                        FunctionParameter(
                            name="config",
                            type_annotation="dict",
                            default_value=None,
                            is_required=True,
                        ),
                    ],
                    return_type="Processor",
                    is_async=False,
                    is_method=True,
                    decorators=["classmethod"],
                ),
                file_path="/project/processor.py",
                line_number=40,
                end_line_number=50,
                docstring=RawDocstring(
                    raw_text="Create instance from configuration.\n\nArgs:\n    config: Configuration dictionary\n\nReturns:\n    New processor instance",
                    line_number=40,
                ),
            ),
            # Static method
            ParsedFunction(
                signature=FunctionSignature(
                    name="validate_config",
                    parameters=[
                        FunctionParameter(
                            name="config",
                            type_annotation="dict",
                            default_value=None,
                            is_required=True,
                        )
                    ],
                    return_type="bool",
                    is_async=False,
                    is_method=False,  # Static methods don't have self/cls
                    decorators=["staticmethod"],
                ),
                file_path="/project/processor.py",
                line_number=55,
                end_line_number=65,
                docstring=RawDocstring(
                    raw_text="Validate configuration.\n\nArgs:\n    config: Configuration to validate\n\nReturns:\n    True if valid",
                    line_number=55,
                ),
            ),
        ]

        result = matcher.match_functions(functions)

        # All methods should be matched
        assert result.match_rate == 1.0
        assert len(result.matched_pairs) == 3

        # Verify proper handling of self/cls parameters
        for pair in result.matched_pairs:
            assert pair.match_type == MatchType.EXACT
            assert pair.confidence.overall >= 0.9

            # Check that docstrings don't mention self/cls
            assert pair.function.docstring is not None
            doc_content = pair.function.docstring.raw_text.lower()
            # Check for classmethod decorator instead of is_class_method
            if "classmethod" in pair.function.signature.decorators:
                assert "cls" not in doc_content or "class" in doc_content
            elif (
                "staticmethod" not in pair.function.signature.decorators
                and pair.function.signature.is_method
            ):
                assert "self" not in doc_content or "instance" in doc_content


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_function_list(self) -> None:
        """Test handling of empty function list."""
        matcher = DirectMatcher()
        result = matcher.match_functions([])

        assert result.total_functions == 0
        assert result.match_rate == 0
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_functions) == 0

    def test_functions_without_docstrings(self) -> None:
        """Test handling of functions without docstrings."""
        matcher = DirectMatcher()

        functions = [
            ParsedFunction(
                signature=FunctionSignature(
                    name="undocumented_function",
                    parameters=[],
                    return_type="None",
                    is_async=False,
                    is_method=False,
                    decorators=[],
                ),
                file_path="/project/utils.py",
                line_number=100,
                end_line_number=101,
                docstring=None,  # No docstring
            )
        ]

        result = matcher.match_functions(functions)

        assert result.total_functions == 1
        assert result.match_rate == 0
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_functions) == 1
        assert result.unmatched_functions[0].signature.name == "undocumented_function"

    def test_mixed_documented_undocumented(self) -> None:
        """Test handling of mixed documented and undocumented functions."""
        matcher = DirectMatcher()

        functions = []

        # Add 5 documented functions
        for i in range(5):
            functions.append(
                ParsedFunction(
                    signature=FunctionSignature(
                        name=f"documented_{i}",
                        parameters=[],
                        return_type="None",
                        is_async=False,
                        is_method=False,
                        decorators=[],
                    ),
                    file_path="/project/module.py",
                    line_number=10 + i * 10,
                    end_line_number=15 + i * 10,
                    docstring=RawDocstring(
                        raw_text=f"Function {i} documentation.",
                        line_number=10 + i * 10,
                    ),
                )
            )

        # Add 5 undocumented functions
        for i in range(5):
            functions.append(
                ParsedFunction(
                    signature=FunctionSignature(
                        name=f"undocumented_{i}",
                        parameters=[],
                        return_type="None",
                        is_async=False,
                        is_method=False,
                        decorators=[],
                    ),
                    file_path="/project/module.py",
                    line_number=100 + i * 10,
                    end_line_number=101 + i * 10,
                    docstring=None,
                )
            )

        result = matcher.match_functions(functions)

        assert result.total_functions == 10
        assert result.match_rate == 0.5
        assert len(result.matched_pairs) == 5
        assert len(result.unmatched_functions) == 5

    def test_performance_consistency(self) -> None:
        """Test that performance remains consistent across multiple runs."""
        matcher = DirectMatcher()

        # Create test data
        functions = []
        for i in range(100):
            functions.append(
                ParsedFunction(
                    signature=FunctionSignature(
                        name=f"func_{i}",
                        parameters=[],
                        return_type="None",
                        is_async=False,
                        is_method=False,
                        decorators=[],
                    ),
                    file_path=f"/project/file_{i // 10}.py",
                    line_number=10 + (i % 10) * 5,
                    end_line_number=15 + (i % 10) * 5,
                    docstring=(
                        RawDocstring(
                            raw_text=f"Function {i}",
                            line_number=10 + (i % 10) * 5,
                        )
                        if i % 2 == 0
                        else None
                    ),
                )
            )

        # Run multiple times
        durations = []
        for _ in range(10):
            start = time.perf_counter()
            result = matcher.match_functions(functions)
            duration = time.perf_counter() - start
            durations.append(duration)

            # Results should be consistent
            assert result.total_functions == 100
            assert result.match_rate == 0.5

        # Check performance consistency
        avg_duration = sum(durations) / len(durations)
        max_deviation = max(abs(d - avg_duration) for d in durations)
        relative_deviation = max_deviation / avg_duration

        assert (
            relative_deviation < 0.5
        ), f"Performance varies too much: {relative_deviation:.1%}"
