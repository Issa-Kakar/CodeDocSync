"""Test exact matching functionality."""

from codedocsync.matcher import DirectMatcher, MatchType
from codedocsync.parser import (
    ParsedFunction,
    FunctionSignature,
    FunctionParameter,
    RawDocstring,
    ParsedDocstring,
    DocstringParameter,
)


class TestExactMatching:
    """Test exact matching capabilities."""

    def test_exact_match_with_docstring(self):
        """Test matching function with its docstring."""
        # Create function with docstring
        func = ParsedFunction(
            signature=FunctionSignature(
                name="calculate_sum",
                parameters=[
                    FunctionParameter("a", "int", None, True),
                    FunctionParameter("b", "int", None, True),
                ],
                return_type="int",
            ),
            docstring=RawDocstring(
                raw_text='"""Calculate sum of two numbers."""', line_number=2
            ),
            file_path="math_utils.py",
            line_number=1,
            end_line_number=3,
            source_code='def calculate_sum(a: int, b: int) -> int:\n    """Calculate sum of two numbers."""\n    return a + b',
        )

        matcher = DirectMatcher()
        result = matcher.match_functions([func])

        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0].match_type == MatchType.EXACT
        assert result.matched_pairs[0].confidence.overall >= 0.95
        assert "Exact match" in result.matched_pairs[0].match_reason

    def test_no_match_without_docstring(self):
        """Test function without docstring is unmatched."""
        func = ParsedFunction(
            signature=FunctionSignature(name="helper_func"),
            docstring=None,  # No docstring!
            file_path="utils.py",
            line_number=10,
            end_line_number=12,
            source_code="def helper_func():\n    pass",
        )

        matcher = DirectMatcher()
        result = matcher.match_functions([func])

        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_functions) == 1
        assert result.unmatched_functions[0] == func

    def test_signature_similarity_calculation(self):
        """Test signature similarity scoring."""
        # Function with matching parameters in docstring
        func = ParsedFunction(
            signature=FunctionSignature(
                name="process_data",
                parameters=[
                    FunctionParameter("data", "List[str]", None, True),
                    FunctionParameter("validate", "bool", "True", False),
                ],
            ),
            docstring=ParsedDocstring(
                format="google",
                summary="Process data with validation.",
                parameters=[
                    DocstringParameter("data", "List[str]", "Data to process"),
                    DocstringParameter("validate", "bool", "Whether to validate"),
                ],
                raw_text='"""Process data with validation."""',
            ),
            file_path="processor.py",
            line_number=5,
            end_line_number=10,
            source_code="def process_data(data: List[str], validate: bool = True): ...",
        )

        matcher = DirectMatcher()
        confidence = matcher._calculate_exact_match_confidence(func)

        assert confidence.signature_similarity == 1.0  # Perfect match
        assert confidence.overall >= 0.95

    def test_multiple_files_matching(self):
        """Test matching across multiple files."""
        functions = [
            # File 1
            ParsedFunction(
                signature=FunctionSignature(name="func1"),
                docstring=RawDocstring(raw_text="Doc1", line_number=2),
                file_path="file1.py",
                line_number=1,
                end_line_number=3,
                source_code="def func1(): pass",
            ),
            # File 2
            ParsedFunction(
                signature=FunctionSignature(name="func2"),
                docstring=RawDocstring(raw_text="Doc2", line_number=2),
                file_path="file2.py",
                line_number=1,
                end_line_number=3,
                source_code="def func2(): pass",
            ),
            # File 1 again
            ParsedFunction(
                signature=FunctionSignature(name="func3"),
                docstring=None,  # No docstring
                file_path="file1.py",
                line_number=5,
                end_line_number=6,
                source_code="def func3(): pass",
            ),
        ]

        matcher = DirectMatcher()
        result = matcher.match_functions(functions)

        assert result.total_functions == 3
        assert len(result.matched_pairs) == 2
        assert len(result.unmatched_functions) == 1

        # Check grouping worked correctly
        stats = matcher.get_stats()
        assert stats["exact_matches"] == 2
        assert stats["no_matches"] == 1

    def test_empty_function_list(self):
        """Test handling of empty function list."""
        matcher = DirectMatcher()
        result = matcher.match_functions([])

        assert result.total_functions == 0
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_functions) == 0
        assert result.match_duration_ms == 0.0

    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(name="documented_func"),
                docstring=RawDocstring(raw_text="Documentation", line_number=2),
                file_path="test.py",
                line_number=1,
                end_line_number=3,
                source_code="def documented_func(): pass",
            ),
            ParsedFunction(
                signature=FunctionSignature(name="undocumented_func"),
                docstring=None,
                file_path="test.py",
                line_number=5,
                end_line_number=6,
                source_code="def undocumented_func(): pass",
            ),
        ]

        matcher = DirectMatcher()
        matcher.match_functions(functions)

        stats = matcher.get_stats()
        assert stats["exact_matches"] == 1
        assert stats["no_matches"] == 1
        assert stats["total_processed"] == 2

    def test_signature_similarity_edge_cases(self):
        """Test edge cases in signature similarity calculation."""
        matcher = DirectMatcher()

        # Function with no parameters
        func_no_params = ParsedFunction(
            signature=FunctionSignature(name="simple_func", parameters=[]),
            docstring=ParsedDocstring(
                format="google",
                summary="Simple function.",
                parameters=[],
                raw_text='"""Simple function."""',
            ),
            file_path="test.py",
            line_number=1,
            end_line_number=2,
            source_code="def simple_func(): pass",
        )

        confidence = matcher._calculate_exact_match_confidence(func_no_params)
        assert confidence.signature_similarity == 1.0

        # Function with RawDocstring (can't check parameters)
        func_raw_doc = ParsedFunction(
            signature=FunctionSignature(
                name="raw_doc_func",
                parameters=[FunctionParameter("x", "int", None, True)],
            ),
            docstring=RawDocstring(raw_text="Raw docstring", line_number=2),
            file_path="test.py",
            line_number=1,
            end_line_number=3,
            source_code="def raw_doc_func(x): pass",
        )

        confidence = matcher._calculate_exact_match_confidence(func_raw_doc)
        assert confidence.signature_similarity == 1.0  # Default for RawDocstring

    def test_parameter_mismatch_confidence(self):
        """Test confidence calculation when parameters don't match."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="mismatched_func",
                parameters=[
                    FunctionParameter("x", "int", None, True),
                    FunctionParameter("y", "int", None, True),
                ],
            ),
            docstring=ParsedDocstring(
                format="google",
                summary="Function with mismatched parameters.",
                parameters=[
                    DocstringParameter("a", "int", "First parameter"),
                    DocstringParameter("b", "int", "Second parameter"),
                ],
                raw_text='"""Function with mismatched parameters."""',
            ),
            file_path="test.py",
            line_number=1,
            end_line_number=5,
            source_code="def mismatched_func(x, y): pass",
        )

        matcher = DirectMatcher()
        confidence = matcher._calculate_exact_match_confidence(func)

        # Should have low signature similarity due to parameter mismatch
        assert confidence.signature_similarity == 0.0  # No intersection
        assert confidence.overall < 0.95  # Below exact match threshold

    def test_partial_parameter_match(self):
        """Test partial parameter matching in confidence calculation."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="partial_match_func",
                parameters=[
                    FunctionParameter("x", "int", None, True),
                    FunctionParameter("y", "int", None, True),
                    FunctionParameter("z", "int", None, True),
                ],
            ),
            docstring=ParsedDocstring(
                format="google",
                summary="Function with partial parameter match.",
                parameters=[
                    DocstringParameter("x", "int", "First parameter"),
                    DocstringParameter("y", "int", "Second parameter"),
                    DocstringParameter("w", "int", "Third parameter"),  # Different name
                ],
                raw_text='"""Function with partial parameter match."""',
            ),
            file_path="test.py",
            line_number=1,
            end_line_number=5,
            source_code="def partial_match_func(x, y, z): pass",
        )

        matcher = DirectMatcher()
        confidence = matcher._calculate_exact_match_confidence(func)

        # Should have 2/3 similarity (x,y match, z/w don't)
        # Jaccard: intersection=2, union=4, so 2/4 = 0.5
        assert confidence.signature_similarity == 0.5
        assert confidence.overall < 0.95  # Below exact match threshold
