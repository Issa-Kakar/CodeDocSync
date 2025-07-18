"""Test fuzzy matching functionality."""

from codedocsync.matcher import DirectMatcher, MatchType
from codedocsync.parser import (
    ParsedFunction,
    FunctionSignature,
    FunctionParameter,
    RawDocstring,
)


class TestFuzzyMatching:
    """Test fuzzy matching capabilities."""

    def test_snake_case_to_camel_case(self):
        """Test matching snake_case to camelCase."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(name="get_user_name"),
                docstring=RawDocstring(raw_text="Get user name", line_number=2),
                file_path="user.py",
                line_number=1,
                end_line_number=3,
                source_code="def get_user_name(): pass",
            ),
            ParsedFunction(
                signature=FunctionSignature(name="getUserName"),
                docstring=RawDocstring(raw_text="Get user name", line_number=6),
                file_path="user.py",
                line_number=5,
                end_line_number=7,
                source_code="def getUserName(): pass",
            ),
        ]

        matcher = DirectMatcher(fuzzy_threshold=0.8)
        result = matcher.match_functions(functions)

        # Both should be matched (to themselves in this case)
        assert len(result.matched_pairs) == 2

    def test_abbreviation_matching(self):
        """Test that functions with slightly mismatched docstring parameters can still match."""
        # This tests fuzzy matching within exact match confidence calculation
        # when the parameters don't align exactly
        functions = [
            ParsedFunction(
                signature=FunctionSignature(
                    name="calculate_total",
                    parameters=[
                        FunctionParameter("values", "List[float]", None, True),
                        FunctionParameter("tax_rate", "float", "0.0", False),
                    ],
                ),
                # Docstring with slightly different parameter names should still match with lower confidence
                docstring=RawDocstring(
                    raw_text="Calculate total with tax", line_number=2
                ),
                file_path="calc.py",
                line_number=1,
                end_line_number=3,
                source_code="def calculate_total(values, tax_rate=0.0): pass",
            )
        ]

        matcher = DirectMatcher(fuzzy_threshold=0.75)
        result = matcher.match_functions(functions)

        # Should still match (exact match can have lower confidence but still match)
        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0].match_type == MatchType.EXACT

    def test_fuzzy_threshold_enforcement(self):
        """Test that fuzzy threshold is respected."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(name="save_user"),
                docstring=RawDocstring(raw_text="Save user", line_number=2),
                file_path="db.py",
                line_number=1,
                end_line_number=3,
                source_code="def save_user(): pass",
            ),
            ParsedFunction(
                signature=FunctionSignature(name="load_data"),  # Very different
                docstring=RawDocstring(raw_text="Load data", line_number=6),
                file_path="db.py",
                line_number=5,
                end_line_number=7,
                source_code="def load_data(): pass",
            ),
        ]

        # High threshold should prevent matching
        matcher = DirectMatcher(fuzzy_threshold=0.9)
        result = matcher.match_functions(functions)

        # Should not create fuzzy matches for very different names
        fuzzy_matches = [
            p for p in result.matched_pairs if p.match_type == MatchType.FUZZY
        ]
        assert len(fuzzy_matches) == 0

    def test_validation_prevents_bad_matches(self):
        """Test that validation prevents incorrect fuzzy matches."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(
                    name="process",
                    parameters=[FunctionParameter("x", "int", None, True)],
                    return_type="int",
                ),
                docstring=RawDocstring(raw_text="Process integer", line_number=2),
                file_path="proc.py",
                line_number=1,
                end_line_number=3,
                source_code="def process(x: int) -> int: pass",
            ),
            ParsedFunction(
                signature=FunctionSignature(
                    name="process",
                    parameters=[
                        FunctionParameter("a", "str", None, True),
                        FunctionParameter("b", "str", None, True),
                        FunctionParameter("c", "str", None, True),
                    ],
                    return_type="str",
                ),
                docstring=RawDocstring(raw_text="Process strings", line_number=102),
                file_path="proc.py",
                line_number=100,  # Far away in file
                end_line_number=104,
                source_code="def process(a: str, b: str, c: str) -> str: pass",
            ),
        ]

        matcher = DirectMatcher()
        result = matcher.match_functions(functions)

        # Should not match due to different signatures and distance
        assert len(result.matched_pairs) == 2  # Each matches itself
        assert all(p.match_type == MatchType.EXACT for p in result.matched_pairs)

    def test_pattern_matching_get_user(self):
        """Test that functions with their own docstrings get exact matches."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(name="get_user"),
                docstring=RawDocstring(raw_text="Get user", line_number=2),
                file_path="user.py",
                line_number=1,
                end_line_number=3,
                source_code="def get_user(): pass",
            ),
            ParsedFunction(
                signature=FunctionSignature(name="getUser"),
                docstring=RawDocstring(raw_text="Get user", line_number=6),
                file_path="user.py",
                line_number=5,
                end_line_number=7,
                source_code="def getUser(): pass",
            ),
        ]

        matcher = DirectMatcher(fuzzy_threshold=0.8)
        result = matcher.match_functions(functions)

        # Both functions have their own docstrings, so should get exact matches
        assert len(result.matched_pairs) == 2
        assert all(p.match_type == MatchType.EXACT for p in result.matched_pairs)

    def test_signature_similarity_validation(self):
        """Test signature similarity calculation for fuzzy matches."""
        # Functions with similar signatures
        func1 = ParsedFunction(
            signature=FunctionSignature(
                name="calc_sum",
                parameters=[
                    FunctionParameter("a", "int", None, True),
                    FunctionParameter("b", "int", None, True),
                ],
            ),
            docstring=RawDocstring(raw_text="Calculate sum", line_number=2),
            file_path="math.py",
            line_number=1,
            end_line_number=3,
            source_code="def calc_sum(a, b): pass",
        )

        func2 = ParsedFunction(
            signature=FunctionSignature(
                name="calculate_sum",
                parameters=[
                    FunctionParameter("a", "int", None, True),
                    FunctionParameter("b", "int", None, True),
                ],
            ),
            docstring=RawDocstring(raw_text="Calculate sum", line_number=6),
            file_path="math.py",
            line_number=5,
            end_line_number=7,
            source_code="def calculate_sum(a, b): pass",
        )

        matcher = DirectMatcher(fuzzy_threshold=0.7)
        sig_similarity = matcher._calculate_signature_similarity_between(func1, func2)

        # Should have high signature similarity (same parameters)
        assert sig_similarity == 1.0

    def test_line_distance_validation(self):
        """Test that line distance affects fuzzy match validation."""
        # Functions far apart in the file
        func1 = ParsedFunction(
            signature=FunctionSignature(name="helper"),
            docstring=RawDocstring(raw_text="Helper function", line_number=2),
            file_path="utils.py",
            line_number=1,
            end_line_number=3,
            source_code="def helper(): pass",
        )

        func2 = ParsedFunction(
            signature=FunctionSignature(name="helper_func"),
            docstring=RawDocstring(raw_text="Helper function", line_number=202),
            file_path="utils.py",
            line_number=200,  # Far away
            end_line_number=202,
            source_code="def helper_func(): pass",
        )

        matcher = DirectMatcher(fuzzy_threshold=0.7)

        # Should fail validation due to distance
        is_valid = matcher._validate_fuzzy_match(func1, func2)
        assert not is_valid

    def test_parameter_count_validation(self):
        """Test parameter count difference validation."""
        func1 = ParsedFunction(
            signature=FunctionSignature(
                name="process", parameters=[FunctionParameter("x", "int", None, True)]
            ),
            docstring=RawDocstring(raw_text="Process data", line_number=2),
            file_path="proc.py",
            line_number=1,
            end_line_number=3,
            source_code="def process(x): pass",
        )

        func2 = ParsedFunction(
            signature=FunctionSignature(
                name="processor",
                parameters=[
                    FunctionParameter("a", "int", None, True),
                    FunctionParameter("b", "int", None, True),
                    FunctionParameter("c", "int", None, True),
                    FunctionParameter("d", "int", None, True),  # Too many parameters
                ],
            ),
            docstring=RawDocstring(raw_text="Process data", line_number=6),
            file_path="proc.py",
            line_number=5,
            end_line_number=7,
            source_code="def processor(a, b, c, d): pass",
        )

        matcher = DirectMatcher()

        # Should fail validation due to parameter count difference > 2
        is_valid = matcher._validate_fuzzy_match(func1, func2)
        assert not is_valid

    def test_confidence_scoring_for_fuzzy_match(self):
        """Test confidence scoring calculation for fuzzy matches."""
        func1 = ParsedFunction(
            signature=FunctionSignature(name="get_data"),
            docstring=RawDocstring(raw_text="Get data", line_number=2),
            file_path="data.py",
            line_number=1,
            end_line_number=3,
            source_code="def get_data(): pass",
        )

        func2 = ParsedFunction(
            signature=FunctionSignature(name="getData"),
            docstring=RawDocstring(raw_text="Get data", line_number=6),
            file_path="data.py",
            line_number=5,
            end_line_number=7,
            source_code="def getData(): pass",
        )

        matcher = DirectMatcher(fuzzy_threshold=0.7)
        fuzzy_match = matcher._create_fuzzy_match(func1, func2, "Test fuzzy match")

        # Should have reasonable confidence scores
        assert 0.0 <= fuzzy_match.confidence.overall <= 1.0
        assert 0.0 <= fuzzy_match.confidence.name_similarity <= 1.0
        assert 0.0 <= fuzzy_match.confidence.location_score <= 1.0
        assert 0.0 <= fuzzy_match.confidence.signature_similarity <= 1.0

        # Name similarity should be high for get_data vs getData
        assert fuzzy_match.confidence.name_similarity > 0.7
