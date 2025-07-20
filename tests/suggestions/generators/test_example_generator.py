"""
Tests for the Example Generation System.

Tests cover example generation, parameter value generation, and usage example
creation for various function scenarios.
"""

from unittest.mock import Mock

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.generators.example_generator import (
    ExampleGenerator,
    ExamplePatternAnalyzer,
    ExampleSuggestionGenerator,
    ExampleTemplate,
    ParameterValueGenerator,
)
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionContext,
    SuggestionType,
)


class TestParameterValueGenerator:
    """Test the parameter value generator."""

    @pytest.fixture
    def generator(self):
        """Create parameter value generator."""
        return ParameterValueGenerator()

    def test_generate_value_by_name(self, generator):
        """Test value generation based on parameter names."""
        test_cases = [
            ("name", '"John Doe"'),
            ("filename", '"example.txt"'),
            ("url", '"https://example.com"'),
            ("email", '"user@example.com"'),
            ("count", "10"),
            ("age", "25"),
            ("data", "[1, 2, 3, 4, 5]"),
        ]

        for param_name, expected_value in test_cases:
            param = Mock()
            param.name = param_name
            param.type_annotation = None
            param.is_required = True
            param.default_value = None

            result = generator.generate_value(param)
            assert result == expected_value

    def test_generate_value_by_type(self, generator):
        """Test value generation based on type annotations."""
        test_cases = [
            ("str", '"example"'),
            ("int", "42"),
            ("float", "3.14"),
            ("bool", "True"),
            ("List[str]", "[1, 2, 3]"),
            ("Dict[str, Any]", '{"key": "value"}'),
        ]

        for type_annotation, expected_value in test_cases:
            param = Mock()
            param.name = "param"
            param.type_annotation = type_annotation
            param.is_required = True
            param.default_value = None

            result = generator.generate_value(param)
            assert result == expected_value

    def test_generate_value_optional_with_default(self, generator):
        """Test value generation for optional parameters with defaults."""
        param = Mock()
        param.name = "timeout"
        param.type_annotation = "int"
        param.is_required = False
        param.default_value = "30"

        result = generator.generate_value(param)
        assert result == "30"

    def test_normalize_type(self, generator):
        """Test type normalization for complex types."""
        test_cases = [
            ("Optional[str]", "str"),
            ("Union[str, None]", "str"),
            ("List[Dict[str, Any]]", "List"),
            ("Dict[str, List[int]]", "Dict"),
        ]

        for input_type, expected_normalized in test_cases:
            result = generator._normalize_type(input_type)
            assert expected_normalized in result

    def test_fallback_value(self, generator):
        """Test fallback value generation."""
        param = Mock()
        param.name = "unknown_param"
        param.type_annotation = "UnknownType"
        param.is_required = True
        param.default_value = None

        result = generator.generate_value(param)
        assert result == '"example"'  # Default fallback


class TestExamplePatternAnalyzer:
    """Test the example pattern analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create example pattern analyzer."""
        return ExamplePatternAnalyzer()

    def test_analyze_function_decorators(self, analyzer):
        """Test analyzing function decorators."""
        function = Mock()
        function.signature = Mock()
        function.signature.decorators = ["property"]

        analysis = analyzer.analyze_function(function)

        assert analysis["is_property"] is True
        assert analysis["is_classmethod"] is False
        assert analysis["is_staticmethod"] is False

    def test_analyze_async_function(self, analyzer):
        """Test analyzing async functions."""
        source_code = """
async def fetch_data():
    return await api_call()
"""

        analysis = analyzer.analyze_function(Mock(), source_code)

        assert analysis["is_async"] is True

    def test_analyze_generator_function(self, analyzer):
        """Test analyzing generator functions."""
        source_code = """
def number_generator():
    for i in range(10):
        yield i
"""

        analysis = analyzer.analyze_function(Mock(), source_code)

        assert analysis["is_generator"] is True

    def test_analyze_side_effects(self, analyzer):
        """Test analyzing functions with side effects."""
        source_code = """
def process_file(filename):
    with open(filename, 'w') as f:
        f.write("data")
    print("Done")
"""

        analysis = analyzer.analyze_function(Mock(), source_code)

        assert analysis["has_side_effects"] is True

    def test_analyze_domain_detection(self, analyzer):
        """Test domain detection from function calls."""
        math_source = """
def calculate():
    return sqrt(x) + sin(y)
"""

        file_source = """
def process():
    with open("file.txt") as f:
        return f.read()
"""

        math_analysis = analyzer.analyze_function(Mock(), math_source)
        file_analysis = analyzer.analyze_function(Mock(), file_source)

        assert math_analysis["domain"] == "math"
        assert file_analysis["domain"] == "file"


class TestExampleGenerator:
    """Test the example generator."""

    @pytest.fixture
    def generator(self):
        """Create example generator."""
        return ExampleGenerator()

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "calculate_sum"

        # Mock parameters
        param1 = Mock()
        param1.name = "a"
        param1.type_annotation = "int"
        param1.is_required = True
        param1.default_value = None

        param2 = Mock()
        param2.name = "b"
        param2.type_annotation = "int"
        param2.is_required = True
        param2.default_value = None

        function.signature.parameters = [param1, param2]
        return function

    def test_generate_basic_example(self, generator, mock_function):
        """Test generating basic usage example."""
        analysis = {
            "is_property": False,
            "is_classmethod": False,
            "is_staticmethod": False,
            "is_async": False,
            "is_generator": False,
            "return_type": "int",
        }

        example = generator._generate_basic_example(mock_function, analysis)

        assert isinstance(example, ExampleTemplate)
        assert example.complexity == "basic"
        assert "calculate_sum(" in example.function_call
        assert "a=" in example.function_call
        assert "b=" in example.function_call

    def test_generate_edge_case_example(self, generator, mock_function):
        """Test generating edge case example."""
        analysis = {
            "is_async": False,
            "return_type": "int",
        }

        example = generator._generate_edge_case_example(mock_function, analysis)

        assert isinstance(example, ExampleTemplate)
        assert example.complexity == "intermediate"
        assert "Edge case" in example.description

    def test_generate_advanced_example(self, generator, mock_function):
        """Test generating advanced example."""
        analysis = {
            "is_async": False,
            "return_type": "int",
        }

        example = generator._generate_advanced_example(mock_function, analysis)

        assert isinstance(example, ExampleTemplate)
        assert example.complexity == "advanced"
        assert "Advanced" in example.description
        assert len(example.setup_code) > 0

    def test_generate_async_example(self, generator):
        """Test generating example for async function."""
        async_function = Mock()
        async_function.signature = Mock()
        async_function.signature.name = "fetch_data"
        async_function.signature.parameters = []

        analysis = {
            "is_async": True,
            "return_type": "dict",
        }

        example = generator._generate_basic_example(async_function, analysis)

        assert "await " in example.function_call
        assert "async context" in " ".join(example.setup_code).lower()

    def test_generate_property_example(self, generator):
        """Test generating example for property."""
        property_function = Mock()
        property_function.signature = Mock()
        property_function.signature.name = "name"
        property_function.signature.parameters = []

        analysis = {
            "is_property": True,
            "return_type": "str",
        }

        example = generator._generate_basic_example(property_function, analysis)

        # Property access should not have parentheses
        assert "name(" not in example.function_call
        assert "instance.name" in example.function_call

    def test_generate_edge_case_values(self, generator):
        """Test generation of edge case parameter values."""
        test_cases = [
            ("count", "0"),
            ("index", "-1"),
            ("items", "[]"),
            ("config", "{}"),
        ]

        for param_name, expected_value in test_cases:
            param = Mock()
            param.name = param_name
            param.type_annotation = None

            result = generator._generate_edge_case_value(param)
            assert result == expected_value

    def test_generate_advanced_values(self, generator):
        """Test generation of advanced parameter values."""
        param = Mock()
        param.name = "config"
        param.type_annotation = "dict"

        result = generator._generate_advanced_value(param)

        # Should be a complex dictionary
        assert "{" in result and "}" in result

    def test_generate_expected_output(self, generator):
        """Test generation of expected output."""
        test_cases = [
            ({"return_type": "str"}, '"result"'),
            ({"return_type": "int"}, "42"),
            ({"return_type": "bool"}, "True"),
            ({"return_type": "None"}, "None"),
            ({"is_generator": True}, "# Generator object"),
        ]

        for analysis, expected_output in test_cases:
            result = generator._generate_expected_output(analysis)
            assert result == expected_output

    def test_generate_multiple_examples(self, generator, mock_function):
        """Test generating multiple examples."""
        examples = generator.generate_examples(mock_function, count=3)

        assert isinstance(examples, list)
        assert len(examples) <= 3  # May be fewer if generation fails

        if examples:
            # Should have different complexity levels
            complexities = [ex.complexity for ex in examples]
            assert "basic" in complexities


class TestExampleSuggestionGenerator:
    """Test the example suggestion generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(
            default_style="google",
            max_line_length=88,
        )

    @pytest.fixture
    def generator(self, config):
        """Create example suggestion generator."""
        return ExampleSuggestionGenerator(config)

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "calculate_area"
        function.line_number = 10

        # Mock parameters
        param1 = Mock()
        param1.name = "width"
        param1.type_annotation = "float"
        param1.is_required = True

        param2 = Mock()
        param2.name = "height"
        param2.type_annotation = "float"
        param2.is_required = True

        function.signature.parameters = [param1, param2]
        function.source_code = """
def calculate_area(width, height):
    return width * height
"""
        return function

    @pytest.fixture
    def mock_docstring(self):
        """Create mock docstring."""
        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Calculate area"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Calculate area."""'
        return docstring

    @pytest.fixture
    def mock_issue(self):
        """Create mock issue."""
        return InconsistencyIssue(
            issue_type="missing_examples",
            severity="low",
            description="Missing usage examples",
            suggestion="Add examples",
            line_number=10,
        )

    def test_add_missing_examples(
        self, generator, mock_function, mock_docstring, mock_issue
    ):
        """Test adding missing examples."""
        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_missing_examples(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.EXAMPLE_UPDATE
        assert suggestion.confidence >= 0.5

        # Should contain example code
        suggested_text = suggestion.suggested_text
        assert "calculate_area(" in suggested_text
        assert "width=" in suggested_text
        assert "height=" in suggested_text

    def test_fix_invalid_example(
        self, generator, mock_function, mock_docstring, mock_issue
    ):
        """Test fixing invalid examples."""
        mock_issue.issue_type = "example_invalid"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._fix_invalid_example(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.EXAMPLE_UPDATE

    def test_update_outdated_example(
        self, generator, mock_function, mock_docstring, mock_issue
    ):
        """Test updating outdated examples."""
        mock_issue.issue_type = "example_outdated"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._update_outdated_example(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.EXAMPLE_UPDATE

    def test_complete_example(
        self, generator, mock_function, mock_docstring, mock_issue
    ):
        """Test completing incomplete examples."""
        mock_issue.issue_type = "example_incomplete"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._complete_example(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.EXAMPLE_UPDATE

    def test_format_example_code(self, generator):
        """Test formatting example code."""
        example = ExampleTemplate(
            setup_code=["data = [1, 2, 3]"],
            function_call="result = process_data(data)",
            expected_output="[2, 4, 6]",
            imports=["import math"],
            description="Basic usage",
            complexity="basic",
        )

        formatted = generator._format_example_code(example)

        assert "import math" in formatted
        assert "data = [1, 2, 3]" in formatted
        assert "result = process_data(data)" in formatted
        assert "# Expected: [2, 4, 6]" in formatted

    def test_no_examples_generated_fallback(
        self, generator, mock_function, mock_docstring, mock_issue
    ):
        """Test fallback when no examples can be generated."""
        # Function without signature
        mock_function.signature = None

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_missing_examples(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence == 0.1  # Fallback suggestion

    def test_unknown_issue_type(self, generator, mock_function, mock_docstring):
        """Test handling unknown issue types."""
        unknown_issue = InconsistencyIssue(
            issue_type="unknown_example_issue",
            severity="low",
            description="Unknown issue",
            suggestion="",
            line_number=10,
        )

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=unknown_issue
        )

        suggestion = generator.generate(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence == 0.1
        assert "Unknown example issue type" in suggestion.description


class TestExampleGeneratorIntegration:
    """Integration tests for example generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(default_style="google")

    @pytest.fixture
    def generator(self, config):
        """Create example suggestion generator."""
        return ExampleSuggestionGenerator(config)

    def test_complete_workflow_mathematical_function(self, generator):
        """Test complete workflow for mathematical function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "calculate_distance"
        function.line_number = 5

        # Parameters
        x1_param = Mock()
        x1_param.name = "x1"
        x1_param.type_annotation = "float"
        x1_param.is_required = True
        x1_param.default_value = None

        y1_param = Mock()
        y1_param.name = "y1"
        y1_param.type_annotation = "float"
        y1_param.is_required = True
        y1_param.default_value = None

        x2_param = Mock()
        x2_param.name = "x2"
        x2_param.type_annotation = "float"
        x2_param.is_required = True
        x2_param.default_value = None

        y2_param = Mock()
        y2_param.name = "y2"
        y2_param.type_annotation = "float"
        y2_param.is_required = True
        y2_param.default_value = None

        function.signature.parameters = [x1_param, y1_param, x2_param, y2_param]
        function.source_code = """
def calculate_distance(x1, y1, x2, y2):
    import math
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
"""

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Calculate distance between two points"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []  # Missing examples
        docstring.raw_text = '"""Calculate distance between two points."""'

        issue = InconsistencyIssue(
            issue_type="missing_examples",
            severity="low",
            description="Function lacks usage examples",
            suggestion="Add examples",
            line_number=5,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify example generation
        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.EXAMPLE_UPDATE
        assert suggestion.confidence >= 0.7

        suggested_text = suggestion.suggested_text

        # Should contain the function call with realistic parameters
        assert "calculate_distance(" in suggested_text
        assert "x1=" in suggested_text
        assert "y1=" in suggested_text
        assert "x2=" in suggested_text
        assert "y2=" in suggested_text

        # Should be properly formatted for docstring
        assert "Examples:" in suggested_text or "Example:" in suggested_text

    def test_complete_workflow_file_processing_function(self, generator):
        """Test complete workflow for file processing function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "process_csv_file"
        function.line_number = 8

        # Parameters with meaningful names
        filepath_param = Mock()
        filepath_param.name = "filepath"
        filepath_param.type_annotation = "str"
        filepath_param.is_required = True
        filepath_param.default_value = None

        encoding_param = Mock()
        encoding_param.name = "encoding"
        encoding_param.type_annotation = "str"
        encoding_param.is_required = False
        encoding_param.default_value = "'utf-8'"

        function.signature.parameters = [filepath_param, encoding_param]
        function.source_code = """
def process_csv_file(filepath, encoding='utf-8'):
    import csv
    with open(filepath, 'r', encoding=encoding) as f:
        return list(csv.reader(f))
"""

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Process CSV file and return rows"
        docstring.examples = []  # Missing examples
        docstring.raw_text = '"""Process CSV file and return rows."""'

        issue = InconsistencyIssue(
            issue_type="missing_examples",
            severity="low",
            description="Add usage examples",
            suggestion="",
            line_number=8,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify file-specific example generation
        assert isinstance(suggestion, Suggestion)

        suggested_text = suggestion.suggested_text

        # Should use appropriate file-related values
        assert "process_csv_file(" in suggested_text
        assert "filepath=" in suggested_text

        # Should use realistic file path
        assert any(
            path_indicator in suggested_text
            for path_indicator in [".csv", ".txt", "/path/", "file"]
        )

    def test_async_function_example_generation(self, generator):
        """Test example generation for async functions."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "fetch_user_data"
        function.line_number = 3

        user_id_param = Mock()
        user_id_param.name = "user_id"
        user_id_param.type_annotation = "int"
        user_id_param.is_required = True
        user_id_param.default_value = None

        function.signature.parameters = [user_id_param]
        function.source_code = """
async def fetch_user_data(user_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/users/{user_id}") as response:
            return await response.json()
"""

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Fetch user data asynchronously"
        docstring.examples = []
        docstring.raw_text = '"""Fetch user data asynchronously."""'

        issue = InconsistencyIssue(
            issue_type="missing_examples",
            severity="low",
            description="Add async examples",
            suggestion="",
            line_number=3,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify async-specific handling
        suggested_text = suggestion.suggested_text

        # Should include await keyword
        assert "await " in suggested_text
        assert "fetch_user_data(" in suggested_text

        # Should mention async context
        assert any(
            async_indicator in suggested_text.lower()
            for async_indicator in ["async", "await"]
        )
