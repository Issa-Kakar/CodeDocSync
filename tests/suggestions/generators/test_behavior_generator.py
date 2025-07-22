"""
Tests for the Behavioral Description Generator.

Tests cover behavioral pattern analysis, description enhancement, and suggestion
generation for function behavior documentation.
"""

from unittest.mock import Mock

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.generators.behavior_generator import (
    BehaviorAnalyzer,
    BehaviorPattern,
    BehaviorSuggestionGenerator,
)
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionContext,
    SuggestionType,
)


class TestBehaviorAnalyzer:
    """Test the behavior analyzer."""

    def test_analyze_control_flow_loops(self) -> None:
        """Test analyzing control flow with loops."""
        source_code = """
def test_func(items) -> None:
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
"""
        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, "test_func")

        pattern_types = [p.pattern_type for p in patterns]
        assert "iteration" in pattern_types
        assert "conditional" in pattern_types

    def test_analyze_data_operations(self) -> None:
        """Test analyzing data manipulation patterns."""
        source_code = """
def test_func(data) -> None:
    processed = [x * 2 for x in data]
    result = {}
    for item in processed:
        result[str(item)] = item
    return result
"""
        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, "test_func")

        pattern_types = [p.pattern_type for p in patterns]
        assert "data_transformation" in pattern_types
        assert "data_creation" in pattern_types

    def test_analyze_side_effects(self) -> None:
        """Test analyzing side effects."""
        source_code = """
def test_func(filename, data) -> None:
    import logging
    logging.info("Processing file")

    with open(filename, 'w') as f:
        f.write(str(data))

    print("Done")
"""
        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, "test_func")

        pattern_types = [p.pattern_type for p in patterns]
        assert "file_io" in pattern_types
        assert "logging" in pattern_types

    def test_analyze_error_handling(self) -> None:
        """Test analyzing error handling patterns."""
        source_code = """
def test_func(value) -> None:
    assert isinstance(value, str), "Value must be string"

    if not value:
        raise ValueError("Value cannot be empty")

    try:
        result = risky_operation(value)
    except Exception as e:
        log_error(e)
        return None

    return result
"""
        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, "test_func")

        pattern_types = [p.pattern_type for p in patterns]
        assert "error_handling" in pattern_types
        assert "input_validation" in pattern_types
        assert "assertion_checks" in pattern_types

    def test_analyze_performance_characteristics(self) -> None:
        """Test analyzing performance-related patterns."""
        source_code = """
def test_func(matrix) -> None:
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(matrix[i][j] * 2)
        result.append(row)
    return result
"""
        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, "test_func")

        pattern_types = [p.pattern_type for p in patterns]
        assert "nested_iteration" in pattern_types

        # Check complexity scoring
        nested_patterns = [p for p in patterns if p.pattern_type == "nested_iteration"]
        assert len(nested_patterns) >= 1
        assert nested_patterns[0].details is not None
        assert "complexity_score" in nested_patterns[0].details

    def test_analyze_recursive_function(self) -> None:
        """Test analyzing recursive functions."""
        source_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, "fibonacci")

        pattern_types = [p.pattern_type for p in patterns]
        assert "recursive" in pattern_types

    def test_analyze_function_purpose_by_name(self) -> None:
        """Test analyzing function purpose based on name."""
        test_cases = [
            ("get_user_data", "Retrieves and returns data"),
            ("calculate_total", "Performs mathematical calculations"),
            ("validate_input", "Validates data against specified criteria"),
            ("create_user", "Creates new objects or data structures"),
            ("process_data", "Processes input data through a series of operations"),
        ]

        analyzer = BehaviorAnalyzer()

        for func_name, expected_pattern in test_cases:
            source_code = f"""
def {func_name}():
    pass
"""
            patterns = analyzer.analyze_behavior(source_code, func_name)

            purpose_patterns = [p for p in patterns if p.pattern_type == "purpose"]
            assert len(purpose_patterns) >= 1
            assert expected_pattern in purpose_patterns[0].description

    def test_looks_like_validation(self) -> None:
        """Test validation pattern detection."""
        analyzer = BehaviorAnalyzer()

        # This is a bit tricky to test directly since _looks_like_validation
        # expects an AST node, but we can test it indirectly through the
        # full analysis
        source_code = """
def test_func(value) -> None:
    if not isinstance(value, str):
        return False
    if len(value) == 0:
        return False
    return True
"""
        patterns = analyzer.analyze_behavior(source_code, "test_func")
        pattern_types = [p.pattern_type for p in patterns]
        assert "input_validation" in pattern_types

    def test_syntax_error_handling(self) -> None:
        """Test handling syntax errors gracefully."""
        source_code = """
def test_func(
    # Incomplete function
"""
        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, "test_func")

        # Should return at least one pattern indicating analysis failure
        assert len(patterns) >= 1
        unknown_patterns = [p for p in patterns if p.pattern_type == "unknown"]
        assert len(unknown_patterns) >= 1
        assert unknown_patterns[0].confidence <= 0.2


class TestBehaviorSuggestionGenerator:
    """Test the behavior suggestion generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(
            default_style="google",
            max_line_length=88,
        )

    @pytest.fixture
    def generator(self, config):
        """Create behavior suggestion generator."""
        return BehaviorSuggestionGenerator(config)

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "process_data"
        function.line_number = 10
        function.source_code = """
def process_data(items):
    result = []
    for item in items:
        if item.is_valid():
            processed = transform(item)
            result.append(processed)
    return result
"""
        return function

    @pytest.fixture
    def mock_docstring(self):
        """Create mock docstring."""
        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Process data"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Process data."""'
        return docstring

    @pytest.fixture
    def mock_issue(self):
        """Create mock issue."""
        return InconsistencyIssue(
            issue_type="description_vague",
            severity="medium",
            description="Description is too vague",
            suggestion="Improve description",
            line_number=10,
        )

    def test_improve_vague_description(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test improving vague description."""
        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._improve_vague_description(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE
        assert suggestion.confidence >= 0.5

        # Should have enhanced description
        suggested_text = suggestion.suggested_text
        assert len(suggested_text) > len(mock_docstring.raw_text)

    def test_improve_outdated_description(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test improving outdated description."""
        mock_issue.issue_type = "description_outdated"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._improve_outdated_description(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE

    def test_add_behavior_description(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test adding missing behavior description."""
        mock_issue.issue_type = "missing_behavior_description"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_behavior_description(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE

    def test_add_side_effects_documentation(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test adding side effects documentation."""
        # Function with side effects
        mock_function.source_code = """
def process_data(items, log_file):
    import logging
    logging.basicConfig(filename=log_file)

    with open("output.txt", "w") as f:
        for item in items:
            processed = item.transform()
            f.write(str(processed))
            logging.info(f"Processed {item}")

    print("Processing complete")
"""
        mock_issue.issue_type = "side_effects_undocumented"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_side_effects_documentation(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE

        # Should mention side effects
        suggested_text = suggestion.suggested_text.lower()
        assert (
            "side effects" in suggested_text
            or "file" in suggested_text
            or "log" in suggested_text
        )

    def test_generate_enhanced_description(self, generator) -> None:
        """Test generation of enhanced descriptions."""
        # Mock patterns for testing
        patterns = [
            BehaviorPattern("purpose", "Processes input data", 0.9),
            BehaviorPattern("iteration", "Iterates through data using for loops", 0.8),
            BehaviorPattern("conditional", "Applies conditional logic", 0.7),
            BehaviorPattern("file_io", "Performs file input/output operations", 0.9),
        ]

        mock_docstring = Mock()
        mock_docstring.summary = "Process data"
        mock_docstring.description = None

        result = generator._generate_enhanced_description(
            patterns, "process_data", mock_docstring, focus_side_effects=False
        )

        assert isinstance(result, str)
        assert len(result) > 20  # Should be substantial
        assert "processes input data" in result.lower()

    def test_generate_enhanced_description_with_side_effects_focus(
        self, generator
    ) -> None:
        """Test enhanced description with side effects focus."""
        patterns = [
            BehaviorPattern("purpose", "Processes input data", 0.9),
            BehaviorPattern("file_io", "Performs file input/output operations", 0.9),
            BehaviorPattern("logging", "Logs information for debugging", 0.8),
        ]

        mock_docstring = Mock()
        mock_docstring.summary = "Process data"

        result = generator._generate_enhanced_description(
            patterns, "process_data", mock_docstring, focus_side_effects=True
        )

        assert "side effects" in result.lower()
        assert any(effect in result.lower() for effect in ["file", "log"])

    def test_generate_basic_purpose(self, generator) -> None:
        """Test generation of basic purpose from function names."""
        test_cases = [
            ("get_user_data", "user data"),
            ("create_new_user", "new user"),
            ("calculate_total_function", "total"),
            ("process_items", "items"),
        ]

        for func_name, expected_words in test_cases:
            result = generator._generate_basic_purpose(func_name)
            assert expected_words in result.lower()
            assert "handles" in result.lower()

    def test_no_source_code_fallback(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test fallback when source code is not available."""
        mock_function.source_code = ""

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._improve_vague_description(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence == 0.1  # Fallback suggestion

    def test_no_patterns_detected(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test handling when no behavioral patterns are detected."""
        # Very simple function that might not trigger pattern detection
        mock_function.source_code = """
def simple_func():
    return 42
"""

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._improve_vague_description(context)

        # Should be a fallback suggestion
        assert suggestion.confidence == 0.1

    def test_unknown_issue_type(self, generator, mock_function, mock_docstring) -> None:
        """Test handling unknown issue types."""
        unknown_issue = InconsistencyIssue(
            issue_type="unknown_behavior_issue",
            severity="medium",
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
        assert "Unknown behavior issue type" in suggestion.description


class TestBehaviorGeneratorIntegration:
    """Integration tests for behavior generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(default_style="google")

    @pytest.fixture
    def generator(self, config):
        """Create behavior suggestion generator."""
        return BehaviorSuggestionGenerator(config)

    def test_complete_workflow_data_processing(self, generator) -> None:
        """Test complete workflow for data processing function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "filter_and_transform_data"
        function.line_number = 5
        function.source_code = '''
def filter_and_transform_data(raw_data, threshold=0.5):
    """Filter data."""
    filtered = []

    # Validate input
    if not isinstance(raw_data, list):
        raise TypeError("Expected list")

    # Process each item
    for item in raw_data:
        if hasattr(item, 'score') and item.score > threshold:
            # Transform the item
            transformed = {
                'id': item.id,
                'value': item.value * 2,
                'normalized_score': item.score / 100
            }
            filtered.append(transformed)

    # Log results
    import logging
    logging.info(f"Processed {len(filtered)} items")

    return filtered
'''

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Filter data"  # Vague
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Filter data."""'

        issue = InconsistencyIssue(
            issue_type="description_vague",
            severity="medium",
            description="Description is too vague",
            suggestion="Improve description clarity",
            line_number=5,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify comprehensive behavior analysis
        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.DESCRIPTION_UPDATE
        assert suggestion.confidence >= 0.6

        suggested_text = suggestion.suggested_text.lower()

        # Should mention key behaviors detected
        assert any(
            word in suggested_text
            for word in ["filter", "transform", "process", "iterate", "conditional"]
        )

        # Should be more detailed than original
        assert len(suggested_text) > len(docstring.raw_text)

    def test_complete_workflow_file_operations(self, generator) -> None:
        """Test complete workflow for file operations with side effects."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "backup_and_process_file"
        function.line_number = 8
        function.source_code = '''
def backup_and_process_file(filepath, backup_dir):
    """Process file."""
    import os
    import shutil
    import logging

    # Create backup
    backup_path = os.path.join(backup_dir, os.path.basename(filepath))
    shutil.copy2(filepath, backup_path)
    logging.info(f"Created backup: {backup_path}")

    # Process original file
    with open(filepath, 'r') as f:
        content = f.read()

    processed_content = content.upper()

    with open(filepath, 'w') as f:
        f.write(processed_content)

    print(f"Processed file: {filepath}")
    return backup_path
'''

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Process file"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Process file."""'

        issue = InconsistencyIssue(
            issue_type="side_effects_undocumented",
            severity="high",
            description="Side effects not documented",
            suggestion="Document side effects",
            line_number=8,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify side effects documentation
        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence >= 0.6

        suggested_text = suggestion.suggested_text.lower()

        # Should mention side effects
        assert "side effects" in suggested_text

        # Should mention specific side effects detected
        assert any(
            effect in suggested_text for effect in ["file", "backup", "log", "modif"]
        )

    def test_performance_pattern_detection(self, generator) -> None:
        """Test detection of performance-related patterns."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "matrix_multiply"
        function.line_number = 3
        function.source_code = '''
def matrix_multiply(a, b):
    """Multiply matrices."""
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            sum_val = 0
            for k in range(len(b)):
                sum_val += a[i][k] * b[k][j]
            row.append(sum_val)
        result.append(row)
    return result
'''

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Multiply matrices"
        docstring.description = None
        docstring.raw_text = '"""Multiply matrices."""'

        issue = InconsistencyIssue(
            issue_type="description_vague",
            severity="medium",
            description="Should mention performance characteristics",
            suggestion="",
            line_number=3,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Should detect nested loops and mention performance
        suggested_text = suggestion.suggested_text.lower()
        assert any(
            perf_word in suggested_text
            for perf_word in ["nested", "loop", "performance"]
        )
