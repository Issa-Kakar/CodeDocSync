"""
Tests for prompt template functionality.

This module tests the prompt templates for all analysis types, ensuring they:
- Generate valid prompts with proper formatting
- Include all required sections and examples
- Validate input parameters correctly
- Produce prompts within token limits
- Map LLM issue types correctly
"""

import pytest

from codedocsync.analyzer.prompt_templates import (
    LLM_ISSUE_TYPE_MAPPING,
    PROMPT_TEMPLATES,
    format_prompt,
    get_available_analysis_types,
    get_prompt_template,
    map_llm_issue_type,
    validate_llm_response,
)


class TestPromptTemplateRetrieval:
    """Test prompt template retrieval functionality."""

    def test_get_prompt_template_valid_types(self):
        """Test getting prompt templates for all valid analysis types."""
        for analysis_type in get_available_analysis_types():
            template = get_prompt_template(analysis_type)
            assert isinstance(template, str)
            assert len(template) > 100  # Templates should be substantial
            assert "Return JSON in this EXACT format" in template

    def test_get_prompt_template_invalid_type(self):
        """Test error handling for invalid analysis types."""
        with pytest.raises(ValueError, match="Unknown analysis type"):
            get_prompt_template("invalid_type")

    def test_get_available_analysis_types(self):
        """Test that all expected analysis types are available."""
        types = get_available_analysis_types()
        expected_types = {
            "behavior_analysis",
            "example_validation",
            "edge_case_analysis",
            "version_analysis",
            "type_consistency",
            "performance_analysis",
        }
        assert set(types) == expected_types

    def test_prompt_templates_constant(self):
        """Test that PROMPT_TEMPLATES constant has all expected templates."""
        assert len(PROMPT_TEMPLATES) == 6
        for analysis_type in get_available_analysis_types():
            assert analysis_type in PROMPT_TEMPLATES


class TestPromptFormatting:
    """Test prompt formatting with real data."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for prompt formatting."""
        return {
            "signature": "def process_data(data: List[Dict], validate: bool = True) -> Dict[str, Any]:",
            "source_code": '''def process_data(data: List[Dict], validate: bool = True) -> Dict[str, Any]:
    """Process input data with validation."""
    if not data:
        return {"count": 0, "items": []}

    if validate:
        for item in data:
            if "id" not in item:
                raise ValueError("Missing id field")

    return {"count": len(data), "items": data}''',
            "docstring": """Process input data with validation.

Args:
    data: List of dictionaries to process
    validate: Whether to validate input data

Returns:
    Dictionary with count and processed items

Raises:
    ValueError: If validation fails""",
            "rule_issues": "No rule issues found",
        }

    def test_format_prompt_all_types(self, sample_data):
        """Test formatting prompts for all analysis types."""
        for analysis_type in get_available_analysis_types():
            prompt = format_prompt(analysis_type=analysis_type, **sample_data)

            # Basic validation
            assert isinstance(prompt, str)
            assert len(prompt) > 500  # Should be substantial

            # Check that all variables were substituted
            assert "{signature}" not in prompt
            assert "{source_code}" not in prompt
            assert "{docstring}" not in prompt
            assert "{rule_issues}" not in prompt

            # Check for key content
            assert sample_data["signature"] in prompt
            assert "process_data" in prompt
            assert "Return JSON in this EXACT format" in prompt

    def test_format_prompt_missing_required_field(self, sample_data):
        """Test error handling for missing required formatting variables."""
        incomplete_data = {k: v for k, v in sample_data.items() if k != "signature"}

        with pytest.raises(KeyError, match="Missing required formatting variable"):
            format_prompt("behavior_analysis", **incomplete_data)

    def test_format_prompt_with_optional_rule_issues(self, sample_data):
        """Test prompt formatting with and without rule issues."""
        # Without rule_issues
        data_without_rules = {
            k: v for k, v in sample_data.items() if k != "rule_issues"
        }
        prompt1 = format_prompt("behavior_analysis", **data_without_rules)

        # With rule_issues
        prompt2 = format_prompt("behavior_analysis", **sample_data)

        assert len(prompt2) >= len(prompt1)  # Should be same or longer
        assert sample_data["rule_issues"] in prompt2

    def test_format_prompt_with_additional_kwargs(self, sample_data):
        """Test prompt formatting with additional keyword arguments."""
        extra_data = {"custom_context": "Additional context information"}

        # For templates that might use additional variables
        prompt = format_prompt("behavior_analysis", **sample_data, **extra_data)
        assert isinstance(prompt, str)

    @pytest.mark.parametrize(
        "analysis_type",
        [
            "behavior_analysis",
            "example_validation",
            "edge_case_analysis",
            "version_analysis",
            "type_consistency",
            "performance_analysis",
        ],
    )
    def test_prompt_contains_required_sections(self, analysis_type, sample_data):
        """Test that prompts contain all required sections."""
        prompt = format_prompt(analysis_type, **sample_data)

        # All prompts should have these sections
        required_sections = [
            "FUNCTION SIGNATURE:",
            "CURRENT DOCUMENTATION:",
            "FUNCTION IMPLEMENTATION:",
            "Return JSON in this EXACT format:",
            "issues",
            "confidence",
            "description",
            "suggestion",
        ]

        for section in required_sections:
            assert section in prompt, f"Missing section '{section}' in {analysis_type}"


class TestPromptContent:
    """Test specific content requirements for each prompt type."""

    def test_behavior_analysis_prompt_content(self):
        """Test behavior analysis prompt has specific requirements."""
        template = get_prompt_template("behavior_analysis")

        required_content = [
            "behavior described in docstring vs actual implementation",
            "Edge cases handled in code but not documented",
            "Error conditions",
            "Side effects",
            "Return value descriptions",
            "behavior_mismatch",
        ]

        for content in required_content:
            assert content in template

    def test_example_validation_prompt_content(self):
        """Test example validation prompt has specific requirements."""
        template = get_prompt_template("example_validation")

        required_content = [
            "code examples in the documentation",
            "Syntax errors in examples",
            "Examples that would raise exceptions",
            "parameter names/types",
            "example_invalid",
        ]

        for content in required_content:
            assert content in template

    def test_edge_case_analysis_prompt_content(self):
        """Test edge case analysis prompt has specific requirements."""
        template = get_prompt_template("edge_case_analysis")

        required_content = [
            "edge cases handled in the implementation",
            "Input validation",
            "Boundary conditions",
            "empty lists",
            "missing_edge_case",
        ]

        for content in required_content:
            assert content in template

    def test_version_analysis_prompt_content(self):
        """Test version analysis prompt has specific requirements."""
        template = get_prompt_template("version_analysis")

        required_content = [
            "version information",
            "deprecation warnings",
            "Outdated version numbers",
            "Deprecated parameters",
            "version_info_outdated",
        ]

        for content in required_content:
            assert content in template

    def test_type_consistency_prompt_content(self):
        """Test type consistency prompt has specific requirements."""
        template = get_prompt_template("type_consistency")

        required_content = [
            "type inconsistencies",
            "type hints",
            "Generic types",
            "Union types",
            "type_documentation_mismatch",
        ]

        for content in required_content:
            assert content in template

    def test_performance_analysis_prompt_content(self):
        """Test performance analysis prompt has specific requirements."""
        template = get_prompt_template("performance_analysis")

        required_content = [
            "performance characteristics",
            "Time complexity",
            "Memory usage",
            "O(n)",
            "performance_mismatch",
        ]

        for content in required_content:
            assert content in template


class TestLLMResponseValidation:
    """Test LLM response validation functionality."""

    def test_validate_llm_response_valid(self):
        """Test validation of valid LLM responses."""
        valid_response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Function behavior doesn't match docs",
                    "suggestion": "Update documentation to reflect actual behavior",
                    "confidence": 0.85,
                    "line_number": 5,
                    "details": {"issue": "mismatch"},
                }
            ],
            "confidence": 0.90,
            "analysis_notes": "Analyzed behavior consistency",
        }

        assert validate_llm_response(valid_response) is True

    def test_validate_llm_response_missing_issues(self):
        """Test validation fails when issues field is missing."""
        invalid_response = {"confidence": 0.90, "analysis_notes": "Test"}

        assert validate_llm_response(invalid_response) is False

    def test_validate_llm_response_invalid_issues_type(self):
        """Test validation fails when issues is not a list."""
        invalid_response = {"issues": "not a list", "confidence": 0.90}

        assert validate_llm_response(invalid_response) is False

    def test_validate_llm_response_missing_confidence(self):
        """Test validation fails when confidence field is missing."""
        invalid_response = {"issues": []}

        assert validate_llm_response(invalid_response) is False

    def test_validate_llm_response_invalid_confidence_type(self):
        """Test validation fails when confidence is not a number."""
        invalid_response = {"issues": [], "confidence": "high"}

        assert validate_llm_response(invalid_response) is False

    def test_validate_llm_response_invalid_issue_structure(self):
        """Test validation fails for issues with missing required fields."""
        invalid_response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Test",
                    # Missing required fields
                }
            ],
            "confidence": 0.90,
        }

        assert validate_llm_response(invalid_response) is False

    def test_validate_llm_response_invalid_issue_confidence(self):
        """Test validation fails for issues with invalid confidence values."""
        invalid_response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Test",
                    "suggestion": "Fix it",
                    "confidence": 1.5,  # Invalid: > 1.0
                    "line_number": 5,
                }
            ],
            "confidence": 0.90,
        }

        assert validate_llm_response(invalid_response) is False

    def test_validate_llm_response_empty_issues(self):
        """Test validation passes for empty issues list."""
        valid_response = {
            "issues": [],
            "confidence": 0.95,
            "analysis_notes": "No issues found",
        }

        assert validate_llm_response(valid_response) is True


class TestIssueTypeMapping:
    """Test LLM issue type mapping functionality."""

    def test_map_llm_issue_type_known_types(self):
        """Test mapping of known LLM issue types."""
        for llm_type, expected_type in LLM_ISSUE_TYPE_MAPPING.items():
            result = map_llm_issue_type(llm_type)
            assert result == expected_type

    def test_map_llm_issue_type_unknown_type(self):
        """Test mapping of unknown LLM issue types defaults correctly."""
        unknown_type = "some_unknown_issue_type"
        result = map_llm_issue_type(unknown_type)
        assert result == "description_outdated"  # Default mapping

    def test_llm_issue_type_mapping_completeness(self):
        """Test that all LLM issue types mentioned in prompts are mapped."""
        # Get all issue types mentioned in prompt templates
        mentioned_types = set()
        for template in PROMPT_TEMPLATES.values():
            # Extract issue types from JSON examples in templates
            import re

            type_matches = re.findall(r'"type":\s*"([^"]+)"', template)
            mentioned_types.update(type_matches)

        # All mentioned types should be in our mapping
        for issue_type in mentioned_types:
            mapped_type = map_llm_issue_type(issue_type)
            assert mapped_type in {
                "parameter_name_mismatch",
                "parameter_missing",
                "parameter_type_mismatch",
                "return_type_mismatch",
                "missing_raises",
                "parameter_order_different",
                "description_outdated",
                "example_invalid",
            }


class TestPromptTokenEstimation:
    """Test token estimation for prompts."""

    @pytest.fixture
    def realistic_data(self):
        """Realistic function data for token estimation."""
        return {
            "signature": "def complex_function(data: Dict[str, List[Any]], options: Optional[ProcessingOptions] = None, validate: bool = True, timeout: float = 30.0) -> ProcessingResult:",
            "source_code": '''def complex_function(data: Dict[str, List[Any]], options: Optional[ProcessingOptions] = None, validate: bool = True, timeout: float = 30.0) -> ProcessingResult:
    """
    Complex function that processes data with various options.

    This function performs multi-step data processing including validation,
    transformation, and result aggregation. It supports various processing
    options and has comprehensive error handling.

    Args:
        data: Dictionary containing lists of data items to process
        options: Optional processing configuration object
        validate: Whether to perform input validation
        timeout: Maximum processing time in seconds

    Returns:
        ProcessingResult object with processed data and metadata

    Raises:
        ValidationError: If input validation fails
        TimeoutError: If processing exceeds timeout
        ProcessingError: If data processing fails

    Example:
        >>> data = {"items": [1, 2, 3], "metadata": ["a", "b", "c"]}
        >>> result = complex_function(data, validate=True)
        >>> print(result.success)
        True
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    if validate:
        for key, values in data.items():
            if not isinstance(values, list):
                raise ValidationError(f"Value for key '{key}' must be a list")

    start_time = time.time()

    try:
        # Processing logic here
        processed_items = []
        for key, values in data.items():
            for value in values:
                processed_item = process_single_item(value, options)
                processed_items.append(processed_item)

                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Processing exceeded {timeout} seconds")

        result = ProcessingResult(
            items=processed_items,
            count=len(processed_items),
            processing_time=time.time() - start_time,
            success=True
        )

        return result

    except Exception as e:
        if isinstance(e, (ValidationError, TimeoutError)):
            raise
        raise ProcessingError(f"Processing failed: {e}")''',
            "docstring": """Complex function that processes data with various options.

This function performs multi-step data processing including validation,
transformation, and result aggregation. It supports various processing
options and has comprehensive error handling.

Args:
    data: Dictionary containing lists of data items to process
    options: Optional processing configuration object
    validate: Whether to perform input validation
    timeout: Maximum processing time in seconds

Returns:
    ProcessingResult object with processed data and metadata

Raises:
    ValidationError: If input validation fails
    TimeoutError: If processing exceeds timeout
    ProcessingError: If data processing fails

Example:
    >>> data = {"items": [1, 2, 3], "metadata": ["a", "b", "c"]}
    >>> result = complex_function(data, validate=True)
    >>> print(result.success)
    True""",
            "rule_issues": "Parameter type mismatch: documented as Dict[str, List[Any]] but implementation suggests more restrictive types",
        }

    def test_prompt_token_estimation(self, realistic_data):
        """Test that prompts stay within reasonable token limits."""
        for analysis_type in get_available_analysis_types():
            prompt = format_prompt(analysis_type, **realistic_data)

            # Rough token estimation: ~4 characters per token
            estimated_tokens = len(prompt) // 4

            # Should be under 3000 tokens for most models
            assert (
                estimated_tokens < 3000
            ), f"{analysis_type} prompt too long: ~{estimated_tokens} tokens"

            # Should be substantial enough to be useful
            assert (
                estimated_tokens > 200
            ), f"{analysis_type} prompt too short: ~{estimated_tokens} tokens"

    def test_prompt_length_consistency(self):
        """Test that all prompts are of reasonable and consistent length."""
        sample_data = {
            "signature": "def test_func(x: int) -> str:",
            "source_code": "def test_func(x: int) -> str:\n    return str(x)",
            "docstring": "Convert integer to string.",
            "rule_issues": "No issues",
        }

        lengths = {}
        for analysis_type in get_available_analysis_types():
            prompt = format_prompt(analysis_type, **sample_data)
            lengths[analysis_type] = len(prompt)

        # All prompts should be substantial
        for analysis_type, length in lengths.items():
            assert length > 500, f"{analysis_type} prompt too short: {length} chars"

        # Variation should be reasonable (no prompt should be 10x longer than another)
        min_length = min(lengths.values())
        max_length = max(lengths.values())
        assert (
            max_length / min_length < 5
        ), f"Prompt length variation too large: {min_length} to {max_length}"


class TestPromptExamples:
    """Test that prompts include proper examples."""

    def test_all_prompts_have_examples(self):
        """Test that all prompt templates include example responses."""
        for analysis_type in get_available_analysis_types():
            template = get_prompt_template(analysis_type)

            # Should have example section
            assert "EXAMPLE" in template.upper(), f"{analysis_type} missing example"

            # Should have JSON structure in example
            assert (
                '"issues"' in template
            ), f"{analysis_type} missing JSON structure in example"
            assert (
                '"confidence"' in template
            ), f"{analysis_type} missing confidence in example"

    def test_examples_have_valid_json_structure(self):
        """Test that examples in prompts contain valid JSON structure."""
        for analysis_type in get_available_analysis_types():
            template = get_prompt_template(analysis_type)

            # Extract JSON from examples
            import re

            json_matches = re.findall(r'\{[^}]*"issues"[^}]*\}', template, re.DOTALL)

            assert len(json_matches) > 0, f"{analysis_type} has no JSON examples"

            # Each JSON example should be parseable (at least structurally)
            for json_str in json_matches:
                # Basic validation - should have proper quotes and braces
                assert json_str.count("{") == json_str.count(
                    "}"
                ), f"Unmatched braces in {analysis_type}"
                assert (
                    '"issues"' in json_str
                ), f"Missing issues field in {analysis_type} example"


@pytest.mark.integration
class TestPromptIntegration:
    """Integration tests for prompt templates with realistic scenarios."""

    def test_prompt_generation_end_to_end(self):
        """Test complete prompt generation workflow."""
        # This would test with actual ParsedFunction and ParsedDocstring objects
        # For now, we'll test the string-based interface

        sample_data = {
            "signature": "def calculate_statistics(numbers: List[float], percentiles: List[float] = [0.25, 0.5, 0.75]) -> StatisticsResult:",
            "source_code": '''def calculate_statistics(numbers: List[float], percentiles: List[float] = [0.25, 0.5, 0.75]) -> StatisticsResult:
    """Calculate comprehensive statistics for a list of numbers."""
    if not numbers:
        return StatisticsResult(count=0, mean=None, std=None, percentiles={})

    import statistics
    result = StatisticsResult(
        count=len(numbers),
        mean=statistics.mean(numbers),
        std=statistics.stdev(numbers) if len(numbers) > 1 else 0,
        percentiles={p: statistics.quantiles(numbers, n=int(1/p)) for p in percentiles}
    )
    return result''',
            "docstring": """Calculate comprehensive statistics for a list of numbers.

Args:
    numbers: List of numeric values to analyze
    percentiles: List of percentile values to calculate (default: quartiles)

Returns:
    StatisticsResult with mean, standard deviation, and percentiles

Raises:
    ValueError: If percentiles are not between 0 and 1""",
            "rule_issues": "Missing raises documentation for empty list handling",
        }

        # Test all analysis types work together
        prompts = {}
        for analysis_type in get_available_analysis_types():
            prompt = format_prompt(analysis_type, **sample_data)
            prompts[analysis_type] = prompt

            # Basic validation that prompt makes sense
            assert "calculate_statistics" in prompt
            assert "StatisticsResult" in prompt
            assert sample_data["rule_issues"] in prompt

        # All prompts should be different
        prompt_texts = list(prompts.values())
        for i, prompt1 in enumerate(prompt_texts):
            for j, prompt2 in enumerate(prompt_texts[i + 1 :], i + 1):
                # Should have some differences (not identical)
                similarity = len(set(prompt1.split()) & set(prompt2.split())) / len(
                    set(prompt1.split()) | set(prompt2.split())
                )
                assert similarity < 0.9, f"Prompts {i} and {j} are too similar"
