"""Unit tests for curated examples expansion."""

import json
from pathlib import Path

import pytest


class TestCuratedExamplesExpansion:
    """Test the expanded curated examples corpus."""

    @pytest.fixture
    def curated_examples(self):
        """Load curated examples from JSON file."""
        data_path = (
            Path(__file__).parent.parent.parent / "data" / "curated_examples.json"
        )
        with open(data_path) as f:
            return json.load(f)

    def test_json_structure_validity(self, curated_examples):
        """Test that JSON structure is valid and contains expected fields."""
        assert "examples" in curated_examples
        assert isinstance(curated_examples["examples"], list)
        assert (
            len(curated_examples["examples"]) >= 75
        )  # Should have at least 75 examples

        required_fields = {
            "function_name",
            "module_path",
            "function_signature",
            "docstring_format",
            "docstring_content",
            "has_params",
            "has_returns",
            "has_examples",
            "complexity_score",
            "quality_score",
            "source",
        }

        for i, example in enumerate(curated_examples["examples"]):
            # Check all required fields are present
            missing_fields = required_fields - set(example.keys())
            assert not missing_fields, f"Example {i} missing fields: {missing_fields}"

            # Validate field types
            assert isinstance(example["function_name"], str)
            assert isinstance(example["module_path"], str)
            assert isinstance(example["function_signature"], str)
            assert isinstance(example["docstring_format"], str)
            assert isinstance(example["docstring_content"], str)
            assert isinstance(example["has_params"], bool)
            assert isinstance(example["has_returns"], bool)
            assert isinstance(example["has_examples"], bool)
            assert isinstance(example["complexity_score"], int)
            assert isinstance(example["quality_score"], int)
            assert isinstance(example["source"], str)

            # Validate docstring format
            assert example["docstring_format"] in ["google", "numpy", "sphinx"]

            # Validate scores
            assert 1 <= example["complexity_score"] <= 5
            assert 1 <= example["quality_score"] <= 5

            # Curated examples should have quality score of 5
            if example["source"] == "curated":
                assert example["quality_score"] == 5

    def test_no_duplicate_function_names(self, curated_examples):
        """Test that there are no duplicate function names."""
        function_names = [ex["function_name"] for ex in curated_examples["examples"]]
        assert len(function_names) == len(
            set(function_names)
        ), "Duplicate function names found"

    def test_category_distribution(self, curated_examples):
        """Test that examples are well distributed across categories."""
        categories = {}
        for example in curated_examples["examples"]:
            if "category" in example:
                category = example["category"]
                categories[category] = categories.get(category, 0) + 1

        # Verify expected categories have adequate representation
        expected_categories = {
            "async_patterns": 20,  # Should have at least 20 async examples
            "rest_api_patterns": 15,  # At least 15 REST API examples
            "data_science_patterns": 10,  # At least 10 data science examples
            "advanced_patterns": 10,  # At least 10 decorator/context manager examples
        }

        for category, min_count in expected_categories.items():
            if category in categories:
                assert (
                    categories[category] >= min_count
                ), f"Category {category} has {categories.get(category, 0)} examples, expected at least {min_count}"

    def test_complexity_score_distribution(self, curated_examples):
        """Test that complexity scores are well distributed."""
        complexity_scores = [
            ex["complexity_score"] for ex in curated_examples["examples"]
        ]

        # Should have examples across different complexity levels
        unique_scores = set(complexity_scores)
        assert len(unique_scores) >= 3, "Complexity scores not diverse enough"

        # Verify reasonable distribution (not all same score)
        score_counts = {
            score: complexity_scores.count(score) for score in unique_scores
        }
        max_count = max(score_counts.values())
        assert (
            max_count < len(complexity_scores) * 0.6
        ), "Too many examples with same complexity score"

    def test_docstring_format_diversity(self, curated_examples):
        """Test that examples use diverse docstring formats."""
        formats = [ex["docstring_format"] for ex in curated_examples["examples"]]

        # Count format usage
        format_counts = {
            "google": formats.count("google"),
            "numpy": formats.count("numpy"),
            "sphinx": formats.count("sphinx"),
        }

        # Each format should have at least some representation
        for format_name, count in format_counts.items():
            assert count > 0, f"No examples using {format_name} format"

        # No single format should dominate completely
        max_format_count = max(format_counts.values())
        assert (
            max_format_count < len(formats) * 0.8
        ), "Docstring formats not diverse enough"

    def test_quality_scores_for_curated_source(self, curated_examples):
        """Test that all curated source examples have quality score of 5."""
        curated_only = [
            ex for ex in curated_examples["examples"] if ex["source"] == "curated"
        ]
        assert len(curated_only) >= 75, "Should have at least 75 curated examples"

        for example in curated_only:
            assert (
                example["quality_score"] == 5
            ), f"Curated example '{example['function_name']}' should have quality_score=5"

    def test_example_content_not_empty(self, curated_examples):
        """Test that example content fields are not empty."""
        for i, example in enumerate(curated_examples["examples"]):
            assert example[
                "function_name"
            ].strip(), f"Example {i} has empty function_name"
            assert example["module_path"].strip(), f"Example {i} has empty module_path"
            assert example[
                "function_signature"
            ].strip(), f"Example {i} has empty function_signature"
            assert example[
                "docstring_content"
            ].strip(), f"Example {i} has empty docstring_content"

            # If has_params is True, docstring should mention parameters
            if example["has_params"]:
                content_lower = example["docstring_content"].lower()
                assert any(
                    keyword in content_lower
                    for keyword in ["args:", "parameters", "params:"]
                ), f"Example '{example['function_name']}' marked has_params=True but no parameters section found"

            # If has_returns is True, docstring should mention returns
            if example["has_returns"]:
                content_lower = example["docstring_content"].lower()
                assert any(
                    keyword in content_lower
                    for keyword in ["returns:", "return:", "yields:"]
                ), f"Example '{example['function_name']}' marked has_returns=True but no returns section found"

            # If has_examples is True, docstring should have examples
            if example["has_examples"]:
                content_lower = example["docstring_content"].lower()
                assert any(
                    keyword in content_lower
                    for keyword in ["example:", "examples:", ">>>"]
                ), f"Example '{example['function_name']}' marked has_examples=True but no examples found"

    def test_async_pattern_examples(self, curated_examples):
        """Test async pattern examples have appropriate signatures."""
        async_examples = [
            ex
            for ex in curated_examples["examples"]
            if ex.get("category") == "async_patterns"
        ]

        assert (
            len(async_examples) >= 20
        ), "Should have at least 20 async pattern examples"

        for example in async_examples:
            # Async functions should have 'async def' in signature
            assert (
                "async def" in example["function_signature"]
                or "AsyncIterator" in example["function_signature"]
                or "AsyncContextManager" in example["function_signature"]
            ), f"Async example '{example['function_name']}' missing async signature"

    def test_rest_api_pattern_examples(self, curated_examples):
        """Test REST API pattern examples have appropriate characteristics."""
        api_examples = [
            ex
            for ex in curated_examples["examples"]
            if ex.get("category") == "rest_api_patterns"
        ]

        assert (
            len(api_examples) >= 15
        ), "Should have at least 15 REST API pattern examples"

        # Check for common REST patterns
        http_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        method_coverage = dict.fromkeys(http_methods, False)

        for example in api_examples:
            content = example["docstring_content"]
            for method in http_methods:
                if method in content:
                    method_coverage[method] = True

        # Should cover at least 3 different HTTP methods
        covered_methods = sum(method_coverage.values())
        assert (
            covered_methods >= 3
        ), f"REST API examples only cover {covered_methods} HTTP methods"
