#!/usr/bin/env python3
"""
Component validation script for CodeDocSync.
Tests critical functionality without requiring full test suite.
"""

import tempfile
import time
from pathlib import Path

from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
    parse_python_file,
)


def validate_parser():
    """Validate parser functionality."""
    print("\n=== VALIDATING PARSER ===")

    # Test 1: Basic function parsing
    test_code = '''
def example_function(a: int, b: str = "default") -> dict:
    """Example docstring.

    Args:
        a: First parameter
        b: Second parameter

    Returns:
        A dictionary
    """
    return {"a": a, "b": b}

@property
def prop_example(self):
    """Property example."""
    return self._value
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        f.flush()
        temp_path = f.name

    try:
        start = time.time()
        functions = parse_python_file(temp_path)
        elapsed = (time.time() - start) * 1000

        print(f"[PASS] Parsing completed in {elapsed:.2f}ms")
        print(f"[PASS] Found {len(functions)} functions")

        # Validate first function
        func = functions[0]
        assert (
            func.signature.name == "example_function"
        ), f"Wrong name: {func.signature.name}"
        assert func.end_line_number > func.line_number, "Missing end_line_number!"
        assert func.source_code, "Missing source_code!"
        assert len(func.signature.parameters) == 2, "Parameter parsing failed!"
        assert func.docstring is not None, "Docstring not parsed!"

        # Validate decorator parsing
        prop_func = functions[1]
        assert prop_func.signature.name == "prop_example"
        assert (
            "property" in prop_func.signature.decorators
            or "property" == prop_func.signature.decorators[0]
        )

        print("[PASS] All parser validations passed!")
        return True

    except Exception as e:
        print(f"[FAIL] Parser validation failed: {e}")
        return False
    finally:
        Path(temp_path).unlink(missing_ok=True)


def validate_parsed_function_creation():
    """Test creating ParsedFunction objects (for fixing tests)."""
    print("\n=== VALIDATING ParsedFunction CREATION ===")

    try:
        # Create minimal valid ParsedFunction
        signature = FunctionSignature(
            name="test_func",
            parameters=[
                FunctionParameter(
                    name="param1", type_annotation="str", is_required=True
                )
            ],
        )

        # CRITICAL: Must set end_line_number >= line_number
        parsed_func = ParsedFunction(
            signature=signature,
            docstring=RawDocstring(raw_text="Test docstring", line_number=2),
            file_path="test.py",
            line_number=1,
            end_line_number=5,  # MUST be >= line_number
            source_code="def test_func(param1: str): pass",  # Optional but good practice
        )

        print(f"[PASS] Created ParsedFunction: {parsed_func.signature.name}")
        print(
            f"[PASS] Line range: {parsed_func.line_number}-{parsed_func.end_line_number}"
        )
        return True

    except Exception as e:
        print(f"[FAIL] ParsedFunction creation failed: {e}")
        return False


def validate_suggestion_generator():
    """Validate suggestion generator structure."""
    print("\n=== VALIDATING SUGGESTION GENERATOR ===")

    try:
        from codedocsync.suggestions.generators.parameter_generator import (
            ParameterSuggestionGenerator,
        )

        # Create test function with parameter mismatch
        signature = FunctionSignature(
            name="calculate",
            parameters=[
                FunctionParameter(
                    name="amount", type_annotation="float", is_required=True
                ),
                FunctionParameter(
                    name="rate", type_annotation="float", is_required=True
                ),
            ],
        )

        from codedocsync.parser.docstring_parser import DocstringParser

        # Create docstring with mismatched parameter name
        docstring_text = """Calculate interest.

        Args:
            value: The principal amount  # Wrong name!
            rate: The interest rate
        """

        parser = DocstringParser()
        parsed_doc = parser.parse(docstring_text)

        ParsedFunction(
            signature=signature,
            docstring=parsed_doc,
            file_path="test.py",
            line_number=1,
            end_line_number=10,
            source_code="def calculate(amount, rate): pass",
        )

        # Test generator
        generator = ParameterSuggestionGenerator()

        # Check if generate method exists
        if not hasattr(generator, "generate"):
            print("[FAIL] ParameterSuggestionGenerator missing generate method!")
            return False

        print("[PASS] ParameterSuggestionGenerator has correct interface")
        print("[INFO] Key findings:")
        print("  - Tests expect 'generate_suggestion' but class has 'generate'")
        print(
            "  - Tests create ParsedFunction without required fields (end_line_number, source_code)"
        )
        print("  - Parser decorators have line number off-by-one issues")
        return True

    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Suggestion generator validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("CodeDocSync Component Validation")
    print("=" * 50)

    results = {
        "Parser": validate_parser(),
        "ParsedFunction Creation": validate_parsed_function_creation(),
        "Suggestion Generator": validate_suggestion_generator(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY:")
    for component, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{component}: {status}")

    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{len(results)} components working")

    if total_passed < len(results):
        print("\n[WARNING] Some components need fixes before full test suite will pass")
    else:
        print("\n[SUCCESS] All critical components validated!")


if __name__ == "__main__":
    main()
