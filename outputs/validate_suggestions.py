import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import parse_python_file
from codedocsync.suggestions import create_suggestion_context, enhance_with_suggestions
from codedocsync.suggestions.generators.parameter_generator import (
    ParameterSuggestionGenerator,
)


def validate_suggestions():
    print("=== Suggestions Module Validation ===\n")

    # Create test file with issues
    with open("test_suggestions.py", "w") as f:
        f.write(
            '''
def authenticate(username: str, password: str) -> bool:
    """Authenticate a user.

    Args:
        email: User email  # WRONG PARAMETER NAME!
        password: User password

    Returns:
        True if authenticated
    """
    return True
'''
        )

    # Parse the function
    functions = parse_python_file("test_suggestions.py")
    func = functions[0]
    print(f"[PASS] Parsed function: {func.signature.name}")

    # Create an issue
    issue = InconsistencyIssue(
        issue_type="parameter_name_mismatch",
        severity="critical",
        description="Parameter 'username' documented as 'email'",
        suggestion="Update docstring parameter from 'email' to 'username'",
        line_number=func.line_number,
        details={
            "expected": "username",
            "actual": "email",
            "file_path": "test_suggestions.py",
            "function_name": "authenticate",
        },
    )

    # Test suggestion creation using parameter generator
    try:
        # Create suggestion context
        context = create_suggestion_context(
            issue=issue, function=func, docstring=func.docstring
        )

        # Use parameter generator directly
        generator = ParameterSuggestionGenerator()
        suggestion = generator.generate(context)

        print(f"[PASS] Created suggestion with confidence: {suggestion.confidence}")
        print(f"  Original: {suggestion.original_text[:50]}...")
        print(f"  Suggested: {suggestion.suggested_text[:50]}...")
    except Exception as e:
        print(f"[FAIL] Suggestion creation failed: {e}")
        import traceback

        traceback.print_exc()

    # Test enhancement
    try:
        enhanced = enhance_with_suggestions(func, [issue])
        print(f"\n[PASS] Enhanced with {len(enhanced.suggestions)} suggestions")
    except Exception as e:
        print(f"[FAIL] Enhancement failed: {e}")

    # Clean up
    os.remove("test_suggestions.py")
    print("\n=== Suggestions Validation Complete ===")


if __name__ == "__main__":
    validate_suggestions()
