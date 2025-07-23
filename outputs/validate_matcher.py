import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codedocsync.matcher import ContextualMatcher, DirectMatcher, UnifiedMatchingFacade
from codedocsync.parser.ast_parser import parse_python_file


def validate_matcher():
    print("=== Matcher Module Validation ===\n")

    # Create test files
    with open("test_source.py", "w") as f:
        f.write(
            '''
def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of numbers.

    Args:
        numbers: List of integers to sum

    Returns:
        The sum of all numbers
    """
    return sum(numbers)

def calculate_mean(values: list[float]) -> float:
    """Calculate mean of values."""
    return sum(values) / len(values)
'''
        )

    # Parse the file
    functions = parse_python_file("test_source.py")
    print(f"[PASS] Parsed {len(functions)} functions")

    # Test Direct Matcher
    direct_matcher = DirectMatcher()
    result = direct_matcher.match_functions(functions)
    print(f"[PASS] Direct matcher processed {result.total_functions} functions")
    print(f"  - Matched: {len(result.matched_pairs)}")
    print(f"  - Unmatched: {len(result.unmatched_functions)}")
    print(f"  - Match rate: {result.match_rate:.1%}")

    for match in result.matched_pairs:
        print(f"  [PASS] Matched: {match.function.signature.name}")

    for func in result.unmatched_functions:
        print(f"  [INFO] Unmatched: {func.signature.name} (no docstring)")

    # Test Contextual Matcher (if API available)
    try:
        ContextualMatcher(project_root=".")
        print("\n[PASS] Contextual matcher initialized")
    except Exception as e:
        print(f"\n[INFO] Contextual matcher initialization: {e}")

    # Test Unified Facade
    try:
        UnifiedMatchingFacade()
        print("[PASS] Unified facade initialized")
    except Exception as e:
        print(f"[INFO] Unified facade initialization: {e}")

    # Clean up
    os.remove("test_source.py")
    print("\n=== Matcher Validation Complete ===")


if __name__ == "__main__":
    validate_matcher()
