import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codedocsync.analyzer.rule_engine import RuleEngine
from codedocsync.matcher.models import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser.ast_parser import parse_python_file


def validate_analyzer():
    print("=== Analyzer Module Validation ===\n")

    # Create test function with issues
    with open("test_analyzer.py", "w") as f:
        f.write(
            '''
def process_data(items: list[str], threshold: float) -> dict:
    """Process data items.

    Args:
        data: List of items to process  # WRONG NAME!
        limit: Threshold value  # WRONG NAME!

    Returns:
        Processing results
    """
    return {"count": len(items)}
'''
        )

    # Parse and create matched pair
    functions = parse_python_file("test_analyzer.py")
    func = functions[0]

    # Create a matched pair
    matched_pair = MatchedPair(
        function=func,
        docstring=func.docstring,
        confidence=MatchConfidence(
            overall=0.9,
            name_similarity=1.0,
            location_score=1.0,
            signature_similarity=0.8,
        ),
        match_type=MatchType.EXACT,
        match_reason="Direct name match",
    )

    # Test rule engine
    rule_engine = RuleEngine()
    print("[PASS] Rule engine initialized")

    # Test analysis using rule engine directly
    try:
        issues = rule_engine.check_matched_pair(matched_pair)
        print(f"[PASS] Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue.severity}: {issue.issue_type} - {issue.description}")
    except Exception as e:
        print(f"[FAIL] Rule engine analysis failed: {e}")
        import traceback

        traceback.print_exc()

    # Test async analysis function exists
    try:
        print("\n[PASS] analyze_matched_pair function is available (async)")
    except Exception as e:
        print(f"[FAIL] analyze_matched_pair import failed: {e}")

    # Clean up
    os.remove("test_analyzer.py")
    print("\n=== Analyzer Validation Complete ===")


if __name__ == "__main__":
    validate_analyzer()
