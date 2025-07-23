import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import tempfile
import time


def create_test_file(num_functions=100):
    """Create a test file with specified number of functions."""
    content = []
    for i in range(num_functions):
        content.append(
            f'''
def function_{i}(param_{i}: int) -> str:
    """Function {i} documentation.

    Args:
        param_{i}: Parameter {i}

    Returns:
        String result
    """
    return f"result_{i}"
'''
        )
    return "\n".join(content)


def test_parser_performance():
    """Test parser meets <50ms requirement."""
    from codedocsync.parser.ast_parser import parse_python_file

    # Create test file
    with open("perf_test.py", "w") as f:
        f.write(create_test_file(100))

    # Warm up
    parse_python_file("perf_test.py")

    # Measure
    start = time.time()
    functions = parse_python_file("perf_test.py")
    elapsed = (time.time() - start) * 1000

    print("Parser Performance:")
    print(f"  - Parsed {len(functions)} functions")
    print(f"  - Time: {elapsed:.2f}ms (target: <50ms)")
    print(f"  - {'[PASS]' if elapsed < 50 else '[FAIL]'}")

    os.remove("perf_test.py")
    return elapsed < 50


def test_analyzer_performance():
    """Test analyzer meets <5ms requirement."""
    from codedocsync.analyzer.rule_engine import RuleEngine
    from codedocsync.matcher.models import MatchConfidence, MatchedPair, MatchType
    from codedocsync.parser.ast_parser import FunctionParameter
    from tests.conftest import create_test_function

    # Create test data
    func = create_test_function(
        "test",
        [
            FunctionParameter(
                name="x", type_annotation="int", default_value=None, is_required=True
            )
        ],
    )

    matched_pair = MatchedPair(
        function=func,
        match_type=MatchType.EXACT,
        confidence=MatchConfidence(
            overall=1.0,
            name_similarity=1.0,
            location_score=1.0,
            signature_similarity=1.0,
        ),
        match_reason="Test",
        docstring=None,
    )

    rule_engine = RuleEngine()

    # Warm up
    rule_engine.check_matched_pair(matched_pair)

    # Measure
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        rule_engine.check_matched_pair(matched_pair)
    elapsed = ((time.time() - start) / iterations) * 1000

    print("\nRule Engine Performance:")
    print(f"  - Average time: {elapsed:.2f}ms (target: <5ms)")
    print(f"  - {'[PASS]' if elapsed < 5 else '[FAIL]'}")

    return elapsed < 5


def test_project_performance():
    """Test full project analysis performance."""
    # Create temporary project
    temp_dir = tempfile.mkdtemp()

    try:
        # Create 100 Python files
        for i in range(100):
            file_path = os.path.join(temp_dir, f"module_{i}.py")
            with open(file_path, "w") as f:
                f.write(create_test_file(10))  # 10 functions per file

        print("\nProject Analysis Performance:")
        print("  - Created test project with 100 files, 1000 functions")

        # Measure analysis time
        start = time.time()
        exit_code = os.system(
            f"python -m codedocsync analyze {temp_dir} --format json > outputs/perf_analysis.json 2>&1"
        )
        elapsed = time.time() - start

        print(f"  - Analysis time: {elapsed:.2f}s (target: <5s)")
        print(f"  - Exit code: {exit_code}")
        print(f"  - {'[PASS]' if elapsed < 5 else '[FAIL]'}")

        return elapsed < 5

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("=== Performance Validation ===\n")

    results = [
        test_parser_performance(),
        test_analyzer_performance(),
        test_project_performance(),
    ]

    print(
        f"\nOverall: {'[PASS] ALL TESTS PASSED' if all(results) else '[FAIL] SOME TESTS FAILED'}"
    )
