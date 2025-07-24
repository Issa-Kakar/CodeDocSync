CLI Test Fix Action Plan
Current Status Assessment
✅ Already Completed

Created comprehensive test fixtures in tests/fixtures/:

simple_project/ - Basic modules with correct and incorrect docs
complex_project/ - Cross-file references and imports
edge_cases/ - Syntax errors, edge cases


Fixed test_match_basic to use real implementation
Added test_match_unified_command with multiple test cases

🔴 Critical Issues Remaining
Based on the FIX_CLI_TESTS_GUIDE.md and current progress, these are the ACTUAL problems:

Mock Dependencies Throughout test_cli.py

Multiple tests use @patch decorators expecting mock returns
Tests check mock.called instead of actual output
Mock data doesn't match real implementation output


Output Format Mismatches

Tests expect plain text, but CLI uses Rich formatting with emojis
JSON output tests may expect old schema
Progress indicators and styling not accounted for


Incorrect Command Options

--no-llm flag renamed to --rules-only
Some commands may have different option names
Profile names might have changed


Zero Coverage Commands

suggest-interactive has no tests
Possibly others missing basic coverage



Exact Fix Instructions
Phase 1: Scan and Fix All Mock-Based Tests
Task: Go through test_cli.py and fix EVERY test that uses mocks.
python# Pattern to find and fix:
@patch("codedocsync.cli.something")
def test_something(mock_thing):
    mock_thing.return_value = Mock(...)
    result = runner.invoke(app, ["command"])
    assert mock_thing.called

# Fix to:
def test_something():
    result = runner.invoke(app, ["command", "tests/fixtures/simple_project"])
    assert result.exit_code == 0
    assert "✨" in result.stdout  # Rich formatting
    assert "actual output text" in result.stdout
Specific tests that likely need fixing:

test_parse_basic - Remove mock, use fixtures
test_analyze_basic - Remove mock, expect real analysis output
test_analyze_with_severity - Check for actual severity markers
test_suggest_basic - Remove mock, expect real suggestions
Any test with @patch decorator

Phase 2: Update Output Expectations
Every assertion about output needs Rich formatting awareness:
python# OLD (wrong):
assert "Analyzing files" in result.stdout

# NEW (correct):
assert "✨ Analyzing" in result.stdout
# OR be more flexible:
assert "Analyzing" in result.stdout  # Partial match ignoring emojis
Common Rich elements to expect:

✨ for starting analysis
✅ for success
❌ for errors/failures
💡 for suggestions
Progress bars shown as [####...]
Colored output (though we just check text content)

Phase 3: Fix Command Options
Search and replace throughout test_cli.py:

--no-llm → --rules-only
Check all command invocations match current CLI interface

Phase 4: Add Missing Critical Tests Only
Only add tests for commands with ZERO coverage:

test_suggest_interactive (if it doesn't exist):

pythondef test_suggest_interactive_accept_flow():
    """Test accepting a suggestion interactively."""
    result = runner.invoke(
        app,
        ["suggest-interactive", "tests/fixtures/simple_project"],
        input="1\ny\n"  # Select first suggestion, accept
    )
    assert result.exit_code == 0
    assert "Suggestion" in result.stdout

def test_suggest_interactive_reject_flow():
    """Test rejecting suggestions."""
    result = runner.invoke(
        app,
        ["suggest-interactive", "tests/fixtures/simple_project"],
        input="1\nn\n"  # Select first suggestion, reject
    )
    assert result.exit_code == 0

Any other command with zero tests (check if these exist first):

check command for CI/CD mode?
init command for configuration?



Phase 5: Verify JSON Output Formats
For any test checking JSON output:
pythondef test_command_json_output():
    result = runner.invoke(app, ["analyze", "tests/fixtures/simple_project", "--format", "json"])
    assert result.exit_code == 0

    # Parse and validate structure
    data = json.loads(result.stdout)
    assert "summary" in data
    assert "issues" in data or "results" in data
    # Don't over-specify - just check key fields exist
Execution Checklist
Work through test_cli.py from top to bottom:
1. TestParseCommand

 Remove all @patch decorators
 Update to use tests/fixtures/simple_project
 Fix output assertions for Rich formatting
 Ensure JSON output test validates real schema

2. TestMatchCommand

 test_match_basic - Already fixed!
 test_match_with_threshold - Remove mocks if any
 Other match tests - Update output expectations

3. TestAnalyzeCommand

 test_analyze_basic - Remove mocks, use fixtures
 test_analyze_with_severity - Check for CRITICAL:, HIGH: etc
 test_analyze_parallel - Verify it uses real parallel processing
 Update all output assertions

4. TestSuggestCommand

 test_suggest_basic - Remove mocks
 Add test_suggest_interactive_* if missing
 Fix option names if changed

5. Global Fixes

 Replace all --no-llm with --rules-only
 Update any hardcoded file paths to use fixtures
 Remove any assert mock.called statements

What NOT to Do
Skip these from the original guide:

Don't add performance benchmarks
Don't add extensive edge case tests
Don't create more fixtures (use existing ones)
Don't add integration tests (you have those separately)
Don't test every possible command combination

Success Criteria
The CLI tests are DONE when:

✅ No more @patch decorators (except for external APIs like OpenAI)
✅ All tests use the fixtures in tests/fixtures/
✅ Output assertions match Rich-formatted output
✅ Command options are current (--rules-only not --no-llm)
✅ Every CLI command has at least one basic test
✅ All tests pass with pytest tests/test_cli.py

Final Notes
Remember: The integration tests prove the implementation works. These CLI tests just need to verify the command-line interface correctly calls the working implementation and formats output properly. Don't overthink it - fix what's broken, use the real implementation, and move on.
Start at the top of test_cli.py and work your way down methodically. Each test should take 2-5 minutes to fix. The entire task should take 2-3 hours maximum.
