import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from codedocsync.matcher.contextual_matcher import ContextualMatcher
from codedocsync.matcher.models import (
    MatchResult,
    MatchedPair,
    MatchType,
    MatchConfidence,
)
from codedocsync.matcher.contextual_models import (
    ImportStatement,
    ImportType,
    ModuleInfo,
    FunctionLocation,
)
from codedocsync.parser.ast_parser import (
    FunctionSignature,
    FunctionParameter,
    ParsedFunction,
)
from codedocsync.parser import RawDocstring


class TestContextualMatcher:
    """Test suite for ContextualMatcher class."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create project structure
            (project_root / "src").mkdir()
            (project_root / "src" / "utils").mkdir()
            (project_root / "tests").mkdir()

            # Create Python files
            (project_root / "src" / "__init__.py").write_text("")
            (project_root / "src" / "main.py").write_text(
                """
def main():
    \"\"\"Main function.\"\"\"
    pass
"""
            )
            (project_root / "src" / "utils" / "__init__.py").write_text("")
            (project_root / "src" / "utils" / "helpers.py").write_text(
                """
def helper_function():
    \"\"\"Helper function.\"\"\"
    pass
"""
            )
            yield project_root

    @pytest.fixture
    def sample_function(self):
        """Create a sample ParsedFunction for testing."""
        signature = FunctionSignature(
            name="test_function",
            parameters=[
                FunctionParameter(
                    name="param1", type_annotation="str", default_value=None
                )
            ],
            return_type="str",
            decorators=[],
        )

        docstring = RawDocstring(raw_text="Test function docstring", line_number=2)

        return ParsedFunction(
            signature=signature,
            docstring=docstring,
            file_path="/test/file.py",
            line_number=1,
            end_line_number=5,
            source_code='def test_function(param1: str) -> str:\n    """Test function docstring"""\n    return param1',
        )

    @pytest.fixture
    def contextual_matcher(self, temp_project):
        """Create a ContextualMatcher instance."""
        return ContextualMatcher(str(temp_project))

    def test_init(self, temp_project):
        """Test ContextualMatcher initialization."""
        matcher = ContextualMatcher(str(temp_project))

        assert matcher.project_root == temp_project
        assert matcher.state is not None
        assert matcher.module_resolver is not None
        assert matcher.function_registry is not None
        assert matcher.import_parser is not None
        assert matcher.integrated_parser is not None
        assert matcher.stats["files_analyzed"] == 0
        assert matcher.stats["imports_resolved"] == 0
        assert matcher.stats["cross_file_matches"] == 0
        assert matcher.stats["moved_functions"] == 0

    def test_analyze_project_with_files(self, contextual_matcher, temp_project):
        """Test project analysis with provided file list."""
        files = [
            str(temp_project / "src" / "main.py"),
            str(temp_project / "src" / "utils" / "helpers.py"),
        ]

        with patch.object(contextual_matcher, "_analyze_file") as mock_analyze:
            with patch.object(
                contextual_matcher, "_resolve_all_imports"
            ) as mock_resolve:
                contextual_matcher.analyze_project(files)

                assert mock_analyze.call_count == 2
                assert mock_resolve.called

    def test_analyze_project_auto_discovery(self, contextual_matcher):
        """Test project analysis with automatic file discovery."""
        with patch.object(
            contextual_matcher, "_discover_python_files"
        ) as mock_discover:
            mock_discover.return_value = ["/test/file.py"]

            with patch.object(contextual_matcher, "_analyze_file") as mock_analyze:
                with patch.object(
                    contextual_matcher, "_resolve_all_imports"
                ) as mock_resolve:
                    contextual_matcher.analyze_project()

                    assert mock_discover.called
                    assert mock_analyze.call_count == 1
                    assert mock_resolve.called

    def test_match_with_context_no_previous_results(
        self, contextual_matcher, sample_function
    ):
        """Test contextual matching without previous direct match results."""
        functions = [sample_function]

        with patch.object(contextual_matcher, "_find_contextual_match") as mock_find:
            mock_find.return_value = None

            result = contextual_matcher.match_with_context(functions)

            assert result.total_functions == 1
            assert len(result.matched_pairs) == 0
            assert len(result.unmatched_functions) == 1
            assert result.unmatched_functions[0] == sample_function

    def test_match_with_context_with_previous_results(
        self, contextual_matcher, sample_function
    ):
        """Test contextual matching with previous direct match results."""
        functions = [sample_function]

        # Create a previous match result with high confidence
        previous_match = MatchedPair(
            function=sample_function,
            match_type=MatchType.EXACT,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=1.0,
                location_score=1.0,
                signature_similarity=1.0,
            ),
            match_reason="Direct match",
        )

        direct_result = MatchResult(
            total_functions=1, matched_pairs=[previous_match], unmatched_functions=[],
        )

        result = contextual_matcher.match_with_context(functions, direct_result)

        # Should keep the previous high-confidence match
        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0] == previous_match

    def test_match_with_context_low_confidence_previous(
        self, contextual_matcher, sample_function
    ):
        """Test contextual matching with low confidence previous results."""
        functions = [sample_function]

        # Create a previous match result with low confidence
        previous_match = MatchedPair(
            function=sample_function,
            match_type=MatchType.FUZZY,
            confidence=MatchConfidence(
                overall=0.5,  # Low confidence
                name_similarity=0.7,
                location_score=0.8,
                signature_similarity=0.6,
            ),
            match_reason="Fuzzy match",
        )

        direct_result = MatchResult(
            total_functions=1, matched_pairs=[previous_match], unmatched_functions=[],
        )

        # Mock contextual match finding
        contextual_match = MatchedPair(
            function=sample_function,
            match_type=MatchType.CONTEXTUAL,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=1.0,
                location_score=0.8,
                signature_similarity=0.9,
            ),
            match_reason="Contextual match",
        )

        with patch.object(contextual_matcher, "_find_contextual_match") as mock_find:
            mock_find.return_value = contextual_match

            result = contextual_matcher.match_with_context(functions, direct_result)

            # Should include both previous and contextual matches
            assert len(result.matched_pairs) == 2
            assert contextual_match in result.matched_pairs

    def test_analyze_file_success(self, contextual_matcher, temp_project):
        """Test successful file analysis."""
        file_path = str(temp_project / "src" / "main.py")

        with patch.object(
            contextual_matcher.module_resolver, "resolve_module_path"
        ) as mock_resolve:
            mock_resolve.return_value = "src.main"

            with patch.object(
                contextual_matcher.import_parser, "build_module_info"
            ) as mock_build:
                mock_module_info = ModuleInfo(
                    module_path="src.main",
                    file_path=file_path,
                    imports=[],
                    exports=set(),
                    functions={},
                )
                mock_build.return_value = mock_module_info

                with patch.object(
                    contextual_matcher.integrated_parser, "parse_file"
                ) as mock_parse:
                    mock_parse.return_value = []

                    with patch.object(
                        contextual_matcher.function_registry, "register_function"
                    ):
                        contextual_matcher._analyze_file(file_path)

                        assert contextual_matcher.stats["files_analyzed"] == 1
                        assert mock_resolve.called
                        assert mock_build.called
                        assert mock_parse.called

    def test_analyze_file_no_module_path(self, contextual_matcher, temp_project):
        """Test file analysis when module path cannot be resolved."""
        file_path = str(temp_project / "src" / "main.py")

        with patch.object(
            contextual_matcher.module_resolver, "resolve_module_path"
        ) as mock_resolve:
            mock_resolve.return_value = None

            # Should handle gracefully without raising exception
            contextual_matcher._analyze_file(file_path)

            assert contextual_matcher.stats["files_analyzed"] == 0

    def test_analyze_file_exception_handling(self, contextual_matcher, temp_project):
        """Test file analysis exception handling."""
        file_path = str(temp_project / "src" / "main.py")

        with patch.object(
            contextual_matcher.module_resolver, "resolve_module_path"
        ) as mock_resolve:
            mock_resolve.side_effect = Exception("Test error")

            # Should handle exception gracefully
            contextual_matcher._analyze_file(file_path)

            assert contextual_matcher.stats["files_analyzed"] == 0

    def test_find_contextual_match_import_match(
        self, contextual_matcher, sample_function
    ):
        """Test finding contextual match for imported function."""
        with patch.object(
            contextual_matcher, "_match_imported_function"
        ) as mock_import:
            mock_match = MatchedPair(
                function=sample_function,
                match_type=MatchType.CONTEXTUAL,
                confidence=MatchConfidence(
                    overall=0.9,
                    name_similarity=1.0,
                    location_score=0.8,
                    signature_similarity=0.9,
                ),
                match_reason="Imported function",
            )
            mock_import.return_value = mock_match

            result = contextual_matcher._find_contextual_match(sample_function)

            assert result == mock_match

    def test_find_contextual_match_moved_function(
        self, contextual_matcher, sample_function
    ):
        """Test finding contextual match for moved function."""
        with patch.object(
            contextual_matcher, "_match_imported_function"
        ) as mock_import:
            mock_import.return_value = None

            with patch.object(
                contextual_matcher, "_match_moved_function"
            ) as mock_moved:
                mock_match = MatchedPair(
                    function=sample_function,
                    match_type=MatchType.CONTEXTUAL,
                    confidence=MatchConfidence(
                        overall=0.85,
                        name_similarity=1.0,
                        location_score=0.7,
                        signature_similarity=0.85,
                    ),
                    match_reason="Moved function",
                )
                mock_moved.return_value = mock_match

                result = contextual_matcher._find_contextual_match(sample_function)

                assert result == mock_match

    def test_find_contextual_match_no_match(self, contextual_matcher, sample_function):
        """Test finding contextual match when no match is found."""
        with patch.object(
            contextual_matcher, "_match_imported_function"
        ) as mock_import:
            mock_import.return_value = None

            with patch.object(
                contextual_matcher, "_match_moved_function"
            ) as mock_moved:
                mock_moved.return_value = None

                with patch.object(
                    contextual_matcher, "_match_cross_file_docs"
                ) as mock_cross:
                    mock_cross.return_value = None

                    result = contextual_matcher._find_contextual_match(sample_function)

                    assert result is None

    def test_match_imported_function_success(self, contextual_matcher, sample_function):
        """Test successful matching of imported function."""
        # Mock module resolution
        with patch.object(
            contextual_matcher.module_resolver, "resolve_module_path"
        ) as mock_resolve:
            mock_resolve.return_value = "test.module"

            # Set up module info with imports
            import_stmt = ImportStatement(
                import_type=ImportType.FROM,
                module_path="other.module",
                imported_names=["test_function"],
                aliases={},
                line_number=1,
            )

            module_info = ModuleInfo(
                module_path="test.module",
                file_path="/test/file.py",
                imports=[import_stmt],
                exports=set(),
                functions={},
            )

            contextual_matcher.state.add_module(module_info)

            # Mock import resolution
            with patch.object(
                contextual_matcher.module_resolver, "resolve_import"
            ) as mock_resolve_import:
                mock_resolve_import.return_value = "other.module"

                # Mock function registry lookup
                location = FunctionLocation(
                    canonical_module="other.module",
                    function_name="test_function",
                    line_number=1,
                    import_paths=set(),
                    is_exported=True,
                )

                with patch.object(
                    contextual_matcher.function_registry, "find_function"
                ) as mock_find:
                    mock_find.return_value = [location]

                    result = contextual_matcher._match_imported_function(
                        sample_function
                    )

                    assert result is not None
                    assert result.match_type == MatchType.CONTEXTUAL
                    assert result.confidence.overall == 0.9
                    assert "Imported from other.module" in result.match_reason

    def test_match_imported_function_no_module_path(
        self, contextual_matcher, sample_function
    ):
        """Test imported function matching when module path cannot be resolved."""
        with patch.object(
            contextual_matcher.module_resolver, "resolve_module_path"
        ) as mock_resolve:
            mock_resolve.return_value = None

            result = contextual_matcher._match_imported_function(sample_function)

            assert result is None

    def test_match_moved_function_success(self, contextual_matcher, sample_function):
        """Test successful matching of moved function."""
        # Mock module resolution
        with patch.object(
            contextual_matcher.module_resolver, "resolve_module_path"
        ) as mock_resolve:
            mock_resolve.return_value = "test.module"

            # Mock function registry lookup
            location = FunctionLocation(
                canonical_module="other.module",
                function_name="test_function",
                line_number=1,
                import_paths=set(),
                is_exported=True,
            )

            with patch.object(
                contextual_matcher.function_registry, "find_moved_function"
            ) as mock_find_moved:
                mock_find_moved.return_value = location

                with patch.object(
                    contextual_matcher, "_calculate_signature_similarity"
                ) as mock_calc:
                    mock_calc.return_value = 0.9  # High similarity

                    result = contextual_matcher._match_moved_function(sample_function)

                    assert result is not None
                    assert result.match_type == MatchType.CONTEXTUAL
                    assert result.confidence.overall == 0.85
                    assert "Function moved from other.module" in result.match_reason
                    assert contextual_matcher.stats["moved_functions"] == 1

    def test_match_moved_function_low_similarity(
        self, contextual_matcher, sample_function
    ):
        """Test moved function matching with low signature similarity."""
        with patch.object(
            contextual_matcher.module_resolver, "resolve_module_path"
        ) as mock_resolve:
            mock_resolve.return_value = "test.module"

            location = FunctionLocation(
                canonical_module="other.module",
                function_name="test_function",
                line_number=1,
                import_paths=set(),
                is_exported=True,
            )

            with patch.object(
                contextual_matcher.function_registry, "find_moved_function"
            ) as mock_find_moved:
                mock_find_moved.return_value = location

                with patch.object(
                    contextual_matcher, "_calculate_signature_similarity"
                ) as mock_calc:
                    mock_calc.return_value = 0.5  # Low similarity

                    result = contextual_matcher._match_moved_function(sample_function)

                    assert result is None

    def test_discover_python_files(self, contextual_matcher, temp_project):
        """Test Python file discovery."""
        files = contextual_matcher._discover_python_files()

        # Should find the Python files we created
        assert len(files) >= 2  # At least main.py and helpers.py
        assert any("main.py" in f for f in files)
        assert any("helpers.py" in f for f in files)

        # Should not include __pycache__ or .git files
        assert not any("__pycache__" in f for f in files)
        assert not any(".git" in f for f in files)

    def test_resolve_all_imports(self, contextual_matcher):
        """Test import resolution tracking."""
        # Add mock module info with imports
        import_stmt = ImportStatement(
            import_type=ImportType.FROM,
            module_path="test.module",
            imported_names=["func"],
            aliases={},
            line_number=1,
        )

        module_info = ModuleInfo(
            module_path="test.module",
            file_path="/test/file.py",
            imports=[import_stmt],
            exports=set(),
            functions={},
        )

        contextual_matcher.state.add_module(module_info)

        contextual_matcher._resolve_all_imports()

        assert contextual_matcher.stats["imports_resolved"] == 1

    def test_create_import_match(self, contextual_matcher, sample_function):
        """Test creating import match result."""
        location = FunctionLocation(
            canonical_module="other.module",
            function_name="test_function",
            line_number=1,
            import_paths=set(),
            is_exported=True,
        )

        import_stmt = ImportStatement(
            import_type=ImportType.FROM,
            module_path="other.module",
            imported_names=["test_function"],
            aliases={},
            line_number=1,
        )

        result = contextual_matcher._create_import_match(
            sample_function, location, import_stmt
        )

        assert result.function == sample_function
        assert result.match_type == MatchType.CONTEXTUAL
        assert result.confidence.overall == 0.9
        assert result.confidence.name_similarity == 1.0
        assert "Imported from other.module" in result.match_reason

    def test_create_moved_match(self, contextual_matcher, sample_function):
        """Test creating moved function match result."""
        location = FunctionLocation(
            canonical_module="other.module",
            function_name="test_function",
            line_number=1,
            import_paths=set(),
            is_exported=True,
        )

        result = contextual_matcher._create_moved_match(sample_function, location)

        assert result.function == sample_function
        assert result.match_type == MatchType.CONTEXTUAL
        assert result.confidence.overall == 0.85
        assert result.confidence.name_similarity == 1.0
        assert "Function moved from other.module" in result.match_reason

    def test_calculate_signature_similarity_placeholder(
        self, contextual_matcher, sample_function
    ):
        """Test signature similarity calculation (placeholder implementation)."""
        location = FunctionLocation(
            canonical_module="other.module",
            function_name="test_function",
            line_number=1,
            import_paths=set(),
            is_exported=True,
        )

        similarity = contextual_matcher._calculate_signature_similarity(
            sample_function, location
        )

        # Should return placeholder value
        assert similarity == 0.85

    def test_match_cross_file_docs_placeholder(
        self, contextual_matcher, sample_function
    ):
        """Test cross-file documentation matching (placeholder implementation)."""
        result = contextual_matcher._match_cross_file_docs(sample_function)

        # Should return None for placeholder implementation
        assert result is None
