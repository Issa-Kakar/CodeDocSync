"""Test suite for contextual matcher models."""

from dataclasses import dataclass

import pytest

from codedocsync.matcher.contextual_models import (
    ContextualMatcherState,
    CrossFileMatch,
    FunctionLocation,
    ImportStatement,
    ImportType,
    ModuleInfo,
)


class TestImportType:
    """Test ImportType enum."""

    def test_import_type_values(self):
        """Test all import type values exist."""
        assert ImportType.STANDARD.value == "standard"
        assert ImportType.FROM.value == "from"
        assert ImportType.RELATIVE.value == "relative"
        assert ImportType.WILDCARD.value == "wildcard"


class TestImportStatement:
    """Test ImportStatement dataclass."""

    def test_valid_import_statement(self):
        """Test creating valid import statement."""
        stmt = ImportStatement(
            import_type=ImportType.FROM,
            module_path="os.path",
            imported_names=["join", "exists"],
            aliases={"join": "path_join"},
            line_number=1,
        )

        assert stmt.import_type == ImportType.FROM
        assert stmt.module_path == "os.path"
        assert stmt.imported_names == ["join", "exists"]
        assert stmt.aliases == {"join": "path_join"}
        assert stmt.line_number == 1
        assert stmt.level == 0

    def test_wildcard_import_validation(self):
        """Test wildcard import validation."""
        # Valid wildcard import
        stmt = ImportStatement(
            import_type=ImportType.WILDCARD,
            module_path="os",
            imported_names=["*"],
            aliases={},
            line_number=1,
        )
        assert stmt.import_type == ImportType.WILDCARD

        # Invalid wildcard import
        with pytest.raises(ValueError, match="Wildcard imports must have exactly one"):
            ImportStatement(
                import_type=ImportType.WILDCARD,
                module_path="os",
                imported_names=["*", "path"],
                aliases={},
                line_number=1,
            )

    def test_relative_import_validation(self):
        """Test relative import validation."""
        # Valid relative import
        stmt = ImportStatement(
            import_type=ImportType.RELATIVE,
            module_path="..utils",
            imported_names=["helper"],
            aliases={},
            line_number=1,
            level=2,
        )
        assert stmt.level == 2

        # Invalid relative import
        with pytest.raises(
            ValueError, match="Only relative imports can have level > 0"
        ):
            ImportStatement(
                import_type=ImportType.STANDARD,
                module_path="os",
                imported_names=[],
                aliases={},
                line_number=1,
                level=1,
            )


class TestModuleInfo:
    """Test ModuleInfo dataclass."""

    def test_module_info_creation(self):
        """Test creating module info."""
        info = ModuleInfo(
            module_path="mypackage.utils",
            file_path="/path/to/utils.py",
            imports=[],
            exports={"helper_func", "UtilityClass"},
            is_package=False,
        )

        assert info.module_path == "mypackage.utils"
        assert info.file_path == "/path/to/utils.py"
        assert info.exports == {"helper_func", "UtilityClass"}
        assert not info.is_package

    def test_get_canonical_name(self):
        """Test canonical name generation."""
        info = ModuleInfo(module_path="mypackage.utils", file_path="/path/to/utils.py")

        canonical = info.get_canonical_name("helper_func")
        assert canonical == "mypackage.utils.helper_func"

    def test_module_info_defaults(self):
        """Test module info with default values."""
        info = ModuleInfo(module_path="test.module", file_path="/test/module.py")

        assert info.imports == []
        assert info.exports == set()
        assert info.functions == {}
        assert not info.is_package


class TestFunctionLocation:
    """Test FunctionLocation dataclass."""

    def test_function_location_creation(self):
        """Test creating function location."""
        location = FunctionLocation(
            canonical_module="mypackage.utils",
            function_name="helper_func",
            line_number=42,
            import_paths={"from mypackage.utils import helper_func"},
            is_exported=True,
        )

        assert location.canonical_module == "mypackage.utils"
        assert location.function_name == "helper_func"
        assert location.line_number == 42
        assert location.is_exported

    def test_function_location_defaults(self):
        """Test function location with default values."""
        location = FunctionLocation(
            canonical_module="test.module", function_name="test_func", line_number=10
        )

        assert location.import_paths == set()
        assert location.is_exported


class TestCrossFileMatch:
    """Test CrossFileMatch dataclass."""

    def test_cross_file_match_creation(self):
        """Test creating cross file match."""

        # We'll use mock objects since we don't have actual ParsedFunction/ParsedDocstring
        @dataclass
        class MockSignature:
            name: str

        @dataclass
        class MockFunction:
            signature: MockSignature

        @dataclass
        class MockDocstring:
            summary: str

        match = CrossFileMatch(
            function=MockFunction(MockSignature("test_func")),
            documentation=MockDocstring("Test function"),
            match_reason="imported_function",
            import_chain=["from utils import test_func"],
            confidence=0.9,
        )

        assert match.function.signature.name == "test_func"
        assert match.documentation.summary == "Test function"
        assert match.match_reason == "imported_function"
        assert match.import_chain == ["from utils import test_func"]
        assert match.confidence == 0.9

    def test_confidence_validation(self):
        """Test confidence validation."""

        @dataclass
        class MockSignature:
            name: str

        @dataclass
        class MockFunction:
            signature: MockSignature

        @dataclass
        class MockDocstring:
            summary: str

        # Valid confidence
        match = CrossFileMatch(
            function=MockFunction(MockSignature("test_func")),
            documentation=MockDocstring("Test function"),
            match_reason="test",
            import_chain=[],
            confidence=0.5,
        )
        assert match.confidence == 0.5

        # Invalid confidence - too low
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CrossFileMatch(
                function=MockFunction(MockSignature("test_func")),
                documentation=MockDocstring("Test function"),
                match_reason="test",
                import_chain=[],
                confidence=-0.1,
            )

        # Invalid confidence - too high
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CrossFileMatch(
                function=MockFunction(MockSignature("test_func")),
                documentation=MockDocstring("Test function"),
                match_reason="test",
                import_chain=[],
                confidence=1.1,
            )


class TestContextualMatcherState:
    """Test ContextualMatcherState dataclass."""

    def test_contextual_matcher_state_creation(self):
        """Test creating contextual matcher state."""
        state = ContextualMatcherState()

        assert state.module_tree == {}
        assert state.import_graph == {}
        assert state.function_registry == {}

    def test_add_module(self):
        """Test adding module to state."""
        state = ContextualMatcherState()

        # Create a module with imports
        import_stmt = ImportStatement(
            import_type=ImportType.FROM,
            module_path="os.path",
            imported_names=["join"],
            aliases={},
            line_number=1,
        )

        module_info = ModuleInfo(
            module_path="test.module",
            file_path="/test/module.py",
            imports=[import_stmt],
        )

        state.add_module(module_info)

        # Check module was added
        assert "test.module" in state.module_tree
        assert state.module_tree["test.module"] == module_info

        # Check import graph was updated
        assert "test.module" in state.import_graph
        assert "os.path" in state.import_graph["test.module"]

    def test_add_module_multiple_imports(self):
        """Test adding module with multiple imports."""
        state = ContextualMatcherState()

        # Create module with multiple imports
        imports = [
            ImportStatement(
                import_type=ImportType.STANDARD,
                module_path="os",
                imported_names=[],
                aliases={},
                line_number=1,
            ),
            ImportStatement(
                import_type=ImportType.FROM,
                module_path="sys",
                imported_names=["argv"],
                aliases={},
                line_number=2,
            ),
        ]

        module_info = ModuleInfo(
            module_path="test.module", file_path="/test/module.py", imports=imports
        )

        state.add_module(module_info)

        # Check both imports are in graph
        assert "test.module" in state.import_graph
        assert "os" in state.import_graph["test.module"]
        assert "sys" in state.import_graph["test.module"]
        assert len(state.import_graph["test.module"]) == 2


class TestIntegration:
    """Test integration between components."""

    def test_complete_workflow(self):
        """Test complete workflow with all components."""
        # Create import statements
        import_stmt = ImportStatement(
            import_type=ImportType.FROM,
            module_path="utils.helpers",
            imported_names=["format_text", "parse_data"],
            aliases={"format_text": "fmt"},
            line_number=1,
        )

        # Create function location
        func_location = FunctionLocation(
            canonical_module="utils.helpers",
            function_name="format_text",
            line_number=15,
            import_paths={"from utils.helpers import format_text"},
            is_exported=True,
        )

        # Create module info
        module_info = ModuleInfo(
            module_path="myapp.processor",
            file_path="/myapp/processor.py",
            imports=[import_stmt],
            exports={"process_data"},
            functions={"format_text": func_location},
        )

        # Create state and add module
        state = ContextualMatcherState()
        state.add_module(module_info)

        # Verify everything is connected
        assert "myapp.processor" in state.module_tree
        assert state.module_tree["myapp.processor"] == module_info
        assert "utils.helpers" in state.import_graph["myapp.processor"]

        # Test canonical name generation
        canonical = module_info.get_canonical_name("process_data")
        assert canonical == "myapp.processor.process_data"
