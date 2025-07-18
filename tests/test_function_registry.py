from matcher.function_registry import FunctionRegistry
from matcher.contextual_models import ModuleInfo
from parser import ParsedFunction, FunctionSignature


class TestFunctionRegistry:
    """Test cases for FunctionRegistry class."""

    def test_init_empty_registry(self):
        """Test initialization of empty registry."""
        registry = FunctionRegistry()
        assert len(registry.functions) == 0
        assert len(registry.by_module) == 0
        assert len(registry.by_name) == 0

    def test_register_function_basic(self):
        """Test basic function registration."""
        registry = FunctionRegistry()

        # Create test function
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module.py",
            line_number=10,
            end_line_number=15,
            source_code="def test_func(): pass",
        )

        # Create module info
        module_info = ModuleInfo(
            module_path="test.module",
            file_path="/path/to/module.py",
            exports={"test_func"},
        )

        # Register function
        canonical_name = registry.register_function(function, module_info)

        # Verify registration
        assert canonical_name == "test.module.test_func"
        assert canonical_name in registry.functions
        assert registry.functions[canonical_name].function_name == "test_func"
        assert registry.functions[canonical_name].canonical_module == "test.module"
        assert registry.functions[canonical_name].line_number == 10
        assert registry.functions[canonical_name].is_exported is True

    def test_register_function_private(self):
        """Test registration of private function."""
        registry = FunctionRegistry()

        # Create private function
        function = ParsedFunction(
            signature=FunctionSignature(
                name="_private_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module.py",
            line_number=20,
            end_line_number=25,
            source_code="def _private_func(): pass",
        )

        # Create module info (private function not in exports)
        module_info = ModuleInfo(
            module_path="test.module", file_path="/path/to/module.py", exports=set()
        )

        # Register function
        canonical_name = registry.register_function(function, module_info)

        # Verify registration
        assert canonical_name == "test.module._private_func"
        assert registry.functions[canonical_name].is_exported is False

    def test_register_function_exported_private(self):
        """Test registration of private function that is explicitly exported."""
        registry = FunctionRegistry()

        # Create private function
        function = ParsedFunction(
            signature=FunctionSignature(
                name="_internal_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module.py",
            line_number=30,
            end_line_number=35,
            source_code="def _internal_func(): pass",
        )

        # Create module info (private function is explicitly exported)
        module_info = ModuleInfo(
            module_path="test.module",
            file_path="/path/to/module.py",
            exports={"_internal_func"},
        )

        # Register function
        canonical_name = registry.register_function(function, module_info)

        # Verify registration
        assert canonical_name == "test.module._internal_func"
        assert registry.functions[canonical_name].is_exported is True

    def test_register_function_duplicate_warning(self, caplog):
        """Test duplicate function registration warning."""
        registry = FunctionRegistry()

        # Create test function
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module.py",
            line_number=10,
            end_line_number=15,
            source_code="def test_func(): pass",
        )

        # Create module info
        module_info = ModuleInfo(
            module_path="test.module",
            file_path="/path/to/module.py",
            exports={"test_func"},
        )

        # Register function twice
        registry.register_function(function, module_info)
        registry.register_function(function, module_info)

        # Check warning was logged
        assert "Duplicate function registration" in caplog.text

    def test_register_function_updates_indices(self):
        """Test that function registration updates secondary indices."""
        registry = FunctionRegistry()

        # Create and register first function
        function1 = ParsedFunction(
            signature=FunctionSignature(
                name="func1", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module1.py",
            line_number=10,
            end_line_number=15,
            source_code="def func1(): pass",
        )

        module_info1 = ModuleInfo(
            module_path="test.module1", file_path="/path/to/module1.py"
        )

        registry.register_function(function1, module_info1)

        # Create and register second function (same name, different module)
        function2 = ParsedFunction(
            signature=FunctionSignature(
                name="func1", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module2.py",
            line_number=20,
            end_line_number=25,
            source_code="def func1(): pass",
        )

        module_info2 = ModuleInfo(
            module_path="test.module2", file_path="/path/to/module2.py"
        )

        registry.register_function(function2, module_info2)

        # Verify indices
        assert "test.module1" in registry.by_module
        assert "test.module2" in registry.by_module
        assert "test.module1.func1" in registry.by_module["test.module1"]
        assert "test.module2.func1" in registry.by_module["test.module2"]

        assert "func1" in registry.by_name
        assert len(registry.by_name["func1"]) == 2
        assert "test.module1.func1" in registry.by_name["func1"]
        assert "test.module2.func1" in registry.by_name["func1"]

    def test_find_function_by_canonical_name(self):
        """Test finding function by canonical name."""
        registry = FunctionRegistry()

        # Register function
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module.py",
            line_number=10,
            end_line_number=15,
            source_code="def test_func(): pass",
        )

        module_info = ModuleInfo(
            module_path="test.module", file_path="/path/to/module.py"
        )

        registry.register_function(function, module_info)

        # Find by canonical name
        result = registry.find_function("test.module.test_func")
        assert len(result) == 1
        assert result[0].canonical_module == "test.module"
        assert result[0].function_name == "test_func"

    def test_find_function_by_name_only(self):
        """Test finding function by name only."""
        registry = FunctionRegistry()

        # Register function
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/module.py",
            line_number=10,
            end_line_number=15,
            source_code="def test_func(): pass",
        )

        module_info = ModuleInfo(
            module_path="test.module", file_path="/path/to/module.py"
        )

        registry.register_function(function, module_info)

        # Find by name only
        result = registry.find_function("test_func")
        assert len(result) == 1
        assert result[0].function_name == "test_func"

    def test_find_function_with_hint_module(self):
        """Test finding function with module hint for relevance sorting."""
        registry = FunctionRegistry()

        # Register same function in different modules
        for i, module_name in enumerate(["pkg.module1", "pkg.module2", "other.module"]):
            function = ParsedFunction(
                signature=FunctionSignature(
                    name="common_func", parameters=[], return_type=None, decorators=[]
                ),
                docstring=None,
                file_path=f"/path/to/{module_name.replace('.', '/')}.py",
                line_number=10 + i,
                end_line_number=15 + i,
                source_code="def common_func(): pass",
            )

            module_info = ModuleInfo(
                module_path=module_name,
                file_path=f"/path/to/{module_name.replace('.', '/')}.py",
            )

            registry.register_function(function, module_info)

        # Find with hint - should prioritize exact match
        result = registry.find_function("common_func", hint_module="pkg.module1")
        assert len(result) == 3
        assert result[0].canonical_module == "pkg.module1"  # Exact match first

        # Find with hint - should prioritize child module
        result = registry.find_function("common_func", hint_module="pkg")
        assert len(result) == 3
        # First two should be from pkg.* modules
        assert result[0].canonical_module.startswith("pkg.")
        assert result[1].canonical_module.startswith("pkg.")

    def test_find_function_nonexistent(self):
        """Test finding nonexistent function."""
        registry = FunctionRegistry()

        result = registry.find_function("nonexistent_func")
        assert len(result) == 0

        result = registry.find_function("nonexistent.module.func")
        assert len(result) == 0

    def test_find_moved_function_single_candidate(self):
        """Test finding moved function with single candidate."""
        registry = FunctionRegistry()

        # Register function in new location
        function = ParsedFunction(
            signature=FunctionSignature(
                name="moved_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/new_module.py",
            line_number=10,
            end_line_number=15,
            source_code="def moved_func(): pass",
        )

        module_info = ModuleInfo(
            module_path="new.module", file_path="/path/to/new_module.py"
        )

        registry.register_function(function, module_info)

        # Create old function reference
        old_function = ParsedFunction(
            signature=FunctionSignature(
                name="moved_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/old_module.py",
            line_number=20,
            end_line_number=25,
            source_code="def moved_func(): pass",
        )

        # Find moved function
        result = registry.find_moved_function(old_function, "old.module")
        assert result is not None
        assert result.canonical_module == "new.module"
        assert result.function_name == "moved_func"

    def test_find_moved_function_multiple_candidates(self):
        """Test finding moved function with multiple candidates."""
        registry = FunctionRegistry()

        # Register function in multiple locations
        for i, module_name in enumerate(["new.module1", "new.module2"]):
            function = ParsedFunction(
                signature=FunctionSignature(
                    name="moved_func", parameters=[], return_type=None, decorators=[]
                ),
                docstring=None,
                file_path=f"/path/to/{module_name.replace('.', '/')}.py",
                line_number=10 + i,
                end_line_number=15 + i,
                source_code="def moved_func(): pass",
            )

            module_info = ModuleInfo(
                module_path=module_name,
                file_path=f"/path/to/{module_name.replace('.', '/')}.py",
            )

            registry.register_function(function, module_info)

        # Create old function reference
        old_function = ParsedFunction(
            signature=FunctionSignature(
                name="moved_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/old_module.py",
            line_number=20,
            end_line_number=25,
            source_code="def moved_func(): pass",
        )

        # Find moved function - should return None for multiple candidates
        result = registry.find_moved_function(old_function, "old.module")
        assert result is None

    def test_find_moved_function_no_candidates(self):
        """Test finding moved function with no candidates."""
        registry = FunctionRegistry()

        # Create old function reference
        old_function = ParsedFunction(
            signature=FunctionSignature(
                name="nonexistent_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/old_module.py",
            line_number=20,
            end_line_number=25,
            source_code="def nonexistent_func(): pass",
        )

        # Find moved function - should return None
        result = registry.find_moved_function(old_function, "old.module")
        assert result is None

    def test_find_moved_function_filters_original_location(self):
        """Test that find_moved_function filters out original location."""
        registry = FunctionRegistry()

        # Register function in original location
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func", parameters=[], return_type=None, decorators=[]
            ),
            docstring=None,
            file_path="/path/to/original.py",
            line_number=10,
            end_line_number=15,
            source_code="def test_func(): pass",
        )

        module_info = ModuleInfo(
            module_path="original.module", file_path="/path/to/original.py"
        )

        registry.register_function(function, module_info)

        # Find moved function - should return None as only original location exists
        result = registry.find_moved_function(function, "original.module")
        assert result is None

    def test_get_module_functions(self):
        """Test getting all functions in a module."""
        registry = FunctionRegistry()

        # Register multiple functions in same module
        for i, func_name in enumerate(["func1", "func2", "func3"]):
            function = ParsedFunction(
                signature=FunctionSignature(
                    name=func_name, parameters=[], return_type=None, decorators=[]
                ),
                docstring=None,
                file_path="/path/to/module.py",
                line_number=10 + i * 5,
                end_line_number=15 + i * 5,
                source_code=f"def {func_name}(): pass",
            )

            module_info = ModuleInfo(
                module_path="test.module", file_path="/path/to/module.py"
            )

            registry.register_function(function, module_info)

        # Get all functions in module
        result = registry.get_module_functions("test.module")
        assert len(result) == 3

        function_names = {loc.function_name for loc in result}
        assert function_names == {"func1", "func2", "func3"}

    def test_get_module_functions_empty_module(self):
        """Test getting functions from empty module."""
        registry = FunctionRegistry()

        result = registry.get_module_functions("nonexistent.module")
        assert len(result) == 0
