"""Test suite for import parser."""

import tempfile
from pathlib import Path

from codedocsync.matcher.import_parser import ImportParser
from codedocsync.matcher.contextual_models import ImportType


class TestImportParser:
    """Test ImportParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ImportParser()

    def test_parse_standard_import(self):
        """Test parsing standard import statements."""
        content = """import os
import sys
import json as js
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(imports) == 3

        # Check os import
        os_import = imports[0]
        assert os_import.import_type == ImportType.STANDARD
        assert os_import.module_path == "os"
        assert os_import.imported_names == []
        assert os_import.aliases == {}
        assert os_import.line_number == 1

        # Check sys import
        sys_import = imports[1]
        assert sys_import.import_type == ImportType.STANDARD
        assert sys_import.module_path == "sys"

        # Check json import with alias
        json_import = imports[2]
        assert json_import.import_type == ImportType.STANDARD
        assert json_import.module_path == "json"
        assert json_import.aliases == {"js": "json"}

    def test_parse_from_import(self):
        """Test parsing from import statements."""
        content = """from os import path
from sys import argv, exit
from json import loads as json_loads, dumps
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(imports) == 3

        # Check os.path import
        os_import = imports[0]
        assert os_import.import_type == ImportType.FROM
        assert os_import.module_path == "os"
        assert os_import.imported_names == ["path"]
        assert os_import.aliases == {}

        # Check sys import with multiple names
        sys_import = imports[1]
        assert sys_import.import_type == ImportType.FROM
        assert sys_import.module_path == "sys"
        assert sys_import.imported_names == ["argv", "exit"]

        # Check json import with alias
        json_import = imports[2]
        assert json_import.import_type == ImportType.FROM
        assert json_import.module_path == "json"
        assert json_import.imported_names == ["loads", "dumps"]
        assert json_import.aliases == {"json_loads": "loads"}

    def test_parse_relative_import(self):
        """Test parsing relative import statements."""
        content = """from . import utils
from .. import helpers
from ...core.models import User
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(imports) == 3

        # Check single dot import
        single_dot = imports[0]
        assert single_dot.import_type == ImportType.RELATIVE
        assert single_dot.module_path == "."
        assert single_dot.imported_names == ["utils"]
        assert single_dot.level == 1

        # Check double dot import
        double_dot = imports[1]
        assert double_dot.import_type == ImportType.RELATIVE
        assert double_dot.module_path == ".."
        assert double_dot.level == 2

        # Check deep relative import
        deep_import = imports[2]
        assert deep_import.import_type == ImportType.RELATIVE
        assert deep_import.module_path == "core.models"
        assert deep_import.imported_names == ["User"]
        assert deep_import.level == 3

    def test_parse_wildcard_import(self):
        """Test parsing wildcard import statements."""
        content = """from os import *
from utils.helpers import *
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(imports) == 2

        # Check os wildcard import
        os_import = imports[0]
        assert os_import.import_type == ImportType.WILDCARD
        assert os_import.module_path == "os"
        assert os_import.imported_names == ["*"]

        # Check utils wildcard import
        utils_import = imports[1]
        assert utils_import.import_type == ImportType.WILDCARD
        assert utils_import.module_path == "utils.helpers"
        assert utils_import.imported_names == ["*"]

    def test_extract_exports_with_all(self):
        """Test extracting exports from __all__."""
        content = """
def helper_func():
    pass

def _private_func():
    pass

class UtilityClass:
    pass

__all__ = ["helper_func", "UtilityClass"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert exports == {"helper_func", "UtilityClass"}
        assert "_private_func" not in exports

    def test_extract_exports_public_names(self):
        """Test extracting public names when no __all__ exists."""
        content = """
def helper_func():
    pass

def _private_func():
    pass

class UtilityClass:
    pass

class _PrivateClass:
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert exports == {"helper_func", "UtilityClass"}
        assert "_private_func" not in exports
        assert "_PrivateClass" not in exports

    def test_extract_exports_with_async_functions(self):
        """Test extracting exports including async functions."""
        content = """
def sync_func():
    pass

async def async_func():
    pass

def _private_func():
    pass

async def _private_async_func():
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert exports == {"sync_func", "async_func"}
        assert "_private_func" not in exports
        assert "_private_async_func" not in exports

    def test_build_module_info_regular_file(self):
        """Test building module info for regular Python file."""
        content = """
import os
from sys import argv

def helper_func():
    pass

__all__ = ["helper_func"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            module_info = self.parser.build_module_info(f.name, "test.module")

        # Clean up
        Path(f.name).unlink()

        assert module_info.module_path == "test.module"
        assert module_info.file_path == f.name
        assert len(module_info.imports) == 2
        assert module_info.exports == {"helper_func"}
        assert not module_info.is_package

    def test_build_module_info_package_file(self):
        """Test building module info for package __init__.py file."""
        content = """
from .utils import helper_func
from .models import DataModel

__all__ = ["helper_func", "DataModel"]
"""

        # Create a temporary directory and __init__.py file
        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = Path(temp_dir) / "__init__.py"
            init_file.write_text(content)

            module_info = self.parser.build_module_info(str(init_file), "mypackage")

        assert module_info.module_path == "mypackage"
        assert module_info.file_path == str(init_file)
        assert len(module_info.imports) == 2
        assert module_info.exports == {"helper_func", "DataModel"}
        assert module_info.is_package

    def test_parse_complex_imports(self):
        """Test parsing complex import combinations."""
        content = """
import os
import sys as system
from pathlib import Path
from typing import List, Dict, Optional
from . import utils
from ..core import models
from json import loads as json_loads, dumps as json_dumps
from collections import *
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(imports) == 8

        # Check various import types are present
        import_types = [imp.import_type for imp in imports]
        assert ImportType.STANDARD in import_types
        assert ImportType.FROM in import_types
        assert ImportType.RELATIVE in import_types
        assert ImportType.WILDCARD in import_types

        # Check aliases are properly captured
        aliases = {}
        for imp in imports:
            aliases.update(imp.aliases)

        assert "system" in aliases
        assert "json_loads" in aliases
        assert "json_dumps" in aliases

    def test_parse_empty_file(self):
        """Test parsing empty file."""
        content = ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        assert imports == []
        assert exports == set()

    def test_parse_syntax_error_file(self):
        """Test parsing file with syntax error."""
        content = """
import os
def invalid_syntax(:
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        # Should handle syntax error gracefully
        assert imports == []
        assert exports == set()

    def test_parse_encoding_issues(self):
        """Test parsing files with encoding issues."""
        # Create file with non-UTF8 content
        content = "# -*- coding: latin-1 -*-\nimport os\n# Special character: \xe9\n"

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(content.encode("latin-1"))
            f.flush()

            imports, exports = self.parser.parse_imports(f.name)

        # Clean up
        Path(f.name).unlink()

        # Should handle encoding gracefully and still parse imports
        assert len(imports) == 1
        assert imports[0].module_path == "os"

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file."""
        imports, exports = self.parser.parse_imports("/nonexistent/file.py")

        assert imports == []
        assert exports == set()


class TestImportParserIntegration:
    """Test integration scenarios."""

    def test_full_module_analysis(self):
        """Test complete module analysis workflow."""
        content = """
'''Module docstring for test module.'''

import os
import sys as system
from pathlib import Path
from typing import List, Dict
from . import utils
from ..models import User

def public_function():
    '''A public function.'''
    pass

def _private_function():
    '''A private function.'''
    pass

class PublicClass:
    '''A public class.'''
    pass

class _PrivateClass:
    '''A private class.'''
    pass

__all__ = ["public_function", "PublicClass"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()

            parser = ImportParser()
            imports, exports = parser.parse_imports(f.name)
            module_info = parser.build_module_info(f.name, "test.analysis")

        # Clean up
        Path(f.name).unlink()

        # Verify imports
        assert len(imports) == 6

        # Check specific imports
        import_modules = [imp.module_path for imp in imports]
        assert "os" in import_modules
        assert "sys" in import_modules
        assert "pathlib" in import_modules
        assert "typing" in import_modules
        assert "." in import_modules
        assert "models" in import_modules

        # Check aliases
        all_aliases = {}
        for imp in imports:
            all_aliases.update(imp.aliases)
        assert "system" in all_aliases

        # Check exports
        assert exports == {"public_function", "PublicClass"}

        # Check module info
        assert module_info.module_path == "test.analysis"
        assert module_info.file_path == f.name
        assert module_info.imports == imports
        assert module_info.exports == exports
        assert not module_info.is_package
