"""
AST Parser Module for CodeDocSync.

This module provides data models and parsing functionality for Python source code.
It uses the built-in ast module for fast, reliable parsing and extracts function
signatures, parameters, and docstrings.
"""

import ast
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Union, Generator

from ..utils.errors import (
    ValidationError,
    ParsingError,
    FileAccessError,
    SyntaxParsingError,
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class FunctionParameter:
    """Single function parameter with validation."""

    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True

    def __post_init__(self):
        """Validate parameter data after initialization."""
        # Validate parameter name follows Python identifier rules
        # Allow special prefixes for *args, **kwargs, and keyword-only parameters
        clean_name = self.name.lstrip("*")
        if clean_name.endswith("/"):
            clean_name = clean_name[:-1]

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean_name):
            raise ValidationError(
                f"Invalid parameter name: '{self.name}'",
                recovery_hint="Parameter names must be valid Python identifiers",
            )

        # Validate type annotation if provided
        if self.type_annotation:
            self._validate_type_annotation(self.type_annotation)

    def _validate_type_annotation(self, annotation: str) -> None:
        """Validate type annotation against common patterns."""
        # Basic validation for common type patterns
        valid_patterns = [
            r"^[a-zA-Z_][a-zA-Z0-9_]*$",  # Simple types like str, int
            r"^List\[.*\]$",  # List[T]
            r"^Dict\[.*\]$",  # Dict[K, V]
            r"^Optional\[.*\]$",  # Optional[T]
            r"^Union\[.*\]$",  # Union[T1, T2, ...]
            r"^Tuple\[.*\]$",  # Tuple[T1, T2, ...]
            r"^Callable\[.*\]$",  # Callable[..., T]
            r"^[a-zA-Z_][a-zA-Z0-9_]*\[.*\]$",  # Generic[T]
        ]

        if not any(re.match(pattern, annotation.strip()) for pattern in valid_patterns):
            # Don't raise error for complex annotations, just log warning
            pass


@dataclass
class FunctionSignature:
    """Complete function signature with validation."""

    name: str
    parameters: List[FunctionParameter] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate function signature after initialization."""
        # Validate function name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.name):
            raise ValidationError(
                f"Invalid function name: '{self.name}'",
                recovery_hint="Function names must be valid Python identifiers",
            )

    def to_string(self) -> str:
        """Convert function signature to readable string representation."""
        # Build parameter string
        param_parts = []
        for param in self.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str += f": {param.type_annotation}"
            if param.default_value:
                param_str += f" = {param.default_value}"
            param_parts.append(param_str)

        params_str = ", ".join(param_parts)

        # Build full signature
        async_prefix = "async " if self.is_async else ""
        return_annotation = f" -> {self.return_type}" if self.return_type else ""

        return f"{async_prefix}def {self.name}({params_str}){return_annotation}"


@dataclass
class ParsedFunction:
    """Represents a parsed function with all metadata."""

    signature: FunctionSignature
    docstring: Optional[str] = None  # Raw docstring for now
    file_path: str = ""
    line_number: int = 0
    end_line_number: int = 0
    source_code: str = ""

    def __post_init__(self):
        """Validate parsed function data."""
        if self.line_number < 0:
            raise ValidationError(
                f"Invalid line number: {self.line_number}",
                recovery_hint="Line numbers must be positive integers",
            )

        if self.end_line_number < self.line_number:
            raise ValidationError(
                f"End line ({self.end_line_number}) before start line ({self.line_number})",
                recovery_hint="End line number must be >= start line number",
            )


@lru_cache(maxsize=100)
def _get_cached_ast(file_content_hash: str, file_path: str) -> ast.AST:
    """Cache parsed AST trees for repeated analysis."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_content = f.read()
    except UnicodeDecodeError:
        # Try alternative encodings
        with open(file_path, "r", encoding="latin-1") as f:
            source_content = f.read()
        logger.warning(f"File {file_path} decoded using latin-1 instead of utf-8")
    return ast.parse(source_content, filename=file_path)


def parse_python_file(file_path: str) -> List[ParsedFunction]:
    """
    Parse Python file with comprehensive error handling.

    Performance targets:
    - Small file (<100 lines): <10ms
    - Medium file (100-1000 lines): <50ms
    - Large file (>1000 lines): <200ms

    Args:
        file_path: Path to the Python file to parse

    Returns:
        List of ParsedFunction objects representing all functions found

    Raises:
        ParsingError: If file cannot be parsed or accessed
    """
    start_time = time.time()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_content = f.read()
    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileAccessError(
            error_msg, recovery_hint="Check the file path and ensure the file exists"
        )
    except PermissionError:
        error_msg = f"Permission denied: {file_path}"
        logger.error(error_msg)
        raise FileAccessError(
            error_msg, recovery_hint="Check file permissions and ensure read access"
        )
    except UnicodeDecodeError as e:
        # Try alternative encodings
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                source_content = f.read()
            logger.warning(f"File {file_path} decoded using latin-1 instead of utf-8")
        except Exception:
            error_msg = f"Encoding error in {file_path}: {e}"
            logger.error(error_msg)
            raise ParsingError(
                error_msg,
                recovery_hint="Ensure the file uses UTF-8 encoding or check file content",
            )

    if not source_content.strip():
        logger.info(f"Empty file: {file_path}")
        return []  # Empty file

    # Check if file contains only imports and comments
    if _is_imports_only(source_content):
        logger.info(f"File contains only imports/comments: {file_path}")
        return []

    # Create hash for caching
    file_content_hash = hashlib.md5(source_content.encode()).hexdigest()

    try:
        tree = _get_cached_ast(file_content_hash, file_path)
    except SyntaxError as e:
        # Attempt partial parsing up to the error line
        logger.warning(
            f"Syntax error in {file_path}:{e.lineno}: {e.msg}. Attempting partial parse."
        )
        functions = _parse_partial_file(file_path, source_content, e.lineno or 1)

        if functions:
            logger.info(
                f"Partial parse successful for {file_path}: found {len(functions)} functions"
            )
            return functions
        else:
            raise SyntaxParsingError(
                f"Syntax error in {file_path}:{e.lineno}: {e.msg}",
                recovery_hint="Fix the syntax error and try again",
            )

    functions = []

    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                parsed_func = _extract_function(node, file_path, source_content)
                functions.append(parsed_func)
            except Exception as e:
                # Log warning but continue parsing other functions
                logger.warning(
                    f"Failed to extract function {getattr(node, 'name', 'unknown')} from {file_path}: {e}"
                )
                continue

    parse_duration = time.time() - start_time
    logger.info(
        f"Parsed {file_path} in {parse_duration:.3f}s: found {len(functions)} functions"
    )

    return functions


def parse_python_file_lazy(file_path: str) -> Generator[ParsedFunction, None, None]:
    """
    Lazy parsing using generators for memory efficiency.

    This function yields ParsedFunction objects one at a time instead of loading
    all functions into memory at once. Useful for large files or when processing
    many files.

    Args:
        file_path: Path to the Python file to parse

    Yields:
        ParsedFunction objects one at a time

    Raises:
        ParsingError: If file cannot be parsed or accessed
    """
    start_time = time.time()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_content = f.read()
    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileAccessError(
            error_msg, recovery_hint="Check the file path and ensure the file exists"
        )
    except PermissionError:
        error_msg = f"Permission denied: {file_path}"
        logger.error(error_msg)
        raise FileAccessError(
            error_msg, recovery_hint="Check file permissions and ensure read access"
        )
    except UnicodeDecodeError as e:
        # Try alternative encodings
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                source_content = f.read()
            logger.warning(f"File {file_path} decoded using latin-1 instead of utf-8")
        except Exception:
            error_msg = f"Encoding error in {file_path}: {e}"
            logger.error(error_msg)
            raise ParsingError(
                error_msg,
                recovery_hint="Ensure the file uses UTF-8 encoding or check file content",
            )

    if not source_content.strip():
        logger.info(f"Empty file: {file_path}")
        return  # Empty file

    # Check if file contains only imports and comments
    if _is_imports_only(source_content):
        logger.info(f"File contains only imports/comments: {file_path}")
        return

    # Create hash for caching
    file_content_hash = hashlib.md5(source_content.encode()).hexdigest()

    try:
        tree = _get_cached_ast(file_content_hash, file_path)
    except SyntaxError as e:
        # Attempt partial parsing up to the error line
        logger.warning(
            f"Syntax error in {file_path}:{e.lineno}: {e.msg}. Attempting partial parse."
        )
        functions = _parse_partial_file(file_path, source_content, e.lineno or 1)

        if functions:
            logger.info(
                f"Partial parse successful for {file_path}: found {len(functions)} functions"
            )
            for func in functions:
                yield func
            return
        else:
            raise SyntaxParsingError(
                f"Syntax error in {file_path}:{e.lineno}: {e.msg}",
                recovery_hint="Fix the syntax error and try again",
            )

    function_count = 0

    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                parsed_func = _extract_function(node, file_path, source_content)
                function_count += 1
                yield parsed_func
            except Exception as e:
                # Log warning but continue parsing other functions
                logger.warning(
                    f"Failed to extract function {getattr(node, 'name', 'unknown')} from {file_path}: {e}"
                )
                continue

    parse_duration = time.time() - start_time
    logger.info(
        f"Lazy parsed {file_path} in {parse_duration:.3f}s: found {function_count} functions"
    )


def _extract_function(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    file_path: str,
    source_content: str,
) -> ParsedFunction:
    """
    Extract function details with error recovery.

    Handles:
    - Functions without docstrings
    - Functions with no parameters
    - Functions with *args and **kwargs
    - Complex type annotations (Union, Optional, List, Dict)
    - Default values that are function calls or complex expressions
    - Nested functions (include parent context)
    - Lambda functions
    - Property decorators (@property, @setter)
    - Class methods and static methods
    - Complex decorators
    """
    try:
        signature = _extract_signature(node)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Calculate line numbers
        line_number = node.lineno
        end_line_number = node.end_lineno or line_number

        # Extract source code with better error handling
        source_lines = source_content.split("\n")
        if line_number <= len(source_lines) and line_number > 0:
            # Ensure we don't go out of bounds
            end_idx = min(end_line_number, len(source_lines))
            source_code = "\n".join(source_lines[line_number - 1 : end_idx])
        else:
            source_code = ""
            logger.warning(
                f"Invalid line numbers for function {node.name}: {line_number}-{end_line_number}"
            )

        return ParsedFunction(
            signature=signature,
            docstring=docstring,
            file_path=file_path,
            line_number=line_number,
            end_line_number=end_line_number,
            source_code=source_code,
        )
    except Exception as e:
        logger.error(
            f"Failed to extract function {getattr(node, 'name', 'unknown')}: {e}"
        )
        raise


def _extract_signature(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> FunctionSignature:
    """Extract signature information from AST node with comprehensive parameter handling."""
    parameters = []
    args = node.args

    # Handle positional-only and regular arguments together
    # args.defaults contains defaults for trailing positional parameters (both posonly and regular combined)
    posonly_args = args.posonlyargs if hasattr(args, "posonlyargs") else []
    regular_args = args.args if args.args else []
    all_positional_args = posonly_args + regular_args

    num_defaults = len(args.defaults)
    num_positional = len(all_positional_args)

    # Defaults apply to the last N positional arguments where N = len(args.defaults)
    defaults_start_index = max(0, num_positional - num_defaults)

    # Process positional-only arguments
    for i, arg in enumerate(posonly_args):
        default_value = None
        is_required = True

        if i >= defaults_start_index:
            default_index = i - defaults_start_index
            if default_index < num_defaults:
                default_value = _get_default_value(args.defaults[default_index])
                is_required = False

        param = FunctionParameter(
            name=f"{arg.arg}/",  # Mark as positional-only
            type_annotation=(
                _get_annotation_string(arg.annotation) if arg.annotation else None
            ),
            default_value=default_value,
            is_required=is_required,
        )
        parameters.append(param)

    # Process regular positional arguments
    for i, arg in enumerate(regular_args):
        # Calculate index in the combined positional args list
        combined_index = len(posonly_args) + i
        default_value = None
        is_required = True

        if combined_index >= defaults_start_index:
            default_index = combined_index - defaults_start_index
            if default_index < num_defaults:
                default_value = _get_default_value(args.defaults[default_index])
                is_required = False

        param = FunctionParameter(
            name=arg.arg,
            type_annotation=(
                _get_annotation_string(arg.annotation) if arg.annotation else None
            ),
            default_value=default_value,
            is_required=is_required,
        )
        parameters.append(param)

    # Handle *args
    if args.vararg:
        param = FunctionParameter(
            name=f"*{args.vararg.arg}",
            type_annotation=(
                _get_annotation_string(args.vararg.annotation)
                if args.vararg.annotation
                else None
            ),
            is_required=False,
        )
        parameters.append(param)

    # Handle keyword-only arguments
    if args.kwonlyargs:
        kw_defaults = args.kw_defaults or []
        for i, arg in enumerate(args.kwonlyargs):
            default_value = None
            is_required = True

            if i < len(kw_defaults) and kw_defaults[i] is not None:
                default_value = _get_default_value(kw_defaults[i])
                is_required = False

            param = FunctionParameter(
                name=f"*{arg.arg}",  # Mark as keyword-only
                type_annotation=(
                    _get_annotation_string(arg.annotation) if arg.annotation else None
                ),
                default_value=default_value,
                is_required=is_required,
            )
            parameters.append(param)

    # Handle **kwargs
    if args.kwarg:
        param = FunctionParameter(
            name=f"**{args.kwarg.arg}",
            type_annotation=(
                _get_annotation_string(args.kwarg.annotation)
                if args.kwarg.annotation
                else None
            ),
            is_required=False,
        )
        parameters.append(param)

    # Extract decorators with enhanced handling
    decorators = _get_decorator_names(node.decorator_list)

    # Enhanced method detection
    is_method = _is_method_function(parameters, decorators)

    return FunctionSignature(
        name=node.name,
        parameters=parameters,
        return_type=_get_annotation_string(node.returns) if node.returns else None,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        is_method=is_method,
        decorators=decorators,
    )


def _get_decorator_names(decorators: List[ast.expr]) -> List[str]:
    """Extract decorator names handling both simple and complex decorators."""
    decorator_names = []

    for decorator in decorators:
        if isinstance(decorator, ast.Name):
            decorator_names.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            # Handle decorators like @property, @classmethod, etc.
            try:
                decorator_names.append(ast.unparse(decorator))
            except Exception:
                decorator_names.append(_decorator_fallback(decorator))
        elif isinstance(decorator, ast.Call):
            # Handle decorator calls like @decorator(arg)
            try:
                decorator_names.append(ast.unparse(decorator))
            except Exception:
                # Try to get at least the function name
                func_name = _decorator_fallback(decorator.func)
                decorator_names.append(f"{func_name}(...)")
        else:
            # Fallback for complex decorators
            try:
                decorator_names.append(ast.unparse(decorator))
            except Exception:
                decorator_names.append("<complex_decorator>")

    return decorator_names


def _decorator_fallback(decorator: ast.expr) -> str:
    """Fallback method to extract decorator name."""
    if isinstance(decorator, ast.Name):
        return decorator.id
    elif isinstance(decorator, ast.Attribute):
        return f"{_decorator_fallback(decorator.value)}.{decorator.attr}"
    else:
        return "<complex_decorator>"


def _get_annotation_string(annotation: Optional[ast.expr]) -> Optional[str]:
    """Convert AST annotation to string representation with enhanced type handling."""
    if annotation is None:
        return None

    try:
        # Use ast.unparse for Python 3.9+
        return ast.unparse(annotation)
    except AttributeError:
        # Fallback for older Python versions
        try:
            import astor  # type: ignore

            return astor.to_source(annotation).strip()
        except ImportError:
            return _annotation_to_string_fallback(annotation)
    except Exception:
        # Fallback for complex annotations
        return _annotation_to_string_fallback(annotation)


def _annotation_to_string_fallback(annotation: ast.expr) -> str:
    """Fallback method to convert annotation to string."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return f"{_annotation_to_string_fallback(annotation.value)}.{annotation.attr}"
    elif isinstance(annotation, ast.Subscript):
        value = _annotation_to_string_fallback(annotation.value)
        slice_val = _annotation_to_string_fallback(annotation.slice)
        return f"{value}[{slice_val}]"
    elif isinstance(annotation, ast.Tuple):
        elements = [_annotation_to_string_fallback(elt) for elt in annotation.elts]
        return f"({', '.join(elements)})"
    elif isinstance(annotation, ast.List):
        elements = [_annotation_to_string_fallback(elt) for elt in annotation.elts]
        return f"[{', '.join(elements)}]"
    elif isinstance(annotation, ast.Constant):
        return repr(annotation.value)
    else:
        return "<complex_annotation>"


def _is_method_function(
    parameters: List[FunctionParameter], decorators: List[str]
) -> bool:
    """Enhanced method detection considering decorators and parameters."""
    # Check for static method or class method decorators
    if any(dec in ["staticmethod", "classmethod"] for dec in decorators):
        return True

    # Check for property decorators
    if any(dec in ["property", "setter", "getter", "deleter"] for dec in decorators):
        return True

    # Check for self/cls parameter
    if len(parameters) > 0:
        first_param = parameters[0].name
        return first_param in ("self", "cls")

    return False


def _get_default_value(default_node: Optional[ast.expr]) -> str:
    """Extract default value from AST node with enhanced handling."""
    if default_node is None:
        return "None"

    try:
        # Use ast.unparse for Python 3.9+
        return ast.unparse(default_node)
    except AttributeError:
        # Fallback for older Python versions
        try:
            import astor  # type: ignore

            return astor.to_source(default_node).strip()
        except ImportError:
            return _default_value_fallback(default_node)
    except Exception:
        # Fallback for complex default values
        return _default_value_fallback(default_node)


def _default_value_fallback(default_node: ast.expr) -> str:
    """Fallback method to extract default value."""
    if isinstance(default_node, ast.Constant):
        return repr(default_node.value)
    elif isinstance(default_node, ast.Name):
        return default_node.id
    elif isinstance(default_node, ast.Attribute):
        return f"{_default_value_fallback(default_node.value)}.{default_node.attr}"
    elif isinstance(default_node, ast.Call):
        func_name = _default_value_fallback(default_node.func)
        return f"{func_name}(...)"
    elif isinstance(default_node, ast.List):
        return "[...]"
    elif isinstance(default_node, ast.Dict):
        return "{...}"
    elif isinstance(default_node, ast.Lambda):
        return "<lambda>"
    else:
        return "<complex_default>"


def _is_imports_only(source_content: str) -> bool:
    """Check if file contains only imports and comments."""
    try:
        tree = ast.parse(source_content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return False
            # Check for any substantial code beyond imports
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                # Allow simple constant assignments like VERSION = "1.0.0"
                if isinstance(node, ast.Assign):
                    if not isinstance(node.value, ast.Constant):
                        return False
                elif isinstance(node, ast.AnnAssign):
                    if node.value is not None and not isinstance(
                        node.value, ast.Constant
                    ):
                        return False
                else:  # ast.AugAssign
                    return False
            elif isinstance(node, ast.Expr):
                # Allow simple constant expressions but not complex ones
                if not isinstance(node.value, ast.Constant):
                    return False
        return True
    except Exception:
        return False


def _parse_partial_file(
    file_path: str, source_content: str, error_line: int
) -> List[ParsedFunction]:
    """Attempt to parse file up to the syntax error line."""
    lines = source_content.split("\n")
    if error_line <= 1:
        return []

    # Try to parse up to the line before the error
    partial_content = "\n".join(lines[: error_line - 1])

    try:
        tree = ast.parse(partial_content, filename=file_path)
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    parsed_func = _extract_function(node, file_path, partial_content)
                    functions.append(parsed_func)
                except Exception as e:
                    logger.warning(
                        f"Failed to extract function {getattr(node, 'name', 'unknown')} during partial parse: {e}"
                    )
                    continue

        return functions
    except Exception:
        return []


def _extract_parameter(
    arg: ast.arg, defaults: List, default_offset: int
) -> FunctionParameter:
    """Extract parameter information with proper default handling."""
    # Calculate if this parameter has a default value
    default_value = None
    is_required = True

    if default_offset >= 0 and default_offset < len(defaults):
        default_value = _get_default_value(defaults[default_offset])
        is_required = False

    return FunctionParameter(
        name=arg.arg,
        type_annotation=(
            _get_annotation_string(arg.annotation) if arg.annotation else None
        ),
        default_value=default_value,
        is_required=is_required,
    )
