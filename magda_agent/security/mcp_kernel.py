import ast
from typing import Any, Dict, Set, Optional

class MCPKernel:
    """
    MCPKernel provides an isolated execution environment for code blocks.
    It uses strict taint tracking to prevent side-channel leaks and unsafe operations
    during execution. Inspired by MCPKernel trends.
    """

    def __init__(self, allowed_builtins: Optional[Set[str]] = None):
        """
        Initializes the MCPKernel with a defined set of allowed builtins.

        Args:
            allowed_builtins: A set of allowed builtin function names. If None, defaults to a safe set.
        """
        self.allowed_builtins = allowed_builtins or {"print", "len", "range", "int", "str", "float", "list", "dict", "set", "tuple", "bool"}
        self.unsafe_calls = {"open", "exec", "eval", "__import__"}

    def is_safe(self, code: str) -> bool:
        """
        Checks if the provided code contains any unsafe operations via AST parsing.

        Args:
            code: The Python code to analyze.

        Returns:
            True if the code is deemed safe, False otherwise.
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.unsafe_calls or func_name not in self.allowed_builtins:
                            return False
                    elif isinstance(node.func, ast.Attribute):
                        # Block accessing methods that might bypass the sandbox
                        attr_name = node.func.attr
                        if attr_name in self.unsafe_calls:
                            return False
                    else:
                        # Block any other callable types (like subscripting dicts to get builtins)
                        return False
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # Blocking all imports for strict isolation by default
                    return False
                elif isinstance(node, ast.Attribute):
                    # Block accessing dangerous attributes
                    if node.attr.startswith('__') and node.attr.endswith('__'):
                        return False
            return True
        except SyntaxError:
            return False

    def execute(self, code: str, globals_dict: Optional[Dict[str, Any]] = None, locals_dict: Optional[Dict[str, Any]] = None) -> Any:
        """
        Executes the provided code block if it passes safety checks.

        Args:
            code: The Python code to execute.
            globals_dict: Optional dictionary to use as globals.
            locals_dict: Optional dictionary to use as locals.

        Returns:
            The result of the execution if safe.

        Raises:
            SecurityError: If the code is deemed unsafe.
        """
        if not self.is_safe(code):
            raise SecurityError("Code contains unsafe operations and was blocked by MCPKernel taint tracking.")

        if globals_dict is None:
            # Create a restricted globals dict
            globals_dict = {"__builtins__": {k: __builtins__[k] for k in self.allowed_builtins if k in __builtins__}}

        if locals_dict is None:
            locals_dict = {}

        try:
            exec(code, globals_dict, locals_dict)
            return locals_dict
        except Exception as e:
             raise RuntimeError(f"Execution failed: {e}")

class SecurityError(Exception):
    """Exception raised for security violations in MCPKernel."""
    pass
