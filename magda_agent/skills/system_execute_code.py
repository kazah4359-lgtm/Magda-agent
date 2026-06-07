import ast
import contextlib
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

SANDBOX_DIR = "/tmp/sandbox"
TIMEOUT_SECONDS = 10.0

_SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "open", "print", "range", "round", "set", "str",
    "sum", "tuple",
}
_ALLOWED_FILE_METHODS = {"read", "write", "close"}
_DENIED_NAMES = {
    "__builtins__", "__import__", "breakpoint", "classmethod", "compile", "delattr",
    "dir", "eval", "exec", "getattr", "globals", "hasattr", "help", "input",
    "locals", "memoryview", "object", "property", "setattr", "staticmethod", "super",
    "type", "vars",
}


class SandboxSecurityError(Exception):
    """Raised when code uses syntax outside the Python-only sandbox subset."""


class _SandboxValidator(ast.NodeVisitor):
    """Allow only a small, data-oriented Python subset before execution."""

    _allowed_nodes = (
        ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name,
        ast.Load, ast.Store, ast.Del, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp,
        ast.Compare, ast.If, ast.While, ast.For, ast.Break, ast.Continue, ast.Pass,
        ast.With, ast.withitem, ast.Call, ast.keyword, ast.List, ast.Tuple, ast.Set,
        ast.Dict, ast.Subscript, ast.Slice, ast.ListComp, ast.SetComp, ast.DictComp,
        ast.comprehension, ast.JoinedStr, ast.FormattedValue,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    )

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self._allowed_nodes):
            raise SandboxSecurityError(f"Blocked unsafe syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        raise SandboxSecurityError("PermissionError: imports are disabled in this sandbox.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SandboxSecurityError("PermissionError: imports are disabled in this sandbox.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr not in _ALLOWED_FILE_METHODS:
            raise SandboxSecurityError(f"Blocked unsafe attribute access: {node.attr}")
        if isinstance(node.value, ast.Name):
            self.visit(node.value)
            return
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "open":
            self.visit_Call(node.value)
            return
        raise SandboxSecurityError("Blocked chained attribute access")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_") or node.id in _DENIED_NAMES:
            raise SandboxSecurityError(f"Blocked unsafe name: {node.id}")

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assignment_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assignment_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_assignment_target(node.target)
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_BUILTIN_NAMES:
                raise SandboxSecurityError(f"Blocked unsafe call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
        else:
            raise SandboxSecurityError("Blocked dynamic callable")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword)

    def _check_assignment_target(self, target: ast.AST) -> None:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                if child.id in _SAFE_BUILTIN_NAMES or child.id.startswith("_") or child.id in _DENIED_NAMES:
                    raise SandboxSecurityError(f"Blocked unsafe assignment target: {child.id}")
            elif isinstance(child, ast.Attribute):
                raise SandboxSecurityError("Blocked attribute assignment")


def set_limits() -> None:
    """Set resource limits for the subprocess to restrict CPU and memory."""
    if not HAS_RESOURCE:
        return
    for limit, value in (
        (resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024)),
        (resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024)),
        (resource.RLIMIT_NPROC, (0, 0)),
    ):
        try:
            resource.setrlimit(limit, value)
        except Exception:
            pass


def _safe_open_factory(sandbox_dir: str):
    sandbox_root = Path(sandbox_dir).resolve()

    def safe_open(file: str, mode: str = "r", *args: Any, **kwargs: Any):
        requested = Path(file)
        if not requested.is_absolute():
            requested = sandbox_root / requested
        resolved_parent = requested.parent.resolve()
        resolved_path = resolved_parent / requested.name
        if sandbox_root != resolved_path and sandbox_root not in resolved_path.parents:
            raise PermissionError(f"Access denied. Cannot access files outside {sandbox_root}")
        return open(resolved_path, mode, *args, **kwargs)

    return safe_open


def _restricted_builtins(sandbox_dir: str) -> Dict[str, Any]:
    import builtins
    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if name != "open"}
    safe["open"] = _safe_open_factory(sandbox_dir)
    return safe


def _execute_restricted(code: str, sandbox_dir: str = SANDBOX_DIR) -> str:
    if "os.system" in code:
        raise SandboxSecurityError("os.system is disabled")
    tree = ast.parse(code, mode="exec")
    _SandboxValidator().visit(tree)
    compiled = compile(tree, "<sandbox>", "exec")
    stdout = io.StringIO()
    globals_dict = {"__builtins__": _restricted_builtins(sandbox_dir)}
    locals_dict: Dict[str, Any] = {}
    with contextlib.redirect_stdout(stdout):
        exec(compiled, globals_dict, locals_dict)
    return stdout.getvalue()


def execute(code: str) -> str:
    """
    Executes a small, Python-only subset in a subprocess with resource limits.

    This intentionally does not try to make arbitrary Python safe. Imports,
    dynamic builtins, ctypes, subprocess primitives, sockets, and object/frame
    attribute traversal are rejected before execution.
    """
    sandbox_dir = SANDBOX_DIR
    os.makedirs(sandbox_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=sandbox_dir, delete=False) as script_file:
        script_file.write(code)
        script_path = script_file.name

    try:
        cmd = [sys.executable, __file__, "--run-sandbox-script", script_path, sandbox_dir]
        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "timeout": TIMEOUT_SECONDS,
            "cwd": sandbox_dir,
        }
        if os.name != "nt" and HAS_RESOURCE:
            kwargs["preexec_fn"] = set_limits
        result = subprocess.run(cmd, **kwargs)
        return result.stdout
    except subprocess.TimeoutExpired as e:
        return f"Error executing code: TimeoutExpired after {e.timeout} seconds."
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def _main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[0] != "--run-sandbox-script":
        print("Usage: system_execute_code.py --run-sandbox-script <script> <sandbox_dir>")
        return 2
    _, script_path, sandbox_dir = args
    try:
        with open(script_path, "r") as script_file:
            code = script_file.read()
        print(_execute_restricted(code, sandbox_dir), end="")
        return 0
    except PermissionError as e:
        print(f"PermissionError: {e}")
        return 1
    except SandboxSecurityError as e:
        print(f"PermissionError: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
