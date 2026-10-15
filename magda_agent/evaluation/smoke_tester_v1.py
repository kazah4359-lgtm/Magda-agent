import ast
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SmokeTesterV1:
    """
    Aider-style post-merge smoke tester V1.
    Performs baseline functional verification on agent-generated code files
    by detecting syntax errors and malformed import structures.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def check_syntax(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a Python file using AST to verify syntax validity.

        Args:
            file_path: Path to the Python file.

        Returns:
            Dict containing 'valid' boolean and optional 'error' message.
        """
        if not os.path.exists(file_path):
            return {"valid": False, "error": f"File not found: {file_path}"}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=file_path)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            error_msg = f"SyntaxError line {e.lineno}: {e.msg}"
            if self.verbose:
                logger.error(f"Syntax error in {file_path}: {error_msg}")
            return {"valid": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Read/Parse Error: {str(e)}"
            if self.verbose:
                logger.error(f"Failed to read/parse {file_path}: {error_msg}")
            return {"valid": False, "error": error_msg}

    def check_imports(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a Python file and verifies import AST nodes.

        Args:
            file_path: Path to the Python file.

        Returns:
            Dict containing 'valid' boolean and optional 'error' message.
        """
        if not os.path.exists(file_path):
            return {"valid": False, "error": f"File not found: {file_path}"}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not alias.name:
                            return {"valid": False, "error": "Empty import name encountered"}
                elif isinstance(node, ast.ImportFrom):
                    if not node.module and node.level == 0:
                        return {"valid": False, "error": "Invalid relative/empty import target"}

            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {"valid": False, "error": f"SyntaxError during import analysis: {e.msg}"}
        except Exception as e:
            return {"valid": False, "error": f"Import check error: {str(e)}"}

    def evaluate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Evaluates a single file for syntax and import errors.

        Args:
            file_path: Path to the file.

        Returns:
            Dict with detailed status of syntax, import checks, and overall validity.
        """
        syntax_res = self.check_syntax(file_path)
        if not syntax_res["valid"]:
            return {
                "file_path": file_path,
                "syntax": False,
                "imports": False,
                "valid": False,
                "error": syntax_res["error"],
            }

        import_res = self.check_imports(file_path)
        if not import_res["valid"]:
            return {
                "file_path": file_path,
                "syntax": True,
                "imports": False,
                "valid": False,
                "error": import_res["error"],
            }

        return {
            "file_path": file_path,
            "syntax": True,
            "imports": True,
            "valid": True,
            "error": None,
        }

    def run_smoke_tests(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Runs post-merge smoke tests across a list of file paths.

        Args:
            file_paths: List of file paths to evaluate.

        Returns:
            Summary dict containing pass/fail result, error list, and per-file details.
        """
        details = {}
        errors = []
        passed_count = 0

        for path in file_paths:
            res = self.evaluate_file(path)
            details[path] = res
            if res["valid"]:
                passed_count += 1
            else:
                errors.append(f"{path}: {res['error']}")

        all_passed = (passed_count == len(file_paths)) and len(file_paths) > 0

        return {
            "success": all_passed,
            "total_files": len(file_paths),
            "passed_files": passed_count,
            "failed_files": len(file_paths) - passed_count,
            "errors": errors,
            "details": details,
        }
