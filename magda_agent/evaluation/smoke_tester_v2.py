import ast
import logging
from typing import List, Dict, Any

class SmokeTesterV2:
    """
    Smoke tester V2 for post-merge codebase analysis, detecting syntax and import errors.
    """

    def check_syntax(self, file_path: str) -> bool:
        """
        Parses the file using the ast module to ensure there are no syntax errors.

        Args:
            file_path: The path of the file to check.

        Returns:
            bool: True if syntax is correct, False otherwise.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=file_path)
            return True
        except SyntaxError as e:
            logging.error(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            logging.error(f"Could not read/parse {file_path}: {e}")
            return False

    def check_imports(self, file_path: str) -> bool:
        """
        Parses the file and ensures that the imports look well-formed.
        For a deep resolvable module check, one could use importlib.util.find_spec,
        but a simple ast walk allows us to flag invalid ast import nodes.

        Args:
            file_path: The path of the file to check.

        Returns:
            bool: True if imports are syntactically valid and well-formed.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not alias.name:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if not node.module and node.level == 0:
                        return False
            return True
        except Exception as e:
            logging.error(f"Error checking imports in {file_path}: {e}")
            return False

    def run_all_checks(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Runs all smoke tests on a list of files.

        Args:
            file_paths: List of file paths to check.

        Returns:
            Dict containing the status for each file.
        """
        results = {}
        all_passed = True
        for path in file_paths:
            syntax_ok = self.check_syntax(path)
            imports_ok = False
            if syntax_ok:
                imports_ok = self.check_imports(path)

            is_valid = syntax_ok and imports_ok
            if not is_valid:
                all_passed = False

            results[path] = {
                "syntax": syntax_ok,
                "imports": imports_ok,
                "valid": is_valid
            }

        return {
            "success": all_passed,
            "details": results
        }
